#!/usr/bin/env python3
"""
Loop-closure drift analysis + correction for a captured session.

Idea: if the recorder ever revisits an earlier viewpoint, the same world
surface should land in the same place. ARCore drift breaks that invariant
— after some time the camera pose has shifted by a few cm / fractions of
a degree, and a frame from "later, in the same spot" produces world
points that are slightly offset from the "earlier, same-spot" frame's
points.

Pipeline:
  1. Load every frame's pose + a sparse backprojected world-point cloud.
  2. Find candidate pairs (i, j) with i < j whose camera origins are
     within `--pose-radius` metres, whose viewing directions differ by
     less than `--angle-deg` degrees, AND whose frame indices are at
     least `--min-time-gap` frames apart (so consecutive frames don't
     dominate).
  3. Run Procrustes ICP between the two clouds for every candidate.
     The ICP transform (R, t) is the correction that would re-align
     frame j's points back onto frame i's reference.
  4. Print the drift distribution (|t|, rotation angle) bucketed by
     time gap so we can see whether ARCore is drifting consistently.
  5. (with --apply) Build a 1-D pose-graph LS over per-frame corrections:
        anchor: δ₀ = 0
        sequential: δₖ ≈ δₖ₋₁  (smoothness, low weight)
        loop:       δⱼ − δᵢ = (rotvec(R), t) measured by ICP
     Solve 6 decoupled scalar problems (3 rotvec dims + 3 translation
     dims) via least-squares; the result is a per-frame SE(3) correction.
  6. (with --apply) Rewrite each .bin under captured_frames/<session>/
        - frames/         → frames_aligned/
        - frames_refined/ → frames_refined_aligned/
     replacing only the viewMatrix bytes; depth + colour + projection +
     Bd are byte-identical, so the existing voxeliser reads them natively.
  7. (with --apply) Re-run pairwise ICP on the corrected loop pairs and
     print the new residual distribution as a sanity check.

Run:
    python tools/loop_closure_analyze.py --session 20260427_123301
    python tools/loop_closure_analyze.py --session 20260427_123301 --apply
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import fusion
import serve

PROJECT_ROOT = ROOT.parent


# ---------------------------------------------------------------------
# Per-frame loading.
# ---------------------------------------------------------------------

def _load_frame_pose_and_points(
    body: bytes,
    *,
    near: float, far: float,
    n_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Decode a frame, return (cam_origin (3,), forward_dir (3,), points (N,3)
    in world). `n_points` is the target sample count."""
    try:
        frame = serve.parse_frame(body)
    except Exception:
        return None
    width = int(frame["width"]); height = int(frame["height"])
    depth = fusion.decode_depth(
        frame["depth"], width, height,
        int(frame["format"]), float(frame["rawValueToMeters"]),
    )
    V = fusion._mat4_from_column_major(frame["viewMatrix"])
    P = fusion._mat4_from_column_major(frame["projectionMatrix"])
    Bd = fusion._mat4_from_column_major(frame["normDepthBufferFromNormView"])
    cam = V[:3, 3].astype(np.float64)
    # World-space −Z column of view (camera looks down −z in view space).
    forward = -V[:3, 2].astype(np.float64)
    forward = forward / max(np.linalg.norm(forward), 1e-9)

    # Backproject every depth pixel that's in [near, far], then subsample.
    P_inv = np.linalg.inv(P)
    Bv    = np.linalg.inv(Bd)
    bx, by = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64),
                         indexing="xy")
    valid = (depth > near) & (depth < far)
    if int(valid.sum()) < 200:
        return None
    bxv = bx[valid]; byv = by[valid]; dv = depth[valid].astype(np.float64)

    # depth-buffer pixel → norm depth-buffer UV
    u_d = 1.0 - (bxv + 0.5) / width
    v_d = (byv + 0.5) / height
    nd_h = np.stack([u_d, v_d, np.zeros_like(u_d), np.ones_like(u_d)], axis=-1)
    nv_h = nd_h @ Bv.T
    safe = np.where(np.abs(nv_h[:, 3]) > 1e-12, nv_h[:, 3], 1.0)
    u = nv_h[:, 0] / safe
    v = nv_h[:, 1] / safe

    x_ndc = 2.0 * u - 1.0
    y_ndc = 2.0 * v - 1.0
    clip = np.stack([x_ndc, y_ndc, -np.ones_like(x_ndc), np.ones_like(x_ndc)], axis=-1)
    view = clip @ P_inv.T
    view3 = view[:, :3] / np.where(np.abs(view[:, 3:4]) < 1e-12, 1.0, view[:, 3:4])
    ray_dirs = view3 / np.linalg.norm(view3, axis=-1, keepdims=True)

    # In view space the depth buffer's metric `d` is along the ray (matches
    # voxel_reconstruct's convention), so world point = cam + d * (V·dir).
    dirs_world = ray_dirs @ V[:3, :3].T
    pts_world = cam[None, :] + dv[:, None] * dirs_world

    if pts_world.shape[0] > n_points:
        idx = rng.choice(pts_world.shape[0], size=n_points, replace=False)
        pts_world = pts_world[idx]

    return cam, forward, pts_world.astype(np.float64)


# ---------------------------------------------------------------------
# Pairwise ICP (Procrustes / Kabsch).
# ---------------------------------------------------------------------

def _icp_pairwise(
    src: np.ndarray, dst: np.ndarray,
    *,
    max_iters: int = 4,
    match_radius: float = 0.30,
    min_matches: int = 200,
) -> dict | None:
    """Iterated point-to-point ICP. `src` is moved onto `dst`. Returns
    {R, t, residual_before, residual_after, matches, fraction, iters}.
    None if there isn't enough overlap to estimate a transform."""
    if src.size == 0 or dst.size == 0:
        return None

    R_acc = np.eye(3, dtype=np.float64)
    t_acc = np.zeros(3, dtype=np.float64)
    src_cur = src.copy()

    tree = cKDTree(dst)
    initial_residual = None
    last_residual = None
    last_matches = 0

    for it in range(max_iters):
        dists, idxs = tree.query(src_cur, k=1, distance_upper_bound=match_radius)
        finite = np.isfinite(dists)
        n_match = int(finite.sum())
        if n_match < min_matches:
            return None
        s = src_cur[finite]
        d = dst[idxs[finite]]
        if it == 0:
            initial_residual = float(np.median(np.linalg.norm(s - d, axis=1)))
        sc = s.mean(axis=0); dc = d.mean(axis=0)
        H = (s - sc).T @ (d - dc)
        U, _S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = dc - R @ sc
        # Apply this iteration's increment to the running transform and src.
        src_cur = src_cur @ R.T + t
        R_acc = R @ R_acc
        t_acc = R @ t_acc + t
        last_matches = n_match
        last_residual = float(np.median(
            np.linalg.norm((src_cur)[finite] - d, axis=1)
        ))
        # Early stop if residual has stopped moving.
        if it > 0 and abs(initial_residual - last_residual) < 1e-4:
            break

    cos_angle = np.clip((np.trace(R_acc) - 1.0) * 0.5, -1.0, 1.0)
    rot_deg = float(np.degrees(np.arccos(cos_angle)))
    return {
        "R": R_acc, "t": t_acc,
        "residual_before": initial_residual,
        "residual_after":  last_residual,
        "matches": last_matches,
        "fraction": last_matches / float(src.shape[0]),
        "rot_deg": rot_deg,
        "translation_m": float(np.linalg.norm(t_acc)),
        "iters": it + 1,
    }


# ---------------------------------------------------------------------
# Pose-graph solver (linearised, 6 decoupled scalar LS problems).
# ---------------------------------------------------------------------

def _rotmat_to_rotvec(R: np.ndarray) -> np.ndarray:
    """SO(3) → so(3). Robust for small rotations (the regime we expect from
    ARCore drift). Returns a (3,) axis-angle vector."""
    cos_a = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(cos_a)
    if angle < 1e-9:
        # Linearised limit: skew(rotvec) ≈ R - I, recover from off-diagonals.
        return np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) * 0.5
    n = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]]) / (2.0 * np.sin(angle))
    return n * angle


def _rotvec_to_rotmat(v: np.ndarray) -> np.ndarray:
    """so(3) → SO(3) via Rodrigues."""
    angle = float(np.linalg.norm(v))
    if angle < 1e-9:
        return np.eye(3) + np.array([[0, -v[2], v[1]],
                                     [v[2], 0, -v[0]],
                                     [-v[1], v[0], 0]])
    k = v / angle
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _solve_corrections(
    n_frames: int,
    loop_results: list[dict],
    *,
    seq_weight: float,
    loop_weight: float,
    anchor_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a 1-D pose-graph LS (decoupled across the 6 SE(3) dims)
    on the per-frame correction. Returns (rotvec_corr (N,3), trans_corr (N,3)).

    Constraints per dim:
        anchor:        x[0] = 0                  (high weight)
        sequential:    x[k] − x[k−1] = 0         (low weight; smoothness)
        loop (i, j):   x[j] − x[i] = measurement (ICP-derived)

    The ICP transform (R, t) produced by `_icp_pairwise` aligns frame j's
    cloud onto frame i's, so applying T_corr = (R, t) to V_j (V_j' = T·V_j)
    nudges j by exactly the amount needed for the cloud to land on i's. In
    linearised SE(3) that means the *delta* between frame j's and frame
    i's corrections equals (rotvec(R), t).
    """
    # Build (n_frames, 1) blocks of constraints once; apply b for each dim.
    n_seq    = max(0, n_frames - 1)
    n_anchor = 1
    n_loop   = len(loop_results)
    n_rows   = n_anchor + n_seq + n_loop
    A = np.zeros((n_rows, n_frames), dtype=np.float64)
    A[0, 0] = anchor_weight
    for k in range(1, n_frames):
        A[n_anchor + k - 1, k]     = seq_weight
        A[n_anchor + k - 1, k - 1] = -seq_weight
    for r, res in enumerate(loop_results):
        row = n_anchor + n_seq + r
        A[row, res["j"]] = loop_weight
        A[row, res["i"]] = -loop_weight

    # Stack RHS: 3 rotvec dims + 3 translation dims as columns.
    rotvecs = np.array([_rotmat_to_rotvec(res["R"]) for res in loop_results])
    trans   = np.array([res["t"] for res in loop_results])
    B = np.zeros((n_rows, 6), dtype=np.float64)
    for r in range(n_loop):
        row = n_anchor + n_seq + r
        B[row, 0:3] = loop_weight * rotvecs[r]
        B[row, 3:6] = loop_weight * trans[r]

    X, *_ = np.linalg.lstsq(A, B, rcond=None)
    rotvec_corr = X[:, 0:3]
    trans_corr  = X[:, 3:6]
    return rotvec_corr, trans_corr


def _apply_correction_to_view_matrix(
    view_col_major: tuple,
    rotvec_corr: np.ndarray,
    trans_corr: np.ndarray,
) -> tuple:
    """Read 16-element column-major float32, return a 16-tuple after
    premultiplying by T_corr = [[R_corr, t_corr], [0, 1]]."""
    V = np.asarray(view_col_major, dtype=np.float64).reshape(4, 4, order="F")
    R_corr = _rotvec_to_rotmat(rotvec_corr)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_corr
    T[:3, 3]  = trans_corr
    V_new = T @ V
    return tuple(V_new.reshape(-1, order="F").astype(np.float32).tolist())


def _rewrite_frame(
    src_path: Path,
    dst_path: Path,
    new_view: tuple,
) -> None:
    body = src_path.read_bytes()
    out  = bytearray(body)
    # First 16 floats of the header are viewMatrix (column-major float32).
    struct.pack_into("<16f", out, 0, *new_view)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_bytes(bytes(out))


def _rewrite_session(
    session_dir: Path,
    rotvec_corr: np.ndarray,
    trans_corr: np.ndarray,
    valid_idx: list[int],
) -> dict:
    """Apply per-frame corrections to every .bin under frames/ and
    frames_refined/ (if present). Returns a dict of counts written."""
    counts = {}
    for variant in ("frames", "frames_refined"):
        in_dir  = session_dir / variant
        out_dir = session_dir / f"{variant}_aligned"
        if not in_dir.exists():
            continue
        bins = sorted(in_dir.glob("frame_*.bin"))
        # Map frame index → correction. Only frames we successfully loaded
        # (in valid_idx) get corrected; the rest are copied unchanged.
        n_written = 0
        for k, fp in enumerate(bins):
            if k in valid_idx:
                # parse just enough to grab the view matrix (16 floats)
                hdr = fp.open("rb").read(64)
                view_now = struct.unpack("<16f", hdr)
                new_view = _apply_correction_to_view_matrix(
                    view_now, rotvec_corr[k], trans_corr[k],
                )
                _rewrite_frame(fp, out_dir / fp.name, new_view)
            else:
                # No correction available — mirror the original file byte-
                # identical so frame indexing stays consistent downstream.
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / fp.name).write_bytes(fp.read_bytes())
            n_written += 1
        counts[variant] = n_written
    return counts


# ---------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--frames-root", default=str(PROJECT_ROOT / "captured_frames"))
    ap.add_argument("--pose-radius", type=float, default=0.30,
                    help="loop candidates: cam-origin distance ≤ this (m)")
    ap.add_argument("--angle-deg", type=float, default=20.0,
                    help="loop candidates: forward-vec angle ≤ this (deg)")
    ap.add_argument("--min-time-gap", type=int, default=30,
                    help="loop candidates: |i - j| ≥ this (frames)")
    ap.add_argument("--n-points", type=int, default=4000,
                    help="points sampled per frame for ICP")
    ap.add_argument("--max-pairs", type=int, default=400,
                    help="cap on candidate pairs we ICP (most distant in "
                         "time first; use --max-pairs 0 for no cap)")
    ap.add_argument("--match-radius", type=float, default=0.30)
    ap.add_argument("--min-matches", type=int, default=400)
    ap.add_argument("--near", type=float, default=0.05)
    ap.add_argument("--far",  type=float, default=8.0)
    ap.add_argument("--apply", action="store_true",
                    help="Solve the pose graph and write corrected frames "
                         "into captured_frames/<session>/frames_aligned/ "
                         "(and frames_refined_aligned/ if present), then "
                         "verify by re-running ICP on the same loop pairs.")
    ap.add_argument("--seq-weight",    type=float, default=0.10,
                    help="weight on smoothness (sequential δ ≈ δ′) edges")
    ap.add_argument("--loop-weight",   type=float, default=1.00,
                    help="weight on loop-closure edges")
    ap.add_argument("--anchor-weight", type=float, default=10.0,
                    help="weight on the δ₀ = 0 anchor")
    args = ap.parse_args()

    session_dir = Path(args.frames_root) / args.session / "frames"
    if not session_dir.exists():
        print(f"no frames at {session_dir}", file=sys.stderr); sys.exit(1)
    frame_paths = sorted(session_dir.glob("frame_*.bin"))
    if not frame_paths:
        print("no frames", file=sys.stderr); sys.exit(1)
    print(f"loading {len(frame_paths)} frames from {session_dir}")

    rng = np.random.default_rng(0)
    cams: list[np.ndarray] = []
    fwds: list[np.ndarray] = []
    pts:  list[np.ndarray] = []
    skipped = 0
    t_load = time.time()
    for fp in frame_paths:
        info = _load_frame_pose_and_points(
            fp.read_bytes(), near=args.near, far=args.far,
            n_points=args.n_points, rng=rng,
        )
        if info is None:
            skipped += 1
            cams.append(None); fwds.append(None); pts.append(None)
            continue
        cam, fwd, p = info
        cams.append(cam); fwds.append(fwd); pts.append(p)
    n = len(cams)
    print(f"  loaded {n - skipped}/{n} usable frames in {time.time()-t_load:.1f} s "
          f"({args.n_points} pts each)")

    # Frame-to-frame motion summary, for context.
    motions = []
    for i in range(1, n):
        if cams[i] is None or cams[i-1] is None: continue
        motions.append(np.linalg.norm(cams[i] - cams[i-1]))
    motions = np.asarray(motions)
    if motions.size:
        print(f"  inter-frame motion: median {motions.mean()*100:.1f} cm, "
              f"max {motions.max()*100:.1f} cm")
    cam_arr = np.array([c if c is not None else np.full(3, np.nan) for c in cams])
    valid = ~np.any(np.isnan(cam_arr), axis=1)
    if valid.any():
        ext = cam_arr[valid].max(axis=0) - cam_arr[valid].min(axis=0)
        print(f"  pose extent: dx={ext[0]:.2f} dy={ext[1]:.2f} dz={ext[2]:.2f} m")

    # ----- find loop candidates -----
    print(f"\nsearching loop candidates "
          f"(pose ≤ {args.pose_radius:.2f} m, angle ≤ {args.angle_deg:.0f}°, "
          f"|Δi| ≥ {args.min_time_gap}) …")
    valid_idx = [i for i, c in enumerate(cams) if c is not None]
    cam_v = np.array([cams[i] for i in valid_idx])
    fwd_v = np.array([fwds[i] for i in valid_idx])
    tree = cKDTree(cam_v)
    cos_thresh = float(np.cos(np.radians(args.angle_deg)))

    pairs: list[tuple[int, int, float]] = []   # (i_global, j_global, time_gap)
    for k, i_glob in enumerate(valid_idx):
        # All neighbours within the position radius (in valid_idx coords).
        nbrs = tree.query_ball_point(cam_v[k], r=args.pose_radius)
        for k2 in nbrs:
            j_glob = valid_idx[k2]
            if j_glob <= i_glob: continue
            if j_glob - i_glob < args.min_time_gap: continue
            cos_a = float(np.dot(fwd_v[k], fwd_v[k2]))
            if cos_a < cos_thresh: continue
            pairs.append((i_glob, j_glob, j_glob - i_glob))
    print(f"  {len(pairs)} candidate pairs")

    if not pairs:
        print("no loop candidates — try widening --pose-radius / --angle-deg / "
              "lowering --min-time-gap")
        return

    # Sort by time gap descending so we ICP the most informative first when
    # we cap.
    pairs.sort(key=lambda x: -x[2])
    if args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]
    print(f"  ICP'ing {len(pairs)} pairs …")

    # ----- pairwise ICP -----
    t_icp = time.time()
    results = []
    for k, (i, j, gap) in enumerate(pairs):
        out = _icp_pairwise(
            pts[j], pts[i],
            match_radius=args.match_radius,
            min_matches=args.min_matches,
        )
        if out is None: continue
        results.append({
            "i": i, "j": j, "gap": gap,
            **out,
        })
        if (k + 1) % max(1, len(pairs) // 10) == 0:
            print(f"    {k+1}/{len(pairs)}  "
                  f"last gap={gap}  t={out['translation_m']*100:.1f} cm  "
                  f"rot={out['rot_deg']:.2f}°  "
                  f"resid {out['residual_before']*100:.1f} → "
                  f"{out['residual_after']*100:.1f} cm")
    print(f"  done in {time.time()-t_icp:.1f} s, "
          f"{len(results)}/{len(pairs)} pairs gave a usable transform")

    if not results:
        print("no usable ICP results — depth/coverage is probably too thin to "
              "register against, try smaller --pose-radius or larger "
              "--match-radius")
        return

    # ----- drift summary -----
    t_arr = np.array([r["translation_m"] for r in results])
    rot_arr = np.array([r["rot_deg"] for r in results])
    gap_arr = np.array([r["gap"] for r in results])
    res_b = np.array([r["residual_before"] for r in results])
    res_a = np.array([r["residual_after"] for r in results])

    def _q(x, q): return float(np.quantile(x, q))

    print("\n=== drift summary across loop pairs ===")
    print(f"  pairs: {len(results)}")
    print(f"  translation [cm]:  median {np.median(t_arr)*100:5.2f}  "
          f"p90 {_q(t_arr, 0.9)*100:5.2f}  max {t_arr.max()*100:5.2f}")
    print(f"  rotation    [deg]: median {np.median(rot_arr):5.2f}  "
          f"p90 {_q(rot_arr, 0.9):5.2f}  max {rot_arr.max():5.2f}")
    print(f"  ICP residual [cm]: before median {np.median(res_b)*100:5.2f}  "
          f"after  median {np.median(res_a)*100:5.2f}  "
          f"(reduction = {(1 - np.median(res_a)/np.median(res_b))*100:.0f}%)")

    # Bucket by time gap so we can see if drift grows with time.
    print("\n=== drift vs time gap (j − i, frames) ===")
    print(f"  {'gap range':>12}  {'pairs':>5}  {'|t| med':>8}  {'|t| p90':>8}  "
          f"{'rot med':>7}  {'rot p90':>7}  {'res→':>7}")
    edges = [args.min_time_gap, 60, 100, 150, 200, 300, 500, 10**9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (gap_arr >= lo) & (gap_arr < hi)
        if int(mask.sum()) < 3: continue
        t_b = t_arr[mask]; r_b = rot_arr[mask]; ra_b = res_a[mask]
        label = f"{lo}–{hi-1}" if hi < 10**9 else f"{lo}+"
        print(f"  {label:>12}  {int(mask.sum()):>5}  "
              f"{np.median(t_b)*100:>7.2f}cm  {_q(t_b, 0.9)*100:>7.2f}cm  "
              f"{np.median(r_b):>6.2f}°  {_q(r_b, 0.9):>6.2f}°  "
              f"{np.median(ra_b)*100:>6.2f}cm")

    # Top offenders — the worst-aligned pairs.
    print("\n=== worst-aligned loop pairs (by translation) ===")
    results_sorted = sorted(results, key=lambda r: -r["translation_m"])
    for r in results_sorted[:10]:
        print(f"  i={r['i']:4d} j={r['j']:4d} gap={r['gap']:4d}  "
              f"|t|={r['translation_m']*100:5.1f} cm  "
              f"rot={r['rot_deg']:4.2f}°  "
              f"matches={r['matches']:5d} ({r['fraction']*100:4.0f}%)  "
              f"residual {r['residual_before']*100:.1f}→{r['residual_after']*100:.1f} cm")

    if not args.apply:
        return

    # ----- pose-graph solve + apply -----
    print("\n=== solving pose-graph corrections ===")
    rotvec_corr, trans_corr = _solve_corrections(
        n,
        results,
        seq_weight=args.seq_weight,
        loop_weight=args.loop_weight,
        anchor_weight=args.anchor_weight,
    )
    # Frames we couldn't load get zero correction. The rewriter passes
    # them through unchanged.
    valid_set = set(valid_idx)
    for k in range(n):
        if k not in valid_set:
            rotvec_corr[k] = 0.0
            trans_corr[k]  = 0.0

    rot_mag = np.linalg.norm(rotvec_corr, axis=1)
    t_mag   = np.linalg.norm(trans_corr,  axis=1)
    print(f"  per-frame correction magnitude:")
    print(f"    |t|  : 0% {0.0:.2f}  median {np.median(t_mag)*100:5.2f} cm  "
          f"p90 {_q(t_mag, 0.9)*100:5.2f}  max {t_mag.max()*100:5.2f}")
    print(f"    |rot|:           median {np.degrees(np.median(rot_mag)):5.2f}°  "
          f"p90 {np.degrees(_q(rot_mag, 0.9)):5.2f}  "
          f"max {np.degrees(rot_mag.max()):5.2f}")
    # Sanity: the magnitude should grow over time (drift accumulates).
    bins_for_print = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    print(f"  correction at sample frames:")
    for k in bins_for_print:
        print(f"    frame {k:4d}:  |t|={t_mag[k]*100:5.2f} cm  "
              f"|rot|={np.degrees(rot_mag[k]):5.2f}°")

    print("\n=== rewriting corrected frames ===")
    session_dir = Path(args.frames_root) / args.session
    counts = _rewrite_session(session_dir, rotvec_corr, trans_corr, valid_idx)
    for variant, n_written in counts.items():
        print(f"  {variant}_aligned/: {n_written} frames")

    # ----- verify: re-run ICP on the same loop pairs after correction -----
    print("\n=== verify: re-ICP loop pairs after correction ===")
    # Apply the correction to the in-memory point clouds so we don't reload
    # all the bins. Each frame's correction T_k is applied to its world points.
    pts_corr = []
    for k in range(n):
        if pts[k] is None:
            pts_corr.append(None)
            continue
        R_k = _rotvec_to_rotmat(rotvec_corr[k])
        t_k = trans_corr[k]
        pts_corr.append(pts[k] @ R_k.T + t_k)
    res_after_apply = []
    for r in results:
        i, j = r["i"], r["j"]
        out = _icp_pairwise(
            pts_corr[j], pts_corr[i],
            match_radius=args.match_radius,
            min_matches=args.min_matches,
        )
        if out is None: continue
        res_after_apply.append(out)
    if res_after_apply:
        t_after  = np.array([x["translation_m"] for x in res_after_apply])
        r_after  = np.array([x["rot_deg"]       for x in res_after_apply])
        ra_after = np.array([x["residual_after"] for x in res_after_apply])
        print(f"  post-correction translation [cm]:  median "
              f"{np.median(t_after)*100:.2f}  p90 {_q(t_after, 0.9)*100:.2f}  "
              f"(was {np.median(t_arr)*100:.2f} / {_q(t_arr, 0.9)*100:.2f})")
        print(f"  post-correction rotation    [deg]: median "
              f"{np.median(r_after):.2f}  p90 {_q(r_after, 0.9):.2f}  "
              f"(was {np.median(rot_arr):.2f} / {_q(rot_arr, 0.9):.2f})")
        print(f"  post-correction ICP residual [cm]: median "
              f"{np.median(ra_after)*100:.2f}  (was {np.median(res_a)*100:.2f})")

    print("\nNext step: run the voxeliser on the aligned variant, e.g.:")
    print(f"  tools/voxel_reverse_rust/target/release/voxel-reverse \\")
    print(f"    --frames-dir captured_frames/{args.session}/frames_aligned \\")
    print(f"    --out captured_frames/{args.session}/voxels_aligned.json")


if __name__ == "__main__":
    main()
