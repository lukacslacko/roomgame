#!/usr/bin/env python3
"""
Feature-driven bundle-adjustment refinement of WebXR poses.

Reads a session's `features_meta*.json` (the per-feature observation list
produced by `feature_ray_reconstruct.py`) plus the per-frame `.bin` files
(view + projection matrices), and jointly refines:

  • per-frame SE(3) pose corrections   (anchored at the first observed frame)
  • per-feature 3D world positions

to minimise pixel reprojection error across all observations. Robust
Huber loss tolerates the residual outliers that survive the Lowe ratio
+ cross-check + per-pair geometric filters in feature_ray_reconstruct.

The optimisation variable is roughly 6·(N−1) + 3·F where N ≈ 200 frames
and F ≈ 15 000 features → ~50 K parameters. Each observation contributes
two residuals; the Jacobian is sparse (9 nonzero columns per residual)
so scipy.optimize.least_squares with jac_sparsity + finite differences
converges in a few minutes on this scale.

Output: a new `frames_<dst>/` directory under `captured_frames/<session>/`
holding the source `.bin` files with the corrected viewMatrix bytes
spliced in (depth, colour, projection, Bd are byte-identical, so the
existing voxelisers + feature_ray_reconstruct read it natively). Re-run

    feature_ray_reconstruct.py --session <id> --frames-variant frames_<dst>

afterwards to get a fresh feature voxelisation against the refined poses.

Why world-frame correction (V_c2w' = T_corr @ V_c2w) rather than
camera-local (V_c2w' = V_c2w @ T_corr): this matches loop_closure_analyze's
convention so a session's frames_aligned/ output composes naturally with
this stage if you decide to chain them.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import fusion  # noqa: E402
import serve as _serve  # noqa: E402

parse_frame = _serve.parse_frame


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

_FRAME_BIN_RE = re.compile(r"frame_(\d+)\.bin")


def _features_meta_filename(frames_variant: str) -> str:
    """Map a frames-variant directory name to the matching features_meta
    sidecar (mirrors feature_ray_reconstruct.py's suffix logic)."""
    if frames_variant == "frames":
        return "features_meta.json"
    suffix = frames_variant.removeprefix("frames_")
    return f"features_meta_{suffix}.json"


def load_session_frames(frames_dir: Path) -> dict[int, dict]:
    """Read every frame_<idx>.bin and return idx → {V_c2w, V_w2c, P_proj,
    cw, ch}. Frames whose colour is missing or whose viewMatrix is not
    invertible are skipped (BA can't use them)."""
    out: dict[int, dict] = {}
    for p in sorted(frames_dir.glob("frame_*.bin")):
        m = _FRAME_BIN_RE.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        try:
            frame = parse_frame(p.read_bytes())
        except Exception:  # noqa: BLE001
            continue
        cw = int(frame["color_width"]); ch = int(frame["color_height"])
        if cw == 0 or ch == 0:
            continue
        V_c2w = fusion._mat4_from_column_major(frame["viewMatrix"])
        P_proj = fusion._mat4_from_column_major(frame["projectionMatrix"])
        try:
            V_w2c = np.linalg.inv(V_c2w)
        except np.linalg.LinAlgError:
            continue
        out[idx] = {
            "V_c2w": V_c2w.astype(np.float64),
            "V_w2c": V_w2c.astype(np.float64),
            "P_proj": P_proj.astype(np.float64),
            "cw": cw, "ch": ch,
        }
    return out


# ---------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------

def build_problem(meta: dict, frame_data: dict[int, dict],
                  *, max_features: int | None = None) -> dict:
    """Flatten features_meta into per-observation arrays. Drops features
    with fewer than 2 surviving observations (after frames missing on
    disk are filtered out) so the BA Jacobian's per-feature block stays
    well-conditioned."""
    obs_frame_ids: list[int] = []
    obs_uv: list[list[float]] = []
    init_pts: list[list[float]] = []
    obs_pt_idx: list[int] = []

    for v_meta in meta.get("voxels", []):
        for f in v_meta.get("features", []):
            pt = len(init_pts)
            init_pts.append(f["world"])
            for ob in f.get("obs", []):
                fi = int(ob["frame"])
                if fi not in frame_data:
                    continue
                obs_frame_ids.append(fi)
                obs_uv.append([float(ob["u"]), float(ob["v"])])
                obs_pt_idx.append(pt)

    obs_frame_ids = np.asarray(obs_frame_ids, dtype=np.int64)
    obs_uv = np.asarray(obs_uv, dtype=np.float64)
    obs_pt_idx = np.asarray(obs_pt_idx, dtype=np.int64)
    init_pts = np.asarray(init_pts, dtype=np.float64)

    # Drop tracks that lost too many observations to qualify.
    n_per_pt = np.bincount(obs_pt_idx, minlength=len(init_pts))
    keep_pt = np.where(n_per_pt >= 2)[0]
    if max_features is not None and len(keep_pt) > max_features:
        rng = np.random.default_rng(0)
        keep_pt = np.sort(rng.choice(keep_pt, max_features, replace=False))
    pt_remap = -np.ones(len(init_pts), dtype=np.int64)
    pt_remap[keep_pt] = np.arange(len(keep_pt))
    keep_obs_mask = pt_remap[obs_pt_idx] >= 0
    obs_frame_ids = obs_frame_ids[keep_obs_mask]
    obs_uv = obs_uv[keep_obs_mask]
    obs_pt_idx = pt_remap[obs_pt_idx[keep_obs_mask]]
    init_pts = init_pts[keep_pt]

    # Frame indexing (0..N-1 over frames that actually appear in obs).
    unique_frames = np.unique(obs_frame_ids)
    frame_to_local = {int(f): i for i, f in enumerate(unique_frames)}
    obs_frame_local = np.fromiter(
        (frame_to_local[int(f)] for f in obs_frame_ids),
        dtype=np.int64, count=len(obs_frame_ids),
    )

    # Anchor: the lowest-numbered observed frame keeps its pose fixed.
    anchor_local = 0   # by construction unique_frames is sorted ascending

    # Per-frame matrices stacked.
    V_w2c = np.stack([frame_data[int(f)]["V_w2c"]   for f in unique_frames])
    P_proj = np.stack([frame_data[int(f)]["P_proj"] for f in unique_frames])
    cw = np.array([frame_data[int(f)]["cw"] for f in unique_frames], dtype=np.float64)
    ch = np.array([frame_data[int(f)]["ch"] for f in unique_frames], dtype=np.float64)

    obs_cw = cw[obs_frame_local]
    obs_ch = ch[obs_frame_local]

    return {
        "n_frames": int(len(unique_frames)),
        "n_pts":    int(len(init_pts)),
        "anchor_local": int(anchor_local),
        "unique_frames": unique_frames,
        "frame_to_local": frame_to_local,
        "V_w2c": V_w2c,
        "P_proj": P_proj,
        "obs_frame": obs_frame_local,
        "obs_pt": obs_pt_idx,
        "obs_uv": obs_uv,
        "obs_cw": obs_cw,
        "obs_ch": obs_ch,
        "init_points": init_pts,
    }


# ---------------------------------------------------------------------
# Parameter packing: all non-anchor frames first ((ω, t) pairs), then
# all 3D points. Anchor frame's correction is implicitly zero.
# ---------------------------------------------------------------------

def _pose_indices(problem: dict) -> np.ndarray:
    """Local frame indices that have pose params (i.e. excluding anchor)."""
    n = problem["n_frames"]
    a = problem["anchor_local"]
    return np.array([i for i in range(n) if i != a], dtype=np.int64)


def initial_x(problem: dict) -> np.ndarray:
    n_pose = problem["n_frames"] - 1
    x = np.zeros(n_pose * 6 + problem["n_pts"] * 3, dtype=np.float64)
    x[n_pose * 6:] = problem["init_points"].reshape(-1)
    return x


def unpack_x(x: np.ndarray, problem: dict
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (omegas (N,3), ts (N,3), points (F,3)) with anchor's
    correction expanded to zeros."""
    n = problem["n_frames"]
    n_pose = n - 1
    pose_block = x[: n_pose * 6].reshape(n_pose, 6)
    points = x[n_pose * 6:].reshape(-1, 3).copy()
    omegas = np.zeros((n, 3), dtype=np.float64)
    ts = np.zeros((n, 3), dtype=np.float64)
    pose_idx = _pose_indices(problem)
    omegas[pose_idx] = pose_block[:, :3]
    ts[pose_idx] = pose_block[:, 3:]
    return omegas, ts, points


# ---------------------------------------------------------------------
# Forward model (vectorised over all observations).
#
# Convention (matches loop_closure_analyze's premultiply convention):
#     V_c2w'_i = T_i @ V_c2w_i,   T_i = [R(ω_i), t_i; 0, 1]
# ⇒  V_w2c'_i = V_w2c_i @ T_i⁻¹
# Forward project:
#     P'  = T_i⁻¹ @ [P; 1] = [R(ω_i)ᵀ (P − t_i); 1]   (world after correction)
#     p_view = V_w2c_i @ [P'; 1]
#     p_clip = P_proj_i @ p_view
#     uv     = ((p_clip[:2] / p_clip[3]) + 1) / 2
# Residual:
#     r = (uv − uv_obs) ⊙ (cw_i, ch_i)   (pixel-space, isotropic)
# ---------------------------------------------------------------------

def _frame_rotmats(omegas: np.ndarray) -> np.ndarray:
    """Per-frame rotvec → 3×3 matrix. cv2.Rodrigues on a 3-vec is fast,
    and we only call it n_frames times per residual evaluation (~200×)."""
    out = np.empty((len(omegas), 3, 3), dtype=np.float64)
    for i, w in enumerate(omegas):
        if not np.any(w):
            out[i] = np.eye(3)
        else:
            out[i], _ = cv2.Rodrigues(w.astype(np.float64))
    return out


def compute_residuals(x: np.ndarray, problem: dict) -> np.ndarray:
    omegas, ts, points = unpack_x(x, problem)
    Rs = _frame_rotmats(omegas)             # (N, 3, 3)

    obs_frame = problem["obs_frame"]
    obs_pt    = problem["obs_pt"]
    M = len(obs_frame)

    P_per     = points[obs_pt]              # (M, 3)
    t_per     = ts[obs_frame]               # (M, 3)
    R_per     = Rs[obs_frame]               # (M, 3, 3)
    V_w2c_per = problem["V_w2c"][obs_frame]   # (M, 4, 4)
    P_proj_per = problem["P_proj"][obs_frame] # (M, 4, 4)

    # P_corrected = R^T @ (P − t)
    diff = P_per - t_per                                # (M, 3)
    Pc   = np.einsum("mji,mj->mi", R_per, diff)         # (M, 3)
    Pc_h = np.concatenate([Pc, np.ones((M, 1))], axis=1)
    p_view = np.einsum("mij,mj->mi", V_w2c_per, Pc_h)     # (M, 4)
    p_clip = np.einsum("mij,mj->mi", P_proj_per, p_view)  # (M, 4)
    safe_w = np.where(np.abs(p_clip[:, 3]) > 1e-9, p_clip[:, 3], 1.0)
    p_ndc_xy = p_clip[:, :2] / safe_w[:, None]
    uv_pred = (p_ndc_xy + 1.0) * 0.5

    r_norm = uv_pred - problem["obs_uv"]
    r = np.empty(2 * M, dtype=np.float64)
    r[0::2] = r_norm[:, 0] * problem["obs_cw"]
    r[1::2] = r_norm[:, 1] * problem["obs_ch"]
    return r


# ---------------------------------------------------------------------
# Jacobian sparsity: each observation row only depends on its frame's
# 6 pose params (or none, for the anchor) + its point's 3 position
# params. Letting scipy's TRF do finite differences with this pattern
# costs ~9 residual evaluations per Jacobian, dominated by vectorised
# numpy einsums.
# ---------------------------------------------------------------------

def build_jac_sparsity(problem: dict) -> csr_matrix:
    M = len(problem["obs_frame"])
    n_pose = problem["n_frames"] - 1
    n_params = n_pose * 6 + problem["n_pts"] * 3

    pose_offset = np.full(problem["n_frames"], -1, dtype=np.int64)
    j = 0
    for i in range(problem["n_frames"]):
        if i == problem["anchor_local"]:
            continue
        pose_offset[i] = j
        j += 6

    rows: list[int] = []
    cols: list[int] = []
    for m in range(M):
        fi  = int(problem["obs_frame"][m])
        pt  = int(problem["obs_pt"][m])
        cs: list[int] = []
        if pose_offset[fi] >= 0:
            cs.extend(range(pose_offset[fi], pose_offset[fi] + 6))
        cs.extend(range(n_pose * 6 + pt * 3, n_pose * 6 + pt * 3 + 3))
        for r_off in (0, 1):
            r = 2 * m + r_off
            rows.extend([r] * len(cs))
            cols.extend(cs)
    data = np.ones(len(rows), dtype=np.uint8)
    return csr_matrix((data, (rows, cols)), shape=(2 * M, n_params))


# ---------------------------------------------------------------------
# Apply corrections to the on-disk frame.bin files.
# ---------------------------------------------------------------------

def write_corrected_frames(omegas: np.ndarray, ts: np.ndarray,
                           problem: dict, src_dir: Path, dst_dir: Path
                           ) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for src_path in sorted(src_dir.glob("frame_*.bin")):
        m = _FRAME_BIN_RE.match(src_path.name)
        if not m:
            continue
        idx = int(m.group(1))
        body = src_path.read_bytes()
        local = problem["frame_to_local"].get(idx)
        if local is None:
            # Frame not in problem (no observations) → carry through unchanged.
            (dst_dir / src_path.name).write_bytes(body)
            continue
        omega = omegas[local]
        t     = ts[local]
        R = (np.eye(3) if not np.any(omega)
             else cv2.Rodrigues(omega.astype(np.float64))[0])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3]  = t
        view_orig = np.array(struct.unpack("<16f", body[:64]), dtype=np.float64)
        V_c2w = view_orig.reshape(4, 4, order="F")
        V_new = T @ V_c2w
        view_new = V_new.reshape(-1, order="F").astype(np.float32).tolist()
        out = bytearray(body)
        struct.pack_into("<16f", out, 0, *view_new)
        (dst_dir / src_path.name).write_bytes(bytes(out))
        n_written += 1
    return n_written


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--session", required=True)
    ap.add_argument("--frames-variant", default="frames",
                    help="Source viewMatrices to refine. Any per-frame .bin "
                         "directory under captured_frames/<session>/ works "
                         "(e.g. `frames`, `frames_aligned`, "
                         "`frames_refined_aligned`, or a previous BA output). "
                         "The matching features_meta sidecar is read from "
                         "the same suffix.")
    ap.add_argument("--dst-variant", default="frames_feature_ba",
                    help="Output dir name under captured_frames/<session>/.")
    ap.add_argument("--max-features", type=int, default=None,
                    help="Cap on point parameter count for quick smoke "
                         "tests. Default: use every feature in the meta.")
    ap.add_argument("--max-nfev", type=int, default=400,
                    help="Hard upper bound on residual evaluations passed "
                         "through to scipy.least_squares. Each Jacobian "
                         "eval = ~9 residual evals.")
    ap.add_argument("--huber", type=float, default=3.0,
                    help="Huber loss scale (pixels). Larger = more "
                         "tolerant of outliers in the residual tail. Set "
                         "to 0 for plain L2 (faster but outlier-sensitive).")
    ap.add_argument("--xtol", type=float, default=1e-6)
    ap.add_argument("--ftol", type=float, default=1e-6)
    ap.add_argument("--gtol", type=float, default=1e-8)
    args = ap.parse_args()

    sess_dir = ROOT / "captured_frames" / args.session
    if not sess_dir.is_dir():
        sys.exit(f"session not found: {sess_dir}")

    frames_dir = sess_dir / args.frames_variant
    if not frames_dir.is_dir():
        sys.exit(f"frames variant not found: {frames_dir}")

    meta_path = sess_dir / _features_meta_filename(args.frames_variant)
    if not meta_path.exists():
        sys.exit(
            f"{meta_path.name} not found in {sess_dir}.\n"
            f"  Run first:\n"
            f"    python tools/feature_ray_reconstruct.py "
            f"--session {args.session} --frames-variant {args.frames_variant}"
        )

    print(f"[load] {meta_path.name}")
    meta = json.loads(meta_path.read_text())
    print(f"[load] frames from {frames_dir}")
    frame_data = load_session_frames(frames_dir)
    print(f"  {len(frame_data)} frames decoded")

    print("[setup] flattening problem")
    problem = build_problem(meta, frame_data, max_features=args.max_features)
    M = len(problem["obs_frame"])
    print(f"  frames={problem['n_frames']:,}  "
          f"features={problem['n_pts']:,}  "
          f"observations={M:,}")
    if M == 0:
        sys.exit("no observations survived; can't run BA.")

    x0 = initial_x(problem)
    n_pose_params = (problem["n_frames"] - 1) * 6
    n_pt_params = problem["n_pts"] * 3
    print(f"  parameters={x0.size:,} ({n_pose_params:,} pose + {n_pt_params:,} point)")

    r0 = compute_residuals(x0, problem)
    r0_pix = np.linalg.norm(r0.reshape(M, 2), axis=1)
    print(f"[init] reprojection error (px): "
          f"p50={np.percentile(r0_pix, 50):.2f} "
          f"p90={np.percentile(r0_pix, 90):.2f} "
          f"p99={np.percentile(r0_pix, 99):.2f} "
          f"max={r0_pix.max():.2f}")

    print("[setup] building Jacobian sparsity pattern")
    t0 = time.time()
    jac_sp = build_jac_sparsity(problem)
    print(f"  shape={jac_sp.shape}  nnz={jac_sp.nnz:,}  "
          f"({time.time()-t0:.1f}s)")

    loss = "huber" if args.huber > 0 else "linear"
    f_scale = max(args.huber, 1.0)
    print(f"[solve] scipy.least_squares method=trf loss={loss} f_scale={f_scale} "
          f"max_nfev={args.max_nfev}")
    t0 = time.time()
    result = least_squares(
        compute_residuals, x0,
        jac_sparsity=jac_sp,
        args=(problem,),
        method="trf",
        loss=loss, f_scale=f_scale,
        xtol=args.xtol, ftol=args.ftol, gtol=args.gtol,
        max_nfev=args.max_nfev,
        verbose=2,
    )
    print(f"[solve] done in {time.time()-t0:.1f}s  "
          f"status={result.status}  nfev={result.nfev}  njev={result.njev}")
    print(f"        message: {result.message}")

    r1 = compute_residuals(result.x, problem)
    r1_pix = np.linalg.norm(r1.reshape(M, 2), axis=1)
    print(f"[done] reprojection error (px): "
          f"p50={np.percentile(r1_pix, 50):.2f} "
          f"p90={np.percentile(r1_pix, 90):.2f} "
          f"p99={np.percentile(r1_pix, 99):.2f} "
          f"max={r1_pix.max():.2f}")

    omegas, ts, _ = unpack_x(result.x, problem)
    omega_norms = np.linalg.norm(omegas, axis=1)
    t_norms     = np.linalg.norm(ts, axis=1)
    print(f"[done] per-frame correction magnitudes: "
          f"|ω| p50={np.degrees(np.median(omega_norms)):.3f}° "
          f"p99={np.degrees(np.percentile(omega_norms, 99)):.3f}° | "
          f"|t| p50={np.median(t_norms)*100:.2f} cm "
          f"p99={np.percentile(t_norms, 99)*100:.2f} cm")

    dst_dir = sess_dir / args.dst_variant
    print(f"[write] corrected frames → {dst_dir}")
    n = write_corrected_frames(omegas, ts, problem, frames_dir, dst_dir)
    print(f"  wrote {n} frames")
    print(f"\nNext step:\n"
          f"  python tools/feature_ray_reconstruct.py "
          f"--session {args.session} --frames-variant {args.dst_variant}")


if __name__ == "__main__":
    main()
