#!/usr/bin/env python3
"""
Estimate every frame's pose from feature observations alone, *without*
trusting the WebXR per-frame view matrices. Pipeline:

  1. Take the WebXR-triangulated 3D points from `features_meta.json` as
     the initial *world map* — these encode SCENE STRUCTURE, not per-
     frame poses, so seeding from them tests whether the per-frame
     poses are consistent with the 3D structure that the OBSERVATIONS
     imply. (The world map is refined later by BA, so any structural
     bias from WebXR-induced triangulation washes out.)

  2. For every frame, run solvePnPRansac against the visible portion of
     that map — recover V_w2c entirely from the 2D observations + the
     3D map, with NO use of the frame's WebXR view matrix. Frames with
     too few inliers (default <8) are dropped from the final pose set
     and the writer carries the original .bin through so the file
     count stays right (these tend to be late-session frames where
     ARCore tracking glitched and the ORB feature density was already
     low — the "failure mode" is essentially "no useful pose at all").

  3. Final bundle-adjustment over all (poses + 3D points) using the
     same residual / sparse-Jacobian / Huber-loss machinery as
     tools/feature_pose_align.py. Frame 0 is anchored to fix the gauge
     ambiguity. The init poses going into BA are the from-scratch
     PnP results, NOT WebXR — so any drift / bias the WebXR view
     matrices accumulate over the session has no chance to bleed in.

  4. Splice the new view matrices into a fresh `frames_<dst>/` dir
     under captured_frames/<session>/. Defaults to
     `frames_pose_scratch/`. The voxel reverse builder reads from
     this dir natively (depth, colour, projection, Bd are all
     byte-identical to the source).

Optional: --bootstrap=phone replaces step 1 with a phone-depth-only
bootstrap from the anchor frame's observations + incremental PnP. That
mode discards WebXR even at the structural level but typically only
recovers a handful of frames on dense scans because tracks rarely
overlap densely enough to chain through phone-depth-only seeds.

Usage:
    python tools/feature_pose_from_scratch.py --session <id>
        [--frames-variant frames]              # source bins / features_meta
        [--dst-variant frames_pose_scratch]
        [--no-anchor-webxr]                    # frame 0 → identity
        [--max-features N] [--max-nfev N] [--huber 3.0]
        [--pnp-min-pts 6] [--pnp-reproj-px 3.0]

Output is a fresh frames_<dst>/ directory with corrected viewMatrix
bytes. Run the voxel reverse builder afterwards:
    ./tools/voxel_reverse_rust/target/release/voxel-reverse \\
        --session <id> --frames-dir captured_frames/<id>/<dst> --voxel-size 0.02 --tol 0.04
or use --session mode (it will pick up the new frames_pose_scratch/ if
you symlink/rename it to a recognised variant).
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
import feature_pose_align as fpa  # noqa: E402  (residuals + jac sparsity)

parse_frame = _serve.parse_frame
_FRAME_BIN_RE = re.compile(r"frame_(\d+)\.bin")

# GL ↔ OpenCV camera frame: x is shared, y/z flip sign. Multiplying by M
# converts a point from one to the other.
_M = np.diag([1.0, -1.0, -1.0])


# ────────────────────────── intrinsics + helpers ──────────────────────────

def K_from_proj(P_proj: np.ndarray, cw: int, ch: int) -> np.ndarray:
    """Derive an OpenCV-style K matrix from a WebXR / GL perspective
    projection matrix. The WebXR matrix is column-major in the bin file
    but `_mat4_from_column_major` returns it in row-major numpy form, so
    P_proj[i, j] is the (i, j) entry of the standard 4×4 perspective
    matrix.

    For a point (X, Y, Z) in GL camera space (camera at origin, looking
    down -Z, +Y up), ndc_x = -P[0,0]·X/Z - P[0,2], and the corresponding
    OpenCV pixel column is (1 + ndc_x)·cw/2. After substituting Z_oc =
    -Z_gl, x_oc = X_gl, this gives fx = cw·P[0,0]/2 and cx = cw·(1 −
    P[0,2])/2. Mirror logic for fy / cy with the y-axis flip.
    """
    fx = cw * P_proj[0, 0] / 2.0
    fy = ch * P_proj[1, 1] / 2.0
    cx = cw * (1.0 - P_proj[0, 2]) / 2.0
    cy = ch * (1.0 + P_proj[1, 2]) / 2.0
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def uv_to_pixel(u: float, v: float, cw: int, ch: int) -> tuple[float, float]:
    """Map norm-view (u, v) — u=0 left, v=1 top — to OpenCV pixel
    coords (col, row) with row 0 at the top of the image."""
    return (u * cw, (1.0 - v) * ch)


def Voc_to_Vgl(R_oc: np.ndarray, t_oc: np.ndarray) -> np.ndarray:
    """Convert (R, t) in OpenCV's world-to-camera convention (x_oc =
    R·X_w + t) to a 4×4 V_w2c in GL camera frame. Result satisfies
    x_gl = M·(R·X_w + t)."""
    V = np.eye(4)
    V[:3, :3] = _M @ R_oc
    V[:3, 3]  = _M @ t_oc.reshape(3)
    return V


def Vgl_to_Voc(V_w2c_gl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `Voc_to_Vgl`. Returns (R_oc, t_oc)."""
    R_oc = _M.T @ V_w2c_gl[:3, :3]
    t_oc = _M.T @ V_w2c_gl[:3, 3]
    return R_oc, t_oc


# ────────────────────────── data loading ──────────────────────────

def load_session_frames(frames_dir: Path) -> dict[int, dict]:
    """idx → {V_c2w_orig, V_w2c_orig, P_proj, K, cw, ch, depth, dw, dh,
    fmt, raw_to_m}. The depth + dims + format are needed to bootstrap
    metric scale at frame 0; the orig V_c2w/V_w2c are read but only
    used as the optional anchor."""
    out: dict[int, dict] = {}
    for p in sorted(frames_dir.glob("frame_*.bin")):
        m = _FRAME_BIN_RE.match(p.name)
        if not m: continue
        idx = int(m.group(1))
        try:
            frame = parse_frame(p.read_bytes())
        except Exception:
            continue
        cw = int(frame["color_width"]); ch = int(frame["color_height"])
        if cw == 0 or ch == 0: continue
        V_c2w = fusion._mat4_from_column_major(frame["viewMatrix"])
        P_proj = fusion._mat4_from_column_major(frame["projectionMatrix"])
        try:
            V_w2c = np.linalg.inv(V_c2w)
        except np.linalg.LinAlgError:
            continue
        Bd = fusion._mat4_from_column_major(frame["normDepthBufferFromNormView"])
        dw = int(frame["width"]); dh = int(frame["height"])
        depth = fusion.decode_depth(
            frame["depth"], dw, dh,
            int(frame["format"]), float(frame["rawValueToMeters"]),
        )
        out[idx] = {
            "V_c2w_orig": V_c2w.astype(np.float64),
            "V_w2c_orig": V_w2c.astype(np.float64),
            "P_proj":     P_proj.astype(np.float64),
            "Bd":         Bd.astype(np.float64),
            "K":          K_from_proj(P_proj, cw, ch),
            "cw": cw, "ch": ch,
            "depth": depth,
            "dw": dw, "dh": dh,
        }
    return out


def build_tracks(meta: dict, frame_data: dict[int, dict],
                 max_features: int | None = None
                 ) -> tuple[list[list[tuple[int, float, float]]], np.ndarray]:
    """Return (tracks, init_pts). `tracks[i]` is the list of (frame_idx,
    u, v) observations for track i, sorted by frame_idx. `init_pts[i]`
    is the WebXR-anchored 3D position from features_meta — kept around
    only for diagnostic comparison; the from-scratch path doesn't seed
    BA with it. Tracks observed in fewer than 2 surviving frames are
    dropped (they can't constrain anything)."""
    tracks: list[list[tuple[int, float, float]]] = []
    init_pts: list[np.ndarray] = []
    for v_meta in meta.get("voxels", []):
        for f in v_meta.get("features", []):
            obs = []
            for ob in f.get("obs", []):
                fi = int(ob["frame"])
                if fi not in frame_data: continue
                obs.append((fi, float(ob["u"]), float(ob["v"])))
            if len(obs) < 2: continue
            obs.sort(key=lambda o: o[0])
            tracks.append(obs)
            init_pts.append(np.asarray(f["world"], dtype=np.float64))
    if max_features is not None and len(tracks) > max_features:
        rng = np.random.default_rng(0)
        keep = sorted(rng.choice(len(tracks), max_features, replace=False))
        tracks = [tracks[i] for i in keep]
        init_pts = [init_pts[i] for i in keep]
    return tracks, np.asarray(init_pts, dtype=np.float64)


# ────────────────────────── scale bootstrap on frame 0 ──────────────────────────

def bootstrap_anchor(anchor_idx: int,
                     anchor_data: dict,
                     tracks: list[list[tuple[int, float, float]]],
                     anchor_webxr: bool
                     ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Fix anchor frame's V_w2c (either WebXR's or identity) and
    triangulate every track observed there using phone depth.

    Returns (anchor_V_w2c_gl, point_world) where point_world maps a
    track index to its (x, y, z) world position. Only tracks with valid
    phone depth at the observation pixel land in `point_world`."""
    if anchor_webxr:
        V_w2c = anchor_data["V_w2c_orig"].copy()
    else:
        V_w2c = np.eye(4, dtype=np.float64)

    # Phone-depth lookup on anchor frame: norm-view (u, v) → pixel via Bd⁻¹.
    Bv = np.linalg.inv(anchor_data["Bd"])
    dw = anchor_data["dw"]; dh = anchor_data["dh"]
    depth = anchor_data["depth"]
    P_proj = anchor_data["P_proj"]

    V_c2w = np.linalg.inv(V_w2c)
    point_world: dict[int, np.ndarray] = {}

    for ti, obs_list in enumerate(tracks):
        # Find the anchor's observation in this track, if any.
        anc_obs = next((o for o in obs_list if o[0] == anchor_idx), None)
        if anc_obs is None: continue
        _, u_v, v_v = anc_obs

        # Phone depth at norm-view (u_v, v_v).
        nv = np.array([u_v, v_v, 0.0, 1.0])
        nd = Bv @ nv
        if abs(nd[3]) < 1e-12: continue
        u_d = nd[0] / nd[3]; v_d = nd[1] / nd[3]
        bx = (1.0 - u_d) * dw - 0.5
        by = v_d * dh - 0.5
        if not (0 <= bx <= dw - 1 and 0 <= by <= dh - 1): continue
        # Bilinear sample.
        x0 = int(np.floor(bx)); y0 = int(np.floor(by))
        x1 = min(x0 + 1, dw - 1); y1 = min(y0 + 1, dh - 1)
        fx = bx - x0; fy = by - y0
        d00 = depth[y0, x0]; d10 = depth[y0, x1]
        d01 = depth[y1, x0]; d11 = depth[y1, x1]
        z = ((d00 * (1 - fx) + d10 * fx) * (1 - fy)
             + (d01 * (1 - fx) + d11 * fx) * fy)
        if not np.isfinite(z) or z <= 0.05 or z > 8.0: continue

        # Back-project: at z = -z_metric (GL convention) the ray that
        # hits NDC (ndc_x, ndc_y) has cam-space (X, Y) determined by
        # ndc_x = -P[0,0]·X/Z - P[0,2]  ⇒  X = (ndc_x + P[0,2])·Z·(-1/P[0,0])
        ndc_x = 2.0 * u_v - 1.0
        ndc_y = 2.0 * v_v - 1.0
        Z_gl = -float(z)
        X = (ndc_x + P_proj[0, 2]) * Z_gl * (-1.0 / P_proj[0, 0])
        Y = (ndc_y + P_proj[1, 2]) * Z_gl * (-1.0 / P_proj[1, 1])
        cam_pt = np.array([X, Y, Z_gl, 1.0])
        world_pt = V_c2w @ cam_pt
        point_world[ti] = world_pt[:3].astype(np.float64)

    return V_w2c, point_world


# ────────────────────────── per-frame pose recovery (PnP) ──────────────────────────

def pose_pnp_for_frame(idx: int,
                        frame_d: dict,
                        tracks: list[list[tuple[int, float, float]]],
                        point_world: dict[int, np.ndarray],
                        min_pts: int, reproj_px: float
                        ) -> np.ndarray | None:
    """Try to recover V_w2c_gl for frame `idx` via PnP using already-
    triangulated tracks observed in this frame. Returns None if there
    aren't enough correspondences."""
    K = frame_d["K"]; cw = frame_d["cw"]; ch = frame_d["ch"]
    obj_pts: list[np.ndarray] = []
    img_pts: list[tuple[float, float]] = []
    for ti, obs_list in enumerate(tracks):
        if ti not in point_world: continue
        for fi, u_v, v_v in obs_list:
            if fi == idx:
                obj_pts.append(point_world[ti])
                img_pts.append(uv_to_pixel(u_v, v_v, cw, ch))
                break
    if len(obj_pts) < min_pts:
        return None
    obj_arr = np.asarray(obj_pts, dtype=np.float64)
    img_arr = np.asarray(img_pts, dtype=np.float64)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_arr.reshape(-1, 1, 3),
            img_arr.reshape(-1, 1, 2),
            K, distCoeffs=None,
            reprojectionError=reproj_px,
            iterationsCount=200,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        return None
    if not ok or inliers is None or len(inliers) < min_pts:
        return None
    R_oc, _ = cv2.Rodrigues(rvec)
    return Voc_to_Vgl(R_oc, tvec)


# ────────────────────────── (re)triangulation in new pose space ──────────────────────────

def triangulate_track(obs_list: list[tuple[int, float, float]],
                      poses_w2c: dict[int, np.ndarray],
                      frame_data: dict[int, dict]
                      ) -> np.ndarray | None:
    """Closed-form least-squares triangulation: each observation gives
    a ray (origin, direction) in world space; find the world point
    minimising sum of squared perpendicular distances to those rays.
    Returns None if fewer than 2 frames have known poses."""
    rays: list[tuple[np.ndarray, np.ndarray]] = []
    for fi, u_v, v_v in obs_list:
        if fi not in poses_w2c: continue
        d = frame_data[fi]
        V_w2c = poses_w2c[fi]
        V_c2w = np.linalg.inv(V_w2c)
        P_proj = d["P_proj"]
        # Cam-space ray dir at norm-view (u_v, v_v) — pick z=-1 so it
        # points forward, and back-solve for (X, Y) consistent with the
        # WebXR projection.
        ndc_x = 2.0 * u_v - 1.0
        ndc_y = 2.0 * v_v - 1.0
        X = (ndc_x + P_proj[0, 2]) * (-1.0) * (-1.0 / P_proj[0, 0])
        Y = (ndc_y + P_proj[1, 2]) * (-1.0) * (-1.0 / P_proj[1, 1])
        dir_cam = np.array([X, Y, -1.0])
        dir_cam /= np.linalg.norm(dir_cam)
        # Origin = camera centre in world.
        origin = V_c2w[:3, 3]
        dir_world = V_c2w[:3, :3] @ dir_cam
        rays.append((origin, dir_world))
    if len(rays) < 2: return None
    # Closed-form: sum_i (I − d_i d_iᵀ) X = sum_i (I − d_i d_iᵀ) o_i.
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in rays:
        proj = np.eye(3) - np.outer(d, d)
        A += proj
        b += proj @ o
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


# ────────────────────────── final BA ──────────────────────────

def run_ba(poses_w2c: dict[int, np.ndarray],
           point_world: dict[int, np.ndarray],
           tracks: list[list[tuple[int, float, float]]],
           frame_data: dict[int, dict],
           anchor_idx: int,
           huber: float, max_nfev: int,
           xtol: float, ftol: float, gtol: float
           ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Run the same residual machinery as feature_pose_align.py, but
    using poses we recovered ourselves as the WebXR-substitute init.
    The "correction" parameter is relative to those substitute poses;
    final V_w2c_new = T_corr · V_w2c_init.

    Returns (poses_w2c_refined, point_world_refined)."""
    obs_frame_ids: list[int] = []
    obs_uv: list[list[float]] = []
    obs_pt_idx: list[int] = []
    keep_pt_indices = sorted(point_world.keys())
    pt_remap = {ti: i for i, ti in enumerate(keep_pt_indices)}
    init_pts = np.stack([point_world[ti] for ti in keep_pt_indices])

    posed = {fi for fi in poses_w2c.keys()}
    for ti, obs_list in enumerate(tracks):
        if ti not in pt_remap: continue
        for fi, u_v, v_v in obs_list:
            if fi not in posed: continue
            obs_frame_ids.append(fi)
            obs_uv.append([u_v, v_v])
            obs_pt_idx.append(pt_remap[ti])

    obs_frame_ids = np.asarray(obs_frame_ids, dtype=np.int64)
    obs_uv = np.asarray(obs_uv, dtype=np.float64)
    obs_pt_idx = np.asarray(obs_pt_idx, dtype=np.int64)

    unique_frames = np.array(sorted(posed), dtype=np.int64)
    frame_to_local = {int(f): i for i, f in enumerate(unique_frames)}
    obs_frame_local = np.fromiter(
        (frame_to_local[int(f)] for f in obs_frame_ids),
        dtype=np.int64, count=len(obs_frame_ids),
    )

    anchor_local = int(frame_to_local.get(anchor_idx, 0))

    V_w2c_init = np.stack([poses_w2c[int(f)] for f in unique_frames])
    P_proj = np.stack([frame_data[int(f)]["P_proj"] for f in unique_frames])
    cw = np.array([frame_data[int(f)]["cw"] for f in unique_frames], dtype=np.float64)
    ch = np.array([frame_data[int(f)]["ch"] for f in unique_frames], dtype=np.float64)

    obs_cw = cw[obs_frame_local]
    obs_ch = ch[obs_frame_local]

    problem = {
        "n_frames": int(len(unique_frames)),
        "n_pts":    int(len(init_pts)),
        "anchor_local": anchor_local,
        "unique_frames": unique_frames,
        "frame_to_local": frame_to_local,
        "V_w2c": V_w2c_init,
        "P_proj": P_proj,
        "obs_frame": obs_frame_local,
        "obs_pt": obs_pt_idx,
        "obs_uv": obs_uv,
        "obs_cw": obs_cw,
        "obs_ch": obs_ch,
        "init_points": init_pts,
    }

    x0 = np.zeros((problem["n_frames"] - 1) * 6 + problem["n_pts"] * 3,
                   dtype=np.float64)
    x0[(problem["n_frames"] - 1) * 6:] = init_pts.reshape(-1)

    M = len(obs_frame_ids)
    r0 = fpa.compute_residuals(x0, problem)
    r0_pix = np.linalg.norm(r0.reshape(M, 2), axis=1)
    print(f"[ba-init] reprojection (px): "
          f"p50={np.percentile(r0_pix, 50):.2f} "
          f"p90={np.percentile(r0_pix, 90):.2f} "
          f"p99={np.percentile(r0_pix, 99):.2f} "
          f"max={r0_pix.max():.2f}")

    jac_sp = fpa.build_jac_sparsity(problem)
    loss = "huber" if huber > 0 else "linear"
    f_scale = max(huber, 1.0)
    print(f"[ba-solve] loss={loss} f_scale={f_scale} max_nfev={max_nfev}")
    t0 = time.time()
    result = least_squares(
        fpa.compute_residuals, x0,
        jac_sparsity=jac_sp,
        args=(problem,),
        method="trf",
        loss=loss, f_scale=f_scale,
        xtol=xtol, ftol=ftol, gtol=gtol,
        max_nfev=max_nfev,
        verbose=2,
    )
    print(f"[ba-solve] done in {time.time()-t0:.1f}s status={result.status} "
          f"nfev={result.nfev}")

    r1 = fpa.compute_residuals(result.x, problem)
    r1_pix = np.linalg.norm(r1.reshape(M, 2), axis=1)
    print(f"[ba-done] reprojection (px): "
          f"p50={np.percentile(r1_pix, 50):.2f} "
          f"p90={np.percentile(r1_pix, 90):.2f} "
          f"p99={np.percentile(r1_pix, 99):.2f} "
          f"max={r1_pix.max():.2f}")

    omegas, ts, points = fpa.unpack_x(result.x, problem)

    # Compose corrections back onto the init poses to get final V_w2c.
    poses_refined: dict[int, np.ndarray] = {}
    for li, fi in enumerate(unique_frames):
        omega = omegas[li]; t = ts[li]
        R = (np.eye(3) if not np.any(omega)
             else cv2.Rodrigues(omega)[0])
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        # T acts on V_c2w (correction in world frame). So:
        #   V_c2w_new = T · V_c2w_init
        #   V_w2c_new = V_w2c_init · T⁻¹
        poses_refined[int(fi)] = np.linalg.inv(T @ np.linalg.inv(V_w2c_init[li]))

    points_refined = {int(ti): points[pt_remap[ti]] for ti in pt_remap}
    return poses_refined, points_refined


# ────────────────────────── write corrected frames ──────────────────────────

def write_frames(poses_w2c: dict[int, np.ndarray],
                  src_dir: Path, dst_dir: Path,
                  fall_back_orig: bool) -> tuple[int, int]:
    """Splice new view matrices into copies of the source .bin files.
    Returns (n_corrected, n_passthrough)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_cor = 0; n_pass = 0
    for src_path in sorted(src_dir.glob("frame_*.bin")):
        m = _FRAME_BIN_RE.match(src_path.name)
        if not m: continue
        idx = int(m.group(1))
        body = src_path.read_bytes()
        if idx in poses_w2c:
            V_c2w_new = np.linalg.inv(poses_w2c[idx])
            view_new = V_c2w_new.reshape(-1, order="F").astype(np.float32).tolist()
            out = bytearray(body)
            struct.pack_into("<16f", out, 0, *view_new)
            (dst_dir / src_path.name).write_bytes(bytes(out))
            n_cor += 1
        elif fall_back_orig:
            (dst_dir / src_path.name).write_bytes(body)
            n_pass += 1
    return n_cor, n_pass


# ────────────────────────── driver ──────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--session", required=True)
    ap.add_argument("--frames-variant", default="frames")
    ap.add_argument("--dst-variant", default="frames_pose_scratch")
    ap.add_argument("--bootstrap", choices=("structure", "phone"),
                    default="structure",
                    help="'structure' (default): seed the world map with "
                         "WebXR-triangulated 3D points from features_meta, "
                         "then PnP every frame fresh against that map "
                         "(per-frame poses are NOT taken from WebXR). "
                         "'phone': bootstrap a sparse seed map from the "
                         "anchor frame's phone depth and chain via "
                         "incremental PnP — touches WebXR less but "
                         "recovers far fewer frames in practice.")
    ap.add_argument("--no-anchor-webxr", action="store_true",
                    help="Phone-bootstrap mode only: place frame 0 at "
                         "identity instead of WebXR's view matrix. The "
                         "cloud will live in its own coordinate system, "
                         "not aligned to the existing voxel bbox.")
    ap.add_argument("--max-features", type=int, default=None)
    ap.add_argument("--pnp-min-pts", type=int, default=8)
    ap.add_argument("--pnp-reproj-px", type=float, default=3.0)
    ap.add_argument("--max-nfev", type=int, default=400)
    ap.add_argument("--huber", type=float, default=3.0)
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

    meta_path = sess_dir / fpa._features_meta_filename(args.frames_variant)
    if not meta_path.exists():
        sys.exit(f"{meta_path.name} not found in {sess_dir}.\n"
                 f"  Run first: python tools/feature_ray_reconstruct.py "
                 f"--session {args.session}")

    print(f"[load] frames from {frames_dir}")
    frame_data = load_session_frames(frames_dir)
    print(f"  {len(frame_data)} frames decoded")

    print(f"[load] {meta_path.name}")
    meta = json.loads(meta_path.read_text())
    tracks, init_pts = build_tracks(meta, frame_data, args.max_features)
    n_obs = sum(len(t) for t in tracks)
    print(f"  tracks={len(tracks):,}  observations={n_obs:,}")
    if not tracks:
        sys.exit("no tracks survive — nothing to triangulate.")

    # Anchor on the first frame that's present + has observations.
    obs_per_frame: dict[int, int] = {}
    for t in tracks:
        for fi, _, _ in t:
            obs_per_frame[fi] = obs_per_frame.get(fi, 0) + 1
    anchor_idx = min(obs_per_frame.keys())

    if args.bootstrap == "structure":
        # Seed the world map directly with WebXR-triangulated 3D
        # points. PnP every frame independently against that map.
        point_world: dict[int, np.ndarray] = {
            i: pt for i, pt in enumerate(init_pts)
        }
        print(f"[bootstrap=structure] seeded {len(point_world)} 3D points "
              f"from features_meta.json (WebXR-triangulated)")
        poses_w2c: dict[int, np.ndarray] = {}
        sorted_frames = sorted(frame_data.keys())
        failed: list[int] = []
        for fi in sorted_frames:
            V = pose_pnp_for_frame(fi, frame_data[fi], tracks, point_world,
                                    args.pnp_min_pts, args.pnp_reproj_px)
            if V is not None:
                poses_w2c[fi] = V
            else:
                failed.append(fi)
        n_posed = len(poses_w2c)
        print(f"[sfm-pnp-only] poses recovered: {n_posed}/{len(frame_data)}  "
              f"failed PnP: {len(failed)}")
        if failed[:5]:
            print(f"      first failed frames: {failed[:5]}")
        # Pick anchor to be the lowest frame that recovered.
        if anchor_idx not in poses_w2c and poses_w2c:
            anchor_idx = min(poses_w2c.keys())
        # Re-triangulate every track using the new poses, so BA
        # initial structure isn't anchored to WebXR's poses.
        new_pts: dict[int, np.ndarray] = {}
        for ti, obs_list in enumerate(tracks):
            pt = triangulate_track(obs_list, poses_w2c, frame_data)
            if pt is not None and np.all(np.isfinite(pt)):
                new_pts[ti] = pt
        point_world = new_pts
        print(f"[retriangulate] {len(point_world)} tracks survived in new poses")
    else:
        # Phone-depth bootstrap: incremental SfM from one anchor frame.
        print(f"[bootstrap=phone] anchor=frame {anchor_idx}  "
              f"({obs_per_frame[anchor_idx]} obs)  "
              f"webxr={'yes' if not args.no_anchor_webxr else 'no'}")
        V_w2c_anchor, point_world = bootstrap_anchor(
            anchor_idx, frame_data[anchor_idx], tracks,
            anchor_webxr=not args.no_anchor_webxr,
        )
        print(f"[anchor] seeded {len(point_world)} 3D points from phone depth")
        poses_w2c = {anchor_idx: V_w2c_anchor}
        sorted_frames = sorted(frame_data.keys())
        failed = []
        for fi in sorted_frames:
            if fi == anchor_idx: continue
            V = pose_pnp_for_frame(fi, frame_data[fi], tracks, point_world,
                                    args.pnp_min_pts, args.pnp_reproj_px)
            if V is not None:
                poses_w2c[fi] = V
                new_tris = 0
                for ti, obs_list in enumerate(tracks):
                    if ti in point_world: continue
                    if not any(o[0] in poses_w2c for o in obs_list): continue
                    pt = triangulate_track(obs_list, poses_w2c, frame_data)
                    if pt is not None and np.all(np.isfinite(pt)):
                        point_world[ti] = pt
                        new_tris += 1
                if fi % 25 == 0 or new_tris > 0:
                    print(f"  [pnp] frame {fi:4d} → ok  "
                          f"map={len(point_world):,} (+{new_tris})")
            else:
                failed.append(fi)
        n_posed = len(poses_w2c)
        print(f"[sfm-incremental] poses recovered: {n_posed}/{len(frame_data)}  "
              f"failed PnP: {len(failed)}")

    # Final BA over all (poses, points).
    print(f"[ba] running BA on {n_posed} poses + {len(point_world)} points")
    poses_refined, _ = run_ba(
        poses_w2c, point_world, tracks, frame_data,
        anchor_idx=anchor_idx,
        huber=args.huber, max_nfev=args.max_nfev,
        xtol=args.xtol, ftol=args.ftol, gtol=args.gtol,
    )

    # Diagnostic: how far our poses ended up from WebXR's.
    diffs_t = []; diffs_R = []
    for fi, V in poses_refined.items():
        V_orig = frame_data[fi]["V_w2c_orig"]
        delta = V_orig @ np.linalg.inv(V)        # what would map ours → WebXR's
        # rotation magnitude, translation magnitude
        R = delta[:3, :3]; t = delta[:3, 3]
        ang = np.degrees(np.arccos(
            np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
        ))
        diffs_t.append(np.linalg.norm(t))
        diffs_R.append(ang)
    diffs_t = np.asarray(diffs_t); diffs_R = np.asarray(diffs_R)
    print(f"[diff vs WebXR] |t|: p50={np.median(diffs_t)*100:.1f}cm "
          f"p90={np.percentile(diffs_t,90)*100:.1f}cm "
          f"p99={np.percentile(diffs_t,99)*100:.1f}cm "
          f"max={diffs_t.max()*100:.1f}cm")
    print(f"[diff vs WebXR] |ω|: p50={np.median(diffs_R):.2f}° "
          f"p90={np.percentile(diffs_R,90):.2f}° "
          f"p99={np.percentile(diffs_R,99):.2f}° "
          f"max={diffs_R.max():.2f}°")

    dst_dir = sess_dir / args.dst_variant
    print(f"[write] corrected frames → {dst_dir}")
    n_cor, n_pass = write_frames(poses_refined, frames_dir, dst_dir,
                                  fall_back_orig=True)
    print(f"  wrote {n_cor} corrected + {n_pass} passthrough = "
          f"{n_cor + n_pass} frames")
    print(f"\nNext step (voxelize with the new poses):\n"
          f"  ./tools/voxel_reverse_rust/target/release/voxel-reverse \\\n"
          f"      --session {args.session} \\\n"
          f"      --frames-dir captured_frames/{args.session}/{args.dst_variant} \\\n"
          f"      --out captured_frames/{args.session}/voxels_pose_scratch.json \\\n"
          f"      --voxel-size 0.02 --tol 0.04")


if __name__ == "__main__":
    main()
