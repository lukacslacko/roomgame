#!/usr/bin/env python3
"""
Classical feature-based offline reconstruction.

Pipeline:
  1. Extract ORB features from every captured frame's RGB image.
  2. For every pair of frames (within --window), Lowe-ratio match the
     descriptors. Each surviving match unions the two keypoints in a
     UnionFind, building "tracks" — sets of keypoints across frames that
     refer to the same physical point.
  3. For each track of length ≥ --min-views, triangulate the world-space
     point that minimises the sum-of-squared perpendicular distances to
     the contributing rays (the closed-form least-squares solution).
  4. Keep tracks whose mean ray residual is below --max-residual; emit
     one small coloured cube per track at the triangulated position
     (median of the contributing observations' RGB).

Output: web/out/features.json — list of (position, colour) entries
consumed by web/featureview.html.

Camera poses, intrinsics, and colour pixels are already on hand from
the captured frames; we don't run bundle adjustment because ARCore's
pose error has been measured at <10 cm over 15 m of walking, well below
the cube-rendering scale.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import fusion
import serve

import cv2  # noqa: E402  — import after path tweak so we know it's available


# ---------------------------------------------------------------------------
# Per-frame work: feature extraction + ray geometry. Picklable for workers.
# ---------------------------------------------------------------------------

def _frame_features(frame_path: str, n_features: int, downscale: float):
    """Decode one frame body, run ORB, and return per-keypoint world-space
    rays + RGB colours. None if the frame has no colour or no descriptors."""
    body = Path(frame_path).read_bytes()
    try:
        frame = serve.parse_frame(body)
    except Exception:
        return None
    if frame["color"] is None:
        return None

    cw = int(frame["color_width"])
    ch = int(frame["color_height"])
    if cw == 0 or ch == 0:
        return None

    color_arr = np.frombuffer(frame["color"], dtype=np.uint8).reshape(ch, cw, 4)[..., :3]
    # OpenCV doesn't care about top-down vs bottom-up — we just need to be
    # consistent with how we map keypoint pixels back to GL view UV later.
    # Keep the buffer's GL bottom-up storage so kp.pt.y maps directly to v.

    if downscale != 1.0:
        new_w = max(1, int(round(cw * downscale)))
        new_h = max(1, int(round(ch * downscale)))
        cv_img = cv2.resize(color_arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        cv_img = color_arr.copy()
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

    detector = cv2.ORB_create(nfeatures=n_features)
    kp, des = detector.detectAndCompute(gray, None)
    if des is None or len(kp) == 0:
        return None

    # Map keypoint coords from the (possibly downscaled) detection image
    # back into the original camera-buffer pixel grid for ray geometry,
    # but read colour from the *full-res* buffer for fidelity.
    scale_x = cw / cv_img.shape[1]
    scale_y = ch / cv_img.shape[0]
    pts = np.array([kp_.pt for kp_ in kp], dtype=np.float64)  # (N, 2) in cv_img coords
    pts_full = pts * np.array([[scale_x, scale_y]])

    # GL view UV ∈ [0, 1]² (bottom-up). Our buffer is stored bottom-up, so
    # u = x / cw, v = y / ch with no flip.
    u = pts_full[:, 0] / cw
    v = pts_full[:, 1] / ch

    V = fusion._mat4_from_column_major(frame["viewMatrix"])
    P = fusion._mat4_from_column_major(frame["projectionMatrix"])
    P_inv = np.linalg.inv(P)
    cam_origin = V[:3, 3].astype(np.float64)

    x_ndc = 2.0 * u - 1.0
    y_ndc = 2.0 * v - 1.0
    clip = np.stack(
        [x_ndc, y_ndc, -np.ones_like(x_ndc), np.ones_like(x_ndc)], axis=-1,
    )
    view4 = clip @ P_inv.T
    view3 = view4[:, :3] / np.where(np.abs(view4[:, 3:4]) < 1e-12, 1.0, view4[:, 3:4])
    view3_h = np.concatenate([view3, np.ones((view3.shape[0], 1))], axis=1)
    world_h = view3_h @ V.T
    world3 = world_h[:, :3] / np.where(np.abs(world_h[:, 3:4]) < 1e-12, 1.0, world_h[:, 3:4])
    dirs = world3 - cam_origin[None, :]
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.maximum(norms, 1e-9)

    # Sample colours at the (full-res, integer-rounded) pixel.
    cx = np.clip(np.floor(pts_full[:, 0]).astype(np.int32), 0, cw - 1)
    cy = np.clip(np.floor(pts_full[:, 1]).astype(np.int32), 0, ch - 1)
    colours = color_arr[cy, cx, :3].copy()

    # Effective focal length in pixels for the colour buffer. For an OpenGL
    # symmetric-frustum projection, P[0,0] = (cw / 2) / f_x in NDC units,
    # i.e. f_x = (cw / 2) * P[0,0]. We use the average of f_x and f_y so
    # downstream code can model 1-pixel-of-noise as ≈ (1 / focal_pix) rad
    # of angular ray noise — that's what powers the per-feature sensitivity
    # score below.
    focal_pix = float(0.5 * (abs(P[0, 0]) * cw + abs(P[1, 1]) * ch) * 0.5)

    return {
        "des": des,                                  # (N, 32) uint8
        "rays_dirs": dirs.astype(np.float32),        # (N, 3)
        "cam_origin": cam_origin.astype(np.float32), # (3,)
        "colours": colours,                          # (N, 3) uint8
        "kp_uv": np.stack([u, v], axis=-1).astype(np.float32),  # (N, 2)
        "kp_count": len(kp),
        "focal_pix": focal_pix,
    }


# ---------------------------------------------------------------------------
# Pairwise matcher worker — descriptors only, runs in a subprocess.
# ---------------------------------------------------------------------------

def _match_pairs(args):
    """args = (i, des_i, list_of (j, des_j), ratio).
    Returns list of (i, j, kp_a_array, kp_b_array). The arrays are the
    Lowe-ratio-passing match indices into frames[i] and frames[j]."""
    i, des_i, others, ratio = args
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    out = []
    for j, des_j in others:
        if des_i is None or des_j is None:
            continue
        try:
            knn = matcher.knnMatch(des_i, des_j, k=2)
        except cv2.error:
            continue
        a_idx = []
        b_idx = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                a_idx.append(m.queryIdx)
                b_idx.append(m.trainIdx)
        if a_idx:
            out.append((i, j, np.array(a_idx, dtype=np.int32),
                              np.array(b_idx, dtype=np.int32)))
    return out


# ---------------------------------------------------------------------------
# UnionFind on a flat node array.
# ---------------------------------------------------------------------------

def _uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _uf_union(parent, x, y):
    rx = _uf_find(parent, x)
    ry = _uf_find(parent, y)
    if rx != ry:
        parent[rx] = ry


# ---------------------------------------------------------------------------
# Multi-view ray triangulation: closed-form least-squares closest point.
# ---------------------------------------------------------------------------

def triangulate_rays(origins: np.ndarray, dirs: np.ndarray):
    """origins (N, 3), dirs (N, 3, unit). Returns (P, residuals, A, M).

    Each ray contributes M_i = I - d_i d_i^T (project perpendicular to the
    ray). The point closest to all rays in the L² sense satisfies
    (Σ M_i) P = Σ M_i o_i. We return A and M alongside P so the caller can
    plug them into the analytical 1-pixel-sensitivity formula without
    re-deriving the same outer products.
    """
    M = np.eye(3)[None, :, :] - dirs[:, :, None] * dirs[:, None, :]  # (N, 3, 3)
    A = M.sum(axis=0)
    b = np.einsum("nij,nj->ni", M, origins).sum(axis=0)
    try:
        P = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, None, None, None
    diffs = P[None, :] - origins
    proj_lens = (diffs * dirs).sum(axis=1)
    perps = diffs - proj_lens[:, None] * dirs
    residuals = np.linalg.norm(perps, axis=1)
    return P, residuals, A, M


def pixel_sensitivity_m(P: np.ndarray, A: np.ndarray, M: np.ndarray,
                        origins: np.ndarray, dirs: np.ndarray,
                        focal_pixels: np.ndarray) -> float:
    """RMS 3D displacement (meters) of the triangulated point P under
    independent isotropic 1-pixel observation noise on every contributing
    ray.

    Derivation. Each ray's keypoint-shift δθᵢ ≈ 1/fᵢ rad rotates dᵢ
    perpendicular to itself; to first order this shifts P by
        δP ≈ A⁻¹ · tᵢ · δdᵢ
    where tᵢ = dᵢᵀ(P − oᵢ) is the depth along ray i and δdᵢ ⊥ dᵢ. With iid
    isotropic noise the position covariance is
        Σ_P = A⁻¹ · ( Σᵢ (tᵢ/fᵢ)² · Mᵢ ) · A⁻¹.
    The scalar √trace(Σ_P) collapses that to a single number with units of
    meters per 1-px-of-noise — high score = ill-conditioned, e.g. all rays
    near-parallel because the camera mostly rotated between observations.
    """
    diffs = P[None, :] - origins
    ts = (diffs * dirs).sum(axis=1)             # (N,) depth along each ray
    alphas = 1.0 / np.maximum(focal_pixels, 1e-9)
    weights = (alphas * ts) ** 2                # (N,)
    B = (weights[:, None, None] * M).sum(axis=0)
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return float("inf")
    Sigma = A_inv @ B @ A_inv
    tr = float(np.trace(Sigma))
    return float(np.sqrt(tr)) if tr > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Main reconstruction.
# ---------------------------------------------------------------------------

def reconstruct(frames_dir: Path, out_path: Path, *,
                n_features: int = 1500,
                downscale: float = 0.5,
                ratio: float = 0.75,
                window: int = 0,
                min_views: int = 3,
                max_residual_m: float = 0.05,
                max_sensitivity_m: float = float("inf"),
                min_depth_m: float = 0.0,
                world_min: tuple[float, float, float] = (-2.5, -0.3, -2.5),
                world_max: tuple[float, float, float] = ( 2.5,  4.7,  2.5),
                cube_size: float = 0.03,
                workers: int = 1,
                max_frames: int | None = None,
                voxel_out_path: Path | None = None,
                meta_out_path: Path | None = None,
                voxel_size: float | None = None,
                grid_shape: tuple[int, int, int] | None = None) -> None:
    frame_paths = sorted(frames_dir.glob("frame_*.bin"))
    if max_frames:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        print(f"No frames found in {frames_dir}")
        return
    print(f"Found {len(frame_paths)} frames in {frames_dir}")

    # ----- pass 1: extract features from every frame --------------------
    t0 = time.time()
    features = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_frame_features, str(p), n_features, downscale): i
                       for i, p in enumerate(frame_paths)}
            results = [None] * len(frame_paths)
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        features = results
    else:
        for i, p in enumerate(frame_paths):
            features.append(_frame_features(str(p), n_features, downscale))
            if (i + 1) % 25 == 0 or i == len(frame_paths) - 1:
                print(f"  features {i+1}/{len(frame_paths)}")
    print(f"Feature extraction: {time.time()-t0:.1f} s")

    # Drop frames that yielded no features.
    valid_idxs = [i for i, f in enumerate(features) if f is not None]
    if not valid_idxs:
        print("No frames yielded features.")
        return
    print(f"  {len(valid_idxs)} of {len(features)} frames have descriptors")

    # ----- pass 2: pairwise matching + UnionFind ------------------------
    # Flat node ids: feature k of frame i has node `node_offset[i] + k`.
    kp_counts = np.array([(f["kp_count"] if f is not None else 0)
                          for f in features], dtype=np.int64)
    node_offset = np.concatenate([[0], np.cumsum(kp_counts)])
    n_total = int(node_offset[-1])
    parent = np.arange(n_total, dtype=np.int64)
    print(f"  total keypoints across frames: {n_total:,}")

    # Build pair list (within sliding window if set; else all pairs).
    pairs = []
    F = len(features)
    win = window if window > 0 else F
    for i in valid_idxs:
        for j in valid_idxs:
            if j <= i:
                continue
            if j - i > win:
                continue
            pairs.append((i, j))
    print(f"  matching {len(pairs):,} frame pairs (window={win})")

    # Group pairs by i so each worker handles one source frame against many
    # targets — saves the cost of re-shipping descriptors to each call.
    grouped: dict[int, list[tuple[int, np.ndarray]]] = {}
    for i, j in pairs:
        grouped.setdefault(i, []).append((j, features[j]["des"]))
    args_list = [(i, features[i]["des"], grouped[i], ratio) for i in grouped]

    t0 = time.time()
    n_match = 0
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for batch in ex.map(_match_pairs, args_list):
                for i, j, ai, bi in batch:
                    qoff = int(node_offset[i])
                    toff = int(node_offset[j])
                    for a, b in zip(ai, bi):
                        _uf_union(parent, qoff + a, toff + b)
                    n_match += len(ai)
    else:
        for args in args_list:
            for i, j, ai, bi in _match_pairs(args):
                qoff = int(node_offset[i])
                toff = int(node_offset[j])
                for a, b in zip(ai, bi):
                    _uf_union(parent, qoff + a, toff + b)
                n_match += len(ai)
    print(f"  {n_match:,} pairwise matches kept; matching took {time.time()-t0:.1f} s")

    # ----- pass 3: collect tracks (root → list of node ids) -------------
    roots = np.array([_uf_find(parent, k) for k in range(n_total)])
    tracks: dict[int, list[int]] = {}
    for k in range(n_total):
        tracks.setdefault(int(roots[k]), []).append(k)

    candidate_tracks = [t for t in tracks.values() if len(t) >= min_views]
    print(f"  {len(candidate_tracks):,} tracks with ≥{min_views} views")

    # Pre-compute a node→(frame, local_idx) map for triangulation.
    node_to_frame = np.empty(n_total, dtype=np.int32)
    node_to_local = np.empty(n_total, dtype=np.int32)
    for i, n in enumerate(kp_counts):
        if n > 0:
            sl = slice(int(node_offset[i]), int(node_offset[i] + n))
            node_to_frame[sl] = i
            node_to_local[sl] = np.arange(n, dtype=np.int32)

    wmin = np.asarray(world_min, dtype=np.float64)
    wmax = np.asarray(world_max, dtype=np.float64)

    # Map each frame's array index back to the integer parsed from its
    # filename (`frame_NNNNNN.bin`) — that's the `frame_idx` the manifest
    # exposes and that the voxelview sends back when the user clicks a
    # frame thumbnail. Falls back to the array index if the filename
    # doesn't follow the convention.
    frame_idx_for_array_pos: list[int] = []
    for p in frame_paths:
        m = re.match(r"frame_(\d+)\.bin", Path(p).name)
        frame_idx_for_array_pos.append(int(m.group(1)) if m else len(frame_idx_for_array_pos))

    out_pos: list[list[float]] = []
    out_col: list[list[int]] = []
    out_views: list[int] = []
    out_residual: list[float] = []
    out_sensitivity: list[float] = []        # meters of P-shift per 1-px iid keypoint noise
    out_frame_idxs: list[list[int]] = []     # parallel to out_pos: frame ids the track was seen in
    out_obs: list[list[tuple[int, float, float]]] = []  # parallel: (frame_idx, u, v) per observation
    n_kept = 0
    n_rejected_residual = 0
    n_rejected_sensitivity = 0
    n_rejected_oob = 0
    n_rejected_behind = 0
    n_rejected_singleframe = 0

    for members in candidate_tracks:
        # Don't count multiple keypoints in the same frame as separate views.
        seen_frames: dict[int, int] = {}
        for m in members:
            seen_frames.setdefault(int(node_to_frame[m]), m)
        if len(seen_frames) < min_views:
            n_rejected_singleframe += 1
            continue

        origins = []
        dirs = []
        cols = []
        focals = []
        track_obs: list[tuple[int, float, float]] = []
        for fi, node in seen_frames.items():
            f = features[fi]
            li = int(node_to_local[node])
            origins.append(f["cam_origin"])
            dirs.append(f["rays_dirs"][li])
            cols.append(f["colours"][li])
            focals.append(float(f["focal_pix"]))
            uv = f["kp_uv"][li]
            track_obs.append((int(frame_idx_for_array_pos[fi]),
                              float(uv[0]), float(uv[1])))
        origins = np.asarray(origins, dtype=np.float64)
        dirs = np.asarray(dirs, dtype=np.float64)
        cols = np.asarray(cols, dtype=np.uint8)
        focals = np.asarray(focals, dtype=np.float64)

        P, residuals, A_mat, M_stack = triangulate_rays(origins, dirs)
        if P is None:
            continue
        if (P < wmin).any() or (P > wmax).any():
            n_rejected_oob += 1
            continue
        # Cheirality: every contributing observation must see P in front of
        # its camera. Spurious matches (e.g. two unrelated keypoints in
        # repeated textures) frequently triangulate to a point whose
        # least-squares fit lands behind one of the views — geometrically
        # impossible for a real feature, so we drop those whole tracks.
        depths = ((P[None, :] - origins) * dirs).sum(axis=1)
        if (depths < min_depth_m).any():
            n_rejected_behind += 1
            continue
        mean_res = float(residuals.mean())
        if mean_res > max_residual_m:
            n_rejected_residual += 1
            continue
        sensitivity = pixel_sensitivity_m(P, A_mat, M_stack, origins, dirs, focals)
        if sensitivity > max_sensitivity_m:
            n_rejected_sensitivity += 1
            continue

        out_pos.append([float(P[0]), float(P[1]), float(P[2])])
        out_col.append([int(c) for c in np.median(cols, axis=0)])
        out_views.append(int(len(seen_frames)))
        out_residual.append(mean_res)
        out_sensitivity.append(sensitivity)
        out_frame_idxs.append(sorted(int(frame_idx_for_array_pos[fi]) for fi in seen_frames))
        out_obs.append(sorted(track_obs))
        n_kept += 1

    print(f"  kept {n_kept:,}; rejected {n_rejected_residual:,} (residual) "
          f"+ {n_rejected_sensitivity:,} (1-px sensitivity > {max_sensitivity_m} m) "
          f"+ {n_rejected_behind:,} (point behind a camera, depth < {min_depth_m} m) "
          f"+ {n_rejected_oob:,} (out of bbox) "
          f"+ {n_rejected_singleframe:,} (single-frame track)")
    if n_kept:
        s_arr = np.asarray(out_sensitivity, dtype=np.float64)
        print(f"  sensitivity (m / 1px): "
              f"p50={np.percentile(s_arr, 50):.4f} "
              f"p90={np.percentile(s_arr, 90):.4f} "
              f"p99={np.percentile(s_arr, 99):.4f} "
              f"max={s_arr.max():.4f}")

    if n_kept == 0:
        print("Nothing to write.")
        return

    payload = {
        "cube_size": cube_size,
        "n_features": n_kept,
        "world_min": wmin.tolist(),
        "world_max": wmax.tolist(),
        "min_views": min_views,
        "max_residual": max_residual_m,
        "max_sensitivity": (max_sensitivity_m
                             if max_sensitivity_m != float("inf") else None),
        "positions": out_pos,
        "colors": out_col,
        "n_views": out_views,
        "residuals": out_residual,
        "sensitivities": out_sensitivity,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    print(f"Wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")

    # ----- voxel-binned output (for the voxelview's "features" variant) ----
    if voxel_out_path is not None:
        if voxel_size is None or voxel_size <= 0:
            raise ValueError("voxel_size must be a positive float for voxel output")
        # Default the grid shape to whatever (world_max - world_min) / voxel_size
        # rounds to. Caller may override (e.g. to match an existing variant on
        # disk) so the voxel grids overlay 1:1 in the viewer.
        if grid_shape is None:
            grid_shape = tuple(
                int(np.ceil((wmax[i] - wmin[i]) / voxel_size)) for i in range(3)
            )
        Nx, Ny, Nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
        # Bin each kept feature into its (ix, iy, iz). Multiple features in
        # the same voxel collapse into one entry (mean colour, union of
        # frame ids, summed feature count).
        positions = np.asarray(out_pos, dtype=np.float64)
        colors_arr = np.asarray(out_col, dtype=np.float64)
        ix = np.floor((positions[:, 0] - wmin[0]) / voxel_size).astype(np.int64)
        iy = np.floor((positions[:, 1] - wmin[1]) / voxel_size).astype(np.int64)
        iz = np.floor((positions[:, 2] - wmin[2]) / voxel_size).astype(np.int64)
        in_grid = (
            (ix >= 0) & (ix < Nx)
            & (iy >= 0) & (iy < Ny)
            & (iz >= 0) & (iz < Nz)
        )
        per_voxel: dict[tuple[int, int, int], dict] = {}
        for k in np.flatnonzero(in_grid):
            key = (int(ix[k]), int(iy[k]), int(iz[k]))
            entry = per_voxel.get(key)
            if entry is None:
                entry = {
                    "color_sum": np.zeros(3, dtype=np.float64),
                    "n": 0,
                    "frames": set(),
                    "residual_sum": 0.0,
                    "sensitivity_min": float("inf"),
                    "features": [],
                }
                per_voxel[key] = entry
            entry["color_sum"] += colors_arr[k]
            entry["n"] += 1
            entry["frames"].update(out_frame_idxs[k])
            entry["residual_sum"] += float(out_residual[k])
            sens_k = float(out_sensitivity[k])
            if sens_k < entry["sensitivity_min"]:
                entry["sensitivity_min"] = sens_k
            entry["features"].append({
                "world": [float(positions[k, 0]),
                           float(positions[k, 1]),
                           float(positions[k, 2])],
                "n_views": int(out_views[k]),
                "residual_mean": float(out_residual[k]),
                "sensitivity": sens_k,
                "obs": [
                    {"frame": int(fi), "u": float(uu), "v": float(vv)}
                    for (fi, uu, vv) in out_obs[k]
                ],
            })

        v_indices: list[list[int]] = []
        v_colors: list[list[int]] = []
        v_meta: list[dict] = []
        for key in sorted(per_voxel.keys()):
            e = per_voxel[key]
            avg = (e["color_sum"] / max(e["n"], 1)).round().clip(0, 255).astype(int)
            v_indices.append([key[0], key[1], key[2]])
            v_colors.append([int(avg[0]), int(avg[1]), int(avg[2])])
            # The voxel's "sensitivity" is the *best* (smallest) sensitivity
            # across the features that fell inside it: if any single
            # contributing track was well-conditioned, the voxel position is
            # reliable even when other contributors are noisy.
            v_meta.append({
                "idx": [key[0], key[1], key[2]],
                "frames": sorted(e["frames"]),
                "n_features": int(e["n"]),
                "residual_mean": float(e["residual_sum"] / max(e["n"], 1)),
                "sensitivity_min": float(e["sensitivity_min"]),
                "features": e["features"],
            })

        wmax_eff = (wmin + np.array([Nx, Ny, Nz]) * voxel_size).tolist()
        voxels_payload = {
            "voxel_size": float(voxel_size),
            "world_min": wmin.tolist(),
            "world_max": wmax_eff,
            "shape": [Nx, Ny, Nz],
            "n_voxels": len(v_indices),
            "indices": v_indices,
            "colors": v_colors,
            "source": "feature_ray_reconstruct",
            "min_views": int(min_views),
            "max_residual": float(max_residual_m),
        }
        voxel_out_path.parent.mkdir(parents=True, exist_ok=True)
        voxel_out_path.write_text(json.dumps(voxels_payload))
        print(f"Wrote {voxel_out_path}  ({voxel_out_path.stat().st_size/1e6:.1f} MB)  "
              f"({len(v_indices)} occupied voxels from {n_kept} features)")

        if meta_out_path is not None:
            meta_payload = {
                "voxel_size": float(voxel_size),
                "world_min": wmin.tolist(),
                "world_max": wmax_eff,
                "shape": [Nx, Ny, Nz],
                "n_voxels": len(v_meta),
                "voxels": v_meta,
            }
            meta_out_path.parent.mkdir(parents=True, exist_ok=True)
            meta_out_path.write_text(json.dumps(meta_payload))
            print(f"Wrote {meta_out_path}  ({meta_out_path.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--frames-dir", default="captured_frames")
    ap.add_argument("--out", default="web/out/features.json")
    ap.add_argument("--n-features", type=int, default=1500,
                    help="ORB nfeatures per frame")
    ap.add_argument("--downscale", type=float, default=0.5,
                    help="run ORB on the camera image scaled by this factor "
                         "(1.0 = native res; 0.5 ≈ 4× faster)")
    ap.add_argument("--ratio", type=float, default=0.75,
                    help="Lowe ratio test cutoff")
    ap.add_argument("--window", type=int, default=0,
                    help="match each frame only against its next N frames "
                         "(0 = all-pairs). Limit if ARCore drift is high or "
                         "you need a faster run.")
    ap.add_argument("--min-views", type=int, default=3)
    ap.add_argument("--max-residual", type=float, default=0.05,
                    help="reject tracks whose mean perpendicular ray-distance "
                         "to the triangulated point exceeds this (m)")
    ap.add_argument("--max-sensitivity", type=float, default=float("inf"),
                    help="reject tracks whose 1-pixel-noise sensitivity "
                         "(σ_1px = √trace(A⁻¹ Σᵢ (tᵢ/fᵢ)² Mᵢ A⁻¹)) exceeds "
                         "this many meters. Filters out features that come "
                         "from frames where the camera mostly rotated, since "
                         "near-parallel rays leave the triangulated depth "
                         "underdetermined. Try 0.05–0.10 m for the kind of "
                         "floating mid-air voxels you see in handheld scans.")
    ap.add_argument("--min-depth", type=float, default=0.0,
                    help="reject a track if the triangulated point sits "
                         "closer than this (m) along any contributing ray "
                         "— including behind the camera. The default 0.0 "
                         "enforces strict cheirality (in front of every "
                         "camera); raise to e.g. 0.05 if you also want to "
                         "discard near-clipping triangulations.")
    ap.add_argument("--world-min", type=float, nargs=3, default=[-2.5, -0.3, -2.5])
    ap.add_argument("--world-max", type=float, nargs=3, default=[ 2.5,  4.7,  2.5])
    ap.add_argument("--cube-size", type=float, default=0.03,
                    help="size of the rendered marker cube (m)")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--session", default=None,
                    help="If set, source frames from "
                         "captured_frames/<session>/frames/ and additionally "
                         "write voxels_features.json + features_meta.json into "
                         "captured_frames/<session>/ — these power the "
                         "voxelview's 'features' variant and per-frame voxel "
                         "highlight overlay.")
    ap.add_argument("--voxel-size", type=float, default=None,
                    help="Voxel grid edge (m) for voxels_features.json. "
                         "Defaults to whatever the reference voxels_<variant>.json "
                         "in the session uses; falls back to 0.05.")
    ap.add_argument("--reference-variant", default=None,
                    help="Take voxel_size, world_min, shape from "
                         "captured_frames/<session>/voxels_<variant>.json. "
                         "Defaults to the best available "
                         "(refined_aligned > aligned > refined > original).")
    args = ap.parse_args()

    # Resolve --session into concrete paths and (optionally) inherit grid
    # parameters from a reference voxel JSON so the features lay over the
    # depth-derived voxels 1:1 in the viewer.
    project_root = Path(__file__).resolve().parent.parent
    voxel_out_path = None
    meta_out_path = None
    voxel_size = args.voxel_size
    grid_shape = None
    world_min_tuple = tuple(args.world_min)
    world_max_tuple = tuple(args.world_max)
    frames_dir = Path(args.frames_dir)
    out_path = Path(args.out)

    if args.session is not None:
        session_dir = project_root / "captured_frames" / args.session
        if not session_dir.is_dir():
            raise SystemExit(f"session not found: {session_dir}")
        frames_dir = session_dir / "frames"
        voxel_out_path = session_dir / "voxels_features.json"
        meta_out_path  = session_dir / "features_meta.json"

        ref_variants = ([args.reference_variant] if args.reference_variant
                        else ["refined_aligned", "aligned", "refined", "original"])
        ref = None
        for v in ref_variants:
            cand = session_dir / f"voxels_{v}.json"
            if cand.exists():
                try:
                    ref = json.loads(cand.read_text())
                    print(f"Inheriting grid from {cand.name}: "
                          f"voxel_size={ref.get('voxel_size')} "
                          f"world_min={ref.get('world_min')} "
                          f"shape={ref.get('shape')}")
                    break
                except (OSError, json.JSONDecodeError) as e:
                    print(f"  failed to read {cand}: {e}")
        if ref is not None:
            if voxel_size is None and ref.get("voxel_size"):
                voxel_size = float(ref["voxel_size"])
            if ref.get("world_min"):
                world_min_tuple = tuple(float(x) for x in ref["world_min"])
            if ref.get("world_max"):
                world_max_tuple = tuple(float(x) for x in ref["world_max"])
            if ref.get("shape"):
                grid_shape = tuple(int(x) for x in ref["shape"])
        if voxel_size is None:
            voxel_size = 0.05

    reconstruct(
        frames_dir, out_path,
        n_features=args.n_features, downscale=args.downscale,
        ratio=args.ratio, window=args.window,
        min_views=args.min_views, max_residual_m=args.max_residual,
        max_sensitivity_m=args.max_sensitivity,
        min_depth_m=args.min_depth,
        world_min=world_min_tuple, world_max=world_max_tuple,
        cube_size=args.cube_size,
        workers=args.workers, max_frames=args.max_frames,
        voxel_out_path=voxel_out_path,
        meta_out_path=meta_out_path,
        voxel_size=voxel_size,
        grid_shape=grid_shape,
    )


if __name__ == "__main__":
    main()
