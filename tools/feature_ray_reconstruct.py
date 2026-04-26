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

    return {
        "des": des,                                  # (N, 32) uint8
        "rays_dirs": dirs.astype(np.float32),        # (N, 3)
        "cam_origin": cam_origin.astype(np.float32), # (3,)
        "colours": colours,                          # (N, 3) uint8
        "kp_count": len(kp),
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
    """origins (N, 3), dirs (N, 3, unit). Returns (P, residuals).

    Each ray contributes the projection matrix M_i = I - d_i d_i^T (project
    perpendicular to the ray). The point closest to all rays in the L²
    sense satisfies (Σ M_i) P = Σ M_i o_i.
    """
    M = np.eye(3)[None, :, :] - dirs[:, :, None] * dirs[:, None, :]  # (N, 3, 3)
    A = M.sum(axis=0)
    b = np.einsum("nij,nj->ni", M, origins).sum(axis=0)
    try:
        P = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, None
    diffs = P[None, :] - origins
    proj_lens = (diffs * dirs).sum(axis=1)
    perps = diffs - proj_lens[:, None] * dirs
    residuals = np.linalg.norm(perps, axis=1)
    return P, residuals


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
                world_min: tuple[float, float, float] = (-2.5, -0.3, -2.5),
                world_max: tuple[float, float, float] = ( 2.5,  4.7,  2.5),
                cube_size: float = 0.03,
                workers: int = 1,
                max_frames: int | None = None) -> None:
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

    out_pos: list[list[float]] = []
    out_col: list[list[int]] = []
    out_views: list[int] = []
    out_residual: list[float] = []
    n_kept = 0
    n_rejected_residual = 0
    n_rejected_oob = 0
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
        for fi, node in seen_frames.items():
            f = features[fi]
            li = int(node_to_local[node])
            origins.append(f["cam_origin"])
            dirs.append(f["rays_dirs"][li])
            cols.append(f["colours"][li])
        origins = np.asarray(origins, dtype=np.float64)
        dirs = np.asarray(dirs, dtype=np.float64)
        cols = np.asarray(cols, dtype=np.uint8)

        P, residuals = triangulate_rays(origins, dirs)
        if P is None:
            continue
        if (P < wmin).any() or (P > wmax).any():
            n_rejected_oob += 1
            continue
        mean_res = float(residuals.mean())
        if mean_res > max_residual_m:
            n_rejected_residual += 1
            continue

        out_pos.append([float(P[0]), float(P[1]), float(P[2])])
        out_col.append([int(c) for c in np.median(cols, axis=0)])
        out_views.append(int(len(seen_frames)))
        out_residual.append(mean_res)
        n_kept += 1

    print(f"  kept {n_kept:,}; rejected {n_rejected_residual:,} (residual) "
          f"+ {n_rejected_oob:,} (out of bbox) "
          f"+ {n_rejected_singleframe:,} (single-frame track)")

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
        "positions": out_pos,
        "colors": out_col,
        "n_views": out_views,
        "residuals": out_residual,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    print(f"Wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


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
    ap.add_argument("--world-min", type=float, nargs=3, default=[-2.5, -0.3, -2.5])
    ap.add_argument("--world-max", type=float, nargs=3, default=[ 2.5,  4.7,  2.5])
    ap.add_argument("--cube-size", type=float, default=0.03,
                    help="size of the rendered marker cube (m)")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    reconstruct(
        Path(args.frames_dir), Path(args.out),
        n_features=args.n_features, downscale=args.downscale,
        ratio=args.ratio, window=args.window,
        min_views=args.min_views, max_residual_m=args.max_residual,
        world_min=tuple(args.world_min), world_max=tuple(args.world_max),
        cube_size=args.cube_size,
        workers=args.workers, max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
