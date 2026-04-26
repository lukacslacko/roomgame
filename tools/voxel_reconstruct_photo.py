#!/usr/bin/env python3
"""
Iterative voxel reconstruction with photometric depth refinement.

Pipeline per iteration:
  1. Build the voxel grid from each frame's *current* per-pixel depth map
     (initialised from the captured depth sensor on iteration 1). Air /
     colour counters and thresholding work exactly as in
     voxel_reconstruct.py.
  2. Threshold to a (Nx, Ny, Nz) "kept" boolean mask.
  3. For every ray (per pixel of every frame), walk along the ray and
     record the first kept voxel index it lands in.
  4. Invert that mapping: voxel -> [(frame_idx, ray_idx), ...]. Two rays
     that share a voxel are observing approximately the same surface
     point from different cameras.
  5. For each ray A, look at its peer rays B sharing the same voxel
     (subject to an angle cutoff). For each peer, do a 1-D photometric
     search along the direction in B's image that corresponds to changing
     A's depth (∂q_B/∂t_A): try a small range of t_A in [t_A−tol, t_A+tol],
     project to B, sample a patch, compute NCC vs. A's patch. Keep the
     t_A that maximises the match. Convert that to a depth delta for A.
  6. Aggregate delta_t suggestions over peers (median, clamped to ±tol)
     and update each frame's depth map.

After all iterations, the final voxel grid build (with the most-refined
depths) is written as web/out/voxels.json.

This file is independent of voxel_reconstruct{,_iter}.py — those are
unchanged.
"""
from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import fusion
import serve


# ---------------------------------------------------------------------------
# Per-frame loading: rays + initial depth + grayscale image, kept in RAM.
# ---------------------------------------------------------------------------

def load_frame(frame_path: Path, *, near: float, far: float, subsample: int):
    """Decode a captured frame body into a dict of:
      pixel_xy:  (M, 2) int32     pixel coords in the colour image
      rays_dirs: (M, 3) float32   unit world directions
      cam_origin: (3,) float32
      depths_init: (M,) float32   initial depth from the sensor (immutable)
      depths_current: (M,) float32  starts == depths_init; mutated by refinement
      rgb: (M, 3) float32         colour at each ray's pixel (full-res)
      gray: (ch, cw) uint8        full-res grayscale image (for NCC patches)
      view, proj: (4, 4) float64  camera matrices (for projecting rays into us)
      cw, ch: int                  colour image dims
      valid_kept_voxel_idx: (M,) int32  set per iteration, −1 = no hit
    """
    body = frame_path.read_bytes()
    try:
        frame = serve.parse_frame(body)
    except Exception:
        return None
    if frame["color"] is None:
        return None

    width = int(frame["width"])
    height = int(frame["height"])
    depth = fusion.decode_depth(
        frame["depth"], width, height,
        int(frame["format"]), float(frame["rawValueToMeters"]),
    )

    cw = int(frame["color_width"])
    ch = int(frame["color_height"])
    if cw == 0 or ch == 0:
        return None
    color_arr = np.frombuffer(frame["color"], dtype=np.uint8).reshape(ch, cw, 4)
    gray = (color_arr[..., :3].astype(np.float32) @
            np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.float32)

    V = fusion._mat4_from_column_major(frame["viewMatrix"])
    P = fusion._mat4_from_column_major(frame["projectionMatrix"])
    Bd = fusion._mat4_from_column_major(frame["normDepthBufferFromNormView"])
    P_inv = np.linalg.inv(P)
    cam_origin = V[:3, 3].astype(np.float64)

    color_sub = color_arr[::subsample, ::subsample]
    sh, sw = color_sub.shape[:2]

    j_grid, i_grid = np.meshgrid(np.arange(sh), np.arange(sw), indexing="ij")
    px_x = (i_grid * subsample + subsample // 2).astype(np.int32)
    px_y = (j_grid * subsample + subsample // 2).astype(np.int32)
    u = (px_x + 0.5) / cw
    v = (px_y + 0.5) / ch

    x_ndc = 2.0 * u - 1.0
    y_ndc = 2.0 * v - 1.0
    clip = np.stack(
        [x_ndc, y_ndc, -np.ones_like(x_ndc), np.ones_like(x_ndc)], axis=-1,
    )
    view4 = clip @ P_inv.T
    view3 = view4[..., :3] / np.where(np.abs(view4[..., 3:4]) < 1e-12, 1.0, view4[..., 3:4])
    view3_h = np.concatenate([view3, np.ones((sh, sw, 1))], axis=-1)
    world_h = view3_h @ V.T
    world3 = world_h[..., :3] / np.where(np.abs(world_h[..., 3:4]) < 1e-12, 1.0, world_h[..., 3:4])
    ray_dirs = world3 - cam_origin[None, None, :]
    ray_lens = np.linalg.norm(ray_dirs, axis=-1, keepdims=True)
    ray_dirs = ray_dirs / np.maximum(ray_lens, 1e-9)

    nv_h = np.stack([u, v, np.zeros_like(u), np.ones_like(u)], axis=-1)
    nd_h = nv_h @ Bd.T
    u_d = nd_h[..., 0] / np.where(np.abs(nd_h[..., 3]) < 1e-12, 1.0, nd_h[..., 3])
    v_d = nd_h[..., 1] / np.where(np.abs(nd_h[..., 3]) < 1e-12, 1.0, nd_h[..., 3])
    bx = np.floor((1.0 - u_d) * width).astype(np.int32)
    by = np.floor(v_d * height).astype(np.int32)
    in_buf = (bx >= 0) & (bx < width) & (by >= 0) & (by < height)
    bx_c = np.clip(bx, 0, width - 1)
    by_c = np.clip(by, 0, height - 1)
    depth_est = depth[by_c, bx_c]

    valid = in_buf & (depth_est > near) & (depth_est < far)

    rgb = color_arr[px_y.ravel(), px_x.ravel(), :3].astype(np.float32)

    rays = ray_dirs.reshape(-1, 3).astype(np.float32)
    depths = depth_est.reshape(-1).astype(np.float32)
    pixel_xy = np.stack([px_x.ravel(), px_y.ravel()], axis=-1).astype(np.int32)
    valid_flat = valid.reshape(-1)

    # Drop invalid rays once and for all so we never look at them again.
    rays = rays[valid_flat]
    depths = depths[valid_flat]
    pixel_xy = pixel_xy[valid_flat]
    rgb = rgb[valid_flat]

    return {
        "pixel_xy": pixel_xy,
        "rays_dirs": rays,
        "cam_origin": cam_origin.astype(np.float32),
        "depths_init": depths.copy(),
        "depths_current": depths.copy(),
        "rgb": rgb,
        "gray": gray,
        "V": V,
        "P": P,
        "V_inv": np.linalg.inv(V),                   # cached for project_to_pixel
        "PV_inv": P @ np.linalg.inv(V),              # combined clip_from_world
        "cw": cw, "ch": ch,
    }


# ---------------------------------------------------------------------------
# Voxel-grid build: same machinery as voxel_reconstruct, but using each
# frame's current per-ray depth array (already filtered + valid) directly.
# Also records, per ray, the first kept voxel index along its line.
# ---------------------------------------------------------------------------

def run_voxel_pass(frames, grid_params, *, kept_mask=None, record_hits=False):
    """One full sweep: accumulate air/colour counters from every frame's
    current depth map. If `record_hits`, also fills frame['hit_voxel'] with
    the flat voxel index of the first kept voxel on each ray (-1 if none),
    in which case `kept_mask` must be provided."""
    Nx, Ny, Nz = grid_params["shape"]
    Ntot = Nx * Ny * Nz
    Ny_Nz = Ny * Nz

    air = np.zeros(Ntot, dtype=np.uint32)
    cc  = np.zeros(Ntot, dtype=np.uint32)
    cs  = np.zeros((Ntot, 3), dtype=np.float64)

    n_steps = grid_params["n_steps"]
    step = grid_params["step"]
    near = grid_params["near"]
    tol = grid_params["tol"]
    voxel_size = grid_params["voxel_size"]
    grid_origin = grid_params["world_min"].astype(np.float32)
    chunk_size = grid_params["chunk_size"]

    t_vals = (near + (np.arange(n_steps, dtype=np.float32) + 0.5) * step).astype(np.float32)
    t_vals_2d = t_vals[None, :]

    for fi, frame in enumerate(frames):
        if frame is None:
            continue
        rays = frame["rays_dirs"]
        depths = frame["depths_current"]
        rgb = frame["rgb"]
        cam_origin = frame["cam_origin"]
        M = rays.shape[0]
        if record_hits:
            hits = np.full(M, -1, dtype=np.int64)

        for cs_i in range(0, M, chunk_size):
            ce_i = min(cs_i + chunk_size, M)
            rays_c = rays[cs_i:ce_i]
            depth_c = depths[cs_i:ce_i]
            rgb_c = rgb[cs_i:ce_i]
            N = ce_i - cs_i

            wp = cam_origin[None, None, :] + t_vals[None, :, None] * rays_c[:, None, :]
            vp = (wp - grid_origin[None, None, :]) / np.float32(voxel_size)
            vidx = np.floor(vp).astype(np.int32)
            vx = vidx[..., 0]; vy = vidx[..., 1]; vz = vidx[..., 2]
            in_grid = (
                (vx >= 0) & (vx < Nx) &
                (vy >= 0) & (vy < Ny) &
                (vz >= 0) & (vz < Nz)
            )

            d = depth_c[:, None]
            air_mask = (t_vals_2d < d - tol) & in_grid
            col_mask = (np.abs(t_vals_2d - d) <= tol) & in_grid

            flat = vx * Ny_Nz + vy * Nz + vz

            if air_mask.any():
                idx_air = flat[air_mask]
                air[:] += np.bincount(idx_air, minlength=Ntot).astype(np.uint32)

            if col_mask.any():
                idx_col = flat[col_mask]
                cc[:] += np.bincount(idx_col, minlength=Ntot).astype(np.uint32)
                mi, _si = np.nonzero(col_mask)
                cols = rgb_c[mi]
                cs[:, 0] += np.bincount(idx_col, weights=cols[:, 0], minlength=Ntot)
                cs[:, 1] += np.bincount(idx_col, weights=cols[:, 1], minlength=Ntot)
                cs[:, 2] += np.bincount(idx_col, weights=cols[:, 2], minlength=Ntot)

            if record_hits and kept_mask is not None:
                vx_safe = np.clip(vx, 0, Nx - 1)
                vy_safe = np.clip(vy, 0, Ny - 1)
                vz_safe = np.clip(vz, 0, Nz - 1)
                kept_at = kept_mask[vx_safe, vy_safe, vz_safe] & in_grid
                has_hit = kept_at.any(axis=1)
                first_hit = kept_at.argmax(axis=1)
                # Flat voxel index of the first hit, or -1 if has_hit is False.
                flat_first = flat[np.arange(N), first_hit]
                hits[cs_i:ce_i] = np.where(has_hit, flat_first, -1)

        if record_hits:
            frame["hit_voxel"] = hits

    return air, cc, cs


# ---------------------------------------------------------------------------
# Photometric refinement.
# ---------------------------------------------------------------------------

def project_to_pixel(P_world: np.ndarray, frame: dict) -> np.ndarray | None:
    """Project 3D world points (S, 3) to image-pixel coords (S, 2) for
    `frame` (which carries the cached V_inv / PV_inv matrices).

    Returns None if all points are behind the camera; otherwise returns
    (S, 2) with NaNs for any point at/behind the camera.
    """
    V_inv = frame["V_inv"]
    P_proj = frame["P"]
    cw = frame["cw"]; ch = frame["ch"]

    h = np.concatenate([P_world, np.ones((P_world.shape[0], 1))], axis=-1)
    view = h @ V_inv.T
    view_xyz = view[:, :3] / np.where(np.abs(view[:, 3:4]) < 1e-12, 1.0, view[:, 3:4])
    behind = view_xyz[:, 2] >= -0.001
    if behind.all():
        return None
    view4 = np.concatenate([view_xyz, np.ones((P_world.shape[0], 1))], axis=-1)
    clip = view4 @ P_proj.T
    cw_ = clip[:, 3:4]
    ndc = clip[:, :3] / np.where(np.abs(cw_) < 1e-12, 1.0, cw_)
    px = (ndc[:, 0] + 1.0) * 0.5 * cw
    py = (1.0 - (ndc[:, 1] + 1.0) * 0.5) * ch
    out = np.stack([px, py], axis=-1)
    if behind.any():
        out[behind] = np.nan
    return out


def extract_patch(gray: np.ndarray, cx: float, cy: float, half: int) -> np.ndarray | None:
    """Sample a (2*half+1)² patch centred at (cx, cy) using nearest-pixel
    rounding. Returns None if any of the patch falls outside the image."""
    ix = int(round(cx))
    iy = int(round(cy))
    h, w = gray.shape
    if ix - half < 0 or ix + half >= w or iy - half < 0 or iy + half >= h:
        return None
    return gray[iy - half:iy + half + 1, ix - half:ix + half + 1]


def normalised_patch(p: np.ndarray) -> np.ndarray | None:
    p = p.astype(np.float32, copy=False)
    m = p.mean()
    s = p.std()
    if s < 1e-3:
        return None
    return (p - m) / s


def photometric_refine_pass(frames, *, tol: float, max_angle_deg: float,
                            patch_half: int, n_search: int,
                            min_ncc: float,
                            max_peers_per_ray: int) -> tuple[int, int]:
    """Update each frame's `depths_current` based on photo-consistency
    with peer rays sharing the same kept voxel. Returns (n_rays_updated,
    n_peer_pairs_evaluated)."""
    Nx, Ny, Nz = frames[0]["_shape"]
    cos_thresh = np.cos(np.radians(max_angle_deg))

    # Build voxel -> [(frame_idx, ray_idx), ...] map.
    voxel_to_rays: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for fi, frame in enumerate(frames):
        if frame is None or "hit_voxel" not in frame:
            continue
        hits = frame["hit_voxel"]
        nz = np.nonzero(hits >= 0)[0]
        for ri in nz:
            voxel_to_rays[int(hits[ri])].append((fi, int(ri)))

    n_voxels_with_peers = sum(1 for ray_list in voxel_to_rays.values() if len(ray_list) >= 2)
    print(f"    {n_voxels_with_peers:,} voxels have ≥2 ray observations "
          f"(out of {len(voxel_to_rays):,} hit voxels)")

    dt_sums = [np.zeros(len(f["rays_dirs"]), dtype=np.float64) if f is not None else None
               for f in frames]
    dt_counts = [np.zeros(len(f["rays_dirs"]), dtype=np.int32) if f is not None else None
                 for f in frames]

    rng = np.random.default_rng(0)
    n_pairs_eval = 0

    for ray_list in voxel_to_rays.values():
        K = len(ray_list)
        if K < 2:
            continue

        for ai, (fi_a, ri_a) in enumerate(ray_list):
            frame_a = frames[fi_a]
            if frame_a is None:
                continue
            d_a = frame_a["rays_dirs"][ri_a]
            o_a = frame_a["cam_origin"]
            t_a = float(frame_a["depths_current"][ri_a])
            p_a_xy = frame_a["pixel_xy"][ri_a]

            patch_a = extract_patch(frame_a["gray"], p_a_xy[0], p_a_xy[1], patch_half)
            if patch_a is None:
                continue
            patch_a_n = normalised_patch(patch_a)
            if patch_a_n is None:
                continue

            ts = t_a + np.linspace(-tol, tol, n_search, dtype=np.float32)
            P_candidates = o_a[None, :] + ts[:, None] * d_a[None, :]   # (S, 3)

            peer_indices = list(range(K))
            peer_indices.remove(ai)
            if len(peer_indices) > max_peers_per_ray:
                peer_indices = list(rng.choice(peer_indices, size=max_peers_per_ray, replace=False))

            best_dts = []
            for bi in peer_indices:
                fi_b, ri_b = ray_list[bi]
                frame_b = frames[fi_b]
                if frame_b is None or fi_b == fi_a:
                    continue
                d_b = frame_b["rays_dirs"][ri_b]
                cos_ang = abs(float(np.dot(d_a, d_b)))
                if cos_ang < cos_thresh:
                    continue

                qs = project_to_pixel(P_candidates.astype(np.float64), frame_b)
                if qs is None:
                    continue

                # Vectorise across the n_search candidates for this peer:
                # extract S patches into one (S, ph, ph) array, normalise,
                # and dot against the (already-normalised) source patch.
                gray_b = frame_b["gray"]
                cw_b = frame_b["cw"]; ch_b = frame_b["ch"]
                ph = 2 * patch_half + 1

                ix = np.round(qs[:, 0])
                iy = np.round(qs[:, 1])
                # Treat NaNs (behind-camera) as out-of-bounds.
                in_b = (
                    np.isfinite(ix) & np.isfinite(iy)
                    & (ix - patch_half >= 0) & (ix + patch_half < cw_b)
                    & (iy - patch_half >= 0) & (iy + patch_half < ch_b)
                )
                if not in_b.any():
                    n_pairs_eval += 1
                    continue
                ix_i = np.where(in_b, ix, 0).astype(np.int32)
                iy_i = np.where(in_b, iy, 0).astype(np.int32)

                # Gather all S patches with one fancy-indexing call.
                # Build (S, ph, ph) row/col offset grid.
                rr = (np.arange(ph) - patch_half)[None, :, None] + iy_i[:, None, None]
                cc_ = (np.arange(ph) - patch_half)[None, None, :] + ix_i[:, None, None]
                patches_b = gray_b[rr, cc_].astype(np.float32)         # (S, ph, ph)
                m = patches_b.mean(axis=(1, 2), keepdims=True)
                d = patches_b - m
                s = np.sqrt((d * d).mean(axis=(1, 2), keepdims=True))
                ok = (s.squeeze((1, 2)) > 1e-3) & in_b
                if not ok.any():
                    n_pairs_eval += 1
                    continue
                norm = d / np.where(s > 1e-3, s, 1.0)                  # (S, ph, ph)
                # NCC = mean of element-wise product against normalised source.
                scores = (norm * patch_a_n[None, :, :]).mean(axis=(1, 2))
                # Mask out invalid candidates with -inf.
                scores = np.where(ok, scores, -np.inf)
                best_s = int(np.argmax(scores))
                best_score = float(scores[best_s])
                n_pairs_eval += 1
                if np.isfinite(best_score) and best_score >= min_ncc:
                    best_dts.append(float(ts[best_s] - t_a))

            if best_dts:
                dt = float(np.median(best_dts))
                dt = max(-tol, min(tol, dt))
                dt_sums[fi_a][ri_a] += dt
                dt_counts[fi_a][ri_a] += 1

    n_updated = 0
    for fi, frame in enumerate(frames):
        if frame is None:
            continue
        c = dt_counts[fi]
        nz = c > 0
        if not nz.any():
            continue
        delta = np.where(nz, dt_sums[fi] / np.maximum(c, 1), 0.0).astype(np.float32)
        # Don't drift further from the original depth than `tol` over the whole run.
        new_depth = frame["depths_current"] + delta
        max_drift = tol  # cumulative drift bound from the initial sensor depth
        original = frame["depths_init"]
        new_depth = np.clip(new_depth, original - max_drift, original + max_drift)
        frame["depths_current"] = new_depth
        n_updated += int(nz.sum())

    return n_updated, n_pairs_eval


# ---------------------------------------------------------------------------
# Main orchestration.
# ---------------------------------------------------------------------------

def reconstruct(frames_dir: Path, out_path: Path, *,
                voxel_size: float = 0.05,
                world_min=(-2.5, -0.3, -2.5),
                world_max=( 2.5,  4.7,  2.5),
                near: float = 0.05,
                far: float = 5.0,
                tol: float = 0.20,
                threshold: float = 0.10,
                subsample: int = 8,
                chunk_size: int = 200_000,
                max_frames: int | None = None,
                iterations: int = 3,
                photo_iterations: int = 1,
                max_angle_deg: float = 30.0,
                patch_half: int = 2,
                n_search: int = 5,
                min_ncc: float = 0.6,
                max_peers_per_ray: int = 6) -> None:
    wmin = np.asarray(world_min, dtype=np.float64)
    wmax = np.asarray(world_max, dtype=np.float64)
    shape = tuple(int(np.ceil((wmax[i] - wmin[i]) / voxel_size)) for i in range(3))
    Nx, Ny, Nz = shape
    Ntot = Nx * Ny * Nz
    print(f"Voxel grid: {shape}  ({Ntot:,} voxels, {voxel_size*100:.1f} cm/edge)")

    frame_paths = sorted(frames_dir.glob("frame_*.bin"))
    if max_frames:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        print(f"No frames found in {frames_dir}")
        return
    print(f"Loading {len(frame_paths)} frames (subsample={subsample})…")
    t0 = time.time()
    frames = []
    for i, fp in enumerate(frame_paths):
        f = load_frame(fp, near=near, far=far, subsample=subsample)
        frames.append(f)
        if (i + 1) % 25 == 0 or i == len(frame_paths) - 1:
            print(f"  loaded {i+1}/{len(frame_paths)}")
    n_loaded = sum(1 for f in frames if f is not None)
    n_rays_total = sum(len(f["rays_dirs"]) for f in frames if f is not None)
    print(f"Loaded {n_loaded} frames, {n_rays_total:,} valid rays "
          f"in {time.time()-t0:.1f} s")

    for f in frames:
        if f is not None:
            f["_shape"] = shape

    step = voxel_size
    n_steps = int(np.ceil((far + tol - near) / step)) + 1
    grid_params = {
        "world_min": wmin,
        "voxel_size": voxel_size,
        "shape": shape,
        "Ntot": Ntot,
        "near": near, "far": far, "tol": tol,
        "step": step, "n_steps": n_steps,
        "subsample": subsample,
        "chunk_size": chunk_size,
    }

    for it in range(1, iterations + 1):
        print(f"\n=== iteration {it}/{iterations} ===")
        t_iter = time.time()

        # Pass 1: build grid using current depths. On iter > 1 we also
        # record per-ray hit voxel against the previous mask.
        kept_mask = frames[0].get("_kept_mask") if frames[0] is not None else None
        record_hits = it < iterations and photo_iterations > 0
        air, cc, cs = run_voxel_pass(
            frames, grid_params,
            kept_mask=kept_mask,
            record_hits=record_hits and (kept_mask is not None),
        )
        # Threshold.
        total = cc.astype(np.float64) + air.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(total > 0, cc.astype(np.float64) / np.maximum(total, 1), 0.0)
        keep_flat = (cc >= 1) & (ratio >= threshold)
        n_kept = int(keep_flat.sum())
        new_kept = keep_flat.reshape(shape)
        print(f"  iter {it}: kept {n_kept:,} voxels "
              f"(ratio≥{threshold}, color_count≥1) — pass took {time.time()-t_iter:.1f} s")

        if it < iterations:
            # Compute per-ray hit voxels against THIS iteration's mask, then
            # run the photometric refinement (optionally multiple sub-passes).
            for sub in range(photo_iterations):
                t_sub = time.time()
                # Re-record hits against the updated mask.
                run_voxel_pass(frames, grid_params, kept_mask=new_kept,
                               record_hits=True)  # discards counters, just sets hit_voxel
                n_upd, n_pairs = photometric_refine_pass(
                    frames, tol=tol, max_angle_deg=max_angle_deg,
                    patch_half=patch_half, n_search=n_search,
                    min_ncc=min_ncc, max_peers_per_ray=max_peers_per_ray,
                )
                print(f"    photo sub-pass {sub+1}/{photo_iterations}: "
                      f"{n_upd:,} rays updated, {n_pairs:,} ray-pair NCCs "
                      f"in {time.time()-t_sub:.1f} s")
            for f in frames:
                if f is not None:
                    f["_kept_mask"] = new_kept

    # Final write-out.
    if n_kept == 0:
        print("Nothing to write.")
        return

    safe = np.maximum(cc[keep_flat, None], 1).astype(np.float64)
    avg_color = (cs[keep_flat] / safe).clip(0, 255).astype(np.uint8)
    flat_idx = np.nonzero(keep_flat)[0]
    iz = flat_idx % Nz
    iy = (flat_idx // Nz) % Ny
    ix = flat_idx // (Ny * Nz)

    payload = {
        "voxel_size": voxel_size,
        "world_min": wmin.tolist(),
        "world_max": (wmin + np.array(shape) * voxel_size).tolist(),
        "shape": list(shape),
        "threshold": threshold,
        "iterations": iterations,
        "photo_iterations": photo_iterations,
        "n_voxels": n_kept,
        "indices": np.stack([ix, iy, iz], axis=-1).astype(int).tolist(),
        "colors":  avg_color.astype(int).tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    print(f"Wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--frames-dir", default="captured_frames")
    ap.add_argument("--out", default="web/out/voxels.json")
    ap.add_argument("--voxel-size", type=float, default=0.05)
    ap.add_argument("--world-min", type=float, nargs=3, default=[-2.5, -0.3, -2.5])
    ap.add_argument("--world-max", type=float, nargs=3, default=[ 2.5,  4.7,  2.5])
    ap.add_argument("--near", type=float, default=0.05)
    ap.add_argument("--far", type=float, default=5.0)
    ap.add_argument("--tol", type=float, default=0.20)
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--subsample", type=int, default=8,
                    help="default 8 — higher than the other reconstructors "
                         "because we keep full per-ray data in RAM and run "
                         "all-pairs photometric matching")
    ap.add_argument("--chunk-size", type=int, default=200_000)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=3,
                    help="outer iterations (one voxel-grid build per iter)")
    ap.add_argument("--photo-iterations", type=int, default=1,
                    help="inner photometric refinement passes per outer iter "
                         "(0 disables refinement; the tool then behaves like "
                         "voxel_reconstruct.py with one extra build).")
    ap.add_argument("--max-angle-deg", type=float, default=30.0,
                    help="reject peer rays whose angle to the source ray "
                         "exceeds this (small-angle approximation breaks "
                         "down past ~30°)")
    ap.add_argument("--patch-half", type=int, default=2,
                    help="patch is (2*half+1)²; default 2 → 5×5")
    ap.add_argument("--n-search", type=int, default=5,
                    help="number of t_A samples in [t−tol, t+tol]")
    ap.add_argument("--min-ncc", type=float, default=0.6,
                    help="reject peer matches with NCC below this")
    ap.add_argument("--max-peers-per-ray", type=int, default=6,
                    help="cap on peer rays evaluated per source ray")
    args = ap.parse_args()

    reconstruct(
        Path(args.frames_dir), Path(args.out),
        voxel_size=args.voxel_size,
        world_min=tuple(args.world_min),
        world_max=tuple(args.world_max),
        near=args.near, far=args.far, tol=args.tol,
        threshold=args.threshold,
        subsample=args.subsample,
        chunk_size=args.chunk_size,
        max_frames=args.max_frames,
        iterations=args.iterations,
        photo_iterations=args.photo_iterations,
        max_angle_deg=args.max_angle_deg,
        patch_half=args.patch_half,
        n_search=args.n_search,
        min_ncc=args.min_ncc,
        max_peers_per_ray=args.max_peers_per_ray,
    )


if __name__ == "__main__":
    main()
