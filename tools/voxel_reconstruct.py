#!/usr/bin/env python3
"""
Offline voxel reconstruction from captured WebXR frames.

For each frame's RGB pixel we:
  1. Compute a world-space ray from the camera through that pixel.
  2. Look up the corresponding depth-buffer pixel and read its
     metres-from-camera value `d`.
  3. Walk along the ray in fixed-size steps:
       - while t ∈ [near, d − tol]:    increment `air_count` for that voxel.
       - while t ∈ [d − tol, d + tol]: increment `color_count` and accumulate
                                       the pixel's RGB into `color_sum`.
       - t > d + tol:                  stop (we've passed the surface).

After all frames, voxels with `color_count / (color_count + air_count) >
threshold` (default 0.10) are kept and rendered as opaque cubes whose colour
is the average of the accumulated RGB.

Output: `web/out/voxels.json` (a list of voxel grid indices + averaged colour),
served as a static file by tools/serve.py and consumed by web/voxelview.html.

Vectorised in numpy: per chunk of pixels, iterate steps t and do one
np.bincount per (air, colour) channel. ~100-line inner loop, dominated by
np.bincount on 1M-element grids.
"""
from __future__ import annotations

# Cap BLAS thread pools to 1 *before* numpy gets imported. With multiprocessing
# we already have W workers; OpenBLAS spawning T threads per worker yields
# W × T contending threads and cratery throughput on Apple Silicon. Single-
# process runs (workers=1) are unaffected because all the heavy work is in
# np.bincount, which is single-threaded anyway.
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

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


def reconstruct(
    frames_dir: Path,
    out_path: Path,
    *,
    voxel_size: float = 0.05,
    world_min: tuple[float, float, float] = (-2.5, -0.3, -2.5),
    world_max: tuple[float, float, float] = (2.5, 4.7, 2.5),
    near: float = 0.05,
    far: float = 8.0,
    tol: float = 0.20,
    threshold: float = 0.10,
    subsample: int = 4,
    chunk_size: int = 200_000,
    max_frames: int | None = None,
    workers: int = 1,
    no_air: bool = False,
    min_color_count: int = 1,
) -> None:
    wmin = np.asarray(world_min, dtype=np.float64)
    wmax = np.asarray(world_max, dtype=np.float64)
    shape = tuple(int(np.ceil((wmax[i] - wmin[i]) / voxel_size)) for i in range(3))
    Nx, Ny, Nz = shape
    Ntot = Nx * Ny * Nz
    print(f"Voxel grid: {shape}  ({Ntot:,} voxels, {voxel_size*100:.1f} cm/edge)")
    print(f"world bbox: {wmin} → {wmax}")

    air_count = np.zeros(Ntot, dtype=np.uint32)
    color_count = np.zeros(Ntot, dtype=np.uint32)
    color_sum = np.zeros((Ntot, 3), dtype=np.float64)

    frame_paths = sorted(frames_dir.glob("frame_*.bin"))
    if max_frames:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        print(f"No frames found in {frames_dir}")
        return
    print(f"Processing {len(frame_paths)} frames from {frames_dir} (workers={workers})")

    # Sampling at voxel_size (rather than half) gives ~1 sample per voxel
    # along each ray — quite enough for an occupancy/free counter, and roughly
    # halves the per-frame bincount work. Sub-voxel jitter is irrelevant
    # because the masks aggregate over many frames.
    step = voxel_size
    n_steps = int(np.ceil((far + tol - near) / step)) + 1
    if no_air:
        # With no-air, we sample only a short window around each ray's d so the
        # heavy work scales with K instead of n_steps. K covers [d-tol, d+tol].
        n_steps_eff = int(np.ceil((2.0 * tol) / step)) + 1
        print(f"  no-air mode: per-ray K={n_steps_eff} samples around d "
              f"(skipping {n_steps - n_steps_eff} pre-depth steps)")
    t0_total = time.time()

    # Bundle every per-frame parameter into one picklable dict so we can ship
    # it to subprocesses without re-deriving anything.
    grid_params = {
        "world_min": wmin,
        "voxel_size": voxel_size,
        "shape": shape,
        "Ntot": Ntot,
        "near": near, "far": far, "tol": tol,
        "step": step, "n_steps": n_steps,
        "subsample": subsample,
        "chunk_size": chunk_size,
        "no_air": no_air,
    }

    if workers <= 1:
        for fi, fp in enumerate(frame_paths):
            body = fp.read_bytes()
            try:
                frame = serve.parse_frame(body)
            except Exception as e:  # noqa: BLE001
                print(f"  frame {fi}: parse error {e}")
                continue
            if frame["color"] is None:
                continue
            t_frame = time.time()
            rays_processed = process_frame(
                frame, air_count, color_count, color_sum,
                wmin, voxel_size, shape, step,
                near=near, far=far, tol=tol,
                subsample=subsample, chunk_size=chunk_size,
                n_steps=n_steps,
                no_air=no_air,
            )
            dt = time.time() - t_frame
            if (fi + 1) % 5 == 0 or fi == 0 or fi == len(frame_paths) - 1:
                print(f"  frame {fi+1:4d}/{len(frame_paths)}: "
                      f"{rays_processed:7d} rays in {dt*1000:.0f} ms; "
                      f"running totals: color {int(color_count.sum()):,}, "
                      f"air {int(air_count.sum()):,}")
    else:
        # Multiprocess path: each worker gets a contiguous slice of the
        # frame list, accumulates its own counter buffers in numpy, and
        # ships them back. Counters are summed in the parent.
        batches = np.array_split(np.array(frame_paths, dtype=object), workers)
        batches = [list(b) for b in batches if len(b) > 0]
        print(f"  splitting into {len(batches)} batches "
              f"(~{len(batches[0])} frames/worker)")

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_worker_process_batch,
                          [str(p) for p in batch], grid_params): bi
                for bi, batch in enumerate(batches)
            }
            done = 0
            for fut in as_completed(futures):
                bi = futures[fut]
                a, c, s, n_ok, dt_w = fut.result()
                air_count += a
                color_count += c
                color_sum += s
                done += 1
                print(f"  batch {bi+1:2d}/{len(batches)} merged "
                      f"({n_ok} frames in {dt_w:.1f} s; "
                      f"merged {done}/{len(batches)})")

    print(f"\nAll frames processed in {time.time()-t0_total:.1f} s")

    # Threshold + average. With no_air the ratio test is vacuous (denominator
    # is just color_count), so the gate effectively reduces to the
    # min_color_count check.
    total = color_count.astype(np.float64) + air_count.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total > 0, color_count.astype(np.float64) / np.maximum(total, 1), 0.0)
    keep = (color_count >= min_color_count) & (ratio >= threshold)
    n_kept = int(keep.sum())
    print(f"Kept {n_kept:,} of {Ntot:,} voxels "
          f"(ratio >= {threshold}, color_count >= {min_color_count})")

    if n_kept == 0:
        print("Nothing to write.")
        return

    # Compute average colour for kept voxels.
    safe_count = np.maximum(color_count[keep, None], 1).astype(np.float64)
    avg_color = (color_sum[keep] / safe_count).clip(0, 255).astype(np.uint8)

    # Convert flat indices back to 3D.
    flat_idx = np.nonzero(keep)[0]
    iz = flat_idx % Nz
    iy = (flat_idx // Nz) % Ny
    ix = flat_idx // (Ny * Nz)

    payload = {
        "voxel_size": voxel_size,
        "world_min": wmin.tolist(),
        "world_max": (wmin + np.array(shape) * voxel_size).tolist(),
        "shape": list(shape),
        "threshold": threshold,
        "n_voxels": n_kept,
        "indices": np.stack([ix, iy, iz], axis=-1).astype(int).tolist(),
        "colors":  avg_color.astype(int).tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    print(f"Wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


def _frame_rays(frame, *, near, far, subsample):
    """Decode a captured frame body into per-pixel rays + depth + RGB.
    Pure numpy; no GPU yet. Returns (rays, depths, rgbs, cam_origin, M_valid)
    where rays/depths/rgbs only contain valid pixels (in-buf, depth in range).
    """
    width = int(frame["width"])
    height = int(frame["height"])
    depth = fusion.decode_depth(
        frame["depth"], width, height,
        int(frame["format"]), float(frame["rawValueToMeters"]),
    )

    cw = int(frame["color_width"])
    ch = int(frame["color_height"])
    color_payload = frame["color"]
    if color_payload is None or cw == 0 or ch == 0:
        return None

    color_arr = np.frombuffer(color_payload, dtype=np.uint8).reshape(ch, cw, 4)

    V  = fusion._mat4_from_column_major(frame["viewMatrix"])
    P  = fusion._mat4_from_column_major(frame["projectionMatrix"])
    Bd = fusion._mat4_from_column_major(frame["normDepthBufferFromNormView"])
    P_inv = np.linalg.inv(P)

    cam_origin = V[:3, 3].astype(np.float64)

    color_sub = color_arr[::subsample, ::subsample]
    sh, sw = color_sub.shape[:2]

    j_grid, i_grid = np.meshgrid(np.arange(sh), np.arange(sw), indexing="ij")
    u = (i_grid * subsample + 0.5 * subsample) / cw
    v = (j_grid * subsample + 0.5 * subsample) / ch

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

    rgb = color_sub[..., :3]

    rays_flat = ray_dirs.reshape(-1, 3)
    depth_flat = depth_est.reshape(-1)
    valid_flat = valid.reshape(-1)
    rgb_flat = rgb.reshape(-1, 3)

    return (
        rays_flat[valid_flat].astype(np.float32),
        depth_flat[valid_flat].astype(np.float32),
        rgb_flat[valid_flat].astype(np.float32),
        cam_origin.astype(np.float32),
    )


def process_frame(frame, air_count, color_count, color_sum,
                 grid_origin, voxel_size, shape, step,
                 *, near, far, tol, subsample, chunk_size, n_steps,
                 no_air: bool = False) -> int:
    """Vectorised numpy backend.

    For each chunk of pixels, builds (chunk × S) sample arrays in one shot and
    runs bincounts over the masked indices. With `no_air=False`, S = n_steps
    spans [near, far+tol] and we count both air and colour. With `no_air=True`,
    S is a small window K around each ray's depth d (covering [d-tol, d+tol]),
    so we skip the air bincount and the pre-depth steps entirely.
    """
    rays_data = _frame_rays(frame, near=near, far=far, subsample=subsample)
    if rays_data is None:
        return 0
    rays_v, depth_v, rgb_v, cam_origin = rays_data
    M = rays_v.shape[0]
    if M == 0:
        return 0

    Nx, Ny, Nz = shape
    Ntot = Nx * Ny * Nz
    Ny_Nz = Ny * Nz

    if no_air:
        K = int(np.ceil((2.0 * tol) / step)) + 1
        offsets = ((np.arange(K, dtype=np.float32) - (K - 1) / 2.0) * step).astype(np.float32)
    else:
        t_vals = (near + (np.arange(n_steps, dtype=np.float32) + 0.5) * step).astype(np.float32)

    for cs in range(0, M, chunk_size):
        ce = min(cs + chunk_size, M)
        rays_c = rays_v[cs:ce]                                    # (N, 3) f32
        depth_c = depth_v[cs:ce]                                  # (N,)   f32
        rgb_c = rgb_v[cs:ce]                                      # (N, 3) f32

        if no_air:
            # Per-ray sample positions at t = d + offset, for offset ∈ ±tol.
            t_per = depth_c[:, None] + offsets[None, :]           # (N, K)
            wp = cam_origin[None, None, :] + t_per[:, :, None] * rays_c[:, None, :]
            vp = (wp - grid_origin[None, None, :].astype(np.float32)) / np.float32(voxel_size)
            vidx = np.floor(vp).astype(np.int32)
            vx = vidx[..., 0]; vy = vidx[..., 1]; vz = vidx[..., 2]
            in_grid = (
                (vx >= 0) & (vx < Nx) &
                (vy >= 0) & (vy < Ny) &
                (vz >= 0) & (vz < Nz)
            )
            in_range = (t_per >= near) & (t_per <= far + tol)
            col_mask = in_grid & in_range
            flat = vx * Ny_Nz + vy * Nz + vz
            if col_mask.any():
                idx_col = flat[col_mask]
                color_count[:] += np.bincount(idx_col, minlength=Ntot).astype(np.uint32)
                mi, _si = np.nonzero(col_mask)
                cols = rgb_c[mi]
                color_sum[:, 0] += np.bincount(idx_col, weights=cols[:, 0], minlength=Ntot)
                color_sum[:, 1] += np.bincount(idx_col, weights=cols[:, 1], minlength=Ntot)
                color_sum[:, 2] += np.bincount(idx_col, weights=cols[:, 2], minlength=Ntot)
            continue

        # Default path: dense [near, far] sweep + air counter.
        wp = cam_origin[None, None, :] + t_vals[None, :, None] * rays_c[:, None, :]
        vp = (wp - grid_origin[None, None, :].astype(np.float32)) / np.float32(voxel_size)
        vidx = np.floor(vp).astype(np.int32)                      # (N, S, 3)
        vx = vidx[..., 0]; vy = vidx[..., 1]; vz = vidx[..., 2]
        in_grid = (
            (vx >= 0) & (vx < Nx) &
            (vy >= 0) & (vy < Ny) &
            (vz >= 0) & (vz < Nz)
        )

        d = depth_c[:, None]
        t = t_vals[None, :]
        air_mask = (t < d - tol) & in_grid
        col_mask = (np.abs(t - d) <= tol) & in_grid

        flat = vx * Ny_Nz + vy * Nz + vz                          # (N, S) int32

        if air_mask.any():
            idx_air = flat[air_mask]
            air_count[:] += np.bincount(idx_air, minlength=Ntot).astype(np.uint32)

        if col_mask.any():
            idx_col = flat[col_mask]
            color_count[:] += np.bincount(idx_col, minlength=Ntot).astype(np.uint32)
            mi, _si = np.nonzero(col_mask)
            cols = rgb_c[mi]
            color_sum[:, 0] += np.bincount(idx_col, weights=cols[:, 0], minlength=Ntot)
            color_sum[:, 1] += np.bincount(idx_col, weights=cols[:, 1], minlength=Ntot)
            color_sum[:, 2] += np.bincount(idx_col, weights=cols[:, 2], minlength=Ntot)

    return M


# ----- multiprocessing worker --------------------------------------------

def _worker_process_batch(frame_paths: list[str], grid_params: dict):
    """Run in a subprocess. Accumulate counters across `frame_paths` using
    the same vectorised numpy `process_frame` and return the buffers."""
    Ntot = grid_params["Ntot"]
    air = np.zeros(Ntot, dtype=np.uint32)
    cc  = np.zeros(Ntot, dtype=np.uint32)
    cs  = np.zeros((Ntot, 3), dtype=np.float64)

    t0 = time.time()
    n_ok = 0
    for fp in frame_paths:
        try:
            body = Path(fp).read_bytes()
            frame = serve.parse_frame(body)
        except Exception:
            continue
        if frame["color"] is None:
            continue
        rays_done = process_frame(
            frame, air, cc, cs,
            grid_params["world_min"], grid_params["voxel_size"],
            grid_params["shape"], grid_params["step"],
            near=grid_params["near"], far=grid_params["far"],
            tol=grid_params["tol"],
            subsample=grid_params["subsample"],
            chunk_size=grid_params["chunk_size"],
            n_steps=grid_params["n_steps"],
            no_air=grid_params.get("no_air", False),
        )
        if rays_done > 0:
            n_ok += 1
    return air, cc, cs, n_ok, time.time() - t0


# =====================================================================
# Reverse mode: iterate voxels, project onto frames.
# =====================================================================

def _prepare_frame_reverse(frame, near: float, far: float):
    """Decode a captured frame body into the per-frame data needed for
    reverse projection (voxel → image plane). Returns None if unusable."""
    if frame.get("color") is None:
        return None
    width = int(frame["width"])
    height = int(frame["height"])
    cw = int(frame["color_width"])
    ch = int(frame["color_height"])
    if cw == 0 or ch == 0 or width == 0 or height == 0:
        return None
    depth = fusion.decode_depth(
        frame["depth"], width, height,
        int(frame["format"]), float(frame["rawValueToMeters"]),
    ).astype(np.float32)
    color_arr = (
        np.frombuffer(frame["color"], dtype=np.uint8)
        .reshape(ch, cw, 4)[..., :3]
        .copy()
    )
    V  = fusion._mat4_from_column_major(frame["viewMatrix"])
    P  = fusion._mat4_from_column_major(frame["projectionMatrix"])
    Bd = fusion._mat4_from_column_major(frame["normDepthBufferFromNormView"])
    return {
        "depth": depth,
        "color": color_arr,
        "width": width, "height": height, "cw": cw, "ch": ch,
        "V_inv": np.linalg.inv(V),
        "P": P, "Bd": Bd,
        "cam_origin": V[:3, 3].astype(np.float64),
    }


def _project_voxels_to_frame(pts3, frame, near, far, tol):
    """Project a batch of voxel centres `pts3` (N, 3 in world) onto a frame.

    Returns (col_mask, air_mask, colors). For each voxel and frame:
      * skip (mask False on both) if outside the image, behind the surface
        (dist > d + tol), out of [near, far+tol], or no valid depth.
      * col_mask True if |dist − d| ≤ tol; `colors[i]` carries the bilinear
        RGB sample at the projected pixel.
      * air_mask True if dist < d − tol.
    """
    cam = frame["cam_origin"]
    V_inv = frame["V_inv"]
    P  = frame["P"]
    Bd = frame["Bd"]
    depth_buf = frame["depth"]
    color_buf = frame["color"]
    cw, ch = frame["cw"], frame["ch"]
    h, w = depth_buf.shape

    N = pts3.shape[0]
    pts4 = np.empty((N, 4), dtype=np.float64)
    pts4[:, :3] = pts3
    pts4[:, 3] = 1.0

    view_h = pts4 @ V_inv.T
    clip_h = view_h @ P.T
    w_h = clip_h[:, 3]
    safe_w = np.where(np.abs(w_h) > 1e-9, w_h, 1.0)
    ndc_x = clip_h[:, 0] / safe_w
    ndc_y = clip_h[:, 1] / safe_w

    u = 0.5 * (ndc_x + 1.0)
    v = 0.5 * (ndc_y + 1.0)
    in_view = (
        (np.abs(w_h) > 1e-9)
        & (view_h[:, 2] < 0.0)            # camera looks down −z (matches forward path)
        & (u >= 0.0) & (u <= 1.0)
        & (v >= 0.0) & (v <= 1.0)
    )

    diff = pts3 - cam[None, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    in_range = (dist >= near) & (dist <= far + tol)

    # Map (u, v) → depth-buffer pixel coords, matching forward path.
    nv = np.empty((N, 4), dtype=np.float64)
    nv[:, 0] = u; nv[:, 1] = v; nv[:, 2] = 0.0; nv[:, 3] = 1.0
    nd = nv @ Bd.T
    nd_w = nd[:, 3]
    safe_ndw = np.where(np.abs(nd_w) > 1e-9, nd_w, 1.0)
    u_d = nd[:, 0] / safe_ndw
    v_d = nd[:, 1] / safe_ndw
    bx = (1.0 - u_d) * w
    by = v_d * h
    in_buf = (bx >= 0.0) & (bx <= w - 1.0) & (by >= 0.0) & (by <= h - 1.0)

    # Bilinear depth sample.
    bxc = np.clip(bx, 0.0, float(w - 1) - 1e-3)
    byc = np.clip(by, 0.0, float(h - 1) - 1e-3)
    bx0 = bxc.astype(np.int32); by0 = byc.astype(np.int32)
    bx1 = np.minimum(bx0 + 1, w - 1); by1 = np.minimum(by0 + 1, h - 1)
    fx = (bxc - bx0).astype(np.float32)
    fy = (byc - by0).astype(np.float32)
    d00 = depth_buf[by0, bx0]; d10 = depth_buf[by0, bx1]
    d01 = depth_buf[by1, bx0]; d11 = depth_buf[by1, bx1]
    depth_est = ((d00 * (1.0 - fx) + d10 * fx) * (1.0 - fy)
                 + (d01 * (1.0 - fx) + d11 * fx) * fy)
    valid_depth = (depth_est > near) & (depth_est < far)

    visible = in_view & in_range & in_buf & valid_depth
    delta = dist.astype(np.float32) - depth_est
    col_mask = visible & (np.abs(delta) <= np.float32(tol))
    air_mask = visible & (delta < -np.float32(tol))
    # delta > tol (behind surface) → not in either mask → discarded for this frame.

    # Bilinear colour sample. Computed for every voxel (cheap), used only
    # where col_mask is set.
    cx_pix = u * cw - 0.5
    cy_pix = v * ch - 0.5
    cxc = np.clip(cx_pix, 0.0, float(cw - 1) - 1e-3)
    cyc = np.clip(cy_pix, 0.0, float(ch - 1) - 1e-3)
    cx0 = cxc.astype(np.int32); cy0 = cyc.astype(np.int32)
    cx1 = np.minimum(cx0 + 1, cw - 1); cy1 = np.minimum(cy0 + 1, ch - 1)
    bxf = (cxc - cx0).astype(np.float32)
    byf = (cyc - cy0).astype(np.float32)
    c00 = color_buf[cy0, cx0].astype(np.float32)
    c10 = color_buf[cy0, cx1].astype(np.float32)
    c01 = color_buf[cy1, cx0].astype(np.float32)
    c11 = color_buf[cy1, cx1].astype(np.float32)
    colors = (
        (c00 * (1 - bxf[:, None]) + c10 * bxf[:, None]) * (1 - byf[:, None])
        + (c01 * (1 - bxf[:, None]) + c11 * bxf[:, None]) * byf[:, None]
    )
    return col_mask, air_mask, colors


_RV_FRAMES = None


def _reverse_worker_init(pickle_path: str) -> None:
    """Pool initializer: load decoded frames once per worker."""
    import pickle
    global _RV_FRAMES
    with open(pickle_path, "rb") as f:
        _RV_FRAMES = pickle.load(f)


def _reverse_worker(args):
    """Process a single voxel chunk against all frames. A chunk is a set of
    `ix` slabs spread across the x dimension (modulo n_chunks). Splitting the
    work into many small chunks (rather than one chunk per worker) lets the
    parent stream per-chunk progress as workers finish, instead of going
    silent for the entire reconstruction."""
    (chunk_id, ix_list, voxel_size, world_min,
     shape, near, far, tol) = args
    Nx, Ny, Nz = shape
    ix_set = np.asarray(ix_list, dtype=np.int32)
    Mx = int(len(ix_set))
    if Mx == 0 or _RV_FRAMES is None:
        return chunk_id, ix_set, \
               np.zeros(0, dtype=np.uint32), \
               np.zeros(0, dtype=np.uint32), \
               np.zeros((0, 3), dtype=np.float64)

    iy = np.arange(Ny, dtype=np.int32)
    iz = np.arange(Nz, dtype=np.int32)
    IX, IY, IZ = np.meshgrid(ix_set, iy, iz, indexing="ij")
    grid_origin = np.asarray(world_min, dtype=np.float64)
    cx = grid_origin[0] + (IX + 0.5) * voxel_size
    cy = grid_origin[1] + (IY + 0.5) * voxel_size
    cz = grid_origin[2] + (IZ + 0.5) * voxel_size
    pts3 = np.stack([cx.ravel(), cy.ravel(), cz.ravel()], axis=-1).astype(np.float64)
    Mtot = pts3.shape[0]

    color_count = np.zeros(Mtot, dtype=np.uint32)
    air_count   = np.zeros(Mtot, dtype=np.uint32)
    color_sum   = np.zeros((Mtot, 3), dtype=np.float64)

    for frame in _RV_FRAMES:
        col_mask, air_mask, colors = _project_voxels_to_frame(
            pts3, frame, near, far, tol,
        )
        if col_mask.any():
            color_count[col_mask] += 1
            color_sum[col_mask] += colors[col_mask]
        if air_mask.any():
            air_count[air_mask] += 1

    return chunk_id, ix_set, color_count, air_count, color_sum


def reconstruct_reverse(
    frames_dir: Path,
    out_path: Path,
    *,
    voxel_size: float = 0.05,
    world_min: tuple[float, float, float] = (-2.5, -0.3, -2.5),
    world_max: tuple[float, float, float] = (2.5, 4.7, 2.5),
    near: float = 0.05,
    far: float = 8.0,
    tol: float = 0.20,
    threshold: float = 0.10,
    min_color_count: int = 1,
    max_frames: int | None = None,
    workers: int = 1,
) -> None:
    """Reverse-projection reconstruction.

    Iterates voxels (sliced by `ix % workers`) and, for each frame, projects
    the voxel centre to the image plane:
      * out-of-frame, behind-surface (dist > d+tol), or no valid depth → skip.
      * |dist − d| ≤ tol → counts as colour for this voxel; bilinear RGB sample.
      * dist < d − tol  → counts as air for this voxel.
    Then the standard ratio threshold + min_color_count gate decides which
    voxels are written out.
    """
    wmin = np.asarray(world_min, dtype=np.float64)
    wmax = np.asarray(world_max, dtype=np.float64)
    shape = tuple(int(np.ceil((wmax[i] - wmin[i]) / voxel_size)) for i in range(3))
    Nx, Ny, Nz = shape
    Ntot = Nx * Ny * Nz
    print(f"Voxel grid: {shape}  ({Ntot:,} voxels, {voxel_size*100:.1f} cm/edge)")
    print(f"world bbox: {wmin} → {wmax}")
    print(f"Reverse mode: iterating voxels, projecting onto each frame")

    frame_paths = sorted(frames_dir.glob("frame_*.bin"))
    if max_frames:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        print(f"No frames found in {frames_dir}")
        return

    print(f"Decoding {len(frame_paths)} frames…")
    frames = []
    t_dec = time.time()
    for fp in frame_paths:
        try:
            frame = serve.parse_frame(fp.read_bytes())
        except Exception as e:  # noqa: BLE001
            print(f"  parse error on {fp.name}: {e}")
            continue
        prep = _prepare_frame_reverse(frame, near, far)
        if prep is not None:
            frames.append(prep)
    print(f"  decoded {len(frames)}/{len(frame_paths)} frames in {time.time()-t_dec:.1f} s")
    if not frames:
        print("No usable frames.")
        return

    color_count = np.zeros(Ntot, dtype=np.uint32)
    air_count   = np.zeros(Ntot, dtype=np.uint32)
    color_sum   = np.zeros((Ntot, 3), dtype=np.float64)
    Ny_Nz = Ny * Nz

    t0 = time.time()
    if workers <= 1:
        ix_set = np.arange(Nx, dtype=np.int32)
        iy = np.arange(Ny, dtype=np.int32)
        iz = np.arange(Nz, dtype=np.int32)
        IX, IY, IZ = np.meshgrid(ix_set, iy, iz, indexing="ij")
        cx = wmin[0] + (IX + 0.5) * voxel_size
        cy = wmin[1] + (IY + 0.5) * voxel_size
        cz = wmin[2] + (IZ + 0.5) * voxel_size
        pts3 = np.stack([cx.ravel(), cy.ravel(), cz.ravel()], axis=-1).astype(np.float64)
        for fi, frame in enumerate(frames):
            col_mask, air_mask, colors = _project_voxels_to_frame(
                pts3, frame, near, far, tol,
            )
            if col_mask.any():
                color_count[col_mask] += 1
                color_sum[col_mask] += colors[col_mask]
            if air_mask.any():
                air_count[air_mask] += 1
            if (fi + 1) % 5 == 0 or fi == 0 or fi == len(frames) - 1:
                elapsed = time.time() - t0
                eta = (elapsed / (fi + 1)) * (len(frames) - fi - 1)
                print(f"  frame {fi+1:4d}/{len(frames)}: "
                      f"running color {int(color_count.sum()):,} "
                      f"air {int(air_count.sum()):,}; "
                      f"elapsed {elapsed:.1f} s; ETA {eta:.1f} s")
    else:
        import pickle
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fp:
            pickle.dump(frames, fp, protocol=pickle.HIGHEST_PROTOCOL)
            pickle_path = fp.name
        size_mb = os.path.getsize(pickle_path) / 1e6
        print(f"  pickled frames to {pickle_path} ({size_mb:.1f} MB)")
        # Free the parent's frame copies so workers don't double up RAM.
        frames.clear()
        try:
            # Split the x range into many small chunks (~4 per worker) so
            # the ex.map() result stream fires often enough to surface a
            # live ETA. Each chunk takes ix values modulo n_chunks, which
            # keeps slabs spread along x and the per-chunk work even.
            chunks_per_worker = 4
            n_chunks = min(max(workers * chunks_per_worker, workers), Nx)
            ix_chunks = [
                np.arange(cid, Nx, n_chunks, dtype=np.int32).tolist()
                for cid in range(n_chunks)
            ]
            print(f"  dispatching {n_chunks} chunks across {workers} workers "
                  f"(~{Nx / n_chunks:.1f} ix slabs/chunk)")
            args_iter = [
                (cid, ix_chunk, voxel_size, tuple(wmin.tolist()),
                 shape, near, far, tol)
                for cid, ix_chunk in enumerate(ix_chunks)
            ]
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_reverse_worker_init,
                initargs=(pickle_path,),
            ) as ex:
                done = 0
                for chunk_id, ix_set, cc, ac, cs_w in ex.map(
                    _reverse_worker, args_iter,
                ):
                    done += 1
                    Mx = int(len(ix_set))
                    if Mx > 0:
                        flat_idx = (
                            ix_set[:, None, None] * Ny_Nz
                            + np.arange(Ny)[None, :, None] * Nz
                            + np.arange(Nz)[None, None, :]
                        ).ravel()
                        color_count[flat_idx] += cc
                        air_count[flat_idx]   += ac
                        color_sum[flat_idx]   += cs_w
                    elapsed = time.time() - t0
                    eta = (elapsed / done) * (n_chunks - done) if done else 0.0
                    pct = 100.0 * done / n_chunks
                    print(f"  chunk {done:3d}/{n_chunks} ({pct:5.1f}%); "
                          f"elapsed {elapsed:6.1f} s; ETA {eta:6.1f} s; "
                          f"running color {int(color_count.sum()):,} "
                          f"air {int(air_count.sum()):,}",
                          flush=True)
        finally:
            try:
                os.unlink(pickle_path)
            except OSError:
                pass

    print(f"\nAll voxels processed in {time.time()-t0:.1f} s")

    total = color_count.astype(np.float64) + air_count.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total > 0,
                         color_count.astype(np.float64) / np.maximum(total, 1),
                         0.0)
    keep = (color_count >= min_color_count) & (ratio >= threshold)
    n_kept = int(keep.sum())
    print(f"Kept {n_kept:,} of {Ntot:,} voxels "
          f"(ratio >= {threshold}, color_count >= {min_color_count})")

    if n_kept == 0:
        print("Nothing to write.")
        return

    safe_count = np.maximum(color_count[keep, None], 1).astype(np.float64)
    avg_color = (color_sum[keep] / safe_count).clip(0, 255).astype(np.uint8)
    flat_idx = np.nonzero(keep)[0]
    iz_o = flat_idx % Nz
    iy_o = (flat_idx // Nz) % Ny
    ix_o = flat_idx // (Ny * Nz)
    payload = {
        "voxel_size": voxel_size,
        "world_min": wmin.tolist(),
        "world_max": (wmin + np.array(shape) * voxel_size).tolist(),
        "shape": list(shape),
        "threshold": threshold,
        "n_voxels": n_kept,
        "indices": np.stack([ix_o, iy_o, iz_o], axis=-1).astype(int).tolist(),
        "colors":  avg_color.astype(int).tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    print(f"Wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", default="captured_frames",
                    help="directory with frame_*.bin captures")
    ap.add_argument("--out", default="web/out/voxels.json",
                    help="output JSON path")
    ap.add_argument("--voxel-size", type=float, default=0.05,
                    help="voxel edge in metres (default 0.05 = 5 cm)")
    ap.add_argument("--world-min", type=float, nargs=3, default=[-2.5, -0.3, -2.5])
    ap.add_argument("--world-max", type=float, nargs=3, default=[ 2.5,  4.7,  2.5])
    ap.add_argument("--near", type=float, default=0.05)
    ap.add_argument("--far", type=float, default=8.0)
    ap.add_argument("--tol", type=float, default=0.20,
                    help="half-width of the colour band around the depth surface (m)")
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="keep voxels with color_count/(color_count+air_count) ≥ this")
    ap.add_argument("--subsample", type=int, default=4,
                    help="take every Nth pixel of the RGB image (1 = all pixels)")
    ap.add_argument("--chunk-size", type=int, default=200_000,
                    help="pixels processed per inner-loop chunk")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="process only the first N frames (debug)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="number of worker processes (1 disables multiprocessing). "
                         "Default = cpu_count − 2.")
    ap.add_argument("--no-air", action="store_true",
                    help="skip filling air along rays before the sensed depth. "
                         "Restricts per-ray sampling to a short window around d "
                         "(faster); no air ratio is computed. Forward mode only.")
    ap.add_argument("--min-color-count", type=int, default=1,
                    help="minimum number of rays that must place colour at a "
                         "voxel for it to be kept (default 1).")
    ap.add_argument("--reverse", action="store_true",
                    help="reverse mode: iterate voxels and project them onto "
                         "frames (instead of shooting rays per pixel). Workers "
                         "split the grid by ix %% n_workers.")
    args = ap.parse_args()

    if args.reverse:
        reconstruct_reverse(
            Path(args.frames_dir), Path(args.out),
            voxel_size=args.voxel_size,
            world_min=tuple(args.world_min),
            world_max=tuple(args.world_max),
            near=args.near, far=args.far, tol=args.tol,
            threshold=args.threshold,
            min_color_count=args.min_color_count,
            max_frames=args.max_frames,
            workers=args.workers,
        )
        return

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
        workers=args.workers,
        no_air=args.no_air,
        min_color_count=args.min_color_count,
    )


if __name__ == "__main__":
    main()
