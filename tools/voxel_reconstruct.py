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

    # Threshold + average.
    total = color_count.astype(np.float64) + air_count.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total > 0, color_count.astype(np.float64) / np.maximum(total, 1), 0.0)
    keep = (color_count >= 1) & (ratio >= threshold)
    n_kept = int(keep.sum())
    print(f"Kept {n_kept:,} of {Ntot:,} voxels "
          f"(ratio >= {threshold}, color_count >= 1)")

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
                 *, near, far, tol, subsample, chunk_size, n_steps) -> int:
    """Vectorised numpy backend.

    For each chunk of pixels, builds (chunk × n_steps) sample arrays in one
    shot and runs five np.bincount calls over the masked indices — one for
    air, one for colour count, three (R/G/B) for the colour sum. That's far
    fewer Python-level loop iterations than the per-step variant, and lets
    numpy's internal C loops do the heavy lifting.
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

    t_vals = (near + (np.arange(n_steps, dtype=np.float32) + 0.5) * step).astype(np.float32)

    for cs in range(0, M, chunk_size):
        ce = min(cs + chunk_size, M)
        rays_c = rays_v[cs:ce]                                    # (N, 3) f32
        depth_c = depth_v[cs:ce]                                  # (N,)   f32
        rgb_c = rgb_v[cs:ce]                                      # (N, 3) f32
        N = ce - cs

        # Sample positions: (N, S, 3) world, (N, S, 3) voxel index.
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

        # Air: one bincount over all (chunk × steps) air samples.
        if air_mask.any():
            idx_air = flat[air_mask]
            air_count[:] += np.bincount(idx_air, minlength=Ntot).astype(np.uint32)

        if col_mask.any():
            idx_col = flat[col_mask]
            color_count[:] += np.bincount(idx_col, minlength=Ntot).astype(np.uint32)
            # rgb_b shape (N, S, 3); pick rows by col_mask.
            # Index trick: broadcast rgb_c (N, 3) to (N, S, 3) implicitly via
            # repeat with col_mask masking. Cheaper to repeat then index.
            mi, si = np.nonzero(col_mask)                          # both (N_col,)
            cols = rgb_c[mi]                                       # (N_col, 3) f32
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
        )
        if rays_done > 0:
            n_ok += 1
    return air, cc, cs, n_ok, time.time() - t0


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
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
