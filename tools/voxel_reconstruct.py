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

import argparse
import json
import sys
import time
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
    print(f"Processing {len(frame_paths)} frames from {frames_dir}")

    step = voxel_size * 0.5
    t0_total = time.time()

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
        )
        dt = time.time() - t_frame
        if (fi + 1) % 5 == 0 or fi == 0 or fi == len(frame_paths) - 1:
            tot_color = int(color_count.sum())
            tot_air = int(air_count.sum())
            print(f"  frame {fi+1:4d}/{len(frame_paths)}: "
                  f"{rays_processed:7d} rays in {dt*1000:.0f} ms; "
                  f"running totals: color {tot_color:,}, air {tot_air:,}")

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


def process_frame(frame, air_count, color_count, color_sum,
                 grid_origin, voxel_size, shape, step,
                 *, near, far, tol, subsample, chunk_size) -> int:
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
        return 0

    color_arr = np.frombuffer(color_payload, dtype=np.uint8).reshape(ch, cw, 4)
    # gl.readPixels rows are bottom-up: color_arr[0] is the bottom row, which
    # corresponds to GL view-V at v=0. Keep that convention here so projection
    # math (y_ndc = 2v − 1 with bottom-up v) lines up with sampling.

    V  = fusion._mat4_from_column_major(frame["viewMatrix"])              # world_from_view
    P  = fusion._mat4_from_column_major(frame["projectionMatrix"])
    Bd = fusion._mat4_from_column_major(frame["normDepthBufferFromNormView"])
    P_inv = np.linalg.inv(P)

    cam_origin = V[:3, 3].astype(np.float64)

    # Subsample the colour image in pixel space; preserve original UV coords
    # so the projection math is unchanged.
    color_sub = color_arr[::subsample, ::subsample]
    sh, sw = color_sub.shape[:2]

    j_grid, i_grid = np.meshgrid(np.arange(sh), np.arange(sw), indexing="ij")
    u = (i_grid * subsample + 0.5 * subsample) / cw   # GL u  ∈ [0, 1]
    v = (j_grid * subsample + 0.5 * subsample) / ch

    # NDC for the pixel centre, on the near plane.
    x_ndc = 2.0 * u - 1.0
    y_ndc = 2.0 * v - 1.0
    clip = np.stack(
        [x_ndc, y_ndc, -np.ones_like(x_ndc), np.ones_like(x_ndc)], axis=-1
    )
    view4 = clip @ P_inv.T
    view3 = view4[..., :3] / np.where(np.abs(view4[..., 3:4]) < 1e-12, 1.0, view4[..., 3:4])
    view3_h = np.concatenate([view3, np.ones((sh, sw, 1))], axis=-1)
    world_h = view3_h @ V.T
    world3 = world_h[..., :3] / np.where(np.abs(world_h[..., 3:4]) < 1e-12, 1.0, world_h[..., 3:4])
    ray_dirs = world3 - cam_origin[None, None, :]
    ray_lens = np.linalg.norm(ray_dirs, axis=-1, keepdims=True)
    ray_dirs = ray_dirs / np.maximum(ray_lens, 1e-9)

    # Look up depth at each pixel through the depth-buffer mapping.
    nv_h = np.stack([u, v, np.zeros_like(u), np.ones_like(u)], axis=-1)
    nd_h = nv_h @ Bd.T
    u_d = nd_h[..., 0] / np.where(np.abs(nd_h[..., 3]) < 1e-12, 1.0, nd_h[..., 3])
    v_d = nd_h[..., 1] / np.where(np.abs(nd_h[..., 3]) < 1e-12, 1.0, nd_h[..., 3])
    bx = np.floor((1.0 - u_d) * width).astype(np.int32)   # column-flip per scan.js
    by = np.floor(v_d * height).astype(np.int32)
    in_buf = (bx >= 0) & (bx < width) & (by >= 0) & (by < height)
    bx_c = np.clip(bx, 0, width - 1)
    by_c = np.clip(by, 0, height - 1)
    depth_est = depth[by_c, bx_c]

    valid = in_buf & (depth_est > near) & (depth_est < far)

    rgb = color_sub[..., :3]                              # (sh, sw, 3) uint8

    rays_flat = ray_dirs.reshape(-1, 3)
    depth_flat = depth_est.reshape(-1)
    valid_flat = valid.reshape(-1)
    rgb_flat = rgb.reshape(-1, 3)

    rays_v = rays_flat[valid_flat]
    depth_v = depth_flat[valid_flat]
    rgb_v = rgb_flat[valid_flat]
    M = rays_v.shape[0]
    if M == 0:
        return 0

    Nx, Ny, Nz = shape
    Ntot = Nx * Ny * Nz

    for cs in range(0, M, chunk_size):
        ce = min(cs + chunk_size, M)
        rays_c = rays_v[cs:ce]
        depth_c = depth_v[cs:ce]
        rgb_c = rgb_v[cs:ce]

        max_t_chunk = depth_c.max() + tol
        n_steps = int(np.ceil((max_t_chunk - near) / step)) + 1

        for s in range(n_steps):
            t = near + (s + 0.5) * step
            if t > max_t_chunk:
                break

            wp = cam_origin[None, :] + t * rays_c
            vp = (wp - grid_origin[None, :]) / voxel_size
            vx = np.floor(vp[:, 0]).astype(np.int32)
            vy = np.floor(vp[:, 1]).astype(np.int32)
            vz = np.floor(vp[:, 2]).astype(np.int32)
            in_grid = (vx >= 0) & (vx < Nx) & (vy >= 0) & (vy < Ny) & (vz >= 0) & (vz < Nz)

            d_band = depth_c
            air_mask = (t < d_band - tol) & in_grid
            col_mask = (np.abs(t - d_band) <= tol) & in_grid

            if np.any(air_mask):
                idx = vx[air_mask] * (Ny * Nz) + vy[air_mask] * Nz + vz[air_mask]
                air_count[:] += np.bincount(idx, minlength=Ntot).astype(np.uint32)

            if np.any(col_mask):
                idx = vx[col_mask] * (Ny * Nz) + vy[col_mask] * Nz + vz[col_mask]
                color_count[:] += np.bincount(idx, minlength=Ntot).astype(np.uint32)
                cols = rgb_c[col_mask].astype(np.float64)
                color_sum[:, 0] += np.bincount(idx, weights=cols[:, 0], minlength=Ntot)
                color_sum[:, 1] += np.bincount(idx, weights=cols[:, 1], minlength=Ntot)
                color_sum[:, 2] += np.bincount(idx, weights=cols[:, 2], minlength=Ntot)

    return M


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
    )


if __name__ == "__main__":
    main()
