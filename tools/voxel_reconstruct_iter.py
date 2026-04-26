#!/usr/bin/env python3
"""
Iterative voxel reconstruction from captured WebXR frames.

Same machinery as voxel_reconstruct.py for a single pass — see that module's
docstring for the air/colour accounting. This tool repeats the pass:

  iter 1:  use the captured depth `d_orig` per pixel as the depth hint.
  iter k:  for every ray, walk the grid built by iter k−1 until we hit
           a voxel that was *kept* (passed the threshold). If we hit one
           at distance `t_hit`, use that as the depth hint instead of
           `d_orig`. If no kept voxel sits on the ray, fall back to
           `d_orig`.

Each iteration zeros the counters and re-runs the full air/colour scan,
so the final grid is built entirely from the most-recent (refined) depth
hints. The hope is that the kept mask quickly converges to a sharper
surface — wherever multiple frames already agree on a surface position,
that consensus pulls each individual ray's contribution onto the same
voxel column instead of letting noisy depth smear them along the ray.

Output: `web/out/voxels_iter.json` (same format as voxels.json) — load
it by pointing voxelview's URL bar at `?src=voxels_iter.json` once that
hook is added, or simply rename the output file to voxels.json.
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
    iterations: int = 2,
) -> None:
    wmin = np.asarray(world_min, dtype=np.float64)
    wmax = np.asarray(world_max, dtype=np.float64)
    shape = tuple(int(np.ceil((wmax[i] - wmin[i]) / voxel_size)) for i in range(3))
    Nx, Ny, Nz = shape
    Ntot = Nx * Ny * Nz
    print(f"Voxel grid: {shape}  ({Ntot:,} voxels, {voxel_size*100:.1f} cm/edge)")
    print(f"world bbox: {wmin} → {wmax}")

    frame_paths = sorted(frames_dir.glob("frame_*.bin"))
    if max_frames:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        print(f"No frames found in {frames_dir}")
        return
    print(f"Processing {len(frame_paths)} frames from {frames_dir} "
          f"(workers={workers}, iterations={iterations})")

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

    # `kept_mask` is the (Nx, Ny, Nz) boolean grid from the previous iteration.
    # Iteration 1 has no prior mask, so depth refinement is a no-op.
    kept_mask: np.ndarray | None = None
    air_count = np.zeros(Ntot, dtype=np.uint32)
    color_count = np.zeros(Ntot, dtype=np.uint32)
    color_sum = np.zeros((Ntot, 3), dtype=np.float64)

    for it in range(1, iterations + 1):
        print(f"\n=== iteration {it}/{iterations} ===")
        air_count.fill(0)
        color_count.fill(0)
        color_sum.fill(0.0)

        t0_iter = time.time()
        run_pass(
            frame_paths, air_count, color_count, color_sum,
            grid_params, workers=workers, kept_mask=kept_mask,
        )
        print(f"  iteration {it} pass took {time.time()-t0_iter:.1f} s")

        # Threshold + summarise.
        total = color_count.astype(np.float64) + air_count.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(total > 0, color_count.astype(np.float64) / np.maximum(total, 1), 0.0)
        keep_flat = (color_count >= 1) & (ratio >= threshold)
        n_kept = int(keep_flat.sum())
        print(f"  iter {it}: kept {n_kept:,} of {Ntot:,} voxels "
              f"(ratio ≥ {threshold}, color_count ≥ 1)")

        if it < iterations:
            # Reshape to 3D so the next iteration can index by (ix, iy, iz).
            kept_mask = keep_flat.reshape(shape)

    if n_kept == 0:
        print("Nothing to write.")
        return

    safe_count = np.maximum(color_count[keep_flat, None], 1).astype(np.float64)
    avg_color = (color_sum[keep_flat] / safe_count).clip(0, 255).astype(np.uint8)

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
        "n_voxels": n_kept,
        "indices": np.stack([ix, iy, iz], axis=-1).astype(int).tolist(),
        "colors":  avg_color.astype(int).tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    print(f"Wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


def run_pass(frame_paths, air_count, color_count, color_sum, grid_params,
             *, workers: int, kept_mask: np.ndarray | None) -> None:
    """One full sweep over `frame_paths`, accumulating into the supplied
    counter buffers. If `kept_mask` is given, each ray's depth hint is
    replaced by the distance to the first kept voxel along the ray (or the
    original depth if no kept voxel lies on the ray)."""
    if workers <= 1:
        for fi, fp in enumerate(frame_paths):
            try:
                body = Path(fp).read_bytes()
                frame = serve.parse_frame(body)
            except Exception as e:  # noqa: BLE001
                print(f"    frame {fi}: parse error {e}")
                continue
            if frame["color"] is None:
                continue
            rays_done = process_frame(
                frame, air_count, color_count, color_sum,
                grid_params["world_min"], grid_params["voxel_size"],
                grid_params["shape"], grid_params["step"],
                near=grid_params["near"], far=grid_params["far"],
                tol=grid_params["tol"],
                subsample=grid_params["subsample"],
                chunk_size=grid_params["chunk_size"],
                n_steps=grid_params["n_steps"],
                kept_mask=kept_mask,
            )
            if (fi + 1) % 10 == 0 or fi == len(frame_paths) - 1:
                print(f"    frame {fi+1:4d}/{len(frame_paths)}: {rays_done:7d} rays")
        return

    batches = np.array_split(np.array(frame_paths, dtype=object), workers)
    batches = [list(b) for b in batches if len(b) > 0]
    print(f"    splitting into {len(batches)} batches "
          f"(~{len(batches[0])} frames/worker)")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_worker_process_batch,
                      [str(p) for p in batch], grid_params, kept_mask): bi
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
            print(f"    batch {bi+1:2d}/{len(batches)} merged "
                  f"({n_ok} frames in {dt_w:.1f} s; "
                  f"merged {done}/{len(batches)})")


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
                 kept_mask: np.ndarray | None = None) -> int:
    """Vectorised numpy backend.

    For each chunk of pixels, builds (chunk × n_steps) sample arrays in one
    shot and runs five np.bincount calls over the masked indices — one for
    air, one for colour count, three (R/G/B) for the colour sum.

    If `kept_mask` (shape (Nx, Ny, Nz), bool) is provided, before the
    air/colour scan we replace each ray's depth hint with the distance
    along the ray to the first kept voxel. Rays with no kept voxel keep
    their original depth.
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
    grid_origin_f32 = grid_origin.astype(np.float32)

    for cs in range(0, M, chunk_size):
        ce = min(cs + chunk_size, M)
        rays_c = rays_v[cs:ce]                                    # (N, 3) f32
        depth_c = depth_v[cs:ce].copy()                           # (N,)   f32 (will refine in-place)
        rgb_c = rgb_v[cs:ce]                                      # (N, 3) f32
        N = ce - cs

        # Sample positions: (N, S, 3) world, (N, S, 3) voxel index.
        wp = cam_origin[None, None, :] + t_vals[None, :, None] * rays_c[:, None, :]
        vp = (wp - grid_origin_f32[None, None, :]) / np.float32(voxel_size)
        vidx = np.floor(vp).astype(np.int32)                      # (N, S, 3)
        vx = vidx[..., 0]; vy = vidx[..., 1]; vz = vidx[..., 2]
        in_grid = (
            (vx >= 0) & (vx < Nx) &
            (vy >= 0) & (vy < Ny) &
            (vz >= 0) & (vz < Nz)
        )

        # Depth refinement: for each ray, find the first sample whose voxel
        # is "kept" in the previous-iteration mask. That sample's t becomes
        # the refined depth; rays with no kept voxel keep their original
        # depth from the captured frame.
        if kept_mask is not None:
            vx_safe = np.clip(vx, 0, Nx - 1)
            vy_safe = np.clip(vy, 0, Ny - 1)
            vz_safe = np.clip(vz, 0, Nz - 1)
            kept_at = kept_mask[vx_safe, vy_safe, vz_safe] & in_grid   # (N, S)
            has_hit = kept_at.any(axis=1)
            # argmax returns the FIRST True index along axis 1; falls back to
            # 0 if no True, but we mask that out via has_hit.
            first_hit = kept_at.argmax(axis=1)
            t_refined = t_vals[first_hit]
            depth_c = np.where(has_hit, t_refined, depth_c).astype(np.float32)

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
            mi, _si = np.nonzero(col_mask)                          # both (N_col,)
            cols = rgb_c[mi]                                       # (N_col, 3) f32
            color_sum[:, 0] += np.bincount(idx_col, weights=cols[:, 0], minlength=Ntot)
            color_sum[:, 1] += np.bincount(idx_col, weights=cols[:, 1], minlength=Ntot)
            color_sum[:, 2] += np.bincount(idx_col, weights=cols[:, 2], minlength=Ntot)

    return M


# ----- multiprocessing worker --------------------------------------------

def _worker_process_batch(frame_paths: list[str], grid_params: dict,
                          kept_mask: np.ndarray | None):
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
            kept_mask=kept_mask,
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
                    help="output JSON path (overwrites the file the standard "
                         "voxel viewer loads — give a different path with "
                         "--out if you want to keep the single-pass result)")
    ap.add_argument("--iterations", type=int, default=2,
                    help="number of refinement passes. 1 = behave exactly "
                         "like voxel_reconstruct.py. Default 2.")
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
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
