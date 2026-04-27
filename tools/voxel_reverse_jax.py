#!/usr/bin/env python3
"""
JAX reverse-projection voxel reconstruction.

Same algorithm as `voxel_reconstruct.py --reverse`, but the per-voxel work
runs through JAX (XLA-compiled). Reverse mode is gather-only (project,
bilinear sample of depth + colour, classify, accumulate per voxel), so the
fused JIT'd kernel is much tighter than the equivalent numpy code.

Pipeline:
  1. Decode all frames on the host (numpy).
  2. Stack frame matrices + depth/colour buffers into device-side jnp arrays.
  3. For each voxel chunk, Python-loop over frames calling a JIT'd
     `_step_one_frame` that adds (col_mask, air_mask, mask*colors) to
     per-voxel running counters.
  4. Pull final counters back to host, threshold, write JSON.

Backend: defaults to CPU. With jax-metal 0.1.1, this kernel does not run
reliably on Metal:
  * `lax.scan` over the projection step → SIGSEGV in the MPS client.
  * `vmap` over frames → hangs at 0% CPU during compile/dispatch.
  * Python-for over JIT'd `_step_one_frame` → runs, but each Metal
    dispatch carries ~4 s of fixed overhead regardless of voxel count,
    so a 50-frame run takes minutes instead of seconds.
The XLA-CPU path on the same code produces a 50-frame × 1 M-voxel
reconstruction in ~1 s, which is faster than the multiprocess numpy
`--reverse` baseline. Override the default with
`JAX_PLATFORMS=METAL python tools/voxel_reverse_jax.py …` to retry once
jax-metal stabilises.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Configure backend before `import jax`. CPU is the default because
# jax-metal 0.1.1 doesn't run this kernel reliably; opt back in with
# `JAX_PLATFORMS=METAL python tools/voxel_reverse_jax.py …`. Note:
# jax-metal registers as "METAL" (uppercase); plain "metal" silently
# falls back to CPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import fusion  # noqa: F401  (kept for symmetry with voxel_reconstruct)
import serve
from voxel_reconstruct import _prepare_frame_reverse  # reuses host-side decoding

import jax
import jax.numpy as jnp


# ----------------------------------------------------------------------
# Frame staging on device.
# ----------------------------------------------------------------------

def _stack_frames_to_device(frames):
    """Stack per-frame numpy arrays into device-side jnp arrays.

    All frames in a session are expected to share the same depth and colour
    resolutions; we assert that here so a mismatch fails loudly rather than
    silently sampling the wrong buffer.
    """
    h, w = frames[0]["depth"].shape
    ch_dim, cw_dim, _ = frames[0]["color"].shape
    for i, f in enumerate(frames[1:], start=1):
        if f["depth"].shape != (h, w):
            raise ValueError(
                f"frame {i} depth shape {f['depth'].shape} != "
                f"frame 0 ({h},{w}); JAX path needs uniform shapes"
            )
        if f["color"].shape != (ch_dim, cw_dim, 3):
            raise ValueError(
                f"frame {i} color shape {f['color'].shape} != "
                f"frame 0 ({ch_dim},{cw_dim},3); JAX path needs uniform shapes"
            )

    V_inv = jnp.asarray(np.stack([f["V_inv"]    for f in frames]).astype(np.float32))
    P     = jnp.asarray(np.stack([f["P"]        for f in frames]).astype(np.float32))
    Bd    = jnp.asarray(np.stack([f["Bd"]       for f in frames]).astype(np.float32))
    cam   = jnp.asarray(np.stack([f["cam_origin"] for f in frames]).astype(np.float32))
    depth = jnp.asarray(np.stack([f["depth"]    for f in frames]).astype(np.float32))
    # Keep colour as uint8 on device (4× smaller than float32) and convert
    # in the JIT'd step.
    color = jnp.asarray(np.stack([f["color"] for f in frames]))
    return {
        "V_inv": V_inv, "P": P, "Bd": Bd, "cam": cam,
        "depth": depth, "color": color,
        "h": h, "w": w, "ch": ch_dim, "cw": cw_dim,
        "n_frames": len(frames),
    }


# ----------------------------------------------------------------------
# Per-frame projection (pure-jnp helper used inside the scan body).
# ----------------------------------------------------------------------

def _project_and_sample(pts3, V_inv, P, Bd, cam, depth_buf, color_buf,
                        near, far, tol):
    """Pure-jnp projection of pts3 onto one frame. Returns
    (col_mask f32, air_mask f32, colors f32). Both masks are 0/1 floats so
    the caller can multiply directly without boolean→float casts."""
    h, w = depth_buf.shape
    ch_dim, cw_dim, _ = color_buf.shape
    N = pts3.shape[0]

    pts4 = jnp.concatenate([pts3, jnp.ones((N, 1), dtype=pts3.dtype)], axis=-1)
    view_h = pts4 @ V_inv.T
    clip_h = view_h @ P.T
    w_h = clip_h[:, 3]
    safe_w = jnp.where(jnp.abs(w_h) > 1e-9, w_h, 1.0)
    ndc_x = clip_h[:, 0] / safe_w
    ndc_y = clip_h[:, 1] / safe_w

    u = 0.5 * (ndc_x + 1.0)
    v = 0.5 * (ndc_y + 1.0)
    in_view = (
        (jnp.abs(w_h) > 1e-9)
        & (view_h[:, 2] < 0.0)
        & (u >= 0.0) & (u <= 1.0)
        & (v >= 0.0) & (v <= 1.0)
    )

    diff = pts3 - cam[None, :]
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
    in_range = (dist >= near) & (dist <= far + tol)

    nv = jnp.stack([u, v, jnp.zeros_like(u), jnp.ones_like(u)], axis=-1)
    nd = nv @ Bd.T
    nd_w = nd[:, 3]
    safe_ndw = jnp.where(jnp.abs(nd_w) > 1e-9, nd_w, 1.0)
    u_d = nd[:, 0] / safe_ndw
    v_d = nd[:, 1] / safe_ndw
    bx = (1.0 - u_d) * w
    by = v_d * h
    in_buf = (bx >= 0.0) & (bx <= w - 1.0) & (by >= 0.0) & (by <= h - 1.0)

    bxc = jnp.clip(bx, 0.0, float(w - 1) - 1e-3)
    byc = jnp.clip(by, 0.0, float(h - 1) - 1e-3)
    bx0 = bxc.astype(jnp.int32); by0 = byc.astype(jnp.int32)
    bx1 = jnp.minimum(bx0 + 1, w - 1); by1 = jnp.minimum(by0 + 1, h - 1)
    fx = bxc - bx0
    fy = byc - by0
    d00 = depth_buf[by0, bx0]; d10 = depth_buf[by0, bx1]
    d01 = depth_buf[by1, bx0]; d11 = depth_buf[by1, bx1]
    depth_est = ((d00 * (1 - fx) + d10 * fx) * (1 - fy)
                 + (d01 * (1 - fx) + d11 * fx) * fy)
    valid_depth = (depth_est > near) & (depth_est < far)

    visible = in_view & in_range & in_buf & valid_depth
    delta = dist - depth_est
    col_mask = visible & (jnp.abs(delta) <= tol)
    air_mask = visible & (delta < -tol)

    cx_pix = u * cw_dim - 0.5
    cy_pix = v * ch_dim - 0.5
    cxc = jnp.clip(cx_pix, 0.0, float(cw_dim - 1) - 1e-3)
    cyc = jnp.clip(cy_pix, 0.0, float(ch_dim - 1) - 1e-3)
    cx0 = cxc.astype(jnp.int32); cy0 = cyc.astype(jnp.int32)
    cx1 = jnp.minimum(cx0 + 1, cw_dim - 1); cy1 = jnp.minimum(cy0 + 1, ch_dim - 1)
    bxf = cxc - cx0
    byf = cyc - cy0
    c00 = color_buf[cy0, cx0].astype(jnp.float32)
    c10 = color_buf[cy0, cx1].astype(jnp.float32)
    c01 = color_buf[cy1, cx0].astype(jnp.float32)
    c11 = color_buf[cy1, cx1].astype(jnp.float32)
    colors = (
        (c00 * (1 - bxf[:, None]) + c10 * bxf[:, None]) * (1 - byf[:, None])
        + (c01 * (1 - bxf[:, None]) + c11 * bxf[:, None]) * byf[:, None]
    )

    return col_mask.astype(jnp.float32), air_mask.astype(jnp.float32), colors


# ----------------------------------------------------------------------
# Whole-chunk JIT: scan over all frames inside one Metal kernel.
# ----------------------------------------------------------------------
#
# The first version dispatched one JIT call per (chunk, frame). On Metal
# each call carried ~4 s of fixed kernel-launch overhead even when the
# voxel chunk was empty, so 50 frames × 5 chunks made the run unusably
# slow. Folding the per-frame loop into a single `lax.scan` lets XLA fuse
# the whole frame sweep into one kernel; we pay one launch per chunk
# instead of n_frames × n_chunks. Memory stays bounded — only the
# per-voxel running counters cross loop iterations, not the
# (n_frames, N, …) intermediate tensors a vmap would materialise.

@jax.jit
def _step_one_frame(cc, ac, cs, pts3,
                    V_inv, P, Bd, cam, depth_buf, color_buf,
                    near, far, tol):
    """Add one frame's contribution to the running counters."""
    cm, am, colors = _project_and_sample(
        pts3, V_inv, P, Bd, cam, depth_buf, color_buf, near, far, tol,
    )
    return cc + cm, ac + am, cs + cm[:, None] * colors


def _process_voxel_chunk(pts3_np, fs, near, far, tol):
    """Run all frames against one voxel chunk; returns numpy counters.

    JAX-Metal note: `lax.scan` over this kernel SIGSEGVs, and `vmap`
    deadlocks somewhere in compile — both behaviours are documented
    Metal-backend sharp edges for non-trivial kernels with mixed
    matmul + gather + uint8 → float32 promotion. The Python loop with a
    JIT'd inner step actually runs, but each kernel launch carries
    several seconds of fixed overhead on Metal (apparently per
    dispatch, regardless of voxel count). For practical runs the CPU
    `--reverse` path in voxel_reconstruct.py is much faster — keep this
    file as a working baseline / reference for when jax-metal matures.
    """
    pts3 = jnp.asarray(pts3_np.astype(np.float32))
    N = pts3.shape[0]
    cc = jnp.zeros(N, dtype=jnp.float32)
    ac = jnp.zeros(N, dtype=jnp.float32)
    cs = jnp.zeros((N, 3), dtype=jnp.float32)
    near_j = jnp.float32(near)
    far_j  = jnp.float32(far)
    tol_j  = jnp.float32(tol)
    for fi in range(fs["n_frames"]):
        cc, ac, cs = _step_one_frame(
            cc, ac, cs, pts3,
            fs["V_inv"][fi], fs["P"][fi], fs["Bd"][fi], fs["cam"][fi],
            fs["depth"][fi], fs["color"][fi],
            near_j, far_j, tol_j,
        )
    cc.block_until_ready()
    return np.asarray(cc), np.asarray(ac), np.asarray(cs)


# ----------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------

def reconstruct_jax(
    frames_dir: Path,
    out_path: Path,
    *,
    voxel_size: float,
    world_min, world_max,
    near: float, far: float, tol: float,
    threshold: float, min_color_count: int,
    max_frames: int | None,
    voxel_chunk: int,
) -> None:
    print(f"JAX backend: {jax.default_backend()}; devices: {jax.devices()}")

    wmin = np.asarray(world_min, dtype=np.float64)
    wmax = np.asarray(world_max, dtype=np.float64)
    shape = tuple(int(np.ceil((wmax[i] - wmin[i]) / voxel_size)) for i in range(3))
    Nx, Ny, Nz = shape
    Ntot = Nx * Ny * Nz
    Ny_Nz = Ny * Nz
    print(f"Voxel grid: {shape}  ({Ntot:,} voxels, {voxel_size*100:.1f} cm/edge)")
    print(f"world bbox: {wmin} → {wmax}")

    frame_paths = sorted(frames_dir.glob("frame_*.bin"))
    if max_frames:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        print(f"No frames found in {frames_dir}")
        return

    print(f"Decoding {len(frame_paths)} frames…")
    t_dec = time.time()
    frames = []
    for fp in frame_paths:
        try:
            frame = serve.parse_frame(fp.read_bytes())
        except Exception as e:  # noqa: BLE001
            print(f"  parse error on {fp.name}: {e}")
            continue
        prep = _prepare_frame_reverse(frame, near, far)
        if prep is not None:
            frames.append(prep)
    print(f"  decoded {len(frames)}/{len(frame_paths)} frames "
          f"in {time.time()-t_dec:.1f} s")
    if not frames:
        print("No usable frames.")
        return

    print("Uploading frames to device…")
    t_up = time.time()
    fs = _stack_frames_to_device(frames)
    fs["depth"].block_until_ready()
    fs["color"].block_until_ready()
    color_mb = fs["color"].nbytes / 1e6
    depth_mb = fs["depth"].nbytes / 1e6
    print(f"  uploaded {fs['n_frames']} frames: depth {depth_mb:.0f} MB "
          f"+ color {color_mb:.0f} MB in {time.time()-t_up:.1f} s "
          f"(depth {fs['h']}×{fs['w']}, color {fs['ch']}×{fs['cw']})")
    # Free host-side frames now that the device has a copy.
    frames.clear()

    color_count = np.zeros(Ntot, dtype=np.uint32)
    air_count   = np.zeros(Ntot, dtype=np.uint32)
    color_sum   = np.zeros((Ntot, 3), dtype=np.float64)

    voxels_per_x_slab = Ny * Nz
    nx_per_chunk = max(1, voxel_chunk // voxels_per_x_slab)
    n_chunks = int(np.ceil(Nx / nx_per_chunk))
    print(f"Voxel chunks: {n_chunks} chunks × up to {nx_per_chunk} ix slabs "
          f"(~{nx_per_chunk * voxels_per_x_slab:,} voxels/chunk)")

    t_proc = time.time()
    iy_grid = np.arange(Ny)
    iz_grid = np.arange(Nz)
    for ci in range(n_chunks):
        ix_start = ci * nx_per_chunk
        ix_end   = min(ix_start + nx_per_chunk, Nx)
        ix_set   = np.arange(ix_start, ix_end)
        Mx       = len(ix_set)

        IX, IY, IZ = np.meshgrid(ix_set, iy_grid, iz_grid, indexing="ij")
        cx = wmin[0] + (IX + 0.5) * voxel_size
        cy = wmin[1] + (IY + 0.5) * voxel_size
        cz = wmin[2] + (IZ + 0.5) * voxel_size
        pts3 = np.stack([cx.ravel(), cy.ravel(), cz.ravel()], axis=-1)

        cc, ac, cs = _process_voxel_chunk(pts3, fs, near, far, tol)

        flat_idx = (
            ix_set[:, None, None] * Ny_Nz
            + iy_grid[None, :, None] * Nz
            + iz_grid[None, None, :]
        ).ravel()
        color_count[flat_idx] += cc.astype(np.uint32)
        air_count[flat_idx]   += ac.astype(np.uint32)
        color_sum[flat_idx]   += cs.astype(np.float64)

        elapsed = time.time() - t_proc
        eta = (elapsed / (ci + 1)) * (n_chunks - ci - 1)
        print(f"  chunk {ci+1:3d}/{n_chunks} "
              f"(ix {ix_start:3d}..{ix_end-1:3d}); "
              f"elapsed {elapsed:6.1f} s; ETA {eta:6.1f} s; "
              f"running color {int(color_count.sum()):,} "
              f"air {int(air_count.sum()):,}",
              flush=True)

    print(f"\nAll voxels processed in {time.time()-t_proc:.1f} s")

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
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--frames-dir", default="captured_frames")
    ap.add_argument("--out", default="web/out/voxels.json")
    ap.add_argument("--voxel-size", type=float, default=0.05)
    ap.add_argument("--world-min", type=float, nargs=3, default=[-2.5, -0.3, -2.5])
    ap.add_argument("--world-max", type=float, nargs=3, default=[ 2.5,  4.7,  2.5])
    ap.add_argument("--near", type=float, default=0.05)
    ap.add_argument("--far", type=float, default=8.0)
    ap.add_argument("--tol", type=float, default=0.20)
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--min-color-count", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--voxel-chunk", type=int, default=1_000_000,
                    help="approximate voxels per device kernel call. Larger = "
                         "fewer dispatches; smaller = lower per-call peak memory.")
    args = ap.parse_args()

    reconstruct_jax(
        Path(args.frames_dir), Path(args.out),
        voxel_size=args.voxel_size,
        world_min=tuple(args.world_min),
        world_max=tuple(args.world_max),
        near=args.near, far=args.far, tol=args.tol,
        threshold=args.threshold,
        min_color_count=args.min_color_count,
        max_frames=args.max_frames,
        voxel_chunk=args.voxel_chunk,
    )


if __name__ == "__main__":
    main()
