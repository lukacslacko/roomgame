"""
Dense marching-cubes mesh extraction with per-vertex RGB colour.

For one-room scans the chunked grid easily fits in a single dense numpy
block (a few MB), so we mesh the whole thing in one pass:

  1. Build a (W+2, H+2, D+2, 4) uint8 dense block covering the active
     bbox plus a 1-voxel rim of zeros (so the iso-surface closes around
     boundary chunks).
  2. Box-blur the `hits` channel with a 3³ kernel — without smoothing the
     mesh fragments into hundreds of single-voxel islands as a noisy
     scanner sprays observations across slightly-different positions.
  3. marching_cubes at iso=1.0 on the smoothed field. The threshold is in
     units of "average hits per cell in the 3³ neighbourhood"; iso=1.0
     means we need at least 27 hits' worth of mass nearby — enough to
     reject single-frame noise while keeping anything observed in a few
     frames.
  4. Per vertex, trilinear-interpolate the (R, G, B) channels weighted by
     the unobserved-cell mask so empty neighbours don't drag the colour
     toward zero.

trimesh writes vertex colours into the GLB's COLOR_0 attribute, which
three.js picks up automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np

from voxel_store import CHUNK_SIZE, CHANNEL_HITS, CHANNELS, VoxelRoom


def _build_dense_block(voxel_room: VoxelRoom) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Stitch all populated chunks into one dense (W+2, H+2, D+2, 4) block.
    Returns (block, vox_origin_int) where vox_origin_int is the world
    voxel index corresponding to block[0, 0, 0, :] (i.e., one voxel
    before the active bbox to give marching cubes a closing rim)."""
    if not voxel_room.chunks:
        return None, None

    keys = np.array(list(voxel_room.chunks.keys()), dtype=np.int64)
    # Bounding box in voxel-index coords (inclusive).
    vmin = keys.min(axis=0) * CHUNK_SIZE
    vmax = (keys.max(axis=0) + 1) * CHUNK_SIZE - 1

    pad = 1
    origin = vmin - pad
    shape = tuple((vmax - vmin + 1 + 2 * pad).tolist())
    block = np.zeros((shape[0], shape[1], shape[2], CHANNELS), dtype=np.uint8)

    for (ci, cj, ck), chunk in voxel_room.chunks.items():
        ox = ci * CHUNK_SIZE - origin[0]
        oy = cj * CHUNK_SIZE - origin[1]
        oz = ck * CHUNK_SIZE - origin[2]
        block[ox:ox + CHUNK_SIZE, oy:oy + CHUNK_SIZE, oz:oz + CHUNK_SIZE, :] = chunk

    return block, origin


def _trilinear_colors(block: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Sample (R, G, B) at each vertex via trilinear interpolation of the
    surrounding 8 cells, weighted by 'observed' presence. `verts` are in
    block coordinates (the raw output of skimage.marching_cubes)."""
    # Clamp integer indices so neighbours +1 stay in range.
    bx = block.shape[0] - 2
    by = block.shape[1] - 2
    bz = block.shape[2] - 2
    x0 = np.clip(np.floor(verts[:, 0]).astype(np.int32), 0, bx); x1 = x0 + 1
    y0 = np.clip(np.floor(verts[:, 1]).astype(np.int32), 0, by); y1 = y0 + 1
    z0 = np.clip(np.floor(verts[:, 2]).astype(np.int32), 0, bz); z1 = z0 + 1
    fx = np.clip(verts[:, 0] - x0, 0.0, 1.0).astype(np.float32)[:, None]
    fy = np.clip(verts[:, 1] - y0, 0.0, 1.0).astype(np.float32)[:, None]
    fz = np.clip(verts[:, 2] - z0, 0.0, 1.0).astype(np.float32)[:, None]

    c000 = block[x0, y0, z0].astype(np.float32)
    c100 = block[x1, y0, z0].astype(np.float32)
    c010 = block[x0, y1, z0].astype(np.float32)
    c110 = block[x1, y1, z0].astype(np.float32)
    c001 = block[x0, y0, z1].astype(np.float32)
    c101 = block[x1, y0, z1].astype(np.float32)
    c011 = block[x0, y1, z1].astype(np.float32)
    c111 = block[x1, y1, z1].astype(np.float32)

    w000 = (1 - fx) * (1 - fy) * (1 - fz)
    w100 = (    fx) * (1 - fy) * (1 - fz)
    w010 = (1 - fx) * (    fy) * (1 - fz)
    w110 = (    fx) * (    fy) * (1 - fz)
    w001 = (1 - fx) * (1 - fy) * (    fz)
    w101 = (    fx) * (1 - fy) * (    fz)
    w011 = (1 - fx) * (    fy) * (    fz)
    w111 = (    fx) * (    fy) * (    fz)

    p = lambda c: (c[:, CHANNEL_HITS:CHANNEL_HITS + 1] > 0).astype(np.float32)
    presence = (
        w000 * p(c000) + w100 * p(c100) + w010 * p(c010) + w110 * p(c110) +
        w001 * p(c001) + w101 * p(c101) + w011 * p(c011) + w111 * p(c111)
    )
    rgb = (
        w000 * p(c000) * c000[:, :3] + w100 * p(c100) * c100[:, :3] +
        w010 * p(c010) * c010[:, :3] + w110 * p(c110) * c110[:, :3] +
        w001 * p(c001) * c001[:, :3] + w101 * p(c101) * c101[:, :3] +
        w011 * p(c011) * c011[:, :3] + w111 * p(c111) * c111[:, :3]
    )
    safe = np.maximum(presence, 1e-6)
    rgb = (rgb / safe).clip(0, 255).astype(np.uint8)
    rgb = np.where(presence > 0, rgb, np.uint8(128))
    return rgb


def extract_mesh(voxel_room: VoxelRoom, *, iso: float = 1.0, smooth_kernel: int = 3):
    """Return (verts, faces, colors) in world metres / RGB uint8, or
    (None, None, None) if there's no surface to mesh."""
    from skimage.measure import marching_cubes
    from scipy.ndimage import uniform_filter

    voxel_size = float(voxel_room.voxel_size)
    with voxel_room.lock():
        block, origin = _build_dense_block(voxel_room)
    if block is None:
        return None, None, None

    hits_field = block[..., CHANNEL_HITS].astype(np.float32)
    if smooth_kernel and smooth_kernel > 1:
        hits_field = uniform_filter(hits_field, size=smooth_kernel)
    if not (hits_field > iso).any():
        return None, None, None

    try:
        verts, faces, _normals, _values = marching_cubes(
            hits_field, level=iso, allow_degenerate=False
        )
    except (ValueError, RuntimeError):
        return None, None, None
    if verts.size == 0 or faces.size == 0:
        return None, None, None

    colors = _trilinear_colors(block, verts)

    # Block coords → world coords. block[0,0,0] sits at voxel index `origin`
    # (one voxel before the active bbox). World = (block_idx + origin) * vox.
    verts_world = ((verts + origin.astype(np.float64)) * voxel_size).astype(np.float32)

    return verts_world, faces.astype(np.int32), colors


def write_glb(verts: np.ndarray, faces: np.ndarray, colors: np.ndarray | None, path: Path | str) -> None:
    import trimesh
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if colors is not None:
        rgba = np.empty((colors.shape[0], 4), dtype=np.uint8)
        rgba[:, :3] = colors
        rgba[:, 3] = 255
        mesh.visual.vertex_colors = rgba
    mesh.export(str(path), file_type="glb")


def remesh_to_glb(
    voxel_room: VoxelRoom, path: Path | str, *, iso: float = 1.0
) -> Optional[dict]:
    verts, faces, colors = extract_mesh(voxel_room, iso=iso)
    if verts is None:
        return None
    write_glb(verts, faces, colors, path)
    return {
        "path": str(path),
        "vertices": int(verts.shape[0]),
        "faces": int(faces.shape[0]),
        "has_colors": colors is not None,
    }
