"""
Per-chunk marching cubes mesh extraction with per-vertex RGB colour.

The voxel grid stores (R, G, B, hits) per cell. Marching cubes runs on
the `hits` channel at iso=0.5, so any cell observed at least once
contributes to the surface. Vertex colours are sampled by trilinear
interpolation among the 8 voxels surrounding each vertex's position
(weighted by hits — empty neighbours get zero weight, so vertices on
the surface boundary get colour from the side that was actually
observed).

The mesh is concatenated into a single (verts, faces, vertex_colours)
triple and exported as GLB via trimesh, which preserves vertex colours
in the COLOR_0 attribute that three.js picks up automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np

from voxel_store import CHUNK_SIZE, CHANNEL_HITS, CHANNELS, VoxelRoom


def _trilinear_colors(block: np.ndarray, verts_block: np.ndarray) -> np.ndarray:
    """Sample (R, G, B) at each vertex from a padded chunk block.

    `block`        : (N+2, N+2, N+2, 4) uint8 — RGBA where A=hits.
    `verts_block`  : (V, 3) float — vertex positions in block coords (the raw
                     output of skimage.marching_cubes, before the -1 shift).

    Returns (V, 3) uint8 RGB.
    """
    # Clamp the integer index, not the float position. Reason: float64 epsilon
    # tricks like `by - 1e-6` round back to `by` when cast to float32 (the
    # dtype skimage produces), making y1 = by + 1 land out of bounds. By
    # clamping the integer floor instead, we guarantee y0 in [0, by-1] and
    # y1 in [1, by]; the interpolation factor is then clipped to [0, 1] for
    # the rare verts that landed exactly on the upper boundary.
    bx_max = block.shape[0] - 2  # last valid index for x0 such that x1 ≤ block.shape[0] - 1
    by_max = block.shape[1] - 2
    bz_max = block.shape[2] - 2
    vp = verts_block
    x0 = np.clip(np.floor(vp[:, 0]).astype(np.int32), 0, bx_max); x1 = x0 + 1
    y0 = np.clip(np.floor(vp[:, 1]).astype(np.int32), 0, by_max); y1 = y0 + 1
    z0 = np.clip(np.floor(vp[:, 2]).astype(np.int32), 0, bz_max); z1 = z0 + 1
    fx = np.clip(vp[:, 0] - x0, 0.0, 1.0).astype(np.float32)[:, None]
    fy = np.clip(vp[:, 1] - y0, 0.0, 1.0).astype(np.float32)[:, None]
    fz = np.clip(vp[:, 2] - z0, 0.0, 1.0).astype(np.float32)[:, None]

    # Pull the 8 neighbour cells (each (V, 4) — R,G,B,hits).
    c000 = block[x0, y0, z0].astype(np.float32)
    c100 = block[x1, y0, z0].astype(np.float32)
    c010 = block[x0, y1, z0].astype(np.float32)
    c110 = block[x1, y1, z0].astype(np.float32)
    c001 = block[x0, y0, z1].astype(np.float32)
    c101 = block[x1, y0, z1].astype(np.float32)
    c011 = block[x0, y1, z1].astype(np.float32)
    c111 = block[x1, y1, z1].astype(np.float32)

    # Weights from interpolation factors.
    w000 = (1 - fx) * (1 - fy) * (1 - fz)
    w100 = (    fx) * (1 - fy) * (1 - fz)
    w010 = (1 - fx) * (    fy) * (1 - fz)
    w110 = (    fx) * (    fy) * (1 - fz)
    w001 = (1 - fx) * (1 - fy) * (    fz)
    w101 = (    fx) * (1 - fy) * (    fz)
    w011 = (1 - fx) * (    fy) * (    fz)
    w111 = (    fx) * (    fy) * (    fz)

    # Hits channel → presence weight: 1 if observed, 0 if not.
    p = lambda c: (c[:, CHANNEL_HITS:CHANNEL_HITS + 1] > 0).astype(np.float32)
    presence = (
        w000 * p(c000) + w100 * p(c100) + w010 * p(c010) + w110 * p(c110) +
        w001 * p(c001) + w101 * p(c101) + w011 * p(c011) + w111 * p(c111)
    )

    # Colour-weighted sum (only count colours from observed cells).
    rgb = (
        w000 * p(c000) * c000[:, :3] + w100 * p(c100) * c100[:, :3] +
        w010 * p(c010) * c010[:, :3] + w110 * p(c110) * c110[:, :3] +
        w001 * p(c001) * c001[:, :3] + w101 * p(c101) * c101[:, :3] +
        w011 * p(c011) * c011[:, :3] + w111 * p(c111) * c111[:, :3]
    )

    # Normalise; vertices with all-empty neighbours get neutral grey (this
    # shouldn't happen since marching_cubes only emits verts near the iso,
    # but be defensive).
    safe = np.maximum(presence, 1e-6)
    rgb = (rgb / safe).clip(0, 255).astype(np.uint8)
    rgb = np.where(presence > 0, rgb, np.uint8(128))
    return rgb


def extract_mesh(voxel_room: VoxelRoom, *, iso: float = 0.5):
    """Run per-chunk marching cubes on the hits channel; sample colour at
    each vertex via trilinear interpolation. Returns (verts, faces, colors)
    in world metres / RGB uint8, or (None, None, None) if no surface."""
    from skimage.measure import marching_cubes

    voxel_size = float(voxel_room.voxel_size)
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    vert_offset = 0

    with voxel_room.lock():
        chunk_keys = list(voxel_room.chunks.keys())
        for key in chunk_keys:
            chunk = voxel_room.chunks[key]
            if not chunk[..., CHANNEL_HITS].any():
                continue
            block = voxel_room.get_padded_block(key)            # (34,34,34,4)
            hits_field = block[..., CHANNEL_HITS].astype(np.float32)
            try:
                verts, faces, _normals, _values = marching_cubes(
                    hits_field, level=iso, allow_degenerate=False
                )
            except (ValueError, RuntimeError):
                continue
            if verts.size == 0 or faces.size == 0:
                continue

            # Centroid-ownership filter: each chunk owns its interior cells
            # [1, 1+CHUNK_SIZE), so shared boundary cells appear in exactly
            # one chunk's mesh (no duplicate triangles).
            tri_centroids = verts[faces].mean(axis=1)
            inside = (
                (tri_centroids[:, 0] >= 1) & (tri_centroids[:, 0] < 1 + CHUNK_SIZE) &
                (tri_centroids[:, 1] >= 1) & (tri_centroids[:, 1] < 1 + CHUNK_SIZE) &
                (tri_centroids[:, 2] >= 1) & (tri_centroids[:, 2] < 1 + CHUNK_SIZE)
            )
            faces = faces[inside]
            if faces.size == 0:
                continue

            # Compact: drop verts not referenced by surviving faces.
            used = np.unique(faces)
            remap = -np.ones(verts.shape[0], dtype=np.int64)
            remap[used] = np.arange(used.shape[0])
            verts_kept = verts[used]
            faces = remap[faces]

            # Sample colour at each surviving vertex (in block coords).
            colors_kept = _trilinear_colors(block, verts_kept)

            # Block coords → world coords.
            ci, cj, ck = key
            chunk_origin_voxels = np.array(
                [ci * CHUNK_SIZE, cj * CHUNK_SIZE, ck * CHUNK_SIZE], dtype=np.float64
            )
            verts_world = ((verts_kept - 1) + chunk_origin_voxels) * voxel_size

            all_verts.append(verts_world.astype(np.float32))
            all_faces.append((faces + vert_offset).astype(np.int32))
            all_colors.append(colors_kept)
            vert_offset += verts_kept.shape[0]

    if not all_verts:
        return None, None, None
    return (
        np.concatenate(all_verts, axis=0),
        np.concatenate(all_faces, axis=0),
        np.concatenate(all_colors, axis=0),
    )


def write_glb(verts: np.ndarray, faces: np.ndarray, colors: np.ndarray | None, path: Path | str) -> None:
    import trimesh
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if colors is not None:
        # trimesh wants RGBA; default alpha 255.
        rgba = np.empty((colors.shape[0], 4), dtype=np.uint8)
        rgba[:, :3] = colors
        rgba[:, 3] = 255
        mesh.visual.vertex_colors = rgba
    mesh.export(str(path), file_type="glb")


def remesh_to_glb(
    voxel_room: VoxelRoom, path: Path | str, *, iso: float = 0.5
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
