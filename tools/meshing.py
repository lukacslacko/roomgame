"""
Per-chunk marching cubes mesh extraction for a chunked VoxelRoom.

For each non-empty chunk:
  1. Build a (34,34,34) padded block via VoxelRoom.get_padded_block.
     The 1-voxel border slabs on each face are filled from the corresponding
     neighbour chunk (or 0 when there's no neighbour). This is what lets
     marching cubes close the iso-surface cleanly across chunk boundaries.
  2. Run skimage.measure.marching_cubes at iso=0.5.
  3. Translate verts from padded-block coords to world coords.

To avoid duplicate triangles where two neighbours both mesh their shared
boundary, we keep only triangles whose centroid lies inside the chunk's own
voxel range (i.e. block coords [1, 33)). That gives one canonical chunk per
boundary.

Output is concatenated to a single (verts, faces) pair and written as GLB
via trimesh.

This module imports skimage and trimesh lazily so a server boot without
those installed still answers /save and /stats correctly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np

from voxel_store import CHUNK_SIZE, VoxelRoom


def extract_mesh(voxel_room: VoxelRoom, *, iso: float = 0.5):
    """Run per-chunk marching cubes; return (verts, faces) in world metres,
    or (None, None) if there's nothing to mesh."""
    from skimage.measure import marching_cubes

    voxel_size = float(voxel_room.voxel_size)
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vert_offset = 0

    # Hold the lock for the whole pass so chunks don't mutate under us.
    with voxel_room.lock():
        chunk_keys = list(voxel_room.chunks.keys())
        for key in chunk_keys:
            chunk = voxel_room.chunks[key]
            if not chunk.any():
                continue
            block = voxel_room.get_padded_block(key)
            try:
                verts, faces, _normals, _values = marching_cubes(
                    block.astype(np.float32), level=iso, allow_degenerate=False
                )
            except (ValueError, RuntimeError):
                # No surface in this chunk (e.g. all values below iso after
                # padding masks them out). Skip silently.
                continue
            if verts.size == 0 or faces.size == 0:
                continue

            # Keep only triangles whose centroid sits in the chunk's own
            # interior block range [1, 1 + CHUNK_SIZE). This dedupes shared
            # boundary cells across neighbouring chunks: each cell appears
            # in exactly one chunk's "owned" range.
            #
            # Ownership convention: a cell at block coord c is owned by the
            # chunk that contains it in [1, 1 + CHUNK_SIZE) — the cells in
            # the [0, 1) and [1 + CHUNK_SIZE, 2 + CHUNK_SIZE) padding bands
            # belong to the neighbour chunks instead.
            tri_centroids = verts[faces].mean(axis=1)  # (F, 3) in block coords
            inside = (
                (tri_centroids[:, 0] >= 1) & (tri_centroids[:, 0] < 1 + CHUNK_SIZE) &
                (tri_centroids[:, 1] >= 1) & (tri_centroids[:, 1] < 1 + CHUNK_SIZE) &
                (tri_centroids[:, 2] >= 1) & (tri_centroids[:, 2] < 1 + CHUNK_SIZE)
            )
            faces = faces[inside]
            if faces.size == 0:
                continue

            # Drop verts not referenced by surviving faces (compact).
            used = np.unique(faces)
            remap = -np.ones(verts.shape[0], dtype=np.int64)
            remap[used] = np.arange(used.shape[0])
            verts = verts[used]
            faces = remap[faces]

            # Block coords → world coords. Block index 1 corresponds to the
            # first chunk-local cell, which sits at world position
            # (chunk_origin_voxels + 0) * voxel_size.
            ci, cj, ck = key
            chunk_origin_voxels = np.array(
                [ci * CHUNK_SIZE, cj * CHUNK_SIZE, ck * CHUNK_SIZE], dtype=np.float64
            )
            verts_world = ((verts - 1) + chunk_origin_voxels) * voxel_size

            all_verts.append(verts_world.astype(np.float32))
            all_faces.append((faces + vert_offset).astype(np.int32))
            vert_offset += verts.shape[0]

    if not all_verts:
        return None, None
    return np.concatenate(all_verts, axis=0), np.concatenate(all_faces, axis=0)


def write_glb(verts: np.ndarray, faces: np.ndarray, path: Path | str) -> None:
    import trimesh
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(str(path), file_type="glb")


def remesh_to_glb(
    voxel_room: VoxelRoom, path: Path | str, *, iso: float = 0.5
) -> Optional[dict]:
    verts, faces = extract_mesh(voxel_room, iso=iso)
    if verts is None:
        return None
    write_glb(verts, faces, path)
    return {
        "path": str(path),
        "vertices": int(verts.shape[0]),
        "faces": int(faces.shape[0]),
    }
