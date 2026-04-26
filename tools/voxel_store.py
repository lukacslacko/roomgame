"""
Chunked sparse voxel storage.

Layout
------
A `VoxelRoom` is a dict mapping integer chunk-coordinate triples
`(ci, cj, ck)` → a `(32, 32, 32)` `uint8` numpy array. Empty chunks aren't
in the dict, so empty space costs nothing.

  voxel_index_xyz  = floor(world_xyz / voxel_size).astype(int32)
  chunk_index_xyz  = voxel_index_xyz >> 5            # arithmetic shift
  local_index_xyz  = voxel_index_xyz & 31

For a 3 cm voxel, one chunk = 96 cm on a side ≈ furniture-scale.
At room scale (~5×5×3 m) a fully-observed scene is ~50–150 chunks
(~1.6–5 MB total), which is tiny.

Cell values
-----------
- 0  = empty / unobserved
- 1+ = observed at least once. Stored as `uint8`, saturates at 255 so we can
  later use the value as a hit count (more hits = more confident).

Insertion is vectorized: world points are bucketed by chunk via
`np.unique(chunks, axis=0, return_inverse=True)`, then a single
fancy-index assignment writes all points within one chunk in one numpy op.

Persistence
-----------
`save(path.npz)` writes `keys` (Nx3 int32), `chunks` (N×32×32×32 uint8),
plus voxel_size and frames metadata. `load` is symmetric. The format is
plain numpy, so it's easy to inspect from the REPL or other tools.
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any
import numpy as np


CHUNK_SIZE = 32  # cells per chunk edge (32³ = 32 KB at uint8)
CHUNK_SHIFT = 5  # log2(CHUNK_SIZE)
CHUNK_MASK = CHUNK_SIZE - 1  # 31


def is_available() -> bool:
    """numpy is the only requirement; if this module imported, we're set."""
    return True


class VoxelRoom:
    """Chunked sparse occupancy grid.

    Threading: insert_points / ingest_frame / extract_mesh / save all take
    a per-instance lock. ThreadingHTTPServer handles requests on a thread
    pool, so concurrent /frame POSTs serialize cleanly here.
    """

    def __init__(self, voxel_size_m: float = 0.03) -> None:
        self.voxel_size = float(voxel_size_m)
        self.chunks: dict[tuple[int, int, int], np.ndarray] = {}
        self._lock = Lock()
        self._frames_ingested = 0

    # ---- ingestion -------------------------------------------------------

    def insert_points(self, world_points: np.ndarray) -> int:
        """Mark voxels covering each world-space point as observed.
        Returns the number of points written (after dedup within a chunk
        is *not* applied — the count reflects total writes, useful as a
        rough rate metric)."""
        if world_points.size == 0:
            return 0
        with self._lock:
            return self._insert_locked(world_points)

    def _insert_locked(self, world_points: np.ndarray) -> int:
        # int32 voxel coords. floor() handles negatives correctly.
        vox = np.floor(world_points / self.voxel_size).astype(np.int32)
        chunk_ijk = vox >> CHUNK_SHIFT      # (N, 3) — Python's >> is arithmetic on signed
        local_ijk = vox & CHUNK_MASK         # (N, 3) — wraps negatives correctly

        # Bucket points by chunk in one pass.
        unique_chunks, inverse = np.unique(chunk_ijk, axis=0, return_inverse=True)
        total_written = 0
        for ci_row, chunk_xyz in enumerate(unique_chunks):
            mask = inverse == ci_row
            locals_here = local_ijk[mask]
            key = (int(chunk_xyz[0]), int(chunk_xyz[1]), int(chunk_xyz[2]))
            chunk = self.chunks.get(key)
            if chunk is None:
                chunk = np.zeros((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)
                self.chunks[key] = chunk
            # Saturating-increment hit count so repeated observations
            # strengthen confidence without overflowing.
            existing = chunk[locals_here[:, 0], locals_here[:, 1], locals_here[:, 2]]
            chunk[locals_here[:, 0], locals_here[:, 1], locals_here[:, 2]] = np.minimum(255, existing.astype(np.int32) + 1).astype(np.uint8)
            total_written += locals_here.shape[0]
        return total_written

    def ingest_frame(self, frame: dict[str, Any]) -> int:
        import fusion  # tools/ on sys.path when serve.py runs as a script
        points, _cam = fusion.frame_to_world_points(frame)
        n = self.insert_points(points)
        with self._lock:
            self._frames_ingested += 1
        return n

    # ---- persistence -----------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if not self.chunks:
                # Save an empty marker so loaders can distinguish "saved
                # nothing" from "file missing".
                np.savez(
                    path,
                    keys=np.zeros((0, 3), dtype=np.int32),
                    chunks=np.zeros((0, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8),
                    voxel_size_m=np.float32(self.voxel_size),
                    chunk_size=np.int32(CHUNK_SIZE),
                    frames_ingested=np.int32(self._frames_ingested),
                )
                return
            keys = np.array(list(self.chunks.keys()), dtype=np.int32)
            data = np.stack(list(self.chunks.values()))  # (N, 32, 32, 32) uint8
            np.savez(
                path,
                keys=keys,
                chunks=data,
                voxel_size_m=np.float32(self.voxel_size),
                chunk_size=np.int32(CHUNK_SIZE),
                frames_ingested=np.int32(self._frames_ingested),
            )

    def load(self, path: Path | str) -> None:
        with self._lock:
            with np.load(path) as f:
                cs = int(f["chunk_size"])
                if cs != CHUNK_SIZE:
                    raise ValueError(
                        f"saved chunk_size={cs} ≠ this build's CHUNK_SIZE={CHUNK_SIZE}; "
                        "rebuild the file or change CHUNK_SIZE"
                    )
                self.voxel_size = float(f["voxel_size_m"])
                self._frames_ingested = int(f["frames_ingested"])
                keys = f["keys"]
                chunks = f["chunks"]
                self.chunks = {
                    (int(k[0]), int(k[1]), int(k[2])): chunks[i].copy()
                    for i, k in enumerate(keys)
                }

    # ---- introspection ---------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            n_chunks = len(self.chunks)
            n_voxels = sum(int(np.count_nonzero(c)) for c in self.chunks.values())
            bbox = None
            if n_chunks > 0:
                keys = np.array(list(self.chunks.keys()), dtype=np.int32)
                vox_min = (keys.min(axis=0) * CHUNK_SIZE).astype(np.int64)
                vox_max = ((keys.max(axis=0) + 1) * CHUNK_SIZE - 1).astype(np.int64)
                bbox = {
                    "voxels_min": vox_min.tolist(),
                    "voxels_max": vox_max.tolist(),
                    "world_min_m": (vox_min * self.voxel_size).tolist(),
                    "world_max_m": ((vox_max + 1) * self.voxel_size).tolist(),
                }
            return {
                "voxels": n_voxels,
                "chunks": n_chunks,
                "voxel_size_m": self.voxel_size,
                "chunk_size": CHUNK_SIZE,
                "frames_ingested": self._frames_ingested,
                "bbox": bbox,
            }

    # ---- meshing helpers (called from meshing.py) ------------------------

    def get_padded_block(self, key: tuple[int, int, int]) -> np.ndarray:
        """Return a (34, 34, 34) uint8 block whose interior [1:33, 1:33, 1:33]
        is the chunk at `key`, with 1-voxel boundary slabs filled from the
        six face-neighbours when present (zero otherwise).

        Padding is essential for marching-cubes to seal the iso-surface
        across chunk boundaries. Caller must hold the lock if mutating in
        parallel; meshing.py grabs it once for the full pass.
        """
        ci, cj, ck = key
        block = np.zeros((CHUNK_SIZE + 2, CHUNK_SIZE + 2, CHUNK_SIZE + 2), dtype=np.uint8)
        block[1:1 + CHUNK_SIZE, 1:1 + CHUNK_SIZE, 1:1 + CHUNK_SIZE] = self.chunks[key]
        # Six face neighbours: copy the 1-cell-thick slab on the touching face.
        n = self.chunks.get((ci - 1, cj, ck))
        if n is not None:
            block[0, 1:1 + CHUNK_SIZE, 1:1 + CHUNK_SIZE] = n[CHUNK_SIZE - 1, :, :]
        n = self.chunks.get((ci + 1, cj, ck))
        if n is not None:
            block[CHUNK_SIZE + 1, 1:1 + CHUNK_SIZE, 1:1 + CHUNK_SIZE] = n[0, :, :]
        n = self.chunks.get((ci, cj - 1, ck))
        if n is not None:
            block[1:1 + CHUNK_SIZE, 0, 1:1 + CHUNK_SIZE] = n[:, CHUNK_SIZE - 1, :]
        n = self.chunks.get((ci, cj + 1, ck))
        if n is not None:
            block[1:1 + CHUNK_SIZE, CHUNK_SIZE + 1, 1:1 + CHUNK_SIZE] = n[:, 0, :]
        n = self.chunks.get((ci, cj, ck - 1))
        if n is not None:
            block[1:1 + CHUNK_SIZE, 1:1 + CHUNK_SIZE, 0] = n[:, :, CHUNK_SIZE - 1]
        n = self.chunks.get((ci, cj, ck + 1))
        if n is not None:
            block[1:1 + CHUNK_SIZE, 1:1 + CHUNK_SIZE, CHUNK_SIZE + 1] = n[:, :, 0]
        return block

    def lock(self) -> Lock:
        """Expose the internal lock so meshing can hold it across the pass
        without re-acquiring on every chunk lookup."""
        return self._lock
