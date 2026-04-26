"""
Chunked sparse voxel storage with per-voxel RGB.

Layout
------
A `VoxelRoom` is a dict mapping integer chunk-coordinate triples
`(ci, cj, ck)` → `(32, 32, 32, 4)` `uint8` numpy arrays. Channels are
`(R, G, B, hits)`. Empty chunks aren't in the dict, so empty space costs
nothing.

  voxel_index_xyz  = floor(world_xyz / voxel_size).astype(int32)
  chunk_index_xyz  = voxel_index_xyz >> 5
  local_index_xyz  = voxel_index_xyz & 31

For 3 cm voxels, one chunk = 96 cm on a side ≈ furniture-scale. At
room scale (~5×5×3 m), a fully-observed scene is ~50–150 chunks
(~6–20 MB total) — still tiny.

Cell semantics
--------------
- `hits == 0` means empty / unobserved. Iso-surface = 0.5.
- `hits >= 1` means observed. Saturates at 255.
- `(R, G, B)` is a running average over all observations:
      new_rgb = (old_rgb * old_hits + observed_rgb) / (old_hits + 1)
  Stable, no flicker, decays toward the mean as hits grow.

Persistence
-----------
`save(path.npz)` writes `keys` (Nx3 int32), `chunks` (N×32×32×32×4 uint8),
plus voxel_size, chunk_size, and frames metadata. Plain numpy → REPL- and
tool-inspectable.
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any
import numpy as np


CHUNK_SIZE = 32          # cells per chunk edge (32³ = 32 768 cells)
CHUNK_SHIFT = 5
CHUNK_MASK = CHUNK_SIZE - 1
CHANNELS = 4             # (R, G, B, hits)
CHANNEL_R, CHANNEL_G, CHANNEL_B, CHANNEL_HITS = 0, 1, 2, 3


def is_available() -> bool:
    return True


class VoxelRoom:
    """Chunked sparse RGB+hits grid."""

    def __init__(self, voxel_size_m: float = 0.02) -> None:
        self.voxel_size = float(voxel_size_m)
        self.chunks: dict[tuple[int, int, int], np.ndarray] = {}
        self._lock = Lock()
        self._frames_ingested = 0

    # ---- ingestion -------------------------------------------------------

    def insert_points(self, world_points: np.ndarray, colors: np.ndarray | None = None) -> int:
        """Mark voxels covering each point as observed and update colour.

        `colors` is an optional (N, 3) uint8 RGB array. If omitted, colours
        are recorded as neutral grey so meshing still has something to show.
        """
        if world_points.size == 0:
            return 0
        if colors is None:
            colors = np.full((world_points.shape[0], 3), 128, dtype=np.uint8)
        elif colors.shape[0] != world_points.shape[0]:
            raise ValueError(f"colors {colors.shape} mismatches points {world_points.shape}")
        with self._lock:
            return self._insert_locked(world_points, colors)

    def _insert_locked(self, world_points: np.ndarray, colors: np.ndarray) -> int:
        vox = np.floor(world_points / self.voxel_size).astype(np.int32)
        chunk_ijk = vox >> CHUNK_SHIFT
        local_ijk = vox & CHUNK_MASK

        unique_chunks, inverse = np.unique(chunk_ijk, axis=0, return_inverse=True)
        total_written = 0
        for ci_row, chunk_xyz in enumerate(unique_chunks):
            mask = inverse == ci_row
            locs = local_ijk[mask]                 # (M, 3)
            cols = colors[mask].astype(np.int32)   # (M, 3) — int32 for safe maths
            key = (int(chunk_xyz[0]), int(chunk_xyz[1]), int(chunk_xyz[2]))
            chunk = self.chunks.get(key)
            if chunk is None:
                chunk = np.zeros((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, CHANNELS), dtype=np.uint8)
                self.chunks[key] = chunk

            # Multiple points may land in the same voxel within one frame —
            # that's fine: each becomes a sequential update of the running
            # average. We do a Python-level loop over them since updating in
            # numpy with overlapping indices isn't well-defined for our
            # accumulator semantics. The number of points per insert is
            # bounded by depth pixel count (~14k), so this is cheap.
            for m in range(locs.shape[0]):
                ix, iy, iz = locs[m]
                cell = chunk[ix, iy, iz]
                old_hits = int(cell[CHANNEL_HITS])
                new_hits = min(255, old_hits + 1)
                if old_hits == 0:
                    cell[CHANNEL_R] = cols[m, 0]
                    cell[CHANNEL_G] = cols[m, 1]
                    cell[CHANNEL_B] = cols[m, 2]
                else:
                    # Running average — uses old_hits (pre-increment) so the
                    # old colour is weighted by what produced it.
                    cell[CHANNEL_R] = (int(cell[CHANNEL_R]) * old_hits + cols[m, 0]) // (old_hits + 1)
                    cell[CHANNEL_G] = (int(cell[CHANNEL_G]) * old_hits + cols[m, 1]) // (old_hits + 1)
                    cell[CHANNEL_B] = (int(cell[CHANNEL_B]) * old_hits + cols[m, 2]) // (old_hits + 1)
                cell[CHANNEL_HITS] = new_hits
            total_written += locs.shape[0]
        return total_written

    def ingest_frame(self, frame: dict[str, Any]) -> int:
        import fusion  # tools/ on sys.path when serve.py runs as a script
        result = fusion.frame_to_world_points(frame, with_colors=True)
        # with_colors=True returns a 3-tuple
        points, _cam, colors = result
        n = self.insert_points(points, colors)
        with self._lock:
            self._frames_ingested += 1
        return n

    # ---- persistence -----------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if not self.chunks:
                np.savez(
                    path,
                    keys=np.zeros((0, 3), dtype=np.int32),
                    chunks=np.zeros((0, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, CHANNELS), dtype=np.uint8),
                    voxel_size_m=np.float32(self.voxel_size),
                    chunk_size=np.int32(CHUNK_SIZE),
                    channels=np.int32(CHANNELS),
                    frames_ingested=np.int32(self._frames_ingested),
                )
                return
            keys = np.array(list(self.chunks.keys()), dtype=np.int32)
            data = np.stack(list(self.chunks.values()))  # (N, 32, 32, 32, 4)
            np.savez(
                path,
                keys=keys,
                chunks=data,
                voxel_size_m=np.float32(self.voxel_size),
                chunk_size=np.int32(CHUNK_SIZE),
                channels=np.int32(CHANNELS),
                frames_ingested=np.int32(self._frames_ingested),
            )

    def load(self, path: Path | str) -> None:
        with self._lock:
            with np.load(path) as f:
                cs = int(f["chunk_size"])
                if cs != CHUNK_SIZE:
                    raise ValueError(
                        f"saved chunk_size={cs} ≠ this build's CHUNK_SIZE={CHUNK_SIZE}"
                    )
                ch = int(f["channels"]) if "channels" in f.files else 1
                if ch != CHANNELS:
                    raise ValueError(
                        f"saved channels={ch} ≠ this build's CHANNELS={CHANNELS} "
                        "(likely an older grid without colour — start a fresh scan)"
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
            n_voxels = sum(int(np.count_nonzero(c[..., CHANNEL_HITS])) for c in self.chunks.values())
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
                "channels": CHANNELS,
                "frames_ingested": self._frames_ingested,
                "bbox": bbox,
            }

    # ---- meshing helpers -------------------------------------------------

    def get_padded_block(self, key: tuple[int, int, int]) -> np.ndarray:
        """Return a (34, 34, 34, 4) uint8 block whose interior
        [1:33, 1:33, 1:33, :] is the chunk at `key`, with 1-voxel boundary
        slabs filled from the six face-neighbours (zero otherwise)."""
        ci, cj, ck = key
        N = CHUNK_SIZE
        block = np.zeros((N + 2, N + 2, N + 2, CHANNELS), dtype=np.uint8)
        block[1:1 + N, 1:1 + N, 1:1 + N, :] = self.chunks[key]
        n = self.chunks.get((ci - 1, cj, ck))
        if n is not None: block[0,     1:1+N, 1:1+N, :] = n[N - 1, :, :, :]
        n = self.chunks.get((ci + 1, cj, ck))
        if n is not None: block[N + 1, 1:1+N, 1:1+N, :] = n[0,     :, :, :]
        n = self.chunks.get((ci, cj - 1, ck))
        if n is not None: block[1:1+N, 0,     1:1+N, :] = n[:, N - 1, :, :]
        n = self.chunks.get((ci, cj + 1, ck))
        if n is not None: block[1:1+N, N + 1, 1:1+N, :] = n[:, 0,     :, :]
        n = self.chunks.get((ci, cj, ck - 1))
        if n is not None: block[1:1+N, 1:1+N, 0,     :] = n[:, :, N - 1, :]
        n = self.chunks.get((ci, cj, ck + 1))
        if n is not None: block[1:1+N, 1:1+N, N + 1, :] = n[:, :, 0,     :]
        return block

    def lock(self) -> Lock:
        return self._lock
