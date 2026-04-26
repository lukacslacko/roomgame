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
        self._frames_rejected_drift = 0
        self._frames_rejected_jump = 0
        self._frames_corrected = 0
        # kd-tree over occupied voxel centres (world m). Rebuilt lazily as
        # the grid grows so we can detect frames whose ARCore pose drifted.
        self._kdtree = None
        self._kdtree_built_at_voxels = 0
        # Last accepted camera position, for pose-jump rejection.
        self._last_cam_origin: np.ndarray | None = None

    def reset(self) -> None:
        """Wipe the grid + drift state. Used by /reset when the user wants
        to throw away a drifted scan and start over without restarting."""
        with self._lock:
            self.chunks.clear()
            self._kdtree = None
            self._kdtree_built_at_voxels = 0
            self._last_cam_origin = None
            # Counters are kept (across-session totals); resets are rare and
            # the absolute frames-ingested number is still informative.

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

    # ---- drift detection -------------------------------------------------
    #
    # When ARCore re-localises mid-scan, the same wall can land at two
    # slightly-different world positions across frames; integrating both
    # produces "ghost" parallel walls. We catch that with a kd-tree over
    # currently-occupied voxel centres: if a new frame has substantial
    # overlap with the existing surface but a large median nearest-neighbour
    # distance, the frame's pose has drifted and we reject it.

    # Tight kd-tree match radius. With a wider radius (e.g. 20 cm) every
    # point finds *some* match in the grid, even when the actual nearest
    # surface is a ghost wall, and ICP-SVD ends up "aligning" to noise.
    # Keeping it tight means only points with a clean correspondence
    # contribute to the rigid-transform estimate — when there's drift
    # inside this radius, ICP fixes it; when there isn't, no spurious
    # rotation gets cooked up.
    DRIFT_MATCH_RADIUS_M = 0.05
    DRIFT_MIN_MATCH_FRACTION = 0.30  # need this much overlap to correct/check
    DRIFT_MIN_MATCHES_FOR_ICP = 500  # min correspondences for stable SVD
    DRIFT_REJECT_DIST_M = 0.10       # median displacement that's beyond ICP
    DRIFT_MIN_VOXELS_FOR_CHECK = 1500  # grid needs to be populated enough
    # ICP gets to nudge per frame, not relocate. ARCore's gradual drift is
    # bounded by a few cm and a fraction of a degree per frame; anything
    # the SVD proposes outside these caps is much more likely a noisy
    # estimate from bad correspondences than a real correction.
    ICP_MAX_TRANSLATION_M = 0.05
    ICP_MAX_ROTATION_DEG = 2.0
    # Pose-jump rejection: ARCore re-localisations cause the camera to
    # teleport between consecutive frames. At streaming cadence (≈5–15 fps)
    # honest motion is bounded by walking pace; anything past 30 cm in a
    # single frame is almost certainly a re-localisation, not the user.
    POSE_JUMP_REJECT_M = 0.30

    def _count_active_voxels(self) -> int:
        n = 0
        for chunk in self.chunks.values():
            n += int(np.count_nonzero(chunk[..., CHANNEL_HITS]))
        return n

    def _maybe_rebuild_kdtree(self) -> None:
        """Rebuild the spatial index when the grid has grown materially.
        Caller must hold self._lock."""
        n = self._count_active_voxels()
        if n < self.DRIFT_MIN_VOXELS_FOR_CHECK:
            return
        if self._kdtree is not None and n < 1.5 * max(1, self._kdtree_built_at_voxels):
            return  # no significant change since last rebuild
        from scipy.spatial import cKDTree
        all_centers: list[np.ndarray] = []
        for (ci, cj, ck), chunk in self.chunks.items():
            occ = np.argwhere(chunk[..., CHANNEL_HITS] > 0)  # (M, 3) local idx
            if occ.size == 0:
                continue
            global_idx = occ.astype(np.int64) + np.array(
                [ci, cj, ck], dtype=np.int64
            ) * CHUNK_SIZE
            centers = (global_idx + 0.5) * self.voxel_size
            all_centers.append(centers)
        if not all_centers:
            self._kdtree = None
            return
        self._kdtree = cKDTree(np.concatenate(all_centers, axis=0))
        self._kdtree_built_at_voxels = n

    def _check_drift(self, world_points: np.ndarray) -> dict | None:
        """Run a single Procrustes-ICP iteration of new world-points against
        the existing voxel-centre kd-tree. Returns None if the tree is empty
        or there's not enough overlap to estimate a rigid transform; otherwise
        the rigid (R, t) plus diagnostic stats. Caller decides whether to
        apply the transform based on overlap and residual.

        Algorithm: Kabsch / Umeyama. For matched pairs (src_i, dst_i),
            R = V · U^T  (with reflection fix from det)
            t = centroid_dst − R · centroid_src
        minimises Σ‖R·src_i + t − dst_i‖² in closed form.
        """
        tree = self._kdtree
        if tree is None or world_points.size == 0:
            return None
        dists, idxs = tree.query(world_points, k=1,
                                 distance_upper_bound=self.DRIFT_MATCH_RADIUS_M)
        finite = np.isfinite(dists)
        n_match = int(finite.sum())
        if n_match < self.DRIFT_MIN_MATCHES_FOR_ICP:
            # Below this the SVD-rotation estimate is too noisy.
            return None

        src = world_points[finite].astype(np.float64)
        dst = tree.data[idxs[finite]].astype(np.float64)
        median_dist_before = float(np.median(np.linalg.norm(src - dst, axis=1)))

        centroid_src = src.mean(axis=0)
        centroid_dst = dst.mean(axis=0)
        H = (src - centroid_src).T @ (dst - centroid_dst)
        U, _S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            # Pure-reflection solution; flip the smallest-singular-value
            # axis to land back in SO(3).
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = centroid_dst - R @ centroid_src

        # Residual after applying the transform (how well does ICP align?).
        src_transformed = src @ R.T + t
        median_dist_after = float(np.median(np.linalg.norm(src_transformed - dst, axis=1)))

        # Rotation magnitude in degrees, for diagnostics.
        cos_angle = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
        rot_deg = float(np.degrees(np.arccos(cos_angle)))

        return {
            "R": R.astype(np.float32),
            "t": t.astype(np.float32),
            "median_dist": median_dist_before,
            "median_dist_after": median_dist_after,
            "rot_deg": rot_deg,
            "match_fraction": float(n_match / world_points.shape[0]),
            "matches": n_match,
        }

    def ingest_frame(self, frame: dict[str, Any]) -> tuple[int, dict | None]:
        """Returns (n_points_written, diagnostic_or_None).

        n_points_written is -1 when the frame was rejected; the
        diagnostic dict identifies why ("jump_m" or "median_dist" /
        "match_fraction"). diagnostic is also returned for accepted
        frames when the kd-tree check ran, so callers can log it.
        """
        import fusion  # tools/ on sys.path when serve.py runs as a script
        points, cam, colors = fusion.frame_to_world_points(frame, with_colors=True)
        if points.size == 0:
            with self._lock:
                self._frames_ingested += 1
            return 0, None

        raw_cam = cam.astype(np.float32)   # before ICP correction; used for
                                            # frame-to-frame jump detection.

        with self._lock:
            # Pose-jump rejection: compare RAW camera position to the
            # previous RAW pose. ARCore re-localisations show up as a
            # sudden jump between consecutive raw poses. Update the anchor
            # on EVERY frame (accept or reject) so we re-sync to the new
            # world frame instead of rejecting in a loop.
            if self._last_cam_origin is not None:
                jump = float(np.linalg.norm(raw_cam - self._last_cam_origin))
                if jump >= self.POSE_JUMP_REJECT_M:
                    self._frames_rejected_jump += 1
                    self._frames_ingested += 1
                    self._last_cam_origin = raw_cam
                    return -1, {"jump_m": jump}
            self._last_cam_origin = raw_cam

            self._maybe_rebuild_kdtree()
            check = self._check_drift(points)

            # Procrustes ICP correction (rotation + translation). When the
            # new frame substantially overlaps existing scan content but
            # its points are systematically offset OR rotated, snap the
            # WHOLE frame to the existing surface before integrating. This
            # is what keeps gradual ARCore drift — translation AND
            # orientation — from accumulating into the fan-of-sheets that
            # builds up otherwise: each frame anchors itself to the
            # existing surface instead of trusting ARCore's drifting world
            # frame absolutely.
            #
            # Only correct when (a) we have enough overlap to trust the
            # measurement and (b) the residual is in single-iteration
            # ICP's reach (≤ 10 cm). Larger residuals indicate something
            # worse than gradual drift — reject those.
            if (check is not None
                and check["match_fraction"] >= self.DRIFT_MIN_MATCH_FRACTION):
                if check["median_dist"] >= self.DRIFT_REJECT_DIST_M:
                    self._frames_rejected_drift += 1
                    self._frames_ingested += 1
                    return -1, check
                # Apply rigid transform. Sanity caps: anything outside
                # ARCore's plausible per-frame drift (a few cm + a
                # fraction of a degree) is almost certainly the SVD
                # fitting to bad correspondences, not real drift. Drop
                # the correction in that case rather than baking in a
                # large spurious motion. Pose-jump rejection earlier
                # already handles the "honest, big" case.
                R = check["R"]
                t = check["t"]
                t_mag = float((t * t).sum() ** 0.5)
                rot_deg = check["rot_deg"]
                if (t_mag > self.ICP_MAX_TRANSLATION_M
                    or rot_deg > self.ICP_MAX_ROTATION_DEG):
                    check["correction_applied"] = False
                else:
                    if rot_deg > 0.05 or t_mag > 0.0005:
                        points = (points @ R.T + t).astype(np.float32)
                        self._frames_corrected += 1
                        check["correction_applied"] = True
                    else:
                        check["correction_applied"] = False

        n = self.insert_points(points, colors)
        with self._lock:
            self._frames_ingested += 1
        return n, check

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
                "frames_rejected_drift": self._frames_rejected_drift,
                "frames_rejected_jump": self._frames_rejected_jump,
                "frames_corrected": self._frames_corrected,
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
