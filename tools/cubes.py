"""
Dense occupancy-cube grid driven by WebXR depth frames.

Each cell in a (Nx, Ny, Nz) grid keeps two saturating counters:
  - `occupied[i,j,k]`: number of frames in which this cube was fully visible
    AND the camera ray for at least one sample point inside the cube
    measured a surface within ±cube_size/2 of that sample's view-z.
  - `free[i,j,k]`: number of frames in which the cube was fully visible AND
    every valid sample measured a surface *beyond* the cube (so the camera
    saw straight through the cube's volume — empty space).

Frames where some samples projected outside the camera frame, had no
depth measurement, or measured something *closer* than the cube (i.e.
the cube is occluded by another surface) abstain — neither counter
ticks. That keeps occupancy = occupied / (occupied + free) honest:
high values mean the cube was directly observed multiple times *and*
the surface kept landing inside it.

The classifier is implemented per "sample point". Each cube is sampled
at 27 positions (a 3×3×3 grid spanning its volume from corner to
corner). Each sample is projected through (viewMatrix, projectionMatrix,
normDepthBufferFromNormView) to a depth-buffer pixel; the measured
metres there is compared with the sample's view-space z-distance.

Decisions are vectorized over (Ncubes × 27) sample points using numpy.
"""
from __future__ import annotations

from threading import Lock
from typing import Any
import numpy as np

import fusion


class CubeGrid:
    """Dense occupancy-cube grid populated from WebXR depth frames."""

    # Defaults sized for a typical room: 6 m east-west, 6 m north-south,
    # 3.2 m floor-to-ceiling (with a small headroom below floor for noise).
    DEFAULT_WORLD_MIN = (-3.0, -0.3, -3.0)
    DEFAULT_WORLD_MAX = ( 3.0,  2.9,  3.0)
    DEFAULT_CUBE_SIZE = 0.25

    def __init__(
        self,
        *,
        world_min: tuple[float, float, float] | None = None,
        world_max: tuple[float, float, float] | None = None,
        cube_size: float = DEFAULT_CUBE_SIZE,
    ) -> None:
        wmin = np.asarray(world_min if world_min is not None else self.DEFAULT_WORLD_MIN, dtype=np.float64)
        wmax = np.asarray(world_max if world_max is not None else self.DEFAULT_WORLD_MAX, dtype=np.float64)
        if cube_size <= 0:
            raise ValueError(f"cube_size must be > 0, got {cube_size}")
        if not np.all(wmax > wmin):
            raise ValueError(f"world_max must exceed world_min: {wmin} → {wmax}")

        self.cube_size = float(cube_size)
        self.shape = tuple(int(np.ceil((wmax[i] - wmin[i]) / self.cube_size)) for i in range(3))
        self.world_min = wmin
        # Snap world_max to the actual cube grid extent (may exceed user wmax slightly).
        self.world_max = (wmin + np.array(self.shape, dtype=np.float64) * self.cube_size)

        self.occupied = np.zeros(self.shape, dtype=np.uint32)
        self.free = np.zeros(self.shape, dtype=np.uint32)
        self.frames = 0
        self._lock = Lock()

        # Precompute the (Ncubes × 27 × 3) sample-point world coordinates so
        # ingest_frame is one big matmul + indexing per frame.
        s = np.array([0.0, 0.5, 1.0])
        ox, oy, oz = np.meshgrid(s, s, s, indexing="ij")
        self._sample_offsets = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1)  # (27, 3)

        ix, iy, iz = np.meshgrid(
            np.arange(self.shape[0]),
            np.arange(self.shape[1]),
            np.arange(self.shape[2]),
            indexing="ij",
        )
        self._cube_index_flat = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1).astype(np.int32)
        cube_corners = wmin[None, :] + self._cube_index_flat.astype(np.float64) * self.cube_size
        self._sample_world_flat = (
            cube_corners[:, None, :] + self.cube_size * self._sample_offsets[None, :, :]
        ).reshape(-1, 3).astype(np.float64)

    @property
    def n_cubes(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    def reset(self) -> None:
        with self._lock:
            self.occupied.fill(0)
            self.free.fill(0)
            self.frames = 0

    def ingest_frame(self, frame: dict[str, Any]) -> dict[str, int]:
        """Update occupied/free counts from a parsed /frame dict.
        Returns a small summary for logging."""
        with self._lock:
            return self._ingest_frame_unlocked(frame)

    def _ingest_frame_unlocked(self, frame: dict[str, Any]) -> dict[str, int]:
        width = int(frame["width"])
        height = int(frame["height"])
        depth = fusion.decode_depth(
            frame["depth"], width, height,
            int(frame["format"]), float(frame["rawValueToMeters"]),
        )  # (H, W) float32, 0 means "no measurement".

        V  = fusion._mat4_from_column_major(frame["viewMatrix"])              # world_from_view
        P  = fusion._mat4_from_column_major(frame["projectionMatrix"])        # clip_from_view
        Bd = fusion._mat4_from_column_major(frame["normDepthBufferFromNormView"])
        V_inv = np.linalg.inv(V)

        pts_world = self._sample_world_flat                                   # (M, 3)
        M = pts_world.shape[0]

        # World → view.
        pts_h  = np.concatenate([pts_world, np.ones((M, 1))], axis=1)
        view_h = pts_h @ V_inv.T
        w = view_h[:, 3:4]
        w_safe = np.where(np.abs(w) < 1e-12, 1.0, w)
        view = view_h[:, :3] / w_safe                                          # (M, 3)
        # In WebGL conventions the camera looks down -Z, so view-space points
        # *in front* of the camera have z<0. The "z-distance" used in the
        # depth buffer (and against which we compare) is therefore -view.z.
        view_dist = -view[:, 2]

        # View → clip → NDC.
        view4 = np.concatenate([view, np.ones((M, 1))], axis=1)
        clip = view4 @ P.T
        cw = clip[:, 3:4]
        cw_safe = np.where(np.abs(cw) < 1e-12, 1.0, cw)
        ndc = clip[:, :3] / cw_safe

        # NDC ∈ [-1, 1]² → normalized view coords ∈ [0, 1]².
        u_v = (ndc[:, 0] + 1.0) * 0.5
        v_v = (ndc[:, 1] + 1.0) * 0.5

        # Apply normDepthBufferFromNormView to (u_v, v_v, 0, 1) → (u_d, v_d).
        nv_h = np.stack([u_v, v_v, np.zeros(M), np.ones(M)], axis=1)
        nd_h = nv_h @ Bd.T
        nw = nd_h[:, 3:4]
        nw_safe = np.where(np.abs(nw) < 1e-12, 1.0, nw)
        u_d = (nd_h[:, 0:1] / nw_safe).ravel()
        v_d = (nd_h[:, 1:2] / nw_safe).ravel()

        # Per fusion.py: Chrome stores the buffer with column index running
        # opposite to the matrix's u_d direction; row follows v_d directly.
        bx = np.floor((1.0 - u_d) * width).astype(np.int32)
        by = np.floor(v_d * height).astype(np.int32)

        in_frame = (
            (view_dist > 0.05)
            & (u_v >= 0.0) & (u_v <= 1.0)
            & (v_v >= 0.0) & (v_v <= 1.0)
            & (u_d >= 0.0) & (u_d <= 1.0)
            & (v_d >= 0.0) & (v_d <= 1.0)
            & (bx >= 0) & (bx < width)
            & (by >= 0) & (by < height)
        )

        bx_clamped = np.clip(bx, 0, width - 1)
        by_clamped = np.clip(by, 0, height - 1)
        measured = depth[by_clamped, bx_clamped]                              # (M,)

        # Reshape (M,) → (Ncubes, 27).
        Nc = M // 27
        in_frame = in_frame.reshape(Nc, 27)
        measured = measured.reshape(Nc, 27)
        view_dist = view_dist.reshape(Nc, 27)

        valid = in_frame & (measured > 0.0)

        # Cube is "fully visible" iff every one of its 27 samples projects
        # into the frame *and* has a valid (non-zero) depth measurement.
        cube_visible = np.all(valid, axis=1)

        # For each visible sample, classify against the cube's surface band.
        half = 0.5 * self.cube_size
        delta = measured - view_dist                                          # >0 → measured beyond sample
        surface  = (np.abs(delta) <= half)
        free_pt  = (delta >  half)
        occluded = (delta < -half)

        any_surface  = np.any(surface  & valid, axis=1)
        any_occluded = np.any(occluded & valid, axis=1)

        is_occupied = cube_visible & any_surface
        is_free     = cube_visible & ~any_surface & ~any_occluded

        if np.any(is_occupied):
            idx = self._cube_index_flat[is_occupied]
            np.add.at(self.occupied, (idx[:, 0], idx[:, 1], idx[:, 2]), 1)
        if np.any(is_free):
            idx = self._cube_index_flat[is_free]
            np.add.at(self.free, (idx[:, 0], idx[:, 1], idx[:, 2]), 1)

        self.frames += 1
        return {
            "frame": self.frames,
            "visible": int(cube_visible.sum()),
            "occupied": int(is_occupied.sum()),
            "free": int(is_free.sum()),
        }

    def state(self, *, threshold: float = 0.25, min_observations: int = 2) -> dict[str, Any]:
        """Snapshot the grid as a JSON-serialisable dict.

        Returns every cube whose total observation count is at least
        `min_observations` *and* whose occupancy ratio exceeds `threshold`.
        That keeps the payload modest while preserving the cubes the viewer
        actually wants to render.
        """
        with self._lock:
            occ = self.occupied.copy()
            free = self.free.copy()
            frames = self.frames
            cube_size = self.cube_size
            world_min = self.world_min.tolist()
            shape = self.shape

        total = occ.astype(np.float32) + free.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(total > 0, occ / np.maximum(total, 1), 0.0)
        keep = (total >= min_observations) & (ratio >= threshold)
        ix, iy, iz = np.nonzero(keep)
        cubes = []
        for a, b, c in zip(ix.tolist(), iy.tolist(), iz.tolist()):
            cubes.append([
                int(a), int(b), int(c),
                int(occ[a, b, c]),
                int(free[a, b, c]),
            ])
        return {
            "cube_size": cube_size,
            "world_min": world_min,
            "shape": list(shape),
            "frames": int(frames),
            "threshold": threshold,
            "min_observations": min_observations,
            "n_returned": len(cubes),
            "cubes": cubes,
        }
