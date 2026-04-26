"""
Convert a parsed WebXR depth frame into world-space points (numpy).

Wire-format primer (matches tools/serve.py FRAME_HEADER_FMT):
  viewMatrix                       — 4x4 column-major, world_from_view
  projectionMatrix                 — 4x4 column-major (clip = P @ view)
  normDepthBufferFromNormView      — 4x4 column-major; transforms normalized
                                     view coords [0,1]² (origin bottom-left)
                                     to normalized depth-buffer coords [0,1]²
                                     (per W3C Depth Sensing §3.6).
  width, height                    — depth buffer dims
  rawValueToMeters                 — multiply raw uint16 / float by this
  format                           — 0=uint16-LA, 1=float32

Conventions (WebGL / WebXR):
  - View space: camera at origin, +X right, +Y up, looking down -Z.
  - Depth value `d` is the Z-DISTANCE from the camera plane (positive),
    not radial distance. A point at depth d has view.z = -d.
  - viewMatrix here = view.transform.matrix in WebXR = world_from_view.

Pipeline per depth pixel (i, j):
  1. Decode raw depth → metres.
  2. Pixel-center → normalized depth-buffer coord u_d, v_d ∈ [0,1].
  3. Apply inverse(normDepthBufferFromNormView) → normalized view coord u_v, v_v.
  4. NDC: x_ndc = 2 u_v − 1, y_ndc = 2 v_v − 1.
  5. Unproject through inverse(projectionMatrix) to get a view-space ray dir.
  6. Scale the ray so view_z = -d. That's the view-space point.
  7. Multiply by viewMatrix → world-space point.

All steps are vectorized over the whole frame (one matmul per stage).
"""
from __future__ import annotations

from typing import Any
import numpy as np


def _mat4_from_column_major(values) -> np.ndarray:
    """16 column-major floats → numpy (4,4) row-major matrix M such that
    M @ v applies the same transform as the original column-major matrix."""
    return np.asarray(values, dtype=np.float64).reshape(4, 4, order="F")


def decode_depth(raw: bytes, width: int, height: int, fmt: int, raw_to_m: float) -> np.ndarray:
    """Decode the raw depth payload into a (height, width) float32 array of metres.

    Pixels with raw==0 are returned as 0.0 — callers should treat them as
    "no measurement" and skip them (see frame_to_world_points).
    """
    if fmt == 0:  # luminance-alpha: little-endian uint16 per pixel
        arr = np.frombuffer(raw, dtype=np.uint16).astype(np.float32)
    elif fmt == 1:  # float32
        arr = np.frombuffer(raw, dtype=np.float32).astype(np.float32)
    else:
        raise ValueError(f"unknown frame format {fmt}")
    if arr.size != width * height:
        raise ValueError(f"depth payload has {arr.size} samples, expected {width*height}")
    return (arr * float(raw_to_m)).reshape(height, width)


def frame_to_world_points(
    frame: dict[str, Any],
    *,
    near_m: float = 0.05,
    far_m: float = 6.0,
    floor_y_m: float = -100.0,   # disabled by default; flip the buffer Y axis
    stride: int = 1,             # was the actual cause of sub-floor points.
    with_colors: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unproject a parsed /frame dict to world-space points.

    Returns (points, view_origin_world) where:
      points: (N, 3) float32 array of world-space points (metres)
      view_origin_world: (3,) float32 — camera position in world coords
                          (useful for later TSDF/ray-carving)

    If `with_colors=True`, also returns a (N, 3) uint8 RGB array sampled from
    the frame's colour image at the same normalised view coords. Pixels with
    no colour (frame['color'] is None) get (128, 128, 128) grey.

    Filters out depth==0 (unobserved) and depth outside [near_m, far_m]
    (defends against ARCore "infinity" readings on glass/sky).

    `stride` lets you subsample for speed: stride=2 → quarter the points.
    """
    width = int(frame["width"])
    height = int(frame["height"])
    depth = decode_depth(
        frame["depth"], width, height, int(frame["format"]), float(frame["rawValueToMeters"])
    )

    V = _mat4_from_column_major(frame["viewMatrix"])             # world_from_view
    P = _mat4_from_column_major(frame["projectionMatrix"])       # clip_from_view
    Bd = _mat4_from_column_major(frame["normDepthBufferFromNormView"])
    P_inv = np.linalg.inv(P)
    Bv = np.linalg.inv(Bd)  # normView_from_normDepthBuffer

    if stride > 1:
        depth = depth[::stride, ::stride]
        height, width = depth.shape

    # Pixel grid → normalized depth-buffer coords (pixel centres).
    # The W3C Depth Sensing spec puts buffer (0,0) at the bottom-left (Y up),
    # but Chrome's actual array storage is row-0-at-top. So row index `j` of
    # the array sits at v_buffer = 1 − (j+0.5)/H, not (j+0.5)/H.
    # Confirmed empirically: with the naive (j+0.5)/H mapping the live
    # depth-overlay's vertical axis was inverted relative to the camera image.
    js = 1.0 - (np.arange(height, dtype=np.float64) + 0.5) / height
    is_ = (np.arange(width, dtype=np.float64) + 0.5) / width
    Ud, Vd = np.meshgrid(is_, js, indexing="xy")  # both shape (H, W)

    # Map [0,1]² → normalized view coords [0,1]² via inverse of B.
    # Bv is a 4x4; we treat (u_d, v_d, 0, 1) as a homogeneous point.
    pts_db = np.stack(
        [Ud, Vd, np.zeros_like(Ud), np.ones_like(Ud)], axis=-1
    )  # (H, W, 4)
    pts_v = pts_db @ Bv.T  # (H, W, 4)
    # Should still be near (u_v, v_v, 0, 1) because Bv only acts on x/y for
    # standard depth orientations, but we divide by w to be safe.
    w_safe = np.where(np.abs(pts_v[..., 3]) < 1e-12, 1.0, pts_v[..., 3])
    Uv = pts_v[..., 0] / w_safe
    Vv = pts_v[..., 1] / w_safe

    # NDC.
    x_ndc = 2.0 * Uv - 1.0
    y_ndc = 2.0 * Vv - 1.0

    # Unproject (x_ndc, y_ndc, -1, 1) through P_inv → view-space ray endpoint
    # on the near plane. Ray from origin (0,0,0) through this endpoint;
    # parameterize as p(t) = endpoint * t. We pick t so that p.z = -d.
    clip = np.stack(
        [x_ndc, y_ndc, -np.ones_like(x_ndc), np.ones_like(x_ndc)], axis=-1
    )  # (H, W, 4)
    view4 = clip @ P_inv.T  # (H, W, 4)
    w_safe = np.where(np.abs(view4[..., 3]) < 1e-12, 1.0, view4[..., 3])
    view3 = view4[..., :3] / w_safe[..., None]  # (H, W, 3) — point on near plane

    # Ray direction (from origin). Scale so view_z = -d (depth in metres).
    eps = 1e-6
    near_z = view3[..., 2]
    near_z_safe = np.where(np.abs(near_z) < eps, eps, near_z)  # avoid /0
    t = -depth / near_z_safe  # scale factor (broadcasts H×W against H×W)

    view_pts = view3 * t[..., None]  # (H, W, 3)
    # Mask: drop unobserved (depth==0), out-of-range pixels, and — crucially —
    # buffer pixels that don't correspond to any visible view ray. The
    # `normDepthBufferFromNormView` matrix maps view coords [0,1]² to a
    # sub-rectangle of the depth buffer; the strip outside that sub-rectangle
    # has Uv/Vv outside [0,1] and unprojecting those produces ghost points at
    # huge view angles (~17% of the buffer on a real Pixel scan).
    mask = (
        (depth > near_m) & (depth < far_m) & np.isfinite(depth)
        & (Uv >= 0.0) & (Uv <= 1.0)
        & (Vv >= 0.0) & (Vv <= 1.0)
    )
    view_pts_kept = view_pts[mask]  # (N, 3)

    # Lift to homogeneous, transform to world.
    view_h = np.concatenate([view_pts_kept, np.ones((view_pts_kept.shape[0], 1))], axis=1)  # (N, 4)
    world_h = view_h @ V.T  # (N, 4)
    world = (world_h[:, :3] / world_h[:, 3:4]).astype(np.float32)

    # Camera origin in world space = V @ (0, 0, 0, 1).
    cam_origin = V[:3, 3].astype(np.float32)

    # Snapshot the per-point view coords so we can keep them aligned with
    # `world` through the floor filter (used below for colour sampling).
    Uv_kept = Uv[mask]
    Vv_kept = Vv[mask]

    # Drop world points below `floor_y_m`. ARCore can return implausibly
    # large depth values (5–6 m) at extreme view angles which then project
    # well below the actual floor. The local-floor reference space puts y=0
    # at the floor, so a small negative tolerance handles depth noise.
    keep = world[:, 1] >= floor_y_m
    world = world[keep]
    Uv_kept = Uv_kept[keep]
    Vv_kept = Vv_kept[keep]

    if not with_colors:
        return world, cam_origin

    # Sample colour at the same normalised view coords as the surviving
    # points. scan.js renders the camera texture into a fullscreen quad in
    # view space, so output pixel (i, j) at center ((i+0.5)/W, (j+0.5)/H)
    # corresponds to the matching view coord directly.
    color_payload = frame.get("color")
    cw = int(frame.get("color_width") or 0)
    ch = int(frame.get("color_height") or 0)
    if color_payload is None or cw <= 0 or ch <= 0:
        rgb = np.full((world.shape[0], 3), 128, dtype=np.uint8)
        return world, cam_origin, rgb

    color_img = np.frombuffer(color_payload, dtype=np.uint8).reshape(ch, cw, 4)
    cx = np.clip(np.floor(Uv_kept * cw).astype(np.int32), 0, cw - 1)
    cy = np.clip(np.floor(Vv_kept * ch).astype(np.int32), 0, ch - 1)
    rgb = color_img[cy, cx, :3].copy()
    return world, cam_origin, rgb
