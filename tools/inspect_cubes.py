#!/usr/bin/env python3
"""
Offline visual inspection of the cube → camera-image projection.

Walks `captured_frames/`, replays every frame through a `CubeGrid`, then
picks the top-K cubes that were *fully visible* (in the cube-grid sense:
all 27 sample points project inside the frame with a valid depth) in
the most frames. For each of those cubes, picks K frames evenly spaced
through its visibility timeline, crops the camera image around the
cube's projected bounding box, draws the cube's 12 edges and 8 corners
on top, and assembles the cells into a single K×K PNG with one cube
per row.

Run with all defaults to inspect `captured_frames/` and write the PNG
to `cube_inspection.png` in the working directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import cubes as cubes_mod
import fusion
import serve


# Cube vertex order: bit 0 = x lo/hi, bit 1 = y lo/hi, bit 2 = z lo/hi.
# Edges connect vertices that differ in exactly one bit.
CUBE_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
]


def visible_cube_indices(grid: cubes_mod.CubeGrid, frame: dict) -> np.ndarray:
    """Global flat indices of cubes fully visible in `frame`.

    A cube is "fully visible" iff all 27 sample points project inside the
    camera frame with a non-zero depth measurement — same gate that
    `CubeGrid.ingest_frame` uses to decide whether to update counters.
    """
    width = int(frame["width"])
    height = int(frame["height"])
    depth = fusion.decode_depth(
        frame["depth"], width, height,
        int(frame["format"]), float(frame["rawValueToMeters"]),
    )
    V = fusion._mat4_from_column_major(frame["viewMatrix"])
    P = fusion._mat4_from_column_major(frame["projectionMatrix"])
    Bd = fusion._mat4_from_column_major(frame["normDepthBufferFromNormView"])
    V_inv = np.linalg.inv(V)

    # Coarse cube-centre cull, mirroring CubeGrid._ingest_frame_unlocked.
    centers = grid._cube_centers
    Nc = centers.shape[0]
    cen_h = np.concatenate([centers, np.ones((Nc, 1))], axis=1)
    cen_view = cen_h @ V_inv.T
    cen_view_xyz = cen_view[:, :3] / np.where(np.abs(cen_view[:, 3:4]) < 1e-12, 1.0, cen_view[:, 3:4])
    cen_dist = -cen_view_xyz[:, 2]
    cen_clip = np.concatenate([cen_view_xyz, np.ones((Nc, 1))], axis=1) @ P.T
    cen_cw = cen_clip[:, 3:4]
    cen_ndc = cen_clip[:, :2] / np.where(np.abs(cen_cw) < 1e-12, 1.0, cen_cw)
    cube_diag = grid.cube_size * np.sqrt(3.0)
    coarse_keep = (
        (cen_dist > 0.05)
        & (cen_dist < grid._max_depth_m + cube_diag)
        & (np.abs(cen_ndc[:, 0]) < 1.4)
        & (np.abs(cen_ndc[:, 1]) < 1.4)
    )
    keep_idx = np.nonzero(coarse_keep)[0]
    if keep_idx.size == 0:
        return np.array([], dtype=np.int64)

    sample_world = grid._sample_world[keep_idx].reshape(-1, 3)
    M = sample_world.shape[0]
    pts_h = np.concatenate([sample_world, np.ones((M, 1))], axis=1)
    view_h = pts_h @ V_inv.T
    w = view_h[:, 3:4]
    view = view_h[:, :3] / np.where(np.abs(w) < 1e-12, 1.0, w)
    view_dist = -view[:, 2]
    view4 = np.concatenate([view, np.ones((M, 1))], axis=1)
    clip = view4 @ P.T
    cw = clip[:, 3:4]
    ndc = clip[:, :3] / np.where(np.abs(cw) < 1e-12, 1.0, cw)
    u_v = (ndc[:, 0] + 1.0) * 0.5
    v_v = (ndc[:, 1] + 1.0) * 0.5
    nv_h = np.stack([u_v, v_v, np.zeros(M), np.ones(M)], axis=1)
    nd_h = nv_h @ Bd.T
    nw = nd_h[:, 3:4]
    u_d = (nd_h[:, 0:1] / np.where(np.abs(nw) < 1e-12, 1.0, nw)).ravel()
    v_d = (nd_h[:, 1:2] / np.where(np.abs(nw) < 1e-12, 1.0, nw)).ravel()
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
    measured = depth[by_clamped, bx_clamped]
    valid = in_frame & (measured > 0.0)
    Ns = keep_idx.size
    valid = valid.reshape(Ns, 27)
    cube_visible = np.all(valid, axis=1)
    return keep_idx[cube_visible].astype(np.int64)


def cube_corners_world(grid: cubes_mod.CubeGrid, cube_flat_idx: int) -> np.ndarray:
    """Return the 8 world-space corners of a cube as (8, 3)."""
    ix, iy, iz = grid._cube_index_flat[cube_flat_idx]
    lo = grid.world_min + np.array([ix, iy, iz], dtype=np.float64) * grid.cube_size
    sz = grid.cube_size
    out = np.zeros((8, 3), dtype=np.float64)
    for k in range(8):
        out[k, 0] = lo[0] + (sz if (k & 1) else 0.0)
        out[k, 1] = lo[1] + (sz if (k & 2) else 0.0)
        out[k, 2] = lo[2] + (sz if (k & 4) else 0.0)
    return out


def project_to_image_px(world_pts: np.ndarray, V: np.ndarray, P: np.ndarray,
                        cw: int, ch: int) -> np.ndarray:
    """Project (N, 3) world points to (N, 2) image pixel coords (PIL: x→right,
    y→down, origin top-left). NaN for any point at/behind the camera."""
    N = world_pts.shape[0]
    V_inv = np.linalg.inv(V)
    h = np.concatenate([world_pts, np.ones((N, 1))], axis=1)
    view = h @ V_inv.T
    view_xyz = view[:, :3] / np.where(np.abs(view[:, 3:4]) < 1e-12, 1.0, view[:, 3:4])
    behind = view_xyz[:, 2] >= -0.001
    view4 = np.concatenate([view_xyz, np.ones((N, 1))], axis=1)
    clip = view4 @ P.T
    cw_h = clip[:, 3:4]
    ndc = clip[:, :3] / np.where(np.abs(cw_h) < 1e-12, 1.0, cw_h)
    px = (ndc[:, 0] + 1.0) * 0.5 * cw
    # PIL Y goes top→bottom, NDC Y goes bottom→top: flip.
    py = (1.0 - (ndc[:, 1] + 1.0) * 0.5) * ch
    out = np.stack([px, py], axis=1)
    out[behind] = np.nan
    return out


def render_cube_in_frame(frame: dict, cube_flat_idx: int, grid: cubes_mod.CubeGrid,
                        cell_w: int, cell_h: int) -> Image.Image | None:
    """Crop the colour image of `frame` around the projected cube and draw
    its 12 edges + 8 corners. Returns a PIL RGB Image of (cell_w, cell_h),
    or None if the frame can't be rendered."""
    color_payload = frame["color"]
    cw = int(frame["color_width"])
    ch = int(frame["color_height"])
    if color_payload is None or cw == 0 or ch == 0:
        return None
    # gl.readPixels returns rows bottom-up; PIL wants top-down. Flip + drop alpha.
    color_arr = np.frombuffer(color_payload, dtype=np.uint8).reshape(ch, cw, 4)
    color_arr = color_arr[::-1, :, :3].copy()
    img = Image.fromarray(color_arr, mode="RGB")

    V = fusion._mat4_from_column_major(frame["viewMatrix"])
    P = fusion._mat4_from_column_major(frame["projectionMatrix"])

    corners = cube_corners_world(grid, cube_flat_idx)
    px = project_to_image_px(corners, V, P, cw, ch)
    if np.any(np.isnan(px)):
        return None

    min_x, min_y = np.min(px, axis=0)
    max_x, max_y = np.max(px, axis=0)
    bw = max_x - min_x
    bh = max_y - min_y
    # Generous padding so we get plenty of context around the cube — the
    # raw camera image is only 320×180 so we want most of the cell area to
    # be real pixels and most of those pixels to be the cube + immediate
    # surroundings.
    pad = max(12.0, 0.6 * max(bw, bh))
    cx_lo = float(min_x - pad)
    cy_lo = float(min_y - pad)
    cx_hi = float(max_x + pad)
    cy_hi = float(max_y + pad)

    # Match the cell's aspect ratio by expanding the shorter axis. This
    # avoids the "wide thin crop in a square cell" letterboxing that wastes
    # most of the cell as dark grey.
    target_aspect = cell_w / cell_h
    crop_w = cx_hi - cx_lo
    crop_h = cy_hi - cy_lo
    if crop_w / crop_h < target_aspect:
        needed = crop_h * target_aspect
        extra = (needed - crop_w) * 0.5
        cx_lo -= extra
        cx_hi += extra
    else:
        needed = crop_w / target_aspect
        extra = (needed - crop_h) * 0.5
        cy_lo -= extra
        cy_hi += extra

    cx_lo_i = int(np.clip(np.floor(cx_lo), 0, cw - 1))
    cy_lo_i = int(np.clip(np.floor(cy_lo), 0, ch - 1))
    cx_hi_i = int(np.clip(np.ceil(cx_hi), 1, cw))
    cy_hi_i = int(np.clip(np.ceil(cy_hi), 1, ch))
    if cx_hi_i <= cx_lo_i or cy_hi_i <= cy_lo_i:
        return None

    crop = img.crop((cx_lo_i, cy_lo_i, cx_hi_i, cy_hi_i)).convert("RGB")

    # Resize to cell, preserving aspect (small letterbox only when the bbox
    # extension hit an image edge and we couldn't keep the requested aspect).
    crop_aspect = crop.width / crop.height
    if abs(crop_aspect - target_aspect) < 1e-3:
        scaled = crop.resize((cell_w, cell_h), Image.LANCZOS)
    elif crop_aspect > target_aspect:
        new_w = cell_w
        new_h = max(1, int(round(cell_w / crop_aspect)))
        scaled = crop.resize((new_w, new_h), Image.LANCZOS)
    else:
        new_h = cell_h
        new_w = max(1, int(round(cell_h * crop_aspect)))
        scaled = crop.resize((new_w, new_h), Image.LANCZOS)

    cell = Image.new("RGB", (cell_w, cell_h), (28, 28, 32))
    ox = (cell_w - scaled.width) // 2
    oy = (cell_h - scaled.height) // 2
    cell.paste(scaled, (ox, oy))

    # Draw the cube edges/corners *after* resizing, scaling pixel coords from
    # the original crop into the resized image. Drawing post-resize keeps
    # the lines crisp at one-pixel-aliased width regardless of zoom factor.
    draw = ImageDraw.Draw(cell)
    sx = scaled.width / (cx_hi_i - cx_lo_i)
    sy = scaled.height / (cy_hi_i - cy_lo_i)
    local = (px - np.array([[cx_lo_i, cy_lo_i]])) * np.array([[sx, sy]])
    local = local + np.array([[ox, oy]])
    line_w = max(1, cell_h // 90)
    for a, b in CUBE_EDGES:
        x0, y0 = local[a]
        x1, y1 = local[b]
        draw.line([(x0, y0), (x1, y1)], fill=(255, 220, 0), width=line_w)
    r = max(2, cell_h // 60)
    for k in range(8):
        x, y = local[k]
        draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=(255, 80, 80), width=1)

    return cell


def pick_evenly(items: list, k: int) -> list:
    """Pick `k` items spaced as evenly as possible across `items`."""
    n = len(items)
    if n <= k:
        return list(items)
    return [items[int(round(i * (n - 1) / (k - 1)))] for i in range(k)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--frames-dir", default="captured_frames",
                    help="directory with frame_NNNNNN.bin bodies")
    ap.add_argument("--cube-size", type=float, default=0.25,
                    help="cube edge length in metres (default 0.25)")
    ap.add_argument("--top-cubes", type=int, default=10,
                    help="number of cubes to inspect (= rows in PNG)")
    ap.add_argument("--frames-per-cube", type=int, default=10,
                    help="number of frames per cube (= columns in PNG)")
    ap.add_argument("--cell-width", type=int, default=320,
                    help="pixel width of each cell in the output PNG")
    ap.add_argument("--cell-aspect", type=float, default=4.0/3.0,
                    help="aspect ratio (w/h) of each cell. Default 4:3 fits "
                         "the typical cube projection without letterboxing.")
    ap.add_argument("--out", default="cube_inspection.png",
                    help="output PNG path")
    ap.add_argument("--limit-frames", type=int, default=None,
                    help="only process the first N frames (debug)")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    frame_paths = sorted(frames_dir.glob("frame_*.bin"))
    if args.limit_frames:
        frame_paths = frame_paths[:args.limit_frames]
    if not frame_paths:
        print(f"No frames found in {frames_dir}")
        return
    print(f"Found {len(frame_paths)} frames in {frames_dir}")

    grid = cubes_mod.CubeGrid(cube_size=args.cube_size)
    print(f"Grid: shape={grid.shape} ({grid.n_cubes} cubes)")

    cube_visibility: dict[int, list[int]] = {}
    parsed_cache: dict[int, dict] = {}

    for fi, fp in enumerate(frame_paths):
        body = fp.read_bytes()
        try:
            frame = serve.parse_frame(body)
        except Exception as e:  # noqa: BLE001
            print(f"  frame {fi}: parse error {e}")
            continue
        if frame["color"] is None:
            continue
        visible = visible_cube_indices(grid, frame)
        for ci in visible:
            cube_visibility.setdefault(int(ci), []).append(fi)
        parsed_cache[fi] = frame
        if (fi + 1) % 25 == 0 or fi == len(frame_paths) - 1:
            print(f"  {fi+1}/{len(frame_paths)}: {len(visible)} visible cubes; "
                  f"tracked {len(cube_visibility)} cubes total")

    if not cube_visibility:
        print("No cubes ever fully visible. Try a smaller --cube-size or "
              "different frames.")
        return

    top = sorted(cube_visibility.items(), key=lambda kv: -len(kv[1]))[:args.top_cubes]
    print(f"\nTop {len(top)} cubes by visibility:")
    for rank, (ci, fis) in enumerate(top):
        idx = grid._cube_index_flat[ci]
        center = grid.world_min + (idx + 0.5) * grid.cube_size
        print(f"  #{rank}: flat_idx={ci} grid={tuple(idx)} "
              f"center=({center[0]:+.2f}, {center[1]:+.2f}, {center[2]:+.2f}) "
              f"frames_visible={len(fis)}")

    K = len(top)
    M = args.frames_per_cube
    cell_w = args.cell_width
    cell_h = max(1, int(round(cell_w / max(0.5, args.cell_aspect))))
    margin = 130

    canvas = Image.new("RGB", (margin + M * cell_w, K * cell_h), (16, 16, 20))
    draw = ImageDraw.Draw(canvas)

    for row, (ci, fis) in enumerate(top):
        picked = pick_evenly(fis, M)
        for col, fi in enumerate(picked[:M]):
            tile = render_cube_in_frame(parsed_cache[fi], ci, grid, cell_w, cell_h)
            if tile is None:
                tile = Image.new("RGB", (cell_w, cell_h), (60, 30, 30))
                d = ImageDraw.Draw(tile)
                d.text((6, 6), "no img", fill=(220, 200, 200))
            canvas.paste(tile, (margin + col * cell_w, row * cell_h))

        idx = grid._cube_index_flat[ci]
        center = grid.world_min + (idx + 0.5) * grid.cube_size
        label = (f"#{row}\n"
                 f"{len(fis)} frames\n"
                 f"({center[0]:+.2f},\n"
                 f" {center[1]:+.2f},\n"
                 f" {center[2]:+.2f})")
        draw.multiline_text((6, row * cell_h + 6), label, fill=(220, 220, 220), spacing=2)

    canvas.save(args.out)
    print(f"\nWrote {args.out} ({canvas.width}x{canvas.height}) "
          f"cell={cell_w}x{cell_h}")


if __name__ == "__main__":
    main()
