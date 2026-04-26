#!/usr/bin/env python3
"""
Offline replay of captured frames. Loads `captured_frames/*.bin` produced by
serve.py's /frame handler, runs them through fusion + voxel_store, and
prints per-frame diagnostics.

This is the debugging entry point: when something looks wrong on a phone
scan, we replay the same bytes here and instrument freely.

Usage:
    python tools/replay.py
    python tools/replay.py --frames captured_frames --voxel-size 0.03
    python tools/replay.py --mesh out/replay.glb        # also remesh
    python tools/replay.py --first 1 --last 10          # subset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add the tools/ dir to sys.path when run as a script.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np

from serve import parse_frame
import fusion
from voxel_store import VoxelRoom


def matrix_dump(label: str, m_col_major) -> str:
    """Full 4x4 matrix dump, plus translation + rotation determinant summary.

    Knowing the actual rotation block tells us if the phone is held portrait
    (90° rotation in normDepthBufferFromNormView) or landscape (identity-ish),
    and whether the world_from_view convention holds (translation matches the
    camera's position above the floor)."""
    M = np.asarray(m_col_major, dtype=np.float64).reshape(4, 4, order="F")
    trans = M[:3, 3]
    R = M[:3, :3]
    det = np.linalg.det(R)
    out = [f"{label}: t=({trans[0]:+.3f},{trans[1]:+.3f},{trans[2]:+.3f}) det(R)={det:+.3f}"]
    for row in M:
        out.append("  " + "  ".join(f"{v:+.4f}" for v in row))
    return "\n".join(out)


def project_center_pixel(frame) -> tuple[np.ndarray, np.ndarray, float]:
    """Sanity check: unproject the center depth pixel and compare its
    distance to the camera against the depth value. Returns (cam, world, depth)."""
    from fusion import frame_to_world_points
    w, h = frame["width"], frame["height"]
    if frame["format"] == 0:
        arr = np.frombuffer(frame["depth"], dtype=np.uint16).astype(np.float32)
    else:
        arr = np.frombuffer(frame["depth"], dtype=np.float32)
    metres = (arr * float(frame["rawValueToMeters"])).reshape(h, w)
    cy, cx = h // 2, w // 2
    d = float(metres[cy, cx])
    pts, cam, _rgb = frame_to_world_points(frame, near_m=0.001, far_m=100.0, with_colors=True)
    # Find the world point closest to the center ray's expected location.
    # As a crude proxy, the closest point to the camera by direction.
    # Just take the median of all points as a rough scene location.
    if pts.size == 0:
        return cam, np.array([float("nan")] * 3), d
    return cam, np.median(pts, axis=0), d


def replay(frames_dir: Path, voxel_size: float, near_m: float, far_m: float,
           first: int | None, last: int | None, mesh_out: Path | None,
           dump_first: bool) -> None:
    files = sorted(frames_dir.glob("frame_*.bin"))
    if not files:
        print(f"No frame files in {frames_dir}", file=sys.stderr)
        sys.exit(1)
    if first is not None:
        files = [f for f in files if int(f.stem.split("_")[1]) >= first]
    if last is not None:
        files = [f for f in files if int(f.stem.split("_")[1]) <= last]
    print(f"Replaying {len(files)} frames from {frames_dir}\n")

    room = VoxelRoom(voxel_size_m=voxel_size)

    for idx, path in enumerate(files):
        body = path.read_bytes()
        frame = parse_frame(body)

        W, H = frame["width"], frame["height"]
        if frame["format"] == 0:
            arr = np.frombuffer(frame["depth"], dtype=np.uint16).astype(np.float32)
        else:
            arr = np.frombuffer(frame["depth"], dtype=np.float32)
        metres = arr * float(frame["rawValueToMeters"])
        nonzero = metres[metres > 0]
        n_nonzero = int(nonzero.size)
        n_total = int(metres.size)

        if dump_first and idx == 0:
            print("---- first frame matrices ----")
            print(matrix_dump("viewMatrix (claimed: world_from_view)", frame["viewMatrix"]))
            print(matrix_dump("projectionMatrix                     ", frame["projectionMatrix"]))
            print(matrix_dump("normDepthBufferFromNormView          ", frame["normDepthBufferFromNormView"]))
            print(f"depth: {W}x{H} fmt={frame['format']} rawToM={frame['rawValueToMeters']:.6f}")
            if n_nonzero:
                print(f"depth values (m): "
                      f"min={nonzero.min():.3f} p10={np.percentile(nonzero,10):.3f} "
                      f"med={np.median(nonzero):.3f} p90={np.percentile(nonzero,90):.3f} "
                      f"max={nonzero.max():.3f}")
                bins = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 1000.0]
                hist, _ = np.histogram(nonzero, bins=bins)
                print("depth histogram (m):")
                for lo, hi, c in zip(bins[:-1], bins[1:], hist):
                    print(f"  [{lo:>5.2f}, {hi:>6.2f}) : {c}")
            print(f"color: {frame.get('color_width')}x{frame.get('color_height')} "
                  f"format={frame.get('color_format')} payload={len(frame['color']) if frame.get('color') else 0}B")
            cam, med, d = project_center_pixel(frame)
            print(f"sanity: cam={cam.tolist()}  median(world_pts)={med.tolist()}")
            print(f"        center depth={d:.3f}m  cam→median dist={np.linalg.norm(med - cam):.3f}m  "
                  f"(should be ≈ depth if pose+projection are consistent)")
            print("------------------------------\n")

        # Same code path as the server's ingest, but with extra reporting.
        pts, cam, rgb = fusion.frame_to_world_points(
            frame, near_m=near_m, far_m=far_m, with_colors=True
        )
        n_world = int(pts.shape[0])
        n_written = room.insert_points(pts, rgb) if n_world else 0
        stats = room.stats()
        color_tag = "rgb" if frame.get("color") is not None else "grey"
        # Median world point per frame: if the same wall is being scanned from
        # different camera positions, the medians should cluster (small std).
        # If the medians follow the camera around, fusion isn't applying the
        # pose's translation/rotation correctly.
        med = np.median(pts, axis=0) if n_world else np.array([np.nan] * 3)
        print(
            f"#{idx+1:3d} {path.name}  "
            f"cam=({cam[0]:+.2f},{cam[1]:+.2f},{cam[2]:+.2f}) "
            f"med_world=({med[0]:+.2f},{med[1]:+.2f},{med[2]:+.2f}) "
            f"dist={np.linalg.norm(med - cam):.2f}m "
            f"depth(nz={100*n_nonzero/max(1,n_total):.0f}% "
            f"med={np.median(nonzero) if n_nonzero else 0:.2f}) "
            f"col={color_tag} "
            f"+{n_written}pts → {stats['voxels']}vox/{stats['chunks']}ch"
        )

    print()
    print("---- final ----")
    print(room.stats())

    if mesh_out is not None:
        import meshing  # noqa: WPS433
        meta = meshing.remesh_to_glb(room, mesh_out)
        print("mesh:", meta)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="captured_frames",
                    help="directory of frame_NNNNNN.bin files (default: captured_frames)")
    ap.add_argument("--voxel-size", type=float, default=0.03,
                    help="voxel edge length in metres (default: 0.03 = 3 cm)")
    ap.add_argument("--near", type=float, default=0.05, help="near filter (m)")
    ap.add_argument("--far",  type=float, default=6.0,  help="far filter (m)")
    ap.add_argument("--first", type=int, default=None, help="first frame index to include")
    ap.add_argument("--last",  type=int, default=None, help="last frame index to include")
    ap.add_argument("--mesh",  default=None, help="if given, write a GLB mesh here after replay")
    ap.add_argument("--brief", action="store_true", help="skip the first-frame matrix dump")
    args = ap.parse_args()

    frames_dir = Path(args.frames)
    if not frames_dir.is_absolute():
        frames_dir = (HERE.parent / frames_dir).resolve()
    mesh_out = Path(args.mesh).resolve() if args.mesh else None
    replay(frames_dir, args.voxel_size, args.near, args.far,
           args.first, args.last, mesh_out, dump_first=not args.brief)


if __name__ == "__main__":
    main()
