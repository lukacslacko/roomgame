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


def matrix_summary(label: str, m_col_major) -> str:
    """One-line summary of a column-major 4x4 matrix."""
    M = np.asarray(m_col_major, dtype=np.float64).reshape(4, 4, order="F")
    trans = M[:3, 3]
    det = np.linalg.det(M[:3, :3])
    return f"{label}: t=({trans[0]:+.2f},{trans[1]:+.2f},{trans[2]:+.2f}) det(R)={det:+.3f}"


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
            print(matrix_summary("viewMatrix (world_from_view)", frame["viewMatrix"]))
            print(matrix_summary("projectionMatrix             ", frame["projectionMatrix"]))
            print(matrix_summary("normDepthBufferFromNormView  ", frame["normDepthBufferFromNormView"]))
            print(f"depth: {W}x{H} fmt={frame['format']} rawToM={frame['rawValueToMeters']:.6f}")
            if n_nonzero:
                print(f"depth values (m): "
                      f"min={nonzero.min():.3f} p10={np.percentile(nonzero,10):.3f} "
                      f"med={np.median(nonzero):.3f} p90={np.percentile(nonzero,90):.3f} "
                      f"max={nonzero.max():.3f}")
                # Histogram of depth ranges to spot whether the [near,far] gate is suspect.
                bins = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 1000.0]
                hist, _ = np.histogram(nonzero, bins=bins)
                print("depth histogram (m):")
                for lo, hi, c in zip(bins[:-1], bins[1:], hist):
                    print(f"  [{lo:>5.2f}, {hi:>6.2f}) : {c}")
            print("------------------------------\n")

        # Run through the same code path the server uses, but tell us how
        # many points pass each filter stage.
        pts, cam = fusion.frame_to_world_points(frame, near_m=near_m, far_m=far_m)

        n_world = int(pts.shape[0])
        n_written = room.insert_points(pts) if n_world else 0
        stats = room.stats()
        print(
            f"#{idx+1:3d} {path.name}  {W}x{H}  "
            f"depth(nz={100*n_nonzero/max(1,n_total):.0f}% "
            f"min={nonzero.min() if n_nonzero else 0:.2f} "
            f"max={nonzero.max() if n_nonzero else 0:.2f}) "
            f"-> world_pts={n_world}  +vox_writes={n_written}  "
            f"total={stats['voxels']}vox/{stats['chunks']}ch"
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
