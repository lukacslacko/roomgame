#!/usr/bin/env python3
"""
Cache the raw (pre-affine) Depth Anything V2 prediction for every frame in
a session, so the voxelview's pixel-cloud mode can apply slider-driven
(a, b) refinement parameters without re-running the model on each tick.

Output layout:
    captured_frames/<session>/model_raw/frame_NNNNNN.f16
    captured_frames/<session>/model_raw/index.json   ← {idx: {w, h}, ...}

Each .f16 file is a tightly packed float16 matrix of shape (h, w) at the
*colour image* resolution — same convention depth_refine.py uses for its
pred_full intermediate (see _resample_model_to_depth_grid). Float16 is
plenty for monocular depth (the model itself runs in fp32 but the dynamic
range of ~0.1 to 8.0 metres needs only ~14 bits of precision and the
slider tuning is the dominant error source here).

Usage:
    python tools/cache_model_raw.py --session <id>
        [--model depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf]
        [--device auto|mps|cuda|cpu]
        [--max-frames N]
        [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import serve  # noqa: E402  (parse_frame)

PROJECT_ROOT = ROOT.parent
DEFAULT_FRAMES_ROOT = PROJECT_ROOT / "captured_frames"

_FRAME_BIN_RE = re.compile(r"frame_(\d+)\.bin")


def cache_session(
    session_dir: Path,
    model_id: str,
    device_str: str,
    max_frames: int | None,
    overwrite: bool,
) -> None:
    in_dir = session_dir / "frames"
    out_dir = session_dir / "model_raw"
    if not in_dir.exists():
        print(f"no frames dir at {in_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(in_dir.glob("frame_*.bin"))
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        print(f"no frames in {in_dir}", file=sys.stderr)
        sys.exit(1)

    # Load existing index so re-runs only need to fill gaps.
    index_path = out_dir / "index.json"
    index: dict[str, dict] = {}
    if index_path.exists() and not overwrite:
        try:
            index = json.loads(index_path.read_text())
        except (OSError, json.JSONDecodeError):
            index = {}

    pending = []
    for fp in frame_paths:
        m = _FRAME_BIN_RE.match(fp.name)
        if not m:
            continue
        idx = int(m.group(1))
        out_path = out_dir / f"frame_{idx:06d}.f16"
        if not overwrite and out_path.exists() and str(idx) in index:
            continue
        pending.append((idx, fp, out_path))

    if not pending:
        print(f"all {len(frame_paths)} frames already cached "
              f"(use --overwrite to redo)")
        return
    print(f"caching {len(pending)} of {len(frame_paths)} frames "
          f"(skipping {len(frame_paths) - len(pending)} already done) "
          f"into {out_dir}")

    import torch
    from transformers import pipeline

    if device_str == "auto":
        if torch.backends.mps.is_available():
            device_str = "mps"
        elif torch.cuda.is_available():
            device_str = "cuda"
        else:
            device_str = "cpu"
    print(f"loading model {model_id} on {device_str}…")
    t0 = time.time()
    pipe = pipeline("depth-estimation", model=model_id, device=device_str)
    print(f"  model ready in {time.time()-t0:.1f} s")

    t_proc = time.time()
    for i, (idx, fp, out_path) in enumerate(pending):
        body = fp.read_bytes()
        try:
            frame = serve.parse_frame(body)
        except Exception as e:  # noqa: BLE001
            print(f"  parse error on {fp.name}: {e}")
            continue
        if frame["color"] is None:
            print(f"  {fp.name}: no colour, skipping")
            continue

        cw = int(frame["color_width"]); ch_ = int(frame["color_height"])
        rgba = np.frombuffer(frame["color"], dtype=np.uint8).reshape(ch_, cw, 4)
        rgb = Image.fromarray(rgba[..., :3])

        result = pipe(rgb)
        pred = result["predicted_depth"]
        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
        elif pred.dim() == 3:
            pred = pred.unsqueeze(0)
        pred_full = torch.nn.functional.interpolate(
            pred, size=(ch_, cw), mode="bilinear", align_corners=False,
        )[0, 0].detach().to("cpu").float().numpy().astype(np.float16)

        out_path.write_bytes(pred_full.tobytes())
        index[str(idx)] = {"w": cw, "h": ch_}

        # Persist the index frequently so a Ctrl-C doesn't lose progress.
        if (i + 1) % 10 == 0 or i + 1 == len(pending):
            index_path.write_text(json.dumps(index, indent=0))

        if (i + 1) % 5 == 0 or i == 0 or i + 1 == len(pending):
            elapsed = time.time() - t_proc
            eta = (elapsed / (i + 1)) * (len(pending) - i - 1)
            print(f"  [{i+1:4d}/{len(pending)}] frame_{idx:06d} "
                  f"{cw}x{ch_} → {out_path.stat().st_size//1024} KiB "
                  f"elapsed {elapsed:5.1f}s ETA {eta:5.1f}s")

    index_path.write_text(json.dumps(index, indent=0))
    print(f"\nWrote {len(pending)} model-raw caches to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--session", required=True)
    ap.add_argument("--frames-root", default=str(DEFAULT_FRAMES_ROOT))
    ap.add_argument("--model",
                    default="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    session_dir = Path(args.frames_root) / args.session
    cache_session(session_dir, args.model, args.device,
                  args.max_frames, args.overwrite)


if __name__ == "__main__":
    main()
