#!/usr/bin/env python3
"""
Cache the raw (pre-affine) monocular-depth prediction for every frame in a
session, so the voxelview / stereo / depth-scatter pages can apply slider-
driven (a, b) refinement parameters without re-running the model on each
tick.

Two model families are supported (mutually-non-exclusive — keep both
caches around to A/B):

  --model-version v2   Depth-Anything-V2 Metric Indoor (default).
                       Output is roughly metres but unscaled; the UI fits
                       (a, b) per frame.
                       Cache dir:   captured_frames/<sess>/model_raw/

  --model-version v3   Depth-Anything-3 (ByteDance, 2025-11). Pick a
                       variant with --v3-variant:
                         nested  → DA3NESTED-GIANT-LARGE (~2 B params),
                                   output is in metres directly.
                         metric  → DA3METRIC-LARGE; we post-multiply by
                                   focal_processed/300 (per the official
                                   FAQ) to convert net output to metres.
                       Cache dir:   captured_frames/<sess>/model_raw_v3/

Output layout (per cache dir):
    frame_NNNNNN.f16    tightly packed float16 (ch_c × cw_c) at colour-image
                        resolution — same convention depth_refine.py and
                        the existing V2 cache use.
    index.json          {"<idx>": {"w": cw, "h": ch}, ...}
    info.json           one-shot summary of which model produced this cache:
                        {"model_version": "v3", "model_id": "...", "variant": "nested"}

Float16 is plenty for monocular depth; the dynamic range of ~0.1 to 8 m
needs only ~14 bits of precision and slider-tuning dominates the error.

V3 install note: the official Depth-Anything-3 package pins `xformers`
(CUDA-only) and hardcodes a CUDA/CPU autocast. To run on Apple Silicon
you'll likely need `--device cpu` (or a manually patched install). If
import fails, this script reports the package import error verbatim so
you can act on it.

Usage:
    # V2 (default — unchanged behaviour)
    python tools/cache_model_raw.py --session <id>

    # V3 nested-giant (metres directly)
    python tools/cache_model_raw.py --session <id> --model-version v3

    # V3 metric (smaller, but needs focal scaling)
    python tools/cache_model_raw.py --session <id> --model-version v3 \
        --v3-variant metric

Common flags:
    [--model <hf-id>]                Override the HF model id explicitly.
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

V2_DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
V3_VARIANT_TO_MODEL = {
    "nested": "depth-anything/DA3NESTED-GIANT-LARGE",
    "metric": "depth-anything/DA3METRIC-LARGE",
}


def cache_dir_for(session_dir: Path, model_version: str) -> Path:
    if model_version == "v2":
        return session_dir / "model_raw"
    if model_version == "v3":
        return session_dir / "model_raw_v3"
    raise ValueError(f"unknown model_version {model_version!r}")


# ────────────────────────── inference adapters ──────────────────────────

class _V2Adapter:
    """HuggingFace pipeline for Depth-Anything-V2 (the existing path)."""
    def __init__(self, model_id: str, device_str: str):
        import torch  # noqa: F401  (used for autocast/device check)
        from transformers import pipeline
        self._pipe = pipeline("depth-estimation", model=model_id,
                               device=device_str)

    def predict(self, rgb_pil: Image.Image, target_hw: tuple[int, int]) -> np.ndarray:
        """Run inference, return (ch, cw) float32 raw prediction."""
        import torch
        result = self._pipe(rgb_pil)
        pred = result["predicted_depth"]
        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
        elif pred.dim() == 3:
            pred = pred.unsqueeze(0)
        ch_, cw = target_hw
        pred_full = torch.nn.functional.interpolate(
            pred, size=(ch_, cw), mode="bilinear", align_corners=False,
        )[0, 0].detach().to("cpu").float().numpy()
        return pred_full


class _V3Adapter:
    """ByteDance-Seed depth_anything_3 wrapper. Lazily imports the package
    so V2 use never has to install it. variant="nested" returns metres
    directly; variant="metric" applies focal/300 post-scaling per the
    official FAQ."""
    def __init__(self, model_id: str, device_str: str, variant: str):
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as e:
            raise ImportError(
                "Depth-Anything-3 not installed.\n"
                "Install it with:\n"
                "    pip install torch torchvision\n"
                "    git clone https://github.com/ByteDance-Seed/Depth-Anything-3 /tmp/da3\n"
                "    pip install -e /tmp/da3\n"
                "Note: the official install pins xformers (CUDA-only); on\n"
                "Apple Silicon you may need to install with --no-deps and\n"
                "add only the actually-used deps. Original error:\n"
                f"    {e}"
            ) from e
        if variant not in ("nested", "metric"):
            raise ValueError(f"unknown v3 variant {variant!r}")
        self._variant = variant
        # Patch around the hardcoded cuda/cpu autocast for non-cuda runs.
        # The package's `forward` calls `torch.autocast(device_type=…, dtype=bf16/fp16)`.
        # On MPS/CPU this fails or is a no-op; bf16/fp16 weights aren't great
        # on the M2 anyway. Easiest robust path: load on CPU/MPS in fp32.
        import torch
        model = DepthAnything3.from_pretrained(model_id)
        if device_str == "mps":
            # MPS doesn't support bfloat16 in autocast; load in fp32 there.
            model = model.to(torch.float32)
        model = model.to(device_str).eval()
        self._model = model
        self._device = device_str

    def predict(self, rgb_pil: Image.Image, target_hw: tuple[int, int]) -> np.ndarray:
        import numpy as np
        import torch
        import torch.nn.functional as F
        # The DA3 inference helper expects PIL or numpy or path; pass PIL.
        with torch.inference_mode():
            try:
                pred = self._model.inference([rgb_pil])
            except RuntimeError as e:
                # Surface common Apple-Silicon failure modes with a clearer hint.
                msg = str(e)
                if "bf16" in msg.lower() or "bfloat16" in msg.lower():
                    raise RuntimeError(
                        "DA3 internal autocast tried bfloat16 (cuda-only path). "
                        "Re-run with --device cpu, or patch depth_anything_3.api "
                        "to skip the autocast wrapper."
                    ) from e
                raise
        depth_proc = pred.depth[0]      # (Hp, Wp) float32 numpy
        if self._variant == "metric":
            # Per the README FAQ: metric_depth = focal_processed/300 * net_output.
            K = pred.intrinsics[0]
            focal = 0.5 * float(K[0, 0] + K[1, 1])
            depth_proc = (focal / 300.0) * depth_proc
        # Resize back to the colour-image resolution (the cache contract).
        ch_, cw = target_hw
        t = torch.from_numpy(np.ascontiguousarray(depth_proc)).unsqueeze(0).unsqueeze(0)
        t_full = F.interpolate(t.float(), size=(ch_, cw),
                                mode="bilinear", align_corners=False)
        return t_full[0, 0].cpu().numpy()


def cache_session(
    session_dir: Path,
    model_version: str,
    model_id: str,
    v3_variant: str,
    device_str: str,
    max_frames: int | None,
    overwrite: bool,
) -> None:
    in_dir = session_dir / "frames"
    out_dir = cache_dir_for(session_dir, model_version)
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

    index_path = out_dir / "index.json"
    info_path = out_dir / "info.json"
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
    if device_str == "auto":
        if torch.backends.mps.is_available():
            device_str = "mps"
        elif torch.cuda.is_available():
            device_str = "cuda"
        else:
            device_str = "cpu"
    print(f"loading {model_version} model {model_id} on {device_str}…")
    t0 = time.time()
    if model_version == "v2":
        adapter = _V2Adapter(model_id, device_str)
    else:
        adapter = _V3Adapter(model_id, device_str, v3_variant)
    print(f"  model ready in {time.time()-t0:.1f} s")

    info_path.write_text(json.dumps({
        "model_version": model_version,
        "model_id": model_id,
        "variant": v3_variant if model_version == "v3" else None,
        "device": device_str,
    }, indent=2))

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

        pred_full = adapter.predict(rgb, target_hw=(ch_, cw)).astype(np.float16)

        out_path.write_bytes(pred_full.tobytes())
        index[str(idx)] = {"w": cw, "h": ch_}

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
    ap.add_argument("--model-version", choices=("v2", "v3"), default="v2",
                    help="Which model family to run (default v2).")
    ap.add_argument("--v3-variant", choices=tuple(V3_VARIANT_TO_MODEL.keys()),
                    default="nested",
                    help="V3 sub-variant: 'nested' = DA3NESTED-GIANT-LARGE "
                         "(metres directly, ~2 B params); 'metric' = "
                         "DA3METRIC-LARGE (smaller, focal-scaled).")
    ap.add_argument("--model", default=None,
                    help="HF model id; if unset, picked from --model-version "
                         "and --v3-variant.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.model is None:
        if args.model_version == "v2":
            model_id = V2_DEFAULT_MODEL
        else:
            model_id = V3_VARIANT_TO_MODEL[args.v3_variant]
    else:
        model_id = args.model

    session_dir = Path(args.frames_root) / args.session
    cache_session(
        session_dir,
        model_version=args.model_version,
        model_id=model_id,
        v3_variant=args.v3_variant,
        device_str=args.device,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
