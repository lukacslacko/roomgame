#!/usr/bin/env python3
"""
Per-frame depth refinement using Depth Anything V2 (Metric Indoor).

For a captured session under captured_frames/<session>/frames/, this script
runs the model on every frame's RGB image, then linearly aligns the model's
output to the phone's lowres depth buffer (a per-frame scale + shift fit on
the valid pixels). The aligned depth is written into
captured_frames/<session>/frames_refined/ in the same wire format the Rust
voxeliser already understands — only the depth payload changes (re-emitted
as float32 metres so we don't quantise twice).

Why this shape and not e.g. depth completion?
    Depth Anything V2 has no public hint/conditioning input — it's an
    encoder-decoder DPT, RGB → relative depth. The "metric" finetune gives
    semi-absolute meters but per-frame scale and offset still drift. The
    phone's lowres ARCore depth is sparse but absolute, so a per-frame
    affine fit anchors the dense model output to the same scale.

Run:
    python tools/depth_refine.py --session <id>

Useful flags:
    --model         HF id (default: depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf)
    --device        cpu | mps | cuda  (default: auto, mps on Apple Silicon)
    --max-frames    debug
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import fusion
import serve

PROJECT_ROOT = ROOT.parent
DEFAULT_FRAMES_ROOT = PROJECT_ROOT / "captured_frames"

FRAME_HEADER_SIZE = serve.FRAME_HEADER_SIZE
DEPTH_FMT_FLOAT32 = 1

_FRAME_BIN_RE = re.compile(r"frame_(\d+)\.bin")


# --------------------------------------------------------------------------
# Frame re-encoding (preserve everything except depth payload + format).
# --------------------------------------------------------------------------

def _encode_refined_body(orig_body: bytes, refined_depth_m: np.ndarray) -> bytes:
    """Take the original frame body and return a new one with the depth
    payload replaced by `refined_depth_m`.

    The refined buffer can be at any resolution we like (we keep the same Bd
    coordinate frame as the original ARCore depth — that matrix maps view-UV
    to normalised depth-buffer-UV regardless of the buffer's pixel count,
    so denser is just denser). We rewrite depth_w / depth_h in the header
    accordingly. Format is forced to float32 metres (rawValueToMeters=1)
    so we don't quantise twice."""
    new_dh, new_dw = refined_depth_m.shape
    header = bytearray(orig_body[:FRAME_HEADER_SIZE])
    struct.pack_into("<I", header, 192, new_dw)            # depth_w
    struct.pack_into("<I", header, 196, new_dh)            # depth_h
    struct.pack_into("<f", header, 200, 1.0)               # rawValueToMeters
    struct.pack_into("<I", header, 204, DEPTH_FMT_FLOAT32) # depth_format

    orig_frame = serve.parse_frame(orig_body)
    new_depth = refined_depth_m.astype(np.float32, copy=False).tobytes()
    expected_bytes = new_dw * new_dh * 4
    if len(new_depth) != expected_bytes:
        raise ValueError(
            f"refined depth payload size mismatch: {len(new_depth)} bytes "
            f"vs expected {expected_bytes} for ({new_dh},{new_dw})"
        )
    color = orig_frame["color"] or b""
    return bytes(header) + new_depth + color


# --------------------------------------------------------------------------
# Model output → phone depth grid (using Bd^{-1} so rays line up).
# --------------------------------------------------------------------------

def _bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinearly sample a (H, W) array at fractional pixel coords.
    Coords outside the image return NaN."""
    h, w = img.shape
    inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x_c = np.clip(x, 0, w - 1 - 1e-3)
    y_c = np.clip(y, 0, h - 1 - 1e-3)
    x0 = np.floor(x_c).astype(np.int32); y0 = np.floor(y_c).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1);       y1 = np.minimum(y0 + 1, h - 1)
    fx = (x_c - x0).astype(np.float32);   fy = (y_c - y0).astype(np.float32)
    s = ((img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy)
         + (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy)
    return np.where(inside, s, np.nan)


def _resample_model_to_depth_grid(
    model_depth_color_grid: np.ndarray,   # (ch, cw)
    cw: int, ch_: int,
    Bd: np.ndarray, dw: int, dh: int,
) -> np.ndarray:
    """Resample a model depth map from the colour-image grid to the phone
    depth-buffer grid via Bd⁻¹ so the same view ray hits the same value in
    both. Output shape: (dh, dw)."""
    Bv = np.linalg.inv(Bd)  # normView ← normDepthBuffer
    bx, by = np.meshgrid(np.arange(dw, dtype=np.float64),
                         np.arange(dh, dtype=np.float64),
                         indexing="xy")
    # depth-buffer pixel → norm depth-buffer UV (matches the forward path)
    u_d = 1.0 - (bx + 0.5) / dw
    v_d = (by + 0.5) / dh
    nd_h = np.stack([u_d, v_d, np.zeros_like(u_d), np.ones_like(u_d)], axis=-1)
    nv_h = nd_h @ Bv.T
    safe_w = np.where(np.abs(nv_h[..., 3]) > 1e-12, nv_h[..., 3], 1.0)
    u = nv_h[..., 0] / safe_w
    v = nv_h[..., 1] / safe_w
    cx = u * cw - 0.5
    cy = v * ch_ - 0.5
    return _bilinear_sample(model_depth_color_grid, cx, cy)


# --------------------------------------------------------------------------
# Robust scale + shift fit.
# --------------------------------------------------------------------------

def _fit_affine(model_d: np.ndarray, phone_d: np.ndarray,
                near: float, far: float) -> tuple[float, float, int]:
    """Fit phone_d ≈ a * model_d + b on valid pixels via OLS, then trim
    outliers (|residual| > 3·MAD·1.4826) and refit once. Returns (a, b, n_used)."""
    mask = (
        (phone_d > near) & (phone_d < far)
        & np.isfinite(model_d)
        & (model_d > 1e-3)
    )
    n = int(mask.sum())
    if n < 100:
        return 1.0, 0.0, n
    x = model_d[mask].astype(np.float64)
    y = phone_d[mask].astype(np.float64)
    a, b = _ols(x, y)
    resid = y - (a * x + b)
    mad = np.median(np.abs(resid))
    if mad > 1e-6:
        keep = np.abs(resid) < 3.0 * mad * 1.4826
        if int(keep.sum()) > 100:
            a2, b2 = _ols(x[keep], y[keep])
            return a2, b2, int(keep.sum())
    return a, b, n


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xm, ym = float(x.mean()), float(y.mean())
    var = float(((x - xm) ** 2).mean())
    if var < 1e-9:
        return 1.0, ym - xm
    cov = float(((x - xm) * (y - ym)).mean())
    a = cov / var
    return a, ym - a * xm


# --------------------------------------------------------------------------
# Feature-anchor fit. Uses BA-triangulated 3D points + their per-frame
# keypoint observations as the affine fit target instead of the WebXR
# depth buffer. Far fewer points (typ. 50–500 per frame vs ~14k phone
# depth pixels) but each anchor is a metric ground-truth point — much
# tighter than ARCore's noisy depth, and immune to the "phone depth has
# its own scale drift" problem the .py header originally complained about.
# --------------------------------------------------------------------------

def _build_feature_anchors(meta_path: Path) -> dict[int, list[tuple[float, float, np.ndarray]]]:
    """Group features_meta observations by frame_idx → [(u, v, P_world), …].
    `u, v` are colour-buffer bottom-up [0, 1]² (the same convention
    feature_ray_reconstruct.py emits)."""
    meta = json.loads(meta_path.read_text())
    by_frame: dict[int, list[tuple[float, float, np.ndarray]]] = {}
    for v_meta in meta.get("voxels", []):
        for f in v_meta.get("features", []):
            P = np.asarray(f["world"], dtype=np.float64)
            for ob in f.get("obs", []):
                fi = int(ob["frame"])
                by_frame.setdefault(fi, []).append(
                    (float(ob["u"]), float(ob["v"]), P),
                )
    return by_frame


def _fit_affine_features(model_pred_full: np.ndarray,
                         anchors: list,
                         V_w2c: np.ndarray,
                         cw: int, ch_: int,
                         near: float, far: float
                         ) -> tuple[float, float, int]:
    """Fit the per-frame model→true depth correction in **disparity
    (1/d) space**:

        1 / true_depth  ≈  a · (1 / model_depth)  +  b

    even though Depth-Anything V2 metric outputs are nominally already
    in metres, monocular depth predictions are approximately linear in
    inverse-depth at the pixel level (the natural representation for
    DPT-style models pre-metric-head). Fitting in depth space gives
    OLS where far anchors dominate the variance — a ~5% relative model
    error at d=4 m contributes 16× the squared residual of the same
    error at d=1 m — and a single noisy far anchor can run away with
    the fit. In disparity space the residual budget is roughly
    constant per anchor and the fit is far more stable, especially
    with the ~150 anchors per frame the BA features give us.

    For each anchor (u, v, P_world):
      • true_depth = −(V_w2c · [P; 1])[z]   (camera-Z, +ve in front)
      • model_depth = bilinear sample of the model's prediction at the
        keypoint pixel (colour-buffer bottom-up grid).

    Returns (a, b, n_used) of the disparity-space fit. The caller
    inverts back to depth as  d_refined = M / (a + b · M)."""
    if not anchors:
        return 1.0, 0.0, 0
    M = len(anchors)
    P = np.stack([a[2] for a in anchors])             # (M, 3)
    P_h = np.concatenate([P, np.ones((M, 1))], axis=1)
    p_view = P_h @ V_w2c.T                            # (M, 4)
    true_depth = -p_view[:, 2]                        # (M,) +ve in front

    us = np.fromiter((a[0] for a in anchors), dtype=np.float64, count=M)
    vs = np.fromiter((a[1] for a in anchors), dtype=np.float64, count=M)
    sample_x = us * cw - 0.5
    sample_y = vs * ch_ - 0.5
    model_at_kp = _bilinear_sample(model_pred_full, sample_x, sample_y)

    mask = (
        np.isfinite(model_at_kp) & (model_at_kp > 1e-3)
        & np.isfinite(true_depth)
        & (true_depth > near) & (true_depth < far)
    )
    n = int(mask.sum())
    if n < 5:
        return 1.0, 0.0, n
    x = 1.0 / model_at_kp[mask].astype(np.float64)    # disparity_model
    y = 1.0 / true_depth[mask].astype(np.float64)     # disparity_true
    a, b = _ols(x, y)
    resid = y - (a * x + b)
    mad = float(np.median(np.abs(resid)))
    if mad > 1e-9 and n >= 10:
        keep = np.abs(resid) < 3.0 * mad * 1.4826
        if int(keep.sum()) >= 5:
            a2, b2 = _ols(x[keep], y[keep])
            return a2, b2, int(keep.sum())
    return a, b, n


def _apply_disparity_affine(pred: np.ndarray,
                            a: float, b: float,
                            near: float, far: float) -> np.ndarray:
    """Map model depth `M` to refined depth via the disparity-space fit:

            d_refined = M / (a + b · M)

    Pixels where `(a + b · M)` is not strictly positive — degenerate
    after the affine — are zeroed out so the rust voxeliser treats them
    as missing measurements (its `d > near` gate then drops them).
    Output is clipped to [0, far]."""
    denom = a + b * pred
    safe = denom > 1e-3
    out = np.where(safe, pred / np.where(safe, denom, 1.0), 0.0)
    out = np.clip(out, 0.0, far).astype(np.float32)
    return out


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------

def refine_session(
    session_dir: Path,
    model_id: str,
    device_str: str,
    near: float,
    far: float,
    max_frames: int | None,
    frames_variant: str = "frames",
    anchor: str = "phone",
) -> None:
    in_dir  = session_dir / frames_variant
    suffix  = frames_variant.removeprefix("frames")  # "" / "_aligned" / "_feature_ba" / …
    out_dir = session_dir / f"frames_refined{suffix}"
    if not in_dir.exists():
        print(f"no frames dir at {in_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional feature-anchor sidecar (only used when anchor='features').
    feature_anchors: dict[int, list] = {}
    if anchor == "features":
        meta_filename = ("features_meta.json" if not suffix
                         else f"features_meta{suffix}.json")
        meta_path = session_dir / meta_filename
        if not meta_path.exists():
            print(f"--anchor features needs {meta_path} — run "
                  f"feature_ray_reconstruct.py first", file=sys.stderr)
            sys.exit(1)
        feature_anchors = _build_feature_anchors(meta_path)
        n_obs = sum(len(v) for v in feature_anchors.values())
        print(f"loaded {n_obs} feature anchors across "
              f"{len(feature_anchors)} frames from {meta_path.name}")

    frame_paths = sorted(in_dir.glob("frame_*.bin"))
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        print(f"no frames in {in_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"refining {len(frame_paths)} frames in {in_dir} → {out_dir}  "
          f"(anchor={anchor})")

    # Lazy-import torch / transformers so a `--help` invocation is cheap.
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

    fits = []
    t_proc = time.time()
    for i, fp in enumerate(frame_paths):
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

        # Run model.
        result = pipe(rgb)
        # `predicted_depth` is a torch tensor; on the metric-indoor finetune
        # the values are already in metres. Resize to (ch_, cw) so the
        # subsequent Bd⁻¹ resample matches the colour image's pixel grid.
        pred = result["predicted_depth"]
        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
        elif pred.dim() == 3:
            pred = pred.unsqueeze(0)
        pred_full = torch.nn.functional.interpolate(
            pred, size=(ch_, cw), mode="bilinear", align_corners=False,
        )[0, 0].detach().to("cpu").float().numpy()

        Bd = fusion._mat4_from_column_major(frame["normDepthBufferFromNormView"])
        dw = int(frame["width"]); dh = int(frame["height"])

        # `fit_space` records whether (a, b) live in depth space or
        # disparity space, so the application formula matches the fit.
        if anchor == "features":
            # BA-feature anchor fit (disparity space — see
            # _fit_affine_features). Falls back to phone-depth OLS in
            # depth space if too few anchors or the resulting scale is
            # implausibly small/large/negative; "copy bytes through" if
            # both fits fail.
            m = _FRAME_BIN_RE.match(fp.name)
            idx = int(m.group(1)) if m else -1
            anchors = feature_anchors.get(idx, [])
            V_c2w = fusion._mat4_from_column_major(frame["viewMatrix"])
            V_w2c = np.linalg.inv(V_c2w)
            a, b, n_used = _fit_affine_features(
                pred_full, anchors, V_w2c, cw, ch_, near, far,
            )
            fit_space = "disparity"
            sane_a = 0.2 <= a <= 5.0
            if n_used < 5 or not sane_a:
                model_on_phone = _resample_model_to_depth_grid(pred_full, cw, ch_, Bd, dw, dh)
                phone_depth = fusion.decode_depth(
                    frame["depth"], dw, dh,
                    int(frame["format"]), float(frame["rawValueToMeters"]),
                )
                a, b, n_used = _fit_affine(model_on_phone, phone_depth, near, far)
                fit_space = "depth"
                if not (0.2 <= a <= 5.0):
                    # Both fits unstable; preserve the source frame's
                    # phone depth verbatim by copying the bytes through.
                    (out_dir / fp.name).write_bytes(body)
                    fits.append((1.0, 0.0, 0))
                    if (i + 1) % 5 == 0 or i == 0 or i + 1 == len(frame_paths):
                        elapsed = time.time() - t_proc
                        eta = (elapsed / (i + 1)) * (len(frame_paths) - i - 1)
                        print(f"  [{i+1:4d}/{len(frame_paths)}] {fp.name} "
                              f"BAD-FIT — copied source through "
                              f"elapsed {elapsed:5.1f}s ETA {eta:5.1f}s")
                    continue
        else:
            # Phone-depth fit (original behaviour, depth space).
            model_on_phone = _resample_model_to_depth_grid(pred_full, cw, ch_, Bd, dw, dh)
            phone_depth = fusion.decode_depth(
                frame["depth"], dw, dh,
                int(frame["format"]), float(frame["rawValueToMeters"]),
            )
            a, b, n_used = _fit_affine(model_on_phone, phone_depth, near, far)
            fit_space = "depth"

        # Apply the fit at colour resolution. Disparity-space fit gets
        # inverted to depth via d = M / (a + b·M); depth-space fit is
        # the original a·M + b.
        if fit_space == "disparity":
            refined_color = _apply_disparity_affine(pred_full, a, b, near, far)
        else:
            refined_color = (a * pred_full + b).astype(np.float32)
        target_dw, target_dh = cw, ch_
        refined_hi = _resample_model_to_depth_grid(
            refined_color, cw, ch_, Bd, target_dw, target_dh,
        )
        # NaNs (out-of-frame after Bd⁻¹) → 0 so downstream treats them as
        # "no measurement".
        refined_hi = np.where(np.isfinite(refined_hi), refined_hi, 0.0)
        refined_hi = np.clip(refined_hi, 0.0, far).astype(np.float32)

        new_body = _encode_refined_body(body, refined_hi)
        (out_dir / fp.name).write_bytes(new_body)

        fits.append((a, b, n_used))
        if (i + 1) % 5 == 0 or i == 0 or i + 1 == len(frame_paths):
            elapsed = time.time() - t_proc
            eta = (elapsed / (i + 1)) * (len(frame_paths) - i - 1)
            print(f"  [{i+1:4d}/{len(frame_paths)}] {fp.name} "
                  f"a={a:6.3f} b={b:+.3f} n={n_used:6d} "
                  f"elapsed {elapsed:5.1f}s ETA {eta:5.1f}s")

    if fits:
        a_arr = np.array([f[0] for f in fits])
        b_arr = np.array([f[1] for f in fits])
        n_arr = np.array([f[2] for f in fits])
        print(f"\nWrote {len(fits)} refined frames to {out_dir}")
        print(f"  scale a: median={np.median(a_arr):.3f} "
              f"p10={np.percentile(a_arr, 10):.3f} "
              f"p90={np.percentile(a_arr, 90):.3f}")
        print(f"  shift b: median={np.median(b_arr):+.3f} "
              f"p10={np.percentile(b_arr, 10):+.3f} "
              f"p90={np.percentile(b_arr, 90):+.3f}")
        print(f"  fit pixels: median={int(np.median(n_arr))}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--session", required=True,
                    help="session id (timestamp dir under captured_frames/)")
    ap.add_argument("--frames-root", default=str(DEFAULT_FRAMES_ROOT),
                    help="root holding session dirs (default: captured_frames)")
    ap.add_argument("--model",
                    default="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
                    help="HuggingFace model id")
    ap.add_argument("--device", default="auto",
                    help="cpu | mps | cuda | auto (default: auto)")
    ap.add_argument("--near", type=float, default=0.05)
    ap.add_argument("--far",  type=float, default=8.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--frames-variant", default="frames",
                    help="Source frames directory under "
                         "captured_frames/<session>/. Use frames_feature_ba "
                         "to refine on top of bundle-adjusted poses; output "
                         "lands in frames_refined<suffix>/ (suffix derived "
                         "from the variant name).")
    ap.add_argument("--anchor", choices=("phone", "features"), default="phone",
                    help="Per-frame affine fit target. `phone` uses the "
                         "WebXR depth buffer (original behaviour). "
                         "`features` uses the BA-triangulated 3D points "
                         "(features_meta_<suffix>.json) — far fewer points "
                         "per frame but each is a metric ground-truth "
                         "anchor, immune to ARCore depth's own scale "
                         "drift. Falls back to phone fit per-frame if a "
                         "frame has <5 feature observations.")
    args = ap.parse_args()

    session_dir = Path(args.frames_root) / args.session
    refine_session(
        session_dir, args.model, args.device,
        args.near, args.far, args.max_frames,
        frames_variant=args.frames_variant,
        anchor=args.anchor,
    )


if __name__ == "__main__":
    main()
