#!/usr/bin/env python3
"""
Multi-view depth refinement: scale Depth Anything V2 outputs frame-to-frame
using inter-frame geometric consistency, instead of fitting each frame's
affine independently against the phone's lowres depth.

Why we tried this:
    The phone-anchored fit in tools/depth_refine.py runs an OLS of phone
    depth × model depth per frame. The phone depth is sparse and noisy,
    so the per-frame (a, b) values jitter (e.g. a ranged 0.39–0.89 on a
    98-frame session). The hypothesis: each frame's geometry then sits
    on a slightly different metric scale, which smears walls when the
    same surface is seen from different views.

    Multi-view alignment instead asks: given frame k's depth (with an
    estimated a_k, b_k) and the captured pose, where does each pixel
    project in frame k+1? It must land at a depth consistent with frame
    k+1's model output. That gives a linear constraint on (a_{k+1},
    b_{k+1}) per matched sample, which we solve with LS.

What actually happened (TL;DR — forward-chain alone is unstable):
    On the test session (217 frames, single end-of-session loop) the
    chain drifted dramatically: a slid from 0.48 at frame 0 to 0.13 by
    frame 19 and went negative around frame 75. Each pair contributes
    a small bias from feature noise + the linearised projection, and
    the bias compounds.
    `--reanchor-every N` (re-fit against phone depth every N frames)
    bounds the drift but is essentially "phone fit + a few chained
    frames between anchors" which converges back toward the per-frame
    phone fit as N → 1.
    The right fix is a global least-squares solve over all frames at
    once, with a smoothness prior on neighbouring (a, b) pairs and a
    soft tie to the phone-depth fit per frame. That's left as future
    work — this script is here as a honest negative result + a
    foundation if someone wants to layer the global LS on top.

Pipeline:
  1. Run Depth Anything V2 once per frame; cache the model depths.
  2. Anchor frame 0's affine — choices via `--anchor`:
       phone     (default): fit (a_0, b_0) against the phone depth on
                            frame 0 only.
       identity:            (a, b) = (1, 0). Pure model output.
  3. Forward chain. For each k = 0 … N−2:
        sample a sparse grid in frame k
        project each sample with R_k = a_k·M_k + b_k → world point
        forward-project into frame k+1 → (u, v, expected_dist)
        accumulate constraint  a_{k+1}·M_{k+1}(u,v) + b_{k+1} = dist
        LS-solve (a_{k+1}, b_{k+1}) with one MAD-trim refit pass.
     `--reanchor-every N` re-fits against phone depth every N frames
     (bounds drift; with N=1 it degenerates to per-frame phone fit).
  4. Apply (a_k, b_k); rewrite each .bin into frames_refined_mv/ at
     colour resolution, same wire format as the phone-anchored refiner
     so the Rust voxeliser reads it natively.

Run:
    python tools/depth_refine_mv.py --session <id>
    python tools/depth_refine_mv.py --session <id> --anchor identity
    python tools/depth_refine_mv.py --session <id> --reanchor-every 10
"""
from __future__ import annotations

import argparse
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

# Reuse the existing helpers; they're already tested.
from depth_refine import (
    _resample_model_to_depth_grid,
    _fit_affine,
    _encode_refined_body,
    DEPTH_FMT_FLOAT32,
)
FRAME_HEADER_SIZE = serve.FRAME_HEADER_SIZE

PROJECT_ROOT = ROOT.parent
DEFAULT_FRAMES_ROOT = PROJECT_ROOT / "captured_frames"


# ---------------------------------------------------------------------
# Per-frame back/forward projection at colour resolution.
# ---------------------------------------------------------------------

def _backproject_grid(
    P: np.ndarray, V: np.ndarray, cw: int, ch: int,
    n_grid: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For a sparse n_grid × n_grid raster of (u, v) in [0, 1]² of frame's
    view UV, return:
        uvs   (G, 2)   the sample UVs
        rays  (G, 3)   unit world rays from cam_origin through each pixel
        cam   (3,)     the camera origin in world coords

    Convention matches voxel_reconstruct.py + depth_refine.py: v = 0 at
    bottom of view, v = 1 at top (OpenGL view space)."""
    g = np.linspace(0.5 / n_grid, 1.0 - 0.5 / n_grid, n_grid, dtype=np.float64)
    U, Vg = np.meshgrid(g, g, indexing="xy")              # (G, G)
    uvs = np.stack([U.ravel(), Vg.ravel()], axis=-1)      # (n_grid², 2)

    P_inv = np.linalg.inv(P)
    cam = V[:3, 3].astype(np.float64)
    x_ndc = 2.0 * uvs[:, 0] - 1.0
    y_ndc = 2.0 * uvs[:, 1] - 1.0
    clip = np.stack([x_ndc, y_ndc,
                     -np.ones_like(x_ndc), np.ones_like(x_ndc)], axis=-1)
    view4 = clip @ P_inv.T
    view3 = view4[:, :3] / np.where(np.abs(view4[:, 3:4]) < 1e-12, 1.0, view4[:, 3:4])
    # ray_dir is the unit vector from cam through the image-plane point in
    # view space; rotate to world space via V's 3×3 sub-block.
    rays_view = view3 / np.linalg.norm(view3, axis=-1, keepdims=True)
    rays_world = rays_view @ V[:3, :3].T
    return uvs, rays_world, cam


def _project_world_to_uv(
    world_pts: np.ndarray, P: np.ndarray, V: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-project world points into a frame's view UV.
    Returns (uvs (N, 2), dist (N,), in_view (N, bool))."""
    cam = V[:3, 3].astype(np.float64)
    V_inv = np.linalg.inv(V)
    n = world_pts.shape[0]
    pts_h = np.concatenate([world_pts, np.ones((n, 1))], axis=-1)
    view_h = pts_h @ V_inv.T
    clip_h = view_h @ P.T
    w = clip_h[:, 3]
    safe = np.where(np.abs(w) > 1e-12, w, 1.0)
    ndc_x = clip_h[:, 0] / safe
    ndc_y = clip_h[:, 1] / safe
    u = 0.5 * (ndc_x + 1.0)
    v = 0.5 * (ndc_y + 1.0)
    in_front = view_h[:, 2] < 0.0
    in_uv    = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    in_view  = in_front & in_uv & (np.abs(w) > 1e-9)
    dist = np.linalg.norm(world_pts - cam[None, :], axis=-1)
    return np.stack([u, v], axis=-1), dist, in_view


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample of a (H, W) array at fractional pixel coords. Out-of-
    bounds returns NaN."""
    h, w = img.shape
    inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    xc = np.clip(x, 0, w - 1 - 1e-3)
    yc = np.clip(y, 0, h - 1 - 1e-3)
    x0 = np.floor(xc).astype(np.int32); y0 = np.floor(yc).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1);     y1 = np.minimum(y0 + 1, h - 1)
    fx = (xc - x0).astype(np.float32);  fy = (yc - y0).astype(np.float32)
    s = ((img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy)
         + (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy)
    return np.where(inside, s, np.nan)


# ---------------------------------------------------------------------
# Pair-wise affine fit.
# ---------------------------------------------------------------------

def _solve_pair(
    M_k: np.ndarray, M_k1: np.ndarray,
    a_k: float, b_k: float,
    P_k: np.ndarray, V_k: np.ndarray,
    P_k1: np.ndarray, V_k1: np.ndarray,
    cw: int, ch: int,
    n_grid: int = 30,
    near: float = 0.1, far: float = 8.0,
) -> tuple[float, float, int]:
    """Given frame k's affine (a_k, b_k), fit (a_{k+1}, b_{k+1}) such that
    sample world points seen from frame k land at consistent depths in
    frame k+1. Returns (a, b, n_used)."""
    uvs, rays, cam = _backproject_grid(P_k, V_k, cw, ch, n_grid=n_grid)
    # Sample model depth at each grid UV in frame k.
    cx_k = uvs[:, 0] * cw - 0.5
    cy_k = uvs[:, 1] * ch - 0.5
    m_k = _bilinear(M_k, cx_k, cy_k)
    valid_k = np.isfinite(m_k) & (m_k > 1e-3)
    if int(valid_k.sum()) < 30:
        return 1.0, 0.0, 0

    d_k = a_k * m_k[valid_k] + b_k
    valid_k_idx = np.where(valid_k)[0]
    rays_v = rays[valid_k_idx]
    pts = cam[None, :] + d_k[:, None] * rays_v
    # Drop anything that lands beyond far / behind near.
    range_ok = (d_k > near) & (d_k < far)
    pts = pts[range_ok]
    if pts.shape[0] < 30:
        return 1.0, 0.0, 0

    uvs_n, dist_n, in_view_n = _project_world_to_uv(pts, P_k1, V_k1)
    if int(in_view_n.sum()) < 30:
        return 1.0, 0.0, 0
    uvs_n = uvs_n[in_view_n]; dist_n = dist_n[in_view_n]

    cx_n = uvs_n[:, 0] * cw - 0.5
    cy_n = uvs_n[:, 1] * ch - 0.5
    m_n = _bilinear(M_k1, cx_n, cy_n)
    finite = np.isfinite(m_n) & (m_n > 1e-3)
    if int(finite.sum()) < 30:
        return 1.0, 0.0, 0
    x = m_n[finite].astype(np.float64)
    y = dist_n[finite].astype(np.float64)
    A = np.stack([x, np.ones_like(x)], axis=-1)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    # MAD-trim outliers and refit once.
    resid = y - (a * x + b)
    mad = float(np.median(np.abs(resid)))
    if mad > 1e-6:
        keep = np.abs(resid) < 3.0 * mad * 1.4826
        if int(keep.sum()) >= 30:
            sol2, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
            a, b = float(sol2[0]), float(sol2[1])
            return a, b, int(keep.sum())
    return a, b, int(finite.sum())


# ---------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------

def _decode_frame_for_model(body: bytes):
    """Returns (frame_dict, rgb_pil, M_decoded_phone_depth_or_None,
    Bd, dw, dh, cw, ch) for use in the pipeline."""
    frame = serve.parse_frame(body)
    if frame["color"] is None:
        return None
    cw = int(frame["color_width"]); ch = int(frame["color_height"])
    rgba = np.frombuffer(frame["color"], dtype=np.uint8).reshape(ch, cw, 4)
    rgb = Image.fromarray(rgba[..., :3])
    Bd = fusion._mat4_from_column_major(frame["normDepthBufferFromNormView"])
    dw = int(frame["width"]); dh = int(frame["height"])
    return frame, rgb, Bd, dw, dh, cw, ch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--frames-root", default=str(DEFAULT_FRAMES_ROOT))
    ap.add_argument("--source-dir", default="frames",
                    help="subdir under captured_frames/<session>/ to read "
                         "(default `frames`; pass `frames_aligned` if you've "
                         "run loop closure first)")
    ap.add_argument("--output-dir", default="frames_refined_mv",
                    help="subdir under captured_frames/<session>/ to write")
    ap.add_argument("--model",
                    default="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--near", type=float, default=0.05)
    ap.add_argument("--far",  type=float, default=8.0)
    ap.add_argument("--n-grid", type=int, default=30,
                    help="√(samples per pair) — 30 → 900 grid points/frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--anchor", choices=["phone", "identity"], default="phone",
                    help="`phone` fits (a₀, b₀) on frame 0's phone depth and "
                         "uses that as the global scale anchor (kept for "
                         "frame 0 only). `identity` skips the phone depth "
                         "entirely and trusts the model's metric output as "
                         "(a, b) = (1, 0) for frame 0.")
    ap.add_argument("--reanchor-every", type=int, default=0,
                    help="Periodically re-fit (a_k, b_k) against the phone "
                         "depth every N frames, capping chain drift to "
                         "≤ N−1 frames. 0 = pure chain (drifts unboundedly "
                         "on long sessions).")
    args = ap.parse_args()

    session_dir = Path(args.frames_root) / args.session
    in_dir  = session_dir / args.source_dir
    out_dir = session_dir / args.output_dir
    if not in_dir.exists():
        print(f"no source dir at {in_dir}", file=sys.stderr); sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    bins = sorted(in_dir.glob("frame_*.bin"))
    if args.max_frames is not None:
        bins = bins[:args.max_frames]
    if not bins:
        print(f"no frames in {in_dir}", file=sys.stderr); sys.exit(1)
    print(f"multi-view refining {len(bins)} frames from {in_dir}")

    # ----- 1. Run Depth Anything per frame -----
    import torch
    from transformers import pipeline
    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else (
                 "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
    print(f"loading model {args.model} on {device}…")
    t0 = time.time()
    pipe = pipeline("depth-estimation", model=args.model, device=device)
    print(f"  model ready in {time.time()-t0:.1f} s")

    M_all: list[np.ndarray] = []   # (N, ch, cw) model depth, or None
    P_all: list[np.ndarray] = []
    V_all: list[np.ndarray] = []
    Bd_all: list[np.ndarray] = []
    dw_all: list[int] = []
    dh_all: list[int] = []
    cw0 = ch0 = None
    bodies: list[bytes] = []

    print(f"running depth model on each frame…")
    t_inf = time.time()
    for i, fp in enumerate(bins):
        body = fp.read_bytes()
        bodies.append(body)
        decoded = _decode_frame_for_model(body)
        if decoded is None:
            M_all.append(None); P_all.append(None); V_all.append(None)
            Bd_all.append(None); dw_all.append(0); dh_all.append(0)
            continue
        frame, rgb, Bd, dw, dh, cw, ch = decoded
        if cw0 is None:
            cw0, ch0 = cw, ch
        elif (cw, ch) != (cw0, ch0):
            raise SystemExit(f"frame {i}: colour res {(cw, ch)} != "
                             f"first frame's {(cw0, ch0)} — non-uniform sessions "
                             f"aren't supported by the multiview chain")
        result = pipe(rgb)
        pred = result["predicted_depth"]
        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
        elif pred.dim() == 3:
            pred = pred.unsqueeze(0)
        pred_full = torch.nn.functional.interpolate(
            pred, size=(ch, cw), mode="bilinear", align_corners=False,
        )[0, 0].detach().to("cpu").float().numpy()
        M_all.append(pred_full)
        V = fusion._mat4_from_column_major(frame["viewMatrix"])
        P = fusion._mat4_from_column_major(frame["projectionMatrix"])
        P_all.append(P); V_all.append(V); Bd_all.append(Bd)
        dw_all.append(dw); dh_all.append(dh)
        if (i + 1) % 10 == 0 or i == len(bins) - 1:
            elapsed = time.time() - t_inf
            eta = (elapsed / (i + 1)) * (len(bins) - i - 1)
            print(f"  [{i+1:4d}/{len(bins)}] {fp.name}  "
                  f"elapsed {elapsed:5.1f}s  ETA {eta:5.1f}s")
    print(f"  done in {time.time()-t_inf:.1f} s")

    # ----- 2. Anchor frame 0 -----
    abs_path = lambda i: bins[i]   # noqa: E731
    a0 = b0 = None
    first_valid = next((i for i, m in enumerate(M_all) if m is not None), None)
    if first_valid is None:
        print("no usable frames"); return
    if args.anchor == "phone":
        # Re-run the existing phone-depth fit on frame 0 only.
        frame0 = serve.parse_frame(bodies[first_valid])
        phone_depth = fusion.decode_depth(
            frame0["depth"], dw_all[first_valid], dh_all[first_valid],
            int(frame0["format"]), float(frame0["rawValueToMeters"]),
        )
        model0_on_phone = _resample_model_to_depth_grid(
            M_all[first_valid], cw0, ch0, Bd_all[first_valid],
            dw_all[first_valid], dh_all[first_valid],
        )
        a0, b0, n_used = _fit_affine(model0_on_phone, phone_depth, args.near, args.far)
        print(f"\nanchor (phone, frame {first_valid}): "
              f"a={a0:.3f} b={b0:+.3f} n_pixels={n_used}")
    else:
        a0, b0 = 1.0, 0.0
        print(f"\nanchor (identity, frame {first_valid}): "
              f"a={a0:.3f} b={b0:+.3f}")

    # ----- 3. Forward chain -----
    a_arr = np.zeros(len(bins))
    b_arr = np.zeros(len(bins))
    n_arr = np.zeros(len(bins), dtype=np.int32)
    a_arr[first_valid] = a0
    b_arr[first_valid] = b0

    def _phone_reanchor(idx: int) -> tuple[float, float, int]:
        """Re-fit (a_k, b_k) against phone depth on frame idx (same logic as
        the anchor step). Used for periodic re-anchoring to bound drift."""
        frame_k = serve.parse_frame(bodies[idx])
        phone_depth_k = fusion.decode_depth(
            frame_k["depth"], dw_all[idx], dh_all[idx],
            int(frame_k["format"]), float(frame_k["rawValueToMeters"]),
        )
        model_k_on_phone = _resample_model_to_depth_grid(
            M_all[idx], cw0, ch0, Bd_all[idx],
            dw_all[idx], dh_all[idx],
        )
        return _fit_affine(model_k_on_phone, phone_depth_k, args.near, args.far)

    print(f"chaining forward, n_grid = {args.n_grid} "
          f"({args.n_grid**2} pixels/pair) "
          f"{'reanchor every ' + str(args.reanchor_every) if args.reanchor_every else 'no reanchor'}…")
    n_chained = 0
    n_reanchored = 0
    for k in range(first_valid, len(bins) - 1):
        if M_all[k] is None or M_all[k+1] is None:
            a_arr[k+1] = a_arr[k]; b_arr[k+1] = b_arr[k]
            continue
        # Periodic re-anchor: if we're at a multiple of reanchor_every,
        # discard the chained value and refit against the phone depth.
        if (args.reanchor_every > 0
                and (k + 1 - first_valid) % args.reanchor_every == 0):
            a_re, b_re, n_re = _phone_reanchor(k + 1)
            a_arr[k+1] = a_re; b_arr[k+1] = b_re; n_arr[k+1] = n_re
            n_reanchored += 1
            if (k - first_valid) % 25 == 0 or k == len(bins) - 2:
                print(f"  reanchor ({k+1:4d}): a={a_re:6.3f} "
                      f"b={b_re:+6.3f}  n_pixels={n_re}")
            continue
        a_next, b_next, n_used = _solve_pair(
            M_all[k], M_all[k+1],
            float(a_arr[k]), float(b_arr[k]),
            P_all[k],  V_all[k],
            P_all[k+1], V_all[k+1],
            cw0, ch0, n_grid=args.n_grid,
            near=args.near, far=args.far,
        )
        if n_used < 30:
            # No usable correspondences — fall back to previous frame's affine.
            a_arr[k+1] = a_arr[k]; b_arr[k+1] = b_arr[k]
            n_arr[k+1] = 0
        else:
            a_arr[k+1] = a_next; b_arr[k+1] = b_next
            n_arr[k+1] = n_used
            n_chained += 1
        if (k - first_valid) % 25 == 0 or k == len(bins) - 2:
            print(f"  pair ({k:4d}, {k+1:4d}): a={a_arr[k+1]:6.3f} "
                  f"b={b_arr[k+1]:+6.3f}  n_used={int(n_arr[k+1])}")
    print(f"  chained {n_chained} pairs, reanchored {n_reanchored} frames")

    # ----- 4. Apply + write -----
    print(f"\nwriting refined frames to {out_dir}…")
    for i, fp in enumerate(bins):
        if M_all[i] is None: continue
        # Apply the chained affine, then resample through Bd⁻¹ at colour res
        # so the wire format matches what depth_refine.py emits.
        refined_color = (a_arr[i] * M_all[i] + b_arr[i]).astype(np.float32)
        refined_hi = _resample_model_to_depth_grid(
            refined_color, cw0, ch0, Bd_all[i], cw0, ch0,
        )
        refined_hi = np.where(np.isfinite(refined_hi), refined_hi, 0.0)
        refined_hi = np.clip(refined_hi, 0.0, args.far).astype(np.float32)
        new_body = _encode_refined_body(bodies[i], refined_hi)
        (out_dir / fp.name).write_bytes(new_body)

    a_used = a_arr[first_valid:]
    b_used = b_arr[first_valid:]
    print(f"\nDone. {len(bins) - first_valid} frames written to {out_dir}.")
    print(f"  a (chain): median {np.median(a_used):.3f}  "
          f"p10 {np.percentile(a_used, 10):.3f}  "
          f"p90 {np.percentile(a_used, 90):.3f}  "
          f"std {a_used.std():.3f}")
    print(f"  b (chain): median {np.median(b_used):+.3f}  "
          f"p10 {np.percentile(b_used, 10):+.3f}  "
          f"p90 {np.percentile(b_used, 90):+.3f}  "
          f"std {b_used.std():.3f}")


if __name__ == "__main__":
    main()
