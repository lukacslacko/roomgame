#!/usr/bin/env python3
"""
3D-Gaussian-splatting depth POC.

Render per-frame depth maps from a single multi-view-consistent set of
3D Gaussians, then save them as a `model_raw`-style cache the existing
Rust voxelizer can consume (`--depth-source model --model-version splat`).

Pipeline:

  1. Initialize Gaussian centres from `features_meta.json` (the multi-
     view-triangulated ORB features). Optionally densify with phone-
     depth backprojections at a strided pixel grid from N keyframes —
     this gives the splat enough coverage to produce full-image depth
     instead of a sparse feature-only sprinkle.

  2. Each Gaussian is isotropic with metric radius `--gaussian-radius`
     (default 5 cm); colour is initialised from the median observed
     pixel for feature points, and from the source pixel for densified
     points; opacity defaults to ~0.7.

  3. Optional photometric optimization (`--photo-iters > 0`): forward-
     render every Nth keyframe at a downsampled resolution, take L1
     loss against the captured RGB, backprop into log-scale + opacity
     + colour. Positions stay fixed (multi-view-triangulated points
     are typically more accurate than the photometric loss can pull
     them via gradient — and fixing positions makes the renderer
     simpler and 5–10× faster).

  4. Render depth at colour-image resolution for *every* frame in the
     session. Per-pixel depth is the weight-normalised average of
     covering Gaussians (weight = opacity × Gaussian footprint at the
     pixel). Pixels with no covering Gaussian end up at depth 0,
     which the Rust voxelizer treats as "no depth" — they don't
     produce voxels.

  5. Save each frame's rendered depth as `frame_NNNNNN.f16` in
     `captured_frames/<session>/model_raw_splat/`, plus an
     `index.json` mapping idx → (cw, ch) and an `info.json` summary.
     This matches the existing model_raw cache format exactly.

This is *not* full 3DGS — there's no anisotropic covariance, no front-
to-back alpha compositing, no densification or pruning during training.
But it's the smallest thing that uses a single multi-view-consistent 3D
representation to render depth from every pose, so we can measure the
voxelization quality of "joint optimization" vs the per-frame depth
approaches we've been chaining together so far.

Usage:
    python tools/gsplat_depth.py --session <id>
        [--frames-variant frames]
        [--gaussian-radius 0.05]
        [--densify-stride 32]              # phone-depth pixel stride; 0 = no densify
        [--densify-keyframes 32]
        [--photo-iters 0]                  # 0 = skip photometric optimization
        [--photo-keyframes 16]
        [--photo-edge 200]
        [--device mps|cpu|cuda]
        [--out-cache model_raw_splat]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import fusion  # noqa: E402
import serve as _serve  # noqa: E402

parse_frame = _serve.parse_frame

_FRAME_BIN_RE = re.compile(r"frame_(\d+)\.bin")


# ────────────────────────── frame loading ──────────────────────────

def load_session(frames_dir: Path) -> dict[int, dict]:
    """idx → {V_w2c, P_proj, Bd, cw, ch, rgb (H,W,3), depth (dh,dw),
    dw, dh}. RGB and depth are kept on CPU as numpy; we only move
    them to the device when needed."""
    out: dict[int, dict] = {}
    for p in sorted(frames_dir.glob("frame_*.bin")):
        m = _FRAME_BIN_RE.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        try:
            f = parse_frame(p.read_bytes())
        except Exception:
            continue
        cw = int(f["color_width"]); ch = int(f["color_height"])
        if cw == 0 or ch == 0 or f["color"] is None:
            continue
        V_c2w = fusion._mat4_from_column_major(f["viewMatrix"])
        try:
            V_w2c = np.linalg.inv(V_c2w)
        except np.linalg.LinAlgError:
            continue
        P_proj = fusion._mat4_from_column_major(f["projectionMatrix"])
        Bd = fusion._mat4_from_column_major(f["normDepthBufferFromNormView"])
        rgba = np.frombuffer(f["color"], dtype=np.uint8).reshape(ch, cw, 4)
        rgb = rgba[..., :3].copy()  # row 0 = scene-bottom (GL row order)
        dw = int(f["width"]); dh = int(f["height"])
        depth = fusion.decode_depth(
            f["depth"], dw, dh,
            int(f["format"]), float(f["rawValueToMeters"]),
        )
        out[idx] = {
            "V_w2c": V_w2c.astype(np.float32),
            "P_proj": P_proj.astype(np.float32),
            "Bd": Bd.astype(np.float32),
            "rgb": rgb, "cw": cw, "ch": ch,
            "depth": depth, "dw": dw, "dh": dh,
        }
    return out


# ────────────────────────── point cloud init ──────────────────────────

def init_from_features(meta: dict, frame_data: dict[int, dict]
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature world position + median observed pixel colour."""
    pos_list: list[np.ndarray] = []
    rgb_list: list[np.ndarray] = []
    for v in meta.get("voxels", []):
        for f in v.get("features", []):
            obs = f.get("obs", [])
            colors = []
            for ob in obs:
                fi = int(ob["frame"])
                if fi not in frame_data:
                    continue
                fd = frame_data[fi]
                cw, ch = fd["cw"], fd["ch"]
                # Norm-view (u, v): u=0 left, v=1 top. The .bin's RGB
                # buffer is GL row order (row 0 = scene bottom = v=0).
                px = int(round(float(ob["u"]) * cw))
                py = int(round(float(ob["v"]) * ch))  # row in GL-ordered rgb
                if 0 <= px < cw and 0 <= py < ch:
                    colors.append(fd["rgb"][py, px])
            if not colors:
                continue
            pos_list.append(np.asarray(f["world"], dtype=np.float32))
            rgb_list.append(np.median(np.asarray(colors), axis=0) / 255.0)
    pos = np.asarray(pos_list, dtype=np.float32)
    rgb = np.asarray(rgb_list, dtype=np.float32).clip(0.0, 1.0)
    return pos, rgb


def densify_from_phone(frame_data: dict[int, dict], n_keyframes: int,
                        stride: int, near: float, far: float
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Backproject every `stride`-th depth-buffer pixel of `n_keyframes`
    evenly-spaced frames into world coords. Colour comes from sampling
    the colour grid at the same norm-view (u, v). This adds dense
    coverage so the rendered depth is full-image, not just feature-
    sprinkle."""
    if n_keyframes <= 0 or stride <= 0:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32))
    sorted_frames = sorted(frame_data.keys())
    step = max(1, len(sorted_frames) // n_keyframes)
    keyframes = sorted_frames[::step][:n_keyframes]
    pts: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for fi in keyframes:
        d = frame_data[fi]
        depth = d["depth"]; Bd = d["Bd"]
        dw = d["dw"]; dh = d["dh"]
        cw = d["cw"]; ch = d["ch"]
        rgb = d["rgb"]
        V_c2w = np.linalg.inv(d["V_w2c"])
        P_proj = d["P_proj"]
        # Iterate strided depth-buffer pixels, sample depth, get norm-view
        # via Bd⁻¹, back-project to camera space, then world space.
        Bv = np.linalg.inv(Bd)
        for by in range(0, dh, stride):
            for bx in range(0, dw, stride):
                z = depth[by, bx]
                if not np.isfinite(z) or z <= near or z >= far:
                    continue
                # depth-buffer pixel → norm-depth-buffer (u_d, v_d).
                # Phone projection has a horizontal flip:
                #   bx = (1 - u_d) * dw  ⇒  u_d = 1 - bx/dw
                # by = v_d * dh
                u_d = 1.0 - (bx + 0.5) / dw
                v_d = (by + 0.5) / dh
                # norm-depth-buffer → norm-view via Bd⁻¹
                nd = np.array([u_d, v_d, 0.0, 1.0])
                nv = Bv @ nd
                if abs(nv[3]) < 1e-12:
                    continue
                u_v = nv[0] / nv[3]
                v_v = nv[1] / nv[3]
                if not (0 < u_v < 1 and 0 < v_v < 1):
                    continue
                # Colour at the norm-view UV on the GL-row-order RGB.
                cx = int(round(u_v * cw))
                cy = int(round(v_v * ch))
                if not (0 <= cx < cw and 0 <= cy < ch):
                    continue
                colour = rgb[cy, cx].astype(np.float32) / 255.0
                # Back-project (u_v, v_v, z) to camera space then world.
                ndc_x = 2.0 * u_v - 1.0
                ndc_y = 2.0 * v_v - 1.0
                Z = -float(z)  # GL: in front means z<0
                # ndc_x = -P[0,0] X/Z - P[0,2]  ⇒  X = -(ndc_x + P[0,2]) Z / P[0,0]
                X = -(ndc_x + P_proj[0, 2]) * Z / P_proj[0, 0]
                Y = -(ndc_y + P_proj[1, 2]) * Z / P_proj[1, 1]
                cam = np.array([X, Y, Z, 1.0], dtype=np.float32)
                world = V_c2w @ cam
                pts.append(world[:3].astype(np.float32))
                cols.append(colour)
    if not pts:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32))
    return np.asarray(pts, dtype=np.float32), np.asarray(cols, dtype=np.float32)


# ────────────────────────── differentiable render ──────────────────────────

def render_frame(pos: torch.Tensor, log_scale: torch.Tensor,
                  opacity: torch.Tensor, color: torch.Tensor,
                  V_w2c: torch.Tensor, P_proj: torch.Tensor,
                  H: int, W: int, *,
                  return_rgb: bool = False,
                  chunk: int = 128) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Forward-render an H×W image from N isotropic 3D Gaussians.

    Returns (depth_img, rgb_img) where:
      depth_img: (H, W) metric depth (positive metres). 0 where no
                 Gaussian covers the pixel.
      rgb_img:   (H, W, 3) in [0, 1] if return_rgb else None.

    `chunk` controls how many Gaussians are processed per inner kernel
    — keeps the (H, W, K) workspace under control on MPS / 16 GB."""
    device = pos.device
    N = pos.shape[0]

    # 1. World → camera-space (GL convention: -z is forward).
    pos_h = torch.cat([pos, torch.ones(N, 1, device=device)], dim=-1)
    pos_cam = pos_h @ V_w2c.T  # (N, 4)
    z = pos_cam[:, 2]
    in_frust = (z < -0.05) & (z > -10.0)
    if not in_frust.any():
        depth_img = torch.zeros(H, W, device=device)
        rgb_img = torch.zeros(H, W, 3, device=device) if return_rgb else None
        return depth_img, rgb_img

    pos_cam = pos_cam[in_frust]
    log_scale_m = log_scale[in_frust]
    opacity_m = opacity[in_frust]
    color_m = color[in_frust] if return_rgb else None
    z_m = pos_cam[:, 2]                    # (N,) negative

    # 2. Project to NDC then to OpenCV-style pixel coords (col, row).
    pos_clip = pos_cam @ P_proj.T
    safe_w = torch.where(pos_clip[:, 3].abs() > 1e-9,
                          pos_clip[:, 3], torch.ones_like(pos_clip[:, 3]))
    ndc_x = pos_clip[:, 0] / safe_w
    ndc_y = pos_clip[:, 1] / safe_w
    px = (ndc_x + 1.0) * 0.5 * W
    py = (1.0 - ndc_y) * 0.5 * H

    # 3. Pixel-space radius from world-space scale (focal/depth pinhole).
    fx = abs(P_proj[0, 0].item()) * W * 0.5
    sigma_px = torch.exp(log_scale_m) * fx / (-z_m).clamp(min=0.05)
    sigma_px = sigma_px.clamp(min=0.5, max=30.0)

    # 4. Reject anything outside (image + 3σ margin).
    margin = 3.0 * sigma_px
    in_view = ((px >= -margin) & (px < W + margin)
                & (py >= -margin) & (py < H + margin))
    px = px[in_view]; py = py[in_view]; z_m = z_m[in_view]
    sigma_px = sigma_px[in_view]; opacity_m = opacity_m[in_view]
    if return_rgb:
        color_m = color_m[in_view]

    Nm = px.shape[0]
    if Nm == 0:
        depth_img = torch.zeros(H, W, device=device)
        rgb_img = torch.zeros(H, W, 3, device=device) if return_rgb else None
        return depth_img, rgb_img

    # 5. Tile-free vectorised splat. We process Gaussians in chunks of
    # `chunk` to keep the (H, W, K) workspace small.
    yy = torch.arange(H, device=device, dtype=torch.float32)[:, None, None]
    xx = torch.arange(W, device=device, dtype=torch.float32)[None, :, None]

    weight_acc = torch.zeros(H, W, device=device)
    depth_num  = torch.zeros(H, W, device=device)
    rgb_num    = torch.zeros(H, W, 3, device=device) if return_rgb else None
    metric_z = -z_m  # (N,) positive metres

    for s in range(0, Nm, chunk):
        e = min(s + chunk, Nm)
        # (1, 1, K) broadcasts
        px_c  = px[s:e][None, None, :]
        py_c  = py[s:e][None, None, :]
        sig_c = sigma_px[s:e][None, None, :]
        op_c  = opacity_m[s:e][None, None, :]
        z_c   = metric_z[s:e][None, None, :]
        # (H, W, K) Gaussian footprint
        d2 = (xx - px_c) ** 2 + (yy - py_c) ** 2
        w = torch.exp(-d2 / (2.0 * sig_c ** 2)) * op_c
        weight_acc = weight_acc + w.sum(dim=-1)
        depth_num  = depth_num  + (w * z_c).sum(dim=-1)
        if return_rgb:
            col_c = color_m[s:e][None, None, :, :]   # (1, 1, K, 3)
            rgb_num = rgb_num + (w[..., None] * col_c).sum(dim=-2)

    safe_w_acc = weight_acc.clamp(min=1e-6)
    depth_img = depth_num / safe_w_acc
    # Pixels with negligible coverage → set depth to 0 (cleaner than NaN
    # for the f16 cache; the Rust voxelizer drops 0/<=0 depths).
    depth_img = torch.where(weight_acc > 1e-3, depth_img, torch.zeros_like(depth_img))
    rgb_img = (rgb_num / safe_w_acc[..., None]) if return_rgb else None
    return depth_img, rgb_img


# ────────────────────────── driver ──────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--session", required=True)
    ap.add_argument("--frames-variant", default="frames")
    ap.add_argument("--gaussian-radius", type=float, default=0.05,
                    help="Initial isotropic radius of each Gaussian (m).")
    ap.add_argument("--densify-stride", type=int, default=4,
                    help="Phone-depth pixel stride for densification "
                         "(0 = features only).")
    ap.add_argument("--densify-keyframes", type=int, default=64,
                    help="Number of evenly-spaced keyframes to take phone "
                         "depth from for densification.")
    ap.add_argument("--photo-iters", type=int, default=0,
                    help="Photometric optimization iterations (0 = skip).")
    ap.add_argument("--photo-keyframes", type=int, default=16)
    ap.add_argument("--photo-edge", type=int, default=200)
    ap.add_argument("--photo-lr", type=float, default=2e-2)
    ap.add_argument("--render-chunk", type=int, default=64,
                    help="Gaussians per kernel chunk; lower if MPS OOMs.")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out-cache", default="model_raw_splat")
    args = ap.parse_args()

    sess_dir = ROOT / "captured_frames" / args.session
    if not sess_dir.is_dir():
        sys.exit(f"session not found: {sess_dir}")

    frames_dir = sess_dir / args.frames_variant
    if not frames_dir.is_dir():
        sys.exit(f"frames variant not found: {frames_dir}")

    meta_path = sess_dir / "features_meta.json"
    if not meta_path.exists():
        sys.exit(f"features_meta.json missing — run "
                 f"tools/feature_ray_reconstruct.py --session {args.session}")

    print(f"[load] frames from {frames_dir}")
    t0 = time.time()
    frame_data = load_session(frames_dir)
    print(f"  {len(frame_data)} frames in {time.time()-t0:.1f}s")
    if not frame_data:
        sys.exit("no usable frames.")

    print(f"[load] {meta_path.name}")
    meta = json.loads(meta_path.read_text())
    feat_pos, feat_rgb = init_from_features(meta, frame_data)
    print(f"  {len(feat_pos)} feature points")

    if args.densify_stride > 0 and args.densify_keyframes > 0:
        print(f"[densify] stride={args.densify_stride}px on "
              f"{args.densify_keyframes} keyframes")
        t0 = time.time()
        d_pos, d_rgb = densify_from_phone(
            frame_data, args.densify_keyframes, args.densify_stride,
            near=0.05, far=8.0,
        )
        print(f"  {len(d_pos)} densified points in {time.time()-t0:.1f}s")
        pos_np = np.concatenate([feat_pos, d_pos], axis=0) if len(d_pos) else feat_pos
        rgb_np = np.concatenate([feat_rgb, d_rgb], axis=0) if len(d_rgb) else feat_rgb
    else:
        pos_np = feat_pos; rgb_np = feat_rgb

    print(f"[splat] {len(pos_np)} Gaussians total")

    device = torch.device(args.device)
    pos = torch.tensor(pos_np, device=device)
    log_scale = torch.full((len(pos),), float(np.log(args.gaussian_radius)),
                            device=device)
    # opacity stored as logit (so sigmoid keeps it in [0, 1] under Adam).
    init_opac = 0.7
    logit_opacity = torch.full((len(pos),),
                                 float(np.log(init_opac / (1 - init_opac))),
                                 device=device)
    # colour stored as logit too.
    rgb_clip = np.clip(rgb_np, 1e-3, 1 - 1e-3)
    color_logit = torch.tensor(np.log(rgb_clip / (1 - rgb_clip)), device=device,
                                 dtype=torch.float32)

    # ── photometric optimization ──
    if args.photo_iters > 0:
        log_scale.requires_grad_()
        logit_opacity.requires_grad_()
        color_logit.requires_grad_()
        optim = torch.optim.Adam(
            [log_scale, logit_opacity, color_logit], lr=args.photo_lr,
        )
        sorted_frames = sorted(frame_data.keys())
        step = max(1, len(sorted_frames) // args.photo_keyframes)
        keyframes = sorted_frames[::step][:args.photo_keyframes]
        print(f"[photo] {args.photo_iters} iters · {len(keyframes)} keyframes "
              f"· edge={args.photo_edge}px · lr={args.photo_lr}")
        t0 = time.time()
        for it in range(args.photo_iters):
            kf = keyframes[it % len(keyframes)]
            d = frame_data[kf]
            cw = d["cw"]; ch = d["ch"]
            ar = cw / ch
            if ar > 1:
                W_t, H_t = args.photo_edge, max(1, int(args.photo_edge / ar))
            else:
                H_t, W_t = args.photo_edge, max(1, int(args.photo_edge * ar))
            # GT: row 0 = scene bottom in the .bin → flip so renderer
            # comparison uses row 0 = top.
            rgb_gt = torch.tensor(d["rgb"][::-1].copy(), dtype=torch.float32,
                                    device=device).permute(2, 0, 1) / 255.0
            rgb_gt = torch.nn.functional.interpolate(
                rgb_gt[None], size=(H_t, W_t),
                mode="bilinear", align_corners=False,
            )[0].permute(1, 2, 0)

            V_w2c = torch.tensor(d["V_w2c"], device=device)
            P_proj = torch.tensor(d["P_proj"], device=device)
            opac = torch.sigmoid(logit_opacity)
            color = torch.sigmoid(color_logit)

            _, rgb_pred = render_frame(
                pos, log_scale, opac, color,
                V_w2c, P_proj, H_t, W_t,
                return_rgb=True, chunk=args.render_chunk,
            )

            loss = (rgb_pred - rgb_gt).abs().mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

            if it % 10 == 0 or it == args.photo_iters - 1:
                print(f"  iter {it:4d}  loss={loss.item():.4f}  "
                      f"({(time.time()-t0):.1f}s)")
        log_scale = log_scale.detach()
        opacity = torch.sigmoid(logit_opacity).detach()
        color = torch.sigmoid(color_logit).detach()
    else:
        opacity = torch.sigmoid(logit_opacity)
        color = torch.sigmoid(color_logit)

    # ── render all frames at colour resolution ──
    out_dir = sess_dir / args.out_cache
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[render] depth → {out_dir}")
    index: dict[str, dict] = {}
    t0 = time.time()
    for i, fi in enumerate(sorted(frame_data.keys())):
        d = frame_data[fi]
        cw = d["cw"]; ch = d["ch"]
        V_w2c = torch.tensor(d["V_w2c"], device=device)
        P_proj = torch.tensor(d["P_proj"], device=device)
        with torch.no_grad():
            depth, _ = render_frame(
                pos, log_scale, opacity, color,
                V_w2c, P_proj, ch, cw,
                return_rgb=False, chunk=args.render_chunk,
            )
        # The render produces row 0 = top of screen. The model_raw cache
        # convention is GL row order (row 0 = scene bottom), matching
        # what `compute_blend_metres` and `parse_frame_with_model_only`
        # in the Rust voxelizer expect. Flip vertically.
        depth_np = torch.flip(depth, dims=[0]).cpu().numpy().astype(np.float16)
        (out_dir / f"frame_{fi:06d}.f16").write_bytes(depth_np.tobytes())
        index[str(fi)] = {"w": cw, "h": ch}
        if (i + 1) % 20 == 0 or i + 1 == len(frame_data):
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(frame_data) - i - 1)
            print(f"  [{i+1:3d}/{len(frame_data)}] frame_{fi:06d}  "
                  f"{cw}x{ch}  elapsed {elapsed:5.1f}s  ETA {eta:5.1f}s")

    (out_dir / "index.json").write_text(json.dumps(index, indent=0))
    (out_dir / "info.json").write_text(json.dumps({
        "method": "feature-splat-depth-poc",
        "n_gaussians": int(len(pos_np)),
        "gaussian_radius_m": args.gaussian_radius,
        "densify_stride": args.densify_stride,
        "densify_keyframes": args.densify_keyframes,
        "photo_iters": args.photo_iters,
        "photo_keyframes": args.photo_keyframes,
        "photo_edge": args.photo_edge,
        "photo_lr": args.photo_lr,
    }, indent=2))
    print(f"\nDone. Voxelize with:\n"
          f"  ./tools/voxel_reverse_rust/target/release/voxel-reverse \\\n"
          f"      --session {args.session} \\\n"
          f"      --voxel-size 0.02 --tol 0.04 \\\n"
          f"      --depth-source model --model-version splat\n"
          f"\n(Or with --frames-dir <pose-variant> + --out for the from-scratch / BA poses.)")


if __name__ == "__main__":
    main()
