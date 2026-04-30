#!/usr/bin/env python3
"""
Proper 3D-Gaussian-splatting training on CUDA via the `gsplat` library.

Successor to `gsplat_depth.py` (the pure-PyTorch / MPS POC). This script
uses the nerfstudio `gsplat` CUDA rasterizer end-to-end:

  - Anisotropic 3D Gaussians (per-axis scale + quaternion orientation).
  - Front-to-back alpha compositing.
  - Photometric L1 (+ optional SSIM) loss against the captured RGB.
  - `gsplat.DefaultStrategy` for densification / opacity reset / pruning.

Pipeline:

  1. Load `features_meta.json`, frame poses, RGB images, intrinsics.
  2. Initialize Gaussian centres from triangulated feature points,
     optionally densified with phone-depth backprojections at strided
     pixel grids on N keyframes.
  3. Train for `--iters` iterations against random frame batches; gsplat
     manages densification / pruning under the hood.
  4. Render depth at colour resolution for *every* frame in the
     session, save as `frame_NNNNNN.f16` in
     `captured_frames/<session>/<--out-cache>/` (default
     `model_raw_splat_cuda`) for the existing Rust voxelizer.

Usage:
    python tools/gsplat_train_cuda.py --session <id> [--iters 7000]
        [--frames-variant frames]
        [--gaussian-radius 0.03]
        [--densify-stride 4 --densify-keyframes 64]
        [--photo-edge 480]
        [--lr-pos 1.6e-4 --lr-scale 5e-3 --lr-quat 1e-3
         --lr-opac 5e-2 --lr-color 2.5e-3]
        [--ssim 0.2]
        [--out-cache model_raw_splat_cuda]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import fusion  # noqa: E402
import serve as _serve  # noqa: E402
from feature_pose_from_scratch import K_from_proj  # noqa: E402

parse_frame = _serve.parse_frame

_FRAME_BIN_RE = re.compile(r"frame_(\d+)\.bin")
# GL → OpenCV camera frame: x shared, y/z flipped.
_M_OC = np.diag([1.0, -1.0, -1.0, 1.0])


# ────────────────────────── frame loading ──────────────────────────

def load_session(frames_dir: Path):
    """Return per-frame dict including OpenCV viewmat + K. RGB is kept
    in GL row order (row 0 = scene bottom) — flipped at the loss/render
    boundaries since gsplat works in row 0 = top."""
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
            V_w2c_gl = np.linalg.inv(V_c2w)
        except np.linalg.LinAlgError:
            continue
        P_proj = fusion._mat4_from_column_major(f["projectionMatrix"])
        Bd = fusion._mat4_from_column_major(f["normDepthBufferFromNormView"])
        rgba = np.frombuffer(f["color"], dtype=np.uint8).reshape(ch, cw, 4)
        rgb = rgba[..., :3].copy()  # GL row order
        dw = int(f["width"]); dh = int(f["height"])
        depth = fusion.decode_depth(
            f["depth"], dw, dh,
            int(f["format"]), float(f["rawValueToMeters"]),
        )
        # OpenCV world->camera = M @ V_gl (flip y/z), and an OpenCV K
        # whose pixel coords match row 0 = top of the captured image.
        V_w2c_oc = _M_OC @ V_w2c_gl
        K = K_from_proj(P_proj, cw, ch)
        out[idx] = {
            "V_w2c_gl": V_w2c_gl.astype(np.float32),
            "V_w2c_oc": V_w2c_oc.astype(np.float32),
            "P_proj":   P_proj.astype(np.float32),
            "Bd":       Bd.astype(np.float32),
            "K":        K.astype(np.float32),
            "rgb": rgb, "cw": cw, "ch": ch,
            "depth": depth, "dw": dw, "dh": dh,
        }
    return out


# ────────────────────────── point init ──────────────────────────

def init_from_features(meta, frame_data):
    pos_list, rgb_list = [], []
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


def densify_from_phone(frame_data, n_keyframes, stride, near=0.05, far=8.0):
    if n_keyframes <= 0 or stride <= 0:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32))
    sorted_frames = sorted(frame_data.keys())
    step = max(1, len(sorted_frames) // n_keyframes)
    keyframes = sorted_frames[::step][:n_keyframes]
    pts, cols = [], []
    for fi in keyframes:
        d = frame_data[fi]
        depth = d["depth"]; Bd = d["Bd"]
        dw, dh = d["dw"], d["dh"]; cw, ch = d["cw"], d["ch"]
        rgb = d["rgb"]
        V_c2w = np.linalg.inv(d["V_w2c_gl"])
        P_proj = d["P_proj"]
        Bv = np.linalg.inv(Bd)
        for by in range(0, dh, stride):
            for bx in range(0, dw, stride):
                z = depth[by, bx]
                if not np.isfinite(z) or z <= near or z >= far:
                    continue
                u_d = 1.0 - (bx + 0.5) / dw
                v_d = (by + 0.5) / dh
                nv = Bv @ np.array([u_d, v_d, 0.0, 1.0])
                if abs(nv[3]) < 1e-12:
                    continue
                u_v, v_v = nv[0] / nv[3], nv[1] / nv[3]
                if not (0 < u_v < 1 and 0 < v_v < 1):
                    continue
                cx = int(round(u_v * cw))
                cy = int(round(v_v * ch))
                if not (0 <= cx < cw and 0 <= cy < ch):
                    continue
                colour = rgb[cy, cx].astype(np.float32) / 255.0
                ndc_x, ndc_y = 2.0 * u_v - 1.0, 2.0 * v_v - 1.0
                Z = -float(z)
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


# ────────────────────────── losses ──────────────────────────

def ssim(x, y, win=11, sigma=1.5):
    """Single-scale SSIM averaged over channels and spatial dims.
    x, y: (B, 3, H, W)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    coords = torch.arange(win, device=x.device, dtype=x.dtype) - (win - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None, :]
    g = g.expand(3, 1, win, win)
    pad = win // 2
    mu_x = F.conv2d(x, g, groups=3, padding=pad)
    mu_y = F.conv2d(y, g, groups=3, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sx = F.conv2d(x * x, g, groups=3, padding=pad) - mu_x2
    sy = F.conv2d(y * y, g, groups=3, padding=pad) - mu_y2
    sxy = F.conv2d(x * y, g, groups=3, padding=pad) - mu_xy
    val = ((2 * mu_xy + C1) * (2 * sxy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sx + sy + C2)
    )
    return val.mean()


# ────────────────────────── driver ──────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--session", required=True)
    ap.add_argument("--frames-variant", default="frames")
    ap.add_argument("--gaussian-radius", type=float, default=0.03)
    ap.add_argument("--densify-stride", type=int, default=4)
    ap.add_argument("--densify-keyframes", type=int, default=64)
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--photo-edge", type=int, default=480,
                    help="Long-edge of the (down-sampled) training "
                         "image; full resolution is too memory-heavy.")
    ap.add_argument("--lr-pos",   type=float, default=1.6e-4)
    ap.add_argument("--lr-scale", type=float, default=5e-3)
    ap.add_argument("--lr-quat",  type=float, default=1e-3)
    ap.add_argument("--lr-opac",  type=float, default=5e-2)
    ap.add_argument("--lr-color", type=float, default=2.5e-3)
    ap.add_argument("--ssim",     type=float, default=0.2,
                    help="Weight of (1-SSIM) term; 0 disables.")
    ap.add_argument("--densify-from-iter", type=int, default=500)
    ap.add_argument("--densify-until-iter", type=int, default=15_000)
    ap.add_argument("--reset-every", type=int, default=3000)
    ap.add_argument("--out-cache",  default="model_raw_splat_cuda")
    ap.add_argument("--device",     default="cuda")
    ap.add_argument("--seed",       type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

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

    print(f"[load] {frames_dir}")
    t0 = time.time()
    frame_data = load_session(frames_dir)
    print(f"  {len(frame_data)} frames in {time.time()-t0:.1f}s")
    if not frame_data:
        sys.exit("no usable frames.")
    print(f"[load] features_meta.json")
    meta = json.loads(meta_path.read_text())
    feat_pos, feat_rgb = init_from_features(meta, frame_data)
    print(f"  {len(feat_pos)} feature points")

    if args.densify_stride > 0 and args.densify_keyframes > 0:
        print(f"[densify-init] stride={args.densify_stride}px on "
              f"{args.densify_keyframes} keyframes")
        d_pos, d_rgb = densify_from_phone(
            frame_data, args.densify_keyframes, args.densify_stride
        )
        print(f"  {len(d_pos)} densified points")
        pos_np = np.concatenate([feat_pos, d_pos], 0) if len(d_pos) else feat_pos
        rgb_np = np.concatenate([feat_rgb, d_rgb], 0) if len(d_rgb) else feat_rgb
    else:
        pos_np, rgb_np = feat_pos, feat_rgb

    N = len(pos_np)
    print(f"[init] {N} Gaussians")

    # ── parameters (gsplat conventions) ──
    means = torch.nn.Parameter(torch.tensor(pos_np, device=device))
    # log scales (3 per Gaussian) — start isotropic at gaussian_radius.
    init_log_scale = math.log(args.gaussian_radius)
    log_scales = torch.nn.Parameter(
        torch.full((N, 3), init_log_scale, device=device)
    )
    # quaternions (wxyz), identity init.
    quats = torch.nn.Parameter(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]] * N, device=device)
    )
    # opacities stored as logit; gsplat takes sigmoid-applied [0,1].
    init_opac = 0.5
    logit_opac = torch.nn.Parameter(
        torch.full((N,), math.log(init_opac / (1 - init_opac)), device=device)
    )
    # Colors stored as logit of [0,1] RGB.
    rgb_clip = np.clip(rgb_np, 1e-3, 1 - 1e-3)
    color_logit = torch.nn.Parameter(
        torch.tensor(np.log(rgb_clip / (1 - rgb_clip)),
                     dtype=torch.float32, device=device)
    )

    # gsplat's DefaultStrategy expects `params` dict so it can clone /
    # split / prune them in place. We register the 5 tensors as a
    # ParameterDict so the strategy can update them.
    splats = torch.nn.ParameterDict({
        "means":      means,
        "scales":     log_scales,   # strategy needs raw log-scales
        "quats":      quats,
        "opacities":  logit_opac,
        "colors":     color_logit,
    }).to(device)

    optimizers = {
        "means":     torch.optim.Adam([splats["means"]],     lr=args.lr_pos),
        "scales":    torch.optim.Adam([splats["scales"]],    lr=args.lr_scale),
        "quats":     torch.optim.Adam([splats["quats"]],     lr=args.lr_quat),
        "opacities": torch.optim.Adam([splats["opacities"]], lr=args.lr_opac),
        "colors":    torch.optim.Adam([splats["colors"]],    lr=args.lr_color),
    }

    # DefaultStrategy: clone large gradients, split too-large, prune low-opacity.
    strategy = DefaultStrategy(
        prune_opa=0.005,
        grow_grad2d=0.0002,
        grow_scale3d=0.01,  # m
        prune_scale3d=0.1,  # m
        refine_start_iter=args.densify_from_iter,
        refine_stop_iter=args.densify_until_iter,
        reset_every=args.reset_every,
        refine_every=100,
        absgrad=True,
        verbose=False,
    )
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=1.0)

    # ── precompute per-frame training-resolution RGB targets ──
    sorted_frames = sorted(frame_data.keys())
    train_imgs: dict[int, dict] = {}
    for fi in sorted_frames:
        d = frame_data[fi]
        cw, ch = d["cw"], d["ch"]
        if cw >= ch:
            W_t = args.photo_edge
            H_t = max(1, int(round(args.photo_edge * ch / cw)))
        else:
            H_t = args.photo_edge
            W_t = max(1, int(round(args.photo_edge * cw / ch)))
        # GT row 0 = top (flip from GL row order)
        rgb_t = torch.tensor(d["rgb"][::-1].copy(), dtype=torch.float32,
                              device=device).permute(2, 0, 1) / 255.0
        rgb_t = F.interpolate(rgb_t[None], size=(H_t, W_t),
                               mode="bilinear", align_corners=False)[0]
        # Scale K to the training resolution.
        K = d["K"].copy()
        K[0, 0] *= W_t / cw; K[0, 2] *= W_t / cw
        K[1, 1] *= H_t / ch; K[1, 2] *= H_t / ch
        train_imgs[fi] = {
            "rgb": rgb_t,                                    # (3, H_t, W_t)
            "viewmat": torch.tensor(d["V_w2c_oc"], device=device),
            "K": torch.tensor(K, device=device),
            "H": H_t, "W": W_t,
        }

    # ── training loop ──
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    log_interval = max(50, args.iters // 100)
    for it in range(1, args.iters + 1):
        fi = int(rng.choice(sorted_frames))
        gt = train_imgs[fi]
        opacities_act = torch.sigmoid(splats["opacities"])
        scales_act = torch.exp(splats["scales"])
        colors_act = torch.sigmoid(splats["colors"])
        rgb_pred, alpha, info = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=scales_act,
            opacities=opacities_act,
            colors=colors_act,
            viewmats=gt["viewmat"][None],   # (1, 4, 4)
            Ks=gt["K"][None],               # (1, 3, 3)
            width=gt["W"], height=gt["H"],
            render_mode="RGB",
            packed=False,
            absgrad=True,
        )
        # rgb_pred: (1, H, W, 3) → (1, 3, H, W) for SSIM convs.
        rgb_pred_chw = rgb_pred.permute(0, 3, 1, 2)
        gt_chw = gt["rgb"][None]
        l1 = (rgb_pred_chw - gt_chw).abs().mean()
        if args.ssim > 0:
            ssim_val = ssim(rgb_pred_chw.clamp(0, 1), gt_chw)
            loss = (1.0 - args.ssim) * l1 + args.ssim * (1.0 - ssim_val)
        else:
            loss = l1

        for o in optimizers.values():
            o.zero_grad(set_to_none=True)
        strategy.step_pre_backward(splats, optimizers, strategy_state, it, info)
        loss.backward()
        for o in optimizers.values():
            o.step()
        strategy.step_post_backward(splats, optimizers, strategy_state, it, info)

        if it % log_interval == 0 or it == 1 or it == args.iters:
            n_now = splats["means"].shape[0]
            elapsed = time.time() - t0
            eta = elapsed / it * (args.iters - it)
            print(f"  iter {it:5d}/{args.iters}  N={n_now:6d}  "
                  f"loss={loss.item():.4f}  l1={l1.item():.4f}  "
                  f"elapsed {elapsed:5.1f}s  ETA {eta:5.1f}s")

    print(f"[train] done — {splats['means'].shape[0]} Gaussians, "
          f"{time.time()-t0:.1f}s")

    # ── render depth at full color resolution for every frame ──
    out_dir = sess_dir / args.out_cache
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[render] {len(frame_data)} depth maps → {out_dir}")
    index: dict[str, dict] = {}
    t0 = time.time()
    means_e = splats["means"].detach()
    quats_e = splats["quats"].detach()
    scales_e = torch.exp(splats["scales"].detach())
    opac_e = torch.sigmoid(splats["opacities"].detach())
    colors_e = torch.sigmoid(splats["colors"].detach())
    for i, fi in enumerate(sorted_frames):
        d = frame_data[fi]
        cw, ch = d["cw"], d["ch"]
        viewmat = torch.tensor(d["V_w2c_oc"], device=device)[None]
        Kt = torch.tensor(d["K"], device=device)[None]
        with torch.no_grad():
            depth_out, _, _ = rasterization(
                means=means_e, quats=quats_e, scales=scales_e,
                opacities=opac_e, colors=colors_e,
                viewmats=viewmat, Ks=Kt,
                width=cw, height=ch,
                render_mode="ED",  # expected (weight-normalized) depth
                packed=False,
            )
        # depth_out: (1, H, W, 1) — expected depth in OpenCV camera Z (positive).
        depth_np = depth_out[0, :, :, 0].cpu().numpy()
        # gsplat renders row 0 = top. Cache convention is GL row order
        # (row 0 = scene bottom): flip vertically.
        depth_np = np.flipud(depth_np).astype(np.float16)
        (out_dir / f"frame_{fi:06d}.f16").write_bytes(depth_np.tobytes())
        index[str(fi)] = {"w": cw, "h": ch}
        if (i + 1) % 20 == 0 or i + 1 == len(frame_data):
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(frame_data) - i - 1)
            print(f"  [{i+1:3d}/{len(frame_data)}] frame_{fi:06d}  "
                  f"{cw}x{ch}  elapsed {elapsed:5.1f}s  ETA {eta:5.1f}s")

    # ── export trained Gaussians as a .splat blob for the browser viewer ──
    # antimatter15/splat format: 32 bytes per Gaussian, sorted by descending
    # scale-product so the streaming viewer renders the largest first.
    #   position  : 3×f32   (12 B)
    #   scale     : 3×f32   (12 B)  in metres
    #   color     : 4×u8    (4 B)   RGBA, alpha = opacity * 255
    #   rotation  : 4×u8    (4 B)   wxyz quat encoded as (q+1)*128 → uint8
    means_w = means_e.cpu().numpy()                # (N, 3) world (GL frame)
    quats_w = quats_e.cpu().numpy()                # (N, 4) wxyz
    scales_w = scales_e.cpu().numpy()              # (N, 3) metres
    opac_w  = opac_e.cpu().numpy()                  # (N,)
    cols_w  = colors_e.cpu().numpy()                # (N, 3) [0,1]
    qnorm = quats_w / (np.linalg.norm(quats_w, axis=1, keepdims=True) + 1e-12)
    sort_key = -(scales_w[:, 0] * scales_w[:, 1] * scales_w[:, 2])
    order = np.argsort(sort_key)
    N_out = len(order)
    buf = bytearray(32 * N_out)
    for j, i in enumerate(order):
        off = 32 * j
        buf[off:off+12]  = means_w[i].astype(np.float32).tobytes()
        buf[off+12:off+24] = scales_w[i].astype(np.float32).tobytes()
        # colour + opacity in RGBA u8.
        rgba = (np.clip(np.concatenate([cols_w[i], [opac_w[i]]]), 0, 1) * 255
                ).round().astype(np.uint8)
        buf[off+24:off+28] = rgba.tobytes()
        # quat in u8 (antimatter15 convention): q ∈ [-1, 1] → round((q+1)*128).
        # Decoder reads back as (v - 128) / 128. Round (not truncate) so q=0
        # decodes back to 0 cleanly; clip to [0, 255] so q=1 doesn't overflow.
        q_u8 = np.clip(np.round((qnorm[i] + 1.0) * 128.0), 0, 255).astype(np.uint8)
        buf[off+28:off+32] = q_u8.tobytes()
    (out_dir / "splat.bin").write_bytes(bytes(buf))
    print(f"[export] splat.bin: {N_out} Gaussians, {len(buf)/1e6:.1f} MB")

    (out_dir / "index.json").write_text(json.dumps(index, indent=0))
    (out_dir / "info.json").write_text(json.dumps({
        "method": "gsplat-cuda-train",
        "n_gaussians_final": int(splats["means"].shape[0]),
        "iters": args.iters,
        "photo_edge": args.photo_edge,
        "ssim_weight": args.ssim,
        "lr": {"pos": args.lr_pos, "scale": args.lr_scale,
               "quat": args.lr_quat, "opac": args.lr_opac,
               "color": args.lr_color},
        "init_radius": args.gaussian_radius,
        "densify_stride": args.densify_stride,
        "densify_keyframes": args.densify_keyframes,
        "densify_from_iter": args.densify_from_iter,
        "densify_until_iter": args.densify_until_iter,
    }, indent=2))
    print(f"\nDone. Voxelize with:\n"
          f"  ./tools/voxel_reverse_rust/target/release/voxel-reverse \\\n"
          f"      --session {args.session} \\\n"
          f"      --voxel-size 0.02 --tol 0.04 \\\n"
          f"      --depth-source model --model-version splat \\\n"
          f"      --frames-dir captured_frames/{args.session}/{args.frames_variant} \\\n"
          f"      --model-cache-dir captured_frames/{args.session}/{args.out_cache}\n")


if __name__ == "__main__":
    main()
