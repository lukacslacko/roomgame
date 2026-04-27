#!/usr/bin/env python3
"""
Camera-parameter sanity check for a captured session.

WebXR's projectionMatrix encodes focal length and principal point as a
standard OpenGL-style 4×4. We've been trusting it implicitly, but if the
colour image we get from `view.camera` was cropped or scaled relative
to the FOV that produced P, every projection downstream will be off —
walls bend, reprojected colour misses by a few px, etc.

This script does two checks:

  1. Intrinsics from P. We decode (fx, fy, cx, cy) in pixels of the
     colour image from each frame's P matrix and report consistency
     across the session. Same camera throughout = these should be near
     identical. Implied FOV and principal-point offset get printed too.

  2. Reprojection error. For pairs of frames a few seconds apart
     (enough baseline for triangulation, plenty of overlap), we ORB-
     match features in the colour images, triangulate via
     cv2.triangulatePoints with the claimed K + the captured poses,
     then reproject and measure pixel error. A correct pinhole +
     undistorted model yields ~1 px median; large errors that grow
     with radius from the principal point are radial-distortion
     fingerprints; a constant offset is a focal-length mismatch.

Coordinate-system note:
    WebXR view-space is right-handed, camera looks down −z, +y up.
    OpenCV camera-space looks down +z, +y down. To use cv2.triangulate
    we apply diag(1, −1, −1) on the WebXR camera-from-world matrix.

Run:
    python tools/camera_check.py --session 20260427_123301
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import fusion
import serve

PROJECT_ROOT = ROOT.parent

# WebXR view (-z forward, +y up) → OpenCV camera (+z forward, +y down).
WEBXR_TO_CV = np.diag([1.0, -1.0, -1.0])


def intrinsics_from_P(P: np.ndarray, image_w: int, image_h: int) -> dict:
    """OpenGL-style 4×4 P → pinhole intrinsics in pixels of the supplied
    image size. Derivation:

        view (X, Y, −near) → clip → ndc → pixel (top-left origin, y down)
        fx_px = P[0,0] · W/2
        fy_px = P[1,1] · H/2
        cx_px = (1 + P[0,2]) · W/2     (image x increases rightward)
        cy_px = (1 − P[1,2]) · H/2     (image y increases downward; OpenGL y is up)

    The (1 ± P[i,2]) sign comes from the y-axis flip between OpenGL view
    (+y up) and pixel coords (y down)."""
    fx = float(P[0, 0]) * image_w / 2.0
    fy = float(P[1, 1]) * image_h / 2.0
    cx = (1.0 + float(P[0, 2])) * image_w / 2.0
    cy = (1.0 - float(P[1, 2])) * image_h / 2.0
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


def _decode_color_to_gray(frame: dict, flip_v: bool) -> np.ndarray:
    """Capture's RGBA buffer → contiguous grayscale uint8 for ORB.

    `flip_v=True` flips top-↔-bottom. The recorder's GL framebuffer
    readPixels stores row 0 at the FB's bottom. Whether that ends up at
    the bottom of the camera image (GL-native texture, no flip) or the
    top (CV-native texture, sampled bottom-up by the shader) depends on
    the WebXR runtime — so we try both and pick the one whose
    reprojection lines up."""
    cw = int(frame["color_width"]); ch = int(frame["color_height"])
    rgba = np.frombuffer(frame["color"], dtype=np.uint8).reshape(ch, cw, 4)
    gray = cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_RGB2GRAY)
    if flip_v:
        gray = cv2.flip(gray, 0)
    return gray


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--frames-root", default=str(PROJECT_ROOT / "captured_frames"))
    ap.add_argument("--time-gap", type=int, default=5,
                    help="frames apart for triangulation pairs (small = small "
                         "baseline = noisy triangulation; large = less overlap)")
    ap.add_argument("--n-pairs", type=int, default=24,
                    help="number of pairs to test (uniformly spaced over the session)")
    ap.add_argument("--n-features", type=int, default=2500,
                    help="ORB features per frame")
    ap.add_argument("--ratio-test", type=float, default=0.75,
                    help="Lowe's ratio for keeping a match")
    args = ap.parse_args()

    session_dir = Path(args.frames_root) / args.session / "frames"
    bins = sorted(session_dir.glob("frame_*.bin"))
    if not bins:
        print(f"no frames at {session_dir}", file=sys.stderr); sys.exit(1)
    print(f"checking {len(bins)} frames in {session_dir}")

    # ----- step 1: intrinsics consistency -----
    intrinsics_rows = []
    color_dims_first = None
    for fp in bins:
        frame = serve.parse_frame(fp.read_bytes())
        if frame["color"] is None: continue
        cw = int(frame["color_width"]); ch = int(frame["color_height"])
        P = fusion._mat4_from_column_major(frame["projectionMatrix"])
        K = intrinsics_from_P(P, cw, ch)
        intrinsics_rows.append((K["fx"], K["fy"], K["cx"], K["cy"], cw, ch))
        if color_dims_first is None:
            color_dims_first = (cw, ch)

    if not intrinsics_rows:
        print("no usable colour frames"); return
    K_arr = np.array(intrinsics_rows)
    cw, ch = color_dims_first
    print(f"\n=== intrinsics from P ({len(K_arr)} frames) ===")
    print(f"  colour image: {cw} × {ch}")
    for i, label in enumerate(["fx (px)", "fy (px)", "cx (px)", "cy (px)"]):
        v = K_arr[:, i]
        print(f"  {label:>9}: mean {v.mean():9.2f}  std {v.std():9.4f}  "
              f"min {v.min():9.2f}  max {v.max():9.2f}")
    fx_mean = float(K_arr[:, 0].mean())
    fy_mean = float(K_arr[:, 1].mean())
    cx_mean = float(K_arr[:, 2].mean())
    cy_mean = float(K_arr[:, 3].mean())
    fov_x_deg = 2.0 * np.degrees(np.arctan(cw / (2.0 * fx_mean)))
    fov_y_deg = 2.0 * np.degrees(np.arctan(ch / (2.0 * fy_mean)))
    fov_diag  = 2.0 * np.degrees(np.arctan(np.hypot(cw, ch) / (2.0 * np.hypot(fx_mean, fy_mean))))
    print(f"  implied FOV: {fov_x_deg:.1f}° horizontal, "
          f"{fov_y_deg:.1f}° vertical, {fov_diag:.1f}° diagonal")
    pp_off_x = cx_mean - cw / 2.0
    pp_off_y = cy_mean - ch / 2.0
    print(f"  principal point offset from centre: "
          f"dx={pp_off_x:+.1f} px, dy={pp_off_y:+.1f} px "
          f"({100*pp_off_x/cw:+.1f}%, {100*pp_off_y/ch:+.1f}%)")
    print(f"  pixel aspect ratio fy/fx = {fy_mean/fx_mean:.4f} "
          f"(square pixels = 1.000; |dev| > 0.01 is suspicious)")

    # ----- step 2: reprojection error -----
    K_mat = np.array([
        [fx_mean, 0,       cx_mean],
        [0,       fy_mean, cy_mean],
        [0,       0,       1.0],
    ], dtype=np.float64)
    orb = cv2.ORB_create(nfeatures=args.n_features)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING)

    n = len(bins)
    if n <= args.time_gap + 1:
        print("not enough frames for the requested time gap"); return
    starts = np.linspace(0, n - 1 - args.time_gap, args.n_pairs).astype(int)

    def run_pass(flip_v: bool):
        """Returns (all_errors_concat, radial_pairs) for this orientation."""
        all_errors_local = []
        radial_local = []
        for k, i in enumerate(starts):
            j = int(i) + args.time_gap
            f_i = serve.parse_frame(bins[int(i)].read_bytes())
            f_j = serve.parse_frame(bins[j].read_bytes())
            if f_i["color"] is None or f_j["color"] is None: continue

            gray_i = _decode_color_to_gray(f_i, flip_v=flip_v)
            gray_j = _decode_color_to_gray(f_j, flip_v=flip_v)
            kp_i, des_i = orb.detectAndCompute(gray_i, None)
            kp_j, des_j = orb.detectAndCompute(gray_j, None)
            if des_i is None or des_j is None: continue

            knn = bf.knnMatch(des_i, des_j, k=2)
            good = []
            for m_pair in knn:
                if len(m_pair) < 2: continue
                if m_pair[0].distance < args.ratio_test * m_pair[1].distance:
                    good.append(m_pair[0])
            if len(good) < 30: continue

            pts_i = np.array([kp_i[m.queryIdx].pt for m in good], dtype=np.float64)
            pts_j = np.array([kp_j[m.trainIdx].pt for m in good], dtype=np.float64)

            V_i = fusion._mat4_from_column_major(f_i["viewMatrix"])
            V_j = fusion._mat4_from_column_major(f_j["viewMatrix"])
            Rt_i = WEBXR_TO_CV @ np.linalg.inv(V_i)[:3, :]
            Rt_j = WEBXR_TO_CV @ np.linalg.inv(V_j)[:3, :]
            P_pix_i = K_mat @ Rt_i
            P_pix_j = K_mat @ Rt_j

            X_h = cv2.triangulatePoints(P_pix_i, P_pix_j, pts_i.T, pts_j.T)
            X = (X_h[:3] / np.where(np.abs(X_h[3:4]) < 1e-12, 1.0, X_h[3:4])).T
            in_front_i = (Rt_i @ np.hstack([X, np.ones((X.shape[0], 1))]).T)[2, :] > 0
            in_front_j = (Rt_j @ np.hstack([X, np.ones((X.shape[0], 1))]).T)[2, :] > 0
            keep = in_front_i & in_front_j
            if int(keep.sum()) < 10: continue
            X = X[keep]; pts_i = pts_i[keep]; pts_j = pts_j[keep]

            proj_i = (P_pix_i @ np.hstack([X, np.ones((X.shape[0], 1))]).T).T
            proj_j = (P_pix_j @ np.hstack([X, np.ones((X.shape[0], 1))]).T).T
            proj_i_px = proj_i[:, :2] / proj_i[:, 2:3]
            proj_j_px = proj_j[:, :2] / proj_j[:, 2:3]

            err_i = np.linalg.norm(proj_i_px - pts_i, axis=1)
            err_j = np.linalg.norm(proj_j_px - pts_j, axis=1)
            all_errors_local.append(err_i); all_errors_local.append(err_j)
            r_i = np.linalg.norm(pts_i - np.array([cx_mean, cy_mean]), axis=1)
            r_j = np.linalg.norm(pts_j - np.array([cx_mean, cy_mean]), axis=1)
            radial_local.extend(zip(r_i.tolist(), err_i.tolist()))
            radial_local.extend(zip(r_j.tolist(), err_j.tolist()))
        if not all_errors_local:
            return None
        return np.concatenate(all_errors_local), np.array(radial_local)

    print(f"\n=== reprojection test (pairs at time gap {args.time_gap}, "
          f"{args.n_pairs} pairs, both orientations) ===")
    res_noflip = run_pass(flip_v=False)
    res_flip   = run_pass(flip_v=True)
    if res_noflip is None and res_flip is None:
        print("no usable pairs"); return
    medians = {
        "no-flip":   None if res_noflip is None else float(np.median(res_noflip[0])),
        "v-flipped": None if res_flip   is None else float(np.median(res_flip[0])),
    }
    print("median reprojection error per orientation:")
    for label, med in medians.items():
        print(f"  {label:>9}: {('—' if med is None else f'{med:6.2f} px')}")
    # Pick the better orientation for the rest of the report.
    valid = [(k, v) for k, v in medians.items() if v is not None]
    valid.sort(key=lambda kv: kv[1])
    best_label = valid[0][0]
    best = res_flip if best_label == "v-flipped" else res_noflip
    err_arr, radial_arr = best
    print(f"  best:    {best_label} ({medians[best_label]:.2f} px median) — "
          f"using this for the radial breakdown.")
    # Drop the worst 5 % so a few feature-detection blunders don't drown the signal.
    err_clean = err_arr[err_arr < np.percentile(err_arr, 95)]
    print(f"\nreprojection error across {len(err_arr)} matched points "
          f"({len(err_arr) - len(err_clean)} dropped as outliers above p95):")
    print(f"  median {np.median(err_clean):5.2f} px  "
          f"p25 {np.percentile(err_clean, 25):5.2f}  "
          f"p75 {np.percentile(err_clean, 75):5.2f}  "
          f"p90 {np.percentile(err_clean, 90):5.2f}  "
          f"max {err_clean.max():5.2f}")

    # Radial-distortion fingerprint: median error per radial bin.
    radial = radial_arr
    radial = radial[radial[:, 1] < np.percentile(radial[:, 1], 95)]
    half_diag = float(np.hypot(cw / 2.0, ch / 2.0))
    edges = np.linspace(0, half_diag, 7)
    print(f"\nreprojection error vs distance from principal point "
          f"(image half-diagonal = {half_diag:.0f} px):")
    print(f"  {'radius (px)':>14}  {'frac':>6}  {'matches':>8}  "
          f"{'med err':>8}  {'p90 err':>8}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (radial[:, 0] >= lo) & (radial[:, 0] < hi)
        if int(mask.sum()) < 20: continue
        e = radial[mask, 1]
        print(f"  {f'{lo:5.0f}–{hi:5.0f}':>14}  "
              f"{(lo + hi)/2.0/half_diag:>5.2f}   "
              f"{int(mask.sum()):>8}  "
              f"{np.median(e):>7.2f}  "
              f"{np.percentile(e, 90):>7.2f}")

    print("\nInterpretation:")
    print("  median ≤ ~1.5 px and flat across radial bins → intrinsics OK.")
    print("  median rises with radius                     → radial distortion.")
    print("  median uniformly large                       → focal-length / principal-point bias.")
    print("  no-flip ≫ v-flipped                          → captured buffer is in GL "
          "convention (row 0 = bottom). The Rust voxeliser uses OpenGL "
          "projection math so it's internally consistent — no fix needed there.")


if __name__ == "__main__":
    main()
