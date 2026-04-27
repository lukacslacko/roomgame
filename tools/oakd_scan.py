#!/usr/bin/env python3
"""
tools/oakd_scan.py — OAK-D Lite room scanner.

A standalone alternative to the Android-Chrome WebXR scanner
(web/scan.html + web/scan.js). Captures aligned depth + RGB frames from a
Luxonis OAK-D Lite over USB-C using the DepthAI v2 Python API,
unprojects them through the device's calibrated intrinsics, tracks the
camera pose with frame-to-grid ICP, and feeds the resulting world
points into the SAME voxel pipeline (tools/voxel_store +
tools/meshing) the WebXR scanner uses.

Final outputs (web/out/room.npz, web/out/room.glb) are byte-identical in
format to what the WebXR scanner produces, so web/game.html "just
works" without any changes.

The WebXR scanner is left fully intact — pick whichever capture front
end fits the situation.

Setup
-----
    cd /Users/lukacs/claude/roomgame
    source .venv/bin/activate
    pip install -r tools/oakd_requirements.txt

Run
---
    python tools/oakd_scan.py                 # default: 640x360 @ 10 fps
    python tools/oakd_scan.py --preview       # live RGB + depth preview
    python tools/oakd_scan.py --no-icp        # disable ICP refinement
    python tools/oakd_scan.py --remesh-every 30   # remesh every 30 frames

Ctrl-C (or ESC in the preview window) stops capture and writes
room.npz + room.glb. Open https://localhost:8080/game.html in another
terminal (running `python tools/serve.py`) to view the result.

Pose tracking
-------------
The OAK-D Lite has no SLAM and (typically) no IMU, so we have no a
priori pose. Strategy:

  - Frame 1 establishes the world frame: camera at world origin, looking
    along world -Z, with world +Y as "up" (standard OpenGL/three.js
    convention).
  - Frame N>1 starts from frame N-1's pose as an initial guess, then
    runs one Procrustes-ICP iteration against the existing voxel grid
    (using voxel_store's drift-detection machinery, which is already
    proven on real WebXR scans). The correction nudges this frame's
    points and updates the running pose so the next frame inherits the
    refined position.

This means you must move the camera SLOWLY between frames — adjacent
frames need to overlap on actual surfaces so ICP can lock the new pose
against existing voxels. Per-frame motion above ~5 cm or ~2° is capped
out by voxel_store's ICP safety limits; faster motion will accumulate
drift. At 10 fps and walking pace (~0.5 m/s) that's exactly the budget
you have. Sweep, don't dart.

Hold the OAK-D upright when you start (USB-C connector pointing down,
or pointing up — pick one and stay consistent). The first frame
defines world Y=up relative to whatever camera attitude it sees.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

# Make tools/ imports work whether the script is run as
#   `python tools/oakd_scan.py`     (cwd = project root)
# or
#   `cd tools && python oakd_scan.py`
TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import meshing  # noqa: E402
from voxel_store import VoxelRoom  # noqa: E402


# OpenCV camera (x_right, y_down, z_forward) → OpenGL/WebXR/three.js
# world axes (x_right, y_up, z_back). This matches what fusion.py
# produces from a WebXR frame, so the resulting voxel grid lines up
# the same way in game.html regardless of which scanner generated it.
CV_TO_GL = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


# ---------------------------------------------------------------------------
# DepthAI v3 pipeline construction
# ---------------------------------------------------------------------------
#
# The OAK-D Lite has three cameras on these board sockets:
#   CAM_A = colour (IMX214, 4K)
#   CAM_B = left mono (OV7251)
#   CAM_C = right mono (OV7251)
#
# DepthAI v3 collapsed ColorCamera/MonoCamera into a single `Camera` node
# whose outputs are requested via `.requestOutput((W, H), type, fps)`.
# Output queues are created directly off the resulting node outputs (no
# more XLinkOut intermediary). The Pipeline owns its own device and is
# itself a context manager — `with dai.Pipeline() as pipeline:` opens
# the device, builds nodes, and tears down on exit.
#
# StereoDepth.setDepthAlign(CAM_A) reprojects the depth map into the
# colour camera's frame on-device (Myriad-X does the warp); setOutputSize
# forces both streams to the same resolution so depth pixel (i, j)
# corresponds to colour pixel (i, j) exactly.

def build_pipeline(pipeline, width: int, height: int, fps: float):
    """Wire a Pipeline (already opened) for aligned depth + RGB.

    Returns the `Sync` node's output — a single queue that emits paired
    `MessageGroup`s containing one RGB frame and one depth frame whose
    timestamps fall within `SYNC_THRESHOLD_MS` of each other. Pulling
    from this single queue is what keeps us at full device-side framerate
    instead of dropping ~half the frames to RGB/depth misalignment.
    """
    import depthai as dai

    # CRITICAL: requestOutput() defaults to ImgResizeMode.CROP, which takes
    # a centre crop of the sensor at native pixel scale. But the
    # `getCameraIntrinsics(socket, W, H)` lookup later assumes the W×H
    # image is an *isotropic scale* of the calibrated frame, so cropping
    # leaves us using fx/fy ~6× too small — a flat wall reconstructs as a
    # cone, and any focus / scale change shows up as two reconstructions
    # at different sizes. STRETCH makes the resize match the intrinsics
    # math (and is loss-less here because the IMX214 1080p mode and the
    # OV7251 mono modes are both 16:9, like our 640×360 request).
    cam_color = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_A
    )
    # Lock IMX214 autofocus too — the calibrated intrinsics correspond to
    # one specific focus position; AF moving mid-scan shifts fx/fy and
    # superimposes two scaled copies of the scene. 130 is approximately
    # hyperfocal for the IMX214.
    cam_color.initialControl.setManualFocus(130)
    color_out = cam_color.requestOutput(
        (width, height), type=dai.ImgFrame.Type.BGR888i,
        resizeMode=dai.ImgResizeMode.STRETCH, fps=fps,
    )

    cam_left = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_B
    )
    left_out = cam_left.requestOutput(
        (width, height), type=dai.ImgFrame.Type.NV12,
        resizeMode=dai.ImgResizeMode.STRETCH, fps=fps,
    )

    cam_right = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_C
    )
    right_out = cam_right.requestOutput(
        (width, height), type=dai.ImgFrame.Type.NV12,
        resizeMode=dai.ImgResizeMode.STRETCH, fps=fps,
    )

    stereo = pipeline.create(dai.node.StereoDepth)
    # v3 renamed HIGH_ACCURACY → ACCURACY (and added FAST_ACCURACY etc).
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ACCURACY)
    stereo.setLeftRightCheck(True)
    # Subpixel disparity is critical for room-scale scanning. Without it,
    # depth is quantised in discrete buckets (~3 cm at 1 m, ~10 cm at 2 m,
    # ~25 cm at 3 m on the OAK-D Lite's 7.5 cm baseline) and a single
    # viewpoint reconstructs as a stack of concentric depth shells. With
    # 5-bit subpixel precision the same buckets shrink by 32×.
    stereo.setSubpixel(True)
    stereo.setSubpixelFractionalBits(5)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(width, height)
    # Tighten the confidence threshold from the default (245, very loose).
    # OAK-D Lite is passive stereo with no IR projector, so on anything
    # less than well-textured surfaces the disparity matcher confidently
    # returns garbage. 200 drops the worst of it.
    stereo.initialConfig.setConfidenceThreshold(200)
    # Post-processing chain. The biggest noise win comes from speckle
    # (drops isolated mismatched-disparity blobs) and temporal (averages
    # across consecutive frames — perfect for slow scanning). Spatial is
    # an edge-preserving smooth that cleans flat surfaces. Threshold
    # caps the depth range so wildly-wrong far disparities are dropped.
    pp = stereo.initialConfig.postProcessing
    pp.speckleFilter.enable = True
    # 28 is the actual ceiling on the OAK-D Lite's stock memory budget;
    # asking for 50 prints a noisy warning and silently clips to 28
    # anyway, so just request 28 directly.
    pp.speckleFilter.speckleRange = 28
    pp.spatialFilter.enable = True
    pp.spatialFilter.holeFillingRadius = 2
    pp.spatialFilter.numIterations = 1
    pp.temporalFilter.enable = True
    # CRITICAL: default persistencyMode is VALID_8_OUT_OF_8 — a pixel
    # must be valid in 8 consecutive frames before it gets emitted at
    # all. Combined with our LRC + confidence + speckle + spatial
    # filters, almost no pixels survive, and the depth preview is
    # nearly black. VALID_2_IN_LAST_4 needs only 2 valid frames out of
    # the most recent 4 — strict enough to drop flicker, loose enough
    # that we actually see the scene.
    pp.temporalFilter.persistencyMode = (
        dai.StereoDepthConfig.PostProcessing.TemporalFilter
        .PersistencyMode.VALID_2_IN_LAST_4
    )
    pp.thresholdFilter.minRange = 200      # mm
    pp.thresholdFilter.maxRange = 5000     # mm
    left_out.link(stereo.left)
    right_out.link(stereo.right)

    sync = pipeline.create(dai.node.Sync)
    # 1.5 frame periods — generous enough that paired frames almost
    # always sync, tight enough that we don't pair stale data.
    sync.setSyncThreshold(timedelta(milliseconds=int(1500.0 / max(fps, 1.0))))
    color_out.link(sync.inputs[SYNC_KEY_RGB])
    stereo.depth.link(sync.inputs[SYNC_KEY_DEPTH])

    return sync.out


SYNC_KEY_RGB = "rgb"
SYNC_KEY_DEPTH = "depth"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def unproject_depth(
    depth_mm: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    near_m: float, far_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(H, W) uint16 mm → (N, 3) float32 OpenCV camera-frame points + the
    flat indices of the kept pixels (so the caller can sample colour at
    the same locations)."""
    h, w = depth_mm.shape
    z = depth_mm.astype(np.float32) * 0.001  # mm → m
    mask = (z > near_m) & (z < far_m)
    flat_idx = np.flatnonzero(mask.ravel())
    if flat_idx.size == 0:
        return np.empty((0, 3), dtype=np.float32), flat_idx
    z_kept = z.ravel()[flat_idx]
    # Pixel centres: (i + 0.5, j + 0.5). Skip the +0.5 to match the
    # WebXR fusion convention (which treats integer (u, v) as the
    # reference point); the half-pixel shift just rotates the mesh by a
    # sub-mm constant.
    j_kept = (flat_idx % w).astype(np.float32)
    i_kept = (flat_idx // w).astype(np.float32)
    x_cam = (j_kept - cx) / fx * z_kept
    y_cam = (i_kept - cy) / fy * z_kept
    pts = np.stack([x_cam, y_cam, z_kept], axis=1).astype(np.float32)
    return pts, flat_idx


def transform(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply rigid p' = R @ p + t to (N, 3) points."""
    return (pts.astype(np.float64) @ R.T + t).astype(np.float32)


# ---------------------------------------------------------------------------
# ICP refinement against the cumulative voxel grid
# ---------------------------------------------------------------------------

def icp_refine(
    world_pts: np.ndarray,
    voxel_room: VoxelRoom,
) -> dict | None:
    """One Procrustes-ICP iteration of `world_pts` against the existing
    grid. Returns the diagnostic dict from voxel_store._check_drift, or
    None if the grid is too sparse / no overlap."""
    with voxel_room.lock():
        # Both of these are intra-package private helpers (leading
        # underscore). They have no side effects beyond rebuilding /
        # querying the kd-tree, so calling them directly from a sibling
        # script is safe and far simpler than synthesising a fake
        # WebXR-style frame just to run through ingest_frame.
        voxel_room._maybe_rebuild_kdtree()
        return voxel_room._check_drift(world_pts)


def apply_correction(
    world_pts: np.ndarray,
    pose_R: np.ndarray, pose_t: np.ndarray,
    check: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Decide whether to apply the ICP-proposed (R, t) correction and
    return (corrected_world_pts, new_pose_R, new_pose_t, msg)."""
    R_corr = check["R"].astype(np.float64)
    t_corr = check["t"].astype(np.float64)
    rot_deg = float(check["rot_deg"])
    t_mag = float(np.linalg.norm(t_corr))

    if (rot_deg > VoxelRoom.ICP_MAX_ROTATION_DEG
            or t_mag > VoxelRoom.ICP_MAX_TRANSLATION_M):
        return (world_pts, pose_R, pose_t,
                f" corr-skipped(t{t_mag*100:.2f}cm/r{rot_deg:.2f}°)")

    if rot_deg <= 0.05 and t_mag <= 0.0005:
        return (world_pts, pose_R, pose_t, "")

    corrected = transform(world_pts, R_corr, t_corr)
    new_R = R_corr @ pose_R
    new_t = R_corr @ pose_t + t_corr
    msg = (f" corr=t{t_mag*100:.2f}cm/r{rot_deg:.2f}°"
           f" → res {check['median_dist_after']*100:.2f}cm")
    return corrected, new_R, new_t, msg


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--voxel-size", type=float, default=0.02,
                    help="voxel edge length in metres (default: 0.02 = 2 cm)")
    ap.add_argument("--width", type=int, default=640,
                    help="aligned depth+RGB frame width (default: 640)")
    ap.add_argument("--height", type=int, default=360,
                    help="aligned depth+RGB frame height (default: 360)")
    ap.add_argument("--fps", type=float, default=15.0,
                    help="device-side capture rate; the loop processes every "
                         "synced pair so this is effectively the integration "
                         "rate too (subject to per-frame ICP+insert cost)")
    ap.add_argument("--near", type=float, default=0.20,
                    help="reject depth values closer than this (m)")
    ap.add_argument("--far", type=float, default=6.0,
                    help="reject depth values farther than this (m)")
    ap.add_argument("--stride", type=int, default=2,
                    help="depth pixel stride (1 = no subsample)")
    ap.add_argument("--no-icp", action="store_true",
                    help="disable ICP refinement (trust raw pose chain)")
    ap.add_argument("--preview", action="store_true",
                    help="show live RGB + colour-mapped depth windows (cv2)")
    ap.add_argument("--remesh-every", type=int, default=0,
                    help="rewrite room.glb every N frames (0 = only on exit)")
    ap.add_argument("--min-hits", type=int, default=5,
                    help="before meshing, drop voxels observed in fewer than "
                         "this many frames. OAK-D passive stereo is noisy on "
                         "untextured surfaces — at default 5, we mesh only "
                         "voxels that survived 5+ frames of confirmation, "
                         "which kills the random-disparity background fog. "
                         "Set 0 to disable, raise it (10–20) for stricter, "
                         "lower for sparser scans.")
    ap.add_argument("--out", type=str, default=None,
                    help="output directory (default: <project>/web/out)")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "web" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "room.npz"
    glb_path = out_dir / "room.glb"

    try:
        import depthai as dai
    except ImportError:
        sys.stderr.write(
            "depthai is not installed. Activate your venv and run:\n"
            "  pip install -r tools/oakd_requirements.txt\n"
        )
        sys.exit(1)

    if args.preview:
        try:
            import cv2  # noqa: F401
        except ImportError:
            sys.stderr.write(
                "--preview requires opencv-python. Install with:\n"
                "  pip install opencv-python\n"
            )
            sys.exit(1)

    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        sys.stderr.write(
            "No OAK device found.\n"
            "  - Is the OAK-D Lite plugged in?\n"
            "  - Is the cable a USB-C *data* cable (many are charge-only)?\n"
            "  - Try replugging; macOS sometimes needs a moment to enumerate.\n"
        )
        sys.exit(1)

    sys.stderr.write(f"Found {len(devices)} OAK device(s):\n")
    for d in devices:
        sys.stderr.write(f"  {_device_id(d)}  state={_device_state(d)}\n")

    voxel_room = VoxelRoom(voxel_size_m=args.voxel_size)

    # Pose: world_from_camera. Identity = camera at world origin in the
    # OpenGL convention (Y up, looking along -Z). We map OpenCV camera
    # points through CV_TO_GL before applying this pose.
    pose_R = np.eye(3, dtype=np.float64)
    pose_t = np.zeros(3, dtype=np.float64)

    # Graceful Ctrl-C: flag the loop, let it write the mesh, exit.
    stop = {"flag": False}
    def _sigint(*_):
        stop["flag"] = True
        sys.stderr.write("\n(stopping — flushing mesh)\n")
    signal.signal(signal.SIGINT, _sigint)

    with dai.Pipeline() as pipeline:
        sync_out = build_pipeline(
            pipeline, args.width, args.height, args.fps,
        )

        device = pipeline.getDefaultDevice()
        sys.stderr.write(
            f"Connected: id={_device_id(device)} "
            f"cams={device.getConnectedCameras()}\n"
        )
        try:
            imu = device.getConnectedIMU()
            sys.stderr.write(f"IMU: {imu or 'none'} (currently unused)\n")
        except (AttributeError, RuntimeError):
            pass

        # `device.readCalibration()` is the on-device factory calibration;
        # `pipeline.getCalibrationData()` is the *staged* (about-to-flash)
        # calibration, which is normally empty. Always prefer the former.
        calib = device.readCalibration()
        K = np.asarray(calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A, args.width, args.height
        ))
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        sys.stderr.write(
            f"Intrinsics @ {args.width}x{args.height}: "
            f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}\n"
        )

        # Single queue of synced (rgb + depth) MessageGroups. blocking=False
        # so when the consumer (this loop) falls behind, old groups are
        # dropped at the device boundary instead of piling up unboundedly.
        q_sync = sync_out.createOutputQueue(maxSize=4, blocking=False)

        pipeline.start()

        sys.stderr.write(
            "\nStreaming. Hold the OAK-D upright and pan the room SLOWLY —\n"
            "adjacent frames need to overlap on real surfaces so ICP can\n"
            "lock the next pose to the existing grid. Ctrl-C (or ESC in\n"
            "the preview window) stops capture and writes room.glb.\n\n"
        )

        frame_idx = 0
        first_real_frame = True
        last_log = time.time()
        fps_window_start = time.time()
        fps_window_count = 0
        fps_recent = 0.0

        while not stop["flag"] and pipeline.isRunning():
            # Block up to 250 ms for the next paired (rgb+depth) group.
            # On timeout get() returns None and we re-check stop / isRunning.
            group = q_sync.get(timedelta(milliseconds=250))
            if group is None:
                continue

            rgb_pkt = group[SYNC_KEY_RGB]
            dep_pkt = group[SYNC_KEY_DEPTH]
            rgb_bgr = rgb_pkt.getCvFrame()      # (H, W, 3) uint8 BGR
            depth_mm = dep_pkt.getFrame()       # (H, W) uint16 mm
            if depth_mm.shape[:2] != rgb_bgr.shape[:2]:
                sys.stderr.write(
                    f"size mismatch: rgb={rgb_bgr.shape} depth={depth_mm.shape}"
                    " — skipping frame\n"
                )
                continue

            if args.stride > 1:
                depth_use = depth_mm[::args.stride, ::args.stride]
                rgb_use = rgb_bgr[::args.stride, ::args.stride]
                fx_u = fx / args.stride
                fy_u = fy / args.stride
                cx_u = cx / args.stride
                cy_u = cy / args.stride
            else:
                depth_use = depth_mm
                rgb_use = rgb_bgr
                fx_u, fy_u, cx_u, cy_u = fx, fy, cx, cy

            pts_cv, flat_idx = unproject_depth(
                depth_use, fx_u, fy_u, cx_u, cy_u, args.near, args.far,
            )
            if pts_cv.shape[0] == 0:
                continue

            # OpenCV camera frame → OpenGL/WebXR convention, then current pose.
            pts_gl = pts_cv @ CV_TO_GL.T.astype(np.float32)
            world_pts = transform(pts_gl, pose_R, pose_t)

            # Sample colour at the same kept pixels (BGR → RGB).
            rgb_flat = rgb_use.reshape(-1, 3)[flat_idx]
            colors_rgb = np.ascontiguousarray(rgb_flat[:, ::-1])

            correction_msg = ""
            if not args.no_icp and not first_real_frame:
                check = icp_refine(world_pts, voxel_room)
                if (check is not None
                        and check["match_fraction"] >= VoxelRoom.DRIFT_MIN_MATCH_FRACTION):
                    if check["median_dist"] >= VoxelRoom.DRIFT_REJECT_DIST_M:
                        # Frame too far from existing surface to ICP-correct.
                        # voxel_store rejects this case; we mirror that
                        # behaviour rather than baking in a 10+ cm jump.
                        correction_msg = (
                            f" REJECTED drift {check['median_dist']*100:.1f}cm"
                            f"@{check['match_fraction']*100:.0f}%"
                        )
                        if _should_log(last_log):
                            sys.stderr.write(
                                f"frame {frame_idx+1} skipped:{correction_msg}\n"
                            )
                            last_log = time.time()
                        continue
                    world_pts, pose_R, pose_t, correction_msg = apply_correction(
                        world_pts, pose_R, pose_t, check,
                    )

            n_written = voxel_room.insert_points(world_pts, colors_rgb)
            first_real_frame = False
            frame_idx += 1
            fps_window_count += 1

            if args.remesh_every > 0 and frame_idx % args.remesh_every == 0:
                _remesh(voxel_room, glb_path)

            now = time.time()
            window = now - fps_window_start
            if window >= 1.0:
                fps_recent = fps_window_count / window
                fps_window_start = now
                fps_window_count = 0
            if now - last_log >= 0.5:
                stats = voxel_room.stats()
                # Total rotation of the running pose, in degrees, so we can
                # tell whether ICP is tracking actual camera rotation or
                # just sitting at identity.
                cos_a = np.clip((np.trace(pose_R) - 1.0) * 0.5, -1.0, 1.0)
                pose_rot_deg = float(np.degrees(np.arccos(cos_a)))
                sys.stderr.write(
                    f"frame {frame_idx} {fps_recent:4.1f}fps "
                    f"pts={n_written} "
                    f"pos=({pose_t[0]:+.2f},{pose_t[1]:+.2f},{pose_t[2]:+.2f}) "
                    f"rot={pose_rot_deg:5.1f}° "
                    f"vox={stats['voxels']} ch={stats['chunks']}{correction_msg}\n"
                )
                last_log = now

            if args.preview:
                _draw_preview(rgb_bgr, depth_mm)
                # cv2.waitKey returning 27 (ESC) flips the stop flag.
                import cv2
                if cv2.waitKey(1) & 0xFF == 27:
                    stop["flag"] = True

    if args.preview:
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

    sys.stderr.write("\nWriting outputs…\n")
    voxel_room.save(npz_path)
    sys.stderr.write(f"  saved {npz_path}\n")
    if args.min_hits > 0:
        dropped = _filter_min_hits(voxel_room, args.min_hits)
        sys.stderr.write(
            f"  dropped {dropped} voxels with hits < {args.min_hits} "
            "before meshing\n"
        )
    res = meshing.remesh_to_glb(voxel_room, glb_path)
    if res is None:
        sys.stderr.write(
            "  no surface to mesh (did the camera see anything in range?\n"
            "   if --min-hits is high, try lowering it)\n"
        )
    else:
        sys.stderr.write(
            f"  wrote {glb_path} ({res['vertices']} verts, {res['faces']} faces)\n"
        )
    sys.stderr.write(
        "\nOpen https://localhost:8080/game.html (or whatever port your\n"
        "tools/serve.py is running on) to view the result.\n"
    )


def _should_log(last_log: float) -> bool:
    return (time.time() - last_log) >= 0.5


def _device_id(d) -> str:
    """Extract a printable ID from a DeviceInfo or a connected Device.
    DepthAI shuffled this across versions: v2 had `mxid` / `getMxId()`,
    v3 has `deviceId` / `getDeviceId()`. Try them all."""
    for attr in ("deviceId", "mxid"):
        val = getattr(d, attr, None)
        if val:
            return str(val)
    for method in ("getDeviceId", "getMxId"):
        if hasattr(d, method):
            try:
                return str(getattr(d, method)())
            except Exception:
                pass
    return "?"


def _device_state(d) -> str:
    state = getattr(d, "state", None)
    if state is None:
        return "?"
    return getattr(state, "name", str(state))


def _filter_min_hits(voxel_room: VoxelRoom, min_hits: int) -> int:
    """Zero out the hits channel on voxels observed fewer than `min_hits`
    times. Returns the count of voxels dropped. Mutates the grid in
    place. Done at oakd_scan layer (not inside voxel_store) because the
    WebXR scanner has different noise characteristics and shouldn't get
    this filter automatically."""
    from voxel_store import CHANNEL_HITS
    dropped = 0
    with voxel_room.lock():
        for chunk in voxel_room.chunks.values():
            mask = (chunk[..., CHANNEL_HITS] > 0) & (chunk[..., CHANNEL_HITS] < min_hits)
            dropped += int(mask.sum())
            chunk[mask, CHANNEL_HITS] = 0
    return dropped


def _remesh(voxel_room: VoxelRoom, glb_path: Path) -> None:
    res = meshing.remesh_to_glb(voxel_room, glb_path)
    if res is None:
        sys.stderr.write("  (remesh: no surface yet)\n")
    else:
        sys.stderr.write(
            f"  remesh → {glb_path.name} "
            f"({res['vertices']}v, {res['faces']}f)\n"
        )


def _draw_preview(rgb_bgr: np.ndarray, depth_mm: np.ndarray) -> None:
    import cv2
    d_clip = np.clip(depth_mm, 200, 4000).astype(np.float32)
    d_norm = ((d_clip - 200) * (255.0 / (4000 - 200))).astype(np.uint8)
    d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
    d_color[depth_mm == 0] = 0
    cv2.imshow("oakd-rgb", rgb_bgr)
    cv2.imshow("oakd-depth", d_color)


if __name__ == "__main__":
    main()
