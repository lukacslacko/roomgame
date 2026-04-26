# roomgame

Phone-scanned voxel room model, rendered as a fly-through on the laptop.
The plan is for it to eventually become a small racing/destruction game
inside the scanned room. Right now it's a working scanner + reconstruction
pipeline, with one unsolved issue (gradual ARCore pose drift).

## Architecture

```
[Android Chrome / WebXR depth + camera]   [OAK-D Lite via USB-C / DepthAI v3]
         │  binary frames                          │  Sync-paired depth + RGB
         ▼  HTTPS POST /frame, ~5–10 fps           ▼  in-process, ~10 fps
[Python serve.py]                         [Python tools/oakd_scan.py]
   ├── parse_frame      (224-byte hdr)             ├── depthai pipeline
   ├── fusion.py        (depth → world)            ├── intrinsics unproject
   │                                                │
   └─────────────► voxel_store.py ◄─────────────────┘
                  (chunked uint8 RGBA grid + ICP)
                          │
                          ▼
                     meshing.py (dense marching cubes → GLB)
                          │
                          ▼  web/out/room.glb
                  [Laptop browser, three.js]
                  └── game.html (fly cam, SSAO, vertex colours, shadows)
```

Two interchangeable capture front ends share the same voxel + meshing
back end. The WebXR scanner integrates over HTTP `POST /frame`; the
OAK-D scanner runs as a standalone process and writes the same
`web/out/room.glb` directly. Pick whichever is convenient.

No build step on either side; three.js loaded via importmap from
jsdelivr. Server is stdlib `ThreadingHTTPServer` with HTTPS auto-cert
generation lifted from the [building-to-be](https://github.com/lukacslacko/building-to-be)
sibling project.

## Quick start

```
cd /Users/lukacs/claude/roomgame
python -m venv .venv && source .venv/bin/activate
pip install -r tools/requirements.txt
python tools/serve.py --https
```

The terminal prints a QR code with the LAN URL. Scan it with **Android
Chrome** (depth-sensing isn't available on iOS Safari). Hit `Enter AR`,
tap to start streaming, tap again to stop. On the laptop open
`https://localhost:8443/game.html` and click `Re-mesh & reload`.

Phone HUD:
- `D` button — toggle the live banded depth overlay (1 m bands).
- `↺` button — wipe the voxel grid and start over.
- Tap anywhere else — toggle streaming.

Laptop viewer:
- `WASD` move, `Space`/`Shift` up/down, arrows look, drag-to-look,
  wheel dollies.

## Pose stability validation (`/pose.html`)

Open `/pose.html` on the phone to validate how much ARCore drifts on
this device:

1. Tap **Enter AR**.
2. Tap anywhere (or the `⊕` button) to mark the current pose as origin.
3. Walk around. Translation (cm) and Euler angles (yaw/pitch/roll, deg)
   are shown live, plus a top-down trajectory plot.
4. Return to the start pose; readings should fall back near zero.
   Whatever residual remains is ARCore's drift on this run.

The page is self-contained — no server-side state, no recording. It's a
sanity check before doing a long capture.

## Coarse occupancy cubes (`/cubes.html`)

The cube scanner is a simpler, more robust alternative to the 2 cm
voxel pipeline. It maintains a dense 3D grid of larger cubes (default
25 cm) with two saturating counters per cube:

- `occupied` ticks each time a frame's depth map shows a surface
  passing through that cube while the cube is fully visible.
- `free` ticks each time the cube is fully visible *and* every valid
  sample reports the surface beyond the cube (i.e. the camera saw
  past the cube into empty space).

Cubes with `occupied / (occupied + free) > threshold` (default 0.25)
are returned by `GET /cubes/state` and rendered live on the phone:

- **Plain mode**: cubes drawn with alpha = mix slider, blended on top
  of camera passthrough.
- **Occluded mode**: per-fragment depth comparison with the live
  WebXR depth buffer hides cube fragments behind real surfaces.

Endpoints:

| Method | Path | Body / query | Effect |
|---|---|---|---|
| POST | `/cubes/start` | `{cube_size?, world_min?, world_max?, reset?}` | Init/replace grid; turn recording on |
| POST | `/cubes/stop`  | — | Turn recording off |
| POST | `/cubes/reset` | — | Wipe counters |
| GET  | `/cubes/state` | `?threshold=0.25&min=2` | JSON: cube list with indices + counts |

While `/cubes/start` is active, every `/frame` ingestion also feeds
the cube grid. Raw frame bodies still land in `captured_frames/` for
later offline reconstruction (planned 5 cm voxel-within-cube pass).

## Alternative scanner: OAK-D Lite

The Android Chrome scanner is the default, but if you have a Luxonis
**OAK-D Lite** depth camera you can scan from the laptop directly over
USB-C — no phone, no HTTPS dance, no ARCore drift.

```
# one-time install (in the same venv as serve.py)
pip install -r tools/oakd_requirements.txt

# plug the OAK-D Lite into a USB-C *data* cable, then:
python tools/oakd_scan.py                   # default 640x360 @ 15 fps
python tools/oakd_scan.py --preview         # live RGB + depth preview
python tools/oakd_scan.py --min-hits 20     # stricter noise filter
python tools/oakd_scan.py --remesh-every 30 # rewrite room.glb every 30 frames
```

Ctrl-C (or ESC in the preview window) stops capture and writes
`web/out/room.npz` + `web/out/room.glb`. Open
`https://localhost:8080/game.html` (run `tools/serve.py` in another
terminal) to fly through the result — the viewer doesn't know or care
which scanner produced the GLB.

What `oakd_scan.py` actually does:

1. Builds a DepthAI v3 pipeline that aligns the stereo-depth output to
   the colour camera on-device (`StereoDepth.setDepthAlign(CAM_A)` +
   `setOutputSize`) so depth pixel `(i, j)` corresponds to RGB pixel
   `(i, j)` exactly, and pairs the two streams via a `dai.node.Sync`
   so we read them as a single timestamp-aligned `MessageGroup`
   (avoids dropping ~half the frames to RGB/depth queue skew).
2. Reads the calibrated intrinsics for that resolution from
   `Device.readCalibration().getCameraIntrinsics(CAM_A, W, H)`.
3. Unprojects each depth pixel (uint16 mm) through `(fx, fy, cx, cy)`
   into camera-frame metres, converts OpenCV camera axes
   `(x_right, y_down, z_forward)` → OpenGL/three.js world axes
   `(x_right, y_up, z_back)` so the voxel grid lines up the same way
   the WebXR scanner produces.
4. Tracks the camera pose by running one Procrustes-ICP iteration
   per frame against the cumulative voxel grid (using
   `voxel_store._check_drift`, the same machinery the WebXR scanner
   uses for drift correction). Frame 1 anchors the world origin.

**Pose-tracking caveats.** OAK-D Lite has no SLAM and the cost-reduced
units have no IMU either, so we have no a priori pose. The script
relies entirely on frame-to-grid ICP, which means:

- Adjacent frames must overlap on real surfaces — pan **slowly**
  (sweep, don't dart). Per-frame motion above ~5 cm or ~2° gets capped
  by `voxel_store`'s ICP safety limits and drift will accumulate.
- Hold the device upright (USB-C connector consistently down or up).
  The first frame defines world Y=up relative to whatever attitude the
  camera has at startup.
- The first ~1500 voxels are inserted with no ICP refinement (drift
  detection needs a populated grid to query against). At 2 cm voxels
  this is roughly the first 1–2 frames; ICP starts kicking in by
  frame 3.

If pose tracking falls over, try `--no-icp` to confirm raw capture is
working, then scan more slowly. `--preview` is invaluable for
sanity-checking what the depth map looks like.

**Hardware notes.** USB-C from a laptop is enough power; no external
adapter needed. Most early-Kickstarter OAK-D Lite units shipped without
the IMU, later units have a BMI270 — `oakd_scan.py` prints which one is
attached on startup but doesn't currently use it.

## What works

- **WebXR depth capture** at 160×90 (`luminance-alpha` uint16) on Pixel.
  Chrome's `XRCPUDepthInformation.data` is an `ArrayBuffer`, not a
  `Uint8Array` — getting that wrong silently sends all-zero payloads
  (no error, no crash). See `scan.js#captureAndSendInner`.
- **WebXR camera-access** at 320×180 (RGBA8) via a small GLSL pass that
  copies the camera texture into a framebuffer and `readPixels` back.
- **Chunked sparse voxel grid** at 2 cm: `dict[(ci,cj,ck)] →
  np.ndarray((32,32,32,4), uint8)` with channels (R, G, B, hits).
  Saturating uint8 hit-count, running RGB average per voxel.
- **Dense marching cubes** on the hits channel; colours via trilinear
  interpolation of RGBA neighbours weighted by presence. trimesh writes
  vertex colours into the GLB's COLOR_0 attribute.
- **Streaming** with natural backpressure (one in-flight `fetch` at a
  time) and `pose.emulatedPosition` filtering.
- **Fly camera** in three.js with WASD/arrows/mouse, ACES tone mapping,
  PCFSoftShadowMap, and an SSAO pass for crevice darkening.

## Critical things we got wrong (and how)

These cost real time. Future-self, please remember:

1. **Buffer's column index is flipped from the matrix's u_d direction.**
   The W3C Depth Sensing spec says buffer coords are Y-up bottom-left;
   Chrome's actual array storage is row-0-at-top *and* col-0-at-right
   relative to the matrix's u_d. Empirical sequence:
   - No flips → "left-right correct, up-down inverted" (Pixel pilot)
   - `by` flipped → both inverted
   - `bx` flipped (current) → both correct
   The matrix has a 90° rotation block (u_d ↔ v_v, v_d ↔ u_v), so
   "flipping the wrong axis" maps to the *other* on-screen axis.
2. **`view.transform.matrix` is `world_from_view`.** Translation is the
   camera's world position. Multiplying view-space points by it gives
   world points (no inversion needed).
3. **WebXR depth values are z-distance, not radial.** Confirmed both
   from the spec and from RANSAC plane-fits on captured frames (RMSE
   1.5–1.8 cm — consistent across walls and floor, so the projection
   isn't bending single frames).
4. **Buffer pixels outside the view's u_v ∈ [0, 1] window have to be
   filtered**, otherwise they unproject to ghost points at huge view
   angles (~17 % of the buffer on a Pixel scan).
5. **Camera-image lifetime is the current XR frame.** `getCameraImage`
   must be called and `readPixels` must complete inside the
   `requestAnimationFrame` callback synchronously.

## Unsolved: cumulative pose drift

The current visible artefact is "fan of sheets" walls — a single
physical wall ends up as several near-parallel surfaces fanned out
from each other. Top-down screenshots show this clearly.

What we ruled out via measurement:
- **Projection error** — single-frame plane-fit RMSE is 1.5–1.8 cm,
  no spatial gradient, so the per-frame geometry is straight.
- **Discrete re-localisation** — pose-jump rejection at 30 cm catches
  the obvious case (verified against a synthetic 0.5 m teleport), but
  on real scans 0–1 jumps trigger out of 240+ frames.

What's actually happening:
- ARCore's reported pose drifts smoothly across frames in both
  translation (~0.3 cm/frame) and orientation (~0.3°/frame).
- Each new frame gets integrated at its slightly-shifted location.
- After 200 frames that's a few cm + a few degrees of accumulated drift
  — exactly enough to fan a wall into 3–4 sheets.

## What's been tried for drift

| Mitigation | Effect |
|---|---|
| Skip `pose.emulatedPosition === true` frames | Helps for hard tracking losses; doesn't catch gradual drift. |
| Pose-jump rejection (≥30 cm in one inter-frame interval) | Catches discrete re-localisations. Anchor updates on rejection too, so we resync to the post-jump frame. |
| kd-tree drift rejection (median nearest-neighbour ≥10 cm) | Catches frames that don't overlap with existing surface at all. Doesn't fire on gradual drift because each frame *does* overlap (with ghost walls). |
| ICP step 1, translation only (median displacement) | Median correction on the user's data was 0.28 cm — too small to matter. The kd-tree picks the closest *ghost* wall, so the median displacement collapses toward zero. |
| Procrustes ICP, full rigid (R + t) with safety caps | Caps |t| ≤ 5 cm, |R| ≤ 2°. On replay, proposed corrections are sane (median 0.28° / 0.63 cm). But on already-corrupted grids, doesn't undo what's there. |

**Key realisation**: ICP is a *prevention* tool, not a *repair* tool.
On a fresh scan starting from empty grid, the early frames build a
clean baseline and ICP can keep subsequent frames anchored. On an
already-multi-layered grid, the kd-tree's "nearest existing voxel"
correspondences are ambiguous and ICP either does nothing or fits
spurious alignments to noise.

## Things to try next

In rough order of likely payoff vs. cost:

1. **Confirm ICP behaviour on a fresh scan.** The replay-on-old-data
   results don't tell us how ICP performs on a clean grid. Tap ↺,
   scan slowly, and watch the server log's `corr=t…cm/r…°` values.
   If they're consistently sub-cm and sub-degree and the resulting
   mesh is single-walled, we're in business.
2. **Multi-iteration ICP per frame.** A single Kabsch step is the
   first iteration; running 3–5 iterations on the same correspondences
   (re-querying the kd-tree each time) typically converges. Each
   iteration is fast (~10 ms).
3. **RANSAC-Procrustes.** Sample subsets of correspondences, find the
   transform with the most inliers. Robust against the "ghost wall"
   problem: only the subset that aligns to the *real* surface gets
   selected.
4. **Local-window ICP.** Instead of matching against the whole grid
   (where ghost walls appear), match against only the last N seconds
   of voxels. The recent voxels were laid down with similar drift to
   the current frame, so the local correspondence is cleaner.
5. **TSDF instead of occupancy.** A truncated signed-distance field
   averages multiple observations smoothly and rejects extreme
   outliers naturally. Bigger code change but standard for SLAM.
6. **Loop closure / pose-graph optimisation.** Detect when the user
   revisits an early-scanned area and globally optimise all poses to
   match. The full SLAM solution.
7. **Save raw frames and post-process.** `captured_frames/` already
   logs every body. We could implement an offline "rebuild" tool that
   does smarter alignment (multi-pass, robust, etc.) without realtime
   constraints. `tools/replay.py` is the start of that.

## File map

```
tools/
  serve.py            — stdlib HTTPS server. Endpoints:
                          POST /frame         ingest a binary frame
                          POST /save          write room.npz
                          POST /reset         wipe voxel grid + drift state
                          POST /remesh        write room.glb
                          GET  /stats         JSON status
                          POST /cubes/start   init coarse cube grid + start recording
                          POST /cubes/stop    stop cube ingestion
                          POST /cubes/reset   wipe cube counters
                          GET  /cubes/state   JSON list of populated cubes
                          POST /log           phone console mirror
  fusion.py           — depth + pose → world points (with colour samples)
  voxel_store.py      — VoxelRoom: chunked sparse RGBA grid + Procrustes ICP
  cubes.py            — CubeGrid: dense coarse-cube occupancy/free counters
  meshing.py          — dense marching cubes + GLB writer with vertex colours
  replay.py           — offline replay of captured_frames/ for debugging
  oakd_scan.py        — alternative scanner using a Luxonis OAK-D Lite over USB-C
  requirements.txt    — numpy, scikit-image, trimesh
  oakd_requirements.txt — depthai, opencv-python, scipy (only for oakd_scan.py)
web/
  index.html          — landing page
  scan.html / .js     — phone WebXR scanner with depth overlay + reset
  pose.html / .js     — phone pose-stability validator (tap-to-anchor, return-to-zero)
  cubes.html / .js    — phone cube scanner + live overlay (plain & depth-occluded)
  game.html / .js     — laptop fly-through viewer with SSAO + shadows
  logbridge.js        — mirrors phone console.log to /log
  styles.css          — shared styles
captured_frames/      — gitignored, holds raw /frame bodies for replay
web/out/              — gitignored, holds room.npz / room.glb
```

## Acknowledgements

Architecture lifted from
[`building-to-be`](https://github.com/lukacslacko/building-to-be) (sibling
WebXR project) — same `ThreadingHTTPServer` template, same
no-build-step three.js-via-CDN approach. That repo's
`tools/serve.py` and `web/ar-xr.js` were the starting point for this
one's server and scanner.
