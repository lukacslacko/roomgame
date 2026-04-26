# roomgame

Phone-scanned voxel room model, rendered as a fly-through on the laptop.
The plan is for it to eventually become a small racing/destruction game
inside the scanned room. Right now it's a working scanner + reconstruction
pipeline, with one unsolved issue (gradual ARCore pose drift).

## Architecture

```
[Android Chrome / WebXR depth + camera]
         │  binary frames (depth + RGBA + pose)
         ▼  HTTPS POST /frame, ~5–10 fps with backpressure
[Python serve.py]
   ├── parse_frame      (224-byte header + payload)
   ├── fusion.py        (depth pixel → world point)
   ├── voxel_store.py   (chunked uint8 RGBA grid + ICP)
   └── meshing.py       (dense marching cubes → GLB)
         │
         ▼  /out/room.glb
[Laptop browser, three.js]
   └── game.html        (fly cam, SSAO, vertex colours, shadows)
```

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
                          POST /frame  ingest a binary frame
                          POST /save   write room.npz
                          POST /reset  wipe grid + drift state
                          POST /remesh write room.glb
                          GET  /stats  JSON status
                          POST /log    phone console mirror
  fusion.py           — depth + pose → world points (with colour samples)
  voxel_store.py      — VoxelRoom: chunked sparse RGBA grid + Procrustes ICP
  meshing.py          — dense marching cubes + GLB writer with vertex colours
  replay.py           — offline replay of captured_frames/ for debugging
  requirements.txt    — numpy, scikit-image, trimesh
web/
  index.html          — landing page
  scan.html / .js     — phone WebXR scanner with depth overlay + reset
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
