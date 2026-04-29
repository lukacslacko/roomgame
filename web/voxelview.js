// Laptop-side viewer for the offline voxel reconstruction.
// Loads /out/voxels.json (written by tools/voxel_reconstruct.py) and renders
// every kept voxel as a real, lit, shadow-casting box with per-instance
// colour. WASD/arrows fly cam, lights, shadows, SSAO — same as cubeview.

import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { SSAOPass } from "three/addons/postprocessing/SSAOPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const stage = document.getElementById("stage");
const reloadBtn = document.getElementById("reloadBtn");
const recenterBtn = document.getElementById("recenterBtn");
const lightingBtn = document.getElementById("lightingBtn");
const voxMeshToggle = document.getElementById("voxMeshToggle");
const stVoxels = document.getElementById("stVoxels");
const stSize = document.getElementById("stSize");
const stMsg = document.getElementById("stMsg");
const sessionSel = document.getElementById("sessionSel");
const variantSel = document.getElementById("variantSel");
const panelToggleBtn = document.getElementById("panelToggleBtn");
const framePanel = document.getElementById("framePanel");
const framePanelList = document.getElementById("framePanelList");
const framePanelCount = document.getElementById("framePanelCount");
const panelClearBtn = document.getElementById("panelClearBtn");
const pixelCloudBtn = document.getElementById("pixelCloudBtn");
const pcToolbar = document.getElementById("pcToolbar");
const pcDepthKindSel = document.getElementById("pcDepthKindSel");
const pcPoseDirSel = document.getElementById("pcPoseDirSel");
const pcStrideSel = document.getElementById("pcStrideSel");
const pcASlider = document.getElementById("pcASlider");
const pcBSlider = document.getElementById("pcBSlider");
const pcAVal = document.getElementById("pcAVal");
const pcBVal = document.getElementById("pcBVal");
const pcFitSpaceSel = document.getElementById("pcFitSpaceSel");
const pcResetBtn = document.getElementById("pcResetBtn");
const pcAffineGroup = document.getElementById("pcAffineGroup");
const pcStatus = document.getElementById("pcStatus");
const pcAutotuneBtn = document.getElementById("pcAutotuneBtn");
const pcAutotuneVoxelBtn = document.getElementById("pcAutotuneVoxelBtn");
const pcAutotuneChamferBtn = document.getElementById("pcAutotuneChamferBtn");
const pcFeaturesToggle = document.getElementById("pcFeaturesToggle");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101015);

const camera = new THREE.PerspectiveCamera(60, 1, 0.05, 200);
camera.rotation.order = "YXZ";
camera.position.set(0, 1.6, 4);

const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
stage.appendChild(renderer.domElement);

const ambient = new THREE.AmbientLight(0xffffff, 0.30);
scene.add(ambient);
const sun = new THREE.DirectionalLight(0xfff1d6, 1.4);
sun.position.set(2.5, 5.0, 2.0);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.near = 0.1;
sun.shadow.camera.far = 30;
sun.shadow.camera.left = -7;
sun.shadow.camera.right = 7;
sun.shadow.camera.top = 7;
sun.shadow.camera.bottom = -7;
sun.shadow.bias = -0.0005;
sun.shadow.normalBias = 0.01;
scene.add(sun);
const hemi = new THREE.HemisphereLight(0xc6d8ff, 0x202028, 0.45);
scene.add(hemi);

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(20, 20),
  new THREE.MeshStandardMaterial({ color: 0x1a1a22, roughness: 0.95, metalness: 0 }),
);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

const grid = new THREE.GridHelper(10, 20, 0x444466, 0x222234);
grid.position.y = 0.001;
scene.add(grid);
scene.add(new THREE.AxesHelper(0.5));

const composer = new EffectComposer(renderer);
composer.renderTarget1.samples = 4;
composer.renderTarget2.samples = 4;
composer.addPass(new RenderPass(scene, camera));
const ssao = new SSAOPass(scene, camera, 0, 0);
ssao.kernelRadius = 0.08;
ssao.minDistance = 0.0008;
ssao.maxDistance = 0.05;
composer.addPass(ssao);
composer.addPass(new OutputPass());

const voxMaterial = new THREE.MeshStandardMaterial({
  color: 0xffffff,
  metalness: 0.0,
  roughness: 0.85,
});
const voxGeometry = new THREE.BoxGeometry(1, 1, 1);

let voxMesh = null;
let lastBBox = null;

function rebuildMesh(payload) {
  // Stash for the frame-debug overlays so they know the voxel grid params
  // (world_min, voxel_size, shape) to use when projecting the selected
  // frames into voxel space.
  lastVoxelPayload = payload;
  if (voxMesh) {
    scene.remove(voxMesh);
    voxMesh.dispose();
    voxMesh = null;
  }
  const indices = payload.indices;
  const colors = payload.colors;
  const wm = payload.world_min;
  const sz = payload.voxel_size;
  if (!indices || indices.length === 0) {
    lastBBox = null;
    return;
  }
  const m = new THREE.InstancedMesh(voxGeometry, voxMaterial, indices.length);
  m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  m.castShadow = true;
  m.receiveShadow = true;
  // InstancedMesh's default frustum cull uses the *base geometry's* bounding
  // sphere (the unit cube ⇒ radius ≈ 0.87 at origin), which doesn't reflect
  // where the instance matrices actually place each voxel. With dense variants
  // the room geometry happens to keep that sphere in view, so it doesn't bite;
  // with the sparse `features` variant the camera often looks at voxels far
  // from origin while the sphere is off-screen, causing the entire mesh to be
  // culled and big chunks of voxels to disappear. Disable per-mesh culling and
  // let the GPU draw — the instance count stays bounded by the room.
  m.frustumCulled = false;

  const tmpMat = new THREE.Matrix4();
  const tmpQuat = new THREE.Quaternion();
  const tmpScale = new THREE.Vector3(sz, sz, sz);
  const tmpPos = new THREE.Vector3();
  const tmpCol = new THREE.Color();
  const bbox = new THREE.Box3();

  for (let k = 0; k < indices.length; k++) {
    const [ix, iy, iz] = indices[k];
    const [r, g, b] = colors[k];
    tmpPos.set(
      wm[0] + (ix + 0.5) * sz,
      wm[1] + (iy + 0.5) * sz,
      wm[2] + (iz + 0.5) * sz,
    );
    tmpMat.compose(tmpPos, tmpQuat, tmpScale);
    m.setMatrixAt(k, tmpMat);
    // setColorAt expects linear values when outputColorSpace is sRGB but the
    // captured RGB is sRGB-encoded camera bytes. SRGBToLinear keeps the
    // displayed colour consistent with what the camera actually saw.
    tmpCol.setRGB(r / 255, g / 255, b / 255).convertSRGBToLinear();
    m.setColorAt(k, tmpCol);
    bbox.expandByPoint(tmpPos);
  }
  m.instanceMatrix.needsUpdate = true;
  if (m.instanceColor) m.instanceColor.needsUpdate = true;
  // Stash the index list so the raycast click handler can map an
  // intersection.instanceId back to the (ix, iy, iz) voxel triple.
  m.userData.voxelIndices = indices;
  // Apply the toolbar checkbox up front so a reload that lands while
  // the user has the voxel mesh hidden doesn't snap it back into view.
  m.visible = !voxMeshToggle || voxMeshToggle.checked;
  scene.add(m);
  voxMesh = m;
  lastBBox = bbox.isEmpty() ? null : bbox;
  // Newly-built mesh defaults to lit settings; re-apply current mode so a
  // reload while in flat mode doesn't snap the cubes back to lit.
  applyLightingMode();
}

// Voxel-mesh visibility: hide the main reconstruction so the user can
// inspect just the per-frame pixel-cloud overlays. Default ON; toggling
// only affects the main mesh, not the per-frame frustum/points overlays.
if (voxMeshToggle) {
  voxMeshToggle.addEventListener("change", () => {
    if (voxMesh) voxMesh.visible = voxMeshToggle.checked;
  });
}

// State: which session/variant the dropdowns are pointing at, plus the
// last fetched /sessions snapshot so we can recompute "available variants
// for current session" without hitting the network on every dropdown change.
let availableSessions = [];

function urlFor(sessionId, variant) {
  if (sessionId) {
    return `/captures/${encodeURIComponent(sessionId)}/voxels_${encodeURIComponent(variant)}.json?t=${Date.now()}`;
  }
  // Legacy fallback: old /out/voxels.json written by the standalone tools.
  return `/out/voxels.json?t=${Date.now()}`;
}

async function refreshSessionList() {
  try {
    const r = await fetch("/sessions");
    if (!r.ok) {
      console.warn(`/sessions HTTP ${r.status}`);
      return;
    }
    const j = await r.json();
    availableSessions = j.sessions || [];
  } catch (e) {
    console.warn("/sessions fetch failed", e);
    availableSessions = [];
  }
  renderSessionDropdown();
}

function renderSessionDropdown() {
  // Newest sessions first; only sessions with at least one voxel JSON are
  // worth selecting, but we still show the rest so the user can spot a
  // session they haven't run the voxeliser on yet.
  const usable = availableSessions
    .filter((s) => (s.variants || []).length > 0)
    .sort((a, b) => (a.id < b.id ? 1 : -1));
  const stale = availableSessions
    .filter((s) => (s.variants || []).length === 0)
    .sort((a, b) => (a.id < b.id ? 1 : -1));

  const previous = sessionSel.value;
  sessionSel.innerHTML = "";
  if (usable.length === 0 && stale.length === 0) {
    const opt = document.createElement("option");
    opt.value = ""; opt.textContent = "(no sessions yet)";
    sessionSel.appendChild(opt);
  } else {
    for (const s of usable) {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = `${s.id}  (${s.variants.join(", ")})`;
      sessionSel.appendChild(opt);
    }
    for (const s of stale) {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = `${s.id}  (no voxels yet)`;
      opt.disabled = true;
      sessionSel.appendChild(opt);
    }
  }

  // Pick previous if still present, otherwise the newest usable session.
  let pick = previous && [...sessionSel.options].some((o) => o.value === previous)
    ? previous
    : (usable[0]?.id || "");
  sessionSel.value = pick;
  syncVariantOptions();
}

// Variant naming convention: any captured_frames/<id>/voxels_<name>.json on
// disk shows up here. We sort the dropdown by a hand-tuned "quality"
// preference so the most processed variant lands at the top, then fall
// back alphabetically. Default selection on a new session goes to the
// top entry; the user's previous choice is preserved if still valid.
const VARIANT_ORDER = ["refined_aligned", "aligned", "refined", "original"];
function _variantSortKey(v) {
  const i = VARIANT_ORDER.indexOf(v);
  return i >= 0 ? [0, i, v] : [1, 0, v];
}

function syncVariantOptions() {
  const sid = sessionSel.value;
  const sess = availableSessions.find((s) => s.id === sid);
  const variants = (sess ? sess.variants : []).slice().sort((a, b) => {
    const ka = _variantSortKey(a), kb = _variantSortKey(b);
    if (ka[0] !== kb[0]) return ka[0] - kb[0];
    if (ka[1] !== kb[1]) return ka[1] - kb[1];
    return ka[2] < kb[2] ? -1 : 1;
  });
  const previous = variantSel.value;
  variantSel.innerHTML = "";
  if (variants.length === 0) {
    const opt = document.createElement("option");
    opt.value = ""; opt.textContent = "(no variants)";
    variantSel.appendChild(opt);
    return;
  }
  for (const v of variants) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    variantSel.appendChild(opt);
  }
  if (previous && variants.includes(previous)) {
    variantSel.value = previous;
  } else {
    variantSel.value = variants[0];   // best-ranked variant
  }
}

// Camera is only repositioned on the very first successful load; subsequent
// reloads (variant switch, session switch, Reload button) keep the user's
// current viewpoint so A/B-ing between variants doesn't snap them away from
// the spot they were inspecting. The shadow bounds are still refreshed
// every time via recenter(false).
let hasLoadedOnce = false;

async function loadVoxels() {
  reloadBtn.disabled = true;
  stMsg.textContent = "loading…";
  const sid = sessionSel.value || "";
  const variant = variantSel.value || "refined";
  try {
    const r = await fetch(urlFor(sid, variant));
    if (!r.ok) {
      stMsg.textContent = sid
        ? `${sid}/${variant}: HTTP ${r.status}`
        : `voxels.json: HTTP ${r.status}`;
      stVoxels.textContent = "—";
      return;
    }
    const payload = await r.json();
    rebuildMesh(payload);
    stVoxels.textContent = (payload.n_voxels || 0).toLocaleString();
    stSize.textContent = `${payload.voxel_size?.toFixed?.(2) ?? "?"} m`;
    const tag = sid ? `${sid} · ${variant}` : "legacy /out/voxels.json";
    stMsg.textContent = `bbox ${payload.shape?.join("×") ?? "?"} · ${tag}`;
    recenter(!hasLoadedOnce);
    hasLoadedOnce = true;
  } catch (e) {
    stMsg.textContent = `error: ${e.message || e}`;
    console.error(e);
  } finally {
    reloadBtn.disabled = false;
  }
}

function recenter(adjustCamera) {
  if (!lastBBox) return;
  const c = lastBBox.getCenter(new THREE.Vector3());
  const sz = lastBBox.getSize(new THREE.Vector3());
  if (adjustCamera) {
    camera.position.set(c.x, Math.max(c.y + sz.y * 0.3, 1.6), c.z + Math.max(2.5, sz.z * 0.9));
    yaw = 0; pitch = 0;
    camera.rotation.set(0, 0, 0);
  }
  const r = Math.max(5, sz.length() * 0.6);
  sun.shadow.camera.left = -r;
  sun.shadow.camera.right = r;
  sun.shadow.camera.top = r;
  sun.shadow.camera.bottom = -r;
  sun.shadow.camera.updateProjectionMatrix();
}

reloadBtn.addEventListener("click", () => refreshSessionList().then(loadVoxels));
recenterBtn.addEventListener("click", () => recenter(true));
sessionSel.addEventListener("change", () => { syncVariantOptions(); loadVoxels(); });
variantSel.addEventListener("change", () => loadVoxels());

// "Flat" mode: kill the directional sun + hemisphere + SSAO and crank the
// ambient up to 1.0 so each voxel renders as its captured colour without
// any shading. Useful when uneven scan coverage (e.g. half-captured ceiling)
// throws the lit version into harsh contrasts.
let flatMode = true;
function applyLightingMode() {
  if (flatMode) {
    ambient.intensity = 1.0;
    sun.intensity = 0.0;
    sun.castShadow = false;
    hemi.intensity = 0.0;
    floor.visible = false;
    if (voxMesh) {
      voxMesh.castShadow = false;
      voxMesh.receiveShadow = false;
    }
    ssao.enabled = false;
    renderer.toneMapping = THREE.NoToneMapping;
    lightingBtn.textContent = "Flat";
  } else {
    ambient.intensity = 0.30;
    sun.intensity = 1.4;
    sun.castShadow = true;
    hemi.intensity = 0.45;
    floor.visible = true;
    if (voxMesh) {
      voxMesh.castShadow = true;
      voxMesh.receiveShadow = true;
    }
    ssao.enabled = true;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    lightingBtn.textContent = "Lit";
  }
}
lightingBtn.addEventListener("click", () => {
  flatMode = !flatMode;
  applyLightingMode();
});
applyLightingMode();   // sync renderer/lighting/button label with the flatMode default

function onResize() {
  const w = stage.clientWidth || 1;
  const h = stage.clientHeight || 1;
  renderer.setSize(w, h, false);
  composer.setSize(w, h);
  ssao.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", onResize);
new ResizeObserver(onResize).observe(stage);
onResize();

// ----- fly cam (lifted from cubeview.js) ----------------------------------
const keys = new Set();
const movementCodes = new Set([
  "KeyW", "KeyA", "KeyS", "KeyD",
  "Space", "ShiftLeft", "ShiftRight",
  "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
]);
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  keys.add(e.code);
  if (movementCodes.has(e.code)) e.preventDefault();
});
window.addEventListener("keyup", (e) => { keys.delete(e.code); });
window.addEventListener("blur", () => keys.clear());

let yaw = 0;
let pitch = 0;
const PITCH_LIMIT = Math.PI / 2 - 0.05;

let dragging = false;
let lastMouse = { x: 0, y: 0 };
// Clicking into the 3D viewer must unfocus any active slider/dropdown so
// the keydown handler (which early-returns when the focus is on an input)
// starts forwarding WASD/arrows to the flycam instead of nudging the
// pixel-cloud sliders. Use pointerdown so it fires on every mouse button
// + touch, not just left-click.
renderer.domElement.addEventListener("pointerdown", () => {
  const a = document.activeElement;
  if (a && a !== document.body && a.blur) a.blur();
});
renderer.domElement.addEventListener("mousedown", (e) => {
  if (e.button !== 0) return;
  dragging = true;
  lastMouse.x = e.clientX;
  lastMouse.y = e.clientY;
  e.preventDefault();
});
window.addEventListener("mouseup", () => { dragging = false; });
window.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  const dx = e.clientX - lastMouse.x;
  const dy = e.clientY - lastMouse.y;
  lastMouse.x = e.clientX;
  lastMouse.y = e.clientY;
  yaw -= dx * 0.0035;
  pitch -= dy * 0.0035;
  if (pitch >  PITCH_LIMIT) pitch =  PITCH_LIMIT;
  if (pitch < -PITCH_LIMIT) pitch = -PITCH_LIMIT;
});
renderer.domElement.addEventListener("wheel", (e) => {
  const dir = new THREE.Vector3();
  camera.getWorldDirection(dir);
  const step = (e.deltaY > 0 ? -1 : 1) * 0.25;
  camera.position.addScaledVector(dir, step);
  e.preventDefault();
}, { passive: false });

const BASE_SPEED_M_PER_S = 1.6;
const SPRINT_MULT = 3.0;
const LOOK_RATE_RAD_PER_S = Math.PI / 2;
let lastTime = performance.now();

const _fwdH = new THREE.Vector3();
const _rightH = new THREE.Vector3();
const _move = new THREE.Vector3();

function updateCamera(dtSec) {
  if (keys.has("ArrowLeft"))  yaw   += LOOK_RATE_RAD_PER_S * dtSec;
  if (keys.has("ArrowRight")) yaw   -= LOOK_RATE_RAD_PER_S * dtSec;
  if (keys.has("ArrowUp"))    pitch += LOOK_RATE_RAD_PER_S * dtSec;
  if (keys.has("ArrowDown"))  pitch -= LOOK_RATE_RAD_PER_S * dtSec;
  if (pitch >  PITCH_LIMIT) pitch =  PITCH_LIMIT;
  if (pitch < -PITCH_LIMIT) pitch = -PITCH_LIMIT;

  camera.rotation.set(pitch, yaw, 0, "YXZ");
  _fwdH.set(-Math.sin(yaw), 0, -Math.cos(yaw));
  _rightH.set(Math.cos(yaw), 0, -Math.sin(yaw));

  _move.set(0, 0, 0);
  if (keys.has("KeyW")) _move.add(_fwdH);
  if (keys.has("KeyS")) _move.sub(_fwdH);
  if (keys.has("KeyD")) _move.add(_rightH);
  if (keys.has("KeyA")) _move.sub(_rightH);
  if (keys.has("Space")) _move.y += 1;
  if (keys.has("ShiftLeft") || keys.has("ShiftRight")) _move.y -= 1;

  if (_move.lengthSq() > 0) {
    _move.normalize();
    const sprint = (keys.has("ShiftLeft") || keys.has("ShiftRight"))
                   && (keys.has("KeyW") || keys.has("KeyA") || keys.has("KeyS") || keys.has("KeyD"));
    const speed = BASE_SPEED_M_PER_S * (sprint ? SPRINT_MULT : 1.0);
    camera.position.addScaledVector(_move, speed * dtSec);
  }
}

renderer.setAnimationLoop(() => {
  const now = performance.now();
  const dtSec = Math.min(0.1, (now - lastTime) / 1000);
  lastTime = now;
  updateCamera(dtSec);
  composer.render();
});

// =====================================================================
// Frame-debug panel: thumbnails on the left, multi-select, frustum + voxel
// highlight overlays in the 3D scene.
// =====================================================================
//
// State: which frames are checked, the manifest cached for the current
// session, and a map of frame_idx → THREE objects added to the scene
// (one entry per selected frame, removed on deselect).
const selectedFrames = new Set();
let frameManifest = null;
let lastVoxelPayload = null;
const frameOverlays = new Map();   // idx → { frustum: LineSegments, highlight: InstancedMesh }
let panelOpen = false;

// Map a UI variant ("refined_aligned", etc.) to the corresponding frames-dir
// name used by the server's frame-debug endpoints. The colour thumbnails
// always come from `frames/` (colour is identical across variants); only
// depth + voxel-projection care about the variant.
function variantToFramesDir(v) {
  switch (v) {
    case "original":         return "frames";
    case "refined":          return "frames_refined";
    case "aligned":          return "frames_aligned";
    case "refined_aligned":  return "frames_refined_aligned";
    case "refined_mv":       return "frames_refined_mv";
    default:                 return "frames";
  }
}

function colorForFrame(idx) {
  // Golden-ratio hue spread keeps adjacent indices visually distinct.
  const hue = ((idx * 0.6180339887) % 1.0);
  const c = new THREE.Color();
  c.setHSL(hue, 0.7, 0.6);
  return c;
}

// Pick the on-disk frames_*/ directory whose depth payload was the
// input to the currently-selected voxel variant. We can't always tell
// which one the rust voxeliser actually used (filename collisions
// between the legacy depth-ICP path and the new BA path), so prefer
// the BA-aligned sibling when present and fall back along the chain.
function pickDepthDir(variant, dirsAvail) {
  const want = (variant || "").toLowerCase();
  const has = (d) => dirsAvail.has(d);
  // Refined variants → look for a frames_refined*/ source.
  if (want.includes("refined") || want === "refined_aligned") {
    if (has("frames_refined_feature_ba")) return "frames_refined_feature_ba";
    if (has("frames_refined_aligned"))    return "frames_refined_aligned";
    if (has("frames_refined_mv"))         return "frames_refined_mv";
    if (has("frames_refined"))            return "frames_refined";
    // No refined depth on disk; fall through to phone depth.
  }
  // Everything else (original, aligned, features*) shows raw phone
  // depth. The pose-corrected siblings (frames_aligned, frames_feature_ba)
  // carry byte-identical depth, so we prefer the simpler `frames/` source.
  if (has("frames")) return "frames";
  if (has("frames_feature_ba")) return "frames_feature_ba";
  return "frames";
}

async function refreshFrameManifest() {
  const sid = sessionSel.value;
  if (!sid) { frameManifest = null; renderFramePanel(); return; }
  try {
    const r = await fetch(
      `/captures/${encodeURIComponent(sid)}/frame-manifest?variant=frames`,
    );
    if (!r.ok) {
      console.warn(`frame-manifest HTTP ${r.status}`);
      frameManifest = null;
    } else {
      frameManifest = await r.json();
    }
  } catch (e) {
    console.warn("frame-manifest fetch failed", e);
    frameManifest = null;
  }
  renderFramePanel();
}

function renderFramePanel() {
  if (!framePanel) return;
  framePanelList.innerHTML = "";
  const frames = frameManifest?.frames ?? [];
  framePanelCount.textContent = `${frames.length} frames · ${selectedFrames.size} selected`;
  if (!frames.length) {
    const empty = document.createElement("div");
    empty.style.padding = "0.6rem";
    empty.style.opacity = "0.55";
    empty.textContent = "(no frames in this session)";
    framePanelList.appendChild(empty);
    return;
  }
  const sid = sessionSel.value;
  const sessInfo = availableSessions.find((s) => s.id === sid);
  // Pick the depth-thumbnail source so it matches the depth that
  // actually went into the voxel variant the user is looking at —
  // otherwise refined-aligned voxels in 3D look nothing like the
  // (blurry, lowres, phone-WebXR) depth thumbnail next to each frame.
  // Mapping: `refined*` variants want model depth from a
  // `frames_refined*` dir (preferring the BA-aligned `_feature_ba`
  // sibling when present), everything else wants the raw phone depth
  // payload that lives in `frames/` (and is byte-identical in the
  // pose-only siblings like `frames_feature_ba/`).
  const dirsAvail = new Set(sessInfo?.frame_dirs ?? []);
  const depthDir = pickDepthDir(variantSel.value || "", dirsAvail);
  const depthLabel = depthDir.startsWith("frames_refined")
    ? "model depth" : "phone depth";
  // Render in batches to keep the DOM responsive on long sessions.
  const frag = document.createDocumentFragment();
  for (const f of frames) {
    const item = document.createElement("div");
    item.className = "frame-item";
    item.dataset.idx = String(f.idx);
    if (selectedFrames.has(f.idx)) item.classList.add("selected");
    const c = colorForFrame(f.idx);
    item.style.setProperty("--swatch-color",
                           `rgb(${(c.r*255)|0},${(c.g*255)|0},${(c.b*255)|0})`);

    // Grid layout: [swatch] [info column] [colour thumb] [depth thumb]
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    item.appendChild(swatch);

    const info = document.createElement("div");
    info.className = "info";
    const idx = document.createElement("div");
    idx.className = "idx";
    idx.textContent = `#${f.idx}`;
    info.appendChild(idx);
    const pose = document.createElement("div");
    pose.className = "pose";
    pose.textContent = `(${f.pose[0].toFixed(2)},\n${f.pose[1].toFixed(2)},\n${f.pose[2].toFixed(2)})`;
    pose.style.whiteSpace = "pre";
    info.appendChild(pose);
    const labelRgb = document.createElement("div");
    labelRgb.className = "label";
    labelRgb.textContent = `rgb · ${depthLabel}`;
    info.appendChild(labelRgb);
    item.appendChild(info);

    const colorWrap = document.createElement("div");
    colorWrap.className = "thumb-wrap";
    colorWrap.dataset.kind = "color";
    const colorImg = document.createElement("img");
    colorImg.loading = "lazy";
    colorImg.alt = `rgb #${f.idx}`;
    colorImg.src = `/captures/${encodeURIComponent(sid)}/frame-thumb/${f.idx}.png?variant=frames&kind=color`;
    colorWrap.appendChild(colorImg);
    item.appendChild(colorWrap);

    const depthWrap = document.createElement("div");
    depthWrap.className = "thumb-wrap";
    depthWrap.dataset.kind = "depth";
    const depthImg = document.createElement("img");
    depthImg.loading = "lazy";
    depthImg.alt = `depth #${f.idx}`;
    depthImg.src = `/captures/${encodeURIComponent(sid)}/frame-thumb/${f.idx}.png?variant=${encodeURIComponent(depthDir)}&kind=depth`;
    depthWrap.appendChild(depthImg);
    item.appendChild(depthWrap);

    // Stash the colour-buffer pixel size so the SVG-marker overlay can use
    // a viewBox of (cw × ch); preserveAspectRatio="xMidYMid meet" then
    // letterboxes identically to the img's `object-fit: contain`.
    item.dataset.cw = String(f.color?.[0] ?? 0);
    item.dataset.ch = String(f.color?.[1] ?? 0);

    // Re-apply markers (if any) for this frame after creating the item.
    applyKeypointMarkersToItem(item, f.idx);

    item.addEventListener("click", () => toggleFrameSelection(f.idx, item));
    frag.appendChild(item);
  }
  framePanelList.appendChild(frag);
}

async function toggleFrameSelection(idx, itemEl) {
  if (selectedFrames.has(idx)) {
    selectedFrames.delete(idx);
    itemEl?.classList.remove("selected");
    removeFrameOverlay(idx);
  } else {
    selectedFrames.add(idx);
    itemEl?.classList.add("selected");
    await addFrameOverlay(idx);
  }
  framePanelCount.textContent = `${frameManifest?.frames?.length ?? 0} frames · ${selectedFrames.size} selected`;
  // In pixel-cloud mode the set of common-features depends on which
  // frames are selected, so refresh the markers + 3D sphere overlay
  // (debounced inside the helper to coalesce rapid clicks).
  if (pcMode && typeof scheduleCommonFeaturesRefresh === "function") {
    scheduleCommonFeaturesRefresh();
  }
}

async function addFrameOverlay(idx) {
  // Pixel-cloud mode replaces the voxel-highlight cubes with a per-pixel
  // THREE.Points cloud — same frustum wireframe, different content. The
  // dispatch happens here so existing call sites (toggleFrameSelection,
  // selectVoxel) reach either renderer.
  if (pcMode) return addPixelCloudOverlay(idx);
  const sid = sessionSel.value;
  const variant = variantSel.value || "original";
  if (!sid || !lastVoxelPayload) return;
  const wm = lastVoxelPayload.world_min;
  const sz = lastVoxelPayload.voxel_size;
  const sh = lastVoxelPayload.shape;
  // Any `features*` variant comes from feature_ray_reconstruct.py — its
  // per-frame voxels are not derived from depth backprojection but from
  // which feature tracks were observed in this frame, so we hit a
  // dedicated endpoint instead of `frame-voxels`. The query param picks
  // which features_meta sidecar (raw poses vs ICP-aligned poses).
  let url;
  if (isFeaturesVariant(variant)) {
    url = `/captures/${encodeURIComponent(sid)}/frame-feature-voxels/${idx}.json`
        + `?variant=${encodeURIComponent(variant)}`;
  } else {
    const variantDir = variantToFramesDir(variant);
    url = `/captures/${encodeURIComponent(sid)}/frame-voxels/${idx}.json`
        + `?variant=${encodeURIComponent(variantDir)}`
        + `&voxel_size=${sz}`
        + `&world_min=${wm.join(",")}`
        + `&shape=${sh.join(",")}`;
  }
  let payload;
  try {
    const r = await fetch(url);
    if (!r.ok) {
      console.warn(`frame-voxels HTTP ${r.status} for #${idx}`);
      return;
    }
    payload = await r.json();
  } catch (e) {
    console.warn("frame-voxels fetch failed", e);
    return;
  }
  const colour = colorForFrame(idx);
  // The features endpoint may return an empty pose/frustum if the original
  // frame file isn't on disk anymore; skip the wireframe in that case.
  let frustum = null;
  if (Array.isArray(payload.frustum_world) && payload.frustum_world.length === 8
      && Array.isArray(payload.pose) && payload.pose.length === 3) {
    frustum = buildFrustumLines(payload.frustum_world, payload.pose, colour);
    // Tag with the frame id so the click handler can identify which frame
    // this wireframe belongs to and jump the flycam to that pose.
    frustum.userData.frameIdx = idx;
    scene.add(frustum);
  }
  const highlight = buildHighlightMesh(payload.indices, colour, sz, wm);
  if (highlight) scene.add(highlight);
  frameOverlays.set(idx, { frustum, highlight });
}

function removeFrameOverlay(idx) {
  const o = frameOverlays.get(idx);
  if (!o) return;
  if (o.frustum) {
    scene.remove(o.frustum);
    o.frustum.geometry.dispose();
    o.frustum.material.dispose();
  }
  if (o.highlight) {
    scene.remove(o.highlight);
    o.highlight.geometry.dispose();
    o.highlight.material.dispose();
  }
  if (o.points) {
    scene.remove(o.points);
    o.points.geometry.dispose();
    o.points.material.dispose();
  }
  frameOverlays.delete(idx);
}

function clearAllFrameOverlays() {
  for (const idx of [...frameOverlays.keys()]) removeFrameOverlay(idx);
  selectedFrames.clear();
  for (const el of framePanelList.querySelectorAll(".frame-item.selected")) {
    el.classList.remove("selected");
  }
  // Active-voxel highlight + per-thumb keypoint markers are tied to the
  // frame selection's "voxel-driven" mode; clearing one clears the other.
  if (typeof clearActiveVoxel === "function") clearActiveVoxel();
  framePanelCount.textContent = `${frameManifest?.frames?.length ?? 0} frames · 0 selected`;
}

function buildFrustumLines(corners, pose, colour) {
  // corners = 8 entries: [0..3] near plane, [4..7] far plane (in CCW order
  // around the image). pose = camera origin in world. We connect:
  //   - 4 edges of the near rectangle
  //   - 4 edges of the far rectangle
  //   - 4 edges from each near corner to the matching far corner
  //   - 4 apex lines from cam origin to far corners (pyramid)
  const positions = [];
  const push = (a, b) => positions.push(a[0], a[1], a[2], b[0], b[1], b[2]);
  for (const [a, b] of [[0,1],[1,2],[2,3],[3,0]]) push(corners[a], corners[b]);
  for (const [a, b] of [[4,5],[5,6],[6,7],[7,4]]) push(corners[a], corners[b]);
  for (let i = 0; i < 4; i++) push(corners[i], corners[i+4]);
  for (let i = 4; i < 8; i++) push(pose, corners[i]);
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const mat = new THREE.LineBasicMaterial({
    color: colour,
    transparent: true,
    opacity: 0.85,
    depthTest: true,
    depthWrite: false,
  });
  return new THREE.LineSegments(geom, mat);
}

const _highlightGeom = new THREE.BoxGeometry(1, 1, 1);
function buildHighlightMesh(triples, colour, voxelSize, worldMin) {
  if (!triples || triples.length === 0) return null;
  // Slightly larger than the original voxels so the highlight rim peeks out;
  // additive transparency keeps occluded voxels visible behind the surface.
  const mat = new THREE.MeshBasicMaterial({
    color: colour,
    transparent: true,
    opacity: 0.30,
    depthWrite: false,
  });
  const mesh = new THREE.InstancedMesh(_highlightGeom, mat, triples.length);
  mesh.frustumCulled = false;   // see voxMesh comment — same culling caveat
  const tmpMat = new THREE.Matrix4();
  const tmpQuat = new THREE.Quaternion();
  const tmpScale = new THREE.Vector3(voxelSize * 1.04, voxelSize * 1.04, voxelSize * 1.04);
  const tmpPos = new THREE.Vector3();
  for (let k = 0; k < triples.length; k++) {
    const [ix, iy, iz] = triples[k];
    tmpPos.set(
      worldMin[0] + (ix + 0.5) * voxelSize,
      worldMin[1] + (iy + 0.5) * voxelSize,
      worldMin[2] + (iz + 0.5) * voxelSize,
    );
    tmpMat.compose(tmpPos, tmpQuat, tmpScale);
    mesh.setMatrixAt(k, tmpMat);
  }
  mesh.instanceMatrix.needsUpdate = true;
  mesh.renderOrder = 5;
  return mesh;
}

function setPanelOpen(open) {
  panelOpen = open;
  framePanel.hidden = !open;
  panelToggleBtn.textContent = open ? "Frames ◂" : "Frames ▸";
  if (open && !frameManifest) refreshFrameManifest();
  // Three.js renderer needs to re-fit when the stage's width changes.
  if (renderer) {
    renderer.setSize(stage.clientWidth, stage.clientHeight);
    composer?.setSize(stage.clientWidth, stage.clientHeight);
    camera.aspect = stage.clientWidth / Math.max(1, stage.clientHeight);
    camera.updateProjectionMatrix();
  }
}

panelToggleBtn.addEventListener("click", () => setPanelOpen(!panelOpen));
panelClearBtn.addEventListener("click", clearAllFrameOverlays);

// Re-fetch the manifest whenever the session changes; clear overlays whenever
// session OR variant changes (the voxel grid params and depth source change
// underneath the existing overlays).
sessionSel.addEventListener("change", () => {
  clearAllFrameOverlays();
  clearActiveVoxel();
  featuresMeta = null;
  featuresLookup = null;
  featuresMetaVariant = null;
  refreshFrameManifest();
  if (isFeaturesVariant(variantSel.value)) prefetchFeaturesMeta();
});
variantSel.addEventListener("change", () => {
  clearAllFrameOverlays();
  clearActiveVoxel();
  // Different features variants point at different meta sidecars; drop
  // the cache so the next click pulls the right one.
  if (featuresMetaVariant !== variantSel.value) {
    featuresMeta = null;
    featuresLookup = null;
    featuresMetaVariant = null;
  }
  if (isFeaturesVariant(variantSel.value)) prefetchFeaturesMeta();
  // Depth-thumbnail source depends on the active variant — re-render
  // the panel so model-vs-phone depth shows up correctly when the user
  // flips between e.g. `aligned` and `refined_aligned`.
  if (frameManifest) renderFramePanel();
});

// =====================================================================
// Voxel-click → feature lookup → frame selection + per-thumb keypoint marker.
// Only meaningful for the "features" variant; harmless otherwise.
// =====================================================================

let featuresMeta = null;       // raw features_meta.json payload for the session
let featuresLookup = null;     // Map<"ix,iy,iz", voxelMetaEntry>
let activeVoxelKey = null;     // currently-clicked "ix,iy,iz", or null
let activeVoxelMesh = null;    // outline mesh highlighting the clicked voxel
let frameMarkers = new Map();  // frame_idx → [{u, v}, ...] (drives SVG overlays)

// Any voxelview variant emitted by feature_ray_reconstruct.py — the base
// `features` plus optional suffixes (e.g. `features_aligned` from
// --frames-variant frames_aligned). The dropdown auto-discovers them
// from voxels_<v>.json on disk; the JS just needs to know which subset
// to route through the per-feature endpoint.
function isFeaturesVariant(v) {
  return v === "features" || v.startsWith("features_");
}

let featuresMetaVariant = null;  // which variant the cached meta belongs to

async function prefetchFeaturesMeta() {
  const sid = sessionSel.value;
  const variant = variantSel.value || "features";
  if (!sid) return;
  if (!isFeaturesVariant(variant)) return;
  if (featuresMeta && featuresLookup && featuresMetaVariant === variant) return;
  try {
    const r = await fetch(
      `/captures/${encodeURIComponent(sid)}/features_meta.json`
      + `?variant=${encodeURIComponent(variant)}`
      + `&t=${Date.now()}`,
    );
    if (!r.ok) {
      featuresMeta = null; featuresLookup = null; featuresMetaVariant = null;
      console.warn(`features_meta HTTP ${r.status}`);
      return;
    }
    featuresMeta = await r.json();
    featuresMetaVariant = variant;
  } catch (e) {
    featuresMeta = null; featuresLookup = null; featuresMetaVariant = null;
    console.warn("features_meta fetch failed", e);
    return;
  }
  featuresLookup = new Map();
  for (const v of (featuresMeta.voxels || [])) {
    const [ix, iy, iz] = v.idx;
    featuresLookup.set(`${ix},${iy},${iz}`, v);
  }
}

// Distinguish a click from a drag-rotate: only treat mouseup as a click
// if the pointer barely moved between mousedown and mouseup.
let downAt = 0;
let downX = 0, downY = 0;
const CLICK_DRAG_PX = 4;
const CLICK_MAX_MS = 350;
renderer.domElement.addEventListener("mousedown", (e) => {
  if (e.button !== 0) return;
  downAt = performance.now();
  downX = e.clientX; downY = e.clientY;
});
renderer.domElement.addEventListener("mouseup", (e) => {
  if (e.button !== 0) return;
  const dt = performance.now() - downAt;
  const dx = e.clientX - downX, dy = e.clientY - downY;
  if (dt > CLICK_MAX_MS) return;
  if (dx*dx + dy*dy > CLICK_DRAG_PX * CLICK_DRAG_PX) return;
  handleSceneClick(e);
});

const _raycaster = new THREE.Raycaster();
// THREE's default Line raycast threshold is 1 world unit — way too loose
// at our 0.02 m voxel scale. 0.04 m gives the user a forgiving click
// target on the pyramid wireframes without bleeding into nearby ones.
_raycaster.params.Line = _raycaster.params.Line || {};
_raycaster.params.Line.threshold = 0.04;
const _ndc = new THREE.Vector2();

async function handleSceneClick(ev) {
  const rect = renderer.domElement.getBoundingClientRect();
  _ndc.x =  ((ev.clientX - rect.left) / rect.width)  * 2 - 1;
  _ndc.y = -((ev.clientY - rect.top)  / rect.height) * 2 + 1;
  _raycaster.setFromCamera(_ndc, camera);

  // 1. Frustum click — works on every variant. Loops over the wireframe
  //    pyramids currently in the scene; the closest hit wins.
  const frustumObjs = [];
  for (const ov of frameOverlays.values()) {
    if (ov.frustum) frustumObjs.push(ov.frustum);
  }
  if (frustumObjs.length) {
    const fHits = _raycaster.intersectObjects(frustumObjs, false);
    if (fHits.length) {
      const idx = fHits[0].object.userData.frameIdx;
      if (idx != null) {
        jumpCameraToFrame(idx);
        return;
      }
    }
  }

  // 2. Voxel click — only meaningful for any features* variant.
  if (!voxMesh) return;
  if (!isFeaturesVariant(variantSel.value)) return;
  await prefetchFeaturesMeta();
  if (!featuresLookup) return;

  const hits = _raycaster.intersectObject(voxMesh, false);
  if (!hits.length) return;
  const triple = voxMesh.userData.voxelIndices?.[hits[0].instanceId];
  if (!triple) return;
  selectVoxel(triple);
}

// Move the flycam to the clicked frame's pose + viewing direction (using
// the manifest's `pose` and `forward`) and scroll the panel so the frame's
// thumbnail is centred. Forward → (yaw, pitch) inverts the same YXZ Euler
// formula used by updateCamera().
function jumpCameraToFrame(idx) {
  const f = frameManifest?.frames?.find((fr) => fr.idx === idx);
  if (!f || !f.pose || !f.forward) return;
  const [px, py, pz] = f.pose;
  const [fx_, fy_, fz_] = f.forward;
  const len = Math.hypot(fx_, fy_, fz_) || 1;
  const nx = fx_ / len, ny = fy_ / len, nz = fz_ / len;
  camera.position.set(px, py, pz);
  yaw   = Math.atan2(-nx, -nz);
  pitch = Math.atan2(ny, Math.hypot(nx, nz));
  if (pitch >  PITCH_LIMIT) pitch =  PITCH_LIMIT;
  if (pitch < -PITCH_LIMIT) pitch = -PITCH_LIMIT;
  camera.rotation.set(pitch, yaw, 0, "YXZ");
  if (!panelOpen) setPanelOpen(true);
  scrollToFrameItem(idx);
}

function selectVoxel(triple) {
  const key = `${triple[0]},${triple[1]},${triple[2]}`;
  // Toggle off if the user re-clicks the same voxel.
  if (activeVoxelKey === key) {
    clearActiveVoxel();
    return;
  }
  const entry = featuresLookup?.get(key);
  if (!entry) {
    clearActiveVoxel();
    return;
  }

  // Replace current selection with the union of frames the voxel's features
  // were observed in, and stash per-frame keypoint UVs for the SVG overlay.
  // Each `entry.features[k].obs[j]` is `{frame, u, v}` in colour-buffer
  // bottom-up coords.
  for (const idx of [...frameOverlays.keys()]) removeFrameOverlay(idx);
  selectedFrames.clear();
  frameMarkers.clear();
  for (const feat of (entry.features || [])) {
    for (const ob of (feat.obs || [])) {
      if (!frameMarkers.has(ob.frame)) frameMarkers.set(ob.frame, []);
      frameMarkers.get(ob.frame).push({ u: ob.u, v: ob.v });
    }
  }

  activeVoxelKey = key;
  drawActiveVoxelHighlight(triple);

  // Open the panel if hidden and add a frustum overlay per selected frame.
  if (!panelOpen) setPanelOpen(true);
  // Frame ids the voxel touches, in the order they were observed.
  const frames = [...frameMarkers.keys()].sort((a, b) => a - b);
  for (const fi of frames) {
    selectedFrames.add(fi);
    addFrameOverlay(fi);   // fire-and-forget; per-frame frustum + voxel highlight
  }

  // Re-apply marker DOM and selected styling to all visible frame items,
  // and scroll the first selected one into view so the user can immediately
  // see what they clicked on.
  rerenderPanelMarkersAndSelection();
  if (frames.length) scrollToFrameItem(frames[0]);
  framePanelCount.textContent =
    `${frameManifest?.frames?.length ?? 0} frames · ${selectedFrames.size} selected · voxel ${triple.join(",")}`;
}

function clearActiveVoxel() {
  activeVoxelKey = null;
  if (activeVoxelMesh) {
    scene.remove(activeVoxelMesh);
    activeVoxelMesh.geometry.dispose();
    activeVoxelMesh.material.dispose();
    activeVoxelMesh = null;
  }
  frameMarkers.clear();
  rerenderPanelMarkersAndSelection();
}

function drawActiveVoxelHighlight(triple) {
  if (activeVoxelMesh) {
    scene.remove(activeVoxelMesh);
    activeVoxelMesh.geometry.dispose();
    activeVoxelMesh.material.dispose();
    activeVoxelMesh = null;
  }
  if (!lastVoxelPayload) return;
  const wm = lastVoxelPayload.world_min;
  const sz = lastVoxelPayload.voxel_size;
  // Wireframe edges around the voxel cell, sized 1.10× so the lines clearly
  // peek out from the underlying coloured cube.
  const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(sz*1.10, sz*1.10, sz*1.10));
  const mat = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2,
                                            depthTest: false, transparent: true, opacity: 0.95 });
  const mesh = new THREE.LineSegments(edges, mat);
  mesh.position.set(
    wm[0] + (triple[0] + 0.5) * sz,
    wm[1] + (triple[1] + 0.5) * sz,
    wm[2] + (triple[2] + 0.5) * sz,
  );
  mesh.renderOrder = 10;
  scene.add(mesh);
  activeVoxelMesh = mesh;
}

function scrollToFrameItem(idx) {
  const el = framePanelList.querySelector(`.frame-item[data-idx="${idx}"]`);
  if (el) el.scrollIntoView({ block: "center", behavior: "smooth" });
}

// Rebuild SVG markers for every visible frame item + sync selected styling.
// Cheap: walks the DOM, no re-layout of the list itself.
function rerenderPanelMarkersAndSelection() {
  for (const item of framePanelList.querySelectorAll(".frame-item")) {
    const idx = Number(item.dataset.idx);
    if (selectedFrames.has(idx)) item.classList.add("selected");
    else                          item.classList.remove("selected");
    applyKeypointMarkersToItem(item, idx);
  }
}

// Build/replace the SVG marker layers for one frame-item. Creates one SVG
// overlay per thumb-wrap (rgb + depth). The viewBox is the colour buffer's
// natural pixel size (cw × ch); circles placed at (u·cw, (1−v)·ch) line up
// with the displayed image because the SVG uses the same `xMidYMid meet`
// aspect rule as `object-fit: contain`. Depth thumbs are resampled into the
// colour image's pixel grid by the server, so the same coords work for both.
function applyKeypointMarkersToItem(item, idx) {
  const cw = Number(item.dataset.cw) || 0;
  const ch = Number(item.dataset.ch) || 0;
  const marks = frameMarkers.get(idx) || [];
  for (const wrap of item.querySelectorAll(".thumb-wrap")) {
    const old = wrap.querySelector("svg.kp-overlay");
    if (old) old.remove();
    if (!marks.length || cw <= 0 || ch <= 0) continue;
    const NS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("class", "kp-overlay");
    svg.setAttribute("viewBox", `0 0 ${cw} ${ch}`);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    // Marker radius scales with image diagonal so it looks the same on
    // landscape vs portrait captures.
    const r = Math.max(8, Math.round(Math.hypot(cw, ch) * 0.014));
    for (const m of marks) {
      const x = m.u * cw;
      const y = (1 - m.v) * ch;
      const ring = document.createElementNS(NS, "circle");
      ring.setAttribute("cx", x); ring.setAttribute("cy", y);
      ring.setAttribute("r", r);
      svg.appendChild(ring);
      const dot = document.createElementNS(NS, "circle");
      dot.setAttribute("class", "dot");
      dot.setAttribute("cx", x); dot.setAttribute("cy", y);
      dot.setAttribute("r", Math.max(2, r * 0.18));
      svg.appendChild(dot);
    }
    wrap.appendChild(svg);
  }
}

// =====================================================================
// Pixel-cloud mode — per-pixel projection of selected frames into 3D.
// Each selected frame is fetched once via /pixel-cloud/<idx>.json (the
// server returns origin + per-pixel ray dirs + raw depths + colors) and
// rendered as a THREE.Points cloud. Sliders for the model affine fit
// `(a, b)` and the depth-vs-disparity space toggle re-compute positions
// CLIENT-SIDE from the cached payload, so dragging them doesn't go back
// to the server. Only depth-source / pose-source / stride changes
// trigger a re-fetch.
// =====================================================================

let pcMode = false;
let pcModelStatus = null;   // { ready, frames: { idx: { w, h } } } or null
// Per-frame affine state survives selection toggles so the user can
// auto-tune one batch, deselect, re-select, and find the same tuning
// in place. Slider edits and Reset overwrite *all* currently-selected
// frames' entries; auto-tune populates them per-frame.
const frameAffine = new Map();   // idx → { a, b, fit_space }
// Common features observed in every currently-selected frame, fetched
// from /common-features. Drives both the per-thumb keypoint overlay
// and the in-3D sphere markers.
let pcCommonFeatures = [];       // [{ world: [x,y,z], obs: { idx: [u,v] } }]
let pcCommonFeaturesMesh = null; // THREE.Points sprites for the 3D markers
let pcCommonFeaturesShown = true;

function pcParams() {
  return {
    depth_kind: pcDepthKindSel.value,
    pose_dir:   pcPoseDirSel.value || "frames",
    stride:     parseInt(pcStrideSel.value, 10) || 4,
    a:          parseFloat(pcASlider.value),
    b:          parseFloat(pcBSlider.value),
    fit_space:  pcFitSpaceSel.value,
  };
}

function affineForFrame(idx) {
  // Per-frame tuning if we have it; otherwise the global slider state.
  // Phone depth is metric and has no slider math attached.
  const stored = frameAffine.get(idx);
  if (stored) return stored;
  const p = pcParams();
  return { a: p.a, b: p.b, fit_space: p.fit_space };
}

function syncPcAffineEnabled() {
  // The (a, b, fit_space, reset) controls only mean something for model
  // depth — phone depth is already in metres. Grey them out otherwise so
  // the user can see the mode is recognised but the controls are inert.
  const en = pcDepthKindSel.value === "model";
  pcAffineGroup.style.opacity = en ? "1" : "0.45";
  for (const el of pcAffineGroup.querySelectorAll("input,select,button")) {
    el.disabled = !en;
  }
}

function updatePcStatus() {
  const p = pcParams();
  let n = 0;
  let dMin = Infinity, dMax = -Infinity;
  for (const ov of frameOverlays.values()) {
    if (!ov.points || !ov.payload) continue;
    n += ov.payload.count;
    const dd = ov.payload.depths;
    for (let i = 0; i < dd.length; i++) {
      const x = dd[i];
      if (x < dMin) dMin = x;
      if (x > dMax) dMax = x;
    }
  }
  // Report the raw depth range across all selected frames. Useful for
  // spotting bad frames where ARCore returned a tight band of values
  // (typical of the first ~15 frames after session start, before the
  // depth API has stabilised) — in that case the cloud projects onto
  // a single near-planar surface at the band's distance, which looks
  // like a "flat plane orthogonal to the central ray" bug but is
  // really just the data.
  const depthTag = (n > 0 && isFinite(dMin))
    ? ` · raw d ${dMin.toFixed(2)}–${dMax.toFixed(2)} m`
    : "";
  let cacheTag = "";
  if ((p.depth_kind === "model" || p.depth_kind === "blend")
      && !pcModelStatus?.ready) {
    cacheTag = " · model cache missing — run cache_model_raw.py";
  }
  pcStatus.textContent =
    `${frameOverlays.size} frames · ${n.toLocaleString()} pts${depthTag} · `
    + `${p.depth_kind} · ${p.pose_dir}${cacheTag}`;
}

async function refreshPcModelStatus() {
  const sid = sessionSel.value;
  if (!sid) { pcModelStatus = null; updatePcStatus(); return; }
  try {
    const r = await fetch(`/captures/${encodeURIComponent(sid)}/pixel-cloud-status`);
    pcModelStatus = r.ok ? await r.json() : { ready: false, frames: {} };
  } catch (e) {
    console.warn("pixel-cloud-status fetch failed", e);
    pcModelStatus = { ready: false, frames: {} };
  }
  // Update the model option in the depth-kind dropdown to reflect cache state.
  const modelOpt = [...pcDepthKindSel.options].find((o) => o.value === "model");
  const blendOpt = [...pcDepthKindSel.options].find((o) => o.value === "blend");
  if (modelOpt) {
    modelOpt.disabled = !pcModelStatus?.ready;
    const n = Object.keys(pcModelStatus?.frames || {}).length;
    modelOpt.textContent = pcModelStatus?.ready
      ? `model (Depth-Anything-V2, ${n} cached)`
      : "model (no cache — run cache_model_raw.py)";
  }
  if (blendOpt) {
    blendOpt.disabled = !pcModelStatus?.ready;
    blendOpt.textContent = pcModelStatus?.ready
      ? "blend (phone low + model high, σ=3%)"
      : "blend (no cache — run cache_model_raw.py)";
  }
  if (!pcModelStatus?.ready
      && (pcDepthKindSel.value === "model" || pcDepthKindSel.value === "blend")) {
    pcDepthKindSel.value = "phone";
    syncPcAffineEnabled();
  }
  updatePcStatus();
}

function populatePosePicker() {
  // Populate from the currently-selected session's frame_dirs (returned
  // by /sessions). Sort so the most likely-useful pose dir is first.
  const sid = sessionSel.value;
  const sess = availableSessions.find((s) => s.id === sid);
  let dirs = (sess?.frame_dirs || []).slice();
  if (!dirs.length) dirs.push("frames");
  const rank = (d) => {
    if (d === "frames")               return 0;
    if (d === "frames_aligned")       return 1;
    if (d.startsWith("frames_feature_ba")) return 2;
    if (d.startsWith("frames_refined"))    return 4;
    return 3;
  };
  dirs.sort((a, b) => (rank(a) - rank(b)) || (a < b ? -1 : 1));
  const previous = pcPoseDirSel.value;
  pcPoseDirSel.innerHTML = "";
  for (const d of dirs) {
    const o = document.createElement("option");
    o.value = d;
    o.textContent = d;
    pcPoseDirSel.appendChild(o);
  }
  if (previous && dirs.includes(previous)) pcPoseDirSel.value = previous;
  else                                     pcPoseDirSel.value = dirs[0];
}

async function fetchPixelCloud(idx) {
  const sid = sessionSel.value;
  const p = pcParams();
  // Blend uses the server-side hole-aware Gaussian detail-injection at a
  // fixed 3% sigma — no slider here. (Sigma can still be tuned on the
  // depth-scatter and stereo pages.)
  const sigmaQ = (p.depth_kind === "blend") ? "&sigma=0.03" : "";
  const url = `/captures/${encodeURIComponent(sid)}/pixel-cloud/${idx}.json`
    + `?depth_kind=${encodeURIComponent(p.depth_kind)}`
    + `&pose_dir=${encodeURIComponent(p.pose_dir)}`
    + `&stride=${p.stride}${sigmaQ}`;
  let r;
  try { r = await fetch(url); }
  catch (e) { console.warn("pixel-cloud fetch failed", e); return null; }
  if (!r.ok) {
    const txt = await r.text().catch(() => "");
    console.warn(`pixel-cloud HTTP ${r.status} for #${idx}: ${txt}`);
    return null;
  }
  return await r.json();
}

function applyAffineToPositions(payload, positionsBuf, a, b, fitSpace) {
  // Phone and blend depths are already metric — sliders are a no-op for
  // both even if the user has nudged them in another mode.
  if (payload.depth_kind === "phone" || payload.depth_kind === "blend") {
    a = 1.0; b = 0.0; fitSpace = "depth";
  }
  const n = payload.count;
  const ox = payload.origin[0], oy = payload.origin[1], oz = payload.origin[2];
  const dirs = payload.dirs;
  const raw  = payload.depths;
  for (let i = 0; i < n; i++) {
    const r = raw[i];
    let d;
    if (fitSpace === "disparity") {
      const denom = a + b * r;
      d = (denom > 1e-3) ? r / denom : 0.0;
    } else {
      d = a * r + b;
    }
    if (!(d > 0) || !isFinite(d)) d = 0.0;
    positionsBuf[3*i  ] = ox + d * dirs[3*i  ];
    positionsBuf[3*i+1] = oy + d * dirs[3*i+1];
    positionsBuf[3*i+2] = oz + d * dirs[3*i+2];
  }
}

function rebuildOverlayPositions(idx) {
  // Re-apply this frame's affine to its cached payload's positions buffer.
  // Used by the auto-tune button + by anything that mutates frameAffine.
  const ov = frameOverlays.get(idx);
  if (!ov || !ov.points || !ov.payload) return;
  const aff = affineForFrame(idx);
  const posAttr = ov.points.geometry.getAttribute("position");
  applyAffineToPositions(ov.payload, posAttr.array, aff.a, aff.b, aff.fit_space);
  posAttr.needsUpdate = true;
}

function buildPointsMesh(payload) {
  const n = payload.count;
  if (!n) return null;
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  // Initial fill at identity affine; the caller re-applies the live
  // slider values right after to match the UI state.
  applyAffineToPositions(payload, positions, 1.0, 0.0, "depth");
  for (let i = 0; i < n; i++) {
    // Convert sRGB-encoded camera bytes to linear so the colour matches
    // the lit voxel rendering's setRGB().convertSRGBToLinear() path.
    const r = (payload.colors[3*i  ] / 255);
    const g = (payload.colors[3*i+1] / 255);
    const b = (payload.colors[3*i+2] / 255);
    colors[3*i  ] = r * r;   // cheap sRGB→linear approx (γ ≈ 2.0)
    colors[3*i+1] = g * g;
    colors[3*i+2] = b * b;
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position",
    new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
  geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  // Disable culling — slider-driven depth changes can push points well
  // outside any precomputed bounding sphere.
  geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);
  const mat = new THREE.PointsMaterial({
    size: 0.025,
    vertexColors: true,
    sizeAttenuation: true,
  });
  const points = new THREE.Points(geom, mat);
  points.frustumCulled = false;
  return points;
}

async function addPixelCloudOverlay(idx) {
  // The user may have toggled mode off / deselected the frame while the
  // fetch was in flight — bail out so we don't add stale objects.
  if (!selectedFrames.has(idx) || !pcMode) return;
  const colour = colorForFrame(idx);
  const payload = await fetchPixelCloud(idx);
  if (!payload) return;
  if (!selectedFrames.has(idx) || !pcMode) return;
  let frustum = null;
  if (Array.isArray(payload.frustum_world) && payload.frustum_world.length === 8
      && Array.isArray(payload.origin) && payload.origin.length === 3) {
    frustum = buildFrustumLines(payload.frustum_world, payload.origin, colour);
    frustum.userData.frameIdx = idx;
    scene.add(frustum);
  }
  const points = buildPointsMesh(payload);
  if (points) {
    scene.add(points);
    // Apply this frame's stored (a, b) — falls back to the global slider
    // values for frames that haven't been auto-tuned or manually set yet.
    const aff = affineForFrame(idx);
    const posAttr = points.geometry.getAttribute("position");
    applyAffineToPositions(payload, posAttr.array, aff.a, aff.b, aff.fit_space);
    posAttr.needsUpdate = true;
  }
  frameOverlays.set(idx, { frustum, points, payload });
  updatePcStatus();
}

async function refetchAllPcOverlays() {
  // Pose / depth_kind / stride change — payloads are stale, fetch fresh.
  // Sliders never trigger this (they reuse the cached payload).
  if (!pcMode) return;
  const ids = [...frameOverlays.keys()];
  for (const idx of ids) removeFrameOverlay(idx);
  for (const idx of ids) await addPixelCloudOverlay(idx);
  updatePcStatus();
}

function recomputeAllPcPositions() {
  // Called on slider input + fit-space toggle. Slider edits write the
  // new (a, b, fit_space) into every selected frame's frameAffine entry
  // — i.e. the slider acts as a "bulk apply to all selected" control.
  // Auto-tune populates per-frame values; sliders are the manual override.
  const p = pcParams();
  for (const idx of frameOverlays.keys()) {
    frameAffine.set(idx, { a: p.a, b: p.b, fit_space: p.fit_space });
    rebuildOverlayPositions(idx);
  }
  pcAVal.textContent = p.a.toFixed(3);
  pcBVal.textContent = (p.b >= 0 ? "+" : "") + p.b.toFixed(3);
  updatePcStatus();
}

function applyPcModeToggle(force) {
  pcMode = (typeof force === "boolean") ? force : !pcMode;
  pcToolbar.hidden = !pcMode;
  pixelCloudBtn.textContent = pcMode ? "Pixel cloud ◂" : "Pixel cloud ▸";
  // Drop existing overlays so we cleanly switch between modes.
  const wasSelected = [...selectedFrames];
  for (const idx of [...frameOverlays.keys()]) removeFrameOverlay(idx);
  if (pcMode) {
    populatePosePicker();
    refreshPcModelStatus();
  } else {
    // Leaving pcMode — drop the common-features 3D markers + thumb
    // keypoint overlays so the non-pcMode UX isn't littered with them.
    removeCommonFeaturesMesh();
    pcCommonFeatures = [];
    frameMarkers.clear();
    rerenderPanelMarkersAndSelection();
  }
  // Re-add overlays under the new mode.
  for (const idx of wasSelected) addFrameOverlay(idx);
  syncPcAffineEnabled();
  if (pcMode) scheduleCommonFeaturesRefresh();
  updatePcStatus();
}

pixelCloudBtn.addEventListener("click", () => applyPcModeToggle());
pcDepthKindSel.addEventListener("change", () => {
  syncPcAffineEnabled();
  refetchAllPcOverlays();
});
pcPoseDirSel.addEventListener("change", () => {
  // Different pose set → different features_meta sidecar → refresh
  // common-features (and the per-thumb keypoint overlays) too.
  refetchAllPcOverlays();
  scheduleCommonFeaturesRefresh();
});
pcStrideSel.addEventListener("change", refetchAllPcOverlays);
pcASlider.addEventListener("input", recomputeAllPcPositions);
pcBSlider.addEventListener("input", recomputeAllPcPositions);
pcFitSpaceSel.addEventListener("change", recomputeAllPcPositions);
pcResetBtn.addEventListener("click", () => {
  pcASlider.value = "1.0";
  pcBSlider.value = "0.0";
  pcFitSpaceSel.value = "depth";
  // Reset clears all selected frames' tuning back to identity.
  for (const idx of frameOverlays.keys()) frameAffine.delete(idx);
  recomputeAllPcPositions();
});

// ---------------------------------------------------------------------
// Common features (across the currently-selected frames) — drives both
// the per-thumb keypoint overlay and the 3D sphere markers. Fetched
// from /captures/<id>/common-features and rebuilt on every selection /
// pose-dir change. Cached so the auto-tune button can use the same
// list without a second fetch.
// ---------------------------------------------------------------------

let pcCommonFetchTimer = null;
function scheduleCommonFeaturesRefresh() {
  // 200 ms debounce so rapid multi-click selections only fetch once.
  if (pcCommonFetchTimer) clearTimeout(pcCommonFetchTimer);
  pcCommonFetchTimer = setTimeout(() => {
    pcCommonFetchTimer = null;
    refreshCommonFeatures();
  }, 200);
}

async function refreshCommonFeatures() {
  // Drop any prior 3D markers + thumb markers so an empty / failed
  // result clears the previous batch.
  removeCommonFeaturesMesh();
  frameMarkers.clear();
  pcCommonFeatures = [];
  if (!pcMode || selectedFrames.size < 1) {
    rerenderPanelMarkersAndSelection();
    updatePcStatus();
    return;
  }
  const sid = sessionSel.value;
  const p = pcParams();
  const frames = [...selectedFrames].sort((a, b) => a - b);
  const url = `/captures/${encodeURIComponent(sid)}/common-features`
    + `?frames=${frames.join(",")}`
    + `&pose_dir=${encodeURIComponent(p.pose_dir)}`;
  let r;
  try { r = await fetch(url); }
  catch (e) { console.warn("common-features fetch failed", e); return; }
  if (!r.ok) {
    // 404 = no matching features_meta sidecar; tell the user via status.
    const txt = await r.text().catch(() => "");
    pcStatus.textContent = `common-features HTTP ${r.status}: ${txt.split("\n")[0] || "?"}`;
    return;
  }
  const payload = await r.json();
  pcCommonFeatures = payload.features || [];
  // Populate thumb markers (existing frameMarkers infrastructure).
  for (const feat of pcCommonFeatures) {
    for (const [frameStr, uv] of Object.entries(feat.obs || {})) {
      const fi = Number(frameStr);
      if (!frameMarkers.has(fi)) frameMarkers.set(fi, []);
      frameMarkers.get(fi).push({ u: uv[0], v: uv[1] });
    }
  }
  rerenderPanelMarkersAndSelection();
  if (pcCommonFeaturesShown) addCommonFeaturesMesh();
  updatePcStatus();
}

function addCommonFeaturesMesh() {
  removeCommonFeaturesMesh();
  if (!pcCommonFeatures.length) return;
  // Cyan points at world positions; large enough to spot against the
  // pixel cloud. depthTest off so they peek through occluding voxels.
  const positions = new Float32Array(pcCommonFeatures.length * 3);
  for (let i = 0; i < pcCommonFeatures.length; i++) {
    const w = pcCommonFeatures[i].world;
    positions[3*i  ] = w[0];
    positions[3*i+1] = w[1];
    positions[3*i+2] = w[2];
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);
  const mat = new THREE.PointsMaterial({
    color: 0x00ffff,
    size: 0.06,
    sizeAttenuation: true,
    depthTest: false,
    depthWrite: false,
    transparent: true,
    opacity: 0.95,
  });
  const points = new THREE.Points(geom, mat);
  points.frustumCulled = false;
  points.renderOrder = 12;
  scene.add(points);
  pcCommonFeaturesMesh = points;
}

function removeCommonFeaturesMesh() {
  if (!pcCommonFeaturesMesh) return;
  scene.remove(pcCommonFeaturesMesh);
  pcCommonFeaturesMesh.geometry.dispose();
  pcCommonFeaturesMesh.material.dispose();
  pcCommonFeaturesMesh = null;
}

pcFeaturesToggle.addEventListener("change", () => {
  pcCommonFeaturesShown = pcFeaturesToggle.checked;
  if (pcCommonFeaturesShown) addCommonFeaturesMesh();
  else                       removeCommonFeaturesMesh();
});

// ---------------------------------------------------------------------
// Auto-tune: server fits per-frame (a, b) to align cached model_raw
// depths with BA-feature world positions, then we apply per-frame.
// ---------------------------------------------------------------------

async function doAutotune() {
  if (!pcMode || selectedFrames.size < 1) return;
  const sid = sessionSel.value;
  const p = pcParams();
  if (p.depth_kind !== "model") {
    pcStatus.textContent = "auto-tune only meaningful for depth=model";
    return;
  }
  const frames = [...selectedFrames].sort((a, b) => a - b);
  pcAutotuneBtn.disabled = true;
  pcAutotuneBtn.textContent = "tuning…";
  try {
    const url = `/captures/${encodeURIComponent(sid)}/pixel-cloud-autotune`
      + `?frames=${frames.join(",")}`
      + `&pose_dir=${encodeURIComponent(p.pose_dir)}`
      + `&fit_space=${encodeURIComponent(p.fit_space)}`;
    const r = await fetch(url);
    if (!r.ok) {
      const txt = await r.text().catch(() => "");
      pcStatus.textContent = `auto-tune HTTP ${r.status}: ${txt.split("\n")[0]}`;
      return;
    }
    const payload = await r.json();
    if (!payload.n_common) {
      pcStatus.textContent =
        `auto-tune: no features common to all ${frames.length} frames — `
        + `pick frames closer in time`;
      return;
    }
    let nFit = 0;
    let residSum = 0;
    let residCount = 0;
    for (const [frameStr, fit] of Object.entries(payload.per_frame)) {
      const fi = Number(frameStr);
      if (fit && typeof fit.a === "number" && typeof fit.b === "number") {
        frameAffine.set(fi, {
          a: fit.a,
          b: fit.b,
          fit_space: fit.fit_space || p.fit_space,
        });
        rebuildOverlayPositions(fi);
        nFit++;
        if (typeof fit.residual_m === "number") {
          residSum += fit.residual_m;
          residCount++;
        }
      }
    }
    const meanR = residCount ? (residSum / residCount) : NaN;
    pcStatus.textContent =
      `auto-tuned ${nFit}/${frames.length} frames · ${payload.n_common} common features · `
      + `median |Δd| ≈ ${isFinite(meanR) ? (meanR * 100).toFixed(1) + " cm" : "?"} · ${p.fit_space}`;
  } catch (e) {
    pcStatus.textContent = `auto-tune error: ${e.message || e}`;
  } finally {
    pcAutotuneBtn.disabled = false;
    pcAutotuneBtn.textContent = "auto-tune";
  }
}

pcAutotuneBtn.addEventListener("click", doAutotune);

// ---------------------------------------------------------------------
// Voxel-overlap auto-tune — feature-free, coarse-to-fine. Sends each
// frame's current (a, b) as the optimiser's initial point so a second
// click can refine an earlier run rather than starting from scratch.
// ---------------------------------------------------------------------

async function doAutotuneVoxel() {
  if (!pcMode || selectedFrames.size < 2) {
    pcStatus.textContent = "voxel auto-tune needs ≥ 2 selected frames";
    return;
  }
  const sid = sessionSel.value;
  const p = pcParams();
  if (p.depth_kind !== "model") {
    pcStatus.textContent = "voxel auto-tune only meaningful for depth=model";
    return;
  }
  const frames = [...selectedFrames].sort((a, b) => a - b);
  // Pack the current per-frame state into the `init=` query so the
  // optimiser warm-starts from the user's manually set / previously
  // auto-tuned values rather than from identity (1, 0).
  const initParts = frames.map((idx) => {
    const aff = affineForFrame(idx);
    return `${idx}:${aff.a.toFixed(4)},${aff.b.toFixed(4)}`;
  }).join(";");

  pcAutotuneVoxelBtn.disabled = true;
  pcAutotuneVoxelBtn.textContent = "tuning…";
  pcStatus.textContent =
    `voxel auto-tune: ${frames.length} frames, coarse→fine 30/15/5 cm…`;
  try {
    const url = `/captures/${encodeURIComponent(sid)}/pixel-cloud-autotune-voxel`
      + `?frames=${frames.join(",")}`
      + `&pose_dir=${encodeURIComponent(p.pose_dir)}`
      + `&fit_space=${encodeURIComponent(p.fit_space)}`
      + `&voxel_sizes=0.30,0.15,0.05`
      + `&stride=8`
      + `&init=${encodeURIComponent(initParts)}`;
    const r = await fetch(url);
    if (!r.ok) {
      const txt = await r.text().catch(() => "");
      pcStatus.textContent = `voxel auto-tune HTTP ${r.status}: ${txt.split("\n")[0]}`;
      return;
    }
    const payload = await r.json();
    let nFit = 0;
    for (const [frameStr, fit] of Object.entries(payload.per_frame || {})) {
      const fi = Number(frameStr);
      if (fit && typeof fit.a === "number" && typeof fit.b === "number") {
        frameAffine.set(fi, {
          a: fit.a,
          b: fit.b,
          fit_space: fit.fit_space || p.fit_space,
        });
        rebuildOverlayPositions(fi);
        nFit++;
      }
    }
    const stages = (payload.stages || [])
      .map((s) => `${(s.voxel_size*100).toFixed(0)}cm:${s.overlap}`)
      .join(" → ");
    pcStatus.textContent =
      `voxel auto-tuned ${nFit}/${frames.length} frames · ${stages} `
      + `· ${p.fit_space}`;
  } catch (e) {
    pcStatus.textContent = `voxel auto-tune error: ${e.message || e}`;
  } finally {
    pcAutotuneVoxelBtn.disabled = false;
    pcAutotuneVoxelBtn.textContent = "auto-tune (voxel)";
  }
}

pcAutotuneVoxelBtn.addEventListener("click", doAutotuneVoxel);

// ---------------------------------------------------------------------
// Chamfer-distance auto-tune — smooth surrogate. Warm-starts from the
// current per-frame state, which is critical: a cold (1, 0) start
// often falls into the "collapse all clouds onto each frame's camera
// origin" basin (the smooth metric, unlike the discrete voxel count,
// rewards collapsing). Following auto-tune (features) → auto-tune
// (chamfer) usually converges to a more honest fit.
// ---------------------------------------------------------------------

async function doAutotuneChamfer() {
  if (!pcMode || selectedFrames.size < 2) {
    pcStatus.textContent = "chamfer auto-tune needs ≥ 2 selected frames";
    return;
  }
  const sid = sessionSel.value;
  const p = pcParams();
  if (p.depth_kind !== "model") {
    pcStatus.textContent = "chamfer auto-tune only meaningful for depth=model";
    return;
  }
  const frames = [...selectedFrames].sort((a, b) => a - b);
  const initParts = frames.map((idx) => {
    const aff = affineForFrame(idx);
    return `${idx}:${aff.a.toFixed(4)},${aff.b.toFixed(4)}`;
  }).join(";");

  pcAutotuneChamferBtn.disabled = true;
  pcAutotuneChamferBtn.textContent = "tuning…";
  pcStatus.textContent =
    `chamfer auto-tune: ${frames.length} frames, thresholds 30→5 cm…`;
  try {
    const url = `/captures/${encodeURIComponent(sid)}/pixel-cloud-autotune-chamfer`
      + `?frames=${frames.join(",")}`
      + `&pose_dir=${encodeURIComponent(p.pose_dir)}`
      + `&fit_space=${encodeURIComponent(p.fit_space)}`
      + `&thresholds=0.30,0.05`
      + `&stride=16`
      + `&init=${encodeURIComponent(initParts)}`;
    const r = await fetch(url);
    if (!r.ok) {
      const txt = await r.text().catch(() => "");
      pcStatus.textContent = `chamfer auto-tune HTTP ${r.status}: ${txt.split("\n")[0]}`;
      return;
    }
    const payload = await r.json();
    let nFit = 0;
    for (const [frameStr, fit] of Object.entries(payload.per_frame || {})) {
      const fi = Number(frameStr);
      if (fit && typeof fit.a === "number" && typeof fit.b === "number") {
        frameAffine.set(fi, {
          a: fit.a,
          b: fit.b,
          fit_space: fit.fit_space || p.fit_space,
        });
        rebuildOverlayPositions(fi);
        nFit++;
      }
    }
    const stages = (payload.stages || [])
      .map((s) => `${(s.threshold_m*100).toFixed(0)}cm:${s.loss?.toFixed(4) ?? "?"}`)
      .join(" → ");
    pcStatus.textContent =
      `chamfer auto-tuned ${nFit}/${frames.length} frames · `
      + `loss ${stages} · ${p.fit_space}`;
  } catch (e) {
    pcStatus.textContent = `chamfer auto-tune error: ${e.message || e}`;
  } finally {
    pcAutotuneChamferBtn.disabled = false;
    pcAutotuneChamferBtn.textContent = "auto-tune (chamfer)";
  }
}

pcAutotuneChamferBtn.addEventListener("click", doAutotuneChamfer);

// (Common-features refresh is triggered from inside toggleFrameSelection
// — see the call to scheduleCommonFeaturesRefresh there.)

// Refresh model-cache status + pose dropdown whenever the active session
// changes. Listeners are independent of the existing session/variant
// handlers above so the order of registration doesn't matter.
sessionSel.addEventListener("change", () => {
  // Per-frame affine entries are tied to frame indices in the *current*
  // session — drop them when the session changes so they don't bleed
  // into the next session's frames-of-the-same-numbers.
  frameAffine.clear();
  pcCommonFeatures = [];
  removeCommonFeaturesMesh();
  refreshPcModelStatus();
  populatePosePicker();
});

// Boot: fetch the session list first so the dropdowns are populated before
// the initial fetch picks the latest session + refined variant.
refreshSessionList().then(() => {
  populatePosePicker();
  refreshPcModelStatus();
  return loadVoxels();
}).then(() => {
  if (isFeaturesVariant(variantSel.value)) prefetchFeaturesMeta();
});
