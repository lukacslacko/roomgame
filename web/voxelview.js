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
  scene.add(m);
  voxMesh = m;
  lastBBox = bbox.isEmpty() ? null : bbox;
  // Newly-built mesh defaults to lit settings; re-apply current mode so a
  // reload while in flat mode doesn't snap the cubes back to lit.
  applyLightingMode();
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
  // Pick the depth-thumbnail source: prefer the model-derived depth in
  // `frames_refined/` (i.e. what depth_refine.py wrote out — that's the
  // depth the user actually wants to inspect for debugging), fall back
  // to the phone's lowres ARCore depth in `frames/` if the session hasn't
  // been through the refiner yet.
  const sessInfo = availableSessions.find((s) => s.id === sid);
  const depthDir = (sessInfo && (sessInfo.n_frames_refined ?? 0) > 0)
                   ? "frames_refined" : "frames";
  const depthLabel = depthDir === "frames_refined" ? "model depth" : "phone depth";
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
}

async function addFrameOverlay(idx) {
  const sid = sessionSel.value;
  const variant = variantSel.value || "original";
  if (!sid || !lastVoxelPayload) return;
  const wm = lastVoxelPayload.world_min;
  const sz = lastVoxelPayload.voxel_size;
  const sh = lastVoxelPayload.shape;
  // The "features" variant comes from feature_ray_reconstruct.py — its
  // per-frame voxels are not derived from depth backprojection but from
  // which feature tracks were observed in this frame, so we hit a
  // dedicated endpoint instead of `frame-voxels`.
  let url;
  if (variant === "features") {
    url = `/captures/${encodeURIComponent(sid)}/frame-feature-voxels/${idx}.json`;
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
  refreshFrameManifest();
  if (variantSel.value === "features") prefetchFeaturesMeta();
});
variantSel.addEventListener("change", () => {
  clearAllFrameOverlays();
  clearActiveVoxel();
  if (variantSel.value === "features") prefetchFeaturesMeta();
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

async function prefetchFeaturesMeta() {
  const sid = sessionSel.value;
  if (!sid) return;
  if (featuresMeta && featuresLookup) return;
  try {
    const r = await fetch(
      `/captures/${encodeURIComponent(sid)}/features_meta.json?t=${Date.now()}`,
    );
    if (!r.ok) {
      featuresMeta = null; featuresLookup = null;
      console.warn(`features_meta HTTP ${r.status}`);
      return;
    }
    featuresMeta = await r.json();
  } catch (e) {
    featuresMeta = null; featuresLookup = null;
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

  // 2. Voxel click — only meaningful for the features variant.
  if (!voxMesh) return;
  if (variantSel.value !== "features") return;
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

// Boot: fetch the session list first so the dropdowns are populated before
// the initial fetch picks the latest session + refined variant.
refreshSessionList().then(loadVoxels).then(() => {
  if (variantSel.value === "features") prefetchFeaturesMeta();
});
