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

function syncVariantOptions() {
  const sid = sessionSel.value;
  const sess = availableSessions.find((s) => s.id === sid);
  const have = new Set(sess ? sess.variants : []);
  for (const opt of variantSel.options) {
    opt.disabled = !have.has(opt.value);
  }
  // Default to refined if present, else original; but keep the user's
  // previous pick when it's still valid.
  if (variantSel.options[variantSel.selectedIndex]?.disabled) {
    if (have.has("refined"))      variantSel.value = "refined";
    else if (have.has("original")) variantSel.value = "original";
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

// Boot: fetch the session list first so the dropdowns are populated before
// the initial fetch picks the latest session + refined variant.
refreshSessionList().then(loadVoxels);
