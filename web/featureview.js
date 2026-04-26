// Laptop-side viewer for the classical feature-based reconstruction.
// Loads /out/features.json (written by tools/feature_ray_reconstruct.py)
// and renders every triangulated feature as a small lit cube. Same fly
// cam + lit/flat toggle as voxelview.

import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { SSAOPass } from "three/addons/postprocessing/SSAOPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const stage = document.getElementById("stage");
const reloadBtn = document.getElementById("reloadBtn");
const recenterBtn = document.getElementById("recenterBtn");
const lightingBtn = document.getElementById("lightingBtn");
const stCount = document.getElementById("stCount");
const stSize = document.getElementById("stSize");
const stMsg = document.getElementById("stMsg");

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
ssao.kernelRadius = 0.06;
ssao.minDistance = 0.0008;
ssao.maxDistance = 0.04;
composer.addPass(ssao);
composer.addPass(new OutputPass());

const featMaterial = new THREE.MeshStandardMaterial({
  color: 0xffffff,
  metalness: 0.0,
  roughness: 0.85,
});
const featGeometry = new THREE.BoxGeometry(1, 1, 1);

let featMesh = null;
let lastBBox = null;

function rebuildMesh(payload) {
  if (featMesh) {
    scene.remove(featMesh);
    featMesh.dispose();
    featMesh = null;
  }
  const positions = payload.positions;
  const colors = payload.colors;
  const sz = payload.cube_size;
  if (!positions || positions.length === 0) {
    lastBBox = null;
    return;
  }
  const m = new THREE.InstancedMesh(featGeometry, featMaterial, positions.length);
  m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  m.castShadow = true;
  m.receiveShadow = true;

  const tmpMat = new THREE.Matrix4();
  const tmpQuat = new THREE.Quaternion();
  const tmpScale = new THREE.Vector3(sz, sz, sz);
  const tmpPos = new THREE.Vector3();
  const tmpCol = new THREE.Color();
  const bbox = new THREE.Box3();

  for (let k = 0; k < positions.length; k++) {
    const [x, y, z] = positions[k];
    const [r, g, b] = colors[k];
    tmpPos.set(x, y, z);
    tmpMat.compose(tmpPos, tmpQuat, tmpScale);
    m.setMatrixAt(k, tmpMat);
    tmpCol.setRGB(r / 255, g / 255, b / 255).convertSRGBToLinear();
    m.setColorAt(k, tmpCol);
    bbox.expandByPoint(tmpPos);
  }
  m.instanceMatrix.needsUpdate = true;
  if (m.instanceColor) m.instanceColor.needsUpdate = true;
  scene.add(m);
  featMesh = m;
  lastBBox = bbox.isEmpty() ? null : bbox;
  applyLightingMode();
}

async function loadFeatures() {
  reloadBtn.disabled = true;
  stMsg.textContent = "loading…";
  try {
    const r = await fetch(`/out/features.json?t=${Date.now()}`);
    if (!r.ok) {
      stMsg.textContent = `features.json: HTTP ${r.status}`;
      stCount.textContent = "—";
      return;
    }
    const payload = await r.json();
    rebuildMesh(payload);
    stCount.textContent = (payload.n_features || 0).toLocaleString();
    stSize.textContent = `${(payload.cube_size ?? 0).toFixed(2)} m`;
    stMsg.textContent = `min-views=${payload.min_views} `
                      + `max-residual=${(payload.max_residual ?? 0).toFixed(3)}m`;
    recenter(true);
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

reloadBtn.addEventListener("click", loadFeatures);
recenterBtn.addEventListener("click", () => recenter(true));

let flatMode = true;
function applyLightingMode() {
  if (flatMode) {
    ambient.intensity = 1.0;
    sun.intensity = 0.0;
    sun.castShadow = false;
    hemi.intensity = 0.0;
    floor.visible = false;
    if (featMesh) {
      featMesh.castShadow = false;
      featMesh.receiveShadow = false;
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
    if (featMesh) {
      featMesh.castShadow = true;
      featMesh.receiveShadow = true;
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
applyLightingMode();

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

// ----- fly cam (lifted from voxelview.js) ---------------------------------
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

loadFeatures();
