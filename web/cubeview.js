// Laptop-side viewer for the coarse occupancy-cube grid.
//
// Polls /cubes/state and renders every returned cube as a real, lit,
// shadow-casting box via three.js InstancedMesh. Per-instance colour
// runs along a teal → yellow ramp by occupancy ratio. Same fly-cam
// controls as game.html.
//
//   W A S D            move (camera-forward projected onto X-Z plane)
//   Space / Shift      world up / world down
//   Arrow keys         yaw / pitch
//   Click + drag       look around
//   Mouse wheel        dolly along the camera's forward
//   Hold Shift+W etc   ~3× speed

import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { SSAOPass } from "three/addons/postprocessing/SSAOPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const stage = document.getElementById("stage");
const reloadBtn = document.getElementById("reloadBtn");
const autoBtn = document.getElementById("autoBtn");
const recenterBtn = document.getElementById("recenterBtn");
const thrSlider = document.getElementById("thr");
const thrVal = document.getElementById("thrVal");
const minObsInput = document.getElementById("minObs");
const stCubes = document.getElementById("stCubes");
const stSize = document.getElementById("stSize");
const stFrames = document.getElementById("stFrames");
const stMsg = document.getElementById("stMsg");

thrSlider.addEventListener("input", () => { thrVal.textContent = (+thrSlider.value).toFixed(2); });

// ----- scene + lights ------------------------------------------------------
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

scene.add(new THREE.AmbientLight(0xffffff, 0.30));
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
scene.add(new THREE.HemisphereLight(0xc6d8ff, 0x202028, 0.45));

// A faint floor plane catches the sun shadow even where there are no cubes,
// which makes the cube positions easier to read in space. Y=0 is the
// local-floor plane reported by ARCore, so this also acts as a ground-truth
// reference.
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

// ----- post-processing -----------------------------------------------------
const composer = new EffectComposer(renderer);
composer.renderTarget1.samples = 4;
composer.renderTarget2.samples = 4;
composer.addPass(new RenderPass(scene, camera));
const ssao = new SSAOPass(scene, camera, 0, 0);
ssao.kernelRadius = 0.12;
ssao.minDistance = 0.0008;
ssao.maxDistance = 0.08;
composer.addPass(ssao);
composer.addPass(new OutputPass());

// ----- cube instanced mesh -------------------------------------------------
//
// One InstancedMesh of unit cubes (BoxGeometry 1×1×1). Each instance gets a
// translation+scale matrix so the cube ends up at its world centre with the
// configured cube_size, and a per-instance colour from the occupancy ramp.
const cubeMaterial = new THREE.MeshStandardMaterial({
  color: 0xffffff,        // white base, modulated by per-instance colour
  metalness: 0.0,
  roughness: 0.85,
});
const cubeGeometry = new THREE.BoxGeometry(1, 1, 1);

let cubeMesh = null;
let lastBBox = null;

function rebuildInstancedMesh(cubes, cubeSize) {
  if (cubeMesh) {
    scene.remove(cubeMesh);
    cubeMesh.dispose();
    cubeMesh = null;
  }
  if (!cubes || cubes.length === 0) {
    lastBBox = null;
    return;
  }
  const m = new THREE.InstancedMesh(cubeGeometry, cubeMaterial, cubes.length);
  m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  m.castShadow = true;
  m.receiveShadow = true;

  const tmpMat = new THREE.Matrix4();
  const tmpQuat = new THREE.Quaternion();
  const tmpScale = new THREE.Vector3(cubeSize, cubeSize, cubeSize);
  const tmpPos = new THREE.Vector3();
  const tmpCol = new THREE.Color();

  const bbox = new THREE.Box3();

  for (let i = 0; i < cubes.length; i++) {
    const c = cubes[i];          // [cx, cy, cz, ratio]
    tmpPos.set(c[0], c[1], c[2]);
    tmpMat.compose(tmpPos, tmpQuat, tmpScale);
    m.setMatrixAt(i, tmpMat);
    rampColor(c[3], tmpCol);
    m.setColorAt(i, tmpCol);
    bbox.expandByPoint(tmpPos);
  }
  m.instanceMatrix.needsUpdate = true;
  if (m.instanceColor) m.instanceColor.needsUpdate = true;

  scene.add(m);
  cubeMesh = m;
  lastBBox = bbox.isEmpty() ? null : bbox;
}

function rampColor(t, out) {
  // teal → green → yellow ramp on occupancy ratio. Matches the phone shader.
  t = Math.max(0, Math.min(1, t));
  if (t < 0.5) {
    const u = t * 2;
    out.setRGB(
      THREE.MathUtils.lerp(0.05, 0.20, u),
      THREE.MathUtils.lerp(0.30, 0.65, u),
      THREE.MathUtils.lerp(0.55, 0.45, u),
    );
  } else {
    const u = (t - 0.5) * 2;
    out.setRGB(
      THREE.MathUtils.lerp(0.20, 0.95, u),
      THREE.MathUtils.lerp(0.65, 0.85, u),
      THREE.MathUtils.lerp(0.45, 0.25, u),
    );
  }
  return out;
}

// ----- /cubes/state polling ------------------------------------------------
let auto = true;
let polling = false;
let pollTimer = null;

autoBtn.addEventListener("click", () => {
  auto = !auto;
  autoBtn.classList.toggle("on", auto);
  if (auto) schedulePoll(); else { clearTimeout(pollTimer); pollTimer = null; }
});
reloadBtn.addEventListener("click", () => poll());
recenterBtn.addEventListener("click", () => recenter(true));
thrSlider.addEventListener("change", () => poll());
minObsInput.addEventListener("change", () => poll());

async function poll() {
  if (polling) return;
  polling = true;
  try {
    const thr = +thrSlider.value;
    const minObs = Math.max(1, +minObsInput.value || 1);
    const r = await fetch(`/cubes/state?threshold=${thr}&min=${minObs}`);
    if (!r.ok) {
      stMsg.textContent = `state ${r.status}`;
      return;
    }
    const s = await r.json();
    if (!s.ready) {
      stMsg.textContent = "no grid yet (POST /cubes/start from the phone)";
      stCubes.textContent = "0";
      stFrames.textContent = "0";
      stSize.textContent = "—";
      rebuildInstancedMesh([], 0);
      return;
    }
    stCubes.textContent = String(s.cubes.length);
    stFrames.textContent = String(s.frames);
    stSize.textContent = `${s.cube_size.toFixed(2)} m`;
    stMsg.textContent = s.recording ? "recording" : "";

    // Convert (ix, iy, iz, occ, free) → (cx, cy, cz, ratio) in world coords.
    const wm = s.world_min, sz = s.cube_size;
    const cubes = new Array(s.cubes.length);
    for (let k = 0; k < s.cubes.length; k++) {
      const [ix, iy, iz, occ, free] = s.cubes[k];
      const tot = occ + free;
      cubes[k] = [
        wm[0] + (ix + 0.5) * sz,
        wm[1] + (iy + 0.5) * sz,
        wm[2] + (iz + 0.5) * sz,
        tot > 0 ? occ / tot : 0,
      ];
    }
    rebuildInstancedMesh(cubes, sz);
  } catch (e) {
    stMsg.textContent = `network err`;
    console.warn(e);
  } finally {
    polling = false;
    if (auto) schedulePoll();
  }
}

function schedulePoll() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(poll, 1000);
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
  // Always grow shadow camera to cover the cube cloud.
  const r = Math.max(5, sz.length() * 0.6);
  sun.shadow.camera.left = -r;
  sun.shadow.camera.right = r;
  sun.shadow.camera.top = r;
  sun.shadow.camera.bottom = -r;
  sun.shadow.camera.updateProjectionMatrix();
}

// ----- resize --------------------------------------------------------------
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

// ----- fly cam (lifted from game.js) --------------------------------------
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

// Kick off.
poll();
