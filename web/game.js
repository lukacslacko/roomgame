// Laptop-side viewer for the scanned room mesh.
//
// Loads /out/room.glb from POST /remesh and renders it with vertex colours.
// Camera is a fly-cam:
//
//   W / A / S / D     forward / strafe-left / back / strafe-right (planar)
//   Space / Shift     world up / world down
//   Arrow keys        yaw / pitch (alternative to mouse drag)
//   Click + drag      look around (yaw + pitch)
//   Mouse wheel       dolly along the camera's forward direction
//   Hold Shift+W etc  ~3× speed
//
// "Forward" for WASD is the camera's forward direction projected onto the
// horizontal plane, so you can't accidentally fly through the floor when
// walking forward while looking down.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { SSAOPass } from "three/addons/postprocessing/SSAOPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const stage = document.getElementById("stage");
const reloadBtn = document.getElementById("reloadBtn");
const reloadOnlyBtn = document.getElementById("reloadOnlyBtn");
const stVerts = document.getElementById("stVerts");
const stFaces = document.getElementById("stFaces");
const stVoxels = document.getElementById("stVoxels");
const stMsg = document.getElementById("stMsg");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101015);

const camera = new THREE.PerspectiveCamera(60, 1, 0.05, 200);
camera.rotation.order = "YXZ";   // yaw then pitch — keeps "world up" stable
camera.position.set(0, 1.6, 2);

const renderer = new THREE.WebGLRenderer({ antialias: false }); // SSAO supplies the AA-feel
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
stage.appendChild(renderer.domElement);

// A slightly off-white ambient + warm directional + cool hemisphere; keeps
// the room readable without being either a sunny outdoor scene or a flat
// matte. Sun casts soft shadows so geometry crevices get visible darkening.
scene.add(new THREE.AmbientLight(0xffffff, 0.35));
const sun = new THREE.DirectionalLight(0xfff1d6, 1.4);
sun.position.set(2.5, 4.0, 2.0);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.near = 0.1;
sun.shadow.camera.far = 30;
sun.shadow.camera.left = -6;
sun.shadow.camera.right = 6;
sun.shadow.camera.top = 6;
sun.shadow.camera.bottom = -6;
sun.shadow.bias = -0.0005;
sun.shadow.normalBias = 0.01;
scene.add(sun);
scene.add(new THREE.HemisphereLight(0xc6d8ff, 0x202028, 0.4));

const grid = new THREE.GridHelper(10, 20, 0x444466, 0x222234);
scene.add(grid);
scene.add(new THREE.AxesHelper(0.5));

// Post-processing chain: render scene → SSAO darkens contact areas
// (under furniture, in corners) → output pass tonemaps + sRGB-encodes.
// MSAA on the composer's internal render targets so polygon edges still
// look clean even though the renderer's own antialias is off.
const composer = new EffectComposer(renderer);
composer.renderTarget1.samples = 4;
composer.renderTarget2.samples = 4;
const renderPass = new RenderPass(scene, camera);
composer.addPass(renderPass);

const ssao = new SSAOPass(scene, camera, 0, 0);     // size set via onResize
ssao.kernelRadius = 0.12;                            // ~12 cm; matches indoor scale
ssao.minDistance = 0.0008;
ssao.maxDistance = 0.08;
composer.addPass(ssao);

composer.addPass(new OutputPass());

const loader = new GLTFLoader();
let currentMesh = null;

function clearMesh() {
  if (!currentMesh) return;
  scene.remove(currentMesh);
  currentMesh.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) o.material.dispose();
  });
  currentMesh = null;
}

async function loadMesh(forceRemesh) {
  stMsg.textContent = forceRemesh ? "re-meshing…" : "loading…";
  reloadBtn.disabled = true;
  reloadOnlyBtn.disabled = true;
  try {
    if (forceRemesh) {
      const r = await fetch("/remesh", { method: "POST" });
      if (!r.ok) {
        stMsg.textContent = `remesh failed: ${r.status}`;
        return;
      }
      const meta = await r.json();
      stVerts.textContent = meta.vertices.toLocaleString();
      stFaces.textContent = meta.faces.toLocaleString();
    }
    const url = `/out/room.glb?t=${Date.now()}`;
    const gltf = await loader.loadAsync(url);
    clearMesh();
    currentMesh = gltf.scene;
    currentMesh.traverse((o) => {
      if (o.isMesh) {
        // Marching-cubes output from skimage doesn't carry vertex normals,
        // and trimesh's GLB writer doesn't always include them. Shading
        // requires them, so compute on load if missing.
        if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();

        const colorAttr = o.geometry?.attributes?.color;
        const hasVertexColors = !!colorAttr;
        if (hasVertexColors && colorAttr.colorSpace !== THREE.SRGBColorSpace) {
          // Camera bytes are sRGB; tell three.js to convert to linear when
          // shading. Without this the colours look washed out under ACES.
          colorAttr.colorSpace = THREE.SRGBColorSpace;
        }

        o.material = new THREE.MeshStandardMaterial({
          color: hasVertexColors ? 0xffffff : 0xc0c8d8,
          vertexColors: hasVertexColors,
          metalness: 0.0,
          roughness: 0.9,
          side: THREE.DoubleSide,
        });
        o.castShadow = true;
        o.receiveShadow = true;
      }
    });
    scene.add(currentMesh);

    // Place the camera at scan-typical eye height in the middle of the scene
    // looking forward. (No more orbit-around-target framing — the user wants
    // to walk through their room.)
    const box = new THREE.Box3().setFromObject(currentMesh);
    if (!box.isEmpty()) {
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      camera.position.set(center.x, Math.min(1.6, center.y + size.y * 0.4), center.z + size.z * 0.6);
      camera.rotation.set(0, 0, 0);   // reset look direction
      yaw = 0; pitch = 0;
      camera.near = 0.05;
      camera.far = Math.max(50, size.length() * 5);
      camera.updateProjectionMatrix();
    }
    stMsg.textContent = "";
  } catch (e) {
    stMsg.textContent = `error: ${e.message || e}`;
    console.error(e);
  } finally {
    reloadBtn.disabled = false;
    reloadOnlyBtn.disabled = false;
  }
}

async function pollStats() {
  try {
    const r = await fetch("/stats");
    if (!r.ok) return;
    const s = await r.json();
    stVoxels.textContent = (s.voxels || 0).toLocaleString();
  } catch {}
}

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

reloadBtn.addEventListener("click", () => loadMesh(true));
reloadOnlyBtn.addEventListener("click", () => loadMesh(false));

setInterval(pollStats, 1000);
pollStats();

fetch("/out/room.glb", { method: "HEAD" })
  .then((r) => { if (r.ok) loadMesh(false); })
  .catch(() => {});

// ----- fly camera ---------------------------------------------------------

const keys = new Set();
const movementCodes = new Set([
  "KeyW", "KeyA", "KeyS", "KeyD",
  "Space", "ShiftLeft", "ShiftRight",
  "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
]);
window.addEventListener("keydown", (e) => {
  // Don't hijack typing in the toolbar's buttons / inputs.
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  keys.add(e.code);
  if (movementCodes.has(e.code)) e.preventDefault();
});
window.addEventListener("keyup", (e) => { keys.delete(e.code); });
window.addEventListener("blur", () => keys.clear());  // avoid stuck keys

let yaw = 0;       // radians, around world Y
let pitch = 0;     // radians, around camera local X (after yaw)
const PITCH_LIMIT = Math.PI / 2 - 0.05;

let dragging = false;
let lastMouse = { x: 0, y: 0 };
renderer.domElement.addEventListener("mousedown", (e) => {
  // Left button only. Prevent text-selection while dragging.
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
  // Dolly along the *full* camera forward direction (not the planar one),
  // so the wheel feels like zooming toward what you're looking at.
  const dir = new THREE.Vector3();
  camera.getWorldDirection(dir);
  const step = (e.deltaY > 0 ? -1 : 1) * 0.25;
  camera.position.addScaledVector(dir, step);
  e.preventDefault();
}, { passive: false });

const BASE_SPEED_M_PER_S = 1.6;   // a comfortable walking pace
const SPRINT_MULT = 3.0;
const LOOK_RATE_RAD_PER_S = Math.PI / 2;  // arrow-key look = 90°/s
let lastTime = performance.now();

const _fwdHorizontal = new THREE.Vector3();
const _rightHorizontal = new THREE.Vector3();
const _move = new THREE.Vector3();

function updateCamera(dtSec) {
  // Arrow-key look: yaw left/right, pitch up/down.
  if (keys.has("ArrowLeft"))  yaw   += LOOK_RATE_RAD_PER_S * dtSec;
  if (keys.has("ArrowRight")) yaw   -= LOOK_RATE_RAD_PER_S * dtSec;
  if (keys.has("ArrowUp"))    pitch += LOOK_RATE_RAD_PER_S * dtSec;
  if (keys.has("ArrowDown"))  pitch -= LOOK_RATE_RAD_PER_S * dtSec;
  if (pitch >  PITCH_LIMIT) pitch =  PITCH_LIMIT;
  if (pitch < -PITCH_LIMIT) pitch = -PITCH_LIMIT;

  // Apply yaw + pitch as Euler (YXZ), then read off the planar basis.
  camera.rotation.set(pitch, yaw, 0, "YXZ");

  // Camera forward is camera-space -Z transformed by the rotation. We want
  // the *horizontal* forward (projected onto the X-Z plane) for WASD; that
  // way pressing W when looking up doesn't lift you off the ground.
  _fwdHorizontal.set(-Math.sin(yaw), 0, -Math.cos(yaw));
  _rightHorizontal.set(Math.cos(yaw), 0, -Math.sin(yaw));

  _move.set(0, 0, 0);
  if (keys.has("KeyW")) _move.add(_fwdHorizontal);
  if (keys.has("KeyS")) _move.sub(_fwdHorizontal);
  if (keys.has("KeyD")) _move.add(_rightHorizontal);
  if (keys.has("KeyA")) _move.sub(_rightHorizontal);
  if (keys.has("Space")) _move.y += 1;
  if (keys.has("ShiftLeft") || keys.has("ShiftRight")) _move.y -= 1;

  if (_move.lengthSq() > 0) {
    _move.normalize();
    // Sprint when both Shift modifiers are held alongside a direction. Since
    // ShiftLeft alone is already "down", sprint requires *another* movement
    // key to be held — which is the natural case anyway.
    const sprint = (keys.has("ShiftLeft") || keys.has("ShiftRight"))
                   && (keys.has("KeyW") || keys.has("KeyA") || keys.has("KeyS") || keys.has("KeyD"));
    const speed = BASE_SPEED_M_PER_S * (sprint ? SPRINT_MULT : 1.0);
    camera.position.addScaledVector(_move, speed * dtSec);
  }
}

renderer.setAnimationLoop((time) => {
  const now = performance.now();
  const dtSec = Math.min(0.1, (now - lastTime) / 1000);
  lastTime = now;
  updateCamera(dtSec);
  composer.render();
});
