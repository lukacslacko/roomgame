// Laptop-side viewer for the scanned room mesh.
//
// Loads /out/room.glb produced by POST /remesh on the server. Orbit camera,
// double-sided material so we can look at the mesh from inside the room (the
// usual case for a scan, since the camera was inside while scanning).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

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
camera.position.set(3, 2.5, 3);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
stage.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1, 0);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const sun = new THREE.DirectionalLight(0xffffff, 1.0);
sun.position.set(3, 5, 4);
scene.add(sun);
scene.add(new THREE.HemisphereLight(0xb8d6ff, 0x202020, 0.4));

// Floor grid for orientation. Drawn at y=0; should sit at the WebXR
// `local-floor` reference space y=0 plane after fusion.
const grid = new THREE.GridHelper(10, 20, 0x444466, 0x222234);
scene.add(grid);
scene.add(new THREE.AxesHelper(0.5));

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
    // Cache-bust: append a timestamp so the browser picks up the freshly
    // written GLB rather than a stale cached copy.
    const url = `/out/room.glb?t=${Date.now()}`;
    const gltf = await loader.loadAsync(url);
    clearMesh();
    currentMesh = gltf.scene;
    currentMesh.traverse((o) => {
      if (o.isMesh) {
        o.material = new THREE.MeshStandardMaterial({
          color: 0xc0c8d8, metalness: 0.05, roughness: 0.9, side: THREE.DoubleSide,
        });
      }
    });
    scene.add(currentMesh);

    // Frame the mesh so something is visible without manual orbit fiddling.
    const box = new THREE.Box3().setFromObject(currentMesh);
    if (!box.isEmpty()) {
      const size = box.getSize(new THREE.Vector3()).length();
      const center = box.getCenter(new THREE.Vector3());
      controls.target.copy(center);
      camera.position.copy(center).add(new THREE.Vector3(size, size * 0.7, size));
      camera.near = Math.max(0.01, size / 1000);
      camera.far = size * 50;
      camera.updateProjectionMatrix();
      controls.update();
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

// Try a quiet initial load of any pre-existing /out/room.glb (HEAD → 200 means
// a previous session left one behind). Skip if not present, no error spam.
fetch("/out/room.glb", { method: "HEAD" })
  .then((r) => { if (r.ok) loadMesh(false); })
  .catch(() => {});

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});
