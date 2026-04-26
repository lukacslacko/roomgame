// WebXR pose-stability validation page.
//
// Tap the screen to define the *current* phone pose as origin. Translation
// and rotation relative to that origin are then displayed live, plus a
// top-down trajectory canvas of recent positions. Walk away and back —
// any non-zero residual when standing at the origin again is ARCore drift.
//
// Conventions:
//   view.transform.matrix is column-major Float32Array[16], world_from_view.
//   - Translation = m[12..14].
//   - 3x3 rotation block: m[col*4 + row], col,row ∈ {0,1,2}.
//   - WebXR uses Y-up; "yaw" below is rotation around Y, pitch around X,
//     roll around Z (YXZ convention).
//
// Relative pose math: with M_o = world_from_view at the moment of anchoring
// and M_n = world_from_view now, we compute origin_from_now = M_o^{-1} M_n.
// For rigid (R, t) this gives:
//   t_rel = R_o^T (t_n - t_o)        (camera offset in origin's local axes)
//   R_rel = R_o^T R_n
//
// No three.js, no HTTP. Self-contained.

import { hookConsoleToServer } from "./logbridge.js";
hookConsoleToServer();

const $ = (id) => document.getElementById(id);
const gate = $("gate");
const gateMsg = $("gateMsg");
const startBtn = $("startBtn");
const overlay = $("xrOverlay");
const hudStatus = $("hudStatus");
const exitBtn = $("exitBtn");
const anchorBtn = $("anchorBtn");
const hudTap = $("hudTap");
const trajCanvas = $("trajCanvas");
const trajCtx = trajCanvas.getContext("2d");

const out = {
  dist: $("rDist"),
  tx: $("rTx"), ty: $("rTy"), tz: $("rTz"),
  yaw: $("rYaw"), pitch: $("rPitch"), roll: $("rRoll"),
  track: $("rTrack"),
};

let session = null;
let xrRefSpace = null;
let gl = null;
let xrCanvas = null;

// Origin pose (world_from_view at anchor time). null until first tap.
let originR = null;        // Float64Array(9), column-major 3x3
let originT = null;        // Float64Array(3)

// Recent (tx, tz) samples for the top-down trajectory plot.
const TRAJ_MAX = 600;
const trajPoints = [];

async function checkSupport() {
  if (!("xr" in navigator)) {
    return { ok: false, reason: "navigator.xr missing — WebXR not available." };
  }
  let arSupported = false;
  try {
    arSupported = await navigator.xr.isSessionSupported("immersive-ar");
  } catch (e) {
    return { ok: false, reason: `isSessionSupported failed: ${e}` };
  }
  if (!arSupported) {
    return { ok: false, reason: "immersive-ar not supported on this device." };
  }
  return { ok: true, reason: "AR available — tap Enter AR, then tap anywhere to set the origin." };
}

async function init() {
  const { ok, reason } = await checkSupport();
  gateMsg.textContent = reason;
  if (ok) {
    startBtn.disabled = false;
    startBtn.addEventListener("click", startSession);
  }
}

async function startSession() {
  startBtn.disabled = true;
  gateMsg.textContent = "Requesting AR session…";

  xrCanvas = document.createElement("canvas");
  xrCanvas.style.position = "fixed";
  xrCanvas.style.inset = "0";
  xrCanvas.style.width = "100%";
  xrCanvas.style.height = "100%";
  document.body.appendChild(xrCanvas);
  gl = xrCanvas.getContext("webgl2", { xrCompatible: true, alpha: true, antialias: false });
  if (!gl) {
    gateMsg.textContent = "WebGL2 unavailable.";
    startBtn.disabled = false;
    return;
  }

  try {
    session = await navigator.xr.requestSession("immersive-ar", {
      requiredFeatures: ["local-floor"],
      optionalFeatures: ["dom-overlay"],
      domOverlay: { root: overlay },
    });
  } catch (e) {
    gateMsg.textContent = `Failed to start AR: ${e.message || e}`;
    startBtn.disabled = false;
    return;
  }

  await gl.makeXRCompatible();
  await session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
  xrRefSpace = await session.requestReferenceSpace("local-floor");

  session.addEventListener("end", onSessionEnd);
  session.addEventListener("select", anchorNow);
  exitBtn.addEventListener("click", () => session && session.end());
  anchorBtn.addEventListener("click", anchorNow);

  hudStatus.textContent = "no origin yet — tap to set";
  gate.style.display = "none";
  overlay.style.display = "flex";

  session.requestAnimationFrame(onXRFrame);
}

function onSessionEnd() {
  console.log("session ended");
  session = null;
  xrRefSpace = null;
  if (xrCanvas?.parentNode) xrCanvas.parentNode.removeChild(xrCanvas);
  xrCanvas = null;
  gl = null;
  originR = null;
  originT = null;
  trajPoints.length = 0;
  overlay.style.display = "none";
  gate.style.display = "";
  gateMsg.textContent = "Session ended. Tap Enter AR to retry.";
  startBtn.disabled = false;
}

function anchorNow() {
  // Capture the most recent pose. Stored fields are written each frame; if
  // we haven't seen one yet, ignore.
  if (!latestR) return;
  originR = new Float64Array(latestR);
  originT = new Float64Array(latestT);
  trajPoints.length = 0;
  hudTap.textContent = "origin set — walk around and come back";
  hudStatus.textContent = "tracking relative pose";
}

let latestR = null;
let latestT = null;
let latestEmulated = false;

function onXRFrame(time, frame) {
  if (!session) return;
  session.requestAnimationFrame(onXRFrame);

  const baseLayer = session.renderState.baseLayer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, baseLayer.framebuffer);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const pose = frame.getViewerPose(xrRefSpace);
  if (!pose || pose.views.length === 0) return;
  const m = pose.views[0].transform.matrix;

  // Pull rotation block + translation as Float64 for the relative-pose math.
  if (!latestR) {
    latestR = new Float64Array(9);
    latestT = new Float64Array(3);
  }
  // Column-major 4x4 → column-major 3x3 (drop row 3 + col 3).
  latestR[0] = m[0];  latestR[1] = m[1];  latestR[2] = m[2];
  latestR[3] = m[4];  latestR[4] = m[5];  latestR[5] = m[6];
  latestR[6] = m[8];  latestR[7] = m[9];  latestR[8] = m[10];
  latestT[0] = m[12]; latestT[1] = m[13]; latestT[2] = m[14];
  latestEmulated = !!pose.emulatedPosition;

  if (originR) {
    // t_rel = R_o^T (t_n - t_o)
    const dx = latestT[0] - originT[0];
    const dy = latestT[1] - originT[1];
    const dz = latestT[2] - originT[2];
    // R_o^T applied to (dx, dy, dz). With column-major storage, R[i,j] (row i,
    // col j) = R[j*3 + i]. R^T[i,j] = R[j,i] = R[i*3 + j]. So:
    const tx = originR[0]*dx + originR[1]*dy + originR[2]*dz;
    const ty = originR[3]*dx + originR[4]*dy + originR[5]*dz;
    const tz = originR[6]*dx + originR[7]*dy + originR[8]*dz;

    // R_rel = R_o^T R_n (column-major, 3x3).
    const Rr = mul3x3T(originR, latestR);

    const distCm = Math.sqrt(tx*tx + ty*ty + tz*tz) * 100;
    const { yaw, pitch, roll } = eulerYXZ(Rr);
    const angleDeg = axisAngleDeg(Rr);

    out.dist.textContent = `|t| = ${distCm.toFixed(1)} cm    ∠R = ${angleDeg.toFixed(1)}°`;
    out.tx.textContent = (tx * 100).toFixed(1);
    out.ty.textContent = (ty * 100).toFixed(1);
    out.tz.textContent = (tz * 100).toFixed(1);
    out.yaw.textContent = (yaw * 180 / Math.PI).toFixed(2);
    out.pitch.textContent = (pitch * 180 / Math.PI).toFixed(2);
    out.roll.textContent = (roll * 180 / Math.PI).toFixed(2);
    out.track.textContent = latestEmulated ? "emulated" : "ok";
    out.track.className = latestEmulated ? "warn" : "ok";

    // Top-down trajectory (X-Z plane in origin frame; Y is up so it falls out).
    trajPoints.push([tx, tz]);
    if (trajPoints.length > TRAJ_MAX) trajPoints.shift();
    drawTrajectory();
  } else {
    out.track.textContent = latestEmulated ? "emulated" : "ok (no origin)";
  }
}

function mul3x3T(A, B) {
  // Returns A^T * B for column-major 3x3 stored as flat [a00,a10,a20,a01,...].
  // A[i,j] = A[j*3+i].  (A^T)[i,j] = A[j,i] = A[i*3+j].
  // C[i,j] = sum_k (A^T)[i,k] B[k,j] = sum_k A[k,i] B[k,j].
  // In column-major output: C[j*3+i] = sum_k A[i*3+k] B[j*3+k].
  const C = new Float64Array(9);
  for (let j = 0; j < 3; j++) {
    for (let i = 0; i < 3; i++) {
      let s = 0;
      for (let k = 0; k < 3; k++) {
        s += A[i*3 + k] * B[j*3 + k];
      }
      C[j*3 + i] = s;
    }
  }
  return C;
}

function eulerYXZ(R) {
  // Column-major R[col*3+row]. Equivalent row-major M[row][col]:
  // M[0][0] M[0][1] M[0][2]   = R[0] R[3] R[6]
  // M[1][0] M[1][1] M[1][2]   = R[1] R[4] R[7]
  // M[2][0] M[2][1] M[2][2]   = R[2] R[5] R[8]
  // YXZ convention (rotate Y then X then Z, applied right-to-left to a vector):
  // R = R_y(yaw) R_x(pitch) R_z(roll)
  //   pitch = asin(-M[1][2]) = asin(-R[7])
  //   yaw   = atan2(M[0][2], M[2][2]) = atan2(R[6], R[8])
  //   roll  = atan2(M[1][0], M[1][1]) = atan2(R[1], R[4])
  const m12 = R[7];
  const pitch = Math.asin(-Math.max(-1, Math.min(1, m12)));
  let yaw, roll;
  if (Math.abs(m12) < 0.99999) {
    yaw  = Math.atan2(R[6], R[8]);
    roll = Math.atan2(R[1], R[4]);
  } else {
    // Gimbal lock — fold roll into yaw.
    yaw  = Math.atan2(-R[2], R[0]);
    roll = 0;
  }
  return { yaw, pitch, roll };
}

function axisAngleDeg(R) {
  // angle = arccos((trace - 1) / 2). trace = R[0,0] + R[1,1] + R[2,2].
  // Column-major flat indexing: diag = R[0], R[4], R[8].
  const tr = R[0] + R[4] + R[8];
  const c  = Math.max(-1, Math.min(1, (tr - 1) / 2));
  return Math.acos(c) * 180 / Math.PI;
}

function drawTrajectory() {
  const w = trajCanvas.width;
  const h = trajCanvas.height;
  trajCtx.clearRect(0, 0, w, h);

  // Auto-fit: compute a square bbox around origin large enough to contain
  // all points, with a 1 m floor so a stationary phone shows a sensible scale.
  let maxR = 1.0;
  for (const [x, z] of trajPoints) {
    const r = Math.max(Math.abs(x), Math.abs(z));
    if (r > maxR) maxR = r;
  }
  // Round up to a sensible number of metres for grid lines.
  const grid = niceStep(maxR);
  const half = Math.ceil(maxR / grid) * grid;

  const cx = w / 2, cy = h / 2;
  const px = (x) => cx + (x / half) * (w / 2 - 8);
  const py = (z) => cy - (z / half) * (h / 2 - 8); // -z forward = up on canvas

  // Grid.
  trajCtx.strokeStyle = "rgba(255,255,255,0.08)";
  trajCtx.lineWidth = 1;
  for (let g = grid; g <= half + 1e-6; g += grid) {
    trajCtx.beginPath();
    trajCtx.arc(cx, cy, (g / half) * (w / 2 - 8), 0, Math.PI * 2);
    trajCtx.stroke();
  }
  // Axes.
  trajCtx.strokeStyle = "rgba(255,255,255,0.18)";
  trajCtx.beginPath();
  trajCtx.moveTo(cx, 4); trajCtx.lineTo(cx, h - 4);
  trajCtx.moveTo(4, cy); trajCtx.lineTo(w - 4, cy);
  trajCtx.stroke();

  // Trajectory polyline.
  if (trajPoints.length > 1) {
    trajCtx.strokeStyle = "rgba(110,196,255,0.85)";
    trajCtx.lineWidth = 1.5;
    trajCtx.beginPath();
    trajCtx.moveTo(px(trajPoints[0][0]), py(trajPoints[0][1]));
    for (let i = 1; i < trajPoints.length; i++) {
      trajCtx.lineTo(px(trajPoints[i][0]), py(trajPoints[i][1]));
    }
    trajCtx.stroke();
  }

  // Origin (green) and current position (red).
  trajCtx.fillStyle = "#6cf28b";
  trajCtx.beginPath();
  trajCtx.arc(cx, cy, 5, 0, Math.PI * 2);
  trajCtx.fill();
  if (trajPoints.length > 0) {
    const [x, z] = trajPoints[trajPoints.length - 1];
    trajCtx.fillStyle = "#ff7777";
    trajCtx.beginPath();
    trajCtx.arc(px(x), py(z), 5, 0, Math.PI * 2);
    trajCtx.fill();
  }

  // Scale label.
  trajCtx.fillStyle = "rgba(255,255,255,0.5)";
  trajCtx.font = "11px ui-monospace, Menlo, monospace";
  trajCtx.fillText(`±${half.toFixed(1)} m   grid ${grid.toFixed(2)} m`, 8, h - 8);
}

function niceStep(maxR) {
  // Pick a grid step that gives 2–5 rings within ±maxR.
  const candidates = [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10];
  for (const c of candidates) if (maxR / c <= 5) return c;
  return candidates[candidates.length - 1];
}

init();
