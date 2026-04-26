// Phone-side recorder for the offline voxel pipeline.
//
// Enters an immersive-AR session (local-floor + cpu-optimized depth + camera-
// access + dom-overlay), and while "Start recording" is on, ships every XR
// frame to the server as a single binary /frame POST: pose matrices, depth
// buffer, and the WebXR camera image at its native resolution.
//
// No cubes are rendered on the phone, no /cubes/state polling — both used to
// burn frame budget for no benefit during capture. Reconstruction happens
// offline via tools/voxel_reconstruct.py and is viewed with cubeview.html /
// voxelview.html on the laptop.
//
// /cubes/start and /cubes/stop are still called so the server's coarse
// cube-grid keeps ingesting frames in parallel — no extra cost on the phone.

import { hookConsoleToServer } from "./logbridge.js";
hookConsoleToServer();

const $ = (id) => document.getElementById(id);
const gate = $("gate");
const gateMsg = $("gateMsg");
const startBtn = $("startBtn");
const overlay = $("xrOverlay");
const hudStatus = $("hudStatus");
const exitBtn = $("exitBtn");
const resetBtn = $("resetBtn");
const recBtn = $("recBtn");
const cubeSizeInput = $("cubeSize");
const hudFrames = $("hudFrames");

// ----- wire format constants (mirror tools/serve.py) ----------------------
const FRAME_HEADER_SIZE = 224;
const FORMAT_UINT16_LA = 0;
const FORMAT_FLOAT32 = 1;
const COLOR_NONE = 0;
const COLOR_RGBA8 = 1;

// ----- session-level state -------------------------------------------------
let session = null;
let xrRefSpace = null;
let gl = null;
let xrCanvas = null;

let recording = false;
let fetchInFlight = false;
let framesSent = 0;
let framesSkipped = 0;

// Camera-image capture (rendered into our framebuffer + read back as RGBA8).
let xrBinding = null;
let camProgram = null, camUCam = null, camVao = null, camFbo = null,
    camOutTex = null, camPixels = null, camOutW = 0, camOutH = 0,
    cameraSetupTried = false;

// ----- support check -------------------------------------------------------
async function checkSupport() {
  if (!("xr" in navigator)) {
    return { ok: false, reason: "navigator.xr missing — WebXR not available." };
  }
  let ok = false;
  try { ok = await navigator.xr.isSessionSupported("immersive-ar"); }
  catch (e) { return { ok: false, reason: `isSessionSupported failed: ${e}` }; }
  if (!ok) return { ok: false, reason: "immersive-ar not supported on this device." };
  return { ok: true, reason: "AR available — tap Enter AR, then Start recording." };
}

async function init() {
  const s = await checkSupport();
  gateMsg.textContent = s.reason;
  if (s.ok) {
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
      requiredFeatures: ["local-floor", "depth-sensing"],
      optionalFeatures: ["camera-access", "dom-overlay"],
      depthSensing: {
        usagePreference: ["cpu-optimized"],
        dataFormatPreference: ["luminance-alpha", "float32"],
      },
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

  try { xrBinding = new XRWebGLBinding(session, gl); }
  catch (e) { console.warn("XRWebGLBinding ctor failed:", e); xrBinding = null; }

  hudStatus.textContent = "ready · tap Start to record";
  gate.style.display = "none";
  overlay.style.display = "flex";

  session.addEventListener("end", onSessionEnd);
  exitBtn.addEventListener("click", () => session && session.end());
  resetBtn.addEventListener("click", onReset);
  recBtn.addEventListener("click", onToggleRecording);

  session.requestAnimationFrame(onXRFrame);
}

function onSessionEnd() {
  console.log("session ended");
  session = null;
  xrRefSpace = null;
  xrBinding = null;
  recording = false;
  fetchInFlight = false;
  if (xrCanvas?.parentNode) xrCanvas.parentNode.removeChild(xrCanvas);
  xrCanvas = null;
  gl = null;
  camProgram = camVao = camFbo = camOutTex = camPixels = null;
  camOutW = camOutH = 0; cameraSetupTried = false;
  overlay.style.display = "none";
  gate.style.display = "";
  gateMsg.textContent = "Session ended. Tap Enter AR to retry.";
  startBtn.disabled = false;
}

// ----- recording control --------------------------------------------------
async function onToggleRecording() {
  if (!recording) {
    recBtn.disabled = true;
    try {
      const cubeSize = Math.max(0.05, Math.min(2.0, +cubeSizeInput.value || 0.25));
      const r = await fetch("/cubes/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cube_size: cubeSize, reset: true }),
      });
      if (!r.ok) {
        hudStatus.textContent = `start failed ${r.status}`;
        recBtn.disabled = false;
        return;
      }
      const meta = await r.json();
      cubeSizeInput.value = String(meta.cube_size);
      cubeSizeInput.disabled = true;
      framesSent = 0;
      framesSkipped = 0;
      hudFrames.textContent = "0";
      recording = true;
      recBtn.classList.add("recording");
      recBtn.textContent = "Stop recording";
      hudStatus.textContent = `recording · cube=${meta.cube_size.toFixed(2)}m grid=${meta.shape.join("×")}`;
    } catch (e) {
      hudStatus.textContent = "start network err";
      console.error("/cubes/start failed", e);
    } finally {
      recBtn.disabled = false;
    }
  } else {
    recBtn.disabled = true;
    recording = false;
    try {
      const r = await fetch("/cubes/stop", { method: "POST" });
      if (r.ok) {
        const meta = await r.json();
        hudStatus.textContent = `stopped after ${meta.frames} frames`;
      } else {
        hudStatus.textContent = `stop failed ${r.status}`;
      }
    } catch (e) {
      hudStatus.textContent = "stop network err";
    } finally {
      recBtn.classList.remove("recording");
      recBtn.textContent = "Start recording";
      cubeSizeInput.disabled = false;
      recBtn.disabled = false;
    }
  }
}

async function onReset() {
  resetBtn.disabled = true;
  try {
    const r = await fetch("/cubes/reset", { method: "POST" });
    hudStatus.textContent = r.ok ? "grid wiped" : `reset failed ${r.status}`;
  } catch (e) {
    hudStatus.textContent = "reset network err";
  } finally {
    resetBtn.disabled = false;
  }
}

// ----- camera-image grab --------------------------------------------------
function compileShader(type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`shader compile failed: ${log}`);
  }
  return sh;
}

function ensureCameraSetup(width, height) {
  if (camOutW === width && camOutH === height && camProgram) return true;
  if (!camProgram) {
    const vs = `#version 300 es
      out vec2 vUV;
      void main() {
        float x = (gl_VertexID == 1) ? 3.0 : -1.0;
        float y = (gl_VertexID == 2) ? 3.0 : -1.0;
        vUV = vec2((x + 1.0) * 0.5, (y + 1.0) * 0.5);
        gl_Position = vec4(x, y, 0.0, 1.0);
      }`;
    const fs = `#version 300 es
      precision mediump float;
      in vec2 vUV;
      uniform sampler2D uCam;
      out vec4 outColor;
      void main() { outColor = texture(uCam, vUV); }`;
    const v = compileShader(gl.VERTEX_SHADER, vs);
    const f = compileShader(gl.FRAGMENT_SHADER, fs);
    const p = gl.createProgram();
    gl.attachShader(p, v); gl.attachShader(p, f);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error(`cam link failed: ${gl.getProgramInfoLog(p)}`);
    }
    camProgram = p;
    camUCam = gl.getUniformLocation(p, "uCam");
    camVao = gl.createVertexArray();
  }
  if (camOutTex) gl.deleteTexture(camOutTex);
  if (camFbo) gl.deleteFramebuffer(camFbo);
  camOutTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, camOutTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  camFbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, camFbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, camOutTex, 0);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    console.error("cam FBO incomplete");
    return false;
  }
  camPixels = new Uint8Array(width * height * 4);
  camOutW = width; camOutH = height;
  return true;
}

function captureCameraRGBA(view, width, height) {
  if (!xrBinding || !view.camera) return null;
  let camTex;
  try { camTex = xrBinding.getCameraImage(view.camera); }
  catch (e) {
    if (!cameraSetupTried) { cameraSetupTried = true; console.warn("getCameraImage failed:", e); }
    return null;
  }
  if (!camTex) return null;
  if (!ensureCameraSetup(width, height)) return null;
  const baseFb = session.renderState.baseLayer.framebuffer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, camFbo);
  gl.viewport(0, 0, width, height);
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.BLEND);
  gl.useProgram(camProgram);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, camTex);
  gl.uniform1i(camUCam, 0);
  gl.bindVertexArray(camVao);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.bindVertexArray(null);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, camPixels);
  gl.bindFramebuffer(gl.FRAMEBUFFER, baseFb);
  return camPixels;
}

// ----- per-frame loop ------------------------------------------------------
function onXRFrame(time, frame) {
  if (!session) return;
  session.requestAnimationFrame(onXRFrame);

  const baseLayer = session.renderState.baseLayer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, baseLayer.framebuffer);
  gl.viewport(0, 0, baseLayer.framebufferWidth, baseLayer.framebufferHeight);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  if (!recording) return;

  const pose = frame.getViewerPose(xrRefSpace);
  if (!pose || pose.views.length === 0) return;
  if (pose.emulatedPosition) {
    framesSkipped++;
    return;
  }

  const view = pose.views[0];
  let depthInfo = null;
  try { depthInfo = frame.getDepthInformation(view); } catch {}
  if (!depthInfo) return;

  if (!fetchInFlight) {
    const fmtCode = (session.depthDataFormat === "float32") ? FORMAT_FLOAT32 : FORMAT_UINT16_LA;
    const bpp = fmtCode === FORMAT_FLOAT32 ? 4 : 2;
    captureAndSend(view, depthInfo, fmtCode, bpp);
  }
}

// ----- /frame submission ---------------------------------------------------
const COLOR_OVERSAMPLE_FALLBACK = 2;

async function captureAndSend(view, depthInfo, formatCode, bytesPerPixel) {
  fetchInFlight = true;
  try { await captureAndSendInner(view, depthInfo, formatCode, bytesPerPixel); }
  finally { fetchInFlight = false; }
}

async function captureAndSendInner(view, depthInfo, formatCode, bytesPerPixel) {
  const w = depthInfo.width, h = depthInfo.height;
  const depthBytes = w * h * bytesPerPixel;

  // Native camera resolution when WebXR exposes it; else 2× depth as a
  // sane fallback (camera-access denied, etc.).
  let cw, ch;
  if (view.camera && view.camera.width > 0 && view.camera.height > 0) {
    cw = view.camera.width;
    ch = view.camera.height;
  } else {
    cw = w * COLOR_OVERSAMPLE_FALLBACK;
    ch = h * COLOR_OVERSAMPLE_FALLBACK;
  }
  const colorPixels = captureCameraRGBA(view, cw, ch);
  const colorBytes = colorPixels ? cw * ch * 4 : 0;
  const colorFormat = colorPixels ? COLOR_RGBA8 : COLOR_NONE;
  const total = FRAME_HEADER_SIZE + depthBytes + colorBytes;

  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);
  let off = 0;
  const writeMat = (m) => { for (let i = 0; i < 16; i++) { dv.setFloat32(off, m[i], true); off += 4; } };
  writeMat(view.transform.matrix);
  writeMat(view.projectionMatrix);
  writeMat(depthInfo.normDepthBufferFromNormView.matrix);
  dv.setUint32(off, w, true); off += 4;
  dv.setUint32(off, h, true); off += 4;
  dv.setFloat32(off, depthInfo.rawValueToMeters, true); off += 4;
  dv.setUint32(off, formatCode, true); off += 4;
  dv.setUint32(off, colorPixels ? cw : 0, true); off += 4;
  dv.setUint32(off, colorPixels ? ch : 0, true); off += 4;
  dv.setUint32(off, colorFormat, true); off += 4;
  dv.setUint32(off, colorBytes, true); off += 4;

  const xrData = depthInfo.data;
  const srcBuf = xrData instanceof ArrayBuffer ? xrData : xrData.buffer;
  const srcOff = xrData instanceof ArrayBuffer ? 0 : xrData.byteOffset;
  if (xrData.byteLength < depthBytes) { console.warn("depth too small"); return; }
  new Uint8Array(buf, FRAME_HEADER_SIZE, depthBytes)
    .set(new Uint8Array(srcBuf, srcOff, depthBytes));
  if (colorPixels) {
    new Uint8Array(buf, FRAME_HEADER_SIZE + depthBytes, colorBytes).set(colorPixels);
  }

  try {
    const r = await fetch("/frame", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: buf,
    });
    if (r.ok) {
      framesSent++;
      hudFrames.textContent = String(framesSent);
    } else {
      hudStatus.textContent = `HTTP ${r.status}`;
    }
  } catch (e) {
    hudStatus.textContent = "network err";
  }
}

init();
