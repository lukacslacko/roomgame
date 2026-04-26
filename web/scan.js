// Phone-side WebXR depth scanner.
//
// Captures one depth frame per tap, packs viewMatrix + projectionMatrix +
// normDepthBufferFromNormView + dimensions + format into a 208-byte binary
// header, appends the raw depth payload, and POSTs it to /frame on the laptop.
//
// No three.js — the AR view is camera passthrough only; we just need the
// WebGL context to satisfy XRWebGLLayer. The HUD is HTML via dom-overlay.

import { hookConsoleToServer } from "./logbridge.js";
hookConsoleToServer();

const FRAME_HEADER_SIZE = 208;
const FORMAT_UINT16_LA = 0;
const FORMAT_FLOAT32 = 1;

const $ = (id) => document.getElementById(id);
const gate = $("gate");
const gateMsg = $("gateMsg");
const startBtn = $("startBtn");
const overlay = $("xrOverlay");
const hudStatus = $("hudStatus");
const hudDepth = $("hudDepth");
const hudFrames = $("hudFrames");
const hudLast = $("hudLast");
const hudTap = $("hudTap");
const exitBtn = $("exitBtn");

let session = null;
let xrRefSpace = null;
let gl = null;
let xrCanvas = null;
let captureRequested = false;
let framesSent = 0;
let lastFrameAt = 0;
let depthBufferLogged = false;

async function checkSupport() {
  if (!("xr" in navigator)) {
    return { ok: false, reason: "navigator.xr missing — WebXR not available in this browser." };
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
  // We can't pre-test depth-sensing without starting a session. Best-effort
  // hint: WebXR Depth Sensing is Android Chrome / Quest browser only.
  return { ok: true, reason: "immersive-ar available — depth support will be confirmed on session start." };
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
  gl = xrCanvas.getContext("webgl2", { xrCompatible: true, alpha: true, antialias: false, preserveDrawingBuffer: false });
  if (!gl) {
    gateMsg.textContent = "WebGL2 unavailable — can't open an AR session.";
    startBtn.disabled = false;
    return;
  }

  try {
    session = await navigator.xr.requestSession("immersive-ar", {
      requiredFeatures: ["local-floor", "depth-sensing"],
      optionalFeatures: ["dom-overlay"],
      depthSensing: {
        usagePreference: ["cpu-optimized"],
        dataFormatPreference: ["luminance-alpha", "float32"],
      },
      domOverlay: { root: overlay },
    });
  } catch (e) {
    gateMsg.textContent = `Failed to start AR (depth-sensing required): ${e.message || e}`;
    console.error("requestSession failed", e);
    startBtn.disabled = false;
    return;
  }

  await gl.makeXRCompatible();
  const baseLayer = new XRWebGLLayer(session, gl);
  await session.updateRenderState({ baseLayer });
  xrRefSpace = await session.requestReferenceSpace("local-floor");

  // The phone may have selected one of the two formats we offered. Save it
  // so we send the right bytes-per-pixel and format code on each frame.
  const fmt = session.depthDataFormat || "luminance-alpha";
  const formatCode = fmt === "float32" ? FORMAT_FLOAT32 : FORMAT_UINT16_LA;
  const bytesPerPixel = formatCode === FORMAT_FLOAT32 ? 4 : 2;
  console.log(`session ready: depthDataFormat=${fmt} usage=${session.depthUsage}`);
  hudStatus.textContent = `depth=${fmt}`;

  session.addEventListener("end", onSessionEnd);
  session.addEventListener("select", () => { captureRequested = true; });
  exitBtn.addEventListener("click", () => session && session.end());

  gate.style.display = "none";
  overlay.style.display = "flex";

  session.requestAnimationFrame((t, f) => onXRFrame(t, f, formatCode, bytesPerPixel));
}

function onSessionEnd() {
  console.log("session ended");
  session = null;
  xrRefSpace = null;
  if (xrCanvas && xrCanvas.parentNode) xrCanvas.parentNode.removeChild(xrCanvas);
  xrCanvas = null;
  gl = null;
  overlay.style.display = "none";
  gate.style.display = "";
  gateMsg.textContent = "Session ended. Tap Enter AR to scan again.";
  startBtn.disabled = false;
}

function onXRFrame(time, frame, formatCode, bytesPerPixel) {
  if (!session) return;
  session.requestAnimationFrame((t, f) => onXRFrame(t, f, formatCode, bytesPerPixel));

  const baseLayer = session.renderState.baseLayer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, baseLayer.framebuffer);
  // Clear to fully transparent so passthrough shows through; the spec says
  // immersive-ar's blend mode is alpha-blend on most devices.
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const pose = frame.getViewerPose(xrRefSpace);
  if (!pose || pose.views.length === 0) return;
  const view = pose.views[0];

  let depthInfo = null;
  try {
    depthInfo = frame.getDepthInformation(view);
  } catch (e) {
    // Some implementations throw if depth isn't ready yet.
    return;
  }
  if (!depthInfo) {
    hudDepth.textContent = "(waiting)";
    return;
  }

  hudDepth.textContent = `${depthInfo.width}×${depthInfo.height}`;

  if (!depthBufferLogged) {
    depthBufferLogged = true;
    const d = depthInfo.data;
    console.log(
      `depth buffer: ${depthInfo.width}×${depthInfo.height} ` +
      `data=${d?.constructor?.name || typeof d} byteLength=${d?.byteLength} ` +
      `rawValueToMeters=${depthInfo.rawValueToMeters}`
    );
  }

  if (captureRequested) {
    captureRequested = false;
    captureAndSend(view, depthInfo, formatCode, bytesPerPixel);
  }
}

async function captureAndSend(view, depthInfo, formatCode, bytesPerPixel) {
  const w = depthInfo.width;
  const h = depthInfo.height;
  const payloadBytes = w * h * bytesPerPixel;
  const total = FRAME_HEADER_SIZE + payloadBytes;

  // Build the wire buffer.
  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);
  let off = 0;

  const writeMat = (m) => {
    for (let i = 0; i < 16; i++) {
      dv.setFloat32(off, m[i], true);
      off += 4;
    }
  };

  // viewMatrix = world_from_view, column-major
  writeMat(view.transform.matrix);
  // projectionMatrix
  writeMat(view.projectionMatrix);
  // normDepthBufferFromNormView (XRRigidTransform.matrix)
  writeMat(depthInfo.normDepthBufferFromNormView.matrix);

  dv.setUint32(off, w, true); off += 4;
  dv.setUint32(off, h, true); off += 4;
  dv.setFloat32(off, depthInfo.rawValueToMeters, true); off += 4;
  dv.setUint32(off, formatCode, true); off += 4;
  if (off !== FRAME_HEADER_SIZE) {
    console.error(`header size mismatch: wrote ${off} bytes, expected ${FRAME_HEADER_SIZE}`);
    return;
  }

  // Copy raw depth bytes from the XR buffer into the wire buffer.
  // Chrome's XRCPUDepthInformation.data is an ArrayBuffer per the WebXR
  // Depth Sensing IDL, but other implementations could plausibly hand back
  // a TypedArray. Handle both. (When this was broken it returned all-zero
  // payloads silently, since `new Uint8Array(undefined, undefined, N)`
  // is `new Uint8Array(0)` — wrong but doesn't throw.)
  const xrData = depthInfo.data;
  const srcBuf = xrData instanceof ArrayBuffer ? xrData : xrData.buffer;
  const srcOff = xrData instanceof ArrayBuffer ? 0 : xrData.byteOffset;
  const srcLen = xrData.byteLength;
  if (srcLen < payloadBytes) {
    console.warn(`depth buffer too small: ${srcLen}B < expected ${payloadBytes}B (${depthInfo.width}×${depthInfo.height}×${bytesPerPixel}); skipping frame`);
    return;
  }
  const src = new Uint8Array(srcBuf, srcOff, payloadBytes);
  new Uint8Array(buf, FRAME_HEADER_SIZE, payloadBytes).set(src);

  hudTap.textContent = "uploading…";
  const t0 = performance.now();
  try {
    const r = await fetch("/frame", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: buf,
    });
    const elapsed = (performance.now() - t0) | 0;
    if (!r.ok) {
      const text = await r.text();
      hudLast.textContent = `HTTP ${r.status}`;
      console.error(`/frame ${r.status}: ${text}`);
    } else {
      framesSent++;
      lastFrameAt = Date.now();
      hudFrames.textContent = String(framesSent);
      hudLast.textContent = `${total}B in ${elapsed}ms`;
    }
  } catch (e) {
    hudLast.textContent = "network err";
    console.error("/frame upload failed", e);
  } finally {
    hudTap.textContent = "tap anywhere to capture a frame";
  }
}

init();
