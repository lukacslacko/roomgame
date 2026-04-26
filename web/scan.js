// Phone-side WebXR depth + colour scanner.
//
// Each tap captures one frame:
//   - depth   : the WebXR Depth Sensing CPU buffer (luminance-alpha or float32)
//   - colour  : the camera image (camera-access feature), downsampled to the
//               same resolution as the depth buffer via a tiny GL pipeline
//   - pose    : viewMatrix, projectionMatrix, normDepthBufferFromNormView
// All packed into one binary body and POSTed to /frame on the laptop. If
// camera-access isn't granted we still send depth (color_format=NONE) so
// scanning works on devices without it.
//
// No three.js — the AR view is camera passthrough only; we manage the GL
// context and a single shader directly. The HUD is HTML via dom-overlay.

import { hookConsoleToServer } from "./logbridge.js";
hookConsoleToServer();

const FRAME_HEADER_SIZE = 224;       // matches FRAME_HEADER_FMT in tools/serve.py
const FORMAT_UINT16_LA = 0;
const FORMAT_FLOAT32 = 1;
const COLOR_NONE = 0;
const COLOR_RGBA8 = 1;

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
const depthOverlay = $("depthOverlay");
const depthToggleBtn = $("depthToggleBtn");
let depthOverlayCtx = null;
let depthOverlayImageData = null;
let depthOverlayVisible = true;
depthToggleBtn?.addEventListener("click", () => {
  depthOverlayVisible = !depthOverlayVisible;
  depthOverlay.classList.toggle("hidden", !depthOverlayVisible);
});

let session = null;
let xrRefSpace = null;
let gl = null;
let xrCanvas = null;
let captureRequested = false;
let framesSent = 0;
let depthBufferLogged = false;

// Camera-access state. Initialised lazily on first frame that has both a
// depth buffer (to know the target size) and a successful getCameraImage.
let xrBinding = null;            // XRWebGLBinding
let camProgram = null;           // GL program
let camProgramUCam = null;       // sampler uniform location
let camVao = null;               // empty VAO for the fullscreen-triangle trick
let camFbo = null;               // framebuffer with attached output texture
let camOutTex = null;            // RGBA8 output texture
let camPixels = null;            // Uint8Array readback buffer
let camOutW = 0, camOutH = 0;
let cameraSetupTried = false;
let cameraAvailable = false;     // true once we've successfully grabbed one image

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
  return { ok: true, reason: "immersive-ar available — depth + camera support confirmed on session start." };
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
      // camera-access: needed for colour. dom-overlay: needed for the HUD.
      optionalFeatures: ["camera-access", "dom-overlay"],
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

  const fmt = session.depthDataFormat || "luminance-alpha";
  const formatCode = fmt === "float32" ? FORMAT_FLOAT32 : FORMAT_UINT16_LA;
  const bytesPerPixel = formatCode === FORMAT_FLOAT32 ? 4 : 2;

  // Bind for camera image reads. We construct it even if camera-access wasn't
  // granted; getCameraImage will throw and we'll fall back gracefully.
  try {
    xrBinding = new XRWebGLBinding(session, gl);
  } catch (e) {
    console.warn("XRWebGLBinding ctor failed:", e);
    xrBinding = null;
  }

  console.log(`session ready: depthDataFormat=${fmt} usage=${session.depthUsage}`);
  hudStatus.textContent = `depth=${fmt} color=?`;

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
  xrBinding = null;
  if (xrCanvas && xrCanvas.parentNode) xrCanvas.parentNode.removeChild(xrCanvas);
  xrCanvas = null;
  gl = null;
  // Drop GL resources tied to that context.
  camProgram = camVao = camFbo = camOutTex = camPixels = null;
  camOutW = camOutH = 0;
  cameraSetupTried = false;
  cameraAvailable = false;
  depthBufferLogged = false;
  overlay.style.display = "none";
  gate.style.display = "";
  gateMsg.textContent = "Session ended. Tap Enter AR to scan again.";
  startBtn.disabled = false;
}

// ----- camera capture helpers ---------------------------------------------

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
  // (Re)build the FBO+texture for the requested size.
  if (!camProgram) {
    // GLSL ES 3.00. The single-triangle trick draws a triangle covering the
    // whole NDC; gl_VertexID 0,1,2 → (-1,-1),(3,-1),(-1,3). vUV is in [0,1].
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
      throw new Error(`link failed: ${gl.getProgramInfoLog(p)}`);
    }
    camProgram = p;
    camProgramUCam = gl.getUniformLocation(p, "uCam");
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
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    console.error(`color FBO incomplete: 0x${status.toString(16)}`);
    return false;
  }
  camPixels = new Uint8Array(width * height * 4);
  camOutW = width;
  camOutH = height;
  return true;
}

/** Returns the camera image as RGBA8 bytes at (width, height), or null if
 *  camera-access isn't available / the texture isn't ready this frame. */
function captureCameraRGBA(view, width, height) {
  if (!xrBinding || !view.camera) return null;
  let camTex = null;
  try {
    camTex = xrBinding.getCameraImage(view.camera);
  } catch (e) {
    if (!cameraSetupTried) {
      cameraSetupTried = true;
      console.warn("getCameraImage failed:", e);
    }
    return null;
  }
  if (!camTex) return null;
  if (!ensureCameraSetup(width, height)) return null;

  // Save current FB+viewport then restore — XR session expects baseLayer.fb.
  const baseFb = session.renderState.baseLayer.framebuffer;

  gl.bindFramebuffer(gl.FRAMEBUFFER, camFbo);
  gl.viewport(0, 0, width, height);
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.BLEND);
  gl.useProgram(camProgram);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, camTex);
  gl.uniform1i(camProgramUCam, 0);
  gl.bindVertexArray(camVao);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.bindVertexArray(null);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, camPixels);

  // Restore baseLayer FBO so XR doesn't notice we ran a side-pass.
  gl.bindFramebuffer(gl.FRAMEBUFFER, baseFb);

  cameraAvailable = true;
  return camPixels;
}

// ----- depth overlay ------------------------------------------------------
//
// Paints the raw cpu-depth buffer onto a 2D canvas with bands alternating
// dark/light per 1 m of depth. Because the depth buffer is in landscape
// sensor orientation but the phone is held portrait, we transpose while
// drawing so the overlay roughly aligns with the camera passthrough behind it
// (assumes a 90° rotation between buffer u and view v, which is what we see
// in practice on Pixels — the normDepthBufferFromNormView matrix had a
// `(0, 1, -0.82, 0)` rotation block when we inspected it).
//
// Colours:
//   depth == 0       → magenta (no measurement / out of confidence range)
//   d ∈ [0, 1) m     → dark
//   d ∈ [1, 2) m     → light
//   d ∈ [2, 3) m     → dark
//   ...
//   d ≥ overlayMaxM  → black (clamped, treated as "far")
function renderDepthOverlay(depthInfo) {
  const W = depthInfo.width;
  const H = depthInfo.height;
  // Canvas size is the *rotated* depth-buffer shape so 1 buffer pixel = 1
  // canvas pixel after the transpose, with no resampling.
  if (depthOverlay.width !== H || depthOverlay.height !== W) {
    depthOverlay.width = H;
    depthOverlay.height = W;
    depthOverlayCtx = depthOverlay.getContext("2d");
    depthOverlayImageData = depthOverlayCtx.createImageData(H, W);
  }
  const ctx = depthOverlayCtx;
  const imgData = depthOverlayImageData;
  const px = imgData.data;

  const xrData = depthInfo.data;
  const buf = xrData instanceof ArrayBuffer ? xrData : xrData.buffer;
  const off = xrData instanceof ArrayBuffer ? 0 : xrData.byteOffset;
  // We only handle the luminance-alpha format here (the only one Chrome
  // returns in practice). Float32 would need a Float32Array source.
  const u16 = new Uint16Array(buf, off, W * H);
  const r2m = depthInfo.rawValueToMeters;

  const overlayMaxM = 8.0;
  for (let by = 0; by < H; by++) {
    for (let bx = 0; bx < W; bx++) {
      const d = u16[by * W + bx] * r2m;
      // Counter-clockwise 90° rotation: buffer (bx, by) → output (by, W-1-bx).
      const ox = by;
      const oy = W - 1 - bx;
      const dst = (oy * H + ox) * 4;
      let r, g, b, a;
      if (d <= 0.0) {
        r = 220; g = 0; b = 220; a = 200;        // no measurement
      } else if (d >= overlayMaxM) {
        r = 0; g = 0; b = 0; a = 200;            // far / out of range
      } else {
        const band = Math.floor(d);
        const v = (band & 1) === 0 ? 35 : 220;   // alternate dark/light
        r = v; g = v; b = v; a = 220;
      }
      px[dst]     = r;
      px[dst + 1] = g;
      px[dst + 2] = b;
      px[dst + 3] = a;
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

// ----- main per-frame loop ------------------------------------------------

function onXRFrame(time, frame, formatCode, bytesPerPixel) {
  if (!session) return;
  session.requestAnimationFrame((t, f) => onXRFrame(t, f, formatCode, bytesPerPixel));

  const baseLayer = session.renderState.baseLayer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, baseLayer.framebuffer);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const pose = frame.getViewerPose(xrRefSpace);
  if (!pose || pose.views.length === 0) return;
  const view = pose.views[0];

  let depthInfo = null;
  try {
    depthInfo = frame.getDepthInformation(view);
  } catch {
    return;
  }
  if (!depthInfo) {
    hudDepth.textContent = "(waiting)";
    return;
  }

  hudDepth.textContent = `${depthInfo.width}×${depthInfo.height}`;

  if (depthOverlayVisible) renderDepthOverlay(depthInfo);

  if (!depthBufferLogged) {
    depthBufferLogged = true;
    const d = depthInfo.data;
    console.log(
      `depth buffer: ${depthInfo.width}×${depthInfo.height} ` +
      `data=${d?.constructor?.name || typeof d} byteLength=${d?.byteLength} ` +
      `rawValueToMeters=${depthInfo.rawValueToMeters} ` +
      `cameraGranted=${!!view.camera}`
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
  const depthBytes = w * h * bytesPerPixel;

  // Capture colour at depth resolution. If camera-access isn't usable,
  // colorPixels is null and we send a NONE-format colour record.
  const colorPixels = captureCameraRGBA(view, w, h);
  const colorBytes = colorPixels ? w * h * 4 : 0;
  const colorFormat = colorPixels ? COLOR_RGBA8 : COLOR_NONE;
  const total = FRAME_HEADER_SIZE + depthBytes + colorBytes;

  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);
  let off = 0;
  const writeMat = (m) => { for (let i = 0; i < 16; i++) { dv.setFloat32(off, m[i], true); off += 4; } };

  writeMat(view.transform.matrix);
  writeMat(view.projectionMatrix);
  writeMat(depthInfo.normDepthBufferFromNormView.matrix);
  dv.setUint32(off, w, true); off += 4;                                  // depth_width
  dv.setUint32(off, h, true); off += 4;                                  // depth_height
  dv.setFloat32(off, depthInfo.rawValueToMeters, true); off += 4;         // rawValueToMeters
  dv.setUint32(off, formatCode, true); off += 4;                          // depth_format
  dv.setUint32(off, colorPixels ? w : 0, true); off += 4;                 // color_width
  dv.setUint32(off, colorPixels ? h : 0, true); off += 4;                 // color_height
  dv.setUint32(off, colorFormat, true); off += 4;                         // color_format
  dv.setUint32(off, colorBytes, true); off += 4;                          // color_byte_length
  if (off !== FRAME_HEADER_SIZE) {
    console.error(`header size mismatch: wrote ${off}, expected ${FRAME_HEADER_SIZE}`);
    return;
  }

  // Depth payload.
  const xrData = depthInfo.data;
  const srcBuf = xrData instanceof ArrayBuffer ? xrData : xrData.buffer;
  const srcOff = xrData instanceof ArrayBuffer ? 0 : xrData.byteOffset;
  if (xrData.byteLength < depthBytes) {
    console.warn(`depth buffer too small: ${xrData.byteLength}B < ${depthBytes}B`);
    return;
  }
  new Uint8Array(buf, FRAME_HEADER_SIZE, depthBytes)
    .set(new Uint8Array(srcBuf, srcOff, depthBytes));

  // Colour payload (if any).
  if (colorPixels) {
    new Uint8Array(buf, FRAME_HEADER_SIZE + depthBytes, colorBytes).set(colorPixels);
  }

  hudTap.textContent = "uploading…";
  hudStatus.textContent = `depth=${formatCode === FORMAT_UINT16_LA ? "u16" : "f32"} color=${colorPixels ? "yes" : "no"}`;

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
