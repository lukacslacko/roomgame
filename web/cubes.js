// Occupancy-cubes phone scanner + viewer.
//
// Two responsibilities in one page:
//
//   1. Recording. While the user has tapped "Start recording", every XR
//      frame is packaged (depth + pose + colour) and POSTed to /frame
//      exactly the same way scan.js does. The server-side cube grid
//      (tools/cubes.py) consumes those frames if /cubes/start has been
//      called (we do that automatically on Start).
//
//   2. Live overlay. Every ~0.6 s we GET /cubes/state and update an
//      instanced cube mesh that renders on top of the camera passthrough.
//
// Two render modes:
//
//   "plain"     — cubes drawn with alpha = mix slider. Slider 0 → invisible
//                 (passthrough only); slider 1 → opaque cubes hide
//                 passthrough behind themselves.
//   "occluded"  — additionally samples the live WebXR depth buffer per
//                 fragment; the cube fragment is discarded where the
//                 measured depth says a real surface is closer than the
//                 cube. Slider still controls cube alpha where visible.
//
// No three.js. Custom WebGL2 with one cube program (instanced) plus the
// XR baseLayer's automatic camera passthrough.

import { hookConsoleToServer } from "./logbridge.js";
hookConsoleToServer();

// ----- DOM ----------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const gate = $("gate");
const gateMsg = $("gateMsg");
const startBtn = $("startBtn");
const overlay = $("xrOverlay");
const hudStatus = $("hudStatus");
const exitBtn = $("exitBtn");
const resetBtn = $("resetBtn");
const recBtn = $("recBtn");
const mixSlider = $("mixSlider");
const mixVal = $("mixVal");
const modeSel = $("modeSel");
const threshSlider = $("threshSlider");
const threshVal = $("threshVal");
const cubeSizeInput = $("cubeSize");
const hudFrames = $("hudFrames");
const hudCubes = $("hudCubes");

mixSlider.addEventListener("input", () => { mixVal.textContent = (+mixSlider.value).toFixed(2); });
threshSlider.addEventListener("input", () => { threshVal.textContent = (+threshSlider.value).toFixed(2); });

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

// Camera-image capture pipeline (lifted from scan.js, slimmed for our use).
let xrBinding = null;
let camProgram = null, camUCam = null, camVao = null, camFbo = null,
    camOutTex = null, camPixels = null, camOutW = 0, camOutH = 0,
    cameraSetupTried = false, cameraAvailable = false;

// Cube rendering pipeline.
let cubeProgram = null;
let cubeProgInfo = null;
let cubeUnitVao = null;        // VAO for the unit-cube mesh + instance attribs
let cubeIndexBuf = null;
let cubeInstanceBuf = null;
let cubeInstanceCount = 0;
let cubeInstanceCapacity = 0;

// CPU depth → GPU texture for the occlusion mode.
let depthTex = null;
let depthTexW = 0, depthTexH = 0;
let lastDepthInfo = null;      // most recent XRCPUDepthInformation

// /cubes/state polling.
let lastStatePollAt = 0;
const STATE_POLL_INTERVAL_MS = 700;
let cubeGridSize = 0.5;
let lastShape = null;
let lastWorldMin = [0, 0, 0];

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

  buildCubeProgram();
  buildUnitCubeMesh();

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
  camOutW = camOutH = 0; cameraSetupTried = false; cameraAvailable = false;
  cubeProgram = null; cubeUnitVao = null; cubeIndexBuf = null;
  cubeInstanceBuf = null; cubeInstanceCount = 0; cubeInstanceCapacity = 0;
  depthTex = null; depthTexW = 0; depthTexH = 0;
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
      const cubeSize = Math.max(0.05, Math.min(2.0, +cubeSizeInput.value || 0.5));
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
      cubeGridSize = meta.cube_size;
      cubeSizeInput.value = String(cubeGridSize);
      cubeSizeInput.disabled = true;
      lastShape = meta.shape;
      lastWorldMin = meta.world_min;
      framesSent = 0;
      framesSkipped = 0;
      hudFrames.textContent = "0";
      hudCubes.textContent = "0";
      recording = true;
      recBtn.classList.add("recording");
      recBtn.textContent = "Stop recording";
      hudStatus.textContent = `recording · cube=${cubeGridSize.toFixed(2)}m grid=${meta.shape.join("×")}`;
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
    if (r.ok) {
      cubeInstanceCount = 0;
      hudCubes.textContent = "0";
      hudStatus.textContent = "grid wiped";
    } else {
      hudStatus.textContent = `reset failed ${r.status}`;
    }
  } catch (e) {
    hudStatus.textContent = "reset network err";
  } finally {
    resetBtn.disabled = false;
  }
}

// ----- camera-image grab (slimmed copy of scan.js) ------------------------
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
  cameraAvailable = true;
  return camPixels;
}

// ----- cube rendering -----------------------------------------------------
//
// Pipeline:
//   - Static buffer with 8 unit-cube vertex positions and 36 indices for
//     the 12 triangles of the cube (two per face).
//   - Per-instance buffer holding (cx, cy, cz, occupancy) for each cube
//     to draw.
//   - Vertex shader: world_pos = instance_center + size * unit_pos;
//     gl_Position = projection * view_inverse * world_pos.
//     view_z (used by occlusion mode) is forwarded to the fragment shader.
//   - Fragment shader: in plain mode emits a colour ramp by occupancy with
//     alpha = uMixAlpha. In occlusion mode it samples the depth texture
//     at the fragment's screen position (mapped through normDepthBufferFromNormView)
//     and discards if the measured depth is closer than the cube's view-z
//     (i.e. there's a real surface in front of this cube fragment).
function buildCubeProgram() {
  const vs = `#version 300 es
    in vec3 aPos;            // unit cube vertex (each component ∈ [-0.5, 0.5])
    in vec4 aInstance;       // cube center in world coords + occupancy ratio
    uniform mat4 uProj;
    uniform mat4 uView;      // view_from_world (inverse of XR view matrix)
    uniform float uCubeSize;
    out float vOccupancy;
    out vec3 vViewPos;
    out vec4 vClip;
    void main() {
      vec3 worldPos = aInstance.xyz + uCubeSize * aPos;
      vec4 viewPos = uView * vec4(worldPos, 1.0);
      vViewPos = viewPos.xyz;
      vec4 clip = uProj * viewPos;
      vClip = clip;
      vOccupancy = aInstance.w;
      gl_Position = clip;
    }`;
  const fs = `#version 300 es
    precision highp float;
    in float vOccupancy;
    in vec3 vViewPos;
    in vec4 vClip;
    uniform float uMixAlpha;            // [0..1] global cube alpha
    uniform int   uMode;                // 0=plain, 1=depth-occluded
    uniform highp usampler2D uDepth;    // R16UI texture, raw uint16 depth values
    uniform float uRawToM;              // multiply texel by this for metres
    uniform vec2  uDepthSize;           // (W, H)
    uniform mat4  uNormDepthFromNormView; // 4x4 column-major
    out vec4 outColor;

    vec3 ramp(float x) {
      // viridis-ish ramp: low=teal/blue, high=yellow.
      x = clamp(x, 0.0, 1.0);
      vec3 lo = vec3(0.05, 0.30, 0.55);
      vec3 mid = vec3(0.20, 0.65, 0.45);
      vec3 hi = vec3(0.95, 0.85, 0.25);
      return (x < 0.5) ? mix(lo, mid, x * 2.0) : mix(mid, hi, (x - 0.5) * 2.0);
    }

    void main() {
      // Compute normalized view coords in [0, 1]^2 from the post-perspective
      // clip-space position. This corresponds to the view's normalised pixel
      // (origin bottom-left, Y up) — i.e. the same (u_v, v_v) the depth
      // mapping matrix expects.
      vec3 ndc = vClip.xyz / vClip.w;
      vec2 u_v = ndc.xy * 0.5 + 0.5;

      vec3 col = ramp(vOccupancy);

      if (uMode == 1) {
        // (u_v, v_v, 0, 1) → (u_d, v_d, *, *) via the WebXR depth buffer mapping.
        vec4 nd = uNormDepthFromNormView * vec4(u_v, 0.0, 1.0);
        vec2 ud_norm = nd.xy / nd.w;

        // Chrome stores the depth buffer with column index reversed relative
        // to the matrix's u_d direction (matches scan.js + fusion.py).
        ivec2 px = ivec2(
          int(floor((1.0 - ud_norm.x) * uDepthSize.x)),
          int(floor(ud_norm.y * uDepthSize.y))
        );
        bool inBuf = (
          ud_norm.x >= 0.0 && ud_norm.x <= 1.0 &&
          ud_norm.y >= 0.0 && ud_norm.y <= 1.0 &&
          px.x >= 0 && px.x < int(uDepthSize.x) &&
          px.y >= 0 && px.y < int(uDepthSize.y)
        );
        if (inBuf) {
          uint raw = texelFetch(uDepth, px, 0).r;
          float measured = float(raw) * uRawToM;
          // vViewPos.z is negative for points in front of the camera.
          float cubeDist = -vViewPos.z;
          // If measured > 0 and the surface is closer than the cube,
          // the cube fragment is occluded; emit alpha=0 so the camera
          // passthrough shows here unchanged.
          if (measured > 0.0 && measured < cubeDist - 0.02) {
            discard;
          }
        }
      }

      outColor = vec4(col, uMixAlpha);
    }`;
  const v = compileShader(gl.VERTEX_SHADER, vs);
  const f = compileShader(gl.FRAGMENT_SHADER, fs);
  cubeProgram = gl.createProgram();
  gl.attachShader(cubeProgram, v);
  gl.attachShader(cubeProgram, f);
  gl.linkProgram(cubeProgram);
  if (!gl.getProgramParameter(cubeProgram, gl.LINK_STATUS)) {
    throw new Error(`cube link failed: ${gl.getProgramInfoLog(cubeProgram)}`);
  }
  cubeProgInfo = {
    aPos: gl.getAttribLocation(cubeProgram, "aPos"),
    aInstance: gl.getAttribLocation(cubeProgram, "aInstance"),
    uProj: gl.getUniformLocation(cubeProgram, "uProj"),
    uView: gl.getUniformLocation(cubeProgram, "uView"),
    uCubeSize: gl.getUniformLocation(cubeProgram, "uCubeSize"),
    uMixAlpha: gl.getUniformLocation(cubeProgram, "uMixAlpha"),
    uMode: gl.getUniformLocation(cubeProgram, "uMode"),
    uDepth: gl.getUniformLocation(cubeProgram, "uDepth"),
    uRawToM: gl.getUniformLocation(cubeProgram, "uRawToM"),
    uDepthSize: gl.getUniformLocation(cubeProgram, "uDepthSize"),
    uNormDepthFromNormView: gl.getUniformLocation(cubeProgram, "uNormDepthFromNormView"),
  };
}

function buildUnitCubeMesh() {
  // Unit cube centred at origin, side 1: vertices at ±0.5 on each axis.
  const positions = new Float32Array([
    -0.5, -0.5, -0.5,   // 0
     0.5, -0.5, -0.5,   // 1
     0.5,  0.5, -0.5,   // 2
    -0.5,  0.5, -0.5,   // 3
    -0.5, -0.5,  0.5,   // 4
     0.5, -0.5,  0.5,   // 5
     0.5,  0.5,  0.5,   // 6
    -0.5,  0.5,  0.5,   // 7
  ]);
  const indices = new Uint16Array([
    0,1,2, 0,2,3,   // -Z face
    4,6,5, 4,7,6,   // +Z face
    0,4,5, 0,5,1,   // -Y face
    2,6,7, 2,7,3,   // +Y face
    1,5,6, 1,6,2,   // +X face
    0,3,7, 0,7,4,   // -X face
  ]);
  cubeUnitVao = gl.createVertexArray();
  gl.bindVertexArray(cubeUnitVao);

  const posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(cubeProgInfo.aPos);
  gl.vertexAttribPointer(cubeProgInfo.aPos, 3, gl.FLOAT, false, 0, 0);

  cubeInstanceBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, cubeInstanceBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(0), gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(cubeProgInfo.aInstance);
  gl.vertexAttribPointer(cubeProgInfo.aInstance, 4, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(cubeProgInfo.aInstance, 1);

  cubeIndexBuf = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cubeIndexBuf);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

  gl.bindVertexArray(null);
}

function setCubeInstances(instanceData /* Float32Array, 4 floats per cube */) {
  cubeInstanceCount = instanceData.length / 4;
  gl.bindBuffer(gl.ARRAY_BUFFER, cubeInstanceBuf);
  if (cubeInstanceCount > cubeInstanceCapacity) {
    gl.bufferData(gl.ARRAY_BUFFER, instanceData, gl.DYNAMIC_DRAW);
    cubeInstanceCapacity = cubeInstanceCount;
  } else {
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, instanceData);
  }
}

// ----- depth texture upload -----------------------------------------------
function ensureDepthTexture(W, H) {
  if (depthTex && depthTexW === W && depthTexH === H) return;
  if (depthTex) gl.deleteTexture(depthTex);
  depthTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, depthTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16UI, W, H, 0, gl.RED_INTEGER, gl.UNSIGNED_SHORT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  depthTexW = W; depthTexH = H;
}

function uploadDepthTexture(depthInfo) {
  const W = depthInfo.width, H = depthInfo.height;
  ensureDepthTexture(W, H);
  const xrData = depthInfo.data;
  const buf = xrData instanceof ArrayBuffer ? xrData : xrData.buffer;
  const off = xrData instanceof ArrayBuffer ? 0 : xrData.byteOffset;
  const u16 = new Uint16Array(buf, off, W * H);
  gl.bindTexture(gl.TEXTURE_2D, depthTex);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, W, H, gl.RED_INTEGER, gl.UNSIGNED_SHORT, u16);
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

  const pose = frame.getViewerPose(xrRefSpace);
  if (!pose || pose.views.length === 0) return;
  const view = pose.views[0];

  let depthInfo = null;
  try { depthInfo = frame.getDepthInformation(view); } catch {}
  if (depthInfo) {
    lastDepthInfo = depthInfo;
    uploadDepthTexture(depthInfo);
  }

  // Recording: send /frame submissions (depth + pose + colour).
  if (recording && depthInfo && !fetchInFlight) {
    if (pose.emulatedPosition) {
      framesSkipped++;
    } else {
      const fmtCode = (session.depthDataFormat === "float32") ? FORMAT_FLOAT32 : FORMAT_UINT16_LA;
      const bpp = fmtCode === FORMAT_FLOAT32 ? 4 : 2;
      captureAndSend(view, depthInfo, fmtCode, bpp);
    }
  }

  // Periodic /cubes/state poll.
  if (time - lastStatePollAt > STATE_POLL_INTERVAL_MS) {
    lastStatePollAt = time;
    pollCubeState();
  }

  // Render cubes.
  if (cubeInstanceCount > 0 && depthInfo && lastShape) {
    renderCubes(view, depthInfo);
  }
}

function renderCubes(view, depthInfo) {
  // WebXR exposes the inverse rigid transform directly; saves us inverting
  // a 4x4 in JS every frame. .matrix is column-major Float32Array(16).
  const invView = view.transform.inverse.matrix;

  gl.useProgram(cubeProgram);
  gl.bindVertexArray(cubeUnitVao);

  gl.uniformMatrix4fv(cubeProgInfo.uProj, false, view.projectionMatrix);
  gl.uniformMatrix4fv(cubeProgInfo.uView, false, invView);
  gl.uniform1f(cubeProgInfo.uCubeSize, cubeGridSize);
  gl.uniform1f(cubeProgInfo.uMixAlpha, +mixSlider.value);
  gl.uniform1i(cubeProgInfo.uMode, modeSel.value === "occluded" ? 1 : 0);

  if (depthTex) {
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, depthTex);
    gl.uniform1i(cubeProgInfo.uDepth, 0);
    gl.uniform2f(cubeProgInfo.uDepthSize, depthInfo.width, depthInfo.height);
    gl.uniform1f(cubeProgInfo.uRawToM, depthInfo.rawValueToMeters);
    gl.uniformMatrix4fv(cubeProgInfo.uNormDepthFromNormView, false,
                        depthInfo.normDepthBufferFromNormView.matrix);
  }

  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  gl.viewport(0, 0,
    session.renderState.baseLayer.framebufferWidth,
    session.renderState.baseLayer.framebufferHeight);
  gl.drawElementsInstanced(gl.TRIANGLES, 36, gl.UNSIGNED_SHORT, 0, cubeInstanceCount);
  gl.bindVertexArray(null);
}

// ----- /cubes/state polling -----------------------------------------------
async function pollCubeState() {
  try {
    const threshold = +threshSlider.value;
    const r = await fetch(`/cubes/state?threshold=${threshold}&min=2`);
    if (!r.ok) return;
    const s = await r.json();
    if (!s.ready) return;
    cubeGridSize = s.cube_size;
    lastShape = s.shape;
    lastWorldMin = s.world_min;
    hudCubes.textContent = String(s.cubes.length);
    if (s.cubes.length === 0) {
      cubeInstanceCount = 0;
      return;
    }
    // Build instance buffer: (cx, cy, cz, occupancy_ratio).
    const data = new Float32Array(s.cubes.length * 4);
    const wm = s.world_min, sz = s.cube_size;
    for (let k = 0; k < s.cubes.length; k++) {
      const [ix, iy, iz, occ, free] = s.cubes[k];
      const total = occ + free;
      const ratio = total > 0 ? occ / total : 0;
      data[k * 4 + 0] = wm[0] + (ix + 0.5) * sz;
      data[k * 4 + 1] = wm[1] + (iy + 0.5) * sz;
      data[k * 4 + 2] = wm[2] + (iz + 0.5) * sz;
      data[k * 4 + 3] = ratio;
    }
    setCubeInstances(data);
  } catch (e) {
    console.warn("/cubes/state poll failed", e);
  }
}

// ----- /frame submissions (mirror of scan.js's path) ----------------------
// Fallback when view.camera doesn't expose dimensions (camera-access denied);
// otherwise we use the WebXR camera's native resolution so the offline
// voxel-reconstruction tool has the sharpest possible per-pixel rays.
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
  // sane default. Pixel phones report 1920×1080 here on a typical session.
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
