// Stereo feature explorer.
// User picks 2 frames; clicks land exactly where placed (no snap).
// Clicking frame 1 starts a feature; clicking frame 2 commits it.
// Two-mark features are LS-triangulated (midpoint of closest approach
// between the two world rays). For each frame, we plot phone / model /
// blend depth at the click against the camera-Z distance from that
// camera to the triangulated point — y=x (rendered as a 45° diagonal,
// since the plot region is a centred square with equal axis scaling)
// is "perfect agreement". Scroll to zoom, drag to pan; "0" resets.

const sessionSel    = document.getElementById("sessionSel");
const poseDirSel    = document.getElementById("poseDirSel");
const depthKindSel  = document.getElementById("depthKindSel");
const reloadBtn     = document.getElementById("reloadBtn");
const clearFeaturesBtn = document.getElementById("clearFeaturesBtn");
const skipBtn       = document.getElementById("skipBtn");
const nextSlotIndicator = document.getElementById("nextSlotIndicator");
const stFeatures    = document.getElementById("stFeatures");
const stMsg         = document.getElementById("stMsg");
const thumbsPanel   = document.getElementById("thumbsPanel");
const slotEls       = [null,
  document.querySelector('.frame-slot[data-slot="1"]'),
  document.querySelector('.frame-slot[data-slot="2"]')];
const metaEls = { 1: document.getElementById("meta1"),
                  2: document.getElementById("meta2") };
const plotEls = { 1: document.getElementById("plot1"),
                  2: document.getElementById("plot2") };
const kindTagEls = { 1: document.getElementById("kindTagX1"),
                     2: document.getElementById("kindTagX2") };
// All slot iteration goes through this constant so changing the
// feature size to 3 in the future is a one-line edit (plus adding
// the slot in HTML).
const SLOTS = [1, 2];

// ---------------------------------------------------------------------
// State
// ---------------------------------------------------------------------

let availableSessions = [];
let frameManifest = null;     // { frames: [{idx, color: [w,h]}, ...] }
// slotFrames[s] = frame index assigned to that slot, or null.
const slotFrames = { 1: null, 2: null };
// Per-slot rendered <img> reference so click handlers can read the
// actual rendered pixel size (object-fit: contain — letterboxed).
const slotImgs   = { 1: null, 2: null };
// Per-slot color-buffer dimensions (cw, ch) from manifest, so we can
// convert mouseX/Y → norm-view UV regardless of letterbox aspect.
const slotImgSize = { 1: null, 2: null };

// features[] entries: { id, hue, slot1: {u,v} | null, slot2: {u,v} | null,
//                       computed: { world: [x,y,z]|null,
//                                   marks: { 1: {phone,model,cam}, 2: …} } }
const features = [];
// pending feature being defined; when complete it gets pushed to features.
let pending = null;
// Which slot is expected next (1 or 2). Always 1 unless we've just
// clicked frame 1 (then 2). Two-mark features triangulate to the
// midpoint of closest approach between the two rays — enough to read
// off model-vs-truth depth at each frame.
let nextSlot = 1;

// Per-slot zoom/pan state. zoom=1 means the wrap fills the slot exactly
// (the image is letterboxed inside it via object-fit: contain). Larger
// zoom = wrap visually bigger, image enlarged. panX/panY are screen-px
// offsets applied as `translate(panX, panY) scale(zoom)`. Resetting to
// zoom=1, panX=panY=0 returns to the fit-to-slot view.
const slotView = {
  1: { zoom: 1, panX: 0, panY: 0 },
  2: { zoom: 1, panX: 0, panY: 0 },
};
const ZOOM_MIN = 1.0;
const ZOOM_MAX = 16.0;
// Drag-vs-click discrimination: if the cursor moved more than this many
// pixels between pointerdown and pointerup, treat the gesture as a pan
// (not a click). Stays small so a deliberate click of an existing
// marker isn't misclassified as a pan even with a slightly shaky hand.
const CLICK_MOVE_PX = 4;

// ---------------------------------------------------------------------
// Session + frame discovery
// ---------------------------------------------------------------------

async function refreshSessionList() {
  try {
    const r = await fetch("/sessions");
    const j = r.ok ? await r.json() : { sessions: [] };
    availableSessions = j.sessions || [];
  } catch (e) {
    console.warn("/sessions failed", e);
    availableSessions = [];
  }
  renderSessionDropdown();
}

function renderSessionDropdown() {
  const usable = availableSessions
    .filter((s) => s.n_frames > 0)
    .sort((a, b) => (a.id < b.id ? 1 : -1));
  const previous = sessionSel.value;
  sessionSel.innerHTML = "";
  if (!usable.length) {
    const opt = document.createElement("option");
    opt.value = ""; opt.textContent = "(no sessions yet)";
    sessionSel.appendChild(opt);
  } else {
    for (const s of usable) {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = `${s.id}  (${s.n_frames} frames)`;
      sessionSel.appendChild(opt);
    }
  }
  if (previous && usable.some((s) => s.id === previous)) {
    sessionSel.value = previous;
  } else if (usable.length) {
    sessionSel.value = usable[0].id;
  }
  populatePosePicker();
}

function populatePosePicker() {
  const sid = sessionSel.value;
  const sess = availableSessions.find((s) => s.id === sid);
  let dirs = (sess?.frame_dirs || []).slice();
  if (!dirs.length) dirs.push("frames");
  const rank = (d) => {
    if (d === "frames")               return 0;
    if (d === "frames_aligned")       return 1;
    if (d.startsWith("frames_feature_ba")) return 2;
    if (d.startsWith("frames_refined"))    return 4;
    return 3;
  };
  dirs.sort((a, b) => (rank(a) - rank(b)) || (a < b ? -1 : 1));
  const previous = poseDirSel.value;
  poseDirSel.innerHTML = "";
  for (const d of dirs) {
    const o = document.createElement("option");
    o.value = d; o.textContent = d;
    poseDirSel.appendChild(o);
  }
  if (previous && dirs.includes(previous)) poseDirSel.value = previous;
  else                                     poseDirSel.value = dirs[0];
}

async function refreshFrameManifest() {
  const sid = sessionSel.value;
  if (!sid) { frameManifest = null; renderThumbsPanel(); return; }
  try {
    const r = await fetch(`/captures/${encodeURIComponent(sid)}/frame-manifest?variant=frames`);
    frameManifest = r.ok ? await r.json() : null;
  } catch (e) {
    console.warn("frame-manifest failed", e);
    frameManifest = null;
  }
  renderThumbsPanel();
}

// ---------------------------------------------------------------------
// Thumbnails (left panel) — clicking a thumb assigns it to the next
// empty slot. Clicking an already-assigned thumb removes it from
// whatever slot it occupies.
// ---------------------------------------------------------------------

const SLOT_CSS = { 1: "#f87171", 2: "#34d399" };

function renderThumbsPanel() {
  thumbsPanel.innerHTML = "";
  const sid = sessionSel.value;
  const frames = frameManifest?.frames ?? [];
  if (!frames.length) {
    const e = document.createElement("div");
    e.style.padding = "0.5rem"; e.style.opacity = "0.55";
    e.textContent = "(no frames)";
    thumbsPanel.appendChild(e);
    return;
  }
  for (const f of frames) {
    const item = document.createElement("div");
    item.className = "thumb-item";
    item.dataset.idx = String(f.idx);
    const slot = slotForFrame(f.idx);
    if (slot) {
      item.classList.add("assigned");
      item.style.setProperty("--slot-color", SLOT_CSS[slot]);
    }
    const img = document.createElement("img");
    img.loading = "lazy";
    img.alt = `frame ${f.idx}`;
    img.src = `/captures/${encodeURIComponent(sid)}/frame-thumb/${f.idx}.png?variant=frames&kind=color`;
    item.appendChild(img);
    if (slot) {
      const badge = document.createElement("span");
      badge.className = "slot-badge";
      badge.textContent = String(slot);
      item.appendChild(badge);
    }
    const tag = document.createElement("span");
    tag.className = "idx-tag";
    tag.textContent = `#${f.idx}`;
    item.appendChild(tag);
    item.addEventListener("click", () => onThumbClick(f.idx));
    thumbsPanel.appendChild(item);
  }
}

function slotForFrame(idx) {
  for (const s of SLOTS) if (slotFrames[s] === idx) return s;
  return 0;
}

function onThumbClick(idx) {
  const existing = slotForFrame(idx);
  if (existing) {
    // Already assigned → unassign that slot.
    setSlotFrame(existing, null);
  } else {
    // Find first empty slot; if none, overwrite slot 1.
    for (const s of SLOTS) {
      if (slotFrames[s] === null) {
        setSlotFrame(s, idx);
        return;
      }
    }
    setSlotFrame(1, idx);
  }
}

function setSlotFrame(slot, idx) {
  slotFrames[slot] = idx;
  // Selecting fresh slots invalidates any half-completed feature.
  pending = null;
  nextSlot = 1;
  features.length = 0;
  renderThumbsPanel();
  renderSlot(slot);
  refreshSlotMetas();
  refreshAllPlots();
  updateUI();
}

// ---------------------------------------------------------------------
// Big slot rendering: load color image at long_edge=1200, attach a
// click handler that converts mouse → norm-view UV.
// ---------------------------------------------------------------------

function renderSlot(slot) {
  const slotEl = slotEls[slot];
  // Clear previous content while preserving the slot-tag + slot-meta spans.
  for (const child of [...slotEl.children]) {
    if (!child.classList.contains("slot-tag") && !child.classList.contains("slot-meta")) {
      slotEl.removeChild(child);
    }
  }
  slotImgs[slot] = null;
  // Reset zoom/pan whenever the assigned frame changes — fresh image,
  // fresh view. (Reassigning to the same frame still resets, which is
  // the friendlier behaviour: clicking a thumb feels like a "load".)
  slotView[slot].zoom = 1;
  slotView[slot].panX = 0;
  slotView[slot].panY = 0;

  const idx = slotFrames[slot];
  const sid = sessionSel.value;
  if (idx == null || !sid) {
    const empty = document.createElement("span");
    empty.className = "slot-empty";
    empty.textContent = "click a thumbnail to assign";
    slotEl.appendChild(empty);
    updateZoomTag(slot);
    return;
  }
  const wrap = document.createElement("div");
  wrap.className = "slot-img-wrap";
  const img = document.createElement("img");
  img.draggable = false;
  img.alt = `frame ${idx}`;
  img.src = `/captures/${encodeURIComponent(sid)}/frame-thumb/${idx}.png?variant=frames&kind=color&long_edge=1600`;
  wrap.appendChild(img);
  // SVG marker layer with the image's color-buffer pixel size as
  // viewBox so circle coords are easy to author. preserveAspectRatio
  // matches `object-fit: contain` on the <img>, keeping markers
  // aligned with the rendered image — including under the panzoom
  // transform (which scales the wrap; SVG and IMG inherit the same).
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "fmarkers");
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  wrap.appendChild(svg);
  slotEl.appendChild(wrap);
  slotImgs[slot] = img;

  // Zoom indicator pinned bottom-right of the slot.
  let zt = slotEl.querySelector(".zoom-tag");
  if (!zt) {
    zt = document.createElement("span");
    zt.className = "zoom-tag";
    slotEl.appendChild(zt);
  }
  updateZoomTag(slot);

  img.addEventListener("load", () => {
    const finfo = frameManifest?.frames?.find((f) => f.idx === idx);
    const cw = finfo?.color?.[0] ?? img.naturalWidth;
    const ch = finfo?.color?.[1] ?? img.naturalHeight;
    slotImgSize[slot] = [cw, ch];
    svg.setAttribute("viewBox", `0 0 ${cw} ${ch}`);
    applySlotTransform(slot);
    redrawSlotMarkers(slot);
  });

  // Pointer/wheel handlers attached at the SLOT level so the wrap can
  // be transformed without losing event capture. img has
  // `pointer-events: none` so the slot reliably owns the gestures.
  attachSlotPanZoom(slot);
}

function applySlotTransform(slot) {
  const slotEl = slotEls[slot];
  const wrap = slotEl.querySelector(".slot-img-wrap");
  if (!wrap) return;
  const v = slotView[slot];
  wrap.style.transform = `translate(${v.panX}px, ${v.panY}px) scale(${v.zoom})`;
}

function updateZoomTag(slot) {
  const slotEl = slotEls[slot];
  const zt = slotEl.querySelector(".zoom-tag");
  if (!zt) return;
  const v = slotView[slot];
  if (slotFrames[slot] == null) { zt.textContent = ""; return; }
  zt.textContent = v.zoom > 1.001
    ? `${v.zoom.toFixed(1)}× · drag to pan · 0=reset`
    : `1× · scroll=zoom · drag=pan`;
}

function attachSlotPanZoom(slot) {
  const slotEl = slotEls[slot];
  if (slotEl.dataset.panzoomAttached === "1") return;
  slotEl.dataset.panzoomAttached = "1";

  // Wheel: zoom around cursor.
  slotEl.addEventListener("wheel", (ev) => {
    if (slotFrames[slot] == null) return;
    ev.preventDefault();
    const v = slotView[slot];
    // 1.15 per "tick" — feels right with both wheel mice and trackpads.
    const factor = Math.exp(-ev.deltaY * 0.0025);
    let newZoom = v.zoom * factor;
    if (newZoom < ZOOM_MIN) newZoom = ZOOM_MIN;
    if (newZoom > ZOOM_MAX) newZoom = ZOOM_MAX;
    if (newZoom === v.zoom) return;
    const slotRect = slotEl.getBoundingClientRect();
    const mx = ev.clientX - slotRect.left;
    const my = ev.clientY - slotRect.top;
    // Wrap-local coord under cursor (pre-transform) at current zoom.
    const xL = (mx - v.panX) / v.zoom;
    const yL = (my - v.panY) / v.zoom;
    // Choose new pan so the same wrap-local point stays under cursor.
    v.panX = mx - xL * newZoom;
    v.panY = my - yL * newZoom;
    v.zoom = newZoom;
    clampPan(slot);
    applySlotTransform(slot);
    updateZoomTag(slot);
  }, { passive: false });

  // Pointer down/move/up: pan if motion exceeds threshold; otherwise
  // treat as a click that places a feature mark.
  let downX = 0, downY = 0;
  let startPanX = 0, startPanY = 0;
  let panning = false;
  let pressed = false;

  slotEl.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    if (slotFrames[slot] == null) return;
    pressed = true;
    panning = false;
    downX = ev.clientX; downY = ev.clientY;
    startPanX = slotView[slot].panX;
    startPanY = slotView[slot].panY;
    slotEl.setPointerCapture(ev.pointerId);
  });
  slotEl.addEventListener("pointermove", (ev) => {
    if (!pressed) return;
    const dx = ev.clientX - downX;
    const dy = ev.clientY - downY;
    if (!panning && (Math.abs(dx) > CLICK_MOVE_PX || Math.abs(dy) > CLICK_MOVE_PX)) {
      panning = true;
      slotEl.classList.add("panning");
    }
    if (panning) {
      slotView[slot].panX = startPanX + dx;
      slotView[slot].panY = startPanY + dy;
      clampPan(slot);
      applySlotTransform(slot);
    }
  });
  const finishPointer = (ev) => {
    if (!pressed) return;
    const wasPanning = panning;
    pressed = false;
    panning = false;
    slotEl.classList.remove("panning");
    try { slotEl.releasePointerCapture(ev.pointerId); } catch (_) {}
    if (!wasPanning) {
      onSlotClick(slot, ev);
    }
  };
  slotEl.addEventListener("pointerup", finishPointer);
  slotEl.addEventListener("pointercancel", finishPointer);
}

function clampPan(slot) {
  // Keep at least a thin sliver of the wrap on-screen so the user
  // can't pan it entirely out of the slot. Allowance grows with zoom
  // so the user can sweep across the whole image at high zoom.
  const slotEl = slotEls[slot];
  const v = slotView[slot];
  const slotRect = slotEl.getBoundingClientRect();
  // Wrap visual size at current zoom is slotRect.width × zoom (and same
  // for height), positioned at (panX, panY) within the slot. Edges:
  //   left   = panX
  //   right  = panX + zoom * slotRect.width
  // We want at least 60 px visible: right > 60 and left < slotRect.width - 60.
  const margin = Math.min(60, slotRect.width / 4);
  const wrapVisualW = slotRect.width  * v.zoom;
  const wrapVisualH = slotRect.height * v.zoom;
  const minPanX = -(wrapVisualW - margin);
  const maxPanX = slotRect.width  - margin;
  const minPanY = -(wrapVisualH - margin);
  const maxPanY = slotRect.height - margin;
  if (v.panX < minPanX) v.panX = minPanX;
  if (v.panX > maxPanX) v.panX = maxPanX;
  if (v.panY < minPanY) v.panY = minPanY;
  if (v.panY > maxPanY) v.panY = maxPanY;
}

function resetSlotView(slot) {
  slotView[slot].zoom = 1;
  slotView[slot].panX = 0;
  slotView[slot].panY = 0;
  applySlotTransform(slot);
  updateZoomTag(slot);
}

// Convert mouseX/Y on a slot's <img> to norm-view UV (0..1)² with
// v=0 at the bottom of the view (norm-view convention). Returns null
// if the click landed in the letterbox area outside the rendered image.
function clickToUV(slot, ev) {
  const img = slotImgs[slot];
  if (!img) return null;
  const rect = img.getBoundingClientRect();
  const finfo = slotImgSize[slot];
  if (!finfo) return null;
  const [cw, ch] = finfo;
  // object-fit: contain — figure out the rendered image rect inside `rect`.
  const rectAR = rect.width / Math.max(1, rect.height);
  const imgAR  = cw / Math.max(1, ch);
  let renderW, renderH, offsetX, offsetY;
  if (imgAR > rectAR) {
    renderW = rect.width;
    renderH = rect.width / imgAR;
    offsetX = 0;
    offsetY = (rect.height - renderH) / 2;
  } else {
    renderH = rect.height;
    renderW = rect.height * imgAR;
    offsetX = (rect.width - renderW) / 2;
    offsetY = 0;
  }
  const x = ev.clientX - rect.left - offsetX;
  const y = ev.clientY - rect.top  - offsetY;
  if (x < 0 || x > renderW || y < 0 || y > renderH) return null;
  const u = x / renderW;            // norm-view u (left → right)
  const v = 1.0 - y / renderH;      // norm-view v (bottom → top)
  return { u, v };
}

function onSlotClick(slot, ev) {
  if (slotFrames[slot] == null) return;
  const uv = clickToUV(slot, ev);
  if (!uv) return;
  // Per the user's spec: clicks land EXACTLY where placed — no NCC
  // auto-snap. The point of zoom/pan is to let the user dial in the
  // location precisely instead of having the algorithm second-guess.
  if (slot === 1) {
    // Always starts a new feature; finalise any in-progress one first.
    if (pending) commitPending();
    pending = newPending();
    pending.slot1 = { u: uv.u, v: uv.v };
    nextSlot = 2;
    redrawSlotMarkers(1);
    refreshAllPlots();
    updateUI();
    return;
  }
  if (slot === 2) {
    if (!pending || !pending.slot1) {
      stMsg.textContent = "click frame 1 first";
      return;
    }
    pending.slot2 = { u: uv.u, v: uv.v };
    // Two marks → triangulation possible → commit and ready for next.
    commitPending();
    updateUI();
    return;
  }
}

function newPending() {
  const id = features.length + 1;
  const hue = ((id * 0.6180339887) % 1.0);
  return {
    id, hue,
    slot1: null, slot2: null,
    // Filled by /triplet-distances after the feature has ≥ 2 marks.
    computed: { world: null, marks: { 1: null, 2: null } },
  };
}

function commitPending() {
  if (!pending) return;
  features.push(pending);
  pending = null;
  nextSlot = 1;
  redrawAllSlotMarkers();
  refreshAllPlots();
}

// ---------------------------------------------------------------------
// Markers on the big frame panels — one circle per feature on each
// slot it has a click in. Numbered, hue-coded. Pending feature drawn
// in white outline so it's visually distinct from committed ones.
// ---------------------------------------------------------------------

function redrawAllSlotMarkers() {
  for (const s of SLOTS) redrawSlotMarkers(s);
}

function redrawSlotMarkers(slot) {
  const slotEl = slotEls[slot];
  const svg = slotEl.querySelector("svg.fmarkers");
  if (!svg) return;
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const dims = slotImgSize[slot];
  if (!dims) return;
  const [cw, ch] = dims;
  // Marker radius scales with image diag.
  const r = Math.max(8, Math.round(Math.hypot(cw, ch) * 0.012));
  const draw = (uv, hue, label, isPending) => {
    if (!uv) return;
    const x = uv.u * cw;
    const y = (1 - uv.v) * ch;
    const NS = "http://www.w3.org/2000/svg";
    const ring = document.createElementNS(NS, "circle");
    ring.setAttribute("class", "ring");
    ring.setAttribute("cx", x); ring.setAttribute("cy", y);
    ring.setAttribute("r", r);
    ring.setAttribute("stroke", isPending ? "#ffffff" : `hsl(${hue*360}deg, 80%, 60%)`);
    // Keep stroke width constant in screen px regardless of zoom so the
    // ring stays a clear, thin outline rather than a thick blob at 10×.
    ring.setAttribute("vector-effect", "non-scaling-stroke");
    svg.appendChild(ring);
    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("class", "dot");
    dot.setAttribute("cx", x); dot.setAttribute("cy", y);
    dot.setAttribute("r", Math.max(2, r * 0.18));
    dot.setAttribute("fill", isPending ? "#ffffff" : `hsl(${hue*360}deg, 80%, 60%)`);
    dot.setAttribute("vector-effect", "non-scaling-stroke");
    svg.appendChild(dot);
    const txt = document.createElementNS(NS, "text");
    txt.setAttribute("x", x + r + 4);
    txt.setAttribute("y", y + 5);
    txt.textContent = label;
    svg.appendChild(txt);
  };
  for (const f of features) {
    const uv = f[`slot${slot}`];
    if (uv) draw(uv, f.hue, `#${f.id}`, false);
  }
  if (pending) {
    const uv = pending[`slot${slot}`];
    if (uv) draw(uv, pending.hue, `#${pending.id}*`, true);
  }
}

// ---------------------------------------------------------------------
// Triangulation-based plots
// ---------------------------------------------------------------------
//
// For every feature with ≥ 2 marks the server triangulates a 3D world
// point (LS over the rays) and reports per-mark (phone_depth, model_depth,
// cam_distance). We plot one panel per frame slot:
//   X = depth-source value at the click on that frame
//   Y = camera-Z distance from that frame's camera to the triangulated
//       world point
// The y=x diagonal is the "perfect agreement" reference. Drift away
// from it shows where the depth source overshoots / undershoots
// reality at that frame.

async function refreshAllPlots() {
  const sid = sessionSel.value;
  const all = pending ? [...features, pending] : features.slice();
  if (!sid || !all.length) {
    for (const slot of SLOTS) drawSlotPlot(slot, []);
    return;
  }
  // Build the request body — features with their (frame, u, v) marks.
  const featuresPayload = all.map((f) => {
    const marks = [];
    for (const s of SLOTS) {
      const uv = f[`slot${s}`];
      const fi = slotFrames[s];
      if (uv && fi != null) marks.push({ frame: fi, u: uv.u, v: uv.v });
    }
    return { id: f.id, marks };
  });
  const url = `/captures/${encodeURIComponent(sid)}/triplet-distances`
    + `?pose_dir=${encodeURIComponent(poseDirSel.value || "frames")}`
    + `&features=${encodeURIComponent(JSON.stringify(featuresPayload))}`;
  let r;
  try { r = await fetch(url); }
  catch (e) { console.warn("triplet-distances failed", e); return; }
  if (!r.ok) {
    console.warn("triplet-distances HTTP", r.status);
    return;
  }
  const j = await r.json();
  // Splat results back into features[*].computed.
  const featsOut = j.features || [];
  for (let i = 0; i < all.length; i++) {
    const f = all[i];
    const out = featsOut[i];
    if (!out) continue;
    f.computed.world = out.world ?? null;
    f.computed.marks = { 1: null, 2: null, 3: null };
    for (const m of (out.marks || [])) {
      // Find which slot this mark belongs to (by frame index).
      for (const s of SLOTS) {
        if (slotFrames[s] === m.frame) {
          f.computed.marks[s] = m;
          break;
        }
      }
    }
  }
  redrawPlots();
}

function redrawPlots() {
  const kind = depthKindSel.value;   // "phone" | "model"
  const fieldName = (kind === "phone") ? "phone_depth" : "model_depth";
  for (const slot of SLOTS) {
    kindTagEls[slot].textContent = kind;
    const pts = [];
    for (const f of features) {
      const m = f.computed.marks[slot];
      if (!m) continue;
      const x = m[fieldName];
      const y = m.cam_distance;
      if (x != null && y != null) pts.push({ x, y, hue: f.hue, id: f.id });
    }
    drawSlotPlot(slot, pts);
  }
}

// Backwards-compat shim: old name still referenced by depthKindSel
// listeners and resize handler.
function drawScatter(slot, points) { drawSlotPlot(slot, points); }

function drawSlotPlot(slot, points) {
  const svg = plotEls[slot];
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const rect = svg.getBoundingClientRect();
  const W = Math.max(120, rect.width);
  const H = Math.max(80,  rect.height);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const padL = 30, padR = 8, padT = 6, padB = 18;
  if (!points.length) {
    const NS = "http://www.w3.org/2000/svg";
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", W/2); t.setAttribute("y", H/2);
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("fill", "#555");
    t.setAttribute("font-size", "11");
    t.textContent = "(no features yet)";
    svg.appendChild(t);
    return;
  }
  // Auto-range with 5% padding; clamp to (0, ~) so negative depths visible.
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const allMin = Math.min(...xs, ...ys);
  const allMax = Math.max(...xs, ...ys);
  const span = Math.max(0.1, allMax - allMin);
  const min = Math.max(0, allMin - 0.05 * span);
  const max = allMax + 0.05 * span;
  const sx = (x) => padL + (x - min) / (max - min) * (W - padL - padR);
  const sy = (y) => H - padB - (y - min) / (max - min) * (H - padT - padB);

  const NS = "http://www.w3.org/2000/svg";
  const axis = document.createElementNS(NS, "g");
  axis.setAttribute("class", "axis");
  // Y axis
  axis.appendChild(makeLine(NS, padL, padT, padL, H - padB));
  // X axis
  axis.appendChild(makeLine(NS, padL, H - padB, W - padR, H - padB));
  // y=x diagonal — perfect agreement reference.
  const xy0 = sx(min), xy1 = sx(max);
  const yy0 = sy(min), yy1 = sy(max);
  axis.appendChild(makeLine(NS, xy0, yy0, xy1, yy1, "yx"));
  // Tick labels at 4 even points
  for (let k = 0; k <= 3; k++) {
    const t = min + (max - min) * (k / 3);
    const xt = sx(t);
    const yt = sy(t);
    const lblX = document.createElementNS(NS, "text");
    lblX.setAttribute("x", xt);
    lblX.setAttribute("y", H - padB + 12);
    lblX.setAttribute("text-anchor", "middle");
    lblX.textContent = t.toFixed(2);
    axis.appendChild(lblX);
    const lblY = document.createElementNS(NS, "text");
    lblY.setAttribute("x", padL - 4);
    lblY.setAttribute("y", yt + 3);
    lblY.setAttribute("text-anchor", "end");
    lblY.textContent = t.toFixed(2);
    axis.appendChild(lblY);
    if (k > 0 && k < 3) {
      axis.appendChild(makeLine(NS, padL, yt, W - padR, yt, "gridline"));
      axis.appendChild(makeLine(NS, xt, padT, xt, H - padB, "gridline"));
    }
  }
  svg.appendChild(axis);

  // Points.
  for (const p of points) {
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("cx", sx(p.x));
    c.setAttribute("cy", sy(p.y));
    c.setAttribute("r", 4);
    c.setAttribute("fill", `hsl(${p.hue*360}deg, 80%, 60%)`);
    c.setAttribute("class", "point");
    const title = document.createElementNS(NS, "title");
    title.textContent = `feature #${p.id} · (${p.x.toFixed(3)}, ${p.y.toFixed(3)}) m`;
    c.appendChild(title);
    svg.appendChild(c);
  }
}

function makeLine(NS, x1, y1, x2, y2, cls) {
  const l = document.createElementNS(NS, "line");
  l.setAttribute("x1", x1); l.setAttribute("y1", y1);
  l.setAttribute("x2", x2); l.setAttribute("y2", y2);
  if (cls) l.setAttribute("class", cls);
  return l;
}

// ---------------------------------------------------------------------
// UI updates
// ---------------------------------------------------------------------

function updateUI() {
  for (const s of SLOTS) {
    slotEls[s].classList.toggle("next-click", nextSlot === s);
  }
  const labels = { 1: "frame 1", 2: "frame 2" };
  nextSlotIndicator.textContent = labels[nextSlot] || "—";
  stFeatures.textContent = String(features.length + (pending ? 1 : 0));
  redrawAllSlotMarkers();
  refreshSlotMetas();
}

function refreshSlotMetas() {
  for (const s of SLOTS) {
    const idx = slotFrames[s];
    if (idx == null) { metaEls[s].textContent = "—"; continue; }
    const f = frameManifest?.frames?.find((fr) => fr.idx === idx);
    const pose = f?.pose;
    metaEls[s].textContent = pose
      ? `#${idx} · pose (${pose[0].toFixed(2)}, ${pose[1].toFixed(2)}, ${pose[2].toFixed(2)})`
      : `#${idx}`;
  }
}

// ---------------------------------------------------------------------
// Skip / clear / cancel
// ---------------------------------------------------------------------

function skipNextSlot() {
  // 2-frame mode: skipping after frame 1 commits a single-mark feature
  // (no triangulation possible, but the click on frame 1 is still
  // recorded; the plots simply won't have a Y for it). Useful when
  // the corresponding point falls outside frame 2's view.
  if (!pending) return;
  if (nextSlot === 2) {
    commitPending();
    stMsg.textContent = "skipped frame 2 — feature committed (no triangulation possible)";
    updateUI();
  }
}

function cancelPending() {
  pending = null;
  nextSlot = 1;
  redrawAllSlotMarkers();
  updateUI();
}

skipBtn.addEventListener("click", skipNextSlot);
clearFeaturesBtn.addEventListener("click", () => {
  features.length = 0;
  pending = null;
  nextSlot = 1;
  redrawAllSlotMarkers();
  refreshAllPlots();
  updateUI();
});
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA"
      || e.target.tagName === "SELECT") return;
  if (e.code === "Space") { e.preventDefault(); skipNextSlot(); }
  else if (e.code === "Escape") cancelPending();
  else if (e.code === "Digit0" || e.code === "Numpad0") {
    // Reset zoom/pan on whichever slot the cursor is over (or all if
    // outside any slot).
    let target = -1;
    for (const s of SLOTS) {
      if (slotEls[s].matches(":hover")) { target = s; break; }
    }
    if (target > 0) resetSlotView(target);
    else for (const s of SLOTS) resetSlotView(s);
  }
});

// ---------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------

reloadBtn.addEventListener("click", () => {
  refreshSessionList().then(refreshFrameManifest);
});
sessionSel.addEventListener("change", () => {
  // New session — wipe everything.
  for (const s of SLOTS) slotFrames[s] = null;
  features.length = 0; pending = null; nextSlot = 1;
  populatePosePicker();
  refreshFrameManifest().then(() => {
    for (const s of SLOTS) renderSlot(s);
    refreshAllPlots();
    updateUI();
  });
});
poseDirSel.addEventListener("change", () => {
  // Pose change: depth values change (different bytes for phone, but
  // also different model_raw bins should be the same — only phone
  // depth depends on pose_dir's byte content). Refresh plots either
  // way.
  refreshAllPlots();
});
depthKindSel.addEventListener("change", () => {
  redrawPlots();
});
window.addEventListener("resize", () => redrawPlots());

// Boot.
refreshSessionList().then(() => {
  refreshFrameManifest().then(() => {
    for (const s of SLOTS) renderSlot(s);
    updateUI();
  });
});
