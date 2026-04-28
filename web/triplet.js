// Triplet feature explorer.
// User picks 3 frames → clicks features in order frame 1 → 2 → 3.
// Clicks on frame 2/3 snap (NCC patch match) to the latest frame-1
// click. Phone vs model depth values at the clicked points are
// scatter-plotted at the bottom so the user can eyeball how
// affine-or-not the depth correspondence is.

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
  document.querySelector('.frame-slot[data-slot="2"]'),
  document.querySelector('.frame-slot[data-slot="3"]')];
const metaEls = { 1: document.getElementById("meta1"),
                  2: document.getElementById("meta2"),
                  3: document.getElementById("meta3") };
const plotEls = { "12": document.getElementById("plot12"),
                  "13": document.getElementById("plot13"),
                  "23": document.getElementById("plot23") };

// ---------------------------------------------------------------------
// State
// ---------------------------------------------------------------------

let availableSessions = [];
let frameManifest = null;     // { frames: [{idx, color: [w,h]}, ...] }
// slotFrames[1..3] = frame index assigned to that slot, or null.
const slotFrames = { 1: null, 2: null, 3: null };
// Per-slot rendered <img> reference so click handlers can read the
// actual rendered pixel size (object-fit: contain — letterboxed).
const slotImgs   = { 1: null, 2: null, 3: null };
// Per-slot color-buffer dimensions (cw, ch) from manifest, so we can
// convert mouseX/Y → norm-view UV regardless of letterbox aspect.
const slotImgSize = { 1: null, 2: null, 3: null };

// features[] entries: { id, color, slot1: {u,v} | null, slot2: ..., slot3: ...,
//                       depths: { phone: [d1,d2,d3], model: [d1,d2,d3] } }
const features = [];
// pending feature being defined; when complete it gets pushed to features.
let pending = null;
// Which slot is expected next (1, 2, or 3). Always 1 unless we've
// just clicked frame 1 (then 2) or frame 2 (then 3).
let nextSlot = 1;

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

const SLOT_CSS = { 1: "#f87171", 2: "#fbbf24", 3: "#34d399" };

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
  for (const s of [1, 2, 3]) if (slotFrames[s] === idx) return s;
  return 0;
}

function onThumbClick(idx) {
  const existing = slotForFrame(idx);
  if (existing) {
    // Already assigned → unassign that slot.
    setSlotFrame(existing, null);
  } else {
    // Find first empty slot; if none, replace slot 1 (and shift).
    for (const s of [1, 2, 3]) {
      if (slotFrames[s] === null) {
        setSlotFrame(s, idx);
        return;
      }
    }
    // All filled — overwrite slot 1.
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
  const idx = slotFrames[slot];
  const sid = sessionSel.value;
  if (idx == null || !sid) {
    const empty = document.createElement("span");
    empty.className = "slot-empty";
    empty.textContent = "click a thumbnail to assign";
    slotEl.appendChild(empty);
    return;
  }
  const wrap = document.createElement("div");
  wrap.className = "slot-img-wrap";
  const img = document.createElement("img");
  img.draggable = false;
  img.alt = `frame ${idx}`;
  img.src = `/captures/${encodeURIComponent(sid)}/frame-thumb/${idx}.png?variant=frames&kind=color&long_edge=1200`;
  wrap.appendChild(img);
  // SVG marker layer; sized to match the rendered image area only,
  // not the letterbox bars. We rely on `object-fit: contain` keeping
  // the image centred; placing the SVG over the entire slot is fine
  // because we convert pixel coords on the *rendered image* into
  // norm-view UVs in the click handler.
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "fmarkers");
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  // viewBox set later when we know the image's natural size.
  wrap.appendChild(svg);
  slotEl.appendChild(wrap);
  slotImgs[slot] = img;

  img.addEventListener("load", () => {
    // Use the manifest-provided color buffer dims so SVG viewBox
    // matches whatever resolution the rendered PNG happened to be.
    const finfo = frameManifest?.frames?.find((f) => f.idx === idx);
    const cw = finfo?.color?.[0] ?? img.naturalWidth;
    const ch = finfo?.color?.[1] ?? img.naturalHeight;
    slotImgSize[slot] = [cw, ch];
    svg.setAttribute("viewBox", `0 0 ${cw} ${ch}`);
    redrawSlotMarkers(slot);
  });
  img.addEventListener("click", (e) => onSlotClick(slot, e));
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

async function onSlotClick(slot, ev) {
  if (slotFrames[slot] == null) return;
  const uv = clickToUV(slot, ev);
  if (!uv) return;
  if (slot === 1) {
    // Always starts a new feature; finalise any in-progress one first.
    if (pending) commitPending();
    pending = newPending();
    pending.slot1 = { u: uv.u, v: uv.v };
    nextSlot = 2;
    redrawSlotMarkers(1);
    fetchPendingDepths();
    updateUI();
    return;
  }
  if (slot === 2) {
    if (!pending || !pending.slot1) {
      stMsg.textContent = "click frame 1 first";
      return;
    }
    if (nextSlot > 2) {
      // Allow re-clicking frame 2 to override an earlier snap.
    }
    const refined = await snapFrom(slot, pending.slot1, uv);
    pending.slot2 = refined;
    nextSlot = 3;
    redrawSlotMarkers(2);
    fetchPendingDepths();
    updateUI();
    return;
  }
  if (slot === 3) {
    if (!pending || !pending.slot1) {
      stMsg.textContent = "click frame 1 first";
      return;
    }
    const refined = await snapFrom(slot, pending.slot1, uv);
    pending.slot3 = refined;
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
    slot1: null, slot2: null, slot3: null,
    depths: { phone: [null, null, null], model: [null, null, null] },
    snapScore: { 2: null, 3: null },
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

async function snapFrom(targetSlot, refUV, initUV) {
  const refFrame = slotFrames[1];
  const tgtFrame = slotFrames[targetSlot];
  const sid = sessionSel.value;
  if (refFrame == null || tgtFrame == null) {
    return { u: initUV.u, v: initUV.v, score: 0, snapped: false };
  }
  const url = `/captures/${encodeURIComponent(sid)}/snap-feature`
    + `?ref_frame=${refFrame}&target_frame=${tgtFrame}`
    + `&ref_u=${refUV.u.toFixed(5)}&ref_v=${refUV.v.toFixed(5)}`
    + `&init_u=${initUV.u.toFixed(5)}&init_v=${initUV.v.toFixed(5)}`
    + `&pose_dir=${encodeURIComponent(poseDirSel.value || "frames")}`
    + `&patch=0.025&radius=0.06`;
  let r;
  try { r = await fetch(url); }
  catch (e) { console.warn("snap fetch failed", e); return { u: initUV.u, v: initUV.v, score: 0, snapped: false }; }
  if (!r.ok) return { u: initUV.u, v: initUV.v, score: 0, snapped: false };
  const j = await r.json();
  if (pending) pending.snapScore[targetSlot] = j.score;
  return { u: j.u, v: j.v, score: j.score, snapped: !!j.snapped };
}

// ---------------------------------------------------------------------
// Markers on the big frame panels — one circle per feature on each
// slot it has a click in. Numbered, hue-coded. Pending feature drawn
// in white outline so it's visually distinct from committed ones.
// ---------------------------------------------------------------------

function redrawAllSlotMarkers() {
  for (const s of [1, 2, 3]) redrawSlotMarkers(s);
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
    svg.appendChild(ring);
    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("class", "dot");
    dot.setAttribute("cx", x); dot.setAttribute("cy", y);
    dot.setAttribute("r", Math.max(2, r * 0.18));
    dot.setAttribute("fill", isPending ? "#ffffff" : `hsl(${hue*360}deg, 80%, 60%)`);
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
// Depth fetch + scatter plots
// ---------------------------------------------------------------------

async function fetchDepthsFor(featuresToFetch, kind) {
  // Build parallel arrays: (frames[], us[], vs[]) and a backref map so
  // we can splat results back into features[*].depths.
  const sid = sessionSel.value;
  if (!sid) return;
  const queryFrames = [];
  const queryUs = [];
  const queryVs = [];
  // Each entry: { feat, slot } — telling us where to write the result.
  const slots = [];
  for (const f of featuresToFetch) {
    for (const s of [1, 2, 3]) {
      const uv = f[`slot${s}`];
      const fi = slotFrames[s];
      if (uv && fi != null) {
        queryFrames.push(fi);
        queryUs.push(uv.u.toFixed(5));
        queryVs.push(uv.v.toFixed(5));
        slots.push({ feat: f, slot: s });
      }
    }
  }
  if (!slots.length) return;
  const url = `/captures/${encodeURIComponent(sid)}/frame-depth-at`
    + `?frames=${queryFrames.join(",")}`
    + `&us=${queryUs.join(",")}`
    + `&vs=${queryVs.join(",")}`
    + `&kind=${encodeURIComponent(kind)}`
    + `&pose_dir=${encodeURIComponent(poseDirSel.value || "frames")}`;
  let r;
  try { r = await fetch(url); }
  catch (e) { console.warn("depth-at failed", e); return; }
  if (!r.ok) return;
  const j = await r.json();
  const depths = j.depths || [];
  for (let i = 0; i < slots.length; i++) {
    const { feat, slot } = slots[i];
    feat.depths[kind][slot - 1] = depths[i];
  }
}

async function refreshAllPlots() {
  // Refetch depths for both kinds (so toggling is instant) for every
  // committed feature plus the pending one if any.
  const all = pending ? [...features, pending] : features.slice();
  if (!all.length) {
    drawScatter("12", []); drawScatter("13", []); drawScatter("23", []);
    return;
  }
  await Promise.all([fetchDepthsFor(all, "phone"),
                     fetchDepthsFor(all, "model")]);
  redrawPlots();
}

async function fetchPendingDepths() {
  // Only the pending feature; fast path while the user is mid-click.
  if (!pending) return;
  await Promise.all([fetchDepthsFor([pending], "phone"),
                     fetchDepthsFor([pending], "model")]);
  redrawPlots();
}

function redrawPlots() {
  const kind = depthKindSel.value;
  // For each pair (i, j), gather (depths[i-1], depths[j-1], hue).
  const pairs = ["12", "13", "23"];
  for (const p of pairs) {
    const i = Number(p[0]), j = Number(p[1]);
    const pts = [];
    for (const f of features) {
      const di = f.depths[kind][i - 1];
      const dj = f.depths[kind][j - 1];
      if (di != null && dj != null) pts.push({ x: di, y: dj, hue: f.hue, id: f.id });
    }
    drawScatter(p, pts);
  }
}

function drawScatter(pairKey, points) {
  const svg = plotEls[pairKey];
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
  for (const s of [1, 2, 3]) {
    slotEls[s].classList.toggle("next-click", nextSlot === s);
  }
  const labels = { 1: "frame 1", 2: "frame 2", 3: "frame 3" };
  nextSlotIndicator.textContent = labels[nextSlot] || "—";
  stFeatures.textContent = String(features.length + (pending ? 1 : 0));
  redrawAllSlotMarkers();
  refreshSlotMetas();
}

function refreshSlotMetas() {
  for (const s of [1, 2, 3]) {
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
  if (!pending) return;
  if (nextSlot === 2) {
    nextSlot = 3;
    stMsg.textContent = `feature #${pending.id}: skipped frame 2 — click frame 3 or skip again`;
    updateUI();
    return;
  }
  if (nextSlot === 3) {
    commitPending();
    stMsg.textContent = "skipped frame 3 — feature committed";
    updateUI();
    return;
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
});

// ---------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------

reloadBtn.addEventListener("click", () => {
  refreshSessionList().then(refreshFrameManifest);
});
sessionSel.addEventListener("change", () => {
  // New session — wipe everything.
  for (const s of [1, 2, 3]) slotFrames[s] = null;
  features.length = 0; pending = null; nextSlot = 1;
  populatePosePicker();
  refreshFrameManifest().then(() => {
    for (const s of [1, 2, 3]) renderSlot(s);
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
    for (const s of [1, 2, 3]) renderSlot(s);
    updateUI();
  });
});
