// Depth-scatter viewer.
// Pick a session + frame; see the RGB photo, the phone-depth thumb, the
// model-depth thumb, and a scatter of (phone, model) at every pixel of
// the (subsampled) colour-image grid. The RGB panel can have an overlay
// (phone, model, or signed diff) blended on top with adjustable opacity.

const sessionSel    = document.getElementById("sessionSel");
const poseDirSel    = document.getElementById("poseDirSel");
const reloadBtn     = document.getElementById("reloadBtn");
const overlayRadios = document.querySelectorAll("input[name='overlay']");
const overlayOpacity = document.getElementById("overlayOpacity");
const stStatus      = document.getElementById("stStatus");
const thumbsPanel   = document.getElementById("thumbsPanel");
const panelRgb      = document.getElementById("panelRgb");
const panelPhone    = document.getElementById("panelPhone");
const panelModel    = document.getElementById("panelModel");
const scatterMeta   = document.getElementById("scatterMeta");
const scatterSvg    = document.getElementById("scatterSvg");

// ---------------------------------------------------------------------
// State
// ---------------------------------------------------------------------

let availableSessions = [];
let frameManifest = null;
let selectedFrame = null;
// Holds the most recent /depth-scatter response so window resizes can
// re-draw without re-fetching. null until the first response arrives.
let lastScatter = null;
// Bumps on every fetch so a slow response from a previously-selected
// frame can't clobber the panel after the user has moved on.
let fetchToken = 0;

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
  if (!sid) { frameManifest = null; renderThumbsPanel(); applyFrameAspect(); return; }
  try {
    const r = await fetch(`/captures/${encodeURIComponent(sid)}/frame-manifest?variant=frames`);
    frameManifest = r.ok ? await r.json() : null;
  } catch (e) {
    console.warn("frame-manifest failed", e);
    frameManifest = null;
  }
  renderThumbsPanel();
  applyFrameAspect();
}

// Set --frame-ar so the image panels size themselves to the captured
// portrait aspect. All frames in a session share dims so reading the
// first one is enough; default falls back to typical phone portrait.
function applyFrameAspect() {
  const f = frameManifest?.frames?.[0];
  if (!f?.color) return;
  const [w, h] = f.color;
  if (w > 0 && h > 0) {
    document.documentElement.style.setProperty("--frame-ar", `${w} / ${h}`);
  }
}

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
    if (selectedFrame === f.idx) item.classList.add("selected");
    const img = document.createElement("img");
    img.loading = "lazy";
    img.alt = `frame ${f.idx}`;
    img.src = `/captures/${encodeURIComponent(sid)}/frame-thumb/${f.idx}.png?variant=frames&kind=color`;
    item.appendChild(img);
    const tag = document.createElement("span");
    tag.className = "idx-tag";
    tag.textContent = `#${f.idx}`;
    item.appendChild(tag);
    item.addEventListener("click", () => selectFrame(f.idx));
    thumbsPanel.appendChild(item);
  }
}

// ---------------------------------------------------------------------
// Frame selection / loading
// ---------------------------------------------------------------------

function selectFrame(idx) {
  selectedFrame = idx;
  // Cheaper than re-rendering the whole strip from scratch.
  for (const item of thumbsPanel.querySelectorAll(".thumb-item")) {
    item.classList.toggle("selected", Number(item.dataset.idx) === idx);
  }
  loadFrame();
}

function loadFrame() {
  const sid = sessionSel.value;
  if (sid == null || selectedFrame == null) return;
  const idx = selectedFrame;
  fetchToken++;
  const myToken = fetchToken;
  // Long-edge of 800 keeps each panel sharp without making the JSON
  // round-trip too slow on the same frame click.
  const longEdge = 800;
  const base = `/captures/${encodeURIComponent(sid)}/frame-thumb/${idx}.png`;
  const qs = `variant=frames&long_edge=${longEdge}`;

  setRgbImg(`${base}?${qs}&kind=color`);
  applyOverlay(getOverlayMode(), idx, longEdge);

  setDepthImg(panelPhone, `${base}?${qs}&kind=phone`);
  setDepthImg(panelModel, `${base}?${qs}&kind=model`);

  fetchScatter(sid, idx, myToken);

  stStatus.textContent = `frame #${idx} · loading…`;
}

function setRgbImg(src) {
  const wrap = panelRgb.querySelector(".panel-img-wrap");
  let img = wrap.querySelector("img.depth-img");
  // The RGB panel always has its base image first; an optional overlay
  // gets layered on top via setOverlayImg().
  if (!img) {
    wrap.querySelector(".panel-empty")?.remove();
    img = document.createElement("img");
    img.className = "depth-img";
    img.alt = "rgb";
    img.draggable = false;
    wrap.insertBefore(img, wrap.firstChild);
  }
  if (img.src !== src) img.src = src;
}

function setDepthImg(panel, src) {
  const wrap = panel.querySelector(".panel-img-wrap");
  let img = wrap.querySelector("img.depth-img");
  if (!img) {
    wrap.querySelector(".panel-empty")?.remove();
    img = document.createElement("img");
    img.className = "depth-img";
    img.alt = panel.querySelector(".panel-tag")?.textContent || "depth";
    img.draggable = false;
    wrap.appendChild(img);
  }
  // model_raw cache might be missing → 409 from server. We surface that
  // with a placeholder; the broken-image icon would be confusing.
  img.onerror = () => {
    wrap.innerHTML = '<span class="panel-empty">no model_raw cache</span>';
  };
  if (img.src !== src) img.src = src;
}

function getOverlayMode() {
  for (const r of overlayRadios) if (r.checked) return r.value;
  return "off";
}

function setOverlayImg(src) {
  // Layered absolutely on top of the RGB so they share rendering bounds.
  let overlay = panelRgb.querySelector("img.overlay-img");
  if (src == null) {
    if (overlay) overlay.remove();
    return;
  }
  if (!overlay) {
    overlay = document.createElement("img");
    overlay.className = "overlay-img";
    overlay.alt = "overlay";
    overlay.draggable = false;
    panelRgb.querySelector(".panel-img-wrap").appendChild(overlay);
  }
  if (overlay.src !== src) overlay.src = src;
  overlay.style.opacity = String(parseFloat(overlayOpacity.value));
}

function applyOverlay(mode, idx, longEdge) {
  if (mode === "off" || idx == null) {
    setOverlayImg(null);
    return;
  }
  const sid = sessionSel.value;
  if (sid == null) return;
  const base = `/captures/${encodeURIComponent(sid)}/frame-thumb/${idx}.png`;
  const src = `${base}?variant=frames&long_edge=${longEdge}&kind=${mode}`;
  setOverlayImg(src);
}

// ---------------------------------------------------------------------
// Scatter fetch + render
// ---------------------------------------------------------------------

async function fetchScatter(sid, idx, token) {
  let r;
  try {
    r = await fetch(`/captures/${encodeURIComponent(sid)}/depth-scatter/${idx}.json?max_samples=5000`);
  } catch (e) {
    if (token === fetchToken) {
      stStatus.textContent = `frame #${idx} · scatter request failed`;
      console.warn("depth-scatter fetch failed", e);
    }
    return;
  }
  if (token !== fetchToken) return;        // user has already moved on
  if (!r.ok) {
    const txt = await r.text();
    stStatus.textContent = `frame #${idx} · scatter HTTP ${r.status}`;
    setScatterMeta(`(server: ${txt.trim()})`);
    drawScatter([]);
    lastScatter = null;
    return;
  }
  const j = await r.json();
  if (token !== fetchToken) return;
  lastScatter = j;
  renderScatterFromState();
  stStatus.textContent =
      `frame #${idx} · ${j.n_valid.toLocaleString()} valid pairs · `
    + `Pearson ${fmtCorr(j.pearson)} · Spearman ${fmtCorr(j.spearman)}`;
}

function renderScatterFromState() {
  const j = lastScatter;
  if (!j) { drawScatter([]); setScatterMeta("(no data)"); return; }
  setScatterMeta(
    `<span><b>${j.n_valid.toLocaleString()}</b> valid pairs (${j.n_returned.toLocaleString()} plotted)</span>`
    + ` <span>Pearson <b>${fmtCorr(j.pearson)}</b></span>`
    + ` <span>Spearman <b>${fmtCorr(j.spearman)}</b></span>`
    + ` <span>phone <b>${fmtRange(j.phone_min, j.phone_max)}</b> m</span>`
    + ` <span>model <b>${fmtRange(j.model_min, j.model_max)}</b></span>`
  );
  drawScatter(j.pairs || []);
}

function fmtCorr(x) {
  if (x == null || !isFinite(x)) return "—";
  return x.toFixed(3);
}
function fmtRange(a, b) {
  if (a == null || b == null) return "—";
  return `${a.toFixed(2)}…${b.toFixed(2)}`;
}
function setScatterMeta(html) {
  scatterMeta.innerHTML = html;
}

function drawScatter(pairs) {
  const svg = scatterSvg;
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const rect = svg.getBoundingClientRect();
  const W = Math.max(160, rect.width);
  const H = Math.max(120, rect.height);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const padL = 42, padR = 12, padT = 10, padB = 30;

  if (!pairs.length) {
    const NS = "http://www.w3.org/2000/svg";
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", W / 2); t.setAttribute("y", H / 2);
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("fill", "#555");
    t.setAttribute("font-size", "12");
    t.textContent = "(no scatter — pick a frame)";
    svg.appendChild(t);
    return;
  }

  const xs = pairs.map((p) => p[0]);
  const ys = pairs.map((p) => p[1]);
  // Clip the visible window to a robust 1-99 percentile per axis so a
  // few crazy values can't squash the bulk of the cloud into a corner.
  // Points outside the window simply don't get drawn.
  const xLo = pct(xs, 0.01), xHi = pct(xs, 0.99);
  const yLo = pct(ys, 0.01), yHi = pct(ys, 0.99);
  // Use a shared range across both axes — both `min` and `max` are
  // identical for x and y. With the same data range AND the same
  // pixel scale (computed below) the y=x diagonal renders at exactly
  // 45° regardless of the panel's aspect.
  const allLo = Math.min(xLo, yLo);
  const allHi = Math.max(xHi, yHi);
  const span  = Math.max(0.1, allHi - allLo);
  const min   = Math.max(0, allLo - 0.05 * span);
  const max   = allHi + 0.05 * span;

  // Equal data-to-pixel scaling: pick the smaller of the available
  // width/height in pixels and use it for both axes. The plot region
  // is then a centred square inside the panel — the diagonal looks
  // 45° as required, and labels still fit in the padding gutters.
  const usableW = W - padL - padR;
  const usableH = H - padT - padB;
  const side    = Math.max(20, Math.min(usableW, usableH));
  const scale   = side / (max - min);
  // Centre the plot region within the available area.
  const x0 = padL + Math.max(0, (usableW - side) / 2);
  const y1 = H - padB - Math.max(0, (usableH - side) / 2);

  const sx = (x) => x0 + (x - min) * scale;
  const sy = (y) => y1 - (y - min) * scale;

  // Plot square's bounds (drawn in viewBox units, which equal CSS px).
  const xR = x0 + side;
  const yT = y1 - side;
  const NS = "http://www.w3.org/2000/svg";
  const axis = document.createElementNS(NS, "g");
  axis.setAttribute("class", "axis");
  axis.appendChild(makeLine(NS, x0, yT, x0, y1));                  // y axis
  axis.appendChild(makeLine(NS, x0, y1, xR, y1));                  // x axis
  // y=x diagonal — with equal axis scaling this is exactly 45°.
  axis.appendChild(makeLine(NS, sx(min), sy(min), sx(max), sy(max), "yx"));
  // Tick labels at 4 even points.
  for (let k = 0; k <= 3; k++) {
    const t = min + (max - min) * (k / 3);
    const xt = sx(t);
    const yt = sy(t);
    const lblX = document.createElementNS(NS, "text");
    lblX.setAttribute("x", xt);
    lblX.setAttribute("y", y1 + 14);
    lblX.setAttribute("text-anchor", "middle");
    lblX.textContent = t.toFixed(2);
    axis.appendChild(lblX);
    const lblY = document.createElementNS(NS, "text");
    lblY.setAttribute("x", x0 - 4);
    lblY.setAttribute("y", yt + 3);
    lblY.setAttribute("text-anchor", "end");
    lblY.textContent = t.toFixed(2);
    axis.appendChild(lblY);
    if (k > 0 && k < 3) {
      axis.appendChild(makeLine(NS, x0, yt, xR, yt, "gridline"));
      axis.appendChild(makeLine(NS, xt, yT, xt, y1, "gridline"));
    }
  }
  svg.appendChild(axis);

  // Axis labels — pinned to the corners of the plot square so they
  // stay readable even when the panel is letterboxed.
  const xlbl = document.createElementNS(NS, "text");
  xlbl.setAttribute("class", "axis-label");
  xlbl.setAttribute("x", xR);
  xlbl.setAttribute("y", y1 - 4);
  xlbl.setAttribute("text-anchor", "end");
  xlbl.textContent = "phone depth (m)";
  svg.appendChild(xlbl);
  const ylbl = document.createElementNS(NS, "text");
  ylbl.setAttribute("class", "axis-label");
  ylbl.setAttribute("x", x0 + 4);
  ylbl.setAttribute("y", yT + 10);
  ylbl.setAttribute("text-anchor", "start");
  ylbl.textContent = "model raw";
  svg.appendChild(ylbl);

  // Points — additive translucent fill so dense regions read as bright
  // clusters. We draw with one <circle> per point; with N≤5000 and
  // r=1.4 this is well within SVG's comfort zone on the laptops we
  // care about.
  for (const [x, y] of pairs) {
    if (x < min || x > max || y < min || y > max) continue;
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("cx", sx(x));
    c.setAttribute("cy", sy(y));
    c.setAttribute("r", 1.4);
    c.setAttribute("class", "point");
    svg.appendChild(c);
  }
}

function pct(arr, p) {
  if (!arr.length) return 0;
  const sorted = arr.slice().sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1,
                       Math.max(0, Math.floor(p * (sorted.length - 1))));
  return sorted[idx];
}

function makeLine(NS, x1, y1, x2, y2, cls) {
  const l = document.createElementNS(NS, "line");
  l.setAttribute("x1", x1); l.setAttribute("y1", y1);
  l.setAttribute("x2", x2); l.setAttribute("y2", y2);
  if (cls) l.setAttribute("class", cls);
  return l;
}

// ---------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------

reloadBtn.addEventListener("click", () => {
  refreshSessionList().then(refreshFrameManifest);
});
sessionSel.addEventListener("change", () => {
  selectedFrame = null; lastScatter = null;
  populatePosePicker();
  refreshFrameManifest();
  // Reset the panels.
  resetPanel(panelRgb,   "(no frame selected)");
  resetPanel(panelPhone, "—");
  resetPanel(panelModel, "—");
  setOverlayImg(null);
  drawScatter([]); setScatterMeta("(no data)");
  stStatus.textContent = "pick a frame →";
});
poseDirSel.addEventListener("change", () => {
  // Pose change doesn't affect depth values served (model_raw is
  // pose-independent; phone depth bytes are baked into the chosen
  // frames/* dir). For now we only use pose_dir to scope the manifest,
  // so just re-fetch.
  refreshFrameManifest();
});
for (const r of overlayRadios) {
  r.addEventListener("change", () => {
    applyOverlay(getOverlayMode(), selectedFrame, 800);
  });
}
overlayOpacity.addEventListener("input", () => {
  const overlay = panelRgb.querySelector("img.overlay-img");
  if (overlay) overlay.style.opacity = String(parseFloat(overlayOpacity.value));
});

window.addEventListener("resize", () => renderScatterFromState());

function resetPanel(panel, emptyText) {
  const wrap = panel.querySelector(".panel-img-wrap");
  wrap.innerHTML = `<span class="panel-empty">${emptyText}</span>`;
}

// Boot.
refreshSessionList().then(() => {
  refreshFrameManifest();
});
