/* R1 Greenhouse Scout - live dashboard. Polls /api/state every second and renders an SVG top-down map,
   robot pose + trail, capture ticks, finding pins, a finding detail panel and the HERMES decision feed.
   Coordinates: house-local metres (aisle_map.json), door at +Y. heading_deg = yaw in the house frame,
   degrees counter-clockwise from +X (driving from the door toward -Y = -90). No dependencies. */
"use strict";

const POLL_MS = 1000;
const ROBOT_WIDTH_M = 0.528, SIDE_MARGIN_M = 0.06;        // keep in sync with r1_nav/aisle_map.py
const SIDE_CAMS = ["mast_left_high", "mast_left_mid", "mast_left_low", "mast_right_high", "mast_right_mid", "mast_right_low"];
const SEVERITY = { disease: 3, suspect: 2, healthy: 1, not_plant: 0 };
const STATUS_LABEL = { healthy: "healthy", suspect: "suspect", disease: "disease found", not_plant: "not a plant" };
const STATE_LABEL = { surveying: "surveying an aisle", transit: "moving between aisles", hold: "holding", revisit: "re-inspecting a spot",
  idle: "idle", docked: "docked, charging", stopped: "stopped", returning: "returning to dock", error: "error" };
const SVG_NS = "http://www.w3.org/2000/svg";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtM = (m) => (m == null || isNaN(m) ? "—" : Number(m).toFixed(1) + " m");
const key = (aisle, metre) => `${aisle}|${(Math.round(Number(metre) * 2) / 2).toFixed(1)}`;   // 0.5 m buckets
const shortAisle = (id) => String(id || "").split("/").pop();
const clock = (ts) => (ts ? new Date(ts * 1000).toLocaleTimeString([], { hour12: false }) : "");
const sideOf = (cam) => (/left/.test(cam) ? "left" : /right/.test(cam) ? "right" : "front");

const ui = {
  state: null, orient: null, mapSig: "", dynSig: "", detailSig: "", captureSig: "", feedSig: "",
  selected: null,            // {aisle, metre} chosen by click; null = auto-follow
  autoSel: null,
  statusTrail: [],           // [{ts, aisle, metre}] seen across polls
  seenDecisions: new Set(), firstFeed: true,
  view: null, geom: null, showSlam: true, lastOk: 0,
};

// ---------------------------------------------------------------- geometry

function buildGeom(map) {
  const aisles = (map.aisles || []).map((a) => {
    const s = a.start_local, e = a.end_local;
    const len = Math.hypot(e[0] - s[0], e[1] - s[1]) || 1;
    return {
      id: a.id, label: a.label, clearance: a.clearance_m, length: a.length_m || len,
      sx: s[0], sy: s[1], ux: (e[0] - s[0]) / len, uy: (e[1] - s[1]) / len,
      ok: a.clearance_m >= ROBOT_WIDTH_M + 2 * SIDE_MARGIN_M,
    };
  });
  const beds = (map.bed_bounds_local_m || []).map((b) => ({ x0: b.min[0], y0: b.min[1], x1: b.max[0], y1: b.max[1] }));
  const xs = beds.flatMap((b) => [b.x0, b.x1]).concat(aisles.flatMap((a) => [a.sx, a.sx + a.ux * a.length]));
  const ys = beds.flatMap((b) => [b.y0, b.y1]).concat(aisles.flatMap((a) => [a.sy, a.sy + a.uy * a.length]));
  const pad = 0.45;
  return { aisles, beds, house: map.source?.feature_id || map.map_id || "",
    xmin: Math.min(...xs) - pad, xmax: Math.max(...xs) + pad, ymin: Math.min(...ys) - pad, ymax: Math.max(...ys) + pad };
}

/* Affine house-local -> SVG px, as an SVG matrix [a b c d e f]: px = a*x + c*y + e, py = b*x + d*y + f.
   Horizontal (desktop): door (+Y) on the left, +X up. Vertical (phone): door at top, +X right. */
function buildView(g, orient) {
  const W = g.xmax - g.xmin, L = g.ymax - g.ymin;
  if (orient === "h") {
    const sL = 1520 / L, sT = Math.min(56, sL * 2.2), ox = 54, oy = 34;
    const m = [0, -sT, -sL, 0, ox + g.ymax * sL, oy + g.xmax * sT];
    return { m, w: ox * 2 + L * sL - 20, h: oy + W * sT + 30, sL, sT, orient };
  }
  const sT = 64, sL = Math.min(22, 820 / L), ox = 42, oy = 34;
  const m = [sT, 0, 0, -sL, ox - g.xmin * sT, oy + g.ymax * sL];
  return { m, w: ox + W * sT + 20, h: oy + L * sL + 24, sL, sT, orient };
}
const P = (v, x, y) => [v.m[0] * x + v.m[2] * y + v.m[4], v.m[1] * x + v.m[3] * y + v.m[5]];
const D = (v, dx, dy) => [v.m[0] * dx + v.m[2] * dy, v.m[1] * dx + v.m[3] * dy];
const mul = (A, B) => [A[0] * B[0] + A[2] * B[1], A[1] * B[0] + A[3] * B[1], A[0] * B[2] + A[2] * B[3],
  A[1] * B[2] + A[3] * B[3], A[0] * B[4] + A[2] * B[5] + A[4], A[1] * B[4] + A[3] * B[5] + A[5]];

function aislePoint(g, aisleId, metre, lateral = 0) {
  const a = g.aisles.find((x) => x.id === aisleId);
  if (!a || metre == null) return null;
  const m = Math.max(0, Math.min(a.length, Number(metre)));
  // lateral > 0 = to the left when facing along the aisle (start -> end)
  return [a.sx + a.ux * m - a.uy * lateral, a.sy + a.uy * m + a.ux * lateral];
}

function rectPx(v, x0, y0, x1, y1) {
  const [ax, ay] = P(v, x0, y0), [bx, by] = P(v, x1, y1);
  return { x: Math.min(ax, bx), y: Math.min(ay, by), width: Math.abs(bx - ax), height: Math.abs(by - ay) };
}

function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(SVG_NS, tag);
  for (const [k, val] of Object.entries(attrs)) if (val != null) n.setAttribute(k, val);
  if (parent) parent.appendChild(n);
  return n;
}

// ---------------------------------------------------------------- data indexing

function indexState(s) {
  const frames = (s.frames || []).slice().sort((a, b) => (a.ts || 0) - (b.ts || 0));
  const findings = s.findings || [];
  const frameById = new Map(frames.map((f) => [f.frame_id, f]));
  const byKey = new Map();           // key -> {aisle, metre, frames: {cam: [frames]}, findings: []}
  const bucket = (aisle, metre) => {
    const k = key(aisle, metre);
    if (!byKey.has(k)) byKey.set(k, { k, aisle, metre: Math.round(Number(metre) * 2) / 2, frames: {}, findings: [], worst: null });
    return byKey.get(k);
  };
  for (const f of frames) {
    if (f.aisle == null || f.metre == null) continue;
    const b = bucket(f.aisle, f.metre);
    (b.frames[f.camera] ||= []).push(f);
  }
  for (const r of findings) {
    const fr = frameById.get(r.frame_id);
    const aisle = r.aisle ?? fr?.aisle, metre = r.metre ?? fr?.metre;
    if (aisle == null || metre == null) continue;
    const b = bucket(aisle, metre);
    b.findings.push(Object.assign({ camera: fr?.camera }, r, { _frame: fr }));
    if (b.worst == null || (SEVERITY[r.status] ?? -1) > (SEVERITY[b.worst] ?? -1)) b.worst = r.status;
  }
  const findingByFrame = new Map();   // Cosmos and RT-DETR may both report a frame: merge issues, worst status wins
  for (const r of findings) {
    const prev = findingByFrame.get(r.frame_id);
    if (!prev) { findingByFrame.set(r.frame_id, r); continue; }
    const worse = (SEVERITY[r.status] ?? -1) > (SEVERITY[prev.status] ?? -1) ? r.status : prev.status;
    findingByFrame.set(r.frame_id, Object.assign({}, prev, r, { status: worse, issues: [...(prev.issues || []), ...(r.issues || [])] }));
  }
  return { frames, findings, frameById, byKey, findingByFrame };
}

// harvest totals from RT-DETR findings: latest finding per (aisle, 0.5 m, camera), so repeat passes do not add up
function harvestTotals(findings) {
  const latest = new Map();
  for (const r of findings) {
    if (r.source !== "rtdetr" || !r.harvest) continue;
    const k = `${key(r.aisle, r.metre)}|${r.camera}`;
    const prev = latest.get(k);
    if (!prev || (r.frame_id ?? 0) >= (prev.frame_id ?? 0)) latest.set(k, r);
  }
  const t = { plants: 0, pass: 0, not_ready: 0, reject: 0, frames: latest.size };
  for (const r of latest.values()) {
    t.pass += r.harvest.pass || 0; t.not_ready += r.harvest.not_ready || 0; t.reject += r.harvest.reject || 0;
    t.plants += (r.plants || []).length;
  }
  return t;
}

// ---------------------------------------------------------------- map rendering

function ensureMap(s) {
  const svg = $("map");
  const orient = $("mapWrap").clientWidth < 700 ? "v" : "h";
  const sig = JSON.stringify([s.aisle_map?.map_id, s.aisle_map?.aisles?.length, orient, s.slam?.topdown_url, s.slam?.topdown_mtime, ui.showSlam]);
  if (!s.aisle_map) { $("mapEmpty").hidden = false; svg.innerHTML = ""; ui.geom = null; ui.mapSig = ""; return false; }
  $("mapEmpty").hidden = true;
  if (sig === ui.mapSig) return true;
  ui.mapSig = sig; ui.dynSig = ""; ui.orient = orient;
  const g = (ui.geom = buildGeom(s.aisle_map));
  const v = (ui.view = buildView(g, orient));
  svg.innerHTML = "";
  svg.setAttribute("viewBox", `0 0 ${v.w.toFixed(0)} ${v.h.toFixed(0)}`);
  const defs = el("defs", {}, svg);
  const hatch = el("pattern", { id: "hatch", width: 7, height: 7, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, defs);
  el("rect", { width: 7, height: 7, fill: "rgba(244,63,94,.06)" }, hatch);
  el("line", { x1: 0, y1: 0, x2: 0, y2: 7, stroke: "rgba(244,63,94,.45)", "stroke-width": 2.5 }, hatch);
  const bedPat = el("pattern", { id: "bedrows", width: 9, height: 9, patternUnits: "userSpaceOnUse" }, defs);
  el("rect", { width: 9, height: 9, fill: "#1c3a2c" }, bedPat);
  el("circle", { cx: 4.5, cy: 4.5, r: 2.2, fill: "#2c5a40" }, bedPat);

  const base = el("g", { id: "base" }, svg);
  el("rect", Object.assign(rectPx(v, g.xmin, g.ymin, g.xmax, g.ymax), { class: "shell", rx: 12 }), base);

  // SLAM top-down overlay: image +x = local +X, top row = max local Y, stretched over bounds_local.
  const sb = s.slam?.bounds_local;
  if (s.slam?.topdown_url && sb?.min && sb?.max && ui.showSlam) {
    const [x0, y0] = sb.min, [x1, y1] = sb.max;
    const M = mul(v.m, [x1 - x0, 0, 0, -(y1 - y0), x0, y1]);
    el("image", { href: s.slam.topdown_url + "?t=" + (s.slam.topdown_mtime || 0), x: 0, y: 0, width: 1, height: 1,
      preserveAspectRatio: "none", class: "slam-img", transform: `matrix(${M.map((n) => n.toFixed(5)).join(" ")})` }, base);
  }
  $("slamToggleWrap").hidden = !s.slam?.topdown_url;

  for (const b of g.beds) el("rect", Object.assign(rectPx(v, b.x0, b.y0, b.x1, b.y1), { class: "bed", fill: "url(#bedrows)", rx: 2 }), base);
  for (const a of g.aisles) {
    const half = a.clearance / 2;
    const p0 = [a.sx - a.uy * half, a.sy + a.ux * half], p1 = [a.sx + a.ux * a.length + a.uy * half, a.sy + a.uy * a.length - a.ux * half];
    el("rect", Object.assign(rectPx(v, p0[0], p0[1], p1[0], p1[1]), { class: a.ok ? "aisle-ok" : "aisle-narrow", "data-aisle": a.id }), base);
    if (a.ok) {
      const [cx0, cy0] = P(v, a.sx, a.sy), [cx1, cy1] = P(v, a.sx + a.ux * a.length, a.sy + a.uy * a.length);
      el("line", { x1: cx0, y1: cy0, x2: cx1, y2: cy1, class: "centerline" }, base);
    }
  }
  // metre axis (relative to the first aisle's start; all aisles share the door-side start in this map)
  const ref = g.aisles.find((a) => a.ok) || g.aisles[0];
  if (ref) {
    const axis = el("g", { class: "axis" }, base);
    for (let m = 0; m <= ref.length + 1e-6; m += 5) {
      const [x, y] = P(v, ref.sx + ref.ux * m, ref.sy + ref.uy * m);
      if (v.orient === "h") {
        el("line", { x1: x, y1: 22, x2: x, y2: 30 }, axis);
        el("text", { x, y: 17, "text-anchor": "middle" }, axis).textContent = m + (m === 0 ? " m" : "");
      } else {
        el("line", { x1: 30, y1: y, x2: 38, y2: y }, axis);
        el("text", { x: 26, y: y + 4, "text-anchor": "end" }, axis).textContent = m;
      }
    }
    // door marker at the aisle head (+Y end)
    const [dx0, dy0] = P(v, g.xmin + 0.6, g.ymax), [dx1, dy1] = P(v, g.xmax - 0.6, g.ymax);
    el("line", { x1: dx0, y1: dy0, x2: dx1, y2: dy1, class: "door-mark" }, base);
    const [tx, ty] = P(v, (g.xmin + g.xmax) / 2, g.ymax);
    const t = el("text", { x: v.orient === "h" ? tx - 12 : tx, y: v.orient === "h" ? ty : ty - 8, class: "door",
      "text-anchor": "middle", transform: v.orient === "h" ? `rotate(-90 ${tx - 12} ${ty})` : null }, base);
    t.textContent = "DOOR";
  }
  // aisle labels inside each aisle, near the far end
  for (const a of g.aisles) {
    const [x, y] = P(v, a.sx + a.ux * (a.length - 0.4), a.sy + a.uy * (a.length - 0.4));
    const txt = `${shortAisle(a.id)} · ${a.clearance.toFixed(2)} m${a.ok ? "" : " · too narrow ✕"}`;
    const attrs = v.orient === "h" ? { x, y: y + 4, "text-anchor": "end" }
      : { x: x + 4, y, "text-anchor": "start", transform: `rotate(-90 ${x + 4} ${y})` };
    el("text", Object.assign(attrs, { class: "aisle-label" + (a.ok ? "" : " narrow") }), base).textContent = txt;
  }
  // honest scale note: the lateral axis is stretched so 0.5 m aisles stay visible
  const ex = v.sT / v.sL;
  if (ex > 1.05) el("text", { x: v.w - 8, y: v.h - 8, "text-anchor": "end", class: "scale-note" }, base).textContent = `lateral scale ×${ex.toFixed(1)}`;
  el("g", { id: "slamTraj" }, svg);
  el("g", { id: "ticks" }, svg);
  el("g", { id: "trail" }, svg);
  el("g", { id: "pins" }, svg);
  buildRobot(svg, v);
  return true;
}

function buildRobot(svg, v) {
  const r = el("g", { id: "robot", class: "robot", style: "opacity:0" }, svg);
  const inner = el("g", { id: "robotInner" }, r);
  const Wpx = Math.max(16, ROBOT_WIDTH_M * v.sT), len = Math.max(26, 0.72 * v.sL);   // R1 footprint, min size for video
  el("circle", { r: len * 0.95, class: "robot-glow" }, inner);
  el("rect", { x: -len / 2, y: -Wpx / 2, width: len, height: Wpx, rx: 5, class: "robot-body" }, inner);
  el("path", { d: `M ${len / 2 + 9} 0 L ${len / 2 + 1} -6 L ${len / 2 + 1} 6 Z`, class: "robot-arrow" }, inner);
  const lbl = el("text", { id: "robotLabel", x: 0, y: 3.5, "text-anchor": "middle", class: "robot-label" }, r);
  lbl.textContent = "R1";
  el("text", { id: "robotMetre", x: 0, y: -Wpx / 2 - 12, "text-anchor": "middle", class: "metre-flag" }, r);
}

function updateRobot(s) {
  const r = $("robot");
  if (!r || !ui.geom) return;
  const st = s.status;
  const pt = st && aislePoint(ui.geom, st.aisle, st.metre);
  if (!pt) { r.style.opacity = 0; return; }
  const v = ui.view;
  const [x, y] = P(v, pt[0], pt[1]);
  const a = ui.geom.aisles.find((q) => q.id === st.aisle);
  let h = st.heading_deg;
  let dir = h != null ? D(v, Math.cos((h * Math.PI) / 180), Math.sin((h * Math.PI) / 180)) : D(v, a.ux, a.uy);
  const ang = (Math.atan2(dir[1], dir[0]) * 180) / Math.PI;
  r.style.opacity = 1;
  r.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
  $("robotInner").setAttribute("transform", `rotate(${ang.toFixed(1)})`);
  $("robotMetre").textContent = fmtM(st.metre);
}

function renderDynamic(s, idx) {
  if (!ui.geom) return;
  const g = ui.geom, v = ui.view;
  const sel = ui.selected || ui.autoSel;
  const sig = JSON.stringify([idx.frames.length, idx.findings.length, ui.statusTrail.length, sel && key(sel.aisle, sel.metre), s.slam?.poses?.length, s.status?.aisle]);
  if (sig === ui.dynSig) return;
  ui.dynSig = sig;
  document.querySelectorAll("#base rect[data-aisle]").forEach((n) => n.classList.toggle("active", n.dataset.aisle === s.status?.aisle));

  // SLAM trajectory (house-local x/y)
  const st = $("slamTraj"); st.innerHTML = "";
  const poses = (s.slam?.poses || []).filter((p) => p && p.x != null && p.y != null);
  if (poses.length > 1 && ui.showSlam) el("polyline", { class: "slam-traj", points: poses.map((p) => P(v, p.x, p.y).map((n) => n.toFixed(1)).join(",")).join(" ") }, st);

  // capture ticks every 0.5 m (side-camera captures)
  const tk = $("ticks"); tk.innerHTML = "";
  for (const b of idx.byKey.values()) {
    const a = g.aisles.find((q) => q.id === b.aisle);
    if (!a || !Object.keys(b.frames).some((c) => c !== "front")) continue;
    const half = Math.min(a.clearance / 2, 0.5) * 0.8;
    const p0 = aislePoint(g, b.aisle, b.metre, half), p1 = aislePoint(g, b.aisle, b.metre, -half);
    const [x0, y0] = P(v, ...p0), [x1, y1] = P(v, ...p1);
    const rev = Object.values(b.frames).some((fs) => fs.some((f) => f.purpose === "revisit"));
    el("line", { x1: x0, y1: y0, x2: x1, y2: y1, class: "tick" + (rev ? " revisit" : "") }, tk);
  }

  // trail: frame positions + status poses, ordered by time; offset by travel direction so back-and-forth reads
  const tr = $("trail"); tr.innerHTML = "";
  const pts = idx.frames.filter((f) => f.camera === "front" || SIDE_CAMS.includes(f.camera)).map((f) => ({ ts: f.ts, aisle: f.aisle, metre: f.metre }))
    .concat(ui.statusTrail).sort((p, q) => (p.ts || 0) - (q.ts || 0));
  let seg = [], lastAisle = null, prev = null;
  const flush = () => { if (seg.length > 1) el("polyline", { class: "trail", points: seg.join(" ") }, tr); seg = []; };
  for (const p of pts) {
    if (p.aisle !== lastAisle) { flush(); lastAisle = p.aisle; prev = null; }
    if (prev && Math.abs(prev.metre - p.metre) < 1e-3) continue;
    const fwd = prev ? p.metre >= prev.metre : true;
    const off = fwd ? -0.14 : 0.14;
    if (prev && (prev.fwd !== fwd)) {                       // turn-around: bridge the two lanes
      const q = aislePoint(g, p.aisle, prev.metre, off); if (q) seg.push(P(v, ...q).map((n) => n.toFixed(1)).join(","));
    }
    const q = aislePoint(g, p.aisle, p.metre, off);
    if (q) seg.push(P(v, ...q).map((n) => n.toFixed(1)).join(","));
    prev = { metre: p.metre, fwd };
  }
  flush();

  // finding pins: one per (aisle, 0.5 m, side), worst status wins; healthy are small dots
  const pn = $("pins"); pn.innerHTML = "";
  const groups = [];
  for (const b of idx.byKey.values()) {
    const a = g.aisles.find((q) => q.id === b.aisle);
    if (!a || !b.findings.length) continue;
    for (const side of ["left", "right", "front"]) {
      const fs = b.findings.filter((r) => sideOf(r.camera || "") === side);
      if (!fs.length) continue;
      const worst = fs.reduce((w, r) => ((SEVERITY[r.status] ?? -1) > (SEVERITY[w] ?? -1) ? r.status : w), null);
      // camera side -> which bed: use the frame heading (left of travel direction)
      const fr = fs[0]._frame;
      let lat = side === "front" ? 0 : a.clearance / 2 + 0.16;
      if (side !== "front") {
        let leftIsPlus = true;
        if (fr && fr.heading_deg != null) {
          const h = (fr.heading_deg * Math.PI) / 180;
          leftIsPlus = Math.cos(h) * a.ux + Math.sin(h) * a.uy >= 0;   // facing along the aisle direction?
        }
        lat *= (side === "left") === leftIsPlus ? 1 : -1;
      }
      groups.push({ b, worst, lat, side });
    }
  }
  groups.sort((p, q) => (SEVERITY[p.worst] ?? -1) - (SEVERITY[q.worst] ?? -1));   // severe drawn last (on top)
  for (const gr of groups) {
    const p = aislePoint(g, gr.b.aisle, gr.b.metre, gr.lat);
    const [x, y] = P(v, ...p);
    const big = gr.worst === "disease" || gr.worst === "suspect";
    const node = el("g", { "data-k": gr.b.k }, pn);
    if (gr.worst === "disease") el("circle", { cx: x, cy: y, r: 9, class: "pin-ring" }, node);
    el("circle", { cx: x, cy: y, r: big ? (gr.worst === "disease" ? 8 : 6.5) : 2.6, class: "pin " + gr.worst }, node);
    el("circle", { cx: x, cy: y, r: big ? 13 : 6, class: "hit" }, node);
    if (sel && key(sel.aisle, sel.metre) === gr.b.k && big) el("circle", { cx: x, cy: y, r: 14, class: "sel-ring" }, node);
    const t = el("title", {}, node);
    t.textContent = `${shortAisle(gr.b.aisle)} @ ${gr.b.metre.toFixed(1)} m · ${gr.side} · ${gr.worst}`;
    node.addEventListener("click", () => select(gr.b.aisle, gr.b.metre, true));
  }
}

// ---------------------------------------------------------------- panels

function thumbHTML(frame, finding, opts = {}) {
  const cam = opts.cam || frame?.camera || "";
  const portrait = /^mast_/.test(cam) ? " portrait" : "";
  if (!frame || !frame.image_url) return `<div class="thumb missing"><span>${esc(cam.replace("mast_", ""))}<br>no image</span></div>`;
  const st = finding?.status || "";
  const boxes = (finding?.issues || []).filter((i) => Array.isArray(i.bbox_norm) && i.bbox_norm.length === 4).map((i) => {
    const [x, y, x1, y1] = i.bbox_norm.map(Number);                                // contract: [x0,y0,x1,y1] normalised
    const w = Math.max(0, x1 - x), h = Math.max(0, y1 - y);
    return `<div class="bbox ${st === "suspect" ? "suspect" : ""} ${x + w > 0.6 ? "r" : ""}" style="left:${x * 100}%;top:${y * 100}%;width:${w * 100}%;height:${h * 100}%">` +
      (opts.labels ? `<span>${esc(String(i.label || "").split(/[(,]/)[0].trim())} ${(Number(i.confidence) * 100).toFixed(0)}%</span>` : "") + "</div>";
  }).join("");
  const rev = frame.purpose === "revisit" ? `<span class="badge revisit">revisit</span>` : "";
  return `<div class="thumb ${esc(st)}${portrait}" data-img="${esc(frame.image_url)}" data-cap="${esc(`${frame.camera} · ${shortAisle(frame.aisle)} @ ${fmtM(frame.metre)} · frame ${frame.frame_id}${st ? " · " + st : ""}`)}">` +
    `<img loading="lazy" src="${esc(frame.image_url)}" alt="${esc(cam)}">${boxes}` +
    `<div class="cap"><span>${esc(cam.replace("mast_", ""))}</span>${rev}</div></div>`;
}

function select(aisle, metre, user) {
  if (user) ui.selected = { aisle, metre };
  ui.detailSig = ""; ui.dynSig = "";
  if (ui.state) render(ui.state);
}

function chooseAuto(idx, s) {
  // auto-follow: newest disease/suspect bucket, else the robot's current metre
  let best = null, bestTs = -1;
  for (const b of idx.byKey.values()) {
    if (b.worst !== "disease" && b.worst !== "suspect") continue;
    const ts = Math.max(...b.findings.map((f) => f.ts || f._frame?.ts || 0)) + (b.worst === "disease" ? 1e6 : 0);
    if (ts > bestTs) { bestTs = ts; best = b; }
  }
  if (best) return { aisle: best.aisle, metre: best.metre };
  if (s.status?.aisle != null) return { aisle: s.status.aisle, metre: s.status.metre };
  return null;
}

function renderDetail(s, idx) {
  const sel = ui.selected || ui.autoSel;
  const b = sel && idx.byKey.get(key(sel.aisle, sel.metre));
  const sig = JSON.stringify([sel && key(sel.aisle, sel.metre), b && b.findings.length, b && Object.values(b.frames).flat().length]);
  if (sig === ui.detailSig) return;
  ui.detailSig = sig;
  $("detailHint").textContent = ui.selected ? "pinned · click map to change" : "auto-follow · click map to pin";
  if (!b) { $("detail").innerHTML = `<div class="empty">No findings yet</div>`; return; }
  const latest = (cam) => (b.frames[cam] || []).slice(-1)[0];
  const findFor = (fr) => fr && idx.findingByFrame.get(fr.frame_id);
  const worst = b.worst || "healthy";
  const cams = SIDE_CAMS.map((c) => thumbHTML(latest(c), findFor(latest(c)), { cam: c, labels: false })).join("");
  const passes = Math.max(0, ...SIDE_CAMS.map((c) => (b.frames[c] || []).length));
  const issues = b.findings.filter((r) => (r.issues || []).length).sort((p, q) => (q.ts || 0) - (p.ts || 0));
  const issueHTML = issues.flatMap((r) => r.issues.map((i) => `
    <div class="issue ${esc(r.status)}">
      <div class="top"><span class="label">${esc(i.label)}</span>
        <span class="conf"><span class="meter"><span style="width:${Math.round((+i.confidence || 0) * 100)}%;background:var(--${r.status === "suspect" ? "suspect" : "disease"})"></span></span>${((+i.confidence || 0) * 100).toFixed(0)}%</span></div>
      ${i.evidence ? `<div class="ev">${esc(i.evidence)}</div>` : ""}
      <div class="meta">${esc(r.camera || "")} · frame ${esc(r.frame_id)} · ${esc(r.purpose || r._frame?.purpose || "inspect")}${r.model ? " · " + esc(r.model) : ""}</div>
    </div>`)).join("");
  // health summaries only; the detector's summary line is about fruit grading and lives in the harvest box
  const sums = [...new Set(b.findings.filter((r) => r.status !== "healthy" && r.source !== "rtdetr").map((r) => r.summary).filter(Boolean))].slice(0, 3);
  const counts = b.findings.reduce((c, r) => ((c[r.status] = (c[r.status] || 0) + 1), c), {});
  const h = harvestTotals(b.findings);
  const harvestHTML = h.frames ? `<div class="harvest-title">Harvest at this metre</div><div class="harvest">
      <span><b>${h.plants}</b> plants</span><span class="pass"><b>${h.pass}</b> ripe · pass</span>
      <span class="notready"><b>${h.not_ready}</b> not ready</span><span class="reject"><b>${h.reject}</b> reject</span></div>` : "";
  $("detail").innerHTML = `
    <div class="detail-title"><span class="where">${esc(shortAisle(b.aisle))} @ ${b.metre.toFixed(1)} m</span>
      <span class="status ${esc(worst)}">${STATUS_LABEL[worst] || esc(worst.replace("_", " "))}</span>
      <span class="muted">${passes > 1 ? passes + " passes · " : ""}${Object.entries(counts).map(([k, n]) => `${n} ${k}`).join(", ")}</span></div>
    ${sums.map((t) => `<div class="summary">${esc(t)}</div>`).join("")}
    <div class="cams">${cams}</div>
    ${issueHTML || `<div class="empty">No issues at this metre</div>`}
    ${harvestHTML}`;
}

function renderCapture(s, idx) {
  // latest side-capture metre + latest front frame
  const side = [...idx.frames].reverse().find((f) => SIDE_CAMS.includes(f.camera));
  const front = [...idx.frames].reverse().find((f) => f.camera === "front");
  const sig = JSON.stringify([side?.frame_id, front?.frame_id, idx.findings.length]);
  if (sig === ui.captureSig) return;
  ui.captureSig = sig;
  if (!side && !front) { $("capture").innerHTML = `<div class="empty">Waiting for frames</div>`; return; }
  const b = side && idx.byKey.get(key(side.aisle, side.metre));
  const pick = (c) => (b?.frames[c] || []).slice(-1)[0];
  $("captureMeta").textContent = side ? `${shortAisle(side.aisle)} @ ${fmtM(side.metre)} · ${clock(side.ts)}` : "";
  const thumbs = (cams) => cams.map((c) => thumbHTML(pick(c), idx.findingByFrame.get(pick(c)?.frame_id), { cam: c, labels: true })).join("");
  $("capture").innerHTML = `
    <div class="front">${thumbHTML(front, null, { cam: "front" })}
      <div class="muted mono" style="margin-top:6px">front · navigation check · ${front ? fmtM(front.metre) : "—"}</div></div>
    <div class="sides"><div class="side-label">Left bed</div>${thumbs(SIDE_CAMS.slice(0, 3))}
      <div class="side-label">Right bed</div>${thumbs(SIDE_CAMS.slice(3))}</div>`;
}

function renderFeed(s) {
  const ds = (s.decisions || []).slice().sort((a, b) => (b.ts || 0) - (a.ts || 0));
  const sig = JSON.stringify([ds.length, ds[0]?.ts]);
  if (sig === ui.feedSig) return;
  ui.feedSig = sig;
  $("feedCount").textContent = ds.length ? `${ds.length} · Nemotron` : "";
  if (!ds.length) { $("feed").innerHTML = `<li class="empty" style="border:0;background:none">No decisions yet</li>`; return; }
  $("feed").innerHTML = ds.map((d, i) => {
    const c = d.command || {};
    const id = `${d.ts}|${c.cmd}`;
    const fresh = !ui.firstFeed && !ui.seenDecisions.has(id);
    ui.seenDecisions.add(id);
    const reply = typeof d.reply === "string" ? { status: d.reply } : d.reply || {};
    const rs = String(reply.status || d.reply_status || "").toLowerCase();
    const args = Object.entries(c).filter(([k]) => !["cmd", "house"].includes(k))
      .map(([k, val]) => `${k}=${k === "aisle" ? shortAisle(val) : typeof val === "object" ? JSON.stringify(val) : val}`).join(" ");
    const clickable = c.aisle != null && c.metre != null;
    return `<li class="${rs} ${fresh ? "fresh" : ""} ${clickable ? "clickable" : ""}" ${clickable ? `data-aisle="${esc(c.aisle)}" data-metre="${esc(c.metre)}"` : ""}>
      <div class="row"><span class="cmd">${esc(c.cmd || "?")}${args ? " " + esc(args) : ""}</span>
        <span>${rs ? `<span class="reply ${esc(rs)}">${esc(rs)}</span> ` : ""}<span class="ts">${clock(d.ts)}</span></span></div>
      ${d.reasoning ? `<div class="why">${esc(d.reasoning)}</div>` : ""}
      ${reply.reason ? `<div class="reason">↳ ${esc(reply.reason)}</div>` : ""}</li>`;
  }).join("");
  ui.firstFeed = false;
}

function renderHeader(s) {
  const st = s.status || {};
  $("house").textContent = ui.geom?.house || st.house || "—";
  const state = String(st.state || "—");
  $("sState").innerHTML = `<span class="state-pill ${esc(state)}">${esc(state)}</span><i>${STATE_LABEL[state] || ""}</i>`;
  const a = ui.geom?.aisles.find((q) => q.id === st.aisle);
  $("sAisle").textContent = st.aisle ? `${shortAisle(st.aisle)}${a?.label ? " · " + a.label : ""}` : "—";
  $("sMetre").textContent = st.metre != null ? `${Number(st.metre).toFixed(2)}${a ? " / " + a.length.toFixed(1) : ""} m` : "—";
  const conf = Number(st.pose_confidence);
  $("sConf").textContent = isNaN(conf) ? "—" : conf.toFixed(2);
  const bar = $("sConfBar");
  bar.style.width = `${Math.max(0, Math.min(1, conf || 0)) * 100}%`;
  bar.style.background = conf < 0.5 ? "var(--disease)" : conf < 0.75 ? "var(--suspect)" : "var(--accent)";
  const batt = Number(st.battery);
  $("sBatt").textContent = isNaN(batt) ? "—" : `${Math.round(batt <= 1 ? batt * 100 : batt)}%`;
  $("sCmd").textContent = st.cmd ? `${st.cmd}${st.legs_remaining != null ? ` · ${st.legs_remaining} legs left` : ""}` : "—";

  const svc = s.services || {};
  const tags = { nemotron: ["NV", "nv"], cosmos: ["NV", "nv"], slam: ["NEBIUS", "nb"] };
  const order = [["nemotron", "Nemotron", "Nebius Token Factory"], ["cosmos", "Cosmos Reason", "Nebius Serverless Endpoint"], ["slam", "SLAM job", "Nebius GPU job"]];
  $("services").innerHTML = order.map(([k, name, where]) => {
    const x = svc[k] || {};
    const [tag, cls] = tags[k];
    return `<div class="svc ${x.live ? "live" : ""}" title="${esc(x.detail || "")}${x.model ? " · " + esc(x.model) : ""} (${esc(x.source || "")})">
      <span class="dot"></span><div><b>${esc(x.name || name)}<span class="tag ${cls}">${tag}</span></b><span class="w">@ ${esc(x.where || where)} · ${x.live ? "live" : "offline"}</span></div></div>`;
  }).join("");
}

function renderKpis(s, idx) {
  const c = { healthy: 0, suspect: 0, disease: 0, not_plant: 0 };
  for (const r of idx.findings) if (r.status in c) c[r.status]++;
  const st = s.status || {};
  const a = ui.geom?.aisles.find((q) => q.id === st.aisle);
  let cov = "—";
  if (a) {
    const ms = new Set(idx.frames.filter((f) => f.aisle === a.id && SIDE_CAMS.includes(f.camera)).map((f) => Math.round(f.metre * 2)));
    cov = `${Math.min(100, Math.round((ms.size / (Math.floor(a.length / 0.5) + 1)) * 100))}%`;
  }
  // k(cssClass, label, hint, value): hint is an optional small secondary label (e.g. which models produced it).
  const k = (cls, label, hint, v) => `<div class="kpi ${cls}"><div class="k">${label}${hint ? `<i>${hint}</i>` : ""}</div><div class="v">${v}</div></div>`;
  const group = (title, hint) => `<div class="kpi-group">${title}${hint ? `<i>${hint}</i>` : ""}</div>`;
  // Two separate jobs, two separate panels: plant health (disease screening) and harvest (fruit grading / count).
  const det = { suspect: 0, disease: 0 };
  for (const r of idx.findings) if (r.source === "rtdetr" && r.status in det) det[r.status]++;
  $("kpis").innerHTML = k("", "Frames", "", idx.frames.length) + k("", "Aisle coverage", "side cameras", cov) +
    group("Plant health · disease screening", "Cosmos + Nemotron-VL + RT-DETR") +
    k("healthy", "Healthy", "", c.healthy) + k("suspect", "Suspect", "", c.suspect) +
    k("disease", "Disease", "", c.disease) + k("", "Not plant", "", c.not_plant) +
    (det.suspect + det.disease ? k("suspect", "Detector flags", "RT-DETR leaf anomalies", det.suspect + det.disease) : "");
  const h = harvestTotals(idx.findings);
  if (h.frames) $("kpis").innerHTML += group("Harvest · fruit grading", "RT-DETR") +
    k("", "Plants", "", h.plants) + k("healthy", "Ripe · pass", "", h.pass) +
    k("suspect", "Not ready", "", h.not_ready) + k("disease", "Reject", "", h.reject);
}

// ---------------------------------------------------------------- main loop

function render(s) {
  const idx = indexState(s);
  ui.autoSel = chooseAuto(idx, s);
  const haveMap = ensureMap(s);
  renderHeader(s);
  if (haveMap) { renderDynamic(s, idx); updateRobot(s); }
  renderKpis(s, idx);
  renderDetail(s, idx);
  renderCapture(s, idx);
  renderFeed(s);
}

async function poll() {
  try {
    const r = await fetch("/api/state", { cache: "no-store" });
    if (!r.ok) throw new Error(r.status);
    const s = await r.json();
    ui.state = s;
    ui.lastOk = Date.now();
    const st = s.status;
    if (st && st.aisle && st.metre != null) {
      const last = ui.statusTrail[ui.statusTrail.length - 1];
      if (!last || last.ts !== st.ts) ui.statusTrail.push({ ts: st.ts || s.server_ts, aisle: st.aisle, metre: Number(st.metre) });
      if (ui.statusTrail.length > 4000) ui.statusTrail.splice(0, 1000);
    }
    render(s);
    const age = st?.ts ? s.server_ts - st.ts : null;
    $("conn").className = "conn " + (age == null || age < 15 ? "ok" : "");
    $("connText").textContent = age == null ? "live · no robot status" : age < 15 ? `live · updated ${Math.max(0, age).toFixed(0)} s ago` : `robot status ${Math.round(age)} s old`;
  } catch (e) {
    $("conn").className = "conn bad";
    $("connText").textContent = "server unreachable";
  } finally {
    setTimeout(poll, POLL_MS);
  }
}

document.addEventListener("click", (e) => {
  const t = e.target.closest(".thumb[data-img]");
  if (t) {
    $("lbImg").innerHTML = t.innerHTML;
    $("lbImg").querySelector(".cap")?.remove();
    $("lbCap").textContent = t.dataset.cap || "";
    $("lightbox").hidden = false;
    return;
  }
  if (e.target.closest("#lightbox")) { $("lightbox").hidden = true; return; }
  const li = e.target.closest(".feed li.clickable");
  if (li) select(li.dataset.aisle, Number(li.dataset.metre), true);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { $("lightbox").hidden = true; ui.selected = null; ui.detailSig = ""; ui.dynSig = ""; if (ui.state) render(ui.state); }
});
$("slamToggle").addEventListener("change", (e) => { ui.showSlam = e.target.checked; ui.mapSig = ""; if (ui.state) render(ui.state); });
window.addEventListener("resize", () => { if (ui.state) render(ui.state); });

poll();
