/* The Word-Order Axis: page runtime.
   Math (KaTeX -> MathML), figure numbering and figrefs, figure/table mounting, tooltips, cross-figure hover-linking,
   contents (in-flow, top bar, side rail), reading progress, sidenotes, theme toggle, redraw on resize and theme.
   Figure modules register with WOA.fig(id, spec); data tables with WOA.table(id, fn). See v4/DESIGN.md. */
(function () {
"use strict";
const W = window.WOA = window.WOA || {};
const doc = document, root = doc.documentElement;
W.errors = W.errors || [];
W.warnings = [];
addEventListener("error", e => W.errors.push(String((e && e.message) || e)));

/* ---------------- registry ---------------- */
W.figs = W.figs || {};
W.tables = W.tables || {};
W.fig = (id, spec) => { W.figs[id] = spec; };
W.table = (id, fn) => { W.tables[id] = fn; };
W.mounted = [];
W.timing = {};
const now = () => (window.performance && performance.now()) || Date.now();

/* ---------------- small utilities ---------------- */
W.el = (tag, cls, text) => { const e = doc.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
W.get = (o, path) => path.split(".").reduce((a, k) => (a == null ? undefined : a[k]), o);
W.motion = () => !(window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches);
W.dur = ms => (W.motion() ? ms : 0);
W.cssVar = n => getComputedStyle(root).getPropertyValue(n.startsWith("--") ? n : "--" + n).trim();
W.isDark = () => { const t = root.getAttribute("data-theme"); return t ? t === "dark" : !!(window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches); };
const MINUS = "−";
W.f = {
  signed: (x, d = 2) => (x == null || !isFinite(x) ? "–" : (x > 0 ? "+" : x < 0 ? MINUS : "") + Math.abs(x).toFixed(d)),
  num: (x, d = 2) => (x == null || !isFinite(x) ? "–" : (x < 0 ? MINUS : "") + Math.abs(x).toFixed(d)),
  pct: (x, d = 0) => (x == null || !isFinite(x) ? "–" : (100 * x).toFixed(d) + "%"),
  pts: (x, d = 1) => (x == null || !isFinite(x) ? "–" : (x > 0 ? "+" : x < 0 ? MINUS : "") + Math.abs(100 * x).toFixed(d)),
  p: x => (x == null || !isFinite(x) ? "–" : x < 0.00015 ? "0.0001" : x < 0.001 ? x.toFixed(4) : x < 0.01 ? x.toFixed(4) : x.toFixed(3)),
  k: k => (k > 0 ? "+" : k < 0 ? MINUS : "") + Math.abs(k)
};

/* rich text for legends and tooltips: a string, or parts where an array element is a subscript (["β", ["OV"]]) */
W.parts = p => {
  if (!Array.isArray(p)) return doc.createTextNode(p == null ? "" : String(p));
  const f = doc.createDocumentFragment();
  p.forEach(x => f.appendChild(Array.isArray(x) ? W.el("sub", null, x[0]) : doc.createTextNode(x)));
  return f;
};

/* languages: steering files call Simplified Chinese zho_Hans, the 34-language files cmn_Hans */
W.norm = c => (c === "zho_Hans" ? "cmn_Hans" : c);
W.lang = code => {
  const c = W.norm(code), L = (W.D.langs || []).find(l => l.code === c);
  const base = L || { code: c, name: c, family: "", script: (c.split("_")[1] || ""), ov: false, post: false };
  const order = (c === "deu_Latn" || c === "nld_Latn") ? "nd" : base.ov ? "ov" : "vo";
  return Object.assign({}, base, { order });
};
W.shortName = code => { const c = W.norm(code); if (c === "cmn_Hans" && code === "zho_Hans") return "Chinese"; return W.lang(c).name; };
W.orderText = o => ({ ov: "object before verb", vo: "verb before object", nd: "no dominant order" })[o] || o;

/* text measurement for label placement (falls back to an estimate where canvas is unavailable) */
let measureCtx = null;
/* widths are cached, and the font stack is read once: reading a CSS variable forces a style recalculation, which made
   the first figure (thousands of measurements while the page is being built) take most of a second */
let fontStack = null;
const widthCache = new Map();
W.textW = (s, size = 12, weight = 400) => {
  const key = size + "|" + weight + "|" + s;
  const hit = widthCache.get(key); if (hit !== undefined) return hit;
  if (measureCtx === null) { try { measureCtx = doc.createElement("canvas").getContext("2d") || false; } catch (e) { measureCtx = false; } }
  if (fontStack === null) fontStack = W.cssVar("--sans") || "sans-serif";
  let wdt = String(s).length * size * 0.56;
  if (measureCtx) { measureCtx.font = `${weight} ${size}px ${fontStack}`; const m = measureCtx.measureText(String(s)); if (m && m.width) wdt = m.width; }
  widthCache.set(key, wdt);
  return wdt;
};
W.resetTextW = () => widthCache.clear();     /* web fonts arrived: widths change */

/* ---------------- math ---------------- */
W.tex = (el, tex, display) => {
  if (!window.katex) return false;
  try {
    el.innerHTML = katex.renderToString(tex, { output: "mathml", displayMode: !!display, throwOnError: false, strict: "ignore" });
    el.classList.add("ready");
    return true;
  } catch (e) { W.errors.push("KaTeX: " + e.message); return false; }
};
/* top-level occurrences of a TeX command (outside braces and environments), e.g. the \qquad between two equations */
function splitTop(tex, cmd) {
  const parts = []; let depth = 0, env = 0, last = 0;
  for (let i = 0; i < tex.length; i++) {
    const c = tex[i];
    if (c === "\\") {
      if (tex.startsWith("\\begin", i)) env++;
      else if (tex.startsWith("\\end", i)) env--;
      else if (!depth && !env && tex.startsWith(cmd, i) && !/[A-Za-z]/.test(tex[i + cmd.length] || "")) { parts.push(tex.slice(last, i)); last = i + cmd.length; i += cmd.length - 1; continue; }
      i++; continue;
    }
    if (c === "{") depth++; else if (c === "}") depth--;
  }
  parts.push(tex.slice(last));
  return parts.map(p => p.trim()).filter(Boolean);
}
W.splitTop = splitTop;
function renderMath(scope) {
  if (!window.katex) { root.classList.add("no-katex"); return; }
  (scope || doc).querySelectorAll(".m:not(.ready), .mb:not(.ready)").forEach(el => {
    const tex = el.textContent;
    if (!el.classList.contains("mb")) { W.tex(el, tex, false); return; }
    /* display math made of several equations side by side (\qquad) is split, so the parts can wrap on narrow screens */
    const parts = splitTop(tex, "\\qquad");
    if (parts.length < 2) { W.tex(el, tex, true); return; }
    el.textContent = ""; el.classList.add("mb-multi");
    let ok = true;
    parts.forEach(p => { const s = W.el("span", "mb-part"); el.appendChild(s); ok = W.tex(s, p, true) && ok; });
    if (ok) el.classList.add("ready"); else { el.textContent = tex; W.tex(el, tex, true); }
  });
}
W.renderMath = renderMath;

/* ---------------- tooltip ---------------- */
let tipEl = null;
function tipNode() { if (!tipEl) { tipEl = doc.getElementById("tip") || doc.body.appendChild(W.el("div", "tip")); tipEl.setAttribute("role", "tooltip"); } return tipEl; }
function keySvg(k) {
  const ns = "http://www.w3.org/2000/svg", s = doc.createElementNS(ns, "svg");
  s.setAttribute("width", 14); s.setAttribute("height", 10); s.setAttribute("aria-hidden", "true");
  let m;
  if (k.line) { m = doc.createElementNS(ns, "line"); m.setAttribute("x1", 0); m.setAttribute("x2", 14); m.setAttribute("y1", 5); m.setAttribute("y2", 5); m.setAttribute("class", "line " + k.line); }
  else { m = doc.createElementNS(ns, "circle"); m.setAttribute("cx", 7); m.setAttribute("cy", 5); m.setAttribute("r", 4); m.setAttribute("class", k.ring ? "ring " + k.ring : (k.dot || "c-ink2")); }
  s.appendChild(m); return s;
}
W.tip = {
  /* spec: {title, rows: [{v, l, key: {line|dot|ring: cls}}], note}; target: pointer event, Element, or {x, y} */
  show(target, spec) {
    const t = tipNode(); t.replaceChildren();
    if (spec.title) { const tt = W.el("div", "tt"); tt.appendChild(W.parts(spec.title)); t.appendChild(tt); }
    (spec.rows || []).forEach(r => {
      const row = W.el("div", "tr");
      if (r.key) row.appendChild(keySvg(r.key));
      if (r.v != null && r.v !== "") { const v = W.el("span", "tv"); v.appendChild(W.parts(r.v)); row.appendChild(v); }
      if (r.l) { const l = W.el("span", "tl"); l.appendChild(W.parts(r.l)); row.appendChild(l); }
      t.appendChild(row);
    });
    if (spec.note) t.appendChild(W.el("div", "tn", spec.note));
    t.hidden = false;
    let x, y, r = t.getBoundingClientRect();
    if (target && target.getBoundingClientRect && !("clientX" in target)) {
      const b = target.getBoundingClientRect(); x = b.right + 10; y = b.top - 4;
      if (x + r.width > innerWidth - 8) x = Math.max(8, b.left - r.width - 10);
    } else if (target && "clientX" in target) { x = target.clientX + 14; y = target.clientY + 14;
      if (x + r.width > innerWidth - 8) x = target.clientX - r.width - 14;
      if (y + r.height > innerHeight - 8) y = target.clientY - r.height - 14;
    } else { x = (target && target.x) || 0; y = (target && target.y) || 0; }
    x = Math.max(8, Math.min(x, innerWidth - r.width - 8)); y = Math.max(8, Math.min(y, innerHeight - r.height - 8));
    t.style.left = x + "px"; t.style.top = y + "px";
  },
  hide() { if (tipEl) tipEl.hidden = true; }
};

/* ---------------- event bus + cross-figure highlighting ---------------- */
const subs = {};
W.on = (topic, fn) => { (subs[topic] = subs[topic] || []).push(fn); };
W.emit = (topic, payload) => { (subs[topic] || []).forEach(fn => { try { fn(payload); } catch (e) { W.errors.push(`${topic}: ${e.message}`); } }); };
let hlKey = "";
/* codes: array of language codes (any spelling), or null to clear. Every element carrying data-lang (space-separated
   codes allowed) in a figure or data table is matched; containers with at least one match get .linked. */
W.highlight = (codes, info) => {
  const set = codes && codes.length ? new Set(codes.map(W.norm)) : null;
  const key = set ? [...set].sort().join(" ") : "";
  if (key === hlKey) return;
  hlKey = key;
  doc.querySelectorAll(".fig-body, table[data-table]").forEach(box => {
    const marks = box.querySelectorAll("[data-lang]");
    let any = false;
    marks.forEach(m => {
      const on = !!set && m.getAttribute("data-lang").split(" ").some(c => set.has(W.norm(c)));
      m.classList.toggle("hl", on); if (on) any = true;
    });
    box.classList.toggle("linked", any);
  });
  W.emit("highlight", { codes: set, info: info || null });
};
W.hl = () => hlKey;

/* ---------------- UI helpers (keyboard-accessible controls) ---------------- */
W.ui = {
  group(parent, label) {
    const g = W.el("div", "ctl");
    if (label) g.appendChild(W.el("span", "ctl-lab", label));
    parent.appendChild(g); return g;
  },
  /* segmented control: options [[value, text, title?]] */
  seg(parent, { label, options, value, onChange, aria }) {
    const g = W.ui.group(parent, label), box = W.el("div", "seg");
    box.setAttribute("role", "group"); box.setAttribute("aria-label", aria || label || "Options");
    let cur = value;
    const btns = options.map(([v, text, title]) => {
      const b = W.el("button", null, text); b.type = "button"; b.dataset.v = v; if (title) b.title = title;
      b.setAttribute("aria-pressed", String(v === cur));
      b.addEventListener("click", () => { if (b.disabled) return; set(v); onChange && onChange(v); });
      b.addEventListener("keydown", ev => {
        const i = btns.indexOf(b), d = ev.key === "ArrowRight" ? 1 : ev.key === "ArrowLeft" ? -1 : 0;
        if (!d) return; ev.preventDefault();
        for (let j = 1; j <= btns.length; j++) { const n = btns[(i + d * j + btns.length) % btns.length]; if (!n.disabled) { n.focus(); n.click(); break; } }
      });
      box.appendChild(b); return b;
    });
    g.appendChild(box);
    function set(v) { cur = v; btns.forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(v)))); }
    return { get: () => cur, set, disable(v, off) { const b = btns.find(x => x.dataset.v === String(v)); if (b) b.disabled = !!off; }, el: g };
  },
  button(parent, { text, icon, label, onClick, pressed, cls }) {
    const b = W.el("button", "btn" + (cls ? " " + cls : "")); b.type = "button";
    if (icon) b.insertAdjacentHTML("beforeend", icon);
    if (text) b.appendChild(doc.createTextNode(text));
    if (label) { b.setAttribute("aria-label", label); b.title = label; }
    if (pressed != null) b.setAttribute("aria-pressed", String(!!pressed));
    b.addEventListener("click", () => onClick && onClick(b));
    parent.appendChild(b); return b;
  },
  /* legend items: {kind: dot|ring|line|band|hull|bar|range, cls, text} */
  legend(parent, items) {
    parent.replaceChildren();
    items.forEach(it => {
      const s = W.el("span", "lg"), ns = "http://www.w3.org/2000/svg", svg = doc.createElementNS(ns, "svg");
      const w = it.kind === "line" || it.kind === "range" ? 18 : it.kind === "band" || it.kind === "hull" ? 18 : 12;
      svg.setAttribute("width", w); svg.setAttribute("height", 12); svg.setAttribute("aria-hidden", "true");
      const add = (tag, attrs) => { const m = doc.createElementNS(ns, tag); for (const k in attrs) m.setAttribute(k, attrs[k]); svg.appendChild(m); return m; };
      if (it.kind === "line") add("line", { x1: 1, x2: 17, y1: 6, y2: 6, class: "line " + it.cls });
      else if (it.kind === "thin") add("line", { x1: 1, x2: 17, y1: 6, y2: 6, class: "line thin " + it.cls });
      else if (it.kind === "band") add("rect", { x: 0, y: 2, width: 18, height: 8, rx: 2, class: it.cls || "band" });
      else if (it.kind === "hull") add("rect", { x: 1, y: 1.5, width: 16, height: 9, rx: 4.5, class: "hullp" });
      else if (it.kind === "bar") add("rect", { x: 1, y: 1, width: 10, height: 10, rx: 2, class: it.cls });
      else if (it.kind === "range") { add("line", { x1: 2, x2: 16, y1: 6, y2: 6, class: "line " + it.cls }); add("line", { x1: 2, x2: 2, y1: 2.5, y2: 9.5, class: "line " + it.cls }); add("line", { x1: 16, x2: 16, y1: 2.5, y2: 9.5, class: "line " + it.cls }); }
      else if (it.kind === "ring") add("circle", { cx: 6, cy: 6, r: 4, class: "ring " + it.cls });
      else if (it.kind === "around") { add("circle", { cx: 6, cy: 6, r: 3, class: it.dot || "c-ink2" }); add("circle", { cx: 6, cy: 6, r: 6, class: "s-ink", fill: "none", "stroke-width": 1.3 }); }
      else add("circle", { cx: 6, cy: 6, r: 4.5, class: it.cls });
      const lab = W.el("span"); lab.appendChild(W.parts(it.text));
      s.appendChild(svg); s.appendChild(lab); parent.appendChild(s);
    });
  }
};

/* keyboard navigation over the marks of one figure: the svg is a single tab stop, arrows move between items */
/* a polite live region: keyboard readouts of figures are announced to screen readers */
let live = null;
W.announce = text => {
  if (!live) { live = W.el("div", "sr-only"); live.setAttribute("role", "status"); live.setAttribute("aria-live", "polite"); doc.body.appendChild(live); }
  live.textContent = text;
};
W.keyNav = (svgNode, getItems, onItem, onLeave) => {
  svgNode.setAttribute("tabindex", "0");
  const lbl = svgNode.getAttribute("aria-label") || "Figure";
  if (!/arrow keys/.test(lbl)) svgNode.setAttribute("aria-label", lbl + ". Use the arrow keys to read the values.");
  let i = -1;
  svgNode.addEventListener("keydown", ev => {
    const items = getItems(); if (!items.length) return;
    const fwd = ev.key === "ArrowRight" || ev.key === "ArrowDown", back = ev.key === "ArrowLeft" || ev.key === "ArrowUp";
    if (ev.key === "Escape") { i = -1; onLeave(); return; }
    if (!fwd && !back) return;
    ev.preventDefault();
    i = fwd ? (i + 1) % items.length : (i - 1 + items.length) % items.length;
    onItem(items[i], i);
    const t = doc.getElementById("tip");
    if (t && !t.hidden) W.announce([...t.children].map(c => c.textContent).join(". "));
  });
  svgNode.addEventListener("blur", () => { i = -1; onLeave(); });
};

/* ---------------- figures ---------------- */
function missing(ctx, msg) {
  ctx.graphic.replaceChildren(W.el("div", "fig-missing", msg));
  W.warnings.push(`figure ${ctx.id}: ${msg}`);
}
W.width = ctx => Math.max(280, Math.floor(ctx.graphic.clientWidth || ctx.fig.clientWidth || 720));
function drawFig(ctx) {
  const spec = W.figs[ctx.id];
  if (!spec || !ctx.ok) return;
  try { spec.draw(ctx); ctx.lastW = ctx.graphic.clientWidth; }
  catch (e) { W.errors.push(`${ctx.id}: ${e.message}`); if (window.console) console.error(e); }
  if (W.started) relayoutSoon();
}
/* figure heights can change after a control is used: re-place sidenotes and re-check long math once things settle */
let relayoutT = 0;
function relayoutSoon() { clearTimeout(relayoutT); relayoutT = setTimeout(() => { fitMath(); layoutNotes(); }, 60); }
W.redraw = ctx => drawFig(ctx);

function mountFigures() {
  const figs = [...doc.querySelectorAll("figure[data-fig]")];
  figs.forEach((fig, i) => {
    const id = fig.getAttribute("data-fig"), n = i + 1;
    fig.dataset.num = String(n);
    let cap = fig.querySelector(":scope > figcaption");
    if (!cap) { cap = W.el("figcaption"); fig.appendChild(cap); }
    if (!cap.querySelector(".fignum")) { cap.prepend(W.el("span", "fignum", `Figure ${n}.`), doc.createTextNode(" ")); }
    const body = W.el("div", "fig-body"), controls = W.el("div", "fig-controls"), legend = W.el("div", "fig-legend"), graphic = W.el("div", "fig-graphic");
    body.append(controls, legend, graphic);
    fig.insertBefore(body, cap);
    const spec = W.figs[id];
    const ctx = { id, n, fig, body, controls, legend, graphic, D: W.D, state: {}, ok: false };
    ctx.width = () => W.width(ctx);
    ctx.redraw = () => drawFig(ctx);
    fig.classList.add("fig-" + ((spec && spec.layout) || "page"));
    if (!spec) { missing(ctx, `No figure module is registered for “${id}”.`); return; }
    const need = spec.needs || [];
    const lacking = need.find(k => W.get(W.D, k) == null);
    if (lacking) {
      if (spec.pending) {
        body.classList.add("is-pending"); fig.classList.add("pending-fig");
        try { spec.pending(ctx, lacking); } catch (e) { W.errors.push(`${id} pending: ${e.message}`); }
        ctx.pendingDraw = spec.pending; W.mounted.push(ctx);
      } else missing(ctx, `The data for this figure (“${lacking}”) is not in this build.`);
      return;
    }
    if (!W.timing["~layout"]) { const tl = now(); void graphic.clientWidth; W.timing["~layout"] = Math.round(now() - tl) || 0.1; }
    const tf = now();
    try { if (spec.init) spec.init(ctx); ctx.ok = true; spec.draw(ctx); ctx.lastW = graphic.clientWidth; }
    catch (e) { ctx.ok = false; W.errors.push(`${id}: ${e.message}`); if (window.console) console.error(e); missing(ctx, "This figure failed to render."); }
    W.timing[id] = Math.round(now() - tf);
    W.mounted.push(ctx);
  });
}
function redrawAll() { W.mounted.forEach(ctx => { if (ctx.ok) drawFig(ctx); else if (ctx.pendingDraw) { try { ctx.pendingDraw(ctx); } catch (e) {} } }); }
W.redrawAll = redrawAll;

function fillFigrefs() {
  doc.querySelectorAll("a.figref").forEach(a => {
    const id = decodeURIComponent((a.getAttribute("href") || "").replace(/^#/, ""));
    const f = id && doc.getElementById(id);
    if (f && f.dataset && f.dataset.num) { a.textContent = `Figure ${f.dataset.num}`; a.classList.remove("figref-missing"); }
    else { a.textContent = "Figure ?"; a.classList.add("figref-missing"); W.warnings.push(`figref to missing ${id}`); }
  });
}

function mountTables() {
  doc.querySelectorAll("table[data-table]").forEach(t => {
    const id = t.getAttribute("data-table"), fn = W.tables[id];
    if (!fn) { t.replaceChildren(); const c = t.createCaption(); c.className = "table-missing"; c.textContent = `No table module for “${id}”.`; W.warnings.push(`table ${id}: no module`); return; }
    try { t.replaceChildren(); fn(t, { D: W.D }); t.dataset.ready = "1"; }
    catch (e) { W.errors.push(`table ${id}: ${e.message}`); const c = t.createCaption(); c.className = "table-missing"; c.textContent = "This table failed to render."; }
  });
}

/* static tables: a column whose body cells are all td.n gets right-aligned header cells too */
function alignStaticTables() {
  doc.querySelectorAll(".article table:not([data-table])").forEach(t => {
    const body = t.tBodies[0], head = t.tHead; if (!body || !head) return;
    const rows = [...body.rows]; if (!rows.length) return;
    const last = head.rows[head.rows.length - 1]; if (!last) return;
    [...last.cells].forEach((th, i) => { if (rows.every(r => r.cells[i] && r.cells[i].classList.contains("n"))) th.classList.add("n"); });
    /* short hyphenated names (Indo-European) should not break at the hyphen in a narrow column */
    rows.forEach(r => [...r.cells].forEach(td => { const t = td.textContent.trim(); if (t.length <= 16 && /\w-\w/.test(t) && !/\s/.test(t)) td.classList.add("nw"); }));
  });
}

function pending() {
  doc.querySelectorAll(".pending[data-pending]").forEach(el => {
    const k = el.getAttribute("data-pending");
    if (W.get(W.D, k) != null) el.hidden = true;
  });
}

/* ---------------- contents, progress, section indicator ---------------- */
let sections = [];
function buildContents() {
  sections = [...doc.querySelectorAll("section[data-title]")];
  if (!sections.length) return;
  const mk = (list, cls) => {
    const ol = W.el("ol");
    list.forEach((s, i) => {
      const li = W.el("li"), a = W.el("a"); a.href = "#" + s.id; a.dataset.sec = s.id;
      a.append(W.el("span", "toc-n", String(i + 1)), W.el("span", null, s.getAttribute("data-title")));
      li.appendChild(a); ol.appendChild(li);
    });
    return ol;
  };
  /* in-flow list after the summary (sections that follow it), numbered */
  const anchor = doc.getElementById("summary") || doc.querySelector("header.front");
  const after = anchor ? sections.filter(s => anchor.compareDocumentPosition(s) & Node.DOCUMENT_POSITION_FOLLOWING && s !== anchor) : sections;
  if (after.length && !doc.querySelector("nav.toc-flow")) {
    const nav = W.el("nav", "toc-flow"); nav.setAttribute("aria-label", "Contents");
    const inner = W.el("div", "toc-inner"); inner.append(W.el("div", "toc-lab", "Contents"), mk(after));
    nav.appendChild(inner);
    if (anchor) anchor.after(nav); else doc.querySelector(".article").prepend(nav);
  }
  const menu = doc.getElementById("tb-menu"), rail = doc.getElementById("toc-rail");
  if (menu) { menu.replaceChildren(mk(sections)); }
  if (rail) { rail.replaceChildren(W.el("div", "toc-lab", "Contents"), mk(sections)); }
  const btn = doc.getElementById("tb-section");
  if (btn && menu) {
    const close = () => { menu.hidden = true; btn.setAttribute("aria-expanded", "false"); };
    btn.addEventListener("click", ev => { ev.stopPropagation(); const open = menu.hidden; menu.hidden = !open; btn.setAttribute("aria-expanded", String(open)); if (open) { const cur = menu.querySelector('[aria-current="true"]') || menu.querySelector("a"); cur && cur.focus(); } });
    menu.addEventListener("click", ev => { if (ev.target.closest("a")) close(); });
    doc.addEventListener("click", ev => { if (!menu.hidden && !menu.contains(ev.target)) close(); });
    doc.addEventListener("keydown", ev => { if (ev.key === "Escape" && !menu.hidden) { close(); btn.focus(); } });
  }
}
let curSec = null, scrollQueued = false;
function onScroll() {
  scrollQueued = false;
  const y = scrollY || doc.documentElement.scrollTop || 0, H = Math.max(1, doc.documentElement.scrollHeight - innerHeight);
  const bar = doc.getElementById("tb-progress"); if (bar) bar.style.width = (100 * Math.min(1, y / H)).toFixed(2) + "%";
  const h1 = doc.querySelector("header.front h1");
  root.classList.toggle("scrolled", h1 ? h1.getBoundingClientRect().bottom < 40 : y > 120);
  let cur = null; const line = innerHeight * 0.32;
  for (const s of sections) { if (!s.offsetHeight) continue; if (s.getBoundingClientRect().top <= line) cur = s; else break; }
  const flow = doc.querySelector("nav.toc-flow"), rail = doc.getElementById("toc-rail");
  if (rail) rail.classList.toggle("on", flow ? flow.getBoundingClientRect().bottom < 0 : y > 400);
  if (cur !== curSec) {
    curSec = cur;
    const label = doc.getElementById("tb-section-label");
    if (label) label.textContent = cur ? cur.getAttribute("data-title") : "Contents";
    doc.querySelectorAll("#tb-menu a, #toc-rail a").forEach(a => a.setAttribute("aria-current", String(!!cur && a.dataset.sec === cur.id)));
  }
}
function queueScroll() { if (!scrollQueued) { scrollQueued = true; requestAnimationFrame(onScroll); } }

/* ---------------- long math ----------------
   MathML does not break lines. Inline math wider than its line becomes its own horizontally scrolling box, and display
   math that scrolls gets a fade at the edge it can scroll toward. */
function fitMath() {
  /* batched: all writes, then all reads, then all writes, so the browser lays the page out a few times, not hundreds */
  const ms = [...doc.querySelectorAll(".m.ready")], mbs = [...doc.querySelectorAll(".mb.ready:not(.mb-multi), .mb-part")];
  ms.forEach(el => el.classList.remove("m-long"));
  mbs.forEach(el => { el.style.fontSize = ""; });
  const longM = ms.filter(el => {
    const box = el.closest("p, li, dd, td, figcaption, .def, .takeaway, aside, div") || el.parentElement;
    if (!box) return false;
    const r = el.getBoundingClientRect(), bb = box.getBoundingClientRect();
    return r.width > bb.width - 4 || r.right > bb.right + 1;
  });
  /* display math a little too wide for its column is set slightly smaller (down to 84%) instead of scrolling */
  const sizes = mbs.map(el => { const sw = el.scrollWidth, cw = el.clientWidth; return cw && sw > cw + 2 && cw / sw >= 0.84 ? (Math.floor(cw / sw * 98) / 100) + "em" : ""; });
  longM.forEach(el => el.classList.add("m-long"));
  mbs.forEach((el, i) => { if (sizes[i]) el.style.fontSize = sizes[i]; });
  const scrollers = [...doc.querySelectorAll(".mb.ready:not(.mb-multi), .mb-part, .tablewrap")];
  const st = scrollers.map(el => { const over = el.scrollWidth > el.clientWidth + 2; return [over, over && el.scrollLeft + el.clientWidth >= el.scrollWidth - 2]; });
  scrollers.forEach((el, i) => {
    const [over, end] = st[i];
    el.classList.toggle("scrolls", over); el.classList.toggle("at-end", end);
    if (over && !el.dataset.fitBound) { el.dataset.fitBound = "1"; el.addEventListener("scroll", () => el.classList.toggle("at-end", el.scrollLeft + el.clientWidth >= el.scrollWidth - 2), { passive: true }); }
  });
}
W.fitMath = fitMath;

/* ---------------- sidenotes ---------------- */
function layoutNotes() {
  const wide = !!(window.matchMedia && matchMedia("(min-width: 1180px)").matches);
  root.classList.toggle("notes-wide", wide);
  doc.querySelectorAll(".article > section").forEach(sec => {
    const notes = [...sec.querySelectorAll(":scope > aside.sidenote")];
    sec.style.paddingBottom = "";
    if (!notes.length) return;
    if (!wide) { notes.forEach(n => { n.style.top = ""; }); return; }
    const blockers = [...sec.querySelectorAll(":scope > figure.fig-page, :scope > figure.fig-wide")].map(f => [f.offsetTop, f.offsetTop + f.offsetHeight]);
    let floor = 0;
    notes.forEach(n => {
      let a = n.previousElementSibling; while (a && a.matches("aside.sidenote")) a = a.previousElementSibling;
      let y = Math.max(a ? a.offsetTop + 4 : 0, floor);
      const h = n.offsetHeight;
      for (let k = 0; k < 3; k++) for (const [t, b] of blockers) if (y < b && y + h > t) y = b + 12;
      n.style.top = y + "px";
      floor = y + h + 16;
    });
    if (floor > sec.offsetHeight) sec.style.paddingBottom = (floor - sec.offsetHeight + 8) + "px";
  });
}
W.layoutNotes = layoutNotes;

/* ---------------- theme ---------------- */
function syncThemeButton() {
  const b = doc.getElementById("tb-theme"); if (!b) return;
  const dark = W.isDark();
  b.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  const moon = b.querySelector(".i-moon"), sun = b.querySelector(".i-sun");
  if (moon) moon.hidden = dark; if (sun) sun.hidden = !dark;
}
function setupTheme() {
  const b = doc.getElementById("tb-theme");
  if (b) b.addEventListener("click", () => {
    const next = W.isDark() ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("woa-theme", next); } catch (e) { /* convenience only */ }
  });
  syncThemeButton();
  let t = 0;
  const changed = () => { clearTimeout(t); t = setTimeout(() => { syncThemeButton(); redrawAll(); }, 30); };
  if (window.MutationObserver) new MutationObserver(changed).observe(root, { attributes: true, attributeFilter: ["data-theme"] });
  if (window.matchMedia) { const mq = matchMedia("(prefers-color-scheme: dark)"); if (mq.addEventListener) mq.addEventListener("change", changed); }
}

/* ---------------- resize ---------------- */
function setupResize() {
  let t = 0;
  const printing = () => !!(window.matchMedia && matchMedia("print").matches);
  const run = () => {
    if (printing()) return;            /* print scales the screen drawing; no redraw at paper width */
    W.mounted.forEach(ctx => {
      const w = ctx.graphic.clientWidth;
      if (w && Math.abs(w - (ctx.lastW || 0)) > 1) { if (ctx.ok) drawFig(ctx); else if (ctx.pendingDraw) { try { ctx.pendingDraw(ctx); } catch (e) {} } ctx.lastW = w; }
    });
    fitMath(); layoutNotes(); queueScroll();
  };
  const later = () => { clearTimeout(t); t = setTimeout(run, 120); };
  if (window.ResizeObserver) { const ro = new ResizeObserver(later); W.mounted.forEach(ctx => ro.observe(ctx.graphic)); }
  addEventListener("resize", later);
}

/* ---------------- start ---------------- */
W.start = () => {
  if (W.started) return; W.started = true;
  const t0 = (window.performance && performance.now()) || 0;
  const el = doc.getElementById("figdata");
  try { W.D = JSON.parse((el && el.textContent) || "{}"); } catch (e) { W.D = {}; W.errors.push("figdata: " + e.message); }
  if (!window.d3) W.errors.push("d3 did not load");
  const front = doc.querySelector("header.front");
  if (front && !front.querySelector(":scope > .front-inner")) { const inner = W.el("div", "front-inner"); while (front.firstChild) inner.appendChild(front.firstChild); front.appendChild(inner); }
  const lap = (k, f) => { const t = now(); f(); W.timing["~" + k] = Math.round(now() - t); };
  lap("math", renderMath);
  buildContents();
  pending();
  alignStaticTables();
  if (window.d3) { lap("figures", mountFigures); lap("tables", mountTables); }
  lap("math2", renderMath);           /* math that figure/table modules inserted */
  fillFigrefs();
  setupTheme();
  lap("fitMath", fitMath);
  lap("notes", layoutNotes);
  setupResize();
  addEventListener("scroll", queueScroll, { passive: true });
  onScroll();
  doc.addEventListener("keydown", ev => { if (ev.key === "Escape") { W.tip.hide(); W.highlight(null); } });
  /* web fonts that arrive after the first draw change text widths: redraw once (not when they were already there) */
  let fontsWere = false;
  try { fontsWere = !!(doc.fonts && doc.fonts.check && doc.fonts.check('600 12px "Noto Sans"') && doc.fonts.status === "loaded"); } catch (e) { fontsWere = false; }
  if (doc.fonts && doc.fonts.ready && !fontsWere) doc.fonts.ready.then(() => { W.resetTextW(); redrawAll(); fitMath(); layoutNotes(); }).catch(() => {});
  addEventListener("load", () => { fitMath(); layoutNotes(); queueScroll(); });
  root.classList.add("woa-ready");
  W.startMs = ((window.performance && performance.now()) || 0) - t0;     /* diagnostic: time to first full render */
};
})();
