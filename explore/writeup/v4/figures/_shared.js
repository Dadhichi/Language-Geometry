/* Shared chart helpers for the figure modules: svg frame, axes, label placement, hulls, dodging, the residual-map view. */
(function () {
"use strict";
const W = window.WOA;

/* an <svg> sized to the figure's graphic width; returns a d3 selection */
W.frame = (ctx, height, label, opts = {}) => {
  const w = opts.width || ctx.width();
  ctx.graphic.replaceChildren();
  const svg = d3.select(ctx.graphic).append("svg")
    .attr("width", w).attr("height", height).attr("viewBox", `0 0 ${w} ${height}`)
    .attr("role", "img").attr("aria-label", label || ctx.fig.querySelector("figcaption b")?.textContent || "Figure");
  return svg;
};

/* axes: recessive hairlines, tabular tick labels */
W.axisX = (g, x, y0, opts = {}) => {
  const ax = d3.axisBottom(x).tickSizeOuter(0);
  if (opts.ticks) ax.ticks(opts.ticks); if (opts.values) ax.tickValues(opts.values); if (opts.format) ax.tickFormat(opts.format);
  ax.tickSizeInner(opts.grid ? -(opts.grid) : 4).tickSizeOuter(0).tickPadding(opts.grid ? 7 : 5);
  const a = g.append("g").attr("class", "axis x").attr("transform", `translate(0,${y0})`).call(ax);
  if (opts.noDomain) a.select(".domain").remove();
  return a;
};
W.axisY = (g, y, x0, opts = {}) => {
  const ax = d3.axisLeft(y).tickSizeOuter(0);
  if (opts.ticks) ax.ticks(opts.ticks); if (opts.values) ax.tickValues(opts.values); if (opts.format) ax.tickFormat(opts.format);
  ax.tickSizeInner(opts.grid ? -(opts.grid) : 4).tickSizeOuter(0).tickPadding(6);
  const a = g.append("g").attr("class", "axis y").attr("transform", `translate(${x0},0)`).call(ax);
  a.select(".domain").remove();
  return a;
};

/* greedy label placement around points; items sorted by priority. Each item: {id, x, y, r, text, size, weight}.
   Returns Map id -> {x, y, anchor, show}. Labels avoid each other, every point, the bounds and given obstacle boxes. */
W.placeLabels = (items, bounds, obstacles = [], extraCircles = []) => {
  const placed = [...obstacles], out = new Map();
  const pts = items.map(it => ({ x: it.x, y: it.y, r: (it.r || 4) + 1.5, id: it.id })).concat(extraCircles.map(c => ({ x: c.x, y: c.y, r: c.r, id: null })));
  const hitBox = (a, b) => a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1];
  const hitCircle = (b, c) => { const cx = Math.max(b[0], Math.min(c.x, b[2])), cy = Math.max(b[1], Math.min(c.y, b[3])); return (cx - c.x) ** 2 + (cy - c.y) ** 2 < c.r * c.r; };
  [...items].sort((a, b) => (b.priority || 0) - (a.priority || 0)).forEach(it => {
    const size = it.size || 12, w = W.textW(it.text, size, it.weight || 400), h = size * 0.92, r = it.r || 4, g = 3.5;
    const cand = it.candidates || [
      [r + g, size * 0.34, "start"], [-(r + g), size * 0.34, "end"], [0, -(r + g + 1), "middle"], [0, r + g + h - 1, "middle"],
      [r + 1, -(r + 1), "start"], [r + 1, r + h - 1, "start"], [-(r + 1), -(r + 1), "end"], [-(r + 1), r + h - 1, "end"],
      [r + g + 6, size * 0.34, "start"], [-(r + g + 6), size * 0.34, "end"], [0, -(r + g + 8), "middle"], [0, r + g + h + 6, "middle"]];
    let best = null;
    for (const [dx, dy, anchor] of cand) {
      const lx = it.x + dx, ly = it.y + dy;
      const x0 = anchor === "start" ? lx : anchor === "end" ? lx - w : lx - w / 2;
      const box = [x0 - 1, ly - h + 1, x0 + w + 1, ly + 3];
      if (box[0] < bounds[0] || box[2] > bounds[2] || box[1] < bounds[1] || box[3] > bounds[3]) continue;
      if (placed.some(p => hitBox(box, p))) continue;
      if (pts.some(c => c.id !== it.id && hitCircle(box, c))) continue;
      best = { x: lx, y: ly, anchor, show: true, box }; break;
    }
    if (best) { placed.push(best.box); out.set(it.id, best); }
    else out.set(it.id, { x: it.x + r + g, y: it.y + size * 0.34, anchor: "start", show: false });
  });
  return out;
};

/* rounded, padded outline around a set of screen points (capsule for two, padded convex hull for more) */
W.hullPath = (pts, pad) => {
  if (!pts.length) return "";
  const ring = [];
  pts.forEach(([x, y]) => { for (let k = 0; k < 14; k++) { const a = (k / 14) * 2 * Math.PI; ring.push([x + pad * Math.cos(a), y + pad * Math.sin(a)]); } });
  const hull = d3.polygonHull(ring);
  return hull ? d3.line().curve(d3.curveCatmullRomClosed.alpha(0.6))(hull) : "";
};
W.hullPoly = (pts, pad) => {
  const ring = [];
  pts.forEach(([x, y]) => { for (let k = 0; k < 10; k++) { const a = (k / 10) * 2 * Math.PI; ring.push([x + pad * Math.cos(a), y + pad * Math.sin(a)]); } });
  return d3.polygonHull(ring) || ring;
};

/* dot dodging for one-dimensional strips: returns levels so that dots of radius r do not overlap */
W.dodge = (vals, x, r) => {
  const placed = [];
  return vals.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v).map(({ v, i }) => {
    let lev = 0; while (placed.some(p => p.lev === lev && Math.abs(x(p.v) - x(v)) < 2 * r + 1)) lev++;
    placed.push({ v, lev }); return { v, i, lev };
  });
};

/* quantile helper ignoring nulls */
W.q = (arr, p) => { const a = arr.filter(v => v != null && isFinite(v)).sort(d3.ascending); return a.length ? d3.quantile(a, p) : null; };

/* ---------------- residual map view (used by overview, map and tree) ----------------
   opts: {svg (d3 sel), x0, y0, w, h, groups: "family"|"script"|"none", groupStyle: "hull"|"lines", labels: bool,
          labelSize, arrow: bool, compact: bool, hullLabels: bool, priority: lang -> number, stretch: max x/y scale ratio}
   The frame follows the layout shown (view.frameFor(xy)); transitions interpolate points and frame together. */
W.mapView = function (opts) {
  const langs = W.D.langs;
  const g = opts.svg.append("g").attr("class", "mapview");
  const gHull = g.append("g"), gArrow = g.append("g"), gPts = g.append("g"), gLab = g.append("g").attr("class", "halo"), gGrp = g.append("g").attr("class", "halo");
  const { x0, y0, w, h } = opts;
  const pad = opts.compact ? 12 : 20, padT = opts.hullLabels ? 32 : pad, stretch = opts.stretch || 1.35;
  const R = opts.compact ? 4 : 5.5, LS = opts.labelSize || (opts.compact ? 10.5 : 11.5);
  let cur = null, fr = null, labelPlan = null, groupsMode = opts.groups || "family";
  const view = { g };

  /* frame for a layout: x stretched at most `stretch` times y, centred; returns {kx, ky, cx, cy} */
  function frameFor(xy) {
    const ex = W.extentOf([xy]), xr = (ex[0][1] - ex[0][0]) || 1, yr = (ex[1][1] - ex[1][0]) || 1;
    const sx = (w - 2 * pad) / xr, sy = (h - pad - padT) / yr;
    const kx = Math.min(sx, stretch * sy), ky = Math.min(sy, kx);
    return { kx, ky, cx: x0 + w / 2 - kx * (ex[0][0] + ex[0][1]) / 2, cy: y0 + padT + (h - pad - padT) / 2 + ky * (ex[1][0] + ex[1][1]) / 2 };
  }
  const X = v => fr.cx + fr.kx * v, Y = v => fr.cy - fr.ky * v;
  view.frameFor = frameFor;
  view.lerpFrame = (a, b, t) => ({ kx: a.kx + (b.kx - a.kx) * t, ky: a.ky + (b.ky - a.ky) * t, cx: a.cx + (b.cx - a.cx) * t, cy: a.cy + (b.cy - a.cy) * t });
  view.frame = () => fr;

  const pts = gPts.selectAll("g.pt").data(langs).join("g").attr("class", "pt").attr("data-lang", d => d.code);
  pts.append("circle").attr("r", R).attr("class", d => "dot c-" + W.lang(d.code).order);
  const labs = gLab.selectAll("text").data(langs).join("text").attr("data-lang", d => d.code).style("font-size", LS + "px").text(d => d.name);
  if (!opts.labels) labs.style("display", "none");

  function groupsOf(mode) {
    if (mode === "none") return [];
    const key = mode === "script" ? (l => l.script) : (l => l.family);
    const SCRIPT = { Latn: "Latin", Cyrl: "Cyrillic", Arab: "Arabic", Deva: "Devanagari" };
    return d3.groups(langs.map((l, i) => ({ l, i })), d => key(d.l)).filter(([, m]) => m.length > 1)
      .map(([name, m]) => ({ key: name, name: mode === "script" ? (SCRIPT[name] || name) : name, idx: m.map(d => d.i), codes: m.map(d => d.l.code) }));
  }
  /* minimum spanning tree of a group's points (Prim), for the "lines" style */
  function mst(P) {
    const n = P.length, inT = [0], edges = [];
    while (inT.length < n) {
      let best = null;
      inT.forEach(a => P.forEach((p, b) => { if (inT.includes(b)) return; const d = (P[a][0] - p[0]) ** 2 + (P[a][1] - p[1]) ** 2; if (!best || d < best[2]) best = [a, b, d]; }));
      edges.push([best[0], best[1]]); inT.push(best[1]);
    }
    return edges;
  }
  function arrowGeom(xy) {
    const ovI = langs.map((l, i) => i).filter(i => langs[i].ov), voI = langs.map((l, i) => i).filter(i => W.lang(langs[i].code).order === "vo");
    const a = [d3.mean(voI, i => xy[i][0]), d3.mean(voI, i => xy[i][1])], b = [d3.mean(ovI, i => xy[i][0]), d3.mean(ovI, i => xy[i][1])];
    return { ax: X(a[0]), ay: Y(a[1]), bx: X(b[0]), by: Y(b[1]) };
  }

  /* plan labels for a target layout: the arrow label and group labels first, then language labels around them.
     Offsets are stored relative to the anchor point, so labels glide with their points during a transition. */
  function planLabels(xy) {
    const keep = fr; fr = frameFor(xy);
    const S = xy.map(p => [X(p[0]), Y(p[1])]);
    const circles = S.map(([x, y]) => ({ x, y, r: R + 2 }));
    const obstacles = [], plan = { lang: new Map(), groups: new Map(), arrow: null };
    const hitBox = (a, b) => a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1];
    const hitC = (b, c) => { const qx = Math.max(b[0], Math.min(c.x, b[2])), qy = Math.max(b[1], Math.min(c.y, b[3])); return (qx - c.x) ** 2 + (qy - c.y) ** 2 < c.r * c.r; };
    const free = box => box[0] >= x0 && box[2] <= x0 + w && box[1] >= y0 - 6 && box[3] <= y0 + h + 6 && !obstacles.some(o => hitBox(box, o)) && !circles.some(c => hitC(box, c));
    const extra = [];
    if (opts.arrow) {
      const A = arrowGeom(xy), tw = W.textW("word-order direction", 11.5, 620), L = Math.hypot(A.bx - A.ax, A.by - A.ay);
      search: for (const f of [.5, .38, .62, .26, .74]) for (const side of [-1, 1]) {
        const mx = A.ax + (A.bx - A.ax) * f, my = A.ay + (A.by - A.ay) * f + (side < 0 ? -8 : 17);
        const box = [mx - tw / 2 - 2, my - 11, mx + tw / 2 + 2, my + 3];
        if (L > tw && free(box)) { plan.arrow = { x: mx, y: my, dx: mx - A.ax, dy: my - A.ay }; obstacles.push(box); break search; }
      }
      for (let i = 0; i <= 14; i++) extra.push({ x: A.ax + (A.bx - A.ax) * i / 14, y: A.ay + (A.by - A.ay) * i / 14, r: 3 });
    }
    if (opts.hullLabels) groupsOf(groupsMode).forEach(gr => {
      const P = gr.idx.map(i => S[i]), tw = W.textW(gr.name.toUpperCase(), 10.5, 600) * 1.1;
      const top = d3.min(P, p => p[1]), bot = d3.max(P, p => p[1]), lx = d3.min(P, p => p[0]), rx = d3.max(P, p => p[0]), mx = (lx + rx) / 2;
      const cands = [[mx, top - 16], [mx, bot + 25], [rx + 18 + tw / 2, (top + bot) / 2 + 4], [lx - 18 - tw / 2, (top + bot) / 2 + 4], [mx, top - 30], [mx, bot + 39]];
      for (const [cx, cy] of cands) {
        const box = [cx - tw / 2 - 2, cy - 10, cx + tw / 2 + 2, cy + 2];
        if (free(box) && !extra.some(c => hitC(box, c))) { plan.groups.set(gr.key, { dx: cx - S[gr.idx[0]][0], dy: cy - S[gr.idx[0]][1] }); obstacles.push(box); break; }
      }
    });
    if (opts.labels) {
      const items = langs.map((l, i) => ({ id: l.code, x: S[i][0], y: S[i][1], r: R, text: l.name, size: LS, priority: opts.priority ? opts.priority(l) : 0 }));
      const lp = W.placeLabels(items, [x0, y0 - 6, x0 + w, y0 + h + 6], obstacles, extra);
      langs.forEach((l, i) => { const p = lp.get(l.code); p.dx = p.x - S[i][0]; p.dy = p.y - S[i][1]; plan.lang.set(l.code, p); });
    }
    fr = keep;
    return plan;
  }

  function render(xy, { keepLabels, frame } = {}) {
    cur = xy; fr = frame || (keepLabels && fr) || frameFor(xy);
    const S = xy.map(p => [X(p[0]), Y(p[1])]);
    pts.attr("transform", (d, i) => `translate(${S[i][0].toFixed(2)},${S[i][1].toFixed(2)})`);
    const groups = groupsOf(groupsMode);
    view.groups = groups;
    if ((opts.groupStyle || "hull") === "hull") {
      gHull.selectAll("path").data(groups, d => d.key).join("path").attr("class", "hullp").attr("data-lang", d => d.codes.join(" "))
        .attr("d", gr => W.hullPath(gr.idx.map(i => S[i]), opts.compact ? 9 : 12));
    } else {
      const segs = groups.flatMap(gr => { const P = gr.idx.map(i => S[i]); return mst(P).map(([a, b]) => ({ gr, a: P[a], b: P[b] })); });
      /* a link that would run through an unrelated point bends around it, so it cannot be misread as joining that point */
      const path = d => {
        const [ax, ay] = d.a, [bx, by] = d.b, vx = bx - ax, vy = by - ay, L2 = vx * vx + vy * vy || 1, L = Math.sqrt(L2);
        let worst = null;
        S.forEach(([px, py]) => { const t = ((px - ax) * vx + (py - ay) * vy) / L2; if (t <= .08 || t >= .92) return;
          const dist = ((px - ax) * vy - (py - ay) * vx) / L; if (Math.abs(dist) < 9 && (!worst || Math.abs(dist) < Math.abs(worst))) worst = dist === 0 ? .01 : dist; });
        if (!worst) return `M${ax.toFixed(1)},${ay.toFixed(1)}L${bx.toFixed(1)},${by.toFixed(1)}`;
        const off = -Math.sign(worst) * 26, nx = vy / L, ny = -vx / L;
        return `M${ax.toFixed(1)},${ay.toFixed(1)}Q${((ax + bx) / 2 + nx * off).toFixed(1)},${((ay + by) / 2 + ny * off).toFixed(1)} ${bx.toFixed(1)},${by.toFixed(1)}`;
      };
      gHull.selectAll("path").data(segs).join("path").attr("class", "grp-line").attr("data-group", d => d.gr.key).attr("d", path)
        .classed("on", d => !!view.hlSet && d.gr.codes.every(c => view.hlSet.has(c)));
      view.segs = segs;
    }
    if (!keepLabels || !labelPlan) labelPlan = planLabels(xy);
    if (opts.labels) labs.each(function (d, i) {
      const p = labelPlan.lang.get(d.code);
      d3.select(this).attr("x", S[i][0] + p.dx).attr("y", S[i][1] + p.dy).attr("text-anchor", p.anchor).classed("off", !p.show).attr("aria-hidden", p.show ? null : "true");
    });
    if (opts.hullLabels) {
      const gl = groups.filter(gr => labelPlan.groups.has(gr.key)).map(gr => { const p = labelPlan.groups.get(gr.key), s0 = S[gr.idx[0]]; return { gr, x: s0[0] + p.dx, y: s0[1] + p.dy }; });
      gGrp.selectAll("text").data(gl, d => d.gr.key).join("text").attr("class", "cap").attr("text-anchor", "middle")
        .attr("x", d => d.x).attr("y", d => d.y).text(d => d.gr.name).attr("data-lang", d => d.gr.codes.join(" "));
    } else gGrp.selectAll("text").remove();
    if (opts.arrow) drawArrow(xy, labelPlan.arrow);
  }
  function drawArrow(xy, lab) {
    gArrow.selectAll("*").remove();
    const { ax, ay, bx, by } = arrowGeom(xy), ang = Math.atan2(by - ay, bx - ax), L = Math.hypot(bx - ax, by - ay);
    const gA = gArrow.append("g").attr("transform", `translate(${ax},${ay}) rotate(${ang * 180 / Math.PI})`);
    gA.append("line").attr("x1", 0).attr("x2", L - 9).attr("class", "s-ink").attr("stroke-width", 1.6).attr("opacity", .8);
    gA.append("path").attr("d", `M${L - 11},-4.5 L${L},0 L${L - 11},4.5 Z`).attr("class", "c-ink").attr("opacity", .8);
    gA.append("circle").attr("r", 3).attr("class", "c-ink").attr("opacity", .8);
    if (lab) gArrow.append("text").attr("x", ax + lab.dx).attr("y", ay + lab.dy).attr("text-anchor", "middle").attr("class", "strong halo").style("font-size", "11.5px").text("word-order direction");
  }
  view.render = render;
  view.setPlan = target => { labelPlan = planLabels(target); };
  view.current = () => cur;
  view.setGroups = mode => { groupsMode = mode; if (cur) { labelPlan = planLabels(cur); render(cur, { keepLabels: true }); } };
  /* nearest language to a screen point, else the smallest group outline (or nearest group line) under it */
  view.pick = (px, py, maxD = 26) => {
    if (!cur) return null;
    let best = -1, bd = maxD * maxD;
    cur.forEach((p, i) => { const d = (X(p[0]) - px) ** 2 + (Y(p[1]) - py) ** 2; if (d < bd) { bd = d; best = i; } });
    if (best >= 0) return { lang: langs[best], i: best };
    if ((opts.groupStyle || "hull") === "lines") {
      let bs = null, bdd = 49;
      (view.segs || []).forEach(s => { const vx = s.b[0] - s.a[0], vy = s.b[1] - s.a[1], L2 = vx * vx + vy * vy || 1; const t = Math.max(0, Math.min(1, ((px - s.a[0]) * vx + (py - s.a[1]) * vy) / L2)); const d = (s.a[0] + t * vx - px) ** 2 + (s.a[1] + t * vy - py) ** 2; if (d < bdd) { bdd = d; bs = s; } });
      return bs ? { group: bs.gr } : null;
    }
    const inside = (view.groups || []).filter(gr => d3.polygonContains(W.hullPoly(gr.idx.map(i => [X(cur[i][0]), Y(cur[i][1])]), 12), [px, py]));
    if (inside.length) { inside.sort((a, b) => a.idx.length - b.idx.length); return { group: inside[0] }; }
    return null;
  };
  view.screen = i => [X(cur[i][0]), Y(cur[i][1])];
  return view;
};

/* the 13 groups of langs.family are one Glottolog family, or one branch of Indo-European (10 families in all) */
W.IE_BRANCHES = new Set(["Germanic", "Romance", "Slavic", "Indo-Iranian"]);
W.groupText = f => (W.IE_BRANCHES.has(f) ? `${f} branch of Indo-European` : `${f} family`);

/* common tooltip for a language */
W.langTip = (target, code, extra = []) => {
  const l = W.lang(code);
  const SCRIPT = { Latn: "Latin", Cyrl: "Cyrillic", Arab: "Arabic", Deva: "Devanagari", Hebr: "Hebrew", Hans: "Simplified Chinese", Hant: "Traditional Chinese", Jpan: "Japanese", Hang: "Hangul", Khmr: "Khmer", Taml: "Tamil", Telu: "Telugu" };
  W.tip.show(target, {
    title: l.name,
    rows: [{ l: `${W.groupText(l.family)} · ${SCRIPT[l.script] || l.script} script` },
           { key: { dot: "c-" + l.order }, l: `WALS 83A: ${W.orderText(l.order)}` },
           { l: `Adpositions: ${l.post ? "postpositions" : "prepositions or mixed"}` }].concat(extra)
  });
};

/* the 2-D extent covering several layouts (for a fixed frame across layers and models) */
W.extentOf = layouts => {
  const xs = [], ys = [];
  layouts.forEach(xy => xy.forEach(p => { xs.push(p[0]); ys.push(p[1]); }));
  return [d3.extent(xs), d3.extent(ys)];
};
/* rescale a layout to a given root-mean-square radius around its centroid */
W.rescale = (xy, rms) => {
  const mx = d3.mean(xy, p => p[0]), my = d3.mean(xy, p => p[1]);
  const r = Math.sqrt(d3.mean(xy, p => (p[0] - mx) ** 2 + (p[1] - my) ** 2)) || 1;
  return xy.map(p => [(p[0] - mx) * rms / r, (p[1] - my) * rms / r]);
};
W.rmsOf = xy => { const mx = d3.mean(xy, p => p[0]), my = d3.mean(xy, p => p[1]); return Math.sqrt(d3.mean(xy, p => (p[0] - mx) ** 2 + (p[1] - my) ** 2)); };

/* simple pending (empty) state: a ghosted chart skeleton with a card */
W.pendingState = (ctx, { title, text, rows = 8, height = 300 }) => {
  const svg = W.frame(ctx, height, title + " (pending)");
  const w = ctx.width(), g = svg.append("g").attr("class", "ghost");
  const m = { l: w < 520 ? 90 : 140, r: 20, t: 18, b: 30 };
  const step = (height - m.t - m.b) / rows;
  for (let i = 0; i < rows; i++) {
    const y = m.t + step * (i + .5);
    g.append("rect").attr("x", 12).attr("y", y - 4).attr("width", m.l - 40 - (i % 3) * 12).attr("height", 8).attr("rx", 4).attr("class", "gfill").attr("stroke", "none");
    g.append("line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", y).attr("y2", y).attr("stroke-width", 1);
  }
  g.append("line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", height - m.b + 6).attr("y2", height - m.b + 6);
  const ov = W.el("div", "pend-overlay"), card = W.el("div", "pend-card");
  card.append(W.el("span", "pend-chip", "Pending"), W.el("b", null, title), W.el("span", null, text));
  ov.appendChild(card); ctx.graphic.appendChild(ov);
};
})();

/* text with subscripts in SVG: parts = ["β", ["OV"], " direction"] (an array element is set as a subscript) */
(function () {
  const W = window.WOA;
  W.subText = (sel, parts) => {
    sel.text(null);
    let down = false;
    parts.forEach(p => {
      const sub = Array.isArray(p), t = sel.append("tspan").text(sub ? p[0] : p);
      if (sub && !down) { t.attr("dy", "0.32em").style("font-size", "0.74em"); down = true; }
      else if (!sub && down) { t.attr("dy", "-0.32em"); down = false; }
    });
    return sel;
  };
  /* the same as plain text, for tooltips and aria */
  W.subPlain = parts => parts.map(p => (Array.isArray(p) ? p[0] : p)).join("");
})();
