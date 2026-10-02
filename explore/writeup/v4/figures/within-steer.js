/* within-steer: noun-object share before the verb against the steering strength k = -3..+3 along the within-language
   direction d_W (dark line), beta_OV at k = +-2 (squares, blue toward verb-first, orange toward object-first) and the band
   of random directions at +-2 (both signs, from the confirmatory run, same prompts and norm). One small multiple per
   language, flexible-order languages first, controls after. Shares from fewer than 20 noun-object pairs are not drawn.
   Below: the pre-registered statistic B, the pooled slope of d_W against the 24 random directions. */
(function () {
"use strict";
const W = window.WOA;
const FLEX = ["deu_Latn", "nld_Latn", "rus_Cyrl", "ukr_Cyrl", "pol_Latn", "hrv_Latn"], CTRL = ["eng_Latn", "spa_Latn", "kor_Hang"];
const KS = [-3, -2, -1, 0, 1, 2, 3];
const KEY = { "-3": "w-3", "-2": "w-", "-1": "w-1", 0: "base", 1: "w+1", 2: "w+", 3: "w+3" };
const MIN_N = 20;

W.fig("within-steer", {
  needs: ["within.rates", "within.match"],
  layout: "page",
  init(ctx) {
    W.ui.legend(ctx.legend, [{ kind: "line", cls: "s-ink", text: ["d", ["W"], ", within-language direction, k = −3 … +3"] },
      { kind: "bar", cls: "c-vo", text: ["β", ["OV"], " at k = −2 (toward verb-first)"] }, { kind: "bar", cls: "c-ov", text: ["β", ["OV"], " at k = +2 (toward object-first)"] },
      { kind: "band", cls: "band2", text: "random directions at ±2, 5th–95th percentile" }]);
  },
  draw(ctx) {
    const D = ctx.D, Wn = D.within, R = Wn.rates, M = Wn.match, F2 = D.freegen2, w = ctx.width();
    const langs = FLEX.filter(c => R[c]).concat(CTRL.filter(c => R[c])), nFlex = FLEX.filter(c => R[c]).length;
    const cols = w >= 900 ? 6 : w >= 560 ? 3 : 2, gx = 14, gy = 30, labW = 34;
    const pw = (w - labW - (cols - 1) * gx) / cols, ph = 148, headH = 24;
    /* rows: flexible languages fill rows first, controls start a new row */
    const cells = []; let row = 0, col = 0;
    langs.forEach((c, i) => {
      if (i === nFlex && col !== 0) { row++; col = 0; }
      cells.push({ c, row, col, head: (i === 0 ? "Flexible order" : i === nFlex ? "Controls, rigid order" : null) });
      col++; if (col === cols) { col = 0; row++; }
    });
    const nRows = d3.max(cells, d => d.row) + 1;
    const rowY = r => r * (ph + gy + headH) + headH;
    const B = Wn.B, stripH = B && B.rand ? 132 : 0;
    const H = nRows * (ph + gy + headH) + stripH;
    const svg = W.frame(ctx, H, "Noun-object share before the verb against steering strength, per language");
    const x0 = d3.scaleLinear().domain([-3.4, 3.4]).range([0, pw]);
    const y0 = d3.scaleLinear().domain([0, 1]).range([ph - 26, 26]);
    const ok = v => v && v[0] != null && v[1] >= MIN_N;
    const panels = [];
    cells.forEach(({ c, row, col, head }) => {
      const ox = labW + col * (pw + gx), oy = rowY(row), r = R[c], m = M[c] || {}, name = W.shortName(c);
      if (head) svg.append("text").attr("x", 0).attr("y", oy - 10).attr("class", "cap").text(head);
      const g = svg.append("g").attr("transform", `translate(${ox},${oy})`);
      g.append("text").attr("x", 0).attr("y", 10).attr("class", "strong").attr("data-lang", c).text(name);
      [0, .5, 1].forEach(t => g.append("line").attr("x1", 0).attr("x2", pw).attr("y1", y0(t)).attr("y2", y0(t)).attr("class", t === 0 ? "baseline" : "gridline"));
      if (col === 0) [0, .5, 1].forEach(t => svg.append("text").attr("x", labW - 6).attr("y", oy + y0(t) + 3.5).attr("text-anchor", "end").attr("class", "small muted").text(t * 100 + "%"));
      KS.forEach(k => g.append("text").attr("x", x0(k)).attr("y", ph - 10).attr("text-anchor", "middle").attr("class", "small muted").text(k === 0 ? "0" : W.f.k(k)));
      /* random band at +-2 */
      const rnd = F2 && F2.rates && F2.rates[c] ? F2.rates[c].rand : null, lo = rnd ? W.q(rnd, .05) : null, hi = rnd ? W.q(rnd, .95) : null;
      if (lo != null) [-2, 2].forEach(k => g.append("rect").attr("x", x0(k) - 9).attr("width", 18).attr("y", y0(hi) - 4).attr("height", Math.max(8, y0(lo) - y0(hi) + 8)).attr("rx", 4).attr("class", "band2"));
      /* unsteered reference */
      if (ok(r.base)) g.append("line").attr("x1", 0).attr("x2", pw).attr("y1", y0(r.base[0])).attr("y2", y0(r.base[0])).attr("class", "s-ink2").attr("stroke-width", 1).attr("opacity", .35);
      /* d_W dose-response, broken where a share rests on too few pairs */
      const pts = KS.map(k => ({ k, v: r[KEY[k]] }));
      g.append("path").datum(pts).attr("class", "line s-ink").attr("d", d3.line().defined(p => ok(p.v)).x(p => x0(p.k)).y(p => y0(p.v[0])));
      pts.filter(p => ok(p.v)).forEach(p => g.append("circle").attr("cx", x0(p.k)).attr("cy", y0(p.v[0])).attr("r", p.k === 0 ? 3.6 : 3.4).attr("class", p.k === 0 ? "ring s-ink" : "dot c-ink"));
      /* beta_OV at +-2: squares beside the d_W points, hollow when under half the text stayed in the language */
      const notes = [];
      [["ov-", -2, "vo", "ov-"], ["ov+", 2, "ov", "ov+"]].forEach(([key, k, cls, mk]) => {
        const v = r[key];
        if (!ok(v)) { notes.push(`β ${k > 0 ? "+" : "−"}2: n = ${v ? v[1] : 0}`); return; }
        const hollow = m[mk] != null && m[mk] < .5, s = 9, cx = x0(k) + (k > 0 ? 8 : -8);
        g.append("rect").attr("x", cx - s / 2).attr("y", y0(v[0]) - s / 2).attr("width", s).attr("height", s).attr("rx", 1.5)
          .attr("class", hollow ? "ring s-" + cls : "dot c-" + cls);
      });
      if (notes.length) g.append("text").attr("x", pw).attr("y", 10).attr("text-anchor", "end").attr("class", "small muted").text(notes.join(" · "));
      const hair = g.append("line").attr("class", "xhair").attr("y1", 16).attr("y2", ph - 24);
      const hit = g.append("rect").attr("class", "hit").attr("x", 0).attr("y", 0).attr("width", pw).attr("height", ph);
      panels.push({ c, name, g, hair, hit, ox, oy, r, m, lo, hi });
    });
    const fmt = v => (v && v[0] != null ? `${W.f.pct(v[0])}` : "–");
    const tipFor = (P, k, ev) => {
      const r = P.r, m = P.m, v = r[KEY[k]], mk = k === 0 ? "base" : KEY[k];
      const rows = [{ key: { line: "s-ink" }, v: ok(v) ? fmt(v) : "–", l: [k === 0 ? "unsteered" : "d", k === 0 ? "" : ["W"], k === 0 ? "" : ` at k = ${W.f.k(k)}`, ` · n = ${v ? v[1] : 0}${!ok(v) ? " (not drawn)" : ""}${m[mk] != null ? ` · ${W.f.pct(m[mk])} in language` : ""}`] }];
      if (Math.abs(k) === 2) {
        const key = k > 0 ? "ov+" : "ov-", ov = r[key];
        rows.push({ key: { dot: k > 0 ? "c-ov" : "c-vo" }, v: ok(ov) ? fmt(ov) : "–", l: ["β", ["OV"], ` at k = ${W.f.k(k)} · n = ${ov ? ov[1] : 0}${!ok(ov) ? " (not drawn)" : ""}${m[key] != null ? ` · ${W.f.pct(m[key])} in language` : ""}`] });
        if (P.lo != null) rows.push({ v: `${W.f.pct(P.lo)}–${W.f.pct(P.hi)}`, l: "random directions at ±2" });
      }
      W.tip.show(ev, { title: `${P.name} · k = ${k === 0 ? "0" : W.f.k(k)}`, rows, note: "share of noun objects placed before their verb" });
    };
    const show = (k, ev, src) => { panels.forEach(P => P.hair.attr("x1", x0(k)).attr("x2", x0(k)).style("opacity", 1)); W.highlight([src.c]); tipFor(src, k, ev); };
    const hide = () => { panels.forEach(P => P.hair.style("opacity", 0)); W.highlight(null); W.tip.hide(); };
    panels.forEach(P => P.hit.on("pointermove", ev => { const [px] = d3.pointer(ev, P.g.node()); const k = KS.reduce((a, b) => (Math.abs(x0(b) - px) < Math.abs(x0(a) - px) ? b : a)); show(k, ev, P); }).on("pointerleave", hide));
    const node = svg.node(), items = panels.flatMap(P => KS.map(k => ({ P, k })));
    W.keyNav(node, () => items, it => { const b = node.getBoundingClientRect(); show(it.k, { clientX: b.left + it.P.ox + x0(it.k) + 12, clientY: b.top + it.P.oy + 30 }, it.P); }, hide);

    /* pre-registered statistic B */
    if (stripH) {
      const top = nRows * (ph + gy + headH) + 18, r0 = 4.5, pts = v => W.f.pts(v, 1);
      const rand = B.rand.filter(v => v != null), all = rand.concat([B.b_w, 0]);
      const sp = d3.scaleLinear().domain(d3.extent(all).map(v => 100 * v)).nice().range([labW + 6, w - 16]);   /* in percentage points */
      const sx = v => sp(100 * v);
      const dod = W.dodge(rand, sx, r0), lev = d3.max(dod, d => d.lev) + 1, base = top + 22 + lev * (2 * r0 + 1);
      const gs = svg.append("g");
      W.subText(gs.append("text").attr("x", 0).attr("y", top).attr("class", "small strong"), w < 640 ? ["Test B: pooled slope of d", ["W"], ", k = ±2"] : ["Pre-registered test B: pooled slope of d", ["W"], " over the six flexible-order languages, k = ±2"]);
      gs.append("line").attr("x1", sx(0)).attr("x2", sx(0)).attr("y1", top + 8).attr("y2", base + 20).attr("class", "zero");
      dod.forEach(d => gs.append("circle").attr("cx", sx(d.v)).attr("cy", base - d.lev * (2 * r0 + 1)).attr("r", r0).attr("class", "dot c-rand-strong")
        .on("pointermove", ev => W.tip.show(ev, { rows: [{ key: { dot: "c-rand-strong" }, v: `${pts(d.v)} pts`, l: "random direction, pooled slope" }] })).on("pointerleave", () => W.tip.hide()));
      gs.append("circle").attr("cx", sx(B.b_w)).attr("cy", base + 14).attr("r", 6.5).attr("class", "dot c-ink")
        .on("pointermove", ev => W.tip.show(ev, { title: ["d", ["W"], ", pooled slope at k = ±2"], rows: [{ key: { dot: "c-ink" }, v: `${pts(B.b_w)} pts`, l: "per unit k" },
          { v: `z = ${B.z.toFixed(2)}`, l: `against ${rand.length} random directions (largest ${pts(B.rand_max)} pts)` },
          Wn.dose ? { v: Object.entries(Wn.dose).map(([k, v]) => `${pts(v)}`).join(" / "), l: `pooled slope at k = ${Object.keys(Wn.dose).map(k => (+k).toFixed(0)).join(" / ")}` } : null,
          { v: B.claim ? "claim holds" : "claim not supported", l: B.evaluable === false ? "not evaluable" : "" }].filter(Boolean) }))
        .on("pointerleave", () => W.tip.hide());
      const lt = `z = ${B.z.toFixed(2)}`, bx = sx(B.b_w), roomR = bx + 12 + W.textW("dW  " + lt, 12, 620) < w;
      W.subText(gs.append("text").attr("x", roomR ? bx + 12 : bx - 12).attr("y", base + 18).attr("text-anchor", roomR ? "start" : "end").attr("class", "strong halo"), ["d", ["W"], ` · ${lt}`]);
      gs.append("text").attr("x", sx(d3.min(rand))).attr("y", base - lev * (2 * r0 + 1) - 6).attr("class", "small muted").text(`${rand.length} random directions`);
      W.axisX(gs, sp, base + 26, { ticks: w < 560 ? 3 : 6, format: d => (Number.isInteger(d) ? W.f.signed(d, 0) : W.f.signed(d, 1)).replace(/^\+0$/, "0") });
      gs.append("text").attr("x", w - 16).attr("y", base + 56).attr("text-anchor", "end").attr("class", "small muted").text("change in noun-object share per unit k, percentage points");
    }
  }
});
})();
