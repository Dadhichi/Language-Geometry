/* within-steer: noun-object share before the verb against the steering strength k, one small multiple per language
   (flexible order first, then controls), with a model control.
   Qwen2.5-7B (explore/steer_within): d_W at k = -3..+3 (dark line), beta_OV at k = +-2 (squares, blue toward verb-first,
   orange toward object-first), random directions at +-2 from the confirmatory run (both signs pooled).
   Llama-3.1-8B (explore/steer_llama): both directions as dose curves over k = -3..+3: d_W dark, beta_OV as squares joined
   by a line that is blue on the verb-first side and orange on the object-first side; random band at -2 and at +2 from
   rates_rand (order r0-, r0+, r1-, ...). Hollow marks: under half the continuations stayed in the prompt language.
   Shares from fewer than 20 noun-object pairs are not drawn. Below: the pre-registered pooled tests with their verdicts. */
(function () {
"use strict";
const W = window.WOA;
const FLEX = ["deu_Latn", "nld_Latn", "rus_Cyrl", "ukr_Cyrl", "pol_Latn", "hrv_Latn"], CTRL = ["eng_Latn", "spa_Latn", "kor_Hang"];
const KS = [-3, -2, -1, 0, 1, 2, 3];
const MIN_N = 20;
const keyW = (m, k) => (k === 0 ? "base" : m === "qwen" ? ({ "-3": "w-3", "-2": "w-", "-1": "w-1", 1: "w+1", 2: "w+", 3: "w+3" })[k] : `w${k > 0 ? "+" : "-"}${Math.abs(k)}`);
const keyO = (m, k) => (k === 0 ? "base" : m === "qwen" ? ({ "-2": "ov-", 2: "ov+" })[k] : `ov${k > 0 ? "+" : "-"}${Math.abs(k)}`);

/* everything the drawing needs for one model */
function model(D, m) {
  if (m === "llama") {
    const S = D.steer_llama; if (!S || !S.rates) return null;
    const band = {};
    Object.entries(S.rates_rand || {}).forEach(([c, arr]) => {
      const neg = arr.filter((v, i) => i % 2 === 0), pos = arr.filter((v, i) => i % 2 === 1);
      band[c] = { "-2": [W.q(neg, .05), W.q(neg, .95)], 2: [W.q(pos, .05), W.q(pos, .95)] };
    });
    const tests = [["L_H1", "L-H1", "β", ["OV"], "German, Russian", false], ["L_B", "L-B", "d", ["W"], "six flexible-order languages", false], ["H2_ov", "H2", "β", ["OV"], "Dutch, Ukrainian, Polish, Croatian", true]]
      .filter(([k]) => S[k]).map(([k, lab, a, b, langs, secondary]) => ({ t: S[k], lab, dir: [a, b], kind: S[k].kind, langs, secondary }));
    return { m, name: "Llama-3.1-8B", layer: S.layer || 16, rates: S.rates, match: S.match || {}, ovKs: KS, band, bandBoth: false, tests };
  }
  const Wn = D.within; if (!Wn || !Wn.rates) return null;
  const F2 = D.freegen2, band = {};
  Object.keys(Wn.rates).forEach(c => { const r = F2 && F2.rates && F2.rates[c] ? F2.rates[c].rand : null; if (r) { const b = [W.q(r, .05), W.q(r, .95)]; band[c] = { "-2": b, 2: b }; } });
  const tests = Wn.B && Wn.B.rand ? [{ t: Object.assign({ kind: "w" }, Wn.B, { b: Wn.B.b_w }), lab: "B", dir: ["d", ["W"]], kind: "w", langs: "six flexible-order languages", secondary: false }] : [];
  return { m: "qwen", name: "Qwen2.5-7B", layer: 14, rates: Wn.rates, match: Wn.match || {}, ovKs: [-2, 2], band, bandBoth: true, tests };
}

function verdict(T, secondary) {
  const t = T.t, z = t.z != null ? t.z.toFixed(2) : "–";
  if (t.evaluable === false) return { text: "not evaluable", strong: false };
  if (t.claim) return { text: `${secondary ? "secondary, holds" : "claim holds"}: z = ${z}`, strong: true };
  const reason = t.rand_max != null && t.b <= t.rand_max ? "below the largest random slope" : "z below 2.58";
  return { text: `no claim: z = ${z}, ${reason}`, strong: false, fail: true };
}

W.fig("within-steer", {
  needs: ["within.rates", "within.match"],
  layout: "page",
  init(ctx) {
    ctx.state.model = "qwen";
    if (ctx.D.steer_llama && ctx.D.steer_llama.rates)
      W.modelSeg(ctx, [["qwen", "Qwen2.5-7B", "pre-registered study at layer 14"], ["llama", "Llama-3.1-8B", "pre-registered replication at layer 16"]]);
    ctx.onModel = () => legend(ctx);
    legend(ctx);
  },
  draw(ctx) {
    const D = ctx.D, w = ctx.width();
    const M = model(D, ctx.state.model) || model(D, "qwen"), R = M.rates, MT = M.match, llama = M.m === "llama";
    const langs = FLEX.filter(c => R[c]).concat(CTRL.filter(c => R[c])), nFlex = FLEX.filter(c => R[c]).length;
    const cols = w >= 900 ? 6 : w >= 560 ? 3 : 2, gx = 14, gy = 30, labW = 34;
    const pw = (w - labW - (cols - 1) * gx) / cols, ph = 148, headH = 24;
    const cells = []; let row = 0, col = 0;
    langs.forEach((c, i) => {
      if (i === nFlex && col !== 0) { row++; col = 0; }
      cells.push({ c, row, col, head: (i === 0 ? "Flexible order" : i === nFlex ? "Controls, rigid order" : null) });
      col++; if (col === cols) { col = 0; row++; }
    });
    const nRows = d3.max(cells, d => d.row) + 1, rowY = r => r * (ph + gy + headH) + headH;
    const narrow = w < 560, rowStrip = 68, stripH = M.tests.length ? 44 + M.tests.length * rowStrip + 40 : 0;
    const H = nRows * (ph + gy + headH) + stripH;
    const svg = W.frame(ctx, H, `Noun-object share before the verb against steering strength per language, ${M.name}`);
    const x0 = d3.scaleLinear().domain([-3.4, 3.4]).range([0, pw]);
    const y0 = d3.scaleLinear().domain([0, 1]).range([ph - 26, 26]);
    const ok = v => v && v[0] != null && v[1] >= MIN_N;
    const panels = [];
    cells.forEach(({ c, row, col, head }) => {
      const ox = labW + col * (pw + gx), oy = rowY(row), r = R[c], mt = MT[c] || {}, name = W.shortName(c);
      if (head) svg.append("text").attr("x", 0).attr("y", oy - 10).attr("class", "cap").text(head);
      const g = svg.append("g").attr("transform", `translate(${ox},${oy})`);
      g.append("text").attr("x", 0).attr("y", 10).attr("class", "strong").attr("data-lang", c).text(name);
      [0, .5, 1].forEach(t => g.append("line").attr("x1", 0).attr("x2", pw).attr("y1", y0(t)).attr("y2", y0(t)).attr("class", t === 0 ? "baseline" : "gridline"));
      if (col === 0) [0, .5, 1].forEach(t => svg.append("text").attr("x", labW - 6).attr("y", oy + y0(t) + 3.5).attr("text-anchor", "end").attr("class", "small muted").text(t * 100 + "%"));
      KS.forEach(k => g.append("text").attr("x", x0(k)).attr("y", ph - 10).attr("text-anchor", "middle").attr("class", "small muted").text(k === 0 ? "0" : W.f.k(k)));
      const bd = M.band[c];
      if (bd) [-2, 2].forEach(k => { const [lo, hi] = bd[k]; if (lo == null) return;
        g.append("rect").attr("x", x0(k) - 9).attr("width", 18).attr("y", y0(hi) - 4).attr("height", Math.max(8, y0(lo) - y0(hi) + 8)).attr("rx", 4).attr("class", "band2"); });
      if (ok(r.base)) g.append("line").attr("x1", 0).attr("x2", pw).attr("y1", y0(r.base[0])).attr("y2", y0(r.base[0])).attr("class", "s-ink2").attr("stroke-width", 1).attr("opacity", .35);
      const notes = [];
      /* beta_OV: in the Llama view a full dose curve (blue toward verb-first, orange toward object-first) */
      const ovPts = M.ovKs.map(k => ({ k, v: r[keyO(M.m, k)], mt: mt[keyO(M.m, k)] }));
      if (llama) {
        const base = { k: 0, v: r.base };
        [[ovPts.filter(p => p.k < 0).concat([base]), "s-vo"], [[base].concat(ovPts.filter(p => p.k > 0)), "s-ov"]].forEach(([seg, cls]) =>
          g.append("path").datum(seg.sort((a, b) => a.k - b.k)).attr("class", "line thin " + cls).attr("stroke-width", 1.4).attr("opacity", .85)
            .attr("d", d3.line().defined(p => ok(p.v)).x(p => x0(p.k)).y(p => y0(p.v[0]))));
      }
      /* d_W */
      const wPts = KS.map(k => ({ k, v: r[keyW(M.m, k)], mt: mt[k === 0 ? "base" : keyW(M.m, k)] }));
      g.append("path").datum(wPts).attr("class", "line s-ink").attr("d", d3.line().defined(p => ok(p.v)).x(p => x0(p.k)).y(p => y0(p.v[0])));
      wPts.filter(p => ok(p.v)).forEach(p => g.append("circle").attr("cx", x0(p.k)).attr("cy", y0(p.v[0])).attr("r", p.k === 0 ? 3.8 : 3.4)
        .attr("class", p.k === 0 ? "dot c-ink2" : p.mt != null && p.mt < .5 ? "ring s-ink" : "dot c-ink"));
      ovPts.forEach(p => {
        if (p.k === 0) return;
        const cls = p.k < 0 ? "vo" : "ov";
        if (!ok(p.v)) { notes.push(p.mt != null && p.mt < .5 ? `β ${W.f.k(p.k)}: ${W.f.pct(p.mt)} stay` : `β ${W.f.k(p.k)}: n = ${p.v ? p.v[1] : 0}`); return; }
        const hollow = p.mt != null && p.mt < .5, s = 9, cx = x0(p.k) + (llama ? 0 : p.k > 0 ? 8 : -8);
        g.append("rect").attr("x", cx - s / 2).attr("y", y0(p.v[0]) - s / 2).attr("width", s).attr("height", s).attr("rx", 1.5).attr("class", hollow ? "ring s-" + cls : "dot c-" + cls);
      });
      if (notes.length) g.append("text").attr("x", pw).attr("y", 10).attr("text-anchor", "end").attr("class", "small muted").text(notes.length > 1 && (narrow || pw < 190) ? `β: ${notes.length} not drawn` : notes.join(" · "));
      const hair = g.append("line").attr("class", "xhair").attr("y1", 16).attr("y2", ph - 24);
      const hit = g.append("rect").attr("class", "hit").attr("x", 0).attr("y", 0).attr("width", pw).attr("height", ph);
      panels.push({ c, name, g, hair, hit, ox, oy, r, mt, bd });
    });
    const tipFor = (P, k, ev) => {
      const r = P.r, mt = P.mt, kw = keyW(M.m, k), v = r[kw], mk = k === 0 ? "base" : kw;
      const pct = x => W.f.pct(x), n = x => (x ? x[1] : 0);
      const rows = [{ key: { line: "s-ink" }, v: ok(v) ? pct(v[0]) : "–", l: [k === 0 ? "unsteered" : "d", k === 0 ? "" : ["W"], k === 0 ? "" : ` at k = ${W.f.k(k)}`, ` · n = ${n(v)}${!ok(v) ? " (not drawn)" : ""}${mt[mk] != null ? ` · ${pct(mt[mk])} in language` : ""}`] }];
      const ko = k !== 0 && M.ovKs.includes(k) ? keyO(M.m, k) : null;
      if (ko) { const ov = r[ko]; rows.push({ key: { dot: k > 0 ? "c-ov" : "c-vo" }, v: ok(ov) ? pct(ov[0]) : "–", l: ["β", ["OV"], ` at k = ${W.f.k(k)} · n = ${n(ov)}${!ok(ov) ? " (not drawn)" : ""}${mt[ko] != null ? ` · ${pct(mt[ko])} in language` : ""}`] }); }
      if (Math.abs(k) === 2 && P.bd && P.bd[k][0] != null) rows.push({ v: `${pct(P.bd[k][0])}–${pct(P.bd[k][1])}`, l: M.bandBoth ? "random directions at ±2 (both signs)" : `random directions at k = ${W.f.k(k)}` });
      W.tip.show(ev, { title: `${P.name} · ${M.name} · k = ${k === 0 ? "0" : W.f.k(k)}`, rows, note: "share of noun objects placed before their verb" });
    };
    const show = (k, ev, src) => { panels.forEach(P => P.hair.attr("x1", x0(k)).attr("x2", x0(k)).style("opacity", 1)); W.highlight([src.c]); tipFor(src, k, ev); };
    const hide = () => { panels.forEach(P => P.hair.style("opacity", 0)); W.highlight(null); W.tip.hide(); };
    panels.forEach(P => P.hit.on("pointermove", ev => { const [px] = d3.pointer(ev, P.g.node()); const k = KS.reduce((a, b) => (Math.abs(x0(b) - px) < Math.abs(x0(a) - px) ? b : a)); show(k, ev, P); }).on("pointerleave", hide));
    const node = svg.node(), items = panels.flatMap(P => KS.map(k => ({ P, k }))), sItems = [];

    /* the pre-registered pooled tests, each with its verdict (a failure is stated, not hidden) */
    if (stripH) {
      const top = nRows * (ph + gy + headH) + 14, r0 = 4.2, pts = v => W.f.pts(v, 1);
      const all = M.tests.flatMap(T => T.t.rand.filter(v => v != null).concat([T.t.b, 0]));
      const lm = narrow ? labW : 200, rm = narrow ? 8 : 230;
      const sp = d3.scaleLinear().domain(d3.extent(all).map(v => 100 * v)).nice().range([lm, w - rm]), sx = v => sp(100 * v);
      svg.append("text").attr("x", 0).attr("y", top).attr("class", "small strong").text(narrow ? `Pooled tests, ${M.name}, k = ±2` : `Pre-registered pooled tests, ${M.name}: slope of the noun-object share between k = −2 and +2`);
      const gs = svg.append("g");
      M.tests.forEach((T, i) => {
        const t = T.t, rand = t.rand.filter(v => v != null), y = top + 30 + i * rowStrip, base = y + 32;
        const dod = W.dodge(rand, sx, r0), lev = d3.max(dod, d => d.lev) + 1, step = Math.min(2 * r0 + 1, 32 / lev);
        const V = verdict(T, T.secondary);
        if (i) gs.append("line").attr("x1", 0).attr("x2", w).attr("y1", y - 8).attr("y2", y - 8).attr("class", "gridline");
        const lab = gs.append("text").attr("x", 0).attr("y", narrow ? y + 2 : base - 4).attr("class", "strong");
        W.subText(lab, [T.lab + (T.secondary ? " (secondary)" : "") + " · ", T.dir[0], T.dir[1]]);
        if (!narrow) gs.append("text").attr("x", 0).attr("y", base + 10).attr("class", "small muted").text(T.langs);
        gs.append("line").attr("x1", sx(0)).attr("x2", sx(0)).attr("y1", y).attr("y2", base + 12).attr("class", "zero");
        dod.forEach(d => gs.append("circle").attr("cx", sx(d.v)).attr("cy", base - d.lev * step).attr("r", r0).attr("class", "dot c-rand-strong"));
        const cls = T.kind === "w" ? "c-ink" : "c-ov";
        const tipT = target => W.tip.show(target, { title: [T.lab + (T.secondary ? " (secondary)" : "") + ": ", T.dir[0], T.dir[1], `, ${T.langs}`], rows: [
          { key: { dot: cls }, v: `${pts(t.b)} pts`, l: "pooled slope per unit k" },
          { key: { dot: "c-rand-strong" }, v: `${pts(t.rand_mean)} ± ${(100 * t.rand_sd).toFixed(1)}`, l: `${rand.length} random directions (largest ${pts(t.rand_max)} pts)` },
          { v: `z = ${t.z != null ? t.z.toFixed(2) : "–"}`, l: `empirical p = ${t.p_emp != null ? t.p_emp.toFixed(3) : "–"}, ${t.n_langs} languages` },
          { v: V.text } ] });
        sItems.push({ tipT, x: sx(t.b), y: base });
        gs.append("circle").attr("cx", sx(t.b)).attr("cy", base).attr("r", 6.5).attr("class", "dot " + cls)
          .on("pointermove", ev => tipT(ev)).on("pointerleave", () => W.tip.hide());
        if (t.rand_max != null) gs.append("line").attr("x1", sx(t.rand_max)).attr("x2", sx(t.rand_max)).attr("y1", base - 12).attr("y2", base + 8).attr("class", "s-ink2").attr("stroke-width", 1);
        gs.append("text").attr("x", narrow ? w : w).attr("y", narrow ? y + 2 : base + 4).attr("text-anchor", "end").attr("class", "small strong").text(narrow ? V.text.replace(", below the largest random slope", " (< max random)").replace(", z below 2.58", " (< 2.58)") : V.text);
      });
      W.axisX(gs, sp, top + 30 + M.tests.length * rowStrip - 4, { ticks: narrow ? 4 : 8, format: d => (Number.isInteger(d) ? W.f.signed(d, 0) : W.f.signed(d, 1)).replace(/^\+0$/, "0") });
      gs.append("text").attr("x", w - rm).attr("y", top + 30 + M.tests.length * rowStrip + 26).attr("text-anchor", "end").attr("class", "small muted").text(narrow ? "pts per unit k" : "change in noun-object share per unit k, percentage points; tick: largest random slope");
    }
    W.keyNav(node, () => items.concat(sItems), it => { const b = node.getBoundingClientRect(); if (it.tipT) { W.highlight(null); it.tipT({ x: b.left + it.x + 12, y: b.top + it.y - 10 }); } else show(it.k, { clientX: b.left + it.P.ox + x0(it.k) + 12, clientY: b.top + it.P.oy + 30 }, it.P); }, hide);
  }
});

function legend(ctx) {
  const llama = ctx.state.model === "llama";
  W.ui.legend(ctx.legend, [{ kind: "line", cls: "s-ink", text: ["d", ["W"], ", within-language direction, k = −3 … +3"] },
    { kind: "bar", cls: "c-vo", text: ["β", ["OV"], llama ? ", k < 0 (toward verb-first)" : " at k = −2 (toward verb-first)"] },
    { kind: "bar", cls: "c-ov", text: ["β", ["OV"], llama ? ", k > 0 (toward object-first)" : " at k = +2 (toward object-first)"] },
    { kind: "dot", cls: "c-ink2", text: "unsteered (k = 0)" }, { kind: "ring", cls: "s-ink2", text: "hollow: under half the text in the prompt language" },
    { kind: "band", cls: "band2", text: llama ? "random directions at −2 and +2, 5th–95th percentile" : "random directions at ±2, 5th–95th percentile" }]);
}
})();
