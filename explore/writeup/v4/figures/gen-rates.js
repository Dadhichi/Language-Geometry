/* gen-rates: object-first share of verb-object pairs in the model's own continuations, per language: unsteered and
   steered both ways, against the band of random directions of the same norm. Controls: direction and strength.
   The row renderer W.genRows is shared with gen2-rates (confirmatory run). */
(function () {
"use strict";
const W = window.WOA;

/* rows: [{code, name, base:[v,n], minus:[v,n], plus:[v,n], mMinus, mPlus, mBase, rand:[...]|null, group?, note?}] */
W.genRows = (ctx, rows, opts) => {
  const w = ctx.width(), narrow = w < 520;
  const rowH = 42, headH = 28, m = { t: 30, r: opts.right ? (narrow ? 50 : 120) : 14, b: 40, l: narrow ? 112 : 132 };
  const groups = opts.groups || null;
  let y = m.t; const pos = [];
  rows.forEach((r, i) => {
    if (groups && (i === 0 || rows[i - 1].group !== r.group)) { pos.push({ head: r.group, y: y + 18 }); y += headH; }
    pos.push({ r, y: y + rowH / 2 }); y += rowH;
  });
  const H = y + m.b;
  const svg = W.frame(ctx, H, opts.label);
  const x = d3.scaleLinear().domain([0, 1]).range([m.l, w - m.r]);
  W.axisX(svg, x, H - m.b + 4, { values: narrow ? [0, .5, 1] : [0, .25, .5, .75, 1], grid: H - m.b - m.t + 10, format: d3.format(".0%") });
  svg.append("text").attr("x", w - m.r).attr("y", m.t - 14).attr("text-anchor", "end").attr("class", "small muted").text(opts.axis);
  if (opts.right && !narrow) svg.append("text").attr("x", w).attr("y", m.t - 14).attr("text-anchor", "end").attr("class", "cap").text(opts.rightHead || "");
  const pct = W.f.pct;
  pos.forEach(p => {
    if (p.head) { svg.append("text").attr("x", 0).attr("y", p.y).attr("class", "cap").text(p.head); return; }
    const r = p.r, yy = p.y, g = svg.append("g").attr("data-lang", W.norm(r.code));
    const lo = W.q(r.rand || [], .05), hi = W.q(r.rand || [], .95);
    if (lo != null && opts.band !== false) g.append("rect").attr("x", x(lo) - 4).attr("width", Math.max(8, x(hi) - x(lo) + 8)).attr("y", yy - 9).attr("height", 18).attr("rx", 9).attr("class", "band2")
      .on("pointermove", ev => W.tip.show(ev, { title: r.name, rows: [{ v: `${pct(lo)}–${pct(hi)}`, l: `random directions at ±${opts.krand || 2}, 5th–95th percentile` }, { v: `${(r.rand || []).filter(v => v != null).length} of ${(r.rand || []).length}`, l: "random conditions with enough pairs" }] }))
      .on("pointerleave", () => W.tip.hide());
    const ok = v => v && v[0] != null && !(v[1] != null && v[1] < (opts.minN || 20));
    const vals = [r.minus, r.base, r.plus].filter(ok).map(v => v[0]);
    if (vals.length > 1) g.append("line").attr("x1", x(d3.min(vals))).attr("x2", x(d3.max(vals))).attr("y1", yy).attr("y2", yy).attr("class", "s-axis").attr("stroke-width", 2);
    g.append("text").attr("x", m.l - 14).attr("y", yy + (r.sub ? -1 : 4)).attr("text-anchor", "end").attr("class", "ink").text(r.name);
    if (r.sub) g.append("text").attr("x", m.l - 14).attr("y", yy + 12).attr("text-anchor", "end").attr("class", "small muted").text(r.sub);
    const pts = [["minus", opts.cMinus || "vo", r.mMinus, opts.kMinus], ["plus", opts.cPlus || "ov", r.mPlus, opts.kPlus], ["base", "base", r.mBase, "unsteered"]];
    pts.forEach(([k, c, match, lab]) => {
      const v = r[k]; if (!v || v[0] == null || (v[1] != null && v[1] < (opts.minN || 20))) return;   /* not shown: fewer than 20 pairs */
      const hollow = match != null && match < .5;
      const el = c === "base" ? g.append("circle").attr("r", 3.6).attr("class", "dot c-ink2")
        : g.append("circle").attr("r", 6).attr("class", hollow ? "ring s-" + c : "dot c-" + c);
      el.attr("cx", x(v[0])).attr("cy", yy);
    });
    if (opts.right) { const t = opts.right(r); if (t) g.append("text").attr("x", w).attr("y", yy + 4).attr("text-anchor", "end").attr("class", t.cls || "small").text(narrow ? t.short || t.text : t.text); }
    g.append("rect").attr("class", "hit").attr("x", 0).attr("width", m.l).attr("y", yy - rowH / 2).attr("height", rowH)
      .on("pointermove", ev => { W.highlight([r.code]); tip(ev, r); }).on("pointerleave", () => { W.highlight(null); W.tip.hide(); });
    g.selectAll("circle").on("pointermove", ev => { W.highlight([r.code]); tip(ev, r); }).on("pointerleave", () => { W.highlight(null); W.tip.hide(); });
  });
  function tip(ev, r) {
    const row = (v, match, key, lab) => (v && v[0] != null && !(v[1] != null && v[1] < (opts.minN || 20)) ? { key, v: pct(v[0]), l: `${lab} · ${v[1]} pairs${match != null ? ` · ${pct(match)} in language` : ""}` } : { key, v: "–", l: `${lab} · fewer than ${opts.minN || 20} pairs${v && v[1] != null ? ` (${v[1]})` : ""}, not shown` });
    W.tip.show(ev, { title: r.name, rows: [row(r.minus, r.mMinus, { dot: "c-" + (opts.cMinus || "vo") }, opts.kMinus), row(r.base, r.mBase, { dot: "c-ink2" }, "unsteered"), row(r.plus, r.mPlus, { dot: "c-" + (opts.cPlus || "ov") }, opts.kPlus)].concat(r.tipExtra || []) });
  }
  const node = svg.node();
  W.keyNav(node, () => rows, (r, i) => { const b = node.getBoundingClientRect(), p = pos.find(q => q.r === r); W.highlight([r.code]); tip({ clientX: b.left + m.l + 10, clientY: b.top + p.y }, r); }, () => { W.highlight(null); W.tip.hide(); });
};

W.fig("gen-rates", {
  needs: ["freegen_x.rates", "freegen_x.match"],
  layout: "wide",
  init(ctx) {
    const S = ctx.state; S.dir = "ov"; S.k = 2;
    S.segDir = W.ui.seg(ctx.controls, { label: "Direction", options: [["ov", "Word order"], ["ie", "Indo-European"]], value: "ov",
      onChange: v => { S.dir = v; if (v === "ie") { S.k = 2; S.segK.set(2); } S.segK.disable(1, v === "ie"); legend(ctx); ctx.redraw(); } });
    S.segK = W.ui.seg(ctx.controls, { label: "Strength", options: [[2, "k = ±2"], [1, "k = ±1"]], value: 2, onChange: v => { S.k = +v; ctx.redraw(); } });
    legend(ctx);
  },
  draw(ctx) {
    const S = ctx.state, D = ctx.D, X = D.freegen_x, k = S.k, dir = S.dir;
    const tab = D.freegen && D.freegen.table;
    const rows = Object.keys(X.rates).map(c => {
      const R = X.rates[c], M = X.match[c];
      let minus, plus, mMinus, mPlus;
      if (dir === "ie") { minus = R["ie-"]; plus = R["ie+"]; mMinus = M.ie_m; mPlus = M.ie_p; }
      else if (k === 2) { minus = R["ov-"]; plus = R["ov+"]; mMinus = M.ov_m; mPlus = M.ov_p; }
      else { minus = R["ov-half"]; plus = R["ov+half"]; mMinus = tab && tab["ov-1"] && tab["ov-1"][c] ? tab["ov-1"][c].match : null; mPlus = tab && tab["ov+1"] && tab["ov+1"][c] ? tab["ov+1"][c].match : null; }
      const low = [mMinus, mPlus].map(v => (v == null ? 1 : v));
      const sub = Math.min(...low) < .8 ? `${W.f.pct(mMinus)} / ${W.f.pct(mPlus)} in language` : null;
      return { code: c, name: W.shortName(c), base: R.base, minus, plus, mMinus, mPlus, mBase: M.base, rand: R.rand, sub };
    }).sort((a, b) => (a.base[0] ?? 0) - (b.base[0] ?? 0));
    const ie = dir === "ie";
    W.genRows(ctx, rows, { label: "Object-first share of verb–object pairs per language, unsteered and steered", axis: "object-first share of counted verb–object pairs",
      kMinus: ie ? "Indo-European direction, k = −2" : `k = −${k} (toward verb-first)`, kPlus: ie ? "Indo-European direction, k = +2" : `k = +${k} (toward object-first)`,
      cMinus: ie ? "muted" : "vo", cPlus: ie ? "gen" : "ov", band: k === 2 });
  }
});
function legend(ctx) {
  const ie = ctx.state.dir === "ie";
  W.ui.legend(ctx.legend, [ie ? { kind: "dot", cls: "c-muted", text: "Indo-European, k = −2" } : { kind: "dot", cls: "c-vo", text: "k < 0, toward verb-first" }, { kind: "dot", cls: "c-ink2", text: "unsteered" },
    ie ? { kind: "dot", cls: "c-gen", text: "Indo-European, k = +2" } : { kind: "dot", cls: "c-ov", text: "k > 0, toward object-first" },
    { kind: "ring", cls: "s-ink2", text: "under half the text in the prompt language" }, { kind: "band", cls: "band2", text: "random directions at ±2, 5th–95th percentile" }]);
}
})();
