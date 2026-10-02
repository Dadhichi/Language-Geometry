/* gen-null: pooled slope of the object-first share for the word-order direction against random directions of the same
   norm (exploratory, per-direction inclusion); a second row for noun objects only. W.nullRows is shared with gen2-null. */
(function () {
"use strict";
const W = window.WOA;
const pts = v => W.f.pts(v, 1);

/* rows: [{label, sub, rand:[...], ov, ci:[lo,hi]|null, ie|null, z, verdict?:{text, strong}}] */
W.nullRows = (ctx, rows, opts) => {
  const w = ctx.width(), narrow = w < 560, r = 4.5;
  const m = { l: narrow ? 0 : Math.max(150, d3.max(rows, R => Math.max(W.textW(R.label, 12, 620), R.sub ? W.textW(R.sub, 10.5) : 0)) + 22), r: narrow ? 12 : 120 };
  const all = rows.flatMap(R => R.rand.filter(v => v != null).concat([R.ov, R.ie, 0], R.ci || []).filter(v => v != null));
  const x = d3.scaleLinear().domain(d3.extent(all)).nice().range([m.l + 6, w - m.r]);
  let y = 14; const lay = [];
  rows.forEach(R => {
    const dod = W.dodge(R.rand.filter(v => v != null), x, r), lev = d3.max(dod, d => d.lev) + 1 || 1;
    const top = y + (narrow ? 20 : 0), base = top + lev * (2 * r + 1) + 4;
    lay.push({ R, dod, top, base, yOV: base + 14 }); y = base + 14 + 30;
  });
  const H = y + 22;
  const svg = W.frame(ctx, H, opts.label);
  W.axisX(svg, x, H - 34, { ticks: narrow ? 5 : 8, format: d => W.f.pts(d, 0).replace(/^\+0$/, "0") });
  svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 6).attr("y2", H - 34).attr("class", "zero");
  svg.append("text").attr("x", w - m.r).attr("y", H - 2).attr("text-anchor", "end").attr("class", "small muted").text(opts.axis);
  const items = [];
  lay.forEach(({ R, dod, top, base, yOV }, i) => {
    if (i) svg.append("line").attr("x1", 0).attr("x2", w).attr("y1", top - 14).attr("y2", top - 14).attr("class", "gridline");
    svg.append("text").attr("x", 0).attr("y", narrow ? top - 6 : base - 2).attr("class", "strong").text(R.label);
    if (R.sub) svg.append("text").attr("x", narrow ? W.textW(R.label, 12, 620) + 8 : 0).attr("y", narrow ? top - 6 : base + 12).attr("class", "small muted").text(R.sub);
    dod.forEach(d => svg.append("circle").attr("cx", x(d.v)).attr("cy", base - d.lev * (2 * r + 1)).attr("r", r).attr("class", "dot c-rand-strong")
      .on("pointermove", ev => W.tip.show(ev, { rows: [{ key: { dot: "c-rand-strong" }, v: `${pts(d.v)} pts`, l: "random direction, same norm" }] })).on("pointerleave", () => W.tip.hide()));
    if (R.ie != null) svg.append("circle").attr("cx", x(R.ie)).attr("cy", yOV).attr("r", 5.5).attr("class", "dot c-gen")
      .on("pointermove", ev => W.tip.show(ev, { rows: [{ key: { dot: "c-gen" }, v: `${pts(R.ie)} pts`, l: "Indo-European direction (control)" }] })).on("pointerleave", () => W.tip.hide());
    if (R.ci) { svg.append("line").attr("x1", x(R.ci[0])).attr("x2", x(R.ci[1])).attr("y1", yOV).attr("y2", yOV).attr("class", "line s-ov");
      R.ci.forEach(v => svg.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", yOV - 5).attr("y2", yOV + 5).attr("class", "line s-ov")); }
    if (R.ov != null) {
      const tipOV = target => W.tip.show(target, { title: `${R.label}: word-order direction`, rows: [{ key: { dot: "c-ov" }, v: `${pts(R.ov)} pts`, l: "per unit k" },
          R.ci ? { v: `${pts(R.ci[0])} to ${pts(R.ci[1])}`, l: "95% prompt bootstrap" } : null,
          R.z != null ? { v: `z = ${R.z.toFixed(1)}`, l: `against ${R.rand.filter(v => v != null).length} random directions` } : null,
          R.ie != null ? { key: { dot: "c-gen" }, v: `${pts(R.ie)} pts`, l: "Indo-European direction" } : null,
          { key: { dot: "c-rand-strong" }, v: `${pts(d3.min(R.rand.filter(v => v != null)))} to ${pts(d3.max(R.rand.filter(v => v != null)))}`, l: "random directions" }].filter(Boolean) });
      items.push({ tipOV, x: x(R.ov), y: yOV });
      svg.append("circle").attr("cx", x(R.ov)).attr("cy", yOV).attr("r", 6.5).attr("class", "dot c-ov")
        .on("pointermove", ev => tipOV(ev)).on("pointerleave", () => W.tip.hide());
      const zt = `z = ${R.z != null ? R.z.toFixed(1) : ""}`, ax = x(R.ci ? Math.max(R.ci[1], R.ov) : R.ov) + 12, right = ax + W.textW(zt, 12, 620) < w;
      if (R.z != null) svg.append("text").attr("x", right ? ax : x(R.ov)).attr("y", right ? yOV + 4 : yOV - 12).attr("text-anchor", right ? "start" : "middle").attr("class", "strong halo").text(zt);
    }
    if (R.verdict && !narrow) svg.append("text").attr("x", w).attr("y", yOV + 4).attr("text-anchor", "end").attr("class", R.verdict.strong ? "strong" : "muted").text(R.verdict.text);
    else if (R.verdict) svg.append("text").attr("x", w - m.r).attr("y", top - 6).attr("text-anchor", "end").attr("class", R.verdict.strong ? "small strong" : "small muted").text(R.verdict.text);
  });
  const node = svg.node();
  W.keyNav(node, () => items, it => { const r = node.getBoundingClientRect(); it.tipOV({ x: r.left + it.x + 12, y: r.top + it.y - 10 }); }, () => W.tip.hide());
};

W.fig("gen-null", {
  needs: ["freegen_x.pooled"],
  layout: "wide",
  init(ctx) {
    W.ui.legend(ctx.legend, [{ kind: "dot", cls: "c-ov", text: "word-order direction" }, { kind: "range", cls: "s-ov", text: "95% prompt bootstrap" },
      { kind: "dot", cls: "c-gen", text: "Indo-European direction" }, { kind: "dot", cls: "c-rand-strong", text: "24 random directions" }]);
  },
  draw(ctx) {
    const X = ctx.D.freegen_x, P = X.pooled, N = X.objtype_nom;
    const rows = [{ label: "All objects", sub: `${P.n_langs_ov || 5} languages`, rand: P.rand, ov: P.b_ov, ci: X.b_ov_boot95 || null, ie: P.b_ie, z: P.z }];
    if (N && N.rand) {
      const nl = N.per_language ? Object.values(N.per_language).filter(v => v != null).length : null;
      rows.push({ label: "Noun objects only", sub: nl ? `${nl} languages` : null, rand: N.rand, ov: N.b_ov, ci: null, ie: null, z: N.z });
    }
    W.nullRows(ctx, rows, { label: "Slope of the object-first share for the word-order direction against random directions", axis: "change in object-first share per unit k, percentage points" });
  }
});
})();
