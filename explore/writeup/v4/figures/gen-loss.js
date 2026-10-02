/* gen-loss: share of continuations that stay in the prompt language under the word-order direction at k = +-2, per
   language; the push against the language's majority order is ringed; unsteered and the random band for comparison. */
(function () {
"use strict";
const W = window.WOA;

W.fig("gen-loss", {
  needs: ["freegen_x.match", "freegen_x.sign_loss"],
  layout: "wide",
  init(ctx) {
    W.ui.legend(ctx.legend, [{ kind: "dot", cls: "c-vo", text: "k = −2, toward verb-first" }, { kind: "dot", cls: "c-ov", text: "k = +2, toward object-first" }, { kind: "dot", cls: "c-ink2", text: "unsteered" },
      { kind: "ring", cls: "s-ink", text: "push against the language's majority order" }, { kind: "band", cls: "band", text: "random directions, 5th–95th percentile" }]);
  },
  draw(ctx) {
    const X = ctx.D.freegen_x, M = X.match, SL = X.sign_loss, w = ctx.width(), narrow = w < 520;
    const rows = Object.keys(M).map(c => {
      const ag = SL.against[c], againstMinus = Math.abs(ag - M[c].ov_m) <= Math.abs(ag - M[c].ov_p);
      return { code: c, name: W.shortName(c), base: M[c].base, minus: M[c].ov_m, plus: M[c].ov_p, against: ag, toward: SL.toward[c], againstMinus, rand: M[c].rand };
    }).sort((a, b) => a.against - b.against);
    const rowH = 36, m = { t: 30, r: narrow ? 14 : 92, b: 40, l: narrow ? 80 : 104 }, H = m.t + rows.length * rowH + m.b;
    const svg = W.frame(ctx, H, "Share of continuations in the prompt language under the word-order direction, per language");
    const x = d3.scaleLinear().domain([0, 1]).range([m.l, w - m.r]);
    W.axisX(svg, x, H - m.b + 4, { values: narrow ? [0, .5, 1] : [0, .25, .5, .75, 1], grid: rows.length * rowH + 10, format: d3.format(".0%") });
    svg.append("text").attr("x", w - m.r).attr("y", H - 4).attr("text-anchor", "end").attr("class", "small muted").text("continuations still in the prompt language");
    if (!narrow) svg.append("text").attr("x", w).attr("y", m.t - 14).attr("text-anchor", "end").attr("class", "cap").text("against − toward");
    rows.forEach((r, i) => {
      const yy = m.t + i * rowH + rowH / 2, g = svg.append("g").attr("data-lang", W.norm(r.code));
      const lo = W.q(r.rand, .05), hi = W.q(r.rand, .95);
      if (lo != null) g.append("rect").attr("x", x(lo) - 4).attr("width", Math.max(8, x(hi) - x(lo) + 8)).attr("y", yy - 9).attr("height", 18).attr("rx", 9).attr("class", "band");
      g.append("line").attr("x1", x(Math.min(r.minus, r.plus))).attr("x2", x(Math.max(r.minus, r.plus))).attr("y1", yy).attr("y2", yy).attr("class", "s-axis").attr("stroke-width", 2);
      g.append("text").attr("x", m.l - 14).attr("y", yy + 4).attr("text-anchor", "end").attr("class", "ink").text(r.name);
      const ax = x(r.againstMinus ? r.minus : r.plus);
      g.append("circle").attr("cx", ax).attr("cy", yy).attr("r", 10).attr("class", "s-ink").attr("fill", "none").attr("stroke-width", 1.4);
      g.append("circle").attr("cx", x(r.minus)).attr("cy", yy).attr("r", 5.5).attr("class", "dot c-vo");
      g.append("circle").attr("cx", x(r.plus)).attr("cy", yy).attr("r", 5.5).attr("class", "dot c-ov");
      g.append("circle").attr("cx", x(r.base)).attr("cy", yy).attr("r", 3.6).attr("class", "dot c-ink2");
      if (i === 0) g.append("text").attr("x", ax).attr("y", yy - 15).attr("text-anchor", "middle").attr("class", "small strong halo").text("against");
      if (!narrow) g.append("text").attr("x", w).attr("y", yy + 4).attr("text-anchor", "end").attr("class", "small").text(`${W.f.pts(r.against - r.toward, 0)} pts`);
      g.append("rect").attr("class", "hit").attr("x", 0).attr("width", w).attr("y", yy - rowH / 2).attr("height", rowH)
        .on("pointermove", ev => { W.highlight([r.code]); tip(ev, r); }).on("pointerleave", () => { W.highlight(null); W.tip.hide(); });
    });
    function tip(ev, r) {
      W.tip.show(ev, { title: r.name, rows: [
        { key: { dot: "c-vo" }, v: W.f.pct(r.minus), l: `k = −2, toward verb-first${r.againstMinus ? " (against)" : " (toward)"}` },
        { key: { dot: "c-ov" }, v: W.f.pct(r.plus), l: `k = +2, toward object-first${r.againstMinus ? " (toward)" : " (against)"}` },
        { key: { dot: "c-ink2" }, v: W.f.pct(r.base), l: "unsteered" },
        { v: `${W.f.pct(W.q(r.rand, .05))}–${W.f.pct(W.q(r.rand, .95))}`, l: "random directions at ±2, 5th–95th percentile" }] });
    }
    const node = svg.node();
    W.keyNav(node, () => rows, (r, i) => { const b = node.getBoundingClientRect(); W.highlight([r.code]); tip({ clientX: b.left + m.l + 8, clientY: b.top + m.t + i * rowH }, r); }, () => { W.highlight(null); W.tip.hide(); });
  }
});
})();
