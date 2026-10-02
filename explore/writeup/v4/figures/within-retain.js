/* within-retain: per language, the share of continuations still in the prompt language at the sign of k that pushes
   it against its own order (k = 2), for the within-language direction d_W (dark dot) and for beta_OV (square, blue when
   the push is toward verb-first, orange when toward object-first); unsteered for comparison. Pre-registered test C. */
(function () {
"use strict";
const W = window.WOA;

W.fig("within-retain", {
  needs: ["within.C.per_language"],
  layout: "wide",
  init(ctx) {
    W.ui.legend(ctx.legend, [{ kind: "dot", cls: "c-ink", text: ["d", ["W"], ", within-language direction"] },
      { kind: "bar", cls: "c-ov", text: ["β", ["OV"], ", pushed toward object-first (k = +2)"] }, { kind: "bar", cls: "c-vo", text: ["β", ["OV"], ", pushed toward verb-first (k = −2)"] },
      { kind: "dot", cls: "c-muted", text: "unsteered" }]);
  },
  draw(ctx) {
    const Wn = ctx.D.within, C = Wn.C, PL = C.per_language, M = Wn.match || {}, w = ctx.width(), narrow = w < 560;
    const rows = Object.keys(PL).map(c => ({ c, name: W.shortName(c), ...PL[c], base: M[c] ? M[c].base : null }))
      .sort((a, b) => a.match_ov - b.match_ov);
    const rowH = 38, m = { t: 32, r: narrow ? 12 : 104, b: 40, l: narrow ? 92 : 128 }, H = m.t + rows.length * rowH + m.b;
    const svg = W.frame(ctx, H, "Share of continuations in the prompt language when pushed against its own order, within-language direction against beta_OV");
    const x = d3.scaleLinear().domain([0, 1]).range([m.l, w - m.r]);
    W.axisX(svg, x, H - m.b + 4, { values: narrow ? [0, .5, 1] : [0, .25, .5, .75, 1], grid: rows.length * rowH + 10, format: d3.format(".0%") });
    svg.append("text").attr("x", w - m.r).attr("y", H - 4).attr("text-anchor", "end").attr("class", "small muted").text("continuations still in the prompt language");
    if (!narrow) W.subText(svg.append("text").attr("x", w).attr("y", m.t - 14).attr("text-anchor", "end").attr("class", "small strong"), ["d", ["W"], " − β", ["OV"]]);
    rows.forEach((r, i) => {
      const yy = m.t + i * rowH + rowH / 2, g = svg.append("g").attr("data-lang", r.c), up = r.against_k > 0;
      g.append("text").attr("x", m.l - 14).attr("y", yy - 1).attr("text-anchor", "end").attr("class", "ink").text(r.name);
      g.append("text").attr("x", m.l - 14).attr("y", yy + 12).attr("text-anchor", "end").attr("class", "small muted").text(`pushed at k = ${W.f.k(r.against_k)}`);
      g.append("line").attr("x1", x(Math.min(r.match_w, r.match_ov))).attr("x2", x(Math.max(r.match_w, r.match_ov))).attr("y1", yy).attr("y2", yy).attr("class", "s-axis").attr("stroke-width", 2);
      if (r.base != null) g.append("circle").attr("cx", x(r.base)).attr("cy", yy).attr("r", 3.4).attr("class", "c-muted");
      const s = 10;
      g.append("rect").attr("x", x(r.match_ov) - s / 2).attr("y", yy - s / 2).attr("width", s).attr("height", s).attr("rx", 1.5).attr("class", "dot c-" + (up ? "ov" : "vo"));
      g.append("circle").attr("cx", x(r.match_w)).attr("cy", yy).attr("r", 5.5).attr("class", "dot c-ink");
      if (!narrow) g.append("text").attr("x", w).attr("y", yy + 4).attr("text-anchor", "end").attr("class", r.match_w > r.match_ov ? "small strong" : "small").text(`${W.f.pts(r.match_w - r.match_ov, 0)} pts`);
      if (i === 0) {
        g.append("text").attr("x", x(r.match_ov)).attr("y", yy - 12).attr("text-anchor", "middle").attr("class", "small strong halo").text(W.f.pct(r.match_ov));
        W.subText(g.append("text").attr("x", x(r.match_w)).attr("y", yy - 12).attr("text-anchor", "middle").attr("class", "small strong halo"), ["d", ["W"], ` ${W.f.pct(r.match_w)}`]);
      }
      g.append("rect").attr("class", "hit").attr("x", 0).attr("width", w).attr("y", yy - rowH / 2).attr("height", rowH)
        .on("pointermove", ev => { W.highlight([r.c]); tip(ev, r); }).on("pointerleave", () => { W.highlight(null); W.tip.hide(); });
    });
    function tip(ev, r) {
      const up = r.against_k > 0;
      W.tip.show(ev, { title: `${r.name} · pushed at k = ${W.f.k(r.against_k)} (toward ${up ? "object" : "verb"}-first)`, rows: [
        { key: { dot: "c-ink" }, v: W.f.pct(r.match_w), l: ["d", ["W"], ", within-language direction"] },
        { key: { dot: "c-" + (up ? "ov" : "vo") }, v: W.f.pct(r.match_ov), l: ["β", ["OV"], ", between-language direction"] },
        r.base != null ? { key: { dot: "c-muted" }, v: W.f.pct(r.base), l: "unsteered" } : null].filter(Boolean),
        note: "share of continuations that a language identifier assigns to the prompt language" });
    }
    const node = svg.node();
    W.keyNav(node, () => rows, (r, i) => { const b = node.getBoundingClientRect(); W.highlight([r.c]); tip({ clientX: b.left + x(r.match_w) + 12, clientY: b.top + m.t + i * rowH }, r); }, () => { W.highlight(null); W.tip.hide(); });
  }
});
})();
