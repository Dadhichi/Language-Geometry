/* steer-langs: per-language preference for the object-first twin, unsteered and steered at k = +2 / -2 (layer 14),
   with each language's pre-registered slope. */
(function () {
"use strict";
const W = window.WOA;

W.fig("steer-langs", {
  needs: ["steering.per_language"],
  layout: "wide",
  init(ctx) {
    W.ui.legend(ctx.legend, [{ kind: "dot", cls: "c-vo", text: "k = −2, toward verb-first" }, { kind: "dot", cls: "c-ink2", text: "unsteered" }, { kind: "dot", cls: "c-ov", text: "k = +2, toward object-first" }]);
  },
  draw(ctx) {
    const St = ctx.D.steering, w = ctx.width();
    const rows = [...St.per_language].sort((a, b) => a.base - b.base);
    const slope = (St.primary && St.primary.per_language) || {};
    const narrow = w < 520, rowH = 34, m = { t: 34, r: narrow ? 8 : 70, b: 40, l: narrow ? 76 : 96 };
    const H = m.t + rows.length * rowH + m.b;
    const svg = W.frame(ctx, H, "Preference for the object-first twin per language, unsteered and steered");
    const ext = d3.extent(rows.flatMap(r => [r.base, r.plus2, r.minus2]));
    const x = d3.scaleLinear().domain([Math.min(ext[0], -1) - 1, Math.max(ext[1], 1) + 1]).nice().range([m.l, w - m.r]);
    W.axisX(svg, x, H - m.b + 4, { ticks: narrow ? 5 : 8, grid: rows.length * rowH + 8, format: d => W.f.signed(d, 0).replace("+0", "0") });
    svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", m.t - 6).attr("y2", H - m.b + 4).attr("class", "zero");
    svg.append("text").attr("x", x(0) - 8).attr("y", m.t - 14).attr("text-anchor", "end").attr("class", "small").text(narrow ? "← verb-first" : "← prefers the verb-first twin");
    svg.append("text").attr("x", x(0) + 8).attr("y", m.t - 14).attr("class", "small").text(narrow ? "object-first →" : "prefers the object-first twin →");
    svg.append("text").attr("x", w - m.r).attr("y", H - 4).attr("text-anchor", "end").attr("class", "small muted").text("log P(object-first) − log P(verb-first), nats");
    if (!narrow) svg.append("text").attr("x", w).attr("y", m.t - 14).attr("text-anchor", "end").attr("class", "small strong").text("slope b");
    rows.forEach((r, i) => {
      const yy = m.t + i * rowH + rowH / 2, g = svg.append("g").attr("data-lang", W.norm(r.code));
      g.append("text").attr("x", m.l - 12).attr("y", yy + 4).attr("text-anchor", "end").attr("class", "ink").text(r.name);
      g.append("line").attr("x1", x(r.base)).attr("x2", x(r.plus2)).attr("y1", yy).attr("y2", yy).attr("class", "line s-ov").attr("opacity", .55);
      g.append("line").attr("x1", x(r.base)).attr("x2", x(r.minus2)).attr("y1", yy).attr("y2", yy).attr("class", "line s-vo").attr("opacity", .55);
      g.append("circle").attr("cx", x(r.minus2)).attr("cy", yy).attr("r", 5.5).attr("class", "dot c-vo");
      g.append("circle").attr("cx", x(r.plus2)).attr("cy", yy).attr("r", 5.5).attr("class", "dot c-ov");
      g.append("circle").attr("cx", x(r.base)).attr("cy", yy).attr("r", 4).attr("class", "dot c-ink2");
      if (!narrow && slope[r.code] != null) g.append("text").attr("x", w).attr("y", yy + 4).attr("text-anchor", "end").attr("class", "small").text(W.f.signed(slope[r.code]));
      g.append("rect").attr("class", "hit").attr("x", 0).attr("width", w).attr("y", yy - rowH / 2).attr("height", rowH)
        .on("pointermove", ev => { W.highlight([r.code]); tip(ev, r); }).on("pointerleave", () => { W.highlight(null); W.tip.hide(); });
    });
    function tip(ev, r) {
      W.tip.show(ev, { title: `${r.name} · ${r.n} pairs`, rows: [
        { key: { dot: "c-ov" }, v: W.f.signed(r.plus2, 1), l: "k = +2, toward object-first" },
        { key: { dot: "c-ink2" }, v: W.f.signed(r.base, 1), l: "unsteered" },
        { key: { dot: "c-vo" }, v: W.f.signed(r.minus2, 1), l: "k = −2, toward verb-first" },
        slope[r.code] != null ? { v: W.f.signed(slope[r.code]), l: "slope b (nats per unit k)" } : null].filter(Boolean) });
    }
    const node = svg.node();
    W.keyNav(node, () => rows, (r, i) => { const b = node.getBoundingClientRect(); W.highlight([r.code]); tip({ clientX: b.left + x(r.plus2) + 12, clientY: b.top + m.t + i * rowH }, r); }, () => { W.highlight(null); W.tip.hide(); });
  }
});
})();
