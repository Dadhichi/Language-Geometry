/* proj: held-out projection of each test language on the word-order direction fitted without it (layer 14, Qwen). */
(function () {
"use strict";
const W = window.WOA;

W.fig("proj", {
  needs: ["steering.heldout_proj"],
  layout: "wide",
  init(ctx) {
    W.ui.legend(ctx.legend, [{ kind: "dot", cls: "c-vo", text: "verb before object" }, { kind: "dot", cls: "c-ov", text: "object before verb" }, { kind: "dot", cls: "c-nd", text: "no dominant order (WALS 83A)" }]);
  },
  draw(ctx) {
    const P = ctx.D.steering.heldout_proj, w = ctx.width();
    const rows = Object.entries(P).map(([code, v]) => ({ code, v, name: W.shortName(code), l: W.lang(code) })).sort((a, b) => a.v - b.v);
    const lo = Math.min(-0.1, d3.min(rows, r => r.v)), hi = Math.max(0.1, d3.max(rows, r => r.v));
    const m = { l: 14, r: 14 };
    const x = d3.scaleLinear().domain([lo - 0.12, hi + 0.12]).nice().range([m.l, w - m.r]);
    /* labels in lanes above the dots, nearest lane that is free */
    const lanes = [], size = 12;
    rows.forEach(r => {
      const tw = W.textW(r.name, size, 500) + 10, x0 = x(r.v) - tw / 2, x1 = x(r.v) + tw / 2;
      let k = 0; while ((lanes[k] || []).some(([a, b]) => x0 < b && x1 > a)) k++;
      (lanes[k] = lanes[k] || []).push([x0, x1]); r.lane = k;
    });
    const nl = lanes.length, laneH = 17, yDot = 28 + nl * laneH + 10, H = yDot + 56;
    const svg = W.frame(ctx, H, "Held-out projection of eight languages on the word-order direction");
    svg.append("rect").attr("x", x(x.domain()[0])).attr("width", x(0) - x(x.domain()[0])).attr("y", 18).attr("height", yDot - 18 + 14).attr("class", "w-vo soft");
    svg.append("rect").attr("x", x(0)).attr("width", x(x.domain()[1]) - x(0)).attr("y", 18).attr("height", yDot - 18 + 14).attr("class", "w-ov soft");
    svg.append("text").attr("x", x(0) - 8).attr("y", 12).attr("text-anchor", "end").attr("class", "small").text(w < 520 ? "← verb-first side" : "← verb-before-object side");
    svg.append("text").attr("x", x(0) + 8).attr("y", 12).attr("class", "small").text(w < 520 ? "object-first side →" : "object-before-verb side →");
    svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 18).attr("y2", yDot + 14).attr("class", "zero");
    W.axisX(svg, x, yDot + 16, { ticks: w < 520 ? 5 : 9, format: d => W.f.signed(d, 1) });
    svg.append("text").attr("x", w - m.r).attr("y", H - 2).attr("text-anchor", "end").attr("class", "small muted").text("projection, in units of the object-first minus verb-first contrast");
    const g = svg.append("g").selectAll("g").data(rows).join("g").attr("data-lang", r => W.norm(r.code));
    g.append("line").attr("x1", r => x(r.v)).attr("x2", r => x(r.v)).attr("y1", r => 28 + (nl - 1 - r.lane) * laneH + 4).attr("y2", yDot - 7).attr("class", "s-axis").attr("stroke-width", 1);
    g.append("text").attr("x", r => x(r.v)).attr("y", r => 28 + (nl - 1 - r.lane) * laneH).attr("text-anchor", "middle").attr("class", "ink halo").text(r => r.name);
    g.append("circle").attr("cx", r => x(r.v)).attr("cy", yDot).attr("r", 6).attr("class", r => "dot c-" + r.l.order);
    g.append("rect").attr("class", "hit").attr("x", r => x(r.v) - 12).attr("width", 24).attr("y", 18).attr("height", yDot - 4)
      .on("pointermove", (ev, r) => { W.highlight([r.code]); W.tip.show(ev, { title: r.name, rows: [{ key: { dot: "c-" + r.l.order }, v: W.f.signed(r.v), l: "projection, with this language left out of the fit" }, { l: `WALS 83A: ${W.orderText(r.l.order)}` }] }); })
      .on("pointerleave", () => { W.highlight(null); W.tip.hide(); });
    const node = svg.node();
    W.keyNav(node, () => rows, r => { const b = node.getBoundingClientRect(); W.highlight([r.code]); W.tip.show({ x: b.left + x(r.v) + 12, y: b.top + yDot }, { title: r.name, rows: [{ v: W.f.signed(r.v), l: "held-out projection" }] }); }, () => { W.highlight(null); W.tip.hide(); });
  }
});
})();
