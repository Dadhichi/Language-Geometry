/* cs: code-switch trajectories. Position of each token's representation on the line from language A's centroid (0) to
   B's (1) around the switch, against the Bayesian posterior of B; small multiples for layers 4, 14, 24, synced crosshair. */
(function () {
"use strict";
const W = window.WOA;

W.fig("cs", {
  needs: ["belief"],
  layout: "page",
  init(ctx) {
    W.ui.legend(ctx.legend, [{ kind: "line", cls: "s-ink", text: "representation, λ (observed)" }, { kind: "line", cls: "s-rand-strong", text: "Bayesian posterior of B, whole sequence" }]);
  },
  draw(ctx) {
    const B = ctx.D.belief, w = ctx.width();
    const layers = Object.keys(B).filter(k => /^\d+$/.test(k) && B[k].observed).map(Number).sort((a, b) => a - b);
    const cols = w >= 700 ? layers.length : 1, gap = 24, ml = 34;
    const pw = cols > 1 ? (w - ml - (cols - 1) * gap) / cols : w - ml, ph = cols > 1 ? 260 : 210;
    const H = (cols > 1 ? ph : layers.length * (ph + 12)) + 22;
    const svg = W.frame(ctx, H, "Position of token representations between two languages around a code switch, layers 4, 14 and 24");
    const panels = [];
    layers.forEach((L, i) => {
      const D = B[L], ts = Object.keys(D.observed).map(Number).sort((a, b) => a - b);
      const ox = cols > 1 ? ml + i * (pw + gap) : ml, oy = cols > 1 ? 0 : i * (ph + 12), mg = { t: 26, b: 30 };
      const g = svg.append("g").attr("transform", `translate(${ox},${oy})`);
      const x = d3.scaleLinear().domain(d3.extent(ts)).range([4, pw - 6]);
      const y = d3.scaleLinear().domain([-0.08, 1.05]).range([ph - mg.b, mg.t]);
      [0, .25, .5, .75, 1].forEach(t => g.append("line").attr("x1", 0).attr("x2", pw).attr("y1", y(t)).attr("y2", y(t)).attr("class", t === 0 || t === 1 ? "zero" : "gridline"));
      if (i === 0 || cols === 1) { [[0, "A"], [.5, "0.5"], [1, "B"]].forEach(([t, s]) => g.append("text").attr("x", -8).attr("y", y(t) + 4).attr("text-anchor", "end").attr("class", s.length === 1 ? "strong" : "small muted").text(s)); }
      ts.filter(t => t % 3 === 0).forEach(t => g.append("text").attr("x", x(t)).attr("y", ph - mg.b + 16).attr("text-anchor", "middle").attr("class", "small muted").text(t > 0 ? "+" + t : t < 0 ? "−" + -t : "0"));
      g.append("line").attr("x1", x(-0.5)).attr("x2", x(-0.5)).attr("y1", mg.t - 4).attr("y2", ph - mg.b).attr("class", "s-ink2").attr("stroke-width", 1);
      g.append("text").attr("x", x(-0.5) + 5).attr("y", mg.t + 6).attr("class", "small ink2").text("switch to B");
      g.append("text").attr("x", 0).attr("y", 12).attr("class", "strong").text(`Layer ${L}`);
      [["bayes", "s-rand-strong"], ["observed", "s-ink"]].forEach(([k, cls]) => g.append("path").datum(ts).attr("class", "line " + cls).attr("d", d3.line().x(t => x(t)).y(t => y(D[k][t]))));
      ts.forEach(t => g.append("circle").attr("cx", x(t)).attr("cy", y(D.observed[t])).attr("r", 2.6).attr("class", "c-ink"));
      if (i === Math.floor((layers.length - 1) / 2) || cols === 1 && i === 0) {
        const t1 = Math.min(6, ts[ts.length - 1]), t2 = Math.min(8, ts[ts.length - 1]);
        g.append("text").attr("x", x(t1)).attr("y", y(D.observed[t1]) + 20).attr("text-anchor", "middle").attr("class", "strong halo").text("representation");
        g.append("text").attr("x", x(t2)).attr("y", y(D.bayes[t2]) + 20).attr("text-anchor", "middle").attr("class", "ink halo").text("Bayesian posterior");
      }
      const hair = g.append("line").attr("class", "xhair").attr("y1", mg.t).attr("y2", ph - mg.b);
      const dots = [g.append("circle").attr("r", 4.5).attr("class", "dot c-ink").style("opacity", 0), g.append("circle").attr("r", 4.5).attr("class", "dot c-rand-strong").style("opacity", 0)];
      const hit = g.append("rect").attr("class", "hit").attr("x", 0).attr("y", mg.t).attr("width", pw).attr("height", ph - mg.t - mg.b);
      panels.push({ L, D, ts, g, x, y, hair, dots, hit, ox, oy });
    });
    svg.append("text").attr("x", w).attr("y", H - 2).attr("text-anchor", "end").attr("class", "small muted").text("tokens relative to the first token of language B");
    const show = (t, ev, src) => {
      panels.forEach(P => { P.hair.attr("x1", P.x(t)).attr("x2", P.x(t)).style("opacity", 1);
        P.dots[0].attr("cx", P.x(t)).attr("cy", P.y(P.D.observed[t])).style("opacity", 1); P.dots[1].attr("cx", P.x(t)).attr("cy", P.y(P.D.bayes[t])).style("opacity", 1); });
      W.tip.show(ev, { title: `Layer ${src.L}, token ${t > 0 ? "+" + t : t}`, rows: [{ key: { line: "s-ink" }, v: src.D.observed[t].toFixed(2), l: "representation (0 = A, 1 = B)" }, { key: { line: "s-rand-strong" }, v: src.D.bayes[t].toFixed(2), l: "posterior of B" }] });
    };
    const hide = () => { panels.forEach(P => { P.hair.style("opacity", 0); P.dots.forEach(d => d.style("opacity", 0)); }); W.tip.hide(); };
    panels.forEach(P => P.hit.on("pointermove", ev => { const [px] = d3.pointer(ev, P.g.node()); const t = Math.max(P.ts[0], Math.min(P.ts[P.ts.length - 1], Math.round(P.x.invert(px)))); show(t, ev, P); }).on("pointerleave", hide));
    const node = svg.node(), P0 = panels[Math.floor((panels.length - 1) / 2)];
    W.keyNav(node, () => P0.ts, t => { const r = node.getBoundingClientRect(); show(t, { clientX: r.left + P0.ox + P0.x(t) + 10, clientY: r.top + P0.oy + 70 }, P0); }, hide);
  }
});
})();
