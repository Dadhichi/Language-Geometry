/* depth: per-layer gain of the family tree and of the word-order split, small multiples for the two models against
   relative depth, geometry control, crosshair synchronised across the panels. */
(function () {
"use strict";
const W = window.WOA;
const MODELS = [["qwen", "Qwen2.5-7B", 28], ["llama", "Llama-3.1-8B", 32]];
const GEOM = [["lda05", "Whitened"], ["causal", "Causal"]];

W.fig("depth", {
  needs: ["profiles"],
  layout: "page",
  init(ctx) {
    const S = ctx.state; S.metric = "lda05";
    const avail = GEOM.filter(([k]) => MODELS.every(([m]) => ctx.D.profiles[m] && ctx.D.profiles[m].per_layer[k]));
    W.ui.seg(ctx.controls, { label: "Geometry", options: avail.map(([k, t]) => [k, t, k === "lda05" ? "within-language whitened, η = ½ (pre-registered)" : k === "causal" ? "causal inner product (pre-registered)" : "exploratory"]),
      value: S.metric, onChange: v => { S.metric = v; ctx.redraw(); } });
    W.ui.legend(ctx.legend, [{ kind: "line", cls: "s-gen", text: "family tree, 20 splits" }, { kind: "line", cls: "s-ov", text: "word-order split" }, { kind: "band", cls: "band2", text: "pre-registered window" }]);
  },
  draw(ctx) {
    const S = ctx.state, D = ctx.D, w = ctx.width();
    const two = w >= 620, gap = 36, pw = two ? (w - gap) / 2 : w, ph = two ? 300 : 250;
    const H = two ? ph : 2 * ph + 28;
    const svg = W.frame(ctx, H, "Gain of the family tree and the word-order split by relative depth");
    const ymax = d3.max(MODELS, ([m]) => d3.max(D.profiles[m].per_layer[S.metric], d => Math.max(d.gen, d.ov))) * 1.1;
    const panels = [];
    MODELS.forEach(([m, name, nb], pi) => {
      const ox = two ? pi * (pw + gap) : 0, oy = two ? 0 : pi * (ph + 28);
      const mg = { t: 30, r: 10, b: 34, l: 38 };
      const g = svg.append("g").attr("transform", `translate(${ox},${oy})`);
      const x = d3.scaleLinear().domain([0, 1]).range([mg.l, pw - mg.r]);
      const y = d3.scaleLinear().domain([0, ymax]).nice().range([ph - mg.b, mg.t]);
      const win = D.profiles[m].window;
      g.append("rect").attr("x", x(win[0] / nb)).attr("width", x(win[win.length - 1] / nb) - x(win[0] / nb)).attr("y", mg.t).attr("height", ph - mg.b - mg.t).attr("class", "band");
      W.axisY(g, y, mg.l, { ticks: 5, grid: pw - mg.l - mg.r, format: d3.format(".2f") });
      W.axisX(g, x, ph - mg.b, { values: [0, .25, .5, .75, 1], format: d3.format(".0%") });
      g.append("text").attr("x", mg.l).attr("y", 13).attr("class", "strong").text(name);
      g.append("text").attr("x", mg.l + W.textW(name, 12, 620) + 8).attr("y", 13).attr("class", "small muted").text(`${nb} blocks`);
      g.append("text").attr("x", pw - mg.r).attr("y", ph - 4).attr("text-anchor", "end").attr("class", "small muted").text("relative depth (layer ÷ blocks)");
      if (pi === 0 || !two) g.append("text").attr("x", mg.l - 30).attr("y", mg.t - 10).attr("class", "small muted").text("gain");
      const rows = D.profiles[m].per_layer[S.metric].map(d => Object.assign({}, d, { rd: d.layer / nb }));
      [["gen", "s-gen"], ["ov", "s-ov"]].forEach(([k, cls]) => g.append("path").datum(rows).attr("class", "line " + cls).attr("d", d3.line().x(d => x(d.rd)).y(d => y(d[k]))));
      /* direct labels (first panel; the legend carries the rest): the first spot, preferring where the series is
         furthest from the other, whose text box touches neither line */
      if (pi === 0) {
        const samples = [];
        ["gen", "ov"].forEach(k => rows.forEach((d, i) => { if (i) { const a = rows[i - 1]; for (let t = 0; t <= 1; t += .1) samples.push([x(a.rd + (d.rd - a.rd) * t), y(a[k] + (d[k] - a[k]) * t)]); } }));
        const placed = [];
        [["gen", "ov", "family tree"], ["ov", "gen", "word order"]].forEach(([k, o, text]) => {
          const tw = W.textW(text, 12, 620);
          const order = [...rows].sort((a, b) => Math.abs(b[k] - b[o]) - Math.abs(a[k] - a[o]));
          for (const d of order) {
            const pos = d[o] > d[k] ? [16, -9] : [-9, 16];
            const hit = pos.map(dy => { const cx = Math.max(mg.l + tw / 2 + 2, Math.min(pw - mg.r - tw / 2, x(d.rd))), cy = y(d[k]) + dy;
              const box = [cx - tw / 2 - 3, cy - 11, cx + tw / 2 + 3, cy + 3];
              const ok = box[1] > mg.t && box[3] < ph - mg.b && !samples.some(([sx, sy]) => sx > box[0] && sx < box[2] && sy > box[1] && sy < box[3]) && !placed.some(p => p[0] < box[2] && p[2] > box[0] && p[1] < box[3] && p[3] > box[1]);
              return ok ? { cx, cy, box } : null; }).find(Boolean);
            if (hit) { placed.push(hit.box); g.append("text").attr("x", hit.cx).attr("y", hit.cy).attr("text-anchor", "middle").attr("class", "strong halo").text(text); break; }
          }
        });
      }
      const hair = g.append("line").attr("class", "xhair").attr("y1", mg.t).attr("y2", ph - mg.b);
      const dots = [["gen", "c-gen"], ["ov", "c-ov"]].map(([k, cls]) => g.append("circle").attr("r", 4.5).attr("class", "dot " + cls).style("opacity", 0));
      const hit = g.append("rect").attr("class", "hit").attr("x", mg.l).attr("y", mg.t).attr("width", pw - mg.l - mg.r).attr("height", ph - mg.b - mg.t);
      panels.push({ m, name, nb, x, y, rows, hair, dots, hit, g, ox, oy });
    });
    const show = (rd, ev, src) => {
      panels.forEach(P => {
        const d = P.rows.reduce((a, b) => (Math.abs(b.rd - rd) < Math.abs(a.rd - rd) ? b : a));
        P.hair.attr("x1", P.x(d.rd)).attr("x2", P.x(d.rd)).style("opacity", 1);
        P.dots[0].attr("cx", P.x(d.rd)).attr("cy", P.y(d.gen)).style("opacity", 1);
        P.dots[1].attr("cx", P.x(d.rd)).attr("cy", P.y(d.ov)).style("opacity", 1);
        if (P === src) W.tip.show(ev, { title: `${P.name}, layer ${d.layer} of ${P.nb}`, rows: [
          { key: { line: "s-gen" }, v: d.gen.toFixed(3), l: `family tree (p = ${W.f.p(d.p_gen)})` },
          { key: { line: "s-ov" }, v: d.ov.toFixed(3), l: `word order (p = ${W.f.p(d.p_ov)})` }], note: "500 permutations per layer; descriptive" });
      });
    };
    const hide = () => { panels.forEach(P => { P.hair.style("opacity", 0); P.dots.forEach(c => c.style("opacity", 0)); }); W.tip.hide(); };
    panels.forEach(P => P.hit.on("pointermove", ev => { const [px] = d3.pointer(ev, P.g.node()); show(P.x.invert(px), ev, P); }).on("pointerleave", hide));
    /* keyboard: one tab stop, arrows step through the layers of the first model */
    const node = svg.node();
    W.keyNav(node, () => panels[0].rows, d => { const r = node.getBoundingClientRect(), P = panels[0]; show(d.rd, { clientX: r.left + P.ox + P.x(d.rd) + 10, clientY: r.top + P.oy + P.y(d.ov) }, P); }, hide);
  }
});
})();
