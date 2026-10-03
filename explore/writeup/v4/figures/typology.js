/* typology: gain of four WALS word-order features, alone (beyond base + family tree) and unique (beyond the other three
   as well); small multiples for the four model x geometry analyses, all on one scale. */
(function () {
"use strict";
const W = window.WOA;
const FEATS = [["OV", "Object–verb", "83A"], ["POST", "Adposition", "85A"], ["GENN", "Genitive", "86A"], ["NADJ", "Adjective", "87A"]];
const PANELS = [["qwen", "causal", "Qwen2.5-7B", "causal"], ["qwen", "lda05", "Qwen2.5-7B", "whitened"], ["llama", "causal", "Llama-3.1-8B", "causal"], ["llama", "lda05", "Llama-3.1-8B", "whitened"]];

W.fig("typology", {
  needs: ["typology_lex_qwen", "typology_lex_llama"],
  layout: "page",
  init(ctx) {
    W.ui.legend(ctx.legend, [{ kind: "dot", cls: "c-rand-strong", text: "alone: beyond base terms and family tree" }, { kind: "dot", cls: "c-ink", text: "unique: beyond the other three features too" }]);
  },
  draw(ctx) {
    const D = ctx.D, w = ctx.width();
    const cols = w >= 900 ? 4 : w >= 500 ? 2 : 1;
    const labW = w < 420 ? 104 : 112, gapX = 22, rowH = 38, ph = 28 + FEATS.length * rowH + 30, gapY = 22;
    const pw = (w - labW - (cols - 1) * gapX) / cols;
    const rows = Math.ceil(PANELS.length / cols), H = rows * ph + (rows - 1) * gapY;
    const svg = W.frame(ctx, H, "Gain of four word-order features, alone and unique, in four analyses");
    const vals = PANELS.flatMap(([m, g]) => FEATS.flatMap(([f]) => [D[`typology_lex_${m}`][g][`A3_${f}_alone`], D[`typology_lex_${m}`][g][`A3_${f}_beyond_others`]]));
    const xmax = Math.ceil(d3.max(vals) * 20) / 20, items = [];
    PANELS.forEach(([m, geo, mname, gname], k) => {
      const c = k % cols, r = Math.floor(k / cols), ox = labW + c * (pw + gapX), oy = r * (ph + gapY);
      const R = D[`typology_lex_${m}`][geo];
      const g = svg.append("g").attr("transform", `translate(${ox},${oy})`);
      const x = d3.scaleLinear().domain([0, xmax]).range([0, pw - 8]);
      const yr = i => 30 + i * rowH + rowH / 2;
      g.append("text").attr("x", 0).attr("y", 12).attr("class", "strong").text(mname);
      g.append("text").attr("x", W.textW(mname, 12, 620) + 6).attr("y", 12).attr("class", "small muted").text(gname);
      W.axisX(g, x, 30 + FEATS.length * rowH + 4, { ticks: Math.max(2, Math.round(pw / 70)), grid: FEATS.length * rowH + 6, format: d3.format(".2f") });
      if (c === 0) FEATS.forEach(([f, name, code], i) => {
        svg.append("text").attr("x", ox - 14).attr("y", oy + yr(i) - 1).attr("text-anchor", "end").attr("class", f === "OV" ? "strong" : "ink").text(name);
        svg.append("text").attr("x", ox - 14).attr("y", oy + yr(i) + 11).attr("text-anchor", "end").attr("class", "small muted").text(`order · WALS ${code}`);
      });
      const best = FEATS.reduce((a, b) => (R[`A3_${b[0]}_beyond_others`] > R[`A3_${a[0]}_beyond_others`] ? b : a))[0];
      FEATS.forEach(([f, name, code], i) => {
        const a = R[`A3_${f}_alone`], u = Math.max(0, R[`A3_${f}_beyond_others`]), yy = yr(i);
        g.append("line").attr("x1", x(u)).attr("x2", x(a)).attr("y1", yy).attr("y2", yy).attr("class", "s-axis").attr("stroke-width", 2);
        g.append("circle").attr("cx", x(a)).attr("cy", yy).attr("r", 4.5).attr("class", "dot c-rand-strong");
        g.append("circle").attr("cx", x(u)).attr("cy", yy).attr("r", 4.5).attr("class", "dot c-ink");
        if (f === best) g.append("text").attr("x", Math.max(x(u), 16)).attr("y", yy - 9).attr("text-anchor", "middle").attr("class", "small strong halo").text(u.toFixed(3));
        const cond = f === "OV" ? [R["A1_OV|POST_gain"], R["A1_OV|POST_p"], "beyond adposition order"] : f === "POST" ? [R["A2_POST|OV_gain"], R["A2_POST|OV_p"], "beyond object–verb order"] : null;
        const tip = target => W.tip.show(target, { title: `${name} order (WALS ${code}) · ${mname}, ${gname}`, rows: [
            { key: { dot: "c-rand-strong" }, v: a.toFixed(3), l: "alone" }, { key: { dot: "c-ink" }, v: R[`A3_${f}_beyond_others`].toFixed(3), l: "unique" }]
            .concat(cond ? [{ v: cond[0].toFixed(3), l: `${cond[2]} (pre-registered, p = ${W.f.p(cond[1])})` }] : []) });
        items.push({ tip, x: ox + x(a), y: oy + yy });
        g.append("rect").attr("class", "hit").attr("x", -6).attr("width", pw).attr("y", yy - rowH / 2).attr("height", rowH)
          .on("pointermove", ev => tip(ev)).on("pointerleave", () => W.tip.hide());
      });
      if (k === PANELS.length - 1 || cols === 1) g.append("text").attr("x", pw - 8).attr("y", ph - 2).attr("text-anchor", "end").attr("class", "small muted").text("gain");
    });
    const node = svg.node();
    W.keyNav(node, () => items, it => { const r = node.getBoundingClientRect(); it.tip({ x: r.left + it.x + 12, y: r.top + it.y - 10 }); }, () => W.tip.hide());
  }
});
})();
