/* within-align: cosine of the within-language word-order direction d_W with the between-language direction beta_OV at
   every layer of Qwen2.5-7B, against the 99.9% band of random directions in the span of the language centroids, with the
   Indo-European control. Two small companion panels on the same layer axis: the split-half reliability of d_W and the
   word-order gain of the OV split (where the between-language axis lives). Layer 0 is degenerate (identical tokens). */
(function () {
"use strict";
const W = window.WOA;
const BOV = ["β", ["OV"]], BIE = ["β", ["IE"]], DW = ["d", ["W"]];

W.fig("within-align", {
  needs: ["within.A_profile"],
  layout: "page",
  init(ctx) {
    W.ui.legend(ctx.legend, [{ kind: "line", cls: "s-ov", text: ["cos(d", ["W"], ", β", ["OV"], "), the between-language word-order direction"] },
      { kind: "line", cls: "s-gen", text: ["cos(d", ["W"], ", β", ["IE"], "), Indo-European control"] },
      { kind: "band", cls: "band2", text: "99.9% of random directions in the span" }]);
  },
  draw(ctx) {
    const D = ctx.D, Wn = D.within, P = Wn.A_profile, w = ctx.width();
    const rows = P.filter(r => r.layer > 0), nl = d3.max(P, r => r.layer);
    const rel = Wn.dW_info && Wn.dW_info.split_half_cos;
    const gain = D.profiles && D.profiles.qwen && (D.profiles.qwen.per_layer.lda05 || D.profiles.qwen.per_layer.euc);
    const nSmall = 1 + (rel ? 1 : 0) + (gain ? 1 : 0), row = w >= 700, gap = 28;
    const mw = w, mh = w >= 700 ? 300 : 260;
    const sw = row ? (w - (nSmall - 1) * gap) / nSmall : w, sh = 138;
    const H = mh + 30 + (row ? sh : nSmall * (sh + 22)) + 4;
    const svg = W.frame(ctx, H, "Cosine of the within-language word-order direction with the between-language direction, by layer");
    const panels = [];

    /* main panel */
    const m = { t: 30, r: 12, b: 34, l: 44 };
    const g = svg.append("g");
    const x = d3.scaleLinear().domain([0, nl]).range([m.l, mw - m.r]);
    const vals = rows.flatMap(r => [r.cos_ov, r.cos_ie, r.null_q999, -r.null_q999]);
    const y = d3.scaleLinear().domain([Math.min(-0.3, d3.min(vals)), Math.max(0.36, d3.max(vals))]).nice().range([mh - m.b, m.t]);
    g.append("rect").attr("x", x(-0.5)).attr("width", x(0.5) - x(-0.5)).attr("y", m.t).attr("height", mh - m.b - m.t).attr("class", "band").attr("opacity", .7);
    W.axisY(g, y, m.l, { ticks: 6, grid: mw - m.l - m.r, format: d => W.f.signed(d, 1).replace("+0.0", "0") });
    g.selectAll(".axis.y .tick line").filter(d => d === 0).attr("class", "zero");
    W.axisX(g, x, mh - m.b, { values: d3.range(0, nl + 1, 4), format: d3.format("d") });
    g.append("path").datum(rows).attr("class", "band2").attr("d", d3.area().x(r => x(r.layer)).y0(r => y(-r.null_q999)).y1(r => y(r.null_q999)).curve(d3.curveMonotoneX));
    g.append("line").attr("x1", m.l).attr("x2", mw - m.r).attr("y1", y(0)).attr("y2", y(0)).attr("class", "baseline");
    const A = Wn.A || P.find(r => r.layer === 14), L14 = A.layer != null ? A.layer : 14;
    g.append("line").attr("x1", x(L14)).attr("x2", x(L14)).attr("y1", m.t - 6).attr("y2", mh - m.b).attr("class", "s-ink2").attr("stroke-width", 1);
    g.append("text").attr("x", x(L14) + 5).attr("y", m.t - 10).attr("class", "small strong").text(`pre-registered: layer ${L14}`);
    [["cos_ie", "s-gen", "c-gen"], ["cos_ov", "s-ov", "c-ov"]].forEach(([k, s, c]) => {
      g.append("path").datum(rows).attr("class", "line " + s).attr("d", d3.line().x(r => x(r.layer)).y(r => y(r[k])));
      g.selectAll(null).data(rows).join("circle").attr("cx", r => x(r.layer)).attr("cy", r => y(r[k])).attr("r", r => (r.layer === L14 ? 5 : 2.6)).attr("class", r => (r.layer === L14 ? "dot " : "") + c);
    });
    /* direct labels: the peak of the alignment, the layer-14 value, the control, the band, layer 0 */
    const peak = rows.reduce((a, b) => (b.cos_ov > a.cos_ov ? b : a));
    g.append("text").attr("x", x(peak.layer)).attr("y", y(peak.cos_ov) - 10).attr("text-anchor", "middle").attr("class", "small strong halo").text(`${W.f.signed(peak.cos_ov)} at layer ${peak.layer}`);
    g.append("text").attr("x", x(L14) - 7).attr("y", y(A.cos_ov) - 9).attr("text-anchor", "end").attr("class", "small strong halo").text(`${W.f.signed(A.cos_ov)}, p = ${W.f.p(A.p)}`);
    const ieLow = rows.filter(r => r.layer >= 6 && r.layer <= 22);
    const ieMid = d3.mean(ieLow, r => r.cos_ie);
    W.subText(g.append("text").attr("x", x(L14) - 8).attr("y", y(d3.min(ieLow, r => r.cos_ie)) + 18).attr("text-anchor", "end").attr("class", "small ink halo"),
      ["Indo-European control β", ["IE"], `: about ${W.f.signed(ieMid)} in the middle layers`]);
    g.append("text").attr("x", x(1.2)).attr("y", y(-0.1)).attr("class", "small muted halo").text("random directions, 99.9%");
    g.append("text").attr("x", m.l - 34).attr("y", m.t - 10).attr("class", "small muted").text("cosine");
    g.append("text").attr("x", mw - m.r).attr("y", mh - 4).attr("text-anchor", "end").attr("class", "small muted").text("layer of Qwen2.5-7B (layer 0 omitted: identical tokens)");
    panels.push({ g, x, ox: 0, oy: 0, top: m.t, bottom: mh - m.b, main: true, marks: [["cos_ov", "c-ov", y], ["cos_ie", "c-gen", y]] });

    /* companion panels on the same layer axis */
    const small = (i, title, data, key, cls, yDom, fmt) => {
      const ox = row ? i * (sw + gap) : 0, oy = mh + 30 + (row ? 0 : i * (sh + 22));
      const mm = { t: 22, r: 10, b: 24, l: 36 };
      const gg = svg.append("g").attr("transform", `translate(${ox},${oy})`);
      const xx = d3.scaleLinear().domain([0, nl]).range([mm.l, sw - mm.r]);
      const yy = d3.scaleLinear().domain(yDom).range([sh - mm.b, mm.t]);
      W.axisY(gg, yy, mm.l, { values: yDom[0] < 0 ? [yDom[0], 0, yDom[1]] : yDom[1] > 0.5 ? [0, 0.5, 1] : yy.ticks(3), grid: sw - mm.l - mm.r, format: fmt });
      W.axisX(gg, xx, sh - mm.b, { values: d3.range(0, nl + 1, 7), format: d3.format("d") });
      W.subText(gg.append("text").attr("x", mm.l - 30).attr("y", 10).attr("class", "small strong"), title);
      gg.append("rect").attr("x", xx(L14) - 0.5).attr("width", 1).attr("y", mm.t).attr("height", sh - mm.b - mm.t).attr("class", "c-ink2").attr("opacity", .45);
      gg.append("path").datum(data).attr("class", "line " + cls).attr("d", d3.line().defined(d => d[key] != null).x(d => xx(d.layer)).y(d => yy(d[key])));
      panels.push({ g: gg, x: xx, ox, oy, top: mm.t, bottom: sh - mm.b, marks: [[key, cls.replace("s-", "c-"), yy]], data });
    };
    let si = 0;
    small(si++, ["Cosine with β", ["OV"], " inside the language span"], rows.map(r => ({ layer: r.layer, span: r.cos_ov_within_span })), "span", "s-ov", [-1, 1], d => W.f.signed(d, 0).replace("+0", "0"));
    if (rel) small(si++, ["Reliability of d", ["W"], ": split-half cosine"], rel.map((v, l) => ({ layer: l, rel: v })).filter(d => d.layer > 0), "rel", "s-ink2", [0, 1], d3.format(".1f"));
    if (gain) small(si++, ["Word-order gain of the OV split (whitened)"], gain.map(d => ({ layer: d.layer, ov: d.ov })), "ov", "s-ov", [0, Math.ceil(d3.max(gain, d => d.ov) * 20) / 20], d3.format(".2f"));

    /* synced crosshair */
    const byL = new Map(P.map(r => [r.layer, r]));
    panels.forEach(p => {
      p.hair = p.g.append("line").attr("class", "xhair").attr("y1", p.top).attr("y2", p.bottom);
      p.dots = p.marks.map(([k, c]) => p.g.append("circle").attr("r", 4.5).attr("class", "dot " + c).style("opacity", 0));
      p.hit = p.g.append("rect").attr("class", "hit").attr("x", p.x.range()[0]).attr("width", p.x.range()[1] - p.x.range()[0]).attr("y", p.top).attr("height", p.bottom - p.top);
    });
    const show = (L, ev) => {
      const r = byL.get(L);
      panels.forEach(p => {
        p.hair.attr("x1", p.x(L)).attr("x2", p.x(L)).style("opacity", 1);
        p.marks.forEach(([k, c, yy], i) => {
          const v = p.main ? (L > 0 ? r[k] : null) : (p.data.find(d => d.layer === L) || {})[k];
          p.dots[i].attr("cx", p.x(L)).attr("cy", v == null ? 0 : yy(v)).style("opacity", v == null ? 0 : 1);
        });
      });
      const rows2 = L === 0 ? [{ l: "Layer 0 is degenerate: the twins have identical tokens, so their pooled embeddings are identical." }] : [
        { key: { line: "s-ov" }, v: W.f.signed(r.cos_ov), l: ["cos(d", ["W"], ", β", ["OV"], `), p = ${W.f.p(r.p)}`] },
        { key: { line: "s-gen" }, v: W.f.signed(r.cos_ie), l: ["cos(d", ["W"], ", β", ["IE"], "), control"] },
        { v: `±${r.null_q999.toFixed(2)}`, l: "99.9% of random directions in the span" },
        { v: W.f.signed(r.cos_ov_within_span), l: ["cosine inside the span (", W.f.pct(r.span_share), " of d", ["W"], " lies there)"] }]
        .concat(rel ? [{ key: { line: "s-ink2" }, v: rel[L].toFixed(2), l: ["split-half reliability of d", ["W"]] }] : [])
        .concat(gain && gain[L] ? [{ v: gain[L].ov.toFixed(3), l: "word-order gain of the OV split" }] : []);
      W.tip.show(ev, { title: `Layer ${L} of ${nl}${L === L14 ? " · pre-registered" : ""}`, rows: rows2 });
    };
    const hide = () => { panels.forEach(p => { p.hair.style("opacity", 0); p.dots.forEach(d => d.style("opacity", 0)); }); W.tip.hide(); };
    panels.forEach(p => p.hit.on("pointermove", ev => { const [px] = d3.pointer(ev, p.g.node()); show(Math.max(0, Math.min(nl, Math.round(p.x.invert(px)))), ev); }).on("pointerleave", hide));
    const node = svg.node();
    W.keyNav(node, () => P.map(r => r.layer), L => { const b = node.getBoundingClientRect(); show(L, { clientX: b.left + x(L) + 12, clientY: b.top + y(0) - 40 }); }, hide);
  }
});
})();
