/* within-align: cosine of the within-language word-order direction d_W with the between-language direction beta_OV at
   every layer, against the 99.9% band of random directions in the span of the language centroids, with the Indo-European
   control. A model control switches the main panel and its companions between Qwen2.5-7B (pre-registered layer 14) and
   the Llama-3.1-8B replication (pre-registered layer 16), each on its own layer axis. Companions: both models on relative
   depth (each with its own null band), the cosine inside the language span, the split-half reliability of d_W and the
   word-order gain of the OV split for the model shown. Layer 0 is degenerate in both models (identical tokens). */
(function () {
"use strict";
const W = window.WOA;
const NAMES = { qwen: "Qwen2.5-7B", llama: "Llama-3.1-8B" };

function model(D, k) {
  const src = k === "llama" ? D.within_llama : D.within;
  if (!src || !src.A_profile) return null;
  const prof = src.A_profile, nl = d3.max(prof, r => r.layer);
  const A = src.A || prof.find(r => r.layer === (src.layer || 14));
  const gainAll = D.profiles && D.profiles[k] ? D.profiles[k].per_layer : null;
  return { k, name: NAMES[k], prof, rows: prof.filter(r => r.layer > 0).map(r => Object.assign({ depth: r.layer / nl }, r)), nl, A,
    Lpre: A && A.layer != null ? A.layer : (src.layer || 14), rel: src.dW_info && src.dW_info.split_half_cos,
    gain: gainAll ? (gainAll.lda05 || gainAll.euc) : null };
}

W.fig("within-align", {
  needs: ["within.A_profile"],
  layout: "page",
  init(ctx) {
    const S = ctx.state; S.model = "qwen";
    if (ctx.D.within_llama && ctx.D.within_llama.A_profile) {
      W.ui.seg(ctx.controls, { label: "Model", options: [["qwen", "Qwen2.5-7B", "pre-registered study, layer 14"], ["llama", "Llama-3.1-8B", "pre-registered replication, layer 16"]], value: "qwen",
        onChange: v => {
          S.model = v;
          const g = ctx.graphic;
          if (!W.motion()) { ctx.redraw(); return; }
          g.style.transition = "opacity .16s"; g.style.opacity = "0";
          clearTimeout(S.t); S.t = setTimeout(() => { ctx.redraw(); g.style.opacity = "1"; }, 170);
        } });
    }
    W.ui.legend(ctx.legend, [{ kind: "line", cls: "s-ov", text: ["cos(d", ["W"], ", β", ["OV"], "), the between-language word-order direction"] },
      { kind: "line", cls: "s-gen", text: ["cos(d", ["W"], ", β", ["IE"], "), Indo-European control"] },
      { kind: "band", cls: "band2", text: "99.9% of random directions in the span" }]);
  },
  draw(ctx) {
    const D = ctx.D, w = ctx.width();
    const M = model(D, ctx.state.model) || model(D, "qwen"), other = model(D, M.k === "qwen" ? "llama" : "qwen");
    const { rows, nl, A, Lpre, rel, gain } = M;
    const both = [M].concat(other ? [other] : []);
    const cols = w >= 900 ? 4 : w >= 560 ? 2 : 1, gap = 28;
    const nSmall = (other ? 1 : 0) + 1 + (rel ? 1 : 0) + (gain ? 1 : 0);
    const mw = w, mh = w >= 700 ? 300 : 260;
    const sw = (w - (Math.min(cols, nSmall) - 1) * gap) / Math.min(cols, nSmall), sh = 138, rowsN = Math.ceil(nSmall / cols);
    const H = mh + 30 + rowsN * sh + (rowsN - 1) * 26 + 4;
    const svg = W.frame(ctx, H, `Cosine of the within-language word-order direction with the between-language direction, by layer, ${M.name}`);
    const panels = [];

    /* main panel: one y scale for both models, so switching does not rescale */
    const m = { t: 40, r: 12, b: 34, l: 44 }, narrow = w < 560;
    const g = svg.append("g");
    const x = d3.scaleLinear().domain([0, nl]).range([m.l, mw - m.r]);
    const vals = both.flatMap(b => b.rows.flatMap(r => [r.cos_ov, r.cos_ie, r.null_q999, -r.null_q999]));
    const y = d3.scaleLinear().domain([Math.min(-0.3, d3.min(vals)), Math.max(0.36, d3.max(vals))]).nice().range([mh - m.b, m.t]);
    g.append("rect").attr("x", x(-0.5)).attr("width", x(0.5) - x(-0.5)).attr("y", m.t).attr("height", mh - m.b - m.t).attr("class", "band").attr("opacity", .7);
    W.axisY(g, y, m.l, { ticks: 6, grid: mw - m.l - m.r, format: d => W.f.signed(d, 1).replace("+0.0", "0") });
    W.axisX(g, x, mh - m.b, { values: d3.range(0, nl + 1, 4), format: d3.format("d") });
    g.append("path").datum(rows).attr("class", "band2").attr("d", d3.area().x(r => x(r.layer)).y0(r => y(-r.null_q999)).y1(r => y(r.null_q999)).curve(d3.curveMonotoneX));
    g.append("line").attr("x1", m.l).attr("x2", mw - m.r).attr("y1", y(0)).attr("y2", y(0)).attr("class", "baseline");
    g.append("line").attr("x1", x(Lpre)).attr("x2", x(Lpre)).attr("y1", 20).attr("y2", mh - m.b).attr("class", "s-ink2").attr("stroke-width", 1);
    const preT = narrow ? `pre-registered: layer ${Lpre}` : `${M.name} · pre-registered: layer ${Lpre}`, preR = x(Lpre) + 5 + W.textW(preT, 10.5, 620) < mw;
    g.append("text").attr("x", preR ? x(Lpre) + 5 : x(Lpre) - 5).attr("y", 14).attr("text-anchor", preR ? "start" : "end").attr("class", "small strong").text(preT);
    [["cos_ie", "s-gen", "c-gen"], ["cos_ov", "s-ov", "c-ov"]].forEach(([k, s, c]) => {
      g.append("path").datum(rows).attr("class", "line " + s).attr("d", d3.line().x(r => x(r.layer)).y(r => y(r[k])));
      g.selectAll(null).data(rows).join("circle").attr("cx", r => x(r.layer)).attr("cy", r => y(r[k])).attr("r", r => (r.layer === Lpre ? 5 : 2.6)).attr("class", r => (r.layer === Lpre ? "dot " : "") + c);
    });
    /* direct labels: peak, pre-registered value, the control (as it is), the band */
    const peak = rows.reduce((a, b) => (b.cos_ov > a.cos_ov ? b : a));
    if (peak.layer !== Lpre) g.append("text").attr("x", x(peak.layer)).attr("y", y(peak.cos_ov) - 10).attr("text-anchor", "middle").attr("class", "small strong halo").text(`peak ${W.f.signed(peak.cos_ov)} at layer ${peak.layer}`);
    const right = peak.layer < Lpre && Lpre - peak.layer <= 4, below = narrow && right;
    g.append("text").attr("x", below ? x(Lpre) - 7 : right ? x(Lpre) + 8 : x(Lpre) - 7).attr("y", below ? y(A.cos_ov) + 17 : y(A.cos_ov) - 9)
      .attr("text-anchor", below ? "end" : right ? "start" : "end").attr("class", "small strong halo").text(`${W.f.signed(A.cos_ov)}, p = ${W.f.p(A.p)}`);
    const mid = rows.filter(r => r.depth >= 0.2 && r.depth <= 0.8), ieMid = d3.mean(mid, r => r.cos_ie);
    const ieLong = ["Indo-European control β", ["IE"], `: about ${W.f.signed(ieMid)} in the middle layers`], ieShort = ["β", ["IE"], ` about ${W.f.signed(ieMid)} (middle layers)`];
    const ieFits = W.textW(W.subPlain(ieLong), 10.5) < x(Lpre) - 8 - m.l;
    W.subText(g.append("text").attr("x", ieFits ? x(Lpre) - 8 : m.l + 4).attr("y", Math.min(mh - m.b - 6, y(d3.min(mid, r => r.cos_ie)) + 18)).attr("text-anchor", ieFits ? "end" : "start").attr("class", "small ink halo"),
      ieFits ? ieLong : ieShort);
    /* band label: the first spot whose text span clears both lines */
    const bl = "random directions, 99.9%", span = W.textW(bl, 10.5) / (x(1) - x(0)), hgt = (y.domain()[1] - y.domain()[0]) * 14 / (y.range()[0] - y.range()[1]);
    const spots = [[1.2, -0.11], [1.2, 0.11], [1.2, -0.15], [nl * 0.62, -0.17], [nl * 0.62, 0.17], [1.2, 0.15]];
    const clear = ([l0, v]) => rows.filter(r => r.layer >= l0 - 0.5 && r.layer <= l0 + span + 0.5).every(r => [r.cos_ov, r.cos_ie].every(c => c < v - 0.012 || c > v + hgt));
    const [bx, by] = spots.find(clear) || spots[0];
    g.append("text").attr("x", Math.min(x(bx), mw - m.r - W.textW(bl, 10.5) - 2)).attr("y", y(by)).attr("class", "small muted halo").text(bl);
    g.append("text").attr("x", m.l - 34).attr("y", m.t - 12).attr("class", "small muted").text("cosine");
    g.append("text").attr("x", mw - m.r).attr("y", mh - 4).attr("text-anchor", "end").attr("class", "small muted").text(`layer of ${M.name} (layer 0 omitted: identical tokens)`);
    panels.push({ g, x, top: m.t, bottom: mh - m.b, main: true, marks: [["cos_ov", "c-ov", y], ["cos_ie", "c-gen", y]] });

    /* companions */
    const place = i => ({ ox: (i % cols) * (sw + gap), oy: mh + 30 + Math.floor(i / cols) * (sh + 26) });
    const mm = { t: 22, r: 10, b: 24, l: 36 };
    let si = 0;
    if (other) {
      /* both models on relative depth, each with its own null band; the model shown is in orange */
      const { ox, oy } = place(si++), gg = svg.append("g").attr("transform", `translate(${ox},${oy})`);
      const xd = d3.scaleLinear().domain([0, 1]).range([mm.l, sw - mm.r]);
      const yd = d3.scaleLinear().domain([-0.3, 0.45]).range([sh - mm.b, mm.t + 12]);
      W.axisY(gg, yd, mm.l, { values: [-0.2, 0, 0.2, 0.4], grid: sw - mm.l - mm.r, format: d => W.f.signed(d, 1).replace("+0.0", "0") });
      W.axisX(gg, xd, sh - mm.b, { values: [0, .5, 1], format: d3.format(".0%") });
      W.subText(gg.append("text").attr("x", mm.l - 30).attr("y", 10).attr("class", "small strong"), ["Both models, relative depth: cos(d", ["W"], ", β", ["OV"], ")"]);
      [other, M].forEach(b => gg.append("path").datum(b.rows).attr("class", "band2").attr("opacity", .55)
        .attr("d", d3.area().x(r => xd(r.depth)).y0(r => yd(-r.null_q999)).y1(r => yd(r.null_q999)).curve(d3.curveMonotoneX)));
      gg.append("line").attr("x1", mm.l).attr("x2", sw - mm.r).attr("y1", yd(0)).attr("y2", yd(0)).attr("class", "baseline");
      gg.append("path").datum(other.rows).attr("class", "line s-ink2").attr("stroke-width", 1.5).attr("d", d3.line().x(r => xd(r.depth)).y(r => yd(r.cos_ov)));
      gg.append("path").datum(M.rows).attr("class", "line s-ov").attr("d", d3.line().x(r => xd(r.depth)).y(r => yd(r.cos_ov)));
      both.forEach(b => {
        const pre = b.rows.find(r => r.layer === b.Lpre);
        if (pre) gg.append("circle").attr("cx", xd(pre.depth)).attr("cy", yd(pre.cos_ov)).attr("r", 3.6).attr("class", "dot " + (b === M ? "c-ov" : "c-ink2"));
      });
      /* key: the model shown in orange, the other in grey */
      [[M, "s-ov", "strong"], [other, "s-ink2", "muted"]].forEach(([b, cls, tc], j) => {
        const kx = mm.l + 4 + j * 74, ky = mm.t + 4;
        gg.append("line").attr("x1", kx).attr("x2", kx + 14).attr("y1", ky).attr("y2", ky).attr("class", "line " + cls);
        gg.append("text").attr("x", kx + 18).attr("y", ky + 3.5).attr("class", "small " + tc).text(b.name.split("-")[0].replace("2.5", ""));
      });
      panels.push({ g: gg, x: d => xd(d / nl), top: mm.t, bottom: sh - mm.b, depthPanel: true, xd, yd, range: [mm.l, sw - mm.r] });
    }
    const small = (title, data, key, cls, yDom, fmt) => {
      const { ox, oy } = place(si++), gg = svg.append("g").attr("transform", `translate(${ox},${oy})`);
      const xx = d3.scaleLinear().domain([0, nl]).range([mm.l, sw - mm.r]);
      const yy = d3.scaleLinear().domain(yDom).range([sh - mm.b, mm.t]);
      W.axisY(gg, yy, mm.l, { values: yDom[0] < 0 ? [yDom[0], 0, yDom[1]] : yDom[1] > 0.5 ? [0, 0.5, 1] : yy.ticks(3), grid: sw - mm.l - mm.r, format: fmt });
      W.axisX(gg, xx, sh - mm.b, { values: d3.range(0, nl + 1, nl > 30 ? 8 : 7), format: d3.format("d") });
      W.subText(gg.append("text").attr("x", mm.l - 30).attr("y", 10).attr("class", "small strong"), title);
      gg.append("rect").attr("x", xx(Lpre) - 0.5).attr("width", 1).attr("y", mm.t).attr("height", sh - mm.b - mm.t).attr("class", "c-ink2").attr("opacity", .45);
      gg.append("path").datum(data).attr("class", "line " + cls).attr("d", d3.line().defined(d => d[key] != null).x(d => xx(d.layer)).y(d => yy(d[key])));
      panels.push({ g: gg, x: xx, top: mm.t, bottom: sh - mm.b, marks: [[key, cls.replace("s-", "c-"), yy]], data });
    };
    small(["Cosine with β", ["OV"], " inside the language span"], rows.map(r => ({ layer: r.layer, span: r.cos_ov_within_span })), "span", "s-ov", [-1, 1], d => W.f.signed(d, 0).replace("+0", "0"));
    if (rel) small(["Reliability of d", ["W"], ": split-half cosine"], rel.map((v, l) => ({ layer: l, rel: v })).filter(d => d.layer > 0), "rel", "s-ink2", [0, 1], d3.format(".1f"));
    if (gain) small([`Word-order gain, ${M.name.split("-")[0].replace("2.5", "")} (whitened)`], gain.map(d => ({ layer: d.layer, ov: d.ov })), "ov", "s-ov", [0, Math.ceil(d3.max(gain, d => d.ov) * 20) / 20], d3.format(".2f"));

    /* synced crosshair, keyed by the layer of the model shown */
    const byL = new Map(M.prof.map(r => [r.layer, r])), byLo = other ? new Map(other.prof.map(r => [r.layer, r])) : null;
    panels.forEach(p => {
      p.hair = p.g.append("line").attr("class", "xhair").attr("y1", p.top).attr("y2", p.bottom);
      p.dots = (p.marks || [[null, "c-ov"]]).map(([k, c]) => p.g.append("circle").attr("r", 4.5).attr("class", "dot " + c).style("opacity", 0));
      const r0 = p.range || p.x.range();
      p.hit = p.g.append("rect").attr("class", "hit").attr("x", r0[0]).attr("width", r0[1] - r0[0]).attr("y", p.top).attr("height", p.bottom - p.top);
    });
    const show = (L, ev) => {
      const r = byL.get(L);
      panels.forEach(p => {
        p.hair.attr("x1", p.x(L)).attr("x2", p.x(L)).style("opacity", 1);
        if (p.depthPanel) { p.dots[0].attr("cx", p.x(L)).attr("cy", L > 0 && r ? p.yd(r.cos_ov) : 0).style("opacity", L > 0 && r ? 1 : 0); return; }
        p.marks.forEach(([k, c, yy], i) => {
          const v = p.main ? (L > 0 && r ? r[k] : null) : (p.data.find(d => d.layer === L) || {})[k];
          p.dots[i].attr("cx", p.x(L)).attr("cy", v == null ? 0 : yy(v)).style("opacity", v == null ? 0 : 1);
        });
      });
      const oL = other ? Math.round(L / nl * other.nl) : null, ro = other && byLo.get(oL);
      const rows2 = L === 0 || !r ? [{ l: "Layer 0 is degenerate: the twins have identical tokens, so their pooled embeddings are identical." }] : [
        { key: { line: "s-ov" }, v: W.f.signed(r.cos_ov), l: ["cos(d", ["W"], ", β", ["OV"], `), p = ${W.f.p(r.p)}`] },
        { key: { line: "s-gen" }, v: W.f.signed(r.cos_ie), l: ["cos(d", ["W"], ", β", ["IE"], "), control"] },
        { v: `±${r.null_q999.toFixed(2)}`, l: "99.9% of random directions in the span" },
        { v: W.f.signed(r.cos_ov_within_span), l: ["cosine inside the span (", W.f.pct(r.span_share), " of d", ["W"], " lies there)"] }]
        .concat(rel && rel[L] != null ? [{ key: { line: "s-ink2" }, v: rel[L].toFixed(2), l: ["split-half reliability of d", ["W"]] }] : [])
        .concat(gain && gain[L] ? [{ v: gain[L].ov.toFixed(3), l: "word-order gain of the OV split" }] : [])
        .concat(ro && oL > 0 ? [{ key: { line: "s-ink2" }, v: W.f.signed(ro.cos_ov), l: [`${other.name}, layer ${oL} (same relative depth): cos(d`, ["W"], ", β", ["OV"], ")"] }] : []);
      W.tip.show(ev, { title: `${M.name}, layer ${L} of ${nl} (${W.f.pct(L / nl)} depth)${L === Lpre ? " · pre-registered" : ""}`, rows: rows2 });
    };
    const hide = () => { panels.forEach(p => { p.hair.style("opacity", 0); p.dots.forEach(d => d.style("opacity", 0)); }); W.tip.hide(); };
    panels.forEach(p => p.hit.on("pointermove", ev => {
      const [px] = d3.pointer(ev, p.g.node());
      const L = p.depthPanel ? Math.round(p.xd.invert(px) * nl) : Math.round(p.x.invert(px));
      show(Math.max(0, Math.min(nl, L)), ev);
    }).on("pointerleave", hide));
    const node = svg.node();
    W.keyNav(node, () => M.prof.map(r => r.layer), L => { const b = node.getBoundingClientRect(); show(L, { clientX: b.left + x(L) + 12, clientY: b.top + y(0) - 40 }); }, hide);
  }
});
})();
