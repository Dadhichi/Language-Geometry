/* within-retain: per language, the share of continuations still in the prompt language at the sign of k that pushes
   it against its own order (k = 2), for the within-language direction d_W (dark dot) and for beta_OV (square, blue when
   the push is toward verb-first, orange when toward object-first); unsteered for comparison. Pre-registered test C,
   with its verdict. Model control: Qwen2.5-7B (within.C) or Llama-3.1-8B (steer_llama.L_C). */
(function () {
"use strict";
const W = window.WOA;

function model(D, m) {
  if (m === "llama") { const S = D.steer_llama; return S && S.L_C ? { m, name: "Llama-3.1-8B", C: S.L_C, match: S.match || {}, Bclaim: S.L_B ? S.L_B.claim : null } : null; }
  const Wn = D.within; return Wn && Wn.C ? { m: "qwen", name: "Qwen2.5-7B", C: Wn.C, match: Wn.match || {}, Bclaim: Wn.B ? Wn.B.claim : null } : null;
}

W.fig("within-retain", {
  needs: ["within.C.per_language"],
  layout: "wide",
  init(ctx) {
    ctx.state.model = "qwen";
    if (ctx.D.steer_llama && ctx.D.steer_llama.L_C)
      W.modelSeg(ctx, [["qwen", "Qwen2.5-7B", "pre-registered study at layer 14"], ["llama", "Llama-3.1-8B", "pre-registered replication at layer 16"]]);
    W.ui.legend(ctx.legend, [{ kind: "dot", cls: "c-ink", text: ["d", ["W"], ", within-language direction"] },
      { kind: "bar", cls: "c-ov", text: ["β", ["OV"], ", pushed toward object-first (k = +2)"] }, { kind: "bar", cls: "c-vo", text: ["β", ["OV"], ", pushed toward verb-first (k = −2)"] },
      { kind: "dot", cls: "c-muted", text: "unsteered" }]);
  },
  draw(ctx) {
    const M = model(ctx.D, ctx.state.model) || model(ctx.D, "qwen"), C = M.C, PL = C.per_language, MT = M.match, w = ctx.width(), narrow = w < 560;
    const rows = Object.keys(PL).map(c => ({ c, name: W.shortName(c), ...PL[c], base: MT[c] ? MT[c].base : null }))
      .sort((a, b) => a.match_ov - b.match_ov || a.match_w - b.match_w);
    const rowH = 38, m = { t: 58, r: narrow ? 12 : 104, b: 40, l: narrow ? 92 : 128 }, H = m.t + rows.length * rowH + m.b;
    const svg = W.frame(ctx, H, `Share of continuations in the prompt language when pushed against its own order, ${M.name}`);
    const x = d3.scaleLinear().domain([0, 1]).range([m.l, w - m.r]);
    /* the verdict of test C, stated in the figure */
    const higher = C.higher != null ? `${C.higher} of ${C.n}` : "", ps = C.p_sign != null ? `sign test p = ${W.f.p(C.p_sign)}` : "";
    const verdict = C.claim ? `Test C holds: d_W keeps the language better in ${higher} (${ps})`
      : M.Bclaim === false ? `Test C not evaluated: it required test B, which did not hold. Descriptive: d_W higher in ${higher} (${ps})`
      : `Test C does not hold: d_W higher in ${higher} (${ps})`;
    const vt = svg.append("text").attr("x", 0).attr("y", 14).attr("class", C.claim ? "small strong" : "small ink");
    W.subText(vt, narrow ? [`${M.name}: `, C.claim ? "test C holds" : M.Bclaim === false ? "test C not evaluated (needs B)" : "test C does not hold", ` · higher in ${higher}`] : verdict.split("d_W").flatMap((p, i) => (i ? ["d", ["W"], p] : [p])));
    W.subText(svg.append("text").attr("x", 0).attr("y", 31).attr("class", "small muted"), [`mean ${W.f.pct(C.mean_w)} under d`, ["W"], ` against ${W.f.pct(C.mean_ov)} under β`, ["OV"]]);
    W.axisX(svg, x, H - m.b + 4, { values: narrow ? [0, .5, 1] : [0, .25, .5, .75, 1], grid: rows.length * rowH + 10, format: d3.format(".0%") });
    svg.append("text").attr("x", w - m.r).attr("y", H - 4).attr("text-anchor", "end").attr("class", "small muted").text("continuations still in the prompt language");
    if (!narrow) W.subText(svg.append("text").attr("x", w).attr("y", m.t - 10).attr("text-anchor", "end").attr("class", "small strong"), ["d", ["W"], " − β", ["OV"]]);
    rows.forEach((r, i) => {
      const yy = m.t + i * rowH + rowH / 2, g = svg.append("g").attr("data-lang", r.c), up = r.against_k > 0;
      g.append("text").attr("x", m.l - 14).attr("y", yy - 1).attr("text-anchor", "end").attr("class", "ink").text(r.name);
      g.append("text").attr("x", m.l - 14).attr("y", yy + 12).attr("text-anchor", "end").attr("class", "small muted").text(`pushed at k = ${W.f.k(r.against_k)}`);
      g.append("line").attr("x1", x(Math.min(r.match_w, r.match_ov))).attr("x2", x(Math.max(r.match_w, r.match_ov))).attr("y1", yy).attr("y2", yy).attr("class", "s-axis").attr("stroke-width", 2);
      if (r.base != null) g.append("circle").attr("cx", x(r.base)).attr("cy", yy).attr("r", 3.4).attr("class", "c-muted");
      const s = 10;
      g.append("rect").attr("x", x(r.match_ov) - s / 2).attr("y", yy - s / 2).attr("width", s).attr("height", s).attr("rx", 1.5).attr("class", "dot c-" + (up ? "ov" : "vo"));
      g.append("circle").attr("cx", x(r.match_w)).attr("cy", yy).attr("r", 5.5).attr("class", "dot c-ink");
      const d = r.match_w - r.match_ov;
      if (!narrow) g.append("text").attr("x", w).attr("y", yy + 4).attr("text-anchor", "end").attr("class", d > 0.0005 ? "small strong" : "small").text(Math.abs(d) < 0.005 ? "tie" : `${W.f.pts(d, 0)} pts`);
      if (i === 0) {
        const far = Math.abs(x(r.match_w) - x(r.match_ov)) > 70;
        g.append("text").attr("x", x(r.match_ov)).attr("y", yy - 12).attr("text-anchor", far ? "middle" : "end").attr("class", "small strong halo").text(W.f.pct(r.match_ov));
        W.subText(g.append("text").attr("x", x(r.match_w) + (far ? 0 : 6)).attr("y", yy - 12).attr("text-anchor", far ? "middle" : "start").attr("class", "small strong halo"), ["d", ["W"], ` ${W.f.pct(r.match_w)}`]);
      }
      g.append("rect").attr("class", "hit").attr("x", 0).attr("width", w).attr("y", yy - rowH / 2).attr("height", rowH)
        .on("pointermove", ev => { W.highlight([r.c]); tip(ev, r); }).on("pointerleave", () => { W.highlight(null); W.tip.hide(); });
    });
    function tip(ev, r) {
      const up = r.against_k > 0;
      W.tip.show(ev, { title: `${r.name} · ${M.name} · pushed at k = ${W.f.k(r.against_k)} (toward ${up ? "object" : "verb"}-first)`, rows: [
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
