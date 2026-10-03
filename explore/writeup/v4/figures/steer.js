/* steer: dose-response of the minimal-pair preference to steering, small multiples for layers 8, 14 and 20 on one
   scale (random directions exist at layer 14), plus the pre-registered statistic: the slope b against 32 random slopes. */
(function () {
"use strict";
const W = window.WOA;
const KS = [-2, -1, 0, 1, 2];

W.fig("steer", {
  needs: ["steering.layers", "steering.rand", "steering.primary"],
  layout: "page",
  init(ctx) {
    W.ui.legend(ctx.legend, [{ kind: "line", cls: "s-ov", text: "word-order direction" }, { kind: "line", cls: "s-gen", text: "Indo-European direction (control)" },
      { kind: "thin", cls: "s-rand-strong", text: "32 random directions (layer 14, k = ±2)" }]);
  },
  draw(ctx) {
    const St = ctx.D.steering, w = ctx.width();
    const layers = Object.keys(St.layers).map(Number).sort((a, b) => a - b);
    const cols = w >= 700 ? layers.length : 1, gap = 26, ml = 44;
    const pw = cols > 1 ? (w - ml - (cols - 1) * gap) / cols : w - ml, ph = cols > 1 ? 270 : 220;
    const P = St.primary, ci = St.ci || null, r = 4.5;
    const all = P.rand_slopes.concat([P.b_ov, St.b_ie, 0]).concat(ci || []);
    const sx = d3.scaleLinear().domain(d3.extent(all)).nice().range([ml + 4, w - 20]);
    const dod = W.dodge(P.rand_slopes, sx, r), lev = d3.max(dod, d => d.lev) + 1;
    const T0 = 22, stripH = 30 + lev * (2 * r + 1) + 18 + 66, H = T0 + (cols > 1 ? ph : layers.length * (ph + 14)) + 26 + stripH;
    const svg = W.frame(ctx, H, "Change in preference for the object-first twin against steering strength, layers 8, 14 and 20");
    const vals = [];
    layers.forEach(L => ["ov", "ie"].forEach(k => KS.forEach(q => q && vals.push(St.layers[L][k][q]))));
    St.rand.forEach(r => vals.push(r["2"], r["-2"]));
    const ym = Math.ceil(d3.max(vals, Math.abs) + 0.3);
    const panels = [];
    layers.forEach((L, i) => {
      const ox = cols > 1 ? ml + i * (pw + gap) : ml, oy = T0 + (cols > 1 ? 0 : i * (ph + 14));
      const mg = { t: 26, b: 30 };
      const g = svg.append("g").attr("transform", `translate(${ox},${oy})`);
      const x = d3.scaleLinear().domain([-2.25, 2.25]).range([4, pw - 4]);
      const y = d3.scaleLinear().domain([-ym, ym]).range([ph - mg.b, mg.t]);
      const yt = y.ticks(6);
      yt.forEach(t => g.append("line").attr("x1", 0).attr("x2", pw).attr("y1", y(t)).attr("y2", y(t)).attr("class", t === 0 ? "zero" : "gridline"));
      if (i === 0 || cols === 1) yt.forEach(t => g.append("text").attr("x", -8).attr("y", y(t) + 3.5).attr("text-anchor", "end").attr("class", "small muted").text(W.f.signed(t, 0).replace("+0", "0")));
      g.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", mg.t).attr("y2", ph - mg.b).attr("class", "zero");
      KS.forEach(k => g.append("text").attr("x", x(k)).attr("y", ph - mg.b + 16).attr("text-anchor", "middle").attr("class", "small muted").text(k === 0 ? "0" : W.f.k(k)));
      g.append("text").attr("x", 0).attr("y", 12).attr("class", "strong").text(`Layer ${L}`);
      if (L === 14) g.append("text").attr("x", W.textW("Layer 14", 12, 620) + 8).attr("y", 12).attr("class", "small muted").text("pre-registered");
      if (L === 14) St.rand.forEach(r => g.append("path").attr("class", "line thin s-rand-strong").attr("opacity", .75).attr("d", d3.line()([[x(-2), y(r["-2"])], [x(0), y(0)], [x(2), y(r["2"])]])));
      const series = [["ie", "gen"], ["ov", "ov"]];
      series.forEach(([k, c]) => {
        const pts = KS.map(q => [q, q === 0 ? 0 : St.layers[L][k][q]]);
        g.append("path").attr("class", "line s-" + c).attr("d", d3.line().x(d => x(d[0])).y(d => y(d[1]))(pts));
        pts.forEach(([q, v]) => { if (q) g.append("circle").attr("cx", x(q)).attr("cy", y(v)).attr("r", 4).attr("class", "dot c-" + c); });
      });
      if (L === 14 || cols === 1 && i === 0) {
        const ov2 = St.layers[L].ov[2], ie2 = St.layers[L].ie[2];
        g.append("text").attr("x", x(2) - 10).attr("y", y(ov2) - 2).attr("text-anchor", "end").attr("class", "strong halo").text("word order");
        g.append("text").attr("x", x(2) - 10).attr("y", y(ie2) + (ie2 < ov2 ? 16 : -10)).attr("text-anchor", "end").attr("class", "ink halo").text("Indo-European");
      }
      const hair = g.append("line").attr("class", "xhair").attr("y1", mg.t).attr("y2", ph - mg.b);
      const hit = g.append("rect").attr("class", "hit").attr("x", 0).attr("y", mg.t).attr("width", pw).attr("height", ph - mg.t - mg.b);
      panels.push({ L, g, x, y, hair, hit, ox, oy });
    });
    const yLab = T0 + (cols > 1 ? ph - 2 : layers.length * (ph + 14) - 12);
    svg.append("text").attr("x", 0).attr("y", 11).attr("class", "small muted").text("Δ log-odds of the object-first twin, nats");
    const narrowCap = w < 640;
    svg.append("text").attr("x", w).attr("y", yLab + 12).attr("text-anchor", "end").attr("class", "small muted").text(narrowCap ? "steering strength k" : "steering strength k (multiples of the direction's norm)");
    const show = (k, ev, src) => {
      panels.forEach(P => P.hair.attr("x1", P.x(k)).attr("x2", P.x(k)).style("opacity", 1));
      const L = src.L, rows = [{ key: { line: "s-ov" }, v: W.f.signed(k ? St.layers[L].ov[k] : 0), l: "word-order direction" }, { key: { line: "s-gen" }, v: W.f.signed(k ? St.layers[L].ie[k] : 0), l: "Indo-European direction" }];
      if (L === 14 && Math.abs(k) === 2) { const v = St.rand.map(r => r[k]); rows.push({ key: { line: "s-rand-strong" }, v: `${W.f.signed(d3.min(v))} to ${W.f.signed(d3.max(v))}`, l: "random directions" }); }
      W.tip.show(ev, { title: `Layer ${L}, k = ${W.f.k(k)}`, rows, note: "mean change in nats, averaged over 8 languages" });
    };
    const hide = () => { panels.forEach(P => P.hair.style("opacity", 0)); W.tip.hide(); };
    panels.forEach(P => P.hit.on("pointermove", ev => { const [px] = d3.pointer(ev, P.g.node()); const k = KS.reduce((a, b) => (Math.abs(P.x(b) - px) < Math.abs(P.x(a) - px) ? b : a)); show(k, ev, P); }).on("pointerleave", hide));
    const node = svg.node(), pk = panels.find(P => P.L === 14) || panels[0];
    W.keyNav(node, () => KS, k => { const r = node.getBoundingClientRect(); show(k, { clientX: r.left + pk.ox + pk.x(k) + 10, clientY: r.top + pk.oy + 60 }, pk); }, hide);

    /* the pre-registered statistic */
    const top = H - stripH + 24;
    const base = top + 14 + lev * (2 * r + 1);
    const gs = svg.append("g");
    gs.append("text").attr("x", 0).attr("y", top - 4).attr("class", "small strong").text(w < 560 ? "Pre-registered slope b, layer 14" : "Pre-registered statistic: slope b at layer 14, against 32 random directions");
    /* verdict by the pre-registered rule: above every random slope and z > 2.33 */
    const rmax = d3.max(P.rand_slopes), holds = P.b_ov > rmax && P.z > 2.33;
    gs.append("text").attr("x", w).attr("y", top - 4).attr("text-anchor", "end").attr("class", "small strong").text(`${holds ? "claim holds" : "no claim"}: z = ${P.z.toFixed(2)}`);
    gs.append("line").attr("x1", sx(rmax)).attr("x2", sx(rmax)).attr("y1", base - 12).attr("y2", base + 8).attr("class", "s-ink2").attr("stroke-width", 1);
    gs.append("line").attr("x1", sx(0)).attr("x2", sx(0)).attr("y1", top + 4).attr("y2", base + 24).attr("class", "zero");
    dod.forEach(d => gs.append("circle").attr("cx", sx(d.v)).attr("cy", base - d.lev * (2 * r + 1)).attr("r", r).attr("class", "dot c-rand-strong")
      .on("pointermove", ev => W.tip.show(ev, { rows: [{ key: { dot: "c-rand-strong" }, v: W.f.signed(d.v), l: "random direction, same norm" }] })).on("pointerleave", () => W.tip.hide()));
    gs.append("circle").attr("cx", sx(St.b_ie)).attr("cy", base + 18).attr("r", 5.5).attr("class", "dot c-gen")
      .on("pointermove", ev => W.tip.show(ev, { rows: [{ key: { dot: "c-gen" }, v: W.f.signed(St.b_ie), l: "Indo-European direction" }] })).on("pointerleave", () => W.tip.hide());
    if (ci) { gs.append("line").attr("x1", sx(ci[0])).attr("x2", sx(ci[1])).attr("y1", base + 18).attr("y2", base + 18).attr("class", "line s-ov");
      ci.forEach(v => gs.append("line").attr("x1", sx(v)).attr("x2", sx(v)).attr("y1", base + 13).attr("y2", base + 23).attr("class", "line s-ov")); }
    gs.append("circle").attr("cx", sx(P.b_ov)).attr("cy", base + 18).attr("r", 6).attr("class", "dot c-ov")
      .on("pointermove", ev => W.tip.show(ev, { title: "Word-order direction", rows: [{ key: { dot: "c-ov" }, v: W.f.signed(P.b_ov), l: "nats per unit k" }, ci ? { v: `${W.f.signed(ci[0])} to ${W.f.signed(ci[1])}`, l: "95% bootstrap interval" } : null, { v: `z = ${P.z.toFixed(2)}`, l: `against ${P.rand_slopes.length} random directions (mean ${W.f.signed(P.rand_mean)}, sd ${P.rand_sd.toFixed(2)})` }].filter(Boolean) }))
      .on("pointerleave", () => W.tip.hide());
    const lx = sx(P.b_ov), lt = "word order", ltw = W.textW(lt, 12, 620);
    gs.append("text").attr("x", Math.min(w - ltw / 2 - 2, lx)).attr("y", base + 4).attr("text-anchor", "middle").attr("class", "strong halo").text(lt);
    const ieLeft = sx(St.b_ie) - 10 - W.textW("Indo-European", 12) >= 0;
    gs.append("text").attr("x", ieLeft ? sx(St.b_ie) - 10 : sx(St.b_ie) + 10).attr("y", base + 22).attr("text-anchor", ieLeft ? "end" : "start").attr("class", "ink halo").text("Indo-European");
    gs.append("text").attr("x", sx(d3.min(P.rand_slopes)) - 2).attr("y", base - lev * (2 * r + 1) + 2).attr("class", "small muted halo").text(`${P.rand_slopes.length} random directions`);
    W.axisX(gs, sx, base + 30, { ticks: w < 520 ? 4 : 8, format: d => W.f.signed(d, 1) });
    gs.append("text").attr("x", w).attr("y", base + 30 + 32).attr("text-anchor", "end").attr("class", "small muted").text(w < 560 ? "slope b, nats per unit k" : "slope b, nats per unit k; tick: largest random slope");
  }
});
})();
