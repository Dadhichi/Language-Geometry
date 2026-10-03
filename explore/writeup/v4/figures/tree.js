/* tree: Glottolog cladogram of the 34 languages (20 internal splits), word-order leaves marked, the OV split as
   bars on the right, and an inset residual map that shows where the hovered clade or language sits. */
(function () {
"use strict";
const W = window.WOA;

function hier(D) {
  const h = d3.hierarchy(D.tree.root, d => d.children);
  h.each(n => { n.codes = n.leaves().map(l => l.data.code); });
  return h;
}

W.fig("tree", {
  needs: ["tree", "langs", "map"],
  layout: "page",
  init(ctx) {
    W.ui.legend(ctx.legend, [{ kind: "dot", cls: "c-vo", text: "verb before object" }, { kind: "dot", cls: "c-ov", text: "object before verb" },
      { kind: "dot", cls: "c-nd", text: "no dominant order" }, { kind: "bar", cls: "c-ov", text: "the word-order (OV) split" }]);
    const S = ctx.state;
    W.on("highlight", ({ codes }) => { if (S.inset) S.inset(codes); });
  },
  draw(ctx) {
    const D = ctx.D, S = ctx.state, w = ctx.width();
    const root = hier(D), leaves = root.leaves(), maxDepth = d3.max(leaves, l => l.depth);
    const side = w >= 760, rh = side ? 18 : 17.5, top = 30, treeH = top + leaves.length * rh + 8;
    const tw = side ? Math.round(w * 0.56) : w;
    const insetW = side ? w - tw - 28 : w, insetH = side ? Math.min(treeH - 40, Math.round(insetW * 0.78)) : Math.round(Math.min(320, w * 0.78));
    const H = side ? treeH : treeH + insetH + 46;
    const svg = W.frame(ctx, H, "Glottolog family tree of the 34 languages with the word-order split");
    const labW = d3.max(leaves, l => W.textW(W.lang(l.data.code).name, 12)) + 6;
    const xOV = tw - 12, xTip = xOV - 18 - labW - 12, x0 = 10;
    const step = (xTip - x0) / maxDepth;
    leaves.forEach((l, i) => { l.y = top + i * rh + rh / 2; });
    root.eachAfter(n => { if (n.children) n.y = (n.children[0].y + n.children[n.children.length - 1].y) / 2; });
    root.each(n => { n.x = n.children ? x0 + n.depth * step : xTip; });

    const g = svg.append("g");
    g.append("text").attr("x", x0).attr("y", 12).attr("class", "cap").text("Glottolog clades");
    g.append("text").attr("x", xOV + 2).attr("y", 12).attr("text-anchor", "middle").attr("class", "cap").text("OV");
    /* links: one elbow per child, tagged with the languages below it, so highlighting lights the path from the root */
    g.append("g").selectAll("path").data(root.links()).join("path").attr("class", "hair").attr("stroke-width", 1.2)
      .attr("data-lang", d => d.target.codes.join(" "))
      .attr("d", d => `M${d.source.x},${d.source.y}V${d.target.y}H${d.target.x}`);
    /* clade labels on the incoming branch, where they fit */
    const internal = root.descendants().filter(n => n.children && n.depth > 0);
    g.append("g").attr("class", "halo").selectAll("text").data(internal).join("text").attr("class", "small muted")
      .attr("x", d => d.x - 5).attr("y", d => d.y - 5).attr("text-anchor", "end").attr("data-lang", d => d.codes.join(" "))
      .text(d => d.data.name).style("display", d => (W.textW(d.data.name, 10.5) <= (d.x - d.parent.x) + 2 ? null : "none"));
    g.append("g").selectAll("circle").data(internal).join("circle").attr("cx", d => d.x).attr("cy", d => d.y).attr("r", 3.2)
      .attr("class", "c-ink2").attr("data-lang", d => d.codes.join(" "));
    /* leaves */
    const lg = g.append("g").selectAll("g").data(leaves).join("g").attr("data-lang", d => d.data.code).attr("transform", d => `translate(${xTip},${d.y})`);
    lg.append("circle").attr("r", 4.5).attr("class", d => "dot c-" + W.lang(d.data.code).order);
    lg.append("text").attr("x", 10).attr("y", 4).attr("class", "ink").text(d => W.lang(d.data.code).name);
    /* OV split: bars over runs of consecutive OV leaves */
    const runs = []; let run = null;
    leaves.forEach((l, i) => { const ov = W.lang(l.data.code).order === "ov"; if (ov) { if (!run) runs.push(run = { a: i, b: i }); else run.b = i; } else run = null; });
    g.append("g").selectAll("rect").data(runs).join("rect").attr("x", xOV).attr("width", 5).attr("rx", 2.5)
      .attr("y", r => top + r.a * rh + 3).attr("height", r => (r.b - r.a + 1) * rh - 6).attr("class", "c-ov")
      .attr("data-lang", r => leaves.slice(r.a, r.b + 1).map(l => l.data.code).join(" "));

    /* hit targets: leaf rows and clade nodes */
    const tipFor = (ev, n) => {
      if (!n.children) { W.highlight([n.data.code]); W.langTip(ev, n.data.code); return; }
      W.highlight(n.codes);
      const ov = n.codes.filter(c => W.lang(c).order === "ov").length;
      W.tip.show(ev, { title: n.data.name === "root" ? "All 34 languages" : n.data.name, rows: [{ v: String(n.codes.length), l: "languages" },
        { key: { dot: "c-ov" }, v: String(ov), l: "object before verb" }], note: n.codes.map(c => W.lang(c).name).join(", ") });
    };
    g.append("g").selectAll("rect").data(leaves).join("rect").attr("class", "hit").attr("x", xTip - 8).attr("width", xOV - xTip + 16)
      .attr("y", d => d.y - rh / 2).attr("height", rh).on("pointermove", (ev, d) => tipFor(ev, d)).on("pointerleave", () => { W.highlight(null); W.tip.hide(); });
    g.append("g").selectAll("rect").data(internal).join("rect").attr("class", "hit")
      .attr("x", d => d.x - 9).attr("width", d => Math.max(18, step * 0.5)).attr("y", d => d.children[0].y).attr("height", d => Math.max(14, d.children[d.children.length - 1].y - d.children[0].y))
      .on("pointermove", (ev, d) => tipFor(ev, d)).on("pointerleave", () => { W.highlight(null); W.tip.hide(); });
    const nodes = root.descendants().filter(n => n.depth > 0);
    const node = svg.node();
    W.keyNav(node, () => nodes, n => { const r = node.getBoundingClientRect(); tipFor({ clientX: r.left + n.x + 10, clientY: r.top + n.y + 6 }, n); }, () => { W.highlight(null); W.tip.hide(); });

    /* inset: where the highlighted languages sit on the residual map */
    const ix = side ? tw + 28 : 0, iy = side ? 34 : treeH + 36;
    const xy = W.rescale(D.map.qwen.xy, 1);
    const gi = svg.append("g");
    gi.append("rect").attr("x", ix).attr("y", iy).attr("width", insetW).attr("height", insetH).attr("rx", 6).attr("class", "band").attr("opacity", .55);
    gi.append("text").attr("x", ix + 12).attr("y", iy - 10).attr("class", "cap").text("On the residual map");
    const view = W.mapView({ svg: gi, x0: ix + 6, y0: iy + 6, w: insetW - 12, h: insetH - 30, groups: "none", labels: false, compact: true });
    view.render(xy);
    const hullG = gi.insert("g", ".mapview"), labG = gi.append("g").attr("class", "halo");
    const hint = gi.append("text").attr("x", ix + insetW / 2).attr("y", iy + insetH - 10).attr("text-anchor", "middle").attr("class", "small muted")
      .text("hover a clade or a language");
    gi.append("text").attr("x", ix + 12).attr("y", iy + insetH - 10).attr("class", "small muted").text("← VO");
    gi.append("text").attr("x", ix + insetW - 12).attr("y", iy + insetH - 10).attr("text-anchor", "end").attr("class", "small muted").text("OV →");
    S.inset = codes => {
      hullG.selectAll("*").remove(); labG.selectAll("*").remove();
      hint.style("display", codes ? "none" : null);
      if (!codes) {   /* at rest: the word-order split, the structure the tree does not contain */
        const pts = D.langs.map((l, i) => i).filter(i => D.langs[i].ov).map(i => view.screen(i));
        hullG.append("path").attr("class", "hullp").attr("d", W.hullPath(pts, 10));
        const bot = d3.max(pts, p => p[1]), mx = d3.mean(pts, p => p[0]);
        labG.append("text").attr("x", Math.min(ix + insetW - 10, mx + 40)).attr("y", Math.min(iy + insetH - 26, bot + 24)).attr("text-anchor", "end").attr("class", "small strong").text("the 11 OV languages");
        return;
      }
      const idx = D.langs.map((l, i) => i).filter(i => codes.has(D.langs[i].code));
      if (!idx.length) return;
      const pts = idx.map(i => view.screen(i));
      if (idx.length > 1) hullG.append("path").attr("class", "hullp on").attr("d", W.hullPath(pts, 9));
      if (idx.length <= 6) idx.forEach(i => { const [sx, sy] = view.screen(i); labG.append("text").attr("x", sx + 7).attr("y", sy - 6).attr("class", "small ink").text(D.langs[i].name); });
    };
    S.inset(null);
  }
});
})();
