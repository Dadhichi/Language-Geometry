/* splits: toy explainer of the split model. A six-leaf tree whose leaves label the rows of the squared-distance
   matrix. Selecting a branch (or the word-order split) fills the cells where delta_S(i,j) = 1; hovering a cell lights
   the path between the two languages, whose branches are exactly the terms of that cell. Notation follows the text. */
(function () {
"use strict";
const W = window.WOA;
const LEAVES = [
  { id: "en", code: "eng_Latn", name: "English" }, { id: "fr", code: "fra_Latn", name: "French" }, { id: "es", code: "spa_Latn", name: "Spanish" },
  { id: "hi", code: "hin_Deva", name: "Hindi" }, { id: "tr", code: "tur_Latn", name: "Turkish" }, { id: "ja", code: "jpn_Jpan", name: "Japanese" }];
const TREE = { id: "root", children: [{ id: "IE", name: "Indo-European", children: [{ leaf: "en" }, { id: "Rom", name: "Romance", children: [{ leaf: "fr" }, { leaf: "es" }] }, { leaf: "hi" }] }, { leaf: "tr" }, { leaf: "ja" }] };
const OV = ["hi", "tr", "ja"];
const nameOf = id => LEAVES.find(l => l.id === id).name;

/* every branch is a split: the leaves below it */
function splitsOf(h) {
  const out = {};
  h.descendants().filter(n => n.parent).forEach(n => {
    const leaves = n.leaves().map(l => l.data.leaf);
    const key = n.data.leaf ? "u:" + n.data.leaf : n.data.id;
    out[key] = { key, kind: n.data.leaf ? "leaf" : "tree", node: n, set: leaves, name: n.data.leaf ? nameOf(n.data.leaf) : n.data.name,
      tex: n.data.leaf ? `u_{\\text{${n.data.leaf}}}` : `\\theta_{\\text{${n.data.id === "IE" ? "IE" : "Rom"}}}` };
  });
  out.OV = { key: "OV", kind: "ov", set: OV, name: "Word order (OV)", tex: "\\theta_{\\text{OV}}" };
  return out;
}
const sep = (S, a, b) => S.set.includes(a) !== S.set.includes(b);

W.fig("splits", {
  needs: [],
  layout: "text",
  init(ctx) {
    const S = ctx.state;
    S.h = d3.hierarchy(TREE, d => d.children); S.splits = splitsOf(S.h); S.sel = "IE"; S.hover = null;
    S.seg = W.ui.seg(ctx.controls, { label: "Split", options: [["IE", "Indo-European"], ["Rom", "Romance"], ["OV", "Word order"], ["u:hi", "Leaf: Hindi"]], value: "IE",
      onChange: v => { S.sel = v; paint(ctx); } });
    W.ui.legend(ctx.legend, [{ kind: "bar", cls: "c-gen", text: "a branch of the family tree, θ_S" }, { kind: "bar", cls: "c-ov", text: "the word-order split, θ_OV" },
      { kind: "bar", cls: "c-ink2", text: "a leaf term, u_i" }]);
  },
  draw(ctx) {
    const S = ctx.state, w = Math.min(ctx.width(), 680);
    const narrow = w < 520;
    const labelW = narrow ? 64 : 84, treeW = narrow ? 70 : Math.round(w * 0.2);
    const cs = Math.max(26, Math.min(48, Math.floor((w - treeW - labelW - 40) / 6)));
    const top = 30, H = top + 6 * cs + 8;
    const svg = W.frame(ctx, H, "Toy tree of six languages and its squared-distance matrix", { width: w });
    const xTip = 6 + treeW, xLab = xTip + 10, xOV = xLab + labelW, xM = xOV + 16;
    const leafY = id => top + LEAVES.findIndex(l => l.id === id) * cs + cs / 2;
    const maxD = d3.max(S.h.leaves(), l => l.depth);
    S.h.eachAfter(n => { n.y = n.data.leaf ? leafY(n.data.leaf) : d3.mean([n.children[0].y, n.children[n.children.length - 1].y]); });
    S.h.each(n => { n.x = n.data.leaf ? xTip : 6 + (n.depth / maxD) * treeW; });

    const gTree = svg.append("g");
    S.edges = gTree.selectAll("path.edge").data(S.h.links()).join("path").attr("class", "edge hair").attr("stroke-width", 1.6)
      .attr("d", d => `M${d.source.x},${d.source.y}V${d.target.y}H${d.target.x}`);
    /* wide invisible hit strokes so branches are easy to click */
    gTree.selectAll("path.hit").data(S.h.links()).join("path").attr("class", "hit").attr("stroke", "transparent").attr("stroke-width", 14).attr("fill", "none")
      .style("cursor", "pointer").attr("d", d => `M${d.source.x},${d.target.y}H${d.target.x}`)
      .on("click", (ev, d) => { S.sel = d.target.data.leaf ? "u:" + d.target.data.leaf : d.target.data.id; S.seg.set(S.sel); paint(ctx); })
      .on("pointermove", (ev, d) => { const sp = S.splits[d.target.data.leaf ? "u:" + d.target.data.leaf : d.target.data.id]; W.tip.show(ev, { title: sp.name, rows: [{ l: "click to select this split" }] }); })
      .on("pointerleave", () => W.tip.hide());
    gTree.selectAll("circle").data(S.h.descendants().filter(n => n.children)).join("circle").attr("cx", d => d.x).attr("cy", d => d.y).attr("r", 3).attr("class", "c-ink2");
    gTree.selectAll("text.clade").data(S.h.descendants().filter(n => n.children && n.parent)).join("text").attr("class", "clade small muted halo")
      .attr("x", d => d.x + 5).attr("y", d => d.children[0].y - 6).text(d => (narrow ? (d.data.id === "IE" ? "IE" : "Rom.") : d.data.name));
    /* leaves */
    const gl = svg.append("g").selectAll("g").data(LEAVES).join("g").attr("transform", d => `translate(${xTip},${leafY(d.id)})`).attr("data-lang", d => d.code);
    gl.append("circle").attr("r", 4.5).attr("class", d => "dot c-" + (OV.includes(d.id) ? "ov" : "vo"));
    S.leafText = gl.append("text").attr("x", 10).attr("y", 4).attr("class", "ink").text(d => d.name);
    /* the word-order split: not a branch, a bar over the OV rows */
    S.ovBar = svg.append("rect").attr("x", xOV).attr("width", 5).attr("rx", 2.5).attr("y", leafY("hi") - cs / 2 + 3).attr("height", 3 * cs - 6).attr("class", "c-ov");
    svg.append("rect").attr("class", "hit").attr("x", xOV - 6).attr("width", 17).attr("y", leafY("hi") - cs / 2).attr("height", 3 * cs).style("cursor", "pointer")
      .on("click", () => { S.sel = "OV"; S.seg.set("OV"); paint(ctx); })
      .on("pointermove", ev => W.tip.show(ev, { title: "Word-order split", rows: [{ l: "Hindi, Turkish, Japanese put the object first; click to select" }] }))
      .on("pointerleave", () => W.tip.hide());
    svg.append("text").attr("x", xOV + 2.5).attr("y", top - 10).attr("text-anchor", "middle").attr("class", "cap").text("OV");
    /* matrix */
    svg.append("text").attr("x", xM + 3 * cs).attr("y", 10).attr("text-anchor", "middle").attr("class", "cap").text(narrow ? "D²" : "squared distance D²ᵢⱼ");
    LEAVES.forEach((l, j) => svg.append("text").attr("x", xM + j * cs + cs / 2).attr("y", top - 8).attr("text-anchor", "middle").attr("class", "small muted").text(l.id));
    const cells = [];
    LEAVES.forEach((a, i) => LEAVES.forEach((b, j) => cells.push({ a: a.id, b: b.id, i, j })));
    S.cells = svg.append("g").selectAll("rect").data(cells).join("rect").attr("x", d => xM + d.j * cs + 1).attr("y", d => top + d.i * cs + 1)
      .attr("width", cs - 2).attr("height", cs - 2).attr("rx", 3)
      .on("pointermove", (ev, d) => { if (d.a !== d.b) { S.hover = d; paint(ctx); } })
      .on("pointerleave", () => { S.hover = null; paint(ctx); });
    const det = W.el("div", "detail"); det.setAttribute("aria-live", "polite");
    ctx.graphic.appendChild(det); S.det = det;
    /* keyboard: the matrix is one tab stop; arrows walk the cells above the diagonal */
    const upper = cells.filter(c => c.i < c.j);
    W.keyNav(svg.node(), () => upper, c => { S.hover = c; paint(ctx); }, () => { S.hover = null; paint(ctx); });
    paint(ctx);
  }
});

function paint(ctx) {
  const S = ctx.state, sp = S.splits[S.sel] || S.splits.IE, hv = S.hover;
  const fill = sp.kind === "ov" ? "c-ov" : sp.kind === "tree" ? "c-gen" : "c-ink2";
  S.cells.attr("class", d => (d.a === d.b ? "c-bg s-axis" : sep(sp, d.a, d.b) ? fill : "band2"))
    .attr("stroke-width", d => (d.a === d.b ? 1 : hv && ((d.a === hv.a && d.b === hv.b) || (d.a === hv.b && d.b === hv.a)) ? 2 : 0))
    .attr("stroke", d => (hv && d.a !== d.b && ((d.a === hv.a && d.b === hv.b) || (d.a === hv.b && d.b === hv.a)) ? "currentColor" : null))
    .style("color", "var(--ink)");
  /* tree: the selected split's branch is drawn in its colour; with a hovered cell, the path between the two leaves */
  S.edges.attr("class", d => {
    const key = d.target.data.leaf ? "u:" + d.target.data.leaf : d.target.data.id, e = S.splits[key];
    if (hv) return "edge line " + (sep(e, hv.a, hv.b) ? "s-ink" : "s-axis");
    return "edge line " + (key === sp.key ? (sp.kind === "tree" ? "s-gen" : "s-ink2") : "s-axis");
  }).attr("stroke-width", d => {
    const key = d.target.data.leaf ? "u:" + d.target.data.leaf : d.target.data.id;
    return hv ? (sep(S.splits[key], hv.a, hv.b) ? 2.6 : 1.4) : (key === sp.key ? 3 : 1.4);
  });
  S.ovBar.attr("opacity", hv ? (sep(S.splits.OV, hv.a, hv.b) ? 1 : .3) : (sp.kind === "ov" ? 1 : .55)).attr("width", (!hv && sp.kind === "ov") || (hv && sep(S.splits.OV, hv.a, hv.b)) ? 7 : 5);
  S.leafText.attr("class", d => ((hv ? (d.id === hv.a || d.id === hv.b) : sp.set.includes(d.id)) ? "strong" : "ink"));
  /* readout */
  const det = S.det; det.replaceChildren();
  const m = W.el("div", "d-math");
  if (hv) {
    const terms = [`u_{\\text{${hv.a}}}`, `u_{\\text{${hv.b}}}`];
    Object.values(S.splits).forEach(e => { if (e.kind !== "leaf" && sep(e, hv.a, hv.b)) terms.push(e.tex); });
    const inner = Object.values(S.splits).filter(e => e.kind === "tree" && sep(e, hv.a, hv.b)).map(e => e.name);
    det.append(W.el("span", "d-step", "Cell"), W.el("span", "d-text",
      `${nameOf(hv.a)} and ${nameOf(hv.b)}: the path between them crosses ${inner.length ? "the " + inner.join(" and ") + " branch" + (inner.length > 1 ? "es" : "") + " and " : ""}their two leaf branches${sep(S.splits.OV, hv.a, hv.b) ? "; the word-order split also separates them" : ""}.`));
    det.appendChild(m); W.tex(m, `D^2_{\\text{${hv.a}},\\text{${hv.b}}}=${terms.join("+")}+\\phi^\\top c_{\\text{${hv.a}},\\text{${hv.b}}}`, true);
  } else {
    const n = sp.set.length * (6 - sp.set.length);
    const what = sp.kind === "leaf" ? `The leaf branch of ${sp.name} separates it from every other language, so its length u adds to the ${n} cells of its row and column: the star term.`
      : sp.kind === "ov" ? `The word-order split is not a branch of the family tree. It separates the three object-first languages from the rest and adds θ_OV to the ${n} pairs with one language on each side.`
      : `The ${sp.name} branch separates {${sp.set.map(nameOf).join(", ")}} from the rest. Its weight is added to the ${n} of 15 pairs with exactly one language in the split; pairs inside or outside get nothing.`;
    det.append(W.el("span", "d-step", sp.kind === "leaf" ? "Leaf" : "Split"), W.el("span", "d-text", what));
    det.appendChild(m);
    const setTex = `\\{${sp.set.map(id => `\\text{${id}}`).join(",")}\\}`;
    W.tex(m, `S=${setTex},\\qquad \\delta_S(i,j)=\\mathbb 1\\big[\\lvert S\\cap\\{i,j\\}\\rvert=1\\big]`, true);
  }
}
})();
