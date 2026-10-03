/* pipeline: explanatory diagram, no data. Seven steps from translated sentences to the split model, each a button with a
   small drawing; the selected step's definition appears below (notation follows the article text). */
(function () {
"use strict";
const W = window.WOA;
const STEPS = [
  { t: "Sentences", s: "the same 2,009 FLORES+ sentences in 34 languages",
    d: "Every language reads translations of the same sentences, so differences between languages are not differences in content. Languages need different numbers of tokens for the same sentence.",
    tex: "x_{i,s},\\qquad i=1,\\dots,34,\\quad s=1,\\dots,n" },
  { t: "Residual stream", s: "read after block ℓ of the model",
    d: "Each sentence runs through the model once; we keep the residual stream at every token after block ℓ (28 blocks in Qwen2.5-7B, 32 in Llama-3.1-8B).",
    tex: "h^\\ell_t(x)\\in\\mathbb R^d,\\qquad t=0,1,\\dots,T" },
  { t: "Mean pool", s: "drop the sink token, average the rest",
    d: "Position 0 is the same sink token in every input and has an unusually large norm in the middle layers, so it is left out. The sentence vector is the mean over the content tokens.",
    tex: "\\bar h^\\ell(x)=\\frac1T\\sum_{t=1}^{T}h^\\ell_t(x)" },
  { t: "Centroid", s: "average over a language's sentences",
    d: "Each language becomes one point: the mean of its sentence vectors.",
    tex: "\\mu_i^\\ell=\\frac1n\\sum_{s=1}^{n}\\bar h^\\ell(x_{i,s})" },
  { t: "Offset", s: "subtract the mean of all 34 centroids",
    d: "What is shared by all languages is removed; the offset is what makes language i different from the average language.",
    tex: "a_i^\\ell=\\mu_i^\\ell-\\frac1N\\sum_{j=1}^{N}\\mu_j^\\ell" },
  { t: "Distances", s: "cross the two FLORES halves",
    d: "Offsets estimated on the two disjoint halves of FLORES are crossed, so sampling noise cancels and the squared distances are unbiased.",
    tex: "K_{ij}=\\tfrac12\\big(\\hat a_i^{(1)\\top}M\\,\\hat a_j^{(2)}+\\hat a_j^{(1)\\top}M\\,\\hat a_i^{(2)}\\big),\\qquad D^2_{ij}=K_{ii}+K_{jj}-2K_{ij}" },
  { t: "Split model", s: "star + splits + surface covariates",
    d: "The 561 squared distances are regressed on a per-language term, one indicator per split of the languages, and covariates for script, shared tokens and length. A split's gain is the share of the remaining error it removes.",
    tex: "D^2_{ij}=u_i+u_j+\\sum_{S}\\theta_S\\,\\delta_S(i,j)+\\phi^\\top c_{ij}+\\epsilon_{ij}" }
];
const NS = "http://www.w3.org/2000/svg";

/* the seven drawings, in a 120 x 72 box; language colours follow word order (English VO, Hindi OV, German none) */
const LANG3 = [["en", "c-vo", 6], ["de", "c-nd", 7], ["hi", "c-ov", 9]];
function glyph(i, svg) {
  const g = d3.select(svg);
  const R = (x, y, w, h, cls, rx = 1.5) => g.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h).attr("rx", rx).attr("class", cls);
  const L = (x1, y1, x2, y2, cls = "s-axis", sw = 1.2) => g.append("line").attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2).attr("class", cls).attr("stroke-width", sw);
  const C = (x, y, r, cls) => g.append("circle").attr("cx", x).attr("cy", y).attr("r", r).attr("class", cls);
  const T = (x, y, s, a = "start") => g.append("text").attr("x", x).attr("y", y).attr("text-anchor", a).text(s);
  if (i === 0) {             /* three translations as token blocks: same content, different token counts */
    LANG3.forEach(([tag, cls, n], r) => {
      const y = 12 + r * 21; T(0, y + 7, tag); C(19, y + 4, 3.2, cls);
      for (let k = 0; k < n; k++) R(27 + k * (88 / 9.2), y, 88 / 9.2 - 2, 8, "band2");
    });
  } else if (i === 1) {      /* a stack of blocks, residual stream through them, layer l highlighted */
    for (let k = 0; k < 7; k++) R(28, 4 + k * 9.4, 52, 6.4, k === 3 ? "c-ink" : "band2", 1.5);
    L(54, 0, 54, 70, "s-ink2", 1.4);
    L(82, 35.8, 104, 35.8, "s-ink", 1.4); g.append("path").attr("d", "M101,32.3 L107,35.8 L101,39.3Z").attr("class", "c-ink");
    T(14, 39, "ℓ", "middle").attr("class", "strong");
  } else if (i === 2) {      /* token vectors; the sink is dropped; the rest average into one vector */
    const shade = (a, b) => ((a * 7 + b * 3) % 5);
    for (let t = 0; t < 6; t++) for (let k = 0; k < 6; k++) R(4 + t * 13, 6 + k * 8, 10, 6.5, t === 0 ? "band" : (shade(t, k) > 2 ? "band2" : "c-rand"), 1);
    L(2, 4, 16, 54, "s-ink2", 1.2); L(16, 4, 2, 54, "s-ink2", 1.2);
    g.append("path").attr("d", "M18,58 v4 h60 v-4").attr("class", "hair");
    L(48, 62, 48, 66, "s-axis"); L(48, 66, 96, 66, "s-axis"); L(96, 66, 96, 54, "s-axis");
    for (let k = 0; k < 6; k++) R(91, 6 + k * 8, 10, 6.5, k % 2 ? "c-ink2" : "c-muted", 1);
    T(108, 30, "h̄");
  } else if (i === 3) {      /* clouds of sentence vectors and their centroids */
    const cen = [[30, 22], [86, 18], [62, 54]];
    const jit = [[-9, -5], [7, -8], [10, 4], [-4, 9], [-12, 4], [3, -1], [12, -3], [-6, -10], [5, 10]];
    cen.forEach(([cx, cy], k) => {
      jit.forEach(([dx, dy], j) => C(cx + dx * (1 + (j % 3) * .15), cy + dy * .9, 1.7, LANG3[k][1]).attr("opacity", .45));
      C(cx, cy, 4.2, "dot " + LANG3[k][1]);
    });
  } else if (i === 4) {      /* offsets: arrows from the grand mean to each centroid */
    const cen = [[30, 22], [86, 18], [62, 54]], m = [59, 31];
    cen.forEach(([cx, cy], k) => {
      const a = Math.atan2(cy - m[1], cx - m[0]), d = Math.hypot(cx - m[0], cy - m[1]) - 6;
      const ex = m[0] + d * Math.cos(a), ey = m[1] + d * Math.sin(a);
      L(m[0], m[1], ex, ey, "s-ink2", 1.3);
      g.append("path").attr("d", `M${ex + 4 * Math.cos(a)},${ey + 4 * Math.sin(a)} L${ex - 3 * Math.cos(a) + 3 * Math.sin(a)},${ey - 3 * Math.sin(a) - 3 * Math.cos(a)} L${ex - 3 * Math.cos(a) - 3 * Math.sin(a)},${ey - 3 * Math.sin(a) + 3 * Math.cos(a)}Z`).attr("class", "c-ink2");
      C(cx, cy, 4.2, "dot " + LANG3[k][1]);
    });
    L(m[0] - 4, m[1] - 4, m[0] + 4, m[1] + 4, "s-ink", 1.5); L(m[0] - 4, m[1] + 4, m[0] + 4, m[1] - 4, "s-ink", 1.5);
  } else if (i === 5) {      /* a symmetric distance matrix with an empty diagonal */
    const n = 6, s = 10.5, x0 = 28, y0 = 4;
    const v = [[0, 2, 4, 3, 4, 3], [2, 0, 4, 3, 4, 4], [4, 4, 0, 1, 3, 3], [3, 3, 1, 0, 3, 2], [4, 4, 3, 3, 0, 1], [3, 4, 3, 2, 1, 0]];
    const cls = ["band", "band2", "c-rand", "c-rand-strong", "c-muted"];
    for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) R(x0 + b * s, y0 + a * s, s - 1.2, s - 1.2, a === b ? "band" : (cls[v[a][b]] || "c-muted"), 1);
    T(18, 37, "D²", "middle");
  } else {                   /* D2 = star + split + ...: three small indicator patterns */
    const s = 5.4, n = 6;
    const mat = (x0, f) => { for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) R(x0 + b * s, 8 + a * s, s - .9, s - .9, a === b ? "band" : f(a, b), .7); };
    mat(0, (a, b) => (a === 1 || b === 1 ? "c-ink2" : "band2"));
    mat(44, (a, b) => ((a < 3) !== (b < 3) ? "c-gen" : "band2"));
    mat(88, (a, b) => ((a % 2 === 0) !== (b % 2 === 0) ? "c-ov" : "band2"));
    T(38.5, 28, "+", "middle"); T(82.5, 28, "+", "middle");
    T(16, 54, "u", "middle"); T(60, 54, "tree", "middle"); T(104, 54, "OV", "middle");
  }
}

W.fig("pipeline", {
  needs: [],
  layout: "wide",
  init(ctx) { ctx.state.sel = 0; },
  draw(ctx) {
    const S = ctx.state;
    ctx.graphic.replaceChildren();
    const ol = W.el("ol", "pipe"); ol.setAttribute("aria-label", "Steps of the analysis");
    const btns = STEPS.map((st, i) => {
      const li = W.el("li"), b = W.el("button", "stage"); b.type = "button";
      b.setAttribute("aria-pressed", String(i === S.sel));
      const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", "-2 -2 124 76"); svg.setAttribute("class", "glyph"); svg.setAttribute("aria-hidden", "true");
      glyph(i, svg);
      const txt = W.el("span", "st-txt");
      txt.append(W.el("span", "st-n", String(i + 1).padStart(2, "0")), W.el("span", "st-t", st.t), W.el("span", "st-s", st.s));
      b.append(svg, txt);
      b.addEventListener("click", () => select(i));
      b.addEventListener("keydown", ev => { const d = ev.key === "ArrowRight" || ev.key === "ArrowDown" ? 1 : ev.key === "ArrowLeft" || ev.key === "ArrowUp" ? -1 : 0; if (d) { ev.preventDefault(); const j = (S.sel + d + STEPS.length) % STEPS.length; select(j); btns[j].focus(); } });
      li.appendChild(b); ol.appendChild(li); return b;
    });
    const det = W.el("div", "detail"); det.setAttribute("aria-live", "polite");
    ctx.graphic.append(ol, det);
    function select(i) {
      S.sel = i;
      btns.forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
      const st = STEPS[i];
      det.replaceChildren(W.el("span", "d-step", `Step ${i + 1} of ${STEPS.length}`), W.el("span", "d-text", st.d));
      const m = W.el("div", "d-math"); det.appendChild(m);
      if (!W.tex(m, "\\displaystyle " + st.tex, false)) m.textContent = st.tex;
    }
    select(S.sel);
  }
});
})();
