/* data tables: primary (pre-registered tree and word-order tests), lex (lexical controls), gen-obj (object type). */
(function () {
"use strict";
const W = window.WOA;
const MODELS = [["qwen", "Qwen2.5-7B"], ["llama", "Llama-3.1-8B"]];
const GEOS = [["causal", "causal"], ["lda05", "whitened"]];
const ALPHA = 0.0125;

function head(t, groups, cols) {
  const th = t.createTHead();
  if (groups) { const r = th.insertRow(); groups.forEach(([txt, span, cls]) => { const c = document.createElement("th"); c.textContent = txt; if (span > 1) c.colSpan = span; c.className = cls || (txt ? "grp" : ""); r.appendChild(c); }); }
  const r = th.insertRow(); cols.forEach(([txt, cls]) => { const c = document.createElement("th"); c.textContent = txt; if (cls) c.className = cls; r.appendChild(c); });
}
function cell(row, text, cls, sub) {
  const td = row.insertCell(); td.textContent = text; if (cls) td.className = cls;
  if (sub) td.appendChild(sub);
  return td;
}
function bar(v, max, cls) { const s = W.el("span", "cell-bar " + (cls || "")); s.style.width = Math.max(1, Math.round(44 * v / max)) + "px"; s.setAttribute("aria-hidden", "true"); return s; }
function pCell(row, p) { const td = cell(row, W.f.p(p), "n"); td.classList.add(p != null && p <= ALPHA ? "pass" : "fail"); td.title = p <= ALPHA ? "below the Bonferroni threshold 0.0125" : "above the Bonferroni threshold 0.0125"; return td; }
function firstCell(row, mname, gname) { const td = row.insertCell(); td.append(W.el("span", "mdl", mname), W.el("span", "geo", gname)); return td; }

W.table("primary", (t, { D }) => {
  head(t, [["", 1, ""], ["Family tree, 20 splits", 3], ["Word-order split", 3], ["", 1, ""]],
    [["Model, geometry"], ["gain", "n"], ["p", "n"], ["p | word order", "n"], ["gain", "n"], ["p", "n"], ["p | family tree", "n"], ["same-language p", "n"]]);
  const b = t.createTBody();
  const gmax = d3.max(MODELS.flatMap(([m]) => GEOS.map(([g]) => Math.max(D.profiles[m].primary[g].H1_gain, D.profiles[m].primary[g].H2_gain))));
  MODELS.forEach(([m, mn]) => GEOS.forEach(([g, gn]) => {
    const p = D.profiles[m].primary[g], r = b.insertRow();
    firstCell(r, mn, gn);
    cell(r, p.H1_gain.toFixed(3), "n").prepend(bar(p.H1_gain, gmax, "gen"));
    pCell(r, p.H1_p); pCell(r, p["H1|OV_p"]);
    cell(r, p.H2_gain.toFixed(3), "n").prepend(bar(p.H2_gain, gmax, "ov"));
    pCell(r, p.H2_p); pCell(r, p["OV|H1_p"]);
    pCell(r, p.H3_p);
  }));
  t.querySelectorAll("tbody td.n .cell-bar").forEach(s => { s.style.marginLeft = "0"; s.style.marginRight = ".5rem"; });
});

W.table("lex", (t, { D }) => {
  head(t, [["", 1, ""], ["", 1, ""], ["+ character overlap", 2], ["+ basic vocabulary (ASJP)", 2]],
    [["Model, geometry"], ["family-tree gain", "n"], ["gain", "n"], ["p", "n"], ["gain", "n"], ["p", "n"]]);
  const b = t.createTBody();
  MODELS.forEach(([m, mn]) => GEOS.forEach(([g, gn]) => {
    const R = D[`typology_lex_${m}`] && D[`typology_lex_${m}`][g]; if (!R) return;
    const r = b.insertRow(), base = R.H1_gain_plainbase;
    firstCell(r, mn, gn);
    cell(r, base.toFixed(3), "n");
    const a = R["B1_H1|LEXTEXT_gain"], c = R["B2_H1|LEXTEXT+ASJP_gain"];
    const tdA = cell(r, a.toFixed(3), "n"); tdA.appendChild(W.el("span", "sub", `${W.f.pct(a / base)} kept`));
    pCell(r, R["B1_H1|LEXTEXT_p"]);
    const tdC = cell(r, c.toFixed(3), "n"); tdC.appendChild(W.el("span", "sub", `${W.f.pct(c / base)} kept`));
    pCell(r, R["B2_H1|LEXTEXT+ASJP_p"]);
  }));
});

W.table("gen-obj", (t, { D }) => {
  const O = D.freegen_obj, X = D.freegen_x, ks = (X && X.kstar) || 2;
  head(t, [["", 1, ""], ["Noun objects, share before the verb", 3], ["Pronoun objects, share before the verb", 3]],
    [["Language"], [`k = −${ks}`, "n k"], ["k = 0", "n k"], [`k = +${ks}`, "n k"], [`k = −${ks}`, "n k"], ["k = 0", "n k"], [`k = +${ks}`, "n k"]]);
  const b = t.createTBody();
  const baseOf = c => (X && X.rates[c] ? X.rates[c].base[0] : 0);
  Object.keys(O).sort((a, c) => baseOf(a) - baseOf(c)).forEach(c => {
    const o = O[c], r = b.insertRow(); r.setAttribute("data-lang", W.norm(c));
    r.addEventListener("pointerenter", () => W.highlight([c])); r.addEventListener("pointerleave", () => W.highlight(null));
    cell(r, W.shortName(c));
    ["nom", "pron"].forEach(cls => ["ov-", "base", "ov+"].forEach(k => {
      const e = o[k] || {}, a = e[`${cls}_ov`] || 0, v = e[`${cls}_vo`] || 0, n = a + v;
      cell(r, n >= 10 ? W.f.pct(a / n) : "–", "n", W.el("span", "sub", `n = ${n}`));
    }));
  });
});
})();
