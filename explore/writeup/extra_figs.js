/* ---------- Figure 7: typology bundle (added when explore/typology_lex results exist) ---------- */
const typoSel = document.getElementById("typo-sel") ? seg("typo-sel", v => drawTypo(v)) : null;
function drawTypo(sel) {
  const [mk, met] = sel.split("|"), R = D[`typology_lex_${mk}`][met];
  const svg = d3.select("#fig-typo"); svg.selectAll("*").remove();
  const feats = [["OV", "Object–verb order", "WALS 83A"], ["POST", "Adposition order", "WALS 85A"], ["GENN", "Genitive order", "WALS 86A"], ["NADJ", "Adjective order", "WALS 87A"]];
  const W = width(svg.node()), H = 280, m = {t: 18, r: 20, b: 46, l: 44};
  svg.attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
  const x0 = d3.scaleBand().domain(feats.map(f => f[0])).range([m.l, W - m.r]).paddingInner(.35).paddingOuter(.2);
  const ymax = 0.36;
  const y = d3.scaleLinear().domain([0, ymax]).range([H - m.b, m.t]);
  svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6).tickSize(-(W - m.l - m.r))).call(s => s.select(".domain").remove());
  svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", css("--axis"));
  const bw = Math.min(24, x0.bandwidth() / 2 - 1);
  const bar = (xx, v, col) => {
    const h = y(0) - y(v), r = Math.min(4, h / 2);
    return svg.append("path").attr("fill", col).attr("d", `M${xx},${y(0)} V${y(v) + r} Q${xx},${y(v)} ${xx + r},${y(v)} H${xx + bw - r} Q${xx + bw},${y(v)} ${xx + bw},${y(v) + r} V${y(0)} Z`);
  };
  feats.forEach(([k, name, code]) => {
    const cx = x0(k) + x0.bandwidth() / 2;
    const a = R[`A3_${k}_alone`], u = R[`A3_${k}_beyond_others`];
    [[cx - bw - 1, a, css("--axis"), "alone (beyond family tree)"], [cx + 1, u, css("--ov"), "unique (beyond the other three)"]].forEach(([xx, v, col, lab]) => {
      bar(xx, Math.max(v, 0.0005), col).on("pointermove", ev => showTip(ev, [[v.toFixed(3), lab], [null, `${name} (${code})`]])).on("pointerleave", hideTip);
    });
    svg.append("text").attr("x", cx).attr("y", H - m.b + 18).attr("text-anchor", "middle").style("fill", css("--ink")).text(name);
    svg.append("text").attr("x", cx).attr("y", H - m.b + 32).attr("text-anchor", "middle").style("fill", css("--muted")).text(code);
  });
  svg.append("text").attr("x", m.l + 4).attr("y", m.t + 2).text("gain");
}
function drawLexTable() {
  const t = document.getElementById("tab-lex"); if (!t) return; t.replaceChildren();
  const head = t.createTHead().insertRow();
  ["Model · geometry", "Genealogy gain", "+ character overlap", "p", "+ basic vocabulary (ASJP)", "p"].forEach(h => { const th = document.createElement("th"); th.textContent = h; head.append(th); });
  const body = t.createTBody();
  for (const [mk, mn] of [["qwen", "Qwen2.5-7B"], ["llama", "Llama-3.1-8B"]]) for (const [g, gn] of [["causal", "causal"], ["lda05", "whitened"]]) {
    const r = D[`typology_lex_${mk}`][g], row = body.insertRow();
    [`${mn} · ${gn}`, r.H1_gain_plainbase.toFixed(3), r["B1_H1|LEXTEXT_gain"].toFixed(3), r["B1_H1|LEXTEXT_p"].toFixed(4),
     r["B2_H1|LEXTEXT+ASJP_gain"].toFixed(3), r["B2_H1|LEXTEXT+ASJP_p"].toFixed(4)].forEach((c, i) => { const td = row.insertCell(); td.textContent = c; if (i) td.className = "n"; });
  }
}
/* ---------- Figure 8: free generation (added when explore/steer_gen results exist) ---------- */
window.drawGen = () => {
  const G = D.freegen; if (!G || !document.getElementById("fig-gen")) return;
  const ks = G.kstar, T = G.table, L = Object.keys(T.base);
  const lname = c => c === "zho_Hans" ? "Chinese" : (langs.find(l => l.code === c) || {name: c}).name;
  const key = (k) => `ov${k > 0 ? "+" : ""}${k}`;
  const rows = L.map(c => ({c, name: lname(c), base: T.base[c], plus: T[key(ks)][c], minus: T[key(-ks)][c]}))
    .filter(r => r.base.n_deps > 0).sort((a, b) => a.base.ov_rate - b.base.ov_rate);
  { const svg = d3.select("#fig-gen"); svg.selectAll("*").remove();
    const W = width(svg.node()), rowH = 30, m = {t: 28, r: 24, b: 36, l: 92}, H = m.t + m.b + rows.length * rowH;
    svg.attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
    const x = d3.scaleLinear().domain([0, 1]).range([m.l, W - m.r]);
    svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(5).tickFormat(d3.format(".0%")).tickSize(-(H - m.b - m.t))).call(s => s.select(".domain").remove());
    svg.append("text").attr("x", W - m.r).attr("y", m.t - 10).attr("text-anchor", "end").text("share of generated verb–object pairs with the object first");
    rows.forEach((r, i) => {
      const yy = m.t + rowH * i + rowH / 2, vals = [r.minus.ov_rate, r.base.ov_rate, r.plus.ov_rate].filter(v => v !== null && !Number.isNaN(v));
      svg.append("text").attr("x", m.l - 10).attr("y", yy + 4).attr("text-anchor", "end").style("fill", css("--ink")).text(r.name);
      svg.append("line").attr("x1", x(d3.min(vals))).attr("x2", x(d3.max(vals))).attr("y1", yy).attr("y2", yy).attr("stroke", css("--axis")).attr("stroke-width", 2);
      [["minus", css("--vo"), `k = −${ks} (toward verb-first)`], ["base", css("--muted"), "unsteered"], ["plus", css("--ov"), `k = +${ks} (toward object-first)`]].forEach(([k, col, lab]) => {
        const v = r[k].ov_rate; if (v === null || Number.isNaN(v)) return;
        svg.append("circle").attr("cx", x(v)).attr("cy", yy).attr("r", 5.5).attr("fill", col).attr("stroke", css("--surface")).attr("stroke-width", 2)
          .on("pointermove", ev => showTip(ev, [[null, r.name], [(100 * v).toFixed(1) + "%", lab], [String(r[k].n_deps), "verb–object pairs counted"], [(100 * r[k].match).toFixed(0) + "%", "continuations still in the prompt language"]])).on("pointerleave", hideTip);
      });
    });
  }
  { const svg = d3.select("#fig-gen-null"); svg.selectAll("*").remove();
    const P = G.primary, W = width(svg.node()), H = 110, m = {t: 26, r: 30, b: 34, l: 30};
    svg.attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
    const all = [...P.rand_slopes, P.b_ov, G.secondary.b_ie];
    const pad = (d3.max(all) - d3.min(all)) * 0.12 || 0.01;
    const x = d3.scaleLinear().domain([d3.min(all) - pad, d3.max(all) + pad]).range([m.l, W - m.r]);
    svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6).tickFormat(d3.format("+.1%")));
    P.rand_slopes.forEach(v => svg.append("circle").attr("cx", x(v)).attr("cy", H - m.b - 22).attr("r", 4.5).attr("fill", css("--rand")).attr("stroke", css("--surface")).attr("stroke-width", 2)
      .on("pointermove", ev => showTip(ev, [[(100 * v).toFixed(2) + " pts", "random direction"]])).on("pointerleave", hideTip));
    [[P.b_ov, css("--ov"), "word-order direction"], [G.secondary.b_ie, css("--gen"), "Indo-European direction"]].forEach(([v, col, lab]) => {
      svg.append("circle").attr("cx", x(v)).attr("cy", H - m.b - 22).attr("r", 6).attr("fill", col).attr("stroke", css("--surface")).attr("stroke-width", 2)
        .on("pointermove", ev => showTip(ev, [[(100 * v).toFixed(2) + " pts", lab]])).on("pointerleave", hideTip);
      svg.append("text").attr("x", x(v)).attr("y", m.t - 6).attr("text-anchor", "middle").style("fill", css("--ink")).text(lab);
    });
  }
};
window.drawExtra = () => {
  if (D.typology_lex_qwen && D.typology_lex_llama && typoSel) { drawTypo(typoSel()); drawLexTable(); }
  if (window.drawGen) window.drawGen();
};
