/* ---------- Figure 3: typology bundle (added when explore/typology_lex results exist) ---------- */
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
/* ---------- Figures 7-8 and object table: free generation (explore/steer_gen; exploratory reading) ---------- */
window.drawGen = () => {
  const X = D.freegen_x; if (!X || !document.getElementById("fig-gen")) return;
  const ks = X.kstar, R = X.rates, M = X.match;
  const lname = c => c === "zho_Hans" ? "Chinese" : (langs.find(l => l.code === c) || {name: c}).name;
  const pct = v => (100 * v).toFixed(0) + "%";
  const q = (a, p) => d3.quantile(a.filter(v => v !== null).sort(d3.ascending), p);
  const rows = Object.keys(R).map(c => ({c, name: lname(c), r: R[c], m: M[c]})).sort((a, b) => a.r.base[0] - b.r.base[0]);
  { const svg = d3.select("#fig-gen"); svg.selectAll("*").remove();
    const W = width(svg.node()), rowH = 40, m = {t: 30, r: 18, b: 34, l: 128}, H = m.t + m.b + rows.length * rowH;
    svg.attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
    const x = d3.scaleLinear().domain([0, 1]).range([m.l, W - m.r]);
    svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(W < 520 ? 3 : 5).tickFormat(d3.format(".0%")).tickSize(-(H - m.b - m.t))).call(s => s.select(".domain").remove());
    svg.append("text").attr("x", W - m.r).attr("y", m.t - 12).attr("text-anchor", "end").text("object-first share of generated verb–object pairs");
    // unsteered last and smaller, so it stays visible when steering does not move a language
    const pts = [["ov-", css("--vo"), `k = −${ks} (toward verb-first)`, 5.5], ["ov+", css("--ov"), `k = +${ks} (toward object-first)`, 5.5], ["base", css("--ink2"), "unsteered", 3.5]];
    rows.forEach((d, i) => {
      const yy = m.t + rowH * i + rowH / 2;
      const lo = q(d.r.rand, .05), hi = q(d.r.rand, .95), nr = d.r.rand.filter(v => v !== null).length;
      svg.append("rect").attr("x", x(lo) - 3).attr("width", Math.max(6, x(hi) - x(lo) + 6)).attr("y", yy - 8).attr("height", 16).attr("rx", 4).attr("fill", css("--wash"))
        .on("pointermove", ev => showTip(ev, [[null, d.name], [`${pct(lo)}–${pct(hi)}`, `random directions at ±${ks}, 5th–95th percentile`], [`${nr} of 48`, "random conditions with ≥ 20 counted pairs"]])).on("pointerleave", hideTip);
      const val = {"ov-": d.r["ov-"], base: d.r.base, "ov+": d.r["ov+"]};
      const mt = {"ov-": d.m.ov_m, base: d.m.base, "ov+": d.m.ov_p};
      const def = Object.values(val).filter(v => v[0] !== null).map(v => v[0]);
      svg.append("line").attr("x1", x(d3.min(def))).attr("x2", x(d3.max(def))).attr("y1", yy).attr("y2", yy).attr("stroke", css("--axis")).attr("stroke-width", 2);
      svg.append("text").attr("x", m.l - 12).attr("y", yy - 2).attr("text-anchor", "end").style("fill", css("--ink")).text(d.name);
      const lost = ["ov-", "ov+"].filter(k => val[k][0] === null || mt[k] < .5).map(k => `${k === "ov-" ? "−" : "+"}${ks}: ${pct(mt[k])} in language`);
      if (lost.length) svg.append("text").attr("x", m.l - 12).attr("y", yy + 11).attr("text-anchor", "end").style("fill", css("--muted")).style("font-size", "10.5px").text(lost.join(" · "));
      pts.forEach(([k, col, lab, rad]) => {
        const [v, n] = val[k]; if (v === null) return;
        const hollow = mt[k] < .5;
        svg.append("circle").attr("cx", x(v)).attr("cy", yy).attr("r", rad).attr("fill", hollow ? css("--surface") : col).attr("stroke", hollow ? col : css("--surface")).attr("stroke-width", 2)
          .on("pointermove", ev => showTip(ev, [[null, d.name], [pct(v), lab], [String(n), "verb–object pairs counted"], [pct(mt[k]), "continuations still in the prompt language"]])).on("pointerleave", hideTip);
      });
    });
  }
  { const svg = d3.select("#fig-gen-null"); svg.selectAll("*").remove();
    const P = X.pooled, ci = X.b_ov_boot95, W = width(svg.node()), r = 4.5, m = {t: 14, r: 24, b: 40, l: W < 520 ? 96 : 150};
    const fmtp = d => `${d > 0 ? "+" : d < 0 ? "−" : ""}${Math.abs(100 * d).toFixed(0)}`;
    const all = [...P.rand, P.b_ov, P.b_ie, ...ci, 0];
    const pad = (d3.max(all) - d3.min(all)) * 0.06;
    const x = d3.scaleLinear().domain([d3.min(all) - pad, d3.max(all) + pad]).range([m.l, W - m.r]);
    const placed = [];                                  // dodge: random dots stack upward instead of overlapping
    [...P.rand].sort(d3.ascending).forEach(v => {
      let lev = 0; while (placed.some(p => p.lev === lev && Math.abs(x(p.v) - x(v)) < 2 * r + 1)) lev++;
      placed.push({v, lev});
    });
    const stack = (d3.max(placed, p => p.lev) + 1) * (2 * r + 1);
    const yR = m.t + stack, yI = yR + 30, yO = yI + 30, H = yO + 22 + m.b;
    svg.attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
    svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6).tickFormat(fmtp));
    svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", m.t - 4).attr("y2", H - m.b).attr("stroke", css("--axis")).attr("stroke-dasharray", "3 3");
    svg.append("text").attr("x", (m.l + W - m.r) / 2).attr("y", H - 6).attr("text-anchor", "middle").style("fill", css("--muted")).text("change in object-first share per unit k (percentage points)");
    [[yR - r, "24 random directions"], [yI, "Indo-European"], [yO, "word order"]].forEach(([yy, lab]) =>
      svg.append("text").attr("x", m.l - 12).attr("y", yy + 4).attr("text-anchor", "end").style("fill", css("--ink")).text(lab));
    placed.forEach(({v, lev}) => svg.append("circle").attr("cx", x(v)).attr("cy", yR - lev * (2 * r + 1)).attr("r", r).attr("fill", css("--rand")).attr("stroke", css("--surface")).attr("stroke-width", 1.5)
      .on("pointermove", ev => showTip(ev, [[fmtp(v) + " pts", "random direction, same norm"]])).on("pointerleave", hideTip));
    svg.append("circle").attr("cx", x(P.b_ie)).attr("cy", yI).attr("r", 6).attr("fill", css("--gen")).attr("stroke", css("--surface")).attr("stroke-width", 2)
      .on("pointermove", ev => showTip(ev, [[fmtp(P.b_ie) + " pts", "Indo-European direction (control)"]])).on("pointerleave", hideTip);
    svg.append("line").attr("x1", x(ci[0])).attr("x2", x(ci[1])).attr("y1", yO).attr("y2", yO).attr("stroke", css("--ov")).attr("stroke-width", 2);
    [ci[0], ci[1]].forEach(v => svg.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", yO - 5).attr("y2", yO + 5).attr("stroke", css("--ov")).attr("stroke-width", 2));
    svg.append("circle").attr("cx", x(P.b_ov)).attr("cy", yO).attr("r", 6).attr("fill", css("--ov")).attr("stroke", css("--surface")).attr("stroke-width", 2)
      .on("pointermove", ev => showTip(ev, [[fmtp(P.b_ov) + " pts", "word-order direction"], [`${fmtp(ci[0])} to ${fmtp(ci[1])}`, "prompt bootstrap, 95%"], [`z = ${P.z.toFixed(1)}`, "against the random directions"]])).on("pointerleave", hideTip);
  }
  const O = D.freegen_obj, t = document.getElementById("tab-gen-obj");
  if (O && t) {
    t.replaceChildren();
    const th = (row, text, span, cls) => { const e = document.createElement("th"); e.textContent = text; if (span) e.colSpan = span; if (cls) e.className = cls; row.append(e); };
    const thead = t.createTHead(), h1 = thead.insertRow(), h2 = thead.insertRow();
    th(h1, ""); th(h1, "Noun objects, share first", 3, "grp"); th(h1, "Pronoun objects, share first", 3, "grp");
    th(h2, "Language"); [`k = −${ks}`, "0", `+${ks}`, `k = −${ks}`, "0", `+${ks}`].forEach(s => th(h2, s, 0, "n"));
    const body = t.createTBody();
    Object.keys(O).sort((a, b) => R[a].base[0] - R[b].base[0]).forEach(c => {
      const o = O[c], row = body.insertRow();
      row.insertCell().textContent = lname(c);
      for (const cls of ["nom", "pron"]) for (const k of ["ov-", "base", "ov+"]) {
        const a = o[k][`${cls}_ov`] || 0, b = o[k][`${cls}_vo`] || 0, td = row.insertCell(), nn = document.createElement("span");
        td.className = "n"; td.textContent = a + b >= 10 ? pct(a / (a + b)) : "–";
        nn.className = "nn"; nn.textContent = `n = ${a + b}`; td.append(nn);
      }
    });
  }
};
window.drawExtra = () => {
  if (D.typology_lex_qwen && D.typology_lex_llama && typoSel) { drawTypo(typoSel()); drawLexTable(); }
  if (window.drawGen) window.drawGen();
};
