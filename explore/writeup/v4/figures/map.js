/* map: the residual map (classical MDS of residual centroid distances), per model and per layer.
   Controls: model (morphs between the two models), outlines (family / script / none), a layer scrubber whose track is
   the per-layer gain profile in the same geometry, a "window" button (pre-registered layer window) and play. */
(function () {
"use strict";
const W = window.WOA;
const MODELS = { qwen: { name: "Qwen2.5-7B", blocks: 28 }, llama: { name: "Llama-3.1-8B", blocks: 32 } };
const KEY = new Set(["eng_Latn", "deu_Latn", "fra_Latn", "rus_Cyrl", "hin_Deva", "tur_Latn", "jpn_Jpan", "kor_Hang", "cmn_Hans", "pes_Arab", "fin_Latn", "tam_Taml", "khm_Khmr", "arb_Arab"]);
const PLAY = '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 1.8v8.4L10 6z" fill="currentColor"/></svg>';
const PAUSE = '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 2h2.2v8H3zM6.8 2H9v8H6.8z" fill="currentColor"/></svg>';

/* all layouts of a model on a common scale (unit RMS radius) */
function layouts(D, model) {
  const out = { win: W.rescale(D.map[model].xy, 1) };
  const L = D.map_layers && D.map_layers[model];
  if (L) Object.keys(L).forEach(l => { out[+l] = W.rescale(L[l].xy, 1); });
  return out;
}

W.fig("map", {
  needs: ["map", "langs"],
  layout: "page",
  init(ctx) {
    const S = ctx.state, D = ctx.D;
    S.model = "qwen"; S.layer = "win"; S.groups = "family"; S.playing = false;
    S.lay = { qwen: layouts(D, "qwen"), llama: layouts(D, "llama") };
    S.hasLayers = !!(D.map_layers && D.map_layers.qwen && D.map_layers.llama);
    S.segModel = W.ui.seg(ctx.controls, { label: "Model", options: [["qwen", "Qwen2.5-7B"], ["llama", "Llama-3.1-8B"]], value: "qwen",
      onChange: v => { const old = S.model; S.model = v; if (S.layer !== "win") S.layer = Math.round(S.layer * MODELS[v].blocks / MODELS[old].blocks); stop(ctx); update(ctx, true); } });
    S.segGroups = W.ui.seg(ctx.controls, { label: "Outline", options: [["family", "Family or branch"], ["script", "Script"], ["none", "None"]], value: "family",
      onChange: v => { S.groups = v; if (S.view) S.view.setGroups(v); legend(ctx); } });
    legend(ctx);
    W.on("highlight", ({ codes }) => {
      if (!S.view) return;
      S.view.hlSet = codes;
      S.view.g.selectAll("path.grp-line").classed("on", d => !!codes && d.gr.codes.every(c => codes.has(c)));
    });
  },
  draw(ctx) {
    const S = ctx.state, w = ctx.width();
    const ex = W.extentOf([S.lay.qwen.win]), aspect = (ex[1][1] - ex[1][0]) / (ex[0][1] - ex[0][0]);
    const mapH = Math.round(w < 560 ? Math.max(300, (w - 40) * aspect + 56) : Math.max(400, Math.min(560, (w - 40) * aspect / 1.35 + 40)));
    const axisH = 34, H = mapH + axisH;
    const svg = W.frame(ctx, H, "Residual map of 34 languages");
    S.svg = svg;
    const view = W.mapView({ svg, x0: 0, y0: 0, w, h: mapH, groups: S.groups, groupStyle: "lines", labels: true,
      priority: l => (KEY.has(l.code) ? 1 : 0) });
    S.view = view;
    const xy = S.lay[S.model][S.layer] || S.lay[S.model].win;
    view.setPlan(xy); view.render(xy, { keepLabels: true, frame: view.frameFor(xy) });
    /* reading guide for the horizontal axis */
    const ga = svg.append("g").attr("transform", `translate(0,${mapH + 12})`);
    ga.append("line").attr("x1", 0).attr("x2", w).attr("class", "baseline");
    ga.append("text").attr("x", 0).attr("y", 17).attr("class", "small").text(w < 520 ? "← verb first" : "← verb-before-object end");
    ga.append("text").attr("x", w).attr("y", 17).attr("text-anchor", "end").attr("class", "small").text(w < 520 ? "object first →" : "object-before-verb end →");
    /* hover + keyboard */
    const node = svg.node();
    svg.on("pointermove", ev => {
      const [px, py] = d3.pointer(ev), hit = view.pick(px, py);
      if (hit && hit.lang) { W.highlight([hit.lang.code]); W.langTip(ev, hit.lang.code); }
      else if (hit && hit.group) { W.highlight(hit.group.codes); W.tip.show(ev, { title: S.groups === "family" ? W.groupText(hit.group.name) : hit.group.name + " script", rows: [{ l: hit.group.codes.map(c => W.lang(c).name).join(", ") }] }); }
      else { W.highlight(null); W.tip.hide(); }
    }).on("pointerleave", () => { W.highlight(null); W.tip.hide(); });
    const order = () => { const c = view.current(); return W.D.langs.map((l, i) => i).sort((a, b) => c[a][0] - c[b][0]); };
    W.keyNav(node, order, (i) => { const l = W.D.langs[i], [sx, sy] = view.screen(i), r = node.getBoundingClientRect();
      W.highlight([l.code]); W.langTip({ x: r.left + sx + 14, y: r.top + sy + 10 }, l.code); }, () => { W.highlight(null); W.tip.hide(); });
    if (S.hasLayers) scrubber(ctx, w); else S.readout = null;
    readout(ctx);
  }
});

function legend(ctx) {
  const g = ctx.state.groups;
  const items = [{ kind: "dot", cls: "c-vo", text: "verb before object" }, { kind: "dot", cls: "c-ov", text: "object before verb" }, { kind: "dot", cls: "c-nd", text: "no dominant order (WALS 83A)" }];
  if (g !== "none") items.push({ kind: "thin", cls: "s-axis", text: g === "family" ? "joins one family, or one branch of Indo-European" : "joins languages written in one script" });
  W.ui.legend(ctx.legend, items);
}

function update(ctx, animate) {
  const S = ctx.state, to = S.lay[S.model][S.layer] || S.lay[S.model].win, view = S.view;
  if (!view) return;
  const from = view.current(), f0 = view.frame(), f1 = view.frameFor(to);
  view.setPlan(to);
  if (!animate || !W.motion() || !from) { view.render(to, { keepLabels: true, frame: f1 }); }
  else {
    S.svg.interrupt("morph").transition("morph").duration(S.playing ? 380 : 560).ease(S.playing ? d3.easeLinear : d3.easeCubicInOut)
      .tween("xy", () => t => view.render(from.map((p, i) => [p[0] + (to[i][0] - p[0]) * t, p[1] + (to[i][1] - p[1]) * t]), { keepLabels: true, frame: view.lerpFrame(f0, f1, t) }));
  }
  S.segModel.set(S.model);
  syncScrub(ctx); readout(ctx);
}

function readout(ctx) {
  const S = ctx.state, D = ctx.D, el = S.readoutEl; if (!el) return;
  const M = MODELS[S.model]; el.replaceChildren();
  const b = (t) => { const x = W.el("b", null, t); el.appendChild(x); };
  const s = (t) => el.appendChild(document.createTextNode(t));
  if (S.layer === "win") {
    const win = D.profiles && D.profiles[S.model] ? D.profiles[S.model].window : null;
    b(win ? `Layers ${win[0]}–${win[win.length - 1]} of ${M.blocks}` : "Pre-registered window"); s(" averaged (pre-registered window)");
    s(` · first two axes hold ${W.f.pct(D.map[S.model].var_share)} of the residual variance`);
  } else {
    const L = D.map_layers[S.model][S.layer];
    b(`Layer ${S.layer} of ${M.blocks}`); s(S.layer === 0 ? " (embedding)" : "");
    s(` · two axes hold ${W.f.pct(L.var_share)} of the variance · similarity to the window map ${L.fit.toFixed(2)}`);
  }
}

/* layer scrubber: a slider whose track shows the per-layer gains (whitened geometry, the map's own) */
function scrubber(ctx, w) {
  const S = ctx.state, D = ctx.D;
  const row = W.el("div", "scrub"); row.style.cssText = "display:flex;flex-wrap:wrap;align-items:center;gap:.6rem 1rem;margin-top:.9rem";
  const btns = W.el("div"); btns.style.cssText = "display:flex;gap:.45rem;align-items:center";
  S.winBtn = W.ui.button(btns, { text: "Window", label: "Show the pre-registered layer window", pressed: S.layer === "win",
    onClick: () => { stop(ctx); S.layer = "win"; update(ctx, true); } });
  S.playBtn = W.ui.button(btns, { icon: PLAY, label: "Play through the layers", cls: "icon", onClick: () => (S.playing ? stop(ctx) : play(ctx)) });
  row.appendChild(btns);
  const tw = w < 640 ? w : w - 190, th = 58;
  const holder = W.el("div"); holder.style.cssText = `flex:1 1 ${Math.min(tw, 300)}px;min-width:0`;
  row.appendChild(holder);
  S.readoutEl = W.el("div", "readout"); S.readoutEl.style.cssText = "flex-basis:100%;margin-top:-.1rem"; row.appendChild(S.readoutEl);
  ctx.graphic.appendChild(row);
  const ww = Math.max(220, Math.floor(holder.clientWidth || tw));
  const svg = d3.select(holder).append("svg").attr("width", ww).attr("height", th).attr("viewBox", `0 0 ${ww} ${th}`)
    .attr("role", "slider").attr("tabindex", 0).attr("aria-label", "Layer").style("touch-action", "none").style("cursor", "pointer");
  S.scrubSvg = svg;
  const draw = () => {
    svg.selectAll("*").remove();
    const nb = MODELS[S.model].blocks, x = d3.scaleLinear().domain([0, nb]).range([8, ww - 8]);
    const prof = D.profiles && D.profiles[S.model] && D.profiles[S.model].per_layer.lda05;
    const win = D.profiles && D.profiles[S.model] ? D.profiles[S.model].window : [8, 20];
    const y = d3.scaleLinear().domain([0, prof ? d3.max(prof, d => Math.max(d.gen, d.ov)) : 1]).range([36, 6]);
    svg.append("rect").attr("x", x(win[0]) - 4).attr("width", x(win[win.length - 1]) - x(win[0]) + 8).attr("y", 2).attr("height", 36).attr("rx", 3)
      .attr("class", S.layer === "win" ? "band2" : "band");
    svg.append("line").attr("x1", x(0)).attr("x2", x(nb)).attr("y1", 38).attr("y2", 38).attr("class", "baseline");
    if (prof) {
      [["gen", "s-gen"], ["ov", "s-ov"]].forEach(([k, cls]) => svg.append("path").datum(prof).attr("class", "line thin " + cls).attr("stroke-width", 1.5)
        .attr("d", d3.line().x(d => x(d.layer)).y(d => y(d[k])).curve(d3.curveMonotoneX)));
    }
    d3.range(0, nb + 1).forEach(l => svg.append("line").attr("x1", x(l)).attr("x2", x(l)).attr("y1", 38).attr("y2", l % 4 === 0 ? 43 : 41).attr("class", "baseline"));
    d3.range(0, nb + 1, 4).forEach(l => svg.append("text").attr("x", x(l)).attr("y", 55).attr("text-anchor", "middle").attr("class", "small muted").text(l));
    if (S.layer !== "win") {
      const g = svg.append("g").attr("transform", `translate(${x(S.layer)},0)`);
      g.append("line").attr("y1", 2).attr("y2", 38).attr("class", "s-ink").attr("stroke-width", 1.5);
      g.append("circle").attr("cy", 38).attr("r", 5.5).attr("class", "c-ink dot");
    }
    svg.attr("aria-valuemin", 0).attr("aria-valuemax", nb).attr("aria-valuenow", S.layer === "win" ? win[0] : S.layer)
      .attr("aria-valuetext", S.layer === "win" ? `layers ${win[0]} to ${win[win.length - 1]} averaged` : `layer ${S.layer} of ${nb}`);
    S.scrubX = x;
  };
  S.drawScrub = draw; draw();
  const setFromPointer = ev => { const [px] = d3.pointer(ev, svg.node()), nb = MODELS[S.model].blocks; const l = Math.max(0, Math.min(nb, Math.round(S.scrubX.invert(px)))); if (l !== S.layer) { S.layer = l; update(ctx, true); } };
  let dragging = false;
  svg.on("pointerdown", ev => { stop(ctx); dragging = true; svg.node().setPointerCapture && svg.node().setPointerCapture(ev.pointerId); setFromPointer(ev); })
    .on("pointermove", ev => { if (dragging) setFromPointer(ev); })
    .on("pointerup pointercancel", () => { dragging = false; });
  svg.on("keydown", ev => {
    const nb = MODELS[S.model].blocks; let l = S.layer === "win" ? (ctx.D.profiles ? ctx.D.profiles[S.model].window[0] : 0) : S.layer;
    if (ev.key === "ArrowRight" || ev.key === "ArrowUp") l = Math.min(nb, l + (S.layer === "win" ? 0 : 1));
    else if (ev.key === "ArrowLeft" || ev.key === "ArrowDown") l = Math.max(0, l - (S.layer === "win" ? 0 : 1));
    else if (ev.key === "Home") l = 0; else if (ev.key === "End") l = nb; else return;
    ev.preventDefault(); stop(ctx); S.layer = l; update(ctx, true);
  });
}
function syncScrub(ctx) {
  const S = ctx.state;
  if (S.drawScrub) S.drawScrub();
  if (S.winBtn) S.winBtn.setAttribute("aria-pressed", String(S.layer === "win"));
}
function play(ctx) {
  const S = ctx.state, nb = MODELS[S.model].blocks;
  S.playing = true; S.playBtn.innerHTML = PAUSE; S.playBtn.setAttribute("aria-label", "Pause"); S.playBtn.title = "Pause";
  if (S.layer === "win" || S.layer >= nb) S.layer = -1;
  const step = () => {
    if (!S.playing) return;
    S.layer += 1; update(ctx, true);
    if (S.layer >= nb) { stop(ctx); return; }
    S.timer = setTimeout(step, W.motion() ? 420 : 700);
  };
  step();
}
function stop(ctx) {
  const S = ctx.state; S.playing = false; clearTimeout(S.timer);
  if (S.playBtn) { S.playBtn.innerHTML = PLAY; S.playBtn.setAttribute("aria-label", "Play through the layers"); S.playBtn.title = "Play through the layers"; }
}
W.mapLayouts = layouts;
})();
