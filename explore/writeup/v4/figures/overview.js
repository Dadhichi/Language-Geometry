/* overview: teaser residual map for the summary (Qwen, pre-registered window), family outlines, word-order direction. */
(function () {
"use strict";
const W = window.WOA;
const KEY = new Set(["eng_Latn", "fra_Latn", "rus_Cyrl", "hin_Deva", "tur_Latn", "jpn_Jpan", "kor_Hang", "cmn_Hans", "pes_Arab", "fin_Latn", "tel_Telu", "khm_Khmr", "arb_Arab", "deu_Latn"]);

W.fig("overview", {
  needs: ["map", "langs"],
  layout: "page",
  init(ctx) {
    W.ui.legend(ctx.legend, [{ kind: "dot", cls: "c-vo", text: "verb before object" }, { kind: "dot", cls: "c-ov", text: "object before verb" },
      { kind: "dot", cls: "c-nd", text: "no dominant order" }, { kind: "hull", text: "one family, or one branch of Indo-European" }]);
  },
  draw(ctx) {
    const D = ctx.D, w = ctx.width();
    const xy = W.rescale(D.map.qwen.xy, 1), ex = W.extentOf([xy]);
    const aspect = (ex[1][1] - ex[1][0]) / (ex[0][1] - ex[0][0]);
    const h = Math.round(w < 560 ? Math.max(300, (w - 40) * aspect + 70) : Math.max(380, Math.min(540, (w - 40) * aspect / 1.35 + 40)));
    const svg = W.frame(ctx, h + 22, "Residual map of 34 languages, outlines for families and branches of Indo-European, and the word-order direction");
    const view = W.mapView({ svg, x0: 0, y0: 0, w, h, groups: "family", groupStyle: "hull", labels: true, hullLabels: w >= 700, arrow: true,
      priority: l => (KEY.has(l.code) ? 1 : 0) });
    view.setPlan(xy); view.render(xy, { keepLabels: true });
    svg.append("text").attr("x", w).attr("y", h + 18).attr("text-anchor", "end").attr("class", "small muted")
      .text(`Qwen2.5-7B · layers ${D.profiles ? D.profiles.qwen.window[0] + "–" + D.profiles.qwen.window.slice(-1)[0] : "8–20"} · classical MDS of residual distances`);
    svg.on("pointermove", ev => {
      const [px, py] = d3.pointer(ev), hit = view.pick(px, py);
      if (hit && hit.lang) { W.highlight([hit.lang.code]); W.langTip(ev, hit.lang.code); }
      else if (hit && hit.group) { W.highlight(hit.group.codes); W.tip.show(ev, { title: W.groupText(hit.group.name), rows: [{ l: hit.group.codes.map(c => W.lang(c).name).join(", ") }] }); }
      else { W.highlight(null); W.tip.hide(); }
    }).on("pointerleave", () => { W.highlight(null); W.tip.hide(); });
    const node = svg.node();
    W.keyNav(node, () => D.langs.map((l, i) => i).sort((a, b) => xy[a][0] - xy[b][0]),
      i => { const [sx, sy] = view.screen(i), r = node.getBoundingClientRect(); W.highlight([D.langs[i].code]); W.langTip({ x: r.left + sx + 14, y: r.top + sy + 10 }, D.langs[i].code); },
      () => { W.highlight(null); W.tip.hide(); });
  }
});
})();
