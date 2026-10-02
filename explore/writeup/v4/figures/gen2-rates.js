/* gen2-rates: confirmatory free-generation run (freegen2): noun-object share before the verb per language, grouped by
   pre-registered set, same encoding as gen-rates. Renders a pending state until D.freegen2 exists. */
(function () {
"use strict";
const W = window.WOA;
const SETS = [["H1", "H1"], ["H2", "H2"], ["controls", "Controls"]];
/* analysis_gen2.py: "up" = object-first share rises under +k, "down" = falls under -k, "none" = no change (controls) */
const PRED = { up: "rises under +k", down: "falls under −k", none: "no change" };

W.fig("gen2-rates", {
  needs: ["freegen2"],
  layout: "wide",
  pending(ctx) {
    W.ui.legend(ctx.legend, [{ kind: "dot", cls: "c-vo", text: "k = −2, toward verb-first" }, { kind: "dot", cls: "c-ink2", text: "unsteered" }, { kind: "dot", cls: "c-ov", text: "k = +2, toward object-first" }, { kind: "band", cls: "band2", text: "random directions" }]);
    W.pendingState(ctx, { title: "Confirmatory run in progress", text: "This figure is drawn from the pre-registered rerun. It appears here when the run finishes and the page is rebuilt.", rows: 9, height: 380 });
  },
  init(ctx) {
    W.ui.legend(ctx.legend, [{ kind: "dot", cls: "c-vo", text: "k = −2, toward verb-first" }, { kind: "dot", cls: "c-ink2", text: "unsteered" }, { kind: "dot", cls: "c-ov", text: "k = +2, toward object-first" },
      { kind: "ring", cls: "s-ink2", text: "under half the text in the prompt language" }, { kind: "band", cls: "band2", text: "random directions, 5th–95th percentile" }]);
  },
  draw(ctx) {
    const F = ctx.D.freegen2, k = F.k || 2, sets = F.sets || {}, per = F.per_language || {};
    const groupOf = c => (SETS.find(([s]) => (sets[s] || []).includes(c)) || [null, "Other"])[1];
    const order = SETS.flatMap(([s]) => sets[s] || []);
    const codes = Object.keys(F.rates).sort((a, b) => {
      const ga = order.indexOf(a), gb = order.indexOf(b);
      const sa = SETS.findIndex(([s]) => (sets[s] || []).includes(a)), sb = SETS.findIndex(([s]) => (sets[s] || []).includes(b));
      if (sa !== sb) return (sa < 0 ? 9 : sa) - (sb < 0 ? 9 : sb);
      return ((F.rates[a].base || [0])[0] ?? 0) - ((F.rates[b].base || [0])[0] ?? 0) || ga - gb;
    });
    const rows = codes.map(c => {
      const R = F.rates[c], M = (F.match || {})[c] || {}, P = per[c];
      const low = [M.ov_m, M.ov_p].map(v => (v == null ? 1 : v));        // say why a point is missing: the model left the language
      const sub = Math.min(...low) < .8 ? `${W.f.pct(M.ov_m)} / ${W.f.pct(M.ov_p)} in language` : null;
      const predTxt = !P ? null : P.pred_ok == null ? "not testable: too few pairs" : P.pred_ok ? "prediction met" : "prediction not met";
      return { code: c, name: W.shortName(c), group: groupOf(c), base: R.base, minus: R["ov-"], plus: R["ov+"], mMinus: M.ov_m, mPlus: M.ov_p, mBase: M.base, rand: R.rand || null, sub,
        tipExtra: P ? [{ v: P.b_ov == null ? "–" : W.f.pts(P.b_ov, 1) + " pts", l: `slope per unit k${P.z != null ? `, z = ${P.z.toFixed(1)}` : ""}` }, { v: PRED[P.pred] || String(P.pred ?? "–"), l: predTxt }] : [] };
    });
    W.genRows(ctx, rows, { label: "Confirmatory run: noun-object share before the verb per language", axis: "share of noun objects before the verb",
      kMinus: `k = −${k} (toward verb-first)`, kPlus: `k = +${k} (toward object-first)`, groups: true, minN: 20,
      right: r => { const P = per[r.code]; if (!P) return null; const pr = PRED[P.pred] ? PRED[P.pred] + ": " : "";
        return P.pred_ok == null ? { text: pr + "too few pairs", short: "–", cls: "small muted" } : { text: pr + (P.pred_ok ? "met" : "not met"), short: P.pred_ok ? "met" : "not met", cls: P.pred_ok ? "small strong" : "small muted" }; },
      rightHead: "prediction (pre-registered)", rightW: 176 });
  }
});
})();
