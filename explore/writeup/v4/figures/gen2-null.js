/* gen2-null: the pre-registered tests of the confirmatory run (H1, H2): pooled slope of the noun-object share for the
   word-order direction against 24 random directions, with the verdict. Pending state until D.freegen2 exists. */
(function () {
"use strict";
const W = window.WOA;

W.fig("gen2-null", {
  needs: ["freegen2"],
  layout: "wide",
  pending(ctx) {
    W.ui.legend(ctx.legend, [{ kind: "dot", cls: "c-ov", text: "word-order direction" }, { kind: "dot", cls: "c-gen", text: "Indo-European direction" }, { kind: "dot", cls: "c-rand-strong", text: "24 random directions" }]);
    W.pendingState(ctx, { title: "Confirmatory run in progress", text: "The pre-registered tests of H1 and H2 appear here when the run finishes and the page is rebuilt.", rows: 2, height: 210 });
  },
  init(ctx) {
    const T = ctx.D.freegen2.tests || {}, ie = Object.values(T).some(t => t && t.b_ie != null);
    W.ui.legend(ctx.legend, [{ kind: "dot", cls: "c-ov", text: "word-order direction" }].concat(ie ? [{ kind: "dot", cls: "c-gen", text: "Indo-European direction" }] : [], [{ kind: "dot", cls: "c-rand-strong", text: "24 random directions" }]));
  },
  draw(ctx) {
    const F = ctx.D.freegen2, T = F.tests || {}, sets = F.sets || {};
    const rows = ["H1", "H2"].filter(h => T[h]).map(h => {
      const t = T[h], langs = (sets[h] || []).map(W.shortName).join(", ");
      const verdict = t.evaluable === false ? { text: "not evaluable", strong: false } : t.claim ? { text: "claim holds", strong: true } : { text: "claim not supported", strong: false };
      return { label: h, sub: langs || (t.n_langs != null ? `${t.n_langs} languages` : null), rand: t.rand || [], ov: t.b_ov, ci: t.ci || t.b_ov_boot95 || null, ie: t.b_ie != null ? t.b_ie : null, z: t.z, verdict };
    });
    W.nullRows(ctx, rows, { label: "Confirmatory tests: slope of the noun-object share, word-order direction against random directions", axis: "change in noun-object share per unit k, percentage points" });
  }
});
})();
