/* pairs: minimal-pair examples. The object (underlined) and the verb (bold) are recovered from the two twins (same
   tokens, one block moved); a control swaps every card between its object-first and verb-first twin with a short
   glide, and each card says which twin is the attested sentence. */
(function () {
"use strict";
const W = window.WOA;
const LANGTAG = { eng: "en", hin: "hi", zho: "zh", tur: "tr", deu: "de", fra: "fr", jpn: "ja", rus: "ru" };

/* split two twins into prefix, object, verb, suffix: the middle of one is a rotation of the middle of the other */
function parse(ov, vo, cjk) {
  const A = cjk ? [...ov] : ov.split(" "), B = cjk ? [...vo] : vo.split(" "), J = cjk ? "" : " ";
  let p = 0; while (p < A.length && p < B.length && A[p] === B[p]) p++;
  let q = 0; while (q < A.length - p && q < B.length - p && A[A.length - 1 - q] === B[B.length - 1 - q]) q++;
  for (let s = p + q; s >= 0; s--) for (let pp = Math.min(p, s); pp >= 0; pp--) {
    const qq = s - pp; if (qq > q) continue;
    const mo = A.slice(pp, A.length - qq), mv = B.slice(pp, B.length - qq);
    if (mo.length !== mv.length || mo.length < 2) continue;
    for (let k = 1; k < mo.length; k++) {
      const rot = mo.slice(k).concat(mo.slice(0, k));
      if (rot.every((t, i) => t === mv[i])) return { pre: A.slice(0, pp).join(J), obj: mo.slice(0, k).join(J), verb: mo.slice(k).join(J), post: A.slice(A.length - qq).join(J), J };
    }
  }
  return null;
}

W.fig("pairs", {
  needs: ["steering.examples"],
  layout: "wide",
  init(ctx) {
    const S = ctx.state; S.show = "OV";
    S.seg = W.ui.seg(ctx.controls, { label: "Show", options: [["OV", "Object-first twin"], ["VO", "Verb-first twin"]], value: "OV", onChange: v => { S.show = v; S.cards.forEach(c => c.set(v)); } });
    const lg = W.el("span", "readout"); lg.innerHTML = "<u>object</u> · <b>verb</b>"; ctx.controls.appendChild(lg);
  },
  draw(ctx) {
    const S = ctx.state, D = ctx.D;
    ctx.graphic.replaceChildren();
    const grid = W.el("div", "pairs"); ctx.graphic.appendChild(grid);
    const per = {}; (D.steering.per_language || []).forEach(p => { per[p.code] = p; });
    S.cards = D.steering.examples.map(ex => {
      const code = ex.lang, short = code.split("_")[0], cjk = /^(zho|cmn|jpn)/.test(code);
      const parts = parse(ex.ov, ex.vo, cjk);
      const card = W.el("div", "pair");
      const head = W.el("div", "pair-h"); head.appendChild(W.el("b", null, W.shortName(code)));
      const tag = W.el("span", "tag"); head.appendChild(tag);
      const flip = W.ui.button(head, { text: "Swap", label: `Swap object and verb (${W.shortName(code)})`, onClick: () => set(cur === "OV" ? "VO" : "OV", true) });
      flip.style.marginLeft = "auto"; flip.style.padding = ".1rem .55rem"; flip.style.fontSize = "11.5px";
      const sent = W.el("div", "pair-s"); sent.lang = LANGTAG[short] || short;
      card.append(head, sent);
      const p = per[code];
      if (p) {
        const f = W.el("div", "pair-f"), pref = p.base >= 0 ? "object-first" : "verb-first";
        f.append(document.createTextNode("Unsteered, the model prefers the "), W.el("b", null, `${pref} twin by ${Math.abs(p.base).toFixed(1)} nats`), document.createTextNode(` on average over the ${p.n} ${W.shortName(code)} pairs.`));
        card.appendChild(f);
      }
      grid.appendChild(card);
      let cur = null, spans = null;
      if (parts) {
        const o = W.el("span", "w obj", parts.obj), v = W.el("span", "w verb", parts.verb);
        spans = { o, v, pre: document.createTextNode(parts.pre + (parts.pre ? parts.J : "")), mid: document.createTextNode(parts.J), post: document.createTextNode((parts.post ? parts.J : "") + parts.post) };
      }
      function set(order, animate) {
        if (order === cur) return;
        const first = spans && animate && W.motion() && cur ? { o: spans.o.getBoundingClientRect(), v: spans.v.getBoundingClientRect() } : null;
        cur = order;
        tag.textContent = order === ex.orig_order ? "attested sentence" : "swapped twin";
        if (!spans) { sent.textContent = order === "OV" ? ex.ov : ex.vo; return; }
        sent.replaceChildren(spans.pre, order === "OV" ? spans.o : spans.v, spans.mid, order === "OV" ? spans.v : spans.o, spans.post);
        if (first) [["o", spans.o], ["v", spans.v]].forEach(([k, el]) => {
          const last = el.getBoundingClientRect(), dx = first[k].left - last.left, dy = first[k].top - last.top;
          if (!dx && !dy) return;
          el.style.transition = "none"; el.style.transform = `translate(${dx}px,${dy}px)`;
          requestAnimationFrame(() => requestAnimationFrame(() => { el.style.transition = ""; el.style.transform = ""; }));
        });
      }
      set(S.show, false);
      return { set: v => set(v, true) };
    });
  }
});
W.parsePair = parse;
})();
