/* Headless check of a built v4 page (jsdom).
   usage: node v4/check.js PAGE.html --libs DIR [--mock]
     DIR holds d3.min.js, katex.min.js and node_modules/jsdom (the scratchpad domtest folder).
   Reports: runtime errors, math rendered, marks per figure, figure numbering, figrefs, data tables, pending state,
   contents. Exit code 1 if anything is wrong. */
const fs = require("fs"), path = require("path");
const args = process.argv.slice(2);
const page = args[0], libs = args[args.indexOf("--libs") + 1];
if (!page || !libs || args.indexOf("--libs") < 0) { console.error("usage: node check.js PAGE.html --libs DIR"); process.exit(2); }
const { JSDOM, VirtualConsole } = require(path.join(libs, "node_modules", "jsdom"));
let html = fs.readFileSync(page, "utf8");
html = html.replace('<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>', () => "<script>" + fs.readFileSync(path.join(libs, "d3.min.js"), "utf8") + "</script>");
html = html.replace('<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11/katex.min.js"></script>', () => "<script>" + fs.readFileSync(path.join(libs, "katex.min.js"), "utf8") + "</script>");
const errors = [], logs = [];
const vc = new VirtualConsole();
vc.on("jsdomError", e => { if (!/Not implemented|Could not parse CSS stylesheet/.test(e.message)) errors.push("jsdom: " + e.message); });
vc.on("error", (...a) => errors.push("console.error: " + a.map(String).join(" ").slice(0, 300)));
vc.on("warn", (...a) => logs.push("warn: " + a.join(" ")));
const dom = new JSDOM("<!doctype html><html><head><meta charset='utf-8'></head><body>" + html + "</body></html>", {
  runScripts: "dangerously", pretendToBeVisual: true, virtualConsole: vc,
  beforeParse(w) {
    w.matchMedia = q => ({ matches: false, media: q, addEventListener() {}, removeEventListener() {} });
    w.addEventListener("error", e => errors.push("window.onerror: " + (e.message || e)));
  }
});
const w = dom.window, d = w.document;
/* exercise interactions a little: hover a language, toggle controls */
function exercise() {
  const W = w.WOA; if (!W) return;
  try {
    W.highlight(["eng_Latn"]); W.highlight(["hin_Deva", "urd_Arab"]); W.highlight(null);
    d.querySelectorAll(".fig-controls .seg button").forEach(b => b.click());
    d.querySelectorAll(".fig-controls .seg button:first-child").forEach(b => b.click());
    d.querySelectorAll(".fig-body .btn").forEach(b => { if (!/Play/.test(b.getAttribute("aria-label") || "")) b.click(); });
    W.redrawAll();
  } catch (e) { errors.push("exercise: " + e.message); }
}
setTimeout(() => {
  exercise();
  setTimeout(() => {
    const W = w.WOA || {};
    let bad = 0;
    const all = d.querySelectorAll(".m,.mb"), ok = d.querySelectorAll(".m.ready,.mb.ready"), kerr = d.querySelectorAll(".katex-error");
    console.log(`math        ${ok.length}/${all.length} rendered, ${kerr.length} KaTeX errors`);
    kerr.forEach(b => console.log("   KaTeX ERR:", (b.getAttribute("title") || b.textContent).slice(0, 120)));
    if (ok.length !== all.length || kerr.length) bad++;
    const figs = [...d.querySelectorAll("figure[data-fig]")];
    const nums = figs.map(f => +f.dataset.num);
    const inOrder = nums.every((n, i) => n === i + 1);
    const capsOK = figs.every(f => { const s = f.querySelector("figcaption .fignum"); return s && s.textContent === `Figure ${f.dataset.num}.`; });
    console.log(`figures     ${figs.length}, numbering ${inOrder && capsOK ? "in order" : "WRONG"} (${nums.join(",")})`);
    if (!inOrder || !capsOK) bad++;
    figs.forEach(f => {
      const id = f.dataset.fig, g = f.querySelector(".fig-graphic");
      const marks = g ? g.querySelectorAll("circle,path,line,rect,text,polygon").length : 0;
      const html = g ? g.querySelectorAll("div,span,b,u").length : 0;
      const pend = f.querySelector(".fig-body.is-pending") ? "pending" : "";
      const miss = f.querySelector(".fig-missing") ? "MISSING: " + f.querySelector(".fig-missing").textContent : "";
      const ctl = f.querySelectorAll(".fig-controls button, .fig-graphic [role=slider], .fig-graphic button").length;
      const ok = !miss && (marks > 0 || html > 0);
      if (!ok) bad++;
      console.log(`  ${String(f.dataset.num).padStart(2)} ${id.padEnd(12)} marks ${String(marks).padStart(4)}  html ${String(html).padStart(3)}  controls ${String(ctl).padStart(2)}  ${pend} ${miss} ${ok ? "" : "<-- EMPTY"}`);
    });
    const refs = [...d.querySelectorAll("a.figref")], unfilled = refs.filter(a => !/^Figure \d+$/.test(a.textContent));
    console.log(`figrefs     ${refs.length}, unfilled ${unfilled.length}${unfilled.length ? " -> " + unfilled.map(a => a.getAttribute("href")).join(" ") : ""}`);
    if (unfilled.length) bad++;
    const tabs = [...d.querySelectorAll("table[data-table]")];
    tabs.forEach(t => { const rows = t.querySelectorAll("tbody tr").length; if (!rows) bad++; console.log(`  table ${t.dataset.table.padEnd(9)} ${rows} rows ${rows ? "" : "<-- EMPTY"}`); });
    const pend = [...d.querySelectorAll(".pending[data-pending]")];
    console.log(`pending     ${pend.length} paragraph(s): ${pend.map(p => p.dataset.pending + (p.hidden ? " hidden" : " shown")).join(", ") || "-"}`);
    console.log(`contents    ${d.querySelectorAll("nav.toc-flow li").length} in-flow, ${d.querySelectorAll("#tb-menu li").length} in menu; sidenotes ${d.querySelectorAll("aside.sidenote").length}`);
    const errs = errors.concat((W.errors || []).map(e => "WOA: " + e));
    (W.warnings || []).forEach(m => console.log("  warning:", m));
    console.log(errs.length ? "ERRORS:\n  " + errs.join("\n  ") : "runtime     no errors");
    if (errs.length) bad++;
    console.log(bad ? `CHECK FAILED (${bad} problem groups)` : "CHECK OK");
    process.exit(bad ? 1 : 0);
  }, 1500);
}, 1200);
