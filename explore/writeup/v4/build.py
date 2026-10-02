"""Assemble the v4 article into one self-contained HTML file.

usage: python v4/build.py [FIGDATA.json] [OUT.html] [--content DIR] [--local-libs DIR] [--mock-freegen2] [--strict]

  FIGDATA.json   figure data (default: explore/writeup/figdata.json)
  OUT.html       output (default: explore/writeup/v4/preview.html)
  --content DIR  content files NN-*.html (default: v4/content; falls back to v4/dev_content with a warning when empty)
  --local-libs   inline d3.min.js and katex.min.js from DIR instead of the CDN tags (offline screenshots and tests only)
  --mock-freegen2  DEV ONLY: add a fake `freegen2` built from `freegen_x` to exercise the confirmatory figures
  --strict       exit non-zero on contract warnings

The page is shell.html + content (file-name order) + runtime.js + figures/*.js + the data, inlined. Only d3 and KaTeX
load from cdnjs and the fonts from Google Fonts (artifact CSP). The output is an HTML fragment (no doctype/head/body):
the artifact host wraps it in its own document skeleton.
"""
import argparse, glob, html.parser, json, os, re, sys

V4 = os.path.dirname(os.path.abspath(__file__))
WRITEUP = os.path.dirname(V4)
D3_TAG = '<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>'
KATEX_TAG = '<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.11/katex.min.js"></script>'
ALLOWED_CLASSES = {"m", "mb", "figref", "tablewrap", "tcap", "sidenote", "def", "lab", "takeaway", "note", "pending",
                   "example", "ex-row", "ex-tag", "ex-text", "ex-gloss", "cite", "refs", "front", "subtitle", "byline", "n"}
CONTRACT_FIGS = ["overview", "pipeline", "map", "splits", "tree", "depth", "typology", "proj", "pairs", "steer",
                 "steer-langs", "gen-rates", "gen-null", "gen-loss", "gen2-rates", "gen2-null", "within-align", "within-steer",
                 "within-retain", "cs"]
CONTRACT_TABLES = ["primary", "lex", "gen-obj"]


class Lint(html.parser.HTMLParser):
    """collects what the markup contract cares about"""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids, self.classes, self.figs, self.tables, self.figrefs, self.sections = [], set(), [], [], [], []
        self.styles = self.scripts = 0
        self.stack, self.figure_caps, self.cur_fig = [], {}, None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.append(a["id"])
        for c in (a.get("class") or "").split():
            self.classes.add(c)
        if "style" in a:
            self.styles += 1
        if tag == "script":
            self.scripts += 1
        if tag == "figure":
            self.cur_fig = a.get("data-fig")
            self.figs.append((a.get("id"), a.get("data-fig")))
            self.figure_caps[a.get("data-fig")] = False
        if tag == "figcaption" and self.cur_fig is not None:
            self.figure_caps[self.cur_fig] = True
        if tag == "table" and "data-table" in a:
            self.tables.append(a["data-table"])
        if tag == "a" and "figref" in (a.get("class") or "").split():
            self.figrefs.append(a.get("href", ""))
        if tag == "section":
            self.sections.append((a.get("id"), a.get("data-title")))

    def handle_endtag(self, tag):
        if tag == "figure":
            self.cur_fig = None


def lint(files, fig_modules, table_ids):
    warn, info = [], []
    L = Lint()
    for f in files:
        L.feed(open(f, encoding="utf-8").read())
    if L.styles:
        warn.append(f"{L.styles} inline style attribute(s) in content (contract: none)")
    if L.scripts:
        warn.append(f"{L.scripts} <script> tag(s) in content (contract: none)")
    extra = sorted(L.classes - ALLOWED_CLASSES)
    if extra:
        warn.append("classes outside the contract: " + ", ".join(extra))
    dup = sorted({i for i in L.ids if L.ids.count(i) > 1})
    if dup:
        warn.append("duplicate ids: " + ", ".join(dup))
    for fid, key in L.figs:
        if key is None:
            info.append(f"figure {fid!r} has no data-fig (static figure, not numbered)")
            continue
        if fid != f"fig-{key}":
            warn.append(f"figure data-fig={key!r} should have id 'fig-{key}' (has {fid!r})")
        if key not in fig_modules:
            warn.append(f"figure {key!r} has no module in figures/")
        if not L.figure_caps.get(key):
            warn.append(f"figure {key!r} has no <figcaption>")
    present = {k for _, k in L.figs}
    for t in L.tables:
        if t not in table_ids:
            warn.append(f"data table {t!r} has no module")
    ids = set(L.ids)
    for h in L.figrefs:
        if not h.startswith("#") or h[1:] not in ids:
            warn.append(f"figref {h!r} points to no element")
    for sid, title in L.sections:
        if not title:
            warn.append(f"section {sid!r} has no data-title")
    missing_figs = [k for k in CONTRACT_FIGS if k not in present]
    if missing_figs:
        info.append("contract figures not (yet) placed: " + ", ".join(missing_figs))
    missing_tabs = [t for t in CONTRACT_TABLES if t not in L.tables]
    if missing_tabs:
        info.append("contract data tables not (yet) placed: " + ", ".join(missing_tabs))
    return warn, info, L


def mock_freegen2(D):
    """DEV ONLY. A fake confirmatory result in the freegen2 schema, built by relabelling the exploratory freegen_x data
    onto the nine languages of the confirmatory pre-registration. Never publish a page built with it."""
    X = D.get("freegen_x")
    if not X:
        return None
    import random
    rnd = random.Random(7)
    src = {"deu_Latn": "deu_Latn", "rus_Cyrl": "rus_Cyrl", "nld_Latn": "deu_Latn", "ukr_Cyrl": "rus_Cyrl", "pol_Latn": "rus_Cyrl",
           "hrv_Latn": "fra_Latn", "eng_Latn": "eng_Latn", "spa_Latn": "fra_Latn", "kor_Hang": "jpn_Jpan"}
    sets = {"H1": ["deu_Latn", "rus_Cyrl"], "H2": ["nld_Latn", "ukr_Cyrl", "pol_Latn", "hrv_Latn"], "controls": ["eng_Latn", "spa_Latn", "kor_Hang"]}
    rates = {c: {k: X["rates"][s_][k] for k in ("base", "ov-", "ov+", "ie-", "ie+")} | {"rand": X["rates"][s_]["rand"]} for c, s_ in src.items()}
    match = {c: X["match"][s_] for c, s_ in src.items()}
    P = X["pooled"]

    def test(b, n, claim):
        rand = [r + rnd.gauss(0, .002) for r in P["rand"]]
        m = sum(rand) / len(rand)
        sd = (sum((r - m) ** 2 for r in rand) / (len(rand) - 1)) ** .5
        return {"b_ov": b, "rand": rand, "rand_mean": m, "rand_sd": sd, "z": (b - m) / sd, "p_emp": 1 / 25,
                "n_langs": n, "evaluable": True, "claim": claim}
    per = {}
    for c, s_ in src.items():
        pl = X["per_language"][s_]
        per[c] = {"b_ov": pl["b_ov"], "rand_mean": pl["rand_mean"], "rand_sd": pl["rand_sd"], "z": pl["z"],
                  "pred": "+" if c in sets["H1"] + sets["H2"] else "0", "pred_ok": (pl["z"] or 0) > 1.5 or c in sets["controls"]}
    return {"MOCK": True, "k": X["kstar"], "langs": list(src), "sets": sets, "rates": rates, "match": match,
            "tests": {"H1": test(P["b_ov"], 2, True), "H2": test(0.004, 4, False)}, "per_language": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("figdata", nargs="?", default=os.path.join(WRITEUP, "figdata.json"))
    ap.add_argument("out", nargs="?", default=os.path.join(V4, "preview.html"))
    ap.add_argument("--content", default=os.path.join(V4, "content"))
    ap.add_argument("--local-libs", default=None)
    ap.add_argument("--mock-freegen2", action="store_true")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    pat = re.compile(r"^\d\d-.*\.html$")
    files = sorted(f for f in glob.glob(os.path.join(a.content, "*.html")) if pat.match(os.path.basename(f)))
    if not files and os.path.abspath(a.content) == os.path.abspath(os.path.join(V4, "content")):
        print("WARNING: v4/content has no NN-*.html files; building from v4/dev_content (designer stub text)")
        files = sorted(f for f in glob.glob(os.path.join(V4, "dev_content", "*.html")) if pat.match(os.path.basename(f)))
    if not files:
        sys.exit(f"no content files NN-*.html in {a.content}")

    fig_files = sorted(glob.glob(os.path.join(V4, "figures", "*.js")))
    fig_modules = set()
    for f in fig_files:
        fig_modules |= set(re.findall(r'(?:WOA|W)\.fig\("([^"]+)"', open(f, encoding="utf-8").read()))
    table_ids = set()
    for f in fig_files:
        table_ids |= set(re.findall(r'(?:WOA|W)\.table\("([^"]+)"', open(f, encoding="utf-8").read()))

    warn, info, L = lint(files, fig_modules, table_ids)

    shell = open(os.path.join(V4, "shell.html"), encoding="utf-8").read()
    content = "\n".join(open(f, encoding="utf-8").read().strip() for f in files)
    runtime = open(os.path.join(V4, "runtime.js"), encoding="utf-8").read()
    figs = "\n".join(f"/* ---- figures/{os.path.basename(f)} ---- */\n" + open(f, encoding="utf-8").read() for f in fig_files)

    D = json.load(open(a.figdata, encoding="utf-8"))
    if a.mock_freegen2:
        D["freegen2"] = mock_freegen2(D)
        print("WARNING: --mock-freegen2: the output contains FAKE confirmatory data (dev only, never publish)")
    data = json.dumps(D, separators=(",", ":"), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")

    for marker in ("<!--CONTENT-->", "/*__RUNTIME__*/", "/*__FIGURES__*/", "__FIGDATA__", D3_TAG, KATEX_TAG):
        if marker not in shell:
            sys.exit(f"shell.html lacks {marker}")
    if "</script" in runtime.lower() or "</script" in figs.lower():
        sys.exit("a script contains '</script'")
    page = (shell.replace("<!--CONTENT-->", content)
                 .replace("/*__RUNTIME__*/", runtime)
                 .replace("/*__FIGURES__*/", figs)
                 .replace("__FIGDATA__", data))
    if a.local_libs:
        for tag, name in ((D3_TAG, "d3.min.js"), (KATEX_TAG, "katex.min.js")):
            page = page.replace(tag, "<script>" + open(os.path.join(a.local_libs, name), encoding="utf-8").read() + "</script>")
        print("NOTE: --local-libs: d3 and KaTeX inlined (not for publishing)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w", encoding="utf-8", newline="\n").write(page)

    print(f"built {a.out}: {len(page) / 1e3:.0f} KB from {len(files)} content files "
          f"({', '.join(os.path.basename(f) for f in files)}), {len(fig_files)} figure files, data keys: {', '.join(D)}")
    print(f"figures in content: {len(L.figs)}; data tables: {len(L.tables)}; figrefs: {len(L.figrefs)}; sections: {len(L.sections)}")
    for m in info:
        print("  info:", m)
    for m in warn:
        print("  WARNING:", m)
    if a.strict and warn:
        sys.exit(1)


if __name__ == "__main__":
    main()
