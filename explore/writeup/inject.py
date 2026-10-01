"""Build the publishable page: template.html + figdata.json (+ optional sections/*.html and extra_figs.js).
usage: python inject.py figdata.json OUT.html"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
tpl = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
data = json.load(open(sys.argv[1], encoding="utf-8"))
blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
html = tpl.replace("__FIGDATA__", blob)
extra = os.path.join(HERE, "extra_figs.js")
html = html.replace("/* __EXTRA_FIGS__ */", open(extra, encoding="utf-8").read() if os.path.exists(extra) else "")
for sec in ("which-body", "gen-body"):
    p = os.path.join(HERE, "sections", f"{sec}.html")
    if os.path.exists(p):                      # a section file brings its own .col / figure wrappers
        html = re.sub(rf"<!--SECTION:{sec}-->.*?<!--/SECTION-->", lambda m: open(p, encoding="utf-8").read(),
                      html, count=1, flags=re.S)
open(sys.argv[2], "w", encoding="utf-8").write(html)
script = html.rsplit("<script>", 1)[1].rsplit("</script>", 1)[0]
open(os.path.join(HERE, "_check.js"), "w", encoding="utf-8").write(script)
print(f"built {sys.argv[2]} ({len(html) / 1e3:.0f} KB)")
