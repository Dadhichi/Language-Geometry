"""PRE-REGISTERED analysis of the free-generation steering run (PREREG.md). usage: python analysis_gen.py OUT_DIR"""
import json, os, sys
import numpy as np

D = sys.argv[1]
conds = json.load(open(os.path.join(D, "conditions.json")))
cal = json.load(open(os.path.join(D, "calibration.json")))
G = [json.loads(l) for l in open(os.path.join(D, "gen.jsonl"), encoding="utf-8")]
ks = cal["kstar"]
LANGS = sorted({g["lang"] for g in G})
MIN_DEPS = 20


def stats(ci, lang, filt=True):
    rows = [g for g in G if g["cond"] == ci and g["lang"] == lang and (g["match"] or not filt)]
    ov, vo = sum(g["n_ov"] for g in rows), sum(g["n_vo"] for g in rows)
    return (ov / (ov + vo) if ov + vo else np.nan), ov + vo, rows


def find(kind, k, r=-1):
    return next(i for i, c in enumerate(conds) if c["kind"] == kind and abs(c["k"] - k) < 1e-9 and c["r"] == r)


base = {g: stats(0, g)[0] for g in LANGS}
needed = [0] + [find(kind, s * ks, r) for kind, r in [("ov", -1), ("ie", -1)] + [("rand", r) for r in range(24)] for s in (-1, 1)]
keep = [g for g in LANGS if all(stats(ci, g)[1] >= MIN_DEPS for ci in needed)]


def slope(kind, r=-1, k=ks, langs=keep, filt=True):
    per = {}
    for g in langs:
        dp = stats(find(kind, k, r), g, filt)[0] - stats(0, g, filt)[0]
        dm = stats(find(kind, -k, r), g, filt)[0] - stats(0, g, filt)[0]
        per[g] = float((dp - dm) / (2 * k))
    return float(np.nanmean(list(per.values()))), per


res = {"kstar": ks, "calibration": cal, "languages_used": keep}
print(f"k* = {ks} (calibration {cal}) | languages with >= {MIN_DEPS} counted deps in every needed condition: {keep}")
b_ov, per = slope("ov")
rand = np.array([slope("rand", r)[0] for r in range(24)])
z = (b_ov - rand.mean()) / rand.std(ddof=1)
p_emp = (1 + (rand >= b_ov).sum()) / (1 + len(rand))
res["primary"] = dict(b_ov=b_ov, per_language=per, rand_mean=float(rand.mean()), rand_sd=float(rand.std(ddof=1)),
                      z=float(z), p_emp=float(p_emp), claim=bool(b_ov > 0 and p_emp < 0.05 and z > 2.33),
                      rand_slopes=rand.tolist())
print(f"PRIMARY OV-rate slope per unit k: b_OV {b_ov:+.4f} | random {rand.mean():+.4f} +- {rand.std(ddof=1):.4f} "
      f"(max {rand.max():+.4f}) | z {z:+.2f} | p_emp {p_emp:.3f} | CLAIM {res['primary']['claim']}")
print("   per language:", {g: round(v, 4) for g, v in per.items()})
b_ie, per_ie = slope("ie")
b_half, _ = slope("ov", k=ks / 2)
b_nofilt, _ = slope("ov", filt=False)
res["secondary"] = dict(b_ie=b_ie, b_ov_half_k=b_half, b_ov_unfiltered=b_nofilt, n_positive=sum(v > 0 for v in per.values()))
print(f"IE control {b_ie:+.4f} | OV at k*/2 {b_half:+.4f} | OV without language filter {b_nofilt:+.4f} | "
      f"positive in {res['secondary']['n_positive']}/{len(per)} languages")
# descriptive table: OV rate, language-match rate, counted deps per condition x language
tab = {}
for name, ci in [("base", 0)] + [(f"ov{s * k:+g}", find("ov", s * k)) for k in (ks / 2, ks) for s in (-1, 1)] + \
        [(f"ie{s * ks:+g}", find("ie", s * ks)) for s in (-1, 1)]:
    tab[name] = {}
    for g in LANGS:
        r_, n_, rows = stats(ci, g)
        allrows = [x for x in G if x["cond"] == ci and x["lang"] == g]
        tab[name][g] = dict(ov_rate=r_, n_deps=n_, match=float(np.mean([x["match"] for x in allrows])))
    print(f"  {name:7s} " + " ".join(f"{g[:3]} {tab[name][g]['ov_rate']:.2f}/{tab[name][g]['match']:.2f}" for g in LANGS))
res["table"] = tab
# prompt-bootstrap 95% interval of b_OV
rs = np.random.RandomState(0)
sents = sorted({g["sent"] for g in G})
bs = []
for _ in range(2000):
    pick = set(rs.choice(sents, len(sents)))
    sub = lambda ci, g: [x for x in G if x["cond"] == ci and x["lang"] == g and x["match"] and x["sent"] in pick]
    vals = []
    for g in keep:
        def rate(ci):
            rr = sub(ci, g); o = sum(x["n_ov"] for x in rr); v = sum(x["n_vo"] for x in rr)
            return o / (o + v) if o + v else np.nan
        vals.append((rate(find("ov", ks)) - rate(find("ov", -ks))) / (2 * ks))
    bs.append(np.nanmean(vals))
res["b_ov_boot95"] = [float(x) for x in np.nanquantile(bs, [0.025, 0.975])]
print(f"b_OV prompt-bootstrap 95% CI {res['b_ov_boot95']}")
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_gen.json"), "w"), indent=1, default=float)
