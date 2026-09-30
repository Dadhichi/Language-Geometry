"""PRE-REGISTERED analysis of the steering run (PREREG.md). usage: python analysis_steer.py SCORES_DIR PAIRS.jsonl"""
import json, os, sys
import numpy as np

D, P = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(P, encoding="utf-8")]
conds = json.load(open(os.path.join(D, "conditions.json")))
z = np.load(os.path.join(D, "scores.npz"))
lang = np.array([r["lang"] for r in rows])
LANGS = sorted(set(lang))
OVL, VOL = {"hin_Deva", "tur_Latn", "jpn_Jpan"}, {"eng_Latn", "fra_Latn", "rus_Cyrl", "zho_Hans"}
m0 = z["cond_0"][:, 0] - z["cond_0"][:, 1]


def delta(ci):
    s = z[f"cond_{ci}"]
    return (s[:, 0] - s[:, 1]) - m0


def find(layer, kind, k, r=-1):
    return next(i for i, c in enumerate(conds) if c["layer"] == layer and c["kind"] == kind and c["k"] == k and c["r"] == r)


def slope(layer, kind, ks, r=-1, sel=None):
    """per-language slope through the origin, then equal-weight mean over languages"""
    d = {k: delta(find(layer, kind, k, r)) for k in ks}
    per = {}
    for g in LANGS:
        m = (lang == g) if sel is None else (lang == g) & sel
        if m.sum() == 0:
            continue
        per[str(g)] = float(sum(k * d[k][m].mean() for k in ks) / sum(k * k for k in ks))
    return float(np.mean(list(per.values()))), per


res = {}
b_ov, per_ov = slope(14, "ov", [-2, 2])
rand = np.array([slope(14, "rand", [-2, 2], r)[0] for r in range(sum(c["kind"] == "rand" and c["k"] == 2 for c in conds))])
zz = (b_ov - rand.mean()) / rand.std(ddof=1)
p_emp = (1 + (rand >= b_ov).sum()) / (1 + len(rand))
res["primary"] = dict(b_ov=b_ov, rand_mean=float(rand.mean()), rand_sd=float(rand.std(ddof=1)), z=float(zz), p_emp=float(p_emp),
                      claim=bool(b_ov > 0 and p_emp < 0.05 and zz > 2.33), per_language=per_ov,
                      rand_slopes=rand.tolist())
print(f"PRIMARY L14 slope(k=+-2): b_OV {b_ov:+.4f} | random {rand.mean():+.4f} +- {rand.std(ddof=1):.4f} (min {rand.min():+.4f}, "
      f"max {rand.max():+.4f}) | z {zz:+.2f} | p_emp {p_emp:.3f} | CLAIM {res['primary']['claim']}")
print("   per language:", {g: round(v, 3) for g, v in per_ov.items()})
# secondary
b_ie, per_ie = slope(14, "ie", [-2, 2])
res["ie_L14"] = dict(b=b_ie, per_language=per_ie)
n_pos = sum(v > 0 for v in per_ov.values())
res["sign_test"] = dict(n_positive=n_pos, n=len(per_ov))
# pair bootstrap of b_OV(L14)
rs = np.random.RandomState(0)
dp, dm = delta(find(14, "ov", 2)), delta(find(14, "ov", -2))
boots = []
for _ in range(5000):
    vals = []
    for g in LANGS:
        ix = np.flatnonzero(lang == g)
        ix = ix[rs.randint(0, len(ix), len(ix))]
        vals.append((2 * dp[ix].mean() - 2 * dm[ix].mean()) / 8)
    boots.append(np.mean(vals))
res["b_ov_boot95"] = [float(x) for x in np.quantile(boots, [0.025, 0.975])]
print(f"IE control L14 b {b_ie:+.4f} | b_OV > 0 in {n_pos}/{len(per_ov)} languages | b_OV 95% CI {res['b_ov_boot95']}")
for L in (8, 14, 20):
    full, _ = slope(L, "ov", [-2, -1, 1, 2])
    fie, _ = slope(L, "ie", [-2, -1, 1, 2])
    ov_l = slope(L, "ov", [-2, -1, 1, 2], sel=np.isin(lang, list(OVL)))[0]
    vo_l = slope(L, "ov", [-2, -1, 1, 2], sel=np.isin(lang, list(VOL)))[0]
    curve = {k: float(np.mean([delta(find(L, "ov", k))[lang == g].mean() for g in LANGS])) for k in (-2, -1, 1, 2)}
    res[f"L{L}"] = dict(b_ov_all_k=full, b_ie_all_k=fie, b_ov_OVlangs=ov_l, b_ov_VOlangs=vo_l, dose_response=curve)
    print(f"L{L:2d}: b_OV(all k) {full:+.4f} | b_IE {fie:+.4f} | OV-langs {ov_l:+.4f} VO-langs {vo_l:+.4f} | "
          f"mean Delta by k " + " ".join(f"{k:+d}:{v:+.3f}" for k, v in curve.items()))
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_steer.json"), "w"), indent=1,
          default=float)
