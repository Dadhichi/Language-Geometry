"""EXPLORATORY (not pre-registered) follow-up to analysis_gen.py. usage: python exploratory_gen.py OUT_DIR

The pre-registered inclusion rule (>= 20 counted dependencies in EVERY needed condition) excluded all 8 languages,
because some of the 24 random directions at k* = 2 push a language out of itself almost entirely. Here inclusion is
decided per direction instead: a language enters the slope of direction d if base, +k* and -k* under d each have
>= 20 counted dependencies in language-matched continuations. Written after seeing the data; read as description.
"""
import collections, json, os, sys
import numpy as np

D = sys.argv[1]
conds = json.load(open(os.path.join(D, "conditions.json")))
cal = json.load(open(os.path.join(D, "calibration.json")))
G = [json.loads(l) for l in open(os.path.join(D, "gen.jsonl"), encoding="utf-8")]
ks = cal["kstar"]
LANGS = sorted({g["lang"] for g in G})
MIN_DEPS, NB = 20, 2000
by = collections.defaultdict(list)
for g in G:
    by[(g["cond"], g["lang"])].append(g)


def find(kind, k, r=-1):
    return next(i for i, c in enumerate(conds) if c["kind"] == kind and abs(c["k"] - k) < 1e-9 and c["r"] == r)


def counts(ci, lang, sents=None):
    """(n_OV, n_VO) summed over language-matched continuations, optionally over a resampled list of sentence ids."""
    rows = {g["sent"]: g for g in by[(ci, lang)] if g["match"]}
    ids = rows.keys() if sents is None else [s for s in sents if s in rows]
    return sum(rows[s]["n_ov"] for s in ids), sum(rows[s]["n_vo"] for s in ids)


def lang_slope(kind, lang, r=-1, k=ks, sents=None):
    cs = [counts(0, lang, sents), counts(find(kind, k, r), lang, sents), counts(find(kind, -k, r), lang, sents)]
    if any(a + b < MIN_DEPS for a, b in cs):
        return np.nan
    rate = [a / (a + b) for a, b in cs]
    return (rate[1] - rate[2]) / (2 * k)


DIRS = [("ov", -1), ("ie", -1)] + [("rand", r) for r in range(24)]
S = {d: np.array([lang_slope(d[0], g, d[1]) for g in LANGS]) for d in DIRS}       # direction -> per-language slope
R = np.array([S[("rand", r)] for r in range(24)])                                 # 24 x 8
res = {"note": "exploratory; per-direction inclusion", "kstar": ks, "langs": LANGS}

# E1: per-language OV slope against the random slopes defined for that language
print(f"E1 per-language slope per unit k (k* = {ks}); rank of OV among defined random directions")
res["per_language"] = {}
for j, g in enumerate(LANGS):
    rj = R[:, j][~np.isnan(R[:, j])]
    b = S[("ov", -1)][j]
    row = dict(b_ov=None if np.isnan(b) else float(b), b_ie=None if np.isnan(S[("ie", -1)][j]) else float(S[("ie", -1)][j]),
               n_rand=int(len(rj)), rand_mean=float(rj.mean()), rand_sd=float(rj.std(ddof=1)),
               n_rand_ge=int((rj >= b).sum()) if not np.isnan(b) else None,
               z=float((b - rj.mean()) / rj.std(ddof=1)) if not np.isnan(b) else None)
    res["per_language"][g] = row
    print(f"  {g}: b_OV {b:+.4f} | IE {S[('ie', -1)][j]:+.4f} | random {rj.mean():+.4f} +- {rj.std(ddof=1):.4f} "
          f"(n={len(rj)}) | random >= OV: {row['n_rand_ge']} | z {row['z'] if row['z'] is None else round(row['z'], 2)}")

# E2: pooled slope with per-direction inclusion (mean over the languages defined for that direction)
pooled = {d: float(np.nanmean(S[d])) for d in DIRS}
rp = np.array([pooled[("rand", r)] for r in range(24)])
b_ov = pooled[("ov", -1)]
z2 = (b_ov - rp.mean()) / rp.std(ddof=1)
p2 = (1 + (rp >= b_ov).sum()) / 25
res["pooled"] = dict(b_ov=b_ov, b_ie=pooled[("ie", -1)], rand=rp.tolist(), rand_mean=float(rp.mean()),
                     rand_sd=float(rp.std(ddof=1)), z=float(z2), p_emp=float(p2),
                     n_langs_ov=int((~np.isnan(S[("ov", -1)])).sum()),
                     n_langs_rand=[int((~np.isnan(S[("rand", r)])).sum()) for r in range(24)])
print(f"E2 pooled (per-direction inclusion): b_OV {b_ov:+.4f} over {res['pooled']['n_langs_ov']} langs | "
      f"IE {pooled[('ie', -1)]:+.4f} | random {rp.mean():+.4f} +- {rp.std(ddof=1):.4f} (max {rp.max():+.4f}) | "
      f"z {z2:+.2f} | p_emp {p2:.3f}")

# E3: paired comparison on the shared language set: for each random r, mean OV slope vs mean random slope
wins, diffs = 0, []
for r in range(24):
    m = ~np.isnan(S[("ov", -1)]) & ~np.isnan(S[("rand", r)])
    d = float(S[("ov", -1)][m].mean() - S[("rand", r)][m].mean())
    diffs.append(d)
    wins += d > 0
res["paired"] = dict(wins=int(wins), of=24, diffs=diffs)
print(f"E3 paired on shared language sets: OV above random in {wins}/24 (median difference {np.median(diffs):+.4f})")

# E4: prompt bootstrap of the pooled OV slope (languages fixed to those defined for OV in the full data)
rng = np.random.default_rng(0)
sent_ids = sorted({g["sent"] for g in G})
ok = [g for j, g in enumerate(LANGS) if not np.isnan(S[("ov", -1)][j])]
bs = []
for _ in range(NB):
    samp = list(rng.choice(sent_ids, len(sent_ids)))
    bs.append(np.nanmean([lang_slope("ov", g, sents=samp) for g in ok]))
res["b_ov_boot95"] = [float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))]
print(f"E4 prompt-bootstrap 95% CI of pooled b_OV: [{res['b_ov_boot95'][0]:+.4f}, {res['b_ov_boot95'][1]:+.4f}]")

# E5: language identity. Match rate per direction, and where off-language continuations go
res["match"] = {}
for g in LANGS:
    m = lambda ci: float(np.mean([x["match"] for x in by[(ci, g)]]))
    rr = [m(find("rand", s * ks, r)) for r in range(24) for s in (-1, 1)]
    res["match"][g] = dict(base=m(0), ov_m=m(find("ov", -ks)), ov_p=m(find("ov", ks)), ie_m=m(find("ie", -ks)),
                           ie_p=m(find("ie", ks)), rand=rr)
n_bad = sum(min(res["match"][g]["rand"]) < .5 for g in LANGS)
# E6: OV rate per language under each steered condition (rates with < MIN_DEPS counted deps reported as None)
def rate(ci, g):
    a, b = counts(ci, g)
    return (a / (a + b) if a + b >= MIN_DEPS else None), a + b


res["rates"] = {}
for g in LANGS:
    e = {name: rate(ci, g) for name, ci in [("base", 0), ("ov-", find("ov", -ks)), ("ov+", find("ov", ks)),
                                            ("ov-half", find("ov", -ks / 2)), ("ov+half", find("ov", ks / 2)),
                                            ("ie-", find("ie", -ks)), ("ie+", find("ie", ks))]}
    e["rand"] = [rate(find("rand", s * ks, r), g)[0] for r in range(24) for s in (-1, 1)]
    res["rates"][g] = e
    rr = [v for v in e["rand"] if v is not None]
    print(f"E6 {g[:3]} OV rate: base {e['base'][0]:.2f} (n {e['base'][1]}) | ov- {e['ov-'][0]} (n {e['ov-'][1]}) | "
          f"ov+ {e['ov+'][0]} (n {e['ov+'][1]}) | random 5-95% [{np.quantile(rr, .05):.2f}, {np.quantile(rr, .95):.2f}] (n {len(rr)})")
print(f"E5 random directions that keep < 50% language match, per language: "
      f"{ {g[:3]: sum(x < .5 for x in res['match'][g]['rand']) for g in LANGS} } (of 48) | languages affected {n_bad}/8")
# E7: is language loss under the OV direction sign-specific? "against" = the sign that pushes a language away from its
# own majority order (minus for languages whose unsteered OV rate > .5, plus otherwise); sign test over languages
ag, tw = [], []
for g in LANGS:
    mm = res["match"][g]
    own_ov = res["rates"][g]["base"][0] > .5
    ag.append(mm["ov_m"] if own_ov else mm["ov_p"])
    tw.append(mm["ov_p"] if own_ov else mm["ov_m"])
nz = [(a, t) for a, t in zip(ag, tw) if abs(a - t) > 1e-9]
lower = sum(a < t for a, t in nz)
from math import comb
p_sign = min(1.0, 2 * sum(comb(len(nz), i) for i in range(lower, len(nz) + 1)) / 2 ** len(nz))
res["sign_loss"] = dict(against=dict(zip(LANGS, ag)), toward=dict(zip(LANGS, tw)), lower=lower, n_nontied=len(nz),
                        p_sign=p_sign, mean_against=float(np.mean(ag)), mean_toward=float(np.mean(tw)))
print(f"E7 language match pushing AGAINST own order {np.mean(ag):.3f} vs TOWARD {np.mean(tw):.3f}; against lower in "
      f"{lower}/{len(nz)} non-tied languages (two-sided sign test p = {p_sign:.3f})")
# E8: OV slope at half strength (no random directions were run at k*/2, so no null)
bh = {g: lang_slope("ov", g, k=ks / 2) for g in LANGS}
res["half_k"] = {g: (None if np.isnan(v) else float(v)) for g, v in bh.items()}
print(f"E8 OV slope at k*/2 per language: { {g[:3]: round(v, 4) for g, v in bh.items()} } | mean over defined "
      f"{np.nanmean(list(bh.values())):+.4f}")
# E9: object type, if the re-parse (reparse_objtype.py) is present. Same slope, counting nominal objects only
op = os.path.join(os.path.dirname(os.path.abspath(__file__)), "objtype.json")
if os.path.exists(op):
    O = json.load(open(op))
    def nslope(g, a, b, cls="nom"):
        cs = [O[g][c] for c in ("base", a, b)]
        n = [c.get(f"{cls}_ov", 0) + c.get(f"{cls}_vo", 0) for c in cs]
        if min(n) < MIN_DEPS:
            return np.nan
        r = [c.get(f"{cls}_ov", 0) / m for c, m in zip(cs, n)]
        return (r[1] - r[2]) / (2 * ks)
    for cls in ("nom", "pron"):
        so = {g: nslope(g, "ov+", "ov-", cls) for g in O}
        sr = np.array([[nslope(g, f"r{r}+", f"r{r}-", cls) for g in O] for r in range(24)])
        bo = np.nanmean(list(so.values()))
        pr = np.nanmean(sr, axis=1)
        res[f"objtype_{cls}"] = dict(b_ov=float(bo), per_language={g: (None if np.isnan(v) else float(v)) for g, v in so.items()},
                                     rand=[float(v) for v in pr], z=float((bo - np.nanmean(pr)) / np.nanstd(pr, ddof=1)),
                                     n_rand_ge=int((pr >= bo).sum()))
        print(f"E9 {cls} objects only: b_OV {bo:+.4f} | per language { {g[:3]: round(v, 3) for g, v in so.items()} } | "
              f"random {np.nanmean(pr):+.4f} +- {np.nanstd(pr, ddof=1):.4f} (max {np.nanmax(pr):+.4f}) | "
              f"z {res[f'objtype_{cls}']['z']:+.2f} | random >= OV {res[f'objtype_{cls}']['n_rand_ge']}/24")
    # repetition (1 - distinct word-bigram ratio), mean over continuations: unsteered vs OV +-k* vs IE vs random
    rep = lambda cs: sum(c["rep_sum"] for c in cs) / sum(c["n_rows"] for c in cs)
    rr = {g: dict(base=rep([O[g]["base"]]), ov=rep([O[g]["ov-"], O[g]["ov+"]]), ie=rep([O[g]["ie-"], O[g]["ie+"]]),
                  rand=rep([O[g][k] for k in O[g] if k[0] == "r" and k[1:-1].isdigit()]),
                  rand_range=[min(rep([O[g][k]]) for k in O[g] if k[0] == "r" and k[1:-1].isdigit()),
                              max(rep([O[g][k]]) for k in O[g] if k[0] == "r" and k[1:-1].isdigit())]) for g in O}
    res["repetition"] = rr
    print("E9 repetition, mean over languages: unsteered %.3f | word order +-k* %.3f | IE %.3f | random %.3f" %
          tuple(np.mean([rr[g][k] for g in rr]) for k in ("base", "ov", "ie", "rand")))
    agree = {g: O[g]["_agree"] for g in O}
    res["objtype_agree"] = agree
    print(f"E9 re-parse reproduces the original counts in { {g[:3]: f'{a}/{b}' for g, (a, b) in agree.items()} }")
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exploratory_gen.json"), "w"), indent=1)
