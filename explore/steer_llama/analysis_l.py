"""PRE-REGISTERED analysis of free-generation steering in Llama-3.1-8B (PREREG.md).
usage: python analysis_l.py OUT_DIR      -> results_l.json + report
       python analysis_l.py --selftest
Helpers (noun-object share, per-direction inclusion, slope, language match, retention test C) are imported from
explore/steer_within/analysis_w.py, so the definitions are identical to the Qwen studies."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "steer_within"))
from analysis_w import load_gen, add_row, share, slope, match, test_C, K, MIN_PAIRS, MIN_RAND, Z_CRIT, F, ALL9

H1 = ["deu_Latn", "rus_Cyrl"]
H2 = ["nld_Latn", "ukr_Cyrl", "pol_Latn", "hrv_Latn"]


def pooled_test(agg, kind, langs, min_langs, n_rand=24):
    """Pooled slope of `kind` over `langs` against the 24 random directions of the same run (same inclusion rule)."""
    bk = {l: slope(agg, kind, l) for l in langs}
    rnd = []
    for r in range(n_rand):
        v = [x for x in (slope(agg, "rand", l, r) for l in langs) if not np.isnan(x)]
        rnd.append(np.mean(v) if v else np.nan)
    rnd = np.array(rnd)
    rd = rnd[~np.isnan(rnd)]
    defined = [x for x in bk.values() if not np.isnan(x)]
    b = float(np.mean(defined)) if defined else np.nan
    ev = bool(len(defined) >= min_langs and len(rd) >= MIN_RAND)
    z = float((b - rd.mean()) / rd.std(ddof=1)) if ev else np.nan
    return dict(kind=kind, langs=langs, b=b, per_language=bk, n_langs=len(defined), rand=rnd.tolist(),
                n_rand_defined=int(len(rd)), rand_mean=float(rd.mean()) if len(rd) else np.nan,
                rand_sd=float(rd.std(ddof=1)) if len(rd) > 1 else np.nan, rand_max=float(rd.max()) if len(rd) else np.nan,
                z=z, p_emp=float((1 + (rd >= b).sum()) / (1 + len(rd))) if ev else np.nan, evaluable=ev,
                claim=bool(ev and b > rd.max() and z > Z_CRIT))


def analyze(D):
    _, agg = load_gen(D)
    res = {"model": "meta-llama/Llama-3.1-8B", "layer": 16, "k": K}
    res["L_H1"] = pooled_test(agg, "ov", H1, 2)                      # primary
    res["L_B"] = pooled_test(agg, "w", F, 3)                          # primary
    res["L_C"] = test_C(agg, res["L_B"]["claim"])                     # primary, conditional on L_B
    res["H2_ov"] = pooled_test(agg, "ov", H2, 3)                      # secondary
    res["dose"] = {kd: {str(k): float(np.nanmean([slope(agg, kd, l, k=k) for l in F])) for k in (1.0, 2.0, 3.0)}
                   for kd in ("ov", "w")}
    names = [("base", "base", 0)] + [(f"{kd}{k:+g}", kd, k) for kd in ("ov", "w") for k in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)]
    res["rates"] = {l: {n: list(share(agg, kd, k, -1, l)) for n, kd, k in names} for l in ALL9}
    res["rates_rand"] = {l: [share(agg, "rand", s * K, r, l)[0] if share(agg, "rand", s * K, r, l)[1] >= MIN_PAIRS else None
                             for r in range(24) for s in (-1, 1)] for l in ALL9}
    res["match"] = {l: {n: match(agg, kd, k, l) for n, kd, k in names} for l in ALL9}
    res["match_rand"] = {l: [match(agg, "rand", s * K, l, r) for r in range(24) for s in (-1, 1)] for l in ALL9}
    res["repetition"] = {l: {n: agg[(kd, float(k), -1, l)]["rep"] / agg[(kd, float(k), -1, l)]["n"] for n, kd, k in names
                             if (kd, float(k), -1, l) in agg} for l in ALL9}
    return res


def report(res):
    for key in ("L_H1", "L_B", "H2_ov"):
        t = res[key]
        print(f"{key} ({t['kind']} over {[l[:3] for l in t['langs']]}): b {t['b']:+.4f} over {t['n_langs']} langs | random "
              f"{t['rand_mean']:+.4f} +- {t['rand_sd']:.4f} max {t['rand_max']:+.4f} (n {t['n_rand_defined']}) | z {t['z']:+.2f} | "
              f"evaluable {t['evaluable']} | CLAIM {t['claim']}")
        print("   per language:", {l[:3]: round(v, 4) for l, v in t["per_language"].items()})
    C = res["L_C"]
    print(f"L_C retention (against sign): w {C['mean_w']:.3f} vs ov {C['mean_ov']:.3f}; w higher in {C['higher']}/{C['n']} "
          f"(sign p {C['p_sign']:.3f}) | CLAIM {C['claim']}")
    print("dose (pooled slope over the flexible set by k):", {kd: {k: round(v, 4) for k, v in d.items()} for kd, d in res["dose"].items()})


def selftest():
    rng = np.random.default_rng(3)
    conds = [dict(kind="base", k=0, r=-1)] + [dict(kind=kd, k=k, r=-1) for kd in ("ov", "w") for k in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)] \
        + [dict(kind="rand", k=s * 2.0, r=r) for r in range(24) for s in (-1, 1)]
    for planted in (True, False):
        agg = {}
        rs = {(r, l): rng.normal(0, .01) for r in range(24) for l in ALL9}
        for ci, c in enumerate(conds):
            for l in ALL9:
                if c["kind"] == "base":
                    sl, keep = 0.0, .95
                elif c["kind"] == "rand":
                    sl, keep = rs[(c["r"], l)], .9
                elif c["kind"] == "ov":
                    sl, keep = (.1 if planted and l in H1 else rng.normal(0, .01)), .6
                else:
                    sl, keep = (.08 if planted and l in F else rng.normal(0, .01)), .95
                p = float(np.clip(.4 + sl * c["k"], .01, .99))
                for s in range(150):
                    n = rng.poisson(.6); o = rng.binomial(n, p)
                    add_row(agg, conds, dict(cond=ci, lang=l, sent=s, cont="a b c", match=bool(rng.random() < keep),
                                             nom_ov=int(o), nom_vo=int(n - o), n_ov=int(o), n_vo=int(n - o)))
        h1, b = pooled_test(agg, "ov", H1, 2), pooled_test(agg, "w", F, 3)
        c = test_C(agg, b["claim"])
        print(f"planted={planted}: L_H1 claim {h1['claim']} (z {h1['z']:+.1f}) | L_B claim {b['claim']} (z {b['z']:+.1f}) | "
              f"L_C claim {c['claim']} ({c['higher']}/{c['n']})")


if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        selftest()
        sys.exit(0)
    res = analyze(sys.argv[1])
    report(res)
    clean = lambda o: {str(k): clean(v) for k, v in o.items()} if isinstance(o, dict) else [clean(v) for v in o] if isinstance(o, list) \
        else (None if isinstance(o, float) and not np.isfinite(o) else float(o) if isinstance(o, np.floating) else bool(o) if isinstance(o, np.bool_) else o)
    json.dump(clean(res), open(os.path.join(HERE, "results_l.json"), "w"), indent=1)
