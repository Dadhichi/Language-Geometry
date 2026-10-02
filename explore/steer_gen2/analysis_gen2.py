"""PRE-REGISTERED analysis of the confirmatory free-generation test (PREREG.md).
usage: python analysis_gen2.py OUT_DIR        -> results_gen2.json + printed report
       python analysis_gen2.py --selftest     -> synthetic power / false-positive check (no data needed)

Measure: noun-object share rho_l(c) = sum nom_ov / sum (nom_ov + nom_vo) over language-matched continuations of
language l under condition c. Slope b_l(v) = (rho_l(+k v) - rho_l(-k v)) / (2k), defined only if base, +k v and -k v
each have >= MIN_PAIRS noun-object pairs (per-direction inclusion). Pooled slope over a language set = mean of the
defined b_l. Claim for a hypothesis: evaluable (>= MIN_LANGS[H] languages defined for the word-order direction and
>= MIN_RAND random directions with a defined pooled slope), b_OV above every defined random pooled slope, and
z = (b_OV - mean_r) / sd_r > Z_CRIT."""
import json, os, sys
import numpy as np

K = 2.0
MIN_PAIRS = 20
SETS = {"H1": ["deu_Latn", "rus_Cyrl"], "H2": ["nld_Latn", "ukr_Cyrl", "pol_Latn", "hrv_Latn"]}
CONTROLS = ["eng_Latn", "spa_Latn", "kor_Hang"]
MIN_LANGS = {"H1": 2, "H2": 3}
MIN_RAND = 20
Z_CRIT = 2.58
# secondary directional predictions: "up" = object-first share rises under +k; "down" = falls under -k
PRED = {"rus_Cyrl": "up", "ukr_Cyrl": "up", "pol_Latn": "up", "hrv_Latn": "up", "deu_Latn": "down", "nld_Latn": "down",
        "eng_Latn": "none", "spa_Latn": "none", "kor_Hang": "none"}


def analyze(conds, G, n_rand=24):
    def find(kind, k, r=-1):
        return next(i for i, c in enumerate(conds) if c["kind"] == kind and abs(c["k"] - k) < 1e-9 and c["r"] == r)
    langs = sorted({g["lang"] for g in G})
    agg = {}
    for g in G:
        a = agg.setdefault((g["cond"], g["lang"]), dict(nov=0, nvo=0, pov=0, pvo=0, aov=0, avo=0, n=0, m=0, rep=0.0))
        a["n"] += 1
        a["rep"] += rep(g["cont"])
        if g["match"]:
            a["m"] += 1
            a["nov"] += g["nom_ov"]; a["nvo"] += g["nom_vo"]
            a["pov"] += g["pron_ov"]; a["pvo"] += g["pron_vo"]
            a["aov"] += g["n_ov"]; a["avo"] += g["n_vo"]
    E = dict(nov=0, nvo=0, pov=0, pvo=0, aov=0, avo=0, n=0, m=0, rep=0.0)

    def share(ci, l, pre="n"):
        a = agg.get((ci, l), E)
        tot = a[pre + "ov"] + a[pre + "vo"]
        return (a[pre + "ov"] / tot if tot else None), tot

    def slope(kind, l, r=-1):
        (s0, n0), (sp, npl), (sm, nm) = share(0, l), share(find(kind, K, r), l), share(find(kind, -K, r), l)
        if min(n0, npl, nm) < MIN_PAIRS:
            return np.nan
        return (sp - sm) / (2 * K)

    DIRS = [("ov", -1), ("ie", -1)] + [("rand", r) for r in range(n_rand)]
    S = {d: {l: slope(d[0], l, d[1]) for l in langs} for d in DIRS}
    res = {"k": K, "langs": langs, "sets": dict(SETS, controls=CONTROLS), "tests": {}, "per_language": {}}
    for H, L in SETS.items():
        L = [l for l in L if l in langs]
        pooled = {d: (np.nanmean([S[d][l] for l in L]) if any(not np.isnan(S[d][l]) for l in L) else np.nan) for d in DIRS}
        rnd = np.array([pooled[("rand", r)] for r in range(n_rand)])
        rd = rnd[~np.isnan(rnd)]
        b = pooled[("ov", -1)]
        n_l = int(sum(not np.isnan(S[("ov", -1)][l]) for l in L))
        ev = bool(n_l >= MIN_LANGS[H] and len(rd) >= MIN_RAND and not np.isnan(b))
        z = float((b - rd.mean()) / rd.std(ddof=1)) if ev else float("nan")
        p_emp = float((1 + (rd >= b).sum()) / (1 + len(rd))) if ev else float("nan")
        res["tests"][H] = dict(b_ov=float(b), b_ie=float(pooled[("ie", -1)]), rand=[float(x) for x in rnd],
                               rand_mean=float(rd.mean()) if len(rd) else float("nan"),
                               rand_sd=float(rd.std(ddof=1)) if len(rd) > 1 else float("nan"), z=z, p_emp=p_emp,
                               n_langs=n_l, n_rand_defined=int(len(rd)), evaluable=ev,
                               claim=bool(ev and b > rd.max() and z > Z_CRIT))
    for l in langs:
        rl = np.array([S[("rand", r)][l] for r in range(n_rand)])
        rl = rl[~np.isnan(rl)]
        b = S[("ov", -1)][l]
        s0, sp, sm = share(0, l)[0], share(find("ov", K), l)[0], share(find("ov", -K), l)[0]
        ok = None
        if not np.isnan(b):
            if PRED[l] == "up":
                ok = bool(sp - s0 > 0)
            elif PRED[l] == "down":
                ok = bool(sm - s0 < 0)
            elif len(rl) >= 10:
                lo, hi = np.quantile(rl, [.025, .975])
                ok = bool(lo <= b <= hi)
        res["per_language"][l] = dict(b_ov=float(b), b_ie=float(S[("ie", -1)][l]), pred=PRED[l], pred_ok=ok,
                                      rand_mean=float(rl.mean()) if len(rl) else float("nan"),
                                      rand_sd=float(rl.std(ddof=1)) if len(rl) > 1 else float("nan"),
                                      z=float((b - rl.mean()) / rl.std(ddof=1)) if len(rl) > 1 and not np.isnan(b) else float("nan"))
    # figure-ready rates (noun-object share, n) and language-match rates, same shape as steer_gen/exploratory_gen.json
    named = {"base": 0, "ov-": find("ov", -K), "ov+": find("ov", K), "ie-": find("ie", -K), "ie+": find("ie", K)}
    res["rates"], res["match"], res["pron"], res["all_obj"], res["repetition"] = {}, {}, {}, {}, {}
    mrate = lambda ci, l: agg.get((ci, l), E)["m"] / max(1, agg.get((ci, l), E)["n"])
    for l in langs:
        res["rates"][l] = {k: list(share(ci, l)) for k, ci in named.items()}
        res["rates"][l]["rand"] = [share(find("rand", s * K, r), l)[0] if share(find("rand", s * K, r), l)[1] >= MIN_PAIRS
                                   else None for r in range(n_rand) for s in (-1, 1)]
        res["pron"][l] = {k: list(share(ci, l, "p")) for k, ci in named.items()}
        res["all_obj"][l] = {k: list(share(ci, l, "a")) for k, ci in named.items()}
        res["match"][l] = dict(base=mrate(0, l), ov_m=mrate(named["ov-"], l), ov_p=mrate(named["ov+"], l),
                               ie_m=mrate(named["ie-"], l), ie_p=mrate(named["ie+"], l),
                               rand=[mrate(find("rand", s * K, r), l) for r in range(n_rand) for s in (-1, 1)])
        rp = lambda cis: sum(agg.get((c, l), E)["rep"] for c in cis) / max(1, sum(agg.get((c, l), E)["n"] for c in cis))
        res["repetition"][l] = dict(base=rp([0]), ov=rp([named["ov-"], named["ov+"]]), ie=rp([named["ie-"], named["ie+"]]),
                                    rand=rp([find("rand", s * K, r) for r in range(n_rand) for s in (-1, 1)]))
    # secondary: sign-specific language loss (against = the sign pushing a language away from its own majority order)
    ag, tw = [], []
    for l in langs:
        s0 = share(0, l)[0]
        s0 = share(0, l, "a")[0] if s0 is None else s0
        if s0 is None:
            continue
        m = res["match"][l]
        ag.append(m["ov_m"] if s0 > .5 else m["ov_p"])
        tw.append(m["ov_p"] if s0 > .5 else m["ov_m"])
    nz = [(a, t) for a, t in zip(ag, tw) if abs(a - t) > 1e-9]
    lower = sum(a < t for a, t in nz)
    from math import comb
    p_sign = min(1.0, 2 * sum(comb(len(nz), i) for i in range(lower, len(nz) + 1)) / 2 ** len(nz)) if nz else float("nan")
    res["sign_loss"] = dict(mean_against=float(np.mean(ag)), mean_toward=float(np.mean(tw)), lower=lower,
                            n_nontied=len(nz), p_sign=p_sign)
    return res


def rep(text):
    w = text.split()
    b = list(zip(w, w[1:]))
    return 1 - len(set(b)) / len(b) if b else 0.0


def report(res):
    for H, t in res["tests"].items():
        print(f"{H} {res['sets'][H]}: b_OV {t['b_ov']:+.4f} over {t['n_langs']} langs | IE {t['b_ie']:+.4f} | random "
              f"{t['rand_mean']:+.4f} +- {t['rand_sd']:.4f} (n {t['n_rand_defined']}) | z {t['z']:+.2f} | p_emp "
              f"{t['p_emp']:.3f} | evaluable {t['evaluable']} | CLAIM {t['claim']}")
    for l, p in res["per_language"].items():
        r = res["rates"][l]
        print(f"  {l}: noun-object share -k {r['ov-'][0]} (n {r['ov-'][1]}) | 0 {r['base'][0]} (n {r['base'][1]}) | +k "
              f"{r['ov+'][0]} (n {r['ov+'][1]}) | slope {p['b_ov']:+.4f} z {p['z']:+.2f} | pred {p['pred']} ok {p['pred_ok']}")
    s = res["sign_loss"]
    print(f"sign-specific language loss: against {s['mean_against']:.3f} vs toward {s['mean_toward']:.3f}, lower in "
          f"{s['lower']}/{s['n_nontied']} (p {s['p_sign']:.3f})")


def synth(rng, effect, langs, n_rand=24, n_sent=150):
    """Synthetic gen rows. Base noun-object share per language; each direction shifts the share by a slope; random
    slopes ~ N(0, 0.01) per (direction, language), like the exploratory data; effect = word-order slope per language."""
    import sys as _s
    _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gen_ov2 import conditions
    conds = conditions(n_rand)
    base = {l: rng.uniform(.02, .9) for l in langs}
    rs = {(r, l): rng.normal(0, .01) for r in range(n_rand) for l in langs}
    G = []
    for ci, c in enumerate(conds):
        for l in langs:
            if c["kind"] == "base":
                sl = 0.0
            elif c["kind"] == "ov":
                sl = effect.get(l, 0.0)
            elif c["kind"] == "ie":
                sl = rng.normal(0, .01)
            else:
                sl = rs[(c["r"], l)]
            p = float(np.clip(base[l] + sl * c["k"], .001, .999))
            for s in range(n_sent):
                n = rng.poisson(0.6)
                ov = rng.binomial(n, p)
                G.append(dict(cond=ci, lang=l, sent=s, cont="a b c d e f", match=bool(rng.random() < .95), nom_ov=int(ov),
                              nom_vo=int(n - ov), pron_ov=0, pron_vo=0, n_ov=int(ov), n_vo=int(n - ov)))
    return conds, G


def selftest():
    langs = SETS["H1"] + SETS["H2"] + CONTROLS
    rng = np.random.default_rng(0)
    # power: exploratory effect sizes for H1 (deu 0.18 as a fall, i.e. +0.18 slope; rus 0.077), half of them for H2
    eff = {"deu_Latn": .18, "rus_Cyrl": .077, "nld_Latn": .09, "ukr_Cyrl": .04, "pol_Latn": .04, "hrv_Latn": .04}
    hits = {"H1": 0, "H2": 0}
    for _ in range(20):
        conds, G = synth(rng, eff, langs)
        r = analyze(conds, G)
        for H in hits:
            hits[H] += r["tests"][H]["claim"]
    print(f"power (20 sims): H1 {hits['H1']}/20, H2 {hits['H2']}/20")
    # false positives: word-order direction behaves like a random direction
    fp = {"H1": 0, "H2": 0}
    N = 100
    for _ in range(N):
        conds, G = synth(rng, {l: rng.normal(0, .01) for l in langs}, langs)
        r = analyze(conds, G)
        for H in fp:
            fp[H] += r["tests"][H]["claim"]
    print(f"false-positive rate under the null ({N} sims): H1 {fp['H1']}/{N}, H2 {fp['H2']}/{N} (bound 1/25 = 4%)")
    return hits, fp


if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        selftest()
        sys.exit(0)
    D = sys.argv[1]
    conds = json.load(open(os.path.join(D, "conditions.json")))
    G = [json.loads(l) for l in open(os.path.join(D, "gen.jsonl"), encoding="utf-8")]
    res = analyze(conds, G)
    report(res)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_gen2.json"), "w"), indent=1)
