"""PRE-REGISTERED analysis of the within-language word-order direction (PREREG.md).
usage: python analysis_w.py W_OUT_DIR GEN2_OUT_DIR DERIVED34_DIR      -> results_w.json + report
       python analysis_w.py --selftest
W_OUT_DIR: this study's job output (dW.npz, dW_info.json, gen/conditions.json, gen/gen.jsonl).
GEN2_OUT_DIR: explore/steer_gen2 job output (its 24 random directions are the null for B; same prompts).
DERIVED34_DIR: 34-language centroids (beta_OV and the language span for the alignment null)."""
import json, os, sys
from math import comb
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prereg34"))
LAYER, K, MIN_PAIRS, MIN_RAND, Z_CRIT, N_NULL, Q_A = 14, 2.0, 20, 20, 2.58, 10000, 0.999
F = ["deu_Latn", "nld_Latn", "rus_Cyrl", "ukr_Cyrl", "pol_Latn", "hrv_Latn"]
ALL9 = F + ["eng_Latn", "spa_Latn", "kor_Hang"]


# ---------------------------------------------------------------- A: alignment
def betas(A, T, fert):
    ov = np.array([i in T.OV for i in range(T.N)], float)
    ie = np.array([i in set(T.GLOTTO[0]) for i in range(T.N)], float)
    scr = np.column_stack([[float(i in S) for i in range(T.N)] for S in T.SCRIPT_SPLITS])
    X = np.column_stack([np.ones(T.N), ov, ie, scr, np.log(fert)])
    B, *_ = np.linalg.lstsq(X, A, rcond=None)
    return B[1], B[2]


def span_basis(A):
    _, s, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)
    return Vt[s > s[0] * 1e-8]


def alignment(d, b_ov, b_ie, U, rng, n_null=N_NULL):
    cos = lambda x, y: float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
    R = rng.normal(size=(n_null, U.shape[0])) @ U
    null = (R @ d) / np.linalg.norm(R, axis=1) / np.linalg.norm(d)
    c = cos(d, b_ov)
    Pd = U.T @ (U @ d)
    return dict(cos_ov=c, cos_ie=cos(d, b_ie), null_q999=float(np.quantile(null, Q_A)), null_sd=float(null.std()),
                p=float((1 + (null >= c).sum()) / (1 + n_null)), span_share=float(Pd @ Pd / (d @ d)),
                cos_ov_within_span=cos(Pd, b_ov))


# ---------------------------------------------------------------- B, C: generation
def load_gen(D):
    conds = json.load(open(os.path.join(D, "conditions.json")))
    agg = {}
    for line in open(os.path.join(D, "gen.jsonl"), encoding="utf-8"):
        g = json.loads(line)
        add_row(agg, conds, g)
    return conds, agg


def add_row(agg, conds, g):
    c = conds[g["cond"]]
    key = (c["kind"], float(c["k"]), c["r"], g["lang"])
    a = agg.setdefault(key, dict(nov=0, nvo=0, aov=0, avo=0, n=0, m=0, rep=0.0, conts={}))
    a["n"] += 1
    a["rep"] += rep(g["cont"])
    a["conts"][g["sent"]] = g["cont"]
    if g["match"]:
        a["m"] += 1
        a["nov"] += g["nom_ov"]; a["nvo"] += g["nom_vo"]; a["aov"] += g["n_ov"]; a["avo"] += g["n_vo"]


def rep(text):
    w = text.split()
    b = list(zip(w, w[1:]))
    return 1 - len(set(b)) / len(b) if b else 0.0


def share(agg, kind, k, r, l, pre="n"):
    a = agg.get((kind, float(k), r, l))
    if a is None:
        return None, 0
    tot = a[pre + "ov"] + a[pre + "vo"]
    return (a[pre + "ov"] / tot if tot else None), tot


def slope(agg, kind, l, r=-1, k=K):
    (_, n0), (sp, npl), (sm, nm) = share(agg, "base", 0, -1, l), share(agg, kind, k, r, l), share(agg, kind, -k, r, l)
    return np.nan if min(n0, npl, nm) < MIN_PAIRS else (sp - sm) / (2 * k)


def match(agg, kind, k, l, r=-1):
    a = agg.get((kind, float(k), r, l))
    return a["m"] / a["n"] if a and a["n"] else np.nan


def test_B(aggW, agg2, langs=F, n_rand=24):
    bw = {l: slope(aggW, "w", l) for l in langs}
    rnd = []
    for r in range(n_rand):
        v = [slope(agg2, "rand", l, r) for l in langs]
        v = [x for x in v if not np.isnan(x)]
        rnd.append(np.mean(v) if v else np.nan)
    rnd = np.array(rnd)
    rd = rnd[~np.isnan(rnd)]
    defined = [x for x in bw.values() if not np.isnan(x)]
    b = float(np.mean(defined)) if defined else np.nan
    ev = bool(len(defined) >= 3 and len(rd) >= MIN_RAND)
    z = float((b - rd.mean()) / rd.std(ddof=1)) if ev else np.nan
    return dict(b_w=b, per_language=bw, n_langs=len(defined), rand=rnd.tolist(), n_rand_defined=int(len(rd)),
                rand_mean=float(rd.mean()) if len(rd) else np.nan, rand_max=float(rd.max()) if len(rd) else np.nan,
                z=z, p_emp=float((1 + (rd >= b).sum()) / (1 + len(rd))) if ev else np.nan, evaluable=ev,
                claim=bool(ev and b > rd.max() and z > Z_CRIT))


def test_C(aggW, claim_B, langs=ALL9):
    rows = {}
    for l in langs:
        s0 = share(aggW, "base", 0, -1, l)[0]
        s0 = share(aggW, "base", 0, -1, l, "a")[0] if s0 is None else s0
        if s0 is None:
            continue
        ag = -K if s0 > .5 else K                                     # the sign that pushes against the language's order
        rows[l] = dict(against_k=ag, match_w=match(aggW, "w", ag, l), match_ov=match(aggW, "ov", ag, l))
    higher = sum(r["match_w"] > r["match_ov"] for r in rows.values())
    nz = sum(r["match_w"] != r["match_ov"] for r in rows.values())
    p = min(1.0, 2 * sum(comb(nz, i) for i in range(higher, nz + 1)) / 2 ** nz) if nz else np.nan
    return dict(per_language=rows, higher=int(higher), n=len(rows), p_sign=p,
                mean_w=float(np.mean([r["match_w"] for r in rows.values()])),
                mean_ov=float(np.mean([r["match_ov"] for r in rows.values()])),
                claim=bool(claim_B and higher >= 8 and len(rows) == 9))


def analyze(W_OUT, GEN2, DER):
    import lib34 as T
    res = {}
    dW = np.load(os.path.join(W_OUT, "dW.npz"))
    cent = np.load(os.path.join(DER, "centroids.npz"))
    fert = np.load(os.path.join(DER, "tokstats.npz"))["fert"]
    rng = np.random.default_rng(0)
    prof = []
    for L in range(dW["order_all"].shape[0]):
        A = cent[f"dev_L{L}"].astype(np.float64)
        b_ov, b_ie = betas(A, T, fert)
        al = alignment(dW["order_all"][L].astype(np.float64), b_ov, b_ie, span_basis(A), rng)
        al["layer"] = L
        al["cos_unnat_ov"] = float(dW["unnat_all"][L] @ b_ov / np.linalg.norm(dW["unnat_all"][L]) / np.linalg.norm(b_ov))
        prof.append(al)
    a14 = prof[LAYER]
    res["A"] = dict(a14, claim=bool(a14["cos_ov"] > 0 and a14["cos_ov"] > a14["null_q999"]))
    res["A_profile"] = prof
    res["dW_info"] = json.load(open(os.path.join(W_OUT, "dW_info.json")))
    condsW, aggW = load_gen(os.path.join(W_OUT, "gen"))
    _, agg2 = load_gen(GEN2)
    res["B"] = test_B(aggW, agg2)
    res["C"] = test_C(aggW, res["B"]["claim"])
    # secondary
    res["dose"] = {str(k): float(np.nanmean([slope(aggW, "w", l, k=k) for l in F])) for k in (1.0, 2.0, 3.0)}
    res["ov_rerun"] = {l: slope(aggW, "ov", l) for l in ALL9}
    res["rates"] = {l: {name: list(share(aggW, kind, k, -1, l)) for name, kind, k in
                        [("base", "base", 0), ("w-", "w", -K), ("w+", "w", K), ("ov-", "ov", -K), ("ov+", "ov", K),
                         ("w-3", "w", -3.0), ("w+3", "w", 3.0), ("w-1", "w", -1.0), ("w+1", "w", 1.0)]} for l in ALL9}
    res["match"] = {l: {name: match(aggW, kind, k, l) for name, kind, k in
                        [("base", "base", 0), ("w-", "w", -K), ("w+", "w", K), ("ov-", "ov", -K), ("ov+", "ov", K),
                         ("w-3", "w", -3.0), ("w+3", "w", 3.0)]} for l in ALL9}
    res["repetition"] = {l: {name: (aggW[(kind, float(k), -1, l)]["rep"] / aggW[(kind, float(k), -1, l)]["n"])
                             for name, kind, k in [("base", "base", 0), ("w", "w", K), ("w_neg", "w", -K), ("ov", "ov", K), ("ov_neg", "ov", -K)]}
                         for l in ALL9}
    same = lambda key: np.mean([aggW[key]["conts"].get(s) == c for s, c in agg2[key]["conts"].items()])
    res["determinism"] = {f"{kind}{k:+g}": float(np.mean([same((kind, float(k), -1, l)) for l in ALL9]))
                          for kind, k in [("base", 0), ("ov", -K), ("ov", K)]}
    return res


def report(res):
    A = res["A"]
    print(f"A alignment L{LAYER}: cos(d_W, beta_OV) {A['cos_ov']:+.3f} | null 99.9% {A['null_q999']:+.3f} (sd {A['null_sd']:.3f}) | "
          f"p {A['p']:.4f} | cos(d_W, beta_IE) {A['cos_ie']:+.3f} | span share {A['span_share']:.3f} | within-span cos "
          f"{A['cos_ov_within_span']:+.3f} | CLAIM {A['claim']}")
    print("  profile cos_ov by layer:", " ".join(f"{p['layer']}:{p['cos_ov']:+.2f}" for p in res["A_profile"]))
    B = res["B"]
    print(f"B reordering (noun objects, flexible set): b_w {B['b_w']:+.4f} over {B['n_langs']} langs | random "
          f"{B['rand_mean']:+.4f} max {B['rand_max']:+.4f} (n {B['n_rand_defined']}) | z {B['z']:+.2f} | evaluable {B['evaluable']} | CLAIM {B['claim']}")
    print("  per language:", {l[:3]: round(v, 4) for l, v in B["per_language"].items()})
    C = res["C"]
    print(f"C retention (against sign): w {C['mean_w']:.3f} vs ov {C['mean_ov']:.3f}; w higher in {C['higher']}/{C['n']} "
          f"(sign p {C['p_sign']:.3f}) | CLAIM {C['claim']}")
    print("  dose (pooled slope over flexible set by k):", res["dose"], "| determinism:", res["determinism"])


def selftest():
    import lib34 as T
    rng = np.random.default_rng(1)
    A = rng.normal(size=(T.N, 64))
    fert = np.exp(rng.normal(size=T.N) * .2)
    b_ov, b_ie = betas(A, T, fert)
    U = span_basis(A)
    hit = alignment(b_ov / np.linalg.norm(b_ov) + 0.3 * rng.normal(size=64) / 8, b_ov, b_ie, U, rng, 4000)
    nulls = [alignment(d, b_ov, b_ie, U, rng, 2000) for d in rng.normal(size=(200, 64))]
    fpr = np.mean([a["cos_ov"] > a["null_q999"] for a in nulls])
    print(f"A selftest: planted d ~ beta_OV -> cos {hit['cos_ov']:+.2f} vs null 99.9% {hit['null_q999']:+.2f}; "
          f"random d -> false-positive rate {fpr:.3f} (target 0.001)")
    # B / C: synthetic generations; w reorders like the exploratory effect and keeps the language, random as null
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "steer_gen2"))
    import analysis_gen2 as A2
    conds2, G2 = A2.synth(rng, {}, ALL9)
    agg2 = {}
    for g in G2:
        add_row(agg2, conds2, g)
    condsW = [dict(kind="base", k=0, r=-1)] + [dict(kind=kd, k=k, r=-1) for kd in ("w", "ov") for k in (-K, K)]
    aggW = {}
    eff = {l: (.12 if l in F else 0.0) for l in ALL9}
    for ci, c in enumerate(condsW):
        for l in ALL9:
            p = float(np.clip(.4 + eff[l] * c["k"] * (c["kind"] != "base"), .01, .99))
            keepw = .95 if c["kind"] in ("base", "w") else .6
            for s in range(150):
                n = rng.poisson(.6); o = rng.binomial(n, p)
                add_row(aggW, condsW, dict(cond=ci, lang=l, sent=s, cont="a b c", match=bool(rng.random() < keepw),
                                           nom_ov=int(o), nom_vo=int(n - o), n_ov=int(o), n_vo=int(n - o)))
    B = test_B(aggW, agg2)
    C = test_C(aggW, B["claim"])
    print(f"B selftest: b_w {B['b_w']:+.3f} z {B['z']:+.1f} claim {B['claim']} | C: w higher in {C['higher']}/{C['n']} claim {C['claim']}")


if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        selftest()
        sys.exit(0)
    res = analyze(*sys.argv[1:4])
    report(res)
    clean = lambda o: {str(k): clean(v) for k, v in o.items()} if isinstance(o, dict) else [clean(v) for v in o] if isinstance(o, list) \
        else (None if isinstance(o, float) and not np.isfinite(o) else float(o) if isinstance(o, (np.floating,)) else bool(o) if isinstance(o, np.bool_) else o)
    json.dump(clean(res), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_w.json"), "w"), indent=1)
