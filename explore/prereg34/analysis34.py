"""PRE-REGISTERED analysis of the 34-language extraction (see PREREG.md; do not change after data arrive --
additions go in a separate, clearly labelled exploratory script).

usage: python analysis34.py DATA_DIR [--n_perm 10000] [--out results34.json]
DATA_DIR holds grams/L{0..28}.npz (keys {metric}_{dt,dd,tt,bdt} and {metric}_dt_sq for the sqrt(ntok) variant)
and tokstats.npz (tok_hist_int [34,34], fert [34]) written by derive34.py.
"""
import argparse, json, os, time
import numpy as np
import lib34 as T

PRIMARY_METRICS = ["causal", "lda05"]
ALL_METRICS = ["euc", "lda01", "lda05", "causal"]
LAYERS_PRIMARY = list(range(8, 21))          # L8..L20 inclusive
ALPHA = 0.05 / 4                             # Bonferroni over {H1, H2} x {causal, lda05}


def load(data):
    n_layers = json.load(open(os.path.join(data, "meta.json")))["n_layers"]     # 29 for Qwen2.5-7B, 33 for Llama-3.1-8B
    grams = {l: np.load(os.path.join(data, "grams", f"L{l}.npz")) for l in range(n_layers)}
    tok = np.load(os.path.join(data, "tokstats.npz"))
    return grams, tok


def avg_gram(grams, metric, key="dt", layers=None):
    layers = LAYERS_PRIMARY if layers is None else layers
    Ks = [0.5 * (grams[l][f"{metric}_{key}"] + grams[l][f"{metric}_{key}"].T) for l in layers]
    return np.mean([K / np.trace(K) for K in Ks], axis=0)


def bases(sp, tok):
    tokd = 1 - tok["tok_hist_int"]
    np.fill_diagonal(tokd, 0)
    lf = np.log(tok["fert"])
    fert = np.abs(lf[:, None] - lf[None, :])
    cov = sp.covcols([tokd, fert])
    main = np.column_stack([sp.star] + [sp.col(S) for S in T.SCRIPT_SPLITS] + [cov])
    noscript = np.column_stack([sp.star, cov])
    return main, noscript


def tests(sp, K, base, base_noscript, n_perm, seed, full=True):
    y = sp.vec(T.d2_from_gram(K))
    out = {}
    tot = ((y - y.mean()) ** 2).sum()
    out["R2_star"] = 1 - T.sse(sp.star, y) / tot
    out["R2_base"] = 1 - T.sse(base, y) / tot
    out["H1_gain"], out["H1_p"], _ = T.perm_relabel(sp, y, base, T.GLOTTO, n_perm, seed)
    out["H2_gain"], out["H2_p"], _ = T.perm_subset(sp, y, base, T.OV, n_perm, seed + 1)
    if not full:
        return out
    b_ov = np.column_stack([base, sp.col(T.OV)])
    b_gl = np.column_stack([base] + [sp.col(S) for S in T.GLOTTO])
    out["H1|OV_gain"], out["H1|OV_p"], _ = T.perm_relabel(sp, y, b_ov, T.GLOTTO, n_perm, seed + 2)
    out["OV|H1_gain"], out["OV|H1_p"], _ = T.perm_subset(sp, y, b_gl, T.OV, n_perm, seed + 3)
    out["OV+deu,nld_gain"], out["OV+deu,nld_p"], _ = T.perm_subset(sp, y, base, T.OV_PLUS_DE_NL, n_perm, seed + 4)
    pct, _ = T.pair_residual_percentiles(sp, y, base_noscript, T.SAME_LANG_PAIRS)
    rs = np.random.RandomState(seed + 5)
    b, _r = T.nnls(base_noscript, y)
    r = y - base_noscript @ b
    ranks = r.argsort().argsort() / (len(r) - 1)
    null = np.array([ranks[rs.choice(len(r), 3, replace=False)].mean() for _ in range(n_perm)])
    out["H3_pair_pct"] = dict(zip(["hin-urd", "hrv-srp", "cmnHans-cmnHant"], map(float, pct)))
    out["H3_mean_pct"] = float(pct.mean())
    out["H3_p"] = float((1 + (null <= pct.mean() + 1e-12).sum()) / (1 + n_perm))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--out", default="results34.json")
    ap.add_argument("--primary", default=None, help="primary layer window 'a-b' (default 8-20 = Qwen2.5-7B)")
    args = ap.parse_args()
    global LAYERS_PRIMARY
    if args.primary:                                   # e.g. "9-23" (Llama replication: same relative depth window)
        a, b = map(int, args.primary.split("-"))
        LAYERS_PRIMARY = list(range(a, b + 1))
    grams, tok = load(args.data)
    sp = T.Space()
    base, base_ns = bases(sp, tok)
    res = {"prereg": {"alpha": ALPHA, "layers": LAYERS_PRIMARY, "metrics": PRIMARY_METRICS, "n_perm": args.n_perm}}
    print(f"== PRIMARY (averaged trace-normalised cross-split Gram, L{LAYERS_PRIMARY[0]}-L{LAYERS_PRIMARY[-1]})")
    for m in PRIMARY_METRICS:
        t0 = time.time()
        r = tests(sp, avg_gram(grams, m), base, base_ns, args.n_perm, seed=17)
        r["H1_reject"], r["H2_reject"] = r["H1_p"] < ALPHA, r["H2_p"] < ALPHA
        # PREREG claim rules: marginal AND conditional test both below ALPHA
        r["claim_genealogy"] = bool(r["H1_reject"] and r["H1|OV_p"] < ALPHA)
        r["claim_wordorder"] = bool(r["H2_reject"] and r["OV|H1_p"] < ALPHA)
        # bootstrap CI of the two primary gains (sentence resampling, averaged Gram per bootstrap draw)
        B = grams[LAYERS_PRIMARY[0]][f"{m}_bdt"].shape[0]
        g1, g2 = [], []
        for b in range(B):
            Kb = np.mean([(lambda K: K / np.trace(K))(0.5 * (grams[l][f"{m}_bdt"][b] + grams[l][f"{m}_bdt"][b].T))
                          for l in LAYERS_PRIMARY], axis=0)
            yb = sp.vec(T.d2_from_gram(Kb))
            g1.append(T.gain(sp, yb, base, T.GLOTTO))
            g2.append(T.gain(sp, yb, base, [T.OV]))
        r["H1_gain_boot90"] = [float(x) for x in np.quantile(g1, [0.05, 0.95])]
        r["H2_gain_boot90"] = [float(x) for x in np.quantile(g2, [0.05, 0.95])]
        res[f"primary_{m}"] = r
        print(f"  {m:6s} R2 star {r['R2_star']:.3f} base {r['R2_base']:.3f} | H1 genealogy gain {r['H1_gain']:.3f} "
              f"p={r['H1_p']:.4f} {'REJECT' if r['H1_reject'] else 'n.s.'} | H2 OV gain {r['H2_gain']:.3f} "
              f"p={r['H2_p']:.4f} {'REJECT' if r['H2_reject'] else 'n.s.'} | H1|OV p={r['H1|OV_p']:.4f} "
              f"OV|H1 p={r['OV|H1_p']:.4f} | H3 mean pct {r['H3_mean_pct']:.3f} p={r['H3_p']:.4f} "
              f"({time.time() - t0:.0f}s)\n         CLAIM genealogy: {r['claim_genealogy']} | CLAIM word order: "
              f"{r['claim_wordorder']}", flush=True)
    print("== SECONDARY: sqrt(ntok) pooling variant (same tests, 2000 permutations)")
    for m in PRIMARY_METRICS:
        r = tests(sp, avg_gram(grams, m, key="dt_sq"), base, base_ns, 2000, seed=23)
        res[f"sqrt_ntok_{m}"] = r
        print(f"  {m:6s} H1 gain {r['H1_gain']:.3f} p={r['H1_p']:.4f} | H2 gain {r['H2_gain']:.3f} p={r['H2_p']:.4f} "
              f"| H3 p={r['H3_p']:.4f}", flush=True)
    print("== DESCRIPTIVE: per layer x metric (500 permutations, no multiplicity claims)")
    rows = []
    for l in range(len(grams)):
        for m in ALL_METRICS:
            K = 0.5 * (grams[l][f"{m}_dt"] + grams[l][f"{m}_dt"].T)
            r = tests(sp, K / np.trace(K), base, base_ns, 500, seed=100 + l, full=False)
            r.update(layer=l, metric=m)
            rows.append(r)
        print("  L%2d " % l + " | ".join(f"{r['metric']} H1 {r['H1_gain']:.2f} (p {r['H1_p']:.3f}) H2 {r['H2_gain']:.2f} "
                                          f"(p {r['H2_p']:.3f})" for r in rows[-4:]), flush=True)
    res["per_layer"] = rows
    json.dump(res, open(args.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
