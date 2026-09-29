"""Tree / Haar / Park statistics per layer x metric on the cross-split centroid Grams (grams/L*.npz)."""
import os, sys, json, time
import numpy as np
import treelib as T

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = np.load(os.path.join(HERE, "..", "D_offsets", "tok_sim.npz"))
LAYERS = list(range(0, 29, 2))
METRICS = ["euc", "lda01", "lda05", "causal"]


def covariates():
    tokd = 1 - TOK["tok_hist_int"]; np.fill_diagonal(tokd, 0)
    lf = np.log(TOK["fert"])
    fert = np.abs(lf[:, None] - lf[None, :])
    scr = np.array([[float(T.SCRIPT[i] != T.SCRIPT[j]) for j in range(T.L)] for i in range(T.L)])
    return dict(tok=tokd, fert=fert, script=scr)


COV = covariates()
TOKTREE = T.upgma_splits(COV["tok"])                                   # 10 splits
TOKTREE3 = [S for S, h in sorted(TOKTREE, key=lambda x: x[1])[:3]]      # 3 tightest token clusters
TREES = dict(T.TREES)
TREES["tokUPGMA"] = [S for S, h in TOKTREE]
TREES["tokUPGMA3"] = TOKTREE3
BASES = {
    "star": T.STAR,
    "star+tok+fert": np.column_stack([T.STAR, T.cov_cols({k: COV[k] for k in ("tok", "fert")})]),
    "star+tok+fert+script": np.column_stack([T.STAR, T.cov_cols(COV)]),
}
KEEP_NOHIN = np.array([(i != 5 and j != 5) for i, j in zip(*T.IU)])


def r2(y, X):
    s, _ = T.sse_nnls(X, y)
    return 1 - s / ((y - y.mean()) ** 2).sum()


def core_stats(K, n_perm=2000, seed=0, full=True):
    """statistics of one 12x12 Gram (cross-split).  full=False: only what the power study needs."""
    D = T.d2_from_gram(K)
    y = D[T.IU]
    out = {}
    out["R2_star"] = r2(y, T.STAR)
    out["R2_star+tok"] = r2(y, np.column_stack([T.STAR, T.cov_cols({"tok": COV["tok"]})]))
    out["R2_star+tok+fert"] = r2(y, BASES["star+tok+fert"])
    for bn in (["star", "star+tok+fert", "star+tok+fert+script"] if full else ["star", "star+tok+fert"]):
        base = BASES[bn]
        g, p, _ = T.perm_null_gain(y, base, TREES["glotto"], n_perm, seed)
        out[f"glotto_gain|{bn}"], out[f"glotto_p|{bn}"] = g, p
        g_, b_, _ = T.gain(y, base, TREES["glotto"])
        for nm, bb in zip(["IE", "Germ", "Rom"], b_):
            out[f"b_{nm}|{bn}"] = float(bb)
        for nm, S in T.GLOTTO_SPLITS.items():
            if True:
                obs, pe, _ = T.single_split_rank(y, base, S)
                out[f"split_{nm}_gain|{bn}"], out[f"split_{nm}_p|{bn}"] = obs, pe
        if full:
            for tn in ("script", "script+Han", "wordorder", "tokUPGMA", "tokUPGMA3"):
                g, p, _ = T.perm_null_gain(y, base, TREES[tn], n_perm // 4, seed + 1)
                out[f"{tn}_gain|{bn}"], out[f"{tn}_p|{bn}"] = g, p
    # Hindi excluded (pairs with hin dropped; relabel among the other 11)
    if full:
        yk = y[KEEP_NOHIN]
        base = np.delete(BASES["star+tok+fert"][KEEP_NOHIN], 5, axis=1)
        s0 = T.sse_nnls(base, yk)[0]

        def gk_(splits):
            Xs = np.column_stack([base] + [T.split_col(S)[KEEP_NOHIN] for S in splits])
            return 1 - T.sse_nnls(Xs, yk)[0] / s0
        obs = gk_(TREES["glotto"])
        rs = np.random.RandomState(seed + 7)
        others = np.array([i for i in range(T.L) if i != 5])
        null = []
        for _ in range(n_perm // 2):
            p = np.arange(T.L); p[others] = rs.permutation(others)
            null.append(gk_(T.relabel(TREES["glotto"], p)))
        out["glotto_gain_noHin|star+tok+fert"] = obs
        out["glotto_p_noHin|star+tok+fert"] = float((1 + (np.array(null) >= obs - 1e-12).sum()) / (1 + len(null)))
    # g(k): empirical K (family-balanced root of the glotto tree) vs ultrametric glotto tree
    w = T.root_weights(TREES["glotto"])
    Ke = T.center(0.5 * (K + K.T), w)
    Kt = T.center(T.ultrametric_K(TREES["glotto"]), w)
    gobs = T.gk(Ke, Kt)
    rs = np.random.RandomState(seed + 3)
    gnull = []
    for _ in range(n_perm // 2):
        p = rs.permutation(T.L)
        sp = T.relabel(TREES["glotto"], p)
        wp = T.root_weights(sp)
        gnull.append(T.gk(T.center(0.5 * (K + K.T), wp), T.center(T.ultrametric_K(sp), wp)))
    gnull = np.array(gnull)
    for q, k in enumerate((1, 2, 3, 4)):
        out[f"g{k}"] = float(gobs[q])
        out[f"g{k}_null"] = float(gnull[:, q].mean())
        out[f"g{k}_p"] = float((1 + (gnull[:, q] >= gobs[q] - 1e-12).sum()) / (1 + len(gnull)))
    # Park orthogonality: mean cos^2 over the 7 hierarchical pairs vs random pseudo-clades (smaller = more orthogonal)
    c, names = T.park_cos(K)
    pn = T.park_null(K, n_perm, seed + 5)
    out["park_meancos2"] = float((c ** 2).mean())
    out["park_null_meancos2"] = float((pn ** 2).mean())
    out["park_p_orth"] = float((1 + ((pn ** 2).mean(1) <= (c ** 2).mean() + 1e-12).sum()) / (1 + len(pn)))
    if full:
        for nm, v in zip(names, c):
            out[f"cos[{nm}]"] = float(v)
        out["NJ"] = " ".join(T.fmt_split(T.canon(S)) for S in T.nj_splits(D))
    return out


def main(layers=LAYERS, metrics=METRICS, n_perm=2000):
    rows = []
    for layer in layers:
        z = np.load(os.path.join(HERE, "grams", f"L{layer}.npz"))
        for m in metrics:
            t0 = time.time()
            K = z[f"{m}_dt"]
            S = core_stats(K, n_perm, seed=layer)
            # split-half / bootstrap stability of the headline numbers
            for tag, KK in (("dd", z[f"{m}_dd"]), ("tt", z[f"{m}_tt"])):
                y = T.d2_from_gram(KK)[T.IU]
                S[f"glotto_gain_{tag}|star+tok+fert"] = T.gain(y, BASES["star+tok+fert"], TREES["glotto"])[0]
            bg, bg0 = [], []
            for Kb in z[f"{m}_bdt"]:
                y = T.d2_from_gram(Kb)[T.IU]
                bg.append(T.gain(y, BASES["star+tok+fert"], TREES["glotto"])[0])
                bg0.append(T.gain(y, BASES["star"], TREES["glotto"])[0])
            S["glotto_gain_boot05|star+tok+fert"], S["glotto_gain_boot95|star+tok+fert"] = np.quantile(bg, [0.05, 0.95])
            S["glotto_gain_boot05|star"], S["glotto_gain_boot95|star"] = np.quantile(bg0, [0.05, 0.95])
            S.update(layer=layer, metric=m)
            rows.append(S)
            print(f"L{layer:2d} {m:6s} R2star={S['R2_star']:.3f} R2+tok+fert={S['R2_star+tok+fert']:.3f} | glotto gain "
                  f"{S['glotto_gain|star']:.3f} p={S['glotto_p|star']:.3f}  |cov {S['glotto_gain|star+tok+fert']:.3f} "
                  f"p={S['glotto_p|star+tok+fert']:.3f} | IE p={S['split_IE_p|star+tok+fert']:.3f} Germ p="
                  f"{S['split_Germ_p|star+tok+fert']:.3f} Rom p={S['split_Rom_p|star+tok+fert']:.3f} | g1..3 "
                  f"{S['g1']:.2f}/{S['g2']:.2f}/{S['g3']:.2f} p {S['g1_p']:.2f}/{S['g2_p']:.2f}/{S['g3_p']:.2f} | park "
                  f"{S['park_meancos2']:.3f} vs {S['park_null_meancos2']:.3f} p={S['park_p_orth']:.2f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        json.dump(rows, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float)
    return rows


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(layers=[int(a) for a in sys.argv[1:]])
    else:
        main()
