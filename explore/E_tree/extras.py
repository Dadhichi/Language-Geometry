"""(a) variance partition of the cross-split D2 among star / token / fertility / genealogy;
(b) data-driven best single splits beyond star+tok+fert (all 2047 bipartitions), with where the Glottolog splits rank;
(c) Park orthogonality after equalising leaf lengths (removes the 'similar norms within a clade' effect)."""
import os, json, itertools
import numpy as np
import treelib as T
import analyze as A

HERE = os.path.dirname(os.path.abspath(__file__))
ALLSPLITS = sorted({T.canon(S) for k in range(2, 7) for S in itertools.combinations(range(T.L), k)})
Ssplit = {k: T.canon(v) for k, v in T.GLOTTO_SPLITS.items()}
Ssplit["SOV"] = T.canon((5, 8, 9)); Ssplit["Latin"] = T.canon(T.TREES["script"][0]); Ssplit["Han"] = T.canon((7, 8))
Ssplit["hin-tur"] = T.canon((5, 9))


def r2(y, X):
    return A.r2(y, X)


rows = []
for layer in A.LAYERS:
    z = np.load(os.path.join(HERE, "grams", f"L{layer}.npz"))
    for m in A.METRICS:
        K = 0.5 * (z[f"{m}_dt"] + z[f"{m}_dt"].T)
        y = T.d2_from_gram(K)[T.IU]
        tok = T.cov_cols({"tok": A.COV["tok"]}); fert = T.cov_cols({"fert": A.COV["fert"]})
        gl = np.column_stack([T.split_col(S) for S in T.TREES["glotto"]])
        R = {}
        R["star"] = r2(y, T.STAR)
        R["star+fert"] = r2(y, np.column_stack([T.STAR, fert]))
        R["star+tok"] = r2(y, np.column_stack([T.STAR, tok]))
        R["star+glotto"] = r2(y, np.column_stack([T.STAR, gl]))
        R["star+tok+fert"] = r2(y, np.column_stack([T.STAR, tok, fert]))
        R["star+fert+glotto"] = r2(y, np.column_stack([T.STAR, fert, gl]))
        R["all"] = r2(y, np.column_stack([T.STAR, tok, fert, gl]))
        # unique shares (of total variance) beyond the rest
        R["uniq_tok"] = R["all"] - R["star+fert+glotto"]
        R["uniq_glotto"] = R["all"] - R["star+tok+fert"]
        base = A.BASES["star+tok+fert"]
        s0 = T.sse_nnls(base, y)[0]
        gains = {S: 1 - T.sse_nnls(np.column_stack([base, T.split_col(S)]), y)[0] / s0 for S in ALLSPLITS}
        order = sorted(gains, key=lambda s: -gains[s])
        rank = {s: i + 1 for i, s in enumerate(order)}
        R["top_splits"] = [(T.fmt_split(s), round(gains[s], 3)) for s in order[:4]]
        for nm, S in Ssplit.items():
            R[f"rank_{nm}"] = rank[S]
            R[f"gain_{nm}"] = gains[S]
        # (c) leaf-equalised Park
        _, leaf = T.sse_nnls(T.STAR, y)
        Keq = K - np.diag(leaf / 2) + np.eye(T.L) * leaf.mean() / 2
        c, _ = T.park_cos(Keq)
        pn = T.park_null(Keq, 2000, layer)
        R["park_eq_meancos2"] = float((c ** 2).mean())
        R["park_eq_null"] = float((pn ** 2).mean())
        R["park_eq_p"] = float((1 + ((pn ** 2).mean(1) <= (c ** 2).mean()).sum()) / 2001)
        R.update(layer=layer, metric=m)
        rows.append(R)
        print(f"L{layer:2d} {m:6s} R2 star {R['star']:.3f} +tok {R['star+tok']:.3f} +fert {R['star+fert']:.3f} "
              f"+glotto {R['star+glotto']:.3f} all {R['all']:.3f} | uniq tok {R['uniq_tok']:.3f} glotto "
              f"{R['uniq_glotto']:.3f} | ranks/2047 IE {R['rank_IE']} Germ {R['rank_Germ']} Rom {R['rank_Rom']} "
              f"SOV {R['rank_SOV']} Han {R['rank_Han']} hin-tur {R['rank_hin-tur']} | top {R['top_splits'][:2]} | "
              f"park_eq p {R['park_eq_p']:.2f}", flush=True)
json.dump(rows, open(os.path.join(HERE, "extras.json"), "w"), indent=1, default=float)
