"""Synthetic validation / power of the tree statistics at 12 languages.
Planted worlds (one BM realisation per replicate, anisotropic increments with the REAL centroid spectrum so the
configuration has the real effective dimension ~5; leaf lengths = real star-fit leaf lengths (Hindi long)):
  glotto   : star + internal splits IE, Germ, Rom          (internal share f of sum_pairs E[D2])
  script   : star + Latin clade + Han clade                 (share f)
  wordorder: star + {hin, jpn, tur}                         (share f)
  tokgeom  : star + Gaussian configuration with E[D2] proportional to the token distance (info-geometry null)
  fertline : star + 1-D configuration along log fertility
  star     : leaves only (f = 0)
Noise: a real sentence-bootstrap deviation of the cross-split Gram (K_b - mean_b K_b), rescaled to the planted trace.
usage: python synth.py <metric> <layer> <reps>"""
import os, sys, json, time
import numpy as np
import treelib as T
import analyze as A

HERE = os.path.dirname(os.path.abspath(__file__))


def setup(metric, layer):
    z = np.load(os.path.join(HERE, "grams", f"L{layer}.npz"))
    K = z[f"{metric}_dt"]
    y = T.d2_from_gram(K)[T.IU]
    _, leaf = T.sse_nnls(T.STAR, y)
    lam = np.clip(np.linalg.eigvalsh(T.center(0.5 * (K + K.T), np.ones(T.L) / T.L)), 0, None)[::-1][:11]
    bdev = z[f"{metric}_bdt"] - z[f"{metric}_bdt"].mean(0)
    return leaf, lam / lam.sum(), bdev, np.trace(K)


def pair_sum(splits_w):
    """sum over pairs of E[D2] contributed by weighted splits"""
    return sum(w * T.split_col(S).sum() for S, w in splits_w)


def planted(world, f, leaf, spec, rs, dim=16):
    """returns X [12, dim] leaf positions (one BM realisation, isotropic increments in `dim` dims)"""
    def inc(var):
        return rs.randn(dim) * np.sqrt(var / dim)
    X = np.stack([inc(l) for l in leaf])                    # leaf edges
    star_sum = T.STAR.sum(0) @ leaf                          # sum_pairs of leaf part of E[D2]
    target = f / (1 - f) * star_sum if f < 1 else 0
    if world in ("glotto", "script", "wordorder"):
        splits = {"glotto": T.TREES["glotto"], "script": T.TREES["script+Han"], "wordorder": T.TREES["wordorder"]}[world]
        if world == "glotto":
            wts = [1.0, 1.0, 1.0]
        else:
            wts = [1.0] * len(splits)
        s = target / pair_sum(list(zip(splits, wts))) if target > 0 else 0
        for S, w_ in zip(splits, wts):
            v = inc(s * w_)
            X[list(S)] += v
    elif world == "tokgeom":
        D2 = A.COV["tok"]
        J = np.eye(T.L) - 1.0 / T.L
        G = -0.5 * J @ D2 @ J
        ev, U = np.linalg.eigh(G)
        ev = np.clip(ev, 0, None)
        Y = U[:, -11:] * np.sqrt(ev[-11:])                    # [12, 11] Euclidean embedding of token geometry
        Y *= np.sqrt(target / T.d2_from_gram(Y @ Y.T)[T.IU].sum()) if target > 0 else 0
        O = np.linalg.qr(rs.randn(dim, 11))[0].T              # random isometric embedding into dim
        X += Y @ O
    elif world == "fertline":
        lf = np.log(A.TOK["fert"]); lf -= lf.mean()
        v = inc(1.0)
        Y = np.outer(lf, v)
        Y *= np.sqrt(target / T.d2_from_gram(Y @ Y.T)[T.IU].sum()) if target > 0 else 0
        X += Y
    return X


def run(metric="euc", layer=14, reps=100, fs=(0.0, 0.05, 0.1, 0.2, 0.3),
        worlds=("glotto", "script", "wordorder", "tokgeom", "fertline"), n_perm=400, dim=16):
    leaf, spec, bdev, trK = setup(metric, layer)
    rs = np.random.RandomState(12345 + layer)
    res = []
    for world in worlds:
        for f in fs:
            if f == 0 and world != "glotto":
                continue                                          # f=0 is the star world (listed once)
            t0 = time.time()
            acc = []
            for r in range(reps):
                X = planted(world, f, leaf, spec, rs, dim)
                K = X @ X.T
                K = K + bdev[rs.randint(len(bdev))] * (np.trace(K) / trK)
                S = A.core_stats(K, n_perm=n_perm, seed=r, full=False)
                acc.append(S)
            keys = [k for k in acc[0] if k.endswith(("_p|star", "_p|star+tok+fert", "_p")) or "p_orth" in k]
            summ = {k: float(np.mean([a[k] < 0.05 for a in acc])) for k in keys}
            for k in ("R2_star", "glotto_gain|star", "glotto_gain|star+tok+fert", "park_meancos2", "park_null_meancos2", "g1"):
                summ["mean_" + k] = float(np.mean([a[k] for a in acc]))
            summ.update(world=world if f > 0 else "star", f=f, metric=metric, layer=layer, reps=reps, dim=dim)
            res.append(summ)
            print(f"{summ['world']:9s} f={f:.2f} R2star={summ['mean_R2_star']:.2f} P(p<.05): IE {summ['split_IE_p|star+tok+fert']:.2f} glotto|star {summ['glotto_p|star']:.2f}  glotto|+tok+fert "
                  f"{summ['glotto_p|star+tok+fert']:.2f}  Germ {summ['split_Germ_p|star+tok+fert']:.2f}  Rom "
                  f"{summ['split_Rom_p|star+tok+fert']:.2f}  g1 {summ['g1_p']:.2f}  park_orth {summ['park_p_orth']:.2f} | "
                  f"gain {summ['mean_glotto_gain|star']:.3f}/{summ['mean_glotto_gain|star+tok+fert']:.3f} park "
                  f"{summ['mean_park_meancos2']:.3f} vs {summ['mean_park_null_meancos2']:.3f} ({time.time() - t0:.0f}s)",
                  flush=True)
    json.dump(res, open(os.path.join(HERE, f"synth_{metric}_L{layer}_d{dim}.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "euc"
    lay = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    dim = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    run(m, lay, reps, dim=dim)
