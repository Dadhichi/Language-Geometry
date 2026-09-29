"""Synthetic validation: run the identical pipeline on data with known structure built from the real
layer-14 content cloud (b_s), the real interaction residuals (sentence-shuffled within language, so they
keep each language's covariance but lose any content link), and PLANTED language offsets.
  S_shift : X = m + a*_i + b_s + e_i,pi(s)        a* additive in script+family, random directions
  S_gauge : X = m + a*_i + b_s G_i + e_i,pi(s)    G_i rotates the top-64 content PCs; a* inside span(W_U)
Centroid-level tests are validated on planted 12-point configurations (additive / random / tree)."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlib, clib

rs = np.random.RandomState(7)
W = np.load(os.path.join(dlib.DATA, "wu_basis.npy")).astype(np.float32)
weig = np.load(os.path.join(dlib.DATA, "wu_eigs.npy")).astype(np.float64)
d, L = 3584, 12
out = {}
ONLY_CENTROID = "--centroid" in sys.argv


def planted_additive(scale, idio=0.2, basis=None, seed=0):
    r = np.random.RandomState(seed)
    dim = d if basis is None else basis.shape[1]
    vec = lambda: r.randn(dim)
    sv = {s: vec() for s in set(clib.SCRIPT)}
    fv = {f: vec() for f in set(clib.FAMILY)}
    gv = {g: 0.6 * vec() for g in set(clib.GENUS)}
    A = np.stack([sv[clib.SCRIPT[i]] + fv[clib.FAMILY[i]] + gv[clib.GENUS[i]] for i in range(L)])
    A = A + np.sqrt(idio) * np.sqrt((A ** 2).sum(1).mean() / dim) * r.randn(L, dim)
    A -= A.mean(0)
    if basis is not None:
        A = A @ basis.T
    return A * np.sqrt(scale / (A ** 2).sum(1).mean())


def planted_random(scale, seed=0):
    A = np.random.RandomState(seed).randn(L, d)
    A -= A.mean(0)
    return A * np.sqrt(scale / (A ** 2).sum(1).mean())


def planted_tree(scale, seed=0, noise=0.0):
    """genealogical tree: root->family->genus->leaf, i.i.d. Gaussian increments on every edge"""
    r = np.random.RandomState(seed)
    fv = {f: r.randn(d) for f in set(clib.FAMILY)}
    gv = {g: r.randn(d) for g in set(clib.GENUS)}
    A = np.stack([fv[clib.FAMILY[i]] + gv[clib.GENUS[i]] + 0.5 * r.randn(d) for i in range(L)])
    A += noise * r.randn(L, d) * np.sqrt((A ** 2).sum(1).mean() / d)
    A -= A.mean(0)
    return A * np.sqrt(scale / (A ** 2).sum(1).mean())


# ---------------------------------------------------------------- centroid-level validation
tok = dict(np.load(os.path.join(dlib.OUT, "tok_sim.npz")))
refs = clib.ref_dists(tok)
cres = {}
for name, A in (("additive_idio0.2", planted_additive(160, 0.2)), ("additive_idio1.0", planted_additive(160, 1.0, seed=1)),
                ("random", planted_random(160, 2)), ("tree", planted_tree(160, 3)), ("tree_noisy", planted_tree(160, 4, noise=1.0))):
    # two noisy "splits" of the same centroids (noise at the real split-half level ~0.1%)
    Ad = A + 0.03 * np.sqrt(160 / d) * rs.randn(L, d)
    At = A + 0.03 * np.sqrt(160 / d) * rs.randn(L, d)
    att = clib.attribute_test(Ad, At, n_perm=500)
    cr = clib.crossing_test(Ad, n_perm=2000)
    D = np.sqrt(np.clip(clib.xdist2(Ad, At), 0, None))
    fp = clib.fourpoint(D)
    Gc = (Ad - Ad.mean(0)) @ (Ad - Ad.mean(0)).T
    nul = clib.gaussian_null_tree(Gc, n=200)
    mf = clib.mantel(D, refs["family"], n_perm=2000)
    ms = clib.mantel(D, refs["script"], n_perm=2000)
    cres[name] = dict(loo_r2={k: (round(v["r2"], 3), round(v["null_p95"], 3), round(v["p"], 4)) for k, v in att.items()},
                      crossing=(round(cr["cos"], 3), round(cr["null_cos_p95"], 3), round(cr["p_cos"], 4)),
                      fourpoint_eps=(round(fp[0], 3), round(float(nul[:, 0].mean()), 3), round(float(np.quantile(nul[:, 0], 0.05)), 3)),
                      cophenetic=(round(clib.coph(D), 3), round(float(nul[:, 2].mean()), 3), round(float(np.quantile(nul[:, 2], 0.95)), 3)),
                      mantel_family=mf, mantel_script=ms)
    print(name, json.dumps(cres[name]), flush=True)
out["centroid"] = cres
json.dump(out, open(os.path.join(dlib.OUT, "synth_centroid.json"), "w"), indent=1)
if ONLY_CENTROID:
    sys.exit()

# ---------------------------------------------------------------- full-pipeline validation at layer 14 geometry
ntok = np.load(os.path.join(dlib.DATA, "ntok_dev.npy"))


def build(split, scenario):
    X = dlib.load(split, 14)
    A = dlib.Anova(X)
    n = X.shape[1]
    Vl = float((A.a ** 2).sum() / L)
    if scenario == "shift":
        astar = planted_additive(Vl, 0.2, seed=11)
    else:
        astar = planted_additive(Vl, 0.2, basis=W, seed=11)          # inside span(W_U)
    P = Pb64 if scenario == "gauge" else None
    r = np.random.RandomState(100)
    Qs = []
    for i in range(L):
        K = r.randn(64, 64); K = (K - K.T) / np.sqrt(2 * 64)
        Qs.append(expm(0.8 * K).astype(np.float32))
    b32 = A.b.astype(np.float32)
    m32 = A.m.astype(np.float32)
    pr_ = np.random.RandomState(5 if split == "dev" else 6)
    for i in range(L):
        e = A.e(i)[pr_.permutation(n)]
        cont = b32 if P is None else b32 + ((b32 @ P) @ (Qs[i] - np.eye(64, dtype=np.float32))) @ P.T
        X[i] = m32 + astar[i].astype(np.float32) + cont + e
    return X, astar


Xtmp = dlib.load("dev", 14)
Atmp = dlib.Anova(Xtmp)
_, Pb = dlib.top_basis_rows(Atmp.b, 64)
Pb64 = Pb[:, :64]
del Xtmp, Atmp
full = {}
for scen in ("shift", "gauge"):
    Xd, astar = build("dev", scen)
    Xt, _ = build("devtest", scen)
    S, rows, arrs = dlib.layer_stats(Xd, Xt, W, weig, ntok_d=ntok, verbose=True)
    del Xd, Xt
    S["planted_W_frac"] = float(((astar @ W) ** 2).sum() / (astar ** 2).sum())
    full[scen] = S
    print(scen, {k: round(v, 4) for k, v in S.items() if isinstance(v, float)}, flush=True)
    att = clib.attribute_test(arrs["a_dev"].astype(np.float64), arrs["a_test"].astype(np.float64), n_perm=500)
    full[scen]["recovered_attr_r2"] = {k: (v["r2"], v["null_p95"]) for k, v in att.items()}
out["full"] = full
json.dump(out, open(os.path.join(dlib.OUT, "synth.json"), "w"), indent=1, default=float)
