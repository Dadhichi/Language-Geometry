"""Belief-simplex sentence-level pilot: shared machinery.

Displacement d_{i,s} = o_{i,s} - a_i (in the 11-d span of dev centroids), o = per-sentence language offset.
Belief model: o_{i,s} = sum_k w_{k,s} a_k + noise, w = posterior over languages.  Specific (vertex-directed) part:
  d_{i,s} = g * Dhat_{i,s} + (generic terms) + noise,   Dhat_{i,s} = sum_{j != i} xt_{ij,s} a_j,
xt = instrument demeaned over j != i within (i,s) [removes uniform = generic shrinkage toward the centre] and
centred over s within each pair (i,j) [pair fixed effects].  Generic terms: per-language free vectors times
covariates (length, total shared evidence, own-belief, digits, punctuation) -> any covariate-linked shift in ANY
direction is absorbed.  g is fitted on dev (GLS in the pooled within-language residual metric), scored on devtest.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANGS = ["eng", "deu", "fra", "spa", "rus", "hin", "arb", "zho", "jpn", "tur", "vie", "ind"]
LATIN = [0, 1, 2, 3, 9, 10, 11]
NONLAT = [4, 5, 6, 7, 8]
L = 12
SCRIPT = np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0])   # jpn separate from zho


def load_instr(split):
    return dict(np.load(os.path.join(HERE, f"instr_{split}.npz")))


def load_layer(layer, variant):
    z = np.load(os.path.join(HERE, "layers", f"L{layer}_{variant}.npz"))
    return {k: z[k] for k in z.files}


def prep(x):
    """x [L,L,n] -> xt: diag 0, demeaned over j != i within (i,s), centred over s within (i,j)"""
    x = x.astype(np.float64).copy()
    m = ~np.eye(L, dtype=bool)
    for i in range(L):
        x[i, i] = np.nan
    x = x - np.nanmean(x, axis=1, keepdims=True)
    for i in range(L):
        x[i, i] = 0.0
    x = x - x.mean(2, keepdims=True)
    x[~m] = 0.0
    return x


def dhat(xt, A):
    """xt [L,L,n], A [L,k] vertex coordinates -> [L,n,k]"""
    return np.einsum("ijs,jk->isk", xt, A)


def covariates(I, names=("logn", "shared", "ownbag", "ownpre", "digit", "punct")):
    n = I["ntok"].shape[1]
    cols = []
    for nm in names:
        if nm == "logn":
            c = np.log(I["ntok"])
        elif nm == "shared":
            c = I["e_tokc"].sum(1) / (L - 1)
        elif nm == "sharedchar":
            c = I["e_char"].sum(1) / (L - 1)
        elif nm == "ownbag":
            c = np.stack([I["B_bag"][i, i] for i in range(L)])
        elif nm == "ownpre":
            c = np.stack([I["B_pre"][i, i] for i in range(L)])
        else:
            c = I[nm]
        cols.append(c)
    Z = np.stack(cols, 2).astype(np.float64)          # [L,n,M]  raw units so dev-fitted Gamma transfers to test
    Z -= Z.mean(1, keepdims=True)
    return Z


def centre(Y):
    return Y - Y.mean(1, keepdims=True)


def residualize(Y, Zc):
    """per language: Y [L,n,k] minus projection on Zc [L,n,M] (Zc already centred)."""
    out = np.empty_like(Y)
    for i in range(Y.shape[0]):
        if Zc is None or Zc.shape[2] == 0:
            out[i] = Y[i]
        else:
            b = np.linalg.lstsq(Zc[i], Y[i], rcond=None)[0]
            out[i] = Y[i] - Zc[i] @ b
    return out


def metric(r, eps=1e-3):
    S = np.einsum("isk,isl->kl", r, r) / (r.shape[0] * r.shape[1])
    S += eps * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    return np.linalg.inv(S)


def ip(X, Y, M):
    return float(np.einsum("isk,kl,isl->", X, M, Y))


def gfit(d, Dh, Zc, M):
    """joint GLS gains for a list of Dhat's after partialling covariates (FWL). returns g, Gamma list"""
    r = residualize(d, Zc)
    R = [residualize(D, Zc) for D in Dh]
    G = np.array([[ip(Ra, Rb, M) for Rb in R] for Ra in R])
    h = np.array([ip(Ra, r, M) for Ra in R])
    g = np.linalg.solve(G, h)
    rem = d - sum(gk * D for gk, D in zip(g, Dh))
    Gam = [np.linalg.lstsq(Zc[i], rem[i], rcond=None)[0] if Zc is not None and Zc.shape[2] else None
           for i in range(d.shape[0])]
    return g, Gam


def apply_gamma(d, Zc, Gam):
    out = d.copy()
    for i in range(d.shape[0]):
        if Gam[i] is not None:
            out[i] -= Zc[i] @ Gam[i]
    return out


def heldout(dt, Dht, Zt, Gam, g, M):
    """held-out: residual after dev-fitted covariate terms; rho per instrument; R2 of dev-fitted specific part"""
    r = apply_gamma(dt, Zt, Gam)
    rr = ip(r, r, M)
    rho = [ip(r, D, M) / np.sqrt(rr * ip(D, D, M)) for D in Dht]
    pred = sum(gk * D for gk, D in zip(g, Dht))
    R2 = 1 - ip(r - pred, r - pred, M) / rr
    return np.array(rho), R2, r


def shuffle_rows(D, rng):
    return np.stack([D[i][rng.permutation(D.shape[1])] for i in range(D.shape[0])])


def jperm(xt, rng):
    """permute the instrument across targets j != i within every (i,s), then re-centre per pair"""
    x = xt.copy()
    n = x.shape[2]
    for i in range(L):
        js = np.array([j for j in range(L) if j != i])
        P = np.argsort(rng.random((n, len(js))), axis=1)
        sub = x[i][js]                         # [11, n]
        x[i][js] = np.take_along_axis(sub.T, P, axis=1).T
    x = x - x.mean(2, keepdims=True)
    for i in range(L):
        x[i, i] = 0
    return x


def bary(Zs, A):
    """barycentric coordinates w.r.t. the vertices A [L,11] (rank 11, rows sum to 0): [L,n,L]"""
    return Zs @ np.linalg.pinv(A) + 1.0 / L


def mantel(Dm, Sm, nperm=5000, rng=None, ctrl=None):
    """Spearman between upper triangles of Dm and Sm (optionally partialling ctrl), label-permutation p (2-sided)"""
    from scipy.stats import rankdata
    iu = np.triu_indices(L, 1)

    def vec(M):
        return rankdata(M[iu])

    def stat(a, b):
        if ctrl is not None:
            c = vec(ctrl)
            X = np.c_[np.ones_like(c), c]
            a = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
            b = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
        return float(np.corrcoef(a, b)[0, 1])
    s0 = stat(vec(Dm), vec(Sm))
    rng = rng or np.random.RandomState(0)
    cnt = 0
    for _ in range(nperm):
        p = rng.permutation(L)
        if abs(stat(vec(Dm[np.ix_(p, p)]), vec(Sm))) >= abs(s0) - 1e-12:
            cnt += 1
    return s0, (cnt + 1) / (nperm + 1)
