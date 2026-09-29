"""Centroid-level analyses (12 language offsets a_i per layer): attribute algebra, metric/tree structure,
Mantel tests against linguistic and tokenizer similarity, depth dynamics.  All ambient / gauge-free."""
import itertools
import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, pearsonr

L = 12
LATIN = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1], float)
CJK = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0], float)
IE = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], float)
GERM = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], float)
ROM = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], float)
SCRIPT = ["Latn", "Latn", "Latn", "Latn", "Cyrl", "Deva", "Arab", "Hani", "Hani", "Latn", "Latn", "Latn"]
GENUS = ["Germ", "Germ", "Rom", "Rom", "Slav", "IndoAr", "Sem", "Sin", "Jap", "Turk", "AuAs", "AuNe"]
FAMILY = ["IE"] * 6 + ["AfAs", "SinoTib", "Japonic", "Turkic", "AuAs", "AuNe"]


def ref_dists(tok=None):
    fam = np.zeros((L, L)); scr = np.zeros((L, L))
    for i, j in itertools.product(range(L), range(L)):
        if i == j:
            continue
        fam[i, j] = 1 if GENUS[i] == GENUS[j] else (2 if FAMILY[i] == FAMILY[j] else 3)
        scr[i, j] = 0 if SCRIPT[i] == SCRIPT[j] else 1
    out = dict(family=fam, script=scr)
    if tok is not None:
        out["token"] = 1 - tok["tok_hist_int"]
        out["charbigram"] = 1 - tok["char_hist_int"]
        lf = np.log(tok["fert"])
        out["fertility"] = np.abs(lf[:, None] - lf[None, :])
        np.fill_diagonal(out["token"], 0); np.fill_diagonal(out["charbigram"], 0)
    return out


def xdist2(ad, at):
    """unbiased squared distances from two independent estimates (dev, test) of the centroids"""
    Dd = ad[:, None, :] - ad[None, :, :]
    Dt = at[:, None, :] - at[None, :, :]
    return (Dd * Dt).sum(-1)


# ------------------------------------------------------------------ attribute algebra
FEATS = {
    "script(Latin,CJK)": [LATIN, CJK],
    "family(IE,Germ,Rom)": [IE, GERM, ROM],
    "script+family": [LATIN, CJK, IE, GERM, ROM],
    "2x2(Latin,IE)": [LATIN, IE],
}


def loo_r2(A_fit, A_eval, F, drop=()):
    """leave-one-language-out prediction of centroid i from attribute features (fit on the other 11,
    min-norm LSQ); error relative to predicting the mean of the other 11.  A_fit/A_eval: [12,d]."""
    idx = [i for i in range(L) if i not in drop]
    num = den = 0.0
    per = {}
    for i in idx:
        o = [j for j in idx if j != i]
        beta = np.linalg.pinv(F[o]) @ A_fit[o]
        e = ((A_eval[i] - F[i] @ beta) ** 2).sum()
        e0 = ((A_eval[i] - A_fit[o].mean(0)) ** 2).sum()
        num += e; den += e0
        per[i] = e / e0
    return 1 - num / den, per


def attribute_test(A_fit, A_eval, n_perm=2000, seed=0, extra=None, drop=()):
    rs = np.random.RandomState(seed)
    feats = dict(FEATS)
    if extra is not None:
        for k, v in extra.items():
            feats[k] = v
    res = {}
    for name, cols in feats.items():
        F = np.column_stack([np.ones(L)] + list(cols))
        r2, per = loo_r2(A_fit, A_eval, F, drop)
        null = np.array([loo_r2(A_fit, A_eval, F[rs.permutation(L)], drop)[0] for _ in range(n_perm)])
        res[name] = dict(r2=r2, null_mean=null.mean(), null_p95=np.quantile(null, 0.95),
                         p=(1 + (null >= r2).sum()) / (1 + n_perm), per=per)
    return res


CELLS = dict(LI=[0, 1, 2, 3], LN=[9, 10, 11], NI=[4, 5], NN=[6, 7, 8])


def crossing(A, cells=CELLS):
    m = {k: A[v].mean(0) for k, v in cells.items()}
    dL, dN = m["LI"] - m["LN"], m["NI"] - m["NN"]         # IE effect within Latin / within non-Latin
    cos = float(dL @ dN / np.sqrt((dL @ dL) * (dN @ dN)))
    inter = float(((dL - dN) ** 2).sum() / ((dL ** 2).sum() + (dN ** 2).sum()))
    return cos, inter


def crossing_test(A, n_perm=5000, seed=0, drop=()):
    rs = np.random.RandomState(seed)
    cells = {k: [i for i in v if i not in drop] for k, v in CELLS.items()}
    cos, inter = crossing(A, cells)
    sizes = [len(v) for v in cells.values()]
    pool = [i for i in range(L) if i not in drop]
    nulls = []
    for _ in range(n_perm):
        p = rs.permutation(pool)
        c, s = {}, 0
        for k, sz in zip(cells, sizes):
            c[k] = list(p[s:s + sz]); s += sz
        nulls.append(crossing(A, c))
    nulls = np.array(nulls)
    return dict(cos=cos, inter=inter, null_cos_mean=nulls[:, 0].mean(), null_cos_p95=np.quantile(nulls[:, 0], 0.95),
                p_cos=(1 + (nulls[:, 0] >= cos).sum()) / (1 + n_perm),
                null_inter_mean=nulls[:, 1].mean(), p_inter=(1 + (nulls[:, 1] <= inter).sum()) / (1 + n_perm))


# ------------------------------------------------------------------ metric / tree structure
def fourpoint(D):
    """mean relative four-point defect eps = (S1-S2)/(S1-S3) over quadruples (0 for a tree metric) and
    Gromov delta relative to diameter"""
    eps, dl = [], []
    for q in itertools.combinations(range(D.shape[0]), 4):
        i, j, k, l = q
        s = sorted([D[i, j] + D[k, l], D[i, k] + D[j, l], D[i, l] + D[j, k]], reverse=True)
        eps.append((s[0] - s[1]) / max(s[0] - s[2], 1e-12))
        dl.append((s[0] - s[1]) / 2)
    return float(np.mean(eps)), float(2 * np.max(dl) / D.max())


def coph(D, method="average"):
    Z = linkage(squareform(D, checks=False), method)
    return float(cophenet(Z, squareform(D, checks=False))[0])


def gaussian_null_tree(G, n=300, seed=0):
    """null: Gaussian point clouds whose centred Gram spectrum matches the observed configuration"""
    rs = np.random.RandomState(seed)
    w = np.clip(np.linalg.eigvalsh(G), 0, None)[::-1]
    out = []
    for _ in range(n):
        X = rs.randn(L, L) * np.sqrt(w / L)[None]         # [points, dims]
        X -= X.mean(0)
        u, s, vt = np.linalg.svd(X, full_matrices=False)  # re-impose the spectrum exactly
        X = u * np.sqrt(w)[: len(s)]
        D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
        out.append((*fourpoint(D), coph(D)))
    return np.array(out)


def mantel(D, R, n_perm=5000, seed=0, method="spearman"):
    """rank-based Mantel test; labels of D permuted (ranks precomputed: permutation only reorders them)"""
    from scipy.stats import rankdata
    rs = np.random.RandomState(seed)
    iu = np.triu_indices(L, 1)
    RD = np.zeros((L, L)); RD[iu] = rankdata(D[iu]); RD = RD + RD.T
    y = rankdata(R[iu]); y = (y - y.mean()) / y.std()

    def corr(v):
        v = (v - v.mean()) / v.std()
        return float((v * y).mean())
    r0 = corr(RD[iu])
    null = np.array([corr(RD[np.ix_(p, p)][iu]) for p in (rs.permutation(L) for _ in range(n_perm))])
    return r0, float((1 + (null >= r0).sum()) / (1 + n_perm))


def mrm(D, refs, n_perm=2000, seed=0):
    """multiple regression on distance matrices (ranks), permutation p for each standardized coef"""
    rs = np.random.RandomState(seed)
    iu = np.triu_indices(L, 1)
    from scipy.stats import rankdata

    def fit(Dm):
        y = rankdata(Dm[iu]); y = (y - y.mean()) / y.std()
        X = np.column_stack([(lambda v: (v - v.mean()) / (v.std() + 1e-12))(rankdata(R[iu])) for R in refs.values()])
        X = np.column_stack([np.ones(len(y)), X])
        b = np.linalg.lstsq(X, y, rcond=None)[0][1:]
        r2 = 1 - ((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2).sum() / (y ** 2).sum()
        return b, r2
    b0, r20 = fit(D)
    nb = []
    for _ in range(n_perm):
        p = rs.permutation(L)
        nb.append(fit(D[np.ix_(p, p)])[0])
    nb = np.array(nb)
    return {k: (float(b0[q]), float((1 + (np.abs(nb[:, q]) >= abs(b0[q])).sum()) / (1 + n_perm)))
            for q, k in enumerate(refs)}, float(r20)


# ------------------------------------------------------------------ depth dynamics
def mds(A):
    u, s, vt = np.linalg.svd(A - A.mean(0), full_matrices=False)
    return u * s


def shape_sim(A, B):
    """1 - Procrustes distance with rotation + scale (gauge-free, ambient rotation allowed)"""
    X, Y = mds(A), mds(B)
    return float(np.linalg.svd(X.T @ Y, compute_uv=False).sum() / np.sqrt((X ** 2).sum() * (Y ** 2).sum()))


def ident_sim(A, B):
    """ambient cosine of the two configurations, no rotation allowed"""
    A = A - A.mean(0); B = B - B.mean(0)
    return float((A * B).sum() / np.sqrt((A ** 2).sum() * (B ** 2).sum()))


def subspace_overlap(A, B):
    qa = np.linalg.svd(A - A.mean(0), full_matrices=False)[2][:11]
    qb = np.linalg.svd(B - B.mean(0), full_matrices=False)[2][:11]
    return float((np.linalg.svd(qa @ qb.T, compute_uv=False) ** 2).mean())
