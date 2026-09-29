"""Tree-symmetry statistics on a 12x12 centroid Gram matrix (all gauge-free, metric supplied through the Gram).

Model (Brownian motion on a phylogeny, leaves exchangeable within clades):
    D2_ij = ||a_i - a_j||^2 = sum_i-leaf + sum_j-leaf + sum_{splits S separating i,j} b_S         (b >= 0)
The star part (one leaf length per language) absorbs per-language 'distance from the centre' (e.g. Hindi's
fertility dilution); the internal splits carry the genealogy.  Covariates (token distance, |dlog fertility|,
script-different) enter as extra non-negative columns.
"""
import itertools
import numpy as np
from scipy.optimize import nnls
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

LANGS = ["eng", "deu", "fra", "spa", "rus", "hin", "arb", "zho", "jpn", "tur", "vie", "ind"]
L = 12
IU = np.triu_indices(L, 1)
NP = len(IU[0])

TREES = {
    "glotto": [(0, 1, 2, 3, 4, 5), (0, 1), (2, 3)],
    "script": [(0, 1, 2, 3, 9, 10, 11)],
    "script+Han": [(0, 1, 2, 3, 9, 10, 11), (7, 8)],
    "wordorder": [(5, 8, 9)],
}
GLOTTO_SPLITS = {"IE": (0, 1, 2, 3, 4, 5), "Germ": (0, 1), "Rom": (2, 3)}
SCRIPT = ["Latn", "Latn", "Latn", "Latn", "Cyrl", "Deva", "Arab", "Hani", "Hani", "Latn", "Latn", "Latn"]


def d2_from_gram(K):
    K = 0.5 * (K + K.T)
    dg = np.diag(K)
    return dg[:, None] + dg[None, :] - 2 * K


def split_col(S, iu=IU, n=L):
    m = np.zeros(n, bool)
    m[list(S)] = True
    return (m[iu[0]] ^ m[iu[1]]).astype(float)


def star_cols(iu=IU, n=L):
    X = np.zeros((len(iu[0]), n))
    X[np.arange(len(iu[0])), iu[0]] = 1
    X[np.arange(len(iu[0])), iu[1]] = 1
    return X


STAR = star_cols()


def upgma_splits(D, method="average"):
    """all non-trivial clusters of a UPGMA tree on distance matrix D, with merge heights"""
    Z = linkage(squareform(D, checks=False), method)
    root = to_tree(Z)
    out = []

    def rec(nd):
        if nd.is_leaf():
            return [nd.id]
        lv = rec(nd.left) + rec(nd.right)
        if 1 < len(lv) < D.shape[0] - 1:
            out.append((tuple(sorted(lv)), nd.dist))
        return lv
    rec(root)
    return out


def cov_cols(cov):
    """cov: dict name -> [L,L] distance-like covariate matrices -> [NP, c]"""
    if not cov:
        return np.zeros((NP, 0))
    return np.column_stack([np.asarray(C)[IU] / np.asarray(C)[IU].std() for C in cov.values()])


def sse_nnls(X, y):
    b, r = nnls(X, y)
    return r ** 2, b


def gain(y, base, splits):
    """fraction of the base-model residual SSE removed by adding the split columns (NNLS throughout)"""
    s0, _ = sse_nnls(base, y)
    Xs = np.column_stack([base] + [split_col(S) for S in splits])
    s1, b = sse_nnls(Xs, y)
    return 1 - s1 / s0, b[base.shape[1]:], s0


def relabel(splits, p):
    return [tuple(sorted(p[i] for i in S)) for S in splits]


def perm_null_gain(y, base, splits, n_perm=2000, seed=0, obs=None):
    rs = np.random.RandomState(seed)
    s0, _ = sse_nnls(base, y)
    if obs is None:
        obs = gain(y, base, splits)[0]
    null = np.empty(n_perm)
    for t in range(n_perm):
        p = rs.permutation(L)
        Xs = np.column_stack([base] + [split_col(S) for S in relabel(splits, p)])
        null[t] = 1 - sse_nnls(Xs, y)[0] / s0
    return obs, float((1 + (null >= obs - 1e-12).sum()) / (1 + n_perm)), null


_SUBSETS = {}


def subsets(k):
    if k not in _SUBSETS:
        _SUBSETS[k] = list(itertools.combinations(range(L), k))
    return _SUBSETS[k]


def single_split_rank(y, base, S):
    """exact null: gain of split S vs every subset of the same size (fraction with gain >= observed) and its signed
    OLS coefficient; for |S|=6, complement pairs give identical columns (counted twice, fine)."""
    s0, _ = sse_nnls(base, y)
    g = []
    for T in subsets(len(S)):
        Xs = np.column_stack([base, split_col(T)])
        g.append(1 - sse_nnls(Xs, y)[0] / s0)
    g = np.array(g)
    obs = 1 - sse_nnls(np.column_stack([base, split_col(S)]), y)[0] / s0
    return float(obs), float((g >= obs - 1e-12).mean()), g


# ---------------------------------------------------------------- tree covariance, Haar / g(k)
def ultrametric_K(splits, depth=None):
    """BM covariance from the root of a rooted tree: unit internal edges, leaf edges padded so every leaf has
    the same depth (exchangeable / ultrametric).  K_ij = number of shared internal edges + (depth if i==j)."""
    K = np.zeros((L, L))
    for S in splits:
        v = np.zeros(L); v[list(S)] = 1
        K += np.outer(v, v)
    if depth is None:
        depth = K.diagonal().max() + 1
    K[np.diag_indices(L)] = depth
    return K


def root_weights(splits, kind="family"):
    """family-balanced root: each top-level clade (maximal split or singleton) gets equal weight, shared equally."""
    if kind == "grand":
        return np.ones(L) / L
    top = [S for S in splits if not any(set(S) < set(T) for T in splits)]
    covered = set().union(*[set(S) for S in top]) if top else set()
    groups = [list(S) for S in top] + [[i] for i in range(L) if i not in covered]
    w = np.zeros(L)
    for g_ in groups:
        w[g_] = 1.0 / (len(groups) * len(g_))
    return w


def center(K, w):
    H = np.eye(L) - np.outer(np.ones(L), w)
    return H @ K @ H.T


def gk(Ka, Kb, ks=(1, 2, 3, 4)):
    ua = np.linalg.eigh(Ka)[1][:, ::-1]
    ub = np.linalg.eigh(Kb)[1][:, ::-1]
    return np.array([np.square(ua[:, :k].T @ ub[:, :k]).sum() / k for k in ks])


# ---------------------------------------------------------------- Park-style hierarchical orthogonality
def park_vectors(clade, g1, g2, w):
    """combination weights (rows) over languages for: root r (weights w), parent P, children g1, g2 (pairs)"""
    def mean(ix):
        c = np.zeros(L); c[list(ix)] = 1.0 / len(ix); return c
    P, G1, G2 = mean(clade), mean(g1), mean(g2)
    e = np.eye(L)
    vec = {
        "P-r": P - w, "G2-P": G2 - P, "G1-P": G1 - P,
        "g2a-g2b": e[g2[0]] - e[g2[1]], "g1a-g1b": e[g1[0]] - e[g1[1]], "G1-G2": G1 - G2,
    }
    pairs = [("P-r", "G2-P"), ("P-r", "G1-P"), ("g2a-g2b", "G2-P"), ("g1a-g1b", "G1-P"),
             ("g2a-g2b", "P-r"), ("g1a-g1b", "P-r"), ("G1-G2", "P-r")]
    return vec, pairs


def park_cos(K, clade=(0, 1, 2, 3, 4, 5), g1=(0, 1), g2=(2, 3), rootkind="family"):
    K = 0.5 * (K + K.T)
    splits = [tuple(clade), tuple(g1), tuple(g2)]
    w = root_weights(splits, rootkind)
    vec, pairs = park_vectors(clade, g1, g2, w)
    out = []
    for a, b in pairs:
        u, v = vec[a], vec[b]
        nu, nv = u @ K @ u, v @ K @ v
        out.append(float(u @ K @ v / np.sqrt(max(nu, 1e-30) * max(nv, 1e-30))))
    return np.array(out), [f"{a}|{b}" for a, b in pairs]


def park_null(K, n=2000, seed=0, rootkind="family"):
    rs = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        p = rs.permutation(L)
        clade = tuple(p[:6]); g1 = tuple(p[:2]); g2 = tuple(p[2:4])
        out.append(park_cos(K, clade, g1, g2, rootkind)[0])
    return np.array(out)


# ---------------------------------------------------------------- neighbour joining on D2 (descriptive)
def nj_splits(D):
    D = D.copy().astype(float)
    nodes = [(i,) for i in range(D.shape[0])]
    splits = []
    while len(nodes) > 3:
        n = len(nodes)
        r = D.sum(1)
        Q = (n - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i > j:
            i, j = j, i
        new = tuple(sorted(nodes[i] + nodes[j]))
        dn = 0.5 * (D[i] + D[j] - D[i, j])
        keep = [k for k in range(n) if k not in (i, j)]
        D2 = np.zeros((n - 1, n - 1))
        D2[:-1, :-1] = D[np.ix_(keep, keep)]
        D2[-1, :-1] = D2[:-1, -1] = dn[keep]
        D = D2
        nodes = [nodes[k] for k in keep] + [new]
        if 1 < len(new) < L - 1:
            splits.append(new)
    return splits


def canon(S):
    """canonical form of an unrooted split: the side not containing language 0... use smaller side, tie -> with 0"""
    S = set(S); C = set(range(L)) - S
    if len(S) < len(C) or (len(S) == len(C) and 0 in S):
        return tuple(sorted(S))
    return tuple(sorted(C))


def fmt_split(S):
    return "{" + ",".join(LANGS[i] for i in S) + "}"
