"""Tree / split statistics on an n x n centroid Gram (generalises explore/E_tree/treelib.py to any n).

Model (Brownian motion on a phylogeny):  D2_ij = l_i + l_j + sum_{splits S separating i,j} b_S + covariates, all
coefficients >= 0 (NNLS).  The star part (one leaf length per language) absorbs each language's distance from the
centre (e.g. fertility dilution); internal splits carry the structure under test.
"""
import numpy as np
from scipy.optimize import nnls

LANGS = ["eng_Latn", "deu_Latn", "nld_Latn", "swe_Latn", "fra_Latn", "spa_Latn", "por_Latn",
         "rus_Cyrl", "ukr_Cyrl", "pol_Latn", "hrv_Latn", "srp_Cyrl",
         "hin_Deva", "urd_Arab", "mar_Deva", "pes_Arab",
         "arb_Arab", "heb_Hebr", "mlt_Latn", "tur_Latn", "azj_Latn", "kaz_Cyrl", "fin_Latn", "ekk_Latn",
         "cmn_Hans", "cmn_Hant", "jpn_Jpan", "kor_Hang", "ind_Latn", "fil_Latn", "vie_Latn", "khm_Khmr",
         "tam_Taml", "tel_Telu"]
N = len(LANGS)
IX = {l: i for i, l in enumerate(LANGS)}
IX.update({l.split("_")[0]: i for i, l in enumerate(LANGS) if not l.startswith("cmn")})


def G(*names):
    return tuple(sorted(IX[x] for x in names))


# Glottolog-based internal splits (same tree as the E_tree power study, synth30.py GLOTTO30)
GLOTTO = [G("eng", "deu", "nld", "swe", "fra", "spa", "por", "rus", "ukr", "pol", "hrv", "srp", "hin", "urd", "mar", "pes"),
          G("eng", "deu", "nld", "swe"), G("eng", "deu", "nld"), G("fra", "spa", "por"), G("spa", "por"),
          G("rus", "ukr", "pol", "hrv", "srp"), G("rus", "ukr"), G("hrv", "srp"),
          G("hin", "urd", "mar", "pes"), G("hin", "urd", "mar"), G("hin", "urd"),
          G("arb", "heb", "mlt"), G("arb", "mlt"), G("tur", "azj", "kaz"), G("tur", "azj"), G("fin", "ekk"),
          G("cmn_Hans", "cmn_Hant"), G("ind", "fil"), G("vie", "khm"), G("tam", "tel")]
# WALS 83A (Order of Object and Verb) value 1 = OV; kaz not coded -> OV by genus (all coded Turkic are OV)
OV = G("hin", "urd", "mar", "pes", "tur", "azj", "kaz", "jpn", "kor", "tam", "tel")
# secondary: + WALS 83A value 3 'no dominant order' (OV in subordinate clauses)
OV_PLUS_DE_NL = tuple(sorted(OV + G("deu", "nld")))
SAME_LANG_PAIRS = [G("hin", "urd"), G("hrv", "srp"), G("cmn_Hans", "cmn_Hant")]
SCRIPT = [l.split("_")[1] for l in LANGS]
SCRIPT_SPLITS = [tuple(i for i in range(N) if SCRIPT[i] == s) for s in sorted(set(SCRIPT))
                 if sum(x == s for x in SCRIPT) > 1]                       # Latn, Cyrl, Arab, Deva


class Space:
    def __init__(self, n=N):
        self.n = n
        self.iu = np.triu_indices(n, 1)
        m = len(self.iu[0])
        self.star = np.zeros((m, n))
        self.star[np.arange(m), self.iu[0]] = 1
        self.star[np.arange(m), self.iu[1]] = 1

    def col(self, S):
        v = np.zeros(self.n, bool)
        v[list(S)] = True
        return (v[self.iu[0]] ^ v[self.iu[1]]).astype(float)

    def vec(self, M):
        return np.asarray(M, float)[self.iu]

    def covcols(self, mats):
        """distance-like covariate matrices -> std-scaled non-negative columns"""
        cols = [self.vec(M) for M in mats]
        return np.column_stack([c / c.std() for c in cols]) if cols else np.zeros((len(self.iu[0]), 0))


def d2_from_gram(K):
    K = 0.5 * (K + K.T)
    g = np.diag(K)
    return g[:, None] + g[None, :] - 2 * K


def sse(X, y):
    return nnls(X, y)[1] ** 2


def gain(sp, y, base, splits):
    """fraction of the base model's residual SSE removed by adding the split columns (NNLS)"""
    s0 = sse(base, y)
    return 1 - sse(np.column_stack([base] + [sp.col(S) for S in splits]), y) / s0


def perm_relabel(sp, y, base, splits, n_perm, seed):
    """null: same tree topology, leaf labels permuted (base fixed)"""
    rs = np.random.RandomState(seed)
    obs = gain(sp, y, base, splits)
    null = np.empty(n_perm)
    for t in range(n_perm):
        p = rs.permutation(sp.n)
        null[t] = gain(sp, y, base, [tuple(sorted(p[i] for i in S)) for S in splits])
    return float(obs), float((1 + (null >= obs - 1e-12).sum()) / (1 + n_perm)), null


def perm_subset(sp, y, base, S, n_perm, seed):
    """null for a single split: random subsets of the same size"""
    rs = np.random.RandomState(seed)
    obs = gain(sp, y, base, [S])
    null = np.array([gain(sp, y, base, [tuple(rs.choice(sp.n, len(S), replace=False))]) for _ in range(n_perm)])
    return float(obs), float((1 + (null >= obs - 1e-12).sum()) / (1 + n_perm)), null


def pair_residual_percentiles(sp, y, base, pairs):
    """percentile (0 = closest) of each pair's NNLS residual among all pairs; low = closer than the base predicts"""
    b, _ = nnls(base, y)
    r = y - base @ b
    order = r.argsort().argsort() / (len(r) - 1)
    idx = {(i, j): k for k, (i, j) in enumerate(zip(*sp.iu))}
    return np.array([order[idx[P]] for P in pairs]), r
