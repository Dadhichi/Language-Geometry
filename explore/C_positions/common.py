"""Shared helpers for investigator C (languages as positions; ambient coordinates)."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"
import itertools
import numpy as np
import scipy.linalg as sl

DATA = "C:/Users/ASUS/Documents/lang-geom"
OUT = os.path.dirname(os.path.abspath(__file__))
LANGS = ["eng", "deu", "fra", "spa", "rus", "hin", "arb", "zho", "jpn", "tur", "vie", "ind"]
SCRIPT = ["Latn", "Latn", "Latn", "Latn", "Cyrl", "Deva", "Arab", "Hans", "Jpan", "Latn", "Latn", "Latn"]
FAMILY = ["Germ", "Germ", "Rom", "Rom", "Slav", "IndAr", "Sem", "Sin", "Jap", "Turk", "AuAs", "AuNe"]
IE = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
NL = 12
PAIRS = list(itertools.combinations(range(NL), 2))
OPAIRS = list(itertools.permutations(range(NL), 2))
TRIPLES = list(itertools.combinations(range(NL), 3))


def load16(split, layer):
    return np.load(f"{DATA}/mean_{split}_L{layer}.f16.npy")


def f32(x):
    return np.asarray(x, dtype=np.float32)


def pooled_basis(X16, mmax):
    """Per-language centring, pooled covariance, top-mmax eigenvectors (descending).
    Returns mu [L,d] (f32), U [d,mmax] (f32), explained fractions [mmax]."""
    Lg, n, d = X16.shape
    mu = np.stack([f32(X16[i]).mean(0) for i in range(Lg)])
    C = np.zeros((d, d), np.float32)
    for i in range(Lg):
        A = f32(X16[i]) - mu[i]
        C += A.T @ A
    C64 = C.astype(np.float64)
    del C
    tot = float(np.trace(C64))
    w, U = sl.eigh(C64, subset_by_index=[d - mmax, d - 1])
    del C64
    return mu, np.ascontiguousarray(U[:, ::-1].astype(np.float32)), w[::-1] / tot


def project(X16, mu, U):
    return np.stack([((f32(X16[i]) - mu[i]) @ U).astype(np.float64) for i in range(len(X16))])


def polar(M):
    u, _, vt = np.linalg.svd(M)
    return u @ vt


def procrustes(A, B):
    return polar(A.T @ B)


def ridge(A, B, rel=1e-3):
    G = A.T @ A
    m = G.shape[0]
    return np.linalg.solve(G + rel * np.trace(G) / m * np.eye(m), A.T @ B)


def gpa(Z, iters=40, Q0=None):
    """min sum_i ||Z_i Q_i - H||^2 over Q_i in O(m).  Z [L,n,m]."""
    Lg, n, m = Z.shape
    if Q0 is None:
        Q = np.stack([np.eye(m)] + [procrustes(Z[i], Z[0]) for i in range(1, Lg)])
    else:
        Q = Q0.copy()
    for _ in range(iters):
        H = np.mean([Z[i] @ Q[i] for i in range(Lg)], 0)
        Q = np.stack([procrustes(Z[i], H) for i in range(Lg)])
    return Q


def log_orth(R):
    """Real skew log of an orthogonal matrix via the commuting sym/skew parts.
    Returns (L skew [m,m], plane angles sorted desc [m//2], det sign)."""
    R = np.asarray(R, dtype=np.float64)
    Sy, Sk = 0.5 * (R + R.T), 0.5 * (R - R.T)
    c, V = np.linalg.eigh(Sy)
    c = np.clip(c, -1.0, 1.0)
    phi = np.arccos(c)
    s = np.sqrt(np.clip(1 - c ** 2, 0, None))
    f = np.where(s > 1e-7, phi / np.maximum(s, 1e-7), 1.0)
    f = np.minimum(f, 1e4)
    Lg = (V * f) @ V.T @ Sk
    Lg = 0.5 * (Lg - Lg.T)
    ang = np.sort(phi)[::-1]
    return Lg, ang[0::2], float(np.sign(np.linalg.det(R)))


def expm_skew(K):
    return sl.expm(K)


def angle_stats(ang):
    a2 = ang ** 2
    tot = a2.sum()
    return dict(sumsq=float(tot), pr=float(tot ** 2 / max((a2 ** 2).sum(), 1e-30)),
                top4=float(np.sort(a2)[::-1][:4].sum() / max(tot, 1e-30)),
                frac_pi=float((ang > 0.9 * np.pi).mean()), max_ang=float(ang.max()))


def rho(pred, tgt):
    return float(((pred - tgt) ** 2).sum() / (tgt ** 2).sum())


def mantel(D, M, reps=2000, seed=0):
    """Pearson corr between upper triangles of D and M, and a permutation p-value (one-sided, D ~ M)."""
    iu = np.triu_indices(len(D), 1)
    x, y = D[iu], M[iu]
    r0 = np.corrcoef(x, y)[0, 1]
    rs = np.random.RandomState(seed)
    cnt = 0
    for _ in range(reps):
        p = rs.permutation(len(D))
        if np.corrcoef(D[np.ix_(p, p)][iu], y)[0, 1] >= r0:
            cnt += 1
    return float(r0), float((cnt + 1) / (reps + 1))


DIFF_SCRIPT = np.array([[float(SCRIPT[i] != SCRIPT[j]) for j in range(NL)] for i in range(NL)])
DIFF_FAMILY = np.array([[float(FAMILY[i] != FAMILY[j]) for j in range(NL)] for i in range(NL)])
