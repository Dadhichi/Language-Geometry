"""holo.py -- holonomy / curvature analysis of pairwise cross-language maps on K12.

Conventions follow fit.py (row vectors, per-language PCA coords, map i->j: z_j ~ z_i R_ij).
Holonomy based at b around (b,p,q):  H = R_bp R_pq R_qb  in O(k), acting on language b's own coords.
Composition error of the ordered triple (b -> p -> q):  D = M_bp M_pq - M_bq  (lives in q coords);
for procrustes D = (H - I) R_bq, so c = ||T_b D||^2/||T_b||^2 = ||T_b (H - I)||^2/||T_b||^2.

Everything reported is gauge invariant: spectra of H (conjugation), singular spectra of T_b(H-I),
commutator norms of holonomies based at the same vertex, and AMBIENT (3584-d) lifts P_q D^T S D P_q^T,
which are invariant under P_q -> P_q O_q.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys
import itertools
import numpy as np
import scipy.linalg as sla

REPO = r"C:\Users\ASUS\OneDrive\Documents\Projects\Language Structure in LLMs\Language-Geometry"
sys.path.insert(0, REPO)

DATA = r"C:\Users\ASUS\Documents\lang-geom"
LANGS = ["eng", "deu", "fra", "spa", "rus", "hin", "arb", "zho", "jpn", "tur", "vie", "ind"]
SCRIPT = ["Latn", "Latn", "Latn", "Latn", "Cyrl", "Deva", "Arab", "Hans", "Jpan", "Latn", "Latn", "Latn"]
FAMILY = ["Germanic", "Germanic", "Romance", "Romance", "Slavic", "IndoAryan", "Semitic", "Sinitic",
          "Japonic", "Turkic", "Austroasiatic", "Austronesian"]
L = 12
TRI = list(itertools.combinations(range(L), 3))
OTRI = list(itertools.permutations(range(L), 3))


def load(split, layer):
    return np.load(os.path.join(DATA, f"mean_{split}_L{layer}.f16.npy"), mmap_mode="r")


# ----------------------------------------------------------------------------- projection

def pca(x, k):
    """x [n,d] float32 -> mean g, top-k basis P [d,k] (descending).  Gram trick; f32 GEMM, f64 eigh.
    Same estimator as fit.pca_basis (exact top-k right singular vectors of the centred cloud)."""
    g = x.mean(0)
    xc = x - g
    G = (xc @ xc.T).astype(np.float64)
    w, U = np.linalg.eigh(G)
    U = U[:, ::-1][:, :k].astype(np.float32)
    w = w[::-1][:k].clip(1e-30)
    P = (xc.T @ U) / np.sqrt(w).astype(np.float32)[None, :]
    return g, np.ascontiguousarray(P, dtype=np.float32)


def project(Xf, Xt, k):
    """Per-language PCA on the fit split.  Returns A [L,n,k], T [L,m,k] (f64, centred on fit mean),
    bases [L,d,k] f32, means [L,d] f32."""
    A, T, P, g = [], [], [], []
    for i in range(Xf.shape[0]):
        xi = np.asarray(Xf[i], dtype=np.float32)
        gi, Pi = pca(xi, k)
        A.append(((xi - gi) @ Pi).astype(np.float64))
        T.append(((np.asarray(Xt[i], dtype=np.float32) - gi) @ Pi).astype(np.float64))
        P.append(Pi)
        g.append(gi)
        del xi
    A, T = np.stack(A), np.stack(T)
    mu = A.mean(1, keepdims=True)
    return A - mu, T - mu, np.stack(P), np.stack(g)


# ----------------------------------------------------------------------------- maps

def polar(M):
    u, _, vt = np.linalg.svd(M)
    return u @ vt


def fit_maps(A, rel=1e-3):
    Ln, n, k = A.shape
    Acat = A.transpose(1, 0, 2).reshape(n, Ln * k)
    C = (Acat.T @ Acat).reshape(Ln, k, Ln, k).transpose(0, 2, 1, 3)       # C[i,j] = A_i^T A_j
    R = np.zeros((Ln, Ln, k, k))
    W = np.zeros((Ln, Ln, k, k))
    I = np.eye(k)
    for i in range(Ln):
        R[i, i] = W[i, i] = I
        G = C[i, i]
        reg = G + rel * np.trace(G) / k * I
        for j in range(Ln):
            if j == i:
                continue
            if j > i:
                R[i, j] = polar(C[i, j])
                R[j, i] = R[i, j].T
            W[i, j] = np.linalg.solve(reg, C[i, j])
    return R, W


def gpa(A, R, iters=30):
    Ln, n, k = A.shape
    Q = np.stack([np.eye(k)] + [R[0, j].T for j in range(1, Ln)])
    for _ in range(iters):
        M = np.matmul(A, Q).mean(0)
        U, _, Vt = np.linalg.svd(np.matmul(A.transpose(0, 2, 1), M))
        Q = U @ Vt
    return Q


# ----------------------------------------------------------------------------- consistent surrogates

def model_O(A, T, R):
    """Orthogonal-gauge (GPA hub) model, exactly as fit.analyze: hat_i = M Q_i^T."""
    Ln, n, k = A.shape
    Q = gpa(A, R)
    M = np.matmul(A, Q).mean(0)
    hat = np.matmul(M[None], Q.transpose(0, 2, 1))
    Mt = np.matmul(T, Q).mean(0)
    hat_t = np.matmul(Mt[None], Q.transpose(0, 2, 1))
    E, Et = A - hat, T - hat_t
    rss = float((E ** 2).sum())
    dof = Ln * n * k - n * k - (Ln - 1) * k * (k - 1) / 2 - Ln * k
    infl = float(np.sqrt(rss / dof / (rss / (Ln * n * k))))
    return dict(hat=hat, E=E, hat_t=hat_t, E_t=Et, infl=infl, infl_t=float(np.sqrt(Ln / (Ln - 1))), Q=Q)


def model_GL(A, T):
    """Flat LINEAR gauge: A_i ~ S G_i, G_i in GL(k).  Least squares = rank-k SVD of [A_0 .. A_L-1]."""
    Ln, n, k = A.shape
    Acat = A.transpose(1, 0, 2).reshape(n, Ln * k)
    U, s, Vt = np.linalg.svd(Acat, full_matrices=False)
    V = Vt[:k]                                                     # [k, Lk]
    S = U[:, :k] * s[:k]
    hat = (S @ V).reshape(n, Ln, k).transpose(1, 0, 2)
    Tcat = T.transpose(1, 0, 2).reshape(T.shape[1], Ln * k)
    St = Tcat @ V.T
    hat_t = (St @ V).reshape(T.shape[1], Ln, k).transpose(1, 0, 2)
    E, Et = A - hat, T - hat_t
    dof = Ln * n * k - n * k - (Ln - 1) * k * k - Ln * k
    infl = float(np.sqrt(Ln * n * k / dof))
    return dict(hat=hat, E=E, hat_t=hat_t, E_t=Et, infl=infl, infl_t=float(np.sqrt(Ln / (Ln - 1))))


def surrogate(Xf, Xt, means, bases, mdl, seed, plant=None, ambient_out=False, k=None):
    """Consistent surrogate built in the AMBIENT space as in fit.analyze_with_null: model part +
    independently permuted, dof-inflated in-subspace residual + the real out-of-subspace part, lifted to
    d dims.  plant: optional dict(add_f=[L,n,d] or None per lang, add_t=...) extra ambient signal.
    Returns ambient f16 arrays (ambient_out) or the re-projected (A, T, bases, means)."""
    rs = np.random.RandomState(seed)
    k = k or bases.shape[2]
    outf, outt, A, T, P2, g2 = [], [], [], [], [], []
    for i in range(Xf.shape[0]):
        P, m = bases[i], means[i]
        res = []
        for X, hat, E, infl in ((Xf, mdl["hat"], mdl["E"], mdl["infl"]),
                                (Xt, mdl["hat_t"], mdl["E_t"], mdl["infl_t"])):
            xc = np.asarray(X[i], dtype=np.float32) - m
            z = (hat[i] + infl * E[i][rs.permutation(len(E[i]))]).astype(np.float32)
            res.append(z @ P.T + (xc - (xc @ P) @ P.T) + m)
            del xc
        if plant is not None:
            if plant["add_f"][i] is not None:
                res[0] += plant["add_f"][i]
                res[1] += plant["add_t"][i]
        if ambient_out:
            outf.append(res[0].astype(np.float16))
            outt.append(res[1].astype(np.float16))
            continue
        gi, Pi = pca(res[0], k)
        A.append(((res[0] - gi) @ Pi).astype(np.float64))
        T.append(((res[1] - gi) @ Pi).astype(np.float64))
        P2.append(Pi)
        g2.append(gi)
    if ambient_out:
        return np.stack(outf), np.stack(outt)
    A, T = np.stack(A), np.stack(T)
    mu = A.mean(1, keepdims=True)
    return A - mu, T - mu, np.stack(P2), np.stack(g2)


# ----------------------------------------------------------------------------- holonomy analysis

def angles_of(H):
    """|arg| of the eigenvalues of an orthogonal matrix, sorted descending (k values; planes appear twice)."""
    return np.sort(np.abs(np.angle(np.linalg.eigvals(H))))[::-1]


def comm_stats(Ks, rs, npair=150):
    """Mean normalised commutator ||[K_a,K_b]||/(||K_a|| ||K_b||) over random pairs of a stack of skews."""
    Kn = Ks / np.linalg.norm(Ks, axis=(1, 2), keepdims=True)
    n = len(Kn)
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    sel = rs.choice(len(pairs), size=min(npair, len(pairs)), replace=False)
    out = []
    for s in sel:
        a, b = pairs[s]
        C = Kn[a] @ Kn[b] - Kn[b] @ Kn[a]
        out.append(np.linalg.norm(C))
    return float(np.mean(out))


def analyze_conn(A, T, seed=0, with_comm=True):
    """All holonomy statistics for one dataset (fit coords A, test coords T)."""
    Ln, n, k = A.shape
    R, W = fit_maps(A)
    lam = (A ** 2).sum(1)                                   # [L,k] (diagonal: PCA coords of the fit data)
    S = np.matmul(T.transpose(0, 2, 1), T)                  # [L,k,k] test second moments
    trS = np.trace(S, axis1=1, axis2=2)
    I = np.eye(k)
    out = {}
    # pairwise rho on test (sanity vs fit.py)
    rho_p, rho_r = np.zeros((Ln, Ln)), np.zeros((Ln, Ln))
    for i in range(Ln):
        for j in range(Ln):
            if i != j:
                rho_p[i, j] = ((T[i] @ R[i, j] - T[j]) ** 2).sum() / trS[j]
                rho_r[i, j] = ((T[i] @ W[i, j] - T[j]) ** 2).sum() / trS[j]
    out["rho_proc"], out["rho_ridge"] = rho_p, rho_r
    # ordered-triple composition errors + target-side Grams
    c_p, c_r = np.full((Ln, Ln, Ln), np.nan), np.full((Ln, Ln, Ln), np.nan)
    Gp_test = np.zeros((Ln, k, k))
    pr_r = np.zeros((Ln, k))
    for b, p, q in OTRI:
        D = R[b, p] @ R[p, q] - R[b, q]
        SD = S[b] @ D
        c_p[b, p, q] = (D * SD).sum() / trS[b]
        Gp_test[q] += D.T @ SD / trS[b]
        D = W[b, p] @ W[p, q] - W[b, q]
        dd = (D * (S[b] @ D)).sum(0) / trS[b]
        c_r[b, p, q] = dd.sum()
        pr_r[q] += dd
    cnt = (Ln - 1) * (Ln - 2)
    Gp_test /= cnt
    out.update(c_proc=c_p, c_ridge=c_r, Gp_test=Gp_test)
    # discrepancy mass by PCA rank of the target language (PCA basis is data-determined => gauge-covariant)
    out["prof_proc"] = np.stack([np.diag(G) for G in Gp_test])
    out["prof_ridge"] = pr_r / cnt
    # holonomy spectra per unordered triangle (base = first vertex; spectrum is base-independent)
    ang = np.zeros((len(TRI), k))
    wsv = np.zeros((len(TRI), 3, k))           # singular spectrum^2 of T_b(H-I)/||T_b|| for each base
    reig = np.zeros((len(TRI), k), dtype=np.complex128)
    for t, (a, b, c) in enumerate(TRI):
        H = R[a, b] @ R[b, c] @ R[c, a]
        ang[t] = angles_of(H)
        for s_, (x, y, z) in enumerate(((a, b, c), (b, c, a), (c, a, b))):
            Hx = R[x, y] @ R[y, z] @ R[z, x]
            D = Hx - I
            wsv[t, s_] = np.sort(np.linalg.eigvalsh(D.T @ S[x] @ D).clip(0))[::-1] / trS[x]
        HW = W[a, b] @ W[b, c] @ W[c, a]
        ev = np.linalg.eigvals(HW)
        reig[t] = ev[np.argsort(-np.abs(np.angle(ev)))]
    out.update(angles=ang, wsv=wsv, ridge_eigs=reig)
    # abelian test: holonomies based at the same vertex
    if with_comm:
        rs = np.random.RandomState(seed)
        cu, cw = np.zeros(Ln), np.zeros(Ln)
        for b in range(Ln):
            Ks, Kw = [], []
            sq = np.sqrt(lam[b] / lam[b].mean())
            for p, q in itertools.combinations([x for x in range(Ln) if x != b], 2):
                H = R[b, p] @ R[p, q] @ R[q, b]
                K = 0.5 * (H - H.T)
                Ks.append(K)
                Kw.append(sq[:, None] * K * sq[None, :])
            cu[b] = comm_stats(np.stack(Ks), rs)
            cw[b] = comm_stats(np.stack(Kw), rs)
        out["comm_unw"], out["comm_w"] = cu, cw
    # hub-frame edge discrepancy (pairwise map vs joint-gauge map)
    Q = gpa(A, R)
    u = np.zeros((Ln, Ln))
    for i in range(Ln):
        for j in range(Ln):
            if i != j:
                D = R[i, j] - Q[i] @ Q[j].T
                u[i, j] = (D * (S[i] @ D)).sum() / trS[i]
    out["u_edge"] = u
    # canonical correlations between languages' retained subspaces (gauge invariant; fit data)
    sd = np.sqrt(lam)
    cc = np.zeros((Ln, Ln, k))
    for i in range(Ln):
        for j in range(i + 1, Ln):
            Cij = (A[i].T @ A[j]) / sd[i][:, None] / sd[j][None, :]
            cc[i, j] = cc[j, i] = np.linalg.svd(Cij, compute_uv=False)
    out["cancorr"] = cc
    out["R"], out["W"] = R, W
    return out


def random_skew_comm(k, n=60, seed=0):
    rs = np.random.RandomState(seed)
    X = rs.randn(n, k, k)
    return comm_stats(0.5 * (X - X.transpose(0, 2, 1)), rs)


def lowrank_skew_comm(k, r, n=60, seed=0):
    """Baseline: skews supported in one common random r-dim subspace (abelian iff r=2)."""
    rs = np.random.RandomState(seed)
    B = np.linalg.qr(rs.randn(k, r))[0]
    X = rs.randn(n, r, r)
    X = 0.5 * (X - X.transpose(0, 2, 1))
    return comm_stats(np.einsum("ka,nab,lb->nkl", B, X, B), rs)


# ----------------------------------------------------------------------------- ambient lifts

def factor(bases, G):
    """Ambient factor F [d, L k] with F F^T = mean_q P_q G_q P_q^T (G_q PSD k x k)."""
    cols = []
    for q in range(len(G)):
        w, V = np.linalg.eigh(0.5 * (G[q] + G[q].T))
        cols.append(bases[q].astype(np.float64) @ (V * np.sqrt(w.clip(0))))
    return (np.concatenate(cols, 1) / np.sqrt(len(G))).astype(np.float32)


def orth(X):
    return np.linalg.qr(np.asarray(X, dtype=np.float64))[0]


def share(B, F):
    """Fraction of tr(F F^T) inside the orthonormal subspace B."""
    return float(((B.T @ F) ** 2).sum() / (F.astype(np.float64) ** 2).sum())


def excess_eig(Fr, Fn_list, top=64):
    """Top eigenpairs of dG = Fr Fr^T - mean_n Fn Fn^T (ambient)."""
    G = Fr.astype(np.float32) @ Fr.T.astype(np.float32)
    for Fn in Fn_list:
        G -= (Fn @ Fn.T) / len(Fn_list)
    d = G.shape[0]
    tr = float(np.trace(G))
    w, V = sla.eigh(G, subset_by_index=[d - top, d - 1], driver="evr")
    return w[::-1].astype(np.float64), V[:, ::-1].astype(np.float32), tr


def capture(V, w, Vd, trd):
    """Fraction of tr(dG_2) captured by subspace V, with dG_2 ~ Vd diag(w) Vd^T (top eigenpairs)."""
    return float(((V.T @ Vd) ** 2 * w[None, :]).sum() / trd)
