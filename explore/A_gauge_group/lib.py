"""Shared helpers for investigator A (which group relates the languages?).

Everything downstream of the per-language PCA is computed from k x k Gram matrices:
    Gf[i,j] = A_i^T A_j / n   (fit split, centred by the fit mean of each language)
    Gt[i,j] = T_i^T T_j       (test split, centred by the FIT mean)
so rho, composition error c, cross-covariance spectra etc. are exact and cheap.
All statistics are invariant under z_i -> z_i O_i (per-language orthogonal basis change).
"""
import os
import sys

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import numpy as np

REPO = r"C:\Users\ASUS\OneDrive\Documents\Projects\Language Structure in LLMs\Language-Geometry"
DATA = r"C:\Users\ASUS\Documents\lang-geom"
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
import fit  # noqa: E402

LANGS = ["eng", "deu", "fra", "spa", "rus", "hin", "arb", "zho", "jpn", "tur", "vie", "ind"]
L = 12


def load(split, layer):
    X = np.load(os.path.join(DATA, f"mean_{split}_L{layer}.f16.npy"), mmap_mode="r")
    out = np.empty(X.shape, dtype=np.float32)
    for i in range(X.shape[0]):
        out[i] = X[i]
    return out


def project(Xf, Xt, k):
    """per-language PCA (fit.pca_basis, exact) -> Zf [L,n,k], Zt [L,m,k], bases [(g,P)]."""
    bases = [fit.pca_basis(Xf[i], k) for i in range(len(Xf))]
    Zf = np.stack([(Xf[i] - g) @ P for i, (g, P) in enumerate(bases)])
    Zt = np.stack([(Xt[i] - g) @ P for i, (g, P) in enumerate(bases)])
    return Zf, Zt, bases


def grams(Zf, Zt):
    mu = Zf.mean(1, keepdims=True)
    A = (Zf - mu).astype(np.float64)
    T = (Zt - mu).astype(np.float64)
    n = A.shape[1]
    return blockgram(A) / n, blockgram(T)


def blockgram(A):
    """A [L,n,k] -> [L,L,k,k] with block (i,j) = A_i^T A_j (BLAS)."""
    Ln, n, k = A.shape
    F = A.transpose(1, 0, 2).reshape(n, Ln * k)
    G = F.T @ F
    return np.ascontiguousarray(G.reshape(Ln, k, Ln, k).transpose(0, 2, 1, 3))


def polar(M):
    u, _, vt = np.linalg.svd(M)
    return u @ vt


def sym_pow(S, p, eps=1e-12):
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    return (V * np.clip(w, eps, None) ** p) @ V.T


# ----------------------------------------------------------------------------- map families

def maps_family(Gf, fam, rel=1e-3):
    """Return W [L,L,k,k] for map family `fam` (identity on the diagonal)."""
    Ln, _, k, _ = Gf.shape
    W = np.zeros((Ln, Ln, k, k))
    I = np.eye(k)
    pre = {}
    if fam.startswith("wproc"):
        a = float(fam.split("_")[1])
        for i in range(Ln):
            pre[i] = (sym_pow(Gf[i, i], -a / 2), sym_pow(Gf[i, i], a / 2))
    for i in range(Ln):
        W[i, i] = I
        for j in range(Ln):
            if i == j:
                continue
            C = Gf[i, j]
            if fam == "proc":
                W[i, j] = polar(C)
            elif fam == "sim":
                R = polar(C)
                W[i, j] = R * (np.trace(R.T @ C) / np.trace(Gf[i, i]))
            elif fam == "ridge":
                G = Gf[i, i]
                W[i, j] = np.linalg.solve(G + rel * np.trace(G) / k * I, C)
            elif fam.startswith("wproc"):
                shrink = fam.endswith("_s")
                Si_m, _ = pre[i]
                Sj_m, Sj_p = pre[j]
                O = polar(Si_m @ C @ Sj_m)
                M = Si_m @ O @ Sj_p
                if shrink:
                    M = M * (np.trace(M.T @ C) / np.trace(M.T @ Gf[i, i] @ M))
                W[i, j] = M
            else:
                raise ValueError(fam)
    return W


def gpa_gram(Gf, E=None, iters=30):
    """Generalised Procrustes in Gram form on (optionally pre-conditioned) coords A_i E_i.
    Returns Q [L,k,k] with Q_0 initialised at identity (English gauge)."""
    Ln, _, k, _ = Gf.shape
    if E is None:
        E = np.stack([np.eye(k)] * Ln)
    G = np.transpose(E, (0, 2, 1))[:, None] @ Gf @ E[None]          # Gram of A_i E_i
    Q = np.stack([np.eye(k)] + [polar(G[0, j]).T for j in range(1, Ln)])
    for _ in range(iters):
        Q = np.stack([polar((G[i] @ Q).sum(0)) for i in range(Ln)])
    return Q, G


def joint_maps(Gf, kind, rel=1e-3):
    """Hub-mediated maps.
    sync_O   : W_ij = Q_i Q_j^T   (orthogonal gauge, GPA hub)                       -- fit.py sync_gpa
    sync_W   : W_ij = S_i^-1 Q_i Q_j^T S_j (orthogonal gauge after per-language whitening)
    hubO_wie : regress A_i -> hub, hub -> A_j (hub from plain GPA)    -- GL hub, Wiener in/out
    hubW_wie : same with hub from whitened GPA
    """
    Ln, _, k, _ = Gf.shape
    I = np.eye(k)
    if kind in ("sync_O", "hubO_wie"):
        E = np.stack([I] * Ln)
    else:
        E = np.stack([sym_pow(Gf[i, i], -0.5) for i in range(Ln)])
    Q, _ = gpa_gram(Gf, E)
    W = np.zeros((Ln, Ln, k, k))
    if kind.startswith("sync"):
        Sp = np.stack([np.linalg.inv(E[i]) for i in range(Ln)])
        for i in range(Ln):
            for j in range(Ln):
                W[i, j] = E[i] @ Q[i] @ Q[j].T @ Sp[j] if i != j else I
        return W
    # hub H = mean_i A_i E_i Q_i  (n x k).  A_i^T H = sum_i' Gf_ii' E_i' Q_i' / L ; H^T H similarly
    F = np.einsum("iab,ibc->iac", E, Q)                      # [L,k,k]
    AH = (Gf @ F[None]).sum(1) / Ln                          # [L,k,k]  A_i^T H / n
    HH = (np.transpose(F, (0, 2, 1))[:, None] @ Gf @ F[None]).sum((0, 1)) / Ln ** 2   # H^T H / n
    B = np.stack([np.linalg.solve(Gf[i, i] + rel * np.trace(Gf[i, i]) / k * I, AH[i]) for i in range(Ln)])
    HHr = HH + rel * np.trace(HH) / k * I
    Gout = np.stack([np.linalg.solve(HHr, AH[j].T) for j in range(Ln)])   # [L,k,k]: hub -> A_j
    for i in range(Ln):
        for j in range(Ln):
            W[i, j] = B[i] @ Gout[j] if i != j else I
    return W


# ----------------------------------------------------------------------------- metrics

def rho_pairs(W, Gt):
    Ln = W.shape[0]
    r = np.full((Ln, Ln), np.nan)
    for i in range(Ln):
        for j in range(Ln):
            if i == j:
                continue
            M = W[i, j]
            num = np.sum((Gt[i, i] @ M) * M) - 2 * np.sum(M * Gt[i, j]) + np.trace(Gt[j, j])
            r[i, j] = num / np.trace(Gt[j, j])
    return r


def comp_err(W, Gt):
    """c_ijm = ||T_i (W_ij W_jm - W_im)||^2 / ||T_i||^2 ; returns [L,L,L] with nan where m in (i,j) or i==j."""
    Ln = W.shape[0]
    c = np.full((Ln, Ln, Ln), np.nan)
    for i in range(Ln):
        G = Gt[i, i]
        e = np.trace(G)
        for j in range(Ln):
            if j == i:
                continue
            V = W[i, j][None] @ W[j] - W[i]              # [L(m),k,k]
            cm = np.einsum("mab,mab->m", G[None] @ V, V) / e
            for m in range(Ln):
                if m not in (i, j):
                    c[i, j, m] = cm[m]
    return c


def crosscov_stats(Gf, rmax=None):
    """Gauge-invariant cross-covariance spectra.  Under an orthogonal gauge with independent noise
    all pairwise cross-covariances C_ij are isospectral (s.v. = hub spectrum).  Returns:
      sv [P,k] sorted singular values per pair (i<j), pairs list,
      var decomposition of log sv across pairs: total, explained by additive a_i+a_j (per-language
      scale = GL/similarity gauge), residual (pair-specific)."""
    Ln, _, k, _ = Gf.shape
    pairs = [(i, j) for i in range(Ln) for j in range(i + 1, Ln)]
    sv = np.stack([np.linalg.svd(Gf[i, j], compute_uv=False) for i, j in pairs])
    rmax = rmax or k
    Y = np.log(sv[:, :rmax])
    X = np.zeros((len(pairs), Ln))
    for p, (i, j) in enumerate(pairs):
        X[p, i] = X[p, j] = 1
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ beta
    tot = Y.var(0)                      # per rank
    return dict(sv=sv, pairs=pairs, v_tot=float(tot.mean()), v_res=float(res.var(0).mean()),
                v_add=float(tot.mean() - res.var(0).mean()), a=beta)


def own_spectra(Gf):
    return np.stack([np.sort(np.linalg.eigvalsh(Gf[i, i]))[::-1] for i in range(Gf.shape[0])])


# ----------------------------------------------------------------------------- null (fit.py recipe, Gram-free)

def null_data(Xf, Xt, Zf, Zt, bases, seed=1000):
    """Consistent orthogonal-gauge surrogate exactly as fit.analyze_with_null builds it (per_lang):
    GPA hub + dof-inflated permuted in-subspace residual + real off-subspace part, lifted to d."""
    Ln, n, k = Zf.shape
    d = Xf.shape[2]
    mu = Zf.mean(1, keepdims=True)
    A = (Zf - mu).astype(np.float64)
    T = (Zt - mu).astype(np.float64)
    Gf = blockgram(A) / n
    Q, _ = gpa_gram(Gf)
    M = np.mean([A[i] @ Q[i] for i in range(Ln)], axis=0)
    Mt = np.mean([T[i] @ Q[i] for i in range(Ln)], axis=0)
    hat = [M @ Q[i].T for i in range(Ln)]
    hat_t = [Mt @ Q[i].T for i in range(Ln)]
    E = [A[i] - hat[i] for i in range(Ln)]
    Et = [T[i] - hat_t[i] for i in range(Ln)]
    rss = sum(float((e ** 2).sum()) for e in E)
    dof = Ln * n * k - n * k - (Ln - 1) * k * (k - 1) / 2 - Ln * k
    infl = float(np.sqrt(rss / max(dof, 1) / (rss / (Ln * n * k)))) if dof > 0 else 1.0
    infl_t = float(np.sqrt(Ln / (Ln - 1)))
    rs = np.random.RandomState(seed)
    Xfn = np.empty_like(Xf)
    Xtn = np.empty_like(Xt)
    for i in range(Ln):
        g, P = bases[i]
        for X, H, Ee, fac, out in ((Xf, hat, E, infl, Xfn), (Xt, hat_t, Et, infl_t, Xtn)):
            Zs = H[i] + fac * Ee[i][rs.permutation(len(Ee[i]))]
            Xc = X[i] - g
            off = Xc - (Xc @ P) @ P.T
            out[i] = (Zs @ P.T).astype(np.float32) + off + g
    return Xfn, Xtn, infl


# ----------------------------------------------------------------------------- one full evaluation

FAMS = ["proc", "sim", "wproc_0.5", "wproc_0.5_s", "wproc_1", "wproc_1_s", "ridge"]
JOINT = ["sync_O", "sync_W", "hubO_wie"]
CFAMS = ["proc", "wproc_0.5", "wproc_1", "ridge"]


def evaluate(Gf, Gt, fams=FAMS, joint=JOINT, cfams=CFAMS, keep_sv=False):
    out = {}
    extra = {}
    for fam in fams + joint:
        W = maps_family(Gf, fam) if fam in fams else joint_maps(Gf, fam)
        r = rho_pairs(W, Gt)
        out[f"rho_{fam}"] = float(np.nanmean(r))
        if fam in cfams:
            c = comp_err(W, Gt)
            out[f"c_{fam}"] = float(np.nanmean(c))
        if keep_sv and fam in ("ridge", "wproc_1", "proc"):
            svs = np.stack([np.linalg.svd(W[i, j], compute_uv=False) for i in range(len(W)) for j in range(len(W)) if i != j])
            extra[f"sv_{fam}"] = svs
            ls = np.log(svs)
            out[f"svstd_{fam}"] = float(ls.std(1).mean())
            out[f"svmed_{fam}"] = float(np.median(svs))
        del W
    cc = crosscov_stats(Gf)
    out.update(cc_vtot=cc["v_tot"], cc_vadd=cc["v_add"], cc_vres=cc["v_res"])
    k = Gf.shape[-1]
    cc32 = crosscov_stats(Gf, rmax=min(32, k))
    out.update(cc32_vtot=cc32["v_tot"], cc32_vadd=cc32["v_add"], cc32_vres=cc32["v_res"])
    # canonical correlations (in-sample) mean
    ccor = []
    for i, j in cc["pairs"]:
        K = sym_pow(Gf[i, i], -0.5) @ Gf[i, j] @ sym_pow(Gf[j, j], -0.5)
        ccor.append(np.linalg.svd(K, compute_uv=False))
    ccor = np.stack(ccor)
    out["cancorr_mean"] = float(ccor.mean())
    extra["cancorr"] = ccor
    extra["cc_sv"] = cc["sv"]
    extra["cc_a"] = cc["a"]
    extra["own"] = own_spectra(Gf)
    return out, extra


# ----------------------------------------------------------------------------- hub models of the CROSS-covariance
# The only noise-free objects (independent per-language noise) are the cross-covariances C_ij, i != j.
# Gauge model: C_ij = F_i F_j^T with F_i = A_i^T Sigma_H^{1/2} (k x r).  Group restrictions:
#   O   : F_i = Q_i Sigma^{1/2}            (orthogonal gauge; per-language metric K_i = F_i^T F_i identical)
#   sim : F_i = s_i Q_i Sigma^{1/2}        (similarity gauge; K_i = s_i^2 K)
#   GL  : F_i free (k x r)                 (linear gauge; K_i arbitrary SPD)
# Every model predicts with the noise-optimal (Wiener) map W_ij = (Sigma_i + lam)^-1 C_ij^model, so ridge's
# shrinkage advantage is granted to every group and only the GROUP of the signal maps is compared.

def wiener(Gf, Cm, rel=1e-3):
    Ln, _, k, _ = Gf.shape
    I = np.eye(k)
    W = np.zeros_like(Gf)
    for i in range(Ln):
        G = Gf[i, i]
        Ginv = np.linalg.inv(G + rel * np.trace(G) / k * I)
        for j in range(Ln):
            W[i, j] = Ginv @ Cm[i, j] if i != j else I
    return W


def hub_O(Gf, scale=False):
    Ln, _, k, _ = Gf.shape
    Q, _ = gpa_gram(Gf)
    X = np.transpose(Q, (0, 2, 1))[:, None] @ Gf @ Q[None]       # Q_i^T C_ij Q_j   (hub frame)
    off = ~np.eye(Ln, dtype=bool)
    s = np.ones(Ln)
    if scale:
        t = np.array([[np.trace(X[i, j]) for j in range(Ln)] for i in range(Ln)])
        P = [(i, j) for i in range(Ln) for j in range(Ln) if i != j]
        D = np.zeros((len(P), Ln + 1))
        y = np.zeros(len(P))
        for p, (i, j) in enumerate(P):
            D[p, i] += 1
            D[p, j] += 1
            D[p, -1] = 1
            y[p] = np.log(max(t[i, j], 1e-12))
        b = np.linalg.lstsq(D, y, rcond=None)[0]
        s = np.exp(b[:Ln] - b[:Ln].mean())
    S = np.mean([X[i, j] / (s[i] * s[j]) for i in range(Ln) for j in range(Ln) if off[i, j]], axis=0)
    S = 0.5 * (S + S.T)
    Cm = np.zeros_like(Gf)
    for i in range(Ln):
        for j in range(Ln):
            if i != j:
                Cm[i, j] = s[i] * s[j] * Q[i] @ S @ Q[j].T
    return Cm, s


def block_fa(Gf, r, iters=40):
    """Inter-battery / multi-block factor analysis: fit the OFF-diagonal blocks C_ij ~= F_i F_j^T by
    iterating the diagonal 'communality' blocks.  Returns F [L,k,r]."""
    Ln, _, k, _ = Gf.shape
    G = Gf.transpose(0, 2, 1, 3).reshape(Ln * k, Ln * k).copy()
    for i in range(Ln):
        G[i * k:(i + 1) * k, i * k:(i + 1) * k] = 0.5 * Gf[i, i]
    for _ in range(iters):
        w, V = np.linalg.eigh(0.5 * (G + G.T))
        w, V = w[-r:], V[:, -r:]
        F = V * np.sqrt(np.clip(w, 0, None))
        for i in range(Ln):
            Fi = F[i * k:(i + 1) * k]
            G[i * k:(i + 1) * k, i * k:(i + 1) * k] = Fi @ Fi.T
    return np.stack([F[i * k:(i + 1) * k] for i in range(Ln)])


def hub_GL(Gf, r):
    F = block_fa(Gf, r)
    Cm = F[:, None] @ np.transpose(F, (0, 2, 1))[None]
    return Cm, F


def metric_stats(F, rm=None):
    """Per-language metrics K_i = F_i^T F_i on the shared r-dim hub, normalised so that mean_i K_i = I.
    Invariant under z_i -> z_i O_i and under any common reparametrisation of the hub.
    Returns: global log-scales g_i, anisotropic log-eigenvalue spectra, commutator size, participation ratio."""
    K = np.transpose(F, (0, 2, 1)) @ F
    if rm is not None:                      # restrict to the rm strongest shared hub directions
        w, U = np.linalg.eigh(K.mean(0))
        U = U[:, -rm:]
        K = U.T[None] @ K @ U[None]
    N = sym_pow(K.mean(0), -0.5)
    Kt = N[None] @ K @ N[None]
    logK = np.stack([_logm(Kt[i]) for i in range(len(Kt))])
    r = Kt.shape[-1]
    g = np.array([np.trace(l) / r for l in logK])              # global log-scale (of the metric = 2 log s_i)
    aniso = [l - np.trace(l) / r * np.eye(r) for l in logK]
    ev = np.stack([np.sort(np.linalg.eigvalsh(a)) for a in aniso])
    pr = np.array([(e ** 2).sum() ** 2 / max((e ** 4).sum(), 1e-30) for e in ev])
    comm = []
    for i in range(len(aniso)):
        for j in range(i + 1, len(aniso)):
            Cc = aniso[i] @ aniso[j] - aniso[j] @ aniso[i]
            comm.append(np.linalg.norm(Cc))
    return dict(g=g, ev=ev, var_global=float(g.var()), var_aniso=float((ev ** 2).mean()),
                pr=float(pr.mean()), comm=float(np.mean(comm)), top_abs=float(np.abs(ev).max(1).mean()))


def _logm(S):
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    return (V * np.log(np.clip(w, 1e-12, None))) @ V.T


def hub_ladder(Gf, Gt, rm_frac=0.25):
    """rho of hub-mediated Wiener predictors under O, sim, GL(r) gauges + pairwise ridge; model-implied c_proc."""
    Ln, _, k, _ = Gf.shape
    out = {}
    CmO, _ = hub_O(Gf)
    out["rho_hubO"] = float(np.nanmean(rho_pairs(wiener(Gf, CmO), Gt)))
    CmS, s = hub_O(Gf, scale=True)
    out["rho_hubSim"] = float(np.nanmean(rho_pairs(wiener(Gf, CmS), Gt)))
    r = k
    CmG, F = hub_GL(Gf, r)
    out["rho_hubGL"] = float(np.nanmean(rho_pairs(wiener(Gf, CmG), Gt)))
    # procrustes maps implied by the fitted GL-hub cross-covariances: do they reproduce c_proc?
    Rm = np.zeros_like(Gf)
    for i in range(Ln):
        for j in range(Ln):
            Rm[i, j] = polar(CmG[i, j]) if i != j else np.eye(k)
    out["c_proc_GLimplied"] = float(np.nanmean(comp_err(Rm, Gt)))
    ms = metric_stats(F, rm=max(4, int(rm_frac * k)))
    out.update(partial_aniso_rho(Gf, Gt, F))
    PR = pair_residual(Gf, CmG)
    out["pairres_GL"] = float(np.nanmean(PR))
    Rproc = maps_family(Gf, "proc")
    out["c_proc_top"], out["c_proc_rest"] = c_localised(Rproc, Gt)
    out["c_procGL_top"], out["c_procGL_rest"] = c_localised(Rm, Gt)
    Wr = maps_family(Gf, "ridge")
    out["c_ridge_top"], out["c_ridge_rest"] = c_localised(Wr, Gt)
    out.update({f"K_{a}": ms[a] for a in ("var_global", "var_aniso", "pr", "comm", "top_abs")})
    out["r_hub"] = r
    return out, dict(scale_sim=s, K_g=ms["g"], K_ev=ms["ev"], pairres=PR)


# ----------------------------------------------------------------------------- extensions: partial-anisotropy ladder,
# pair-specific residual, localisation of the procrustes composition error

def partial_aniso_rho(Gf, Gt, F, ps=(0, 1, 2, 4, 8, 16, 32)):
    """Similarity gauge + the p largest anisotropic stretch directions of each language's metric.
    p=0 -> similarity gauge (with GL-fitted orientation), p=r -> full GL hub model."""
    Ln, k, r = F.shape
    K = np.transpose(F, (0, 2, 1)) @ F
    Kbar = K.mean(0)
    N = sym_pow(Kbar, -0.5)
    Ninv2 = Kbar                                            # N^-2
    out = {}
    U, Kt = [], []
    for i in range(Ln):
        Ft = F[i] @ N
        u, s, vt = np.linalg.svd(Ft, full_matrices=False)
        U.append(u @ vt)                                   # polar factor, Ft = U_i Kt_i^{1/2}
        Kt.append(Ft.T @ Ft)
    for p in ps:
        if p > r:
            continue
        Kh = []
        for i in range(Ln):
            w, V = np.linalg.eigh(0.5 * (Kt[i] + Kt[i].T))
            lw = np.log(np.clip(w, 1e-12, None))
            g = lw.mean()
            a = lw - g
            keep = np.argsort(-np.abs(a))[:p]
            lw2 = np.full_like(lw, g)
            lw2[keep] = lw[keep]
            Kh.append((V * np.exp(0.5 * lw2)) @ V.T)        # (K^(p))^{1/2}
        Cm = np.zeros_like(Gf)
        for i in range(Ln):
            for j in range(Ln):
                if i != j:
                    Cm[i, j] = U[i] @ Kh[i] @ Ninv2 @ Kh[j] @ U[j].T
        out[f"rho_hubGL_p{p}"] = float(np.nanmean(rho_pairs(wiener(Gf, Cm), Gt)))
    return out


def pair_residual(Gf, Cm):
    Ln = Gf.shape[0]
    R = np.full((Ln, Ln), np.nan)
    for i in range(Ln):
        for j in range(Ln):
            if i != j:
                R[i, j] = np.sum((Gf[i, j] - Cm[i, j]) ** 2) / np.sum(Gf[i, j] ** 2)
    return R


def c_localised(W, Gt, frac=0.25):
    """split c into the part carried by language i's top-(frac k) PCA coordinates and the rest."""
    Ln, _, k, _ = W.shape
    r = max(1, int(frac * k))
    top, rest = [], []
    for i in range(Ln):
        G = Gt[i, i]
        e = np.trace(G)
        Gtop = np.zeros_like(G)
        Gtop[:r, :r] = G[:r, :r]
        Grest = np.zeros_like(G)
        Grest[r:, r:] = G[r:, r:]
        for j in range(Ln):
            if j == i:
                continue
            V = W[i, j][None] @ W[j] - W[i]
            ct = np.einsum("mab,mab->m", Gtop[None] @ V, V) / e
            cr = np.einsum("mab,mab->m", Grest[None] @ V, V) / e
            msk = np.array([m not in (i, j) for m in range(Ln)])
            top.append(ct[msk])
            rest.append(cr[msk])
    return float(np.mean(np.concatenate(top))), float(np.mean(np.concatenate(rest)))
