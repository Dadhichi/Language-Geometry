"""Investigator D: the language coordinate in AMBIENT residual space.

Two-way decomposition of X[i,s] (language i, sentence s, parallel):
    X[i,s] = m + a_i + b_s + e_{i,s}
a_i = mu_i - m  (language main effect / centroid offset), b_s = c_s - m (shared content),
e = interaction (content-dependent part of the language offset + idiosyncratic translation noise).
Per-sentence language offset o_{i,s} = X[i,s] - c_s = a_i + e_{i,s}.
All quantities live in the ambient d-space shared by languages and layers (gauge-free).
Memory: e is never materialised for all languages at once (rows are recomputed per language).
"""
import os, time
import numpy as np
from scipy.stats import spearmanr

DATA = r"C:\Users\ASUS\Documents\lang-geom"
OUT = os.path.dirname(os.path.abspath(__file__))
LANGS = ["eng_Latn", "deu_Latn", "fra_Latn", "spa_Latn", "rus_Cyrl", "hin_Deva",
         "arb_Arab", "zho_Hans", "jpn_Jpan", "tur_Latn", "vie_Latn", "ind_Latn"]
LAYERS = list(range(0, 29, 2))


def load(split, layer):
    X = np.load(os.path.join(DATA, f"mean_{split}_L{layer}.f16.npy"), mmap_mode="r")
    return np.asarray(X, dtype=np.float32)


def sq(Y):
    return float(np.square(Y).sum(dtype=np.float64))


def pr(w):
    w = np.clip(np.asarray(w, dtype=np.float64), 0, None)
    return float(w.sum() ** 2 / (w ** 2).sum())


def top_basis_rows(Y, k):
    """Y [N,d] (already centred as desired) -> (eigvals desc [N], top-k right singular vectors [d,k])"""
    Y64 = Y.astype(np.float64)
    w, U = np.linalg.eigh(Y64 @ Y64.T)
    w, U = w[::-1], U[:, ::-1]
    k = min(k, int((w > 1e-9 * w[0]).sum()))
    P = (Y64.T @ U[:, :k]) / np.sqrt(w[:k])
    return w, P.astype(np.float32)


def orth(A, tol=1e-6):
    """orthonormal basis [d,r] of the row span of A [r',d] (drops null directions)"""
    u, s, vt = np.linalg.svd(A.astype(np.float64), full_matrices=False)
    r = int((s > tol * s[0]).sum())
    return vt[:r].T.astype(np.float32), s


def ridge_fit(U, Y, rel):
    G = U.T.astype(np.float64) @ U
    lam = rel * np.trace(G) / G.shape[0]
    return np.linalg.solve(G + lam * np.eye(G.shape[0]), U.T.astype(np.float64) @ Y).astype(np.float32)


def procrustes(A, B):
    u, _, vt = np.linalg.svd(A.T.astype(np.float64) @ B, full_matrices=False)
    return (u @ vt).astype(np.float32)


class Anova:
    def __init__(self, X):
        self.X = X
        self.L, self.n, self.d = X.shape
        self.m = X.mean((0, 1), dtype=np.float64)
        self.mu = X.mean(1, dtype=np.float64)
        c = X.mean(0, dtype=np.float64)
        self.a, self.b = self.mu - self.m, c - self.m
        self._mu32 = self.mu.astype(np.float32)
        self._b32 = self.b.astype(np.float32)

    def e(self, i):
        r = self.X[i] - self._mu32[i]
        r -= self._b32
        return r

    def parts(self):
        Vl = float((self.a ** 2).sum() / self.L)
        Vs = float((self.b ** 2).sum() / self.n)
        Vi = sum(sq(self.e(i)) for i in range(self.L)) / (self.L * self.n)
        return Vl, Vs, Vi


def layer_stats(Xd, Xt, W, weig, ntok_d=None, rng_seed=0, do_p1=True, verbose=False, ks=(64, 256)):
    """Everything per layer.  Xd [L,n,d] fit split, Xt [L,m,d] test split (BOTH ARE MODIFIED IN PLACE),
    W [d,512] unembedding basis.  Returns (scalars dict, per-language rows list, arrays dict)."""
    t0 = time.time()
    L, n, d = Xd.shape
    nt = Xt.shape[1]
    S, rows, arrs = {}, [dict(lang=LANGS[i]) for i in range(L)], {}
    Ad, At = Anova(Xd), Anova(Xt)
    a_d, a_t, b_d, b_t = Ad.a, At.a, Ad.b, At.b
    arrs.update(a_dev=a_d.astype(np.float32), a_test=a_t.astype(np.float32),
                m_dev=Ad.m.astype(np.float32), m_test=At.m.astype(np.float32))

    # ---------------- 1. variance decomposition (per-cell mean squared norms)
    for tag, A_ in (("dev", Ad), ("test", At)):
        Vl, Vs, Vi = A_.parts()
        T = Vl + Vs + Vi
        S[f"V_lang_{tag}"], S[f"V_sent_{tag}"], S[f"V_int_{tag}"] = Vl, Vs, Vi
        S[f"f_lang_{tag}"], S[f"f_sent_{tag}"], S[f"f_int_{tag}"] = Vl / T, Vs / T, Vi / T
        S[f"inv_frac_{tag}"] = Vl / (Vl + Vi)          # sentence-invariant share of the per-sentence offset
        S[f"lang_over_content_{tag}"] = Vl / Vs
        S[f"norm_m_{tag}"] = float(np.sqrt((A_.m ** 2).sum()))
    S["V_lang_x"] = float((a_d * a_t).sum() / L)     # unbiased (cross-split) language variance
    S["a_split_cos"] = float((a_d * a_t).sum() / np.sqrt((a_d ** 2).sum() * (a_t ** 2).sum()))
    m32 = Ad.m.astype(np.float32)
    tot = sum(sq(Xd[i] - m32) for i in range(L)) / (L * n)
    S["anova_check"] = (S["V_lang_dev"] + S["V_sent_dev"] + S["V_int_dev"]) / tot
    # sentence-shuffle control: permute sentences independently per language -> shared content vanishes
    rs = np.random.RandomState(rng_seed)
    Xs = np.stack([Xd[i][rs.permutation(n)] for i in range(L)])
    Vl_s, Vs_s, Vi_s = Anova(Xs).parts()
    del Xs
    S["f_sent_shuf"] = Vs_s / (Vl_s + Vs_s + Vi_s)
    S["inv_frac_shuf"] = Vl_s / (Vl_s + Vi_s)
    for i in range(L):
        rows[i].update(a2_dev=float((a_d[i] ** 2).sum()), a2_x=float((a_d[i] * a_t[i]).sum()),
                       a2_over_content=float((a_d[i] ** 2).sum() / S["V_sent_dev"]),
                       e2_over_content=sq(Ad.e(i)) / n / S["V_sent_dev"])

    # ---------------- 2. spectra / effective dimension
    wa = np.clip(np.linalg.eigvalsh(a_d @ a_d.T)[::-1], 0, None)
    wx = np.linalg.eigvalsh(0.5 * (a_d @ a_t.T + a_t @ a_d.T))[::-1]
    S["PR_a"], S["PR_a_x"] = pr(wa), pr(np.clip(wx, 0, None))
    for r in (1, 2, 3, 5):
        S[f"a_top{r}"] = float(wa[:r].sum() / wa.sum())
        S[f"a_top{r}_x"] = float(wx[:r].sum() / wx.sum())
    arrs["a_eig"], arrs["a_eig_x"] = wa, wx
    Ce = np.zeros((d, d))
    for i in range(L):
        E = Ad.e(i)
        Ce += (E.T @ E).astype(np.float64)
    we = np.clip(np.linalg.eigvalsh(Ce)[::-1], 0, None)
    Ce += n * (a_d.T @ a_d)                               # -> second moment of per-sentence offsets o = a + e
    wo = np.clip(np.linalg.eigvalsh(Ce)[::-1], 0, None)
    del Ce
    S["PR_e"], S["PR_o"] = pr(we), pr(wo)
    for r in (11, 64, 256):
        S[f"e_top{r}"] = float(we[:r].sum() / we.sum())
        S[f"o_top{r}"] = float(wo[:r].sum() / wo.sum())
    arrs["e_eig"], arrs["o_eig"] = we.astype(np.float32), wo.astype(np.float32)
    wb, Pb = top_basis_rows(b_d, 256)
    wb = np.clip(wb, 0, None)
    S["PR_b"] = pr(wb)
    for r in (11, 64, 256):
        S[f"b_top{r}"] = float(wb[:r].sum() / wb.sum())
    arrs["b_eig"] = wb[:256].astype(np.float32)
    arrs["Pb32"] = Pb[:, :32]

    # ---------------- 3. language subspace vs content subspace (bases from dev, scored on test)
    Ua, _ = orth(a_d)                                     # [d, 11]
    r = Ua.shape[1]
    S["rank_a"] = r
    et2 = sum(sq(At.e(i)) for i in range(L))
    S["e_in_langspan"] = sum(sq(At.e(i) @ Ua) for i in range(L)) / et2      # vs r/d isotropic
    S["iso_r_over_d"] = r / d
    bt2 = float((b_t ** 2).sum())
    a_t32, b_t32 = a_t.astype(np.float32), b_t.astype(np.float32)
    for k in (11, 64, 256):
        P = Pb[:, :k]
        S[f"a_in_content{k}"] = sq(a_t32 @ P) / sq(a_t32)      # language energy in top-k content PCs
        S[f"b_in_content{k}"] = sq(b_t32 @ P) / bt2            # content's own held-out capture
        S[f"e_in_content{k}"] = sum(sq(At.e(i) @ P) for i in range(L)) / et2
        S[f"iso_{k}"] = k / d
    cosang = np.linalg.svd(Ua.T @ Pb[:, :r], compute_uv=False)
    S["cos2_lang_content11_mean"] = float((cosang ** 2).mean())
    S["cos_lang_content11_max"] = float(cosang.max())
    arrs["cos_lang_content"] = cosang
    trCb = float((b_d ** 2).sum()) / n
    q = [float(((b_d @ a_t[i]) ** 2).sum() / n / (a_t[i] ** 2).sum() / (trCb / d)) for i in range(L)]
    S["rayleigh_content_along_lang"] = float(np.mean(q))
    # own-direction share of the interaction: does the offset only change its LENGTH across sentences?
    tcoef = np.zeros((L, n), np.float32)
    lognt = np.log(ntok_d) if ntok_d is not None else None
    for i in range(L):
        na = float(np.sqrt((a_d[i] ** 2).sum()))
        u = (a_d[i] / na).astype(np.float32)
        E = Ad.e(i)
        proj = E @ u
        tcoef[i] = 1 + proj / na
        rows[i]["t_cv"] = float(tcoef[i].std() / tcoef[i].mean())
        rows[i]["e_own_dir_share"] = float((proj.astype(np.float64) ** 2).sum() / sq(E))
        if lognt is not None:
            rows[i]["t_vs_logntok"] = float(spearmanr(tcoef[i], lognt[i])[0])
            rows[i]["t_vs_relfert"] = float(spearmanr(tcoef[i], lognt[i] - lognt.mean(0))[0])
            onorm = np.sqrt(((E + a_d[i].astype(np.float32)) ** 2).sum(1))
            rows[i]["onorm_vs_logntok"] = float(spearmanr(onorm, lognt[i])[0])
            rows[i]["enorm_vs_logntok"] = float(spearmanr(np.sqrt((E ** 2).sum(1)), lognt[i])[0])
    arrs["tcoef"] = tcoef
    S["e_own_dir_share_mean"] = float(np.mean([r_["e_own_dir_share"] for r_ in rows]))
    S["t_cv_mean"] = float(np.mean([r_["t_cv"] for r_ in rows]))

    # ---------------- 4. unembedding alignment (energy fraction in span(W) = top-512 W_U input directions)
    sw = np.sqrt(weig).astype(np.float32)

    def wfrac(Y, cols=slice(None)):
        return sq(Y @ W[:, cols]) / sq(Y)

    def wgain(Y, c0=0):
        return sq((Y @ W[:, c0:]) * sw[c0:]) / sq(Y)

    for nm, Y in (("a", a_t32), ("b", b_t32), ("m", At.m[None].astype(np.float32))):
        S[f"W_{nm}"] = wfrac(Y)
        S[f"W64_{nm}"] = wfrac(Y, slice(0, 64))
        S[f"W1_{nm}"] = wfrac(Y, slice(0, 1))
        S[f"Wgain_{nm}"] = wgain(Y)
        S[f"Wgain_skip0_{nm}"] = wgain(Y, 1)
    S["W_e"] = sum(sq(At.e(i) @ W) for i in range(L)) / et2
    S["W64_e"] = sum(sq(At.e(i) @ W[:, :64]) for i in range(L)) / et2
    S["Wgain_skip0_e"] = sum(sq((At.e(i) @ W[:, 1:]) * sw[1:]) for i in range(L)) / et2
    S["W_iso"], S["W64_iso"] = W.shape[1] / d, 64 / d
    for i in range(L):
        rows[i]["W_a"] = wfrac(a_t32[i][None])
    if verbose:
        print(f"   stats part1 {time.time() - t0:.0f}s", flush=True)

    # ---------------- 5. cross-language prediction on the TEST split (parameters from dev)
    mu_d = Ad.mu.astype(np.float32)
    Yd = Xd; Yd -= mu_d[:, None, :]                        # in place: centred by fit means (as fit.py)
    Yt = Xt; Yt -= mu_d[:, None, :]
    del Ad, At

    def gram(Y):
        K = np.zeros((L, L))
        for i in range(L):
            for j in range(i, L):
                K[i, j] = K[j, i] = float((Y[i] * Y[j]).sum(dtype=np.float64))
        return K
    Kt, Kd = gram(Yt), gram(Yd)
    sy = Yt.sum(1, dtype=np.float64)
    dm = mu_d.astype(np.float64)
    rid, rsh, rsc = [], [], []
    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            den = Kt[j, j]
            num_sh = Kt[i, i] + Kt[j, j] - 2 * Kt[i, j]
            dmu = dm[i] - dm[j]
            num_id = num_sh + nt * (dmu ** 2).sum() + 2 * ((sy[i] - sy[j]) * dmu).sum()
            al = Kd[i, j] / Kd[i, i]
            num_sc = Kt[j, j] - 2 * al * Kt[i, j] + al ** 2 * Kt[i, i]
            rid.append(num_id / den); rsh.append(num_sh / den); rsc.append(num_sc / den)
    S["rho_identity_amb"], S["rho_shift_amb"], S["rho_scaledshift_amb"] = map(float, map(np.mean, (rid, rsh, rsc)))
    S["frac_err_removed_by_shift"] = 1 - S["rho_shift_amb"] / S["rho_identity_amb"]
    arrs["Kt"], arrs["Kd"] = Kt, Kd

    if do_p1:   # cosine retrieval in the target's centred frame: pred - mu_j vs y_j,t
        Yn = Yt / np.linalg.norm(Yt, axis=2, keepdims=True)
        Ynf = Yn.reshape(-1, d)
        p1_id, p1_sh = [], []
        ar = np.arange(nt)
        for i in range(L):
            Sc = (Yt[i] @ Ynf.T).reshape(nt, L, nt)
            for j in range(L):
                if j == i:
                    continue
                p1_sh.append(float((Sc[:, j, :].argmax(1) == ar).mean()))
                off = (dm[i] - dm[j]).astype(np.float32) @ Yn[j].T
                p1_id.append(float(((Sc[:, j, :] + off[None]).argmax(1) == ar).mean()))
            del Sc
        del Yn, Ynf
        S["p1_identity_amb"], S["p1_shift_amb"] = float(np.mean(p1_id)), float(np.mean(p1_sh))

    # per-language PCA (fit.py pipeline): shift rung done right (ambient identity), procrustes, ridge
    for k in ks:
        B = [top_basis_rows(Yd[i], k)[1] for i in range(L)]
        Zd = [Yd[i] @ B[i] for i in range(L)]
        Zt = [Yt[i] @ B[i] for i in range(L)]
        rs_full, rs_tr, rpol, rp, rr, cap, agree = [], [], [], [], [], [], []
        P1 = {nm: [] for nm in ("shift_trunc", "polar_id", "proc", "ridge")}
        ar = np.arange(nt)
        for j in range(L):
            den = sq(Zt[j])
            cap.append(den / sq(Yt[j]))
            tn = Zt[j] / np.linalg.norm(Zt[j], axis=1, keepdims=True)
            for i in range(L):
                if i == j:
                    continue
                rs_full.append(sq(Yt[i] @ B[j] - Zt[j]) / den)          # shift, no source truncation
                Mid = B[i].T @ B[j]                                       # ambient identity in per-lang coords
                Rid = procrustes(np.eye(k, dtype=np.float32), Mid)        # nearest orthogonal map to it
                R = procrustes(Zd[i], Zd[j])
                Wr = ridge_fit(Zd[i], Zd[j], 1e-3)
                agree.append(float(np.trace(R.T @ Rid)) / k)              # 1 iff fitted rotation == identity
                for nm, M, lst in (("shift_trunc", Mid, rs_tr), ("polar_id", Rid, rpol), ("proc", R, rp), ("ridge", Wr, rr)):
                    pred = Zt[i] @ M
                    lst.append(sq(pred - Zt[j]) / den)
                    P1[nm].append(float(((pred @ tn.T).argmax(1) == ar).mean()))
        S[f"rho_polar_id_k{k}"] = float(np.mean(rpol))
        S[f"proc_vs_identity_k{k}"] = float(np.mean(agree))
        for nm, v in P1.items():
            S[f"p1_{nm}_k{k}"] = float(np.mean(v))
        S[f"rho_shift_k{k}"], S[f"rho_shift_trunc_k{k}"] = float(np.mean(rs_full)), float(np.mean(rs_tr))
        S[f"rho_proc_k{k}"], S[f"rho_ridge_k{k}"] = float(np.mean(rp)), float(np.mean(rr))
        S[f"captured_k{k}"] = float(np.mean(cap))
        del B, Zd, Zt
    if verbose:
        print(f"   prediction {time.time() - t0:.0f}s", flush=True)

    # ---------------- 6. content-dependence of the offset (LOO content hub, ambient output)
    # target y_i,s (centred by dev mean) vs LOO content z_i,s = mean_{i' != i} y_i',s ; residual r = y - z
    kk = 256
    Pc = Pb[:, :kk]
    sumd, sumt = Yd.sum(0), Yt.sum(0)
    half = np.random.RandomState(rng_seed + 1).permutation(n)
    h1, h2 = half[: n // 2], half[n // 2:]
    rels = (1e-3, 1e-2, 1e-1, 1.0)
    cv = np.zeros(len(rels))
    for i in range(L):
        zd = (sumd - Yd[i]) / (L - 1)
        ud, rd = zd @ Pc, Yd[i] - zd
        for q_, rel in enumerate(rels):
            cv[q_] += sq(rd[h2] - ud[h2] @ ridge_fit(ud[h1], rd[h1], rel))
    rel_best = rels[int(np.argmin(cv))]
    S["ridge_rel"] = rel_best
    GU, UR, Wi = np.zeros((kk, kk)), np.zeros((kk, d)), []
    for i in range(L):
        zd = (sumd - Yd[i]) / (L - 1)
        ud, rd = zd @ Pc, Yd[i] - zd
        GU += ud.T.astype(np.float64) @ ud
        UR += ud.T.astype(np.float64) @ rd
        Wi.append(ridge_fit(ud, rd, rel_best))
    Wsh = np.linalg.solve(GU + rel_best * np.trace(GU) / kk * np.eye(kk), UR).astype(np.float32)
    res = dict(shift=0.0, alpha=0.0, ridge_shared=0.0, ridge_perlang=0.0, den=0.0)
    for i in range(L):
        zt, yt = (sumt - Yt[i]) / (L - 1), Yt[i]
        zd = (sumd - Yd[i]) / (L - 1)
        al = float((zd * Yd[i]).sum(dtype=np.float64) / (zd * zd).sum(dtype=np.float64))
        den = sq(yt)
        ut = zt @ Pc
        r_sh, r_al = sq(yt - zt), sq(yt - al * zt)
        r_rs, r_rp = sq(yt - zt - ut @ Wsh), sq(yt - zt - ut @ Wi[i])
        for nm, v in (("shift", r_sh), ("alpha", r_al), ("ridge_shared", r_rs), ("ridge_perlang", r_rp), ("den", den)):
            res[nm] += v
        rows[i].update(hub_resid_shift=r_sh / den, hub_resid_perlang=r_rp / den, hub_alpha=al)
    for nm in ("shift", "alpha", "ridge_shared", "ridge_perlang"):
        S[f"hub_resid_{nm}"] = res[nm] / res["den"]
    if verbose:
        print(f"   done {time.time() - t0:.0f}s", flush=True)
    S["seconds"] = time.time() - t0
    return S, rows, arrs
