"""Investigator C: languages as positions (RoPE analogue) in AMBIENT coordinates.

    python run_layer.py --layer 14 [--quick]

For one layer: the real data and four synthetic worlds built from the real layer (same hub, real mean
offsets, permuted real residuals as noise, calibrated to the real noise level and rotation effect):
    identity   : X_i = mu_i + H + noise                        (pure shift)
    abelian    : X_i = mu_i + H exp(theta_i M) + noise         (one-parameter group, 4 planes, 1-D positions)
    nonabelian : X_i = mu_i + H Q_i + noise, Q_i independent   (consistent non-abelian gauge, 16 planes)
    scaling    : X_i = mu_i + H exp(Y_i) + noise, Y_i symmetric (GL gauge; Q3 control)
All worlds go through the identical pipeline: pooled PCA U from the world's own dev data, maps fit on dev,
scored on devtest.  Writes out/layer{L}.json.
"""
import argparse
import json
import sys
import time

import numpy as np
import scipy.linalg as sl

from common import (DIFF_FAMILY, DIFF_SCRIPT, LANGS, NL, OPAIRS, OUT, PAIRS, TRIPLES, angle_stats, f32,
                    gpa, load16, log_orth, mantel, polar, pooled_basis, procrustes, project, rho, ridge)

MMAX = 512
NTOK = None
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# ============================================================================ Q1: shift rung, full d

def q1_ladder(Xf16, Xt16, mu, U, fr, dual=True, wu=None):
    """Full-d ladder, held-out: identity, shift, shift+Procrustes in U (identity off U), shift+ridge in U
    (identity / zero off U), full-d kernel ridge toward I / toward 0.  Low memory: languages converted lazily."""
    Lg = len(Xf16)
    A = lambda i: f32(Xf16[i]) - mu[i]
    T = lambda i: f32(Xt16[i]) - mu[i]
    n, nt, d = Xf16.shape[1], Xt16.shape[1], Xf16.shape[2]
    ET = np.array([(T(i).astype(np.float64) ** 2).sum() for i in range(Lg)])
    trS = np.array([(A(i).astype(np.float64) ** 2).sum() for i in range(Lg)]) / n
    out = {}
    sh = np.zeros((Lg, Lg))
    idr = np.zeros((Lg, Lg))
    for i, j in PAIRS:
        D = (T(i) - T(j)).astype(np.float64)
        s2 = (D ** 2).sum()
        sh[i, j] = sh[j, i] = s2
        dm = (mu[i] - mu[j]).astype(np.float64)
        cr = 2 * D.sum(0) @ dm
        idr[i, j] = s2 + cr + nt * dm @ dm
        idr[j, i] = s2 + cr + nt * dm @ dm
    off = np.array([[float(((mu[i] - mu[j]).astype(np.float64) ** 2).sum()) for j in range(Lg)] for i in range(Lg)])
    rr = lambda M: float(np.mean([M[i, j] / ET[j] for i, j in OPAIRS]))
    out["rho_identity"] = rr(idr)
    out["rho_shift"] = rr(sh)
    out["offset2_over_trS"] = float(np.mean([off[i, j] / trS[j] for i, j in OPAIRS]))
    mbar = mu.mean(0)
    out["offset_from_centroid2_over_trS"] = float(np.mean([((mu[i] - mbar) ** 2).sum() / trS[i] for i in range(Lg)]))
    out["global_mean2_over_trS"] = float(np.mean([(mbar ** 2).sum() / trS[i] for i in range(Lg)]))
    out["trS"] = trS.tolist()
    out["rho_identity_pair"] = [[idr[i, j] / ET[j] if i != j else 0 for j in range(Lg)] for i in range(Lg)]
    out["rho_shift_pair"] = [[sh[i, j] / ET[j] if i != j else 0 for j in range(Lg)] for i in range(Lg)]
    dms = np.stack([mu[i] - mu[j] for i, j in PAIRS]).astype(np.float64)
    for m in (64, 128, 256, 512):
        Um = U[:, :m].astype(np.float64)
        out[f"offset_in_U{m}"] = float(((dms @ Um) ** 2).sum() / (dms ** 2).sum())
        out[f"content_in_U{m}"] = float(fr[:m].sum())
    if wu is not None:
        wu64 = wu.astype(np.float64)
        out["offset_in_WU512"] = float(((dms @ wu64) ** 2).sum() / (dms ** 2).sum())
        out["content_in_WU512"] = float(sum(((A(i).astype(np.float64) @ wu64) ** 2).sum() for i in range(Lg))
                                        / sum(trS * n))
        out["random512_expect"] = 512 / d
    for m in (64, 128, 256, 512):
        Um = U[:, :m]
        Z = np.stack([(A(i) @ Um).astype(np.float64) for i in range(Lg)])
        Zt = np.stack([(T(i) @ Um).astype(np.float64) for i in range(Lg)])
        pr, rg, rg0, pu, sh_u = [], [], [], [], []
        Rs = {}
        for i, j in PAIRS:
            Rs[(i, j)] = procrustes(Z[i], Z[j])
            Rs[(j, i)] = Rs[(i, j)].T
        for i, j in OPAIRS:
            R = Rs[(i, j)]
            W = ridge(Z[i], Z[j])
            base = sh[i, j] - ((Zt[i] - Zt[j]) ** 2).sum()
            pr.append((base + ((Zt[i] @ R - Zt[j]) ** 2).sum()) / ET[j])
            rg.append((base + ((Zt[i] @ W - Zt[j]) ** 2).sum()) / ET[j])
            rg0.append((ET[j] - (Zt[j] ** 2).sum() + ((Zt[i] @ W - Zt[j]) ** 2).sum()) / ET[j])
            pu.append(((Zt[i] @ R - Zt[j]) ** 2).sum() / (Zt[j] ** 2).sum())
            sh_u.append(((Zt[i] - Zt[j]) ** 2).sum() / (Zt[j] ** 2).sum())
        out[f"rho_procU{m}_I"] = float(np.mean(pr))
        out[f"rho_ridgeU{m}_I"] = float(np.mean(rg))
        out[f"rho_ridgeU{m}_0"] = float(np.mean(rg0))
        out[f"inU{m}_rho_shift"] = float(np.mean(sh_u))
        out[f"inU{m}_rho_proc"] = float(np.mean(pu))
        out[f"energy_inU{m}"] = float(np.mean([(Zt[j] ** 2).sum() / ET[j] for j in range(Lg)]))
        del Z, Zt, Rs
    if dual:
        rs = np.random.RandomState(0)
        perm = rs.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        grid = [1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
        errI, err0 = np.zeros(len(grid)), np.zeros(len(grid))
        sel = [OPAIRS[k] for k in rs.choice(len(OPAIRS), 12, replace=False)]
        for i, j in sel:
            Ai, Aj = A(i).astype(np.float64), A(j).astype(np.float64)
            Aa, Ab = Ai[a], Ai[b]
            w, V = np.linalg.eigh(Aa @ Aa.T)
            Kba = Ab @ Aa.T @ V
            VdA, VAj = V.T @ (Aj[a] - Aa), V.T @ Aj[a]
            for g, rel in enumerate(grid):
                B = Kba / (w + rel * w.mean())
                errI[g] += ((Ab + B @ VdA - Aj[b]) ** 2).sum()
                err0[g] += ((B @ VAj - Aj[b]) ** 2).sum()
            del Ai, Aj
        lamI, lam0 = grid[int(np.argmin(errI))], grid[int(np.argmin(err0))]
        out["dual_lam_I"], out["dual_lam_0"] = lamI, lam0
        rI, r0 = [], []
        for i in range(Lg):
            Ai, Ti = A(i), T(i)
            Ai64 = Ai.astype(np.float64)
            w, V = np.linalg.eigh(Ai64 @ Ai64.T)
            Kt = (Ti.astype(np.float64) @ Ai64.T) @ V
            del Ai64
            BI = ((Kt / (w + lamI * w.mean())) @ V.T).astype(np.float32)
            B0 = ((Kt / (w + lam0 * w.mean())) @ V.T).astype(np.float32)
            BAi = BI @ Ai
            for j in range(Lg):
                if j == i:
                    continue
                Aj, Tj = A(j), T(j)
                rI.append(float((((Ti - Tj) + BI @ Aj - BAi).astype(np.float64) ** 2).sum() / ET[j]))
                r0.append(float(((B0 @ Aj - Tj).astype(np.float64) ** 2).sum() / ET[j]))
        out["rho_dualridge_I"] = float(np.mean(rI))
        out["rho_dualridge_0"] = float(np.mean(r0))
    return out


# ============================================================================ Q2: relative-position axiom

def schur_planes(K):
    """Orthogonal V whose consecutive column pairs span the invariant 2-planes of skew K."""
    T, Z = sl.schur(K, output="real")
    m = K.shape[0]
    cols, singles, k = [], [], 0
    while k < m:
        if k + 1 < m and abs(T[k + 1, k]) > 1e-12:
            cols += [k, k + 1]
            k += 2
        else:
            singles.append(k)
            k += 1
    cols += singles
    return Z[:, cols]


def torus_phi(Y, iters=40):
    """Closed-form alternating fit of per-language plane angles given planes (Y_i = Z_i V)."""
    Lg, n, m = Y.shape
    P = m // 2
    Yp = Y[..., : 2 * P].reshape(Lg, n, P, 2)
    phi = np.zeros((Lg, P))
    for _ in range(iters):
        c, s = np.cos(phi)[:, None], np.sin(phi)[:, None]
        x, y = Yp[..., 0], Yp[..., 1]
        H = np.stack([x * c - y * s, x * s + y * c], -1).mean(0)
        M11 = (x * H[None, ..., 0]).sum(1)
        M12 = (x * H[None, ..., 1]).sum(1)
        M21 = (y * H[None, ..., 0]).sum(1)
        M22 = (y * H[None, ..., 1]).sum(1)
        phi = np.arctan2(M12 - M21, M11 + M22)
    return phi


def rotmats(V, phi):
    """Q_i = V D(phi_i) V^T with D block [[c, s], [-s, c]] (row-vector right multiplication)."""
    m = V.shape[0]
    P = phi.shape[1]
    Qs = []
    for i in range(len(phi)):
        D = np.eye(m)
        c, s = np.cos(phi[i]), np.sin(phi[i])
        idx = np.arange(P) * 2
        D[idx, idx], D[idx, idx + 1], D[idx + 1, idx], D[idx + 1, idx + 1] = c, s, -s, c
        Qs.append(V @ D @ V.T)
    return np.stack(Qs)


def torus_fit(Z, V0, phi0, rank=None, steps=100, lr_s=2e-3, lr_p=1e-2):
    import torch
    torch.set_num_threads(4)
    Zt = torch.tensor(Z, dtype=torch.float32)
    Lg, n, m = Zt.shape
    P = m // 2
    V0t = torch.tensor(V0, dtype=torch.float32)
    S = torch.zeros(m, m, requires_grad=True)
    if rank is None:
        ph = torch.tensor(phi0, dtype=torch.float32, requires_grad=True)
        params = [{"params": [S], "lr": lr_s}, {"params": [ph], "lr": lr_p}]
        getphi = lambda: ph
    else:
        pc = phi0 - phi0.mean(0)
        u, s, vt = np.linalg.svd(pc, full_matrices=False)
        th = torch.tensor(u[:, :rank] * s[:rank], dtype=torch.float32, requires_grad=True)
        om = torch.tensor(vt[:rank], dtype=torch.float32, requires_grad=True)
        params = [{"params": [S], "lr": lr_s}, {"params": [th, om], "lr": lr_p}]
        getphi = lambda: th @ om
    E = float((Zt ** 2).sum())
    opt = torch.optim.Adam(params)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        V = V0t @ torch.linalg.matrix_exp(S - S.T)
        phv = getphi()
        B = (Zt @ V)[..., : 2 * P].reshape(Lg, n, P, 2)
        c, s = torch.cos(phv)[:, None, :], torch.sin(phv)[:, None, :]
        x, y = B[..., 0], B[..., 1]
        Br = torch.stack([x * c - y * s, x * s + y * c], -1)
        loss = ((Br - Br.mean(0)) ** 2).sum() / E
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    with torch.no_grad():
        V = (V0t @ torch.linalg.matrix_exp(S - S.T)).double().numpy()
        phv = getphi().double().numpy()
        extra = (th.double().numpy(), om.double().numpy()) if rank is not None else None
    V = polar(V)
    return V, phv, losses, extra


def pair_rho(Zt, Qs):
    return float(np.mean([rho(Zt[i] @ Qs[i] @ Qs[j].T, Zt[j]) for i, j in OPAIRS]))


def q2(Zf, Zt, m, do_torus=True, truth=None, n_anchor=12, heavy=True):
    Lg, n, _ = Zf.shape
    out = {}
    R, Ls = {}, {}
    st, dets = [], []
    for i, j in PAIRS:
        R[(i, j)] = procrustes(Zf[i], Zf[j])
        R[(j, i)] = R[(i, j)].T
        Lij, ang, dt = log_orth(R[(i, j)])
        Ls[(i, j)], Ls[(j, i)] = Lij, -Lij
        st.append(angle_stats(ang))
        dets.append(dt)
    P = m // 2
    for k in st[0]:
        out["ang_" + k] = float(np.mean([s[k] for s in st]))
    out["ang_pr_frac"] = out["ang_pr"] / P
    out["n_det_neg"] = int(np.sum(np.array(dets) < 0))
    out["L_norm2"] = float(np.mean([(Ls[p] ** 2).sum() for p in PAIRS]))
    out["L_dist"] = [[float(np.sqrt((Ls[(i, j)] ** 2).sum())) if i != j else 0.0 for j in range(Lg)]
                     for i in range(Lg)]
    # held-out rho in U
    out["rho_shift"] = float(np.mean([rho(Zt[i], Zt[j]) for i, j in OPAIRS]))
    out["rho_proc"] = float(np.mean([rho(Zt[i] @ R[(i, j)], Zt[j]) for i, j in OPAIRS]))
    Q = gpa(Zf, iters=30 if heavy else 15)
    out["rho_gpa"] = pair_rho(Zt, Q)
    # split-half stability of the logs, and noise-debiased statistics: inner products between the two
    # independent half-sample estimates keep the signal and cancel (independent, zero-mean) noise
    rs = np.random.RandomState(1)
    perm = rs.permutation(n)
    ha, hb = perm[: n // 2], perm[n // 2:]
    Ra_, Rb_, La_, Lb_ = {}, {}, {}, {}
    for i, j in PAIRS:
        Ra_[(i, j)], Rb_[(i, j)] = procrustes(Zf[i][ha], Zf[j][ha]), procrustes(Zf[i][hb], Zf[j][hb])
        Ra_[(j, i)], Rb_[(j, i)] = Ra_[(i, j)].T, Rb_[(i, j)].T
        La_[(i, j)], Lb_[(i, j)] = log_orth(Ra_[(i, j)])[0], log_orth(Rb_[(i, j)])[0]
        La_[(j, i)], Lb_[(j, i)] = -La_[(i, j)], -Lb_[(i, j)]
    cors, sig, top4s = [], [], []
    for p in PAIRS:
        La, Lb = La_[p], Lb_[p]
        num = (La * Lb).sum()
        cors.append(num / np.sqrt((La ** 2).sum() * (Lb ** 2).sum()))
        sig.append(num)
        M = La.T @ Lb
        ev = np.linalg.eigvalsh(0.5 * (M + M.T))[::-1]
        top4s.append(ev[:8].sum() / max(ev.sum(), 1e-12))
    out["splithalf_L_corr"] = float(np.mean(cors))
    out["splithalf_L_signal2"] = float(np.mean(sig))    # ~ ||L_true||^2 (at n/2)
    out["sh_top4_planes"] = float(np.mean(top4s))       # signal fraction in the top-4 planes (debiased spectrum)
    # commutators: raw, normalised by a Haar-conjugated partner, and split-half debiased
    Os = [np.linalg.qr(rs.randn(m, m))[0] for _ in range(2)]
    comm = lambda X, Y: X @ Y - Y @ X
    raw, nrm = [], []
    sh_num, sh_den = 0.0, 0.0
    for a in range(min(n_anchor, Lg)):
        others = [j for j in range(Lg) if j != a]
        for j, k in [(j, k) for jj, j in enumerate(others) for k in others[jj + 1:]]:
            X, Y = Ls[(a, j)], Ls[(a, k)]
            c = comm(X, Y)
            cn = np.sqrt((c ** 2).sum())
            raw.append(cn / np.sqrt((X ** 2).sum() * (Y ** 2).sum()))
            ref = [(comm(X, O @ Y @ O.T) ** 2).sum() for O in Os]
            nrm.append(cn / np.sqrt(np.mean(ref)))
            sh_num += (comm(La_[(a, j)], La_[(a, k)]) * comm(Lb_[(a, j)], Lb_[(a, k)])).sum()
            sh_den += np.mean([(comm(La_[(a, j)], O @ La_[(a, k)] @ O.T)
                                * comm(Lb_[(a, j)], O @ Lb_[(a, k)] @ O.T)).sum() for O in Os])
    out["comm_raw"] = float(np.mean(raw))
    out["comm_norm"] = float(np.mean(nrm))
    out["sh_comm"] = float(sh_num / sh_den)
    # debiased additivity / holonomy (relative to the debiased log signal)
    na, nh, dt_ = 0.0, 0.0, 0.0
    for i, j, k in TRIPLES:
        ada = La_[(i, j)] + La_[(j, k)] - La_[(i, k)]
        adb = Lb_[(i, j)] + Lb_[(j, k)] - Lb_[(i, k)]
        na += (ada * adb).sum()
        hla = log_orth(Ra_[(i, j)] @ Ra_[(j, k)] @ Ra_[(k, i)])[0]
        hlb = log_orth(Rb_[(i, j)] @ Rb_[(j, k)] @ Rb_[(k, i)])[0]
        nh += (hla * hlb).sum()
        dt_ += np.mean([(La_[p] * Lb_[p]).sum() for p in ((i, j), (j, k), (i, k))])
    out["sh_add"] = float(na / dt_)
    out["sh_hol"] = float(nh / dt_)
    del Ra_, Rb_, La_, Lb_
    # triples: abelian additivity vs holonomy (non-abelian composition error)
    eadd, ehol, cc, cab = [], [], [], []
    for i, j, k in TRIPLES:
        den = np.mean([(Ls[p] ** 2).sum() for p in ((i, j), (j, k), (i, k))])
        eadd.append(((Ls[(i, j)] + Ls[(j, k)] - Ls[(i, k)]) ** 2).sum() / den)
        Hl = log_orth(R[(i, j)] @ R[(j, k)] @ R[(k, i)])[0]
        ehol.append((Hl ** 2).sum() / den)
        Ti = Zt[i]
        Ei = (Ti ** 2).sum()
        cc.append(((Ti @ (R[(i, j)] @ R[(j, k)] - R[(i, k)])) ** 2).sum() / Ei)
        if heavy:
            cab.append(((Ti @ (sl.expm(Ls[(i, j)] + Ls[(j, k)]) - R[(i, k)])) ** 2).sum() / Ei)
    out["e_add"] = float(np.mean(eadd))
    out["e_hol"] = float(np.mean(ehol))
    out["c_comp"] = float(np.mean(cc))
    out["c_abelian"] = float(np.mean(cab)) if cab else float("nan")
    # distances / positions from the GPA logs (anchor-free)
    Dg = np.array([[np.sqrt((log_orth(Q[i] @ Q[j].T)[0] ** 2).sum()) if i != j else 0 for j in range(Lg)]
                   for i in range(Lg)])
    out["gpa_dist"] = Dg.tolist()
    out["mantel_gpa_script"] = mantel(Dg, DIFF_SCRIPT)
    out["mantel_gpa_family"] = mantel(Dg, DIFF_FAMILY)
    if not do_torus:
        return out
    # ---- commuting-generator (torus) model: Q_i = V D(phi_i) V^T, shared planes V, per-language angles
    rs2 = np.random.RandomState(2)
    K = sum(rs2.randn() * log_orth(Q[0] @ Q[j].T)[0] for j in range(1, Lg))
    V0 = schur_planes(K)
    phi0 = torus_phi(np.stack([Zf[i] @ V0 for i in range(Lg)]))
    out["rho_torus_init"] = pair_rho(Zt, rotmats(V0, phi0))
    V, phi, losses, _ = torus_fit(Zf, V0, phi0, rank=None)
    Qt = rotmats(V, phi)
    out["rho_torus"] = pair_rho(Zt, Qt)
    out["loss_torus"] = [losses[0], losses[-1]]
    out["torus_phi"] = phi.tolist()
    phic = np.angle(np.exp(1j * (phi - np.angle(np.exp(1j * phi).mean(0)))))   # circular centring
    Dphi = np.array([[np.sqrt((np.angle(np.exp(1j * (phi[i] - phi[j]))) ** 2).sum()) for j in range(Lg)]
                     for i in range(Lg)])
    out["torus_dist"] = Dphi.tolist()
    out["mantel_torus_script"] = mantel(Dphi, DIFF_SCRIPT)
    out["mantel_torus_family"] = mantel(Dphi, DIFF_FAMILY)
    sv = np.linalg.svd(phic - phic.mean(0), compute_uv=False) ** 2
    out["phi_sv_frac"] = (sv / sv.sum()).tolist()
    # low-rank positions: phi_i = theta_i Omega (languages as points in R^r)
    for r in (1, 2, 4):
        Vr, phr, lr_, (th, om) = torus_fit(Zf, V, phic, rank=r)
        out[f"rho_torus_r{r}"] = pair_rho(Zt, rotmats(Vr, phr))
        out[f"loss_torus_r{r}"] = [lr_[0], lr_[-1]]
        out[f"theta_r{r}"] = th.tolist()
        if r == 1:
            t = th[:, 0] * np.sign(np.linalg.norm(om))
            out["theta1_order"] = [LANGS[k] for k in np.argsort(t)]
            if truth is not None and "theta" in truth:
                out["theta1_corr_truth"] = float(abs(np.corrcoef(t, truth["theta"])[0, 1]))
            Dt = np.abs(t[:, None] - t[None, :])
            out["mantel_theta1_script"] = mantel(Dt, DIFF_SCRIPT)
            out["mantel_theta1_family"] = mantel(Dt, DIFF_FAMILY)
        if r == 2:
            Dt = np.sqrt(((th[:, None] - th[None]) ** 2).sum(-1))
            out["mantel_theta2_script"] = mantel(Dt, DIFF_SCRIPT)
            out["mantel_theta2_family"] = mantel(Dt, DIFF_FAMILY)
    return out


# ============================================================================ Q3: Jordan-type classification

def logm_gen(W):
    lam, P = np.linalg.eig(W)
    negreal = int(np.sum((np.abs(lam.imag) < 1e-9) & (lam.real < 0)))
    Lc = (P * np.log(lam.astype(complex))) @ np.linalg.inv(P)
    return Lc.real, lam, negreal


def q3(Zf, Zt):
    Lg, n, m = Zf.shape
    W, lW, lam = {}, {}, {}
    neg, hen, cplx, absl = 0, [], [], []
    for i, j in OPAIRS:
        W[(i, j)] = ridge(Zf[i], Zf[j])
        lW[(i, j)], lam[(i, j)], nr = logm_gen(W[(i, j)])
        neg += nr
        l = lam[(i, j)]
        fro = (W[(i, j)] ** 2).sum()
        hen.append(np.sqrt(max(fro - (np.abs(l) ** 2).sum(), 0)) / np.sqrt(fro))
        cplx.append(float(np.mean(np.abs(l.imag) > 1e-8)))
        absl.append(float(np.mean(np.abs(l))))
    out = dict(n_negreal_eig=neg, henrici=float(np.mean(hen)), frac_complex_eig=float(np.mean(cplx)),
               mean_abs_eig=float(np.mean(absl)))
    skw = lambda X: 0.5 * (X - X.T)
    sym = lambda X: 0.5 * (X + X.T)
    fr_rot, fr_sc, nX, nD, trD, raw_rot, raw_sc = [], [], [], [], [], [], []
    rW, rR, rX, rK, rS, rO = [], [], [], [], [], []
    Xs = {}
    for i, j in PAIRS:
        X = 0.5 * (lW[(i, j)] - lW[(j, i)])        # antisymmetrised generator: W_ij ~ exp(X - D), W_ji ~ exp(-X - D)
        Xs[(i, j)] = X
        D = -0.5 * (lW[(i, j)] + lW[(j, i)])       # symmetric-in-pair attenuation (regression dilution)
        K, S = skw(X), sym(X)
        x2 = (X ** 2).sum()
        fr_rot.append((K ** 2).sum() / x2)
        fr_sc.append((S ** 2).sum() / x2)
        nX.append(x2)
        nD.append((D ** 2).sum())
        trD.append(np.trace(D) / m)
        l2 = (lW[(i, j)] ** 2).sum()
        raw_rot.append((skw(lW[(i, j)]) ** 2).sum() / l2)
        raw_sc.append((sym(lW[(i, j)]) ** 2).sum() / l2)
        eX, eK, eS = sl.expm(X), sl.expm(K), sl.expm(S)
        Rp = procrustes(Zf[i], Zf[j])
        for (a, b, sgn) in ((i, j, 1), (j, i, -1)):
            tgt = Zt[b]
            rW.append(rho(Zt[a] @ W[(a, b)], tgt))
            rR.append(rho(Zt[a] @ (Rp if sgn > 0 else Rp.T), tgt))
            rX.append(rho(Zt[a] @ (eX if sgn > 0 else np.linalg.inv(eX)), tgt))
            rK.append(rho(Zt[a] @ (eK if sgn > 0 else eK.T), tgt))
            rS.append(rho(Zt[a] @ (eS if sgn > 0 else np.linalg.inv(eS)), tgt))
            rO.append(rho(Zt[a] @ polar(W[(a, b)]), tgt))
    # isotropic part and per-language log-scale positions: tr(X_ij)/m ~ s_j - s_i
    t = np.zeros((Lg, Lg))
    iso_fr, rot_tl = [], []
    for (i, j), X in Xs.items():
        t[i, j] = np.trace(X) / m
        t[j, i] = -t[i, j]
        S = sym(X)
        S_tl = S - t[i, j] * np.eye(m)
        iso_fr.append(m * t[i, j] ** 2 / max((S ** 2).sum(), 1e-30))
        k2 = (skw(X) ** 2).sum()
        rot_tl.append(k2 / (k2 + (S_tl ** 2).sum()))
    s_pos = t.mean(0)
    fitted = s_pos[None, :] - s_pos[:, None]
    iu = np.triu_indices(Lg, 1)
    out["iso_frac_of_sym"] = float(np.mean(iso_fr))
    out["frac_rot_vs_traceless"] = float(np.mean(rot_tl))
    out["scale_pos"] = s_pos.tolist()
    out["scale_pos_R2"] = float(1 - ((t - fitted)[iu] ** 2).sum() / (t[iu] ** 2).sum())
    if NTOK is not None:
        out["scale_pos_corr_logntok"] = float(np.corrcoef(s_pos, np.log(NTOK))[0, 1])
    out["scale_pos_corr_logsd"] = float(np.corrcoef(s_pos, 0.5 * np.log([(Zf[i] ** 2).sum() for i in range(Lg)]))[0, 1])
    out["rho_iso_scale"] = float(np.mean([rho(Zt[i] * np.exp(t[i, j]), Zt[j]) for i, j in OPAIRS]))
    # split-half debiased signal energy of the rotation (skew) vs scaling (symmetric) parts of X
    rs = np.random.RandomState(3)
    perm = rs.permutation(n)
    halves = (perm[: n // 2], perm[n // 2:])
    Xh = []
    for h in halves:
        lWh = {p: logm_gen(ridge(Zf[p[0]][h], Zf[p[1]][h]))[0] for p in OPAIRS}
        Xh.append({(i, j): 0.5 * (lWh[(i, j)] - lWh[(j, i)]) for i, j in PAIRS})
    sK = np.mean([(skw(Xh[0][p]) * skw(Xh[1][p])).sum() for p in PAIRS])
    sS = np.mean([(sym(Xh[0][p]) * sym(Xh[1][p])).sum() for p in PAIRS])
    sI = np.mean([np.trace(Xh[0][p]) * np.trace(Xh[1][p]) / m for p in PAIRS])
    out["sh_signal_rot"], out["sh_signal_scale"], out["sh_signal_iso"] = float(sK), float(sS), float(sI)
    out["sh_frac_rot"] = float(sK / (sK + sS))
    out.update(frac_rot=float(np.mean(fr_rot)), frac_scale=float(np.mean(fr_sc)), X_norm2=float(np.mean(nX)),
               D_norm2=float(np.mean(nD)), mean_log_atten=float(np.mean(trD)),
               raw_log_frac_rot=float(np.mean(raw_rot)), raw_log_frac_scale=float(np.mean(raw_sc)),
               rho_shift=float(np.mean([rho(Zt[i], Zt[j]) for i, j in OPAIRS])),
               rho_ridge=float(np.mean(rW)), rho_proc=float(np.mean(rR)), rho_expX=float(np.mean(rX)),
               rho_expK_rot=float(np.mean(rK)), rho_expS_scale=float(np.mean(rS)),
               rho_polarO=float(np.mean(rO)))
    return out


# ============================================================================ synthetic worlds

def gen_Q(kind, scale, s, seed=7):
    rs = np.random.RandomState(seed)
    info = {}
    if kind == "identity":
        Qs = [np.eye(s) for _ in range(NL)]
    elif kind == "abelian":
        P0 = np.linalg.qr(rs.randn(s, s))[0][:, :8]
        theta = rs.permutation(np.linspace(-1, 1, NL))
        omega = np.array([1.0, 0.8, 0.6, 0.4])
        Qs = []
        for i in range(NL):
            D = np.eye(8)
            for p in range(4):
                a = scale * theta[i] * omega[p]
                c, sn = np.cos(a), np.sin(a)
                D[2 * p:2 * p + 2, 2 * p:2 * p + 2] = [[c, sn], [-sn, c]]
            Qs.append(np.eye(s) + P0 @ (D - np.eye(8)) @ P0.T)
        info["theta"] = theta.tolist()
    elif kind == "nonabelian":
        Qs = []
        for i in range(NL):
            G = rs.randn(s, s)
            G = G - G.T
            Qs.append(sl.expm(scale * G / np.linalg.norm(G)))
    elif kind == "scaling":
        Qs = []
        for i in range(NL):
            G = rs.randn(s, s)
            G = G + G.T
            G -= np.trace(G) / s * np.eye(s)
            Qs.append(sl.expm(scale * G / np.linalg.norm(G)))
    else:
        raise ValueError(kind)
    return Qs, info


def calibrate_scale(kind, HtS, target, s):
    def f(sc):
        Qs, _ = gen_Q(kind, sc, s)
        return np.mean([((HtS @ (Qs[i] - Qs[j])) ** 2).sum() for i, j in PAIRS])
    lo, hi = 0.0, 3.0
    if f(hi) < target:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


def hub(X16, mu):
    H = np.zeros(X16.shape[1:], np.float32)
    for i in range(len(X16)):
        H += f32(X16[i]) - mu[i]
    return H / len(X16)


def make_world(Xf16, Xt16, mu, Sb, Qs, noise_s, seed):
    rs = np.random.RandomState(seed)
    Lg = len(Xf16)
    Hf, Ht = hub(Xf16, mu), hub(Xt16, mu)
    HfS, HtS = Hf @ Sb, Ht @ Sb
    Wf, Wt = np.empty_like(Xf16), np.empty_like(Xt16)
    for i in range(Lg):
        dQ = (Qs[i] - np.eye(len(Qs[i]))).astype(np.float32)
        Ef = (f32(Xf16[i]) - mu[i] - Hf)[rs.permutation(Hf.shape[0])]
        Et = (f32(Xt16[i]) - mu[i] - Ht)[rs.permutation(Ht.shape[0])]
        Wf[i] = (mu[i] + Hf + (HfS @ dQ) @ Sb.T + noise_s * Ef).astype(np.float16)
        Wt[i] = (mu[i] + Ht + (HtS @ dQ) @ Sb.T + noise_s * Et).astype(np.float16)
    return Wf, Wt


# ============================================================================ driver

def run_world(name, Xf16, Xt16, wu, truth=None, quick=False, full_q2=True, dual=True):
    res = {}
    mu, U, fr = pooled_basis(Xf16, MMAX)
    res["pca_frac"] = fr[:512].tolist()
    log(f"  {name}: basis")
    res["q1"] = q1_ladder(Xf16, Xt16, mu, U, fr, dual=dual and not quick, wu=wu)
    q = res["q1"]
    log(f"  {name}: Q1 id={q['rho_identity']:.3f} shift={q['rho_shift']:.3f} procU128+I={q['rho_procU128_I']:.3f} "
        f"ridgeU256+I={q['rho_ridgeU256_I']:.3f} dualI={q.get('rho_dualridge_I', np.nan):.3f} "
        f"dual0={q.get('rho_dualridge_0', np.nan):.3f} off2/trS={q['offset2_over_trS']:.3f}")
    res["q2"], res["q3"] = {}, {}
    for m in (64, 128):
        Zf, Zt = project(Xf16, mu, U[:, :m]), project(Xt16, mu, U[:, :m])
        if full_q2:
            o = q2(Zf, Zt, m, do_torus=(m <= 128), truth=truth, heavy=(m <= 128 or not quick))
            res["q2"][m] = o
            log(f"  {name} m={m}: rho shift/proc/gpa/torus/r1/r2/r4 = {o['rho_shift']:.3f}/{o['rho_proc']:.3f}/"
                f"{o['rho_gpa']:.3f}/{o.get('rho_torus', np.nan):.3f}/{o.get('rho_torus_r1', np.nan):.3f}/"
                f"{o.get('rho_torus_r2', np.nan):.3f}/{o.get('rho_torus_r4', np.nan):.3f} |L|2={o['L_norm2']:.2f} "
                f"PRfrac={o['ang_pr_frac']:.2f} top4={o['ang_top4']:.2f} sh-corr={o['splithalf_L_corr']:.2f} "
                f"comm={o['comm_raw']:.3f}/{o['comm_norm']:.3f} eadd={o['e_add']:.4f} ehol={o['e_hol']:.4f} "
                f"c={o['c_comp']:.4f} cab={o['c_abelian']:.4f} || SH: top4={o['sh_top4_planes']:.2f} comm={o['sh_comm']:.3f} "
                f"add={o['sh_add']:.3f} hol={o['sh_hol']:.3f} sig2={o['splithalf_L_signal2']:.2f} "
                f"th1={o.get('theta1_order')} th1truth={o.get('theta1_corr_truth', np.nan):.2f}")
        if m <= 128:
            o3 = q3(Zf, Zt)
            res["q3"][m] = o3
            log(f"  {name} m={m} Q3: rot={o3['frac_rot']:.2f} scale={o3['frac_scale']:.2f} |X|2={o3['X_norm2']:.3f} "
                f"|D|2={o3['D_norm2']:.3f} logatt={o3['mean_log_atten']:.3f} henrici={o3['henrici']:.3f} "
                f"cplx={o3['frac_complex_eig']:.2f} rho W/R/expX/expK/expS/polarO={o3['rho_ridge']:.3f}/"
                f"{o3['rho_proc']:.3f}/{o3['rho_expX']:.3f}/{o3['rho_expK_rot']:.3f}/{o3['rho_expS_scale']:.3f}/"
                f"{o3['rho_polarO']:.3f}/iso {o3['rho_iso_scale']:.3f} || SH rot/scale/iso={o3['sh_signal_rot']:.3f}/"
                f"{o3['sh_signal_scale']:.3f}/{o3['sh_signal_iso']:.3f} shfrac_rot={o3['sh_frac_rot']:.2f} "
                f"isofrac={o3['iso_frac_of_sym']:.2f} sposR2={o3['scale_pos_R2']:.2f} "
                f"corr(s,logntok)={o3.get('scale_pos_corr_logntok', np.nan):.2f} corr(s,logsd)={o3['scale_pos_corr_logsd']:.2f}")
        del Zf, Zt
    return res, mu, U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dual_identity", action="store_true")
    ap.add_argument("--worlds", default="real,identity,abelian,nonabelian,scaling")
    args = ap.parse_args()
    layer = args.layer
    global NTOK
    NTOK = np.load("C:/Users/ASUS/Documents/lang-geom/ntok_dev.npy").astype(np.float64).mean(1)
    wu = np.load("C:/Users/ASUS/Documents/lang-geom/wu_basis.npy").astype(np.float32)
    Xf16, Xt16 = load16("dev", layer), load16("devtest", layer)
    log(f"layer {layer} loaded {Xf16.shape} {Xt16.shape}")
    results = {"layer": layer}
    worlds = args.worlds.split(",")
    real, mu, U = run_world("real", Xf16, Xt16, wu, quick=args.quick)
    results["real"] = real
    # calibration from the real data at m=128
    U128 = U[:, :128]
    Zt = project(Xt16, mu, U128)
    Hz = Zt.mean(0)
    h = float((Hz ** 2).sum())
    e = float(np.mean([((Zt[i] - Hz) ** 2).sum() for i in range(NL)]))
    E_real = float(np.mean([(Zt[i] ** 2).sum() for i in range(NL)]))
    r128 = real["q2"][128]
    rho_star = r128["rho_gpa"]
    delta = max(r128["rho_shift"] - r128["rho_gpa"], 0.05)
    noise_s = float(np.sqrt(rho_star * h / (e * (2 - rho_star))))
    Sb = np.ascontiguousarray(U[:, :32])
    Ht = hub(Xt16, mu)
    HtS = (Ht @ Sb).astype(np.float64)
    del Zt
    results["calib"] = dict(rho_star=rho_star, delta_target=delta, noise_s=noise_s,
                            real_delta=r128["rho_shift"] - r128["rho_gpa"])
    log(f"calibration: rho*={rho_star:.3f} delta={delta:.3f} noise_s={noise_s:.3f}")
    for kind in [w for w in worlds if w != "real"]:
        sc = 0.0 if kind == "identity" else calibrate_scale(kind, HtS, delta * E_real, 32)
        Qs, info = gen_Q(kind, sc, 32)
        Wf, Wt = make_world(Xf16, Xt16, mu, Sb, Qs, noise_s, seed=11)
        r, _, _ = run_world(kind, Wf, Wt, wu, truth=info, quick=args.quick, full_q2=(kind != "scaling"),
                            dual=(kind == "identity" and args.dual_identity))
        r["scale"], r["info"] = sc, info
        results[kind] = r
        del Wf, Wt
        with open(f"{OUT}/layer{layer}{'_quick' if args.quick else ''}.json", "w") as f:
            json.dump(results, f)
    with open(f"{OUT}/layer{layer}{'_quick' if args.quick else ''}.json", "w") as f:
        json.dump(results, f)
    log("done")


if __name__ == "__main__":
    main()
