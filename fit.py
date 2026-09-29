#!/usr/bin/env python
"""
fit.py -- per-layer geometry of cross-language maps inside one model.

    python fit.py --data DATA_DIR --pooling mean --fit_split tedfit --test_splits devtest,tedtest \
        --k full --device cuda --out RESULTS_DIR [--mlp] [--layers 0,8,16] [--null_reps 2]
    python fit.py --selftest        # synthetic scenarios on CPU; asserts expected orderings

Conventions (row vectors): map i->j is z_j ~= (z_i - mu_i) R_ij + mu_j.  R_ij in O(d) (procrustes)
or GL(d) (ridge).  Maps are fit on --fit_split and scored on every --test_splits.

REQUIREMENT: n_fit > d for k=full.  With n < d an orthogonal map is unidentified off the data
span, and projecting into a shared PCA subspace manufactures inconsistency whenever per-language
rotations move energy across the cutoff (see selftest 'truncation').  Use PCA k only as a
secondary view.

Tables written to --out (all keyed by layer, k, and where relevant split):
    pairs.csv      src, tgt, method in {identity, shift, procrustes, ridge, mlp, sync_gpa}, rho, p1
    triples.csv    i, j, m, c_proc, c_ridge  (||A_i (R_ij R_jm - R_im)||^2 / ||A_i||^2; shifts cancel exactly)
    pivot.csv      pivot, src, tgt, delta = rho(direct) - rho(via pivot)
    sync.csv       cocycle = mean_{i!=j} tr(Q_i^T R_ij Q_j)/d with Q from joint (GPA) alignment; 1 iff consistent
                   spectral_frac (exact, only when d <= 1024), sync_gap = rho(joint) - rho(pairwise)
    fertility.csv  Spearman of per-sentence procrustes residual with |log fertility ratio| and log length
    energy.csv     per-language fraction of fit energy captured by the PCA basis (k runs only)
    null.csv       same statistics on consistent synthetic languages built from one real cloud
    summary.csv    one row per (layer, k, split)
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr


# ----------------------------------------------------------------------------- linear algebra backend

class LA:
    """numpy on CPU; torch on --device cuda for the O(d^3) pieces."""

    def __init__(self, device="cpu"):
        self.t = None
        if device != "cpu":
            import torch
            torch.backends.cuda.matmul.allow_tf32 = False
            self.t, self.dev = torch, torch.device(device)

    def to(self, X):
        if not self.t:
            return X
        if isinstance(X, self.t.Tensor):  # np.ascontiguousarray cannot read a CUDA tensor
            return X.to(self.dev, self.t.float32)
        return self.t.as_tensor(np.ascontiguousarray(X, dtype=np.float32), device=self.dev)

    def np(self, X):
        return X.cpu().numpy() if self.t else X

    def polar(self, M):
        if self.t:
            u, _, vh = self.t.linalg.svd(self.to(M))
            return self.np(u @ vh)
        u, _, vt = np.linalg.svd(M)
        return u @ vt

    def procrustes(self, A, B):
        """R in O(d) minimising ||A R - B||_F."""
        if self.t:
            return self.polar((self.to(A).T @ self.to(B)))
        return orthogonal_procrustes(A, B)[0].astype(np.float32)

    def ridge(self, A, B, rel=1e-3):
        d = A.shape[1]
        if self.t:
            a, b = self.to(A), self.to(B)
            G = a.T @ a
            lam = rel * self.t.trace(G) / d
            return self.np(self.t.linalg.solve(G + lam * self.t.eye(d, device=self.dev), a.T @ b))
        G = A.T @ A
        return np.linalg.solve(G + rel * np.trace(G) / d * np.eye(d), A.T @ B).astype(np.float32)

    def matmul(self, A, B):
        return self.np(self.to(A) @ self.to(B)) if self.t else A @ B

    # batched helpers: the same expressions run on numpy arrays or torch tensors
    def stack(self, xs):
        return self.t.stack(xs) if self.t else np.stack(xs)

    def tr(self, X):
        """Swap the last two axes."""
        return X.transpose(-1, -2) if self.t else np.swapaxes(X, -1, -2)

    def p1(self, pred, tn):
        """P@1 per batch entry: pred [B,m,d] (uncentred), tn [B,m,d] row-normalised targets -> [B]."""
        pn = pred / ((pred ** 2).sum(-1) ** 0.5)[..., None].clip(1e-9)
        best = (pn @ self.tr(tn)).argmax(-1)
        if self.t:
            return (best == self.t.arange(pred.shape[1], device=self.dev)).float().mean(-1)
        return (best == np.arange(pred.shape[1])).mean(-1)

    def gpa(self, A_list, Q0, iters=30):
        """min sum_i ||A_i Q_i - M||^2, Q_i in O(d).  Returns Q [L,d,d] (numpy).  Batched SVD over languages."""
        A, Q = self.to(np.stack(A_list)), self.to(np.stack(Q0))
        At = self.tr(A)
        svd = self.t.linalg.svd if self.t else np.linalg.svd
        for _ in range(iters):
            M = (A @ Q).mean(0)
            u, _, vh = svd(At @ M)
            Q = u @ vh
        return self.np(Q).astype(np.float32)


# ----------------------------------------------------------------------------- basics

def pca_basis(X, k, la=None):
    """X [N,d] -> mean g [d], basis P [d,k] (None when k is None).  Exact top-k principal directions from
    the eigendecomposition of the smaller Gram matrix (N x N when N < d), in float64, on the backend."""
    g = X.mean(0)
    if k is None or k >= X.shape[1]:
        return g, None
    t = la.t if la is not None else None
    if t:
        Xc, eigh = t.as_tensor(X - g, device=la.dev).double(), t.linalg.eigh
    else:
        Xc, eigh = (X - g).astype(np.float64), np.linalg.eigh
    n, d = Xc.shape
    if n < d:
        w, U = eigh(Xc @ Xc.T)                                   # ascending
        P = (Xc.T @ U[:, -k:]) / (w[-k:].clip(1e-30) ** 0.5)     # right singular vectors
    else:
        P = eigh(Xc.T @ Xc)[1][:, -k:]
    P = la.np(P.flip(-1)) if t else P[:, ::-1]                   # descending variance
    return g, np.ascontiguousarray(P.astype(np.float32))


def make_projector(Xfit, k, mode, la=None):
    """Returns f(X [L,n,d]) -> [L,n,k].  mode 'shared': one basis for all languages (NOT consistency-
    preserving under per-language rotations); 'per_lang': each language's own top-k basis, which
    preserves consistency under the gauge model because a rotation of the cloud rotates its basis."""
    L, n, d = Xfit.shape
    if k is None:
        return lambda X: X, None
    if mode == "shared":
        g, P = pca_basis(Xfit.reshape(-1, d), k, la)
        return (lambda X: np.einsum("lnd,dk->lnk", X - g, P)), [P] * L
    bases = [pca_basis(Xfit[i], k, la) for i in range(L)]
    return (lambda X: np.stack([(X[i] - bases[i][0]) @ bases[i][1] for i in range(L)])), [b[1] for b in bases]


def mlp_fit_predict(A, B, A_test, seed=0, hidden=None, epochs=300, device="cpu"):
    import torch
    torch.manual_seed(seed)
    k = A.shape[1]
    hidden = hidden or min(2 * k, 4096)
    n = len(A)
    perm = np.random.RandomState(seed).permutation(n)
    va, tr = perm[: n // 10], perm[n // 10:]
    At, Bt = torch.tensor(A, device=device), torch.tensor(B, device=device)
    net = torch.nn.Sequential(torch.nn.Linear(k, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, k)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_state, patience = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        for s in range(0, len(tr), 256):
            idx = tr[s:s + 256]
            opt.zero_grad()
            ((net(At[idx]) - Bt[idx]) ** 2).mean().backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(((net(At[va]) - Bt[va]) ** 2).mean())
        if v < best - 1e-6:
            best, best_state, patience = v, {a: b.clone() for a, b in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 20:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        return net(torch.tensor(A_test, device=device)).cpu().numpy()


def spectral_frac(Rb, L, d, la):
    """Top-d eigenvalue mass of the block matrix of pairwise rotations / (L d); 1 iff consistent.
    Rb[i] = stack_j R_ij with identity on the diagonal, so block (i, j) of G is R_ij.  float64."""
    G = la.stack(Rb)                                   # [L, L, d, d] -> rows (i, a), cols (j, b)
    if la.t:
        G = G.permute(0, 2, 1, 3).reshape(L * d, L * d).double()
        w = la.np(la.t.linalg.eigvalsh(0.5 * (G + G.T)))
    else:
        G = G.transpose(0, 2, 1, 3).reshape(L * d, L * d).astype(np.float64)
        w = np.linalg.eigvalsh(0.5 * (G + G.T))
    return float(w[-d:].sum() / (L * d))


# ----------------------------------------------------------------------------- core

def analyze(Zfit, tests, k_label, la, langs=None, ntok_fit=None, do_mlp=False, seed=0,
            mlp_device="cpu", verbose=False):
    """Zfit [L,n,d]; tests: {split: (Z [L,m,d], ntok [L,m] or None)}.  Returns dict of DataFrames."""
    L, n_fit, d = Zfit.shape
    langs = langs or [str(i) for i in range(L)]
    mu = Zfit.mean(1)
    A = Zfit - mu[:, None, :]
    T = {s: Z - mu[:, None, :] for s, (Z, _) in tests.items()}

    # ---- pairwise maps
    t0 = time.time()
    R, W = {}, {}
    for i, j in itertools.combinations(range(L), 2):
        R[(i, j)] = la.procrustes(A[i], A[j])
        R[(j, i)] = R[(i, j)].T
        W[(i, j)] = la.ridge(A[i], A[j])
        W[(j, i)] = la.ridge(A[j], A[i])
    if verbose:
        print(f"    pairwise fits {time.time() - t0:.0f}s", flush=True)

    # ---- joint alignment (GPA), init at the English/first-language gauge
    Q0 = [np.eye(d, dtype=np.float32)] + [R[(0, j)].T for j in range(1, L)]   # R_0j = Q_0 Q_j^T with Q_0 = I
    Q = la.gpa([A[i] for i in range(L)], Q0)

    # Everything below runs batched on the backend (GPU with --device cuda): Rb[i] = stack_j R_ij with
    # identity on the (never scored) diagonal, so T_i @ Rb[i] maps language i to every target at once.
    I = np.eye(d, dtype=np.float32)
    Rb = [la.to(np.stack([R[(i, j)] if j != i else I for j in range(L)])) for i in range(L)]
    Wb = [la.to(np.stack([W[(i, j)] if j != i else I for j in range(L)])) for i in range(L)]
    Qb = la.to(Q)
    QbT = la.tr(Qb)
    off = [np.arange(L) != i for i in range(L)]
    # tr(Q_i^T R_ij Q_j) = sum(Q_i * (R_ij Q_j))
    cocycle = float(np.mean(np.concatenate([la.np((Qb[i][None] * (Rb[i] @ Qb)).sum((1, 2)))[off[i]]
                                            for i in range(L)])) / d)
    spec = spectral_frac(Rb, L, d, la) if d <= 1024 else np.nan

    methods = ["identity", "shift", "procrustes", "ridge", "sync_gpa"] + (["mlp"] if do_mlp else [])
    mu_b = la.to(mu)[:, None, :]
    pairs, trip, piv, fert, summ = [], [], [], [], []
    for split, (Ztest, ntok) in tests.items():
        # centred predictions C = pred - mu_j, so pred - tgt = C - T_j and rho = ||C - T_j||^2 / ||T_j||^2
        Tt, Zt = la.to(T[split]), la.to(Ztest)
        energy = (Tt ** 2).sum((1, 2))                                     # [L]
        tn = Zt / ((Zt ** 2).sum(-1) ** 0.5)[..., None].clip(1e-9)          # row-normalised targets for P@1
        D, Dr, res_proc = [], [], []
        for i in range(L):
            C = {"identity": Zt[i] - mu_b, "shift": Tt[i][None], "procrustes": Tt[i] @ Rb[i],
                 "ridge": Tt[i] @ Wb[i], "sync_gpa": (Tt[i] @ Qb[i]) @ QbT}
            if do_mlp:
                C["mlp"] = la.to(np.stack([mlp_fit_predict(A[i], A[j], T[split][i], seed=seed, device=mlp_device)
                                           if j != i else T[split][i] for j in range(L)]))
            rh = {k_: la.np(((c - Tt) ** 2).sum((1, 2)) / energy) for k_, c in C.items()}
            p1 = {k_: la.np(la.p1(c + mu_b, tn)) for k_, c in C.items()}
            D.append(C["procrustes"])
            Dr.append(C["ridge"])
            res_proc.append(((C["procrustes"] - Tt) ** 2).sum((1, 2)))
            per_sent = la.np(((C["procrustes"] - Tt) ** 2).sum(-1) / (Tt ** 2).sum(-1))   # [L, m]
            for j in np.flatnonzero(off[i]):
                for name in methods:
                    pairs.append(dict(split=split, src=langs[i], tgt=langs[j], method=name,
                                      rho=float(rh[name][j]), p1=float(p1[name][j])))
                if ntok is not None:
                    r = per_sent[j]
                    fert.append(dict(split=split, src=langs[i], tgt=langs[j],
                                     spearman_fert=spearmanr(r, np.abs(np.log(ntok[i] / ntok[j])))[0],
                                     spearman_len=spearmanr(r, np.log(ntok[i] + ntok[j]))[0]))
        # ---- composition + pivot, batched over m: via_ijm = T_i R_ij R_jm, direct_im = T_i R_im = D[i][m]
        t0 = time.time()
        c_p, c_r, dl = (np.zeros((L, L, L)) for _ in range(3))
        for i, j in itertools.permutations(range(L), 2):
            via = D[i][j] @ Rb[j]                                          # [L(m), n, d]
            via_r = Dr[i][j] @ Wb[j]
            c_p[i, j] = la.np(((via - D[i]) ** 2).sum((1, 2)) / energy[i])
            c_r[i, j] = la.np(((via_r - Dr[i]) ** 2).sum((1, 2)) / energy[i])
            # rho(direct) - rho(via pivot j), both scored against T_m
            dl[i, j] = la.np((res_proc[i] - ((via - Tt) ** 2).sum((1, 2))) / energy)
        for i, j in itertools.permutations(range(L), 2):
            for m in range(L):
                if m in (i, j):
                    continue
                trip.append(dict(split=split, i=langs[i], j=langs[j], m=langs[m],
                                 c_proc=float(c_p[i, j, m]), c_ridge=float(c_r[i, j, m])))
                piv.append(dict(split=split, pivot=langs[j], src=langs[i], tgt=langs[m], delta=float(dl[i, j, m])))
        if verbose:
            print(f"    triples ({split}) {time.time() - t0:.0f}s", flush=True)

    pairs, trip, piv, fert = map(pd.DataFrame, (pairs, trip, piv, fert))

    # ---- split-half estimation noise (scale reference)
    rs = np.random.RandomState(seed)
    half = rs.permutation(n_fit)
    ha, hb = half[: n_fit // 2], half[n_fit // 2:]
    first = next(iter(tests))
    sh = []
    for i, m in list(itertools.permutations(range(L), 2))[: min(L * (L - 1), 20)]:
        D = la.matmul(T[first][i], la.procrustes(A[i][ha], A[m][ha]) - la.procrustes(A[i][hb], A[m][hb]))
        sh.append(float((D ** 2).sum() / (T[first][i] ** 2).sum()))

    # ---- joint-model decomposition (used by the null builder in analyze_with_null)
    AQ = [la.matmul(A[i], Q[i]) for i in range(L)]
    M_fit = np.mean(AQ, axis=0)
    hat_fit = [la.matmul(M_fit, Q[i].T) for i in range(L)]
    E_fit = [A[i] - hat_fit[i] for i in range(L)]
    hat_t, E_t = {}, {}
    for s in tests:
        M_t = np.mean([la.matmul(T[s][i], Q[i]) for i in range(L)], axis=0)
        hat_t[s] = [la.matmul(M_t, Q[i].T) for i in range(L)]
        E_t[s] = [T[s][i] - hat_t[s][i] for i in range(L)]
    # dof-corrected residual scales: on the fit split the hub (n d), the L-1 free rotations and the
    # centering absorb noise; on a test split only the hub mean does (rotations are fixed).
    rss = sum(float((e ** 2).sum()) for e in E_fit)
    dof = L * n_fit * d - n_fit * d - (L - 1) * d * (d - 1) / 2 - L * d
    infl = float(np.sqrt(rss / max(dof, 1) / (rss / (L * n_fit * d)))) if dof > 0 else 1.0
    infl_t = float(np.sqrt(L / (L - 1)))
    model = dict(mu=mu, hat_fit=hat_fit, E_fit=E_fit, hat_t=hat_t, E_t=E_t, infl=infl, infl_t=infl_t)
    null = pd.DataFrame()

    # ---- summary per split
    for split in tests:
        ps = pairs[pairs.split == split]
        m = ps.groupby("method")["rho"].mean()
        p1 = ps.groupby("method")["p1"].mean()
        ts = trip[trip.split == split]
        fs = fert[fert.split == split] if len(fert) else fert
        row = dict(split=split, k=k_label, n_fit=n_fit, n_test=tests[split][0].shape[1], d=d,
                   rho_identity=m["identity"], rho_shift=m["shift"], rho_procrustes=m["procrustes"],
                   rho_ridge=m["ridge"], rho_mlp=m.get("mlp", np.nan), rho_sync=m["sync_gpa"],
                   p1_identity=p1["identity"], p1_shift=p1["shift"], p1_procrustes=p1["procrustes"],
                   p1_ridge=p1["ridge"], p1_mlp=p1.get("mlp", np.nan), p1_sync=p1["sync_gpa"],
                   c_proc_mean=ts["c_proc"].mean(), c_proc_p95=ts["c_proc"].quantile(0.95),
                   c_ridge_mean=ts["c_ridge"].mean(), splithalf=float(np.mean(sh)),
                   cocycle=cocycle, spectral_frac=spec, sync_gap=m["sync_gpa"] - m["procrustes"],
                   null_resid_inflation=infl,
                   fert_mean=fs["spearman_fert"].mean() if len(fs) else np.nan,
                   len_mean=fs["spearman_len"].mean() if len(fs) else np.nan)
        pm = piv[piv.split == split].groupby("pivot")["delta"].mean()
        for p in pm.index:
            row[f"pivot_{p}"] = pm[p]
        summ.append(row)
    summ = pd.DataFrame(summ)
    sync = pd.DataFrame([dict(k=k_label, cocycle=cocycle, spectral_frac=spec)])
    for df in (pairs, trip, piv, fert):
        df.insert(0, "k", k_label)
    return dict(pairs=pairs, triples=trip, pivot=piv, fertility=fert, null=null, sync=sync, summary=summ,
                model=model)


# ----------------------------------------------------------------------------- projection + null

def subspace_stability(Xfit, k, seed=0, la=None):
    """Per-language split-half agreement of the top-k subspace: mean cos^2 of principal angles (1 = stable)."""
    L, n, d = Xfit.shape
    rs = np.random.RandomState(seed)
    perm = rs.permutation(n)
    out = []
    for i in range(L):
        _, Pa = pca_basis(Xfit[i][perm[: n // 2]], k, la)
        _, Pb = pca_basis(Xfit[i][perm[n // 2:]], k, la)
        out.append(float(((Pa.T @ Pb) ** 2).sum() / k))
    return out


def analyze_with_null(Xfit, Xtests, k, mode, la, null_reps=0, **kw):
    """Xfit [L,n,d]; Xtests {split: (X [L,m,d], ntok)}.  Projects (per-language or shared PCA, or none),
    runs analyze, then builds a consistent null IN THE AMBIENT SPACE: joint-model fit + dof-inflated
    permuted in-subspace residual + permuted out-of-subspace residual, lifted back to d dims, and pushed
    through the same projection and analysis.  So the null carries subspace-estimation noise too."""
    L, n, d = Xfit.shape
    proj, bases = make_projector(Xfit, k, mode, la)
    Zf = proj(Xfit)
    tests = {s: (proj(X), nt) for s, (X, nt) in Xtests.items()}
    out = analyze(Zf, tests, "full" if k is None else k, la, **kw)
    mdl = out["model"]
    if bases is not None:
        out["energy"] = pd.DataFrame(dict(k=k, pca=mode, lang=kw.get("langs") or list(range(L)),
                                          captured=((Zf - Zf.mean(1, keepdims=True)) ** 2).sum((1, 2))
                                          / ((Xfit - Xfit.mean(1, keepdims=True)) ** 2).sum((1, 2)),
                                          stability=subspace_stability(Xfit, k, la=la)))
    rows = []
    for rep in range(null_reps):
        rsn = np.random.RandomState(1000 + rep)

        def lift(hat, E, X, i, m, infl):
            # consistent part + permuted, inflated in-subspace residual, back in d dims.  The
            # out-of-subspace part of the real data is kept as is (conservative: any pair-specific
            # structure living in the tail is also present in the null).
            Zs = hat + infl * E[rsn.permutation(len(E))]
            if bases is None:
                return Zs + m
            P = bases[i]
            Xc = X - m
            off = Xc - (Xc @ P) @ P.T
            return Zs @ P.T + off + m
        # projection means: per-language means for per_lang, global mean for shared
        mf = Xfit.mean(1) if mode == "per_lang" or bases is None else np.repeat(Xfit.reshape(-1, d).mean(0)[None], L, 0)
        Xf_null = np.stack([lift(mdl["hat_fit"][i], mdl["E_fit"][i], Xfit[i], i, mf[i], mdl["infl"])
                            for i in range(L)])
        Xt_null = {s: (np.stack([lift(mdl["hat_t"][s][i], mdl["E_t"][s][i], Xtests[s][0][i], i, mf[i],
                                      mdl["infl_t"]) for i in range(L)]), None) for s in Xtests}
        proj_n, _ = make_projector(Xf_null, k, mode, la)
        o = analyze(proj_n(Xf_null), {s: (proj_n(X), None) for s, (X, _) in Xt_null.items()},
                    "full" if k is None else k, la, seed=kw.get("seed", 0) + rep)
        for _, r in o["summary"].iterrows():
            rows.append(dict(rep=rep, split=r["split"], c_proc=r["c_proc_mean"], c_ridge=r["c_ridge_mean"],
                             cocycle=r["cocycle"], spectral_frac=r["spectral_frac"], sync_gap=r["sync_gap"],
                             rho_procrustes=r["rho_procrustes"], rho_ridge=r["rho_ridge"]))
    null = pd.DataFrame(rows)
    if len(null):
        for s in out["summary"]["split"].unique():
            ns = null[null.split == s]
            sel = out["summary"]["split"] == s
            for col, val in (("null_c_proc_mean", ns["c_proc"].mean()), ("null_c_proc_max", ns["c_proc"].max()),
                             ("null_c_ridge_max", ns["c_ridge"].max()), ("null_cocycle_min", ns["cocycle"].min()),
                             ("null_spectral_min", ns["spectral_frac"].min()),
                             ("null_sync_gap_max", ns["sync_gap"].max()),
                             ("null_rho_procrustes", ns["rho_procrustes"].mean())):
                out["summary"].loc[sel, col] = val
        null.insert(0, "k", "full" if k is None else k)
    out["null"] = null
    out.pop("model")
    return out


# ----------------------------------------------------------------------------- real-data driver

def load_layer(data, pooling, split, layer):
    p = os.path.join(data, f"{pooling}_{split}_L{layer}.f16.npy")
    return np.asarray(np.load(p, mmap_mode="r"), dtype=np.float32)


def run_real(args):
    with open(os.path.join(args.data, "meta.json")) as f:
        meta = json.load(f)
    langs, n_layers = meta["langs"], meta["n_layers"]
    if len(langs) < 3:
        sys.exit(f"need >= 3 languages (composition and pivot tests use triples); {args.data} has {langs}")
    la = LA(args.device)
    test_splits = args.test_splits.split(",")
    ntok = {s: np.load(os.path.join(args.data, f"ntok_{s}.npy")).astype(np.float64) for s in test_splits}
    ntok_fit = np.load(os.path.join(args.data, f"ntok_{args.fit_split}.npy")).astype(np.float64)
    layers = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    ks = [None if x == "full" else int(x) for x in args.k.split(",")]
    os.makedirs(args.out, exist_ok=True)
    acc = {}

    for layer in layers:
        t0 = time.time()
        Xf = load_layer(args.data, args.pooling, args.fit_split, layer)
        Xt = {s: load_layer(args.data, args.pooling, s, layer) for s in test_splits}
        L, n_fit, d = Xf.shape
        for k in ks:
            if k is None and n_fit <= d:
                print(f"WARNING: n_fit={n_fit} <= d={d}; full-d maps are not identified. Skipping k=full.")
                continue
            out = analyze_with_null(Xf, {s: (Xt[s], ntok[s]) for s in test_splits}, k, args.pca, la,
                                    null_reps=args.null_reps, langs=langs, ntok_fit=ntok_fit, do_mlp=args.mlp,
                                    mlp_device=args.device, verbose=args.verbose)
            for name, df in out.items():
                df.insert(0, "layer", layer)
                acc.setdefault(name, []).append(df)
            for _, s in out["summary"].iterrows():
                print(f"L{layer:2d} k={str(k):>4} {s.split:8s} rho id/shift/proc/ridge/sync="
                      f"{s.rho_identity:.3f}/{s.rho_shift:.3f}/{s.rho_procrustes:.3f}/{s.rho_ridge:.3f}/{s.rho_sync:.3f} "
                      f"p1={s.p1_procrustes:.3f} c={s.c_proc_mean:.4f}(null {s.get('null_c_proc_mean', np.nan):.4f}) "
                      f"cocycle={s.cocycle:.4f}(null {s.get('null_cocycle_min', np.nan):.4f}) "
                      f"gap={s.sync_gap:+.4f} fert={s.fert_mean:+.3f} [{time.time() - t0:.0f}s]", flush=True)
        for name in acc:
            pd.concat(acc[name], ignore_index=True).to_csv(os.path.join(args.out, f"{name}.csv"), index=False)


# ----------------------------------------------------------------------------- self-test

def synth(kind, L=6, n_fit=400, n_test=300, d=96, noise=0.15, seed=0):
    rs = np.random.RandomState(seed)
    spec = 1.0 / np.sqrt(np.arange(1, d + 1))
    Hd = (rs.randn(n_fit, d) * spec).astype(np.float32)
    Ht = (rs.randn(n_test, d) * spec).astype(np.float32)
    s = float(np.sqrt((Hd ** 2).mean()))
    script = [0, 0, 0, 1, 1, 1]
    family = [0, 0, 1, 0, 1, 1]
    Fw = {g: (rs.randn(d, d) / np.sqrt(d)).astype(np.float32) for g in ("s0", "s1", "f0", "f1")}

    def F(H, g):
        return 1.0 * s * np.tanh(H @ Fw[g] / s)

    Zd, Zt = [], []
    for i in range(L):
        mu = (rs.randn(d) * spec * 3).astype(np.float32)
        if kind == "shift":
            fd, ft = Hd, Ht
        elif kind == "gauge":
            Q = np.linalg.svd(rs.randn(d, d))[0].astype(np.float32)
            fd, ft = Hd @ Q, Ht @ Q
        elif kind == "linear_gauge":
            M = (np.eye(d) + 0.6 * rs.randn(d, d) / np.sqrt(d)).astype(np.float32)
            fd, ft = Hd @ M, Ht @ M
        elif kind == "family":
            gs, gf = f"s{script[i]}", f"f{family[i]}"
            fd, ft = Hd + F(Hd, gs) + F(Hd, gf), Ht + F(Ht, gs) + F(Ht, gf)
        else:
            raise ValueError(kind)
        Zd.append(fd + mu + noise * s * rs.randn(n_fit, d).astype(np.float32))
        Zt.append(ft + mu + noise * s * rs.randn(n_test, d).astype(np.float32))
    return np.stack(Zd), np.stack(Zt)


def selftest(do_mlp=False):
    la = LA("cpu")
    res = {}
    cols = ["rho_shift", "rho_procrustes", "null_rho_procrustes", "rho_ridge", "p1_procrustes", "c_proc_mean",
            "null_c_proc_max", "c_ridge_mean", "null_c_ridge_max", "splithalf", "cocycle", "null_cocycle_min",
            "spectral_frac", "sync_gap", "null_sync_gap_max"]
    scenarios = (("shift", None, "shared"), ("gauge", None, "shared"), ("linear_gauge", None, "shared"),
                 ("family", None, "shared"), ("gauge", 32, "shared"), ("gauge", 32, "per_lang"),
                 ("family", 32, "per_lang"))
    for kind, k, mode in scenarios:
        Xd, Xt = synth(kind)
        out = analyze_with_null(Xd, {"test": (Xt, None)}, k, mode, la, null_reps=3, do_mlp=do_mlp)
        s = out["summary"].iloc[0]
        name = kind if k is None else f"{kind}_k{k}_{mode}"
        res[name] = s
        print(f"\n[{name}]")
        for c in cols:
            print(f"  {c:20s} {s[c]:.4f}")
    sh, ga, lg, fa = (res[n] for n in ("shift", "gauge", "linear_gauge", "family"))
    tsh, tpl, fpl = res["gauge_k32_shared"], res["gauge_k32_per_lang"], res["family_k32_per_lang"]
    checks = [
        ("shift: translation alone ~ procrustes", sh.rho_shift - sh.rho_procrustes < 0.03),
        ("shift: all pivots ~ neutral", all(abs(v) < 0.02 for c, v in sh.items() if c.startswith("pivot_"))),
        ("gauge: rotation needed", ga.rho_shift > 3 * ga.rho_procrustes),
        ("gauge: composition inside null", ga.c_proc_mean < 2 * ga.null_c_proc_max),
        ("gauge: cocycle ~ 1", ga.cocycle > 0.98 and ga.cocycle > ga.null_cocycle_min - 0.01),
        ("gauge: joint ~ pairwise", ga.sync_gap < 2 * max(ga.null_sync_gap_max, 0.005)),
        ("linear_gauge: ridge beats procrustes", lg.rho_ridge < 0.7 * lg.rho_procrustes),
        ("linear_gauge: ridge maps compose (inside null)", lg.c_ridge_mean < 2 * lg.null_c_ridge_max),
        ("family: procrustes composition > null", fa.c_proc_mean > 2 * fa.null_c_proc_max),
        ("family: cocycle below null", fa.cocycle < fa.null_cocycle_min - 0.002),
        ("truncation: shared PCA k=32 breaks the gauge model (rho and c far above per-language PCA)",
         tsh.rho_procrustes > 5 * tpl.rho_procrustes and tsh.c_proc_mean > 20 * tpl.c_proc_mean),
        ("per-language PCA k=32: gauge artefact small vs family signal (c/rho < 5% vs > 15%)",
         tpl.c_proc_mean / tpl.rho_procrustes < 0.05 and fpl.c_proc_mean / fpl.rho_procrustes > 0.15),
        ("null calibration: rho within 15% at full d; conservative (0.85x-1.8x) under per-language PCA",
         abs(ga.null_rho_procrustes / ga.rho_procrustes - 1) < 0.15
         and 0.85 < tpl.null_rho_procrustes / tpl.rho_procrustes < 1.8),
        ("per-language PCA k=32: gauge artefact covered by the null", tpl.c_proc_mean < tpl.null_c_proc_max),
        ("per-language PCA k=32 still detects family structure", fpl.c_proc_mean > 3 * fpl.null_c_proc_max),
    ]
    print()
    ok = True
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok &= bool(passed)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--pooling", default="mean", choices=["mean", "last"])
    ap.add_argument("--fit_split", default="tedfit")
    ap.add_argument("--test_splits", default="devtest,tedtest")
    ap.add_argument("--k", default="full", help="comma list of ints and/or 'full'")
    ap.add_argument("--pca", default="per_lang", choices=["per_lang", "shared"])
    ap.add_argument("--layers", default="all")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mlp", action="store_true")
    ap.add_argument("--null_reps", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest(do_mlp=args.mlp) else 1)
    if not (args.data and args.out):
        ap.error("--data and --out required")
    run_real(args)


if __name__ == "__main__":
    main()
