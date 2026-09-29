"""One-factor (rank-1 + diagonal) test of the cross-language congruence matrix, ambient and gauge-free.
C_ij = sum_s <y_i,s, y_j,s> / sqrt(K_ii K_jj), y = x - mu_i (language-centred).  Model 'shared content with a
per-language gain, independent language-specific residuals':  y_i,s = g_i b_s + e_i,s  =>  C_ij = lam_i lam_j
(i != j), lam_i = g_i ||b|| / sqrt(g_i^2 ||b||^2 + ||e_i||^2)  (Spearman's one-factor model; tetrads vanish).
Residuals R_ij = C_ij - lam_i lam_j expose pair-specific shared structure (script / family / tokens).
lam fitted on dev; residuals scored on devtest; reproducibility = corr(R_dev, R_test)."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlib, clib

L = 12
iu = np.triu_indices(L, 1)
tok = dict(np.load(os.path.join(dlib.OUT, "tok_sim.npz")))
same_script = np.array([[clib.SCRIPT[i] == clib.SCRIPT[j] for j in range(L)] for i in range(L)])[iu]
same_fam = np.array([[clib.FAMILY[i] == clib.FAMILY[j] for j in range(L)] for i in range(L)])[iu]
tokov = tok["tok_hist_int"][iu]


def onefactor(C, iters=200):
    lam = np.sqrt(np.clip(np.linalg.eigh(C)[1][:, -1] ** 2 * np.linalg.eigh(C)[0][-1], 0, None))
    for _ in range(iters):
        M = C.copy(); np.fill_diagonal(M, lam ** 2)
        w, V = np.linalg.eigh(M)
        lam = np.abs(V[:, -1]) * np.sqrt(max(w[-1], 0))
    return lam


def congr(K):
    s = np.sqrt(np.diag(K))
    return K / s[:, None] / s[None]


def perm_p(res, ind, n=5000, seed=0):
    rs = np.random.RandomState(seed)
    obs = res[ind].mean() - res[~ind].mean()
    null = []
    for _ in range(n):
        p = rs.permutation(L)
        M = np.zeros((L, L)); M[iu] = ind; M = M + M.T
        indp = M[np.ix_(p, p)][iu].astype(bool)
        null.append(res[indp].mean() - res[~indp].mean())
    return float(obs), float((1 + (np.array(null) >= obs).sum()) / (1 + n))


rows = []
for l in dlib.LAYERS:
    A = dict(np.load(os.path.join(dlib.OUT, "layers", f"L{l}.npz")))
    Cd, Ct = congr(A["Kd"]), congr(A["Kt"])
    lam = onefactor(Cd)
    Rd = (Cd - np.outer(lam, lam))[iu]
    Rt = (Ct - np.outer(lam, lam))[iu]
    off = Ct[iu]
    r = dict(layer=l, mean_congruence=float(off.mean()), sd_congruence=float(off.std()),
             rms_resid_test=float(np.sqrt((Rt ** 2).mean())),
             frac_offdiag_var_explained=float(1 - (Rt ** 2).sum() / ((off - off.mean()) ** 2).sum()),
             split_noise_rms=float(np.sqrt(((Cd - Ct)[iu] ** 2).mean())),
             resid_reproducibility=float(np.corrcoef(Rd, Rt)[0, 1]))
    r["script_excess"], r["script_p"] = perm_p(Rt, same_script)
    r["family_excess"], r["family_p"] = perm_p(Rt, same_fam)
    r["resid_vs_tokoverlap_spearman"] = float(pd.Series(Rt).corr(pd.Series(tokov), method="spearman"))
    for i, lg in enumerate(dlib.LANGS):
        r[f"lam_{lg[:3]}"] = float(lam[i])
    rows.append(r)
T = pd.DataFrame(rows).set_index("layer")
T.to_csv(os.path.join(dlib.OUT, "onefactor.csv"))
pd.set_option("display.width", 250)
print(T[[c for c in T if not c.startswith("lam_")]].round(3).to_string())
print(T[[c for c in T if c.startswith("lam_")]].round(3).to_string())
# the largest residual pairs at L14
A = dict(np.load(os.path.join(dlib.OUT, "layers", "L14.npz")))
Ct = congr(A["Kt"]); lam = onefactor(congr(A["Kd"]))
R = Ct - np.outer(lam, lam)
order = np.argsort(R[iu])[::-1]
print("L14 top positive residual pairs:", [(dlib.LANGS[iu[0][q]][:3], dlib.LANGS[iu[1][q]][:3], round(float(R[iu][q]), 3)) for q in order[:8]])
print("L14 top negative residual pairs:", [(dlib.LANGS[iu[0][q]][:3], dlib.LANGS[iu[1][q]][:3], round(float(R[iu][q]), 3)) for q in order[-5:]])
