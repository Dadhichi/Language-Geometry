"""aligned.py LAYER -- 'transported truncation' test.
Instead of each language's own top-k PCA slice, give every language the SAME hub slice carried into its
own coordinates by the fitted gauge: coords_i = X_i P_i^{kb} Q_i U_k, where Q_i is the GPA gauge fitted
in kb=256 dims (dev) and U_k the hub's top-k principal directions.  Fit procrustes maps in these coords
(dev), score composition c on devtest.  If the per-language-PCA curvature is slice misalignment, c drops
to the map-noise level; genuine pair-specific structure (planted world V1) should survive.
Datasets: real, O256 flat surrogate, V1-like planted world (flat128 + group latent)."""
import sys
import time
import numpy as np
import holo as H

layer = int(sys.argv[1]) if len(sys.argv) > 1 else 14
KB = 256
t0 = time.time()
Xf, Xt = H.load("dev", layer), H.load("devtest", layer)


def c_of(A, T):
    A = A - A.mean(1, keepdims=True)
    R, _ = H.fit_maps(A)
    S = np.matmul(T.transpose(0, 2, 1), T)
    trS = np.trace(S, axis1=1, axis2=2)
    cs = [((R[b, p] @ R[p, q] - R[b, q]) * (S[b] @ (R[b, p] @ R[p, q] - R[b, q]))).sum() / trS[b] for b, p, q in H.OTRI]
    return float(np.mean(cs))


def aligned_c(A, T, ks=(32, 64, 128)):
    A = A - A.mean(1, keepdims=True)
    R, _ = H.fit_maps(A)
    Q = H.gpa(A, R)
    M = np.matmul(A, Q).mean(0)
    w, U = np.linalg.eigh(M.T @ M)
    U = U[:, ::-1]
    out = {}
    for k in ks:
        QU = Q @ U[:, :k]                       # [L, kb, k]
        Ak = np.matmul(A, QU)
        Tk = np.matmul(T, QU)
        out[f"aligned_k{k}"] = c_of(Ak, Tk)
        out[f"perlang_k{k}"] = c_of(A[:, :, :k], T[:, :, :k])
    return out


A, T, P, g = H.project(Xf, Xt, KB)
res = {"real": aligned_c(A, T)}
print("real", res["real"], f"[{time.time()-t0:.0f}s]", flush=True)
R, _ = H.fit_maps(A)
for kb in (256, 128):
    Ak, Tk, Pk = A[:, :, :kb] - A[:, :, :kb].mean(1, keepdims=True), T[:, :, :kb], np.ascontiguousarray(P[:, :, :kb])
    mdl = H.model_O(Ak, Tk, H.fit_maps(Ak)[0])
    An, Tn, Pn, gn = H.surrogate(Xf, Xt, g, Pk, mdl, seed=4242 + kb, k=KB)
    res[f"O{kb}"] = aligned_c(An, Tn)
    print(f"O{kb}", res[f"O{kb}"], f"[{time.time()-t0:.0f}s]", flush=True)
    if kb == 128:
        # planted world: flat128 surrogate + group-shared latent (as validate.py V1), analysed the same way
        Df, Dt = H.surrogate(Xf, Xt, g, Pk, mdl, seed=78, ambient_out=True)
        rs = np.random.RandomState(5)
        lam = (A[:, :, :64] ** 2).mean(1)
        Fc = np.concatenate([P[i, :, :64].astype(np.float64) * np.sqrt(lam[i])[None] for i in range(12)], 1)
        w, U = np.linalg.eigh(Fc.T @ Fc)
        C = Fc @ U[:, ::-1][:, :64] / np.sqrt(w[::-1][:64])
        Sst = H.orth(C @ rs.randn(64, 4)).astype(np.float32)
        groups = {7: 0, 8: 0, 0: 1, 1: 1, 2: 1, 3: 1}
        Nf = {q: rs.randn(Df.shape[1], 4).astype(np.float32) for q in (0, 1)}
        Nt = {q: rs.randn(Dt.shape[1], 4).astype(np.float32) for q in (0, 1)}
        Df, Dt = Df.astype(np.float32), Dt.astype(np.float32)
        for i, q in groups.items():
            amp = float(np.sqrt(lam[i, 10]))
            Df[i] += amp * Nf[q] @ Sst.T
            Dt[i] += amp * Nt[q] @ Sst.T
        A1, T1, _, _ = H.project(Df, Dt, KB)
        del Df, Dt
        res["V1planted"] = aligned_c(A1, T1)
        print("V1planted", res["V1planted"], f"[{time.time()-t0:.0f}s]", flush=True)
        A2, T2, _, _ = H.project(*H.surrogate(Xf, Xt, g, Pk, mdl, seed=78, ambient_out=True), KB)
        res["V2flat128"] = aligned_c(A2, T2)
        print("V2flat128", res["V2flat128"], f"[{time.time()-t0:.0f}s]", flush=True)
import json
json.dump(res, open(f"out/aligned_L{layer}.json", "w"), indent=1)
