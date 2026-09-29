"""validate.py -- synthetic worlds built from the real L14 cloud, pushed through run.run_dataset at k=64.
  V0 'aligned':   flat O-gauge in exactly 64 dims (= fit.py null world). Per-language top-64 slices align.
  V2 'flat128':   flat O-gauge in 128 dims + real tail; per-language top-64 truncation cuts different slices.
  V1 'planted':   V2 + genuine curvature: a group-shared latent (r=4, in a fixed ambient subspace S* inside
                  the top-64 pooled content span) added to {zho,jpn} and to {eng,deu,fra,spa} (independent
                  latents per group, independent per split).  Maps inside a group can carry it, maps across
                  groups cannot -> curvature on mixed triangles, located in S*.
"""
import os
import sys
import time
import numpy as np
import holo as H
import run

LAYER = 14
t0 = time.time()
Xf, Xt = H.load("dev", LAYER), H.load("devtest", LAYER)
ntok = np.load(os.path.join(H.DATA, "ntok_dev.npy")).astype(np.float64)
which = sys.argv[1] if len(sys.argv) > 1 else "V0,V2,V1"

A0, T0, P0, g0 = H.project(Xf, Xt, 128)


def world(kb, seed):
    A = A0[:, :, :kb] - A0[:, :, :kb].mean(1, keepdims=True)
    T = T0[:, :, :kb]
    P = np.ascontiguousarray(P0[:, :, :kb])
    R, _ = H.fit_maps(A)
    mdl = H.model_O(A, T, R)
    return H.surrogate(Xf, Xt, g0, P, mdl, seed=seed, ambient_out=True)


for v in which.split(","):
    if v == "V0":
        Df, Dt = world(64, 77)
        run.run_dataset(Df, Dt, ntok, [64], 256, "V0", LAYER, t0=t0)
    elif v in ("V2", "V1"):
        Df, Dt = world(128, 78)
        extra = None
        if v == "V1":
            rs = np.random.RandomState(5)
            lam = (A0[:, :, :64] ** 2).mean(1)
            Fc = np.concatenate([P0[i, :, :64].astype(np.float64) * np.sqrt(lam[i])[None] for i in range(12)], 1)
            w, U = np.linalg.eigh(Fc.T @ Fc)
            C = Fc @ U[:, ::-1][:, :64] / np.sqrt(w[::-1][:64])
            Sst = H.orth(C @ rs.randn(64, 4)).astype(np.float32)            # planted ambient subspace
            groups = {7: 0, 8: 0, 0: 1, 1: 1, 2: 1, 3: 1}
            Nf = {g: rs.randn(Df.shape[1], 4).astype(np.float32) for g in (0, 1)}
            Nt = {g: rs.randn(Dt.shape[1], 4).astype(np.float32) for g in (0, 1)}
            Df = Df.astype(np.float32)
            Dt = Dt.astype(np.float32)
            for i, g in groups.items():
                amp = float(np.sqrt(lam[i, 10]))
                Df[i] += amp * Nf[g] @ Sst.T
                Dt[i] += amp * Nt[g] @ Sst.T
            Df, Dt = Df.astype(np.float16), Dt.astype(np.float16)
            extra = {"planted": Sst}
        run.run_dataset(Df, Dt, ntok, [64], 256, v, LAYER, extra_refs=extra, t0=t0)
    del Df, Dt
print(f"done [{time.time()-t0:.0f}s]")
