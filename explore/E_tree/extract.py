"""Per layer: language centroid offsets a_i = mu_i - m (dev, devtest, sentence-bootstraps), and their 12x12 Gram
matrices under four metrics (everything downstream is a function of these Grams):
  euc     : <u,v> = u.v
  lda01   : u^T W_a^-1 v, W = pooled within-language covariance (dev), shrinkage a=0.1 toward (trW/d) I
  lda05   : same, a=0.5
  causal  : u^T diag(g) Cov(gamma) diag(g) v   (Park et al. causal inner product, logit-lens proxy)
Cross-split Gram K_dt = A_dev G A_test^T is unbiased for the noiseless Gram (independent sentences).
Bootstraps: B resamples of sentences (same indices across languages, since sentences are parallel), dev and test
independently.   usage: python extract.py 14 0 2 ...  (layers)"""
import os, sys, time
import numpy as np

DATA = r"C:\Users\ASUS\Documents\lang-geom"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grams")
os.makedirs(OUT, exist_ok=True)
L, d, B = 12, 3584, 100


def load(split, layer):
    X = np.load(os.path.join(DATA, f"mean_{split}_L{layer}.f16.npy"), mmap_mode="r")
    return np.asarray(X, dtype=np.float32)


def centroids(X, rs, want_W):
    n = X.shape[1]
    mu = X.mean(1, dtype=np.float64)
    A = mu - mu.mean(0)
    cnt = rs.multinomial(n, np.ones(n) / n, size=B).astype(np.float32) / n     # [B, n]
    Ab = np.stack([cnt @ X[i] for i in range(L)], 1).astype(np.float64)       # [B, L, d]
    Ab -= Ab.mean(1, keepdims=True)
    W = None
    if want_W:
        W = np.zeros((d, d))
        for i in range(L):
            Y = X[i] - mu[i].astype(np.float32)
            W += (Y.T @ Y).astype(np.float64)
            del Y
        W /= L * (n - 1)
    return A, Ab, W


def main(layers):
    Craw = np.load(os.path.join(DATA, "covgamma_norm.npy"))
    for layer in layers:
        t0 = time.time()
        rs = np.random.RandomState(1000 + layer)
        X = load("dev", layer)
        Ad, Adb, W = centroids(X, rs, True)
        del X
        X = load("devtest", layer)
        At, Atb, _ = centroids(X, rs, False)
        del X
        w, V = np.linalg.eigh(W)
        w = np.clip(w, 0, None)
        wbar = w.mean()
        stack = np.concatenate([Ad[None], At[None], Adb, Atb], 0)                  # [2+2B, L, d]
        S = stack.reshape(-1, d)
        res = {}

        def grams(T, name):
            Z = T.reshape(2 + 2 * B, L, -1)
            zd, zt, zdb, ztb = Z[0], Z[1], Z[2:2 + B], Z[2 + B:]
            res[f"{name}_dd"] = zd @ zd.T
            res[f"{name}_tt"] = zt @ zt.T
            res[f"{name}_dt"] = zd @ zt.T
            res[f"{name}_bdt"] = np.einsum("bid,bjd->bij", zdb, ztb)
            res[f"{name}_bdd"] = np.einsum("bid,bjd->bij", zdb, zdb)
        grams(S, "euc")
        SV = S @ V
        for a, nm in ((0.1, "lda01"), (0.5, "lda05")):
            grams(SV / np.sqrt((1 - a) * w + a * wbar)[None], nm)
        del SV
        # causal: T = C^{1/2}; Gram = A C A^T -> use Cholesky-free route: Z = S @ C, K = Z S^T
        SC = S @ Craw
        Z = SC.reshape(2 + 2 * B, L, d)
        St = stack
        res["causal_dd"] = Z[0] @ St[0].T
        res["causal_tt"] = Z[1] @ St[1].T
        res["causal_dt"] = Z[0] @ St[1].T
        res["causal_bdt"] = np.einsum("bid,bjd->bij", Z[2:2 + B], St[2 + B:])
        res["causal_bdd"] = np.einsum("bid,bjd->bij", Z[2:2 + B], St[2:2 + B])
        del SC, Z
        res["W_eig"] = w[::-1].astype(np.float32)
        np.savez(os.path.join(OUT, f"L{layer}.npz"), **res)
        del W, V, S, stack, Adb, Atb
        print(f"L{layer} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
