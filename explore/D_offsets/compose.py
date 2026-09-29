"""Does the known non-composition of per-language Procrustes maps follow from an AMBIENT IDENTITY gauge seen
through per-language truncation?  For each map family (fitted Procrustes; polar(B_i^T B_j) = the ambient
identity made orthogonal; fitted ridge) compute held-out rho, P@1 and composition inconsistency
c = ||T_i (R_ij R_jm - R_im)||^2 / ||T_i||^2 (fit.py definition), on
  real     : the real layer
  shift    : synthetic, ambient identity gauge: X = m + a_i + b_s + e_i,pi(s)  (real b, real e shuffled
             within language -> language-specific covariance, no content link).  Exactly consistent in ambient.
  gauge    : synthetic, per-language rotation of the top-64 content PCs (consistent in ambient too).
Usage: python compose.py 14 4 20"""
import os, sys, json, itertools, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlib

L = 12


def build(split, layer, scenario, Pc=None):
    X = dlib.load(split, layer)
    if scenario == "real":
        return X
    A = dlib.Anova(X)
    n = X.shape[1]
    b32, m32, a32 = A.b.astype(np.float32), A.m.astype(np.float32), A.a.astype(np.float32)
    r = np.random.RandomState(100)
    Qs = []
    for i in range(L):
        K = r.randn(64, 64); K = (K - K.T) / np.sqrt(2 * 64)
        Qs.append(expm(0.8 * K).astype(np.float32))
    pr_ = np.random.RandomState(5 if split == "dev" else 6)
    for i in range(L):
        e = A.e(i)[pr_.permutation(n)]
        cont = b32 if scenario == "shift" else b32 + ((b32 @ Pc) @ (Qs[i] - np.eye(64, dtype=np.float32))) @ Pc.T
        X[i] = m32 + a32[i] + cont + e          # real language offsets, real content, shuffled real residual
    return X


def run(Xd, Xt, ks=(64, 256)):
    mu = Xd.mean(1, keepdims=True)
    Xd -= mu; Xt -= mu
    out = {}
    nt = Xt.shape[1]
    ar = np.arange(nt)
    for k in ks:
        B = [dlib.top_basis_rows(Xd[i], k)[1] for i in range(L)]
        Zd = [Xd[i] @ B[i] for i in range(L)]
        Zt = [Xt[i] @ B[i] for i in range(L)]
        maps = {"proc": {}, "polar_id": {}, "ridge": {}, "shift_trunc": {}}
        for i, j in itertools.permutations(range(L), 2):
            M = B[i].T @ B[j]
            maps["shift_trunc"][i, j] = M
            maps["polar_id"][i, j] = dlib.procrustes(np.eye(k, dtype=np.float32), M)
            maps["proc"][i, j] = dlib.procrustes(Zd[i], Zd[j])
            maps["ridge"][i, j] = dlib.ridge_fit(Zd[i], Zd[j], 1e-3)
        for nm, R in maps.items():
            rho, p1, c = [], [], []
            for i in range(L):
                Ti = Zt[i]
                ei = dlib.sq(Ti)
                P = {j: Ti @ R[i, j] for j in range(L) if j != i}
                for j in P:
                    rho.append(dlib.sq(P[j] - Zt[j]) / dlib.sq(Zt[j]))
                    tn = Zt[j] / np.linalg.norm(Zt[j], axis=1, keepdims=True)
                    p1.append(float(((P[j] @ tn.T).argmax(1) == ar).mean()))
                for j, m in itertools.permutations([q for q in range(L) if q != i], 2):
                    c.append(dlib.sq(P[j] @ R[j, m] - P[m]) / ei)
            out[f"{nm}_k{k}"] = dict(rho=float(np.mean(rho)), p1=float(np.mean(p1)), c=float(np.mean(c)))
        # how far the fitted rotation is from the ambient identity
        out[f"proc_vs_identity_k{k}"] = float(np.mean([np.trace(maps["proc"][p].T @ maps["polar_id"][p]) / k
                                                      for p in maps["proc"]]))
        # per-language subspace agreement between languages (mean cos^2 of principal angles)
        out[f"cross_lang_subspace_k{k}"] = float(np.mean([(np.linalg.svd(B[i].T @ B[j], compute_uv=False) ** 2).mean()
                                                         for i, j in itertools.combinations(range(L), 2)]))
        del B, Zd, Zt, maps
    return out


if __name__ == "__main__":
    layers = [int(x) for x in sys.argv[1:]] or [14]
    path = os.path.join(dlib.OUT, "compose.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for l in layers:
        Xtmp = dlib.load("dev", l)
        Pc = dlib.top_basis_rows(dlib.Anova(Xtmp).b, 64)[1][:, :64]
        del Xtmp
        for scen in ("real", "shift", "gauge"):
            t0 = time.time()
            Xd, Xt = build("dev", l, scen, Pc), build("devtest", l, scen, Pc)
            o = run(Xd, Xt)
            del Xd, Xt
            res[f"L{l}_{scen}"] = o
            print(f"L{l} {scen} ({time.time() - t0:.0f}s): " + " | ".join(
                f"{k}: " + (f"{v:.3f}" if isinstance(v, float) else ",".join(f"{a}={b:.3f}" for a, b in v.items()))
                for k, v in o.items()), flush=True)
            json.dump(res, open(path, "w"), indent=1)
