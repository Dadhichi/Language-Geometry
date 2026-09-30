"""Validation of analysis34.tests on planted worlds (run before the real data exist).
Worlds (Brownian motion in `dim` dims): leaf ('star') steps with lognormal lengths + one high-fertility outlier,
plus script-clade steps (nuisance, always present), plus (a) nothing, (b) the Glottolog tree with internal share c,
(c) the OV split with share c.  Reports rejection rates at the pre-registered alpha for H1 and H2."""
import sys
import numpy as np
import lib34 as T
import analysis34 as A

sp = T.Space()
N = T.N


def world(kind, c, rs, dim=32):
    leaf = np.exp(rs.normal(0, 0.35, N))
    leaf[T.IX["hin"]] *= 3
    X = rs.normal(0, 1, (N, dim)) * np.sqrt(leaf)[:, None]
    for S in T.SCRIPT_SPLITS:                                       # nuisance script structure
        X[list(S)] += rs.normal(0, np.sqrt(0.15 * leaf.mean()), dim)
    splits = {"null": [], "glotto": T.GLOTTO, "ov": [T.OV]}[kind]
    for S in splits:
        X[list(S)] += rs.normal(0, np.sqrt(c * leaf.mean()), dim)
    noise = lambda: rs.normal(0, 0.05, (N, dim))
    Xd, Xt = X + noise(), X + noise()                                # dev / devtest estimates
    Xd -= Xd.mean(0); Xt -= Xt.mean(0)
    return Xd @ Xt.T


def tokstats(rs):
    H = np.clip(0.1 + 0.05 * rs.random((N, N)), 0, 1)
    H = (H + H.T) / 2
    for S in T.SCRIPT_SPLITS:
        for i in S:
            for j in S:
                H[i, j] += 0.2
    np.fill_diagonal(H, 1)
    fert = np.exp(rs.normal(3.5, 0.3, N))
    return {"tok_hist_int": H, "fert": fert}


def main(reps=40, n_perm=300):
    rs = np.random.RandomState(0)
    tok = tokstats(rs)
    base, base_ns = A.bases(sp, tok)
    for kind, c in (("null", 0.0), ("glotto", 0.1), ("ov", 0.1), ("glotto", 0.2), ("ov", 0.2)):
        r1 = r2 = 0
        for k in range(reps):
            K = world(kind, c, rs)
            out = A.tests(sp, K / np.trace(K), base, base_ns, n_perm, seed=k, full=False)
            r1 += out["H1_p"] < 0.05
            r2 += out["H2_p"] < 0.05
        print(f"{kind:6s} c={c:.1f}: reject H1 {r1 / reps:.2f}  H2 {r2 / reps:.2f}  (alpha .05, {n_perm} perms)", flush=True)


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
