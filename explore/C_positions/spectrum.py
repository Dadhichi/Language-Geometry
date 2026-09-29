"""Post-hoc, noise-debiased rotation-signal statistics (same worlds as run_layer.py, rebuilt from the stored
calibration).  For each world and m:
  * pooled signal spectrum: M = sum_pairs sym(L^a_p^T L^b_p) from independent half-sample logs; its
    eigenvalues are the plane-usage of the reproducible rotation signal (noise cancels in expectation).
    Report the fraction in the top 8/16/32 dims (4/8/16 planes).
  * split-half commutator normalised by a random rotation WITHIN the pooled signal subspace (top-32 dims),
    so a low-rank signal is compared with a random low-rank signal of the same support: 0 = commuting.
    python spectrum.py --layers 2,14,26
"""
import argparse
import json

import numpy as np

from common import NL, OUT, PAIRS, f32, load16, log_orth, pooled_basis, procrustes, project
from run_layer import gen_Q, make_world


def stats(Zf, m, seed=1):
    n = Zf.shape[1]
    rs = np.random.RandomState(seed)
    perm = rs.permutation(n)
    ha, hb = perm[: n // 2], perm[n // 2:]
    La, Lb = {}, {}
    for i, j in PAIRS:
        La[(i, j)] = log_orth(procrustes(Zf[i][ha], Zf[j][ha]))[0]
        Lb[(i, j)] = log_orth(procrustes(Zf[i][hb], Zf[j][hb]))[0]
        La[(j, i)], Lb[(j, i)] = -La[(i, j)], -Lb[(i, j)]
    M = sum(0.5 * (La[p].T @ Lb[p] + Lb[p].T @ La[p]) for p in PAIRS)
    ev, V = np.linalg.eigh(M)
    ev, V = ev[::-1], V[:, ::-1]
    tot = ev.sum()
    out = {f"pooled_top{k}": float(ev[:k].sum() / tot) for k in (8, 16, 32)}
    out["signal_trace"] = float(tot / len(PAIRS))
    # per-pair aggregated top-8 fraction (ratio of sums)
    t8, tt = 0.0, 0.0
    for p in PAIRS:
        e = np.linalg.eigvalsh(0.5 * (La[p].T @ Lb[p] + Lb[p].T @ La[p]))[::-1]
        t8 += e[:8].sum()
        tt += e.sum()
    out["pair_top8"] = float(t8 / tt)
    Vs = V[:, :32]
    Os = []
    for _ in range(3):
        Q = np.linalg.qr(rs.randn(32, 32))[0]
        Os.append(Vs @ Q @ Vs.T + np.eye(m) - Vs @ Vs.T)
    comm = lambda X, Y: X @ Y - Y @ X
    num, den = 0.0, 0.0
    for a in range(NL):
        others = [j for j in range(NL) if j != a]
        for jj, j in enumerate(others):
            for k in others[jj + 1:]:
                num += (comm(La[(a, j)], La[(a, k)]) * comm(Lb[(a, j)], Lb[(a, k)])).sum()
                den += np.mean([(comm(La[(a, j)], O @ La[(a, k)] @ O.T) * comm(Lb[(a, j)], O @ Lb[(a, k)] @ O.T)).sum()
                                for O in Os])
    out["sh_comm_sub"] = float(num / den)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="2,14,26")
    args = ap.parse_args()
    res = {}
    for layer in [int(x) for x in args.layers.split(",")]:
        R = json.load(open(f"{OUT}/layer{layer}.json"))
        Xf16, Xt16 = load16("dev", layer), load16("devtest", layer)[:, :8].copy()
        mu, U, _ = pooled_basis(Xf16, 128)
        Sb = np.ascontiguousarray(U[:, :32])
        res[layer] = {}
        for world in ("real", "identity", "abelian", "nonabelian"):
            if world == "real":
                Wf = Xf16
            else:
                Qs, _ = gen_Q(world, R[world]["scale"], 32)
                Wf, Wt = make_world(Xf16, Xt16[:, :8], mu, Sb, Qs, R["calib"]["noise_s"], seed=11)
                del Wt
            muw, Uw, _ = pooled_basis(Wf, 128)
            res[layer][world] = {}
            for m in (64, 128):
                o = stats(project(Wf, muw, Uw[:, :m]), m)
                res[layer][world][m] = o
                print(f"L{layer} {world:10s} m={m}: " + " ".join(f"{k}={v:.3f}" for k, v in o.items()), flush=True)
            del Wf
        with open(f"{OUT}/spectrum_{args.layers.replace(',', '_')}.json", "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
