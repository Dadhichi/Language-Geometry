"""Synthetic validation: plant known belief-type (vertex-directed) pulls vs generic shrinkage on top of REAL-spread
noise (real in-span displacements, sentence-shuffled within language so they carry no link to the text), run the
identical pipeline.  Worlds (A = real dev centroids, W_{i,s} = sum_j e_{ij,s}):
  null       d = noise
  S_e(g0)    o = (1 - g0 W) a_i + g0 sum_j e_ij a_j + noise           (prompt's convex-combination world)
  G_e(g0)    same total weight g0 W but spread uniformly over all other vertices (= shrinkage to the centre)
  G_nl       generic shrinkage, nonlinear in length and evidence: d = -k (exp(-ntok/25) + 3 W^2) a_i + noise
  S_bag/S_pre(g0)  o = sum_k B_k a_k (ideal-observer belief as barycentric weights, gain g0) + noise
"""
import sys, json
import numpy as np
import flib as F
import pipeline as P

layers = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [14]
Idev, Itest = F.load_instr("dev"), F.load_instr("devtest")
res = {}
for layer in layers:
    z = F.load_layer(layer, "raw")
    Zd, Zt = z["Z_dev"][..., :11].astype(np.float64), z["Z_test"][..., :11].astype(np.float64)
    A = Zd.mean(1)
    ctx = P.Ctx(Idev, Itest, A)
    rng = np.random.RandomState(1)
    noise = {"dev": F.shuffle_rows(F.centre(Zd), rng), "test": F.shuffle_rows(F.centre(Zt), rng)}
    I = {"dev": Idev, "test": Itest}
    sd_edge = np.sqrt(np.mean([np.sum((A[i] - A[j]) ** 2) for i in range(12) for j in range(i)]))
    sd_noise = np.sqrt(np.mean(np.sum(noise["test"] ** 2, 2)))
    print(f"=== L{layer}: rms edge length {sd_edge:.3f}, rms in-span displacement {sd_noise:.3f}", flush=True)

    def conv(W8, g0):          # o = sum_k w_k a_k with w = (1-g0*sum_j W_ij) e_i + g0 W_ij ; return d = o - a_i
        out = {}
        for sp in W8:
            W = W8[sp].astype(np.float64).copy()
            for i in range(12):
                W[i, i] = 0
            out[sp] = g0 * (np.einsum("ijs,jk->isk", W, A) - W.sum(1)[..., None] * A[:, None, :])
        return out

    def unif(W8, g0):
        out = {}
        for sp in W8:
            W = W8[sp].astype(np.float64).copy()
            for i in range(12):
                W[i, i] = 0
            tot = W.sum(1)                                         # [12,n]
            out[sp] = -g0 * (12 / 11) * tot[..., None] * A[:, None, :]
        return out

    def belief(nm, g0):        # barycentric weights = (1-g0) e_i + g0 B
        out = {}
        for sp in I:
            B = I[sp][nm].astype(np.float64)
            out[sp] = g0 * (np.einsum("ijs,jk->isk", B, A) - A[:, None, :])
        return out

    E = {sp: I[sp]["e_tokc"] for sp in I}
    worlds = {"null": {sp: 0 for sp in I}}
    for g0 in (0.25, 1.0):
        worlds[f"S_e({g0})"] = conv(E, g0)
        worlds[f"G_e({g0})"] = unif(E, g0)
    k = 0.3
    worlds["G_nl"] = {sp: -k * (np.exp(-I[sp]["ntok"] / 25.0) + 3 * (E[sp].sum(1) / 11) ** 2)[..., None]
                      * A[:, None, :] for sp in I}
    for g0 in (0.25, 1.0):
        worlds[f"S_bag({g0})"] = belief("B_bag", g0)
        worlds[f"S_pre({g0})"] = belief("B_pre", g0)
    worlds["S_early(1.0)"] = belief("B_early", 1.0)
    worlds["S_LL(1.0)"] = {sp: 1.0 * F.dhat(ctx.xt[sp]["e_tokc_LL"], A) for sp in I}
    for wn, sig in worlds.items():
        dd = noise["dev"] + sig["dev"]
        dt = noise["test"] + sig["test"]
        amp = np.sqrt(np.mean(np.sum(F.centre(dt - noise["test"]) ** 2, 2))) if not isinstance(sig["test"], int) else 0
        out = P.run(ctx, dd, dt, nperm=25, seed=2)
        print(f"-- world {wn}  (rms planted {amp:.4f} vs noise {sd_noise:.3f})\n" + P.fmt(out), flush=True)
        res[f"L{layer}_{wn}"] = {k_: {kk: np.asarray(v).tolist() for kk, v in r.items()} for k_, r in out.items()}
json.dump(res, open(F.os.path.join(F.HERE, f"synth_{'_'.join(map(str, layers))}.json"), "w"), indent=1)
