"""Follow-up checks:
(1) head-final split {hin,jpn,tur}: exact rank among all 220 3-subsets, bases star+tok+fert and +script;
    also {deu,hin,jpn,tur} among 495 4-subsets.
(2) Hindi-excluded glotto gain: does it survive adding the head-final split {jpn,tur} (+hin, masked) to the base?
(3) upper bound on the genealogical share f: P(synthetic gain <= observed gain) under planted glotto worlds."""
import os, json
import numpy as np
import treelib as T
import analyze as A
import synth as S

HERE = os.path.dirname(os.path.abspath(__file__))
SOV = (5, 8, 9)
out = {"layers": {}}
for layer in (4, 8, 14, 20, 24):
    z = np.load(os.path.join(HERE, "grams", f"L{layer}.npz"))
    for m in A.METRICS:
        K = z[f"{m}_dt"]
        y = T.d2_from_gram(K)[T.IU]
        r = {}
        for bn in ("star+tok+fert", "star+tok+fert+script"):
            base = A.BASES[bn]
            r[f"SOV_p|{bn}"] = T.single_split_rank(y, base, SOV)[1]
            r[f"OV4_p|{bn}"] = T.single_split_rank(y, base, (1, 5, 8, 9))[1]
        # Hindi excluded, with and without a head-final covariate (jpn,tur split) in the base
        keep = A.KEEP_NOHIN
        yk = y[keep]
        rs = np.random.RandomState(layer)
        others = np.array([i for i in range(T.L) if i != 5])
        for tag, extra in (("plain", []), ("+OV(jpn,tur)", [T.split_col((8, 9))]),
                           ("+OV+script", [T.split_col((8, 9))] + [T.cov_cols({"s": A.COV["script"]})[:, 0]])):
            base = np.delete(np.column_stack([A.BASES["star+tok+fert"]] + extra)[keep], 5, axis=1)
            s0 = T.sse_nnls(base, yk)[0]

            def g_(splits):
                Xs = np.column_stack([base] + [T.split_col(Sx)[keep] for Sx in splits])
                return 1 - T.sse_nnls(Xs, yk)[0] / s0
            obs = g_(T.TREES["glotto"])
            null = []
            for _ in range(1000):
                p = np.arange(T.L); p[others] = rs.permutation(others)
                null.append(g_(T.relabel(T.TREES["glotto"], p)))
            r[f"noHin_gain|{tag}"] = obs
            r[f"noHin_p|{tag}"] = float((1 + (np.array(null) >= obs - 1e-12).sum()) / 1001)
            # which split carries it: gain of IE-minus-hin alone
            r[f"noHin_IEonly_gain|{tag}"] = g_([T.TREES["glotto"][0]])
        out["layers"][f"L{layer}_{m}"] = r
        print(f"L{layer} {m}: " + " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)

# (3) f upper bound from gain distributions (no permutations needed)
obs = {r["metric"] + str(r["layer"]): r["glotto_gain|star+tok+fert"] for r in json.load(open(os.path.join(HERE, "results.json")))}
for dim in (16, 50):
    for m, layer in (("euc", 14), ("lda01", 14), ("causal", 14), ("euc", 8), ("lda01", 8)):
        leaf, spec, bdev, trK = S.setup(m, layer)
        rs = np.random.RandomState(99)
        line = []
        for f in (0.0, 0.02, 0.05, 0.1, 0.2):
            g = []
            for rep in range(300):
                X = S.planted("glotto", f, leaf, spec, rs, dim)
                Kx = X @ X.T
                yx = T.d2_from_gram(Kx)[T.IU]
                g.append(T.gain(yx, A.BASES["star+tok+fert"], T.TREES["glotto"])[0])
            g = np.array(g)
            line.append((f, float(np.mean(g <= obs[m + str(layer)]))))
        out[f"fbound_{m}_L{layer}_d{dim}"] = line
        print(f"dim {dim} {m} L{layer} obs gain {obs[m + str(layer)]:.3f}  P(gain<=obs | f): " +
              " ".join(f"f={f}:{p:.2f}" for f, p in line), flush=True)
json.dump(out, open(os.path.join(HERE, "checks.json"), "w"), indent=1)
