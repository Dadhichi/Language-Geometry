"""Robustness: leave-source-out instruments (remove the sentence's content-driven propensity to share evidence with
target j, estimated from the OTHER sources' renderings), all layers, raw + sqrtn."""
import sys, os, json
import numpy as np
import flib as F
import pipeline as P

layers = [int(x) for x in sys.argv[1].split(",")]
variants = sys.argv[2].split(",")
Idev, Itest = F.load_instr("dev"), F.load_instr("devtest")
res = {}
for layer in layers:
    for variant in variants:
        z = F.load_layer(layer, variant)
        Zd, Zt = z["Z_dev"][..., :11].astype(np.float64), z["Z_test"][..., :11].astype(np.float64)
        ctx = P.Ctx(Idev, Itest, Zd.mean(1), extra=True)
        out = P.run(ctx, Zd, Zt, sets=(("e_tokc_X",), ("B_bag_X",), ("B_pre_X",), ("B_bag_X", "B_pre_X")),
                    nperm=40, seed=layer)
        res[f"L{layer}_{variant}"] = {k: {kk: np.asarray(v).tolist() for kk, v in r.items()} for k, r in out.items()}
        print(f"L{layer} {variant}\n" + P.fmt(out), flush=True)
        json.dump(res, open(os.path.join(F.HERE, f"robust_{'_'.join(map(str, layers))}.json"), "w"), indent=1)
