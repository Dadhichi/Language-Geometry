"""Belief (cumulative prefix) vs position-weighted bag attribution with an EXTENDED generic-control set
(own-weight deficits of every candidate observer: bag, pre, early, first, late), synthetic worlds + real layers."""
import sys
import numpy as np
import flib as F
import pipeline as P

COVS = ("logn", "shared", "ownbag", "ownpre", "digit", "punct", "ownearly", "ownfirst", "ownlate")
_orig = F.covariates


def cov_ext(I, names=COVS):
    I = dict(I)
    for nm in ("early", "first", "late"):
        I["own" + nm] = np.stack([I["B_" + nm][i, i] for i in range(12)])
    return _orig(I, names)


F.covariates = cov_ext
Idev, Itest = F.load_instr("dev"), F.load_instr("devtest")
SETS = (("B_pre",), ("B_bag", "B_pre"), ("B_early", "B_late", "B_pre"), ("B_first", "B_pre"))
for layer in [int(x) for x in sys.argv[1].split(",")]:
    z = F.load_layer(layer, "raw")
    Zd, Zt = z["Z_dev"][..., :11].astype(np.float64), z["Z_test"][..., :11].astype(np.float64)
    A = Zd.mean(1)
    ctx = P.Ctx(Idev, Itest, A)
    print(f"=== L{layer} real\n" + P.fmt(P.run(ctx, Zd, Zt, sets=SETS, nperm=0)), flush=True)
    if layer in (4, 14, 24):
        rng = np.random.RandomState(1)
        noise = {"dev": F.shuffle_rows(F.centre(Zd), rng), "test": F.shuffle_rows(F.centre(Zt), rng)}
        I = {"dev": Idev, "test": Itest}
        for nm in ("B_early", "B_pre", "B_bag"):
            sig = {sp: np.einsum("ijs,jk->isk", I[sp][nm].astype(np.float64), A) - A[:, None, :] for sp in I}
            out = P.run(ctx, noise["dev"] + sig["dev"], noise["test"] + sig["test"], sets=SETS, nperm=0)
            print(f"--- L{layer} synth world {nm}(1.0)\n" + P.fmt(out), flush=True)
