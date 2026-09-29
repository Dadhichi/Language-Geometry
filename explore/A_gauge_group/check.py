"""Sanity check: my Gram-based pipeline reproduces fit.py numbers at L14 (summary.csv)."""
import time
import numpy as np
from lib import load, project, grams, evaluate, null_data

t0 = time.time()
layer = 14
Xf, Xt = load("dev", layer), load("devtest", layer)
print("loaded", time.time() - t0)
for k in (64,):
    Zf, Zt, bases = project(Xf, Xt, k)
    Gf, Gt = grams(Zf, Zt)
    print("proj", time.time() - t0)
    o, _ = evaluate(Gf, Gt, keep_sv=True)
    print("eval", time.time() - t0)
    print({a: round(b, 4) for a, b in o.items()})
    Xfn, Xtn, infl = null_data(Xf, Xt, Zf, Zt, bases)
    Zfn, Ztn, _ = project(Xfn, Xtn, k)
    del Xfn, Xtn
    Gfn, Gtn = grams(Zfn, Ztn)
    o, _ = evaluate(Gfn, Gtn, keep_sv=True)
    print("null infl", infl, time.time() - t0)
    print({a: round(b, 4) for a, b in o.items()})
# expected (summary.csv L14 k64): proc .3965 ridge .2728 sync .3995 c_proc .0771 c_ridge .0297 ; null c_proc .0165 c_ridge .028
