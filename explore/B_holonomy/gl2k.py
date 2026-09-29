"""gl2k.py LAYER -- flat LINEAR gauge fitted in kb=2k dims, lifted, re-PCA'd and analysed at k.
Does 'flat GL in 2k + per-language truncation' reproduce BOTH the magnitude and WHICH triangles are curved?"""
import sys, json, time, numpy as np, holo as H
from summarize import tri_c, design, cv_r2, fit_coef
layer = int(sys.argv[1]) if len(sys.argv) > 1 else 14
t0 = time.time()
Xf, Xt = H.load("dev", layer), H.load("devtest", layer)
A0, T0, P0, g0 = H.project(Xf, Xt, 256)
res = {}
import os
for k, kb in ((64, 128), (128, 256)):
    if not os.path.exists(f"out/L{layer}_k{k}_dev.npz"):
        continue
    z = np.load(f"out/L{layer}_k{k}_dev.npz")
    cr = tri_c(z["real_c_proc"])
    A = A0[:, :, :kb] - A0[:, :, :kb].mean(1, keepdims=True)
    mdl = H.model_GL(A, T0[:, :, :kb])
    An, Tn, Pn, _ = H.surrogate(Xf, Xt, g0, np.ascontiguousarray(P0[:, :, :kb]), mdl, seed=321 + kb, k=k)
    st = H.analyze_conn(An, Tn, seed=3, with_comm=False)
    cn = tri_c(st["c_proc"])
    y = cr - cn
    Xv = design("vertex")
    bv = fit_coef(Xv, y)
    ang = st["angles"]; th2 = ang ** 2
    res[f"GL{kb}->k{k}"] = dict(c=float(np.nanmean(st["c_proc"])), c_real=float(np.nanmean(z["real_c_proc"])),
        tri_corr=float(np.corrcoef(cr, cn)[0, 1]), excess_mean=float(y.mean()), r2cv_vertex=float(cv_r2(Xv, y)),
        vertex_eff={H.LANGS[i]: round(float(bv[1 + i]), 4) for i in np.argsort(-bv[1:])},
        infl=mdl["infl"], theta_max_deg=float(np.degrees(ang[:, 0].mean())),
        pr_planes=float((th2.sum(1) ** 2 / (th2 ** 2).sum(1) / 2).mean()),
        c_ridge=float(np.nanmean(st["c_ridge"])))
    print(k, kb, res[f"GL{kb}->k{k}"], f"[{time.time()-t0:.0f}s]", flush=True)
json.dump(res, open(f"out/gl2k_L{layer}.json", "w"), indent=1)
