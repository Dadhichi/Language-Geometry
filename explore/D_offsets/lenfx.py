"""Length / fertility effect on the per-sentence offset.
Within each language: slope of log ||e_{i,s}||^2 on log ntok_{i,s} (slope -1 = pure averaging of independent
token-level noise under mean pooling), and the same for the content-aligned part.  Also: does the constant
language offset a_i itself scale with 1/ntok (a 'sink-like' or position-dependent residue)?  We test the
latter by splitting each language's sentences into short/long halves and comparing the two half-centroids."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlib

ntok = np.load(os.path.join(dlib.DATA, "ntok_dev.npy"))
out = {}
for l in [int(x) for x in sys.argv[1:]] or [0, 8, 14, 20, 28]:
    X = dlib.load("dev", l)
    A = dlib.Anova(X)
    L, n, d = X.shape
    rec = {}
    for i in range(L):
        e = A.e(i)
        le = np.log((e.astype(np.float64) ** 2).sum(1))
        lt = np.log(ntok[i])
        slope = np.polyfit(lt, le, 1)[0]
        # short vs long half: language offset estimated from each half, relative to the SAME sentences' content
        order = np.argsort(ntok[i])
        sh, lo = order[: n // 2], order[n // 2:]
        a_sh = X[i][sh].mean(0, dtype=np.float64) - X[:, sh].mean((0, 1), dtype=np.float64)
        a_lo = X[i][lo].mean(0, dtype=np.float64) - X[:, lo].mean((0, 1), dtype=np.float64)
        # the other languages' sentences sorted by language i's length are the same sentences (parallel)
        rec[dlib.LANGS[i]] = dict(slope_loge2_logntok=float(slope),
                                  offset_norm_ratio_long_over_short=float(np.sqrt((a_lo ** 2).sum() / (a_sh ** 2).sum())),
                                  offset_cos_long_short=float(a_lo @ a_sh / np.sqrt((a_lo ** 2).sum() * (a_sh ** 2).sum())),
                                  mean_ntok_short=float(ntok[i][sh].mean()), mean_ntok_long=float(ntok[i][lo].mean()))
    out[l] = rec
    sl = np.array([r["slope_loge2_logntok"] for r in rec.values()])
    ra = np.array([r["offset_norm_ratio_long_over_short"] for r in rec.values()])
    co = np.array([r["offset_cos_long_short"] for r in rec.values()])
    print(f"L{l}: slope log||e||^2~log ntok: mean {sl.mean():.2f} [{sl.min():.2f},{sl.max():.2f}] | "
          f"||a_long||/||a_short|| mean {ra.mean():.3f} [{ra.min():.3f},{ra.max():.3f}] | cos(a_long,a_short) mean {co.mean():.3f} min {co.min():.3f}",
          flush=True)
    del X, A
json.dump(out, open(os.path.join(dlib.OUT, "lenfx.json"), "w"), indent=1)
