"""Real data vs synthetic gauge worlds, all pushed through the identical per-language-PCA pipeline.

    python run.py --layer 14 --kinds real,R2iso,R2shuf,R1,Rd --ks 64,128 [--tune_k 64]

Synthetic worlds (hub H = real eng_Latn cloud at this layer, fit=dev n=997, test=devtest n=1012):
  R2iso  : X_i = H + sigma * isotropic Gaussian noise (ambient d=3584)      -- exact orthogonal gauge (identity;
           an ambient rotation Q_i is a no-op for every per-language-PCA, basis-invariant statistic)
  R2shuf : X_i = H + beta * shuffle(real language i cloud)                     -- exact orthogonal gauge,
           language-specific anisotropic noise with a realistic spectrum
  R1     : X_i = H S_i + beta * shuffle(real i); S_i = I + V (exp(eps D_i) - I) V^T, V = 32 random directions
           inside the hub's top-128 PCA subspace, D_i ~ N(0,1) diag                -- true SPD (GL) gauge
  Rd     : X_i = H + gamma * Z_script(i) + beta * shuffle(real i): script-shared latent (Latin 7 langs, Han 2)
           that is NOT a function of the hub                                           -- no single hub (rung d)
The noise level (sigma / beta) is tuned so that procrustes rho at --tune_k matches the real value.
Every dataset also gets the fit.py-style consistent null (GPA hub + permuted residuals) through the same code.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import lib
from lib import load, project, grams, evaluate, null_data, hub_ladder, rho_pairs, maps_family

SCRIPT = [0, 0, 0, 0, 1, 2, 3, 4, 4, 0, 0, 0]     # Latin, Cyrl, Deva, Arab, Han(zho,jpn)


def real_rho_proc(layer, k):
    s = pd.read_csv(os.path.join(lib.REPO, "results", "qwen25_7b_mean", "summary.csv"))
    s = s[(s.layer == layer) & (s.k.astype(str) == str(k)) & (s.split == "devtest")]
    return float(s.rho_procrustes.iloc[0])


def make(kind, level, layer, seed=0, eps=0.5, gamma=0.5):
    rs = np.random.RandomState(seed)
    Rf = np.load(os.path.join(lib.DATA, f"mean_dev_L{layer}.f16.npy"), mmap_mode="r")
    Rt = np.load(os.path.join(lib.DATA, f"mean_devtest_L{layer}.f16.npy"), mmap_mode="r")
    Hf, Ht = np.asarray(Rf[0], np.float32), np.asarray(Rt[0], np.float32)
    Ln, n, d = Rf.shape
    m = Rt.shape[1]
    Xf = np.empty((Ln, n, d), np.float32)
    Xt = np.empty((Ln, m, d), np.float32)
    scale_iso = np.sqrt(((Hf - Hf.mean(0)) ** 2).sum(1).mean() / d)
    V = None
    if kind == "R1":
        _, P = lib.fit.pca_basis(Hf, 128)
        V = P @ np.linalg.qr(rs.randn(128, 128))[0][:, :32]          # 32 random directions in top-128
        V = V.astype(np.float32)
    if kind == "Rd":
        src = {0: 4, 4: 10, 2: 7, 3: 7, 1: 5}                        # which real cloud serves as group latent
        perm_g = {g: (rs.permutation(n), rs.permutation(m)) for g in set(SCRIPT)}
        Zg = {}
        for g in set(SCRIPT):
            a = np.asarray(Rf[src[g]], np.float32)
            b = np.asarray(Rt[src[g]], np.float32)
            mu = a.mean(0)
            Zg[g] = ((a - mu)[perm_g[g][0]], (b - mu)[perm_g[g][1]])
    for i in range(Ln):
        for X, H, R, N in ((Xf, Hf, Rf, n), (Xt, Ht, Rt, m)):
            base = H
            if kind == "R1":
                D = (np.exp(eps * rs.randn(32)) - 1).astype(np.float32)
                base = H + ((H @ V) * D) @ V.T
            if kind == "R2iso":
                noise = level * scale_iso * rs.randn(N, d).astype(np.float32)
            else:
                a = np.asarray(R[i], np.float32)
                a = a - np.asarray(Rf[i], np.float32).mean(0) if X is Xt else a - a.mean(0)
                noise = level * a[rs.permutation(N)]
            X[i] = base + noise
            if kind == "Rd":
                X[i] += gamma * Zg[SCRIPT[i]][0 if X is Xf else 1]
    # note: for R1 the stretch D_i is drawn separately for fit and test -> must be identical; fix below
    return Xf, Xt


def make_R1(level, layer, seed=0, eps=0.5):
    """R1 with the SAME S_i on fit and test."""
    rs = np.random.RandomState(seed)
    Rf = np.load(os.path.join(lib.DATA, f"mean_dev_L{layer}.f16.npy"), mmap_mode="r")
    Rt = np.load(os.path.join(lib.DATA, f"mean_devtest_L{layer}.f16.npy"), mmap_mode="r")
    Hf, Ht = np.asarray(Rf[0], np.float32), np.asarray(Rt[0], np.float32)
    Ln, n, d = Rf.shape
    m = Rt.shape[1]
    _, P = lib.fit.pca_basis(Hf, 128)
    V = (P @ np.linalg.qr(rs.randn(128, 128))[0][:, :32]).astype(np.float32)
    Xf = np.empty((Ln, n, d), np.float32)
    Xt = np.empty((Ln, m, d), np.float32)
    for i in range(Ln):
        D = (np.exp(eps * rs.randn(32)) - 1).astype(np.float32)
        mu_i = np.asarray(Rf[i], np.float32).mean(0)
        for X, H, R, N in ((Xf, Hf, Rf, n), (Xt, Ht, Rt, m)):
            a = np.asarray(R[i], np.float32) - mu_i
            X[i] = H + ((H @ V) * D) @ V.T + level * a[rs.permutation(N)]
    return Xf, Xt


def build(kind, level, layer):
    if kind == "R1":
        return make_R1(level, layer)
    return make(kind, level, layer)


def quick_rho(Xf, Xt, k):
    Zf, Zt, _ = project(Xf, Xt, k)
    Gf, Gt = grams(Zf, Zt)
    return float(np.nanmean(rho_pairs(maps_family(Gf, "proc"), Gt)))


def tune(kind, layer, k, target, lo, hi, iters=6, tol=0.008):
    """bisection in log(level) on procrustes rho (monotone increasing in the noise level)."""
    hist = []
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        Xf, Xt = build(kind, mid, layer)
        r = quick_rho(Xf, Xt, k)
        del Xf, Xt
        hist.append((mid, r))
        print(f"   tune {kind} level={mid:.4f} rho={r:.4f} target={target:.4f}", flush=True)
        if abs(r - target) < tol:
            break
        if r < target:
            lo = mid
        else:
            hi = mid
    return min(hist, key=lambda t: abs(t[1] - target))[0], hist


def analyse(Xf, Xt, ks, tag, layer, rows, extras, do_null=True):
    kmax = max(ks)
    for k in ks:
        t0 = time.time()
        Zf, Zt, bases = project(Xf, Xt, k)
        Gf, Gt = grams(Zf, Zt)
        o, ex = evaluate(Gf, Gt, keep_sv=True)
        h, hx = hub_ladder(Gf, Gt)
        o.update(h)
        rows.append(dict(layer=layer, data=tag, which="data", k=k, **o))
        extras[f"{tag}_k{k}"] = {**{a: b for a, b in ex.items() if a in ("cancorr", "own", "sv_ridge", "sv_wproc_1")}, **hx}
        print(f"  [{tag} k={k}] " + " ".join(f"{a}={b:.4f}" for a, b in o.items() if isinstance(b, float) and (a.startswith("rho") or a.startswith("c_") or a.startswith("K_"))) + f" ({time.time() - t0:.0f}s)", flush=True)
        if do_null:
            Xfn, Xtn, infl = null_data(Xf, Xt, Zf, Zt, bases)
            Zfn, Ztn, _ = project(Xfn, Xtn, k)
            del Xfn, Xtn
            Gfn, Gtn = grams(Zfn, Ztn)
            on, exn = evaluate(Gfn, Gtn, keep_sv=True)
            hn, hxn = hub_ladder(Gfn, Gtn)
            on.update(hn)
            rows.append(dict(layer=layer, data=tag, which="null", k=k, infl=infl, **on))
            extras[f"{tag}_null_k{k}"] = {**{a: b for a, b in exn.items() if a in ("cancorr", "own")}, **hxn}
            print(f"  [{tag} NULL k={k}] c_proc={on['c_proc']:.4f} c_ridge={on['c_ridge']:.4f} rho_proc={on['rho_proc']:.4f} rho_ridge={on['rho_ridge']:.4f} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--kinds", default="real,R2iso,R2shuf,R1,Rd")
    ap.add_argument("--ks", default="64,128")
    ap.add_argument("--tune_k", type=int, default=64)
    ap.add_argument("--no_null", action="store_true")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    layer = args.layer
    rows, extras, levels = [], {}, {}
    out_csv = os.path.join(lib.OUT, f"res_L{layer}_{args.kinds.replace(',', '-')}_{args.ks.replace(',', '-')}.csv")
    target = real_rho_proc(layer, args.tune_k)
    ranges = dict(R2iso=(0.3, 6.0), R2shuf=(0.1, 2.0), R1=(0.05, 2.0), Rd=(0.05, 2.0))
    for kind in args.kinds.split(","):
        t0 = time.time()
        if kind == "real":
            Xf, Xt = load("dev", layer), load("devtest", layer)
        else:
            lvl, hist = tune(kind, layer, args.tune_k, target, *ranges[kind])
            levels[kind] = dict(level=lvl, hist=hist)
            Xf, Xt = build(kind, lvl, layer)
        print(f"== L{layer} {kind} ready ({time.time() - t0:.0f}s)", flush=True)
        analyse(Xf, Xt, ks, kind, layer, rows, extras, do_null=not args.no_null)
        del Xf, Xt
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        np.save(out_csv.replace(".csv", "_extras.npy"), extras, allow_pickle=True)
        with open(out_csv.replace(".csv", "_levels.json"), "w") as f:
            json.dump(levels, f)
    print("done", out_csv)


if __name__ == "__main__":
    main()
