"""run.py --layer L [--ks 64,128] [--kbmax 256]
Real data + a HIERARCHY of flat surrogates through one identical pipeline:
  O{kb}:  flat orthogonal gauge (GPA hub) fitted in each language's top-kb PCA subspace (kb in {k,2k,4k}),
          lifted to the ambient space (fit.analyze_with_null construction), re-PCA'd and analysed at k.
          O{k} is exactly fit.py's null.  O{kb>k} is flat in kb dims, so any curvature it shows at k is
          manufactured by per-language top-k truncation.
  GL{k}:  flat linear (GL) gauge, A_i ~ S G_i, rank-k hub (analysed at k).
Saves out/L{layer}_k{k}_{tag}.npz."""
import argparse
import os
import time
import numpy as np
import holo as H

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
KEEP = ["c_proc", "c_ridge", "angles", "wsv", "ridge_eigs", "comm_unw", "comm_w", "u_edge", "rho_proc",
        "rho_ridge", "prof_proc", "prof_ridge", "cancorr"]


def pack(prefix, st, bases, dst):
    for key in KEEP:
        if key in st:
            dst[f"{prefix}_{key}"] = st[key]
    dst[f"{prefix}_Fp_test"] = H.factor(bases, st["Gp_test"]).astype(np.float16)


def refs_for(A, bases, means, ntok, k):
    wu = np.load(os.path.join(H.DATA, "wu_basis.npy")).astype(np.float64)
    r = {"wu16": wu[:, :16], "wu64": wu[:, :64]}
    r["means"] = H.orth((means - means.mean(0)).T)[:, :11]
    lam = (A ** 2).mean(1)
    Fc = np.concatenate([bases[i].astype(np.float64) * np.sqrt(lam[i])[None] for i in range(len(bases))], 1)
    w, U = np.linalg.eigh(Fc.T @ Fc)
    U, w = U[:, ::-1], w[::-1]
    C = Fc @ U[:, :64] / np.sqrt(w[:64])
    r["content16"], r["content64"] = C[:, :16], C[:, :64]
    if ntok is not None:
        fert = []
        for i in range(len(bases)):
            y = np.log(ntok[i]) - np.log(ntok[i]).mean()
            G = A[i].T @ A[i]
            beta = np.linalg.solve(G + 1e-3 * np.trace(G) / k * np.eye(k), A[i].T @ y)
            fert.append(bases[i].astype(np.float64) @ beta)
        r["fert"] = H.orth(np.stack(fert, 1))
    # each language's own truncation edge (last quarter of retained PCs) vs core (first quarter)
    r["edge"] = H.orth(np.concatenate([bases[i][:, 3 * k // 4:] for i in range(len(bases))], 1).astype(np.float64))
    r["core"] = H.orth(np.concatenate([bases[i][:, :k // 4] for i in range(len(bases))], 1).astype(np.float64))
    return r


def run_dataset(Xf, Xt, ntok, ks, kbmax, tag, layer, extra_refs=None, nO=1, with_gl=True, t0=None):
    t0 = t0 or time.time()
    kp = max(kbmax, max(ks))
    A0, T0, P0, g0 = H.project(Xf, Xt, kp)
    print(f"[{tag} L{layer}] projected k={kp} [{time.time()-t0:.0f}s]", flush=True)
    res = {k: {} for k in ks}
    sl = lambda A, k: A[:, :, :k] - A[:, :, :k].mean(1, keepdims=True)
    for k in ks:
        A, T, P = sl(A0, k), T0[:, :, :k], np.ascontiguousarray(P0[:, :, :k])
        dst = res[k]
        for nm, B in refs_for(A, P, g0, ntok, k).items():
            dst[f"ref_{nm}"] = B.astype(np.float32)
        if extra_refs:
            for nm, B in extra_refs.items():
                dst[f"ref_{nm}"] = B.astype(np.float32)
        st = H.analyze_conn(A, T, seed=0)
        pack("real", st, P, dst)
        if with_gl:
            mG = H.model_GL(A, T)
            An, Tn, Pn, _ = H.surrogate(Xf, Xt, g0, P, mG, seed=900 + k)
            pack(f"GL{k}", H.analyze_conn(An, Tn, seed=5), Pn, dst)
            del An, Tn, Pn, mG
        print(f"[{tag} L{layer}] k={k} real c={np.nanmean(st['c_proc']):.4f} cR={np.nanmean(st['c_ridge']):.4f} "
              f"rho={np.nanmean(st['rho_proc'][~np.eye(12, dtype=bool)]):.3f}"
              + (f" GL{k} c={np.nanmean(dst[f'GL{k}_c_proc']):.4f}" if with_gl else "")
              + f" [{time.time()-t0:.0f}s]", flush=True)
    kbs = sorted({kb for k in ks for kb in (k, 2 * k, 4 * k) if kb <= kbmax})
    for kb in kbs:
        A, T, P = sl(A0, kb), T0[:, :, :kb], np.ascontiguousarray(P0[:, :, :kb])
        R, _ = H.fit_maps(A)
        mO = H.model_O(A, T, R)
        del R
        targets = [k for k in ks if kb in (k, 2 * k, 4 * k)]
        for rep in range(nO):
            An, Tn, Pn, _ = H.surrogate(Xf, Xt, g0, P, mO, seed=1000 + kb + 31 * rep, k=max(targets))
            for k in targets:
                sn = H.analyze_conn(sl(An, k), Tn[:, :, :k], seed=rep + 1)
                pack(f"O{kb}r{rep}", sn, np.ascontiguousarray(Pn[:, :, :k]), res[k])
                res[k][f"O{kb}r{rep}_infl"] = mO["infl"]
                print(f"[{tag} L{layer}] O{kb}->k{k} rep{rep}: c={np.nanmean(sn['c_proc']):.4f} "
                      f"cR={np.nanmean(sn['c_ridge']):.4f} rho={np.nanmean(sn['rho_proc'][~np.eye(12, dtype=bool)]):.3f} "
                      f"infl={mO['infl']:.3f} [{time.time()-t0:.0f}s]", flush=True)
            del An, Tn, Pn
        del mO
    for k in ks:
        np.savez_compressed(os.path.join(OUT, f"L{layer}_k{k}_{tag}.npz"), **res[k])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--ks", default="64,128")
    ap.add_argument("--kbmax", type=int, default=256)
    ap.add_argument("--fit", default="dev")
    ap.add_argument("--nO", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ts = "devtest" if args.fit == "dev" else "dev"
    Xf, Xt = H.load(args.fit, args.layer), H.load(ts, args.layer)
    ntok = np.load(os.path.join(H.DATA, f"ntok_{args.fit}.npy")).astype(np.float64)
    run_dataset(Xf, Xt, ntok, [int(x) for x in args.ks.split(",")], args.kbmax, args.fit, args.layer, nO=args.nO)


if __name__ == "__main__":
    main()
