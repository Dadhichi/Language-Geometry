"""summarize.py [glob] -- read out/L*_k*_*.npz and print gauge-invariant holonomy summaries per dataset."""
import glob
import itertools
import json
import os
import re
import sys
import numpy as np
import holo as H

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
TRI = H.TRI
EDGES = list(itertools.combinations(range(12), 2))
EIDX = {e: n for n, e in enumerate(EDGES)}


def tri_c(c):
    """per-unordered-triangle mean over the 6 orderings of c[b,p,q]."""
    return np.array([np.mean([c[x, y, z] for x, y, z in itertools.permutations(t)]) for t in TRI])


def design(kind):
    if kind == "vertex":
        X = np.zeros((len(TRI), 12))
        for n, t in enumerate(TRI):
            X[n, list(t)] = 1
    else:
        X = np.zeros((len(TRI), len(EDGES)))
        for n, (a, b, c) in enumerate(TRI):
            for e in ((a, b), (a, c), (b, c)):
                X[n, EIDX[e]] = 1
    return np.concatenate([np.ones((len(TRI), 1)), X], 1)


def cv_r2(X, y, folds=10, seed=0, lam=1e-6):
    rs = np.random.RandomState(seed)
    idx = rs.permutation(len(y))
    pred = np.zeros_like(y)
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        b = np.linalg.solve(X[tr].T @ X[tr] + lam * np.eye(X.shape[1]), X[tr].T @ y[tr])
        pred[te] = X[te] @ b
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def fit_coef(X, y, lam=1e-6):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def ang_stats(ang):
    th2 = ang ** 2                                        # [T, k] sorted desc, planes counted twice
    tot = th2.sum(1)
    pr = tot ** 2 / (th2 ** 2).sum(1) / 2                 # participation ratio in planes
    top1 = th2[:, :2].sum(1) / tot
    top3 = th2[:, :6].sum(1) / tot
    return dict(theta_rms_deg=float(np.degrees(np.sqrt(tot.mean() / ang.shape[1]))),
                theta_max_deg=float(np.degrees(ang[:, 0].mean())), pr_planes=float(pr.mean()),
                top1plane=float(top1.mean()), top3planes=float(top3.mean()))


def wsv_stats(w):
    w = w.reshape(-1, w.shape[-1])
    tot = w.sum(1)
    return dict(c_base=float(tot.mean()), pr_w=float((tot ** 2 / (w ** 2).sum(1)).mean()),
                top1_w=float((w[:, 0] / tot).mean()), top3_w=float((w[:, :3].sum(1) / tot).mean()))


def ridge_stats(ev):
    mod, arg = np.abs(ev), np.abs(np.angle(ev))
    return dict(ridge_logmod=float(np.log(mod).mean()), ridge_arg_deg=float(np.degrees(arg).mean()),
                ridge_frac_rot=float((arg > np.radians(5)).mean()))


def summarize_file(path, verbose=True):
    z = np.load(path)
    m = re.search(r"L(\d+)_k(\d+)_(\w+)\.npz", os.path.basename(path))
    layer, k, tag = int(m.group(1)), int(m.group(2)), m.group(3)
    names = sorted({key.split("_")[0] for key in z.files if key.endswith("_c_proc")},
                   key=lambda s: (s != "real", len(s), s))
    refs = {key[4:]: z[key].astype(np.float64) for key in z.files if key.startswith("ref_")}
    rows = {}
    off = ~np.eye(12, dtype=bool)
    for nm in names:
        r = dict(c_proc=float(np.nanmean(z[f"{nm}_c_proc"])), c_ridge=float(np.nanmean(z[f"{nm}_c_ridge"])),
                 rho=float(z[f"{nm}_rho_proc"][off].mean()), rho_ridge=float(z[f"{nm}_rho_ridge"][off].mean()))
        prof = z[f"{nm}_prof_proc"].mean(0)
        q = np.array([prof[a:b].sum() for a, b in ((0, k // 4), (k // 4, k // 2), (k // 2, 3 * k // 4), (3 * k // 4, k))])
        r["prof_quart_frac"] = (q / q.sum()).round(3).tolist()
        r["c_tophalf"] = float(q[:2].sum())
        r.update(ang_stats(z[f"{nm}_angles"]))
        r.update(wsv_stats(z[f"{nm}_wsv"]))
        r.update(ridge_stats(z[f"{nm}_ridge_eigs"]))
        r["comm_unw"] = float(z[f"{nm}_comm_unw"].mean())
        r["comm_w"] = float(z[f"{nm}_comm_w"].mean())
        if f"{nm}_cancorr" in z.files:
            cc = z[f"{nm}_cancorr"][off]                      # [132, k] descending
            r["cc_quart"] = [round(float(cc[:, a:b].mean()), 3) for a, b in
                             ((0, k // 4), (k // 4, k // 2), (k // 2, 3 * k // 4), (3 * k // 4, k))]
            r["n_cc_gt05"] = float((cc > 0.5).sum(1).mean())
        F = z[f"{nm}_Fp_test"].astype(np.float64)
        r["share"] = {rn: round(H.share(B, F), 4) for rn, B in refs.items()}
        rows[nm] = r
    out = dict(layer=layer, k=k, tag=tag, rows=rows)
    # ---- per-triangle attribution of excess over each null
    ct = {nm: tri_c(z[f"{nm}_c_proc"]) for nm in names}
    Xv, Xe = design("vertex"), design("edge")
    att = {}
    for nm in names:
        if nm == "real":
            continue
        y = ct["real"] - ct[nm]
        bv = fit_coef(Xv, y)
        be = fit_coef(Xe, y)
        same_script = np.array([H.SCRIPT[a] == H.SCRIPT[b] for a, b in EDGES])
        att[nm] = dict(excess_mean=float(y.mean()), excess_sd=float(y.std()),
                       frac_tri_pos=float((y > 0).mean()),
                       r2cv_vertex=float(cv_r2(Xv, y)), r2cv_edge=float(cv_r2(Xe, y)),
                       vertex_eff={H.LANGS[i]: round(float(bv[1 + i]), 4) for i in np.argsort(-bv[1:])},
                       edge_top=[(H.LANGS[EDGES[i][0]] + "-" + H.LANGS[EDGES[i][1]], round(float(be[1 + i]), 4))
                                 for i in np.argsort(-be[1:])[:6]],
                       edge_samescript_minus_other=float(be[1:][same_script].mean() - be[1:][~same_script].mean()))
    out["attrib"] = att
    # per-triangle correlation real vs null (does the null reproduce WHICH triangles are curved?)
    out["tri_corr"] = {nm: float(np.corrcoef(ct["real"], ct[nm])[0, 1]) for nm in names if nm != "real"}
    out["tri_real_top"] = [("-".join(H.LANGS[i] for i in TRI[t]), round(float(ct["real"][t]), 4))
                           for t in np.argsort(-ct["real"])[:5]]
    # per-vertex mean c of triangles containing v
    out["vertex_c"] = {nm: {H.LANGS[v]: round(float(ct[nm][[v in t for t in TRI]].mean()), 4) for v in range(12)}
                       for nm in names}
    # hub-frame edge discrepancy
    out["u_edge_mean"] = {nm: float(z[f"{nm}_u_edge"][off].mean()) for nm in names}
    # ---- ambient excess spectra: real minus each null (test-weighted discrepancy covariance)
    Fr = z["real_Fp_test"].astype(np.float32)
    ex = {}
    for nm in names:
        if nm == "real":
            continue
        w, V, tr = H.excess_eig(Fr, [z[f"{nm}_Fp_test"].astype(np.float32)], top=32)
        cum = np.cumsum(w) / tr
        ex[nm] = dict(tr_excess=tr, cum_top=[round(float(cum[i - 1]), 3) for i in (1, 2, 4, 8, 16, 32)],
                      top_ref_overlap={rn: round(float(((B.T @ V[:, :8]) ** 2).sum() / 8), 3) for rn, B in refs.items()})
        np.save(path.replace(".npz", f"_excessV_{nm}.npy"), V[:, :16])
    out["excess"] = ex
    return out


def fmt(o):
    lines = [f"==== L{o['layer']} k={o['k']} [{o['tag']}]"]
    hdr = ["c_proc", "c_ridge", "rho", "c_tophalf", "theta_rms_deg", "theta_max_deg", "pr_planes", "top1plane",
           "top3planes", "pr_w", "top1_w", "top3_w", "comm_unw", "comm_w", "ridge_logmod", "ridge_arg_deg"]
    lines.append(f"{'':8s}" + "".join(f"{h[:11]:>12s}" for h in hdr))
    for nm, r in o["rows"].items():
        lines.append(f"{nm:8s}" + "".join(f"{r[h]:12.4f}" for h in hdr))
    for nm, r in o["rows"].items():
        lines.append(f"  {nm:8s} quartile shares of c (PC rank of target) {r['prof_quart_frac']}  shares {r['share']}"
                     + (f"  cancorr quartiles {r['cc_quart']} n(cc>0.5)={r['n_cc_gt05']:.1f}" if 'cc_quart' in r else ""))
    for nm, a in o["attrib"].items():
        lines.append(f"  excess vs {nm}: mean {a['excess_mean']:.4f} sd {a['excess_sd']:.4f} pos {a['frac_tri_pos']:.2f} "
                     f"R2cv vertex {a['r2cv_vertex']:.2f} edge {a['r2cv_edge']:.2f} tri_corr {o['tri_corr'][nm]:.2f}")
        lines.append(f"      vertex eff {a['vertex_eff']}")
        lines.append(f"      top edges {a['edge_top']} same-script minus other {a['edge_samescript_minus_other']:.4f}")
    for nm, e in o["excess"].items():
        lines.append(f"  ambient excess vs {nm}: tr {e['tr_excess']:.4f} cum top-(1,2,4,8,16,32) {e['cum_top']} "
                     f"top8 overlap {e['top_ref_overlap']}")
    lines.append(f"  real top triangles {o['tri_real_top']}")
    lines.append(f"  vertex mean c (real) {o['vertex_c']['real']}")
    return "\n".join(lines)


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else "L*_k*_*.npz"
    allo = []
    for p in sorted(glob.glob(os.path.join(OUT, pat))):
        o = summarize_file(p)
        allo.append(o)
        print(fmt(o), flush=True)
    tag = sys.argv[2] if len(sys.argv) > 2 else "summary"
    json.dump(allo, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1, default=float)
    print("rand-skew comm baseline k=64/128:", round(H.random_skew_comm(64), 3), round(H.random_skew_comm(128), 3),
          " common-2plane:", round(H.lowrank_skew_comm(64, 2), 3), " common-8dim:", round(H.lowrank_skew_comm(64, 8), 3))
