"""Aggregate per-layer stats + run centroid-level analyses on the real data."""
import os, sys, json, glob
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlib, clib
import functools
print = functools.partial(print, flush=True)

HERE = dlib.OUT
lay = sorted(int(os.path.basename(p)[1:-5]) for p in glob.glob(os.path.join(HERE, "layers", "L*.json")))
S = pd.DataFrame([json.load(open(os.path.join(HERE, "layers", f"L{l}.json")))["S"] for l in lay]).set_index("layer")
R = pd.concat([pd.DataFrame(json.load(open(os.path.join(HERE, "layers", f"L{l}.json")))["rows"]) for l in lay])
S.to_csv(os.path.join(HERE, "by_layer.csv"))
R.to_csv(os.path.join(HERE, "by_layer_lang.csv"), index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def show(title, cols):
    print(f"\n== {title}")
    print(S[[c for c in cols if c in S]].round(3).to_string())


show("variance decomposition (dev); inv_frac = sentence-invariant share of the per-sentence offset",
     ["f_lang_dev", "f_sent_dev", "f_int_dev", "f_sent_shuf", "inv_frac_dev", "inv_frac_test", "lang_over_content_dev",
      "a_split_cos", "anova_check", "norm_m_dev"])
show("effective dimension", ["PR_a", "PR_a_x", "a_top1", "a_top2", "a_top3", "a_top5", "PR_o", "o_top11", "PR_e",
                              "e_top11", "e_top64", "PR_b", "b_top11"])
show("language subspace vs content subspace (iso = random direction)",
     ["a_in_content11", "a_in_content64", "a_in_content256", "b_in_content64", "e_in_content64", "iso_64",
      "cos2_lang_content11_mean", "cos_lang_content11_max", "rayleigh_content_along_lang", "e_in_langspan",
      "iso_r_over_d", "e_own_dir_share_mean", "t_cv_mean"])
show("prediction (devtest): ambient shift vs per-language-PCA maps",
     ["rho_identity_amb", "rho_shift_amb", "rho_scaledshift_amb", "frac_err_removed_by_shift", "p1_identity_amb",
      "p1_shift_amb", "rho_shift_k64", "rho_shift_trunc_k64", "rho_polar_id_k64", "rho_proc_k64", "rho_ridge_k64",
      "proc_vs_identity_k64", "p1_shift_trunc_k64", "p1_polar_id_k64", "p1_proc_k64", "p1_ridge_k64"])
show("prediction k=256", ["rho_shift_trunc_k256", "rho_polar_id_k256", "rho_proc_k256", "rho_ridge_k256",
                          "proc_vs_identity_k256", "p1_shift_trunc_k256", "p1_polar_id_k256", "p1_proc_k256", "p1_ridge_k256"])
show("content dependence of offsets (LOO-hub residual / target energy)",
     ["hub_resid_shift", "hub_resid_alpha", "hub_resid_ridge_shared", "hub_resid_ridge_perlang", "ridge_rel"])
show("unembedding: energy fraction in span(W_U top-512) [iso=0.143] / top-64 [iso=0.018]",
     ["W_a", "W_b", "W_e", "W_m", "W_iso", "W64_a", "W64_b", "W64_e", "Wgain_skip0_a", "Wgain_skip0_b", "Wgain_skip0_e"])
S["W_a_over_b"] = S["W_a"] / S["W_b"]
S["Wgain_a_over_b"] = S["Wgain_skip0_a"] / S["Wgain_skip0_b"]
show("unembedding ratios", ["W_a_over_b", "Wgain_a_over_b"])

# per-language tables
print("\n== per-language ||a_i||^2 / content variance, by layer")
print(R.pivot(index="layer", columns="lang", values="a2_over_content")[dlib.LANGS].round(2).to_string())
print("\n== per-language hub alpha (content gain), by layer")
print(R.pivot(index="layer", columns="lang", values="hub_alpha")[dlib.LANGS].round(2).to_string())
print("\n== per-language W_U fraction of the offset, by layer")
print(R.pivot(index="layer", columns="lang", values="W_a")[dlib.LANGS].round(2).to_string())
print("\n== within-language Spearman(per-sentence offset norm, log ntok), by layer")
print(R.pivot(index="layer", columns="lang", values="enorm_vs_logntok")[dlib.LANGS].round(2).to_string())

# ---------------------------------------------------------------- centroid-level analyses
tok = dict(np.load(os.path.join(HERE, "tok_sim.npz")))
refs = clib.ref_dists(tok)
fert = tok["fert"]
A = {l: dict(np.load(os.path.join(HERE, "layers", f"L{l}.npz"))) for l in lay}
rows_attr, rows_tree, rows_mantel = [], [], []
lf = np.log(fert)
extra = {"fertility": [lf], "script+family+fert": [clib.LATIN, clib.CJK, clib.IE, clib.GERM, clib.ROM, lf]}
for l in lay:
    ad, at = A[l]["a_dev"].astype(np.float64), A[l]["a_test"].astype(np.float64)
    Qb = np.linalg.svd(np.vstack([ad, at]), full_matrices=False)[2].T      # exact: all ops live in this span
    ad, at = ad @ Qb, at @ Qb
    for drop in ((), (5,)):
        att = clib.attribute_test(ad, at, n_perm=300, extra=extra, drop=drop)
        for k, v in att.items():
            rows_attr.append(dict(layer=l, drop_hin=bool(drop), model=k, r2=v["r2"], null_mean=v["null_mean"],
                                  null_p95=v["null_p95"], p=v["p"]))
        cr = clib.crossing_test(ad, n_perm=2000, drop=drop)
        crt = clib.crossing(at)
        rows_attr.append(dict(layer=l, drop_hin=bool(drop), model="crossing_cos", r2=cr["cos"], null_mean=cr["null_cos_mean"],
                              null_p95=cr["null_cos_p95"], p=cr["p_cos"], test_split=crt[0]))
    D2 = clib.xdist2(ad, at)
    D = np.sqrt(np.clip(D2, 0, None))
    Gc = (ad - ad.mean(0)) @ (ad - ad.mean(0)).T
    nul = clib.gaussian_null_tree(Gc, n=200, seed=l)
    fp = clib.fourpoint(D)
    rows_tree.append(dict(layer=l, eps4=fp[0], eps4_null=nul[:, 0].mean(), eps4_null_p05=np.quantile(nul[:, 0], 0.05),
                          delta_rel=fp[1], delta_rel_null=nul[:, 1].mean(), coph=clib.coph(D), coph_null=nul[:, 2].mean(),
                          coph_null_p95=np.quantile(nul[:, 2], 0.95)))
    for name, Rm in refs.items():
        r0, p = clib.mantel(D, Rm, n_perm=2000)
        rows_mantel.append(dict(layer=l, ref=name, spearman=r0, p=p))
    Dnh = np.delete(np.delete(D, 5, 0), 5, 1)
    mr, r2 = clib.mrm(D, {k: refs[k] for k in ("family", "script", "token", "fertility")}, n_perm=500)
    for k, (b, p) in mr.items():
        rows_mantel.append(dict(layer=l, ref=f"MRM_{k}", spearman=b, p=p, r2=r2))
TA, TT, TM = pd.DataFrame(rows_attr), pd.DataFrame(rows_tree), pd.DataFrame(rows_mantel)
TA.to_csv(os.path.join(HERE, "attributes.csv"), index=False)
TT.to_csv(os.path.join(HERE, "tree.csv"), index=False)
TM.to_csv(os.path.join(HERE, "mantel.csv"), index=False)
print("\n== attribute algebra: LOO R2 (vs predicting mean of the other 11), label-permutation null")
for drop in (False, True):
    t = TA[TA.drop_hin == drop]
    print(f"-- drop_hin={drop}")
    print(t.pivot(index="layer", columns="model", values="r2").round(3).to_string())
    print(t.pivot(index="layer", columns="model", values="null_p95").round(3).to_string())
    print(t.pivot(index="layer", columns="model", values="p").round(3).to_string())
print("\n== tree / metric")
print(TT.round(3).to_string())
print("\n== Mantel (Spearman) and MRM coefficients")
print(TM.pivot(index="layer", columns="ref", values="spearman").round(2).to_string())
print(TM.pivot(index="layer", columns="ref", values="p").round(3).to_string())

# ---------------------------------------------------------------- depth dynamics
rows_dep = []
for q in range(len(lay)):
    l = lay[q]
    ad, at = A[l]["a_dev"].astype(np.float64), A[l]["a_test"].astype(np.float64)
    r = dict(layer=l, split_shape=clib.shape_sim(ad, at), split_ident=clib.ident_sim(ad, at))
    if q + 1 < len(lay):
        bd = A[lay[q + 1]]["a_dev"].astype(np.float64)
        r.update(next_shape=clib.shape_sim(ad, bd), next_ident=clib.ident_sim(ad, bd), next_subspace=clib.subspace_overlap(ad, bd),
                 next_shape_null=np.mean([clib.shape_sim(ad, bd[np.random.RandomState(s).permutation(12)]) for s in range(200)]))
        iu = np.triu_indices(12, 1)
        D1 = np.sqrt(np.clip(clib.xdist2(ad, at), 0, None))
        D2_ = np.sqrt(np.clip(clib.xdist2(bd, A[lay[q + 1]]["a_test"].astype(np.float64)), 0, None))
        r["next_rsa"] = float(np.corrcoef(D1[iu], D2_[iu])[0, 1])
    last = A[lay[-1]]["a_dev"].astype(np.float64); first = A[lay[0]]["a_dev"].astype(np.float64)
    mid = A[14]["a_dev"].astype(np.float64) if 14 in A else ad
    r.update(shape_vs_L0=clib.shape_sim(ad, first), ident_vs_L0=clib.ident_sim(ad, first),
             shape_vs_L14=clib.shape_sim(ad, mid), ident_vs_L14=clib.ident_sim(ad, mid),
             shape_vs_last=clib.shape_sim(ad, last), ident_vs_last=clib.ident_sim(ad, last))
    rows_dep.append(r)
TD = pd.DataFrame(rows_dep).set_index("layer")
TD.to_csv(os.path.join(HERE, "depth.csv"))
print("\n== depth dynamics of the centroid configuration (shape = Procrustes w/ rotation+scale; ident = no rotation)")
print(TD.round(3).to_string())
# mean pairwise distance / sqrt(content variance), by layer
dd = []
for l in lay:
    D2 = clib.xdist2(A[l]["a_dev"].astype(np.float64), A[l]["a_test"].astype(np.float64))
    iu = np.triu_indices(12, 1)
    dd.append(dict(layer=l, mean_pair_dist2_over_content=float(D2[iu].mean() / S.loc[l, "V_sent_dev"]),
                   mean_pair_dist2_over_total=float(D2[iu].mean() / (S.loc[l, "V_sent_dev"] + S.loc[l, "V_int_dev"] + S.loc[l, "V_lang_dev"]))))
print(pd.DataFrame(dd).set_index("layer").round(3).to_string())
