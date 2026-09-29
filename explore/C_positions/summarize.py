"""Collect out/layer*.json into CSV tables: q1.csv, q2.csv, q3.csv, positions.csv."""
import glob
import json
import os

import numpy as np
import pandas as pd

from common import LANGS, OUT

rows1, rows2, rows3, rowsp = [], [], [], []
for f in sorted(glob.glob(os.path.join(OUT, "layer*.json"))):
    if "quick" in f:
        continue
    R = json.load(open(f))
    layer = R["layer"]
    for world in ("real", "identity", "abelian", "nonabelian", "scaling"):
        if world not in R:
            continue
        w = R[world]
        q = w["q1"]
        best = min(q.get("rho_dualridge_I", 9), q["rho_ridgeU512_I"], q["rho_ridgeU256_I"])
        rows1.append(dict(layer=layer, world=world, rho_identity=q["rho_identity"], rho_shift=q["rho_shift"],
                          procU128_I=q["rho_procU128_I"], procU256_I=q["rho_procU256_I"],
                          procU512_I=q["rho_procU512_I"], ridgeU256_I=q["rho_ridgeU256_I"],
                          ridgeU512_I=q["rho_ridgeU512_I"], ridgeU512_0=q["rho_ridgeU512_0"],
                          dual_I=q.get("rho_dualridge_I", np.nan), dual_0=q.get("rho_dualridge_0", np.nan),
                          shift_share=(q["rho_identity"] - q["rho_shift"]) / (q["rho_identity"] - best),
                          off2_trS=q["offset2_over_trS"], offc2_trS=q["offset_from_centroid2_over_trS"],
                          off_inU256=q["offset_in_U256"], content_inU256=q["content_in_U256"],
                          off_inU64=q["offset_in_U64"], content_inU64=q["content_in_U64"],
                          off_inWU=q.get("offset_in_WU512"), content_inWU=q.get("content_in_WU512")))
        for m, o in w.get("q2", {}).items():
            rows2.append(dict(layer=layer, world=world, m=int(m), **{k: o.get(k) for k in (
                "rho_shift", "rho_proc", "rho_gpa", "rho_torus", "rho_torus_r1", "rho_torus_r2", "rho_torus_r4",
                "L_norm2", "splithalf_L_corr", "splithalf_L_signal2", "ang_pr_frac", "ang_top4", "sh_top4_planes",
                "comm_raw", "comm_norm", "sh_comm", "e_add", "e_hol", "sh_add", "sh_hol", "c_comp", "c_abelian",
                "n_det_neg", "ang_frac_pi", "theta1_corr_truth")},
                mantel_t1_script_p=o.get("mantel_theta1_script", [np.nan, np.nan])[1],
                mantel_t1_family_p=o.get("mantel_theta1_family", [np.nan, np.nan])[1],
                mantel_torus_script=o.get("mantel_torus_script", [np.nan, np.nan])[0],
                mantel_torus_script_p=o.get("mantel_torus_script", [np.nan, np.nan])[1],
                mantel_torus_family_p=o.get("mantel_torus_family", [np.nan, np.nan])[1],
                mantel_gpa_script=o["mantel_gpa_script"][0], mantel_gpa_script_p=o["mantel_gpa_script"][1],
                mantel_gpa_family=o["mantel_gpa_family"][0], mantel_gpa_family_p=o["mantel_gpa_family"][1],
                theta1_order=" ".join(o.get("theta1_order", []))))
            if "theta_r1" in o:
                for li, lang in enumerate(LANGS):
                    rowsp.append(dict(layer=layer, world=world, m=int(m), lang=lang,
                                      theta1=o["theta_r1"][li][0], theta2a=o["theta_r2"][li][0],
                                      theta2b=o["theta_r2"][li][1]))
        for m, o in w.get("q3", {}).items():
            rows3.append(dict(layer=layer, world=world, m=int(m), **{k: o.get(k) for k in (
                "sh_signal_rot", "sh_signal_scale", "sh_signal_iso", "sh_frac_rot", "frac_rot", "frac_scale",
                "frac_rot_vs_traceless", "iso_frac_of_sym", "X_norm2", "D_norm2", "mean_log_atten", "henrici",
                "frac_complex_eig", "mean_abs_eig", "n_negreal_eig", "rho_shift", "rho_proc", "rho_ridge",
                "rho_expX", "rho_expK_rot", "rho_expS_scale", "rho_iso_scale", "rho_polarO", "scale_pos_R2",
                "scale_pos_corr_logntok", "scale_pos_corr_logsd")},
                scale_pos=" ".join(f"{LANGS[k]}:{v:+.3f}" for k, v in enumerate(o["scale_pos"]))))

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 120)
for name, rows in (("q1", rows1), ("q2", rows2), ("q3", rows3), ("positions", rowsp)):
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    print(f"\n==== {name} ({len(df)} rows)")
    if len(df):
        print(df.round(4).to_string(index=False))

# ---- positions vs confounds: r=1 rotation position, log-volume (isotropic scale) position, log token count
nt = np.log(np.load("C:/Users/ASUS/Documents/lang-geom/ntok_dev.npy").astype(np.float64).mean(1))
rowsc = []
for f in sorted(glob.glob(os.path.join(OUT, "layer*.json"))):
    if "quick" in f:
        continue
    R = json.load(open(f))
    for world in ("real", "identity", "abelian", "nonabelian"):
        if world not in R:
            continue
        for m in ("64", "128"):
            o2, o3 = R[world]["q2"].get(m), R[world]["q3"].get(m)
            if o2 is None or "theta_r1" not in o2:
                continue
            th = np.array(o2["theta_r1"])[:, 0]
            sp = np.array(o3["scale_pos"])
            rowsc.append(dict(layer=R["layer"], world=world, m=int(m),
                              corr_theta1_scalepos=abs(np.corrcoef(th, sp)[0, 1]),
                              corr_theta1_logntok=abs(np.corrcoef(th, nt)[0, 1]),
                              corr_scalepos_logntok=np.corrcoef(sp, nt)[0, 1],
                              phi_sv1=o2["phi_sv_frac"][0], phi_sv2=o2["phi_sv_frac"][1]))
df = pd.DataFrame(rowsc)
df.to_csv(os.path.join(OUT, "position_confounds.csv"), index=False)
print("\n==== position confounds")
print(df.round(3).to_string(index=False))
