"""Collect res_*.csv into tables.csv and print the key comparisons."""
import glob
import os

import numpy as np
import pandas as pd

import lib

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
fs = sorted(glob.glob(os.path.join(lib.OUT, "res_L*.csv")))
df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
df["which"] = df["which"].fillna("null")
# latest run wins for duplicated (layer, data, which, k)
df = df.drop_duplicates(["layer", "data", "which", "k"], keep="last")
df.to_csv(os.path.join(lib.OUT, "tables.csv"), index=False)

d = df[df.which == "data"].set_index(["layer", "k", "data"])
n = df[df.which == "null"].set_index(["layer", "k", "data"])
T = pd.DataFrame(index=d.index)
T["rho_proc"] = d.rho_proc
T["rho_ridge"] = d.rho_ridge
T["ridge/proc"] = d.rho_ridge / d.rho_proc
T["c_proc"] = d.c_proc
T["c_proc/null"] = d.c_proc / n.c_proc
T["c_ridge/null"] = d.c_ridge / n.c_ridge
T["c_wproc1/null"] = d.c_wproc_1 / n.c_wproc_1
T["hubO"] = d.rho_hubO
T["hubSim"] = d.rho_hubSim
T["hubGL"] = d.rho_hubGL
T["O-Sim"] = d.rho_hubO - d.rho_hubSim
T["Sim-GL"] = d.rho_hubSim - d.rho_hubGL
T["GL-ridge"] = d.rho_hubGL - d.rho_ridge
T["Kglob"] = d.K_var_global
T["Kaniso"] = d.K_var_aniso
T["Kcomm"] = d.K_comm
T["cc32_add"] = d.cc32_vadd
T["cc_res"] = d.cc_vres
print(T.sort_index().round(4).to_string())
T.sort_index().round(5).to_csv(os.path.join(lib.OUT, "table_main.csv"))

pcols = [c for c in d.columns if c.startswith("rho_hubGL_p")]
if pcols:
    P = d[["rho_hubSim"] + pcols + ["rho_hubGL", "rho_ridge"]].dropna(subset=pcols[:1])
    print("\npartial anisotropy ladder (rho, Wiener-predicted; p = # anisotropic stretch directions per language)")
    print(P.sort_index().round(4).to_string())
    P.sort_index().round(5).to_csv(os.path.join(lib.OUT, "table_partial.csv"))
loc = [c for c in ["c_proc_top", "c_proc_rest", "c_procGL_top", "c_ridge_top", "c_ridge_rest", "pairres_GL"] if c in d.columns]
if loc:
    X = d[loc].dropna().join(n[loc].dropna(), rsuffix="_null")
    print("\nlocalisation of composition error (top 25% PCA coords of source language vs rest)")
    print(X.sort_index().round(4).to_string())
    X.sort_index().round(5).to_csv(os.path.join(lib.OUT, "table_local.csv"))
