"""Per-language gauge parameters from the real-data runs: similarity scale s_i (cross-covariance hub fit),
anisotropic stretch of the per-language metric K_i, vs fertility.  Also pair-specific residual by script."""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

import lib

ntok = np.load(os.path.join(lib.DATA, "ntok_dev.npy")).astype(float).mean(1)
SCRIPT = ["Latn", "Latn", "Latn", "Latn", "Cyrl", "Deva", "Arab", "Hans", "Jpan", "Latn", "Latn", "Latn"]
rows = []
for f in sorted(glob.glob(os.path.join(lib.OUT, "res_L*_real_*_extras.npy"))):
    ex = np.load(f, allow_pickle=True).item()
    layer = int(os.path.basename(f).split("_")[1][1:])
    for key, v in ex.items():
        if "null" in key or "scale_sim" not in v:
            continue
        k = int(key.split("_k")[-1])
        s = np.log(v["scale_sim"])
        g = v["K_g"] / 2                     # metric log-scale -> log length scale
        aniso = np.sqrt((v["K_ev"] ** 2).mean(1))
        top = np.abs(v["K_ev"]).max(1)
        own_var = np.log(v["own"].sum(1))
        r_f = spearmanr(s, np.log(ntok))[0]
        rp = pearsonr(s, np.log(ntok))[0]
        r_g = spearmanr(g, s)[0]
        rows.append(dict(layer=layer, k=k, rho_s_fert_spearman=r_f, rho_s_fert_pearson=rp, rho_Kg_vs_s=r_g,
                         rho_aniso_fert=spearmanr(aniso, np.log(ntok))[0],
                         rho_s_ownvar=spearmanr(s, own_var)[0],
                         s_range=float(np.exp(s.max() - s.min())),
                         **{f"logs_{lib.LANGS[i]}": round(float(s[i]), 3) for i in range(12)},
                         **{f"aniso_{lib.LANGS[i]}": round(float(aniso[i]), 3) for i in range(12)}))
        if "pairres" in v:
            PR = v["pairres"]
            same = [PR[i, j] for i in range(12) for j in range(12) if i != j and SCRIPT[i] == SCRIPT[j]]
            diff = [PR[i, j] for i in range(12) for j in range(12) if i != j and SCRIPT[i] != SCRIPT[j]]
            rows[-1].update(pairres_same_script=float(np.mean(same)), pairres_diff_script=float(np.mean(diff)))
df = pd.DataFrame(rows).sort_values(["layer", "k"])
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
print(df.round(3).to_string())
df.to_csv(os.path.join(lib.OUT, "table_per_language.csv"), index=False)
