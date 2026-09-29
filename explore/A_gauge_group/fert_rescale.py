"""Fertility control for the similarity scale: rescale each mean-pooled sentence vector by (ntok/ntok_ref)^p
(p=0.5 ~ turning the mean into a length-normalised sum), then rerun the identical pipeline.  If the per-language
scale s_i is a mean-pooling dilution effect, Kglob and the O->Sim gap should collapse."""
import os
import sys

import numpy as np
import pandas as pd

import lib
from lib import load
from run import analyse

layer = int(sys.argv[1])
p = float(sys.argv[2])
k = int(sys.argv[3]) if len(sys.argv) > 3 else 64
Xf, Xt = load("dev", layer), load("devtest", layer)
nf = np.load(os.path.join(lib.DATA, "ntok_dev.npy")).astype(np.float32)
nt = np.load(os.path.join(lib.DATA, "ntok_devtest.npy")).astype(np.float32)
ref = float(nf.mean())
for i in range(12):
    Xf[i] *= ((nf[i] / ref) ** p)[:, None]
    Xt[i] *= ((nt[i] / ref) ** p)[:, None]
rows, extras = [], {}
analyse(Xf, Xt, [k], f"real_ntok{p}", layer, rows, extras, do_null=True)
out = os.path.join(lib.OUT, f"res_L{layer}_fert{p}_{k}.csv")
pd.DataFrame(rows).to_csv(out, index=False)
np.save(out.replace(".csv", "_extras.npy"), extras, allow_pickle=True)
s = np.log(extras[f"real_ntok{p}_k{k}"]["scale_sim"])
print("log scale per language:", dict(zip(lib.LANGS, np.round(s, 3))))
print("corr with log fertility:", np.corrcoef(s, np.log(nf.mean(1)))[0, 1])
