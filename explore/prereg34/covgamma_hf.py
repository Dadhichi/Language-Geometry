"""Runs on the Colab VM. Park et al. causal inner product for any HF causal LM, same recipe as explore/E_tree/covgamma.py
(Qwen): gamma = lm_head rows restricted to real (non-special/non-padding) tokens, Cov with (N-1), then
C_norm = diag(g) Cov(gamma) diag(g), g = final RMSNorm weight. Reads only the needed tensors from the cached shards.
usage: python covgamma_hf.py MODEL N_REAL_ROWS OUT.npy   (Llama-3.1-8B: rows < 128000; ids >= 128000 are special)"""
import json, sys
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

model, n_real, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
idx = json.load(open(hf_hub_download(model, "model.safetensors.index.json")))["weight_map"]


def get(name):
    with safe_open(hf_hub_download(model, idx[name]), framework="pt") as f:
        return f.get_tensor(name)


dev = "cuda" if torch.cuda.is_available() else "cpu"
W = get("lm_head.weight")[:n_real].to(dev).float()
g = get("model.norm.weight").to(dev).float()
mean = W.mean(0)
X = W - mean
C = (X.T.double() @ X.double()) / (n_real - 1)
Cn = C * g.double()[:, None] * g.double()[None, :]
w = torch.linalg.eigvalsh(Cn)
print(f"{model}: gamma rows {n_real}/{get('lm_head.weight').shape[0]}, d={W.shape[1]}, C_norm eig max {w[-1]:.4g} "
      f"min {w[0]:.4g} PR {(w.sum() ** 2 / (w ** 2).sum()):.1f}")
np.save(out, Cn.float().cpu().numpy())
