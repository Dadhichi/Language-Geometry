"""Steering vectors for Llama-3.1-8B at layer 16 (PREREG.md), CPU.
1. beta_OV, beta_IE and 24 random span directions per language: explore/steer_gen2/directions2.py on the 34 Llama dev
   centroids at layer 16 (held out: OLS over the other 33 languages; random seed and procedure as in steer_gen2).
2. Within-language direction "w": d_W at layer 16 from the Llama pair study (explore/steer_within_llama; held out for
   German, Russian and English, the pair languages), rescaled to that language's ||beta_OV||.
usage: python make_dirs_l.py DERIVED_L34_DIR DW_LLAMA_NPZ OUT.npz"""
import os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "steer_gen2"))
from make_prompts2 import LANGS

LAYER = 16


def main():
    derived, dw_path, out = sys.argv[1:4]
    base = out + ".base.npz"
    subprocess.run([sys.executable, os.path.join(HERE, "..", "steer_gen2", "directions2.py"), derived, base,
                    "--R", "24", "--layer", str(LAYER)], check=True)
    d = dict(np.load(base))
    dW = np.load(dw_path)
    cos = lambda x, y: float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
    for g in LANGS:
        v = dW[f"order_ho_{g}"][LAYER] if f"order_ho_{g}" in dW.files else dW["order_all"][LAYER]
        b = d[f"L{LAYER}_{g}_ov"]
        d[f"L{LAYER}_{g}_w"] = (v / np.linalg.norm(v) * np.linalg.norm(b)).astype(np.float32)
        print(f"{g}: ||beta_OV|| {np.linalg.norm(b):.3f} | ||d_W|| natural {np.linalg.norm(v):.3f} | cos(d_W, beta_OV) {cos(v, b):+.3f}")
    np.savez(out, **d)
    os.remove(base)
    os.replace(os.path.splitext(base)[0] + "_info.json", os.path.splitext(out)[0] + "_info.json")


if __name__ == "__main__":
    main()
