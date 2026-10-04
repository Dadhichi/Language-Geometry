"""EXPLORATORY CPU check: bias/noise split of the divergence of each language from its English parallel (Qwen2.5-7B,
12 languages, FLORES devtest, mean pooling, layers 0..28 step 2). Framework F1 (offset bias) vs F2 (lossy encoding)
in PROPOSAL.md.

For language l and layer L, with centroids from dev and sentences from devtest:
  total_l  = mean_s ||x_ls - x_en,s||_M^2
  bias_l   = ||mu_l - mu_en||_M^2                      (removed exactly by an offset swap)
  noise_l  = mean_s ||(x_ls - mu_l) - (x_en,s - mu_en)||_M^2  (what an offset swap cannot remove)
  bias share = bias / (bias + noise)
M = identity (euc) and M = whitened (pooled within-language covariance of dev, shrinkage 0.5, as in the article).
Also the noise of a language relative to English's own sentence spread: noise_l / mean_s ||x_en,s - mu_en||^2.
Writes biasnoise12.json. Nothing here is pre-registered.
"""
import json, os
import numpy as np

ROOT = "C:/Users/ASUS/Documents/lang-geom"
LANGS = ["eng", "deu", "fra", "spa", "rus", "hin", "arb", "zho", "jpn", "tur", "vie", "ind"]
LAYERS = list(range(0, 29, 2))


def main():
    out = {"langs": LANGS, "layers": LAYERS, "euc": [], "wh": []}
    for L in LAYERS:
        D = np.asarray(np.load(f"{ROOT}/mean_dev_L{L}.f16.npy", mmap_mode="r"), dtype=np.float64)
        T = np.asarray(np.load(f"{ROOT}/mean_devtest_L{L}.f16.npy", mmap_mode="r"), dtype=np.float64)
        mu = D.mean(1)
        Y = (D - mu[:, None]).reshape(-1, D.shape[2])
        W = Y.T @ Y / (len(Y) - len(LANGS))
        del Y
        w, V = np.linalg.eigh(W)
        w = np.clip(w, 0, None)
        s = np.sqrt(0.5 * w + 0.5 * w.mean())
        rows = {}
        for geom in ("euc", "wh"):
            f = (lambda X: X) if geom == "euc" else (lambda X: (X @ V) / s)
            mut = f(mu)
            Tc = f(T - mu[:, None])
            en_spread = (Tc[0] ** 2).sum(1).mean()
            r = []
            for i in range(len(LANGS)):
                bias = ((mut[i] - mut[0]) ** 2).sum()
                noise = ((Tc[i] - Tc[0]) ** 2).sum(1).mean()
                r.append({"bias": float(bias), "noise": float(noise), "share": float(bias / (bias + noise)) if i else 0.0,
                          "noise_rel_en": float(noise / en_spread)})
            rows[geom] = r
        out["euc"].append(rows["euc"])
        out["wh"].append(rows["wh"])
        print(f"L{L:2d} bias share (euc) " + " ".join(f"{LANGS[i]}:{rows['euc'][i]['share']:.2f}" for i in range(1, 12)) +
              " | noise/en-spread (wh) " + " ".join(f"{rows['wh'][i]['noise_rel_en']:.2f}" for i in range(1, 12)), flush=True)
        del D, T
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "biasnoise12.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
