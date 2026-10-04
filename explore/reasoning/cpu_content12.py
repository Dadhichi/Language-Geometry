"""EXPLORATORY CPU check for the reasoning-gap proposal (PROPOSAL.md, section 'CPU checks').

Uses the 12-language sentence-level Qwen2.5-7B data (FLORES dev/devtest, mean pooling, layers 0..28 step 2):
  C:/Users/ASUS/Documents/lang-geom/mean_{dev,devtest}_L{l}.f16.npy  [12, n, 3584], ntok_{split}.npy
Per layer and language l (vs English), on devtest with centroids from dev:
  cos_en  : mean cosine between centred parallel sentences, cos(x_ls - mu_l, x_en,s - mu_en)
  p1_en   : retrieval P@1 l -> en among 1,012 devtest sentences (centred cosine)
  r2_en   : held-out R^2 of a dual ridge map from centred l-vectors to centred English vectors (fit dev, score devtest;
            ridge strength picked on a dev split). 'Lossy encoding' proxy: share of English content variance that is
            linearly recoverable from language l.
  r2_en_sq: same after rescaling each centred sentence vector by sqrt(T_s / mean T_l) (undoes mean-pool dilution).
Length confound test ('is the English axis a pooling/position artefact?'):
  axis e = unit(mu_en - mean_l mu_l) (dev). Between-language slope: projection of the 12 centroids on e against mean
  log T_l. Within-language slope: projection of each sentence on e against its own log T_s, pooled within languages.
  If the within slope explains the between slope, the fertility ordering of offsets could be an artefact of averaging
  position-dependent states; if within ~ 0, it is a property of the language.
Norms: ||mu_l||, ||mu_l - mean mu||, rms ||x_s - mu_l|| (sentence-level content+noise).
Writes content12.json. Nothing here is pre-registered.
"""
import json, os
import numpy as np

ROOT = "C:/Users/ASUS/Documents/lang-geom"
LANGS = ["eng", "deu", "fra", "spa", "rus", "hin", "arb", "zho", "jpn", "tur", "vie", "ind"]
LAYERS = list(range(0, 29, 2))
ALPHAS = (0.01, 0.03, 0.1, 0.3, 1.0)


def load(split, L):
    return np.load(f"{ROOT}/mean_{split}_L{L}.f16.npy", mmap_mode="r")


def dual_ridge_r2(Xd, Yd, Xt, Yt, alpha):
    G = Xd @ Xd.T
    lam = alpha * np.trace(G) / len(G)
    A = np.linalg.solve(G + lam * np.eye(len(G)), Yd)
    P = (Xt @ Xd.T) @ A
    return 1 - ((Yt - P) ** 2).sum() / (Yt ** 2).sum()


def pick_alpha_r2(Xd, Yd, Xt, Yt):
    n = len(Xd)
    m = int(0.8 * n)
    best = max(ALPHAS, key=lambda a: dual_ridge_r2(Xd[:m], Yd[:m], Xd[m:], Yd[m:], a))
    return dual_ridge_r2(Xd, Yd, Xt, Yt, best), best


def main():
    nt = {s: np.load(f"{ROOT}/ntok_{s}.npy").astype(float) for s in ("dev", "devtest")}
    out = {"langs": LANGS, "layers": LAYERS, "fert": nt["dev"].mean(1).tolist()}
    keys = ("cos_en", "p1_en", "r2_en", "r2_en_sq", "norm_mu", "norm_a", "rms_within")
    for k in keys:
        out[k] = []
    out["len_between_slope"], out["len_within_slope"], out["len_within_slope_by_lang"] = [], [], []
    for L in LAYERS:
        D = np.asarray(load("dev", L), dtype=np.float64)
        T = np.asarray(load("devtest", L), dtype=np.float64)
        mu = D.mean(1)                                   # [12, d]
        Dc = D - mu[:, None]
        Tc = T - mu[:, None]
        row = {k: [] for k in keys}
        for i in range(len(LANGS)):
            a, b = Tc[i], Tc[0]
            an = a / np.linalg.norm(a, axis=1, keepdims=True)
            bn = b / np.linalg.norm(b, axis=1, keepdims=True)
            S = an @ bn.T
            row["cos_en"].append(float(np.diag(S).mean()))
            row["p1_en"].append(float((S.argmax(1) == np.arange(len(S))).mean()))
            if i == 0:
                row["r2_en"].append(1.0)
                row["r2_en_sq"].append(1.0)
            else:
                r2, _ = pick_alpha_r2(Dc[i], Dc[0], Tc[i], Tc[0])
                row["r2_en"].append(float(r2))
                sd = np.sqrt(nt["dev"][i] / nt["dev"][i].mean())[:, None]
                st = np.sqrt(nt["devtest"][i] / nt["devtest"][i].mean())[:, None]
                sd0 = np.sqrt(nt["dev"][0] / nt["dev"][0].mean())[:, None]
                st0 = np.sqrt(nt["devtest"][0] / nt["devtest"][0].mean())[:, None]
                r2s, _ = pick_alpha_r2(Dc[i] * sd, Dc[0] * sd0, Tc[i] * st, Tc[0] * st0)
                row["r2_en_sq"].append(float(r2s))
            row["norm_mu"].append(float(np.linalg.norm(mu[i])))
            row["norm_a"].append(float(np.linalg.norm(mu[i] - mu.mean(0))))
            row["rms_within"].append(float(np.sqrt((Dc[i] ** 2).sum(1).mean())))
        for k in keys:
            out[k].append(row[k])
        # length confound on the English axis
        e = mu[0] - mu.mean(0)
        e /= np.linalg.norm(e)
        projc = (mu - mu.mean(0)) @ e                    # centroid projections
        lT = np.log(nt["dev"])                           # [12, n]
        lTm = lT.mean(1)
        bb = np.polyfit(lTm, projc, 1)[0]
        num, den, byl = 0.0, 0.0, []
        for i in range(len(LANGS)):
            p = Dc[i] @ e
            x = lT[i] - lTm[i]
            num += (x * p).sum()
            den += (x * x).sum()
            byl.append(float((x * p).sum() / (x * x).sum()))
        out["len_between_slope"].append(float(bb))
        out["len_within_slope"].append(float(num / den))
        out["len_within_slope_by_lang"].append(byl)
        print(f"L{L:2d} cos_en " + " ".join(f"{x:.2f}" for x in row["cos_en"]) +
              f" | r2_en " + " ".join(f"{x:.2f}" for x in row["r2_en"][1:]) +
              f" | slope between {bb:+.2f} within {num / den:+.2f}", flush=True)
        del D, T, Dc, Tc
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "content12.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
