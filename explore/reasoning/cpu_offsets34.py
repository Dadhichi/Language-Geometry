"""EXPLORATORY CPU check for the reasoning-gap proposal (PROPOSAL.md, section 'CPU checks').

Question: how far is each language's mid-depth offset from English, and does that distance track tokenizer fertility,
family, or word order in a way that could predict a reasoning-gap ordering?

Reads the existing 34-language cross-half Gram matrices (derive34.py output; no GPU, no new extraction):
  C:/Users/ASUS/Documents/lang-geom/q34/derived34   (Qwen2.5-7B, 29 layers)
  C:/Users/ASUS/Documents/lang-geom/l34/derivedL34  (Llama-3.1-8B, 33 layers)
For each model, geometry (causal, lda05 = whitened eta 0.5, euc) and layer:
  K      = symmetrised cross-half Gram of the 34 offsets (unbiased for a_i^T M a_j)
  star_i = K_ii / mean_j K_jj           (squared distance of language i from the centre of the 34, relative)
  d2en_i = D2(i, eng) / mean_pairs D2   (unbiased squared distance to English, relative)
  cosen_i= K_{i,eng} / sqrt(K_ii K_eng)  (alignment of i's offset with English's offset)
Window = the pre-registered window (Qwen L8-20, Llama L9-23) of trace-normalised Grams, as in the article.
Writes offsets34.json next to this file and prints a summary. Nothing here is pre-registered.
"""
import json, os
import numpy as np
from scipy.stats import spearmanr

ROOT = "C:/Users/ASUS/Documents/lang-geom"
MODELS = {"qwen": (f"{ROOT}/q34/derived34", 29, range(8, 21)),
          "llama": (f"{ROOT}/l34/derivedL34", 33, range(9, 24))}
LANGS = ["eng_Latn", "deu_Latn", "nld_Latn", "swe_Latn", "fra_Latn", "spa_Latn", "por_Latn",
         "rus_Cyrl", "ukr_Cyrl", "pol_Latn", "hrv_Latn", "srp_Cyrl", "hin_Deva", "urd_Arab", "mar_Deva", "pes_Arab",
         "arb_Arab", "heb_Hebr", "mlt_Latn", "tur_Latn", "azj_Latn", "kaz_Cyrl", "fin_Latn", "ekk_Latn",
         "cmn_Hans", "cmn_Hant", "jpn_Jpan", "kor_Hang", "ind_Latn", "fil_Latn", "vie_Latn", "khm_Khmr",
         "tam_Taml", "tel_Telu"]
N = len(LANGS)
EN = 0
IE = set(range(16))
OV = {12, 13, 14, 15, 19, 20, 21, 26, 27, 32, 33}
GEOMS = ("causal", "lda05", "euc")


def symK(g, geom):
    K = g[f"{geom}_dt"].astype(float)
    return 0.5 * (K + K.T)


def stats(K):
    dg = np.diag(K)
    D2 = dg[:, None] + dg[None, :] - 2 * K
    iu = np.triu_indices(N, 1)
    star = dg / dg.mean()
    d2en = D2[:, EN] / D2[iu].mean()
    cosen = K[:, EN] / np.sqrt(np.clip(dg, 1e-30, None) * dg[EN])
    return star, d2en, cosen


def main():
    out = {}
    for m, (path, nl, win) in MODELS.items():
        fert = np.load(f"{path}/tokstats.npz")["fert"]
        lf = np.log(fert)
        res = {"fert": fert.tolist()}
        for geom in GEOMS:
            Kw = None
            prof = {"spearman_d2en_logfert": [], "eng_star_rank": [], "eng_star": [], "zho_star_rank": []}
            per_layer = []
            for L in range(nl):
                K = symK(np.load(f"{path}/grams/L{L}.npz"), geom)
                if L in win:
                    Kw = K / np.trace(K) if Kw is None else Kw + K / np.trace(K)
                star, d2en, cosen = stats(K)
                others = np.arange(1, N)
                prof["spearman_d2en_logfert"].append(float(spearmanr(d2en[others], lf[others])[0]))
                prof["eng_star_rank"].append(int((star < star[EN]).sum()) + 1)   # 1 = closest to centre
                prof["eng_star"].append(float(star[EN]))
                prof["zho_star_rank"].append(int((star < star[24]).sum()) + 1)
                per_layer.append({"star": star.tolist(), "d2en": d2en.tolist(), "cosen": cosen.tolist()})
            Kw /= len(win)
            star, d2en, cosen = stats(Kw)
            others = np.arange(1, N)
            # regression of log star on log fertility over all 34; standardised residuals of eng and cmn_Hans
            X = np.column_stack([np.ones(N), lf])
            y = np.log(star)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ beta
            z = r / r.std(ddof=2)
            # d2en against log fertility, |dlog f|, IE, OV (33 non-English languages); OLS with all four
            Z = np.column_stack([np.ones(N - 1), lf[others], [i in IE for i in others], [i in OV for i in others]]).astype(float)
            b2, *_ = np.linalg.lstsq(Z, d2en[others], rcond=None)
            pred = Z @ b2
            r2 = 1 - ((d2en[others] - pred) ** 2).sum() / ((d2en[others] - d2en[others].mean()) ** 2).sum()
            # partial Spearman of d2en with OV after removing log fertility (rank residuals)
            def resid(a, b):
                A = np.column_stack([np.ones(len(b)), b])
                return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
            rd = resid(d2en[others], lf[others])
            res[geom] = {
                "window": {"star": star.tolist(), "d2en": d2en.tolist(), "cosen": cosen.tolist(),
                           "spearman_d2en_logfert": float(spearmanr(d2en[others], lf[others])[0]),
                           "spearman_star_logfert": float(spearmanr(star, lf)[0]),
                           "spearman_cosen_logfert": float(spearmanr(cosen[others], lf[others])[0]),
                           "eng_star_rank": int((star < star[EN]).sum()) + 1,
                           "logstar_on_logfert_beta": beta.tolist(), "eng_resid_z": float(z[EN]), "cmn_resid_z": float(z[24]),
                           "d2en_ols_coef[1,logf,IE,OV]": b2.tolist(), "d2en_ols_R2": float(r2),
                           "d2en_resid_logf_mean_OV_minus_VO": float(rd[[i - 1 for i in others if i in OV]].mean()
                                                                    - rd[[i - 1 for i in others if i not in OV]].mean()),
                           "d2en_resid_logf_mean_IE_minus_nonIE": float(rd[[i - 1 for i in others if i in IE]].mean()
                                                                        - rd[[i - 1 for i in others if i not in IE]].mean())},
                "profile": prof, "per_layer": per_layer}
        out[m] = res
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "offsets34.json"), "w"), indent=0)

    for m in out:
        print(f"\n===== {m}")
        fert = np.array(out[m]["fert"])
        for geom in GEOMS:
            w = out[m][geom]["window"]
            print(f"-- {geom}: spearman(d2en, log f) = {w['spearman_d2en_logfert']:+.2f}; spearman(star, log f) = "
                  f"{w['spearman_star_logfert']:+.2f}; spearman(cosen, log f) = {w['spearman_cosen_logfert']:+.2f}; "
                  f"eng star rank {w['eng_star_rank']}/34; log-star~log-f residual z: eng {w['eng_resid_z']:+.2f}, "
                  f"cmn {w['cmn_resid_z']:+.2f}")
            print(f"   d2en OLS [1, logf, IE, OV] = {np.round(w['d2en_ols_coef[1,logf,IE,OV]'], 3)}, R2 {w['d2en_ols_R2']:.2f}; "
                  f"resid(logf) OV-VO {w['d2en_resid_logf_mean_OV_minus_VO']:+.3f}, IE-nonIE {w['d2en_resid_logf_mean_IE_minus_nonIE']:+.3f}")
            p = out[m][geom]["profile"]
            print("   per-layer spearman(d2en, log f):", " ".join(f"{x:+.2f}" for x in p["spearman_d2en_logfert"]))
            print("   per-layer eng star rank:", p["eng_star_rank"])
        w = out[m]["causal"]["window"]
        w2 = out[m]["lda05"]["window"]
        order = np.argsort(w["d2en"])
        print("   language  fert  d2en(causal) cosen(causal)  d2en(lda05) cosen(lda05)  star(causal) star(lda05)")
        for i in order:
            print(f"   {LANGS[i]:9s} {fert[i]:6.1f} {w['d2en'][i]:8.3f} {w['cosen'][i]:+8.3f}   {w2['d2en'][i]:8.3f} {w2['cosen'][i]:+8.3f}"
                  f"   {w['star'][i]:7.3f} {w2['star'][i]:7.3f}")


if __name__ == "__main__":
    main()
