"""Steering directions from the 34-language mean-pooled centroids (explore/prereg34 derived data), per layer and per
held-out test language h (h excluded from the regression, so the direction is not fit on the language it steers):
    A_j (centroid offset, d) = b0 + b_OV * OV_j + b_IE * IE_j + sum_s b_s * script_s(j) + b_f * log fert_j + e
over the 33 languages j != h (OLS, dev centroids). Directions: beta_OV (OV minus VO, holding family IE, script and
fertility fixed) and beta_IE (control). Random controls: R unit vectors drawn uniformly in the span of the 33
centered centroids, scaled to ||beta_OV||, seed fixed. Steering adds k * direction (k in multiples of that norm).
usage: python directions.py DERIVED34_DIR OUT.npz [--layers 8,14,20] [--R 32]"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prereg34"))
import lib34 as T

TEST = {"eng_Latn": "eng_Latn", "deu_Latn": "deu_Latn", "fra_Latn": "fra_Latn", "rus_Cyrl": "rus_Cyrl",
        "hin_Deva": "hin_Deva", "tur_Latn": "tur_Latn", "jpn_Jpan": "jpn_Jpan", "zho_Hans": "cmn_Hans"}
IE = set(T.GLOTTO[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("derived")
    ap.add_argument("out")
    ap.add_argument("--layers", default="8,14,20")
    ap.add_argument("--R", type=int, default=32)
    args = ap.parse_args()
    cent = np.load(os.path.join(args.derived, "centroids.npz"))
    tok = np.load(os.path.join(args.derived, "tokstats.npz"))
    langs = [str(x) for x in tok["langs"]]
    assert langs == T.LANGS
    ov = np.array([i in T.OV for i in range(T.N)], float)
    ie = np.array([i in IE for i in range(T.N)], float)
    scr = np.column_stack([[float(i in S) for i in range(T.N)] for S in T.SCRIPT_SPLITS])
    lf = np.log(tok["fert"])
    out, info = {}, {}
    rs = np.random.RandomState(0)
    for l in [int(x) for x in args.layers.split(",")]:
        A = cent[f"dev_L{l}"].astype(np.float64)                          # [34, d] centred offsets
        for name, code in TEST.items():
            h = T.LANGS.index(code)
            keep = [j for j in range(T.N) if j != h]
            X = np.column_stack([np.ones(T.N), ov, ie, scr, lf])[keep]
            B, *_ = np.linalg.lstsq(X, A[keep], rcond=None)
            b_ov, b_ie = B[1], B[2]
            U, s, Vt = np.linalg.svd(A[keep] - A[keep].mean(0), full_matrices=False)
            span = Vt[s > s[0] * 1e-8]                                     # [<=32, d]
            R = rs.normal(size=(args.R, span.shape[0])) @ span
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            nrm = np.linalg.norm(b_ov)
            out[f"L{l}_{name}_ov"] = b_ov.astype(np.float32)
            out[f"L{l}_{name}_ie"] = (b_ie / np.linalg.norm(b_ie) * nrm).astype(np.float32)
            out[f"L{l}_{name}_rand"] = (R * nrm).astype(np.float32)
            # descriptive: where does the held-out language's own centroid sit along beta_OV?
            proj = (A @ b_ov) / nrm ** 2
            info[f"L{l}_{name}"] = dict(norm_ov=float(nrm), norm_ie_raw=float(np.linalg.norm(b_ie)),
                                        cos_ov_ie=float(b_ov @ b_ie / nrm / np.linalg.norm(b_ie)),
                                        heldout_proj=float(proj[h]),
                                        mean_proj_OV=float(proj[ov == 1].mean()), mean_proj_VO=float(proj[ov == 0].mean()))
        print(f"L{l}: ||beta_OV|| (eng held out) {info[f'L{l}_eng_Latn']['norm_ov']:.3f}, cos(OV, IE) "
              f"{info[f'L{l}_eng_Latn']['cos_ov_ie']:+.3f}; held-out projections: " +
              " ".join(f"{n[:3]} {info[f'L{l}_{n}']['heldout_proj']:+.2f}" for n in TEST), flush=True)
    np.savez(args.out, **out)
    json.dump(info, open(os.path.splitext(args.out)[0] + "_info.json", "w"), indent=1)


if __name__ == "__main__":
    main()
