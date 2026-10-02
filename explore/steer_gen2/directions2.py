"""Layer-14 steering directions for the 9 languages of the confirmatory test, by the same procedure as
explore/steer_ov/directions.py: for each test language h, OLS over the other 33 languages of the 34-language dev
centroids, A_j = b0 + b_OV OV_j + b_IE IE_j + sum_s b_s script_s(j) + b_f log fert_j + e; beta_OV is never fit on
the language it steers. IE control rescaled to ||beta_OV||; R random unit vectors in the span of the 33 centred
centroids, scaled to ||beta_OV|| (RandomState(0), languages in LANGS order).
usage: python directions2.py DERIVED34_DIR OUT.npz [--R 24]"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prereg34"))
import lib34 as T
from make_prompts2 import LANGS

IE = set(T.GLOTTO[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("derived")
    ap.add_argument("out")
    ap.add_argument("--R", type=int, default=24)
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()
    cent = np.load(os.path.join(args.derived, "centroids.npz"))
    tok = np.load(os.path.join(args.derived, "tokstats.npz"))
    assert [str(x) for x in tok["langs"]] == T.LANGS
    ov = np.array([i in T.OV for i in range(T.N)], float)
    ie = np.array([i in IE for i in range(T.N)], float)
    scr = np.column_stack([[float(i in S) for i in range(T.N)] for S in T.SCRIPT_SPLITS])
    X = np.column_stack([np.ones(T.N), ov, ie, scr, np.log(tok["fert"])])
    A = cent[f"dev_L{args.layer}"].astype(np.float64)
    rs = np.random.RandomState(0)
    out, info = {}, {}
    for code in LANGS:
        h = T.LANGS.index(code)
        keep = [j for j in range(T.N) if j != h]
        B, *_ = np.linalg.lstsq(X[keep], A[keep], rcond=None)
        b_ov, b_ie = B[1], B[2]
        _, s, Vt = np.linalg.svd(A[keep] - A[keep].mean(0), full_matrices=False)
        span = Vt[s > s[0] * 1e-8]
        R = rs.normal(size=(args.R, span.shape[0])) @ span
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        nrm = np.linalg.norm(b_ov)
        L = f"L{args.layer}_{code}"
        out[f"{L}_ov"] = b_ov.astype(np.float32)
        out[f"{L}_ie"] = (b_ie / np.linalg.norm(b_ie) * nrm).astype(np.float32)
        out[f"{L}_rand"] = (R * nrm).astype(np.float32)
        proj = (A @ b_ov) / nrm ** 2
        info[L] = dict(norm_ov=float(nrm), cos_ov_ie=float(b_ov @ b_ie / nrm / np.linalg.norm(b_ie)),
                       heldout_proj=float(proj[h]), mean_proj_OV=float(proj[ov == 1].mean()),
                       mean_proj_VO=float(proj[ov == 0].mean()))
        print(f"{code}: ||beta_OV|| {nrm:.3f} | cos(OV, IE) {info[L]['cos_ov_ie']:+.3f} | held-out projection "
              f"{info[L]['heldout_proj']:+.2f} (OV mean {info[L]['mean_proj_OV']:+.2f}, VO mean {info[L]['mean_proj_VO']:+.2f})")
    np.savez(args.out, **out)
    json.dump(info, open(os.path.splitext(args.out)[0] + "_info.json", "w"), indent=1)


if __name__ == "__main__":
    main()
