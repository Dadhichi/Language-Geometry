"""PRE-REGISTERED analysis: does the within-language word-order direction align with beta_OV in Llama-3.1-8B?
(PREREG.md). Same test as claim A of explore/steer_within (functions imported from analysis_w.py), at layer 16 of
32 blocks (relative depth 0.5, as Qwen's layer 14 of 28).
usage: python analysis_wl.py W_OUT_DIR DERIVED_L34_DIR   -> results_wl.json + report"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "steer_within"))
sys.path.insert(0, os.path.join(HERE, "..", "prereg34"))
from analysis_w import betas, span_basis, alignment
import lib34 as T

LAYER = 16


def analyze(W_OUT, DER):
    dW = np.load(os.path.join(W_OUT, "dW.npz"))
    cent = np.load(os.path.join(DER, "centroids.npz"))
    tok = np.load(os.path.join(DER, "tokstats.npz"))
    assert [str(x) for x in tok["langs"]] == T.LANGS
    rng = np.random.default_rng(0)
    prof = []
    nL = dW["order_all"].shape[0]
    for L in range(nL):
        A = cent[f"dev_L{L}"].astype(np.float64)
        b_ov, b_ie = betas(A, T, tok["fert"])
        al = alignment(dW["order_all"][L].astype(np.float64), b_ov, b_ie, span_basis(A), rng)
        al["layer"], al["depth"] = L, L / (nL - 1)
        al["cos_unnat_ov"] = float(dW["unnat_all"][L] @ b_ov / np.linalg.norm(dW["unnat_all"][L]) / np.linalg.norm(b_ov))
        prof.append(al)
    a = prof[LAYER]
    res = dict(model="meta-llama/Llama-3.1-8B", layer=LAYER, A=dict(a, claim=bool(a["cos_ov"] > 0 and a["cos_ov"] > a["null_q999"])),
               A_profile=prof, dW_info=json.load(open(os.path.join(W_OUT, "dW_info.json"))))
    qp = os.path.join(HERE, "..", "steer_within", "results_w.json")             # Qwen profile, for the descriptive comparison
    if os.path.exists(qp):
        q = json.load(open(qp))["A_profile"]
        res["qwen_profile_by_depth"] = [dict(depth=p["layer"] / (len(q) - 1), cos_ov=p["cos_ov"]) for p in q]
    return res


def report(res):
    A = res["A"]
    print(f"Llama A alignment L{LAYER}: cos(d_W, beta_OV) {A['cos_ov']:+.3f} | null 99.9% {A['null_q999']:+.3f} (sd {A['null_sd']:.3f}) | "
          f"p {A['p']:.4f} | cos(d_W, beta_IE) {A['cos_ie']:+.3f} | span share {A['span_share']:.3f} | within-span cos "
          f"{A['cos_ov_within_span']:+.3f} | CLAIM {A['claim']}")
    print("  profile cos_ov by layer:", " ".join(f"{p['layer']}:{p['cos_ov']:+.2f}" for p in res["A_profile"]))
    print("  above null 99.9% at layers:", [p["layer"] for p in res["A_profile"] if p["cos_ov"] > p["null_q999"]])
    print("  split-half by layer:", [round(x, 2) for x in res["dW_info"]["split_half_cos"]])


if __name__ == "__main__":
    res = analyze(sys.argv[1], sys.argv[2])
    report(res)
    clean = lambda o: {str(k): clean(v) for k, v in o.items()} if isinstance(o, dict) else [clean(v) for v in o] if isinstance(o, list) \
        else (None if isinstance(o, float) and not np.isfinite(o) else float(o) if isinstance(o, np.floating) else bool(o) if isinstance(o, np.bool_) else o)
    json.dump(clean(res), open(os.path.join(HERE, "results_wl.json"), "w"), indent=1)
