"""Within-language word-order direction (PREREG.md). GPU.
1. Mean-pooled residual states (content tokens, sink dropped; extract.Capture/pool, the same pooling as the language
   centroids) at every layer for both twins of the 999 UD minimal pairs (explore/steer_ov/pairs.jsonl).
2. Per pair p: diff_p = h(OV twin) - h(VO twin). Weighted least squares per layer and coordinate,
       diff_p = d_order + s_p * d_unnat + e_p,   s_p = +1 if the attested order is VO (the OV twin is the unnatural one),
                                                 s_p = -1 if the attested order is OV,
   weights 1 / n_pairs(language) so every language counts equally. d_order is the within-language word-order
   direction; d_unnat absorbs "this twin is the scrambled one". Fit on all 8 pair languages and, for each pair
   language, with that language held out.
3. Steering vectors for the 9 generation languages (explore/steer_gen2): kind "w" = d_order at layer 14 (held out
   if the language has pairs), rescaled to that language's ||beta_OV||; kind "ov" copied from steer_dirs2.npz.
Outputs OUT/dW.npz (order_all, unnat_all [L+1, d]; order_ho_<lang>; cell means), OUT/steer_dirsW.npz, OUT/dW_info.json.
usage: python pairstates.py --pairs pairs.jsonl --dirs2 steer_dirs2.npz --out OUT [--limit N]"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from extract import locate, sink_token_id, Capture

GEN_LANGS = ["deu_Latn", "nld_Latn", "rus_Cyrl", "ukr_Cyrl", "pol_Latn", "hrv_Latn", "eng_Latn", "spa_Latn", "kor_Hang"]


def wls_order(diff, s, lang, keep=None):
    """diff [P, L, d], s [P] in {+1,-1}, lang [P] codes. Returns d_order, d_unnat [L, d] (weights 1/n_lang)."""
    idx = np.arange(len(s)) if keep is None else np.asarray(keep)
    ls = lang[idx]
    w = np.array([1.0 / (ls == l).sum() for l in ls])
    X = np.column_stack([np.ones(len(idx)), s[idx]])
    XtW = X.T * w                                                     # [2, P]
    coef = np.linalg.solve(XtW @ X, XtW @ diff[idx].reshape(len(idx), -1))   # [2, L*d]
    return coef[0].reshape(diff.shape[1:]), coef[1].reshape(diff.shape[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--dirs2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    if args.limit:
        seen = {}
        rows = [r for r in rows if seen.setdefault(r["lang"], 0) < args.limit and not seen.__setitem__(r["lang"], seen[r["lang"]] + 1)]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
                                                 token=os.environ.get("HF_TOKEN"),
                                                 device_map={"": 0} if dev.type == "cuda" else None).eval()
    blocks, embed, _, _, _ = locate(model)
    cap = Capture(blocks, embed)
    sink, _ = sink_token_id(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    seqs = [[sink] + tok(r[k], add_special_tokens=False)["input_ids"] for r in rows for k in ("ov", "vo")]
    nL = len(blocks) + 1
    H = np.zeros((len(seqs), nL, model.config.hidden_size), np.float32)
    order = np.argsort([len(s) for s in seqs])                         # length-sorted batches
    t0 = time.time()
    for b in range(0, len(order), args.batch):
        ib = order[b:b + args.batch]
        T = max(len(seqs[i]) for i in ib)
        ids = torch.full((len(ib), T), pad, dtype=torch.long)
        mask = torch.zeros((len(ib), T), dtype=torch.bool)
        for j, i in enumerate(ib):
            ids[j, :len(seqs[i])] = torch.tensor(seqs[i])
            mask[j, :len(seqs[i])] = True
        cap.mask = mask.to(dev)
        with torch.inference_mode():
            model(input_ids=ids.to(dev), attention_mask=mask.to(dev))
        for l in range(nL):
            H[ib, l] = cap.out[l][0].cpu().numpy()                       # pooled mean [B, d]
    cap.close()
    print(f"states: {len(seqs)} sequences x {nL} layers ({time.time() - t0:.0f}s)", flush=True)
    diff = H[0::2] - H[1::2]                                            # [P, L, d]  OV twin - VO twin
    s = np.array([1.0 if r["orig_order"] == "VO" else -1.0 for r in rows])
    lang = np.array([r["lang"] for r in rows])
    out = {}
    out["order_all"], out["unnat_all"] = wls_order(diff, s, lang)
    for l in sorted(set(lang)):
        out[f"order_ho_{l}"], _ = wls_order(diff, s, lang, keep=np.where(lang != l)[0])
        for o, so in (("VO", 1.0), ("OV", -1.0)):                      # cell means, for any later re-estimation
            m = (lang == l) & (s == so)
            if m.any():
                out[f"cell_{l}_{o}"] = diff[m].mean(0)
    # split-half reliability of d_order at every layer (pairs split at random within each language)
    rng = np.random.default_rng(0)
    half = np.zeros(len(rows), bool)
    for l in set(lang):
        ii = np.where(lang == l)[0]
        half[rng.permutation(ii)[:len(ii) // 2]] = True
    a, _ = wls_order(diff, s, lang, keep=np.where(half)[0])
    b, _ = wls_order(diff, s, lang, keep=np.where(~half)[0])
    cos = lambda x, y: float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
    info = dict(n_pairs=len(rows), langs=sorted(set(lang)), n_layers=nL,
                split_half_cos=[cos(a[l], b[l]) for l in range(nL)],
                norm_order=[float(np.linalg.norm(out["order_all"][l])) for l in range(nL)],
                norm_unnat=[float(np.linalg.norm(out["unnat_all"][l])) for l in range(nL)],
                cos_order_unnat=[cos(out["order_all"][l], out["unnat_all"][l]) for l in range(nL)])
    np.savez(os.path.join(args.out, "dW.npz"), **{k: v.astype(np.float32) for k, v in out.items()})
    # steering vectors at layer 14
    d2 = np.load(args.dirs2)
    sv = {}
    for g in GEN_LANGS:                          # held out when the language has pairs (German, Russian, English)
        v = out.get(f"order_ho_{g}", out["order_all"])[args.layer]
        b_ov = d2[f"L{args.layer}_{g}_ov"]
        sv[f"L{args.layer}_{g}_w"] = (v / np.linalg.norm(v) * np.linalg.norm(b_ov)).astype(np.float32)
        sv[f"L{args.layer}_{g}_ov"] = b_ov
        info[f"cos_w_ov_{g}"] = cos(v, b_ov)
    np.savez(os.path.join(args.out, "steer_dirsW.npz"), **sv)
    json.dump(info, open(os.path.join(args.out, "dW_info.json"), "w"), indent=1)
    print(f"d_order L{args.layer}: norm {info['norm_order'][args.layer]:.3f} | split-half cos {info['split_half_cos'][args.layer]:+.3f} | "
          f"cos(order, unnat) {info['cos_order_unnat'][args.layer]:+.3f} | cos(w, beta_OV held out): " +
          " ".join(f"{g[:3]} {info[f'cos_w_ov_{g}']:+.3f}" for g in GEN_LANGS), flush=True)


if __name__ == "__main__":
    main()
