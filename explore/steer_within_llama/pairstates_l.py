"""Within-language word-order direction for Llama-3.1-8B (PREREG.md). GPU. No steering.
Same pairs, pooling (extract.Capture: mean over content tokens, sink dropped) and estimator (weighted least squares
separating word order from twin unnaturalness, languages weighted equally; explore/steer_within/pairstates.wls_order)
as the Qwen study. Outputs OUT/dW.npz (order_all, unnat_all [L+1, d]; order_ho_<lang>; cell means) and
OUT/dW_info.json (split-half reliability, norms, cos(order, unnat) per layer).
usage: python pairstates_l.py --pairs pairs.jsonl --out OUT [--model meta-llama/Llama-3.1-8B] [--limit N]"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "steer_within"))
from extract import locate, sink_token_id, Capture
from pairstates import wls_order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
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
    order = np.argsort([len(s) for s in seqs])
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
            H[ib, l] = cap.out[l][0].cpu().numpy()
    cap.close()
    print(f"states: {len(seqs)} sequences x {nL} layers ({time.time() - t0:.0f}s)", flush=True)
    diff = H[0::2] - H[1::2]
    s = np.array([1.0 if r["orig_order"] == "VO" else -1.0 for r in rows])
    lang = np.array([r["lang"] for r in rows])
    out = {}
    out["order_all"], out["unnat_all"] = wls_order(diff, s, lang)
    for l in sorted(set(lang)):
        out[f"order_ho_{l}"], _ = wls_order(diff, s, lang, keep=np.where(lang != l)[0])
        for o, so in (("VO", 1.0), ("OV", -1.0)):
            m = (lang == l) & (s == so)
            if m.any():
                out[f"cell_{l}_{o}"] = diff[m].mean(0)
    rng = np.random.default_rng(0)
    half = np.zeros(len(rows), bool)
    for l in set(lang):
        ii = np.where(lang == l)[0]
        half[rng.permutation(ii)[:len(ii) // 2]] = True
    a, _ = wls_order(diff, s, lang, keep=np.where(half)[0])
    b, _ = wls_order(diff, s, lang, keep=np.where(~half)[0])
    cos = lambda x, y: float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
    info = dict(model=args.model, n_pairs=len(rows), langs=sorted(set(lang)), n_layers=nL,
                split_half_cos=[cos(a[l], b[l]) for l in range(nL)],
                norm_order=[float(np.linalg.norm(out["order_all"][l])) for l in range(nL)],
                norm_unnat=[float(np.linalg.norm(out["unnat_all"][l])) for l in range(nL)],
                cos_order_unnat=[cos(out["order_all"][l], out["unnat_all"][l]) for l in range(nL)])
    np.savez(os.path.join(args.out, "dW.npz"), **{k: v.astype(np.float32) for k, v in out.items()})
    json.dump(info, open(os.path.join(args.out, "dW_info.json"), "w"), indent=1)
    print(f"d_order: split-half cos at L16 {info['split_half_cos'][16]:+.3f} | norm {info['norm_order'][16]:.3f}", flush=True)


if __name__ == "__main__":
    main()
