"""Causal steering test of the OV word-order direction (PREREG.md). Runs on a GPU.
For each condition (layer l, direction kind, strength k, random index) and every minimal pair, add k * v_lang to the
residual stream output of block l at every position (v_lang = the direction estimated with that pair's language held
out, steer_dirs.npz) and score both variants: log p(content tokens | sink) summed. Output scores.npz:
cond_{c} -> [n_pairs, 2] (ov, vo) and conditions.json.
usage: python steer_score.py --pairs pairs.jsonl --dirs steer_dirs.npz --out OUT_DIR [--model Qwen/Qwen2.5-7B]"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from extract import locate, sink_token_id

KS = [-2, -1, 1, 2]
KS_RAND = [-2, 2]


def conditions(primary=14, secondary=(8, 20), n_rand=32):
    c = [dict(layer=None, kind="base", k=0, r=-1)]
    for l in [primary, *secondary]:
        for kind in ("ov", "ie"):
            c += [dict(layer=l, kind=kind, k=k, r=-1) for k in KS]
    c += [dict(layer=primary, kind="rand", k=k, r=r) for r in range(n_rand) for k in KS_RAND]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--n_rand", type=int, default=32)
    ap.add_argument("--batch_tokens", type=int, default=12000)
    ap.add_argument("--limit", type=int, default=0, help="debug: pairs per language")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    if args.limit:
        seen = {}
        rows = [r for r in rows if seen.setdefault(r["lang"], 0) < args.limit and not seen.__setitem__(r["lang"], seen[r["lang"]] + 1)]
    dirs = np.load(args.dirs)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
                                                 token=os.environ.get("HF_TOKEN"),
                                                 device_map={"": 0} if dev.type == "cuda" else None).eval()
    blocks, embed, fnorm, head, _ = locate(model)
    base = model.base_model
    sink, _ = sink_token_id(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    W = head.weight                                                            # [V, d]
    seqs = []
    for r in rows:
        for key in ("ov", "vo"):
            seqs.append([sink] + tok(r[key], add_special_tokens=False)["input_ids"])
    langs = sorted({r["lang"] for r in rows})
    by_lang = {l: [i for i, r in enumerate(rows) if r["lang"] == l] for l in langs}
    steer = {"vec": None}

    def hook(_m, _i, out):
        if steer["vec"] is None:
            return out
        if isinstance(out, (tuple, list)):
            return (out[0] + steer["vec"],) + tuple(out[1:])
        return out + steer["vec"]

    def score(idx_seqs):
        res = np.zeros(len(idx_seqs), np.float64)
        lens = np.array([len(seqs[i]) for i in idx_seqs])
        order = np.argsort(-lens, kind="stable")
        s = 0
        while s < len(order):
            b = max(1, min(256, args.batch_tokens // int(lens[order[s]])))
            sel = order[s:s + b]; s += b
            L = int(lens[sel].max())
            ids = torch.full((len(sel), L), pad_id, dtype=torch.long)
            mask = torch.zeros((len(sel), L), dtype=torch.bool)
            for r_, j in enumerate(sel):
                q = seqs[idx_seqs[j]]
                ids[r_, :len(q)] = torch.tensor(q); mask[r_, :len(q)] = True
            ids, mask = ids.to(dev), mask.to(dev)
            with torch.inference_mode():
                hs = base(input_ids=ids, attention_mask=mask.long(), use_cache=False).last_hidden_state
                valid = mask[:, 1:]
                h = hs[:, :-1][valid]                                          # predicts ids[:, 1:][valid]
                tgt = ids[:, 1:][valid]
                lp = torch.cat([torch.log_softmax((c @ W.T).float(), -1).gather(1, t[:, None])[:, 0]
                                for c, t in zip(h.split(2048), tgt.split(2048))])
                rowid = torch.arange(len(sel), device=dev)[:, None].expand(-1, L - 1)[valid]
                tot = torch.zeros(len(sel), device=dev, dtype=torch.float64).index_add_(0, rowid, lp.double())
            res[sel] = tot.cpu().numpy()
        return res

    conds = conditions(n_rand=args.n_rand)
    json.dump(conds, open(os.path.join(args.out, "conditions.json"), "w"), indent=1)
    out = {}
    handles = {}
    t0 = time.time()
    for ci, c in enumerate(conds):
        sc = np.zeros((len(rows), 2))
        for lang in langs:
            pidx = by_lang[lang]
            if c["kind"] == "base":
                steer["vec"] = None
            else:
                l = c["layer"]
                if l not in handles:
                    handles[l] = blocks[l - 1].register_forward_hook(hook)
                v = dirs[f"L{l}_{lang}_{c['kind']}"]
                if c["kind"] == "rand":
                    v = v[c["r"]]
                steer["vec"] = torch.as_tensor(c["k"] * v, device=dev, dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32)
            # only the hook of the current layer may be active: remove others
            for l_, h_ in list(handles.items()):
                if c["kind"] == "base" or l_ != c["layer"]:
                    h_.remove(); del handles[l_]
            seq_idx = [2 * i + j for i in pidx for j in (0, 1)]
            s = score(seq_idx)
            sc[pidx, 0], sc[pidx, 1] = s[0::2], s[1::2]
        out[f"cond_{ci}"] = sc.astype(np.float32)
        if ci % 5 == 0 or ci == len(conds) - 1:
            np.savez(os.path.join(args.out, "scores.npz"), **out)
            print(f"cond {ci + 1}/{len(conds)} {c} mean margin(ov-vo) {np.mean(sc[:, 0] - sc[:, 1]):+.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    np.savez(os.path.join(args.out, "scores.npz"), **out)
    json.dump(dict(model=args.model, n_pairs=len(rows), langs=langs, sink=sink), open(os.path.join(args.out, "meta.json"), "w"))
    print("done")


if __name__ == "__main__":
    main()
