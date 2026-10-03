"""Free-generation steering in Llama-3.1-8B (PREREG.md). GPU.
Same prompts (explore/steer_gen2/prompts2.jsonl), decoding, language ID and object-type parse as the Qwen studies;
generation is the frozen explore/steer_gen/gen_ov.generate with the hook moved to the output of block 16 (layer 16
of 32, relative depth 0.5 as Qwen's layer 14 of 28). Conditions: unsteered; beta_OV ("ov") and the within-language
direction ("w") at k = -3, -2, -1, +1, +2, +3; 24 random directions at k = -2, +2. 61 conditions, 82,350 continuations.
usage: python gen_llama.py --prompts prompts2.jsonl --dirs steer_dirsL.npz --out OUT [--limit N] [--n_rand 24]"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "steer_gen2"))
import gen_ov2 as G2

G = G2.G
LAYER = 16


def conditions(n_rand=24, layer=LAYER):
    c = [dict(layer=None, kind="base", k=0, r=-1)]
    c += [dict(layer=layer, kind=kd, k=k, r=-1) for kd in ("ov", "w") for k in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)]
    c += [dict(layer=layer, kind="rand", k=s * 2.0, r=r) for r in range(n_rand) for s in (-1, 1)]
    return c


def generate(args, prompts, conds, ckpt=None, layer=LAYER):
    """explore/steer_gen/gen_ov.generate, with the steering hook on block `layer` (1-based output index)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
                                                 token=os.environ.get("HF_TOKEN"),
                                                 device_map={"": 0} if dev.type == "cuda" else None).eval()
    blocks = G.locate(model)[0]
    dirs = np.load(args.dirs)
    steer = {"vec": None}

    def hook(_m, _i, out):
        if steer["vec"] is None:
            return out
        if isinstance(out, (tuple, list)):
            return (out[0] + steer["vec"],) + tuple(out[1:])
        return out + steer["vec"]
    handle = blocks[layer - 1].register_forward_hook(hook)
    assert all(c["layer"] in (None, layer) for c in conds)
    langs = sorted({p["lang"] for p in prompts})
    out, done = [], set()
    if ckpt and os.path.exists(ckpt):
        prev = [json.loads(l) for l in open(ckpt, encoding="utf-8")]
        n_per = {}
        for g in prev:
            n_per[g["cond"]] = n_per.get(g["cond"], 0) + 1
        done = {c for c, n in n_per.items() if n == len(prompts)}
        out = [g for g in prev if g["cond"] in done]
        with open(ckpt, "w", encoding="utf-8") as f:
            for g in out:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"resume: {len(done)} complete conditions loaded from {ckpt}", flush=True)
    t0 = time.time()
    for ci, c in enumerate(conds):
        if ci in done:
            continue
        n0 = len(out)
        for lang in langs:
            idx = [i for i, p in enumerate(prompts) if p["lang"] == lang]
            if c["kind"] == "base":
                steer["vec"] = None
            else:
                v = dirs[f"L{layer}_{lang}_{c['kind']}"]
                v = v[c["r"]] if c["kind"] == "rand" else v
                steer["vec"] = torch.as_tensor(c["k"] * v, device=dev, dtype=model.dtype)
            for s in range(0, len(idx), args.batch):
                b = idx[s:s + args.batch]
                enc = tok([prompts[i]["prompt"] for i in b], return_tensors="pt", padding=True).to(dev)
                with torch.inference_mode():
                    g = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
                for i, row in zip(b, g[:, enc["input_ids"].shape[1]:]):
                    cont = tok.decode(row, skip_special_tokens=True).split("\n")[0]
                    out.append(dict(cond=ci, lang=lang, sent=prompts[i]["sent"], cont=cont))
        if ckpt:
            with open(ckpt, "a", encoding="utf-8") as f:
                for g in out[n0:]:
                    f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"gen cond {ci + 1}/{len(conds)} {c} ({time.time() - t0:.0f}s) e.g. {out[-1]['cont'][:60]!r}", flush=True)
    handle.remove()
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--n_rand", type=int, default=24)
    ap.add_argument("--max_new", type=int, default=40)
    ap.add_argument("--batch", type=int, default=75)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    prompts = [json.loads(l) for l in open(args.prompts, encoding="utf-8")]
    if args.limit:
        cnt = {}
        prompts = [p for p in prompts if cnt.setdefault(p["lang"], 0) < args.limit and not cnt.__setitem__(p["lang"], cnt[p["lang"]] + 1)]
    conds = conditions(args.n_rand)
    json.dump(conds, open(os.path.join(args.out, "conditions.json"), "w"), indent=1)
    gens = generate(args, prompts, conds, ckpt=os.path.join(args.out, "gen_raw.jsonl"))
    gens.sort(key=lambda g: (g["cond"], g["lang"], g["sent"]))
    gens = G.lid_match(gens)
    gens = G2.parse(args, prompts, gens)
    with open(os.path.join(args.out, "gen.jsonl"), "w", encoding="utf-8") as f:
        for g in gens:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print("done")


if __name__ == "__main__":
    main()
