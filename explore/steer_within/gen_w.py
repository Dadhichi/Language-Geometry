"""Free generation steered by the within-language word-order direction (PREREG.md). GPU.
Same prompts (explore/steer_gen2/prompts2.jsonl), generation (explore/steer_gen/gen_ov.generate), language ID and
object-type parse (explore/steer_gen2/gen_ov2.parse) as the confirmatory study. Conditions: unsteered; within-language
direction "w" at k = -3, -2, -1, +1, +2, +3; between-language direction "ov" at k = -2, +2 (re-run in this job for
a paired comparison and a determinism check against steer_gen2).
usage: python gen_w.py --prompts prompts2.jsonl --dirs steer_dirsW.npz --out OUT [--limit N]"""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "steer_gen2"))
import gen_ov2 as G2

G = G2.G


def conditions(layer=14):
    c = [dict(layer=None, kind="base", k=0, r=-1)]
    c += [dict(layer=layer, kind="w", k=k, r=-1) for k in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)]
    c += [dict(layer=layer, kind="ov", k=k, r=-1) for k in (-2.0, 2.0)]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--max_new", type=int, default=40)
    ap.add_argument("--batch", type=int, default=75)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    prompts = [json.loads(l) for l in open(args.prompts, encoding="utf-8")]
    if args.limit:
        cnt = {}
        prompts = [p for p in prompts if cnt.setdefault(p["lang"], 0) < args.limit and not cnt.__setitem__(p["lang"], cnt[p["lang"]] + 1)]
    conds = conditions()
    json.dump(conds, open(os.path.join(args.out, "conditions.json"), "w"), indent=1)
    gens = G.generate(args, prompts, conds, ckpt=os.path.join(args.out, "gen_raw.jsonl"))
    gens.sort(key=lambda g: (g["cond"], g["lang"], g["sent"]))
    gens = G.lid_match(gens)
    gens = G2.parse(args, prompts, gens)
    with open(os.path.join(args.out, "gen.jsonl"), "w", encoding="utf-8") as f:
        for g in gens:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print("done")


if __name__ == "__main__":
    main()
