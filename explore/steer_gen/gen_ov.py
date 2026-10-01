"""Free-generation steering test (PREREG.md). GPU. Stage 1: for each condition, greedy-continue every prompt (40 new
tokens, cut at the first newline) with k * v added to block-l output at every position (v = the prompt language's
held-out direction from explore/steer_ov/steer_dirs.npz). Stage 2: language-ID each continuation (langid). Stage 3:
Stanza UD parse of prompt + continuation; count verb-object dependencies (UPOS VERB head, deprel obj*) whose verb and
object words both lie in the generated part: OV if the object precedes the verb.
Output OUT/gen.jsonl rows {cond, lang, sent, cont, lid, match, n_ov, n_vo}; OUT/conditions.json.
usage: python gen_ov.py --prompts prompts.jsonl --dirs steer_dirs.npz --out OUT [--n_rand 24]"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from extract import locate

STANZA = {"eng_Latn": "en", "deu_Latn": "de", "fra_Latn": "fr", "rus_Cyrl": "ru", "hin_Deva": "hi",
          "tur_Latn": "tr", "jpn_Jpan": "ja", "zho_Hans": "zh-hans"}
LANGID = {"eng_Latn": "en", "deu_Latn": "de", "fra_Latn": "fr", "rus_Cyrl": "ru", "hin_Deva": "hi",
          "tur_Latn": "tr", "jpn_Jpan": "ja", "zho_Hans": "zh"}
CJK = {"jpn_Jpan", "zho_Hans"}


def conditions(n_rand, kstar, layer=14):
    h = kstar / 2
    c = [dict(layer=None, kind="base", k=0, r=-1)]
    c += [dict(layer=layer, kind="ov", k=k, r=-1) for k in (-kstar, -h, h, kstar)]
    c += [dict(layer=layer, kind="ie", k=k, r=-1) for k in (-kstar, kstar)]
    c += [dict(layer=layer, kind="rand", k=k, r=r) for r in range(n_rand) for k in (-kstar, kstar)]
    return c


CAL_KS = (2.0, 1.0, 0.5)


def calib_conditions(layer=14):
    c = [dict(layer=None, kind="base", k=0, r=-1)]
    c += [dict(layer=layer, kind="rand", k=s * k, r=r) for k in CAL_KS for r in (0, 1, 2) for s in (-1, 1)]
    return c


def lid_match(gens):
    import langid
    langid.set_languages(sorted(set(LANGID.values())))
    for g in gens:
        lid = langid.classify(g["cont"])[0] if len(g["cont"].strip()) >= 3 else "none"
        g["lid"], g["match"] = lid, lid == LANGID[g["lang"]]
    return gens


def generate(args, prompts, conds):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
                                                 token=os.environ.get("HF_TOKEN"),
                                                 device_map={"": 0} if dev.type == "cuda" else None).eval()
    blocks = locate(model)[0]
    dirs = np.load(args.dirs)
    steer = {"vec": None}

    def hook(_m, _i, out):
        if steer["vec"] is None:
            return out
        if isinstance(out, (tuple, list)):
            return (out[0] + steer["vec"],) + tuple(out[1:])
        return out + steer["vec"]
    handle = blocks[13].register_forward_hook(hook)                         # block 14 output (layer index 14)
    assert all(c["layer"] in (None, 14) for c in conds)
    langs = sorted({p["lang"] for p in prompts})
    out = []
    t0 = time.time()
    for ci, c in enumerate(conds):
        for lang in langs:
            idx = [i for i, p in enumerate(prompts) if p["lang"] == lang]
            if c["kind"] == "base":
                steer["vec"] = None
            else:
                v = dirs[f"L14_{lang}_{c['kind']}"]
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
        print(f"gen cond {ci + 1}/{len(conds)} {c} ({time.time() - t0:.0f}s) e.g. {out[-1]['cont'][:60]!r}", flush=True)
    handle.remove()
    del model
    torch.cuda.empty_cache()
    return out


def parse(args, prompts, gens):
    import stanza
    pmap = {(p["lang"], p["sent"]): p["prompt"] for p in prompts}
    lid_match(gens)
    for g in gens:
        g["n_ov"] = g["n_vo"] = 0
    t0 = time.time()
    for lang in sorted(STANZA):
        try:                                 # mwt exists only for some languages (fr, de, tr, ...)
            nlp = stanza.Pipeline(STANZA[lang], processors="tokenize,mwt,pos,lemma,depparse", use_gpu=True,
                                  verbose=False, download_method=None)
        except Exception:
            nlp = stanza.Pipeline(STANZA[lang], processors="tokenize,pos,lemma,depparse", use_gpu=True,
                                  verbose=False, download_method=None)
        rows = [g for g in gens if g["lang"] == lang and g["cont"].strip()]
        sep = "" if lang in CJK else " "
        texts = [pmap[(lang, g["sent"])] + sep + g["cont"].lstrip() for g in rows]
        starts = [len(pmap[(lang, g["sent"])]) + len(sep) for g in rows]
        for a in range(0, len(texts), 256):
            docs = nlp.bulk_process([stanza.Document([], text=t) for t in texts[a:a + 256]])
            for g, d, st in zip(rows[a:a + 256], docs, starts[a:a + 256]):
                for sent in d.sentences:
                    w = {x.id: x for x in sent.words}
                    gen = {x.id: (x.parent.start_char if x.parent.start_char is not None else -1) >= st for x in sent.words}
                    for x in sent.words:
                        if x.deprel and x.deprel.split(":")[0] == "obj" and x.head in w and w[x.head].upos == "VERB":
                            if gen[x.id] and gen[x.head]:
                                if x.id < x.head:
                                    g["n_ov"] += 1
                                else:
                                    g["n_vo"] += 1
        print(f"parsed {lang}: {len(rows)} texts ({time.time() - t0:.0f}s)", flush=True)
        del nlp
    return gens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
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
    # ---- calibration (random directions only; PREREG rule): k* = largest k in CAL_KS whose random-direction
    # language-match rate is >= 80% of the base rate, on the first 20 prompts per language
    cnt = {}
    cal_prompts = [p for p in prompts if cnt.setdefault(p["lang"], 0) < 20 and not cnt.__setitem__(p["lang"], cnt[p["lang"]] + 1)]
    cal_conds = calib_conditions()
    cal = lid_match(generate(args, cal_prompts, cal_conds))
    rate = lambda pred: float(np.mean([g["match"] for g in cal if pred(cal_conds[g["cond"]])]))
    base_rate = rate(lambda c: c["kind"] == "base")
    rates = {k: rate(lambda c, k=k: c["kind"] == "rand" and abs(c["k"]) == k) for k in CAL_KS}
    kstar = next((k for k in CAL_KS if rates[k] >= 0.8 * base_rate), min(CAL_KS))
    json.dump(dict(base_rate=base_rate, rates=rates, kstar=kstar), open(os.path.join(args.out, "calibration.json"), "w"), indent=1)
    print(f"CALIBRATION base match {base_rate:.3f} | random-direction match by k {rates} -> k* = {kstar}", flush=True)
    conds = conditions(args.n_rand, kstar)
    json.dump(conds, open(os.path.join(args.out, "conditions.json"), "w"), indent=1)
    gens = generate(args, prompts, conds)
    with open(os.path.join(args.out, "gen_raw.jsonl"), "w", encoding="utf-8") as f:
        for g in gens:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    gens = parse(args, prompts, gens)
    with open(os.path.join(args.out, "gen.jsonl"), "w", encoding="utf-8") as f:
        for g in gens:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print("done")


if __name__ == "__main__":
    main()
