"""Confirmatory free-generation steering test (PREREG.md). GPU. Reuses the frozen generation and language-ID code of
explore/steer_gen/gen_ov.py; differs in the conditions (fixed k = +-2, no calibration) and in the parse, which
splits every counted verb-object dependency by the object's UPOS: nominal (NOUN, PROPN), pronominal (PRON), other.
Output OUT/gen.jsonl rows {cond, lang, sent, cont, lid, match, n_ov, n_vo, nom_ov, nom_vo, pron_ov, pron_vo,
oth_ov, oth_vo}; OUT/conditions.json.
usage: python gen_ov2.py --prompts prompts2.jsonl --dirs steer_dirs2.npz --out OUT [--n_rand 24] [--limit N]"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "steer_gen"))
import gen_ov as G

G.STANZA = {"deu_Latn": "de", "nld_Latn": "nl", "rus_Cyrl": "ru", "ukr_Cyrl": "uk", "pol_Latn": "pl", "hrv_Latn": "hr",
            "eng_Latn": "en", "spa_Latn": "es", "kor_Hang": "ko"}
G.LANGID = {"deu_Latn": "de", "nld_Latn": "nl", "rus_Cyrl": "ru", "ukr_Cyrl": "uk", "pol_Latn": "pl", "hrv_Latn": "hr",
            "eng_Latn": "en", "spa_Latn": "es", "kor_Hang": "ko"}
K = 2.0
CLASSES = ("nom", "pron", "oth")


def conditions(n_rand, k=K, layer=14):
    c = [dict(layer=None, kind="base", k=0, r=-1)]
    c += [dict(layer=layer, kind="ov", k=s * k, r=-1) for s in (-1, 1)]
    c += [dict(layer=layer, kind="ie", k=s * k, r=-1) for s in (-1, 1)]
    c += [dict(layer=layer, kind="rand", k=s * k, r=r) for r in range(n_rand) for s in (-1, 1)]
    return c


def obj_class(upos):
    return "nom" if upos in ("NOUN", "PROPN") else "pron" if upos == "PRON" else "oth"


def parse(args, prompts, gens):
    """Same counting rule as gen_ov.parse (VERB head, deprel obj*, both words in the generated part, OV if the object
    precedes the verb), with the counts split by object class."""
    import stanza
    pmap = {(p["lang"], p["sent"]): p["prompt"] for p in prompts}
    keys = ["n_ov", "n_vo"] + [f"{c}_{o}" for c in CLASSES for o in ("ov", "vo")]
    for g in gens:
        for k in keys:
            g[k] = 0
    t0 = time.time()
    for lang in sorted(G.STANZA):
        part = os.path.join(args.out, f"parsed_{lang}.json")
        if os.path.exists(part):                                            # resume
            saved = json.load(open(part))
            for g in gens:
                if g["lang"] == lang:
                    g.update(saved[f"{g['cond']}_{g['sent']}"])
            print(f"parsed {lang}: loaded from checkpoint", flush=True)
            continue
        try:
            nlp = stanza.Pipeline(G.STANZA[lang], processors="tokenize,mwt,pos,lemma,depparse", use_gpu=True,
                                  verbose=False, download_method=None)
        except Exception:
            nlp = stanza.Pipeline(G.STANZA[lang], processors="tokenize,pos,lemma,depparse", use_gpu=True,
                                  verbose=False, download_method=None)
        rows = [g for g in gens if g["lang"] == lang and g["cont"].strip()]
        texts = [pmap[(lang, g["sent"])] + " " + g["cont"].lstrip() for g in rows]
        starts = [len(pmap[(lang, g["sent"])]) + 1 for g in rows]
        for a in range(0, len(texts), 256):
            docs = nlp.bulk_process([stanza.Document([], text=t) for t in texts[a:a + 256]])
            for g, d, st in zip(rows[a:a + 256], docs, starts[a:a + 256]):
                for sent in d.sentences:
                    w = {x.id: x for x in sent.words}
                    gen = {x.id: (x.parent.start_char if x.parent.start_char is not None else -1) >= st for x in sent.words}
                    for x in sent.words:
                        if x.deprel and x.deprel.split(":")[0] == "obj" and x.head in w and w[x.head].upos == "VERB" \
                                and gen[x.id] and gen[x.head]:
                            o = "ov" if x.id < x.head else "vo"
                            g[f"n_{o}"] += 1
                            g[f"{obj_class(x.upos)}_{o}"] += 1
        json.dump({f"{g['cond']}_{g['sent']}": {k: g[k] for k in keys} for g in gens if g["lang"] == lang}, open(part, "w"))
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
    conds = conditions(args.n_rand)
    json.dump(conds, open(os.path.join(args.out, "conditions.json"), "w"), indent=1)
    gens = G.generate(args, prompts, conds, ckpt=os.path.join(args.out, "gen_raw.jsonl"))
    gens.sort(key=lambda g: (g["cond"], g["lang"], g["sent"]))
    gens = G.lid_match(gens)
    gens = parse(args, prompts, gens)
    with open(os.path.join(args.out, "gen.jsonl"), "w", encoding="utf-8") as f:
        for g in gens:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print("done")


if __name__ == "__main__":
    main()
