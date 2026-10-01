"""EXPLORATORY (not pre-registered): is the free-generation OV shift real reordering of nominal objects?

Re-parses a subset of conditions on CPU with the same Stanza pipeline and counting rule as gen_ov.py, and splits each
counted verb-object dependency by the object's UPOS: nominal (NOUN, PROPN) vs pronominal (PRON) vs other. French and
other Romance languages place object clitics before the verb, so a steering effect that only produced more pronouns
would raise the OV rate without reordering anything. Also records a repetition score per continuation (1 - distinct
word-bigram ratio) to check that steered text is not degenerate.
usage: python reparse_objtype.py OUT_DIR [--n_rand 24] [--gpu] [--prompts P] [--out objtype.json]
OUT_DIR holds conditions.json, calibration.json and gen.jsonl (or gen.jsonl.gz) from gen_ov.py.
"""
import argparse, collections, gzip, json, os, time
import stanza

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("D")
ap.add_argument("--n_rand", type=int, default=24)
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--prompts", default=os.path.join(HERE, "prompts.jsonl"))
ap.add_argument("--out", default=os.path.join(HERE, "objtype.json"))
args = ap.parse_args()
D, N_RAND = args.D, args.n_rand
STANZA = {"eng_Latn": "en", "deu_Latn": "de", "fra_Latn": "fr", "rus_Cyrl": "ru", "hin_Deva": "hi"}
conds = json.load(open(os.path.join(D, "conditions.json")))
ks = json.load(open(os.path.join(D, "calibration.json")))["kstar"]
gp = os.path.join(D, "gen.jsonl")
G = [json.loads(l) for l in (open(gp, encoding="utf-8") if os.path.exists(gp) else gzip.open(gp + ".gz", "rt", encoding="utf-8"))]
pmap = {(p["lang"], p["sent"]): p["prompt"] for p in map(json.loads, open(args.prompts, encoding="utf-8"))}


def find(kind, k, r=-1):
    return next(i for i, c in enumerate(conds) if c["kind"] == kind and abs(c["k"] - k) < 1e-9 and c["r"] == r)


CI = {"base": 0, "ov-": find("ov", -ks), "ov+": find("ov", ks), "ie-": find("ie", -ks), "ie+": find("ie", ks)}
CI.update({f"r{r}{'+' if s > 0 else '-'}": find("rand", s * ks, r) for r in range(N_RAND) for s in (-1, 1)})
inv = {v: k for k, v in CI.items()}


def rep(text):
    w = text.split()
    b = list(zip(w, w[1:]))
    return 1 - len(set(b)) / len(b) if b else 0.0


out_path = args.out
out = json.load(open(out_path)) if os.path.exists(out_path) else {}
t0 = time.time()
for lang, code in STANZA.items():
    if lang in out:
        continue
    stanza.download(code, verbose=False)
    try:
        nlp = stanza.Pipeline(code, processors="tokenize,mwt,pos,lemma,depparse", use_gpu=args.gpu, verbose=False,
                              download_method=None)
    except Exception:
        nlp = stanza.Pipeline(code, processors="tokenize,pos,lemma,depparse", use_gpu=args.gpu, verbose=False,
                              download_method=None)
    rows = [g for g in G if g["lang"] == lang and g["cond"] in inv and g["cont"].strip()]
    texts = [pmap[(lang, g["sent"])] + " " + g["cont"].lstrip() for g in rows]
    starts = [len(pmap[(lang, g["sent"])]) + 1 for g in rows]
    agg = collections.defaultdict(lambda: collections.Counter())
    agree = [0, 0]
    for a in range(0, len(texts), 64):
        docs = nlp.bulk_process([stanza.Document([], text=t) for t in texts[a:a + 64]])
        for g, d, st in zip(rows[a:a + 64], docs, starts[a:a + 64]):
            c = agg[inv[g["cond"]]]
            c["n_rows"] += 1
            c["rep_sum"] += rep(g["cont"])
            n_ov = n_vo = 0
            for sent in d.sentences:
                w = {x.id: x for x in sent.words}
                gen = {x.id: (x.parent.start_char if x.parent.start_char is not None else -1) >= st for x in sent.words}
                for x in sent.words:
                    if x.deprel and x.deprel.split(":")[0] == "obj" and x.head in w and w[x.head].upos == "VERB" \
                            and gen[x.id] and gen[x.head]:
                        cls = "nom" if x.upos in ("NOUN", "PROPN") else "pron" if x.upos == "PRON" else "oth"
                        order = "ov" if x.id < x.head else "vo"
                        n_ov += order == "ov"
                        n_vo += order == "vo"
                        if g["match"]:
                            c[f"{cls}_{order}"] += 1
            agree[0] += (n_ov, n_vo) == (g["n_ov"], g["n_vo"])
            agree[1] += 1
    out[lang] = {k: dict(v) for k, v in agg.items()}
    out[lang]["_agree"] = agree
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"{lang}: {len(rows)} texts, counts identical to the VM parse in {agree[0]}/{agree[1]} ({time.time() - t0:.0f}s)",
          flush=True)
    del nlp
