"""Object-verb minimal pairs from Universal Dependencies test treebanks.

For each sentence, the first verb v (UPOS VERB) with an 'obj' dependent o such that
  * the object subtree of o is a contiguous span that contains no punctuation,
  * the verb complex (v + adjacent following aux/aux:pass/cop/compound:prt children) is adjacent to the object span,
  * neither span starts the sentence, the object span has <= 6 words, the sentence has 5-40 words,
  * the sentence has no multi-word tokens (e.g. French 'du') and no empty nodes,
produce the original sentence and the variant with the two spans swapped. Both variants are rebuilt from tokens by
the same detokeniser (so formatting never differs between them): no spaces for zho/jpn; otherwise single spaces,
SpaceAfter=No honoured only where the original neighbour is kept, no space before closing punctuation.
Output: pairs.jsonl rows {lang, sent_id, orig_order ('VO'|'OV'), ov, vo} (ov/vo = the OV-order and VO-order text).
usage: python make_pairs.py UD_DIR OUT.jsonl [--n 150]"""
import argparse, json, os, random

FILES = {"eng_Latn": "en_ewt", "deu_Latn": "de_gsd", "fra_Latn": "fr_gsd", "rus_Cyrl": "ru_syntagrus",
         "hin_Deva": "hi_hdtb", "tur_Latn": "tr_imst", "jpn_Jpan": "ja_gsd", "zho_Hans": "zh_gsd"}
NOSPACE = {"jpn_Jpan", "zho_Hans"}
VERB_EXT = {"aux", "aux:pass", "cop", "compound:prt", "mark", "fixed"}
CLOSE = set(".,;:!?)]}%»”…")
CLAUSAL = {"root", "conj", "ccomp", "xcomp", "advcl", "parataxis", "csubj"}   # clause-level verbs only


def sentences(path):
    sent, meta = [], {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if sent:
                yield meta, sent
            sent, meta = [], {}
        elif line.startswith("#"):
            if "=" in line:
                k, v = line[1:].split("=", 1)
                meta[k.strip()] = v.strip()
        else:
            c = line.split("\t")
            if "-" in c[0]:
                meta["mwt"] = True
                continue
            if "." in c[0]:
                meta["empty"] = True
                continue
            sent.append(dict(id=int(c[0]), form=c[1], upos=c[3], head=int(c[6]), dep=c[7],
                             noskip="SpaceAfter=No" in c[9]))
    if sent:
        yield meta, sent


def subtree(sent, i):
    kids = {}
    for t in sent:
        kids.setdefault(t["head"], []).append(t["id"])
    out, stack = [], [i]
    while stack:
        x = stack.pop()
        out.append(x)
        stack += kids.get(x, [])
    return sorted(out)


def detok(tokens, order, lang):
    """tokens: list of dicts in original order (index = id-1); order: list of ids in the new order"""
    if lang in NOSPACE:
        return "".join(tokens[i - 1]["form"] for i in order)
    s = ""
    for k, i in enumerate(order):
        t = tokens[i - 1]
        s += t["form"]
        if k + 1 < len(order):
            nxt = tokens[order[k + 1] - 1]
            keep_nospace = t["noskip"] and order[k + 1] == i + 1
            if not keep_nospace and not (nxt["upos"] == "PUNCT" and nxt["form"][0] in CLOSE) \
                    and not nxt["form"].startswith(("'", "n't", "’")):
                s += " "
    return s


def pair(meta, sent, lang):
    if meta.get("mwt") or meta.get("empty") or not 5 <= len(sent) <= 40:
        return None
    for v in sent:
        if v["upos"] != "VERB" or v["dep"].split(":")[0] not in CLAUSAL:
            continue
        objs = [t for t in sent if t["head"] == v["id"] and t["dep"].split(":")[0] == "obj"]
        for o in objs:
            span = subtree(sent, o["id"])
            if span != list(range(span[0], span[-1] + 1)) or len(span) > 6:
                continue
            if any(sent[i - 1]["upos"] == "PUNCT" for i in span) or v["id"] in span:
                continue
            # verb complex: v + right-adjacent functional tokens whose head is already in the complex
            # (Japanese chains: 名乗っ <-mark- て <-fixed- い)
            vs, ve = v["id"], v["id"]
            while ve + 1 <= len(sent) and sent[ve]["head"] in range(vs, ve + 1) and sent[ve]["dep"] in VERB_EXT \
                    and ve + 1 not in span:
                ve += 1
            if ve + 1 == span[0]:
                orig, a, b = "VO", list(range(vs, ve + 1)), span              # verb span a, then object span b
            elif span[-1] + 1 == vs:
                orig, a, b = "OV", span, list(range(vs, ve + 1))              # object span a, then verb span b
            else:
                continue
            if min(a[0], b[0]) == 1:
                continue
            ids = [t["id"] for t in sent]
            lo, hi = a[0], b[-1]
            swapped = ids[:lo - 1] + b + a + ids[hi:]
            t_orig, t_swap = detok(sent, ids, lang), detok(sent, swapped, lang)
            ov, vo = (t_swap, t_orig) if orig == "VO" else (t_orig, t_swap)
            return dict(lang=lang, sent_id=meta.get("sent_id", ""), orig_order=orig, ov=ov, vo=vo)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ud")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    rows = []
    for lang, stem in FILES.items():
        cand = [p for p in (pair(m, s, lang) for m, s in sentences(os.path.join(args.ud, f"{stem}-ud-test.conllu"))) if p]
        random.Random(0).shuffle(cand)
        keep = cand[:args.n]
        rows += keep
        n_vo = sum(p["orig_order"] == "VO" for p in keep)
        print(f"{lang}: {len(cand)} candidate pairs, kept {len(keep)} (orig VO {n_vo}, OV {len(keep) - n_vo})")
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
