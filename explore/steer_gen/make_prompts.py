"""Generation prompts: the first ~40% of FLORES+ devtest sentences (whole words; characters for zho/jpn), 150 per
language for the 8 steering languages. usage: python make_prompts.py SENTENCES_DEVTEST_JSON OUT.jsonl [--n 150]"""
import argparse, json, random

LANGS = ["eng_Latn", "deu_Latn", "fra_Latn", "rus_Cyrl", "hin_Deva", "tur_Latn", "jpn_Jpan", "zho_Hans"]
CJK = {"jpn_Jpan", "zho_Hans"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sentences")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--frac", type=float, default=0.4)
    args = ap.parse_args()
    S = json.load(open(args.sentences, encoding="utf-8"))
    idx = list(range(len(S["ids"])))
    random.Random(0).shuffle(idx)
    pick = idx[:args.n]                                   # same sentences in every language (parallel prompts)
    with open(args.out, "w", encoding="utf-8") as f:
        for lang in LANGS:
            for s in pick:
                t = S["text"][lang][s]
                if lang in CJK:
                    p = t[:max(6, round(args.frac * len(t)))]
                else:
                    w = t.split()
                    p = " ".join(w[:max(3, round(args.frac * len(w)))])
                f.write(json.dumps(dict(lang=lang, sent=s, prompt=p), ensure_ascii=False) + "\n")
    print(f"{len(LANGS) * len(pick)} prompts")


if __name__ == "__main__":
    main()
