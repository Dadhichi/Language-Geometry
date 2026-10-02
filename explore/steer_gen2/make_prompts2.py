"""Prompts for the confirmatory free-generation test (PREREG.md): the first ~40% of the words of 150 FLORES+ devtest
sentences that were NOT used in explore/steer_gen (fresh data), the same sentences in all 9 languages.
Selection: the steer_gen shuffle (random.Random(0) over the 1,012 devtest indices) continued past its first 150.
usage: python make_prompts2.py SENTENCES_DEVTEST_JSON OUT.jsonl [--n 150]"""
import argparse, json, os, random

LANGS = ["deu_Latn", "nld_Latn", "rus_Cyrl", "ukr_Cyrl", "pol_Latn", "hrv_Latn", "eng_Latn", "spa_Latn", "kor_Hang"]
OLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "steer_gen", "prompts.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sentences")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--frac", type=float, default=0.4)
    args = ap.parse_args()
    S = json.load(open(args.sentences, encoding="utf-8"))
    used = {json.loads(l)["sent"] for l in open(OLD, encoding="utf-8")}
    idx = list(range(len(S["ids"])))
    random.Random(0).shuffle(idx)
    pick = [s for s in idx if s not in used][:args.n]
    assert len(pick) == args.n and not used & set(pick)
    with open(args.out, "w", encoding="utf-8") as f:
        for lang in LANGS:
            for s in pick:
                w = S["text"][lang][s].split()                       # all 9 languages separate words by spaces
                f.write(json.dumps(dict(lang=lang, sent=s, prompt=" ".join(w[:max(3, round(args.frac * len(w)))])),
                                   ensure_ascii=False) + "\n")
    print(f"{len(LANGS) * len(pick)} prompts; {len(used)} steer_gen sentences excluded")


if __name__ == "__main__":
    main()
