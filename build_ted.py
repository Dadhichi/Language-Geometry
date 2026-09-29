#!/usr/bin/env python
"""
build_ted.py -- n-way parallel TED sentences as extra FLORES-style splits (tedfit, tedtest).

    python build_ted.py --out /content/flores200_dataset --n_fit 6000 --n_test 1000

Writes {out}/tedfit/{lang}.tedfit and {out}/tedtest/{lang}.tedtest, one sentence per line, aligned
across languages.  Then:
    python extract.py ... --local_dir /content/flores200_dataset --splits tedfit,tedtest

Why: fit.py needs n_fit well above the working dimension.  FLORES gives ~1000 sentences; TED gives
thousands of 12-way parallel sentences (spoken register: keep FLORES devtest as the out-of-domain
test and tedtest as the in-domain one).

Source: OPUS TED2020 en-xx pairs (plain, untokenized subtitles), intersected on the English side; for the
default 12 languages this gives ~17.6k 12-way sentences after the length filter (Hindi and Indonesian are
the bottleneck).  Not ted_multi (Qi et al. 2018): its download URL is dead, and its text is Moses-tokenized
('we &apos;ve ...', space-segmented CJK), unlike the natural text in FLORES.
"""
import argparse
import io
import os
import random
import urllib.request
import zipfile

FLORES2TED = {"eng_Latn": "en", "deu_Latn": "de", "fra_Latn": "fr", "spa_Latn": "es", "rus_Cyrl": "ru",
              "hin_Deva": "hi", "arb_Arab": "ar", "zho_Hans": "zh-cn", "jpn_Jpan": "ja", "tur_Latn": "tr",
              "vie_Latn": "vi", "ind_Latn": "id", "por_Latn": "pt", "ita_Latn": "it", "kor_Hang": "ko",
              "nld_Latn": "nl", "pol_Latn": "pl", "ukr_Cyrl": "uk", "pes_Arab": "fa", "ell_Grek": "el",
              "heb_Hebr": "he", "ces_Latn": "cs", "ron_Latn": "ro"}
OPUS_CODE = {"zh-cn": "zh_cn"}


def from_opus(langs, cache="/content/opus_ted2020"):
    os.makedirs(cache, exist_ok=True)
    en_index = None
    tables = {}
    for lang in langs:
        code = FLORES2TED[lang]
        if code == "en":
            continue
        oc = OPUS_CODE.get(code, code)
        pair = "-".join(sorted(["en", oc]))
        url = f"https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/{pair}.txt.zip"
        path = os.path.join(cache, f"{pair}.zip")
        if not os.path.exists(path):
            print("downloading", url)
            urllib.request.urlretrieve(url, path)
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            en_f = [n for n in names if n.endswith(".en")][0]
            xx_f = [n for n in names if n.endswith("." + oc)][0]
            en_lines = io.TextIOWrapper(z.open(en_f), encoding="utf-8").read().split("\n")
            xx_lines = io.TextIOWrapper(z.open(xx_f), encoding="utf-8").read().split("\n")
        table = {}
        for e, x in zip(en_lines, xx_lines):
            e, x = e.strip(), x.strip()
            if e and x and e not in table:
                table[e] = x
        tables[lang] = table
        en_index = set(table) if en_index is None else en_index & set(table)
        print(f"  {lang}: {len(table)} pairs, intersection {len(en_index)}")
    rows = [{**{"eng_Latn": e}, **{l: tables[l][e] for l in tables}} for e in sorted(en_index)]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default=("eng_Latn,deu_Latn,fra_Latn,spa_Latn,rus_Cyrl,hin_Deva,"
                                        "arb_Arab,zho_Hans,jpn_Jpan,tur_Latn,vie_Latn,ind_Latn"))
    ap.add_argument("--n_fit", type=int, default=6000)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--min_words", type=int, default=6)
    ap.add_argument("--max_words", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    langs = args.langs.split(",")
    rows = from_opus(langs)
    print(f"{len(rows)} {len(langs)}-way parallel sentences before filtering")
    seen, keep = set(), []
    for r in rows:
        en = r["eng_Latn"]
        nw = len(en.split())
        if en in seen or nw < args.min_words or nw > args.max_words or any(not r[l] for l in langs):
            continue
        seen.add(en)
        keep.append(r)
    print(f"{len(keep)} after dedup and length filter")
    random.Random(args.seed).shuffle(keep)
    need = args.n_fit + args.n_test
    if len(keep) < need:
        print(f"WARNING: only {len(keep)} sentences; shrinking splits proportionally")
        args.n_test = max(200, len(keep) // 7)
        args.n_fit = len(keep) - args.n_test
    for split, chunk in (("tedfit", keep[: args.n_fit]), ("tedtest", keep[args.n_fit: args.n_fit + args.n_test])):
        os.makedirs(os.path.join(args.out, split), exist_ok=True)
        for l in langs:
            with open(os.path.join(args.out, split, f"{l}.{split}"), "w", encoding="utf-8") as f:
                f.write("\n".join(r[l].replace("\n", " ") for r in chunk) + "\n")
        print(f"wrote {split}: {len(chunk)} sentences x {len(langs)} languages")


if __name__ == "__main__":
    main()
