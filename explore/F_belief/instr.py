"""Investigator F: text-only instruments for the belief-simplex test (Qwen2.5 tokenizer, FLORES dev/devtest).

Per split, arrays indexed [i (source lang), j (other lang), s]:
  e_tok   fraction of i's token positions whose token id occurs anywhere in j's rendering of s
  e_tokc  same, counting only content tokens (letters/digits; punctuation/space-only tokens never count)
  e_char  fraction of i's non-space characters that occur in j's rendering of s
Per (i,s): ntok, digit-token fraction, script fractions of letters, kanji/kana fractions.
Ideal-observer beliefs over the 12 languages (unigram naive Bayes, leave-sentence-out counts over dev+devtest,
backoff to per-language script-class frequencies; uniform prior):
  B_pre[i,:,s]  time-average over positions t of P(lang | tokens_1..t)   (Bayesian prefix belief, mean-pooled)
  B_bag[i,:,s]  time-average over t of P(lang | token_t alone)          (bag-of-tokens, no context)
"""
import os, sys, json, unicodedata
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOFF = os.path.join(os.path.dirname(HERE), "D_offsets")
sys.path.insert(0, os.path.join(DOFF, "pylib"))
from tokenizers import Tokenizer

DATA = r"C:\Users\ASUS\Documents\lang-geom"
LANGS = ["eng_Latn", "deu_Latn", "fra_Latn", "spa_Latn", "rus_Cyrl", "hin_Deva",
         "arb_Arab", "zho_Hans", "jpn_Jpan", "tur_Latn", "vie_Latn", "ind_Latn"]
L = len(LANGS)
tok = Tokenizer.from_file(os.path.join(DOFF, "tokenizer.json"))
BETA = 20.0   # strength of script-class backoff

CLASSES = ["latin", "cyril", "deva", "arab", "han", "kana", "other_letter", "digit", "punct"]


def char_class(ch):
    if ch.isdigit():
        return "digit"
    if not ch.isalpha():
        return "punct"
    o = ord(ch)
    if o < 0x250 or 0x1E00 <= o <= 0x1EFF:
        return "latin"
    if 0x400 <= o <= 0x52F:
        return "cyril"
    if 0x900 <= o <= 0x97F:
        return "deva"
    if 0x600 <= o <= 0x6FF or 0x750 <= o <= 0x77F or 0xFB50 <= o <= 0xFEFF:
        return "arab"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF:
        return "han"
    if 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F:
        return "kana"
    return "other_letter"


def tok_class(s):
    """class of a decoded token string: first letter class if any letter, else digit if any digit, else punct"""
    cl = [char_class(c) for c in s]
    for c in cl:
        if c not in ("digit", "punct"):
            return c
    return "digit" if "digit" in cl else "punct"


def main():
    texts = {}
    for split in ("dev", "devtest"):
        texts[split] = json.load(open(os.path.join(DATA, f"sentences_{split}.json"), encoding="utf-8"))["text"]
    enc = {sp: [[e.ids for e in tok.encode_batch(texts[sp][lg], add_special_tokens=False)] for lg in LANGS]
           for sp in texts}
    for sp in enc:
        nt = np.load(os.path.join(DATA, f"ntok_{sp}.npy"))
        mine = np.array([[len(x) for x in enc[sp][i]] for i in range(L)])
        print(sp, "ntok match:", (mine == nt).mean(), "max diff", np.abs(mine - nt).max(), flush=True)

    # token classes over the used vocabulary + class sizes over the whole vocab
    V = tok.get_vocab_size()
    vocab_cls = np.array([CLASSES.index(tok_class(tok.decode([t]))) for t in range(V)], dtype=np.int8)
    Vc = np.bincount(vocab_cls, minlength=len(CLASSES)).astype(np.float64)
    print("vocab class sizes", dict(zip(CLASSES, Vc.astype(int))), flush=True)
    content = np.isin(vocab_cls, [CLASSES.index(c) for c in CLASSES if c != "punct"])
    is_digit = vocab_cls == CLASSES.index("digit")

    # corpus counts per language (both splits)
    C = np.zeros((L, V), np.float64)
    for sp in enc:
        for i in range(L):
            for ids in enc[sp][i]:
                np.add.at(C[i], ids, 1)
    N = C.sum(1)
    # class frequencies per language (for the backoff), with add-1
    Ccls = np.zeros((L, len(CLASSES)))
    for i in range(L):
        Ccls[i] = np.bincount(vocab_cls, weights=C[i], minlength=len(CLASSES)) + 1
    Pcls = Ccls / Ccls.sum(1, keepdims=True)
    p0 = Pcls[:, vocab_cls] / Vc[vocab_cls][None]          # [L, V] backoff p(tok|lang)
    print("backoff built", flush=True)

    for sp in enc:
        n = len(enc[sp][0])
        e_tok = np.zeros((L, L, n), np.float32)
        e_tokc = np.zeros((L, L, n), np.float32)
        e_char = np.zeros((L, L, n), np.float32)
        B_pre = np.zeros((L, L, n), np.float32)
        B_bag = np.zeros((L, L, n), np.float32)
        B_last = np.zeros((L, L, n), np.float32)
        B_first = np.zeros((L, L, n), np.float32)
        B_early = np.zeros((L, L, n), np.float32)
        B_late = np.zeros((L, L, n), np.float32)
        ntok = np.zeros((L, n), np.int32)
        digit = np.zeros((L, n), np.float32)
        punct = np.zeros((L, n), np.float32)
        scr = np.zeros((L, n, len(CLASSES)), np.float32)   # char-class fractions of non-space chars
        for s in range(n):
            ids = [np.asarray(enc[sp][i][s]) for i in range(L)]
            sets = [set(x.tolist()) for x in ids]
            chars = [texts[sp][LANGS[i]][s] for i in range(L)]
            csets = [set(c for c in t if not c.isspace()) for t in chars]
            own = np.zeros((L, 0))
            # leave-sentence-out counts: remove this sentence's tokens in every language
            allids = np.unique(np.concatenate(ids))
            own = np.zeros((L, len(allids)))
            for k in range(L):
                u, c = np.unique(ids[k], return_counts=True)
                own[k, np.searchsorted(allids, u)] = c
            Cl = C[:, allids] - own
            Nl = N - own.sum(1)
            logp_all = np.log((Cl + BETA * p0[:, allids]) / (Nl + BETA)[:, None])   # [L, |allids|]
            for i in range(L):
                t = ids[i]
                T = len(t)
                ntok[i, s] = T
                digit[i, s] = is_digit[t].mean()
                punct[i, s] = (~content[t]).mean()
                cc = [char_class(c) for c in chars[i] if not c.isspace()]
                scr[i, s] = np.bincount([CLASSES.index(c) for c in cc], minlength=len(CLASSES)) / max(len(cc), 1)
                lp = logp_all[:, np.searchsorted(allids, t)]          # [L, T]
                cum = np.cumsum(lp, 1)
                post = np.exp(cum - cum.max(0, keepdims=True)); post /= post.sum(0, keepdims=True)
                bag = np.exp(lp - lp.max(0, keepdims=True)); bag /= bag.sum(0, keepdims=True)
                B_pre[i, :, s] = post.mean(1)
                B_bag[i, :, s] = bag.mean(1)
                B_last[i, :, s] = post[:, -1]
                B_first[i, :, s] = bag[:, 0]
                B_early[i, :, s] = bag[:, :3].mean(1)
                B_late[i, :, s] = bag[:, 3:].mean(1) if T > 3 else bag.mean(1)
                for j in range(L):
                    if j == i:
                        continue
                    sh = np.fromiter((x in sets[j] for x in t.tolist()), bool, T)
                    e_tok[i, j, s] = sh.mean()
                    e_tokc[i, j, s] = (sh & content[t]).mean()
                    e_char[i, j, s] = np.mean([c in csets[j] for c in chars[i] if not c.isspace()])
            if s % 200 == 0:
                print(sp, s, flush=True)
        np.savez_compressed(os.path.join(HERE, f"instr_{sp}.npz"), e_tok=e_tok, e_tokc=e_tokc, e_char=e_char,
                            B_pre=B_pre, B_bag=B_bag, B_last=B_last, B_first=B_first, B_early=B_early, B_late=B_late, ntok=ntok, digit=digit, punct=punct,
                            scr=scr, classes=np.array(CLASSES))
        np.set_printoptions(precision=3, suppress=True, linewidth=220)
        print(sp, "mean e_tokc [i,j]\n", e_tokc.mean(2))
        print(sp, "mean B_pre [i,j]\n", B_pre.mean(2))
        print(sp, "mean B_bag [i,j]\n", B_bag.mean(2))
        print(sp, "mean B_last own:", np.array([B_last[i, i].mean() for i in range(L)]))
        print(sp, "jpn kanji frac mean", scr[8, :, CLASSES.index("han")].mean(),
              "rus latin frac mean", scr[4, :, 0].mean(), flush=True)


if __name__ == "__main__":
    main()
