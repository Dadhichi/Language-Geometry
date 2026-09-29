"""Tokenizer-level similarity between languages (Qwen2.5 tokenizer on FLORES dev+devtest text).
Outputs tok_sim.npz: tok_hist_int (histogram intersection of token-frequency distributions),
tok_jacc (Jaccard of token-type sets), char_hist_int (char-bigram histogram intersection), fert (tokens/sent)."""
import os, sys, json
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pylib"))
from tokenizers import Tokenizer
import dlib

tok = Tokenizer.from_file(os.path.join(HERE, "tokenizer.json"))
L = len(dlib.LANGS)
tc, cc, fert = [], [], []
for lang in dlib.LANGS:
    txt = []
    for split in ("dev", "devtest"):
        txt += json.load(open(os.path.join(dlib.DATA, f"sentences_{split}.json"), encoding="utf-8"))["text"][lang]
    enc = tok.encode_batch(txt, add_special_tokens=False)
    c = Counter()
    for e in enc:
        c.update(e.ids)
    tc.append(c)
    fert.append(np.mean([len(e.ids) for e in enc]))
    ch = Counter()
    for t in txt:
        ch.update(t[k:k + 2] for k in range(len(t) - 1))
    cc.append(ch)


def hist_int(A, B):
    na, nb = sum(A.values()), sum(B.values())
    return sum(min(A[t] / na, B[t] / nb) for t in A.keys() & B.keys())


H = np.array([[hist_int(tc[i], tc[j]) for j in range(L)] for i in range(L)])
J = np.array([[len(tc[i].keys() & tc[j].keys()) / len(tc[i].keys() | tc[j].keys()) for j in range(L)] for i in range(L)])
C = np.array([[hist_int(cc[i], cc[j]) for j in range(L)] for i in range(L)])
fert = np.array(fert)
np.savez(os.path.join(HERE, "tok_sim.npz"), tok_hist_int=H, tok_jacc=J, char_hist_int=C, fert=fert)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print("fertility (tokens/sentence):", dict(zip(dlib.LANGS, fert.round(1))))
print("token histogram intersection\n", H)
print("char-bigram histogram intersection\n", C)
