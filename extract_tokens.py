#!/usr/bin/env python
"""
extract_tokens.py -- token-level residual states for the belief-simplex test (explore/IDEAS.md candidate 2,
explore/prereg_belief/PREREG.md).  Three products, all in --out:

1. states_L{l}.f16.npy [n_tok, s+n_pc]: every content position of every natural sentence, projected onto a
   per-layer basis P_l = [S_l | U_l]; S_l = orthonormal span of the language-centroid offsets (--span npz, key
   L{l}, [d, s]); U_l = top n_pc principal directions of token states orthogonal to S_l, estimated in a pilot pass
   (--pilot sentences per language).  full_L{l}.f16.npy: full-width states at --full_layers for the first --full_n
   sentences per language (to measure projection loss).  index.npz: lang, sent, pos, token id per row.
2. tagscore.f16.npy [n_tok_tag, n_langs]: log p(token_t | TAG(lang_k) + prefix) for the first --tag_len content
   tokens of every sentence under every language tag; tagindex.npz.  P(language | prefix) follows by cumulative
   sums over positions plus a prior.
3. codeswitch: sequences = first fraction p of sentence s in language A (word units; characters for zho/jpn) +
   the remaining 1-p of the same sentence in language B.  cs_states_L{l}.f16.npy (same bases), cs_index.npz
   (pair, direction, sentence, p, position, token, segment 0=A/1=B), cs_tagscore.f16.npy [n_tok, 2] under the
   A and B tags.

Every input is [sink] + tokens, as in extract.py (sink = BOS if present else newline; position 0 never stored).
"""
import argparse, json, os, time
import numpy as np
import torch

from extract import load_split, locate, sink_token_id

NAMES = {"eng_Latn": "English", "deu_Latn": "German", "fra_Latn": "French", "spa_Latn": "Spanish",
         "rus_Cyrl": "Russian", "hin_Deva": "Hindi", "arb_Arab": "Arabic", "zho_Hans": "Chinese",
         "jpn_Jpan": "Japanese", "tur_Latn": "Turkish", "vie_Latn": "Vietnamese", "ind_Latn": "Indonesian"}
TAG = "The following text is written in {}.\n"
CJK = {"zho_Hans", "jpn_Jpan"}
CS_PAIRS = [("spa_Latn", "fra_Latn"), ("eng_Latn", "deu_Latn"), ("tur_Latn", "ind_Latn"), ("zho_Hans", "jpn_Jpan"),
            ("eng_Latn", "zho_Hans"), ("eng_Latn", "hin_Deva"), ("rus_Cyrl", "eng_Latn"), ("arb_Arab", "eng_Latn")]


class Grab:
    """forward hooks keeping the raw residual stream [B,T,d] of the selected layers (0 = embedding output)"""

    def __init__(self, blocks, embed, layers):
        self.out, self.handles = {}, []
        for l in layers:
            mod = embed if l == 0 else blocks[l - 1]
            self.handles.append(mod.register_forward_hook(self._mk(l)))

    def _mk(self, l):
        def hook(_m, _i, out):
            self.out[l] = out[0] if isinstance(out, (tuple, list)) else out
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def batches(lens, budget, max_b):
    order = np.argsort(-np.asarray(lens), kind="stable")
    s = 0
    while s < len(order):
        b = max(1, min(max_b, budget // int(lens[order[s]])))
        yield order[s:s + b]
        s += b


def pad(seqs, idx, pad_id, dev):
    L = max(len(seqs[i]) for i in idx)
    ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(idx), L), dtype=torch.bool)
    for r, i in enumerate(idx):
        ids[r, :len(seqs[i])] = torch.tensor(seqs[i])
        mask[r, :len(seqs[i])] = True
    return ids.to(dev), mask.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--span", required=True, help="npz with L{l}: [d, s] orthonormal centroid-offset span")
    ap.add_argument("--langs", default=",".join(NAMES))
    ap.add_argument("--split", default="devtest")
    ap.add_argument("--layers", default=",".join(str(l) for l in range(0, 29, 2)))
    ap.add_argument("--n_pc", type=int, default=117)
    ap.add_argument("--pilot", type=int, default=100)
    ap.add_argument("--full_layers", default="4,14,24")
    ap.add_argument("--cs_layers", default="2,4,8,12,14,16,20,24")
    ap.add_argument("--full_n", type=int, default=50)
    ap.add_argument("--tag_len", type=int, default=24)
    ap.add_argument("--cs_n", type=int, default=200)
    ap.add_argument("--cs_p", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--max_sent", type=int, default=0, help="debug: sentences per language")
    ap.add_argument("--batch_tokens", type=int, default=8192)
    ap.add_argument("--flores", default="hf")
    ap.add_argument("--local_dir", default=None)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    langs = args.langs.split(",")
    layers = [int(x) for x in args.layers.split(",")]
    full_layers = [int(x) for x in args.full_layers.split(",") if x]
    cs_layers = [int(x) for x in args.cs_layers.split(",") if x]
    assert set(cs_layers) <= set(layers), "cs_layers must be a subset of layers"
    if args.max_sent:
        args.smoke = args.max_sent

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, token=token)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), token=token,
                                                 device_map={"": dev.index or 0} if dev.type == "cuda" else None)
    model.eval()
    blocks, embed, fnorm, head, _ = locate(model)
    base = model.base_model
    sink, _ = sink_token_id(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    d = int(model.config.hidden_size)

    ids_, sents = load_split(args, langs, args.split, token)
    n = len(ids_)
    enc = {l: [[sink] + e for e in tok(sents[l], add_special_tokens=False)["input_ids"]] for l in langs}
    span = np.load(args.span)
    S = {l: torch.as_tensor(span[f"L{l}"], dtype=torch.float32, device=dev) for l in layers}
    s_dim = S[layers[0]].shape[1]
    grab = Grab(blocks, embed, sorted(set(layers) | set(full_layers)))

    def run(seqs, fn):
        lens = [len(x) for x in seqs]
        for idx in batches(lens, args.batch_tokens, 256):
            ids, mask = pad(seqs, idx, pad_id, dev)
            with torch.inference_mode():
                grab.out.clear()
                hs = base(input_ids=ids, attention_mask=mask.long(), use_cache=False).last_hidden_state
                fn(idx, ids, mask, hs)

    # ---- pass 1: PCA of token states orthogonal to the language span (pilot sentences)
    t0 = time.time()
    pil = [enc[l][s] for l in langs for s in range(min(args.pilot, n))]
    acc = {l: [torch.zeros(d, dtype=torch.float64, device=dev), torch.zeros(d, d, dtype=torch.float32, device=dev), 0]
           for l in layers}

    def pca_fn(idx, ids, mask, hs):
        m = mask.clone(); m[:, 0] = False
        for l in layers:
            x = grab.out[l][m].float()
            x = x - (x @ S[l]) @ S[l].T
            acc[l][0] += x.sum(0).double(); acc[l][1] += x.T @ x; acc[l][2] += x.shape[0]
    run(pil, pca_fn)
    P = {}
    for l in layers:
        mu = (acc[l][0] / acc[l][2]).float()
        C = acc[l][1] / acc[l][2] - torch.outer(mu, mu)
        w, V = torch.linalg.eigh(C.double())
        U = V[:, -args.n_pc:].flip(1).float()
        P[l] = torch.cat([S[l], U], 1)                                     # [d, s + n_pc]
    np.savez(os.path.join(args.out, "bases.npz"), **{f"L{l}": P[l].cpu().numpy() for l in layers})
    del acc
    print(f"pilot PCA done ({len(pil)} seqs) {time.time() - t0:.0f}s", flush=True)

    # ---- pass 2: natural sentences, every content position, projected; full width for a subset
    def store_states(seqs, meta_rows, prefix, lays):
        bufs = {l: [] for l in lays}
        fbufs = {l: [] for l in full_layers}
        rows = []

        def fn(idx, ids, mask, hs):
            m = mask.clone(); m[:, 0] = False
            for l in lays:
                bufs[l].append((grab.out[l][m].float() @ P[l]).half().cpu())
            for l in full_layers:
                keep = torch.tensor([meta_rows[i].get("full", False) for i in idx], device=dev)
                mm = m & keep[:, None]
                if mm.any():
                    fbufs[l].append(grab.out[l][mm].half().cpu())
            for r, i in enumerate(idx):
                L_i = int(mask[r].sum())
                for pos in range(1, L_i):
                    rows.append((i, pos, int(ids[r, pos])))
        run(seqs, fn)
        rows = np.array(rows, dtype=np.int64)
        for l in lays:
            np.save(os.path.join(args.out, f"{prefix}states_L{l}.f16.npy"), torch.cat(bufs[l]).numpy())
        for l in full_layers:
            if fbufs[l]:
                np.save(os.path.join(args.out, f"{prefix}full_L{l}.f16.npy"), torch.cat(fbufs[l]).numpy())
        return rows

    t0 = time.time()
    seqs, meta_rows = [], []
    for li, l in enumerate(langs):
        for s in range(n):
            seqs.append(enc[l][s]); meta_rows.append(dict(lang=li, sent=s, full=s < args.full_n))
    rows = store_states(seqs, meta_rows, "", layers)
    lang = np.array([meta_rows[i]["lang"] for i in rows[:, 0]]); sent = np.array([meta_rows[i]["sent"] for i in rows[:, 0]])
    full = np.array([meta_rows[i]["full"] for i in rows[:, 0]])
    np.savez(os.path.join(args.out, "index.npz"), seq=rows[:, 0], lang=lang, sent=sent, pos=rows[:, 1],
             token=rows[:, 2], full=full)
    print(f"natural states: {len(rows)} tokens {time.time() - t0:.0f}s", flush=True)

    # ---- tag scoring: log p(text token | TAG(k) + prefix), first tag_len content tokens
    Wf = head.weight.float()                                                # [V, d], converted once

    def score(seqs_text, tag_ids_list):
        """seqs_text[i]: content token ids; tag_ids_list[j]: tag token ids; returns [n_tok, n_tags] float16"""
        outs = []
        for tag_ids in tag_ids_list:
            full_seqs = [[sink] + tag_ids + t for t in seqs_text]
            off = 1 + len(tag_ids)
            res = [None] * len(full_seqs)

            def fn(idx, ids, mask, hs):
                for r, i in enumerate(idx):
                    L_t = len(seqs_text[i])
                    h = hs[r, off - 1: off - 1 + L_t].float()                      # predicts tokens off..off+L_t-1
                    lp = torch.cat([(c @ Wf.T).log_softmax(-1) for c in h.split(256)])
                    tgt = ids[r, off: off + L_t]
                    res[i] = lp.gather(1, tgt[:, None])[:, 0].half().cpu()
            run(full_seqs, fn)
            outs.append(torch.cat(res))
        return torch.stack(outs, 1).numpy()

    t0 = time.time()
    tags = [tok(TAG.format(NAMES[l]), add_special_tokens=False)["input_ids"] for l in langs]
    txt, trow = [], []
    for li, l in enumerate(langs):
        for s in range(n):
            t = enc[l][s][1:1 + args.tag_len]
            txt.append(t); trow += [(li, s, p + 1) for p in range(len(t))]
    ts = score(txt, tags)
    trow = np.array(trow)
    np.save(os.path.join(args.out, "tagscore.f16.npy"), ts)
    np.savez(os.path.join(args.out, "tagindex.npz"), lang=trow[:, 0], sent=trow[:, 1], pos=trow[:, 2],
             langs=np.array(langs), tag_template=TAG)
    print(f"tag scoring: {ts.shape} {time.time() - t0:.0f}s", flush=True)

    # ---- code-switch sequences
    t0 = time.time()
    ps = [float(x) for x in args.cs_p.split(",")]
    units = lambda l, s: list(sents[l][s]) if l in CJK else sents[l][s].split()
    join = lambda l, u: "".join(u) if l in CJK else " ".join(u)
    cs_seqs, cs_meta, cs_txt = [], [], []
    for pi, (A, B) in enumerate(CS_PAIRS):
        for direction, (a, b) in enumerate(((A, B), (B, A))):
            if a not in langs or b not in langs:
                continue
            for s in range(min(args.cs_n, n)):
                ua, ub = units(a, s), units(b, s)
                for p in ps:
                    pa, pb = join(a, ua[:max(1, round(p * len(ua)))]), join(b, ub[round(p * len(ub)):])
                    sep = "" if (a in CJK and b in CJK) else " "
                    ta = tok(pa, add_special_tokens=False)["input_ids"]
                    tb = tok(sep + pb, add_special_tokens=False)["input_ids"]
                    cs_seqs.append([sink] + ta + tb)
                    cs_txt.append(ta + tb)
                    cs_meta.append(dict(pair=pi, dir=direction, sent=s, p=p, nA=len(ta), a=langs.index(a),
                                        b=langs.index(b)))
    rows = store_states(cs_seqs, [dict(full=False)] * len(cs_seqs), "cs_", cs_layers)
    seg = np.array([int(pos - 1 >= cs_meta[i]["nA"]) for i, pos, _ in rows])
    get = lambda k: np.array([cs_meta[i][k] for i in rows[:, 0]])
    np.savez(os.path.join(args.out, "cs_index.npz"), seq=rows[:, 0], pos=rows[:, 1], token=rows[:, 2], seg=seg,
             pair=get("pair"), dir=get("dir"), sent=get("sent"), p=get("p"), a=get("a"), b=get("b"), nA=get("nA"))
    # tag scores under the A and B tags only, aligned to the stored rows by (sequence, position)
    out_ab = np.zeros((len(rows), 2), np.float16)
    for k in (0, 1):
        per_seq = {}
        for li in range(len(langs)):
            members = [i for i, m in enumerate(cs_meta) if (m["a"] if k == 0 else m["b"]) == li]
            if not members:
                continue
            sc = score([cs_txt[i] for i in members], [tags[li]])[:, 0]
            o = 0
            for i in members:
                per_seq[i] = sc[o:o + len(cs_txt[i])]; o += len(cs_txt[i])
        out_ab[:, k] = [per_seq[i][pos - 1] for i, pos, _ in rows]
    np.save(os.path.join(args.out, "cs_tagscore.f16.npy"), out_ab)
    print(f"code-switch: {len(cs_seqs)} seqs, {len(rows)} tokens {time.time() - t0:.0f}s", flush=True)
    json.dump(dict(model=args.model, langs=langs, layers=layers, full_layers=full_layers, cs_layers=cs_layers,
                   n_pc=args.n_pc,
                   s_dim=s_dim, split=args.split, n_sent=n, tag_template=TAG, tag_len=args.tag_len,
                   cs_pairs=CS_PAIRS, cs_p=ps, sink=sink),
              open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    grab.close()
    print("done")


if __name__ == "__main__":
    main()
