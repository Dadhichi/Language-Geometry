#!/usr/bin/env python
"""
extract.py -- per-layer residual-stream representations of FLORES-200 sentences.

Colab usage:
    python extract.py --model Qwen/Qwen2.5-7B --tag qwen25_7b \
        --out /content/drive/MyDrive/lang_geom --flores hf
    python extract.py --model meta-llama/Llama-3.1-8B --tag llama31_8b \
        --out /content/drive/MyDrive/lang_geom --flores /content/flores200_dataset

Quick end-to-end check (2 languages, 16 sentences per split):
    python extract.py --model Qwen/Qwen2.5-7B --tag smoke --out /content/scratch_out \
        --langs eng_Latn,deu_Latn --smoke 16

Output layout: {out}/{tag}/   (one file per layer so fit.py reads a layer contiguously)
    meta.json                        model, languages, layer count, d, sink token, per-split sentence ids
    mean_{split}_L{layer}.f16.npy    [n_lang, n_sent, d]   masked mean over content tokens
    last_{split}_L{layer}.f16.npy    [n_lang, n_sent, d]   last content token
    ntok_{split}.npy                 [n_lang, n_sent]      content-token count (sink excluded)
    normdiag_{split}.f16.npy         [n_lang, n_sent, n_layers+1, 2]  (sink norm / median content norm,
                                                                      max content norm / median content norm)
    wu_basis.npy, wu_eigs.npy        top-512 right singular vectors of (final_norm.weight * lm_head.weight)

Splits are added incrementally: running again with --splits tedfit,tedtest on the same --tag appends
those splits to meta.json and leaves existing ones alone (use --overwrite to redo a split).
--flores is 'hf' or a local FLORES-200 dir; --local_dir (optional) is checked first for every split,
which is how the TED splits written by build_ted.py are picked up.

Layer index 0 is the embedding output (residual stream entering block 1); index i is the
output of decoder block i. Hooks are used rather than output_hidden_states because HF applies
the final norm to the last entry of output_hidden_states.

Every input is [sink] + tokens(sentence), where sink = BOS if the tokenizer has one, else the
newline token. Position 0 is always dropped from pooling, so all models are treated identically
and the massive-norm attention-sink token never enters the mean.

Files are written to --tmp (local disk) with np.memmap and copied to --out at the end; writing
memmaps directly onto a Drive FUSE mount is slow and occasionally corrupts.
"""
import argparse
import functools
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

DEFAULT_LANGS = ("eng_Latn,deu_Latn,fra_Latn,spa_Latn,rus_Cyrl,hin_Deva,"
                 "arb_Arab,zho_Hans,jpn_Jpan,tur_Latn,vie_Latn,ind_Latn")


# ----------------------------------------------------------------------------- data

def load_flores_local(root, lang, split):
    """FLORES-200 tarball layout: {root}/{split}/{lang}.{split}, one sentence per line."""
    cands = [os.path.join(root, split, f"{lang}.{split}"), os.path.join(root, f"{lang}.{split}")]
    for p in cands:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                lines = [l.rstrip("\n") for l in f]
            return list(range(len(lines))), lines
    raise FileNotFoundError(f"no FLORES file for {lang}/{split} under {root}; tried {cands}")


# FLORES+ renamed some FLORES-200 codes; outputs keep the FLORES-200 name
FLORES_PLUS_ALIASES = {"zho_Hans": "cmn_Hans", "zho_Hant": "cmn_Hant"}


def load_flores_hf(lang, split, token):
    """openlanguagedata/flores_plus (maintained) or facebook/flores; both are click-through gated."""
    from datasets import load_dataset
    errs = []
    codes = [lang] + ([FLORES_PLUS_ALIASES[lang]] if lang in FLORES_PLUS_ALIASES else [])
    # 1. flores_plus, per-language config
    for code in codes:
        try:
            ds = load_dataset("openlanguagedata/flores_plus", code, split=split, token=token)
            return _rows_to_lists(ds, "text")
        except Exception as e:  # noqa: BLE001
            errs.append(f"flores_plus[{code}]: {e}")
    # 2. flores_plus, single split with all languages -> filter
    try:
        ds = load_dataset("openlanguagedata/flores_plus", split=split, token=token)
        for code in codes:
            iso3, iso15924 = code.split("_")
            sub = ds.filter(lambda r: r["iso_639_3"] == iso3 and r["iso_15924"] == iso15924)
            if len(sub) > 0:
                return _rows_to_lists(sub, "text")
        errs.append("flores_plus[filter]: 0 rows")
    except Exception as e:  # noqa: BLE001
        errs.append(f"flores_plus[filter]: {e}")
    # 3. facebook/flores
    try:
        ds = load_dataset("facebook/flores", lang, split=split, token=token)
        return _rows_to_lists(ds, "sentence")
    except Exception as e:  # noqa: BLE001
        errs.append(f"facebook/flores[{lang}]: {e}")
    raise RuntimeError("could not load FLORES for %s/%s:\n  " % (lang, split) + "\n  ".join(errs))


def _rows_to_lists(ds, text_field):
    ids = [int(r["id"]) for r in ds]
    txt = [r[text_field] for r in ds]
    order = np.argsort(ids)
    return [ids[i] for i in order], [txt[i] for i in order]


def load_split(args, langs, split, token):
    ids_ref, sents = None, {}
    for lang in langs:
        if args.local_dir and os.path.exists(os.path.join(args.local_dir, split, f"{lang}.{split}")):
            ids, txt = load_flores_local(args.local_dir, lang, split)
        elif args.flores == "hf":
            ids, txt = load_flores_hf(lang, split, token)
        else:
            ids, txt = load_flores_local(args.flores, lang, split)
        if args.smoke:
            ids, txt = ids[:args.smoke], txt[:args.smoke]
        if ids_ref is None:
            ids_ref = ids
        elif ids != ids_ref:
            raise RuntimeError(f"sentence ids for {lang}/{split} do not match {langs[0]}")
        sents[lang] = txt
    return ids_ref, sents


# ----------------------------------------------------------------------------- model plumbing

def _find(model, paths):
    for p in paths:
        try:
            return functools.reduce(getattr, p.split("."), model), p
        except AttributeError:
            continue
    raise AttributeError(f"none of {paths} found on {type(model).__name__}")


def locate(model):
    blocks, bp = _find(model, ["model.layers", "transformer.h", "model.decoder.layers", "gpt_neox.layers"])
    # residual stream entering block 1: embedding output (BLOOM applies a LayerNorm right after embeddings)
    embed, ep = _find(model, ["transformer.word_embeddings_layernorm", "model.embed_tokens",
                              "model.decoder.embed_tokens", "gpt_neox.embed_in"])
    fnorm, fp = _find(model, ["model.norm", "transformer.ln_f", "model.decoder.final_layer_norm",
                              "gpt_neox.final_layer_norm"])
    head, hp = _find(model, ["lm_head", "embed_out"])
    if "gemma" in type(model).__name__.lower():
        print("WARNING: Gemma scales embeddings by sqrt(d) after embed_tokens; layer 0 here is unscaled.")
    return blocks, embed, fnorm, head, dict(blocks=bp, embed=ep, final_norm=fp, head=hp)


class Capture:
    """Forward hooks storing the residual stream after the embedding and after every block."""

    def __init__(self, blocks, embed):
        self.buf = {}
        self.handles = [embed.register_forward_hook(self._mk(0))]
        for i, b in enumerate(blocks):
            self.handles.append(b.register_forward_hook(self._mk(i + 1)))

    def _mk(self, idx):
        def hook(_m, _inp, out):
            self.buf[idx] = out[0] if isinstance(out, (tuple, list)) else out
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def sink_token_id(tok):
    if tok.bos_token_id is not None:
        return int(tok.bos_token_id), "bos"
    ids = tok.encode("\n", add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"newline is not a single token ({ids}); pick a sink token manually")
    return int(ids[0]), "newline"


def wu_basis(head, fnorm, top=512, chunk=16384):
    """Top right singular vectors of W = diag(gain) applied to lm_head rows: eigh of W^T W."""
    W = head.weight.detach()
    g = getattr(fnorm, "weight", None)
    d = W.shape[1]
    C = torch.zeros(d, d, dtype=torch.float32, device=W.device)
    for s in range(0, W.shape[0], chunk):
        w = W[s:s + chunk].float()
        if g is not None:
            w = w * g.detach().float()[None, :]
        C += w.T @ w
    evals, evecs = torch.linalg.eigh(C)  # ascending
    idx = torch.arange(d - 1, d - 1 - top, -1, device=W.device)
    return evecs[:, idx].cpu().numpy().astype(np.float32), evals[idx].cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------------------- pooling

@torch.no_grad()
def pool(h, mask):
    """h [B,T,d] (any dtype), mask [B,T] bool with sink at position 0.
    Returns mean [B,d], last [B,d], normdiag [B,2] all float32."""
    h = h.float()
    content = mask.clone()
    content[:, 0] = False
    cm = content.float()
    n = cm.sum(1, keepdim=True).clamp(min=1.0)
    mean = (h * cm[..., None]).sum(1) / n
    lengths = mask.sum(1)  # sink + content
    last = h[torch.arange(h.shape[0], device=h.device), lengths - 1]
    norms = h.norm(dim=-1)  # [B,T]
    cn = torch.where(content, norms, torch.full_like(norms, float("nan")))
    med = torch.nanmedian(cn, dim=1).values.clamp(min=1e-6)
    mx = torch.where(content, norms, torch.zeros_like(norms)).max(dim=1).values
    diag = torch.stack([norms[:, 0] / med, mx / med], dim=1)
    return mean, last, diag


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="output subfolder name")
    ap.add_argument("--out", required=True, help="final output root (e.g. Drive)")
    ap.add_argument("--tmp", default="/content/scratch", help="local scratch root")
    ap.add_argument("--flores", default="hf", help="'hf' or path to flores200_dataset dir")
    ap.add_argument("--local_dir", default=None, help="dir checked first for {split}/{lang}.{split} files")
    ap.add_argument("--langs", default=DEFAULT_LANGS)
    ap.add_argument("--splits", default="dev,devtest")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--smoke", type=int, default=0, help="use only N sentences per split")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if token is None:
        try:
            from google.colab import userdata  # type: ignore
            token = userdata.get("HF_TOKEN")
        except Exception:  # noqa: BLE001
            token = None

    langs = args.langs.split(",")
    splits = args.splits.split(",")
    tmp = os.path.join(args.tmp, args.tag)
    out = os.path.join(args.out, args.tag)
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    meta_path = os.path.join(out, "meta.json")
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta["model"] != args.model or meta["langs"] != langs:
            sys.exit(f"{meta_path} exists for a different model/language set; use another --tag")
        if not args.overwrite:
            splits = [s for s in splits if s not in meta["splits"]]
            if not splits:
                sys.exit("all requested splits already extracted; pass --overwrite to redo")
        print(f"appending splits {splits} to existing {out}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, token=token)
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, token=token,
                                                 device_map={"": 0})
    model.eval()
    blocks, embed, fnorm, head, paths = locate(model)
    n_layers = len(blocks) + 1
    d = int(model.config.hidden_size)
    sink_id, sink_kind = sink_token_id(tok)
    base = model.base_model
    dev = next(model.parameters()).device
    print(f"model {args.model}: {len(blocks)} blocks, d={d}, sink={sink_kind}({sink_id}), paths={paths}")

    # unembedding geometry (once per model)
    if not os.path.exists(os.path.join(out, "wu_basis.npy")):
        ub, ue = wu_basis(head, fnorm)
        np.save(os.path.join(tmp, "wu_basis.npy"), ub)
        np.save(os.path.join(tmp, "wu_eigs.npy"), ue)

    if meta is None:
        meta = dict(model=args.model, langs=langs, n_layers=n_layers, d=d, sink_token_id=sink_id,
                    sink_kind=sink_kind, dtype=args.dtype, paths=paths, splits={}, max_tokens=args.max_tokens,
                    layer0="embedding output (pre block 1); layer i = output of block i",
                    pooling="mean: masked mean over positions 1..len-1; last: position len-1",
                    layout="{pooling}_{split}_L{layer}.f16.npy -> [n_lang, n_sent, d]")

    cap = Capture(blocks, embed)
    for split in splits:
        ids, sents = load_split(args, langs, split, token)
        n = len(ids)
        meta["splits"][split] = dict(n=n, ids=ids)
        shape = (len(langs), n, d)
        f_mean = [np.lib.format.open_memmap(os.path.join(tmp, f"mean_{split}_L{layer}.f16.npy"), mode="w+",
                                            dtype=np.float16, shape=shape) for layer in range(n_layers)]
        f_last = [np.lib.format.open_memmap(os.path.join(tmp, f"last_{split}_L{layer}.f16.npy"), mode="w+",
                                            dtype=np.float16, shape=shape) for layer in range(n_layers)]
        f_diag = np.lib.format.open_memmap(os.path.join(tmp, f"normdiag_{split}.f16.npy"), mode="w+",
                                           dtype=np.float16, shape=(len(langs), n, n_layers, 2))
        ntok = np.zeros((len(langs), n), dtype=np.int32)

        for li, lang in enumerate(langs):
            t0 = time.time()
            enc = [tok.encode(s, add_special_tokens=False)[: args.max_tokens - 1] for s in sents[lang]]
            enc = [[sink_id] + e for e in enc]
            ntok[li] = [len(e) - 1 for e in enc]
            order = np.argsort([-len(e) for e in enc])  # long first: fail fast on OOM
            for s in range(0, n, args.batch):
                idx = order[s:s + args.batch]
                L = max(len(enc[i]) for i in idx)
                ids_t = torch.full((len(idx), L), tok.pad_token_id if tok.pad_token_id is not None else 0,
                                   dtype=torch.long)
                mask = torch.zeros((len(idx), L), dtype=torch.bool)
                for r, i in enumerate(idx):
                    ids_t[r, :len(enc[i])] = torch.tensor(enc[i])
                    mask[r, :len(enc[i])] = True
                ids_t, mask = ids_t.to(dev), mask.to(dev)
                with torch.inference_mode():
                    cap.buf.clear()
                    base(input_ids=ids_t, attention_mask=mask.long(), use_cache=False)
                    for layer in range(n_layers):
                        m, l, dg = pool(cap.buf[layer], mask)
                        f_mean[layer][li, idx, :] = m.cpu().numpy().astype(np.float16)
                        f_last[layer][li, idx, :] = l.cpu().numpy().astype(np.float16)
                        f_diag[li, idx, layer, :] = dg.cpu().numpy().astype(np.float16)
            print(f"  {split} {lang}: {n} sents, mean {ntok[li].mean():.1f} tok, {time.time() - t0:.0f}s")
        for f in f_mean + f_last + [f_diag]:
            f.flush()
        del f_mean, f_last, f_diag
        np.save(os.path.join(tmp, f"ntok_{split}.npy"), ntok)
        with open(os.path.join(tmp, f"sentences_{split}.json"), "w", encoding="utf-8") as f:
            json.dump(dict(ids=ids, text={l: sents[l] for l in langs}), f, ensure_ascii=False)
    cap.close()

    if os.path.abspath(tmp) != os.path.abspath(out):
        print(f"copying {tmp} -> {out}")
        shutil.copytree(tmp, out, dirs_exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    print("done")


if __name__ == "__main__":
    main()
