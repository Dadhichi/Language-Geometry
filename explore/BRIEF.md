# Shared brief: hunting for mathematically elegant structure in cross-language maps (Qwen2.5-7B)

You are one of four parallel investigators. Read this whole file, then `CLAUDE.md` in the repo, before writing code.

## Project in one paragraph
Inside one multilingual LLM (Qwen2.5-7B, 28 blocks, d=3584), we study the maps between languages' sentence
representations (mean-pooled residual stream, FLORES-200 parallel sentences). Rungs: (a) constant shift,
(b) shift + per-language rotation (a "gauge": pairwise maps factor as R_ij = Q_i Q_j^T, orthogonal group
synchronization), (c) linear non-orthogonal gauge (GL), (d) no single shared hub (pair-specific structure).
Repo: `C:\Users\ASUS\OneDrive\Documents\Projects\Language Structure in LLMs\Language-Geometry` (read
`CLAUDE.md` there: conventions, and the "design decisions found the hard way" -- do not re-learn them).
`fit.py` is importable and has `LA`, `pca_basis`, `make_projector`, `analyze`, `analyze_with_null`, `synth`.
DO NOT modify anything in the repo. Work only in your own directory (given in your prompt).

## Inspiration: Jane Street, "Using group theory to explore the space of positional encodings" (Apr 2026)
Axioms (linearity, relative-position invariance A(s-t) = F(s)^T G(t), continuity) force a one-parameter
group A(tau) = exp(tau M); classifying M by Jordan form enumerates all encodings: complex eigenvalues ->
rotations (RoPE), real eigenvalues -> exponential decay/scaling, complex with real part -> damped rotation
(RetNet/Mamba-3), defective (Jordan blocks) -> polynomial/shear (ALiBi-like). The lesson we want: state a
clean axiom, derive the forced algebraic form, classify it, then check the data. The analogue here:
"the map between two languages depends only on their relative position", R_ij = exp((theta_j - theta_i) M)
(or a commuting family M_p), and a Jordan-type classification of what a cross-language map is made of
(rotation vs scaling vs shear vs translation). Elegance is the goal, but only if the data support it.

## Data (local, read-only): `C:\Users\ASUS\Documents\lang-geom\`
- `mean_{split}_L{layer}.f16.npy` -> float16 [12 langs, n_sent, 3584]. splits: `dev` (n=997, use to FIT),
  `devtest` (n=1012, use to TEST). Layers available: 0,2,4,...,28 (0 = embedding output, l = output of
  block l). Sentences are parallel: row s is the same sentence in every language.
- `ntok_{split}.npy` [12, n] content-token counts (fertility; Hindi ~120 tok/sent vs ~30-50 others).
- `sentences_{split}.json` {ids, text{lang: [..]}}; `meta.json`.
- `wu_basis.npy` [3584, 512]: top-512 right singular vectors (columns) of final_norm.weight * lm_head.weight
  (the unembedding's dominant input directions); `wu_eigs.npy` [512] their eigenvalues of W^T W.
- `normdiag_{split}.f16.npy` [12, n, 29, 2]: (sink norm / median content norm, max / median content norm).
Language order (index: code, script, family):
 0 eng_Latn Latin Germanic(IE) | 1 deu_Latn Latin Germanic(IE) | 2 fra_Latn Latin Romance(IE) |
 3 spa_Latn Latin Romance(IE)  | 4 rus_Cyrl Cyrillic Slavic(IE) | 5 hin_Deva Devanagari Indo-Aryan(IE) |
 6 arb_Arab Arabic Semitic     | 7 zho_Hans Han Sinitic        | 8 jpn_Jpan Kana/Han Japonic |
 9 tur_Latn Latin Turkic       | 10 vie_Latn Latin Austroasiatic | 11 ind_Latn Latin Austronesian

## Environment
Python with numpy/scipy/pandas/scikit-learn/torch(cpu)/matplotlib:
`C:/Users/ASUS/AppData/Local/Temp/claude/c--Users-ASUS-OneDrive-Documents-Projects-Language-Structure-in-LLMs/fa049fb0-6ccd-47bf-be70-a43fa5fa9e65/scratchpad/venv/Scripts/python`
Use `PYTHONIOENCODING=utf-8`. Windows + Git Bash. Import fit.py via `sys.path.insert(0, <repo>)`.
RESOURCES ARE SHARED with three other agents: only ~3.7 GB RAM free in total. Set
`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`, load ONE layer at a time
(`np.load(..., mmap_mode='r')`, convert per language / project immediately), keep peak RAM < 1.2 GB,
delete arrays you are done with. Keep each script run < 15 min; total budget ~1.5-2 h of wall time.

## What is already known (FLORES dev->devtest, mean pooling, per-language PCA; ratios at layers 12-16)
- Depth profile U-shaped: procrustes residual rho (||pred-tgt||^2/||tgt-mu||^2) 1.48 (L0) -> ~0.40
  (L14-19, k=64) -> 0.81 (L28); P@1 retrieval up to 0.95-0.98 at L14-19; last layer breaks.
- Procrustes maps do NOT compose: c = ||T_i(R_ij R_jm - R_im)||^2/||T_i||^2 is 4.5x / 2.5x / 1.5x the
  consistent null at k=64/128/256; cocycle 0.95 vs null 0.995 (k=64). Null spread is tiny (~2%).
- Yet the joint (GPA hub) model predicts almost as well as pairwise maps (sync_gap/rho = 0.2-0.7%).
- Ridge (GL) beats procrustes by ~30% in rho, and ridge maps compose within the null (from L3 at k=256).
- Per-language subspaces are poorly estimated at n=997: split-half subspace agreement 0.40-0.60; at
  k=256 split-half map noise (0.39) is ~ the residual (0.53). So "ridge composes, procrustes doesn't"
  could be a real linear (GL) gauge OR an artefact of noisy per-language subspaces. Unresolved.
- Early TED (n_fit=6000) numbers: at L0/L4 k=256 the null tightens (0.41/0.21) while real c stays
  (0.87/0.53) -> excess composition error grows with more data, i.e. it looks REAL at least early on.
- Pivots: every pivot hurts (delta<0); English least bad in L0-10, French/Spanish from L11; Chinese poor
  (rank 8-9/12) despite being a dominant Qwen language; Hindi worst (tokenizer fertility suspected).
- Last-token pooling shows the same qualitative picture (weaker alignment, peak at L19-22).

## Methodological rules (non-negotiable)
1. GAUGE INVARIANCE. A per-language PCA basis is arbitrary: any statistic in per-language coordinates must
   be invariant under z_i -> z_i O_i for arbitrary, independent O_i in O(k). Eigenvalues/eigenvectors of a
   single map R_ij between two different languages' coordinates are NOT meaningful. Meaningful: singular
   values of maps, loop/holonomy spectra (R_ij R_jm R_mi acts on language i's own space), principal angles,
   CCA spectra, anything computed in the AMBIENT residual space. The ambient 3584-d space is shared by all
   languages AND all layers (residual stream), so identity/shift/eigen-structure are meaningful there -- but
   n_fit=997 < d, so restrict to a shared subspace or regularize, and remember CLAUDE.md design decision 1:
   a SHARED truncation manufactures composition inconsistency, so never use shared-subspace composition as
   evidence without a synthetic control run through the identical pipeline.
2. NULLS AND CONTROLS. Every claimed structure must beat an appropriate null: a consistent surrogate (as in
   fit.py analyze_with_null: GPA hub + permuted residuals), sentence-shuffle or label-permutation controls,
   and held-out evaluation (fit on dev, score on devtest). Check at >= 2 layers and >= 2 values of k.
3. SYNTHETIC FIRST. Before trusting a new statistic on real data, run it on synthetic data where the
   answer is known (fit.synth scenarios, or build your own with a realistic spectrum, e.g. use a real
   language's cloud as the hub and apply known transformations + matched noise). Report both.
4. Estimation noise at n=997 is large. Prefer statistics that are stable under split-half resampling;
   report split-half variability.
5. Fertility confound: Hindi's token counts differ ~3x; check that a structure is not just a length effect.
6. A documented dead end is a valid, valuable outcome. Do not oversell. Distinguish "supported",
   "refuted", "inconclusive (needs TED n=6000 / full d)".

## Deliverable
Save scripts + outputs + `report.md` in your directory. Your FINAL MESSAGE is the report (<= ~700 words):
hypothesis as a crisp axiom/algebraic form; what was tested (with the synthetic validation); key numbers
with their nulls/controls, by layer; verdict; the most elegant true statement you can make; caveats; and the
single most decisive follow-up experiment (what data/compute it needs, e.g. TED n=6000 or full d).
