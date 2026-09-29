# Structure hunt, 2026-09-29 (Qwen2.5-7B, FLORES dev->devtest, mean pooling, layers 0..28 step 2)

Four independent investigations (brief: `BRIEF.md`; code and tables in `A_gauge_group/`, `B_holonomy/`,
`C_positions/`, `D_offsets/`). Scripts read activations from `C:\Users\ASUS\Documents\lang-geom\` (local copy of
`$DATA/qwen25_7b` mean_{dev,devtest}_L{0,2,..,28}, meta, ntok, sentences, wu_basis). Every statistic was
validated on synthetic worlds run through the identical pipeline and compared with nulls/controls.

## The law, in order of size
x_{i,s} ~= mu_i + e^{s_i} b_s + (small non-commuting rotation) + e_{i,s}   (ambient residual coordinates)

1. **Translation (first order).** A constant per-language offset removes 82-91% of the removable cross-language
   residual at every layer (C). Subtracting language centroids alone gives P@1 0.99 at L14, 0.97 at L24, 0.80 at
   L4 (D; re-verified independently). In fit.py's own per-language-PCA pipeline the identity-gauge map
   R = B_i^T B_j beats fitted Procrustes at every layer but L28 (L14: rho 0.33 vs 0.40) (D).
   Offsets: ~5-dim (participation ratio 5-9), mostly outside content (12% in top-64 content PCs vs 56% for
   content), sentence-length invariant (cos >= 0.97), smallest at L12-16 (U-shape), configuration shape stable
   across depth but re-embedded at L0->L2 and L26->L28 (D).
2. **Isotropic dilation (second order) = mean-pooling dilution.** Per-language scale s_i ~= -1/2 log ntok_i
   (corr -0.92..-0.96 at every layer; A, C, D independently). L14: hin x0.64, tur x0.85, eng x1.21, zho x1.18.
   Control with nothing fitted: rescale each sentence vector by sqrt(ntok) -> scale variance -80%, O->Sim gain
   0.047 -> 0.007, Hindi 0.64 -> 1.12 (A). One scalar per language beats a full rotation held-out (C, L14:
   .286 vs .327, shift-only .343).
3. **Rotation (third order, small).** ~2-4% of variance mid-depth, ~15% at L26; spread over ~16 planes,
   non-commuting (commutator 0.55-0.72 vs 0 abelian / 0.68-0.80 random), yet composing (C).
   Residual anisotropy beyond scalar scale: Hindi and Turkish (rms log-stretch .28/.19 vs floor ~.09 at L14),
   survives sqrt(ntok) rescaling (A). Shear/non-normality at noise level (C).

## The earlier "Procrustes maps do not compose (c = 1.5-4.5x null)" is an artefact
Per-language top-k PCA truncation + a mis-specified null. fit.py's null assumes every language's top-k subspace
holds the same content; real subspaces overlap only ~0.64 (mean cos^2, k=64). Evidence:
- Exactly consistent synthetic worlds through the same pipeline reproduce the real c (D: identity world 0.091
  vs real 0.077 vs fit.py null 0.017 at L14 k=64; A: orthogonal worlds bracket real c at every layer; B: flat
  O(2k) surrogate 0.099 vs real 0.077).
- Aligned-slice test (every language gets the hub's top-k carried into its coordinates): real c 0.0016 vs flat
  surrogate 0.0020 (L14 k=64); same at L4, L26 (B).
- The spurious holonomy has the truncation signature: 1-3 planes turned by ~pi at each language's spectral edge
  (real 158 deg, flat-truncated 167-169 deg, fit.py null 32 deg) (B). Its per-language amplitude tracks fertility
  (r = .95 with log ntok) (B).
- In shared ambient coordinates maps compose within the consistent synthetic worlds (C).
- "Ridge composes" is uninformative: c_ridge sits at its null in every world, including a true-GL world (A, B).
  Ridge advantage over Procrustes = noise shrinkage (47-80%) + scalar scale (15-40%) + anisotropy (7-13%) (A).

## Refuted or unsupported
- RoPE-like relative-position axiom for rotations R_ij = exp(sum_p (theta_j - theta_i)_p M_p): refuted (C). The
  only 1-D "position" found is the dilation, i.e. log token count (|r| .93-.95), a pooling artefact.
- Low-rank / abelian / shared-subspace curvature; script/family structure in curvature (B).
- Additive attributes mu_i = c + script + family: LOO R^2 -0.27..-0.44 vs -0.01 planted (D).
- Language offsets rotating into the unembedding late: offsets are never more W_U-aligned than content
  (ratio 0.90-1.28) (D); C sees ~2x chance in W_U top-512, but content is similarly enriched.

## Open / exploratory
- Centroid metric reads tokenizer (L0) -> family (L8-24, Mantel p <= .04, Hindi excluded) -> script (L28) (D);
  C's 1-D positions show no script/family organisation. Weak (12 languages).
- A second factor grouping head-final hin/tur/jpn vs eng/fra/spa at L8-24 (p ~ .004), found post hoc (D).

## Prior art to check before claiming anything (CLAUDE.md working rule)
Language-centroid subtraction improving cross-lingual retrieval/neutrality is known for encoders (e.g. Libovicky
et al. 2020, "On the language neutrality of pre-trained multilingual representations"; Chang, Tu & Bergen 2022,
"The geometry of multilingual language model representations": language-sensitive vs language-neutral axes).
Candidate contributions: the decoder-LLM depth profile; the pooling-dilution law s ~ ntok^-1/2; and the
methodological result that per-language PCA truncation manufactures non-composition (with the aligned-slice fix).

## Consequences for the pipeline
1. fit.py's consistent null is wrong for per-language truncation. Replace with an ambient null
   (x = m + a_i + g_i b + permuted e_i, or a flat O/GL(kb >> k) surrogate) run through the identical pipeline,
   and/or add B's aligned-slice analysis. Until then, c vs null_c in summary.csv is not evidence.
2. Normalise pooling for length: sqrt(ntok) * mean (= sum/sqrt(n)) is exact from existing files; per-token-
   normalised means need re-extraction.
3. Move the primary analysis to shared ambient coordinates: shift, then scalar gain, then (regularised) rotation.
4. Decisive: TED n_fit=6000 > d, full-d ambient maps (A100): is R_ij ~= I after shift + gain? Does the small
   non-commuting rotation and the Hindi/Turkish anisotropy survive length-normalised pooling?
