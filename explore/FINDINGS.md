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

# Pilots of IDEAS.md candidates 1-2 on the same 12-language data (2026-09-29; E_tree/, F_belief/)

## E: tree symmetry / Haar (candidate 1) -- inconclusive, leaning unsupported
Four metrics (Euclidean, LDA shrink .1/.5, Park causal IP from the real lm_head: diag(g) Cov(gamma) diag(g);
unembedding + covariance saved in C:\Users\ASUS\Documents\lang-geom\). Unbiased dev x devtest Grams; NNLS tree fits
on D^2; exact clade enumeration + 2000 relabellings; synthetic Brownian-on-tree validation (false positives 2-4%).
- Power at 12 languages is low: 18-42% for a realistic internal-branch share; Park orthogonality 4-9%.
- Mid-depth geometry is a STAR (per-language own-branch, set by fertility; Hindi far) explaining 51-83% of D^2;
  Glottolog splits add 0.02-0.11 (p .07-.29); genealogy's unique share <= 0.015 at L4-22.
- What survives token/script controls: {hin, jpn, tur} (head-final) is the best 3-set of 220 (p = 1/220) at L4-20
  in all four metrics, 26-49% of the remaining misfit; runners-up add deu (OV). Replicates D post hoc (same data).
- Decisive spec: 34 FLORES languages (nested IE incl. hin/urd/mar/pes; Semitic, Turkic, Finnic, Sinitic x2 scripts,
  Austronesian, Austroasiatic, Dravidian; OV spread across families; same-language two-script pairs), analysis fixed
  in advance (L8-20 averaged Gram, causal + LDA05, 1e4 relabellings, script clades as nuisance): power 80-86% at
  c = 0.1. See E_tree/ report numbers in the session log.

## F: belief simplex at sentence level (candidate 2) -- weak form supported, strong form inconclusive
Offsets o_{i,s} = x_{i,s} - mean_i' x_{i',s} in the 11-d centroid span; instruments from text (shared-token /
shared-char fractions; unigram naive-Bayes observer: bag vs cumulative-prefix posteriors); gain g fitted on dev with
target-demeaning (removes generic shrinkage exactly) + controls, scored on devtest; synthetic worlds recover
planted gains 0.25/1.00 and attribute bag vs prefix correctly.
- Evidence-specific pull toward the right vertices at every layer (z = 10-120), gain 0.1-0.3 (not 1), largest L0-14.
- L0 pull is pure bag (as predicted without context); from L4 a cumulative-prefix term (0.10-0.16 at L8-14) appears
  beyond early-token position (Bayesian-like), but a context-free bag term (0.2-0.3) persists (non-Bayesian, or a
  local-span language variable).
- zho-jpn edge is the most belief-like: prefix pull 0.6-0.8 at L4-12 while the bag pull vanishes mid-depth.
- Packing (vertex geometry ~ textual confusability) holds beyond script L0-24 (p <= .03); exception: eng-zho is the
  closest pair at L4-26 despite no shared evidence -> a dominant-language axis (cf. Qwen's en+zh pretraining).
- Sentence averaging hides the simplex interior (span holds 2-6% of sentence-specific offset energy mid-depth).
- Decisive spec (A100, < 1 GPU-hour, ~10 GB): token-level FLORES devtest x 12 langs at 15 layers (128-d projection),
  the model's own P(language | prefix) via 12 language-tag prompts, code-switch continua (8 pairs x 200 sentences x
  11 switch points), word-swap mixtures p = 0..1, ambiguity probes. Tests: posterior-affine map beats one-hot and
  token-identity baselines at ambiguous positions; post-switch paths stay on the A-B edge and follow the logistic of
  the cumulative log-likelihood ratio under the natural-text map without refitting.

# PRE-REGISTERED 34-language test (explore/prereg34; prereg 0d40bdd + d976f88, results 1e1d796)
Qwen2.5-7B, 34 FLORES+ languages, mean pooling, primary = L8-L20 averaged cross-split Gram, NNLS split model over
star + script splits + token distance + |dlog fertility|, 1e4 permutations, Bonferroni alpha 0.0125. Synthetic
validation beforehand: conditional tests separate planted genealogy from planted word order.

| metric | R2 star / base | H1 genealogy (gain, p) | H1given OV p | H2 OV (gain, p) | OV given H1 p | H3 same-lang 2-script p |
|---|---|---|---|---|---|---|
| causal | .52 / .85 | .096, .0005 | .0007 | .114, .0002 | .0001 | .0078 |
| lda05  | .75 / .92 | .288, .0001 | .0001 | .195, .0001 | .0001 | .0001 |

=> BOTH pre-registered claims hold in both metrics: the mid-depth language geometry carries genealogical tree
structure AND an object-verb word-order axis, each beyond the other, beyond script, token overlap and fertility.
Sentence-bootstrap 90% intervals are ~+-0.001 (sampling noise negligible). Same-language/two-script pairs
(hin-urd, hrv-srp, cmn Hans-Hant) are among the closest 0.4-0.7% of 561 pairs under lda05 (7-21% under causal).
Secondary: sqrt(ntok) pooling keeps H1/H2 (p <= .0025) but H3 weakens (p .08-.11); adding deu/nld (WALS 'no
dominant order', V2 main clauses) to OV weakens it (gain .070 vs .114): main-clause OV order is what aligns.
Depth (descriptive): genealogy gain is largest early (L1-L5, .25-.47) and late (L23-L27, .26-.49) and smallest
mid-depth (~.08-.10 causal); OV gain peaks mid-late (L10-L20, .10-.24) and vanishes at L26-L28.

EXPLORATORY robustness (exploratory34.py, after seeing results): adding great-circle geography (WALS coordinates)
and leave-one-family-out (11 families): OV|H1 significant in all 24 runs (p <= .005; weakest dropping Turkic,
gain .044, or Indo-Iranian/Dravidian, .07); H1 significant in all (weakest dropping all of IE, n=18, p=.046 causal).

Caveats: (1) genealogy may partly be cognate/lexical overlap that unigram token distance misses -- the early/late
peaks fit a lexical origin; the mid-depth part is the interesting residue. (2) OV stands for a correlated
typological bundle (OV ~ postpositions ~ head-final); this design cannot say which feature. (3) One model, mean
pooling, no training-data-size covariate. Next: Llama-3.1-8B replication; cognate-controlled lexical distance;
WALS 85A (adpositions) vs 83A to separate the bundle; per-layer figure.

## Pre-registered REPLICATION on Llama-3.1-8B (PREREG_llama.md d392d70; results34_llama.json)
Same languages/tree/OV/tests/claim rules; primary window L9-L23 of 32 blocks (same relative depth); causal metric from
Llama's lm_head (rows < 128000). Both claims replicate in both metrics:

| metric | R2 star / base | H1 genealogy (gain, p) | H1 given OV p | H2 OV (gain, p) | OV given H1 p | H3 p |
|---|---|---|---|---|---|---|
| causal | .62 / .86 | .294, .0001 | .0001 | .281, .0001 | .0001 | .0001 |
| lda05  | .73 / .91 | .340, .0001 | .0001 | .239, .0001 | .0001 | .0001 |

Same depth profile as Qwen: OV gain rises from ~0 at L0 to a mid-depth peak (L10-L16, causal .27-.37) and fades to
~.01-.05 in the last third; genealogy is high at L0 (lexical), ~.2 mid-depth, and highest in the late layers
(L18-L30, .4-.6). sqrt(ntok) pooling: H1/H2 hold (p <= .0015; lda05 OV gain shrinks to .017), H3 .017 / .165.
Two models with different tokenizers, data and training runs show the same two structures at the same relative depth.

# PRE-REGISTERED token-level belief-simplex test (explore/prereg_belief; prereg d8f2d66, data at c1658c2, results 99f9a0b)
Qwen2.5-7B, FLORES devtest x 12 languages, every token, 15 layers projected on [11-d language span | 117 PCs];
the model's own P(language | prefix) from 12 language-tag-conditioned log-likelihoods; 28,800 code-switch sequences.
- P1 (graded belief) SUPPORTED: at ambiguous positions (max posterior < .9; 26k test positions) an affine map from the
  posterior predicts span coordinates much better than the argmax one-hot (mid-depth MSE difference 55.6, p = .0001;
  holds at every layer). Secondary: the TRUE language one-hot beats the posterior (the internal state is more certain
  than the tag-prompt estimate of the model's belief); token identity + position beats the posterior alone, but adding
  the posterior improves on token identity at every layer (e.g. L14 102 -> 85): belief carries information beyond the
  current token.
- P2 (Bayesian code-switch) NOT SUPPORTED: after a switch the representation jumps to the new language within ~1-1.5
  tokens; a step fits far better than sigmoid(cumulative LLR) (mid-depth MSE .11 vs .25). One Bayes-like trace: the
  lag grows with the length of the preceding A prefix (Spearman +.13, p = .0005).
- EXPLORATORY (exploratory_belief.py): a leaky integrator of the same LLR (best half-life 8 tokens) does not rescue it
  (MSE .21 vs step .11; step + leaky .123 vs .126). Reading: the residual stream tracks the language of the LOCAL span
  (near-step at switches, small hysteresis growing with prefix length) with graded uncertainty when evidence is
  ambiguous -- not a whole-document Bayesian posterior. Caveat: the tag-prompt posterior is a noisy proxy for the
  model's belief (the true label beats it), which weakens every test that uses it.
