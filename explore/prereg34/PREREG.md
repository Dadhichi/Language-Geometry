# Pre-registration: tree vs word-order structure in 34-language centroid geometry (Qwen2.5-7B)

Written and committed 2026-09-30 BEFORE the 34-language activations were extracted. The analysis code
(`lib34.py`, `analysis34.py`) is frozen at this commit; anything added after the data arrive goes into a separate,
explicitly exploratory script. Motivation: explore/IDEAS.md candidates 1 and 3; the 12-language pilot
(explore/E_tree) was underpowered (18-42%) and found a post-hoc head-final split {hin, jpn, tur}.

## Data
Qwen/Qwen2.5-7B, FLORES+ dev (997) and devtest (1012) sentences, mean pooling over content tokens (sink dropped),
residual stream after the embedding and after every block (29 layers), extracted by `extract.py` at this commit.
34 languages (FLORES+ codes):
eng deu nld swe (Germanic) | fra spa por (Romance) | rus ukr pol hrv srp (Slavic) | hin urd mar pes (Indo-Iranian)
| arb heb mlt (Semitic) | tur azj kaz (Turkic) | fin ekk (Finnic) | cmn_Hans cmn_Hant (Chinese, two scripts) | jpn |
kor | ind fil (Austronesian) | vie khm (Austroasiatic) | tam tel (Dravidian).

## Statistic
Per layer and metric, the cross-split Gram of centroid offsets K = A_dev M A_devtest^T (A = language means minus
their mean; unbiased for the noiseless Gram). Metrics: causal (M = diag(g) Cov(gamma) diag(g), Park et al. causal
inner product from Qwen's lm_head, g = final RMSNorm weight) and lda05 (M = W_0.5^-1, W = pooled within-language
covariance on dev, shrinkage 0.5 toward (trW/d) I). PRIMARY input: the average over layers 8..20 of the
trace-normalised Grams, then D2_ij = K_ii + K_jj - 2 K_ij over the 561 pairs.

Model: D2_ij = l_i + l_j (star) + sum over script splits {Latn, Cyrl, Arab, Deva} + beta_tok * token distance
+ beta_fert * |dlog tokens/sentence| + tested splits; all coefficients >= 0 (NNLS). Token distance = 1 - histogram
intersection of the languages' Qwen-token frequency distributions on FLORES dev+devtest. Gain = fraction of the base
residual SSE removed by the tested splits.

## Hypotheses and tests (10^4 permutations each)
- H1 (genealogy): gain of the 20 Glottolog internal splits (lib34.GLOTTO). Null: leaf labels permuted.
- H2 (word order): gain of the OV split (WALS 83A value 1: hin urd mar pes tur azj jpn kor tam tel; kaz not coded,
  OV by genus). Null: random 11-language subsets.
- Primary family: {H1, H2} x {causal, lda05}, Bonferroni alpha = 0.0125.
- Because the OV set overlaps whole families, the marginal tests leak into each other (synthetic check: a planted
  tree makes H2 reject ~0.3 of the time; planted OV makes H1 reject 0.1-0.2). Therefore the CLAIMS are defined as:
  * "genealogy structure": H1 rejects AND H1|OV (genealogy over base + OV split) has p < 0.0125, same metric.
  * "word-order structure": H2 rejects AND OV|H1 (OV over base + all 20 Glottolog splits) has p < 0.0125.
  A claim must hold in at least one primary metric and not reverse sign in the other.
- H3 (secondary, same language / two scripts): hin-urd, hrv-srp, cmn_Hans-cmn_Hant should be closer than predicted
  by star + token + fertility (no script terms). Statistic: mean percentile of their residuals among all pairs;
  null: random pair triples. alpha 0.05.
- Secondary (reported, not claimed): OV + deu/nld (WALS 83A 'no dominant order'); sqrt(ntok)-rescaled pooling
  variant; per-layer x metric profiles (euc, lda01, lda05, causal) with 500 permutations; bootstrap 90% intervals of
  the primary gains (50 sentence resamples).

## Validation before data (synthetic worlds, same code; synth_check34.log, synth_cond34.log)
Planted BM worlds with a star, a high-fertility outlier and nuisance script clades. Marginal tests (40 reps, 300
perms, alpha .05): null H1 .05 / H2 .10; genealogy c=.1: H1 .93 / H2 .28; OV c=.1: H1 .12 / H2 .65; genealogy c=.2:
H1 1.00 / H2 .30; OV c=.2: H1 .23 / H2 .95. Conditional tests (60 reps, 200 perms; synth_cond34.log, completed
before any 34-language data were downloaded): null H1|OV .07 / OV|H1 .03; genealogy c=.1: .92 / .08; OV c=.1:
.10 / .50; genealogy c=.2: 1.00 / .12; OV c=.2: .02 / .93. The conditional tests separate the two structures, so
the claim rules above stand unchanged.

## What would change our mind
Genealogy claim with no word-order claim -> emergent phylogeny (candidate 1). Word-order claim without genealogy ->
typological parameter geometry (candidate 3). Neither -> the mid-depth geometry is a star + script/token effects at
34 languages (a documented dead end for candidates 1 and 3). H3 positive -> language identity beyond script.
