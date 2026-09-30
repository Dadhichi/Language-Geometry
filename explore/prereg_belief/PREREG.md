# Pre-registration: is language identity a belief simplex at the token level? (Qwen2.5-7B)

Written and committed 2026-09-30 BEFORE the token-level data were extracted. Code frozen at this commit:
`extract_tokens.py` (repo root), `analysis_belief.py` (here). Motivation: explore/IDEAS.md candidate 2; the
sentence-level pilot (explore/F_belief) found an evidence-specific pull (gain 0.1-0.3) with a cumulative-prefix
component, strongest on the zho-jpn edge, but sentence averaging hides the simplex interior.

## Data (extract_tokens.py)
FLORES+ devtest, 12 languages (eng deu fra spa rus hin arb zho jpn tur vie ind), every content position, 15 layers
(0..28 step 2), projected onto P_l = [S_l | U_l]: S_l = 11-d span of the 12 language-centroid offsets (mean-pooled
FLORES dev centroids, span12.npz), U_l = top 117 PCs of token states orthogonal to S_l (pilot of 100 sentences/lang).
Model's own evidence: log p(token | "The following text is written in {Language}.\n" + prefix) for all 12 tags,
first 24 content tokens of every sentence. Code-switch: first fraction p in {.1..,.9} of sentence s in language A
(word units; characters for zho/jpn) + the remaining 1-p of the same sentence in B; 8 pairs x 2 directions x 200
sentences; states at 8 layers; log-likelihoods under the A and B tags.

## Definitions
z_t = S_l^T h_t (language-span coordinates). Posterior pi_t = softmax_k(sum_{tau<=t} log p(x_tau | tag_k)),
uniform prior. Ambiguous position: max_k pi_t < 0.9. Sentences split 50/50 into fit / test halves (seed 0),
same split for every model; all maps are ridge regressions (lambda chosen by 5-fold CV on the fit half).

## Primary tests (Bonferroni alpha = 0.025 each)
P1 (belief-affine): on ambiguous test positions, the affine map z ~ pi (12 posterior probabilities) has lower
  held-out MSE than z ~ onehot(argmax pi). Statistic: MSE difference averaged over layers 8,10,12,14,16; one-sided
  p from a paired bootstrap over test sentences (10^4).
P2 (Bayesian code-switch): with the natural-text centroids c_A, c_B (fit half, per layer, span coordinates, no
  refitting on code-switched data), lambda_t = (c_B - c_A).(z_t - c_A)/||c_B - c_A||^2 at every position of each
  code-switched sequence (before and after the switch). c_k = mean span coordinates of language k's natural tokens.
  (i) held-out MSE of lambda ~ a + b sigmoid(LLR_t) (LLR_t = cumulative log p(.|B) - log p(.|A) over the whole
  sequence) is lower than lambda ~ a + b step_t (step = 1 after the switch); (ii) lag: tokens after the switch until
  lambda > 0.5 increases with the A-prefix length (Spearman > 0 pooled over pairs). Both (i) and (ii) must hold,
  averaged over layers 8,12,14,16; one-sided p by bootstrap over sentences.

## Secondary (reported, not claimed)
Per-layer profiles of P1/P2; P1 against the true-language one-hot and against token-identity + position baselines
(ridge on the layer-0 embedding of the current token, projected to span coordinates, plus position one-hot), and
whether pi adds to token identity; per-pair P2 (zho-jpn expected strongest); projection loss (full-width subset);
shared-token displacement (Latin-script tokens inside non-Latin sentences move toward the Latin-script centroid).

## What would change our mind
P1 and P2 both positive -> the residual stream carries a graded, Bayes-like belief over languages (candidate 2).
P1 negative / snap in P2 -> language identity is a discrete switch driven by token identity, and the sentence-level
pull was averaging, not belief.
