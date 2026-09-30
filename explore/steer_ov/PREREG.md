# Pre-registration: is the OV word-order direction causally used? (Qwen2.5-7B)

Written and committed 2026-10-01 BEFORE the steering run. Frozen code: make_pairs.py, directions.py, steer_score.py,
analysis_steer.py (this commit). Motivation: explore/prereg34 found (and Llama replicated) an OV word-order axis in
mid-depth language geometry. Represented is not the same as used; this tests use.

## Materials
- Minimal pairs (make_pairs.py, pairs.jsonl): UD test treebanks, 8 languages (eng, fra, rus, zho: VO; hin, tur, jpn:
  OV; deu: mixed), 999 pairs. A clause-level verb (with its adjacent functional suffixes) and its contiguous object
  subtree are swapped; both variants rebuilt by one detokeniser. Each pair: the OV-order and the VO-order text.
- Directions (directions.py, from explore/prereg34 Qwen dev centroids): per layer and per test language h, OLS over
  the 33 languages != h of centroid offsets on [1, OV, IE, script dummies, log fertility]; beta_OV = OV coefficient,
  beta_IE = IE coefficient rescaled to ||beta_OV|| (structured control); 32 random unit directions in the span of the
  33 centered centroids, scaled to ||beta_OV|| (null). Before any steering, held-out projections on beta_OV at L14:
  hin +.48, tur +.64, jpn +.40 (OV) vs eng -.20, fra -.33, rus -.14 (VO), deu +.05, zho +.03.

## Intervention and measurement (steer_score.py)
Add k * v (v = the pair language's held-out direction) to the output of block l at every position; score both
variants: sum of log p(content tokens | sink). Margin m = log p(OV variant) - log p(VO variant); Delta = m(k) - m(0)
per pair. Conditions: L14 (primary), L8 and L20 (secondary); beta_OV and beta_IE at k in {-2,-1,1,2}; the 32 random
directions at L14, k in {-2,2}.

## Primary test
Slope b = sum_k k * mean(Delta_k) / sum_k k^2 over k in {-2, 2}, computed per language (mean over its pairs) and
averaged over the 8 languages with equal weight. PRIMARY CLAIM ("the OV direction is causally used for word order"):
b_OV(L14) > 0, larger than all 32 random-direction slopes (empirical p = 1/33 = .03 < .05), and
z = (b_OV - mean b_rand) / sd b_rand > 2.33.

## Secondary (reported, not claimed)
b_OV vs b_IE at L14; number of the 8 languages with b_OV > 0 (sign test); OV-language vs VO-language slopes;
full dose-response over k in {-2,-1,1,2}; layers 8 and 20; pair-bootstrap 95% interval of b_OV(L14).

## What would change our mind
Primary claim holds -> moving a representation along the OV axis shifts the model's word-order preferences: the
axis is used, not just represented. Fails (b_OV inside the random-direction null) -> the axis is descriptive
(a correlate of which languages these are), at least at the tested layer and scale.
