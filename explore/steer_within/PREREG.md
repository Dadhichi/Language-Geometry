# Pre-registration: a within-language word-order direction (Qwen2.5-7B)

Written and committed 2026-10-03, BEFORE any data for this study. Frozen code in this commit: `pairstates.py`,
`gen_w.py`, `analysis_w.py`, `runW.sh` (they reuse the frozen code of `explore/steer_gen`, `explore/steer_gen2`
and `extract.py`). A debug run (3 pairs and 2 prompts per language) runs first on the VM; its numbers are not used.

## Why this study
The word-order direction beta_OV is estimated BETWEEN languages (difference of language centroids, family, script
and fertility held fixed). Steering along it reorders noun objects in German and Russian (explore/steer_gen2, H1),
but pushed against a language's order it often makes the model leave the language (Polish 27%, Croatian 37%
language match at k = +2; Japanese, Turkish, Chinese in explore/steer_gen). So beta_OV carries part of language
identity. A direction estimated WITHIN languages, from sentences that differ only in word order, cannot carry
language identity. This study asks (A) whether such a direction is aligned with beta_OV, and (B, C) whether it
reorders noun objects while keeping the language.

## The within-language direction
Pairs: the 999 UD minimal pairs of `explore/steer_ov/pairs.jsonl` (8 languages; each pair = an attested sentence
and its twin with the verb and its object subtree swapped, same tokens). States: mean-pooled residual stream over
content tokens with the sink dropped, at every layer (the pooling of the language centroids). For each pair p:
diff_p = h(OV twin) - h(VO twin). Weighted least squares per layer and coordinate,
  diff_p = d_order + s_p d_unnat + e_p,  s_p = +1 if the attested order is VO, -1 if it is OV,
with weights 1/n(language), so every language counts equally. d_unnat absorbs the difference between an attested
sentence and its scrambled twin; d_order is the within-language word-order direction d_W. It is fit on all 8 pair
languages (for A) and with each pair language held out (for steering German, Russian and English).
Synthetic check: with planted order and unnaturalness vectors and the real pair counts, the estimator recovers
both (cos 0.998, 1.000); a plain mean of the differences would not (cos 0.92 with order, 0.46 with unnaturalness).

## A. Alignment (primary)
At layer 14: cos(d_W, beta_OV), with beta_OV fit on all 34 languages (same regression as explore/steer_ov). Null:
cos(d_W, r) for 10,000 random unit vectors r uniform in the span of the 34 centred dev centroids (the space in which
beta_OV lives). CLAIM A: cos(d_W, beta_OV) > 0 and above the 99.9th percentile of the null.
Secondary: the per-layer profile; cos(d_W, beta_IE) (control, expected near 0); the share of d_W inside the span;
the cosine within the span; split-half reliability of d_W; cos(d_unnat, beta_OV).

## B. Reordering (primary)
Free generation as in explore/steer_gen2: the same 1,350 prompts (9 languages), greedy, 40 tokens, steering at block
14, every position. d_W is rescaled to ||beta_OV|| of the prompt language. Conditions: unsteered; d_W at k = -3, -2,
-1, +1, +2, +3; beta_OV at k = -2, +2 (re-run for a paired comparison). Measure and inclusion as in steer_gen2:
noun-object share before the verb in language-matched continuations; a language enters the slope of a direction if
the unsteered, +2 and -2 conditions each have >= 20 noun-object pairs. Statistic: pooled slope
b(d_W) = mean over the included flexible-order languages {German, Dutch, Russian, Ukrainian, Polish, Croatian} of
(share(+2) - share(-2)) / 4. Null: the 24 random directions of explore/steer_gen2 (same prompts, same norm), pooled
over the same set with their own inclusion. CLAIM B: evaluable (>= 3 languages included for d_W, >= 20 random
directions defined), b(d_W) above every defined random pooled slope, and z > 2.58.

## C. Language retention (primary, conditional on B)
For each of the 9 languages, "against" = the sign of k that pushes it away from its own majority order (unsteered
noun-object share > .5 means object-first; all objects if no noun pair). Compare the language-match rate under
d_W and under beta_OV at the against sign, k = 2, in this run. CLAIM C: B is claimed, and d_W keeps the language
better than beta_OV in at least 8 of the 9 languages (two-sided sign test p = 0.039).

## Secondary (reported, not claimed)
Dose-response of the pooled slope at k = 1, 2, 3; per-language slopes of d_W and beta_OV; Polish and Croatian at
k = +2 (match and noun-object share); repetition; destinations of off-language continuations; determinism: the
share of continuations of the unsteered and beta_OV +-2 conditions identical to explore/steer_gen2.

## What would change our mind
A holds: languages are arranged along the same direction that, inside one language, separates object-first from
verb-first sentences -- the typological axis is the model's own word-order feature. A fails: the between-language
axis is a different direction from the within-language order feature. B and C hold: word order can be steered
without changing language, and beta_OV's language switching came from its language-identity component. B fails: the
within-language feature is not causal for generation at this strength (or not in this form).
