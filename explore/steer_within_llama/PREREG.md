# Pre-registration: replication of the within-language alignment in Llama-3.1-8B

Written and committed 2026-10-03, BEFORE any Llama pair states exist. Frozen code in this commit:
`pairstates_l.py`, `analysis_wl.py`, `runWL.sh`; they import the frozen estimator (`explore/steer_within/
pairstates.wls_order`) and alignment test (`explore/steer_within/analysis_w.py`). A debug run (3 pairs per
language) runs first on the VM; its numbers are not used.

## Question
In Qwen2.5-7B, the within-language word-order direction d_W (the word-order component of the difference between
the OV and VO twins of 999 UD minimal pairs) is aligned with the between-language word-order direction beta_OV in the
middle layers (explore/steer_within, claim A: cos +0.232 at layer 14 of 28, above the 99.9th percentile of random
directions in the language span, p = 0.0002). Does the same hold in Llama-3.1-8B, a model trained independently?

## Method (identical to the Qwen study except the model and the layer)
* d_W: same 999 pairs, same pooling (mean over content tokens, sink dropped), same weighted least squares
  (diff_p = d_order + s_p d_unnat, weights 1/n per language), fit on all 8 pair languages, at every layer 0-32.
* beta_OV, beta_IE: the same regression on the 34-language Llama dev centroids (`l34/derivedL34`), all 34
  languages, at every layer.
* Null: cos(d_W, r) for 10,000 random unit vectors r uniform in the span of the 34 centred Llama dev centroids.
* Primary layer: 16 of 32 blocks (relative depth 0.5, as Qwen's layer 14 of 28).

## Claim
CLAIM (replication): cos(d_W, beta_OV) > 0 at layer 16 and above the 99.9th percentile of the null.

## Secondary (reported, not claimed)
The per-layer profile (expected: near zero in the first and last layers, maximal in the middle) and its comparison
with Qwen's profile on relative depth; cos(d_W, beta_IE) (in Qwen -0.18, an unexpected result; here we report it
without a prediction); the share of d_W inside the span and the cosine within the span; split-half reliability;
cos(d_unnat, beta_OV).

## What would change our mind
Holds: the alignment of the within-language word-order feature with the between-language typological axis is not a
property of one model. Fails: it may be specific to Qwen (or to its training data), and the Qwen result should be
read as a single-model finding.
