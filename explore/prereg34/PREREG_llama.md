# Pre-registered replication on Llama-3.1-8B (written before extraction)

Replicates PREREG.md (Qwen2.5-7B; results 1e1d796) on meta-llama/Llama-3.1-8B with NO changes to languages,
tree (lib34.GLOTTO), OV set (WALS 83A), script splits, covariates, tests, permutation counts or claim rules.
Model-specific inputs, fixed now:
- Layers: Llama has 32 blocks (33 residual-stream layers). PRIMARY window = the same relative depth as Qwen's
  L8-L20 of 28 blocks (0.29-0.71): L9-L23 (`analysis34.py --primary 9-23`).
- Causal metric: C = diag(g) Cov(gamma) diag(g) from Llama's lm_head rows with id < 128000 (ids >= 128000 are the
  256 special/reserved tokens), Cov with N-1, g = model.norm.weight (covgamma_hf.py; same recipe as Qwen).
- Sink token: BOS (128000), dropped from pooling, as extract.py does for every model with a BOS.
- Token distance and fertility covariates use Llama's own tokenizer (derive34.py).
Replication criterion: for each claim (genealogy; word order), the claim rule holds in at least one primary metric
and neither primary metric shows a significant effect in the opposite direction; H3 at alpha .05. A claim that
holds in Qwen but not in Llama is reported as model-specific. Descriptive: per-layer profiles on relative depth.
