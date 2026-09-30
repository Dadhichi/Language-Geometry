# lang-geom — geometry of cross-language maps inside multilingual LLMs

## Question
Inside one multilingual model, are the maps between languages' sentence representations (a) a constant
shift, (b) shift + per-language rotation (a "gauge": pairwise maps factor as Q_i Q_j^T), (c) linear but
non-orthogonal, (d) not mediated by any single shared hub (pair-specific structure by script/family),
and how does this change with depth? Secondary: is English a preferred pivot, and does that track the
dominant pretraining language (Llama-3.1-8B vs Qwen2.5-7B, whose dominant languages are en+zh)?

Prior art to position against (do NOT claim these as new): composition inconsistency of pairwise
Procrustes maps is known for static word embeddings (Nakashole & Flauger 2017; Kementchedjhieva et al.
CoNLL 2018; Alaux et al. ICLR 2019 hyperalignment). Zhou, Salhan, Arnett, Korhonen (arXiv 2608.27115,
Aug 2026) run an identity/Procrustes/affine/MLP ladder per layer on FLORES-200 but BETWEEN separately
trained monolingual models, with no shift-only rung, no composition test, no pivot test. Wendler et al.
2024 and Ravisankar et al. EACL 2026 establish English-as-latent-pivot by activation patching. CIF (arXiv
2608.26357) fuses target→English projections into one shared operator; read it for the fusion loss.
The math here is orthogonal group synchronization (Singer 2011; Wang & Singer 2013), not gauge theory;
use plain names in any write-up.

## Files
- `extract.py`  — per-layer residual-stream reps (hooks on embedding + every block; mean pool over
  content tokens with position 0 = sink dropped; last token; token counts; norm diagnostics; top-512
  right singular vectors of final_norm ⊙ lm_head). Layout: `{pooling}_{split}_L{layer}.f16.npy`
  → [n_lang, n_sent, d]. Splits are appended incrementally to the same --tag.
- `build_ted.py` — n-way parallel TED sentences → FLORES-style `tedfit`/`tedtest` splits (needs n_fit ≫ k).
- `plot.py`     — depth curves + pivot heatmap from a results dir.
- `fit.py`      — the analysis. `python fit.py --selftest` must pass (17 checks) after ANY edit to it.
- `colab/run.ipynb` — clone/pull repo, install, extract, fit, push results.
- `results/<model>_<pooling>/*.csv` — outputs (summary.csv is the headline table, one row per layer/k/split).

## Conventions
Row vectors. Map i→j: z_j ≈ (z_i − μ_i) R_ij + μ_j, R_ij ∈ O(k) (procrustes) or GL(k) (ridge).
Fit on --fit_split, score on every --test_splits. Joint alignment by generalized Procrustes initialised
at the English gauge; `cocycle` = mean_{i≠j} tr(Q_i^T R_ij Q_j)/k (1 iff consistent). Composition
inconsistency c = ||A_i (R_ij R_jm − R_im)||² / ||A_i||² on held-out data (mean shifts cancel exactly).
Default language set (crossed on script × family, all in Qwen2.5 and Aya-Expanse):
eng_Latn deu_Latn fra_Latn spa_Latn rus_Cyrl hin_Deva arb_Arab zho_Hans jpn_Jpan tur_Latn vie_Latn ind_Latn.

## Design decisions found the hard way (self-test evidence; do not revert)
1. NEVER fit maps in a SHARED PCA subspace. On exactly consistent synthetic data it manufactured c = 0.19
   vs null 0.01, because per-language rotations move energy across the cutoff. Use `--pca per_lang`
   (each language's own top-k basis; consistency-preserving) or full d with n_fit ≫ d.
2. Estimation variance explodes as n_fit → d. Primary analysis: per-language PCA k ∈ {64,128,256} (FLORES
   dev/devtest suffices). TED (build_ted.py) extends to k = 512–1024 and a full-d check.
3. Null = residual-permutation bootstrap under the fitted joint model, dof-inflated (hub + L−1 rotations
   absorb noise), built in the AMBIENT space and re-projected so it carries subspace-estimation noise.
   Calibrated within ~10% at full d; conservative (over-noisy) under truncation. An isotropic-noise null
   was wrong by 3–5×.
4. c does NOT detect nonlinearity: invertible per-language functions of one hub compose exactly. c ≫ null
   means structure not mediated by a single hub (e.g. script/family-shared components). MLP ≫ linear means
   nonlinear-but-hub-mediated. c_ridge ≈ null with c_proc ≫ null means a linear non-orthogonal gauge.
5. Zhou et al. found Procrustes wins retrieval (P@1) while affine/MLP win MSE — report both, and never
   read an MSE gain alone as "curvature".
6. Prepend the same sink token for every model (BOS if present, else newline) and always drop position 0:
   Qwen adds no BOS, and the position-0 residual has a massive norm mid-depth.
7. Use hooks, not output_hidden_states (HF applies the final norm to the last entry). Never load in 8-bit.
8. PER-LANGUAGE truncation also manufactures composition error, and fit.py's consistent null does not model it
   (it assumes every language's top-k subspace holds the same content; real overlap ~0.64 at k=64). Exactly
   consistent worlds (identity, orthogonal, flat O(2k)) run through the same pipeline reproduce the real
   c = "1.5-4.5x null" (explore/FINDINGS.md). So c vs null_c from fit.py is NOT evidence against a single hub.
   Use an ambient null (x = m + a_i + g_i b + permuted e_i, or flat O/GL(kb >> k)) or the aligned-slice test.
9. Mean pooling dilutes content by ~ntok^-1/2: per-language scale s_i ~= -1/2 log ntok (corr -0.93..-0.96).
   Normalise (sqrt(ntok) * mean) before reading any scale/GL effect as geometry.

## Outcome map (what each pattern means)
- shift ≈ procrustes at mid-depth, c inside null, pivots symmetric → constant-shift model; content is at the ends.
- procrustes ≪ shift, c inside null, cocycle ≈ null, sync_gap ≈ 0 → rung (b), per-language gauge.
- ridge ≪ procrustes, c_ridge inside null → linear non-orthogonal gauge (synchronise over GL, not O).
- c ≫ null, pivot_eng ≥ 0 while other pivots < 0 → English is the gauge that makes maps compose; check
  whether zho does the same in Qwen (dominant-language hypothesis) or not (English-specific).
- c ≫ null, no pivot fixes it → no single hub; check fertility correlation first (tokenizer artefact).
- Depth prediction: procrustes residual U-shaped; early residual tracks fertility; late residual concentrates
  in the W_U subspace (energy.csv / wu diagnostic).

## Status
- 2026-09-30 PRE-REGISTERED RESULTS (explore/FINDINGS.md bottom; explore/prereg34, explore/prereg_belief):
  34 FLORES languages, mid-depth centroid geometry carries BOTH a Glottolog genealogy tree and a WALS-83A OV
  word-order axis, each beyond the other and beyond script/token overlap/fertility; same-language two-script pairs
  among the closest pairs. Replicated on Llama-3.1-8B (larger effects, same relative-depth profile: OV peaks
  mid-depth, genealogy peaks early + late). Token level (Qwen): graded belief over languages at ambiguous positions
  (P1 supported) but code-switches are near-step (~1 token) with small prefix-length hysteresis (P2 not supported).
  Colab jobs run via the Colab CLI (memory: colab-cli-setup); ~5 CU spent.
- 2026-09-29 STRUCTURE HUNT (explore/FINDINGS.md; four independent investigations, all synthetic-validated):
  in ambient coordinates x_{i,s} ~= mu_i + e^{s_i} b_s + small rotation + noise. Translation removes 82-91% of
  the removable residual (centroid subtraction alone: P@1 0.99 at L14); s_i is mean-pooling dilution
  (~ -1/2 log ntok); rotations are small (2-4% mid-depth), non-commuting, but compose. The per-language-PCA
  "maps do not compose" result is an artefact (design decision 8). RoPE-like relative-position rotations,
  structured curvature, additive script/family attributes, and late rotation into W_U: refuted.
  The summary.csv c/null, cocycle/null and pivot/null columns from runs so far are therefore not evidence.
- Repo: github.com/Dadhichi/Language-Geometry, working branch `claude/compassionate-newton-vufc0c`.
  colab/run.ipynb clones that branch and pushes results/ back to it (pull --rebase first).
- fit.py: selftest 17/17 on CPU (numpy path, ~1 min). summary.csv has null_pivot_<lang> (mean pivot delta under
  the consistent null; read pivot_X - null_pivot_X, since raw deltas are all < 0 and track per-language
  estimation noise). Synthetic "hub0" scenario checks it. run_real exercised on fake data in the extract.py
  layout (3 and 12 languages, per_lang k and full). Needs >= 3 languages (exits otherwise).
  Torch/GPU path (`--device cuda`) ran on Colab (smoke test). analyze() is batched on the backend
  (pairs scored against all targets at once, triples batched over m, GPA batched SVD, spectral_frac on
  device in float64); PCA is exact via the Gram matrix in float64 on the backend (was randomized SVD on
  CPU, ~64% of runtime). Checked against the pre-batching version: every basis-invariant statistic
  (procrustes/ridge/sync rho+P@1, c, pivots, cocycle, spectral, nulls) agrees to ~1e-7, numpy and torch.
  Local CPU, FLORES-shaped L=12 d=3584 k=256 2 nulls: 256 s → 77 s (torch-cpu); selftest 3 min → 48 s.
  CAVEAT: under --pca per_lang, rho_identity / rho_shift (and their P@1) compare languages in DIFFERENT
  per-language bases whose signs/orientation are arbitrary, so they are meaningless AS COMPUTED (they changed
  when the PCA solver changed). The correct per-language-PCA version maps through ambient space,
  R = B_i^T B_j (explore/D_offsets); it beats fitted Procrustes at L2-L26.
  Full-d on GPU stores all L² maps (R and W) on device: ~15 GB at d=3584, L=12 → use an A100 for k=full.
- build_ted.py: OPUS TED2020 only (ted_multi URL dead + Moses-tokenized text); tested locally: 17,642 12-way
  sentences after the 6-50 word filter -> tedfit 6000 / tedtest 1000.
- extract.py: ran on Colab for Qwen2.5-7B FLORES dev+devtest (mean+last).
  extract.py loads Qwen2.5-7B on Colab but OOMs on a T4 (bf16 weights ~15.2 GB > 14.6 GiB): use L4/A100.
  Throughput: pooling happens inside the hooks, batches are budgeted by padded tokens (--batch_tokens,
  default 16k for a 24 GB L4) with OOM → split-batch fallback. vLLM-style engines don't help: no
  generation, and they don't expose per-layer residuals. Model download/load dominates wall time.
  FLORES+ names Chinese cmn_Hans (verified; the zho_Hans→cmn_Hans alias works).
  2026-09-29: smoke test passed end-to-end on Colab (Qwen2.5-1.5B, 3 langs, 16 sents; extract.py
  pre-batching-rewrite + fit.py --device cuda). FLORES+ needs the HF gate accepted AND, for fine-grained
  tokens, "read access to public gated repos" ticked.
- Storage (f16, mean+last, all layers, 12 langs): Qwen2.5-7B ~5 GB per FLORES split, Llama-3.1-8B ~6.5 GB;
  a 7000-sentence TED fit+test ~35 GB for Qwen. Check Drive quota before full runs.
- plot.py: `python plot.py results/<run>` -> depth_<split>.png (6 panels by layer, one line per k, dashed =
  null) and pivots_<split>.png (excess over null_pivot when present). PNGs are gitignored.
- Not yet written: W_U energy diagnostic in run_real (function
  `wu_energy` exists in an earlier draft; re-add), MLP rung only via --mlp.

## Next steps
Done: smoke test; Qwen FLORES mean+last fit (results/qwen25_7b_{mean,last}); TED extraction (on Drive);
structure hunt (explore/). In progress: TED mean fit k=256,512 (old null -> read maps/rho, not c/null).
1. Fix the null (design decision 8): ambient identity+offset+gain null and/or flat O/GL(kb >> k) surrogate,
   plus the aligned-slice statistic (explore/B_holonomy/aligned.py). Synthetic scenario first; selftest green.
2. Length-normalised pooling (sqrt(ntok) * mean; exact from existing files) as a fit.py option.
3. Ambient-coordinate analysis as the primary rung ladder: identity -> shift -> shift+gain -> +rotation -> GL,
   on TED (n_fit=6000 > d), full d on an A100. Decisive question: is R_ij ~= I after shift + gain?
4. Re-read literature on centroid subtraction / language-neutral axes (Libovicky et al. 2020; Chang et al.
   2022) before claiming novelty; candidate contributions: decoder depth profile, pooling-dilution law,
   truncation-manufactured non-composition.
5. Llama-3.1-8B (gate) for the dominant-language question; pivots only after the null is fixed.

## Working rules
- Read primary sources before calling anything open. A documented dead end is a valid outcome.
- Keep `python fit.py --selftest` green. Add a synthetic scenario before adding a statistic.
- Activations live on Drive, never in git. Only CSVs and code are committed.
- Point fit.py --out at Drive (then copy into results/ to push): a reclaimed runtime loses /content.
