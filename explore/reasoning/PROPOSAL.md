# Why do multilingual LLMs reason worse outside English? A mechanistic research proposal

Status: proposal (2026-10-04). Nothing here is a result, except the CPU checks in Section 3, which are exploratory.
No GPU job was run for this proposal. Scripts: `explore/reasoning/cpu_*.py`. Outputs: `offsets34.json`,
`content12.json`, `biasnoise12.json`.

Evidence labels used throughout:
- **[Lit]** established in published work (verified at the linked source).
- **[Ours]** a finding of this project (article v4 / `explore/FINDINGS.md`; pre-registered unless noted).
- **[CPU]** an exploratory check run for this proposal.
- **[Spec]** speculation or hypothesis.

---

## 0. Summary

**Problem.** Base 7-8B models solve far fewer grade-school math problems in some languages than in English. Published
MGSM numbers for Qwen2.5-7B (base, 3-shot CoT): English 81.6%, Telugu 9.2%, Swahili 17.2% [Lit: EMMA-500 Gen 2, Table 26].
Two families of explanation compete:
- the problem is read *into* the model's shared space badly (encoding);
- the problem is read well, but the language's identity displaces the state away from where English-trained circuits
  work, or writing the chain of thought in that language costs accuracy (bias or decoding).

**This project's angle.** We already know the mid-depth geometry of 34 languages in both models [Ours]:
- a sentence state is approximately content + a constant per-language offset $a_\ell$;
- the offsets carry a family tree and a causal head-direction (word-order) axis;
- token-level language identity switches within about one token.

That geometry lets us split the state of a non-English problem, relative to its English parallel, into a **constant
language-level part** $b_\ell\approx a_\ell-a_{\rm en}$ (removable by adding one vector) and a **problem-specific part**
$n_{\ell,p}$ (not removable by any constant). A second-order expansion of the downstream answer margin turns that split
into a **bias-variance decomposition of the reasoning gap** (Section 4).

**Recommended framework.** The bias-variance ("rung") framework, with the **lossy-reading hypothesis F2** as the lead
hypothesis: the gap comes mainly from problem-specific errors in reading the quantities and their roles. Two cheaper
alternatives stay explicitly testable:
- F1, the constant offset misleads English-tuned circuits;
- F3, the scratchpad tax of writing the chain of thought in the language.

Why F2 leads:
1. *Literature, input side.* Translating the input closes most of the gap. Mistral-7B-Instruct low-resource MGSM goes
   from 8.0 to 39.4 with Google translation [Lit: Liu et al. 2025]. Understanding failures dominate the gap of
   reasoning models [Lit: Kang et al. 2026].
2. *Literature, steering side.* Training-free mean-difference steering toward English gains only +0.1 to +7.5 points
   [Lit: INCLINE's CAA row; MRRE; Zhao et al. 2025].
3. *Our CPU checks.* The constant offset is geometrically large: 29-61% of the squared divergence from English at
   Qwen layer 14. But languages also differ strongly in the problem-specific residual (whitened residual relative to
   English's own spread: French 0.49, Hindi 0.88, Turkish 0.85).

The question is which part is *causal*, and that is decidable cheaply.

**First experiment (E1, about 2 Colab units, one model).** A rung decomposition under teacher forcing on MGSM (250
parallel problems, 11 languages). Score the log-probability of the numbers in the gold English solution, given the
problem in language $\ell$. That fixes the output channel to English and removes F3 by design. Then compare five
interventions at mid-depth, all estimated from task-independent FLORES data or from the English parallel:
1. *repair*: add $a_{\rm en}-a_\ell$;
2. *linear repair*: a per-language near-identity map;
3. *induce*: add $a_\ell-a_{\rm en}$ to the **English** problem;
4. *content-only numeral patching*: put English content into the states of the numeral tokens, keeping the language offset;
5. *nulls*: random same-norm span directions and lateral swaps to Spanish.

Reading of the outcomes:
- F1 predicts repair and induce each move at least 25% of the gap.
- F2 predicts both stay near 0 while numeral patching closes the gap.
- Numbers copied from the problem versus numbers computed in a step tell us whether the loss is in reading or in
  computing.

**Novelty, frankly.** The offset swap itself is not new as an intervention:
- LRP2 (2023) used the same formula with the exact inverse at a later layer;
- ShifCon (2025) put the shift and shift-back inside fine-tuning and measured MGSM;
- MRRE (2026) is a training-free near-equivalent with MGSM gains of +3.2 (Qwen2.5-7B-Instruct) and +7.5
  (Llama-3.1-8B-Instruct).

What is new is the following:
- the *decomposition* of the gap into constant, language-linear and problem-specific parts, from task-independent
  vectors;
- the *induce* (sufficiency) test;
- numeral-anchored content patching on math reasoning, with a copy/compute split;
- a 32-language atlas linking the gap to the offset geometry and to tokenizer contrasts between the two models;
- a typological binding hypothesis that uses our causal word-order directions;
- an intervention whose benefit F2 predicts quantitatively (Section 7).

---

## 1. The problem and the evidence

### 1.1 The size of the gap [Lit]

**MGSM** (Shi et al., ICLR 2023, <https://arxiv.org/abs/2210.03057>):
- The first 250 problems of the GSM8K test set, each needing 2-8 steps.
- Translated by paid professional native translators into Bengali, Chinese, French, German, Japanese, Russian, Spanish,
  Swahili, Telugu and Thai.
- Answers are kept as Arabic numerals in all languages.
- Settings: DIRECT, NATIVE-CoT, EN-CoT (English chain of thought for any input) and TRANSLATE-EN (Google-translated
  input).
- PaLM-540B (Table 3):
  - English: 62.4.
  - NATIVE-CoT average 48.1 (Swahili 35.2, Telugu 45.6, Bengali 46.0).
  - EN-CoT average 51.3.
  - TRANSLATE-EN average 55.0.
- Their claim: "no strong correlation" between accuracy and the language's frequency in pretraining at that scale. The
  four underrepresented languages averaged 44.9% against 47.9% for the high-resource six.
- GPT-3 (text-davinci-002) NATIVE-CoT shows the small-model picture: English 53.6, Telugu 0.4, Bengali 6.4.

**Base 7-8B models** (EMMA-500 Gen 2, Ji et al., <https://arxiv.org/abs/2506.00469>, Table 26; lm-eval-harness,
3-shot CoT):

| Model | EN | DE | FR | ES | RU | ZH | JA | TH | BN | SW | TE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 81.6 | 69.6 | 68.0 | 73.6 | 74.8 | 72.8 | 60.0 | 56.8 | 25.6 | 17.2 | 9.2 |
| Llama-3.1-8B | 52.8 | 42.0 | 34.8 | 42.8 | 37.2 | 1.6* | 2.8* | 35.6 | 19.2 | 25.2 | 5.6 |

\* These look like answer-extraction failures, not ability.

Published averages for the same base model differ a lot by harness. Qwen2.5-7B base MGSM:
- 57.8 (8-shot CoT, Qwen2.5 report Table 4, <https://arxiv.org/abs/2412.15115>);
- 41.1 (Babel, Table 4, <https://arxiv.org/abs/2503.00865>);
- 55.6 (EMMA-500 Gen 2).

**We must measure under one fixed protocol ourselves.**

**Measurement artefacts can create apparent gaps.** Peter, Vilar, Domhan, Malkin, Freitag (2025,
<https://arxiv.org/abs/2511.05162>) report translation errors in MGSM and sensitivity to answer extraction. After fixing
both, "the aforementioned language gap mostly disappears" (for the models they study); they release MGSM-Rev2. At 7-8B
the Telugu and Swahili gaps (60-70 points) are far too large to be artefacts, but mid-resource gaps of 5-10 points could
be.

### 1.2 What predicts the gap [Lit]

**Data share.**
- MEGA (Ahuja et al., EMNLP 2023, <https://arxiv.org/abs/2303.12528>): Pearson 0.893 between GPT-3.5-Turbo XNLI
  accuracy and a pretraining-size proxy.
- Language Ranker (<https://arxiv.org/abs/2404.11553>) reports a strong correlation with the language's proportion in
  pretraining.
- Shi et al. found no strong correlation at 540B.

**Tokenizer fertility.**
- Petrov et al. (NeurIPS 2023, <https://arxiv.org/abs/2305.15425>): tokenized lengths differ up to 15x between languages.
- Ahia et al. (EMNLP 2023, <https://arxiv.org/abs/2305.13707>): speakers are "overcharged while obtaining poorer results".
- Rust et al. (ACL 2021, <https://arxiv.org/abs/2012.15613>): a dedicated tokenizer matters as much as data size.
- MEGA, Fig. 5: fertility against performance, Spearman -0.75 to -0.98 across tasks.
- "The Token Tax" (<https://arxiv.org/abs/2509.05486>): 8-18 accuracy points per extra token per word on AfriMMLU.
- Every one of these papers notes that fertility and data share are confounded. We found no paper that regresses MGSM
  accuracy on fertility.

### 1.3 What this project already knows [Ours] and why it matters here

1. **Additive offsets.** At mid-depth, $\bar h^l(x_{\ell,s})\approx\mu_0+a_\ell+\kappa_\ell\psi_s+\varepsilon$.
   - A constant per-language offset removes 82-91% of the removable cross-language error.
   - Subtracting centroids alone retrieves the right translation among 1,012 sentences with P@1 0.99 at Qwen layer 14.
   - The remaining maps are small, non-commuting rotations that compose.
   - $\kappa_\ell$ is a mean-pooling artefact ($\propto f_\ell^{-1/2}$).
2. **Structure inside the offsets.**
   - The family tree is strongest at the first and last layers.
   - A head-direction (OV, postposition) axis is strongest at 25-71% of depth.
   - Both hold in Qwen2.5-7B and Llama-3.1-8B, beyond script, shared tokens and fertility.
3. **The word-order axis is used.**
   - Steering along $\beta_{\rm OV}$ shifts the preferred order of the same words (z = 4.99).
   - It changes the order Qwen writes in German and Russian (z = 7.83).
   - A within-language word-order direction $d_W$ lies along it (cos +0.23 Qwen, +0.32 Llama) and reorders without
     leaving the language.
4. **Language identity is token-local.** At a code switch, the representation's position on the line between the two
   languages jumps within about 1.2 tokens. When evidence is ambiguous it is graded.
5. **Machinery and lessons.**
   - Steering hooks at a block for every position, with greedy generation, language ID and Stanza parsing.
   - Nulls from random directions of matched norm inside the language span.
   - Pre-registration with frozen code.
   - Known traps: per-language PCA truncation, pooling dilution, ratio artefacts, inclusion rules that cannot be met.

Each piece bears on the reasoning gap:
- (1) gives the bias/noise split.
- (2) says where in depth to intervene: mid-depth for processing, the edges for reading and writing.
- (3) gives a structured, causal component of the offset to test (F1b).
- (4) says an offset edit must be applied at every position and every decoding step, and that the model will
  re-derive language identity from the tokens unless it is clamped.

---

## 2. Literature map

### 2.1 Mechanistic accounts [Lit]

| Account | Key evidence (verified) | Where the loss is | Prediction for "move mid-layer states toward English" |
|---|---|---|---|
| Latent English / English-biased concept space | Wendler et al., ACL 2024 (<https://aclanthology.org/2024.acl-long.820/>): logit lens on Llama-2 shows input, concept and output phases; middle layers decode the right concept "but giving higher probability to its version in English". Schut, Gal, Farquhar 2025 (<https://arxiv.org/abs/2502.15603>): English steering vectors outperform target-language vectors. Zhao et al., NeurIPS 2024 (MWork, <https://arxiv.org/abs/2402.18815>) | middle bias plus decoding | helps moderately |
| Pivot set by the training mix | Wu et al., ICLR 2025, "Semantic Hub" (<https://arxiv.org/abs/2411.04986>): hub interpretable in the *dominant* pretraining language (Chinese for Baichuan-2). Zhong et al. 2024 (<https://arxiv.org/abs/2408.10811>): Japanese-heavy models have dual latent languages | middle | helps; for Qwen, Chinese may serve as well as English |
| Language-agnostic concepts | Dumas et al., ACL 2025 (<https://aclanthology.org/2025.acl-long.1536/>): patching the *mean* of a concept across languages improves translation. Brinkmann et al., NAACL 2025 (<https://aclanthology.org/2025.naacl-long.312/>): shared grammatical features. Lindsey et al. 2025, "Biology" (<https://transformer-circuits.pub/2025/attribution-graphs/biology.html>): language-specific features at the beginning and end, more language-agnostic in the middle, more sharing in the larger model, English output privileged | edges | small gain; a cross-language mean may beat English |
| Language neurons at the edges | Tang et al., ACL 2024 (<https://arxiv.org/abs/2402.16438>): about 1% of neurons, "skewed U" over depth. Kojima et al., NAACL 2024 (<https://aclanthology.org/2024.naacl-long.384/>). Bandarkar et al., ICLR 2025, layer swapping (<https://arxiv.org/abs/2410.01335>): language lives in the bottom and top layers, math in the middle | reading and writing | little gain; fix the edges |
| Translation barrier at output | Bafna et al. 2025 (<https://arxiv.org/abs/2506.22724>): for Llama-3.1-8B, most word-translation failures (TLP > 50% for 78% of pairs) happen at the final translation step. Wang et al., ACL 2025 (<https://aclanthology.org/2025.acl-long.253/>): factual errors at the final language transition; a shortcut from a middle layer helps (LLaMA2 71.47 to 76.08) | decoding | little effect on output accuracy |
| Understanding failure | Kang et al. 2026 (<https://arxiv.org/abs/2510.27269>): for reasoning models, "failures in the understanding stage account for most of the gap"; English input lifts Swahili from 29.3 to 88.0 (Qwen3-4B, PolyMath-Low); reasoning correlates with xx-to-en translation quality at r = 0.951. Bafna et al.: Telugu sources fail at task solving | encoding | helps only if the content was read |
| Middle misalignment | Ravisankar et al., EACL 2026 (<https://aclanthology.org/2026.eacl-long.225/>): on NLU tasks, wrong answers go with low middle-layer alignment to English; patching English activations fixes errors. Lim, Aji, Cohn 2025 (<https://arxiv.org/abs/2505.13141>): mean-difference steering toward English helps small models, not large ones. Lu et al., EMNLP 2025 (<https://aclanthology.org/2025.emnlp-main.762/>): 78.3% of factual-recall errors never reach the English pathway; steering vectors give +19.04 points on average | middle | helps, more for small models |
| Variance, not bias | Piratla et al. 2025 (<https://arxiv.org/abs/2510.15551>): gaps "dominantly due to unbiased error"; ensembles cut gaps by up to 12 points. Marchisio et al., EMNLP 2024 (<https://aclanthology.org/2024.emnlp-main.380/>): language confusion occurs at flat next-token distributions | everywhere | only through sharpening; ensembles do as well |
| Knowledge stored per language | Qi, Fernández, Bisazza, EMNLP 2023 (<https://arxiv.org/abs/2310.10378>): consistency is predicted by subword overlap. Hu et al., NAACL 2025 (<https://aclanthology.org/2025.naacl-long.72/>): knowledge-free reasoning transfers above 90% | middle FFN knowledge | does not help knowledge items |
| Thinking language | Yong et al. 2025 (<https://arxiv.org/abs/2505.05408>): s1-7B MGSM falls from 74.8 to 69.9 when forced to think in-language. Qi et al. 2025 (<https://arxiv.org/abs/2505.22888>): controlling the thinking language costs accuracy. Park et al. 2025 (<https://arxiv.org/abs/2506.05850>): RL makes chains drift to the dominant language as accuracy rises | decoding / scratchpad | (orthogonal) |

**Where the accounts disagree.**
- *Pivot.* Some say English (Schut; MWork; Gurnee et al. 2026, <https://transformer-circuits.pub/2026/workspace/>, where
  swapping English lens coordinates changes a Chinese answer). Others say "concepts biased toward English" (Wendler,
  Brinkmann), language-agnostic means (Dumas), or the dominant training language (Wu; Zhong).
- *Locus.* Decoding (Bafna, Wang: lexical and factual tasks), encoding (Kang: long-CoT math), middle (Ravisankar, Lu,
  Lim), or no single stage (Piratla). The task matters: word translation and facts are decoding-heavy, multi-step math
  is reading-heavy.
- *Scale.* Anthropic finds more sharing with scale. Lim et al. find larger models working in language-specific
  subspaces, where steering stops helping.

### 2.2 Interventions that already improve multilingual reasoning [Lit]

| Family | Method | Typical MGSM effect at 7-8B (verified) |
|---|---|---|
| Prompting | EN-CoT (Shi et al.) | Mistral-7B-Instruct 0-shot: high-resource 23.1 to 37.3, low-resource 8.0 to 13.1 (Liu et al., NAACL 2025, <https://arxiv.org/abs/2403.10258>). Qwen2.5-7B-Instruct 8-shot: 63.4 to 63.9 (Yong et al.) |
| Prompting | Translate-test (Artetxe, Goswami, Bhosale, Fan, Zettlemoyer, EMNLP 2023, <https://arxiv.org/abs/2305.14240>; Google MT in Liu et al.) | Mistral-7B-Instruct: low-resource 8.0 to 39.4 |
| Prompting | Self-translate (Etxaniz et al., NAACL 2024, <https://arxiv.org/abs/2308.01223>) | LLaMA-2-13B base: 13.2 to 19.2. Can hurt: MathOctopus-7B 39.35 to 21.40 (INCLINE, Table 2) |
| Prompting | XLT (Huang et al. 2023, <https://arxiv.org/abs/2305.07004>) | Mistral-7B-Instruct: high / low 43.0 / 15.0 |
| Fine-tuning | QAlign (<https://arxiv.org/abs/2401.07817>), MAPO (<https://arxiv.org/abs/2401.06838>), MindMerger (<https://arxiv.org/abs/2405.17386>), LangBridge (<https://arxiv.org/abs/2401.10695>), xCoT (<https://arxiv.org/abs/2401.07037>), mCoT (<https://arxiv.org/abs/2406.02301>) | QAlign LLaMA-2-7B 38.4 to 49.6 (Swahili 5.2 to 40.4). MindMerger 38.3 to 57.3 |
| Fine-tuning plus shift | **ShifCon** (ACL 2025, <https://arxiv.org/abs/2410.19453>): $h\leftarrow h-v_\ell+v_{\rm en}$ at the start of a middle band, the exact inverse at its end, plus a contrastive loss and multilingual SFT | LLaMA-2-7B high / low: 44.9 / 29.5 to 48.2 / 35.1; English 59.8 to 58.2 |
| Merging | Layer swapping (Bandarkar et al.) | Llama-3.1-8B: Swahili 29.5 to 32.4, Telugu 20.1 to 22.7, Japanese 42.7 to 38.5 (worse) |
| Activation, training-free | **LRP2** (Xu, Li, Xiong, EMNLP 2023, <https://arxiv.org/abs/2311.03788>): exactly $h-v_\ell+v_{\rm en}$, inverse later; factual probing only, oracle layers | mLAMA BLOOM-560m: 18.0 to 21.7 |
| Activation, training-free | **MRRE** (Li et al., ACL 2026, <https://arxiv.org/abs/2511.23231>): an English-minus-target "reasoning vector" at layer 20 (from MGSM response states), then a target-language anchoring vector at layer 23 | non-English average: Qwen2.5-7B-Instruct 67.85 to 71.04; Llama-3.1-8B-Instruct 55.9 to 63.36. Without the anchor, language consistency falls to 26.2% |
| Activation | Mean difference (CAA row of INCLINE, ACL 2025, <https://arxiv.org/abs/2410.12462>); INCLINE learned linear map | MathOctopus-7B: CAA 39.35 to 39.43; INCLINE 42.85; Google MT 46.70 |
| Activation | Language-subspace projection (Zhao et al., NeurIPS 2025, <https://arxiv.org/abs/2505.15257>) | Qwen2.5-7B-Instruct 69.82 to 70.76 (Bengali drops 67.6 to 64.4) |
| Activation | Learned steering vectors (Mahmoud et al., <https://arxiv.org/abs/2505.12584>) | 5-model average 38.1 to 42.2 (Google MT 44.6); hurts English prompts |
| Activation | Opposite direction on demonstrations (Kirtane & Huang, <https://arxiv.org/abs/2602.02326>) | Qwen2.5-7B-Instruct 68.95 to 71.83 |
| Centering, encoders | Libovický et al. 2020 (<https://arxiv.org/abs/2004.05160>); LSAR (Xie et al., EMNLP 2022, <https://arxiv.org/abs/2401.05792>) | retrieval gains |

**Reading of the table.**
- Input-side fixes (translation, alignment fine-tuning with question translation) give tens of points for low-resource
  languages.
- Training-free constant-vector edits give a few points at best. The plain mean difference (CAA) gives about zero.
- MRRE's positive result uses vectors from MGSM reasoning states (in-domain, content-laden) rather than language means,
  and selects layers on a sweep.

This contrast is itself a puzzle that the bias-variance framework can resolve: *is the offset causally weak because the
model was trained to be invariant to it?*

---

## 3. Exploratory CPU checks on existing data [CPU]

All checks use local data only: the 34-language cross-half Gram matrices and offsets of both models, and the 12-language
sentence-level Qwen activations. None is pre-registered. With 34 languages, a Spearman correlation must exceed about
0.46 to be detected with 80% power.

**C1. Distance to English tracks fertility and data share** (`cpu_offsets34.py`, `cpu_ccshare.py`). Unbiased mid-depth
squared distance to English, $d^2_{\rm en}(\ell)$ (relative units; pre-registered window: Qwen L8-20, Llama L9-23):

| Model, geometry | Spearman with log fertility (33 languages) | Spearman with -log Common Crawl share | Partial on CC, given fertility | Partial on fertility, given CC |
|---|---|---|---|---|
| Qwen causal | +0.92 | +0.85 | +0.52 | +0.76 |
| Qwen whitened | +0.80 | +0.80 | +0.47 | +0.45 |
| Llama causal | +0.70 | +0.79 | +0.53 | +0.18 |
| Llama whitened | +0.58 | +0.63 | +0.34 | +0.16 |

Other results:
- *OLS fit.* OLS of $d^2_{\rm en}$ on [log f, Indo-European, OV] has R² 0.86-0.94. Indo-European languages are closer
  to English beyond fertility (coefficient -0.19 to -0.38). The OV group is **not** farther from English beyond
  fertility and family (coefficient -0.11 to +0.12).
- *Closest languages.* Qwen causal: Chinese (0.33), Spanish, Portuguese, French. Farthest: Telugu (3.09), Khmer (3.04),
  Tamil, Marathi. Llama: Spanish, Portuguese, French, German closest; Chinese only 13th; Khmer, Telugu, Tamil farthest.
  This matches Qwen's Chinese-heavy training (cf. the Semantic Hub account).
- *Data proxy.* Common Crawl CC-MAIN-2023-50 primary-language shares
  (<https://commoncrawl.github.io/cc-crawl-statistics/plots/languages>) are a proxy, not either model's mix.

**C2. Not a pooling or length artefact** (`cpu_content12.py`). On the English axis
$e=(\mu_{\rm en}-\bar\mu)/\lVert\cdot\rVert$, a sentence's own token count moves its projection by only 8-20% of what
the between-language fertility difference would predict (layer 14: within-language slope -2.3 against between-language
slope -14.2). The fertility ordering of the offsets is a property of the languages, not of averaging position-dependent
states.

**C3. Tokenizer contrast between the models** (`cpu_crossmodel.py`). The same language has different fertility in the
two tokenizers (Hebrew 39.6 vs 96.2 tokens per sentence; Hindi 118.8 vs 66.7). Across 33 languages, the model
difference in log $d^2_{\rm en}$ tracks the model difference in log fertility: Spearman +0.42 (permutation p = 0.013,
causal), +0.50 (p = 0.0025, whitened), +0.38 (p = 0.03, Euclidean). A worse tokenization of the same language goes with
a mid-depth state farther from English. This is still confounded with each model's data mix, because tokenizers follow
data.

**C4. English is an outlier mid-depth but the centre late** (`cpu_offsets34.py`).
- *Mid-depth.* English's offset is among the *farthest* from the centre of the 34: rank 28/34 (Qwen causal), 19/34
  (Llama causal). It is farther than its low fertility predicts (residual z +1.25 to +2.11 over the six
  model-geometry combinations). Chinese behaves the same way (+1.50 to +2.19).
- *Late band.* English becomes the language *closest* to the centre:
  - Qwen whitened: rank 1 at layers 24-28; Qwen causal: rank 1 at layer 27 (rank 2 at 26).
  - Llama whitened: rank 1 at every layer from 19 to 32; Llama causal: rank 1 at layers 19-27.
  - In Llama's causal geometry it falls back to rank 34 at the last layer.
- *Comparison.* The Llama late band (from layer 19, relative depth 0.59) overlaps the region where Wendler et al. find the
  English-biased "concept" phase (Llama-2-70B layers 41-70 of 80, depth 0.51-0.88).
- *Reading.* The late-middle layers pull every language toward English. Mid-depth (where our word-order axis lives) is
  not English-centred.
- *Consequence.* There are two candidate bands for an intervention: the pre-registered window, and the late "English
  band".

**C5. Content survives reading in every language; how well differs** (`cpu_content12.py`; Qwen; 12 languages; FLORES
devtest; centroids from dev). All at layer 14:

| | fr | de | es | ru | zh | vi | id | ar | ja | tr | hi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| retrieval P@1 to English | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 |
| centred cosine with English parallel | .81 | .77 | .78 | .76 | .76 | .75 | .75 | .71 | .70 | .62 | .57 |
| held-out ridge R² (English content from language) | .50 | .48 | .47 | .46 | .45 | .45 | .46 | .43 | .41 | .38 | .36 |

- R² peaks at layer 20 in every language (Chinese ties at layer 28 without rescaling). It reaches 90% of its peak at
  layers 14-16 in every language.
- **Hindi and Turkish converge almost as early as French, only less completely.** They reach 90% of their peak within
  two layers of French, for both R² and centred cosine. The "depth budget" variant (F2') gets no support here.
- Rescaling for pooling dilution changes R² by about 0.01.
- Sentence-level retrieval is at ceiling, so FLORES cannot show a reading failure at the level of whole sentences. A
  math problem needs the exact quantities and roles, which is what E1 measures.

**C6. Bias/noise split of the divergence from English** (`cpu_biasnoise12.py`; Qwen layer 14).
- *Bias share.* $\lVert\mu_\ell-\mu_{\rm en}\rVert^2$ as a fraction of the mean squared distance to the English
  parallel is:
  - Euclidean: Chinese 0.29, Spanish 0.37, French 0.42, German 0.45, Japanese 0.44, Russian 0.47, Arabic 0.48,
    Turkish 0.50, Hindi 0.61;
  - whitened: 0.33-0.45.
- *Noise.* The problem-specific remainder, relative to English's own sentence spread (whitened): French 0.49,
  Spanish 0.57, German 0.59, Russian 0.61, Indonesian 0.62, Vietnamese 0.63, Chinese 0.69, Arabic 0.71, Japanese 0.75,
  Turkish 0.85, Hindi 0.88.
- *Fertility.* Bias share follows fertility (Spearman +0.82, 11 languages). The noise does not (+0.16).
- *Reading.* Both components are substantial. The "hard" languages (Hindi, Turkish, Japanese) have the most
  problem-specific noise. Geometry alone cannot say which component is *causal*.

**C7. Published gaps against geometry** (`cpu_gap_vs_geometry.py`). With the 7 non-English MGSM languages in our set
(Qwen base, EMMA-500 accuracies), Spearman of the logit gap with $d^2_{\rm en}$ is +0.57 (causal). Without Telugu it is
+0.31. Telugu dominates, and 7 languages cannot detect anything below |ρ| ≈ 0.85. This motivates the 32-language atlas
(E2b) rather than any claim.

**C8. Scale of an offset swap.** At Qwen layer 14, $\lVert a_\ell-a_{\rm en}\rVert$ (unbiased, raw units) ranges from
12.9 (Chinese) to 43.2 (Khmer); Telugu is 42.2. Our word-order steering used $\lVert\beta_{\rm OV}\rVert\approx 12$ at
$k=\pm2$, where some random directions already pushed a language out of itself. A full swap for Telugu is a push of
about $3.5\lVert\beta_{\rm OV}\rVert$, so random same-norm nulls will be destructive. Lateral swaps (real language
offsets of similar norm) are the informative control. In Llama the raw mid-depth norms are 6-8 times smaller (2.1-5.3
at layer 16): calibrate per model, in units of each model's offset spread, as in the steering studies.

---

## 4. Frameworks

### 4.0 Notation and the gap identity

A model has blocks $1..L$ and residual stream $h^l_t\in\mathbb R^d$. A problem $p$ has a rendering $x(p,\ell)$ in
language $\ell$ (same quantities). The model writes a solution in output language $o$. Accuracy is
$A(\ell,o)=\mathbb E_p\,\mathbb 1[\hat y(p;\ell,o)=y^*_p]$. The native-CoT gap in log-odds splits exactly:

$$\Gamma_\ell=\operatorname{logit}A(\text{en},\text{en})-\operatorname{logit}A(\ell,\ell)
=\underbrace{\big[\operatorname{logit}A(\text{en},\text{en})-\operatorname{logit}A(\ell,\text{en})\big]}_{\Gamma^{\rm in}_\ell\ \text{(reading and processing)}}
+\underbrace{\big[\operatorname{logit}A(\ell,\text{en})-\operatorname{logit}A(\ell,\ell)\big]}_{\Gamma^{\rm out}_\ell\ \text{(scratchpad in }\ell)} .$$

The relative state of a non-English rendering (positions aligned, e.g. at the numerals, which MGSM keeps as Arabic
digits) is

$$h^l(p,\ell)-h^l(p,\text{en})=b^l_\ell+n^l_{\ell,p},\qquad b^l_\ell=\mathbb E_p[\cdot]\approx a^l_\ell-a^l_{\rm en},\qquad \operatorname{Cov}(n^l_{\ell,p})=\Sigma^l_\ell. \tag{1}$$

- $b_\ell$ is the **language-level (bias)** part. Its zeroth rung is a constant (the offset). Its first rung adds a
  language-level linear distortion: near-identity small rotations and anisotropy, as in the structure hunt.
- $n$ is the **problem-specific (noise)** part.

Let $m(h)$ be the downstream margin of a reasoning step (the log-odds of the correct next number against the best
alternative), with gradient $g_p$ and Hessian $H_p$ at the English state. To second order, with a probit link:

$$\mathbb E[m_\ell]\approx m_{\rm en}+g_p^\top b_\ell+\tfrac12 b_\ell^\top H_p b_\ell+\tfrac12\operatorname{tr}(H_p\Sigma_\ell),\qquad
\operatorname{Var}[m_\ell]\approx g_p^\top\Sigma_\ell g_p,\qquad
A_\ell\approx\mathbb E_p\,\Phi\!\Big(\tfrac{\mathbb E[m_\ell]}{\sqrt{s^2+g_p^\top\Sigma_\ell g_p}}\Big). \tag{2}$$

Equation (2) is a bias-variance decomposition of $\Gamma^{\rm in}_\ell$:
- F1 says the bias terms dominate;
- F2 says the variance terms dominate;
- F3 says $\Gamma^{\rm in}_\ell$ is small and $\Gamma^{\rm out}_\ell$ dominates.

The frameworks can be mixed; the experiments estimate the shares.

**Rung ladder.** The project's structure hunt ordered cross-language maps as translation, then gain, then rotation, then
general linear. That order becomes an ordered set of *repairs* $T^{(r)}_\ell$ applied at mid-depth:
- $T^{(0)}$: add $a_{\rm en}-a_\ell$;
- $T^{(1)}$: $h\mapsto\mu_{\rm en}+(I+\Delta_\ell)(h-\mu_\ell)$ with $\Delta_\ell$ ridge-shrunk toward 0. The gain rung
  is a pooling artefact and must not be applied to token states.
- $T^{(\infty)}$: content-only patching from the English parallel.

The share of the gap closed by rung $r$, beyond rung $r-1$, is the gap's projection on that order of distortion. All
language-level maps are estimated on FLORES and are therefore independent of the task.

### 4.1 F1: Offset bias (English-tuned circuits read a displaced operating point)

**Assumptions.**
- (A1) Content is read adequately ($\Sigma_\ell$ small).
- (A2) The circuits that bind quantities and compute were shaped mostly on English-context states, so their thresholds
  are set near $\mu_0+a_{\rm en}$.
- (A3) Qwen2.5 and Llama-3.1 use RMSNorm, which rescales but does not centre, so a constant offset reaches every
  sublayer.

**Mechanism.** A gated MLP unit reads $u=\sqrt d\,\gamma\odot h/\lVert h\rVert$, so relative to English

$$\Delta z_k=\underbrace{\tfrac{\sqrt d}{\lVert h\rVert}\,\tilde w_k^\top(a_\ell-a_{\rm en})}_{\text{threshold shift}}
+\underbrace{\Big(\tfrac{\lVert h_{\rm en}\rVert}{\lVert h_\ell\rVert}-1\Big)\tfrac{\sqrt d}{\lVert h_{\rm en}\rVert}\,\tilde w_k^\top c}_{\text{gain change on content}},\qquad \tilde w_k=\gamma\odot w_k .$$

- *Gain route.* This one is weak. At Qwen layer 14 the sentence-mean norm is about 40-43 and offsets are 10-21 [CPU],
  so the gain factor changes by at most about 8% (Hindi). Token states are larger still, so the gain change is smaller
  there.
- *Threshold route.* Its size depends on how much the reasoning read-out loads on the language span $S$ (33 dimensions
  of 3,584). Summed downstream, $\Delta m_\ell=g^\top b_\ell+\tfrac12 b_\ell^\top Hb_\ell$.
  - *First-order regime.* $g$ has a component along $S$, and the loss is linear in $b_\ell$.
  - *Second-order regime.* The model is tuned at English along $S$ ($g^\top b\approx0$, $H\preceq0$), so
    $\Delta m_\ell\approx-\tfrac12\lVert b_\ell\rVert^2_{|H|}$. The gap is a quadratic form of the offset difference in
    a *task metric* $\bar H$. That metric is the reasoning analogue of the causal inner product the project already
    uses for next-token prediction.

**Predictions.**
- **P1.1 Repair.** Adding $a_{\rm en}-a_\ell$ closes a large share of $\Gamma^{\rm in}_\ell$.
- **P1.2 Induce (sufficiency).** Adding $a_\ell-a_{\rm en}$ to *English* problems reproduces the language's gap.
- **P1.3 Dose.** Along $\lambda(a_{\rm en}-a_\ell)$, $\lambda\in\{0,0.5,1,1.5\}$:
  - second order gives a quadratic with its optimum at $\lambda=1$ (75% of the $\lambda=1$ repair already at $0.5$, and
    a symmetric loss at $1.5$);
  - first order gives a monotone curve through $\lambda>1$.
- **P1.4 Lateral swaps** act as accuracy coordinates. Swapping to Spanish helps Telugu by about
  $\Gamma_{\rm te}-\Gamma_{\rm es}$.
- **P1.5 Specificity.** Random same-norm span directions do not repair.
- **P1.6 Locus.** Numbers copied from the problem keep a small gap; computed numbers carry it.
- **P1.7 Offset blindness.** The read-out gradient $g_p$ loads on $S$ well above the chance share
  $\dim S/d\approx0.009$.

**Falsifier.** Repair and induce both below 10% of the gap, or inside the range of the lateral and random controls. A
content-dependent bias $b_{\ell,p}$ would remain possible, but that is F2-like in practice.

### 4.2 F2: Lossy reading (problem-specific errors at the quantities and their roles), the lead hypothesis

**Assumptions.**
- (B1) The model is approximately invariant to its own offsets, because it saw every language with its offset during
  training: $g^\top b_\ell\approx0$ and $b^\top\bar Hb$ small.
- (B2) Reading a problem into the shared space adds problem-specific error $n_{\ell,p}$. It grows with tokenizer
  fertility (more pieces to compose per word and per number) and falls with the language's data share. C1 and C3 show
  both covariates organise the mid-depth geometry.
- (B3) The errors that matter are at the quantities and the relations between them: who has what, more or less, per
  unit.

**Model.** A problem has $K_p$ quantity-role facts. Each is misread with probability $q_\ell=\Phi(-\delta/\sigma_\ell)$,
where $\sigma^2_\ell$ is the per-fact share of $\operatorname{tr}(\bar H\Sigma_\ell)$. Then

$$A_\ell(p)\approx A_{\rm en}(p)\,(1-q_\ell)^{K_p},\qquad \log A_\ell(p)-\log A_{\rm en}(p)\approx -K_p\,q_\ell .$$

The gap scales with the number of quantities, not with the number of operations, and a problem's own reading error
predicts its failure.

**Predictions.**
- **P2.1** Within a language, the divergence of the problem's numeral-token states from its English parallel *after
  offset removal*, $d_{\ell,p}=\lVert n^l_{\ell,p}\rVert_M$, predicts failure (odds ratio per SD above 1.5) beyond
  language and problem fixed effects.
- **P2.2** Repair and induce stay inside the controls.
- **P2.3** The gap is already present at numbers copied from the problem: the model cannot retrieve the right quantity
  in the right role.
- **P2.4** The gap grows with $K_p$ (interaction of quantities with language); the number of operations adds little
  given $K_p$.
- **P2.5** Content-only patching at the numerals repairs; offset edits do not.
- **P2.6** Translate-test helps exactly when the MT system reads better than the model. With round-trip checks,
  $\sigma_{\rm MT}<\sigma_\ell$.
- **P2.7** If the noises of two renderings are independent, averaging their content states divides the variance:
  $\sigma^2_{\rm avg}=(\sigma^2_\ell+\sigma^2_{\rm MT})/4$. That is smaller than both if and only if
  $1/3<\sigma^2_{\rm MT}/\sigma^2_\ell<3$, which is a quantitative prediction for the intervention in Section 7.

**Falsifier.** $d_{\ell,p}$ does not predict failure at adequate power, and numeral patching closes no more than the
language-level rungs.

*F2' (depth budget).* High-fertility languages need more layers to reach the shared space. C5 shows no delay (90% of
peak at layers 14-16 in every language), so F2' is deprioritised.

### 4.3 F3: The scratchpad tax (writing and re-reading the chain of thought in the language)

**Assumptions.**
- (C1) Mid-depth content and offsets are adequate.
- (C2) The late layers map an English-centred latent (C4; Wendler et al.) to tokens of the language, with per-token
  error $\epsilon_\ell$ (the translation barrier).
- (C3) A native chain of thought re-reads its own text, so every step passes the writing channel and the reading channel
  again.

**Model.** For an $n$-step solution with $\tau$ number-bearing tokens per step,
$A(\ell,\ell;n)\approx A(\ell,\text{en};n)(1-\epsilon_\ell)^{\tau n}$. So $\Gamma^{\rm out}_\ell$ grows linearly in $n$.

**Predictions.**
- **P3.1** English chain of thought for a non-English problem closes most of the gap.
- **P3.2** $\Gamma^{\rm out}_\ell$ grows with the number of steps (GSM8K annotations give $n$); $\Gamma^{\rm in}_\ell$
  does not.
- **P3.3** Teacher-forced English solutions show a small gap.
- **P3.4** Native-CoT errors are surface errors: a correct intermediate value in a logit or tuned lens, a wrong number
  written.
- **P3.5** A mid-band offset swap with restoration does nothing; without restoration it acts like English chain of
  thought.

The literature already leans against F3 as the main term for math. Shi et al. report EN-CoT 51.3 against NATIVE-CoT
48.1 for PaLM-540B. Yong et al. report 63.9 against 63.4 for Qwen2.5-7B-Instruct. Kang et al. call generation
"marginal". F3 is therefore measured (E2a), not built on.

### 4.4 F1b: Typological binding [Spec]

The mid-depth offset contains a head-direction axis that the model *uses* [Ours]. Hypothesis: circuits that bind
quantities to roles use order cues tuned to head-initial English ("X minus Y", "3 more than Y"). In head-final languages
the standard of comparison and the operands arrive in another order (Japanese "Y yori 3 ooi", literally "Y-than 3
more"). A binding step on a non-commutative operation (subtraction, division, comparison) swaps its operands with
probability

$$s_\ell=s_0+\gamma\,\langle a_\ell-a_{\rm en},\hat\beta_{\rm OV}\rangle .$$

**Predictions.**
- **P1b.1** Among errors, operand-order errors ($b-a$ for $a-b$, $b/a$ for $a/b$, inverted comparisons) are enriched in
  OV languages against resource-matched VO languages. In Qwen: Japanese and Korean against Chinese.
- **P1b.2** Steering *English* problems along $+d_W$ (toward object-first; the direction that keeps the language)
  raises operand-order errors on non-commutative steps and not on commutative ones.
- **P1b.3** Removing only the $\hat\beta_{\rm OV}$ component of $a_\ell-a_{\rm en}$ in OV problems lowers operand-order
  errors.

C1 found that OV languages are not farther from English beyond fertility and family, so this mechanism, if real,
explains a small part of the gap. It is cheap to test with directions we already have.

### 4.5 Discriminating predictions

| Instrument | F1 offset bias | F2 lossy reading | F3 scratchpad tax |
|---|---|---|---|
| Teacher-forced English solution: gap at computed numbers | large | large | small |
| Same, at numbers copied from the problem | small | large | small |
| Repair $+(a_{\rm en}-a_\ell)$, mid-band | closes much | ~0 | ~0 |
| Linear rung $T^{(1)}$ beyond the offset | some | ~0 | ~0 |
| Induce: English problem $+(a_\ell-a_{\rm en})$ | reproduces the gap | ~0 | ~0 |
| Dose $\lambda\in\{0.5,1,1.5\}$ | optimum at 1, or monotone | flat | flat |
| Lateral swap to Spanish | partial, proportional to $\Gamma_\ell-\Gamma_{\rm es}$ | ~0 | ~0 |
| Content-only numeral patch | little beyond repair | closes most | ~0 (nothing to close) |
| Per-problem $d_{\ell,p}$ predicts failure | weakly | strongly | weakly |
| Gap against $K_p$ / against steps $n$ | neither specifically | grows with $K_p$ | grows with $n$ (native CoT only) |
| English chain of thought (generation) | closes $\Gamma^{\rm out}$ only | closes $\Gamma^{\rm out}$ only | closes most |
| Read-out gradient load on the language span | above chance | at chance | at chance |

---

## 5. Measurement

### 5.1 Benchmarks and languages

| Set | Items | Translation | Overlap with our 34 | Use |
|---|---|---|---|---|
| MGSM (<https://arxiv.org/abs/2210.03057>; HF `juletxara/mgsm`, CC-BY-SA-4.0 on the mirror) | 250 x 11 | professional, Arabic numerals kept | 8: eng, deu, fra, spa, rus, cmn_Hans, jpn, tel (+ ben, swh, tha need FLORES centroids) | primary (E1, E2a); use MGSM-Rev2 corrections where released |
| `fair-forward/gsm-autotranslated` (Pomerenke et al., <https://arxiv.org/abs/2507.08538>) | 250 per language (inferred to be MGSM's items) | Google Translate | 24 more: nld, swe, por, ukr, pol, hrv, srp, hin, urd, mar, pes, arb, heb, mlt, tur, azj, kaz, fin, ekk, kor, ind, vie, khm, tam | 32-language atlas (E2b), with translation type as a covariate and round-trip filtering |
| PolyMath (<https://arxiv.org/abs/2504.18428>) | 125 x 4 levels x 18 | GPT-4o + expert calibration | 13 | held-out evaluation of interventions (Low level) |
| MCLM MT-MATH100 (<https://arxiv.org/abs/2502.17407>) | 100 x 55 | GPT-4o | 29 | harder robustness set |
| BenchMAX math (<https://arxiv.org/abs/2502.07346>) | MGSM + 6 languages | Google Translate + manual check | 12 | cross-check |
| MSVAMP (<https://aclanthology.org/2024.findings-emnlp.411/>) | 1,000 x 10 | Google Translate + native review | 7 | dev set for choosing layers and strength (disjoint from MGSM) |

MGSM plus the auto-translated set covers 32 of our 34 languages (missing: Filipino, Traditional Chinese). That turns
the project's 34-language geometry into a testable predictor of the gap.

### 5.2 Controls

- **Problem difficulty.** Items are parallel, so every comparison is paired by item with item fixed effects. English
  correctness enters as a difficulty covariate.
- **Translation quality.**
  - Human MGSM is primary.
  - For machine-translated sets, translate back to English (NLLB-200 or the model itself). Flag an item if its
    round-trip English solution is wrong while the original English is right.
  - Report "round-trip English accuracy" as the ceiling attributable to translation.
- **Fertility.** Log tokens per problem as a covariate. The cross-model tokenizer contrast (C3) separates tokenizer
  from language within a language.
- **Memorisation.** GSM8K test items may be in pretraining. Run *number-perturbed MGSM*: substitute new quantities
  consistently in all languages (MGSM keeps Arabic digits) and recompute answers from the GSM8K calculator annotations.
  Report the gap on original and perturbed items.
- **Answer parsing.**
  - Extract the number after a fixed marker.
  - Normalise native digits (Bengali, Devanagari, Thai, Arabic-Indic, full-width), thousands and decimal separators.
  - Report strict and lenient parsers, and the parse-failure rate as an outcome per language (cf. Llama ja/zh in Table
    1.1).
- **Output language.** fastText or langid on the chain of thought. Report accuracy conditional on output language.
  EN-CoT is its own condition.
- **Protocol.**
  - Base models: fixed $k$-shot (MGSM's native exemplars for NATIVE-CoT; English exemplars for EN-CoT).
  - Greedy decoding, at most 384 new tokens, stop at the next "Question:".
  - The same template for all languages.
  - Freeze the protocol before any run (Section 1.1 shows harness choices move averages by 15 points).

### 5.3 Power (`cpu_power.py`)

- **Paired accuracy** (exact McNemar, alpha 0.05):
  - 250 problems per language detects a 10-point change (power 0.87-0.99), not a 5-point one (0.33-0.49);
  - pooled over 10 languages (2,500 pairs), 3 points has power 0.85-0.97.

  So **per-language accuracy claims need a 10-point effect; pooled claims can be finer.**
- **Teacher-forced log-probabilities** (paired, continuous): standardised shifts of 0.18 per language and 0.056 pooled
  are detectable. That is why E1 uses them.
- **Per-problem failure prediction** (logistic): odds ratio 1.5 per SD has power 0.87 with 250 problems; 1.3 needs the
  pooled 2,500.
- **Cross-language correlation**: |r| ≥ 0.85 with 8 languages, 0.79 with 10, 0.46 with 34. Cross-language correlations
  are therefore secondary evidence. The inference rests on within-item interventions.
- **Random-direction nulls**: 24 directions give a smallest p of 0.04; 32 give 0.03.

---

## 6. Mechanistic experiments

All experiments use the existing hook machinery: add a vector to the output of a block at every position, or replace
states at chosen positions. Layers are named for Qwen (28 blocks); Llama uses the same relative depth (14 of 28 ↔ 16 of
32). Vectors come from FLORES dev centroids, which are task-independent and already on disk for 34 languages. Bengali,
Swahili and Thai need a small extraction.

### E1. Bias or noise? A rung decomposition of the reading gap (first experiment; Section 8)

**Design.**
- *Prompt.* Fixed English scaffold, zero-shot: `Question: {problem in ℓ}\nAnswer (in English): {gold English GSM8K
  solution, calculator annotations removed}`. Only the problem text changes across languages.
- *Scored tokens.* The numerals of the solution, in three classes:
  - **copy**: the value occurs in the problem;
  - **carry**: it occurred earlier in the solution;
  - **computed**: first occurrence as the result of an operation, including the final answer.

  The primary outcome is the mean log-probability of computed-numeral tokens per problem, $C_\ell(p)$.
- *Interventions* (layer 14 primary, all positions):
  - R0 repair $+(a_{\rm en}-a_\ell)$;
  - R1 linear rung;
  - P content-only numeral patch, $h_{\ell,t}\leftarrow h_{{\rm en},t'}-\mu_{\rm en}+\mu_\ell$ at aligned numeral tokens
    (Qwen splits digits, so alignment is exact when the digit strings match);
  - I induce on English;
  - D dose $\lambda\in\{0.5,1.5\}$;
  - X lateral to Spanish;
  - N 24 random span directions of norm $\lVert a_{\rm en}-a_\ell\rVert$ (100-problem subset).

  Secondary: a depth sweep {4, 8, 20, 24} including the late "English band" (C4); in-domain offsets (cross-fitted on
  half the MGSM items) against FLORES offsets; and the gradient load of $g_p$ on the language span (P1.7) on 100
  problems per language.
- *Hygiene.* While running, record the hooked model's realised mid-depth centroid. This shows whether the model undoes
  the shift, which C4 and the token-local identity make plausible.

**Outcome map.**

| Outcome | Reading |
|---|---|
| R0 ≥ 0.25 of the gap and I ≥ 0.25 | F1 |
| R0, R1, I < 0.1 and P ≥ 0.25 beyond R1 | F2, and the loss sits at the quantities |
| Copy gap ≈ computed gap | reading (F2) |
| Copy gap ≪ computed gap | processing (F1 or reasoning proper) |
| Teacher-forced gap ≈ 0 while the generation gap is large (E2a) | F3 |

### E2a. Accuracy factorial (generation)

**Design.** MGSM, 11 languages, base Qwen. Factors:
- input language {en, ℓ};
- chain-of-thought language {en, ℓ};
- mid-band edit {none, the best rung from E1 with exact restoration}.

**Baselines.** Translate-test with NLLB-200, self-translate, XLT.

**Outputs.**
- The exact decomposition $\Gamma=\Gamma^{\rm in}+\Gamma^{\rm out}$ per language.
- Whether $\Gamma^{\rm out}$ grows with the number of solution steps (P3.2).
- Whether $\Gamma^{\rm in}$ grows with $K_p$ (P2.4).
- Per-problem agreement between E1's teacher-forced margins and generation correctness. This validates the proxy; the
  check is pre-registered.
- The number-perturbed replicate.

### E2b. A 32-language gap atlas

**Design.** MGSM + gsm-autotranslated, both base models. EN-CoT 4-shot with English exemplars (no native exemplars
needed). Greedy accuracy, plus E1's teacher-forced gap for all 32 languages.

**Regression.** Gap on:
- mid-depth $d^2_{\rm en}$;
- bias share;
- log fertility;
- log CC share;
- Indo-European, OV;
- translation type.

**Within-language test.** Using the cross-model contrast, does the language whose tokenization is worse *in that
model* lose more *in that model* (C3 applied to accuracy)? n = 32 gives power for |r| ≥ 0.47.

### E3. Locating failures

- **Numeral-anchored divergence profiles.** For every problem, the depth profile of $d_{\ell,p}$ at the numeral tokens
  (offset removed). Fit a mixed logistic model of correctness on $d_{\ell,p}$ at each layer: the layer where it predicts
  best is where reading fails (P2.1).
- **Patching depth sweep.** P at layers {4, 8, 12, 16, 20, 24} shows where English content first fixes the answer.
- **Lens on native CoT.** A logit or tuned lens on the numbers of native chains of thought (P3.4): is the correct
  intermediate value latent while a wrong one is written?

### E4. Typological binding (F1b)

- **Steering.** GSM8K English test items with a non-commutative step. Steer along $d_W$ ($k\in\{\pm2,\pm3\}$, the
  strengths that reordered German and Dutch without leaving the language), $\beta_{\rm OV}$, and 24 random directions.
- **Error classification.** Classify operand-order errors from the calculator annotations.
- **Observational test.** Compare operand-order errors in Japanese and Korean against Chinese in MGSM generations (from
  E2a).

---

## 7. Interventions that follow from the framework

Every intervention must beat four baselines, under the same protocol:
1. native CoT;
2. EN-CoT prompting;
3. translate-test (NLLB-200; Google MT where allowed);
4. self-translate.

It should also beat or explain MRRE (+3.2 non-English average on Qwen2.5-7B-Instruct) and the language-subspace
projection (+0.9). For that comparison, run Qwen2.5-7B-Instruct as well. Choose layers and strength on a disjoint dev
set (MSVAMP), never on MGSM; prior work selected on or near the test set.

**If F1 holds: FLORES offset transplant with exact restoration.**
- *Edit.* $h\leftarrow h+(a^{l_1}_{\rm en}-a^{l_1}_\ell)$ at $l_1$; subtract the same vector at $l_2$.
- *Band.* Either the pre-registered window, or a band ending before the late English band.
- *Variants.* Clamping the 33 span coordinates at every layer of the band (robust to the model re-deriving language from
  tokens); partial $\lambda$; a non-English pivot (Spanish, French; Yong et al. found French a close second for
  reasoning).
- *Novelty.* Low as an edit (LRP2, ShifCon). The contribution is a clean, task-independent, held-out test.

**If F2 holds (expected): repair the reading, not the offset.**
1. **Dual-read averaging** [Spec].
   - *Procedure.* Self-translate the problem to English (one extra pass), run both renderings, and at one mid layer
     replace the numeral-token states of the original with the offset-corrected average
     $\mu_\ell+\tfrac12[(h_\ell-\mu_\ell)+(h_{e}-\mu_{\rm en})]$. Keep the language's offset, so the chain of thought
     and the answer stay in the language.
   - *Prediction (P2.7).* It beats both native CoT and translate-test exactly when the two readings' noise variances
     are within a factor of 3. E1 and E3 measure both variances beforehand, so this is a quantitative, falsifiable
     prediction of F2.
   - *Prior art.* Dumas et al. found that averaging a concept across languages helps translation; this is new for
     reasoning.
2. **Reference-free routing.** At inference, the divergence between the original and the self-translated reading at the
   numerals flags a probable misreading by one of them. Route those items to translate-test, or vote. Compare with
   Kang et al.'s selective translation, which translates about 20% of inputs.
3. **Content-anchored alignment fine-tuning** (later, about 3-6 units). A LoRA on the early blocks with a loss that
   aligns *offset-removed* numeral-token states with the English parallel and leaves the offsets alone, so that output
   language is preserved by construction. This is related to Liu & Niehues (ACL 2025,
   <https://aclanthology.org/2025.acl-long.778/>) and ShifCon. The difference is the offset-removed, numeral-anchored
   target that F2 calls for.

**If F3 holds: English scratchpad, native answer.** Write the reasoning in English, then translate only the final
answer sentence.

**Side effects to monitor.**
- Output language: the share of the chain of thought and answer in the target language. MRRE without its anchor falls
  to 26.2%.
- Fluency: the bigram repetition rate used in the steering studies; perplexity under the unsteered model.
- English accuracy: the edit is applied only when the input language is not English; report it anyway with a langid
  gate.
- Non-math ability: FLORES perplexity in the language, Belebele.
- Number formatting: native digits.

---

## 8. Plan, pre-registration sketch and costs

### 8.1 First experiment: E1 on Qwen2.5-7B (base)

**Pre-registration sketch** (to be frozen as `explore/reasoning/PREREG_E1.md` with code and a synthetic check before
any extraction).

**Data.**
- MGSM test, 250 items x {en, de, fr, es, ru, zh, ja, th, bn, sw, te}; MGSM-Rev2 corrections where released.
- Gold English solutions: GSM8K test items 1-250.
- Offsets: FLORES dev centroids at layer 14 (existing for 8 languages; bn, sw, th extracted in the same job).

**Primary statistics** (gap measured in nats on computed numerals):
- $G_\ell=\mathbb E_p[C_{\rm en}(p)-C_\ell(p)]$.
- Repair share $\rho_X=\sum_\ell\big(\mathbb E_p C^{X}_\ell-\mathbb E_p C_\ell\big)/\sum_\ell G_\ell$ for
  $X\in\{\text{R0},\text{R1},\text{P}\}$.
- Induce share $\iota=\sum_\ell\big(\mathbb E_p C_{\rm en}-\mathbb E_p C^{\rm I}_{{\rm en}\to\ell}\big)/\sum_\ell G_\ell$.
- Sums run over the languages with $G_\ell\ge0.1$ nats.

**Hypotheses and thresholds.** Bonferroni over H1 and H2, so alpha 0.005 each.
- **H1 (F1).** All of:
  - $\rho_{\rm R0}\ge0.25$;
  - $\rho_{\rm R0}$ above every random-direction repair (24 directions);
  - $z>2.58$;
  - $\iota\ge0.25$.
- **H2 (F2).** $\rho_{\rm P}-\rho_{\rm R1}\ge0.25$, with the item-bootstrap 99.5% interval above 0.
- **H3 (secondary).** Within language, numeral divergence $d_{\ell,p}$ predicts $C_\ell(p)$ (mixed model, language and
  item effects, slope < 0, p < 0.01).
- **Falsifiers.** F1 (constant form) is rejected if $\rho_{\rm R0}<0.10$ and $\iota<0.10$. F2 is rejected if
  $\rho_{\rm P}<0.10$, or if H3's slope is ≥ 0 at the power computed in 5.3. If neither H1 nor H2 holds and
  $G_\ell$ is small, the gap is not in reading: go to F3 (E2a).

**Secondary (no claim).** Dose shape; lateral proportionality; copy/compute split; depth sweep; FLORES against in-domain
offsets; gradient load on the span; realised centroid shift.

**Validation before data.** Synthetic worlds through the frozen scorer:
- a planted constant-offset harm;
- planted problem-specific noise;
- neither.

Check that the statistics separate the three, as for every earlier pre-registration.

**Compute** (L4, bf16).
- *Forward tokens.* About 81 language-conditions x 250 problems x about 275 tokens ≈ 5.6M tokens. The random null is
  24 directions x 10 languages x 100 problems ≈ 6.6M. The depth sweep is about 1.1M. Total about 13M forward tokens.
- *Extra passes.* 100 gradient passes per language, and a FLORES centroid extraction for 3 languages.
- *Time.* About 1.0-1.3 GPU-hours, so **about 2 compute units**. Under the 10-unit approval threshold.
- *Llama-3.1-8B replication (E1-L).* About 2 units, run only after Qwen's result.

### 8.2 Next two steps

1. **E2a + E2b, about 6-8 units for both models** (needs the user's approval if run as one job above 10 units; split
   it).
   - Accuracy factorial with baselines on MGSM.
   - Number-perturbed replicate.
   - 32-language atlas with round-trip filtering.
   - This step connects E1's mechanism to accuracy and turns the 34-language geometry into a predictor of the gap.
2. **The intervention that E1 favours, about 4-6 units.**
   - Under F2: dual-read averaging and reference-free routing.
   - Under F1: transplant with exact restoration.
   - Head-to-head against EN-CoT, translate-test, self-translate and an MRRE re-implementation.
   - On Qwen2.5-7B base and Instruct; settings chosen on MSVAMP; evaluated on MGSM and PolyMath-Low.

E3 (about 2 units) and E4 (about 1 unit) run alongside, whenever E1's outcome makes them informative. Total programme:
about 15-20 units of the 168 remaining.

### 8.3 Decision tree

- **H1 holds.** Transplant intervention. Ask whether the gap is linear or quadratic in the offset (dose), and whether
  the task metric matches the causal or the whitened geometry (E2b regression).
- **H2 holds.** Locate the failing layer (E3), test P2.4 (quantities against steps), then dual-read averaging.
- **Neither holds and the teacher-forced gap is small.** F3: run E2a first, then the English-scratchpad intervention.
- **Gap near 0 even in generation for mid-resource languages.** Report it as consistent with Peter et al., and
  concentrate on Telugu, Swahili and Bengali.

---

## 9. Risks

- **Teacher forcing is a proxy.** Mitigation: E2a pre-registers the per-problem agreement between margins and
  generation correctness.
- **FLORES offsets may not be math offsets.**
  - The structure hunt found offsets invariant to sentence length (cos ≥ 0.97) on one corpus only.
  - Mitigation: E1 compares FLORES offsets with cross-fitted MGSM offsets. MRRE's in-domain vectors might work for a
    domain reason.
- **Large pushes.** Telugu's swap is about $3.5\lVert\beta_{\rm OV}\rVert$ in Qwen (C8), so generic damage could mask
  repair. Mitigations: dose, lateral swaps and restoration; clamping in the span as a gentler variant.
- **The model undoes the edit.** Language identity is re-derived from tokens within about one token [Ours]. Mitigation:
  record the realised centroid shift, and use per-layer clamping in the band.
- **Numeral alignment.** Number words, separators and Llama's 3-digit chunks can break alignment. Patch only exact digit
  matches; report the excluded share.
- **Ceiling and heterogeneity.**
  - Qwen's high-resource gaps are small (logit 0.4-0.7). The signal is in Telugu, Swahili, Bengali and Japanese.
  - Pooling can be driven by one language. Report every language, and leave-one-language-out.
- **Contamination.** GSM8K test may be memorised in English. Run the number-perturbed control.
- **Mis-specification.** The second-order expansion (2) can be wrong for large offsets. The dose curve is the check.
- **Base and instruct differ.** Most intervention papers use instruct models. Run Instruct in step 2 for comparability.
- **Novelty.** The edit is prior art (LRP2, ShifCon, MRRE). Claims must be about the decomposition and its tests (Section
  10).
- **Lessons from this project's earlier studies.**
  - No inclusion rule that every random direction must satisfy.
  - Ratios such as "share closed" can move when the denominator moves. Report numerator and denominator.
  - Synthetic validation of every statistic first.

---

## 10. Novelty: what is known and what would be new

**Known [Lit].**
- Non-English mid-layer states are closer to English in the late-middle layers (Wendler; Schut; Semantic Hub).
- Misalignment with English predicts NLU errors, and patching English activations fixes them (Ravisankar).
- Understanding failures dominate long-CoT reasoning gaps (Kang).
- Translate-test is a strong baseline (Liu; Shi).
- Shifting $-\mu_\ell+\mu_{\rm en}$ and back is used in fine-tuning (ShifCon) and for factual probing (LRP2).
- Training-free mean-difference steering helps reasoning modestly (MRRE) or not at all (CAA in INCLINE).
- Larger models resist English steering (Lim).
- Variance rather than bias dominates some gaps (Piratla, on output distributions).

**New if the experiments run as planned.**
1. A quantitative bias-variance (rung) decomposition of a reasoning gap. The constant offset, a language-level linear
   map and the problem-specific remainder are estimated from task-independent data. The *induce* test asks whether
   the offset is sufficient to cause the gap in English.
2. Localisation at the quantities: numeral-anchored content patching on math reasoning, and the copy/compute split
   separating reading from computing. Ravisankar et al. patched NLU tasks at sentence-level positions.
3. A principled explanation of why constant offsets help little, if F2 holds: offsets are geometrically large (29-61%
   of the divergence, C6) but the read-out is blind to them (P1.7). That would reconcile CAA ≈ 0 with MRRE > 0 through
   in-domain against FLORES offsets.
4. A 32-language atlas linking the gap to the offset geometry, plus the tokenizer contrast between two models within a
   language (C3). This is the first test of whether *within-language* tokenization quality predicts the reasoning gap.
5. A typological binding test: does a causal word-order feature participate in operand binding? (F1b; no prior work
   found.)
6. An intervention whose benefit is *predicted* from measured noise levels (P2.7), rather than tuned.

**Not new.** The edit $h-\mu_\ell+\mu_{\rm en}$ and its restoration; English chain of thought; translate-test;
middle-layer alignment fine-tuning.

---

## 11. What we already have

| Planned step | Existing code | Existing data | New work needed |
|---|---|---|---|
| Offsets at every layer, 34 languages | `explore/prereg34/derive34.py` (offsets, Grams, bootstraps) | `lang-geom/q34/derived34/centroids.npz`, `l34/derivedL34/centroids.npz` (`dev_L*`, `devtest_L*`) | add ben, swh, tha (+ en raw centroid) with `extract.py` |
| Hooked extraction with pooling in hooks, sink dropped | `extract.py` (`locate`, `sink_token_id`, `pool`, token-budget batching, OOM split) | | pool over problem spans and at numeral positions |
| Teacher-forced scoring under a steering hook | `explore/steer_ov/steer_score.py` (block hook at every position, batched log-probs) | | per-token outputs and numeral classes; patch mode |
| Random span directions of matched norm, held-out directions | `explore/steer_ov/directions.py`, `explore/steer_gen2/directions2.py`, `explore/steer_llama/make_dirs_l.py` | `steer_dirs*.npz` | norm = $\lVert a_{\rm en}-a_\ell\rVert$ per language |
| Greedy generation with hooks, checkpointing, language ID | `explore/steer_gen/gen_ov.py`, `explore/steer_gen2/gen_ov2.py`, `explore/steer_llama/gen_llama.py` | | answer parser, native-digit normalisation |
| Pooled slopes, z against random, bootstraps, inclusion per direction | `explore/steer_gen2/analysis_gen2.py`, `bootstrap_gen2.py`, `explore/steer_llama/analysis_l.py` | | repair and induce shares |
| Geometries (causal, whitened), unbiased distances | `explore/prereg34/lib34.py`, `derive34.py`; `explore/E_tree/covgamma.py` | `grams/L*.npz`, `covgamma_f32.npy` | task-metric comparison (E2b) |
| Token-level language-span projection | `explore/prereg_belief/` | `span12.npz`, `tok/tok_out` | language identity along chains of thought (side effect) |
| Word-order directions | `explore/steer_within/analysis_w.py`, `pairstates.py` | `dW.npz`, `dW_llama.npz`, `steer_dirs2.npz`, `steer_dirsL.npz` | E4 |
| Typology and family labels | `explore/prereg34/lib34.py` (GLOTTO, OV), `explore/typology_lex/` | | |
| CPU checks for this proposal | `explore/reasoning/cpu_offsets34.py`, `cpu_content12.py`, `cpu_biasnoise12.py`, `cpu_crossmodel.py`, `cpu_ccshare.py`, `cpu_gap_vs_geometry.py`, `cpu_power.py` | `offsets34.json`, `content12.json`, `biasnoise12.json` | |
| Colab job pattern | `explore/*/run*.sh`, `colab/run.ipynb`, the orchestrator's Colab CLI | | `runE1.sh` |

---

## Appendix A. Reproducing the CPU checks

Python: `C:/Users/ASUS/AppData/Local/Temp/claude/.../scratchpad/venv/Scripts/python` (numpy, scipy). Run from
`explore/reasoning/`, with `PYTHONIOENCODING=utf-8` and `OMP_NUM_THREADS=4`.

| Script | Runtime | Reads | Writes / prints |
|---|---|---|---|
| `cpu_offsets34.py` | about 1 min | `q34/derived34`, `l34/derivedL34` | `offsets34.json`; C1, C4, C8 tables |
| `cpu_crossmodel.py`, `cpu_ccshare.py`, `cpu_gap_vs_geometry.py` | seconds | `offsets34.json` | C1, C3, C7 |
| `cpu_content12.py` | about 25 min | 12-language sentence activations | `content12.json`; C2, C5 |
| `cpu_biasnoise12.py` | about 10 min | 12-language sentence activations | `biasnoise12.json`; C6 |
| `cpu_power.py` | about 5 min | | Section 5.3 |

## Appendix B. Numbers quoted from the literature

Every number in Sections 1-2 and 7 comes from the cited page or PDF, as opened during this proposal's literature
check.

- **HTML-only extractions** (not re-checked in the PDF): Ravisankar's 41.2%/68.9% patching figures and Kirtane &
  Huang's MGSM numbers.
- **Figures only.** Lim et al. report gains only in figures, so we quote no number.
- **Transcription.** The EMMA-500 table was transcribed from the PDF.
- **Interpretation.** The Llama ja/zh values are our reading (extraction failures), not the authors'.
- **Venues** taken from arXiv comments rather than proceedings pages:
  - Semantic Hub (ICLR 2025);
  - When Less Language is More (NeurIPS 2025);
  - Kang et al. (Findings of ACL 2026);
  - XLT (Findings of EMNLP 2023);
  - LangBridge (ACL 2024).
- **Not verified.** The MGSM license on the original repository, and the license of `fair-forward/gsm-autotranslated`.
  The fact that its 250 rows are exactly MGSM's items is inferred from the row count.
