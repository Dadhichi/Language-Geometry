# Pre-registration: free-generation steering in Llama-3.1-8B

Written and committed 2026-10-03, BEFORE any Llama generation. Frozen code in this commit: `make_dirs_l.py`,
`gen_llama.py`, `analysis_l.py`, `runL.sh`; they reuse the frozen code of `explore/steer_gen` (generation loop,
language ID), `explore/steer_gen2` (prompts, directions procedure, object-type parse) and `explore/steer_within`
(analysis helpers). A debug run (2 prompts per language, 1 random direction) runs first; its numbers are not used.

## Why
All causal results so far are from Qwen2.5-7B: the between-language word-order direction beta_OV reorders noun
objects in free generation (steer_gen2 H1), and the within-language direction d_W reorders them while keeping the
language (steer_within B, C). The alignment of d_W with beta_OV replicates in Llama-3.1-8B (steer_within_llama).
This study asks whether the causal results replicate in Llama too.

## Materials and intervention
* Prompts: `explore/steer_gen2/prompts2.jsonl` (1,350: 150 fresh FLORES+ devtest sentences x 9 languages: German,
  Dutch, Russian, Ukrainian, Polish, Croatian, English, Spanish, Korean).
* Layer: output of block 16 of 32 (relative depth 0.5, as Qwen's 14 of 28).
* Directions (`make_dirs_l.py`; vectors in `steer_dirsL.npz`, info in `steer_dirsL_info.json`): beta_OV and beta_IE by
  the steer_gen2 procedure on the 34 Llama dev centroids at layer 16, each test language held out; 24 random unit
  directions in the span of the 33 centred centroids, scaled to ||beta_OV||; d_W at layer 16 from
  `explore/steer_within_llama` (held out for German, Russian and English), rescaled to ||beta_OV||.
  In units of the spread of the language centroids, ||beta_OV|| is 0.80 in Llama and 0.73 in Qwen, so k = 2 is a
  comparable push in both models.
* Greedy decoding, 40 new tokens, cut at the first newline. Conditions: unsteered; beta_OV and d_W at
  k = -3, -2, -1, +1, +2, +3; 24 random directions at k = -2, +2. 61 conditions, 82,350 continuations.

## Measurement and statistic (identical to steer_gen2 / steer_within)
Noun-object share (UPOS NOUN/PROPN objects before their VERB head, both in the generated text) over continuations
identified as the prompt language (langid restricted to the 9 languages). Slope b_l(v) = (share(+2v) - share(-2v))/4,
defined if the unsteered, +2 and -2 conditions each have >= 20 noun-object pairs. Pooled slope = mean of the defined
b_l over a language set. Null: the 24 random directions of this run, pooled over the same set with their own inclusion.

## Hypotheses (primary)
* L-H1 (beta_OV replication of steer_gen2 H1): set {German, Russian}; evaluable if both have a defined slope and >= 20
  random directions are defined. CLAIM: pooled slope above every defined random pooled slope and z > 2.58.
* L-B (d_W reorders, replication of steer_within B): set {German, Dutch, Russian, Ukrainian, Polish, Croatian};
  evaluable if >= 3 languages are defined and >= 20 random directions. CLAIM as for L-H1.
* L-C (d_W keeps the language, replication of steer_within C), conditional on L-B: at the sign that pushes each of
  the 9 languages against its own majority order (k = 2), the language-match rate under d_W exceeds that under
  beta_OV in at least 8 of 9 languages.
Synthetic check (`python analysis_l.py --selftest`): planted effects are claimed; with no effect none is.

## Secondary (reported, not claimed)
beta_OV over {Dutch, Ukrainian, Polish, Croatian} (the steer_gen2 H2 set); dose-response of both directions at
k = 1, 2, 3; per-language shares and language-match rates; repetition; comparison with the Qwen results.

## What would change our mind
All hold: the causal role of the word-order axis, and the separation of word order from language identity by the
within-language direction, are not specific to Qwen. L-H1 holds and L-B fails: in Llama the between-language
direction reorders but the within-language feature is not sufficient at this strength. L-H1 fails: the free-generation
effect of beta_OV is model-specific (or needs another strength in Llama).
