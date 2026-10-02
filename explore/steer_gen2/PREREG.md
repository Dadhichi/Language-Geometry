# Pre-registration: confirmatory free-generation test of the word-order direction (Qwen2.5-7B)

Written and committed 2026-10-02, BEFORE any generation for this study. Frozen code in this commit:
`make_prompts2.py`, `directions2.py`, `gen_ov2.py` (reuses the frozen generation and language-ID code of
`explore/steer_gen/gen_ov.py`), `analysis_gen2.py`, `runG2.sh`. A debug run (2 prompts per language, 1 random
direction) runs first on the VM to test the pipeline; its numbers are not used.

## Why this study
The pre-registered free-generation test (`explore/steer_gen`, prereg b140a1d) could not be evaluated: its inclusion
rule required every one of 53 conditions to be usable and excluded all 8 languages. Exploratory analysis of that
run (`explore/steer_gen/exploratory_gen.py`, `reparse_objtype.py`, FINDINGS.md) found that the word-order direction
reorders NOUN objects in German (72% -> 14% before the verb at k = -2) and Russian (1% -> 36% at k = +2), that the
French effect was an object-type artefact (clitic pronouns), and that the effect appears at k = 2 but not at k = 1.
This study tests those findings on fresh prompts (H1) and on four new languages (H2), with the analysis choices that
the exploratory analysis made after the fact now fixed in advance.

## Materials
* Prompts (`prompts2.jsonl`, 1,350): 150 FLORES+ devtest sentences NOT used in `explore/steer_gen` (the same
  `random.Random(0)` shuffle, continued past its first 150 after removing the used ones), the same sentences in 9
  languages, first 40% of the words.
* Languages. Flexible order (the grammar lets a noun object stand on either side of its lexical verb):
  German, Dutch (verb-final in subordinate and non-finite clauses, verb-second in main clauses); Russian, Ukrainian,
  Polish, Croatian (verb-object basic order with free scrambling). Rigid controls: English and Spanish (noun objects
  after the verb), Korean (verb-final).
* Directions (`directions2.py`, layer 14): for each test language h, beta_OV and beta_IE from OLS over the other 33
  languages of the 34-language dev centroids (same procedure as `explore/steer_ov/directions.py`; reproduces its
  German, English and Russian directions exactly); beta_IE rescaled to ||beta_OV||; 24 random unit directions in the
  span of the 33 centred centroids, scaled to ||beta_OV||, `RandomState(0)` in the language order of `LANGS`.

## Intervention and conditions
Greedy decoding, 40 new tokens, cut at the first newline. Add k * v to the output of block 14 at every position.
Fixed strength k = +-2 (no calibration). 53 conditions: unsteered; word-order direction at -2, +2; Indo-European
control at -2, +2; 24 random directions at -2, +2. 71,550 continuations.

## Measurement
Language identity: langid restricted to the 9 languages; only continuations identified as the prompt language count.
Parse: Stanza UD (default package of each language) of prompt + continuation. A counted pair is a dependency with
relation `obj` (any subtype) whose head has UPOS VERB, with both words in the generated text. It is object-first if
the object precedes the verb. Primary measure: the NOUN-object share, objects with UPOS NOUN or PROPN:
rho_l(c) = sum nom_ov / sum (nom_ov + nom_vo) over the matched continuations of language l under condition c.

## Statistic and inclusion
b_l(v) = (rho_l(+2v) - rho_l(-2v)) / 4, defined only if the unsteered, +2v and -2v conditions each have >= 20
noun-object pairs (inclusion per direction and language). Pooled slope over a set L: mean of the defined b_l(v),
l in L. The null distribution for a set is the pooled slopes of the 24 random directions over the same set (each
with its own per-direction inclusion; random directions with no defined language are dropped).

## Hypotheses (primary; each tested separately)
* H1 (replication on fresh prompts): L = {German, Russian}.
* H2 (generalization to new languages): L = {Dutch, Ukrainian, Polish, Croatian}.
A hypothesis is EVALUABLE if the word-order direction has a defined slope in >= 2 (H1) or >= 3 (H2) languages and
>= 20 of the 24 random directions have a defined pooled slope. CLAIM: evaluable, pooled b(beta_OV) above every
defined random pooled slope, and z = (b(beta_OV) - mean_r) / sd_r > 2.58 (one-sided 0.005; Bonferroni over the two
hypotheses at 0.01). Not evaluable => no claim, reported as such.
Synthetic check (`python analysis_gen2.py --selftest`): with the exploratory effect sizes (and half of them for H2)
both hypotheses are claimed in 20/20 simulations; with the word-order direction drawn like a random one, the claim
rate is 0/100 (H1) and 1/100 (H2).

## Secondary (reported, not claimed)
* Per-language directional predictions: Slavic languages: rho(+2) > rho(0); German and Dutch: rho(-2) < rho(0);
  controls: b_l(beta_OV) inside the central 95% of that language's random slopes.
* Indo-European control slope; pronoun-object and all-object shares; language-match rates per condition.
* Sign-specific language loss: language-match rate when pushed against vs toward the language's own majority order
  (noun-object unsteered share > .5 = object-first), sign test over languages.
* Repetition (1 - distinct word-bigram ratio) per condition group.

## What would change our mind
H1 and H2 hold: the word-order direction controls the order in which the model writes noun objects, in languages
whose grammar allows the choice, at a strength where it also costs fluency. H1 holds, H2 fails: the effect is
specific to German and Russian (or to their representations), not a general property of flexible-order languages.
H1 fails: the exploratory result does not replicate on fresh prompts.
