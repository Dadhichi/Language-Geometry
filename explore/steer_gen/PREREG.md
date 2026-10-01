# Pre-registration: does the OV direction change word order in FREE GENERATION? (Qwen2.5-7B)

Written and committed 2026-10-01 BEFORE any generation with the 7B model. Frozen code: make_prompts.py, gen_ov.py,
analysis_gen.py (this commit). Follows explore/steer_ov (pre-registered: the OV direction shifts log-probability
preferences between fixed-token OV/VO variants, z = 4.99). This tests the stronger claim: word order in the model's
own continuations. A debug run on 3 prompts per language with 1 random direction may be executed first to test the
Stanza pipeline on the VM; its numbers are not used.

## Materials and intervention
1,200 prompts (prompts.jsonl): the first ~40% of 150 FLORES+ devtest sentences, the same sentences in 8 languages
(eng deu fra rus hin tur jpn zho). Greedy decoding, 40 new tokens, cut at the first newline. Steering: add k * v at
the output of block 14, every position, v = the prompt language's held-out direction (explore/steer_ov
steer_dirs.npz: beta_OV, beta_IE rescaled to ||beta_OV||, and random span directions of the same norm; r = 0..23).

## Strength k* (fixed rule, decided by the script, random directions only)
Calibration on the first 20 prompts per language with random directions r = 0, 1, 2 at k in {+-2, +-1, +-0.5}:
k* = the largest k whose language-match rate (langid of the continuation == prompt language) is >= 80% of the
unsteered rate; if none qualifies, k* = 0.5. Conditions: base; OV at +-k*, +-k*/2; IE at +-k*; 24 random at +-k*.

## Measurement
langid on the continuation (restricted to the 8 languages); Stanza UD parse of prompt + continuation; count
dependencies with a VERB head and deprel obj* whose verb and object both lie in the generated text: OV if the object
precedes the verb. OV rate per (condition, language) = sum n_OV / sum (n_OV + n_VO) over language-matched
continuations. Languages with < 20 counted dependencies in any needed condition are excluded from all averages.

## Primary test
Slope per language b = (Delta(+k*) - Delta(-k*)) / (2 k*), Delta = OV rate minus the unsteered OV rate; averaged
over languages with equal weight. CLAIM ("the OV direction changes word order in free generation"): b_OV > 0, above
all 24 random-direction slopes (p_emp = 1/25 = .04), and z > 2.33.

## Secondary (reported, not claimed)
IE control slope; OV at k*/2; OV slope without the language filter; per-language rates and language-match rates;
sign count over languages; prompt-bootstrap 95% CI of b_OV.

## What would change our mind
Claim holds -> the axis controls word order in production, not only preferences between given strings. Fails ->
the effect is confined to scoring fixed variants (or needs strengths that break fluency), which bounds how the axis
is used.
