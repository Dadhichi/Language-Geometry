# NUMBERS.md — source of every number in `v4/content/*.html`

Paths are relative to `explore/` unless they start with `writeup/` (= `explore/writeup/`) or are repository-root files.
"log L" = line L of the log file. "D." = a key of `writeup/figdata.json`. Derived values give the arithmetic.
Model names (Qwen2.5-7B, Llama-3.1-8B), WALS feature codes (83A, …), citation numbers and layer indices that only name
a pre-registered design element are not listed separately where the design document is cited once.

## 00-front (summary, overview caption)

| Number in text | Meaning | Source |
|---|---|---|
| 2,009 sentences | 997 dev + 1,012 devtest | derived; writeup/template.html line 149; FINDINGS.md line 108 (FLORES dev/devtest) |
| 34 languages | language set | prereg34/PREREG.md "Data"; prereg34/lib34.py `LANGS` |
| 11 object-before-verb languages | size of S_OV | prereg34/lib34.py `OV`; prereg34/PREREG.md H2 |
| all p ≤ 0.0007; threshold 0.0125 | max p over the 4 claim tests × 4 analyses = H1\|OV p Qwen causal 0.0007; Bonferroni α | prereg34/results34.json primary_causal."H1\|OV_p" = 0.00070; results34.log line 2; results34_llama.json; `prereg.alpha` = 0.0125 |
| 25% to 71% of depth | layer of max word-order gain / blocks: Qwen causal L7/28 = 0.25, Qwen whitened L20/28 = 0.714, Llama L16/32 = 0.50 | D.profiles.{qwen,llama}.per_layer.{causal,lda05}[*].ov (= prereg34/results34*.json per_layer H2_gain) |
| 3 of 4 analyses (bundle; OV largest unique share) | A1 and A2 both p < .0125 in 3/4; OV largest A3 unique in 3/4 | typology_lex/results_tl_qwen.log, results_tl_llama.log; FINDINGS.md lines 186–196 |
| 33 other languages; layer 14 | held-out OLS direction | steer_ov/PREREG.md "Directions" |
| 999 sentences; 8 languages | minimal pairs | D.steering.n_pairs = 999; steer_ov/PREREG.md |
| z = 4.99; 32 random directions | primary z | steer_ov/results_steer.json primary.z = 4.989; results_steer.log line 1 |
| 1% to 36% (Russian), 72% to 14% (German) | noun-object share before verb, base → +2 / base → −2 | steer_gen/objtype.json rus_Cyrl.base nom_ov 1 / (1+79) = 0.0125; ov+ 14/(14+25) = 0.359; deu_Latn.base 50/(50+19) = 0.725; ov- 5/(5+32) = 0.135; FINDINGS.md lines 223–224 |
| all 8 languages excluded | primary not evaluable | steer_gen/results_gen.log line 7 (`languages ... : []`); results_gen.json languages_used = [] |
| +15.0 points per unit of strength; z = 7.83; German 66% → 8%; Russian 4% → 58% | confirmatory H1 | see "09-gen, confirmatory results" |
| four new languages; Dutch, Ukrainian, Polish, Croatian | confirmatory H2 | steer_gen2/PREREG.md |
| cosine +0.23 at layer 14, p = 0.0002; z = 4.56; 98% vs 76% | within-language study | see "10-within" |
| within about one token | mean lag 0.79–1.67 tokens | prereg_belief/results_belief.json P2.*.mean_lag; results_belief.log lines 18–25 |

## 01-intro

| Number | Meaning | Source |
|---|---|---|
| 20 splits | Glottolog internal splits | prereg34/lib34.py `GLOTTO` (20 tuples) |
| 11 OV vs 23 others | S_OV and complement | lib34.py `OV`; 34 − 11 = 23 |

## 02-setup

| Number | Meaning | Source |
|---|---|---|
| 28 blocks, width 3,584; 32 blocks, width 4,096 | model sizes | writeup/template.html line 148; prereg34/PREREG_llama.md ("32 blocks") |
| more than 200 languages | FLORES+ coverage | writeup/template.html line 149 |
| 997 / 1,012 sentences | dev / devtest | writeup/template.html line 149; steer_gen2/make_prompts2.py docstring ("1,012 devtest indices") |
| 10 families, 12 scripts | Glottolog top-level families (Indo-European, Afro-Asiatic, Turkic, Uralic, Sino-Tibetan, Japonic, Koreanic, Austronesian, Austroasiatic, Dravidian); scripts Latn Cyrl Deva Arab Hebr Hans Hant Jpan Hang Khmr Taml Telu | derived from prereg34/lib34.py `LANGS` (script suffixes) and the family list in prereg34/PREREG.md "Data". NOTE: template.html line 149 says "13 families" (see inconsistencies) |
| 11 OV languages from 5 families | Indo-European, Turkic, Japonic, Koreanic, Dravidian | lib34.py `OV` |
| three same-language pairs | Hindi–Urdu, Croatian–Serbian, Chinese ×2 | lib34.py `SAME_LANG_PAIRS` |
| 151,665 tokens; 128,000 tokens | real vocabulary rows for M_causal | E_tree/covgamma.py docstring (`VREAL = 151665`); prereg34/PREREG_llama.md ("rows with id < 128000"); template line 157 |
| η = 1/2 | shrinkage of whitened geometry | prereg34/PREREG.md "Statistic" (lda05) |
| layers 8–20; 9–23; relative depth 0.29 to 0.71 | primary windows | prereg34/PREREG.md; PREREG_llama.md ("0.29-0.71") |
| 50 bootstrap resamples; 90% half-widths below 0.001 | sentence bootstrap | prereg34/derive34.py `--B 50`; results34.json primary_*.H1_gain_boot90 / H2_gain_boot90, results34_llama.json (largest half-width: Llama causal H2 [0.28037, 0.28183] → 0.0007); FINDINGS.md line 120 |

## 03-first (pilot; exploratory)

| Number | Meaning | Source |
|---|---|---|
| 12 languages; layers 0–28 step 2 | pilot data | FINDINGS.md line 1; CLAUDE.md line 37 (language list) |
| 82–91% | share of removable cross-language error removed by a constant offset | FINDINGS.md line 11 |
| −0.92 to −0.96 | corr(log scale, log tokens) at every layer | FINDINGS.md line 19 (CLAUDE.md line 62 says −0.93..−0.96; see inconsistencies) |
| 2–4%; about 15% at layer 26 | rotation share of variance | FINDINGS.md line 23 |
| 1,012 devtest sentences; 99.1% | retrieval candidates; P@1 of centroid-shift matching at layer 14 | D_offsets/tables.md, second table, row 14, column "P@1 shift (amb)" = 0.991; computed in D_offsets/dlib.py lines 252–267 on the test split; FINDINGS.md line 12 ("0.99 at L14") |
| 80% | variance of log scale removed by √T rescaling | FINDINGS.md line 20 ("scale variance -80%") |
| 1.5 to 4.5 times | real composition error vs old null | FINDINGS.md line 28 |
| about 1,000 sentences | size of dev half (997) | see 02-setup |
| top-64; 0.64 | mean squared cosine overlap of per-language subspaces, k = 64 | FINDINGS.md line 30; CLAUDE.md line 59 |
| 0.091 / 0.077 / 0.017 (layer 14, q = 64) | consistent synthetic / real / old null | FINDINGS.md line 32; D_offsets/tables.md third table, rows "14 real" (0.077) and "14 synth: ambient identity" (0.091) |
| 0.0016 against 0.0020 | aligned-slice test, real vs flat surrogate | FINDINGS.md lines 34–35 |

## 04-map

| Number | Meaning | Source |
|---|---|---|
| 52% / 75% | R² of the star alone, Qwen causal / whitened | results34.json primary_causal.R2_star = 0.520, primary_lda05.R2_star = 0.749; results34.log lines 2, 4 |
| layers 8–20 / 9–23 | windows | prereg34/PREREG.md, PREREG_llama.md |
| 28% (Qwen), 25% (Llama) | share of positive eigenvalues in the first two MDS coordinates | D.map.qwen.var_share = 0.284; D.map.llama.var_share = 0.254 (writeup/build_data.py `residual_map`) |
| step 5 | orientation rule | writeup/build_data.py `residual_map` (OV mean to the right) |

## 05-tests

| Number | Meaning | Source |
|---|---|---|
| 561 pairs | 34·33/2 | derived; lib34.py `Space` |
| 4 script splits (Latin, Cyrillic, Arabic, Devanagari) | scripts used by > 1 language | lib34.py `SCRIPT_SPLITS` |
| 85% / 92% | R² of base terms, Qwen causal / whitened | results34.json primary_*.R2_base = 0.846 / 0.920 |
| 20 splits; 11 Indo-European + 9 other | Glottolog splits | lib34.py `GLOTTO` |
| B = 10^4; smallest p 0.0001 | permutations | prereg34/PREREG.md; results34.json prereg.n_perm = 10000 |
| 0.0125 = 0.05/4 | Bonferroni | analysis34.py `ALPHA`; PREREG.md |
| α = 0.05 (same-language test) | secondary threshold | prereg34/PREREG.md H3 |
| simulation table (0.05, 0.10, 0.93, 0.28, 0.12, 0.65, 1.00, 0.30, 0.23, 0.95 / 0.07, 0.03, 0.92, 0.08, 0.10, 0.50, 1.00, 0.12, 0.02, 0.93) | rejection rates | prereg34/synth_check34.log lines 1–5; synth_cond34.log lines 1–5 |
| 40 datasets, 300 permutations; 60 datasets, 200 permutations | simulation sizes | prereg34/PREREG.md "Validation before data" |
| all p ≤ 0.0007 | see 00-front | results34.json, results34_llama.json |
| 0.520, 0.846, 0.085, 0.103 | Qwen causal: R2_star, R2_base, "H1\|OV_gain", "OV\|H1_gain" | results34.json primary_causal (0.5205, 0.8462, 0.0847, 0.1028) |
| 0.749, 0.920, 0.306, 0.215 | Qwen whitened | results34.json primary_lda05 (0.7489, 0.9198, 0.3061, 0.2149) |
| 0.615, 0.864, 0.324, 0.312 | Llama causal | results34_llama.json primary_causal (0.6152, 0.8644, 0.3236, 0.3118) |
| 0.727, 0.905, 0.356, 0.257 | Llama whitened | results34_llama.json primary_lda05 (0.7269, 0.9048, 0.3557, 0.2572) |
| p ≤ 0.0078 (same-language) | max H3_p | results34.json primary_causal.H3_p = 0.0078 |
| closest 0.7% (whitened, both models) | max H3_pair_pct lda05 = 0.00714 | results34.json / results34_llama.json primary_lda05.H3_pair_pct |
| closest 1.1% (Llama causal) | max = hin-urd 0.0107 | results34_llama.json primary_causal.H3_pair_pct |
| 7th to 21st percentile (Qwen causal) | 0.073, 0.084, 0.214 | results34.json primary_causal.H3_pair_pct |
| 2,000 permutations | exploratory and √T variant | prereg34/exploratory34.py `--n_perm 2000`; analysis34.py (sqrt variant, 2000) |
| all four p ≤ 0.0045 (√T) | max over H1, H2, H1\|OV, OV\|H1 p in sqrt_ntok_* of both models = Llama lda05 OV\|H1 0.0045 | results34.json, results34_llama.json sqrt_ntok_* |
| word-order gain 0.046 and 0.058 (Qwen); 0.075 and 0.017 (Llama) | √T H2_gain causal / lda05 | results34.log lines 7–8; results34_llama.log lines 7–8 |
| p = 0.08 to 0.16; Llama causal 0.017 | √T H3_p (0.112, 0.083, 0.017, 0.165) | same lines. NOTE: 0.165 rounds to 0.16 (0.1649) |
| 0.114 → 0.070 | Qwen causal OV gain without / with German+Dutch | results34.json primary_causal H2_gain 0.1142, "OV+deu,nld_gain" 0.0700; FINDINGS.md line 123 |
| falls in every analysis | 0.195→0.134; 0.281→0.182; 0.239→0.152 | results34*.json primary_* "OV+deu,nld_gain" |
| 0.083 (p = 0.001), 0.242 (p = 0.0005); 0.101 and 0.213 (p = 0.0005) | geography, all languages: H1_gain/p and OV\|H1_gain/p | prereg34/exploratory34.log lines 1, 13 |
| 10 families + Indo-Iranian branch; 24 runs | leave-one-out configurations (11 drops + full set) × 2 geometries | exploratory34.log lines 1–24 |
| p ≤ 0.005; smallest gain 0.044 (Turkic) | OV\|H1 max p = 0.0040 (drop IE, causal); min gain 0.0439 | exploratory34.log lines 2, 4 |
| p ≤ 0.046; 18 languages | H1 max p (drop IE, causal) | exploratory34.log line 2 |

## 06-depth (descriptive)

| Number | Meaning | Source |
|---|---|---|
| 500 permutations; η = 0.1, 0.5 | per-layer tests, four geometries | analysis34.py (per-layer loop: 500); PREREG.md "Secondary" |
| 0.02 or less at layer 0 | word-order gain L0: 0.017, 0.021, 0.007, 0.002 | D.profiles.*.per_layer.*[0].ov; results34.log line 10 (L 0); results34_llama.log line 10 (L 0) |
| 25% to 71% | see 00-front | |
| below 0.06 at last layer | 0.005, 0.021, 0.052, 0.041 | D.profiles.*.per_layer.*[-1].ov; results34.log "L28"; results34_llama.log "L32" |
| table: 0.46 (3), 0.08, 0.47 (26), 0.02, 0.16 (7, 25%), 0.005 | Qwen causal: max gen over layers 0–7; min gen in window (L13 0.077); max gen over layers 21–28; ov L0; max ov; ov L28 | D.profiles.qwen.per_layer.causal (= results34.json per_layer, metric causal) |
| 0.47 (1), 0.25, 0.49 (27), 0.02, 0.24 (20, 71%), 0.02 | Qwen whitened (window min L16 0.253) | D.profiles.qwen.per_layer.lda05 |
| 0.47 (2), 0.20, 0.48 (26), 0.007, 0.37 (16, 50%), 0.05 | Llama causal (window min L9 0.199; last 0.0519) | D.profiles.llama.per_layer.causal |
| 0.52 (0), 0.26, 0.63 (26), 0.002, 0.30 (16, 50%), 0.04 | Llama whitened (window min L15 0.263) | D.profiles.llama.per_layer.lda05 |
| late rise from layer 17 | Llama gen 0.23 (L16) → 0.37 (L17) causal; 0.30 → 0.42 lda05 | D.profiles.llama.per_layer.* |
| at most 0.002 (Qwen), 0.008 (Llama) | max \|gl_plain − gl_lex\| over layers and geometries | derived from typology_lex/results_tl_qwen.json `profile` (0.0016) and results_tl_llama.json `profile` (0.0079) |

## 07-which

| Number | Meaning | Source |
|---|---|---|
| 11, 12, 15, 10 languages | sizes of S_OV, S_POST, S_GEN, S_NADJ | typology_lex/PREREG.md (A); analysis_tl.py `POST`, `GENN`, `NADJ` |
| A1/A2 table: 0.005 (0.023), 0.101 (0.0001); 0.060 (0.0001), 0.079 (0.0001); 0.092 (0.0001), 0.130 (0.0001); 0.074 (0.0002), 0.084 (0.0001) | conditional gains and p | results_tl_qwen.json / results_tl_llama.json {causal,lda05}."A1_OV\|POST_gain/_p", "A2_POST\|OV_gain/_p"; results_tl_*.log lines 1, 3 |
| 0.080, 0.119, 0.102; 0.057 vs 0.007 | unique gains (A3 beyond others) | results_tl_qwen.json lda05.A3_OV_beyond_others 0.0799; llama causal 0.1193, lda05 0.1025; qwen causal A3_POST_beyond_others 0.0573, A3_OV_beyond_others 0.0066 |
| 10,000 permutations; 0.0125 | test settings | typology_lex/PREREG.md |
| p = 0.023 | A1 Qwen causal | results_tl_qwen.json causal."A1_OV\|POST_p" = 0.0230 |
| 40-item lists | ASJP | typology_lex/PREREG.md (B) |
| 0.66 | Pearson correlation of the two lexical distances over 561 pairs | results_tl_qwen.json lex_corr.text_vs_asjp = 0.655 (analysis_tl.py line 126) |
| at most 0.001; 0.294 → 0.293 | family-tree gain plain vs + LEX | results_tl_*.json *.H1_gain_plainbase vs "B1_H1\|LEXTEXT_gain" (max change 0.0002); llama causal 0.2936 → 0.2935 |
| p ≤ 0.0011 (B1) | max B1 p | results_tl_qwen.json causal."B1_H1\|LEXTEXT_p" = 0.0011 |
| 2% to 56% | share of gain absorbed by LEX+ASJP: (0.0963−0.0947)/0.0963 = 1.7%; (0.2884−0.1711)/0.2884 = 40.7%; (0.2936−0.1432)/0.2936 = 51.2%; (0.3395−0.1503)/0.3395 = 55.7% | derived from results_tl_*.json H1_gain_plainbase and "B2_H1\|LEXTEXT+ASJP_gain" |
| p ≤ 0.0002 (B2) | max B2 p | results_tl_qwen.json causal."B2_H1\|LEXTEXT+ASJP_p" = 0.0002 |

## 08-steer

| Number | Meaning | Source |
|---|---|---|
| 3,584 coordinates | Qwen width | template line 148 |
| norm 11.6 to 12.1 | ‖β_OV‖ at L14 over 8 held-out fits (11.593 Russian … 12.141 Japanese) | steer_ov/steer_dirs_info.json L14_*.norm_ov |
| cosine −0.03 to +0.02 | cos(β_OV, β_IE) at L14 (−0.0296 German … +0.0208 French) | steer_dirs_info.json L14_*.cos_ov_ie |
| +0.50 to +0.53; −0.24 to −0.25 | mean projection of OV / other languages | steer_dirs_info.json L14_*.mean_proj_OV (0.498–0.531), mean_proj_VO (−0.238 to −0.254) |
| +0.48, +0.64, +0.40, −0.20, −0.33, −0.14, +0.05, +0.03 | held-out projections Hindi, Turkish, Japanese, English, French, Russian, German, Chinese | D.steering.heldout_proj; steer_ov/PREREG.md "Directions" |
| 999 pairs; at most 6 words; 5 to 40 words; 150 per language | pair construction | steer_ov/make_pairs.py docstring and `--n 150`; D.steering.n_pairs |
| k ∈ {−2, −1, +1, +2}; layers 8, 14, 20 | conditions | steer_ov/PREREG.md; steer_score.py `KS` |
| 32 random directions; p = 1/33 = 0.03; z > 2.33 | null and claim rule | steer_ov/PREREG.md "Primary test" |
| −3.06, −1.19, +0.98, +3.03 | mean Δ at L14 by k | results_steer.json L14.dose_response; results_steer.log line 5 |
| +1.52; 1.40 to 1.65 | b_OV; 95% pair bootstrap | results_steer.json primary.b_ov = 1.5224; b_ov_boot95 [1.3966, 1.6538] |
| −0.05 ± 0.32; max +1.16 | random mean ± sd, max | results_steer.json primary.rand_mean −0.0528, rand_sd 0.3157; results_steer.log line 1 (max +1.1556) |
| z = 4.99; p = 0.03 | | results_steer.json primary.z 4.989, p_emp 0.0303 |
| all 8 languages positive | sign test | results_steer.json sign_test 8/8 |
| −0.25 | IE slope L14 | results_steer.json ie_L14.b = −0.2543 |
| range of random slopes −0.47 to +1.16 | used implicitly ("inside the range") | results_steer.log line 1 (min −0.4735, max +1.1556) |
| 1.06, 1.43, 1.00 | slopes over all k at L8, L14, L20 | results_steer.json L8/L14/L20.b_ov_all_k |
| 0.88 (Japanese) to 2.96 (Chinese) | per-language slopes | results_steer.json primary.per_language |
| 150, 150, 150, 150, 134, 133, 71, 61 | pairs per language | D.steering.per_language[*].n |
| −14.1 → −3.5 (+2), −15.3 (−2) Chinese; +9.0 → +3.4 (−2), +8.9 (+2) Hindi | per-language margins | steer_ov/exploratory_steer.log lines 14, 10 |
| 33.8 nats; 17.1 (range 1.9 to 95.9) | loss of log p(attested) at k = +2; random directions | exploratory_steer.log lines 2, 6 |
| 96%; 9.3%; 10.0%; 8.8% | baseline preference for attested order; flips | exploratory_steer.log lines 1–4 |

## 09-gen

| Number | Meaning | Source |
|---|---|---|
| 150 sentences, 8 languages, 40%, 1,200 prompts | first study prompts | steer_gen/PREREG.md; make_prompts.py |
| 40 new tokens | generation length | steer_gen/PREREG.md |
| 24 random directions; 55 conditions; 66,000 continuations | conditions | steer_gen/PREREG.md; FINDINGS.md line 204 |
| 20 prompts, 3 random directions, k ∈ {±0.5, ±1, ±2}, 80% | calibration rule | steer_gen/PREREG.md "Strength k*" |
| k* = 2; 88.8% vs 98.8% | calibration result | results_gen.log line 7 (0.8875 vs 0.9875); D.freegen.calibration |
| z > 2.33; 20 pairs; 53 conditions | pre-registered claim and inclusion | steer_gen/PREREG.md; analysis_gen.py `needed` |
| 7 of 8 languages, between 3 and 9 of 48 | random conditions with < 50% match (deu 9, eng 4, fra 5, hin 6, jpn 3, rus 6, tur 6, zho 0) | exploratory_gen.log line 21 |
| 8% (Chinese at +2) | match rate | results_gen.log line 15 (zho 0.00/0.08); exploratory_gen.json match.zho_Hans.ov_p = 0.08 |
| 3 of 24 directions | calibration directions | steer_gen/PREREG.md |
| five languages | 𝓛 for β_OV | exploratory_gen.json pooled.n_langs_ov = 5 |
| 40% (Japanese −2), 55% (Turkish −2) | match rates | results_gen.log line 14 (jpn 0.40, tur 0.55) |
| 9% → 51%; 97%; 2%–18% (Russian) | all-object OV rate base → +2; match at +2; random 5–95% | exploratory_gen.log line 18; results_gen.log lines 11, 15 |
| 82% → 60% (German), 97% → 51% (Hindi) at −2 | all-object OV rate | exploratory_gen.log lines 13, 16 |
| 8% at −2 (Russian) | all-object rate | exploratory_gen.log line 18 (0.082) |
| 5th–95th percentile; 48 random conditions | figure band | writeup/extra_figs.js `drawGen`; D.freegen_x.rates.*.rand |
| +6.8 per unit k; 27 points | pooled b_OV = 0.0681; ×4 | exploratory_gen.json pooled.b_ov; derived 0.068 × 4 = 0.27 |
| +0.1 ± 0.7 (largest +1.1); +0.8 | random mean ± sd, max; IE | exploratory_gen.json pooled.rand_mean 0.0011, rand_sd 0.0067, max(rand) 0.0108, b_ie 0.0077 |
| z = 10.0; 24 of 24 | pooled z; paired comparison | exploratory_gen.json pooled.z 10.02; paired.wins 24/24 |
| +4.3 to +8.6 | prompt bootstrap 95% | exploratory_gen.json b_ov_boot95 [0.0430, 0.0862] |
| z = 5.9, 4.1, 2.2, 1.9, 0.2 | per-language z (Russian, French, Hindi, German, English) | exploratory_gen.json per_language.*.z |
| 94% to 97% | re-parse agreement (7440/7936 = 93.8% … 7695/7940 = 96.9%) | exploratory_gen.json objtype_agree; exploratory_gen.log line 27 |
| fewer than 10 pairs not shown | table rule | writeup/extra_figs.js (`a + b >= 10`) — designer may change; keep in sync |
| 72% → 14%; 96% and 92% (German) | noun objects base → −2; pronouns base → −2 | objtype.json deu_Latn: nom 50/69, 5/37; pron 46/48, 46/50 |
| 1% → 36% (Russian) | see 00-front | objtype.json rus_Cyrl |
| 218 noun objects; 6 left vs 100 | French nouns over base, −2, +2 (100 + 112 + 6) | objtype.json fra_Latn.{base,ov-,ov+}.nom_vo; FINDINGS.md line 225 |
| 92% → 53%; 17 pairs; 19 of 20 | Hindi nouns 24/26 → 9/17; other objects at −2 (oth_vo 19, oth_ov 1) | objtype.json hin_Deva.base, ov- |
| +8.6; +0.3 ± 1.0; largest +2.8; z = 8.2 | noun-only pooled slope (English, German, Russian) | exploratory_gen.json objtype_nom (b_ov 0.0858, z 8.21, max rand 0.0275); exploratory_gen.log line 24; FINDINGS.md line 227 (mean 0.003 ± 0.010) |
| 4.6%, 7.5%, 6.8%, 6.1% | repetition means | exploratory_gen.log line 26 |
| 0% to 25% | range of single random-condition means | exploratory_gen.json repetition.*.rand_range (min 0.0 Russian, max 0.252 Hindi) |
| k = ±1: +15.7 (Hindi), at most +1.5 | slope at k*/2 | exploratory_gen.json half_k (hin 0.1566, rus 0.0152) |
| 64% vs 96%; 7 languages; p = 0.016; 99% (Hindi) | sign-specific language loss | exploratory_gen.json sign_loss (mean_against 0.642, mean_toward 0.963, lower 7/7, p_sign 0.0156; hin 0.987/0.987) |
| 150 prompts per language and condition | design | steer_gen/PREREG.md |
| Confirmatory: 150 sentences, 9 languages, 1,350 prompts | | steer_gen2/PREREG.md "Materials"; prompts2.jsonl (1,350 lines, 0 overlap with steer_gen/prompts.jsonl; checked) |
| k = ±2; 53 conditions; 71,550 continuations; 24 random | | steer_gen2/PREREG.md "Intervention and conditions" (53 × 1,350 = 71,550) |
| 40 new tokens | | steer_gen2/PREREG.md |
| −0.08, −0.13, −0.21, −0.16, −0.35, +0.52 | held-out projections Dutch, Ukrainian, Polish, Croatian, Spanish, Korean | steer_gen2/steer_dirs2_info.json L14_*.heldout_proj (−0.0757, −0.1261, −0.2116, −0.1587, −0.3544, +0.5153) |
| 20 noun-object pairs; /4 | inclusion; slope denominator 2k = 4 | steer_gen2/PREREG.md "Statistic and inclusion" |
| ≥ 2 (H1), ≥ 3 of 4 (H2); 20 of 24 random | evaluability | steer_gen2/PREREG.md "Hypotheses"; analysis_gen2.py `MIN_LANGS`, `MIN_RAND` |
| z > 2.58; one-sided 0.005; 0.01 | claim threshold | steer_gen2/PREREG.md |
| central 95% | control prediction | steer_gen2/PREREG.md "Secondary"; analysis_gen2.py (quantiles .025/.975) |
| +18.1 (German), +7.7 (Russian) | exploratory noun-object slope per language | exploratory_gen.json objtype_nom.per_language (0.1805, 0.0768) |
| 20 of 20; 0 of 100 (H1), 1 of 100 (H2) | synthetic power and false-positive rate | steer_gen2/PREREG.md "Synthetic check"; git commit 5146777 message |
| commit 5146777 | confirmatory pre-registration | `git log` (commit "steer_gen2: pre-registered confirmatory free-generation test (before any data)") |

## 09-gen, confirmatory results (added after the run; sources `steer_gen2/results_gen2.json`, `results_gen2.log`, `bootstrap_gen2.json`; same data in D.freegen2)

| Number in text | Meaning | Source |
|---|---|---|
| 71,550 continuations | planned and produced | steer_gen2/PREREG.md; orchestrator message (run finished) |
| +15.0 points per unit k (H1) | pooled slope German + Russian | results_gen2.json tests.H1.b_ov = 0.1503; results_gen2.log line 1 |
| 21 of 24 random defined; +0.2 ± 1.9; largest +4.2 | H1 null | tests.H1.n_rand_defined 21, rand_mean 0.0018, rand_sd 0.0190, max(tests.H1.rand) 0.0417 |
| z = 7.83; p = 1/22 = 0.045 | H1 test | tests.H1.z 7.825, p_emp 0.0455; claim true |
| 66% (68 pairs) → 8% (38 pairs) at −2; 73% at +2 (German) | noun-object shares | rates.deu_Latn base [0.662, 68], ov- [0.079, 38], ov+ [0.727, 66]; results_gen2.log line 3 |
| 4% (78) → 58% (36) at +2; 3% at −2 (Russian) | noun-object shares | rates.rus_Cyrl base [0.038, 78], ov+ [0.583, 36], ov- [0.029, 69]; results_gen2.log line 9 |
| 60 points between k = −2 and +2 | 4 × 0.1503 | derived |
| z 4.97 (German), 16.7 (Russian) | per-language z | per_language.deu_Latn.z 4.965; rus_Cyrl.z 16.74 |
| −0.8 (H1 set), +0.9 (H2 set) | Indo-European control pooled slope | tests.H1.b_ie −0.0078; tests.H2.b_ie +0.0087 |
| +11.8 to +18.2 (H1); +8.3 to +18.7 (H2, two languages) | descriptive prompt bootstrap 95% (2,000 draws) | bootstrap_gen2.json H1 [0.1181, 0.1824], H2 [0.0825, 0.1866]; bootstrap_gen2.py `--B 2000` |
| 72% → 14%; 1% → 36% (first study) | comparison | see 09-gen (objtype.json) |
| only 2 of 4 languages; minimum 3 | H2 not evaluable | tests.H2.n_langs 2, evaluable false; analysis_gen2.py `MIN_LANGS` |
| 27% (Polish), 37% (Croatian) at +2 | share in prompt language | match.pol_Latn.ov_p 0.273; match.hrv_Latn.ov_p 0.367 |
| 2 and 17 noun-object pairs; minimum 20 | at +2 | rates.pol_Latn.ov+ [0.0, 2]; rates.hrv_Latn.ov+ [0.118, 17]; results_gen2.log lines 4, 7 |
| 14.4% / 16.0% vs 3.8% / 5.3% | repetition under word-order direction vs unsteered (Polish / Croatian) | repetition.pol_Latn.ov 0.144, base 0.038; repetition.hrv_Latn.ov 0.160, base 0.053 |
| 70% (91) → 10% (51) at −2; z 4.94 (Dutch) | | rates.nld_Latn base [0.703, 91], ov- [0.098, 51]; per_language.nld_Latn.z 4.936 |
| 3% (62) → 54% (26) at +2; z 12.3 (Ukrainian) | | rates.ukr_Cyrl base [0.032, 62], ov+ [0.538, 26]; per_language.ukr_Cyrl.z 12.33 |
| +15.0 (H2, Dutch + Ukrainian); largest random +4.0; 24 random defined | descriptive | tests.H2.b_ov 0.1499; max(tests.H2.rand) 0.0400; n_rand_defined 24 |
| English 0%–1%; Spanish 0% | controls, three conditions | rates.eng_Latn (0.0 / 0.009 / 0.0), rates.spa_Latn (0.0 ×3); per_language pred_ok true |
| Korean 100%; 71% at −2; no noun-object pair | | rates.kor_Hang base [1.0, 109], ov+ [1.0, 61], ov- [null, 0]; match.kor_Hang.ov_m 0.713 |
| all four checkable directional predictions hold | | per_language.{deu,nld,rus,ukr}.pred_ok true; pol, hrv null |
| 76% against, 95% toward; 7 of 9; p = 0.18 | sign-specific language loss | sign_loss mean_against 0.759, mean_toward 0.954, lower 7, n_nontied 9, p_sign 0.180; results_gen2.log line 12 |
| 6 to 18 of 48 random conditions with < 20 noun pairs | per language (deu 8, eng 6, hrv 18, kor 9, nld 14, pol 12, rus 6, spa 10, ukr 8) | derived: count of null in results_gen2.json rates.*.rand |
| "some random directions push a language out of itself" | 3 to 15 of 48 random conditions per language with < 50% in prompt language | derived: results_gen2.json match.*.rand (deu 9, eng 6, hrv 15, kor 3, nld 13, pol 8, rus 6, spa 9, ukr 7) |
| 2.76 Colab compute units | cost of the confirmatory run (13-methods) | orchestrator message (see "missing") |

## 10-within (pre-registered study `explore/steer_within`, commit 14fa22e; sources `steer_within/results_w.json` (= D.within), `steer_within/results_w.log`, `steer_within/dW_info.json`, `steer_within/PREREG.md`, FINDINGS.md last block)

| Number in text | Meaning | Source |
|---|---|---|
| Japanese, Turkish, Chinese; Polish, Croatian leave the language | motivation | steer_gen (09-gen); steer_gen2 match (27%, 37%) |
| commit 14fa22e | pre-registration | orchestrator message; FINDINGS.md last block ("prereg 14fa22e") |
| 999 pairs, 8 languages | pair set | dW_info.json n_pairs 999, langs (8) |
| s_p = ±1; weights 1/n | estimator | steer_within/PREREG.md "The within-language direction" |
| cos 0.998 and 1.000; plain mean 0.92 and 0.46 | synthetic check | steer_within/PREREG.md |
| cosine 0.54 between d_W and d_U at layer 14 | cos_order_unnat[14] = 0.540 | dW_info.json |
| 0.85 to 0.94 (layers 2–28); 0.54 (layer 1); layer 0 | split-half reliability: min 0.851 (L2), max 0.940 (L7); L1 0.544; L0 0.133 | dW_info.json split_half_cos. NOTE: FINDINGS.md and the orchestrator say 0.85–0.93; the maximum is 0.9396 |
| layer 0 degenerate | norm_order[0] = 0.0011 | dW_info.json norm_order |
| natural norm 1.24 | norm_order[14] = 1.236 | dW_info.json |
| about 9.4 times | ‖β_OV‖ (11.59–11.84) / 1.236 = 9.38–9.58 | derived: steer_gen2/steer_dirs2_info.json L14_*.norm_ov |
| 10,000 random unit vectors; 33-dimensional span; 99.9th percentile | null for A | PREREG.md "A"; analysis_w.py `N_NULL`, `Q_A` |
| +0.232; +0.206; null sd 0.071; p = 0.0002 | claim A | steer_within/results_w.json A.cos_ov 0.2323, null_q999 0.2055, null_sd 0.0706, p 0.0002; results_w.log line 1 |
| 16% of squared norm; within-span cosine +0.58 | | A.span_share 0.160; A.cos_ov_within_span 0.581 |
| sqrt(0.16) × 0.58 ≈ 0.23 | consistency check | derived (0.400 × 0.581 = 0.232) |
| cos(d_U, β_OV) = +0.003 | | A.cos_unnat_ov 0.0032 |
| below 0.06 at layers 1–5 | 0.040, 0.036, −0.021, 0.016, 0.057 | A_profile[1..5].cos_ov |
| exceeds 99.9th percentile at layers 13–23 | cos_ov > null_q999 exactly at layers 13–23 (also p ≤ 0.001 there) | derived from A_profile |
| maximum +0.30 at layer 19 | 0.304 | A_profile[19].cos_ov |
| +0.04 at layer 27; −0.20 at layer 28 (p = 0.97) | | A_profile[27].cos_ov 0.042; A_profile[28].cos_ov −0.202, p 0.974 |
| within-span maximum +0.72 at layer 23 | | A_profile[23].cos_ov_within_span 0.716. NOTE: FINDINGS.md says "+0.68 at L22"; L22 is 0.675 but L23 is larger |
| −0.18 at layer 14; about 2.5 null sd | cos_ie / null_sd = −0.177 / 0.071 = −2.50 | A.cos_ie; derived |
| −0.15 to −0.24 at layers 6–18 | min −0.236 (L7), max −0.148 (L9) | A_profile[6..18].cos_ie |
| 4 of 16 Indo-European are OV; 7 of 18 others | sample composition | prereg34/lib34.py `OV`, `GLOTTO[0]` |
| cosine of β_OV and β_IE −0.03 to +0.02 | held-out fits at layer 14 | steer_gen2/steer_dirs2_info.json cos_ov_ie (−0.030 … +0.024); steer_ov/steer_dirs_info.json |
| 1,350 prompts, 9 languages; k ∈ {±1, ±2, ±3}; β_OV at ±2 | design B | PREREG.md "B" |
| 20 noun-object pairs; ≥ 3 languages; ≥ 20 random; z > 2.58 | claim rule B | PREREG.md "B"; analysis_w.py constants |
| 100% identical continuations | determinism | steer_within/results_w.json determinism {base+0: 1.0, ov-2: 1.0, ov+2: 1.0}; results_w.log line 6 |
| all 6 languages included; +3.5 points per unit k | pooled slope | B.n_langs 6; B.b_w 0.03486 |
| −0.1 ± 0.8; largest +1.7; 24 defined | random null (steer_gen2 directions, flexible set) | B.rand_mean −0.0009, B.rand_max 0.0165, B.n_rand_defined 24; sd 0.0078 = std(B.rand, ddof 1), derived |
| z = 4.56; p = 1/25 = 0.04 | | B.z 4.561; B.p_emp 0.04 |
| German +10.3, Dutch +8.5, Croatian +1.5, Ukrainian +0.6, Russian +0.1, Polish 0.0 | per-language slopes | B.per_language |
| German 39/48/66/89/89%; Dutch 23/45/70/79/90% at k = −3/−2/0/+2/+3 | noun-object shares under d_W | rates.deu_Latn w-3 0.3947 (38), w- 0.483 (60), base 0.662 (68), w+ 0.894 (47), w+3 0.894 (47); rates.nld_Latn w-3 0.232 (56), w- 0.455 (55), base 0.703 (91), w+ 0.795 (78), w+3 0.897 (68) |
| German 8%/73%, Dutch 10%/78% under β_OV | rerun at ±2 | rates.*.ov-, ov+ (identical to steer_gen2) |
| 38 to 91 pairs | n of the table cells | rates.* (min 38, max 91) |
| German 71%/70%, Dutch 72%/73% at k = −1/+1 | | rates.deu_Latn w-1 0.712, w+1 0.705; nld_Latn w-1 0.720, w+1 0.734 |
| −0.05 / +3.5 / +5.1 points per unit k at k = 1/2/3 | dose | steer_within/results_w.json dose {1.0: −0.00047, 2.0: 0.0349, 3.0: 0.0506} |
| Russian 4% → 58% under β_OV | | rates.rus_Cyrl base 0.038, ov+ 0.583 |
| k = +3: Croatian 5% → 33%, Polish 3% → 18%, Ukrainian 3% → 11% | secondary | rates.hrv_Latn base 0.053, w+3 0.333 (21); pol_Latn 0.026, 0.179 (28); ukr_Cyrl 0.032, 0.108 (37) |
| English and Spanish 0%–2%; Korean 100% at every d_W strength | controls | rates.eng_Latn (max 0.016 at w+1), spa_Latn (0.0), kor_Hang (1.0) |
| 98% vs 76%; 8 of 9; Dutch tie 99%; p = 0.008 | claim C | C.mean_w 0.981, C.mean_ov 0.759, C.higher 8, C.per_language.nld_Latn match_w = match_ov = 0.993, C.p_sign 0.0078 (two-sided, ties left out: 8 of 8) |
| Polish 98% vs 27%; Croatian 95% vs 37%; Korean 100% vs 71% | | C.per_language |
| repetition 5.2% / 9.3% (d_W, k = +2) vs 24.8% / 26.0% (β_OV) vs 3.8% / 5.3% (unsteered), Polish / Croatian | | steer_within/results_w.json repetition.pol_Latn {w 0.052, ov 0.248, base 0.038}, hrv_Latn {w 0.093, ov 0.260, base 0.053} |
| cosine +0.23, z = 4.56, 98% vs 76% | takeaway | see above |
| cosine of steering d_W with β_OV 0.20–0.23 (flexible languages) | German 0.210, Dutch 0.233, Russian 0.199, Ukrainian 0.232, Polish 0.232, Croatian 0.234 | dW_info.json cos_w_ov_* |
| k_eff = 0.40 to 0.47 at k = 2 | 2 × cos | derived (EXPLORATORY argument). NOTE: the orchestrator's message says 0.42–0.47; Russian gives 2 × 0.199 = 0.40 |
| German 71/66/70%, Russian 7/4/5% at k = −1/0/+1 | | rates.deu_Latn, rates.rus_Cyrl (w-1 0.075, base 0.038, w+1 0.054) |
| German 8% at −2, Russian 58% at +2 under β_OV | | rates.*.ov-, ov+ |
| 8 pair languages; German and Russian have pairs | limits | dW_info.json langs |

## 11-tokens (file renamed from 10-tokens)

| Number | Meaning | Source |
|---|---|---|
| 12 languages; 15 layers (0–28 step 2); 11-dimensional span; 24 tokens; 0.9; 5-fold | design | prereg_belief/PREREG.md |
| layers 8–16 (8, 10, 12, 14, 16); 10,000 resamples; 0.025 | P1 statistic | prereg_belief/PREREG.md "Primary tests" |
| 55.6; p = 0.0001 | P1 primary | results_belief.json P1.primary_mean_diff 55.636, primary_p 0.0001; results_belief.log line 17 |
| 26,435 | ambiguous test positions | results_belief.json n_ambiguous_test |
| 147.1 vs 203.6 | layer-14 MSE posterior vs one-hot | results_belief.json P1."14".mse_post / mse_onehot |
| 102.4 → 84.7 | token+position, + posterior (L14) | results_belief.json P1."14".mse_tok / mse_tok_post |
| every layer from 2 to 28 | mse_tok_post < mse_tok for L2…L28 (L0 both ≈ 0) | results_belief.log lines 3–16 (line 2 = L0) |
| 115.4 | true-language one-hot (L14) | results_belief.json P1."14".mse_true |
| 28,800; 8 pairs; 200 sentences; 9 switch points | code-switch design (8 × 2 × 200 × 9) | prereg_belief/PREREG.md; FINDINGS.md line 153 |
| layers 8, 12, 14, 16; α = 0.025 | P2 rule | prereg_belief/PREREG.md; analysis_belief.py `MID_P2`, claim line 152 |
| 0.117 vs 0.250 (layer 14) | step vs sigmoid MSE | results_belief.json P2."14".mse_step 0.1166, mse_sigmoid_llr 0.2496 |
| p = 1.0 | P2 fit test | results_belief.json P2.primary_fit_p = 1.0 (primary_fit_diff −0.140) |
| ρ = +0.13; p = 0.0005 | lag–prefix Spearman | results_belief.json P2.primary_lag_rho 0.131, primary_lag_p 0.0005 |
| 0.44, 0.73, 0.80; 0.14, 0.19, 0.23; 0.41 after 12 tokens | figure values at layer 14 | D.belief."14".observed["0","1","2"], bayes["0","1","2","12"] |
| 1.24 tokens; 0.79 to 1.67 | mean lag L14; range over layers | results_belief.json P2.*.mean_lag |
| H = 8; 0.218 vs 0.117 | leaky observer | prereg_belief/exploratory_belief.json "14" (best_half_life 8, mse_leaky 0.2177, mse_step 0.1166) |

## 12-dead (file renamed from 11-dead)

| Number | Meaning | Source |
|---|---|---|
| 0.55–0.72 vs 0.68–0.80; about 16 planes | commutator; planes | FINDINGS.md lines 23–24 |
| \|r\| = 0.93 to 0.95 | 1-D position vs log token count | FINDINGS.md lines 44–45 |
| R² −0.27 to −0.44 | additive attributes | FINDINGS.md line 47 |
| 0.90 to 1.28 | unembedding alignment ratio | FINDINGS.md line 48 |
| 18%–42%; 80%–86% | power at 12 / 34 languages | FINDINGS.md lines 79, 87 |
| 0.117 vs 0.250; 218 | see 10-tokens, 09-gen | |
| 27%, 37%; 2 of 4 languages, minimum 3 | confirmatory H2 | see "09-gen, confirmatory results" |
| 7 of 7, p = 0.016; 7 of 9, p = 0.18 | sign-specific language loss, first vs confirmatory study | exploratory_gen.json sign_loss; results_gen2.json sign_loss |

## 13-discussion (file renamed from 12-discussion)

| Number | Meaning | Source |
|---|---|---|
| eight languages from four families | English, German, French, Russian, Hindi (Indo-European), Turkish (Turkic), Japanese (Japonic), Chinese (Sino-Tibetan) | steer_ov/PREREG.md language list. NOTE: template line 339 says "five families" |
| 33.8 nats | see 08-steer | exploratory_steer.log line 2 |
| z = 7.83; k = 2; two languages | confirmatory H1 | see "09-gen, confirmatory results" |
| three languages (Persian, Finnish, Estonian) | bundle breakers | typology_lex/PREREG.md (A) |
| four features | 83A, 85A, 86A, 87A | typology_lex/PREREG.md |

## 14-methods (file renamed from 13-methods)

| Number | Meaning | Source |
|---|---|---|
| commits 0d40bdd, d976f88, d392d70, d8f2d66, 5e5e095, b140a1d, a9c7488, 5146777 | pre-registrations | FINDINGS.md lines 108, 137, 151, 169, 185, 201; `git log` for 5146777 and for the d976f88 message ("conditional-test validation (before data download)") |
| z = 4.99; z = 10.0 | outcomes | see 08-steer, 09-gen |
| 151,665; 128,000 | see 02-setup | |
| η = 0.1 | per-layer extra geometry | prereg34/derive34.py (lda01) |
| 50 resamples | bootstrap | derive34.py `--B 50` |
| 2,000 permutations | √T variant; exploratory | analysis34.py; exploratory34.py |
| 34 star indicators; 4 script splits; 561 values | regression design | analysis34.py `bases`; lib34.py |
| 10^−12 tolerance | tie rule | lib34.py `perm_relabel` / `perm_subset` (`obs - 1e-12`) |
| 10% of distance; 80%–86%; 18%–42% | power study | FINDINGS.md lines 79, 87; writeup/template.html line 361 |
| Kazakh (83A, 85A, 86A), Azerbaijani (85A), Maltese (87A) | genus-assigned values | typology_lex/PREREG.md (A) |
| 40-item lists | ASJP | typology_lex/PREREG.md |
| 10,000 permutations | typology tests | typology_lex/PREREG.md |
| layers 8, 14, 20; 32 random directions | steering directions | steer_ov/directions.py; PREREG.md |
| 150 pairs per language | pair cap | steer_ov/make_pairs.py `--n 150` |
| n − 1; /33; 5,000 resamples | statistics | steer_ov/analysis_steer.py (`ddof=1`, p_emp, `range(5000)`) |
| 1,012 indices; 150 + 150; 40%, ≥ 3 words, ≥ 6 characters | prompt selection | steer_gen/make_prompts.py; steer_gen2/make_prompts2.py |
| 3 characters | langid minimum | steer_gen/gen_ov.py `lid_match` |
| 2,000 resamples | prompt bootstrap | steer_gen/exploratory_gen.py `NB = 2000` |
| 48 random conditions | re-parse | steer_gen/reparse_objtype.py (`--n_rand 24`, ±) |
| 2 prompts, 1 random direction | confirmatory debug run | steer_gen2/PREREG.md; runG2.sh |
| 11-dimensional span; 117 components; 24 tokens; 283,900 positions; 0.9; 8 layers | token study | prereg_belief/PREREG.md; results_belief.json n_window = 283900 |
| about 30 minutes; about 10 L4 GPU-hours | compute | writeup/template.html line 365 ONLY (see "missing") |
| z = 7.83; 2 of 4 languages | confirmatory outcome (pre-registration table) | results_gen2.json tests |
| 2,000 draws; 150 sentences | descriptive bootstrap of the confirmatory run | steer_gen2/bootstrap_gen2.py |
| 2.76 compute units | confirmatory run | orchestrator message (see "missing") |

## Missing or weakly sourced

* **Compute time** ("about 30 minutes" per extraction, "about 10 L4 GPU-hours" in total): only in `writeup/template.html` line 365.
  No log or result file records it (CLAUDE.md line 82 records "~6.7 CU spent" at an earlier date). Keep or drop at the orchestrator's choice.
* **Per-language fertility values** (to back "Telugu and Khmer are far from the centre"): the sentence is qualitative and comes from
  `writeup/template.html` line 185; `tokstats.npz` (fert) exists only in the local data directory, not in the repository results.
* **Pairs used in the code-switch study**: the 8 language pairs are not listed in the pre-registration or result files in the repository, so the text does not name them.
* **Destination languages of Polish and Croatian continuations** (09-gen, H2 paragraph): counted from the local file `C:/Users/ASUS/Documents/lang-geom/gen2/final/gen2_out/gen.jsonl` (field `lid`, conditions word-order ±2). Polish at k = +2: English 45, Polish 41, Croatian 27, German 21, Korean 7 of 150; Croatian at k = +2: Croatian 55, English 44, German 29, none 5, Spanish 5. Not in a repository result file; the text gives no counts, only "most".
* **Compute of the confirmatory run** (2.76 Colab compute units, 13-methods): from the orchestrator's message; not recorded in a repository file.
* **Exploratory steering cost range** ("1.9 to 95.9 nats"): `exploratory_steer.log` line 6 gives the range but the script that produced it is not in `explore/steer_ov/`, so it is unclear whether the range is over directions or over pairs. The text states the range without saying which.
* **gen-loss figure**: the caption says the unsteered share is shown "for comparison"; this depends on the designer's figure (data `D.freegen_x.match.*.base` exists).
* **gen-obj table**: the caption rule "fewer than 10 pairs not shown" mirrors the old `extra_figs.js`; the designer must keep it or the caption must change.
* **Cost of the within-language study** (0.69 Colab compute units, 14-methods): from the orchestrator's message and FINDINGS.md last block ("Cost 0.69 CU"); not in a result file.
* **Split-half reliability range**: the orchestrator's message and FINDINGS.md give 0.85–0.93; `dW_info.json` gives a maximum of 0.9396 (layer 7) at layers 2–28, so the text says 0.85 to 0.94.
* **Within-span cosine maximum**: FINDINGS.md says +0.68 at layer 22; `results_w.json` A_profile gives 0.716 at layer 23. The text uses the result file.
* **k_eff range**: the orchestrator's message says 0.42–0.47; with Russian (cosine 0.199) the range over the six flexible languages is 0.40–0.47. The text uses 0.40 to 0.47.

## Round-2 additions to other files

| Number in text | File | Source |
|---|---|---|
| cosine +0.23, p = 0.0002; z = 4.56; 98% vs 76% | 00-front bullet | see "10-within" |
| about 9.4 times; 8 languages | 13-discussion limitation | see "10-within" |
| row 14fa22e: +0.232, p = 0.0002; z = 4.56; 8 of 9, p = 0.008; −0.18 | 14-methods table | see "10-within" |
| 999 pairs; 10,000 null vectors; 10,001; 1,350 prompts; 40 tokens; 9 conditions; 3 pairs and 2 prompts (debug) | 14-methods "Within-language direction" | steer_within/PREREG.md; analysis_w.py; derived 1 + 6 + 2 = 9 conditions |
| 0.69 compute units | 14-methods compute | orchestrator message; FINDINGS.md last block |
| −0.18 at layer 14; 2.5 null sd; −0.15 to −0.24 at layers 6–18 | 12-dead new row | see "10-within" |
