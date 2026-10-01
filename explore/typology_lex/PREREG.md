# Pre-registration: (A) which typological parameter is the axis? (B) is genealogy just shared vocabulary?

Written and committed 2026-10-01 before computing these statistics. Data: the already-extracted 34-language Grams of
explore/prereg34 for Qwen2.5-7B (primary window L8-L20) and Llama-3.1-8B (L9-L23); same NNLS split model, metrics
(causal, lda05), base (star + script splits + token distance + |dlog fertility|) and permutation machinery (lib34).
Code: analysis_tl.py (this commit). 10^4 permutations per test.

## (A) Separate the typology bundle
OV languages here are also postpositional and genitive-first. Splits (WALS CLDF values; uncoded cases by genus, the
rule already used for kaz in prereg34):
- OV (83A = OV; kaz by genus): hin urd mar pes tur azj kaz jpn kor tam tel
- POST (85A = Postpositions; azj, kaz by genus): hin urd mar tur azj kaz fin ekk jpn kor tam tel
- GENN (86A = Genitive-Noun; kaz by genus): swe hin urd mar tur azj kaz fin ekk cmn_Hans cmn_Hant jpn kor tam tel
- NADJ (87A = Noun-Adjective; mlt by genus): fra spa por pes arb heb mlt ind vie khm
OV and POST disagree only on pes (OV, prepositions), fin and ekk (VO, postpositions): power is limited.
Tests, each with base + all 20 Glottolog splits (genealogy held fixed):
- A1: OV | POST  -- order of object and verb beyond adposition order (random 11-subset null)
- A2: POST | OV  -- adposition order beyond object-verb order (random 12-subset null)
- A3 (descriptive): single-split gains of OV, POST, GENN, NADJ, and of each beyond the other three.
Claim "the axis is specifically object-verb order": A1 p < .0125 in both models (>= 1 metric each) while A2 is not.
Claim "head-direction bundle": both A1 and A2 significant. "Inseparable at 34 languages": neither.

## (B) Lexical control for genealogy
Two lexical distances between languages, added to the base:
- LEX_TEXT: 1 - mean Jaccard similarity of character 3-gram sets of the romanised (unidecode, lowercased) parallel
  FLORES dev+devtest sentences -- shared words, cognates, loans and names as the model sees them.
- LEX_ASJP: LDND (Wichmann et al. 2010) on the ASJP 40-item lists: mean normalised Levenshtein distance between
  same-concept forms (min over synonyms) divided by the mean over different-concept pairs. Doculects fixed in advance:
  ENGLISH STANDARD_GERMAN DUTCH SWEDISH FRENCH SPANISH PORTUGUESE RUSSIAN UKRAINIAN POLISH CROATIAN SERBOCROATIAN(srp)
  HINDI URDU MARATHI PERSIAN STANDARD_ARABIC MODERN_HEBREW MALTESE TURKISH AZERBAIJANI_NORTH KAZAKH FINNISH ESTONIAN
  MANDARIN(both cmn) JAPANESE KOREAN INDONESIAN TAGALOG(fil) VIETNAMESE KHMER TAMIL TELUGU.
Tests: B1 = H1 (20 Glottolog splits, leaf-relabelling null) with base + LEX_TEXT; B2 = base + LEX_TEXT + LEX_ASJP.
LEX_ASJP is itself a genealogical signal (basic vocabulary is how families are found), so B2 is deliberately
over-conservative. Claim "genealogy beyond shared vocabulary in the input": B1 p < .0125 in both models
(>= 1 metric). Descriptive: per-layer genealogy gain with and without LEX_TEXT (does the early/late peak shrink?).
