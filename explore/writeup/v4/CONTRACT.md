# Write-up v4 — contract between the writing agent, the design agent and the orchestrator

Goal: rebuild the article "The Word-Order Axis" (current build: `explore/writeup/word-order-axis.html`, sources
`template.html`, `sections/*.html`, `extra_figs.js`) as a Transformer Circuits–style article
(https://transformer-circuits.pub). Same science, same verdicts. Better prose, better figures, more interaction.

## Roles and file ownership (do not edit files you do not own)

| Role | Owns | Never touches |
|---|---|---|
| Writer | `v4/content/*.html`, `v4/content/NUMBERS.md` | CSS, JS, `build_data.py`, experiment code |
| Designer | `v4/shell.html`, `v4/runtime.js`, `v4/figures/*.js`, `v4/build.py`, `v4/dev_content/`, `v4/DESIGN.md`; may ADD keys to `explore/writeup/build_data.py` (never change or remove existing keys) | `v4/content/`, experiment code, the published file |
| Orchestrator | experiments, `figdata.json` regeneration, final build, verification, publishing, git | — |

The old pipeline (`template.html`, `inject.py`, `sections/`, `extra_figs.js`) stays as reference. Do not edit it.
Nobody except the orchestrator commits, pushes or publishes.

## Build

`python v4/build.py FIGDATA.json OUT.html [--content v4/content]` assembles `shell.html` + every
`content/NN-*.html` in file-name order + `runtime.js` + `figures/*.js` + the figure data into ONE self-contained HTML
file. Default OUT for development: `v4/preview.html`. The orchestrator builds the final page to
`explore/writeup/word-order-axis.html` (that path is the published artifact; do not write it).
Data: `python explore/writeup/build_data.py explore/writeup/figdata.json` (needs local data under
`C:\Users\ASUS\Documents\lang-geom`). Python: the scratchpad venv
`C:/Users/ASUS/AppData/Local/Temp/claude/c--Users-ASUS-OneDrive-Documents-Projects-Language-Structure-in-LLMs/fa049fb0-6ccd-47bf-be70-a43fa5fa9e65/scratchpad/venv/Scripts/python`.

## Markup contract (content files)

Each file `v4/content/NN-id.html` holds exactly one `<section id="ID" data-title="Short TOC title"> … </section>`
(file `00-front.html` holds the front matter: `<header class="front">` with `<h1>`, `<p class="subtitle">`,
`<dl class="byline">` (`<dt>`/`<dd>` pairs), then `<section id="summary" data-title="Summary">`).
No inline styles, no scripts, no classes other than those below.

| Element | Markup | Who renders it |
|---|---|---|
| Inline math | `<span class="m">TeX</span>` | runtime: KaTeX → MathML |
| Display math | `<div class="mb">TeX</div>` | runtime: KaTeX → MathML, scrolls horizontally if wide |
| Figure | `<figure id="fig-ID" data-fig="ID"><figcaption><b>Takeaway sentence.</b> What is plotted, encodings, n.</figcaption></figure>` | runtime inserts controls + graphic BEFORE the figcaption and prefixes "Figure N." in document order |
| Figure reference | `<a class="figref" href="#fig-ID"></a>` | runtime fills "Figure N" |
| Data table | `<div class="tablewrap"><table data-table="ID"></table></div>` + optional `<p class="tcap">` caption after it | designer code fills the table |
| Static table | `<div class="tablewrap"><table>…</table></div>` (writer writes rows; `<td class="n">` for numbers) | CSS only |
| Sidenote | `<aside class="sidenote">…</aside>` directly after the paragraph it annotates | right margin on wide screens, inline box on narrow |
| Definition box | `<div class="def"><span class="lab">Label</span>…</div>` | CSS |
| Key result | `<div class="takeaway">…</div>` (max one per section) | CSS |
| Caveat | `<p class="note">…</p>` | CSS |
| Pending result | `<p class="pending" data-pending="freegen2">…</p>` | runtime hides it once `D.freegen2` exists |
| Example block | `<div class="example" lang="ru"><div class="ex-row"><span class="ex-tag">prompt</span> <span class="ex-text">…</span> <span class="ex-gloss">(…)</span></div> …</div>`; verbs `<b>`, objects `<u>` | CSS |
| Citation | `<a class="cite" href="#rN">[N]</a>`; list in the last section as `<ol class="refs"><li id="rN">…</li></ol>` | CSS |

TeX notes: KaTeX syntax; escape `<` as `&lt;` and `&` as `&amp;`; MathML spacing treats `=-2` as binary minus, so
write signed numbers grouped: `k={-2}`, `={+6.8}`.

## Sections and figures (IDs are fixed; the writer may add `<h3>` subsections; designer may add figures only in agreement with the orchestrator)

| File | Section id | Figures / data tables (ID → data in `figdata.json`) |
|---|---|---|
| 00-front | summary | `overview` (optional teaser: residual map with family hulls and the word-order direction; `map`, `langs`, `tree`) |
| 01-intro | intro | — |
| 02-setup | setup | `pipeline` (explanatory diagram, no data: sentences → model → residual stream at layer ℓ → mean pool (drop sink) → language centroid → offsets → distance matrix D² → split model) |
| 03-first | first | — (the per-language PCA trap and mean-pooling dilution; writer may use a static table) |
| 04-map | map | `map` (residual map; model toggle; NEW: layer scrubber over per-layer maps `map_layers`; colour by family / word order / script; hover-linked to `tree`) |
| 05-tests | tests | `splits` (interactive toy explainer: click a branch of a 6-leaf tree → highlight the split S and the cells of D² that δ_S adds to; no data); `tree` (Glottolog tree of the 34 languages as a cladogram, word-order leaves marked, NEW data `tree`); data table `primary` (`profiles.*.primary`) |
| 06-depth | depth | `depth` (per-layer gains, genealogy vs word order, both models, `profiles`) |
| 07-which | which | `typology` (`typology_lex_*`); data table `lex` (`typology_lex_*`) |
| 08-steer | steer | `proj` (`steering.heldout_proj`), `pairs` (`steering.examples`), `steer` (`steering.layers`, `steering.rand`), `steer-langs` (`steering.per_language`) |
| 09-gen | gen | `gen-rates` (`freegen_x.rates`, `freegen_x.match`), `gen-null` (`freegen_x.pooled`, `freegen_x.b_ov_boot95`), data table `gen-obj` (`freegen_obj`), `gen-loss` (`freegen_x.sign_loss`, `freegen_x.match`); confirmatory: `gen2-rates`, `gen2-null` (`freegen2`, pending, schema below) |
| 10-within | within | `within-align` (`within.A_profile`, `within.A`, `within.dW_info.split_half_cos`; optionally `profiles` for the OV-gain shape), `within-steer` (`within.rates`, `within.match`, random band from `freegen2.rates[lang].rand`), `within-retain` (`within.C.per_language`) |
| 11-tokens | tokens | `cs` (`belief`) |
| 12-dead | dead | static table |
| 13-discussion | discussion | — |
| 14-methods | methods | static pre-registration table; references |

### Round 2 (2026-10-03): section 10-within (pre-registered study `explore/steer_within`, commit 14fa22e)
Data key `within` = `explore/steer_within/results_w.json`:
`A` (layer 14: cos_ov, cos_ie, null_q999, null_sd, p, span_share, cos_ov_within_span, claim), `A_profile` (one entry
per layer 0–28, same fields + `cos_unnat_ov`; layer 0 is degenerate: the twins have identical tokens, so their
pooled embeddings are identical — grey it out or omit it), `dW_info` (`split_half_cos`, `norm_order`, `norm_unnat`,
`cos_order_unnat`, `cos_w_ov_<lang>` per layer/language), `B` (b_w, per_language, rand [24], rand_mean, rand_max,
z, evaluable, claim), `C` (per_language {against_k, match_w, match_ov}, higher, n, p_sign, mean_w, mean_ov, claim),
`rates` {lang: {base, "w-3", "w-1", "w-", "w+1", "w+", "w+3", "ov-", "ov+": [noun-object share | null, n]}}
("w-"/"w+" = k ∓2), `match` {lang: {base, "w-", "w+", "w-3", "w+3", "ov-", "ov+"}}, `repetition`, `dose`
{"1.0", "2.0", "3.0"}, `determinism`.
* `within-align`: cosine of the within-language direction d_W with beta_OV per layer, the null (random directions in
  the language span; draw ±null_q999 as a band), the Indo-European control cosine; the pre-registered layer 14
  marked. Small multiples, not a dual axis, if the within-span cosine or the OV-gain shape is shown too.
* `within-steer`: per language, noun-object share against k ∈ {−3, −2, −1, 0, 1, 2, 3} for d_W, with beta_OV at
  ±2 as separate marks and the random band at ±2; groups flexible (German, Dutch, Russian, Ukrainian, Polish,
  Croatian) and controls (English, Spanish, Korean); do not draw a share with n < 20; tooltips with n and
  language-match rate.
* `within-retain`: per language, language-match rate at the sign that pushes against its own order, d_W vs
  beta_OV (k = 2).

## Figure data that does not exist yet

* `map_layers` (designer adds to `build_data.py`): `{qwen: {L: {xy, var_share}}, llama: {...}}` for every stored layer,
  same procedure as `residual_map()` with the window `"L-L"`, same orientation rule (OV languages to the right).
  Procrustes-align each layer's 2-D map to the primary map so points do not flip between layers.
* `tree` (designer adds): the 20 Glottolog splits from `explore/prereg34/lib34.py` (`T.GLOTTO`, language index
  sets) as lists of language codes, plus a nested structure for drawing a cladogram.
* `freegen2` (orchestrator adds when the confirmatory run finishes): same shape as `freegen_x` where possible:
  `{k, langs, sets: {H1: [...], H2: [...], controls: [...]}, rates: {lang: {base: [share|null, n], "ov-", "ov+", "ie-", "ie+": [...], rand: [48 × share|null]}}, match: {lang: {base, ov_m, ov_p, ie_m, ie_p, rand: [48]}}, tests: {H1|H2: {b_ov, rand: [24], rand_mean, rand_sd, z, p_emp, n_langs, evaluable, claim}}, per_language: {lang: {b_ov, rand_mean, rand_sd, z, pred, pred_ok}}}`.
  Shares are NOUN-object shares (object before verb). Until it exists, `gen2-*` figures render a "pending" state.

## Platform constraints (the page is published as a claude.ai artifact)

* Scripts only from `cdnjs.cloudflare.com` (preferred), `cdn.jsdelivr.net/npm/`, `unpkg.com`; stylesheets only from
  Google Fonts; everything else inline. KaTeX CSS and fonts are blocked: render KaTeX with `output: "mathml"`.
* Colours as tokens on `:root`; dark mode under `@media (prefers-color-scheme: dark)` guarded by
  `:root:not([data-theme="light"])`, and again under `:root[data-theme="dark"]`; `body` has an explicit background.
* Works at 390 px width with a 16 px side gutter and no horizontal page scroll; `<title>` stays "The Word-Order Axis".
* `localStorage` only for conveniences, inside try/catch. Total size well under 16 MB.

## Facts

The science is fixed. Sources: the current article text (`template.html`, `sections/*.html`), `explore/FINDINGS.md`,
`CLAUDE.md`, `explore/*/PREREG*.md`, result files in `explore/*/results*.json` / `*.log`, `figdata.json`.
Every number in the text must come from a source; the writer lists each in `NUMBERS.md` (number → file → key/line).
Keep every "pre-registered" / "exploratory" label and every negative result.
