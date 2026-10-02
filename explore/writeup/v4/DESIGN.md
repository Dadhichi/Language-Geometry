# The Word-Order Axis, v4: design notes

Owner: design agent. Files: `shell.html`, `runtime.js`, `figures/*.js`, `build.py`, `check.js`, `dev_content/`,
this file, and two added keys in `../build_data.py` (`map_layers`, `tree`). The markup contract is `CONTRACT.md`.

## Build and verify

```
PY=C:/Users/ASUS/AppData/Local/Temp/claude/c--Users-ASUS-OneDrive-Documents-Projects-Language-Structure-in-LLMs/fa049fb0-6ccd-47bf-be70-a43fa5fa9e65/scratchpad/venv/Scripts/python
$PY explore/writeup/build_data.py explore/writeup/figdata.json        # adds map_layers + tree (needs the local data)
python explore/writeup/v4/build.py                                     # figdata.json + v4/content -> v4/preview.html
python explore/writeup/v4/build.py FIGDATA.json OUT.html [--content DIR] [--strict]
node explore/writeup/v4/check.js OUT.html --libs <scratchpad>/domtest  # jsdom check: errors, math, marks, numbering, figrefs, tables
```

* `build.py` concatenates `shell.html` + `content/NN-*.html` (file-name order) + `runtime.js` + `figures/*.js` + the data
  (as `<script type="application/json">`, `<` escaped). Only d3 7.9.0 and KaTeX 0.16.11 come from cdnjs, fonts from
  Google Fonts. Output is an HTML fragment (no doctype/head/body): the artifact host adds the skeleton.
* It lints the content against the contract and prints warnings: inline styles, scripts, classes outside the contract,
  duplicate ids, `figure` id/data-fig mismatch, figures or data tables without a module, figrefs to nothing, sections
  without `data-title`, contract figures not yet placed. `--strict` turns warnings into a non-zero exit.
* If `v4/content` has no `NN-*.html`, it falls back to `v4/dev_content` and says so loudly.
* Dev-only flags: `--local-libs DIR` inlines `d3.min.js` and `katex.min.js` (offline screenshots, tests);
  `--mock-freegen2` adds a FAKE `freegen2` (exploratory data relabelled onto the nine confirmatory languages) so the
  confirmatory figures can be exercised. Never publish a page built with either flag; both print a warning.
* Screenshots: `scratchpad/shoot4.py PAGE OUTPREFIX --themes light,dark --widths 1280,390 --groups map,steer ...`
  (headless Edge, fonts inlined from `scratchpad/fonts`, phone width via a 390 px iframe, `--script` to drive
  interactions). `scratchpad/overflow.py PAGE` reports horizontal overflow at 390 px.

## Design direction

Transformer Circuits / Distill conventions, with a calmer, more typographic surface than v3.

* **Grid.** A 680 px reading column set left of centre on a 1080 px page; a 212 px right gutter for sidenotes on
  screens ≥ 1180 px. Every `section` is a CSS grid with named lines (`page`, `text`, `gutter`); figures declare
  `page` (1080), `wide` (text column + gutter, 932) or `text` (680). Captions always sit on the text column
  (subgrid where supported). Below 1180 px the gutter folds away; at 390 px there is a 16 px gutter and no horizontal
  page scroll (checked).
* **Type.** Noto Sans for everything (body 17 px / 1.68, headings 640–720 weight with slight negative tracking, chart
  text 12 px), Noto Sans Mono for codes and example tags. Noto was chosen because it is one family designed to cover
  every script in the article (Latin, Cyrillic, Devanagari, Arabic, CJK...), which keeps examples consistent with the
  text. Math is KaTeX → MathML in the platform math font (Cambria Math / STIX Two Math / Latin Modern Math).
* **Front matter.** A tinted band with the title, a subtitle and a byline grid (labels above values; becomes a two-column
  list on phones). The runtime wraps the header children in `.front-inner` to put them on the page grid.
* **Navigation.** A slim fixed top bar: article title (appears once the H1 scrolls away), the current section as a button
  that opens a contents menu (keyboard: Esc closes, focus moves to the current entry), a theme toggle and a 2 px reading
  progress line. A numbered contents list follows the summary. On screens ≥ 1500 px a contents rail appears in the left
  margin once the in-flow list scrolls out of view.
* **Sidenotes.** Inline boxes by default; on wide screens the runtime moves each `aside.sidenote` into the gutter level
  with the paragraph it annotates, stacking notes and stepping them past wide figures.
* **Blocks.** `.def` and `.pending` are quiet tinted panels with small-caps labels; `.takeaway` is a rule-topped "Key
  result" block; `.note` is a hairline-indented aside; `.example` is a two-column gloss table (mono tag, text, gloss).
* **Long math.** MathML cannot break lines. Inline math wider than its line becomes its own horizontally scrolling box
  (`.m-long`), and display math or tables that scroll get a fade on the edge they can scroll toward.
* **Themes.** All colours are tokens on `:root`; dark values are repeated under `prefers-color-scheme: dark` (guarded by
  `:root:not([data-theme="light"])`) and under `:root[data-theme="dark"]`. The toggle writes `data-theme` and remembers it
  in `localStorage` (try/catch; the host's own `data-theme` wins on load). SVG marks take colour from CSS classes
  (`c-ov`, `s-gen`, `band2`...), so a theme switch needs no redraw (the runtime redraws anyway).
* **Print.** Bar, rail, controls and tooltips hidden; figures and tables avoid page breaks.

## Colour

Colour encodes meaning only, and the meaning is the same in every figure:

| token | light | dark | meaning |
|---|---|---|---|
| `--vo` | `#2a78d6` | `#3987e5` | verb before object; push toward verb-first (k < 0) |
| `--ov` | `#eb6834` | `#d95926` | object before verb; push toward object-first (k > 0); the OV split |
| `--gen` | `#1baf7a` | `#199e70` | genealogy / family tree; the Indo-European control direction |
| `--mixed` | `#8a8983` | `#8a8983` | WALS "no dominant order" (German, Dutch) |
| `--rand`, `--rand-strong` | `#c3c2b7`, `#a8a79e` | `#55544e`, `#6c6b64` | random directions (null) |
| `--ink`, `--ink2` | `#121211`, `#5c5b56` | `#f4f3ee`, `#b4b3ab` | non-semantic series (code-switch figure), unsteered points |

Validated with the dataviz skill's `validate_palette.js`, all-pairs (scatter rules), against the actual surfaces:

```
light, surface #ffffff: lightness PASS, chroma PASS, CVD PASS (worst #1baf7a↔#eb6834 ΔE 9.2 deutan),
  normal-vision PASS (worst 24.0), contrast WARN (#1baf7a 2.82:1 -> relief: every green series is direct-labelled
  or in the legend and values are in tooltips/tables)
dark, surface #131312: all PASS (worst CVD ΔE 9.4, normal-vision 20.9, all >= 3:1)
```

Greys are neutral folds, not categorical slots. Text contrast: body 14.2:1 (light) / 13.0:1 (dark); secondary 6.8 / 8.8;
muted (ticks, small caps) 4.8 / 5.3. The 13 groups of `langs.family` (10 families, Indo-European split into four branches)
are never coloured (13 hues cannot be told apart); they are drawn as outlines or links and named "one family, or one
branch of Indo-European" in legends and tooltips (`WOA.groupText`).

## Runtime (`runtime.js`)

`window.WOA` (`W` in the modules):

* `W.fig(id, {needs, layout, init(ctx), draw(ctx), pending(ctx)})`: register a figure. `needs` lists data paths
  (`"steering.examples"`); if one is missing, `pending(ctx)` renders a designed empty state (or a "data not in this
  build" note). `ctx` has `fig, body, controls, legend, graphic, D, state, width(), redraw()`.
* `W.table(id, (table, {D}) => ...)`: fill `table[data-table=id]`.
* Start-up order: theme → wrap front matter → KaTeX → contents → hide `.pending[data-pending=k]` when `D[k]` exists →
  right-align static-table headers over numeric columns → mount figures (insert controls/legend/graphic before the
  figcaption, prefix "Figure N." in document order) → tables → KaTeX for inserted math → fill `a.figref` → fit long
  math → sidenotes → resize observer → scroll tracking.
* Hover-linking: every mark that stands for a language carries `data-lang` (space-separated codes allowed;
  `zho_Hans` is normalised to `cmn_Hans`). `W.highlight(codes)` dims everything else in every figure and data table that
  contains a match and outlines the matches; modules can listen with `W.on("highlight", ...)`. Hovering a language,
  a family outline/link, a clade, a table row or a dot strip highlights it everywhere.
* Tooltips: `W.tip.show(eventOrElementOrXY, {title, rows: [{v, l, key}], note})`, values first, line/dot keys,
  `textContent` only. Every figure that has hover tooltips is also keyboard-navigable: the SVG is one tab stop and the
  arrow keys walk its marks (`W.keyNav`), Esc clears.
* Controls: `W.ui.seg` (segmented buttons, `aria-pressed`, arrow keys), `W.ui.button`, `W.ui.legend`.
* Redraw on width change (ResizeObserver), theme change, and when web fonts finish loading. `W.errors` collects runtime
  errors (the checker reads it).

## Figures

| id | layout | what it shows | interaction |
|---|---|---|---|
| `overview` | page | Qwen window residual map, outlines for the 13 family/branch groups with collision-free labels, arrow from the mean VO to the mean OV language | hover/keyboard: language or group, linked everywhere |
| `pipeline` | wide | seven steps (sentences → residual stream → mean pool → centroid → offset → distances → split model), each a small drawing; definition + formula of the selected step below (notation as in the text: u, θ, φ) | click or arrow keys to select a step |
| `map` | page | residual map per model and per layer (`map_layers`), MST links joining each family/branch (or script), bent around unrelated points | model toggle morphs Qwen ↔ Llama; outline toggle (family or branch / script / none); layer scrubber whose track is the per-layer gain profile (whitened geometry) with the pre-registered window; Window button; play; the frame and points animate together; readout gives the variance share of the two axes and the Procrustes similarity to the window map |
| `splits` | text | toy tree of six languages whose leaves label the rows of D²; selecting a branch (or the OV bar) fills the cells where δ_S = 1; hovering a cell lights the path between the two languages and writes D²_ij as the sum of its terms | segmented control + click branches/bar; arrow keys walk the cells |
| `tree` | page | Glottolog cladogram (20 clades named where they fit), leaf dots by WALS order, OV split as bars on the right; inset residual map shows the hovered clade (at rest: the 11 OV languages) | hover clade/leaf, linked everywhere; arrow keys |
| `depth` | page | small multiples (Qwen, Llama) of per-layer gain of tree and OV split against relative depth, window band, direct labels placed off both lines | geometry toggle (Whitened / Causal); crosshair synced across panels with p-values |
| `typology` | page | four small multiples (model × geometry), one row per WALS feature: alone (ring) vs unique (dot), largest unique value labelled | hover rows: values + the pre-registered A1/A2 conditional gains and p |
| `proj` | wide | held-out projections of the 8 test languages on a line split into verb-first / object-first halves, labels in lanes | hover/keyboard, linked |
| `pairs` | wide | four minimal pairs; object (underlined) and verb (bold) are recovered from the two twins; says which twin is attested and the language's mean unsteered preference | Show: object-first / verb-first twin (all cards glide), per-card Swap |
| `steer` | page | small multiples for layers 8, 14, 20 (one scale; random fan at 14), plus the pre-registered statistic: slope b against 32 random slopes, IE slope, OV slope with bootstrap interval and z | crosshair synced across panels; keyboard |
| `steer-langs` | wide | per-language preference unsteered (ring) and at k = ±2, plus each language's slope b | hover/keyboard, linked |
| `gen-rates` | wide | object-first share per language: k = −2, unsteered, k = +2, random band; hollow = under half in language; points with n < 20 not drawn; in-language shares noted under the name when low | Direction (word order / Indo-European), Strength (±2 / ±1) |
| `gen-null` | wide | pooled slope for the word-order direction vs 24 random (dodged) and IE, bootstrap interval, z; second row: noun objects only (`freegen_x.objtype_nom`) | hover |
| `gen-loss` | wide | share still in the prompt language at k = ±2 per language, the push against the majority order ringed, unsteered shown, random band; right column against − toward | hover/keyboard, linked |
| `gen2-rates` | wide | confirmatory noun-object shares, grouped H1 / H2 / controls, prediction met or not; pending state until `D.freegen2` | hover |
| `gen2-null` | wide | H1 and H2 tests: word-order slope vs 24 random, z, verdict (claim holds / not supported / not evaluable); IE dot only if `tests.H.b_ie` exists; pending state | hover |
| `cs` | page | small multiples for layers 4, 14, 24: representation λ_t (ink) vs Bayesian posterior (grey) around the switch | crosshair synced; keyboard |

| `within-align` | page | cosine of the within-language direction d_W with β_OV per layer (orange) and with β_IE (green, about −0.17 in the middle layers, shown as it is), ±99.9th percentile of the random-in-span null as a band, layer 14 marked, peak labelled; layer 0 (degenerate) left empty; a row of three small multiples on the same layer axis: cosine inside the language span, split-half reliability of d_W, word-order gain of the OV split (whitened) | crosshair synced across all four panels; tooltip with p, null, span share; keyboard |
| `within-steer` | page | one small multiple per language (flexible order first, then controls): noun-object share against k = −3…+3 for d_W (ink line, unsteered ring, faint unsteered reference line), β_OV at ±2 as squares (blue/orange, hollow under 50% in language), random band at ±2 (both signs, from `freegen2.rates[lang].rand`); shares from < 20 pairs not drawn and noted in the panel corner; strip below: pre-registered test B, pooled slope of d_W against 24 random | crosshair synced across panels by k; tooltip with n and language match; titles linked by language; keyboard walks language × k |
| `within-retain` | wide | per language, share of continuations in the prompt language at the "against" sign (k = 2) for d_W (ink dot) and β_OV (square coloured by push direction), unsteered for comparison, d_W − β_OV on the right | hover/keyboard, linked |

Data tables: `primary` (gain with inline bar, p, p beyond the other structure, same-language p; Bonferroni 0.0125 in the
cell title), `lex` (genealogy gain with character overlap and ASJP added, share kept), `gen-obj` (noun / pronoun shares
before the verb at k = −2, 0, +2; "–" below 10 pairs, n always printed; rows linked).

## Data added to `build_data.py`

* `map_layers`: `{qwen|llama: {L: {xy, var_share, rms, fit}}}` for every stored layer (29 / 33). Each layer map is
  `residual_map(derived, "L-L")` (same procedure and OV-right orientation as the window map), then rotated/reflected by
  orthogonal Procrustes onto the window map (no scaling). `rms` is the layer map's RMS radius, `fit` the Procrustes
  congruence with the window map. The figure rescales every layout to unit RMS. Existing keys are unchanged
  (verified byte-identical).
* `tree`: `{splits: [{name, codes}] (the 20 GLOTTO splits of lib34 with names), root: nested {name, children | code},
  ov: [codes]}`.

Colour identity added in round 2: **ink (dark in light mode, white in dark mode) = the within-language direction d_W**;
squares = β_OV, coloured by the direction it pushes (blue toward verb-first, orange toward object-first), as everywhere
else. Legends and tooltips accept subscripts (`["β", ["OV"]]`); SVG labels use `W.subText`.

## For the writer and the orchestrator

* Captions describing a control that is not there (figures use small multiples instead, which show all cases at once):
  `typology` ("Select the model and geometry with the control"), `steer` ("The control selects layer 8, 14 or 20"),
  `cs` ("The control selects layer 4, 14 or 24").
* `cs` caption says "Blue: the representation"; the line is dark ink (blue means verb-first everywhere else).
* `map` caption says "colour shows family, word order (WALS 83A) or script; lines join closely related languages".
  Colour is always word order; the Outline control switches the joining lines between family-or-branch, script and none.
* `steer` also shows the pre-registered slope statistic as a strip under the panels; `gen-null` has a second row
  for noun objects only (z = 8.2 in the text). Captions may mention them.
* TeX in `08-steer.html`: `k\in\{-2,-1,+1,+2\}` renders with binary-operator spacing; write `\{{-2},{-1},{+1},{+2}\}`.
* `freegen2`: `gen2-null` draws the Indo-European dot from `tests.H.b_ie` and the interval from `tests.H.ci`.
  `gen2-rates` spells out `per_language.pred` (analysis_gen2.py: "up" = rises under +k, "down" = falls under −k,
  "none" = no change) with met / not met / too few pairs.
* Round 2 (`within`): the content files were renumbered (10-within, 11-tokens … 14-methods); `build.py` takes any
  `NN-*.html` in file-name order, and `dev_content` mirrors the numbering with a stub `10-within.html`.
* The figure modules count n from `[share, n]` and hide points with n < 20 (gen-rates, gen2-rates).

## Known issues

* Map labels: at phone width many language labels do not fit and are hidden (they appear on hover/focus and when the
  language is highlighted from another figure).
* Group labels in `overview` are placed greedily; for crossing groups (Germanic / Semitic) a label can sit nearer a
  neighbouring outline. Hover resolves it.
* Sidenotes are positioned by script on wide screens; if a sidenote is longer than the paragraphs and figures that
  follow it in its section, the section grows to fit it.
* Equations that are wider than the column scroll horizontally (MathML has no line breaking).
* Headless screenshots were taken with the fonts inlined; on a slow network the page first renders in the fallback
  stack (Segoe UI / system-ui) and redraws the figures when Noto Sans arrives.
