# Idea gathering, 2026-09-29: candidate mathematical structures for "languages" inside an LLM

Goal (far out, don't pivot): does a natural, emergent structure among languages form in pretraining (e.g. the
Indo-European tree, typological parameters), analogous to numbers on circles/helices? Built on explore/FINDINGS.md
(mid-depth: x ~= mu_i + e^{s_i} b + small rotation; centroids ~5-dim; additive script+family failed; weak family
Mantel at mid-depth; post-hoc head-final factor hin/tur/jpn). Three literature scouts (Anthropic circuits thread;
geometric/group structure in mech interp; emergent language/typology structure). All sources below were opened
by a scout; arXiv/ACL links.

## Template borrowed from the counting-manifold paper
A latent variable of the data-generating process gets a low-dim manifold whose shape is fixed by a symmetry or a
distinguishability/packing trade-off (circulant similarity -> Fourier -> "ringing"); computation manipulates it
(QK rotates one manifold onto another); sparse features tile it. Method: class-mean vectors -> PCA -> overlay
feature decoders -> probes' cosine matrix -> ablate top-k PC subspace vs random subspace; patch a - mu_orig + mu_c.
(https://transformer-circuits.pub/2025/linebreaks/index.html; numbers-on-a-circle with "+k" as a rank-2 rotation
is the MOLT illustration in https://transformer-circuits.pub/2025/bulk-update/index.html; mod-10/100 addition
features in https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

## Candidates (ranked)
1. **Tree symmetry -> Haar wavelets** (analogue of cyclic symmetry -> Fourier). If languages in a clade are
   exchangeable given the clade, the centroid covariance is ultrametric (Brownian motion on the phylogeny:
   K_ij = shared branch length); its eigenvectors are Haar wavelets on the tree, PCs split subtrees coarse->fine
   (Nava & Wyart 2026, arXiv 2605.23821, statistic g(k) = (1/k)||U_k^T V_k||_F^2); Park et al. orthogonality
   parent _|_ (child - parent) under the causal inner product (arXiv 2406.01506; for Qwen: <l,l'> = l^T Cov(gamma) l'
   with gamma = lm_head rows, l = final-RMSNorm output; mid-layer use is a proxy -> cross-check with pooled
   within-language covariance). Tests: NNLS tree-covariance fit; g(k) vs label-permuted trees, script tree,
   token-overlap tree. Needs ~25-35 languages (>=2-3 per clade) + two-script pairs. Prior: trees recovered from
   LM language distributions (Rakshit et al. 2026, arXiv 2607.22699, no partial controls); within-family simplex
   under causal IP, Latin-script Germanic/Romance only (Marinov et al. 2026, arXiv 2606.14347).
2. **Belief-state simplex over language identity** (closest to the counting paper). Transformers linearly embed
   belief states over latents of the generating process (Shai et al. NeurIPS 2024, arXiv 2405.15943). Prediction:
   prefix representations = affine image of P(language | prefix); unambiguous text at vertices, ambiguous prefixes on
   faces (Latin-script face; Han-only prefixes on the zho-jpn edge); tokens move the point by Bayes updates.
   Family structure as a packing consequence: 12 languages in ~5 dims -> confusable languages placed close.
   Continuity test (Anthropic Jul 2024 update's manifold-vs-features criterion): code-mixed FLORES sentences with a
   fraction p of words swapped A->B trace smooth paths (manifold) vs a sigmoid snap (discrete features); control
   token counts. Needs token-level re-extraction (cheap).
3. **Parameter hypercube** (typology as coordinates). mu_i = c + sum_p pi_ip v_p over binary WALS/lang2vec
   parameters; parallelogram law for single-parameter flips. Family- and script-held-out CV per Ostling & Kurfali
   2023 (arXiv 2301.08115: only OV, adposition, numeral-noun order survive proper CV). Prior: language subspace
   correlates with URIEL syntax (Xie et al., arXiv 2401.05792). Our hin/tur/jpn factor is the pilot.
4. **Fiber bundle / content-dependent offset.** Half of per-sentence offset energy is sentence-specific (D).
   Structured function of content (numbers, names, punctuation where scripts/tokenizers differ) or noise?
   Runnable now on local data.
5. **Information-geometry null (mandatory control).** Centroid geometry ~= geometry of the languages' token
   distributions (JS/Fisher). Tokenizer vocabulary overlap alone gives Mantel r = 0.33 with genealogy
   (arXiv 2601.18791). Every "emergent" claim must beat it (partial Mantel / matrix regression).
6. **Depth as manipulation / two language codes.** Configuration keeps its shape through depth but is re-embedded
   at L0->L2 and L26->L28 (D): read-in/read-out. Anthropic's global-workspace paper finds a reportable language-name
   vector ("Spanish") separable from the production language (2026/workspace): test whether our offsets are the
   production code, the reportable code, or both (patch a - mu_i + mu_j; measure output language vs language-name
   logit / J-lens). Early language-detection features and language-agnostic middle (biology.html);
   "more closely-related languages are represented more-similarly" (feature IoU, 2025/september-update).
7. **Languages as group actions (MOLT analogue).** Translation "into X" as sparse low-rank transforms; ICA on
   log-maps for shared generators (successor-head meta-heads analogue, 2024/september-update). Our maps are
   ~identity + small non-commuting rotation, so expect little here unless re-framed at token level.

## Shared method checklist
emotions-paper concept vectors (class mean - grand mean, project out top PCs of neutral text) -> cosine/RSA vs
typology/tree/script/token-overlap matrices with partial Mantel; ablate top-k vs random-k subspaces; mean-swap
patching with norm-matched random controls; label permutations (only 66 pairs at 12 languages -> add languages).

## Resources
lang2vec (github.com/antonisa/lang2vec; ISO 639-3; syntax/phonology/inventory/fam/geo + precomputed distances),
URIEL+ (pip urielplus; adds Grambank, script and morphological distances), Grambank (doi:10.5281/zenodo.7740139),
WALS TSV, Glottolog 5.3 Newick. Extra FLORES languages: script-pair controls urd_Arab (vs hin), azb_Arab/azj_Latn,
kas_Arab/kas_Deva, zho_Hant, arb_Latn; crossing: pes_Arab, mlt_Latn, heb_Hebr, amh_Ethi, kor_Hang, tam_Taml,
ben_Beng, mar_Deva, eus_Latn, fin/hun/est_Latn, kaz_Cyrl/uzn_Latn, ukr/bul/srp_Cyrl vs pol/ces/hrv_Latn, ell_Grek,
hye_Armn, kat_Geor, tgl_Latn, ita/por/nld/swe (verify word orders in WALS).
