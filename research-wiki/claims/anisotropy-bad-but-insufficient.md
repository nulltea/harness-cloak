---
type: claim
node_id: claim:anisotropy-bad-but-insufficient
name: "Anisotropy harms RANTEXT, but isotropy alone is insufficient"
description: ""
node_type: claim
status: drafted
provenance: "docs/perturbation-sota.md; results/e2e_ab.json; experiments exp:geom-diag-qwen3emb, exp:ab-phi-emb-vs-matrix"
tags: ["anisotropy", "geometry", "rantext"]
date: 2026-06-29
added: 2026-06-29T13:33:58Z
---

# Anisotropy harms RANTEXT, but isotropy alone is insufficient

**status:** `revised 2026-06-30` — the original retention-based Statement below is superseded by the
floor-corrected analysis in **Honest scope**; read that first.

## Statement (revised)
**Raw `cos(orig,repl)` retention is dominated by the anisotropy floor and must not be used to rank φ.**
Floor-corrected retention `(cos−floor)/(1−floor)` collapses the cross-model spread from ~0.36 to ~0.02
(qwen/glove/cf/Phrase-BERT all 0.43–0.45). The binding, *floor-free* requirement is **synonym-structured
local neighbourhoods (P2)**, measured by WordNet syn_prec on a fixed query set: only the counter-fitting
retrofit moves it materially (glove→cf +0.061 [.058,.063]); isotropy, low-eff-dim, and paraphrase-training
do not. An isotropy-only transform (whitened GloVe) *lowers* corrected retention and leaves syn_prec flat
→ isotropy is not the lever. Anisotropy's harm to the *absolute-distance* formulation is real but largely
sidesteppable by rank-based (k-NN) selection. At ε=3 / matched candidate load, the P2 edge does **not**
reach end-to-end (pilot, n=25). Grounded in Mu&Viswanath 2018, Ethayarajh 2019, Gao 2019, Xu 2020,
Asghar 2024 + `results/phi_geometry_ab.json`, `results/phi_e2e_ci.json`.

<details><summary>Original Statement (superseded — relied on the retention artifact)</summary>

Embedding anisotropy is detrimental to RANTEXT's absolute-distance mechanism … an isotropic LLM
embed_tokens matrix … yields WORSE perturbation retention (0.23 vs 0.79) than an anisotropic
sentence-embedding space. [The 0.23-vs-0.79 gap is now known to be the anisotropy floor, not synonym
quality — see Honest scope.]
</details>

## Honest scope
**Update 2026-06-29 (sub-vocab A/B, `results/phi_geometry_ab.json`):** the original evidence — the
matrix's "retention 0.23 vs 0.79" gap — is now known to be an **anisotropy artifact**. Raw
`cos(orig,repl)` rides the random-pair floor, so anisotropic φ (qwen, floor 0.63) score high for free.
**Anisotropy-corrected** retention `(cos−floor)/(1−floor)` is **close** (0.43–0.45) across qwen, glove,
counter-fitted, Phrase-BERT — the cross-model spread collapses from ~0.36 raw to ~0.02 — but it is NOT a
formal tie within 0.03 (equivalence margin 0.076), driven by the whitened-GloVe outlier (0.377). So this
claim must NOT be defended on retention. It survives on the floor-free **WordNet syn_prec@10** (fixed
7,486-query set, paired CI): counter-fitting lifts it +0.061 [.058,.063] (glove→cf, same base vectors),
while isotropy / low-dim / paraphrase-training (Phrase-BERT 0.105) do not — P2 is the improvable axis,
reached only by explicit synonym-fitting. Banned wording: "isotropy lowered retention" (it didn't, once
floor-corrected). Absolute syn_prec is low (~1.5/10) for all φ → P2 headroom is real but small.

## Evidence chain
- `exp:ab-phi-emb-vs-matrix`, `exp:geom-diag-qwen3emb` — original (retention-based) evidence, now
  reinterpreted as floor-confounded.
- `results/phi_geometry_ab.json` + `scripts/phi_geometry_ab.py` — 4-φ sub-vocab A/B with `ret_corr` and
  `synonym_precision`; the floor-free measure isolating counter-fitting.
- e2e arm (C3, `results/e2e_ab_subvocab.json`): **resolved — masked.** e2e utility/leakage tied across
  all 4 φ; the counter-fitting syn_prec edge does not survive the extraction LLM. ⇒ φ (Lever 2) has low
  leverage on end-to-end utility at ε=3/|C_r|=5%. (n=5 docs, seed=0 — directional.)

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

