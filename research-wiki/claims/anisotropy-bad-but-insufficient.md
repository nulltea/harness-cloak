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

**status:** `drafted`

## Statement
Embedding anisotropy is detrimental to RANTEXT's absolute-distance mechanism: high mean pairwise cosine compresses the Euclidean distance dynamic range, collapsing the random-radius candidate set and flattening the exponential-mechanism scores toward uniform sampling (G1/G2). However, isotropy is necessary-not-sufficient: an isotropic LLM embed_tokens matrix that is more distance-concentrated and lacks local synonym structure yields WORSE perturbation retention (0.23 vs 0.79) than an anisotropic sentence-embedding space. The binding requirements are distance dynamic range + synonym-structured local neighbourhoods; anisotropy's harm is specific to the absolute-distance formulation and is largely sidesteppable by rank-based (k-NN) candidate selection. Grounded in Mu&Viswanath 2018, Ethayarajh 2019, Gao 2019, Xu 2020 (Mahalanobis), Asghar 2024 (d_X curse).

## Honest scope
_TODO: what this claim does NOT say; banned wordings; flagged imports._

## Evidence chain
_TODO: proof obligations, jury verdicts, provenance pointers._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

