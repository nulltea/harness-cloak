---
type: experiment
node_id: exp:geom-diag-qwen3emb
title: "Geometry diagnostics: naive RANTEXT + qwen3-embedding"
idea_id: "idea:naive-rantext-qwen3"
verdict: no
confidence: high
date: ""
hardware: "CPU (API embeddings via llama-swap)"
duration: ""
provenance: "src/inferdpt/diagnostics.py on data/vocab (qwen3-embedding-0.6b, 12k)"
added: 2026-06-29T12:32:21Z
tags: ["geometry", "curse-of-dimensionality", "anisotropy"]
---

# Geometry diagnostics: naive RANTEXT + qwen3-embedding

**verdict:** `no`  ·  **confidence:** `high`  ·  tests `idea:naive-rantext-qwen3`

## Metrics
noise_scale=1.0 faithful: |C_r|/V=1.00 at ε=1..14, norm_entropy=1.00, cos(orig,repl)=0.61==random-pair baseline 0.613 → uniform random replacement, zero utility, ε inert. Geometry: random-pair cosine 0.613 (anisotropic), distance rel-spread 0.088 (p1..p99=0.67..1.04), kNN CoV~0.10. Calibrated noise_scale=0.38: |C_r|/V=0.07 but ε=3 still near-uniform (entropy 0.99, cos 0.75); retention only ε>=10.

## Reasoning
Curse of dimensionality + anisotropy of qwen3 single-token embeddings collapse the Euclidean random-radius mechanism to uniform sampling; noise radius (~2.0) exceeds the distance-shell max (~1.04) and Z(ε) saturates, so ε is inert.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

