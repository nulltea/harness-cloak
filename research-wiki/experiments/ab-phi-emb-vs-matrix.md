---
type: experiment
node_id: exp:ab-phi-emb-vs-matrix
title: "A/B: qwen3-embedding vs Qwen3-1.7B embed_tokens as φ"
idea_id: "idea:llm-matrix-phi"
verdict: no
confidence: high
date: ""
hardware: "host .venv GPU (matrix build) + API (gen/extract/scorer)"
duration: ""
provenance: "src/inferdpt/eval.py + results/e2e_ab.json; diagnostics.py"
added: 2026-06-29T13:33:58Z
tags: ["anisotropy", "ab", "geometry"]
---

# A/B: qwen3-embedding vs Qwen3-1.7B embed_tokens as φ

**verdict:** `no`  ·  **confidence:** `high`  ·  tests `idea:llm-matrix-phi`

## Metrics
At ε=3, calibrated to 5% |C_r|: qwen3-embedding anisotropy_cos=0.613 rel_spread=0.089 retention=0.788; Qwen3-matrix anisotropy_cos=0.019 rel_spread=0.017 retention=0.227. e2e leakage tied 0.42; utility 0.61 vs 0.64 (extraction re-grounding masks it).

## Reasoning
Isotropic LLM matrix is MORE distance-concentrated and lacks synonym-structured neighbourhoods, so removing anisotropy lowered retention. Isotropy necessary-not-sufficient.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

