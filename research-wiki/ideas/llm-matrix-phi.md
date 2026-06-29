---
type: idea
node_id: idea:llm-matrix-phi
title: "Use an LLM input-embedding matrix as φ to fix RANTEXT geometry"
stage: piloted
outcome: negative
added: 2026-06-29T13:33:58Z
based_on: ["paper:xu2020_differentially_private_text"]
target_gaps: ["gap:G2"]
tags: ["anisotropy", "llm-matrix"]
---

# Use an LLM input-embedding matrix as φ to fix RANTEXT geometry

**stage:** `piloted`  ·  **outcome:** `negative`

## Thesis
Swap qwen3-embedding for an isotropic LLM embed_tokens matrix (Qwen3-1.7B) to remove anisotropy and improve perturbation faithfulness.

## Key risks
Isotropy may not help if the matrix is more distance-concentrated or lacks local synonym structure.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

