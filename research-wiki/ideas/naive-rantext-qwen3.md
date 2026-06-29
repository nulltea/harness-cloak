---
type: idea
node_id: idea:naive-rantext-qwen3
title: "Naive RANTEXT with qwen3-embedding as φ"
stage: archived
outcome: negative
added: 2026-06-29T12:32:21Z
based_on: ["paper:tong2023_inferdpt_privacypreserving_inference"]
target_gaps: ["gap:G1", "gap:G2"]
tags: ["rantext", "qwen3", "baseline"]
---

# Naive RANTEXT with qwen3-embedding as φ

**stage:** `archived`  ·  **outcome:** `negative`

## Thesis
Use the reference RANTEXT mechanism unchanged with qwen3-embedding-0.6b (single-token strings) as φ for privacy-preserving black-box inference.

## Key risks
Mechanism assumes a well-spread embedding geometry (G1/G2); qwen3 single-token vectors may be too concentrated/anisotropic.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

