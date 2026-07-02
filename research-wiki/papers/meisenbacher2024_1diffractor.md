---
type: paper
node_id: paper:meisenbacher2024_1diffractor
title: "1-Diffractor: Efficient and Utility-Preserving Text Obfuscation Leveraging Word-Level Metric Differential Privacy"
authors: ["Stephen Meisenbacher", "Maulik Chevli", "Florian Matthes"]
year: 2024
venue: "ACM IWSPA 2024"
external_ids:
  arxiv: "2405.01678"
  doi: null
  s2: null
tags: ["word-level-mdp", "efficiency", "obfuscation", "rd4"]
added: 2026-07-02T00:00:00Z
---

# 1-Diffractor: Efficient Utility-Preserving Text Obfuscation via Word-Level Metric DP

## One-line thesis
Word-level metric-LDP obfuscation that selects perturbation candidates from one-dimensional embedding
lists via a geometric-distribution "diffraction", ~15× faster and lighter than prior MLDP mechanisms.

## Key Results
- >15× throughput and lower memory vs. previous word-level MLDP, with competitive utility/privacy.

## Relevance to This Project
**Why surfaced:** the *efficiency* branch of the RD4 space in
[`docs/research/learned-substitution.md`](../../docs/research/learned-substitution.md). **Fit:** shows
the cost axis — the fully learned/contextual RD4 methods (DP-MLM, DP-ST) are heavy; 1-Diffractor marks
the fast, word-level end (but stays word-level, so F1a persists).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
