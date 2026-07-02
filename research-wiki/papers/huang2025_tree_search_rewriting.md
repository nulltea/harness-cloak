---
type: paper
node_id: paper:huang2025_tree_search_rewriting
title: "Zero-Shot Privacy-Aware Text Rewriting via Iterative Tree Search"
authors: ["Shuo Huang", "Xingliang Yuan", "Gholamreza Haffari", "Lizhen Qu"]
year: 2025
venue: "EMNLP 2025"
external_ids:
  arxiv: "2509.20838"
  doi: null
  s2: null
tags: ["text-rewriting", "tree-search", "reward-model", "zero-shot", "rd4"]
added: 2026-07-02T00:00:00Z
---

# Zero-Shot Privacy-Aware Text Rewriting via Iterative Tree Search

## One-line thesis
Obfuscate/delete private segments by a zero-shot, reward-model-guided *tree search* over rewrites,
preserving coherence and naturalness — inference-time search instead of finetuning.

## Method
Incremental sentence rewriting through a structured tree search guided by a reward model, dynamically
exploring the rewriting space per privacy-sensitive segment.

## Key Results
- Outperforms rule-based redaction/scrubbing baselines on privacy-utility balance and naturalness.

## Relevance to This Project
**Why surfaced:** round-2 (arXiv + Semantic Scholar) discovery — the **search-based** branch of RD4 in
[`docs/research/learned-substitution.md`](../../docs/research/learned-substitution.md). **Fit:** a
reward-guided, training-free alternative to the finetuned (DP-MLM) and RL ([AgentStealth](shao2025_agentstealth.md))
substitutors; complements [RUPTA](yang2025_rupta.md)'s evaluator-guided optimization at inference time.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
