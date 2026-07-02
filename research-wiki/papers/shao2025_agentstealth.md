---
type: paper
node_id: paper:shao2025_agentstealth
title: "AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text"
authors: ["Chenyang Shao", "Tianxing Li", "Chenhao Pu", "Fengli Xu", "Yong Li"]
year: 2025
venue: "NeurIPS 2025 (under review)"
external_ids:
  arxiv: "2506.22508"
  doi: null
  s2: null
tags: ["llm-anonymization", "reinforcement-learning", "adversarial", "local-model", "rd4"]
added: 2026-07-02T00:00:00Z
---

# AgentStealth: RL-reinforced LLM anonymization on a local model

## One-line thesis
Anonymize user text with a *locally deployed small* LLM trained by in-context contrastive learning +
supervised adaptation + **online RL** that uses the model's own adversarial feedback.

## Method
Adversarial anonymization workflow (in-context contrastive learning + adaptive utility-aware control) →
supervised adaptation → online RL loop where internal adversarial feedback iteratively improves
anonymization. Small, locally deployable model; code released.

## Key Results
- +12.3% anonymization effectiveness and +6.8% utility over baselines.

## Relevance to This Project
**Why surfaced:** round-2 (Semantic Scholar + arXiv) discovery — a **new RD4 sub-family (RL-guided)** in
[`docs/research/learned-substitution.md`](../../docs/research/learned-substitution.md). **Fit:** the
self-improving, on-device version of the [Staab](staab2024_llm_anonymizers.md)/[RUPTA](yang2025_rupta.md)
adversarial loop — arguably the strongest local-substitutor direction; trades formal DP for empirical
gains and adds reward-hacking risk.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
