---
type: paper
node_id: paper:haarnoja2018_sac_algorithms_applications
title: "Soft Actor-Critic Algorithms and Applications"
authors: ["Tuomas Haarnoja", "Aurick Zhou", "Kristian Hartikainen", "et al."]
year: 2018
venue: "arXiv"
external_ids:
  arxiv: "1812.05905"
  doi: null
  s2: null
tags: ["sac", "target-entropy", "temperature-auto-tuning", "maximum-entropy-rl"]
added: 2026-07-30T00:00:00Z
---

# Soft Actor-Critic Algorithms and Applications

## One-line thesis
Replaces the fixed entropy temperature with a dual variable auto-tuned to hold a target entropy, because a fixed coefficient is brittle and reward-scale-dependent.

## Key Results
- Constrained formulation: maximize return subject to a minimum expected entropy; temperature becomes a Lagrange multiplier.
- Robust across reward scales where fixed temperatures fail.

## Relevance to This Project
Surfaced for the entropy-floor solution family (root-cause link 3). The canonical auto-tuned entropy-floor mechanism: enforce a small target entropy so softmax margins stay bounded without pinning which action wins - lambda-0 stays free to optimize utility, and the bounded additive controller becomes a deterministic tie-breaker. Replaces our hand-set beta=0.01 with a principled dual variable.
