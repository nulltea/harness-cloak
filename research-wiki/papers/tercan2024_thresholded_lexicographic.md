---
type: paper
node_id: paper:tercan2024_thresholded_lexicographic
title: "Thresholded Lexicographic Ordered Multiobjective Reinforcement Learning"
authors: ["Alperen Tercan", "Vinayak S. Prabhu"]
year: 2024
venue: "ECAI 2024"
external_ids:
  arxiv: "2408.13493"
  doi: null
  s2: null
tags: ["morl", "lexicographic", "thresholds", "policy-gradient-projection"]
added: 2026-07-31T00:00:00Z
---

# Thresholded Lexicographic Ordered Multiobjective Reinforcement Learning

## One-line thesis
Thresholded-lexicographic value methods lack a Bellman equation (no convergence guarantee), and a single global threshold is wrong because the indifference value varies by state; proposes Lexicographic Projection Optimization.

## Key Results
- A global epsilon is the known-wrong choice: per-decision thresholds required.
- Value-based thresholded lexicographic learning has no convergence theory.

## Relevance to This Project
Round-3 sweep: directly warns that a global epsilon in our tie rule would reproduce the global-dial failure with a hand-set constant — the per-decision, instrument-derived epsilon (and the exact-tie core needing none) is the compliant form.
