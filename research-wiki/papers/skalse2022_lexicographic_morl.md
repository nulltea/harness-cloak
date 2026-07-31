---
type: paper
node_id: paper:skalse2022_lexicographic_morl
title: "Lexicographic Multi-Objective Reinforcement Learning"
authors: ["Joar Skalse", "Lewis Hammond", "Charlie Griffin", "Alessandro Abate"]
year: 2022
venue: "IJCAI 2022"
external_ids:
  arxiv: "2212.13769"
  doi: null
  s2: null
tags: ["morl", "lexicographic", "tau-slack", "tie-breaking"]
added: 2026-07-31T00:00:00Z
---

# Lexicographic Multi-Objective Reinforcement Learning

## One-line thesis
Filter actions to a tau-slack near-optimal set of the primary objective, then optimize the next objective inside it; admissible iff tau is below the minimum gap between DISTINCT-value actions, and when unverifiable the primary loss is bounded by tau/(1-gamma); recommends a relative, decaying slack rather than a global constant.

## Key Results
- The action-selection rule our evidence-driven tie-break instantiates.
- Admissibility condition: exact ties contribute nothing to the minimum gap — breaking exact ties is FREE.
- Proposition 1: worst-case primary loss tau/(1-gamma) — the honest utility-cost statement to preregister.

## Relevance to This Project
Surfaced in the round-3 sweep as the formal home of the tie-ownership mechanism: our design is this rule with tau derived from the measurement instrument's noise floor and the set built from runtime counterfactual probes instead of learned value estimates — that combination appears unpublished.
