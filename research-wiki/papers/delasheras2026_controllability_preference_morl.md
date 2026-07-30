---
type: paper
node_id: paper:delasheras2026_controllability_preference_morl
title: "Controllability in Preference-Conditioned Multi-Objective Reinforcement Learning"
authors: ["G. de las Heras Molins", "et al."]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2605.10585"
  doi: null
  s2: null
tags: ["morl", "preference-conditioning", "controllability", "evaluation"]
added: 2026-07-30T00:00:00Z
---

# Controllability in Preference-Conditioned Multi-Objective Reinforcement Learning

## One-line thesis
Preference-conditioned agents can score well on standard MORL metrics while being insensitive to the preference input - the symbolic interface between user intent and behavior is broken - so controllability must be measured as a first-class quantity.

## Key Results
- Standard MORL evaluation structurally misses preference-insensitivity.
- Proposes a complementary controllability metric (sensitivity of realized behavior to the conditioning input).

## Relevance to This Project
Surfaced for root-cause link 4: our lambda dial going inert on tie-dominated documents IS preference-insensitivity, invisible to aggregate metrics. Prescribes our acceptance test: report measured sensitivity of realized behavior to lambda per seed and cycle (our synchronous profile snapshot), for ANY candidate fix - a fix can stabilize training while leaving lambda inert.
