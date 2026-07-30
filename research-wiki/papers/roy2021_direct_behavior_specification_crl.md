---
type: paper
node_id: paper:roy2021_direct_behavior_specification_crl
title: "Direct Behavior Specification via Constrained Reinforcement Learning"
authors: ["Julien Roy", "Roger Girgis", "Joshua Romoff", "Pierre-Luc Bacon", "Christopher Pal"]
year: 2021
venue: "arXiv/ICML 2022"
external_ids:
  arxiv: "2112.12228"
  doi: null
  s2: null
tags: ["constrained-rl", "multiplier-divergence", "normalization"]
added: 2026-07-30T00:00:00Z
---

# Direct Behavior Specification via Constrained Reinforcement Learning

## One-line thesis
A hard-to-satisfy constraint drives the learned multiplier to endlessly increasing magnitude, destabilizing policy updates until the critic diverges and performance collapses; mitigated by multiplier normalization.

## Key Results
- Documented upward-divergence failure of learned multipliers (the attested failure direction; no gaming/collapse-to-zero reported anywhere).
- Multiplier normalization as the fix.

## Relevance to This Project
The decisive negative result for the learned-gain option: on our reward-tied actions where utility resists indefinitely, an uncapped learned alpha could climb without bound — softplus bounds below, not above. A ceiling/normalization on the gain is a precondition of shipping it, not an optimization.
