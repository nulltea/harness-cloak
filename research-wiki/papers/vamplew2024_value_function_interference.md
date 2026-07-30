---
type: paper
node_id: paper:vamplew2024_value_function_interference
title: "Issues with Value-Based Multi-objective Reinforcement Learning: Value Function Interference and Overestimation Sensitivity"
authors: ["Peter Vamplew", "Cameron Foale", "Richard Dazeley"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2402.06266"
  doi: null
  s2: null
tags: ["morl", "value-interference", "tie-breaking", "scalarization"]
added: 2026-07-30T00:00:00Z
---

# Issues with Value-Based Multi-objective Reinforcement Learning: Value Function Interference and Overestimation Sensitivity

## One-line thesis
When a scalarizing utility maps widely varying value-vectors to near-identical utility, the learned value function suffers interference and converges to sub-optimal policies; random tie-breaking among equally optimal actions is itself an instability source, and deterministic tie-breaking measurably mitigates it.

## Key Results
- Value-function interference demonstrated in value-based MORL when the utility function is many-to-one over value-vectors.
- Random tie-breaking between reward-equal actions destabilizes learning; deterministic tie-breaking alleviates it.

## Relevance to This Project
Surfaced while categorizing the lambda-3 coin-flip root cause (decision log 2026-07-30). Closest published account of our links 1->2->4 as one causal chain, from a scalarized-multi-objective setting like ours: the reward-tie set (U(L0)==U(keep) exactly) is the locus of the pathology, not a benign edge case, and whatever resolves ties should be deterministic and stated - supporting bounded-margin + calibrated-controller tie-breaking or an explicit learned owner, never drift.
