---
type: paper
node_id: paper:lin2026_tepo_token_level_policy_optimization
title: "Token-Level Policy Optimization: Linking Group-Level Rewards to Token-Level Aggregation via Sequence-Level Likelihood"
authors: ["Xingyu Lin", "Yilin Wen", "Du Su", "Jinchang Hou", "et al."]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2604.12736"
  doi: null
  s2: null
tags: ["rl", "kl-masking", "selective-regularization", "token-level"]
added: 2026-07-30T00:00:00Z
---

# Token-Level Policy Optimization: Linking Group-Level Rewards to Token-Level Aggregation via Sequence-Level Likelihood

## One-line thesis
Applies its KL regularization selectively - only to tokens with positive advantage and decreasing entropy - to avoid the undifferentiated-regularization failure mode.

## Key Results
- Precedent for gating a KL anchor on per-decision runtime statistics (advantage sign, entropy trend) instead of a global constant.

## Relevance to This Project
Surfaced for the anchor-selectivity family: the literature-closest precedent to anchoring only where the reward is silent. Our gap analysis found nobody regularizes specifically ON the reward-indifference set - TEPO runtime-statistic gating is the mechanism template if we build KL-on-ties.
