---
type: paper
node_id: paper:cui2025_entropy_mechanism_rl
title: "The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models"
authors: ["Ganqu Cui", "et al."]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2505.22617"
  doi: null
  s2: null
tags: ["entropy-collapse", "policy-gradient", "saturation", "logit-drift"]
added: 2026-07-30T00:00:00Z
---

# The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models

## One-line thesis
Policy-entropy change is governed by the covariance between action log-probability and logit change (proportional to advantage), so consistent advantages drive monotone entropy decay and unbounded margin growth; a fixed entropy bonus is structurally outrun, and the remedy is clamping the covariance source (Clip-Cov / KL-Cov).

## Key Results
- Derived covariance law for entropy dynamics under policy-gradient-family updates; fitted H->0 decay with a performance ceiling.
- Fixed-magnitude entropy bonuses die under saturation by construction (they compete with confidence-growing drift).
- Clip-Cov/KL-Cov bound the drift source rather than rewarding entropy.

## Relevance to This Project
Surfaced for root-cause link 3 (logit ranges 9->300, beta=0.01 entropy bonus dead). Gives the mechanistic theory for WHY our entropy bonus failed (not mis-tuned - outrun) and motivates floor/clamp-style interventions (entropy floor, covariance clipping) over bonus retuning; the bounded additive controller regains authority exactly when margins are bounded.
