---
type: paper
node_id: paper:calvofullana2021_state_augmented_constrained_rl
title: "State Augmented Constrained Reinforcement Learning: Overcoming the Limitations of Learning with Rewards"
authors: ["Miguel Calvo-Fullana", "Santiago Paternain", "Luiz F. O. Chamon", "Alejandro Ribeiro"]
year: 2021
venue: "IEEE TAC / arXiv"
external_ids:
  arxiv: "2102.11941"
  doi: null
  s2: null
tags: ["constrained-rl", "lagrangian", "state-augmentation", "multiplier"]
added: 2026-07-30T00:00:00Z
---

# State Augmented Constrained Reinforcement Learning: Overcoming the Limitations of Learning with Rewards

## One-line thesis
The optimal constrained policy provably cannot be induced by ANY fixed weighted linear combination of rewards; multipliers must become state variables the policy conditions on, with dual dynamics running at execution.

## Key Results
- Impossibility result for fixed scalar weightings in constrained RL.
- State-augmented formulation: condition the policy on the multiplier and keep it moving.

## Relevance to This Project
Surfaced as corroboration for the learned state-conditioned controller gain: the strongest theoretical case that our single global alpha is structurally insufficient for per-decision privacy/utility trade-offs — exactly the conditional-negative-transfer argument, proved rather than argued.
