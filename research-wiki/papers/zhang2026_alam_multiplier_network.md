---
type: paper
node_id: paper:zhang2026_alam_multiplier_network
title: "ALaM: Augmented Lagrangian Multiplier Network for State-wise Safety in RL"
authors: ["Zhang et al."]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2605.00667"
  doi: null
  s2: null
tags: ["constrained-rl", "multiplier-network", "oscillation", "state-wise-safety"]
added: 2026-07-30T00:00:00Z
---

# ALaM: Augmented Lagrangian Multiplier Network for State-wise Safety in RL

## One-line thesis
Per-state Lagrange multiplier networks trained by standard dual ascent oscillate severely — network generalization propagates local overshoots to adjacent states, and scalar-multiplier stabilizers (PID) do not transfer; fixed by a quadratic augmented-Lagrangian penalty plus supervised regression to a dual target.

## Key Results
- SAC-ALaM beats safe-RL baselines on safety AND return once stabilized.
- Learned per-state multipliers end up well-calibrated for risk identification (auditability asset).

## Relevance to This Project
Closest published architecture to the proposed learned controller gain, carrying its sharpest warning: going from scalar to state-conditioned is not a free generalization; oscillation dynamics change class. Our gain is trained by ordinary gradients (not dual ascent), which likely sidesteps this — but the alpha-field smoothness check it motivates goes into the spike gates.
