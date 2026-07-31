---
type: paper
node_id: paper:peri2025_nonconflicting_energy_minimization
title: "Non-conflicting Energy Minimization in Reinforcement Learning based Robot Control"
authors: ["Skand Peri", "et al."]
year: 2025
venue: "CoRL 2025 (oral)"
external_ids:
  arxiv: "2509.01765"
  doi: null
  s2: null
tags: ["gradient-projection", "secondary-objective", "hyperparameter-free"]
added: 2026-07-31T00:00:00Z
---

# Non-conflicting Energy Minimization in Reinforcement Learning based Robot Control

## One-line thesis
Applies a secondary objective (energy) only along policy-gradient directions that do not impact the primary task — hyperparameter-free projection; no thresholds, sets, or probe budgets.

## Key Results
- The rival design to the measured-tie-set mechanism: same intent, realized in gradient space.

## Relevance to This Project
Round-3 sweep: the design our fork entry must answer. Our written answer: projection needs a primary gradient to project against — on EXACT ties the utility gradient is identically zero, so projection passes the full undifferentiated count gradient and reproduces the v13 global-saturation dynamics; it cannot localize per-decision exact ties whose correct resolution is already KNOWN from measurement.
