---
type: paper
node_id: paper:mandal2023_performative_rl
title: "Performative Reinforcement Learning"
authors: ["Debmalya Mandal", "Stelios Triantafyllou", "Goran Radanovic"]
year: 2023
venue: "ICML 2023"
external_ids:
  arxiv: "2207.00046"
  doi: null
  s2: null
tags: ["performativity", "feedback-loops", "stability"]
added: 2026-07-31T00:00:00Z
---

# Performative Reinforcement Learning

## One-line thesis
When the deployed policy shifts the distribution its own measurements come from, convergence to a performatively stable point requires sufficiently slow/regularized updates.

## Key Results
- The formal name for control consuming its own measurement stream; the theory behind freezing/lagging probe-derived statistics.

## Relevance to This Project
Round-3 sweep: justifies the one-cycle application lag and stop-gradient, slowly-updated form of the tie bonus (the probes' distribution depends on the policy the bonus steers).
