---
type: paper
node_id: paper:mason2020_all_epsilon_good_arms
title: "Finding All epsilon-Good Arms in Stochastic Bandits"
authors: ["Blake Mason", "Lalit Jain", "Ardhendu Tripathy", "Robert Nowak"]
year: 2020
venue: "NeurIPS 2020"
external_ids:
  arxiv: "2006.08850"
  doi: null
  s2: null
tags: ["bandits", "pure-exploration", "epsilon-good-set", "confidence-bounds"]
added: 2026-07-31T00:00:00Z
---

# Finding All epsilon-Good Arms in Stochastic Bandits

## One-line thesis
Formalizes returning ALL arms within epsilon of the best — a measured indifference set with confidence guarantees — via UCB/LUCB-hybrid and instance-optimal algorithms maintaining explicit good/bad sets across rounds.

## Key Results
- The statistics of the object our rule maintains: membership by confidence interval, never point estimate.

## Relevance to This Project
Round-3 sweep: prescribes the CI-based membership test for the measured utility-equivalence sets (a pair enters only when the bound on |Delta-U| clears the threshold).
