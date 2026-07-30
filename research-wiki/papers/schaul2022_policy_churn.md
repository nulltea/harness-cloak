---
type: paper
node_id: paper:schaul2022_policy_churn
title: "The Phenomenon of Policy Churn"
authors: ["Tom Schaul", "Andre Barreto", "John Quan", "Georg Ostrovski"]
year: 2022
venue: "arXiv/NeurIPS 2022"
external_ids:
  arxiv: "2206.00730"
  doi: null
  s2: null
tags: ["policy-churn", "shared-parameters", "deep-rl", "instability"]
added: 2026-07-30T00:00:00Z
---

# The Phenomenon of Policy Churn

## One-line thesis
In deep value-based RL the greedy action flips in a large fraction of states within a handful of updates, driven by shared-network updates from unrelated states.

## Key Results
- Measured churn rates: greedy-action flips pervasive and rapid, mostly caused by updates on other states.

## Relevance to This Project
Surfaced for root-cause link 2 cross-document channel: D2N005 keep-vs-L0 margins moving after OTHER documents optimizer steps is policy churn through the shared tower - the same phenomenon in policy-gradient form. Motivates churn diagnostics (same-checkpoint greedy-action flips on untouched documents) in any adjudication.
