---
type: paper
node_id: paper:lin2021_random_weighting_multitask
title: "Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning"
authors: ["Baijiong Lin", "Feiyang Ye", "Yu Zhang", "Ivor Tsang"]
year: 2021
venue: "TMLR / arXiv"
external_ids:
  arxiv: "2111.10603"
  doi: null
  s2: null
tags: ["multi-task", "loss-weighting", "ablation-discipline", "negative-result"]
added: 2026-07-30T00:00:00Z
---

# Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning

## One-line thesis
Twelve state-of-the-art dynamic loss/gradient weighting methods are matched by randomly sampled weights across five image datasets and two XTREME tasks.

## Key Results
- Learned/dynamic weighting often contributes nothing over random weighting.

## Relevance to This Project
Mandatory-control discipline for any learned-gain adoption: random-alpha and fixed-rule-alpha ablations must run before claiming the LEARNING (not just the per-decision variation) is doing the work — directly per this paper's litmus test.
