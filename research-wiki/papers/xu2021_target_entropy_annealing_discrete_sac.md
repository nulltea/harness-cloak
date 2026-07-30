---
type: paper
node_id: paper:xu2021_target_entropy_annealing_discrete_sac
title: "Target Entropy Annealing for Discrete Soft Actor-Critic"
authors: ["Yaosheng Xu", "Dailin Hu", "Litian Liang", "et al."]
year: 2021
venue: "NeurIPS 2021 DeepRL WS / arXiv"
external_ids:
  arxiv: "2112.02852"
  doi: null
  s2: null
tags: ["sac", "target-entropy", "discrete-actions", "saturation"]
added: 2026-07-30T00:00:00Z
---

# Target Entropy Annealing for Discrete Soft Actor-Critic

## One-line thesis
In discrete SAC a low constant target entropy crashes the temperature early, the actor learns logits of a deterministic greedy policy, and because deterministic-policy logits saturate, the algorithm is prone to early overfitting that is difficult to unlearn.

## Key Results
- Target entropy is a sensitive, task-dependent constant; the standard 98%-of-max heuristic is often unsatisfiable.
- The entropy mechanism itself can CAUSE logit saturation when the target is low.

## Relevance to This Project
The most damaging evidence against the entropy-floor version of the margin-bounding fix in our discrete-menu setting: it trades our hand-set beta for an equally hand-set, saturation-hazardous target — motivating logit soft-capping as the family representative instead.
