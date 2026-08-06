---
type: paper
node_id: paper:abdolmaleki2020_distributional_view_multiobjective
title: "A Distributional View on Multi-Objective Policy Optimization"
authors: ["Abbas Abdolmaleki", "Sandy H. Huang", "Leonard Hasenclever", "Michael Neunert", "H. Francis Song", "Martina Zambelli", "Murilo F. Martins", "Nicolas Heess", "Raia Hadsell", "Martin Riedmiller"]
year: 2020
venue: "arXiv"
external_ids:
  arxiv: "2005.07513"
  doi: null
  s2: null
tags: ["morl", "scale-invariance", "policy-distillation", "preference"]
added: 2026-08-05T10:46:47Z
---

# A Distributional View on Multi-Objective Policy Optimization

## One-line thesis
MO-MPO learns an improved action distribution per objective and distills them into one policy using scale-invariant KL preference budgets.

## Problem / Gap
Weighted reward sums are fragile when objectives use different units or scales, and a single scalarized policy update can prematurely collapse toward the numerically dominant objective.

## Method
MO-MPO trains one action-value critic and one nonparametric improved action distribution per objective. Objective preferences are encoded as per-objective KL budgets, and a shared parametric policy is fitted to the objective-specific distributions by supervised KL minimization under an additional policy trust region.

## Key Results
- Preference settings are invariant to positive rescaling of objective rewards.
- Recovers nondominated policies on benchmark and high-dimensional robotic-control tasks.
- Shows greater robustness to reward-scale imbalance than linear scalarization.

## Assumptions
- Each objective has a learnable action-value function evaluated under the shared policy.
- Sampled nonparametric action distributions adequately represent useful policy improvements.
- Relative KL budgets are a meaningful user interface for objective preferences.

## Limitations / Failure Modes
- It trades objectives rather than enforcing strict lexicographic priority.
- It is substantially more machinery than ranker v2's discrete action policy.
- Preference KL budgets still require selection and do not guarantee per-document utility retention.

## Reusable Ingredients
- Separate objective critics and objective-specific policy-improvement distributions.
- Scale-invariant preference control through KL budgets.
- Distillation of multiple objective updates into one deployable policy.

## Open Questions
- Can exact discrete count targets replace one learned objective critic without breaking the MO-MPO derivation?
- Would KL-budget conditioning remain controllable on ranker v2's small document corpus?

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
MO-MPO is the strongest literature-backed alternative if HarnessCloak wants a smooth, finite preference menu rather than strict utility-first behavior. It addresses cross-objective scale directly but does not provide the required zero-utility-loss semantics.

## Abstract (original)

> Many real-world problems require trading off multiple competing objectives. However, these objectives are often in different units and/or scales, which can make it challenging for practitioners to express numerical preferences over objectives in their native units. In this paper we propose a novel algorithm for multi-objective reinforcement learning that enables setting desired preferences for objectives in a scale-invariant way. We propose to learn an action distribution for each objective, and we use supervised learning to fit a parametric policy to a combination of these distributions. We demonstrate the effectiveness of our approach on challenging high-dimensional real and simulated robotics tasks, and show that setting different preferences in our framework allows us to trace out the space of nondominated solutions.
