---
type: paper
node_id: paper:achiam2017_constrained_policy_optimization
title: "Constrained Policy Optimization"
authors: ["Joshua Achiam", "David Held", "Aviv Tamar", "Pieter Abbeel"]
year: 2017
venue: "arXiv"
external_ids:
  arxiv: "1705.10528"
  doi: null
  s2: null
tags: ["constrained-rl", "trust-region", "policy-optimization"]
added: 2026-08-05T10:46:46Z
---

# Constrained Policy Optimization

## One-line thesis
CPO performs trust-region policy updates that approximately maximize reward while enforcing near constraint satisfaction at each iteration.

## Problem / Gap
Deep policy-search methods optimized reward without a practical mechanism for maintaining expected-cost constraints during learning. Earlier constrained methods either lacked per-update guarantees or imposed assumptions that did not scale to neural policies.

## Method
CPO approximately solves a trust-region policy update: maximize a first-order reward surrogate subject to first-order expected-cost constraints and a quadratic KL-divergence bound. It solves the low-dimensional dual, reconstructs the natural-gradient update, and uses backtracking line search plus a recovery direction when the local constrained problem is infeasible.

## Key Results
- Derives a bound connecting policy-return changes to average policy divergence.
- Provides near-constraint-satisfaction guarantees for each policy update under the paper's approximations.
- Demonstrates neural constrained policy search on simulated locomotion tasks.

## Assumptions
- Constraints are expectations of auxiliary cumulative costs in a CMDP.
- The trust-region surrogate and sampled gradients adequately approximate the true policy update.
- Practical guarantees inherit error from finite samples, function approximation, and line search.

## Limitations / Failure Modes
- Guarantees expected constraint satisfaction, not a separate guarantee for every document or state.
- The second-order Fisher-vector products and constrained line search add implementation cost.
- An exact zero-slack boundary is vulnerable to sampling and approximation error; the practical method tightens constraints for safety.

## Reusable Ingredients
- Separate reward and cost critics rather than combining raw reward scales.
- Trust-region updates that bound behavioral change while enforcing a utility constraint.
- Explicit recovery updates when the current local approximation is infeasible.

## Open Questions
- Can a document-conditional constraint be enforced without learning one multiplier per training document?
- How conservative must a utility-retention constraint be under a quantized, cached reward?

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
CPO is the strongest established alternative to additive utility-plus-count logits: optimize count through the actor while constraining measured document utility. Its expected-CMDP guarantee does not by itself meet HarnessCloak's per-document utility requirement.

## Abstract (original)

> For many applications of reinforcement learning it can be more convenient to specify both a reward function and constraints, rather than trying to design behavior through the reward function. For example, systems that physically interact with or around humans should satisfy safety constraints. Recent advances in policy search algorithms (Mnih et al., 2016, Schulman et al., 2015, Lillicrap et al., 2016, Levine et al., 2016) have enabled new capabilities in high-dimensional control, but do not consider the constrained setting. We propose Constrained Policy Optimization (CPO), the first general-purpose policy search algorithm for constrained reinforcement learning with guarantees for near-constraint satisfaction at each iteration. Our method allows us to train neural network policies for high-dimensional control while making guarantees about policy behavior all throughout training. Our guarantees are based on a new theoretical result, which is of independent interest: we prove a bound relating the expected returns of two policies to an average divergence between them. We demonstrate the effectiveness of our approach on simulated robot locomotion tasks where the agent must satisfy constraints motivated by safety.
