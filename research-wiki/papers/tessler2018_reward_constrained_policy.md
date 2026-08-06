---
type: paper
node_id: paper:tessler2018_reward_constrained_policy
title: "Reward Constrained Policy Optimization"
authors: ["Chen Tessler", "Daniel J. Mankowitz", "Shie Mannor"]
year: 2018
venue: "arXiv"
external_ids:
  arxiv: "1805.11074"
  doi: null
  s2: null
tags: ["constrained-rl", "primal-dual", "multi-timescale"]
added: 2026-08-05T10:46:47Z
---

# Reward Constrained Policy Optimization

## One-line thesis
RCPO uses multi-timescale primal-dual policy optimization to learn constraint-satisfying policies from a separate penalty signal.

## Problem / Gap
Many useful behavioral constraints are not differentiable with respect to policy parameters, so their gradients cannot be inserted directly into ordinary policy optimization.

## Method
RCPO introduces a discounted penalty signal and a Lagrange multiplier trained on a slower timescale than the actor and critic. The actor optimizes reward minus the learned penalty weight, while dual updates increase pressure when the expected constraint is violated.

## Key Results
- Proves convergence under stochastic-approximation and multi-timescale assumptions.
- Empirically learns constraint-satisfying policies from complex penalty signals.

## Assumptions
- Actor, critic, and multiplier updates operate on separated learning timescales.
- The penalty return is a valid estimator of the specified constraint.
- The constrained problem is feasible within the represented policy class.

## Limitations / Failure Modes
- Dual dynamics can oscillate or become slow when primal and dual timescales are poorly separated.
- A global multiplier enforces an expectation and can hide state- or document-specific violations.
- The learned scalar still combines objective gradients and therefore depends on accurate advantage estimates.

## Reusable Ingredients
- Multi-timescale optimization rather than ad hoc gradient detachment.
- A distinct constraint critic and dual update driven by measured violation.
- A direct route for secondary-objective gradients into the actor without contaminating the primary critic.

## Open Questions
- Does a state-conditioned multiplier generalize utility constraints to held-out documents?
- Can exact count expectations be combined with sampled utility constraints without destabilizing dual learning?

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
RCPO supplies the principled version of the gradient-path unification that ranker v2 lacks: count and utility-constraint gradients meet in the actor, while the utility estimator remains utility-only. It does not justify the current global `alpha`, whose update is not a conventional constraint violation signal.

## Abstract (original)

> Solving tasks in Reinforcement Learning is no easy feat. As the goal of the agent is to maximize the accumulated reward, it often learns to exploit loopholes and misspecifications in the reward signal resulting in unwanted behavior. While constraints may solve this issue, there is no closed form solution for general constraints. In this work we present a novel multi-timescale approach for constrained policy optimization, called `Reward Constrained Policy Optimization' (RCPO), which uses an alternative penalty signal to guide the policy towards a constraint satisfying one. We prove the convergence of our approach and provide empirical evidence of its ability to train constraint satisfying policies.
