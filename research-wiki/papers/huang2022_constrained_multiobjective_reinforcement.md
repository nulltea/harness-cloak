---
type: paper
node_id: paper:huang2022_constrained_multiobjective_reinforcement
title: "A Constrained Multi-Objective Reinforcement Learning Framework"
authors: ["Sandy Huang", "Abbas Abdolmaleki", "Giulia Vezzani", "Philemon Brakel", "Daniel J. Mankowitz", "Michael Neunert", "Steven Bohez", "Yuval Tassa", "Nicolas Heess", "Martin Riedmiller", "Raia Hadsell"]
year: 2022
venue: "CoRL 2021 / PMLR 164"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["constrained-rl", "morl", "preference-learning", "lp3"]
added: 2026-08-05T10:46:47Z
---

# A Constrained Multi-Objective Reinforcement Learning Framework

## One-line thesis
LP3 jointly learns policy preferences and policies, subsuming Lagrangian relaxation and enabling distributions over constraint-satisfying operating points.

## Problem / Gap
Constrained RL commonly learns one linear-scalarization weight, which can be scale-sensitive, sample-inefficient, and unable to recover useful policies on concave Pareto fronts or under uncertain constraint thresholds.

## Method
LP3 separates policy learning from preference learning. One module trains policies for objective preferences using a chosen MORL method; another updates either one preference or a preference distribution according to constraint satisfaction. The paper instantiates this with MO-MPO and with a distribution over preference settings.

## Key Results
- Unifies Lagrangian relaxation as linear scalarization plus one learned preference.
- Learns multiple constraint-satisfying policies in one run when the desired threshold is not fixed in advance.
- Reports improved sample efficiency and coverage over its linear-scalarization baseline on robotic-control tasks.

## Assumptions
- Constraint satisfaction can be scored for candidate preference settings.
- The underlying MORL optimizer can train a policy for the sampled preferences.
- A distribution over preferences is acceptable as the learned solution concept.

## Limitations / Failure Modes
- It does not impose strict lexicographic utility priority.
- Learned preference search adds another optimization loop and more policy-evaluation cost.
- Constraint satisfaction is still evaluated in expectation rather than guaranteed per document.

## Reusable Ingredients
- Separate the question "which preference is feasible?" from "how is a policy improved for that preference?"
- Train one preference-conditioned policy over several operating points.
- Use non-linear/scale-invariant MORL inside a constrained outer loop.

## Open Questions
- Can a finite utility-budget menu replace continuous preference search without losing controllability?
- How much document diversity is needed before one conditioned policy generalizes its operating points?

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
LP3 is relevant to the original requirement for one ranker that supports several user-selectable privacy/utility settings. It is better treated as a second-stage extension after one constrained operating point works, not as the first repair for the current training pathology.
