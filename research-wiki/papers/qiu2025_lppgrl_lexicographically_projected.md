---
type: paper
node_id: paper:qiu2025_lppgrl_lexicographically_projected
title: "LPPG-RL: Lexicographically Projected Policy Gradient Reinforcement Learning with Subproblem Exploration"
authors: ["Ruiyu Qiu", "Rui Wang", "Guanghui Yang", "Xiang Li", "Zhijiang Shao"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2511.08339"
  doi: null
  s2: null
tags: ["lexicographic", "morl", "policy-gradient", "projection"]
added: 2026-08-05T10:46:47Z
---

# LPPG-RL: Lexicographically Projected Policy Gradient Reinforcement Learning with Subproblem Exploration

## One-line thesis
LPPG-RL projects lower-priority policy gradients into directions that preserve higher-priority objectives, with convergence claims in continuous lexicographic MORL.

## Problem / Gap
Prior lexicographic RL methods either depend on heuristic thresholds, are restricted to discrete action spaces, or combine gradients without efficiently enforcing the stated priority order.

## Method
LPPG-RL sequentially projects each lower-priority policy gradient into directions compatible with higher-priority objectives. It formulates projection as a convex feasibility problem, accelerates it with Dykstra's algorithm, and adds subproblem exploration to counter gradient vanishing.

## Key Results
- Provides convergence claims and a lower bound on policy improvement under the paper's conditions.
- Reports better performance than compared continuous lexicographic MORL methods in a 2D navigation environment.

## Assumptions
- Objective policy gradients and the projection constraints are estimated accurately enough to identify feasible directions.
- The local projected direction predicts finite-step objective changes.
- The tested optimization geometry transfers beyond the small continuous-control benchmark.

## Limitations / Failure Modes
- Evidence is currently a 2025 preprint with experiments concentrated in a 2D navigation domain.
- Projection protects objectives only through local gradient geometry; noisy or zero primary gradients weaken the constraint.
- It does not solve incorrect utility ordering or missing cardinal utility measurements.

## Reusable Ingredients
- Sequential priority-aware gradient projection.
- Explicit infeasibility detection when no lower-priority improvement direction preserves higher priorities.
- Subproblem exploration for vanishing-gradient regimes.

## Open Questions
- Does projection remain stable with document-level, quantized, high-variance utility returns?
- How should a projected update be validated against complete-document utility rather than its local surrogate?

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
LPPG-RL is a trainable lexicographic alternative that avoids a deterministic deployment selector. It is a higher-risk fallback because ranker v2's utility gradient is exactly zero on ties and noisy elsewhere, making the projection geometry less trustworthy than a measured constraint.

## Abstract (original)

> Lexicographic multi-objective problems, which consist of multiple conflicting subtasks with explicit priorities, are common in real-world applications. Despite the advantages of Reinforcement Learning (RL) in single tasks, extending conventional RL methods to prioritized multiple objectives remains challenging. In particular, traditional Safe RL and Multi-Objective RL (MORL) methods have difficulty enforcing priority orderings efficiently. Therefore, Lexicographic Multi-Objective RL (LMORL) methods have been developed to address these challenges. However, existing LMORL methods either rely on heuristic threshold tuning with prior knowledge or are restricted to discrete domains. To overcome these limitations, we propose Lexicographically Projected Policy Gradient RL (LPPG-RL), a novel LMORL framework which leverages sequential gradient projections to identify feasible policy update directions, thereby enabling LPPG-RL broadly compatible with all policy gradient algorithms in continuous spaces. LPPG-RL reformulates the projection step as an optimization problem, and utilizes Dykstra's projection rather than generic solvers to deliver great speedups, especially for small- to medium-scale instances. In addition, LPPG-RL introduces Subproblem Exploration (SE) to prevent gradient vanishing, accelerate convergence and enhance stability. We provide theoretical guarantees for convergence and establish a lower bound on policy improvement. Finally, through extensive experiments in a 2D navigation environment, we demonstrate the effectiveness of LPPG-RL, showing that it outperforms existing state-of-the-art continuous LMORL methods.
