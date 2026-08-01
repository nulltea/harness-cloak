---
type: paper
node_id: paper:wray2015_lexicographic_mdp_slack
title: "Multi-Objective MDPs with Conditional Lexicographic Reward Preferences"
authors: ["Kyle Hollins Wray", "Shlomo Zilberstein", "Abdel-Illah Mouaddib"]
year: 2015
venue: "AAAI 2015"
external_ids:
  arxiv: null
  doi: "10.1609/aaai.v29i1.9647"
  s2: null
tags: ["morl", "lexicographic", "slack", "scalarization-impossibility", "state-dependent-preferences"]
added: 2026-08-01T00:00:00Z
---

# Multi-Objective MDPs with Conditional Lexicographic Reward Preferences

## One-line thesis
Defines the Lexicographic MDP: at each state the action set for objective i is restricted to those within a slack of objective i−1's optimum, and proves that the resulting optimal policy **may not exist anywhere in the solution space of the linearly scalarized MOMDP** — lexicographic-with-slack is a genuinely distinct optimization problem, not a scalarization with a hard-to-find weight.

## Why surfaced
Surfaced in the additive-vs-lexicographic adjudication (2026-08-01) as the missing formal statement behind the composition question. Skalse 2022 gives the τ-slack rule and its worst-case cost; Tercan 2024 gives the warning against a global threshold. Neither states the *impossibility*: that no choice of scalarization weight reproduces the lexicographic solution. Wray et al. do, constructively.

## Key Results
- **Proposition 6 (the load-bearing one).** The optimal policy of an LMDP may not exist in the space of solutions captured by its corresponding scalarized MOMDP's policies π_w. The proof constructs a MOMDP whose states impose *conflicting weight requirements* — no single w satisfies all of them at once. This is a per-state-conflict impossibility, not the usual convex-hull/concave-front argument.
- **Slack is per-objective and the ordering is state-conditioned.** The LMDP tuple carries δ = ⟨δ_1,…,δ_k⟩ (slack per objective) and a partition of states, each with its own objective ordering — so both the ordering and the tolerance are allowed to vary by state, which is the design Tercan 2024 argues is required.
- **Proposition 1 — the slack-to-loss conversion.** Per-decision deviation η_i and total value loss δ_i are related by η_i = (1−γ)δ_i: allowing each state to pick an action η_i-suboptimal for the primary bounds the primary's total value loss by δ_i. This is the inverse of the Skalse bound (τ/(1−γ)) — it says how to *set* the per-decision slack from a total primary-loss budget, which is the direction a practitioner actually needs. It is an explicitly worst-case accounting ("each state selects an action as far from optimal as it can") and the authors note it can be relaxed in practice.
- **Convergence (Proposition 2) holds only under a partition-dependent assumption** — the restricted-action Bellman operator B_i is a contraction, but LVI's convergence needs an extra assumption about the state partition. Consistent with Tercan's finding that thresholded-lexicographic value methods lack a general Bellman theory.
- Runtime cost of LVI over weighted VI is a small constant factor — the filter itself is not the expensive part.

## Relevance to This Project
This is the paper that answers whether our additive controller `z(a) = u(a) + α·g(λ)·p̂(a)` is *misspecified* rather than *mistuned*. Proposition 6 says yes: the preference we actually hold — "among actions that preserve every QA-probed answer, prefer the higher anonymity count; never trade a preserved answer for count" — is a lexicographic-with-slack preference, and there is no α that represents it. Our five failed scale-control rounds were searching a space that provably does not contain the target. The conflicting-weight construction is our situation exactly: at an exact tie the required α is unbounded (any finite α can be out-argued by a large enough arbitrary tower margin), while at the utility cliff the required α is small and finite (or the policy buys count with real answer loss). One global scalar must be both.

The slack-to-loss conversion is the honest-cost statement to preregister alongside Skalse's. In our terms, with the per-decision instrument floor ε = 0.044, the worst case is *every* filtered decision spending its whole 0.044 in the same direction; the aggregate primary loss is then bounded by the accumulation of those per-decision slacks, not by 0.044 itself. Our episodes are short and undiscounted-per-document rather than γ-discounted, so the accounting to preregister is per-document (number of filtered decisions × 0.044 as the loose bound), and the measured aggregate document utility drop is the quantity that must be reported against it.

Design question it bears on: composition of the utility head and the count controller — additive scalarization versus an explicit ε-optimal filter, and whether the slack tolerance may be global (it may not; Wray makes it per-objective and state-conditioned, agreeing with [tercan2024_thresholded_lexicographic](tercan2024_thresholded_lexicographic.md)).

## Related pages
- [skalse2022_lexicographic_morl](skalse2022_lexicographic_morl.md) — the RL-side τ-slack rule and the τ/(1−γ) worst-case bound; Wray is the planning-side origin with the scalarization-impossibility proof.
- [tercan2024_thresholded_lexicographic](tercan2024_thresholded_lexicographic.md) — why a single global threshold is wrong.
- [calvofullana2021_state_augmented_constrained_rl](calvofullana2021_state_augmented_constrained_rl.md) — the constrained-RL analogue of Proposition 6 (no fixed weighting induces the constrained optimum).

## Sources
- AAAI Proceedings 29(1):3418–3424, [DOI 10.1609/aaai.v29i1.9647](https://doi.org/10.1609/aaai.v29i1.9647)
