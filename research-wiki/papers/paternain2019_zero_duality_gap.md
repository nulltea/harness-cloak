---
type: paper
node_id: paper:paternain2019_zero_duality_gap
title: "Constrained Reinforcement Learning Has Zero Duality Gap"
authors: ["Santiago Paternain", "Luiz F. O. Chamon", "Miguel Calvo-Fullana", "Alejandro Ribeiro"]
year: 2019
venue: "NeurIPS 2019"
external_ids:
  arxiv: "1910.13393"
  doi: null
  s2: null
tags: ["constrained-rl", "lagrangian", "duality", "scalarization-equivalence", "shadow-price"]
added: 2026-08-01T00:00:00Z
---

# Constrained Reinforcement Learning Has Zero Duality Gap

## One-line thesis
Despite non-convexity, constrained RL has zero duality gap under Slater's condition, so the constrained problem and a *weighted-reward* problem trace the same Pareto front — the weights being the optimal Lagrange multipliers; but the map from constraint level to weight is non-trivial, unknown a priori, and the weight is a shadow price that is zero exactly when the constraint is slack.

## Why surfaced
Surfaced to answer the direct question in the additive-vs-lexicographic adjudication (2026-08-01): *when does a Lagrangian multiplier reduce to exactly our global α, and what changes when the constraint is active versus slack?* This is the paper that legitimizes the equivalence — and, read carefully, the paper that shows why our single calibrated α is not the multiplier the equivalence refers to.

## Key Results
- **Theorem 1.** If the rewards r_i are bounded and Slater's condition holds, strong duality holds for the constrained RL problem: P* = D*. Proof route: the perturbation function P(ξ) = max{V_0(π) : V_i(π) ≥ c_i + ξ_i} is *concave* over the space of all (occupancy-measure / randomized) policies, and Fenchel–Moreau plus Slater gives zero gap. Concavity comes from mixing two policies — the argument is over randomized policies, not deterministic ones.
- **The scalarization equivalence, stated by the authors themselves.** "The trade-offs expressed by the w_i … are the same as those expressed by the specifications c_i in the sense that they trace the same Pareto front. Nevertheless … the relationship between c_i and w_i is not trivial and … specifying the constrained problem is often considerably simpler." So a weighted sum *can* reach the constrained optimum — at the right weight, which is the optimal dual variable λ*, an unknown of the problem.
- **Theorem 2 — parametrization is nearly free.** With an ε-universal policy parametrization (e.g. a neural net), the parametrized dual value satisfies P* ≥ D*_θ ≥ P* − (B_{r0} + ‖λ*‖_1 B_r)ε/(1−γ). The suboptimality scales with ‖λ*‖_1: **a large optimal multiplier amplifies every approximation error in the policy head.**
- **Theorem 3.** Dual ascent converges in a bounded number of steps, with the bound scaling in ‖λ_0 − λ*_θ‖² — i.e. the multiplier must be *found*, and how long that takes depends on how far the initialization is from the true shadow price.
- The maximizer of the Lagrangian at λ* need not be unique; strong duality lives in the space of all policies, where mixtures are available.
- The introduction's own framing: hand-designed composite/weighted rewards are the practice this work is meant to replace — duality *derives* the weights rather than having a practitioner tune them.

## Relevance to This Project
Our controller is literally a Lagrangian in disguise. Write the intended problem as: maximize count p̂ subject to utility u ≥ u* − ε. Its Lagrangian is p̂ + λ·(u − u* + ε); dividing by λ (λ > 0) gives u + (1/λ)·p̂ — our `z(a) = u(a) + α·g(λ_dial)·p̂(a)` with **α = 1/λ**, i.e. our global gain is a reciprocal multiplier. Paternain's theorem says this form is not *wrong in principle*: at α = 1/λ* the scalarized and constrained problems coincide. That is the strongest available defence of additive scalarization, and it must be stated before the refutation.

The refutation is what happens to λ* under our saturating primary. By complementary slackness λ* = 0 whenever the constraint is slack — and on a reward tie the constraint is maximally slack (every action, including the highest-count one, preserves every probed answer, so u = u* and the ε band is untouched). λ* = 0 means α = 1/λ* is **unbounded**: the correct composition on a tie is pure count, no utility term at all. At the utility cliff, where a coarser level breaks a probed answer, the constraint binds and λ* is strictly positive and finite, so α is finite and small. A single global α is therefore required to be simultaneously infinite and finite, which is the additive design's failure expressed in dual variables rather than in logits. Our measurement that 39% of decisions sit at or below the reader floor is a measurement of *how often the constraint is slack* — and it is the majority-adjacent regime, not an edge case.

Two further consequences bear directly on our runs. First, Theorem 2's ‖λ*‖_1 term says that when the multiplier is large the parametrized solution's suboptimality is amplified — the mirror of our observed pathology, where a *bounded* α is out-argued by unbounded tower margins. Second, strong duality is proved over randomized policies via mixing; our deployment picks a greedy action per decision, so even at the exact λ* the deterministic per-decision argmax need not realize the constrained optimum. That is the same gap [calvofullana2021_state_augmented_constrained_rl](calvofullana2021_state_augmented_constrained_rl.md) closes by making the multiplier a state variable — and it is the reason a *state-conditioned* λ (equivalently our learned gain head, which measured degenerate in the RL-ranker v13 screening) is the principled repair of the Lagrangian route rather than an add-on.

Design question it bears on: whether to keep additive scalarization with a better-anchored scale, move to an explicit constrained/filter formulation, or adopt a learned (state-conditioned) multiplier — and specifically what value the multiplier must take on measured ties.

## Related pages
- [calvofullana2021_state_augmented_constrained_rl](calvofullana2021_state_augmented_constrained_rl.md) — same group; no *fixed* weighting induces the constrained optimum, multipliers must be state variables.
- [roy2021_direct_behavior_specification_crl](roy2021_direct_behavior_specification_crl.md) — attested failure of learned multipliers: unbounded upward divergence against a resistant constraint (our tie regime is exactly "resistant").
- [zhang2026_alam_multiplier_network](zhang2026_alam_multiplier_network.md) — per-state multiplier networks and their oscillation dynamics.
- [wray2015_lexicographic_mdp_slack](wray2015_lexicographic_mdp_slack.md) — the complementary impossibility on the lexicographic side.

## Sources
- [arXiv 1910.13393](https://arxiv.org/abs/1910.13393), NeurIPS 2019.
