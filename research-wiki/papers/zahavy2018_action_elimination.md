---
type: paper
node_id: paper:zahavy2018_action_elimination
title: "Learn What Not to Learn: Action Elimination with Deep Reinforcement Learning"
authors: ["Tom Zahavy", "Matan Haroush", "Nadav Merlis", "Daniel J. Mankowitz", "Shie Mannor"]
year: 2018
venue: "NeurIPS 2018"
external_ids:
  arxiv: "1809.02121"
  doi: null
  s2: null
tags: ["action-filtering", "inference-time-set", "contextual-bandit", "confidence-bounds", "generalization-to-unseen-states"]
added: 2026-08-01T00:00:00Z
---

# Learn What Not to Learn: Action Elimination with Deep Reinforcement Learning

## One-line thesis
Train a network to predict a per-(state, action) elimination signal from an auxiliary supervision channel, wrap its last-layer features in a linear contextual bandit, and eliminate an action only when the *lower* confidence bound of its predicted elimination signal exceeds the validity threshold — which guarantees, with probability 1−δ, that no action belonging to some optimal policy is ever removed, on states never seen during training.

## Why surfaced
Surfaced for the hardest part of the lexicographic-filter question (2026-08-01): a filter needs a per-instance notion of which actions are admissible, and at deployment we cannot measure it. This is the only architecture found in the sweep that *builds* that notion — a learned per-state action filter, deployed on unseen states, with an explicit statistical guarantee that the filter does not delete the action it should have kept. It is the published skeleton of our proposed utility-equivalence critic.

## Key Results
- **Problem setup.** After acting, the agent observes a binary elimination signal e(s,a) = 1 iff "any optimal policy in state s will never choose a". Definition 1 pins the safety invariant: the valid set contains every state-action that is part of *some* optimal policy, so **only strictly suboptimal state-actions may be eliminated**. Admissible (what the algorithm keeps) is distinguished from valid (what it must keep).
- **Explicit separation margin.** The binary signal is relaxed to expectations: E[e(s,a)] ≤ ℓ for valid actions and ≥ u for invalid ones, with ℓ < u. The band [ℓ, u] is a declared indifference gap in signal space, and the authors note ℓ must be known (ℓ ≈ 0.5 "should suffice" in practice) while u need not be — only performance, not correctness, depends on u.
- **The elimination rule (Eq. 2).** With x(s) the AEN's last-layer features, realizability E[e_t(s,a)] = θ*_a^T x(s), and the Abbasi-Yadkori–Pál–Szepesvári self-normalized bound, eliminate a at s iff θ̂^T x(s) − √(β_{t−1}(δ̃) x(s)^T V̄^{−1}_{t−1,a} x(s)) > ℓ. Elimination requires the *conservative* end of the interval to clear the threshold; the union bound over k actions uses δ̃ = δ/k. "This ensures that with probability 1−δ we never eliminate any valid action."
- **Proposition 1.** Q-learning restricted to admissible actions (updating and taking maxima only over the admissible set) still converges to Q*, provided elimination follows the concentration bound — the filter and the value learner can be trained concurrently without breaking either.
- **Explicitly a generalization mechanism, not a lookup.** The stated motivation is that a model trained on the observed elimination signal can "generalize to unseen states" — the signal is immediate (contextual-bandit-fast) whereas the reward is delayed, so what to exclude is learnable long before what to prefer.
- Empirically: >1000 discrete actions in Zork, considerable speedup and added robustness over vanilla DQN.
- Rejected alternatives the paper names: folding the elimination signal into the reward as a shaping penalty (tricky to tune, slow, still explores the bad actions) and interleaved policy-gradient updates on both signals (the two models couple and neither converges cleanly).

## Relevance to This Project
This is the structural template for the deployment-generalization gap. Our evidence ledger, verified-tie labels, and the hard lexicographic oracle are all per-document and empty on an unseen document; the counterfactual probe that produced ΔU is exactly an "external signal available during training but not at inference". Zahavy et al. solve that shape of problem: learn a head that predicts the signal, then *use a confidence bound, not the point estimate*, to make the set decision at inference. Our proposed utility-equivalence critic q_U is the same construction with the elimination signal replaced by the measured ΔU and the validity threshold ℓ replaced by the measured reader-resolution floor 0.044 — and it inherits the paper's asymmetry: build the set conservatively so the error is *keeping too many actions* (falling back to ordinary controller behaviour), never *dropping a utility-preserving one*.

Three transfers are immediate. (1) The safety invariant is the right one to preregister — our filter must never remove an action that actually preserves every probed answer, and the guarantee must be one-sided in that direction; this is the same asymmetry the frozen-rule Clopper–Pearson certification in the ties-by-design spec already encodes, and Zahavy supplies the online, per-decision version of it. (2) The ℓ < u separation band tells us what the 0.044 floor is doing formally: it is a declared indifference gap that makes the classification problem well-posed, and only the *lower* side needs to be known — matching our position that 0.044 comes from the instrument rather than from tuning. (3) The rejected alternatives are our own history: reward-shaping the count signal into the objective is precisely our additive controller, and the paper's objection (untunable, slow, does not stop the bad action being chosen) is the abstract form of five failed scale-control rounds.

Two honest disanalogies. Zahavy's guarantee rests on a *realizability* assumption (the expected signal is linear in the learned features) that is unverifiable for our features, and their state stream is i.i.d.-ish within one game whereas our filter must transfer across documents — the exchangeability caveat already flagged via [barber2023_conformal_beyond_exchangeability](barber2023_conformal_beyond_exchangeability.md). And their signal marks *strictly suboptimal* actions, whereas ours marks *equivalent* ones; the sets are complementary, but the guarantee direction (never remove a keeper) is identical.

Design question it bears on: what a lexicographic filter needs at inference on an unseen document, and how to get it — a supervised, calibrated per-instance predictor plus a conservative confidence-bound set rule, rather than probing or a per-document ledger.

## Related pages
- [mason2020_all_epsilon_good_arms](mason2020_all_epsilon_good_arms.md) — the pure-exploration version of the same object (CI-based membership in an ε-good set), for the measurable regime.
- [katzsamuels2019_true_sample_complexity_good_arms](katzsamuels2019_true_sample_complexity_good_arms.md) — verifiable (fixed threshold) versus unverifiable (ε below an unknown max) set definitions.
- [bates2021_risk_controlling_prediction_sets](bates2021_risk_controlling_prediction_sets.md) — the distribution-free alternative to Zahavy's realizability-dependent bound.
- [baram2021_action_redundancy](baram2021_action_redundancy.md), [asadi2019_state_action_equivalence](asadi2019_state_action_equivalence.md) — action-equivalence structure the filter exploits.

## Sources
- [arXiv 1809.02121](https://arxiv.org/abs/1809.02121), NeurIPS 2018, pp. 3566–3577.
