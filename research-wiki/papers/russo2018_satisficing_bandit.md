---
type: paper
node_id: paper:russo2018_satisficing_bandit
title: "Satisficing in Time-Sensitive Bandit Learning"
authors: ["Daniel Russo", "Benjamin Van Roy"]
year: 2018
venue: "arXiv 2018; Mathematics of Operations Research 47(4):2815-2839 (2022)"
external_ids:
  arxiv: "1803.02855"
  doi: "10.1287/moor.2021.1229"
  s2: null
tags: ["satisficing", "rate-distortion", "information-cost", "epsilon-optimality", "measurement-resolution"]
added: 2026-08-01T00:00:00Z
---

# Satisficing in Time-Sensitive Bandit Learning

## Why this paper was surfaced

Our reader has a measured resolution floor of 0.044 utility units; below it, differences are unidentifiable rather than merely unmeasured. We wanted the formal statement of what it *costs* to stop targeting the exact optimum and target instead "anything within a stated tolerance", and whether that reframing is principled or merely convenient. This paper turns exactly that trade into a rate-distortion statement.

## One-line thesis

Targeting a *satisficing* action — one within a declared distortion `D` of optimal — instead of the optimum replaces the learning cost `I(θ; A*)` with the rate-distortion function `ℛ(D) = min I(Ã; θ) s.t. E[R* − μ(Ã, θ)] ≤ D`, which can be arbitrarily smaller; the resulting regret bound is explicit, and the benefit grows without bound as the optimum becomes expensive to identify.

## Key Results

- **Satisficing regret bound (Theorem 1).** For any policy `ψ`, distortion `D ≥ 0` and satisficing action `Ã`: `SRegret(α, ψ, D) ≤ sqrt( Γ(Ã, ψ) · I(Ã; θ) / (1 − α²) )`, where `Γ` is a discounted information ratio (regret per unit of information) and `I(Ã; θ)` is the information needed to identify the satisficing target.
- **The information cost is a rate-distortion function (Theorem 2).** Under a uniform information-ratio bound, `SRegret(α, ψ, D) ≤ sqrt( Γ_U · ℛ(D) / (1 − α²) )`. The price of learning is set by the *tolerance you declare*, not by the difficulty of the exact optimum.
- **Unbounded separation, infinite-armed example.** With `K → ∞` deterministic arms, identifying the optimum costs `I(θ; A*) = log K → ∞`, while the satisficing target `Ã = min{a : θ_a ≥ 1 − ε}` costs finite information. Thompson sampling incurs discounted regret `O(1/(1−α))`; satisficing Thompson sampling `O(1/sqrt(1−α))` — the ratio diverges as `α → 1`.
- **Linear bandits.** `SRegret ≤ sqrt( ℛ(D) · p / (2(1 − α²)) )` for dimension `p`; whenever `ℛ(D) ≪ log|𝒜|` this strictly improves on the standard bound.
- **Mechanism: sample the satisficing action, not the optimal one.** Satisficing Thompson sampling posterior-samples the *first action exceeding a threshold* rather than the argmax — the tolerance is built into the target of inference, not into a post-hoc acceptance test.

## Relevance to This Project

**It is the affirmative case for declaring the 0.044 band equivalent by policy.** Our situation is the paper's premise pushed to its limit: separating two actions inside the reader's resolution floor does not merely require a lot of information, it requires information the instrument cannot emit — `ℛ(D)` for `D` below the floor is unattainable at any sample size. The paper's framing says that adopting a distortion level and optimizing to it is a principled formulation with an explicit regret account, not a shortcut: you name `D`, you accept `≤ D` of primary regret, and you buy a strictly cheaper learning problem. Our `D` is not a tuning knob but an independently measured instrument property, which is the strongest version of the argument — the tolerance is not chosen to make the numbers work.

**It makes the *cost* of the policy declaration explicit and small.** Declaring the sub-floor band equivalent concedes at most 0.044 utility units per decision in the worst case, and by construction that concession is unobservable to the reader. This is the same accounting as the tau-slack admissibility statement in [skalse2022_lexicographic_morl.md](skalse2022_lexicographic_morl.md) ([arXiv 2212.13769](https://arxiv.org/abs/2212.13769)) — bounded primary loss `τ/(1−γ)` in exchange for a well-defined secondary objective — but arrived at from the information side rather than the ordering side, and it explains *why* the trade is favorable rather than merely admissible.

**Where it does not reach us.** The bound is about the cost of *learning* a satisficing action under a known likelihood; it assumes the reward is observable at every step and that the distortion is measured against the true `μ`. At our deployment there is no reward observation at all — the reader is not in the loop on a user document — so nothing here supplies the missing utility signal. The paper justifies the tolerance and prices it; it does not tell us how to recognize the satisficing set without measuring. That remains our open problem, and it is not this literature's problem.

## Design question it bears on

Should sub-floor differences be learned (a magnitude head predicting where in the band a pair lies) or declared equivalent by policy? This paper supports *declared*: below the tolerance there is no regret to recover, and targeting the finer distinction pays a strictly higher information cost for a benefit bounded by the tolerance itself. Concretely it argues for the ε-insensitive deadband in the critic's magnitude loss being the *right* modeling choice rather than a convenience — fitting inside the band is fitting noise at positive information cost and zero return — and against any later attempt to sharpen the tie-break by learning sub-floor ordering.

## Caveats

- The formal object is a Bayesian bandit with a known prior and per-step reward observation; our menus are contextual, our reward is measured offline through a cache, and our tolerance is a property of a text-scoring instrument rather than a distortion measure on `θ`.
- `Ã` is chosen by the designer in their examples (e.g. first arm above a threshold); the paper does not solve "identify the satisficing set from features" — which is the part v15 failed at.
- The regret account is for the primary objective only. It says nothing about how to order actions *inside* the satisficing set, which is where our entire privacy mechanism lives.

## Sources

- [arXiv 1803.02855](https://arxiv.org/abs/1803.02855)
- [Mathematics of Operations Research 47(4):2815-2839, DOI 10.1287/moor.2021.1229](https://doi.org/10.1287/moor.2021.1229)
- Precursor: [Time-Sensitive Bandit Learning and Satisficing Thompson Sampling, arXiv 1704.09028](https://arxiv.org/abs/1704.09028).
- Related in this wiki: [mason2020_all_epsilon_good_arms.md](mason2020_all_epsilon_good_arms.md) ([arXiv 2006.08850](https://arxiv.org/abs/2006.08850)) — identifying the whole ε-good set rather than one satisficing member; [bartok2014_partial_monitoring_classification.md](bartok2014_partial_monitoring_classification.md) ([DOI 10.1287/moor.2014.0663](https://doi.org/10.1287/moor.2014.0663)) — the identifiability side of the same question.
