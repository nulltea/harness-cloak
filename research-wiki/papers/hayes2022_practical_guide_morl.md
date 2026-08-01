---
type: paper
node_id: paper:hayes2022_practical_guide_morl
title: "A Practical Guide to Multi-Objective Reinforcement Learning and Planning"
authors: ["Conor F. Hayes", "Roxana Rădulescu", "Eugenio Bargiacchi", "Johan Källström", "Matthew Macfarlane", "Mathieu Reymond", "Timothy Verstraeten", "Luisa M. Zintgraf", "Richard Dazeley", "Fredrik Heintz", "Enda Howley", "Athirai A. Irissappane", "Patrick Mannion", "Ann Nowé", "Gabriel Ramos", "Marcello Restelli", "Peter Vamplew", "Diederik M. Roijers"]
year: 2022
venue: "Autonomous Agents and Multi-Agent Systems 36(26)"
external_ids:
  arxiv: "2103.09568"
  doi: "10.1007/s10458-022-09552-y"
  s2: null
tags: ["morl", "scalarization", "utility-based-approach", "convex-hull", "solution-concept"]
added: 2026-08-01T00:00:00Z
---

# A Practical Guide to Multi-Objective Reinforcement Learning and Planning

## One-line thesis
The scalarization function is not a design convenience but a *model of the user's utility*, and it must be derived before the algorithm is chosen; a linear scalarization commits you to the claim that the user's utility is a weighted sum, which recovers only the convex hull of the Pareto front and is inadequate whenever the true preference is thresholded, lexicographic, or otherwise non-linear.

## Why surfaced
Surfaced in the additive-vs-lexicographic adjudication (2026-08-01) as the field's consensus statement of the failure modes of linear scalarization, and — more usefully — as the *procedure* that would have caught our error before it was implemented. The utility-based approach says: write down the user's utility first, then pick the solution concept, then pick the algorithm. We picked an additive controller first.

## Key Results
- **The convex-hull limit.** For a positively-weighted linear utility u(V^π) = w^T V^π, the undominated set is exactly the convex hull CH(Π) = {π | ∃w, ∀π′ : w^T V^π ≥ w^T V^π′}. Policies whose value vectors lie in concave regions of the Pareto front are unreachable for every weight — the classical expressiveness ceiling of linear scalarization.
- **The utility-based approach versus the axiomatic approach.** The axiomatic approach assumes the answer is the Pareto front; the utility-based approach derives the solution concept from what is actually known about the user's utility and which policy classes are permitted. Prescribed steps: collect a priori utility information → decide the admissible policy class → derive the solution concept → *then* select an algorithm matching that concept → give the user a selection mechanism if several policies survive.
- **Linear scalarization is a substantive modelling assumption, and often a wrong one.** Many applications use it because it turns the MOMDP into an ordinary MDP so existing convergence proofs apply; the guide notes it is a suitable representation only in domains where the objectives are natively commensurable (their example: everything denominated in money), and cites Vamplew et al. 2008 for its inadequacy elsewhere.
- **The cost of leaving linearity.** Non-linear scalarization violates the additive-return assumption at the heart of the Bellman equation, so Q-values and action selection may have to be conditioned on an augmented state; this is the technical price of a thresholded or lexicographic utility, and the guide names it plainly rather than hiding it.
- **Lexicographic MDPs are positioned as the model for ordered objectives with tolerance** — the guide's own summary of Wray et al. 2015: a specified ordering over objectives, allowed to be state-dependent, "incorporating the concept of slack, which allows some degree of loss in the primary objective in order to obtain gains in secondary objectives".
- Stochastic mixtures over the deterministic stationary convex coverage set suffice to construct the Pareto front, so the CCS — not the full PF — is the practical object; relevant because it means "randomization buys expressiveness" is a general fact, not a technicality of constrained RL.

## Relevance to This Project
Our composition question is a utility-modelling question that we answered implicitly. The user's actual utility in this system is: *`out_final` must preserve every QA-probed answer; subject to that, generalize as far as the anonymity count allows; the λ dial moves how aggressively we spend the tolerance.* That is a thresholded-lexicographic utility, and by the convex-hull result no weight w — no α — represents it. The guide's framing turns our five rounds of scale control into a diagnosable category error: we kept adjusting w inside a family that cannot contain the target, because the additive form was chosen for algorithmic convenience (it keeps the problem a single-objective MDP with standard convergence properties) rather than derived from the preference.

The guide also prices the alternative honestly, which matters for our fork. Leaving linearity breaks the additive-return assumption behind the Bellman equation, so a correct non-linear composition generally needs state augmentation — in our architecture that is precisely the λ-conditioning we already have plus a per-decision notion of the admissible set, i.e. the equivalence critic. It is not a free swap. And the CCS/mixture observation is a caution for our greedy deployment: our per-decision argmax is a deterministic policy, and part of what scalarization can express is only reachable by randomizing, so comparisons must be run against the deterministic-policy solution concept we actually ship.

Design question it bears on: whether the additive controller is a tuning problem or a misspecified solution concept, and what the acceptance criterion for a replacement is — the guide's answer is that the criterion is fidelity to the stated user utility, measured as the realized policy's behaviour under the dial, which matches our controllability-as-first-class-metric position.

## Related pages
- [wray2015_lexicographic_mdp_slack](wray2015_lexicographic_mdp_slack.md) — the constructive impossibility (LMDP optima outside the scalarized solution space) that the guide summarizes.
- [vamplew2024_value_function_interference](vamplew2024_value_function_interference.md) — what goes wrong *inside* a value-based learner when a scalarizing utility is many-to-one.
- [delasheras2026_controllability_preference_morl](delasheras2026_controllability_preference_morl.md) — measuring whether the preference input actually controls behaviour.
- [skalse2022_lexicographic_morl](skalse2022_lexicographic_morl.md), [tercan2024_thresholded_lexicographic](tercan2024_thresholded_lexicographic.md) — the RL-side lexicographic machinery.

## Sources
- [arXiv 2103.09568](https://arxiv.org/abs/2103.09568) · JAAMAS 36(26), 2022, [DOI 10.1007/s10458-022-09552-y](https://doi.org/10.1007/s10458-022-09552-y)
