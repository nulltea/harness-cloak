---
type: paper
node_id: paper:davidson1970_bradley_terry_ties
title: "On Extending the Bradley-Terry Model to Accommodate Ties in Paired Comparison Experiments"
authors: ["Roger R. Davidson"]
year: 1970
venue: "Journal of the American Statistical Association 65(329), 317-328"
external_ids:
  arxiv: null
  doi: "10.1080/01621459.1970.10481082"
  s2: null
tags: ["preference-model", "ties", "bradley-terry", "choice-axiom", "equivalence-critic"]
added: 2026-07-31T00:00:00Z
---

# On Extending the Bradley-Terry Model to Accommodate Ties in Paired Comparison Experiments

Venue link: [JASA 65(329), 317-328](https://www.tandfonline.com/doi/abs/10.1080/01621459.1970.10481082) (DOI [10.1080/01621459.1970.10481082](https://doi.org/10.1080/01621459.1970.10481082)).

## Why this paper was surfaced

It is the second canonical tie model and the direct competitor to Rao-Kupper (1967). It matters for us because its critique of Rao-Kupper is a *scale-consistency* critique: Rao-Kupper's threshold breaks Luce's choice axiom, so the ratio of win probabilities stops being λ_i/λ_j once ties are allowed. Since our critic q_U is a learned network whose output scale is free to drift during training, whether the tie term is scale-consistent or not is a live implementation concern, not a philosophical one. Both modern tie-aware RLHF papers we registered implement *both* models, so knowing which one to pick requires this paper.

## One-line thesis

Ties should be assigned probability proportional to the *geometric mean* of the two win probabilities — the unique way to admit ties while preserving Luce's choice axiom P(i ≻ j)/P(j ≻ i) = λ_i/λ_j — giving a one-parameter alternative to Rao-Kupper with the same asymptotic efficiency and, for balanced designs, the ML ranking of the familiar 2-points-for-a-win / 1-for-a-tie score.

## Key Results

- **The model.** P(i ≻ j) = λ_i / (λ_i + λ_j + 2ν √(λ_i λ_j)) and P(i ~ j) = 2ν √(λ_i λ_j) / (λ_i + λ_j + 2ν √(λ_i λ_j)), with ν ≥ 0; ν = 0 recovers Bradley-Terry. The defining requirement is P(i ~ j) = 2ν √(P(i ≻ j) P(j ≻ i)).
- **Why not Rao-Kupper.** Davidson starts from Luce's choice axiom, which demands P(i ≻ j)/P(j ≻ i) = λ_i/λ_j; the Rao-Kupper model *fails* this. Davidson's added requirement (tie probability proportional to the geometric mean of the preference probabilities) yields the unique model satisfying both the choice axiom and a nonzero tie class.
- **Estimation.** ML estimates via an iterative procedure that, under a weak assumption, converges monotonically to the likelihood-equation solution.
- **Scoring equivalence.** For a balanced paired-comparison experiment the ML ranking coincides with the ranking from the 2/1/0 win/tie/loss score system.
- **Efficiency.** The likelihood-ratio test of equal preferences has the same asymptotic efficiency as under the Rao-Kupper model — the two tie models are statistically comparable; the choice between them is structural, not about power.
- **Margin form (as used in RLHF).** With d = r_i − r_j: P(i ≻ j) = 1 / (1 + e^{−d} + 2ν e^{−d/2}) and P(i ~ j) = 2ν e^{−d/2} · P(i ≻ j). Equally matched items (λ_i = λ_j) tie with probability P(tie) = ν · P(no tie), so ν = 1 makes a genuine tie exactly as likely as a decided outcome.

## Relevance to This Project

**Scale-consistency is the reason to prefer Davidson for a learned critic.** Rao-Kupper's threshold α is an additive shift on the margin d, so the tie band has a *fixed width in output units*: if q_U's outputs shrink or expand during training (there is no constraint pinning their scale, only differences are supervised), the effective tie band in ΔU units moves with it. Davidson's ν multiplies a geometric-mean term and preserves the win-ratio structure, so the model is invariant to the overall multiplicative rescaling of the λ's, i.e. to an additive shift of all rewards — the drift mode a difference-supervised critic actually has.

**But Rao-Kupper's parameter is the one our measurement names.** Our tie threshold 0.044 is a *margin* in ΔU units, and Rao-Kupper's α is exactly a margin in latent-score units, so α = 0.044 is a direct, non-fitted assignment. Davidson's ν has no such reading: it sets the *rate* of ties among matched items (a base rate, ~39% ties in our data), not a resolution threshold. Mapping our measured noise floor into ν requires assuming a relation between margin and tie rate; mapping it into α does not. This is the concrete trade-off: Davidson buys scale-invariance, Rao-Kupper buys a directly measurable parameter.

**Gradient behaviour on the tie stratum differs.** In margin form, both models produce a tie-gradient scale factor that is an *odd* function of d, driving d → 0 for tied pairs, but Davidson's is milder and Rao-Kupper's is steeper (Chen et al. 2025, Sec. 2.3.1). Given that ~39% of our decisions carry tie-class evidence, the milder pull is the safer default for the live-pair signal not to be swamped.

## Design question it bears on

If we adopt a tie-aware likelihood, which one — Rao-Kupper (threshold on the margin, directly equals our measured 0.044, not scale-invariant) or Davidson (choice-axiom-consistent, scale-invariant, parameter is a tie *rate* we would have to derive)? Davidson also tells us the two are asymptotically equally efficient, so this choice cannot be settled on statistical power and must be settled on parameter interpretability and training stability.

## Caveats

- Like Rao-Kupper, this is a fixed-item model with no covariates; the contextual neural version is a modern reinterpretation.
- Davidson's ν has no measurement-instrument reading. Setting it from our noise floor requires an extra modelling step (e.g. match the model's implied tie rate to the observed 39%), which reintroduces a fitted quantity — precisely the kind of per-method knob our empirical-honesty rule forbids using to equalize a secondary quantity, unless it is fixed once and held across all compared arms.
- The 2/1/0 scoring equivalence holds only for balanced designs; our action pairs are not sampled in a balanced round-robin, so that convenience does not transfer.

## Sources

- Davidson, R. R. (1970). *On Extending the Bradley-Terry Model to Accommodate Ties in Paired Comparison Experiments.* JASA 65(329), 317-328. DOI [10.1080/01621459.1970.10481082](https://doi.org/10.1080/01621459.1970.10481082).
- Rao-Kupper comparison and margin-form restatements: [raokupper1967_ties_paired_comparisons.md](raokupper1967_ties_paired_comparisons.md) (DOI [10.1080/01621459.1967.10482901](https://doi.org/10.1080/01621459.1967.10482901)) and [chen2025_dpo_ties.md](chen2025_dpo_ties.md) ([arXiv 2409.17431](https://arxiv.org/abs/2409.17431)).
