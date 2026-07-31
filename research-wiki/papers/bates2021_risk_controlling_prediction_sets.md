---
type: paper
node_id: paper:bates2021_risk_controlling_prediction_sets
title: "Distribution-Free, Risk-Controlling Prediction Sets"
authors: ["Stephen Bates", "Anastasios Angelopoulos", "Lihua Lei", "Jitendra Malik", "Michael I. Jordan"]
year: 2021
venue: "Journal of the ACM 68(6), Article 43"
external_ids:
  arxiv: "2101.02703"
  doi: "10.1145/3478535"
  s2: null
tags: ["conformal-prediction", "risk-control", "distribution-free", "calibration", "high-probability-bound", "concentration-inequalities", "abstention", "equivalence-critic"]
added: 2026-07-31T00:00:00Z
---

# Distribution-Free, Risk-Controlling Prediction Sets

## One-line thesis
Choose a set-size (or threshold) parameter λ̂ on a held-out calibration set by inverting a finite-sample upper confidence bound on the risk, and obtain a **high-probability** guarantee P(R(λ̂) ≤ α) ≥ 1 − δ over the draw of the calibration set — strictly stronger than controlling E[R], and the concentration inequalities it uses are what tell you how much slack a few-hundred-point calibration set actually leaves.

## Why this was surfaced
Our preregistered canonicalization gate ([`docs/specs/RL/ties-by-design.md`](../../docs/specs/RL/ties-by-design.md), Evaluation section) is "≥95% of predicted-tie pairs measure |ΔU| ≤ 0.044 on held-out documents" — a *rate* claim with a threshold, not an interval claim. [angelopoulos2024_conformal_risk_control](angelopoulos2024_conformal_risk_control.md) ([arXiv 2208.02814](https://arxiv.org/abs/2208.02814)) controls that rate only **in expectation** over the calibration draw, which means the rule we actually ship — fit from one particular calibration set — has no confidence statement attached to it at all. RCPS is the version that does: it gives (α, δ) control, so the gate can read "false-equivalence rate ≤ 5% with 90% confidence", which is what a privacy-product claim needs and what an audit can check. It is also the paper that supplies the explicit small-n arithmetic.

## Method
Given a nested family of predictors T_λ (larger λ = larger set = smaller risk) and a bounded risk R(λ) = E[L(T_λ(X), Y)] non-increasing in λ, compute a pointwise upper confidence bound R̂⁺(λ) valid at level δ from n calibration points, then take λ̂ = inf{λ : R̂⁺(λ') ≤ α for all λ' ≥ λ}. Monotonicity makes the family of tests nested, so no multiplicity correction is needed and the single-λ bound transfers to the selected λ̂. The bounds used are: **Hoeffding–Bentkus** (the pointwise maximum of the Hoeffding bound and the Bentkus binomial-tail bound — the latter is much tighter in the small-risk regime we live in), and the **Waudby-Smith–Ramdas** betting/empirical-Bernstein bound for low-variance losses.

## Key Results
- **Theorem (RCPS validity).** For i.i.d. (or exchangeable) calibration data and a valid pointwise UCB, P(R(λ̂) ≤ α) ≥ 1 − δ. Distribution-free, finite-sample, black-box in the underlying model.
- Bentkus/binomial-tail dominates Hoeffding by a large factor when α is small. Concretely at n = 200 and δ = 0.1, plain Hoeffding slack is √(log(1/δ)/2n) = 0.076 — larger than the α = 0.05 we are trying to certify, i.e. Hoeffding alone cannot certify the gate at all at this n. Exact binomial inversion can: it admits up to 5 violations in 200 accepted pairs (2.5% empirical) to certify ≤5% at 90% confidence.
- Demonstrated across hierarchical classification, multi-label FNR control, image segmentation, and protein structure prediction; the framework is agnostic to how λ indexes the predictor.
- The expectation-controlled variant (later generalized as conformal risk control) is the δ → free special case; RCPS is what you use when you must state a confidence level.

## Assumptions
Exchangeability of calibration and test points. Loss bounded (WLOG in [0,1]) and monotone in λ. The guarantee is **marginal over the test distribution** and high-probability over the calibration draw — it is not conditional on a subgroup, and it does not bound the risk on any particular test instance.

## Limitations / Failure Modes
- Distribution shift between calibration and deployment voids validity outright; the paper offers no shift robustness (see [barber2023_conformal_beyond_exchangeability](barber2023_conformal_beyond_exchangeability.md), [arXiv 2202.13415](https://arxiv.org/abs/2202.13415)).
- The (α, δ) pair costs sample size: high-probability control is materially more expensive than expectation control at n in the low hundreds (2.5 points of slack versus 0.5 points at n = 200 — see below).
- The risk controlled is the *marginal* risk over all test points. A **selective** risk — error rate *conditional on the rule choosing to act* — is a ratio of two random quantities and is not directly a monotone bounded loss, so the vanilla theorem does not apply to it without care.
- i.i.d. calibration points. Pairs clustered within a document are not i.i.d., which inflates the effective variance relative to the nominal n.

## Relevance to This Project
RCPS supplies the *guarantee* half of the conservative canonicalization rule, where [romano2019_conformalized_quantile_regression](romano2019_conformalized_quantile_regression.md) ([arXiv 1905.03222](https://arxiv.org/abs/1905.03222)) supplies the accept/abstain *mechanism*. The clean composition: freeze the interval-based accept rule on one split, then on a second, untouched held-out-document split compute an exact binomial (Clopper–Pearson) upper confidence bound on the violation rate among the pairs the frozen rule accepted. Because the rule is frozen, the accepted calibration pairs are draws from the accept-conditional distribution and the plain binomial bound on selective risk is exactly valid — this sidesteps the ratio problem in the limitation above, at the price that the effective sample size is the *accepted* count, not the total pair count.

The small-sample arithmetic is the reason this page exists. To certify false-equivalence rate ≤ 0.05 at δ = 0.1, the exact binomial bound permits: 0 violations at n_accepted = 50 (5.0 points of slack, and the gate is uncertifiable below n = 45 even with zero violations), 1 in 100 (4.0 points), 5 in 200 (2.5 points), 9 in 300 (2.0 points). At δ = 0.05 the minimum feasible n rises to 59 and the n = 300 allowance drops to 8 violations. So at the n ≈ 100–300 the ledger plausibly yields, the honest gate is **not** "empirical precision ≥ 95%" but "empirical precision ≥ 97–98%" — the finite-sample slack eats 2 to 4 percentage points, and a run that measures exactly 95% on 200 accepted pairs *fails* a properly calibrated 95%/90%-confidence gate. Preregistering the raw 95% number without this correction would be a soft version of the calibration-trick failure the project's honesty rules forbid.

The design question it bears on: **is the abstention rule calibrated as a risk-controlled threshold or as per-pair intervals, and at what confidence?** RCPS says: whichever mechanism produces the accept decision, the shipped claim must come from a δ-level bound on a frozen rule, and at our n that bound is loose enough to change the gate's numeric target.

## Reusable Ingredients
- Nested-family + UCB-inversion recipe: calibrates any single scalar knob (the interval width multiplier, a confidence-score threshold, a per-type risk threshold) with an (α, δ) guarantee.
- Hoeffding–Bentkus and WSR bounds as the concrete UCBs; for a 0/1 violation indicator, exact Clopper–Pearson is tighter still and is what we should actually use.
- The pattern of *freezing the rule on split A and certifying on split B*, which is how a selective (accept-conditional) risk becomes a plain binomial parameter.

## Open Questions
Whether the ledger yields enough held-out *documents* (not pairs) for the exchangeable unit to be the document; a document-clustered bound over tens of documents is far looser than a pair-level bound over hundreds of pairs, and the difference between those two numbers is the honest measure of our exchangeability debt.

## Sources
- [arXiv 2101.02703](https://arxiv.org/abs/2101.02703)
- [JACM 68(6):43, doi 10.1145/3478535](https://doi.org/10.1145/3478535)
