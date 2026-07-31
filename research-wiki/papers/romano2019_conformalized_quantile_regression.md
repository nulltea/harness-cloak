---
type: paper
node_id: paper:romano2019_conformalized_quantile_regression
title: "Conformalized Quantile Regression"
authors: ["Yaniv Romano", "Evan Patterson", "Emmanuel J. Candès"]
year: 2019
venue: "NeurIPS 2019 (Advances in Neural Information Processing Systems 32, pp. 3538–3548)"
external_ids:
  arxiv: "1905.03222"
  doi: null
  s2: null
tags: ["conformal-prediction", "quantile-regression", "prediction-intervals", "distribution-free", "calibration", "heteroscedasticity", "equivalence-critic"]
added: 2026-07-31T00:00:00Z
---

# Conformalized Quantile Regression

## One-line thesis
Fit a quantile-regression model for the α/2 and 1−α/2 conditional quantiles, then conformalize its interval by adding a single scalar correction learned on a held-out calibration set — giving a finite-sample, distribution-free marginal coverage guarantee whose interval *width varies with the input*, unlike the constant-width intervals of ordinary split conformal.

## Why this was surfaced
The equivalence critic q_U(s,a) in [`docs/specs/RL/ties-by-design.md`](../../docs/specs/RL/ties-by-design.md) option (i) is a regression head whose *point* prediction of ΔU is compared against the 0.044 reader-noise floor. A point estimate cannot support a conservative accept/abstain rule: "predicted |ΔU| ≤ 0.044" says nothing about how wrong the prediction might be on an unseen document. CQR is the standard machinery for converting a regression head into a per-input interval with a distribution-free guarantee, and its heteroscedastic adaptivity is exactly the property we need — the critic's error is not uniform across pairs (exact ties are dense and easy, live pairs near the floor are sparse and hard), so a constant-width band would either accept confidently-wrong hard pairs or refuse easy ones.

## Method
Split the data into a proposal set and a calibration set. On the proposal set fit any quantile regressor q̂_{α_lo}, q̂_{α_hi} (α_lo = α/2, α_hi = 1−α/2). On the n calibration points compute the conformity score E_i = max{q̂_{α_lo}(X_i) − Y_i, Y_i − q̂_{α_hi}(X_i)} — a signed "how far outside the nominal interval did the truth fall", negative when comfortably inside. Take Q = the ⌈(n+1)(1−α)⌉/n empirical quantile of {E_i} and output C(x) = [q̂_{α_lo}(x) − Q, q̂_{α_hi}(x) + Q]. A negative Q *shrinks* an over-wide nominal interval; a positive Q inflates an over-confident one.

## Key Results
- **Theorem 1 (marginal validity).** Under exchangeability of the calibration and test points, P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 − α, for *any* quantile-regression fit, any distribution, any finite n. No assumption that the quantile model is correct — a bad model yields a valid but wide interval.
- **Theorem 2 (upper bound).** If the scores are almost surely distinct, coverage is also ≤ 1 − α + 1/(n+1), so the procedure is not conservative by more than 1/(n+1) — the finite-sample discreteness cost is entirely characterized.
- Empirically shorter intervals than split conformal and locally-weighted conformal on 11 benchmark regression datasets, with better *conditional* coverage across the input space, because the width tracks the conditional spread rather than a global residual scale.
- The correction is one scalar. All adaptivity comes from the underlying quantile model; conformalization only repairs its miscalibration.

## Assumptions
Exchangeability of calibration and test points (i.i.d. suffices). The guarantee is **marginal** over the joint draw of calibration set and test point — not conditional on x, not conditional on any subgroup, and not conditional on the realized calibration set. Conditional coverage is provably unattainable distribution-free without further assumptions; CQR only *approximates* it via the quantile model.

## Limitations / Failure Modes
- Coverage holds on average over inputs: a systematically hard region can be badly under-covered while the marginal number is met. For us that means a document type where the critic is unreliable can be systematically over-canonicalized without the marginal guarantee noticing.
- The interval is symmetric in the correction Q applied to both ends, so a model that is asymmetrically miscalibrated pays width on the good side.
- Says nothing about the *rate at which the interval is narrow enough to act on*. Coverage is guaranteed; usefulness (interval ⊂ ±0.044) is an empirical property of the fit.
- Requires an honest data split: reusing proposal data for calibration voids validity.

## Relevance to This Project
This is the construction for the accept half of the conservative canonicalization rule. Instead of thresholding the critic's point estimate, fit pinball-loss quantile heads on the evidence ledger for the pairwise difference q_U(s,a_i) − q_U(s,a_j), conformalize on held-out *documents*, and accept a pair into the predicted equivalence set only when the whole conformalized interval lies inside [−0.044, +0.044]; otherwise abstain to the ordinary controller. That inverts the failure mode we care about: a wide interval (critic unsure) can no longer produce a confident false equivalence, it produces an abstention, and abstention costs only privacy-ordering determinism, never utility. It also makes the preregistered gate a *derived* property rather than a hope — the miscoverage level α of the interval is the knob that trades recall of the equivalence set against false-equivalence rate, and 0.044 stays a measured physical constant rather than becoming a tuned threshold (which the project's no-calibration-tricks rule forbids).

The design question it bears on: **what is calibrated in the abstention rule — the per-pair uncertainty interval, or a global accept threshold?** CQR answers "the interval", which is the version that survives heteroscedastic critic error. Its caveat is the one we must preregister: the guarantee is marginal and exchangeability-bound, and our calibration pairs come from training documents while deployment pairs come from unseen documents, which is precisely the gap [barber2023_conformal_beyond_exchangeability](barber2023_conformal_beyond_exchangeability.md) quantifies.

## Reusable Ingredients
- Pinball-loss twin heads + one scalar conformal correction: a drop-in uncertainty wrapper for any of the project's regression heads (the critic, a distilled span-risk head, a utility predictor) with no retraining of the base model.
- The 1/(n+1) two-sided characterization is the cleanest available statement of "how much does a few-hundred-point calibration set cost me" — at n = 200 the conformal quantile index is ⌈201·0.95⌉ = 191, i.e. the realized level is 95.5%, a 0.5-point over-coverage penalty.

## Open Questions
Whether the ledger's ~39% near-zero-ΔU population makes the lower quantile head degenerate (both quantile heads collapsing onto 0), which would produce intervals that are narrow for the wrong reason. Stratified batching across exact-tie / sub-noise / live pairs is the mitigation already noted in the ties-by-design spec.

## Sources
- [arXiv 1905.03222](https://arxiv.org/abs/1905.03222)
- [NeurIPS 2019 proceedings](https://papers.nips.cc/paper/8613-conformalized-quantile-regression)
