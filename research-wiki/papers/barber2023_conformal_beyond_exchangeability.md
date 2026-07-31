---
type: paper
node_id: paper:barber2023_conformal_beyond_exchangeability
title: "Conformal Prediction Beyond Exchangeability"
authors: ["Rina Foygel Barber", "Emmanuel J. Candès", "Aaditya Ramdas", "Ryan J. Tibshirani"]
year: 2023
venue: "Annals of Statistics 51(2), 816–845"
external_ids:
  arxiv: "2202.13415"
  doi: "10.1214/23-AOS2276"
  s2: null
tags: ["conformal-prediction", "distribution-shift", "exchangeability", "coverage-gap", "weighted-conformal", "calibration", "deployment-generalization"]
added: 2026-07-31T00:00:00Z
---

# Conformal Prediction Beyond Exchangeability

## One-line thesis
When calibration and test data are not exchangeable, weighted-quantile conformal prediction still holds with an explicit additive **coverage gap**: coverage ≥ 1 − α − Σ_i w̃_i · d_TV(Z, Z^i), where Z^i is the dataset with calibration point i swapped for the test point — so non-exchangeability degrades the guarantee by a computable amount instead of destroying it, and the degradation is small exactly when the reweighting downweights the least-exchangeable points.

## Why this was surfaced
Both halves of our proposed canonicalization rule — the conformalized interval ([romano2019_conformalized_quantile_regression](romano2019_conformalized_quantile_regression.md), [arXiv 1905.03222](https://arxiv.org/abs/1905.03222)) and the risk-controlled certification ([bates2021_risk_controlling_prediction_sets](bates2021_risk_controlling_prediction_sets.md), [arXiv 2101.02703](https://arxiv.org/abs/2101.02703)) — assume calibration and deployment points are exchangeable. Ours are not, in two distinct ways, and the deployment-generalization gap is the whole reason the equivalence critic exists ([`docs/specs/RL/ties-by-design.md`](../../docs/specs/RL/ties-by-design.md)). Every other conformal paper in this collection states only "shift voids the guarantee". This one is the reference that says *by how much*, which is the difference between an unfalsifiable caveat and a preregisterable one.

## Method
Two orthogonal generalizations of split conformal. (1) **Weighted quantiles**: assign fixed (data-independent) weights w_i to calibration points and take the weighted empirical quantile of the conformity scores instead of the unweighted one; weights that favour points believed closer to the test distribution shrink the coverage gap. (2) **Nonsymmetric algorithms**: a randomization/swap construction that permits a fitting algorithm which treats data points asymmetrically (e.g. recency-weighted training), which vanilla conformal forbids because its proof needs symmetry in the fitted model.

## Key Results
- **Main theorem.** For weights w̃_i normalized over the calibration set, P(Y_{n+1} ∈ Ĉ) ≥ 1 − α − Σ_{i=1}^{n} w̃_i · d_TV(Z, Z^i). The penalty is a weighted average of total-variation distances between the observed data sequence and the sequence with point i and the test point exchanged — zero under exchangeability, recovering the classical bound exactly.
- The coverage gap is driven by *how unexchangeable the specific swapped points are*, not by a global distributional distance: swapping a calibration point drawn from the same regime as the test point contributes nothing, so a shift concentrated in a minority of calibration points costs only its weight.
- Unweighted conformal is the special case w̃_i = 1/n, whose gap is the plain average TV distance — i.e. classical split conformal is already "robust with a penalty", not brittle; the weighted version reduces the penalty when you can guess which points are relevant.
- Simulations and real data (electricity demand, election forecasting) show substantially retained coverage under drift, where fixed-window conformal loses coverage badly.
- Honest limitation stated by the authors: the TV term is generally **not estimable** from the data. It is a bound explaining and bounding the damage, not a computable correction you can add to a shipped interval.

## Assumptions
Weights must be fixed in advance (data-independent), or the argument requires the extra randomization machinery. Scores from a symmetric algorithm unless the nonsymmetric construction is used. The result bounds marginal coverage only.

## Limitations / Failure Modes
- The coverage gap is an unknown quantity in practice. You cannot report "our coverage is ≥ 0.93 because the gap is 0.02"; you can only argue the gap is small on structural grounds and then *measure* held-out coverage empirically.
- Weighting helps only if you have a credible prior about which calibration points resemble the test point. With no such signal, uniform weights are the best you can do and the gap is whatever it is.
- Addresses coverage of prediction intervals; the extension to (α, δ) risk control under shift is not covered here.

## Relevance to This Project
This paper is the source of the honest caveat we must preregister for the canonicalization gate. Our non-exchangeability has two components and they are not equally severe.

The first is **document-level clustering**. Calibration pairs are not independent draws: dozens of pairs come from the same document, sharing its reader behaviour, its lattice, and its critic features. A pair-level binomial bound over n ≈ 200 pairs therefore overstates its own precision, because the effective number of independent units is closer to the number of held-out documents. This is not distribution shift, it is variance under-counting, and the fix is structural: make the *document* the exchangeable unit for the certifying split — compute one per-document violation rate and bound the mean over documents — accepting a much looser bound at n_doc in the tens, and report the pair-level number as the optimistic companion. The honest gate is the document-level one.

The second is **document-distribution shift**: calibration documents come from the training corpus, deployment documents are unseen and, in a tailorable privacy product, may come from an entirely different user domain with different sensitive types, different lattice depth, and different reader difficulty. Barber et al. tell us the guarantee survives as 1 − α − (average TV between our corpus and the deployment stream), which is exactly the term nobody can estimate. The preregistered statement therefore has to be: *the ≥95% false-equivalence bound is valid for documents exchangeable with the held-out calibration split; for a user domain unlike the corpus, the bound degrades by an unmeasured amount and the only defence is that the failure mode is bounded in cost* — a false equivalence costs utility on the order of the true |ΔU|, and the abstain-by-default rule means an out-of-domain document with an unreliable critic produces wide intervals and abstentions rather than confident false ties. Weighted conformal is not usable for us out of the box, since we have no credible per-document relevance weights at deployment for an unseen document, but it is the upgrade path if per-user calibration documents ever become available (weight the corpus split toward documents resembling the user's).

The design question it bears on: **what does the calibrated abstention guarantee actually survive when the deployment documents are new?** Answer: the mechanism survives (abstention is still conservative, still costs no utility), the *number* does not survive unconditionally, and the correct response is to make the certifying unit the document and to state the residual shift term as an acknowledged unmeasured gap rather than pretending exchangeability holds.

## Reusable Ingredients
- The coverage-gap decomposition as the standard way to phrase "our calibration corpus is not the deployment distribution" in a preregistration, with a citation instead of hand-waving.
- The document-as-exchangeable-unit reframing (cluster the calibration statistic at the level that is actually i.i.d.), which is the cheap and correct fix for our clustering problem.
- Weighted-quantile conformal as the future per-user-calibration path.

## Open Questions
How many held-out documents the evidence ledger can supply for a document-level bound, and whether a document-clustered gate at n_doc ≈ 20–40 is tight enough to be informative at all — if not, the honest report is a pair-level number plus an explicit statement that its effective sample size is unknown.

## Sources
- [arXiv 2202.13415](https://arxiv.org/abs/2202.13415)
- [Annals of Statistics 51(2):816–845, doi 10.1214/23-AOS2276](https://doi.org/10.1214/23-AOS2276)
