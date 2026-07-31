---
type: paper
node_id: paper:kong2020_deep_hurdle_networks
title: "Deep Hurdle Networks for Zero-Inflated Multi-Target Regression: Application to Multiple Species Abundance Estimation"
authors: ["Shufeng Kong", "Junwen Bai", "Jae Hee Lee", "Di Chen", "Andrew Allyn", "Michelle Stuart", "Malin Pinsky", "Katherine Mills", "Carla P. Gomes"]
year: 2020
venue: "IJCAI 2020"
external_ids:
  arxiv: "2010.16040"
  doi: "10.24963/ijcai.2020/603"
  s2: null
tags: ["zero-inflated-regression", "hurdle-model", "two-stage", "degenerate-predictor", "equivalence-critic", "evaluation-metric"]
added: 2026-07-31T00:00:00Z
---

# Deep Hurdle Networks for Zero-Inflated Multi-Target Regression

## One-line thesis
When a regression target has a large atom at zero, factor the likelihood the way the classical hurdle model does — a Bernoulli head for "is it zero or positive" times a zero-truncated density for the magnitude given positive — and learn both heads end to end inside one network; and because a pooled RMSE on zero-inflated data is *maximized by the degenerate all-zeros predictor*, report a zero-inflated RMSE that scores the zero part and the positive part separately.

## Why this paper was surfaced
Surfaced as the ML-side canonical treatment of a zero-inflated regression target, for the counterfactual utility-equivalence critic `q_U(s, a)` in `docs/specs/RL/ties-by-design.md` (option i). Our ΔU distribution is exactly hurdle-shaped: ~18% of measured decisions are exactly ΔU = 0 (true utility-equivalent action pairs — the labels the critic exists to produce), ~21% are sub-noise (0 < |ΔU| ≤ 0.044, below the independently measured reader floor), and the rest are live with |ΔU| up to ~0.7. The spec's own stated risk is that a single Huber head collapses to predicting ≈0 everywhere; this paper both names that degeneracy and gives the two structural remedies — factor the model, and fix the metric.

## Key Results
- **Hurdle factorization (§3.1).** For target `y_j ≥ 0`, let `y′_j = 1[y_j > 0] ~ Bernoulli(p_j)` and let `f(y_j | y_j > 0)` be a zero-truncated density; the likelihood is `L(y_j) = Pr(y′_j = 1) · f(y_j | y_j > 0)`. The two components are **independent** in the classical model — which is exactly the property the deep version sets out to break, because the presence and magnitude signals share structure.
- **Deep Hurdle Network (DHN).** A shared encoder produces latent features; a multivariate probit head models the joint presence/absence pattern across all targets, and a multivariate log-normal head models the positive magnitudes. The two are coupled by a penalty on the difference between their covariance matrices — so the "will it be zero" head and the "how large" head are tied together rather than trained as two disconnected models, and the whole thing trains end to end on GPU.
- **The degeneracy is stated as the reason for a new metric.** Verbatim: "Since the data considered here are zero-inflated, using standard RMSE might not be appropriate. Models can produce degenerate results by simply predicting a near-zero vector for each test data point." Their **zRMSE** therefore mixes an error term over the true-zero indices `I_0` (penalizing predicted magnitude where the truth is zero) with an error term over the true-positive indices `I_+`, combined by a weight α: `sqrt( α·Σ_{j∈I_0} ŷ_j² / |I_0| + (1−α)·Σ_{j∈I_+} (y_j − ŷ_j)² / |I_+| )`, averaged over the test set. "A model cannot cheat by predicting near-zero vectors by ignoring the positive parts."
- **Results.** SBTS (fish) — ACC / zRMSE / time(min): DHN 0.65 / 1.71 / 57 vs the statistical multi-level zero-inflated log-normal MLZILN 0.52 / 1.96 / 218, vs plain multi-target regressors MTRS 0.31 / 3.50 / 76, MORF 0.45 / 2.17 / 87, RLTC 0.40 / 2.96 / 96, MOSVR 0.32 / 3.16 / 120, MMR 0.47 / 2.84 / 55. eBird (birds): DHN 0.59 / 0.96 / 45 vs MLZIP 0.50 / 1.39 / 186, others 1.75–2.87 zRMSE.
- DHN cuts error 12.8% below MLZILN and 30.9% below MLZIP (α = 0.5), and is 4× faster to train than the statistical baselines.
- **The clean ablation-style finding:** the zero-inflation-aware models (MLZILN/MLZIP and DHN) all beat the zero-agnostic multi-target regressors on zRMSE, and the sweep over α (Fig. 3) shows *why* — "zero-inflated models tend to perform better for positive parts, while other nonzero-inflated models tend to underestimate the positive parts." Modelling the atom is what stops the shrink-toward-zero bias on the live cases. That is the failure we are trying to avoid, isolated and measured.

## Relevance to This Project
This is the paper that most directly matches our target's shape, and the transferable content is two things, of which the second is nearly free.

**The metric (take this regardless of which loss we pick).** Our critic's natural evaluation — Huber or MAE of predicted pairwise difference against measured ΔU, pooled over all evidence rows — is the exact metric this paper rejects, and for the exact reason: with 39% of the mass at or near zero, the all-zeros predictor scores well, so a good pooled number is not evidence the critic works. The zRMSE construction gives us the shape of the right metric: score the exact-tie rows by *how much nonzero magnitude the critic wrongly predicts* and the live rows by ordinary regression error, and report them separately rather than blended (α is a reporting choice, not a tuned knob — report the curve or report both terms). Downstream this is also what turns tie behavior into a supervised, measurable property (precision/recall on the predicted equivalence set), which `ties-by-design.md` already lists as the reason to prefer the critic over emergent-RL alternatives.

**The factorization (the structural recommendation).** The hurdle form matches our semantics far better than any reweighting scheme: our zero atom is a *distinct event* (the two actions are utility-equivalent), not an over-populated region of a continuum, so it should be a Bernoulli outcome and not a value the regressor tries to hit. A tie head `P(|ΔU| ≤ 0.044 | s, a_i, a_j)` plus a magnitude head trained only on the live rows separates the two questions the critic is asked and stops the tie mass from dragging the magnitude head's scale down. Two adaptations are needed. (a) Our hurdle boundary is not zero but the measured 0.044 noise floor — sub-noise pairs are operationally indistinguishable from exact ties at inference time, so the classifier's positive class should be "live" (|ΔU| > 0.044), which merges the two near-zero strata into one class (~39%) and leaves a well-balanced 39/61 binary problem instead of an 18% minority. This is a *strength* at our sample size: a 39/61 binary split is learnable from a few hundred examples where a three-way stratified regression is not. (b) DHN's shared-covariance coupling is a multi-target device (hundreds of species) and does not apply to our single scalar ΔU; the transferable part is the shared encoder — one trunk, two heads — which we get for free since `q_U` already shares the actor's features.

The honest caveat: DHN's evidence is at n in the tens of thousands of sites, not hundreds of pairs, and it validates a *joint* multivariate construction we are not using. It supports "factor the atom out of the regression, and never trust a pooled error on zero-inflated data"; it does not by itself establish that a two-head split beats one Huber head at n = a few hundred. That is what our ablation has to settle.

## Design question it bears on
Whether the equivalence critic is one regression head (with the zero mass handled by sampling or loss weights) or two heads — a tie/live classifier at the 0.044 floor plus a magnitude regressor on the live rows. And, prior to that question and independent of its answer, what metric can even detect the collapse: this paper's zRMSE argument says the pooled Huber we would naturally reach for cannot.

## Sources
- [arXiv 2010.16040](https://arxiv.org/abs/2010.16040) · [doi:10.24963/ijcai.2020/603](https://doi.org/10.24963/ijcai.2020/603)
- Companion framings of the same skew as a *continuous* imbalance rather than an atom: [yang2021_deep_imbalanced_regression.md](yang2021_deep_imbalanced_regression.md) ([arXiv 2102.09554](https://arxiv.org/abs/2102.09554)) and [ren2022_balanced_mse.md](ren2022_balanced_mse.md) ([arXiv 2203.16427](https://arxiv.org/abs/2203.16427))
