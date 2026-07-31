---
type: paper
node_id: paper:ren2022_balanced_mse
title: "Balanced MSE for Imbalanced Visual Regression"
authors: ["Jiawei Ren", "Mingyuan Zhang", "Cunjun Yu", "Ziwei Liu"]
year: 2022
venue: "CVPR 2022 (Oral)"
external_ids:
  arxiv: "2203.16427"
  doi: "10.1109/CVPR52688.2022.00777"
  s2: null
tags: ["imbalanced-regression", "balanced-mse", "loss-design", "logit-adjustment", "equivalence-critic"]
added: 2026-07-31T00:00:00Z
---

# Balanced MSE for Imbalanced Visual Regression

## One-line thesis
MSE is the negative log-likelihood of a Gaussian predictive distribution *under a uniform label prior*; when the training label distribution `p_train(y)` is skewed, minimizing MSE fits `p_train(y|x)` rather than the balanced `p_bal(y|x)`, and the correction is one extra term — subtract `log ∫ N(y′; y_pred, Σ) p_train(y′) dy′` — which is the regression analogue of logit adjustment in imbalanced classification, and whose batch Monte-Carlo estimate (BMC) reduces to an in-batch softmax/contrastive loss requiring no knowledge of `p_train` at all.

## Why this paper was surfaced
Surfaced as the strongest loss-side alternative to LDS-style inverse-density reweighting for training the counterfactual utility-equivalence critic `q_U(s, a)` (`docs/specs/RL/ties-by-design.md`, option i) on our zero-inflated ΔU targets (~18% exact zero, ~21% sub-noise below the 0.044 reader floor, remainder live to ~0.7). Its BMC variant is directly attractive at our scale because it needs no density estimate over ΔU — and its derivation is also the clearest statement of *why* an unbalanced regression head collapses toward the modal target, which is the exact degeneracy we are trying to preregister against.

## Key Results
- **The statistical identity.** Writing MSE as `−log N(y; y_pred, σ²I)` presumes a balanced label prior. Under a skewed `p_train`, the Bayes-consistent loss for recovering `p_bal(y|x)` is `L = −log N(y; y_pred, Σ) + log ∫_Y N(y′; y_pred, Σ) p_train(y′) dy′` (Definition 3.1). The paper shows that substituting a softmax for `p_bal(y|x)` in the discrete case recovers exactly the logit-adjustment loss, giving imbalanced classification and imbalanced regression one shared derivation for the first time.
- **Three implementations, distinguished by what they assume about `p_train`.** *GAI* (GMM analytical integration) models `p_train(y)` as a Gaussian mixture and integrates in closed form — needs an explicit density fit. *BNI* (bin-based numerical integration) reuses a KDE estimate of `p_train` at bin centres — needs the DIR-style binning. *BMC* (batch Monte-Carlo) treats the labels already in the minibatch as samples from `p_train` and needs **no prior knowledge of the label distribution**; it collapses to `L = −log [ exp(−‖y_pred − y‖² / τ) / Σ_{y′ ∈ B_y} exp(−‖y_pred − y′‖² / τ) ]` with temperature `τ = 2σ²_noise`, i.e. classify the true label against the other labels in the batch.
- `σ_noise` is a **learnable** `torch.nn.Parameter` (typically on its own learning rate), not a hand-set knob — the loss self-calibrates its own scale.
- IMDB-WIKI-DIR, balanced MAE (bMAE, all / many / medium / few): vanilla 13.92 / 7.32 / 15.93 / 32.78; RRT+LDS 13.09 / 7.30 / 14.05 / 30.26; **BMC 12.69 / 7.59 / 12.90 / 28.28**; **GAI 12.66 / 7.65 / 12.68 / 28.14**. BMC matches the density-aware GAI while using no density estimate.
- Plain MAE on the same run: vanilla 8.06 / 7.23 / 15.12 / 26.33 vs BMC 8.08 / 7.52 / 12.47 / 23.29 — overall MAE is *unchanged or marginally worse* while tail error drops substantially. The paper is explicit that the gain is a head↔tail trade measured by balanced metrics, and it reports bMAE precisely because unbalanced MAE hides it.
- The authors note (§ on synthetic/IHMR settings) that when the batch is small or unrepresentative, **BMC gives an inaccurate estimate of `p_train(y)`**; in that regime they fix `σ_noise = 1` and fall back to BNI. This is the paper's own scale caveat and the one that binds hardest on us.
- First general treatment of *high-dimensional* imbalanced regression (human mesh recovery, where the label is pose + shape parameters), which is why BMC's density-free form matters — no KDE is available in high dimensions.

## Relevance to This Project
Balanced MSE is the most theoretically clean answer to "how do I stop the head from collapsing to the modal target", and the one that adds no tuned knob — which matters for us given the project's hard rule against per-model calibration fudges. Two properties recommend it: it needs no explicit density model over ΔU (BMC), and `σ_noise` is learned rather than set, so we do not introduce a second threshold alongside the independently measured 0.044 reader floor.

Three problems block it as our default. First, **BMC's `p_train` estimate is the batch**, and our batches will be small (the whole dataset is a few hundred to a few thousand ΔU pairs). The paper itself flags exactly this failure and retreats to BNI when it bites. With, say, 64 pairs per batch and 39% of them near zero, the in-batch softmax denominator is a very noisy Monte-Carlo estimate of a distribution with an atom at zero — the estimator is worst precisely where our mass is concentrated. Second, and more fundamental: Balanced MSE's target is `p_bal(y|x)`, the predictive distribution **under a uniform prior over the target**. That is the right objective when the label distribution is a sampling artifact — more adult faces were photographed than infant faces, and we want an age estimator that is fair across ages. It is the *wrong* objective for us: our ΔU=0 mass is not a sampling artifact, it is the empirical fact that a large fraction of action pairs really are utility-equivalent. Flattening the prior over ΔU deliberately throws away a correct and useful prior, and at deployment the critic's job is to decide "is this pair inside ±0.044", a decision whose optimal threshold depends on the true base rate of ties. Third, the loss is `p_train`-corrected but still unimodal-Gaussian: it has no mechanism for an atom at zero, so it cannot represent "exactly equivalent" as a distinct outcome from "small continuous difference".

Net: Balanced MSE (BMC) is the correct **ablation** for the equivalence critic — it is the strongest reweighting-free baseline, it is ~5 lines of code, and if it matches a hurdle/two-stage design then the extra structure is unjustified and should be deleted. It is not the right default, because balancing away the tie prior is exactly the thing we must not do.

## Design question it bears on
Whether the ΔU zero mass should be **balanced away** (treated as an over-represented region of a continuous target, this paper's framing) or **modelled** (treated as a distinct atom carrying the label we most care about). The paper's derivation makes the choice explicit rather than implicit: `p_bal` vs `p_train` is a modelling commitment about whether the observed skew is artifact or signal. For the equivalence critic it is signal, which is what pushes the recommendation toward the hurdle formulation and leaves Balanced MSE as the ablation that tests whether that extra structure earns its keep.

## Sources
- [arXiv 2203.16427](https://arxiv.org/abs/2203.16427) · [doi:10.1109/CVPR52688.2022.00777](https://doi.org/10.1109/CVPR52688.2022.00777)
- Code: https://github.com/jiawei-ren/BalancedMSE
- Builds directly on the DIR benchmark of [yang2021_deep_imbalanced_regression.md](yang2021_deep_imbalanced_regression.md) ([arXiv 2102.09554](https://arxiv.org/abs/2102.09554))
