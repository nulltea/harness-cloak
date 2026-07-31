---
type: paper
node_id: paper:yang2021_deep_imbalanced_regression
title: "Delving into Deep Imbalanced Regression"
authors: ["Yuzhe Yang", "Kaiwen Zha", "Ying-Cong Chen", "Hao Wang", "Dina Katabi"]
year: 2021
venue: "ICML 2021 (Long Oral), PMLR v139"
external_ids:
  arxiv: "2102.09554"
  doi: null
  s2: null
tags: ["imbalanced-regression", "label-distribution-smoothing", "reweighting", "equivalence-critic", "small-sample"]
added: 2026-07-31T00:00:00Z
---

# Delving into Deep Imbalanced Regression

## One-line thesis
Deep regression on a skewed continuous target collapses toward the dense region of label space; because targets are continuous, the fix is not class-style inverse-frequency reweighting over bins but *smoothing* — convolve the empirical label density with a symmetric kernel to get an **effective** density, reweight the loss by its (square-root) inverse (LDS), and separately kernel-smooth the per-bin feature first and second moments so sparse bins borrow statistics from their neighbours (FDS).

## Why this paper was surfaced
Surfaced while designing the counterfactual utility-equivalence critic `q_U(s, a)` (`docs/specs/RL/ties-by-design.md`, option i), whose regression target ΔU is heavily zero-inflated: ~18% exact ΔU=0, ~21% sub-noise (0 < |ΔU| ≤ 0.044), the remainder live up to ~0.7. The stated risk in that spec — "the ~39% near-zero population pushes the critic toward predicting zero everywhere" — is textbook deep imbalanced regression, and this is the paper that named the problem, built the benchmarks, and supplied the two standard mitigations. It is the reference point against which our proposed ad-hoc macro-balancing across the three ΔU strata has to be justified.

## Key Results
- Defines **Deep Imbalanced Regression (DIR)** and curates five benchmarks with the now-standard *many-shot / medium-shot / few-shot* label-region split: IMDB-WIKI-DIR (191.5K images, age 0–186), AgeDB-DIR (12.2K, age 0–101), STS-B-DIR (5.2K sentence pairs, similarity 0–5), NYUD2-DIR (50K images, depth 0.7–10 m), SHHS-DIR (1,892 subjects, health score 0–100).
- **LDS**: effective label density is the empirical density convolved with a symmetric kernel, `p̃(y′) ≜ ∫ k(y, y′) p(y) dy`; loss weights are the inverse (INV) or square-root inverse (SQINV) of `p̃`. The paper's own results use SQINV, i.e. deliberately *damped* inverse-density weighting, not full inverse.
- **FDS**: per-target-bin feature statistics are kernel-smoothed across neighbouring bins and applied as a whitening/re-colouring transform `z̃ = Σ̃_b^{1/2} Σ_b^{-1/2} (z − μ_b) + μ̃_b`, so a sparse bin's feature distribution is regularized toward its neighbours'.
- IMDB-WIKI-DIR MAE (all / many / medium / few): vanilla 8.06 / 7.23 / 15.12 / 26.33 → SQINV+LDS+FDS 7.78 / 7.20 / 12.61 / 22.19. The few-shot gain (26.33 → 22.19, 15.7% relative) dominates; overall MAE moves only 0.28.
- AgeDB-DIR MAE: vanilla 7.77 / 6.62 / 9.55 / 13.67 → SQINV+LDS+FDS 7.55 / 7.01 / 8.24 / 10.79. Note the **many-shot regression** (6.62 → 7.01): the rebalancing is a real trade, it buys tail accuracy with head accuracy, it does not come free.
- The paper's diagnostic finding: test error tracks *effective* (smoothed) label density far more tightly than raw empirical density — the empirical histogram is the wrong quantity to reweight against when the target is continuous.

## Relevance to This Project
Three things carry over directly to `q_U`. (1) **The failure mode is the documented one, and the diagnosis is quantitative, not qualitative** — the "predicts ≈0 everywhere" risk in `ties-by-design.md` is DIR's head-collapse, and this paper's many/medium/few reporting convention is exactly how we should evaluate the critic: never a single pooled Huber/MAE number over all ΔU pairs, always split by stratum (exact-tie / sub-noise / live), since a pooled number is maximized by the degenerate constant-zero predictor that has 39% of the mass. Adopting the stratified metric is arguably the cheapest and most important thing to take from this paper, independent of which mitigation we pick.

(2) **The continuity argument cuts against naive stratum reweighting.** LDS exists because a hard bin histogram misstates the true density when neighbouring targets are semantically adjacent. Our three strata are separated by a hard threshold at 0.044, so a per-stratum weight is precisely the bin-histogram approach LDS warns about: a pair at |ΔU| = 0.043 and one at 0.045 land in different strata and get different weights, despite being indistinguishable at the measurement noise floor. If we reweight, the weight should be a smooth function of |ΔU| (kernel-smoothed effective density over the ΔU axis), not a step function at the stratum boundary.

(3) **But the zero atom is not a density trough, and that is where the analogy breaks.** DIR assumes the target is continuous and that sparse regions are *under-observed* — smoothing is legitimate because a neighbouring label is evidence about a rare one. Our ΔU=0 mass is a genuine atom: exact ties are a discrete, semantically distinct event (the actions are utility-equivalent), not a densely observed interval that happens to be popular. Kernel-smoothing across 0 blurs the atom into the sub-noise continuum, which destroys the one distinction the critic exists to make. So LDS-style reweighting is the right *tool shape* for the sub-noise↔live boundary and the wrong shape for the tie↔non-tie boundary. FDS is separately a poor fit at our scale: it estimates per-bin feature covariances with running momentum updates, which needs many samples per bin — with a few hundred to a few thousand total pairs those covariance estimates are noise.

## Design question it bears on
Whether the equivalence critic's zero-inflation is handled *inside* the loss (density-based reweighting over the ΔU axis) or *outside* it (stratified batch composition, or a separate tie head). This paper argues the loss-side fix is enough when the target is genuinely continuous, and its own AgeDB many-shot regression is the honest warning that reweighting is a trade, not a free win — which matters for us because the "head" of our distribution is the live pairs whose sign and magnitude the actor actually needs.

## Sources
- [arXiv 2102.09554](https://arxiv.org/abs/2102.09554)
- Code and DIR benchmarks: https://github.com/YyzHarry/imbalanced-regression
