---
type: paper
node_id: paper:angelopoulos2024_conformal_risk_control
title: "Conformal Risk Control"
authors: ["Anastasios N. Angelopoulos", "Stephen Bates", "Adam Fisch", "Lihua Lei",
          "Tal Schuster"]
year: 2024
venue: "ICLR 2024 (arXiv 2022)"
external_ids:
  arxiv: "2208.02814"
  doi: null
  s2: null
tags: ["conformal-prediction", "risk-control", "distribution-free", "calibration",
       "exchangeability"]
added: 2026-07-04T00:00:00Z
---

# Conformal Risk Control

## One-line thesis
Extends split conformal prediction from coverage (0/1 loss) to any bounded, monotone loss:
pick a threshold on a held-out calibration set and get a distribution-free, finite-sample
guarantee E[loss] ≤ α — for any underlying model, without retraining it.

## Problem / Gap
Split conformal prediction guarantees marginal coverage of prediction sets but nothing about
task losses beyond miscoverage (false-negative rate, set-valued risks, monotone utilities).
Practitioners need post-hoc guarantees on arbitrary risks for black-box models.

## Method
Given a parameterized family of predictions C_λ growing monotonically in λ and a bounded loss
L(C_λ(x), y) non-increasing in λ, choose λ̂ on n exchangeable calibration points such that the
adjusted empirical risk ≤ α; then E[L(C_λ̂(X_{n+1}), Y_{n+1})] ≤ α for the next exchangeable
draw. Coverage is the special case L = 1{y ∉ C_λ}.

## Key Results
- Finite-sample, distribution-free control of any monotone bounded risk; tightness up to
  O(1/n).
- Demonstrated on FNR control in multi-label classification and segmentation, graph distance
  in hierarchical prediction, and F1-related risks in open-domain QA.

## Assumptions
- Exchangeability of calibration and test data (i.i.d. suffices); loss bounded and monotone
  in the threshold; guarantee is **marginal** (on average over draws), not per-instance or
  per-subgroup.

## Limitations / Failure Modes
- Distribution shift between calibration and deployment voids the guarantee; per-condition
  (conditional) guarantees are impossible in general without further assumptions.
- Controls the expectation, not the tail: individual instances can violate arbitrarily as
  long as the average holds.

## Reusable Ingredients
- Recipe for wrapping any learned risk scorer with a per-type threshold shift so that
  "fraction of over-threshold items admitted" ≤ δ with finite-sample validity — applicable to
  a distilled span-risk head as an admission rule.

## Open Questions
Whether per-user document distributions are close enough to a shared calibration corpus for
the exchangeability assumption to be credible in a privacy product.

## Relevance to This Project
Surfaced while evaluating replacements for the inference-time `walk_risk` LM probe (see
`docs/research/inference-risk-enforcement.md`): the distilled-risk-head option needs its
"predicted-safe" mask to carry a stated guarantee, and conformal risk control is the
strongest available wrapper — it makes precise what such a mask can promise (marginal
expected violation rate under exchangeability) and therefore also what it cannot (worst-case,
per-document, or under user-domain shift). This bounds the option's suitability for a hard
privacy floor and is the reason it is rated below the structural (counting-based) mask.
