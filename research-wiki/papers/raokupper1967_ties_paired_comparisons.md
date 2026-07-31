---
type: paper
node_id: paper:raokupper1967_ties_paired_comparisons
title: "Ties in Paired-Comparison Experiments: A Generalization of the Bradley-Terry Model"
authors: ["P. V. Rao", "Lawrence L. Kupper"]
year: 1967
venue: "Journal of the American Statistical Association 62(317), 194-204"
external_ids:
  arxiv: null
  doi: "10.1080/01621459.1967.10482901"
  s2: null
tags: ["preference-model", "ties", "bradley-terry", "threshold-parameter", "equivalence-critic"]
added: 2026-07-31T00:00:00Z
---

# Ties in Paired-Comparison Experiments: A Generalization of the Bradley-Terry Model

Venue link: [JASA 62(317), 194-204](https://www.tandfonline.com/doi/abs/10.1080/01621459.1967.10482901) (DOI [10.1080/01621459.1967.10482901](https://doi.org/10.1080/01621459.1967.10482901)).

## Why this paper was surfaced

Our ranker's utility head is being trained on measured counterfactual utility gaps ΔU whose evidence splits into three strata: exact ties (|ΔU| ≤ 1e-9), sub-noise pairs (0 < |ΔU| ≤ 0.044, the reader noise floor), and live pairs (|ΔU| up to ~0.7). The open question is whether "the two actions are indistinguishable" deserves its own likelihood term rather than being encoded as a regression target of 0. Rao-Kupper is the canonical answer in the paired-comparison literature: it is the original model that adds a *threshold of perception* to Bradley-Terry so that "no measurable difference" is a first-class observable outcome instead of a coin-flip win. Every modern tie-aware RLHF objective we found (BTT in Liu et al. 2024, DPO-RK in Chen et al. 2025) is this model in disguise.

## One-line thesis

Bradley-Terry forces every comparison into a strict win/loss, which is arbitrary when two treatments genuinely do not differ; adding a single threshold parameter θ ≥ 1 (equivalently a sensory threshold α = log θ on the latent difference) yields a three-outcome model in which ties carry their own probability mass, and θ = 1 recovers Bradley-Terry exactly.

## Key Results

- **The model.** With strengths λ_i (in RLHF form λ = e^r), P(i ≻ j) = λ_i / (λ_i + θ λ_j), P(j ≻ i) = λ_j / (λ_j + θ λ_i), and P(i ~ j) = (θ² − 1) λ_i λ_j / [(λ_i + θ λ_j)(λ_j + θ λ_i)], with θ ≥ 1. The three probabilities sum to 1 and the tie probability is symmetric in i, j.
- **The threshold is a margin deadband, not a fudge factor.** Rao and Kupper derive the model from Bradley's integral form P(i ≻ j) = ½ ∫_{−(r_i − r_j)}^{∞} sech²(y/2) dy by shifting the integration limit by a sensitivity threshold α: P(i ≻ j) = σ(d_ij − α) with θ = e^α. A judge declares a tie exactly when the perceived difference falls below α. So the threshold parameter lives in the *same units as the latent score difference* — it is a measurement-resolution parameter of the comparator, not of the items being compared.
- **Motivation is measurement noise, stated explicitly.** Judges "may not be able to express any real preference" when their "sense of perception is not sharp enough" to detect small differences; and (as later quoted by Chen et al.) "any model which does not allow for the possibility of ties is not making full use of the information contained in the no-preference class."
- **Estimation.** Maximum-likelihood estimates of the λ's and θ jointly, with a likelihood-ratio test of the equal-preference hypothesis; Davidson (1970) shows the LR test has the same asymptotic efficiency under either tie model.
- **θ = 1 is nested.** The tie-aware model contains Bradley-Terry as the θ → 1 boundary case, so a tie likelihood is never *less* expressive than the plain preference likelihood — it strictly adds a parameter that plain BT pins to "no ties are possible".

## Relevance to This Project

Three things transfer directly to the equivalence critic q_U(s, a).

**The tie threshold is exactly our noise floor.** In Rao-Kupper the threshold α is the resolution limit of the *comparator*, below which the judge cannot express a preference. In our setting the comparator is the reader-based utility measurement and its resolution limit is measured: 0.044. So α ≡ 0.044 in ΔU units (up to the fixed scale factor between critic output units and ΔU units — if q_U is trained to output ΔU-scale values, α = 0.044 with no free parameter at all). This is the rare case where the tie-model parameter that the statistics literature *fits* is instead *pinned by an independent measurement* on our side, which removes the usual objection to tie models (that θ is a nuisance parameter absorbing labeling idiosyncrasy).

**It legitimizes treating our sub-noise stratum as ties rather than as small signed preferences.** Our 21% sub-noise pairs are precisely Rao-Kupper's "difference below the threshold of perception" case: the sign is measured but is not information. Under this model the correct likelihood contribution for such a pair is the tie term, not a win term with a tiny margin, and not necessarily a hard regression target of 0.

**It gives a principled place for our exact-tie stratum.** Exact ties (18%) are the θ-dominated regime: under Rao-Kupper the tie probability is maximized at λ_i = λ_j, so exact-tie evidence pushes q_U(s, a_i) − q_U(s, a_j) toward zero without asserting that the difference is *exactly* zero — a softer, better-calibrated constraint than a squared/Huber penalty on the difference.

## Design question it bears on

Should the equivalence critic's loss be plain (Huber) regression on ΔU differences, or a tie-aware preference likelihood? Rao-Kupper is the reference point for option (b): it says a threshold parameter on the latent margin is the minimal, nested, single-parameter way to make "indistinguishable" a modeled outcome, and that this threshold denotes comparator resolution — which we have measured rather than fitted.

## Caveats

- The model is defined over item *strengths* on a shared latent scale with no covariates; the contextual/neural version (q_U(s, a) as a network) is the modern reinterpretation, not part of the 1967 paper.
- Rao-Kupper's tie probability at λ_i = λ_j is (θ − 1)/(θ + 1) — bounded, and *not* invariant to the overall scale of the λ's in the way Davidson's is (see Davidson 1970 for the choice-axiom critique). If our critic's output scale drifts during training, a fixed θ changes its effective meaning; Davidson's variant is the scale-consistent alternative.
- Rao-Kupper assumes ties are symmetric and generated by the comparator, not by the annotator abstaining strategically. Our measured ΔU ties satisfy this by construction.

## Sources

- Rao, P. V. and Kupper, L. L. (1967). *Ties in Paired-Comparison Experiments: A Generalization of the Bradley-Terry Model.* JASA 62(317), 194-204. DOI [10.1080/01621459.1967.10482901](https://doi.org/10.1080/01621459.1967.10482901).
- The margin-form restatement P(i ≻ j) = σ(d_ij − α), θ = e^α, as used in RLHF, is taken from [chen2025_dpo_ties.md](chen2025_dpo_ties.md) ([arXiv 2409.17431](https://arxiv.org/abs/2409.17431)), Sec. 2.2.
