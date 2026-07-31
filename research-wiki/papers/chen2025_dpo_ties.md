---
type: paper
node_id: paper:chen2025_dpo_ties
title: "On Extending Direct Preference Optimization to Accommodate Ties"
authors: ["Jinghong Chen", "Guangyu Yang", "Weizhe Lin", "Jingbiao Mei", "Chenxu Lyu", "Bill Byrne"]
year: 2025
venue: "NeurIPS 2025"
external_ids:
  arxiv: "2409.17431"
  doi: null
  s2: null
tags: ["ties", "dpo", "preference-model", "rao-kupper", "davidson", "regularization", "equivalence-critic"]
added: 2026-07-31T00:00:00Z
---

# On Extending Direct Preference Optimization to Accommodate Ties

## Why this paper was surfaced

This is the modern, empirically decisive paper on the exact fork we face: what happens when tied/near-tied comparison pairs are used to train a preference objective. It does the controlled experiment we would otherwise have to run ourselves — same tied pairs, fed to (i) the plain preference likelihood and (ii) Rao-Kupper / Davidson tie-aware likelihoods — and reports both the failure mode of (i) and the gains of (ii). It also states, in gradient terms, exactly what the tie parameter *does*: it sets the width of a deadband in the margin. That is the property our measured 0.044 noise floor is supposed to buy.

## One-line thesis

Replacing Bradley-Terry in DPO with the Rao-Kupper or Davidson tie models turns previously discarded tied pairs from a liability into a useful, KL-regularizing training signal: tied pairs added to plain DPO *hurt* task performance, while the same pairs added under a tie-aware likelihood match or beat clear-preference-only DPO at lower KL to the reference policy.

## Key Results

- **Plain preference likelihood degrades when fed ties.** DPO trained on clear pairs + tied pairs underperforms DPO on clear pairs alone at nearly every KL level, on WMT21 ZH-EN and IWSLT17 FR-EN (BLEURT) and TL;DR (PairRM win-rate). The frontier shifts "down and to the left": less task performance, more regularization. This is the empirical justification for the field's common practice of discarding ties — and the baseline the tie-aware variants have to beat.
- **Tie-aware likelihood removes the degradation and adds gains.** DPO-RK(CP+TP) and DPO-D(CP+TP) reach DPO(CP)-level task performance at *smaller* KL, and at matched KL exceed DPO(CP). On ALMA-R ZH-EN translation, using conflicting-annotator pairs (2302 pairs previously discarded) as ties: COMET 79.66 (DPO) → 80.63 (DPO-RK) / 80.38 (DPO-D); XCOMET 88.87 → 90.40 / 90.09; KIWI-XXL 74.12 → 75.77 / 75.54.
- **Ties recover data that strict-preference training must throw away.** On GSM8K self-training, 30.9% of the training set had all 8 sampled responses correct — no clear preference exists, so DPO-ST discards them. Labeling those as ties: DPO(CP) 76.4-83.7% across β ∈ {0.1…1.0} vs DPO-RK(CP+TP) 83.5-84.4% and DPO-D(CP+TP) up to 84.5% (base model 70.9%). Tie-compatible variants win at *every* β value.
- **Tie evidence acts as a reference-anchoring regularizer, with theory.** Under ideal-DPO policy theory, a pair with true preference probability γ = 0.5 implies ideal reward margin d* = β log(γ/(1−γ)) = 0, i.e. the ideal policy keeps the reference model's likelihood ratio on tied pairs. Measured: KL to reference 2.258 (DPO) → 1.762 (DPO-RK) / 1.465 (DPO-D), with Preservation Rate (fraction of questions the reference answered correctly that survive training) 95.19% → 97.11% / 97.65%, and higher overall accuracy.
- **The tie parameter is a gradient deadband.** For tied pairs the RK/Davidson gradient scale factors are odd functions of the margin d that drive d → 0; the parameters α_RK and ν_D "control the width of the band in reward margin where there is little gradient contribution from tied pairs. However, for tied pairs whose difference in reward fall outside the band, the gradient updates work to reduce the margin." Rao-Kupper's factor is the more aggressive of the two. Plain DPO has no such mechanism.
- **The tie parameter is set, not fitted, and performance is insensitive to it.** They fix ν_RK = 3 (α_RK = log 3) and ν_D = 1 from the principle that evenly matched items should tie with probability ½, and report empirically that performance is not sensitive to α_RK or ν_D within a sensible range.
- **Tie-aware training also improves implicit reward accuracy** on held-out clear pairs and ties, i.e. the benefit is not confined to the tie class.
- **Regularization strength scales with tie fraction.** The measured regularization effect is proportional to the percentage of tied pairs in the training set (half the NMT pairs are ties, 1/8 for TL;DR — and the effect is correspondingly weaker on TL;DR).

## Relevance to This Project

**It is a direct empirical verdict on our stratum mix.** Our tie-class strata (18% exact + 21% sub-noise = 39%) sit between their TL;DR setting (12.5% ties, weak effect) and their NMT setting (50% ties, strong effect), so we should expect a substantial, measurable effect from how ties are handled — this is not a second-order design detail.

**It identifies the failure mode of ignoring the tie structure.** Their DPO(CP+TP) arm is the analogue of feeding sub-noise pairs to a signed objective: the model spends capacity fitting sign information that is measurement noise, and task performance drops. Our sub-noise stratum has exactly this property — the sign of a 0.01 ΔU is not information at a 0.044 noise floor.

**The deadband semantics is the property Huber regression lacks.** Huber on the difference with target ΔU penalizes *any* deviation from the measured value, including within the noise band: a sub-noise pair with ΔU = 0.01 is a hard target of 0.01, and an exact tie is a hard target of 0.0. A tie likelihood instead gives near-zero gradient anywhere inside the band and pulls only from outside it. For transferability of q_U — where we want "these two actions are interchangeable" to generalize, not "their gap is 0.01" — the deadband is the semantically correct object, and this paper shows it also behaves better empirically.

**The regularization result reframes what tie evidence is for.** Under their theory, tie evidence does not carry magnitude information at all: it constrains the margin to zero and thereby anchors the policy to the reference. Read across to us: the 39% tie-class evidence is not 39% of the regression signal, it is a *regularizer on q_U* that keeps the critic from inventing spurious differentiation. That is directly relevant given our v13 finding that the learned gain head degenerated into a bound-pinned global constant — an objective whose tie mass acts as an explicit no-differentiation constraint is a different (and better-posed) mechanism than one where flatness is an unintended attractor.

**Their parameter-insensitivity finding lowers the risk of adopting a tie likelihood.** The main objection to option (b) would be that we are introducing a new hyperparameter with an arbitrary value. They report insensitivity within a sensible range *and* set it from a principle rather than a sweep; we can do better still, since our value is an independently measured instrument property (0.044) rather than a principle.

## Design question it bears on

(a) Huber difference regression vs (b) tie-aware preference likelihood vs (c) hybrid, for a critic trained on mixed exact-tie / sub-noise / live evidence. This paper is the strongest evidence against treating tied and near-tied pairs with the same machinery as decided pairs, and the source of the deadband reading of the tie threshold that makes our measured 0.044 a natural parameter value. It does *not* settle (a) vs (b) directly, because DPO has no regression-on-ΔU alternative — DPO's only signal is comparison outcomes, whereas we have measured ΔU *magnitudes* on live pairs, which is strictly more information than any of their arms uses.

## Caveats

- Everything is DPO (implicit reward via policy likelihood ratios), not an explicit auxiliary critic head trained by supervised regression. The margin d_θ there is β log-ratio; ours is a direct network output. The likelihood algebra transfers; the optimization dynamics may not.
- Their "ties" are constructed: near-neighbours in a metric ranking, or conflicting automatic-metric annotations, not measured indistinguishability with a known noise floor. Our tie definition is stronger.
- Gains are modest in absolute terms (≈1 COMET point, ≈1 point GSM8K) though consistent in sign across β and across three tasks.
- They never compare against regression on the metric gaps they already have (BLEURT/XCOMET score differences), which is the arm we most want to see. That comparison remains unresolved in the literature we found.

## Sources

- [arXiv 2409.17431](https://arxiv.org/abs/2409.17431) — Chen, Yang, Lin, Mei, Lyu, Byrne. *On Extending Direct Preference Optimization to Accommodate Ties.* NeurIPS 2025.
- Tie models it builds on: [raokupper1967_ties_paired_comparisons.md](raokupper1967_ties_paired_comparisons.md) (DOI [10.1080/01621459.1967.10482901](https://doi.org/10.1080/01621459.1967.10482901)), [davidson1970_bradley_terry_ties.md](davidson1970_bradley_terry_ties.md) (DOI [10.1080/01621459.1970.10481082](https://doi.org/10.1080/01621459.1970.10481082)).
- Companion reward-modeling treatment of the same tie model: [liu2024_reward_learning_ties.md](liu2024_reward_learning_ties.md) ([arXiv 2410.05328](https://arxiv.org/abs/2410.05328)).
