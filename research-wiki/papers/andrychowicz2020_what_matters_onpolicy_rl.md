---
type: paper
node_id: paper:andrychowicz2020_what_matters_onpolicy_rl
title: "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study"
authors: ["Marcin Andrychowicz", "Anton Raichuk", "Piotr Stanczyk", "Manu Orsini", "Sertan Girgin", "Raphael Marinier", "Leonard Hussenot", "Matthieu Geist", "Olivier Pietquin", "Marcin Michalski", "Sylvain Gelly", "Olivier Bachem"]
year: 2020
venue: "arXiv 2020; ICLR 2021 (as 'What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study')"
external_ids:
  arxiv: "2006.05990"
  doi: null
  s2: null
tags: ["actor-critic", "shared-trunk", "gradient-interference", "policy-head-init", "ablation-study", "scale-anchoring"]
added: 2026-08-01T00:00:00Z
---

# What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study

## Why this paper was surfaced

The scale-anchoring fix puts a *second* objective — regression onto measured utility — on the same trunk as a policy-gradient objective. The first-order risk is gradient interference between them. This is the largest controlled study of exactly that architectural choice: >50 design choices, >250,000 agents trained, on-policy actor-critic, with the shared-vs-separate trunk isolated as a variable.

## One-line thesis

Across five continuous-control environments, **separate policy and value networks outperform a shared trunk** on four of five, and the scale of the policy head's last-layer initialization is one of the highest-impact choices in the entire study.

## Key Results

- **Separate beats shared.** "Separate value and policy networks appear to lead to better performance on four out of five environments." Their explicit recommendation: *"Use a wide value MLP (no layers shared with the policy) but tune the policy width (it might need to be narrower than the value MLP)."* The value objective wants more capacity than the policy objective does — sharing forces one width on two different needs.
- **Policy-head initialization scale is decisive.** *"The initial policy appears to have a surprisingly high impact on the training performance... initializing the policy MLP with smaller weights in the last layer alone boosts the performance on Humanoid by 66%."* Recommendation: **initialize the last policy layer with 100× smaller weights.** The *scale* at which a policy head starts, independent of what it represents, changes outcomes by tens of percent.
- **Value normalization is strongly environment-dependent.** It is "crucial for good performance on HalfCheetah and Humanoid" and "significantly hurts the performance on Walker2d." Recommendation is to always use observation normalization and to *check* whether value normalization helps — i.e. the target scale of a value head is a live variable with no universally safe setting.
- **Scale of evidence.** >250k agents across five environments; recommendations are ranked by measured effect, not by argument.

## Relevance to This Project

**It is the main cost statement for the auxiliary-head-on-shared-trunk option, and it points the wrong way.** The largest study of this architecture says the policy and value objectives are better off *not* sharing layers. Applied to us: a ΔU-regression head hung off the same trunk as the utility policy head inherits a measured, replicated penalty in the on-policy setting. This does not veto the design — our trunk is also doing feature extraction over documents and a separate trunk doubles that cost, and Farebrother's linear-probe result argues the other way for cross-entropy value objectives — but it means *shared trunk must be an arm with a measured comparison, not the assumed default*.

**It also supplies the cheapest concrete precaution.** If the ΔU head shares the trunk, the paper's ranked findings say the two things to control are (a) the relative *width* of the two heads and (b) the *initialization scale* of the policy head's last layer. The second is directly on-topic: our tower's failure is margins growing to +41..+225 on tie pairs, and a 100×-smaller policy-head init is a one-line, evidence-backed intervention on the initial margin scale that costs nothing and predates any of the softcap/gain machinery we built.

**And it is an honest caution against expecting the anchoring head to be self-normalizing.** Value normalization helped strongly in two environments and hurt strongly in a third; the scale of a value target is not a solved problem even when the target is measured. Our ΔU is in fixed physical units, which is a real advantage over their returns — but the paper is a warning that "the head now outputs real units" does not by itself imply "the head's gradients are well-scaled relative to the policy loss".

## Design question it bears on

Auxiliary regression head on the **shared trunk** vs a **separate** anchoring network (or regressing the policy head itself). This paper is the strongest empirical evidence for separation in on-policy actor-critic, and the source of the two hyperparameters to control if we share anyway.

## Caveats

- Continuous-control MuJoCo with Gaussian policies; ours is a discrete per-menu categorical policy over a shared representation of *documents*, where feature reuse across heads is far more valuable than in a 17-dimensional state space. The transfer of "separate is better" is not automatic.
- Their value head is a bootstrapped critic feeding the advantage estimator — it is *inside* the policy-gradient loop. Our anchoring head would be trained on externally measured ΔU and would not feed the advantage at all, which is a materially different coupling (arguably a safer one).
- The study predates and does not cover the classification-style value losses that the strongest current evidence favours; "separate networks" was measured for MSE critics.
- The four-of-five result is a majority, not a sweep; one environment preferred sharing.

## Sources

- [arXiv 2006.05990](https://arxiv.org/abs/2006.05990) — Andrychowicz et al.
- Complementary evidence on the shared-parameter channel between unrelated decisions: [schaul2022_policy_churn.md](schaul2022_policy_churn.md) ([arXiv 2206.00730](https://arxiv.org/abs/2206.00730)).
- The counter-argument for sharing when the auxiliary loss is cross-entropy: [farebrother2024_stop_regressing_classification.md](farebrother2024_stop_regressing_classification.md) ([arXiv 2403.03950](https://arxiv.org/abs/2403.03950)).
