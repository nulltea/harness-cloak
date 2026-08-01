---
type: paper
node_id: paper:wang2024_helpsteer2_preference
title: "HelpSteer2-Preference: Complementing Ratings with Preferences"
authors: ["Zhilin Wang", "Alexander Bukharin", "Olivier Delalleau", "Daniel Egert", "Gerald Shen", "Jiaqi Zeng", "Oleksii Kuchaiev", "Yi Dong"]
year: 2024
venue: "ICLR 2025"
external_ids:
  arxiv: "2410.01257"
  doi: null
  s2: null
tags: ["reward-modeling", "regression-vs-preference", "scaled-bradley-terry", "calibration", "scale-anchoring"]
added: 2026-08-01T00:00:00Z
---

# HelpSteer2-Preference: Complementing Ratings with Preferences

## Why this paper was surfaced

This is the only head-to-head, matched-data comparison we found between a reward head trained by **regression onto absolute measured scores** and one trained by **preference likelihood only** — the exact fork the scale-anchoring hypothesis turns on. It also contains the closest published relative of our counterfactual pair loss, which lets us say precisely what our loss does and does not inherit.

## One-line thesis

Bradley-Terry and regression reward models had never been compared on matched data; when they are, regression is competitive or better, a **magnitude-scaled** BT loss beats plain BT, and the best model is a *composition* — a regression model used to initialize a scaled-BT model.

## Key Results

- **Scaled BT.** `L_SBT = −m · log σ(r(x,y_c) − r(x,y_r))`, where `m ∈ {1,2,3}` is the annotated preference magnitude. The margin **multiplies the loss** — it reweights how hard each pair is pushed, but imposes no constraint on the size of the reward gap.
- **Margin BT.** `L_MBT = −log σ(r(x,y_c) − r(x,y_r) − m)`, the margin **inside** the sigmoid, which demands the reward gap exceed `m` — i.e. it is a scale constraint, not a reweighting.
- **RewardBench head-to-head (matched data, from scratch).** SteerLM Regression 93.0 > Scaled BT 92.7 > plain BT 91.5 = Margin BT 91.5. Regression on absolute ratings is the single best from-scratch objective; scaling BT by magnitude recovers most of the gap; putting the margin *inside* the sigmoid buys nothing over plain BT.
- **Composition wins.** Initializing Scaled BT from the trained Helpfulness-only regression model (plus ExPO) reaches **94.1**, first of 140+ reward models as of 1 Oct 2024; the resulting RM drives REINFORCE RLHF to 85.0 on Arena Hard.
- **Paradigm-by-purpose.** The paper states regression models suit **absolute-threshold filtering and interpretability**, while BT models maximize RLHF ranking accuracy — the two objectives buy different properties and the paper recommends having both.

## Relevance to This Project

**Our counterfactual pair loss is Scaled BT with the log removed — and that is the whole problem.** `−ΔU·(q−0.5)` weights the ordinal push by measured magnitude, exactly like `L_SBT`'s `m·`, and like `L_SBT` it never asks the margin to *equal* anything. The paper's own results say magnitude-as-weight is a real improvement over plain BT (92.7 vs 91.5) but still loses to regression on absolute values (93.0) — so our loss sits on the correct side of the plain-BT line and the wrong side of the regression line. That is a directly measured, matched-data verdict on the design we currently have.

**Margin BT is the negative control for "just constrain the gap".** Pushing the margin inside the sigmoid — the closest thing in the literature to "make margins mean something by enforcing a minimum gap" — scored identically to plain BT (91.5). Whatever anchoring buys, it is not bought by a hinge on the margin. That is direct evidence against the evidence-hinge family of fixes we have been building, and evidence for regressing values rather than constraining differences.

**The winning recipe is composition, not replacement.** The best configuration keeps *both* objectives, but in sequence: learn the scale first by regression, then refine the ordering by scaled preference training on top. Mapped to our tower, that is "pretrain/co-train the head against measured utility so its output is denominated in utility units, then let the policy-gradient signal sharpen the ordering inside that scale" — and it is evidence that the two objectives are complementary rather than mutually destructive, at least when they are staged rather than summed.

**Its filtering claim is exactly our controller's requirement.** "Regression models suit absolute-threshold filtering" is the property we need and do not have: our design wants "margin below the 0.044 measurement floor ⇒ measured tie ⇒ controller owns the decision". That is an absolute threshold on the head's output, and this paper's own recommendation is that only a regression-trained head supports it.

## Design question it bears on

Whether to keep the ΔU-weighted ordinal loss, replace it with regression onto measured ΔU, or compose them — and whether an alternative "enforce a margin" formulation would do instead. The measured answer: compose (regression first, magnitude-weighted ordinal on top); do **not** rely on a margin hinge.

## Caveats

- Domain is LLM helpfulness reward modelling with **human-annotated integer magnitudes** (1-3), not continuous instrument-measured utility differences with a known noise floor. Our magnitudes are better-behaved in units but noisier near zero.
- The gap between regression and Scaled BT is 0.3 points on RewardBench — small, single benchmark, no confidence intervals reported in what we verified. The safe reading is "regression is at least as good, and composition is clearly best", not "regression dominates".
- The winning number (94.1) includes ExPO weight extrapolation, which is orthogonal to the objective comparison and inflates it.
- No ties. Their magnitudes start at 1; the "measured equal" case that dominates our problem (18% exact, +21% within the floor) has no counterpart in this dataset.
- 70B-scale models and a large preference set; nothing here addresses regression capacity on our data scale.

## Sources

- [arXiv 2410.01257](https://arxiv.org/abs/2410.01257) — Wang, Bukharin, Delalleau, Egert, Shen, Zeng, Kuchaiev, Dong. ICLR 2025.
- The theory-side companion on why the likelihood family is a free choice: [sun2024_rethinking_bradley_terry.md](sun2024_rethinking_bradley_terry.md) ([arXiv 2411.04991](https://arxiv.org/abs/2411.04991)).
- Tie-aware objectives neither paper covers: [liu2024_reward_learning_ties.md](liu2024_reward_learning_ties.md) ([arXiv 2410.05328](https://arxiv.org/abs/2410.05328)); [chen2025_dpo_ties.md](chen2025_dpo_ties.md) ([arXiv 2409.17431](https://arxiv.org/abs/2409.17431)).
