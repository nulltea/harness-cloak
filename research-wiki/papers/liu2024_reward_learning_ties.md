---
type: paper
node_id: paper:liu2024_reward_learning_ties
title: "Reward Learning From Preference With Ties"
authors: ["Jinsong Liu", "Dongdong Ge", "Ruihao Zhu"]
year: 2024
venue: "arXiv preprint (no published venue found)"
external_ids:
  arxiv: "2410.05328"
  doi: null
  s2: null
tags: ["ties", "reward-modeling", "bradley-terry", "rao-kupper", "preference-strength", "bias", "equivalence-critic"]
added: 2026-07-31T00:00:00Z
---

# Reward Learning From Preference With Ties

## Why this paper was surfaced

It is the reward-modeling (not policy-optimization) counterpart to the DPO-with-ties work, and it answers the question our critic design actually hinges on: what happens to learned reward *differences* when tied pairs are forced into strict win/loss labels. Since q_U is trained on and evaluated by differences, a theorem quantifying the distortion of exactly that quantity is the closest thing in the literature to a specification of the error we would commit by ignoring our tie strata.

## One-line thesis

If the true preference process admits ties (Rao-Kupper / "BTT") but the dataset forces every pair into a strict preference, then even with infinite data and the true prompt/response distributions the maximum-likelihood Bradley-Terry reward model systematically *attenuates* preference strength Δr by a closed-form, sigmoid-shaped bias term; modeling ties (or subtracting the bias) removes it.

## Key Results

- **BTT = Rao-Kupper in reward form.** p(y₁ ≻ y₂) = e^{r₁} / (e^{r₁} + θ e^{r₂}) and p(y₁ = y₂) = (θ² − 1) e^{r₁} e^{r₂} / [(e^{r₁} + θ e^{r₂})(θ e^{r₁} + e^{r₂})], θ ≥ 1; θ = 1 gives Bradley-Terry. Larger θ means more tie mass.
- **Theorem 4.2 (what BT converges to).** Fitting BT to a tie-broken dataset converges to r̂ with p_r̂(y₁ ≻ y₂) = p^θ_{r*}(y₁ ≻ y₂) + ½ p^θ_{r*}(y₁ = y₂): the BT model absorbs half of each tie's mass into each direction. Ties are not ignored, they are *smeared into the preference probabilities*, which is what shifts the differences.
- **Theorem 4.3 (the bias, in closed form).** Δr̂ = Δr* + log[ (2θ + (1+θ²) e^{−Δr*}) / (1 + θ² + 2θ e^{−Δr*}) ]. The bias has sign opposite to Δr*, so preference strength is attenuated (shrunk toward zero); it is bounded in absolute value by log((1+θ²)/(2θ)).
- **The bound is not a reassurance.** Over the range in which 83.6% of HH-RLHF mean preference strengths fall (Δr* ∈ [−0.6, 2.94]), the ratio |bias| / |Δr*| reaches roughly 0.1 at θ = 2, ~0.45 at θ = 5, and ~0.65 at θ = 10 (Fig. 1) — i.e. relative distortion of tens of percent in the regime that matters.
- **Correction algorithm.** Since the bias is monotone in Δr*, they invert it: at each step, solve the nonlinear equation for Δr* given the current Δr_ψ and subtract the bias from the loss margin. They note this is a variant of adaptive-margin reward modeling / ODPO-style offset DPO.
- **Empirics.** (i) Synthetic ground-truth reward: BTT-trained reward models have consistently smaller preference-strength bias than BT-trained ones, gap growing with θ (Δ = 0.0206 / 0.0237 / 0.0353 at θ = 2 / 5 / 10). (ii) DPO with bias-correction offset on HH-RLHF (Pythia-160M, one epoch, reward preference accuracy): 0.5333 at θ = 1 (≡ plain DPO) vs 0.5583 / 0.6042 / 0.5958 at θ = 2 / 5 / 10 — >10% relative gain at the optimal θ. (iii) Ties in HH-RLHF are labeled by Llama3-70B and Qwen2-72B; the paper documents real HH-RLHF pairs whose mean preference strength across 10 independently trained reward models is ~0.0 with SD 0.22-0.36, i.e. pairs that are ties in all but the label.
- **Acknowledged limitation.** No real human-labeled tie dataset exists at scale; all tie labels here are LLM-simulated.

## Relevance to This Project

**It names the exact error mode of the strict-preference framing.** If we were to convert our sub-noise stratum into signed preference labels, Theorem 4.3 says the learned differences would be *shrunk* by a factor that depends on the tie rate — with a 39% tie-class share, θ is large, and the attenuation is in the tens of percent. A critic whose differences are attenuated is a critic that under-separates action pairs it should separate, which is the exact pathology we already observed in the v13 gain-head degeneracy (differentiation never emerging). This is the strongest argument in the literature against a signed-preference-only formulation for our data.

**It also, read carefully, argues that plain difference regression is *not* the thing being indicted.** The bias in Theorem 4.3 arises from label forcing: the tie's probability mass gets split ½/½ into win/loss because the *label space* has no tie symbol. Regression on measured ΔU has no such forcing — ΔU = 0 is directly representable and there is no probability mass to smear. So this paper's headline result is an argument against strict-preference training on tied data, not against regression. What regression does wrong is different and milder: it treats the sub-noise sign as a target rather than as noise.

**Their correction is a margin offset — the same object as Rao-Kupper's α.** Their fix, framed as "adaptive margin"/ODPO offset, is structurally the same as the tie threshold: shift the effective margin. That convergence across two independent papers is why the tie threshold is the right single parameter to expose in our loss, and why the value should come from our measured 0.044 rather than a sweep.

**Their θ is fitted, ours is measured.** They sweep θ ∈ {1, 2, 5, 10} and pick the best by test accuracy — a fitted nuisance parameter. In our setting the tie threshold is a property of the reader-based utility measurement (0.044), pinned before training. Under our empirical-honesty rule this matters: a swept θ tuned per arm would be exactly the per-model calibration knob we forbid, while a measured threshold held fixed across all compared arms is legitimate.

## Design question it bears on

Whether the tie strata need a dedicated likelihood term (b/c) or can be absorbed as regression targets (a). The paper supplies the quantitative cost of the worst option — forcing ties into signed preferences — and supplies the bias-correction/margin-offset view that makes the tie threshold interpretable. It does not test regression on measured utility gaps, so it does not directly rule for or against (a).

## Caveats

- Empirical scale is weak: Pythia-160M, one epoch, reward accuracy 0.53 at baseline (barely above chance), and a 2.8B follow-up at only the best θ. The theory is the contribution; the numbers are illustrative.
- All tie labels are LLM-simulated, and the tie *generation* assumption (Assumption 4.1: annotators break ties uniformly at random) is what drives the ½/½ smearing in Theorem 4.2. Our ties are not annotator-broken at all, so the bias theorem applies to us only in the hypothetical where we convert sub-noise ΔU into signed labels.
- No published venue found; treat as a preprint.
- θ is a global constant across all pairs; a measurement-noise-derived threshold is arguably state-dependent (our reader noise floor could vary by document), which neither this paper nor the classics address.

## Sources

- [arXiv 2410.05328](https://arxiv.org/abs/2410.05328) — Liu, Ge, Zhu. *Reward Learning From Preference With Ties.*
- Tie model it adopts: [raokupper1967_ties_paired_comparisons.md](raokupper1967_ties_paired_comparisons.md) (DOI [10.1080/01621459.1967.10482901](https://doi.org/10.1080/01621459.1967.10482901)).
- Policy-side counterpart with stronger empirics: [chen2025_dpo_ties.md](chen2025_dpo_ties.md) ([arXiv 2409.17431](https://arxiv.org/abs/2409.17431)).
