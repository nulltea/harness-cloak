---
type: paper
node_id: paper:farebrother2024_stop_regressing_classification
title: "Stop Regressing: Training Value Functions via Classification for Scalable Deep RL"
authors: ["Jesse Farebrother", "Jordi Orbay", "Quan Vuong", "Adrien Ali Taiga", "Yevgen Chebotar", "Ted Xiao", "Alex Irpan", "Sergey Levine", "Pablo Samuel Castro", "Aleksandra Faust", "Aviral Kumar", "Rishabh Agarwal"]
year: 2024
venue: "ICML 2024 (PMLR 235:13049-13071)"
external_ids:
  arxiv: "2403.03950"
  doi: null
  s2: null
tags: ["value-regression", "hl-gauss", "cross-entropy", "representation-grounding", "plasticity", "scale-anchoring"]
added: 2026-08-01T00:00:00Z
---

# Stop Regressing: Training Value Functions via Classification for Scalable Deep RL

## Why this paper was surfaced

If we add a head that regresses measured utility, the obvious implementation is an MSE head — and the obvious objection is that MSE value heads are notoriously brittle on noisy, non-stationary targets, which is exactly what our measured ΔU is (0.044 measurement-resolution floor, 39% of pairs at or below it). This paper is the load-bearing evidence on *how* to make a value-regression head work at scale, and on what a regression head does to the shared representation it sits on.

## One-line thesis

Replace MSE value regression with cross-entropy over a **fixed categorical support in the target's own units** (HL-Gauss: project the scalar target onto bins as a Gaussian, then minimize cross-entropy); this keeps the head's output denominated in real units while behaving like a classification loss, and it improves performance and scalability across every domain tested.

## Key Results

- **HL-Gauss construction.** Fixed support of `m` evenly-spaced bins over `[v_min, v_max]`, bin width `ς`. The scalar target is smeared as a Gaussian of standard deviation σ and integrated over each bin; loss is cross-entropy against the predicted categorical. Recommended hyperparameter is the **ratio** `σ/ς`, default `0.75`, which spreads mass over ~6 neighbouring bins. Prediction is the support-weighted mean — still a number in target units.
- **Three measured mechanisms** (diagnostic experiments, not speculation): **(1) robustness to noise** — HL-Gauss degrades more slowly than MSE as reward corruption increases; **(2) better representations** — linear probes on frozen features show cross-entropy-trained trunks retain more capacity for downstream value learning; **(3) plasticity under non-stationarity** — in offline SARSA with *stationary* targets "most of the benefit from HL-Gauss compared to MSE vanishes", isolating non-stationarity as the source of the gain.
- **Magnitudes.** ~30% IQM improvement over MSE with SoftMoE on 20 Atari games; 1.8-2.1× on multi-task offline Atari with ResNet-101 (40 games); 67% peak improvement in robotic manipulation with Q-Transformer; 40% better Wordle success rate; 70% of the gap to AlphaZero-with-search closed on chess without search.
- **The support is a required, stated constant.** `[v_min, v_max]` must be chosen; the method does not discover the range. This is a knob, but an *auditable* one in the target's units.

## Relevance to This Project

**It removes the strongest practical objection to the scale-anchoring fix.** "Regress the head on measured ΔU" invites the reply that MSE on a noisy, drifting signal will destabilize a trunk that is simultaneously carrying a policy-gradient objective. This paper's answer is that the *loss form*, not the *idea of regression*, is what fails: a cross-entropy-over-fixed-support head keeps the units and drops the MSE pathologies, and the measured mechanism (plasticity under non-stationary targets) is precisely our regime — our ΔU targets come from a live reader and the trunk is being moved by an RL objective simultaneously.

**It supplies the specific construction for a ΔU head.** Our anchoring target is a *difference* with a known measurement floor of 0.044 and a bounded plausible range, so the fixed support is not a guess: it is set from the utility scale, and the smoothing width σ can be pinned to the measurement resolution rather than tuned. That is an unusually clean fit — the hyperparameter the paper says to tune (`σ/ς`) has a physical value here, which means the anchoring head introduces a constant that is *measured*, not calibrated, satisfying the project's no-calibration-knob rule.

**It is also the strongest evidence for the auxiliary-head-on-shared-trunk direction, from the representation side.** The linear-probe result says a cross-entropy value objective *improves* the shared representation's usefulness rather than merely competing with the primary loss. That is the specific benefit an auxiliary regression head would need to deliver to justify the interference risk of sharing a trunk with the policy objective.

**And it quietly argues the anchored head should be the auxiliary one, not the policy head itself.** HL-Gauss outputs a distribution over a support and reads out a mean; that is not a softmax over actions and cannot be the action policy. Anchoring by this route necessarily means a *second* head, sharing the trunk, whose output the policy head is asked to track — not a reinterpretation of the policy logits.

## Design question it bears on

If we anchor, what loss does the anchoring head use? Answer from this paper: categorical cross-entropy over a fixed support in utility units (HL-Gauss), not MSE — with σ set from the 0.044 measurement floor and the support set from the utility range. It also bears on the shared-vs-separate-trunk question, on the "sharing helps" side.

## Caveats

- Every experiment is a *bootstrapped* value function in sequential RL. Our ΔU targets are directly measured, not bootstrapped, so the non-stationarity mechanism that drives most of the gain is weaker for us (our non-stationarity comes from the trunk moving, not from the target regressing on itself). The offline-SARSA ablation says the benefit shrinks when targets are stationary — an honest discount on the expected size of the win.
- No experiment in the paper has the regression head coexisting with a policy-gradient loss on the same trunk; the interference question is out of scope here.
- The support range is a real hyperparameter with a real failure mode (targets outside `[v_min, v_max]` are clipped by construction).
- Data scale is enormous (Atari, chess, robotics). Nothing here speaks to a 63-document training set.

## Sources

- [ICML 2024 / PMLR 235:13049-13071](https://proceedings.mlr.press/v235/farebrother24a.html) — Farebrother et al.
- [arXiv 2403.03950](https://arxiv.org/abs/2403.03950)
