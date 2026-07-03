---
type: paper
node_id: paper:gao2022_reward_overoptimization
title: "Scaling Laws for Reward Model Overoptimization"
authors: ["Leo Gao", "John Schulman", "Jacob Hilton"]
year: 2022
venue: "arXiv"
external_ids:
  arxiv: "2210.10760"
  doi: null
  s2: null
tags: ["rlhf", "reward-modeling", "goodhart", "overoptimization", "rl-background", "d2"]
added: 2026-07-02T19:40:00Z
---

# Scaling laws for reward model overoptimization

## One-line thesis

Optimizing a policy against a frozen proxy reward model improves the proxy score monotonically
while the *gold* (true) reward rises then falls — Goodhart's law, measured: the divergence follows
smooth functional forms in the KL distance from the initial policy, with coefficients scaling in
reward-model size and data.

## Problem / Gap

RLHF optimizes against a learned, frozen reward model that imperfectly proxies the true objective;
"optimizing too hard hurts" was folklore, not measured (human labels too expensive to probe it).

## Method

Synthetic gold-standard setup: a large fixed "gold" reward model plays the human; a smaller proxy
reward model is trained on gold-labeled preferences; a policy is then optimized against the proxy
by (a) best-of-n sampling and (b) RL (PPO). Gold score is tracked as a function of
`KL(policy ‖ init)`.

## Key Results

- Gold reward follows `d(α_bon − β_bon·d)` for best-of-n and `d(α_RL − β_RL·log d)` for RL, where
  `d = √KL(policy‖init)` — i.e. **rises, peaks, then degrades** while proxy reward keeps climbing.
- Coefficients scale smoothly with proxy-RM parameter count and dataset size: bigger/better-fed
  proxies push the peak further out but never remove it.
- RL is less KL-efficient than best-of-n at extracting proxy reward — but both Goodhart.
- KL penalty studied as the standard mitigation knob.

## Assumptions

- The proxy is *frozen* during optimization (no re-labeling loop) — exactly the setting of any
  frozen reward head or frozen judge.
- Synthetic gold model stands in for ground truth.

## Limitations / Failure Modes

- Synthetic gold ≠ real human preference drift; absolute numbers don't transfer, the *shape* does.
- Single-domain (preference reward); coefficients are not portable to other reward types.

## Reusable Ingredients

- The canonical citation and functional form for "training against a frozen proxy Goodharts".
- KL-vs-gold-reward curve as the standard diagnostic plot for proxy-reward training runs.
- Best-of-n as a lower-optimization-pressure alternative that Goodharts more slowly per KL.

## Open Questions

- How fast does overoptimization set in for *rule-based/task-grounded* proxies (vs learned RMs)?
  Directly relevant to our model-free surrogate options.

## Claims

_None registered._

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

**Why surfaced:** background for the round-trip GRPO plan
([`2026-07-02-roundtrip-grpo-training.md`](../../docs/plans/2026-07-02-roundtrip-grpo-training.md)):
the plan trains a substitutor against frozen reward components (frozen remote task model, frozen
MTI attack head, frozen utility proxy). This paper is the measured demonstration that *any* frozen
proxy Goodharts under enough optimization pressure, and supplies the mitigation vocabulary the
plan uses: KL leash, held-out (different-family) evaluation models, and low-pressure optimizers
(best-of-n) as slower-Goodharting alternatives. It is the reason the plan treats "policy exploits
the local attack head's blind spots" and "policy overfits Qwen3.6's quirks" as one phenomenon with
one playbook.

## Abstract (original)

> In reinforcement learning from human feedback, it is common to optimize against a reward model
> trained to predict human preferences. Because the reward model is an imperfect proxy, optimizing
> its value too much can hinder ground truth performance, in accordance with Goodhart's law. This
> effect has been frequently observed, but not carefully measured due to the expense of collecting
> human preference data. In this work, we use a synthetic setup in which a fixed "gold-standard"
> reward model plays the role of humans, providing labels used to train a proxy reward model. We
> study how the gold reward model score changes as we optimize against the proxy reward model
> using either reinforcement learning or best-of-n sampling. We find that this relationship
> follows a different functional form depending on the method of optimization, and that in both
> cases its coefficients scale smoothly with the number of reward model parameters. We also study
> the effect on this relationship of the size of the reward model dataset, the number of reward
> model and policy parameters, and the coefficient of the KL penalty added to the reward in the
> reinforcement learning setup. We explore the implications of these empirical results for
> theoretical considerations in AI alignment.
