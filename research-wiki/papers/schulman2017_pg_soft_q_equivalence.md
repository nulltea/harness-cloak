---
type: paper
node_id: paper:schulman2017_pg_soft_q_equivalence
title: "Equivalence Between Policy Gradients and Soft Q-Learning"
authors: ["John Schulman", "Xi Chen", "Pieter Abbeel"]
year: 2017
venue: "arXiv (no published venue)"
external_ids:
  arxiv: "1704.06440"
  doi: null
  s2: null
tags: ["policy-gradient", "soft-q-learning", "entropy-regularization", "logit-semantics", "shift-invariance", "scale-anchoring"]
added: 2026-08-01T00:00:00Z
---

# Equivalence Between Policy Gradients and Soft Q-Learning

## Why this paper was surfaced

The scale-anchoring hypothesis rests on a claim about semantics: that policy logits trained by policy gradient are *not* value estimates, and their differences carry no units. This paper is the precise statement of when that claim is false — i.e. the exact conditions under which a policy's logits *do* equal values in reward units — and therefore the sharpest available diagnosis of what our utility tower is missing.

## One-line thesis

Under entropy (KL-to-reference) regularization with temperature τ, the optimal policy is `π(a|s) ∝ π̄(a|s)·exp(Q(s,a)/τ)`, so soft Q-learning and policy gradient are the same algorithm in different coordinates; equivalently `Q_θ(s,a) = V_θ(s) + τ·log(π_θ(a|s)/π̄(a|s))` — the logits are Q-values divided by τ, determined only up to the per-state additive constant `V_θ(s)`.

## Key Results

- **Exponential-family form (Eq. 28).** `π(a|s) ∝ π̄(a|s) exp(Q(s,a)/τ)`. Logit differences within a state equal **advantage differences divided by τ**. Units exist — but only through τ.
- **Per-state additive freedom (Eq. 57).** `Q_θ(s,a) = V_θ(s) + τ log(π_θ(a|s)/π̄(a|s))`. The state-value term is an arbitrary additive constant per state that cancels in the policy; the policy identifies Q only up to it. This is exactly the "identifiable only up to a within-menu additive shift" property we measured, stated as theory rather than as a bug.
- **τ is a rescaling of reward, not a free stability knob.** The paper notes τ "can be eliminated by rescaling the rewards" and is kept only for dimensional clarity. Temperature and reward scale are the same degree of freedom — you cannot tune one for stability without changing what a logit means.
- **The equivalence is a fixed-point statement.** It characterizes the entropy-regularized *optimum*; nothing asserts that a partially-converged policy-gradient network's logits approximate `Q/τ` at any point during training.

## Relevance to This Project

**It says the anchoring we want is not exotic — it is the property an entropy-regularized policy has *at its fixed point*, and it explains precisely why ours doesn't have it.** Three conditions have to hold for `u(a_i)−u(a_j)` to mean "(U(a_i)−U(a_j))/τ": the objective must be entropy/KL-regularized toward a fixed reference with a **known, fixed τ**; the policy must be near its fixed point; and the reward must be the quantity you want the units of. Our tower fails all three usefully. The entropy coefficient is a small fixed bonus (β=0.01) that the entropy-collapse literature says is structurally outrun, the KL anchor is intermittent and directional, and the LOO advantage signal is not a soft-Bellman target — so the paper's mapping simply does not apply, and the additive per-menu freedom (which is *real and unavoidable*, Eq. 57) is the only part of the picture we inherit.

**It sharpens what scale anchoring buys.** The additive within-menu shift is not a defect to be removed — it is a genuine invariance of any softmax policy, and a regression head will not eliminate it. What anchoring can fix is the *scale* (the τ), not the *offset*. That is the right target anyway: our controller `z(a) = u(a) + α·g(λ)·p̂(a)` is itself shift-invariant per menu, so only the margin scale ever competed with the bounded shift. The correct framing of the fix is therefore "fix τ" — either by regressing differences onto measured ΔU (setting τ = 1 utility unit) or by a calibrated likelihood with a fixed temperature — not "remove the shift".

**It also predicts a weak free anchor we may already have.** Entropy regularization pulls tied actions toward equal logits (the fixed point for `Q(a_i)=Q(a_j)` is margin 0), which is the one force in our current objective that opposes tie-margin drift. That it lost — margins reached +41..+225 on tie pairs — is measured evidence that β=0.01 is far below any τ at which the equivalence would bind.

## Design question it bears on

Whether the fix is (i) making the existing entropy/KL regularization principled enough that the soft-Q equivalence supplies units for free, or (ii) adding an explicit cardinal regression signal. This paper makes (i) a legitimate option in principle — with the caveat that it requires τ large enough to bind and a genuine soft-Bellman target, both of which change the training objective more than adding a regression head does.

## Caveats

- Preprint, no venue. Widely cited but the equivalence is a theoretical result on the entropy-regularized fixed point, with limited experiments.
- The equivalence holds at optimality; it gives no guarantee about mid-training logits, which is where all our pathologies live.
- The setting is sequential RL with bootstrapping. Our per-menu decision is a contextual bandit, which makes the mapping *simpler*, not weaker — but the paper does not treat the bandit case explicitly.

## Sources

- [arXiv 1704.06440](https://arxiv.org/abs/1704.06440) — Schulman, Chen, Abbeel.
- Related entropy-mechanism evidence in this wiki: [cui2025_entropy_mechanism_rl.md](cui2025_entropy_mechanism_rl.md) ([arXiv 2505.22617](https://arxiv.org/abs/2505.22617)); [haarnoja2018_sac_algorithms_applications.md](haarnoja2018_sac_algorithms_applications.md) ([arXiv 1812.05905](https://arxiv.org/abs/1812.05905)).
