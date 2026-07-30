---
type: training-experiment
status: running
created: 2026-07-30
model: semantic-v1 policy (controller_production BC/ExIt warm starts reused,
  gap-scaled controller + switch-calibrated alpha init, Arm C counterfactual
  broadcast; per-arm addition — utility-logit soft-cap OR profile-sensitivity
  regularizer)
dataset: aci 4-doc controller-strength spike set (D2N005/D2N027/D2N063/D2N031),
  frozen environment sha256:4cc754a7, qa-utility-runtime-v2 policy denominator
result: pending
tags: [rl, ranker-v2, tie-ownership, logit-softcap, sensitivity-regularizer]
companion: ../../docs/research/reward-ties-and-controller-authority.md
---

# RL-ranker v12 — tie-ownership screening: logit soft-cap vs sensitivity regularizer

Screening the two approved arms of the tie-ownership fork (decision log
2026-07-30; taxonomy and literature in the companion research doc). Both are
capacity-free, single-constant mechanisms; the learned controller gain
remains the preregistered escalation.

## Arms

- **soft-cap**: `--utility-logit-softcap 25` — margins can never exceed 50
  (the preregistered saturation gate), so the calibrated controller shift
  deterministically owns reward ties. One auditable constant; production
  precedent Gemma 2 (cap-tanh).
- **sensitivity**: `--profile-sensitivity-reg 0.1` — per-decision adjacent-
  profile KL is pulled toward target·Δg, with the target measured from the
  calibrated warm start (KL per unit g of the controller ramp). Profile
  separation becomes an explicit objective; gradient reaches alpha and the
  utility shape. Coefficient 0.1 is this run's single hyperparameter choice.

Both arms: production trainer, 8 epochs, seeds 47 then 17 (collapsed mode
first), Arm C counterfactual coverage, gap-scaled controller, switch-
calibrated alpha init, NO KL-anchor flags (collapse-trigger default),
synchronous profile eval on, early kill on definitive gate failure at
cycle 1. Implementation commit 42c0db8.

## Evaluation & success criteria (preregistered)

Per arm on the synchronous readout: lambda-zero converges (per-doc utility
loss <= 0.044 vs the fixed control, and lambda-zero P falls toward the
unanchored trajectory rather than pinning — the v11 failure); D2N005
synchronous Delta P(lambda-3) >= 0.20 by the final cycle with cycle range
<= 0.10; cross-seed final difference <= 0.10; no menu-logit range > 50 and no
persistent action probability > 0.999; frontier regret reported (item-7
gate); flat-decision expected privacy >= 0.50 at lambda-3.

## Results

pending

## Cost

Local only, shared iGPU, 4 chains sequential (~1.5h each, cache-backed).

## Artifacts

results/ranker_v2/architecture/tie_ownership/{train,control,kl-ref,epochs}-
{softcap,sensitivity}-s{47,17}.*, screening logs. Predecessors: RL-ranker v11
(KL anchor, both directions failed by over-anchoring), v10 (Arm C stage 2).
