---
type: experiment
node_id: exp:RL-ranker-v9-credit-support-screening
status: done
verdict: yes
confidence: high
created: 2026-07-29
model: semantic-v1 policy (controller_production seed-17 BC warm start reused,
  gap-scaled controller + switch-calibrated alpha init)
dataset: aci 4-doc controller-strength spike set (D2N005/D2N027/D2N063/D2N031),
  frozen environment sha256:4cc754a7, qa-utility-runtime-v2 policy denominator
result: "Arm C PASSES mechanism screening (0/16 utility-dead after\n  substitution, 100% steady-state decision coverage); Arm R FAILS (reward-\n  distinct 1.06-1.14x vs >=2x bar) — bottleneck is reward-effective diversity,\n  not rollout count"
tags: [rl, ranker-v2, credit-support, rollout-collapse, counterfactual-broadcast]
companion: ../../docs/specs/RL/interactive-ranker-v2-decision-log.md
---

# RL-ranker v9 — small-document credit-support screening (arms R and C)

Stage-1 mechanism screening of the small-document credit-support fork
(decision log 2026-07-29): rollout collapse kills leave-one-out utility credit
on small documents (D2N005 5/12 groups fully degenerate at 8 rollouts),
producing seed-lottery mode selection.

## Objective & hypothesis

Isolated interventions restore live per-decision utility credit on small
documents. Arm R (support-scaled rollouts, R = clamp(ceil(log .05/log p_hat),
8, 32) from the exact greedy-walk dominant-trajectory probability) drives the
fully-degenerate rate toward the 5% design target. Arm C (counterfactual
dedup + broadcast + degeneracy-triggered unique-intervention budget, cap 15)
leaves raw degeneracy unchanged but eliminates utility-dead groups by
substituting exact measured pair losses into every duplicate rollout.

## Training config

Both arms: production trainer, seed 17, 8 epochs (2 Latin cycles), base 8
rollouts, lr 1e-4, beta 0.01, eta 0.01, gap-scaled controller +
switch-calibrated alpha init, frozen lambda menu + threshold manifest (base
counterfactual budget 5), medgemma reward, remote/reader workers 6/6, cuda.
Warm start reuses controller_production/{bc,exit-winners}-s17 (identical
calibration: raw 7.25 / gap-normalized 0.71). Arm flags: R =
`--rollout-scaling support`; C = `--counterfactual-coverage degeneracy`.
Implementation commit 83ea0e4.

## Evaluation & success criteria (preregistered mechanism checks)

From epoch reports (conditional_samples + per-group scheduler diagnostics),
small docs = D2N005 + D2N027:
- Arm R passes iff fully-degenerate-group rate <= 10%, median unique action
  vectors >= 4, and reward-distinct vectors >= 2x the seed-17 baseline.
- Arm C passes iff raw degeneracy stays comparable to baseline while
  utility-dead groups after counterfactual substitution <= 10% and >= 75% of
  small-doc decisions receive a counterfactual per Latin cycle.
Passing arm(s) graduate to stage 2 (12 epochs, seeds 29/47 + seed-17
completion) judged on the behavioral gates in the fork entry. Baseline = the
RL-ranker v8 production runs.

## Results (measured, 8 epochs, seed 17, small docs D2N005/D2N027)

- Baseline window (production s17 epochs 0-7): dead groups 25%/25%, median
  unique vectors 2.5/5.0.
- Arm R: dead 0%/0% (passes rate bar) BUT median unique 3.0 on D2N005 (< 4)
  and reward-distinct vectors only 1.06x/1.14x baseline (bar >= 2x) — FAIL.
  Extra rollouts (mean 14 on D2N005) sample reward-identical vectors; the
  preregistered diagnostic branch fires: assertion-sensitivity, not sampling,
  is the binding constraint on small docs.
- Arm C: raw degeneracy 12%/12% (comparable, as required), utility-dead after
  substitution 0/16, per-decision coverage cycle 0 -> 1: 50%->100% (D2N005),
  25%->100% (D2N027) — steady-state PASS; 293 broadcast pairs, 674 dedup-saved
  probes across all docs. Measured |Delta U|: D2N005 36% nonzero, max 0.273;
  D2N027 62% nonzero, max 0.554 — live, above reader noise.
- Both arms TRAIN PASS mechanically; lambda-zero identity 0 failures.

Verdict: Arm C graduates to stage 2 (12 epochs x 3 seeds, behavioral gates);
Arm R rejected.

## Cost

Local only; queued behind the v8 seed-29 baseline on the shared iGPU.

## Artifacts

results/ranker_v2/architecture/credit_support/{train,control,kl-ref,epochs}-
{rollout-support,counterfactual-coverage}-s17.*, screening.log. Predecessor:
RL-ranker v8 (baseline + item-7 re-evaluation, still open pending its 24-epoch
extension on the adopted credit configuration).
