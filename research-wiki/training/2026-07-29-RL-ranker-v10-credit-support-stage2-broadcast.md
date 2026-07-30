---
type: training-experiment
status: done
created: 2026-07-29
model: semantic-v1 policy (per-seed BC warm starts, gap-scaled controller +
  switch-calibrated alpha init, counterfactual dedup/broadcast/coverage)
dataset: aci 4-doc controller-strength spike set (D2N005/D2N027/D2N063/D2N031),
  frozen environment sha256:4cc754a7, qa-utility-runtime-v2 policy denominator
result: "Mixed: D2N027 seed-stable (SD 0.036 PASS), lambda-zero\n  non-inferior, regret tightens 0.064-0.069 (still > 0.044); D2N005 fails\n  primary gate (SD 0.190) — residual variance is switch-quantum discreteness +\n  one fully reward-flat decision, not dead credit"
tags: [rl, ranker-v2, credit-support, counterfactual-broadcast, multi-seed]
companion: ../../docs/specs/RL/interactive-ranker-v2-decision-log.md
---

# RL-ranker v10 — credit-support stage 2: Arm C behavioral confirmation

Stage 2 of the small-document credit-support fork. Arm C (counterfactual
dedup + broadcast + degeneracy-triggered coverage) passed mechanism screening
in RL-ranker v9 (0/16 utility-dead groups after substitution, 100%
steady-state per-decision coverage); Arm R was rejected (reward-identical
rollouts, 1.06-1.14x reward-distinct vs the >= 2x bar).

## Objective & hypothesis

With exact per-decision credit alive on small documents, the seed lottery
closes: small-doc Delta P(lambda-3) becomes seed-stable, and frontier regret
improves toward the item-7 floor.

## Training config

Production trainer, 12 epochs (3 Latin cycles), seeds 17/29/47, base 8
rollouts, `--counterfactual-coverage degeneracy`, gap-scaled controller +
switch-calibrated alpha init, frozen lambda menu + threshold manifest (base
budget 5, degeneracy-widened to 5*ceil(D/5) cap 15), lr 1e-4, beta 0.01,
eta 0.01, medgemma reward, cuda. Seeds 17/29 reuse controller_production
BC/ExIt warm starts (exact baseline comparability); seed 47 rebuilds BC +
ExIt first (its baseline chain was cancelled per scope decision).
Implementation commit 83ea0e4; screening adjudication 4e40afb.

## Evaluation & success criteria (preregistered behavioral gates, all 3 seeds)

- Small-doc Delta P(lambda-3) sample SD <= 0.07 and cross-seed range <= 0.15
  (baseline: SD ~0.24 / range 0.48 on D2N005, SD ~0.14 / range 0.26 on D2N027).
- Controller-fork responsiveness items still pass (monotone P, lambda-zero
  exact identity, placeholder < 95%, no non-finite values).
- Conditional lambda-zero utility within 0.044 of the fixed control.
- Frontier regret (item 7, median <= 0.044 from cycle >= 1) reported per seed —
  adjudicated for the controller-strength fork, not this one.

## Results (measured, 12 epochs x seeds 17/29/47)

- All seeds TRAIN PASS; lambda-zero identity 0 failures; monotone profile
  response every seed; conditional lambda-zero utility gap to fixed control
  0.003/0.041/0.040 (all within the 0.044 floor).
- D2N027 final-cycle Delta P(lambda-3): +0.10/+0.14/+0.07 — SD 0.036 PASS,
  range 0.072 PASS (baseline SD ~0.14, range 0.26).
- D2N005: +0.42/+0.29/+0.05 — SD 0.190 FAIL, range 0.374 FAIL.
- Median frontier regret: 0.0682/0.0644/0.0693 (baseline 0.0674/0.0723) —
  tighter across seeds, still above the item-7 floor 0.044.
- Sensitivity probe (cached single-decision pairs): D2N005 has 3/4 decisions
  reward-live (median spans 0.07-0.27) and exactly one fully reward-flat
  decision (38/38 contexts span 0.000). Residual D2N005 spread is the
  discrete lambda-3 switch quantum (~0.25 P per decision at D=4) interacting
  with per-seed switch-threshold drift under the global scalar alpha.

Verdict: Arm C adopted as infrastructure (strict improvement, no
regressions); fork remains open for D=4-class docs — decision escalated (per-
decision controller calibration vs accepted variance bound vs reward
enrichment for flat decisions).

## Cost

Local only, shared iGPU, sequential seeds; utility cache carries all
previously scored vectors.

## Artifacts

results/ranker_v2/architecture/credit_support/{train,control,kl-ref,epochs}-
stage2-s{17,29,47}.*, stage2.log. Predecessors: RL-ranker v8 (baseline),
RL-ranker v9 (screening).
