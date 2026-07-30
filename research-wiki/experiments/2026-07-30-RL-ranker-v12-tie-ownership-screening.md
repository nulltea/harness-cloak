---
type: experiment
node_id: exp:RL-ranker-v12-tie-ownership-screening
status: done
created: 2026-07-30
model: semantic-v1 policy (controller_production BC/ExIt warm starts reused,
  gap-scaled controller + switch-calibrated alpha init, Arm C counterfactual
  broadcast; per-arm addition — utility-logit soft-cap OR profile-sensitivity
  regularizer)
dataset: aci 4-doc controller-strength spike set (D2N005/D2N027/D2N063/D2N031),
  frozen environment sha256:4cc754a7, qa-utility-runtime-v2 policy denominator
result: "All four capacity-free arms fail; cap+no-gap solve scale and lambda-zero freedom and find a stable equilibrium with insufficient separation — escalation to the learned controller gain is triggered"
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

- **softcap seed 47 (done, interim adjudication).** Boundedness fully works: max menu-logit range 38.2 anywhere (gate 50; the same seed exploded to 277-327 uncapped), lambda-zero converges freely (D2N005 lambda-zero P -> ~0.0 by epoch 1 — the v11 over-anchoring failure is absent), synchronous Delta P(lambda-3) >= 0.20 from epoch 1 (final +0.25), no non-finites. TWO GATES FAIL: (1) cycle stability — lambda-3 synchronous P swings 0.20..0.56 across epochs, and the DETERMINISTIC greedy path confirms it is real policy churn, not sampling (greedy_P: 0.56 -> 0.25 -> 0.34 -> 0.56 -> 0.25 -> 0.15 -> 0.00 -> 0.00); (2) by epochs 6-7 the greedy lambda-3 policy is ALL-KEEP (greedy_P = 0.00) while sampled P reads 0.22-0.50 — logit ranges shrank to 2-4 and the apparent separation comes from distribution softness around an indecisive near-tie, not a decisive private preference. Sampled Delta P alone is gameable by softness; greedy-path stability is added to the adjudication of all arms. Verdict so far: the cap is necessary-not-sufficient — boundedness solved, tie OWNERSHIP still absent.
- **sensitivity seed 47 (done).** Complementary-opposite failure: epochs 0-4 show REAL tie ownership (Delta P +0.21..+0.32 with genuine greedy separation, lambda-zero argmax 0.00-0.28 vs lambda-3 argmax 0.34-0.52 — the softcap arm never achieved this), then the logit ranges explode anyway (54 -> 81 -> 124, max 170; gate 50 FAIL) and Delta P collapses to +0.01 before partially recovering (+0.13 final; gate FAIL). Mechanism identified: with the GAP-SCALED controller the shift grows with the range, making the adjacent-profile KL the regularizer targets roughly scale-invariant — it pins the shape of the lambda-response while the scale race runs underneath, and extreme sharpness eventually collapses the shape too. Measured target 0.2295 KL per unit g.
- **Adjudication of the single arms: both fail, on exactly complementary halves** — softcap = bounds without ownership; sensitivity = ownership without bounds. The pre-declared composition trigger is met: the composed arm (softcap 25 + sensitivity 0.1) ran on seed 47 (the moot sensitivity-17 single arm and the softcap-17 confirmation were cancelled per time-to-signal review — a second seed of a rejected single arm buys nothing; composed-17 is gated on composed-47).
- **composed seed 47 (done): FAIL, new failure shape.** Best epochs of any arm (0-4: Delta P +0.13 -> +0.42 with real greedy separation, greedy lambda-0 at 0.00 vs lambda-3 at 0.38-0.56; flat-decision expected privacy ~0.6; ranges bounded <= 42 — the scale channel is fully controlled), then epoch-cadence collapse-recover oscillation: epochs 5 and 7 total lambda-3 collapse (greedy 0/0, flat 0.01-0.05, all profiles ~0.00), partial sampled recovery at 6 with greedy still dead. With scale controlled, the flip driver is the utility SHAPE: per-epoch credit on the shared tower swings the keep-margin across the shift line faster than the sensitivity term (coeff 0.1, measured target 0.1867) pulls back. Ranges 5.6-11 during the dead ep6 recovery = the downward-authority regime of gap-scaling. Gates: ranges PASS, lambda-zero freedom PASS; final Delta P, cycle stability (both readouts), and flat-decision privacy FAIL.
- Per the preregistered no-gap ablation (decision log 2026-07-30), A1's failure triggered the no-gap arm unconditionally on seed 47 (alpha applied 5.35 in raw units on capped logits; measured sensitivity target 0.1633).
- **no-gap seed 47 (done): FAILS the gate set but uniquely finds a stable equilibrium.** Early churn transient (epochs 2-3 collapse, greedy 0/0) then recovery (+0.34) and — unprecedented across all arms — THREE consecutive greedy-stable, lambda-ordered epochs (5-7: greedy lambda-0 = 0.18, lambda-3 = 0.34 constant; sampled Delta P +0.11..+0.15; ranges settled 15-20; flat-decision ~0.5). The authority-wandering hypothesis is CONFIRMED in mechanism (constant absolute shift -> an equilibrium exists and the system stays in it) but the equilibrium's separation magnitude fails the gate (final Delta P +0.12 < 0.20): the fixed shift calibrated at warm-start ranges (~10) lost switchable fraction as ranges grew to ~17 under training — raw switch thresholds scale with the range even when the shift does not.
- **Screening verdict (v12 complete): all four capacity-free arms fail** — softcap (bounds, no ownership), sensitivity (ownership, no bounds), composed (both, but epoch-cadence shape churn outruns the damping), no-gap (stable equilibrium, insufficient separation magnitude). Scale control and lambda-zero freedom are solved by cap+no-gap; the residual problem is per-decision authority against utility-shape churn on the shared tower. The preregistered escalation applies: learned state-conditioned controller gain (with the literature-mandated ceiling, non-dual-ascent training, gain-field smoothness check, and random/fixed-rule-gain controls).
- Operational: the sensitivity arm's first launch crashed on a diagnostics-family consistency check (profile_sensitivity missing from absolute_weighted_mass) — fixed and pinned before rerun.

## Cost

Local only, shared iGPU, 4 chains sequential (~1.5h each, cache-backed).

## Artifacts

results/ranker_v2/architecture/tie_ownership/{train,control,kl-ref,epochs}-
{softcap,sensitivity}-s{47,17}.*, screening logs. Predecessors: RL-ranker v11
(KL anchor, both directions failed by over-anchoring), v10 (Arm C stage 2).
