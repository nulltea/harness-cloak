---
type: dev-log
status: current
created: 2026-07-12
updated: 2026-07-12
tags: [rl, reward, privacy, qa-build, codesign, notes]
companion: [docs/specs/RL/training-task-env.md]
---

# Notes: reward-side implications of the QA-build coverage redesign

Advice only — reward design is a separate workstream (Timo's, WIP in `src/cloak/anonymity.py`).
Logged here per request; the QA-build session decides nothing below.

## What the QA build will guarantee (once the coverage redesign lands)

1. Every detected span is in the env span set (admission decoupled from probing) — the policy
   acts on all spans, so a per-span privacy term applies everywhere.
2. The probed subset is class-stratified under a fixed reader budget (~13 kept probes/doc),
   rotating across builds/epochs — every span class accumulates utility signal over training.
3. Per-span coverage metadata in the artifact: `{class, polarity, probed, unprobed_reason,
   verdicts}` — the reward can distinguish "utility measured" from "utility unmeasured this
   round."

## The unprobed-span problem, from the reward side

A count-rewarding privacy term is unopposed on unprobed spans; without mitigation the policy
drifts to placeholder there and generalizes the pattern by context. Options the coverage
metadata enables (not mutually exclusive):

- **Class-mean utility prior:** for an unprobed span, substitute the running mean utility of
  its class's probed siblings (same type × polarity × action). Unbiased if stratification is
  honest; keeps the privacy term opposed everywhere.
- **Count-curve proxy:** utility-loss proxy monotone in the anonymity-set growth
  (e.g. −α·log(k_chosen/k_finest)). Zero-cost, model-free; note it is the same variable as the
  privacy term, so the two must differ in shape, not just sign, or they cancel into a constant.
- **Mask the privacy term on unprobed spans** (reward only where opposed): simplest, but
  wastes privacy gradient and biases toward finer actions on unprobed spans.

## One warning: the count-weight knob vs the no-calibration-tricks rule

A hand-tuned weight `w` in `utility + w·log k` is one knob governing the privacy↔utility
operating point. Re-tuning `w` per corpus/mechanism to make a secondary quantity look right
is exactly the calibration-trick failure the repo banned (RANTEXT `noise_scale` incident).
Whatever form the term takes, move operating points only via an explicit privacy target and
report realized privacy against an attacker — never equalize post-hoc. Framings where the
weight is a dual variable adapted to hit a stated target (rather than a taste constant)
dissolve the "which side gets higher reward" question into choosing the operating point;
sweeping the target then yields the Pareto curve directly.

## Cost facts the reward side can rely on

- Teacher: one call per sampled span + one per doc (decisions), one-time, cached, pv-tagged.
- Reader: the recurring wall — every kept probe is a serial read per rollout (~1/s). The
  kept-probe budget, not admission width, controls RL wall-time.
- Per-span credit: ladder parts are per-span; decisions smear over span_ids; counterfactuals
  are the spec's credit mechanism. A per-span privacy lookup adds zero reader/teacher cost.
