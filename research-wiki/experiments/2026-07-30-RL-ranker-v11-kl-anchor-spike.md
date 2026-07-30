---
type: experiment
node_id: exp:RL-ranker-v11-kl-anchor-spike
status: done
verdict: no
confidence: high
created: 2026-07-30
model: semantic-v1 policy (controller_production BC/ExIt warm starts reused,
  gap-scaled controller + switch-calibrated alpha init, Arm C counterfactual
  broadcast, always-on KL to the calibrated reference)
dataset: aci 4-doc controller-strength spike set (D2N005/D2N027/D2N063/D2N031),
  frozen environment sha256:4cc754a7, qa-utility-runtime-v2 policy denominator
result: "Both KL directions at uniform eta over-anchor: lambda-3 held\n  near reference and sharpening bounded, but lambda-zero pinned to BC (item 6\n  failing) — reference-too-restrictive branch; escalation decision pending"
tags: [rl, ranker-v2, kl-anchor, tie-drift, saturation, coin-flip]
companion: ../../docs/specs/RL/interactive-ranker-v2-decision-log.md
---

# RL-ranker v11 — KL-anchor spike (forward vs reverse, always-on)

Fix spike for the lambda-3 coin-flip root cause (decision log 2026-07-30):
utility-tied action pairs (L0 == keep exactly on D2N005) inherit a keep-ward
prior through shared parameters, unbounded softmax sharpening amplifies it past
the bounded controller shift, and nothing pushes back (count gradient reaches
only alpha; the KL collapse trigger is aggregate-level, never fired, and the
forward direction loses its gradient under saturation while the calibrated
references are healthy anchors, E[P|lambda-3] ~0.66).

## Objective & hypothesis

An always-on low-weight KL to the calibrated reference (eta=0.01 from epoch 0)
owns the reward-silent ties and bounds margin drift, making lambda-3 behavior
deterministic on D2N005 without dulling reward-live learning. Reverse KL
(ref||pi) should dominate if saturation recovery matters (its ~(pi-ref)
gradient survives sharp policies; measured >100x forward at +-12 logits).

## Training config

Production trainer, 12 epochs, base 8 rollouts, Arm C
(--counterfactual-coverage degeneracy), gap-scaled controller +
switch-calibrated alpha init, lr 1e-4, beta 0.01, eta 0.01, no gradient
clipping (kept null — clipping is a separate later ablation, never bundled).
Arms: --kl-schedule always-on with --kl-direction forward | reverse.
Seeds 17 (healthy high-P mode) and 47 (collapsed all-keep mode, logit ranges
277-327 at stage-2 end) — prevention and recovery in one design.
AMENDMENT (2026-07-30, before any chain completed; time-to-signal): screening
runs at 8 epochs (2 cycles), seed 47 FIRST in both arms (the decisive
recovery test), with per-cycle live adjudication and early kill on definitive
failure; criteria 1-3 read over the 2 available cycles. The surviving arm
gets the full 12-epoch two-seed confirmation (folded into the item-7 rerun).
The first forward-s17 chain was stopped at epoch ~1 and its scored vectors
remain in the shared cache.
--synchronous-profile-eval on: every epoch report carries a same-checkpoint
4-profile readout (64 sampled vectors per profile, count-score only — no
remote calls; per-decision expected privacy score and utility-logit range).
Implementation commit 2b5ff6f.

## Evaluation & success criteria (preregistered, from the decision log)

On the synchronous readout, per arm, both seeds:
1. D2N005 sampled Delta P(lambda-3 vs lambda-0) >= 0.20 after every completed
   cycle;
2. D2N005 lambda-3 sampled-P range across cycles <= 0.10;
3. cross-seed final lambda-3 sampled-P difference <= 0.10;
4. reward-flat decision (atrial fibrillation) expected privacy score >= 0.50
   at lambda-3;
5. median frontier regret <= 0.044 held (item-7 gate, from the usual groups);
6. conditional lambda-zero utility within 0.044 of the fixed control;
7. no placeholder collapse, no non-finite values;
8. D2N005 utility-logit ranges grow <= 3x the calibrated reference's (9-15).

Adjudication: both pass -> adopt forward (smaller semantic change); only
reverse -> adopt reverse; behavior stable but regret/lambda-zero degrade ->
escalate to learned context-conditioned controller gain; neither stops the
range explosion -> one isolated max_grad_norm=1.0 arm before touching
controller capacity.

## Results

- FORWARD arm, seed 47: KILLED at cycle 1 per the early-kill rule — item 1
  failed definitively (synchronous D2N005 Delta P(lambda-3) = +0.02..+0.05
  across epochs 0-3 vs the >= 0.20 bar; "every completed cycle" cannot be
  recovered). The anchor itself works mechanically: logit ranges bounded at
  9-18 (reference 9-15; unanchored stage-2 exploded to 277-327), flat-decision
  expected privacy ~0.53, no drift. Failure mode: OVER-ANCHORING — a
  symmetric tug-of-war that suppresses all profile separation (lambda-zero
  stuck at P~0.46 instead of converging to utility-optimal ~0; lambda-3 at
  ~0.50, BELOW the reference's 0.66). Forward KL at eta=0.01 slows learning
  everywhere rather than owning only the reward-silent directions.
- REVERSE arm, seed 47: KILLED at cycle 1 — same signature as forward
  (synchronous D2N005 Delta P(lambda-3) = +0.03..+0.04). The readout localizes
  the failure: lambda-3 IS held near the reference (0.49-0.53 vs the
  unanchored collapse to 0.084) and ranges stay bounded (7-16), but
  lambda-zero is pinned at ~0.45 instead of converging to utility-optimal ~0
  (D2N031 lambda-zero likewise stuck 0.38-0.47) — uniform-eta KL freezes the
  utility side, so profile separation never opens. Item 6 (lambda-zero
  non-inferiority vs the freely-trained control) was headed for failure on
  both arms.
- Adjudication: the preregistered "reference too restrictive" branch fires
  for BOTH directions at uniform eta=0.01. The stabilization goals all hold
  (bounded sharpening, flat-decision privacy ~0.5, no drift); the damage is
  confined to anchoring profiles whose job is to LEAVE the reference
  (lambda-zero). Caveat recorded: pass-rule item 1 (>=0.20 per cycle,
  synchronous) was preregistered without a synchronous baseline trajectory —
  an unanchored run may also fail it at cycle 1 — but the trajectory evidence
  (lambda-zero pinned, no separation trend, lambda-3 drifting down not up)
  supports the branch decision independent of that bar's calibration.
- Operational note: the first reverse-s47 attempt crashed on a 429 caused by
  a failed kill leaving forward-s47 running concurrently (~15 min overlap;
  both trainers hammered llama-swap). Contaminated partial artifacts were
  deleted; the adjudicated reverse run executed on a verified-empty GPU.
  LLMClient now retries 429s with capped backoff (commit 0daddeb).

## Cost

Local only, shared iGPU, 4 chains sequential, heavily cache-backed.

## Artifacts

results/ranker_v2/architecture/kl_anchor/{train,control,kl-ref,epochs}-
{forward,reverse}-s{17,47}.*, spike.log. Predecessors: RL-ranker v10
(stage 2), root-cause adjudication in the decision log (2026-07-30).
