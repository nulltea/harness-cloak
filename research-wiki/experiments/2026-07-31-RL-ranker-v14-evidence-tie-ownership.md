---
type: experiment
node_id: exp:RL-ranker-v14-evidence-tie-ownership
verdict: ""
confidence: ""
created: 2026-07-31
model: semantic-v1 policy (controller_production BC/ExIt warm starts, no-gap trio base — softcap 25, no gap-scaling, sensitivity 0.1 — plus evidence controller gain and the tie-margin hinge)
dataset: aci 4-doc controller-strength spike set, frozen environment sha256:4cc754a7, qa-utility-runtime-v2 policy denominator
result: pending
tags: [rl, ranker-v2, tie-ownership, evidence-supervised, lexicographic, hinge]
companion: ../../docs/specs/RL/interactive-ranker-v2-decision-log.md
---

# RL-ranker v14 — evidence-supervised tie ownership screening

The round-3 adjudicated mechanism (decision log 2026-07-31): route the trainer's own counterfactual evidence into controller authority. Verifiable-core tie labels (>=3 distinct contexts at exactly zero delta-U, contradiction exit at 0.044, one-cycle lag) drive a controller-only hinge on the max-lambda sampling margin of each qualified pair; the gradient split sends the hinge (and lambda>0 sampling terms) to the gain residual, the global count loss to global alpha only, and nothing to the utility tower. No hard gain ceiling (the hinge is self-limiting; weak L2 penalty instead). The hard lexicographic oracle rides every synchronous snapshot as a ceiling and evidence-sufficiency diagnostic.

## Arms (screening discipline: seed 47 first, early kills, then confirmation seeds for the survivor)

- **A (online)**: `--tie-mode online --tie-coefficient 1.0 --tie-margin 0.1 --controller-gain evidence --controller-gain-lr 1e-2` on the no-gap trio; labels lagged one cycle.
- **B (cycle-projected)**: `--tie-mode cycle --controller-gain-lr 0.0 --tie-projection-lr 1e-2` — residual frozen within cycles; <=25 full-batch controller-only hinge updates at each cycle boundary (isolates whether timescale mismatch remains load-bearing after evidence routing).

Both: 12 epochs, 8 rollouts, Arm C counterfactual coverage, synchronous profile eval, alpha-init switch-calibrated (raw), no KL-anchor flags. Implementation commit b5b2018.

## Gates (round-3 preregistration)

Greedy Delta P(lambda-3) >= 0.20 holding over the final three snapshots, no all-KEEP collapse in them, nondecreasing greedy privacy across profiles (sampled separation report-only); lambda-zero loss <= 0.044 every document; median frontier regret <= 0.044; >= 90% of evidence-qualified tie constraints satisfying the 0.1 greedy margin; no costly pair mislabeled as a tie; nonzero cross-decision gain spread; ranges <= 50; behavior evaluated without evidence override (the hinge trains the policy; no lookup at inference). Early kills: zero hinge gradient after cycle 1; gain spread < 1e-3 after cycle 2 despite >= 5 qualified decisions; two consecutive greedy lambda-3 collapses; any lambda-zero loss > 0.044; effective gain > 12 with constraints unsatisfied.

## Results

- **Arm A (online), 12 epochs, done.** The mechanism WORKS: from the first qualified labels (epoch 4; only 2 constraints all run — no bootstrap in this chain) the greedy lambda-3 policy tracks the hard oracle exactly at most epochs (0.10/0.10, 0.18/0.18, 0.34/0.34, 0.43/0.43), and when it slips (epochs 9-10: policy 0.18 vs oracle 0.28) the hinge readout names the unsatisfied constraint and it re-converges. First arm in the campaign whose lambda-3 behavior is explained by an auditable quantity at every epoch. Greedy stability gate still FAILS (final-three greedy Delta P 0.08/0.18/0.34; series oscillates 0.43->0.18->0.18->0.34). Gain field remains decision-uniform (rides alpha_raw 4.48 -> 2.66 -> 6.10) — with 2 constraints there was little to sculpt.
- **Arm B (cycle-projected), epoch-8 intermediary.** The boundary projection moves the controller decisively (alpha 5.35 frozen through cycle 0, then 13.58 after the first projection, 10.06 after the second) and the policy tracks the oracle EXACTLY AT ALL NINE EPOCHS with zero constraint violations. Cleaner than online. THE TWIST: the oscillation did not disappear — it moved upstream into the oracle itself (policy and oracle wobble together: lambda-3 greedy 0.52/0.34/0.00/0.00/0.15/0.34/0.34/0.52/0.18), i.e. the utility tower's greedy base churns among utility-equivalent actions and/or lands outside qualified tie coverage; lambda-zero wobbles in sympathy between tied actions (utility-free P movement). Evidence-to-authority routing is solved in both arms; the residual instability is tower churn among ties plus evidence-coverage gaps — a different, smaller problem than the fork opened with. alpha reached 13.58 (past the >12 early-kill line, which conditions on UNSATISFIED constraints — all satisfied, so no kill; the hinge's self-limiting property holding without a bound).
- Interim verdict: cycle projection is the stronger arm on current evidence; final adjudication (with per-decision decomposition of the residual wobble: covered ties vs uncovered ties vs sub-noise pairs) after the chain completes.

## Artifacts

results/ranker_v2/architecture/tie_ownership/{train,control,kl-ref,epochs}-evidence-{online,cycle}-s47.* plus logs. Predecessors: v13 (learned gain rejected), round-3 decision-log entry, docs/research/reward-ties-and-controller-authority.md.
