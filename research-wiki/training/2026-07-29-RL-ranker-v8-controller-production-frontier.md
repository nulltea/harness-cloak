---
type: training-experiment
status: planned
created: 2026-07-29
model: semantic-v1 policy (fresh BC warm start per seed, pinned frozen encoder
  thomas-sounack/BioClinical-ModernBERT-base, direct-count privacy signal,
  gap-scaled controller + switch-calibrated alpha init)
dataset: aci 4-doc controller-strength spike set (D2N005, D2N027, D2N063,
  D2N031; D = 4/12/18/22), frozen environment sha256:4cc754a7, utility
  artifact qa-utility-runtime-v2 policy denominator
result: pending
tags: [rl, ranker-v2, controller-strength, frontier-regret, production-trainer]
companion: ../../docs/specs/RL/interactive-ranker-v2-decision-log.md
---

# RL-ranker v8 — production-trainer re-evaluation of the frontier-regret criterion

Final adoption gate for the controller-strength fork (decision log 2026-07-28,
adjudicated OPEN 2026-07-29). The spike passed responsiveness (items 1–6, 8) on
all seeds but failed the utility-efficiency criterion — median utility regret vs
the cached (U, P) frontier plateaued at 0.063–0.084 against the 0.044
reader-noise floor — under a simplified loop that omitted counterfactual credit
and KL, the production mechanisms built for frontier-tracking. This run re-judges
that criterion under the real trainer.

## Objective & hypothesis

Hypothesis: with counterfactual utility credit and the KL anchor enabled, the
gap-scaled, switch-calibrated policy closes the per-decision utility gap and the
median frontier regret falls to ≤ 0.044 on all three seeds. If it fails here
too, the fork's recorded next step applies: iterate on per-decision credit, not
on alpha.

## Training data / config

- Docs: aci/D2N005, D2N027, D2N063, D2N031 (the spike's composition-diverse
  set), frozen full environment; scope demotion + zero-signal filter as in
  production.
- Per seed s in {17, 29, 47} (spike screening + confirmation seeds), full
  production chain, one chain at a time on the shared iGPU:
  1. `bc` — 3 epochs, lr 1e-4, seed s;
  2. `exit-collect` — 4 rollouts/doc from the BC checkpoint;
  3. `train` — 12 epochs (3 Latin cycles over the frozen 4-profile lambda
     menu), 8 rollouts/group, lr 1e-4, beta 0.01, eta 0.01, counterfactual
     budget from the frozen threshold manifest, device cuda,
     `--controller-gap-scaling utility-gap --alpha-init switch-calibrated`.
     Per-decision alpha routing stays OFF (normalization fork closed: current
     mix retained; spike arms did not use it).
- Reward: medgemma-4b-it via llama-swap (RT + QA pins unchanged), utility cache
  results/ranker_v2/cache/utility-results.jsonl, remote/reader workers 6/6.

## Selection & operating point

No selection: single preregistered configuration (the adopted candidate), fixed
hyperparameters mirroring the spike wherever the trainer has the same knob.

## Evaluation & success criteria

The spike's preregistered instrument, unchanged. Pool: every cached
full-coverage action vector per doc, P = mean per-decision profile score
(ProfileCountTargets), U = cached utility. Groups: per (doc, profile, epoch)
from epoch-report `conditional_samples` hashes joined back to the cache;
group P/U = mean over the group's rollouts; eligible groups are lambda > 0 and
cycle >= 1 (epoch >= 4). Regret = max{U_c : P_c >= P_group − 1e-9} − U_group.
PASS iff median regret <= 0.044 per seed, all three seeds, with items 1–6 and 8
of the preregistered rule (lambda-zero identity, monotone P, adjacent gaps,
Delta P(lambda-3) >= 0.20, >=75% nonnegative paired movement, placeholder < 95%,
finiteness/stability) not regressing vs the spike.

## Deviations

- 2026-07-29, before any seed completed: the first seed-17 attempt crashed at
  epoch 8 — the lambda-zero D2N005 group converged to all-KEEP (legitimate pure
  utility optimization), which has zero adjacent-capable counterfactual pairs,
  and `schedule_counterfactuals` treated the frozen budget (5, one endpoint
  slot) as exact demand and raised. Fixed as a production defect, not a
  protocol change: the scheduler now degrades per slot (drop lookahead ->
  retype to the other probe kind -> stop), so all-keep groups get keep->level
  endpoint probes and healthy groups behave bit-identically (strict path
  untouched). Pinned by two tests in test_counterfactual_credit.py; epoch
  reports gain scheduled/slot_shortfall/retyped_slots diagnostics. No reward
  values or preregistered evaluation quantities were touched. Seed 17 restarts
  from epoch 0 (deterministic replay through the cache reproduces the first
  8 epochs).

## Results

pending

## Ablations

None (confirmation run, not a search).

## Cost

Local only (medgemma via llama-swap, shared iGPU); no paid calls.

## Risks & caveats

The frontier pool now contains ~2.3k spike-era vectors including
exploration-enriched arms — the same in-hindsight bar the spike faced; a PASS
here is conservative evidence. 4-doc ACI subset: no generalization claims; the
readout is the selected count score (shaping proxy), never realized privacy.
ExIt may return 0 winners (BC near ceiling) as in v7 — the chain is still the
production chain.

## Artifacts

results/ranker_v2/architecture/controller_production/{bc,exit-winners,train,
control,kl-ref}-s{17,29,47}.*, epochs-s{seed}.jsonl, train-s{seed}.log;
evaluation script output alongside. Predecessor: RL-ranker v7 (hybrid smoke);
spike record in the decision log (controller-strength fork).

## Sources

docs/specs/RL/interactive-ranker-v2-decision-log.md (controller-strength fork);
scripts/spikes/objective_normalization_spike.py (spike loop + calibration,
graduated to cloak.ranker.semantic in commit 0dafd2d).
