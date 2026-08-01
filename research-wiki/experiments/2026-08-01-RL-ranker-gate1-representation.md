---
type: experiment
node_id: exp:RL-ranker-gate1-representation
verdict: "FAIL on the primary comparison — an anchored probe on frozen features does not beat the tower's own margin on held-out tie discrimination. The learning-curve diagnostic is UNDERPOWERED (minimum resolvable AUC difference 0.159 versus an observed range of 0.138), so it cannot say whether the cause is representation or evidence support."
confidence: "high on the primary failure (three arms, consistent across two held-out splits and per-document); none on the cause"
created: 2026-08-01
model: frozen adopted v14 cycle-projected checkpoint (semantic-v1, softcap 25, no gap-scaling); probes trained on detached utility-head input features, policy never updated
dataset: linked-restricted delta-U evidence (evidence-rows-linked.json, 10,459 pairs), document-disjoint train/calibration/certification
result: negative — anchoring delivers calibrated scale (slope ~1.0 vs the actor's 0.010) but loses ordering (live sign agreement 0.75 vs 0.86) and does not improve held-out tie discrimination
tags: [rl, ranker-v2, gate, representation, anchoring, scale-identifiability, underpowered]
companion: ../../docs/research/tie-ownership-root-cause-and-solution-space.md
---

# Gate 1 — representation gate for utility-scale anchoring

## Objective & hypothesis

Round-4 recommends anchoring the utility tower's margins in measured ΔU units. Gate 1 asks the precondition: **can the existing representation support being a utility estimator at all, on documents it has not seen?** Hypothesis: a probe regressing linked-restricted ΔU from the tower's own input features will discriminate ties better than the raw actor margin the controller races today.

Self-adjudicating by construction: both quantities are computed from the same features on the same held-out rows, so if the actor margin already discriminates as well, anchoring buys nothing without a second model needing to be trained.

## Design

Probes on detached utility-head input features, trained with the ε-insensitive Huber (ε = 0.044) on the pairwise difference, 1500 full-batch epochs, Adam 1e-3, seed 47. Three arms: `linear`; `mlp` matched to the actor head's own architecture (Linear→GELU→Linear — a linear probe is strictly weaker than the margin it is compared against, which invalidated a first attempt); and `actor-init`, the same MLP warm-started from the tower's utility head. Baseline: the softcapped actor margin. Splits are document-disjoint (4 pinned train documents, 27 calibration, 36 certification). Reporting is stratified — structurally derivable rows are an easy class and are excluded from the headline discrimination number.

## Results

**Noise ceiling is not binding.** Within-pair across-context variance on the train split implies max achievable R² = **0.873**.

**Train fit is strong** (MLP / actor-init): tie-AUC 0.826 / 0.850, live sign agreement 0.929 / 0.927, calibration slope 1.044 / 1.041, live MAE 0.054 / 0.053. The representation *can* express ΔU on documents it has seen — this is not a capacity, optimisation, or noise-ceiling failure.

**Certification (147 rows, 102 judged after excluding derivable):**

| arm | tie-AUC | live sign agreement | calibration slope | live MAE |
|---|---|---|---|---|
| linear probe | 0.406 | 0.639 | 1.18 | 0.139 |
| MLP (matched to actor head) | 0.525 | 0.722 | 1.03 | 0.131 |
| MLP warm-started from actor head | 0.628 | 0.750 | 1.23 | 0.121 |
| **actor margin (baseline)** | 0.592 | **0.861** | 0.010 | 10.57 |

Calibration split reverses the one apparent win: actor 0.733 versus 0.606 for the best anchored arm. Per-document, anchored beats actor on 2–3 of 15 certification documents and 0–1 of 11 calibration documents.

**Two findings that survive the failure.**

1. **Anchoring delivers scale, reliably, on held-out data.** Every anchored arm lands calibration slope ≈1.0–1.2 with MAE ≈0.12 in ΔU units, against the actor's slope 0.010 and MAE 10.57. The tower's margins are ~100× off-scale — the partial-identifiability prediction confirmed empirically, not just theoretically.
2. **Anchoring loses ordering.** The actor keeps live-pair sign agreement 0.861 versus 0.72–0.75 for every anchored arm including the warm-started one. The tower knows *which* action is better; regression learns *how much* and forgets *which*. This independently reproduces why the reward-modelling literature composes regression with ranking rather than substituting one for the other, and it settles that part of the design regardless of what else happens.

**Learning-curve diagnostic — underpowered, no conclusion.** Retraining the actor-init probe on 1/2/3/4 training documents gives certification tie-AUC 0.518 / 0.655 / 0.616 / 0.525 (761 / 4,214 / 7,157 / 10,206 rows). Non-monotonic and *declining* after two documents. Power analysis on the certification split (66 tie / 36 live, Hanley–McNeil): SE of a single AUC = **0.0575**, minimum resolvable difference at 95% = **0.159 AUC**, observed range = **0.138**, and the spread of the four points (sd 0.068) is indistinguishable from noise alone (0.058). The curve is consistent with a flat line plus noise; it can support neither "breadth helps" nor "the representation is dead."

## Verdict

Gate 1 **fails** its primary comparison: anchoring on frozen features does not beat the margin it was meant to replace. The cause is not identifiable with current data. Combined with v15, this is the second independent experiment to find that a features-to-equivalence map does not transfer across documents at our scale — and the learning curve shows we now lack the statistical power to diagnose why. **The binding constraint is evidence volume, in training (4 documents) and in evaluation (102 judged held-out rows), not modelling ideas.**

## Next steps (not run)

The preregistered escalation was the unfrozen-trunk arm, but it is not worth running: it would be evaluated on the same 102 rows and so inherits the same 0.159 AUC resolution floor. Any of these first: buy evaluation breadth (more documents with cache coverage) so the instrument can resolve what we are asking it; pursue structural derivation, which needs no prediction; or price local QA regeneration at deployment, which would remove the transfer requirement entirely.

## Artifacts

`results/ranker_v2/architecture/equivalence_critic/gate1-report-{linear,mlp,actor-init}.json`; implementation `scripts/spikes/equivalence_critic_screening.py gate1` (commit 5402cf4). Predecessor: [v15 equivalence-critic screening](2026-07-31-RL-ranker-v15-equivalence-critic.md). Companion: [round-4 research report](../../docs/research/tie-ownership-root-cause-and-solution-space.md).
