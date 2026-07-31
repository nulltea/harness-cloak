---
type: experiment
node_id: exp:RL-ranker-v15-equivalence-critic
verdict: ""
confidence: ""
created: 2026-07-31
model: equivalence-critic heads (logistic tie gate + epsilon-insensitive magnitude head) on the frozen adopted v14 cycle-projected policy trunk; policy weights untouched
dataset: counterfactual evidence rows (v14 chain ledger + corpus-wide cache-mined single-decision pairs), document-disjoint train/calibration/certification splits
result: pending
tags: [rl, ranker-v2, tie-ownership, equivalence-critic, hurdle, epsilon-insensitive, conformal, deployment-generalization]
companion: ../../docs/specs/RL/ties-by-design.md
---

# RL-ranker v15 — equivalence-critic screening (offline, held-out tie ownership)

## Objective & hypothesis

The deployment-generalization gap (ties-by-design spec, section 5): tie ownership on training documents is solved by per-document evidence (v14 cycle projection), but the evidence ledger is empty on unseen documents, so held-out tie-breaking degenerates to global alpha racing arbitrary interpolated tower margins. Hypothesis: a small supervised critic trained on measured counterfactual ΔU evidence predicts utility-equivalence on *held-out documents* well enough to build a certified-conservative canonicalization gate, making lexicographic tie behavior transferable without per-document probes. This is a supervised screening — the policy is NOT retrained; the critic bolts onto the frozen adopted v14 cycle checkpoint and canonicalization is inference-time.

## Design (formalized by the 2026-07-31 literature round; full rationale in the companion spec, fork section)

- **Architecture**: shared trunk = frozen, detached policy decision features (same pooled inputs as the gain head); two heads: (1) logistic tie gate, BCE on 1[|ΔU| ≤ 0.044], all evidence rows; (2) magnitude head q_U, ε-insensitive Huber `Huber(max(0, |r| − 0.044))` on the pairwise difference residual, all rows (deadband self-masks tie rows).
- **No batch rebalancing**: gate at the 0.044 boundary makes classes ~39/61; uniform sampler.
- **Splits**: document-disjoint train / calibration / certification, split by document, never by pair. Certification documents must have utility-cache coverage (measured ΔU is the ground truth).
- **Acceptance rule**: gate-probability threshold chosen on the calibration split, then frozen; equivalence set on a menu = accepted actions by distance-from-max; abstain (ordinary controller) below threshold.
- **Certification**: exact one-sided Clopper–Pearson bound on the accept-conditional false-equivalence rate, computed on the certification split with the *document* as the unit (mean of per-document violation rates); pair-level number reported as optimistic companion only.

## Run plan

0. **Feasibility count (preregistered gate, before any training)**: mine all corpus-wide cache-covered single-decision pairs (extend `bootstrap_tie_evidence_from_cache` beyond the 4 training docs) + the v14 chain evidence rows; report row counts per stratum (exact-tie / sub-noise / live) and per split. Kill if the certification split cannot plausibly yield ≥ 45 accepted pairs (Clopper–Pearson infeasible below that even at zero violations — undersized, not failed).
1. **Primary arm**: hurdle critic as specified, trained on the train split.
2. **Ablation A (loss form, the comparison the literature lacks)**: ε = 0 (plain Huber) vs ε = 0.044, everything else fixed.
3. **Ablation B (structure-free baseline)**: single Huber head with Balanced MSE, uniform sampler; if it matches the hurdle gate on tie precision/recall at 0.044, the gate head is unjustified and gets deleted.
4. **Comparison mechanism (only if the gate score is miscalibrated on the calibration split)**: CQR twin-quantile intervals with split-conformal correction; accept iff interval ⊂ ±0.044.
5. **Behavioral leg**: bolt the certified gate onto the frozen v14 cycle checkpoint; synchronous profile snapshot on held-out documents with canonicalization active at λ>0 and exact identity at λ=0.

## Evaluation & success criteria (preregistered)

- **Never pooled**: gate precision / recall / reliability curve separately from live-row MAE; gate recall split by exact-tie vs sub-noise at evaluation (training merges them).
- **Primary gate**: certified false-equivalence risk ≤ 5% at confidence 90% (document-level Clopper–Pearson). At realistic accepted-pair counts (n ≈ 100–300) this demands empirical precision 97–98% — 95% empirical on 200 pairs FAILS. Recall reported, never optimized.
- **Secondary gates**: held-out tie-oracle agreement ≥ 90% on cache-verified ties; live-pair order consistency; λ=0 exact identity; λ=0 utility non-inferiority within 0.044; no held-out utility regression from false canonicalization; greedy ΔP(λmax − λ0) ≥ 0.20 on tiny held-out documents.
- **Kill criteria**: gate degenerates to base-rate predictor (held-out AUC ≈ 0.5); feasibility count fails (step 0); certified bound infeasible on the realized accepted count.
- **Preregistered caveat**: the certified bound holds for documents exchangeable with the calibration split; under user-domain shift it degrades unmeasurably. Structural mitigation: unfamiliar documents produce low gate confidence → abstention → ordinary controller behavior, never confident false ties.

## Cost

CPU-dominant supervised training on hundreds–thousands of pairs (small heads, frozen trunk); GPU minutes for feature extraction and the behavioral snapshot leg, no RL training. Fits the perf gate trivially; feasibility count is free (cache-only).

## Risks & caveats

- Same sparse evidence as v14 — held-out interpolation is exactly the open question; the experiment is designed so a negative answer is a measured precision/recall number, not an ambiguous RL curve.
- Reader-noise common-mode refinement: if exact ties are traceable to identical reader input/output, noise cancels and ε=0 is the tighter constraint for that stratum; check during step 0, do not assume.
- Sanity check to leave behind: a deliberately mis-fit critic must abstain, not accept (measured violation rate under the CP bound on held-out data).

## Results

pending

## Artifacts

pending (planned: results/ranker_v2/architecture/equivalence_critic/). Predecessors: v14 (evidence-supervised tie ownership; cycle projection adopted), tie-structure census, ties-by-design spec. Literature: registered pages raokupper1967, davidson1970, chen2025_dpo_ties, liu2024_reward_learning_ties, sun2024_rethinking_bradley_terry, yang2021_deep_imbalanced_regression, ren2022_balanced_mse, kong2020_deep_hurdle_networks, romano2019_conformalized_quantile_regression, bates2021_risk_controlling_prediction_sets, barber2023_conformal_beyond_exchangeability.
