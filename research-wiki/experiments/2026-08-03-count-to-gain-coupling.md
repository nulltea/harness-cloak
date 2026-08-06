---
type: experiment
node_id: exp:count-to-gain-coupling
verdict: "reject as insufficient — count->gain coupling does not restore lambda-controllability (M=0.00 in both arms); it raised the two documents with surviving traction (0.15->0.21, 0.18->0.29) and left the two collapsed ones bit-identical, so the pathology is upstream of controller routing"
confidence: "high on the null result (validity check passed live: count->gain exactly 0.0 vs 0.120; arms bit-identical on the governing document), low on generalization (one seed, four documents, lambda0 gate unevaluated)"
created: 2026-08-03
model: semantic-v1 policy (controller_production BC/ExIt warm starts, no-gap base — softcap 25, no gap-scaling, sensitivity 0.1 — evidence controller gain, online tie-margin hinge)
dataset: aci 4-doc controller set (D2N005, D2N027, D2N031, D2N063), frozen environment, qa-utility-runtime-v2 denominator, existing utility cache (unchanged)
result: "M=0.00 both arms, primary gate fails; differential gain on D2N031/D2N063 only; oscillation shape unchanged"
tags: [rl, ranker-v2, controller, gradient-routing, count-to-gain, tie-ownership]
companion: ../../docs/research/tie-ownership-root-cause-and-solution-space.md
---

# Count-to-gain coupling — does the one isolated gradient edge cause the controller pathology?

## Objective & hypothesis

In **evidence** gain mode the controller has exactly **one uniquely isolated gradient edge**: the dense expected-count objective never reaches the state-conditioned gain residual, because the count branch alone detaches it (`src/cloak/ranker/semantic.py`, evidence-mode branch). Everything else is already coupled — `alpha_raw` receives count *and* λ>0 policy gradients, and the residual receives the hinge *and* λ>0 policy gradients.

**What this experiment does NOT rest on.** It would be wrong to motivate it with v13. In `learned` mode `alpha_count_value` stays `None`, so the count objective flows through the controller built from the *attached* residual: **v13's learned gain was trained by count and still degenerated**, its measured failure being a dominant common count gradient pinning a *bounded* (tanh-ceilinged) gain to a global constant. So "the count gradient could not differentiate the gain" is false as a v13 explanation, and the outcome "gain goes uniform" has a precedent rather than being a surprise. What is untested is whether coupling helps the **unbounded evidence gain in the presence of the hinge** — a different object from v13's bounded, hinge-free gain.

**Hypothesis.** Within the current softcap / no-gap / online-hinge stack, detaching the *unbounded evidence* gain residual from the dense count objective is the proximate cause of unstable high-λ privacy behaviour on D2N005.

**Deciding quantity.** `M = min` over seeds {17, 29, 47} and the final three snapshots of greedy `P(λ3) − P(λ0)` on aci/D2N005, against v14's existing requirement `M ≥ 0.20`. No new constants.

**Decision fed.** Whether to remove the detach and retain count→gain coupling in the current stack.

**Explicitly not addressed:** whether the hinge can be deleted; whether the three-way gradient separation is the architectural root cause; held-out-document generalization; calibration of the local tie floor; the anchored-utility redesign; production matched-realized-privacy claims. A clean pass shows only that the mechanism behaves this way **on these four documents under the cached deterministic reward** — it is not a model-fitness result and it does not license the word "works" without that scope attached.

## Arms

One logged option, one code path, no branch duplication:

- **control** — `--count-to-gain detached` (current routing: count trains `alpha_raw` only)
- **treatment** — `--count-to-gain coupled` (count also trains the per-decision residual)

The arms are distinguished by `count_to_gain` in `training_config`, **not** by an architecture-pin retag. An earlier draft retagged `controller_transform`; the pre-run review correctly rejected that — detaching changes gradients, not forward values (`softplus(a + r)` and `softplus(a + r.detach())` are numerically identical), so retagging would give two numerically identical policies different architecture pins and stop them sharing one KL-reference artifact.

The control is **re-run on current HEAD**, not read off v14's archived numbers: tie labels and reader-call counts changed after the measurement revision, so comparing to the archive would confound the un-detach with those changes.

## Configuration (identical except the one flag)

```
--doc-id aci/D2N005  --doc-id aci/D2N027  --doc-id aci/D2N031  --doc-id aci/D2N063
--device auto  --remote-workers 6  --reader-workers 6
--max-epochs 8  --rollouts 8  --learning-rate 1e-4  --beta 0.01  --eta 0.01
--alpha-utility-routing none  --controller-gap-scaling none  --alpha-init switch-calibrated
--rollout-scaling fixed  --counterfactual-coverage degeneracy
--kl-schedule collapse-trigger  --kl-direction forward
--synchronous-profile-eval  --synchronous-profile-samples 16
--utility-logit-softcap 25  --profile-sensitivity-reg 0.1
--controller-gain evidence  --controller-gain-hidden 32  --controller-gain-lr 1e-2
--controller-gain-bound 1.5  --tie-mode online  --tie-coefficient 1.0  --tie-margin 0.1
--tie-min-contexts 3  --gain-penalty 1e-3  --tie-evidence-bootstrap  --batched-rollouts
```

Identical artifact hashes, initial checkpoints, schedules, RNG streams and KL references. Document set hash `sha256:c50016fe…` (emitted by the preflight; the trainer must load exactly these four). Seeds 17 and 29 were preregistered but not run — see Results.

**λ-zero control: once per seed, not per arm.** `--skip-lambda-zero-control` is now the default. The control trains on a single λ=0 profile where the forward pass takes the exact-identity branch and the controller and tie machinery are inert, so `count_to_gain` provably cannot affect it: it varies only with seed and documents, and training it per arm doubled every A/B for an identical result. Produce it once per seed with `--no-skip-lambda-zero-control`.

**Document selection is explicit, not `--max-docs`.** `--max-docs 4` slices the first four *sorted* retained documents, which is `D2N001–D2N004` on this environment — not the intended set, and not what predecessor runs used.

**Cache isolation, and why not `--cache-only`.** The pre-run review asked for `--cache-only` to stop the arms mutating a shared cache. That flag is the wrong instrument: it raises `CacheOnlyMissError` on the first miss, and an RL run necessarily misses on fresh rollouts. Isolation is achieved instead by giving each arm its **own copy of the utility cache** — the cache is keyed by action vector and grows as an arm explores, so a shared file would let whichever arm ran first feed rows to the second. Both arms start from a byte-identical 9,440-row copy. The **LLM cache is shared on purpose**: it is content-addressed by prompt, so sharing it is deterministic and isolating it would force full regeneration.

**Budget (8 epochs, not 12).** An arm must finish inside an hour or the development loop is unusable. Three launch defects were found and fixed before the measured run, none of them in the reward logic: `--remote-workers`/`--reader-workers` left at their code default of 1 (every prior run passed 6; now the default), and the semantic extractor recomputing `_candidate_windows` — an O(tokens x window sizes) rapidfuzz scan — for identical `(fill, text)` pairs on every rollout. Memoizing it and `_semantic_scores` (pure functions of their arguments, so bit-identical results) took the observed pace from ~2 to ~16 utility evaluations per minute. Moving the MiniLM encoder to the GPU was considered and **rejected for now**: fp32 reduction order differs by device, near-tied overlapping candidate windows would reorder, and `extractor_pin` does not record the device — so CPU- and GPU-computed rows would share one identity in the cache.

**Staged execution.** Run the paired seed 47 first; continue with paired seeds 17 and 29 only if the treatment arm survives the existing early kills. Maximum remains six runs.

**Mechanism decisions, and why each is held rather than deleted.** Round 4 prescribed deleting the hinge, cycle projection, softcap, capped gain, gap-scaling, KL anchor and sensitivity regularizer — but those deletions were conditional on replacing the actor/controller stack with the anchored-utility redesign, so they are not prerequisites here and applying them now would change the architecture instead of isolating one edge.

- **Hinge KEPT, `--tie-mode online`.** Present identically in both arms, so the paired difference is still caused by the new edge (including its interaction with the hinge). Cost: the experiment cannot claim "count alone replaces the hinge".
- **Cycle projection OFF.** It fires up to 25 gain-only optimizer steps at cycle boundaries, which would swamp the incremental count update.
- **Softcap KEPT at 25.** Bounds the competing tower scale; does not cap the evidence gain.
- **`--controller-gain-bound` IRRELEVANT.** It applies only to `learned` mode; `evidence` mode is deliberately unbounded (the v13 tanh ceiling is what froze differentiation). Raising it would do nothing.
- **Sensitivity KEPT at 0.1.** It uses `policy.alpha` = `softplus(alpha_raw)`, so it never trains `gain_head`.
- **`--gain-penalty` KEPT at 1e-3.** The residual is unbounded; removing its only regularizer would move two variables. If count gradients are dominated by the penalty, family-specific gradient norms will expose it.
- **Gap-scaling OFF**, **KL on `collapse-trigger`** (never activated in archived v14; activation in one arm is a reportable treatment-mediated collapse response).

**Logging asymmetry, disclosed.** Per-epoch progress reporting (`_report_epoch_progress`) was added while the detached arm was mid-run, so the detached arm produced no per-epoch lines and the coupled arm did. The change prints a line and rewrites a `.partial` report file: it consumes no RNG, touches no policy state, and cannot alter training. Progress for the detached arm was read from its per-epoch checkpoint's `epoch` field instead. Measured pace: **~5 min/epoch, ~40 min for 8 epochs per arm.**

## Preflight — both checks executed 2026-08-03, both PASS

**Gradient-routing validity (the preregistered check).** `src/cloak/tests/test_semantic_ranker.py::test_count_to_gain_coupling_moves_exactly_one_gradient_edge`: the count objective's gradient into `gain_head` is **exactly 0.0** under `detached` and **nonzero** under `coupled`; `alpha_raw` is reached in both arms; the utility head receives **exactly zero** count gradient in both. If this ever fails, no outcome of the arm is interpretable.

**Bootstrap statistic parity (the blocking fix).** `bootstrap_tie_evidence_from_cache` previously emitted seed records carrying only the document-level `delta_u` — the statistic that reported nonzero on 65% of provably-tied pairs — while online rows carried the revised statistics. Epoch-0 labels would have been qualified under a *different measurement contract* than every later round, so the hinge would change meaning mid-run. Seed records now recompute `delta_u_attributed`, `delta_u_linked` and `movement_l1` from the cached component scores; this is post-hoc aggregation, no reward call is repeated and the utility cache is untouched.

Measured effect of the parity fix on the seed ledger, **scoped to this run's four documents** (`scripts/spikes/count_to_gain_arm_preflight.py`, 440 pairs / 10,206 records, document set hash `sha256:c50016fe…`):

| contract | qualifying tie pairs | decisions |
|---|---|---|
| legacy (document delta only) | 35 | 10 |
| revised (both-agree + movement) | 34 | 10 |

1 pair lost, 0 gained — conservative. Independently recomputed by the pre-run integrity audit with the same loaders and the same result.

Over the whole 63-document environment the same fix gives 40 → 38 pairs and 12 → 11 decisions (638 pairs / 10,459 records). That figure is reported for completeness only: **it does not describe this experiment**, and an earlier draft of this record quoted it as if it did.

## Inherited state, declared

- Reader dedup is live (~78% fewer context reader calls, values identical).
- Scope-matched substitution is **inert**: the attributed set is the full weight set, so the complement is empty and `delta_u` is the total document delta.
- Context preflight / delivered audit quota is **fully disarmed** (`delivered_audit_fraction = 0.0` ⇒ quota `None` ⇒ every pair admitted).
- `TIE_EXIT_BOUND = 0.044` is a document-unit figure compared against the movement statistic. It over-disqualifies rather than manufacturing false ties, so it costs power. **It is not automatically comparison-neutral:** evidence coverage is policy-dependent, so the two arms can accumulate different qualified sets during training. Qualified-pair and qualified-decision counts are therefore reported per arm and per epoch, and any arm difference in coverage is reported alongside the outcome rather than assumed away.

## Gates & preregistered decision

Reuse v14's gates verbatim (round-3 preregistration, reproduced so nothing is silently weakened): greedy `ΔP(λ3) ≥ 0.20` holding over the final three snapshots; **no all-KEEP collapse** in them; nondecreasing greedy privacy across profiles (sampled separation report-only); λ-zero loss ≤ 0.044 every document; median frontier regret ≤ 0.044; **≥ 90% of evidence-qualified tie constraints satisfying the 0.1 greedy margin**; **no costly pair mislabeled as a tie**; nonzero cross-decision gain spread; ranges ≤ 50; **behaviour evaluated without evidence override** (the hinge trains the policy; no lookup at inference).

Early kills, also verbatim: zero hinge gradient after cycle 1; gain spread < 1e-3 after cycle 2 despite ≥ 5 qualified decisions; two consecutive greedy λ3 collapses; any λ-zero loss > 0.044; **effective gain > 12 with constraints unsatisfied** (the bound is conditional on unsatisfied constraints — v14 exceeded it with all constraints satisfied and correctly did not kill).

1. **Validity** — count-family gradient into `gain_head` exactly zero in control, finite-nonzero in treatment, for λ>0 groups. Fails ⇒ the run is void.
2. **Adopt coupling** — treatment has `M ≥ 0.20`, passes every v14 gate, and control fails the primary gate.
3. **Retain detach** — both arms pass; the new edge is unnecessary.
4. **Reject as insufficient** — both arms fail, or treatment improves but stays below 0.20.
5. **Reject as harmful** — treatment violates λ0 utility, frontier regret, monotonicity, range, or an early-kill gate.

## Pre-run review and audit (2026-08-03)

Codex Sol High returned **NEEDS-CHANGES** and the integrity audit (gpt-5.5, xhigh, independent thread) returned **FAIL**. Both are addressed above; what they caught:

- **Wrong documents.** `--max-docs 4` selects `D2N001–D2N004`. Replaced with four explicit `--doc-id` flags plus `--cache-only`, and the preflight is now scoped to the same set and emits a document-set hash. The parity figures in this record were previously all-environment numbers.
- **Bootstrap failed open.** Missing `component_scores` defaulted to `{}`, producing zero movement — a manufactured tie. Now skipped, and both utilities are recomputed with `document_utility` rather than trusting the cached scalar. Covered by `test_bootstrap_records_match_the_online_evidence_formula`.
- **The validity check was unmeasurable at runtime.** `gain_head` was absent from the semantic parameter groups, so per-family gradient norms could not report the count→gain edge; a `gain` group was added. The unit test now backpropagates a non-flat count objective (probabilities weighted by exact counts) instead of `sum(count_log_probs)`, which is insensitive to a uniform logit shift.
- **Wrong pin.** Coupling changes gradients, not forward values, so retagging `controller_transform` was removed — it would have given numerically identical policies different architecture pins and blocked a shared KL reference. The distinction is pinned in `training_config`, and validation was hoisted out of the gain-mode branch so `coupled` with `--controller-gain none` is rejected.
- **Overstated history and weakened gates.** The v13 motivation was false (corrected above), "mechanism proof" was scope-bound, the conservative-qualification claim was qualified for policy-dependent coverage, and v14's gates are now reproduced verbatim.
- **Leg D labelling.** Its held-out target was called the "TRUE effect"; it is a cached model/scorer quantity. Relabelled as a proxy target in `scripts/spikes/credit_attribution_leg_d.py`.

**One review recommendation declined:** Sol High proposed filing this as `RL-ranker v16` under `research-wiki/training/`. `training/` is reserved for production full runs; a six-run diagnostic screening on four documents belongs in `research-wiki/experiments/`, where it stays.

## Cost

Six runs (2 arms × 3 seeds), 12 epochs each, 4 documents. Utility cache survives completely — no remote generation and no reader calls beyond cache misses. GPU: one process at a time.

## Results — seed 47, both arms, 8 epochs each (~57 min per arm)

**Preregistered outcome 4: REJECT AS INSUFFICIENT.** `M = 0.00` in *both* arms, against the required `≥ 0.20`.

**Validity check PASSED on live training** (final epoch): `count → gain` is **exactly 0.000000** detached and **0.120497** coupled; `count → utility` is **0.0 in both**; `lambda_zero_identity_failures = 0` in both. The arms differ in the intended edge and nothing else, so the null result is interpretable.

**Greedy ΔP (λ3 − λ0), final three snapshots (the gate window):**

| document | detached e5,e6,e7 | min | coupled e5,e6,e7 | min |
|---|---|---|---|---|
| D2N005 | +0.00 +0.00 +0.34 | **0.00** | +0.00 +0.00 +0.34 | **0.00** |
| D2N027 | +0.16 +0.16 +0.19 | 0.16 | +0.16 +0.16 +0.19 | 0.16 |
| D2N031 | +0.33 +0.15 +0.17 | 0.15 | +0.33 +0.26 +0.21 | **0.21** |
| D2N063 | +0.47 +0.20 +0.18 | 0.18 | +0.45 +0.33 +0.29 | **0.29** |

**The differential finding, which is the useful part.** Coupling **improved the two documents that still had controller traction** (D2N031 0.15 → 0.21, D2N063 0.18 → 0.29, both now above the 0.20 bar) and left the two that had collapsed or plateaued **bit-identical to the control** (D2N005 and D2N027 match the detached arm at every one of the final three snapshots). So the count→gain edge can raise authority where separation still exists but cannot restore a document that has gone to zero — and D2N005, which governs `M`, is exactly such a document.

That identity across arms on D2N005/D2N027 is itself evidence: those outcomes are determined by something the coupling cannot reach, consistent with the round-6 reading that the oscillation lives **upstream in the utility tower** (greedy base churn among utility-equivalent actions) rather than in controller routing.

**Trajectory shape unchanged.** Both arms peak mid-training (+0.25 to +0.60 at e1-e4) and decay to +0.15..+0.33 by e5-e7; D2N005 oscillates 0.34 → 0.15 → 0.34 → 0.15 → 0.00 → 0.00 → 0.34 in both. The arms are near-identical at e0-e1 (zero-initialised residual) and diverge from e2, confirming proper pairing.

**Other diagnostics (final epoch).** Global alpha inert in both (5.3516 detached, 5.3519 coupled). Loss 2.298 → 1.766 with coupling. `tie_margin` mass 10.23 → 8.06 while `count` mass 0.58 → 0.64: the hinge still dominates the gain residual by ~12×, so my preregistered magnitude prediction (that coupling would be swamped) was **half right** — it was not swamped into invisibility, but it was too weak to move the collapsed documents.

**Not evaluated: the λ0 utility non-inferiority gate.** Seed 47's λ-zero control pass was killed to save wall-clock, so there is no "trained at λ=0 only" baseline and preregistered outcome 5 (*reject as harmful*) could not be tested. `lambda_zero_identity_failures = 0` and unchanged λ0 privacy show no gross λ0 damage, but the gate itself is open. One control run per seed with `--no-skip-lambda-zero-control` closes it.

**Scope.** One seed, four documents, within-document only. Per the preregistration this supports a mechanism claim about these documents under the cached deterministic reward — not a model-fitness or generalization claim. Seeds 17 and 29 were not run: with `M = 0.00` fixed by two zero-separation epochs on D2N005, additional seeds cannot change the primary verdict, only its confidence.

### Defects found by this run (both mine, both fixed)

1. **A monitoring write killed an arm after epoch 1.** `_report_epoch_progress` caught only `OSError`, so a `TypeError` from tuple-keyed report dicts propagated out of the epoch callback. Now catches everything, with `skipkeys=True`; regression test `test_epoch_progress_reporting_never_fails_a_run`.
2. **A tuple-keyed diagnostic destroyed the authoritative report of a completed run.** Scope-matched substitution left `attributed_sets`, keyed by `(rollout_index, decision_id)`, inside `scheduler_diagnostics`, which is JSON-serialised into the epoch report — so `_write_epoch_reports` raised *after* all 8 epochs finished. Root cause fixed (the internal handoff is popped once consumed) and `_write_epoch_reports` hardened as a backstop. Both arms' reports were recovered from their per-epoch checkpoints; no training was repeated.

## Artifacts

- Preflight: `results/ranker_v2/architecture/count_to_gain/arm-preflight.json`
- Preflight script: `scripts/spikes/count_to_gain_arm_preflight.py`
- Routing test: `src/cloak/tests/test_semantic_ranker.py::test_count_to_gain_coupling_moves_exactly_one_gradient_edge`

## Sources

Design adjudicated by Codex Sol High rounds 6–7 (2026-08-03). Root-cause analysis and the deletion/repurposing ranking this experiment deliberately does *not* implement: [tie-ownership root cause and solution space](../../docs/research/tie-ownership-root-cause-and-solution-space.md), [ties-by-design](../../docs/specs/RL/ties-by-design.md). Predecessor: [RL-ranker v14 evidence tie ownership](2026-07-31-RL-ranker-v14-evidence-tie-ownership.md).
