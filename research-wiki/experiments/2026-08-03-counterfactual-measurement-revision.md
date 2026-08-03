---
type: experiment
node_id: exp:counterfactual-measurement-revision
verdict: ""
confidence: ""
created: 2026-08-03
model: no model training — this revises the ΔU measurement instrument used by every ranker experiment
dataset: results/ranker_v2/cache/utility-results.jsonl (10,459 single-decision pairs), aci-full.utility
result: pending
tags: [reward, counterfactual, measurement, targeting, normalization, roundtrip, prompt, ranker-v2]
companion: ../../docs/issues/counterfactual-delta-u-measurement.md
---

# Counterfactual ΔU measurement revision — targeted context scoring, linked-mass normalization, delivered-noise reduction

## Objective

Every ranker experiment since v13 read its results through one instrument: the counterfactual ΔU. Round-4 established that instrument is contaminated and mis-normalized. This revision makes it targeted, correctly normalized, and diagnoses the residual noise channel — so that the next modelling experiment measures the world rather than the measurement.

## Verified preconditions (all measured 2026-08-03, cache-only)

**Full-span substitution works correctly (Q3 double-check, PASS).** `assemble_action_vector` requires an action for every policy decision and applies all of them plus fixed decisions; `reader_excerpt` cuts from that fully-assembled `doc_p`. Verified on `aci/D2N001` (11 decisions): **10/10** other decisions' chosen fills are present in the assembled document. Counterfactuals therefore DO absorb other policy decisions — the excerpt a reader judges already contains every other decision's generalization. Empirical confirmation: **12.7%** of total ΔU variance is within-pair across-context (mean within-group variance 0.00074 against total 0.0058), i.e. cross-decision interaction is real but secondary to `(decision, action-pair)` identity, which explains 87.3%.

**Context re-scoring is untargeted (A1).** `prepare_utility_scoring` builds `context_work` from *every* context assertion of the document and `_call_runtime_reader` issues a reader call for each. Median 8 context assertions per document (max 34), so a probe costs ~16 context reader calls. On the verified document only **5 of 14** excerpts changed, and **all 5 contained the flipped span**; the other 9 were byte-identical. With the reader at `temperature=0.0`, a byte-identical excerpt yields an identical answer, so those calls are **provably** wasted — skipping is exact, not approximate.

**Delivered-family noise is not stochastic.** Generation is pinned at `temperature=0`, single-flight, with a content-addressed disk cache, so `out_final` is deterministic given `doc_p`. The movement is therefore greedy-decode *input sensitivity*, not sampling. Characterized over 10,459 pairs: 63% of pairs show ≥1 non-linked delivered flip, median 2 flips per affected pair (29% of a document's delivered assertions at once), p90 4, max 9. **100% of the 13,824 flips are `contains` contracts; `required_sections` flipped zero times.** So the note's section structure is stable — this is *paraphrase drift*, not format collapse.

**The roundtrip uses the loose prompt.** `_template(job)` returns `SCHEMA_TEMPLATE` only when `job["template"] == "schema"`, but both the pin (`roundtrip.py:87`) and the generation call (`roundtrip.py:283`) pass `{"corpus": corpus}` with no such key. So the ranker always uses `CLINICAL_NOTE` ("using standard note sections" — no format specification), and `SCHEMA_NOTE` — which mandates sections *and* line formats like `"problem — category — status"` — is unreachable on this path. No caller in `scripts/` or `src/cloak/reward/` ever requests it.

## Revision items

**R1 — targeted context re-scoring (A2+A3).** For a probe at decision X, re-score only those context assertions whose excerpt differs between the two assembled documents; reuse the score for byte-identical excerpts. Selection is a string comparison on the already-assembled `doc_p`, requires no model, and is exact under `temperature=0`. Expected saving on the verified document: 9/14 (64%) of context reader calls. Note this subsumes the A2/A3 distinction — "span is subject/object" and "span merely occurs in the excerpt" both reduce to "the excerpt changed", which is the provable criterion.

**R2 — normalize per-decision attribution by linked mass.** Replace `Σ_linked w·Δs / W` with `Σ_linked w·Δs / W_L` where `W_L` is the decision's own linked weight, giving "what fraction of this decision's obligations the flip broke" in [−1, 1]. Measured effect: p75 |ΔU| 0.0302 → 0.2500 (×8.3), p90 0.1066 → 0.4565, fraction above the 0.044 floor 17% → 32%. The *objective* keeps `W` for cross-document comparability, and document-level utility remains the final arbiter for a composed action vector (required, because per-decision attribution deliberately excludes the measured cross-decision spillover).

**R3 — reduce delivered-family drift.** Two independent candidates, both of which change the reward pin (see Cost):
- *(a) Constrain generation.* Wire `SCHEMA_NOTE` into the ranker roundtrip. Its mandated line formats constrain **phrasing**, which is what actually drifts — note that specifying sections alone would not have helped, since `required_sections` never flipped.
- *(b) Make delivered scoring paraphrase-robust.* 606 of 723 delivered assertions score by `contains`, i.e. exact substring matching. That is the lexical-matcher anti-pattern this project has already ruled out elsewhere: a `contains` flip measures the matcher as much as the model. Replacing it with an entailment or embedding judgement would remove drift sensitivity at its source.

**R4 — keep the assembly as-is.** Verified correct; do not switch to locally-patched originals, which would forfeit the measured 12.7% cross-decision component.

## Measurements & success criteria

1. **R1 equivalence (blocking).** On pairs already in the cache, targeted re-scoring must reproduce the untargeted ΔU **exactly** (bitwise on the skipped assertions, since their inputs are byte-identical). Any mismatch means the skip criterion is unsound and R1 is withdrawn.
2. **R1 saving.** Report reader calls avoided per probe, corpus-wide, as the concrete cost reduction available for buying evidence breadth.
3. **R2 restratification.** Re-derive the census strata under `W_L`; report the shift in exact / sub-floor / live counts and the fraction of decisions that become measurable.
4. **R3 diagnosis (before any prompt change).** Extract concrete examples separating (i) real per-decision signal — a linked `contains` flip on the flipped span's own assertions — from (ii) drift — a non-linked `contains` flip. Report contract text and what changed, **counts and contract identifiers only, no clinical text**, per the clinical-echo handling rule.
5. **R3 arm, if funded.** Re-score a small fixed document set under `SCHEMA_NOTE` and measure whether the non-linked delivered flip rate falls from its current 63% of pairs.

## Cost and the cache warning

R1, R2, and measurement 4 are **cache-only and free** — no remote calls, no reader calls, no GPU.

R3 is not. The task prompt is part of the reward pin (`TASK_PROMPT_PIN_VERSION`, `template_hash`), and cached rewards are valid only under the pin that produced them. **Switching to `SCHEMA_NOTE` invalidates the entire utility cache** — every scored action vector in `utility-results.jsonl`, which is the accumulated output of the whole campaign and the source of all 10,459 evidence pairs. The same applies to replacing `contains` scoring. R3 therefore requires an explicit budget decision and should be scoped to a small pinned subset first, not the corpus.

## Risks & caveats

- R1's soundness rests on the reader being deterministic at `temperature=0` for identical input. Measurement 1 tests exactly this and blocks on it.
- R2 changes the units of every threshold downstream, including the 0.044 floor, which was measured at document granularity and must be re-measured under `W_L` before any gate reuses it.
- Context assertions are 66.9% of the reward denominator (median 68% per document), so a context-only probe channel measures a *majority* but not all of utility; delivered coverage cannot simply be dropped.
- The 18% systematically-signed residue remains split between dependency under-declaration and real spillover. A free triage rule now exists — does the flipped span appear in the assertion's excerpt — and should be applied before any artifact repair.

## Results

pending

## Artifacts

Verification scripts in session scratch; reproducible from `results/ranker_v2/architecture/equivalence_critic/evidence-rows-linked.json`. Predecessors: [Gate 1](2026-08-01-RL-ranker-gate1-representation.md), [v15 screening](2026-07-31-RL-ranker-v15-equivalence-critic.md). Companion issues: [counterfactual ΔU measurement](../../docs/issues/counterfactual-delta-u-measurement.md), [QA dependency under-declaration](../../docs/issues/qa-dependency-underdeclaration.md).
