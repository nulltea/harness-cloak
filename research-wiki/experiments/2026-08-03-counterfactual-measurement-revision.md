---
type: experiment
node_id: exp:counterfactual-measurement-revision
verdict: "Linked-mass normalization ADOPTED — the ambiguous sub-floor band collapses 1204 -> 85 pairs (93%) and live pairs nearly double, with exact ties invariant as required. Targeted context re-scoring FAILS its blocking exactness gate on 13 of 88,619 checks (0.015%), concentrated in 4 assertions and traceable to reader_refresh mismatch (5) plus genuine reader non-determinism (8); conditional adoption proposed, not unconditional."
confidence: "high — both are exhaustive offline computations over the full cache under a single uniform reward pin; no sampling"
created: 2026-08-03
model: no model training — this revises the ΔU measurement instrument used by every ranker experiment
dataset: results/ranker_v2/cache/utility-results.jsonl (10,459 single-decision pairs), aci-full.utility
result: normalization adopted (sub-floor 1204->85, live 1291->2410, p75 |delta| 0.030->0.250); targeted re-scoring would skip 78% of context reader calls but failed the zero-violation gate on 13/88,619 checks
tags: [reward, counterfactual, measurement, targeting, normalization, roundtrip, prompt, ranker-v2]
companion: ../../docs/issues/counterfactual-delta-u-measurement.md
---

# Counterfactual ΔU measurement revision — targeted context scoring, linked-mass normalization, delivered-noise reduction

## Objective

Every ranker experiment since v13 read its results through one instrument: the counterfactual ΔU. Round-4 established that instrument is contaminated and mis-normalized. This revision makes it targeted, correctly normalized, and diagnoses the residual noise channel — so that the next modelling experiment measures the world rather than the measurement.

## Verified preconditions (all measured 2026-08-03, cache-only)

**Full-span substitution works correctly (Q3 double-check, PASS).** `assemble_action_vector` requires an action for every policy decision and applies all of them plus fixed decisions; `reader_excerpt` cuts from that fully-assembled `doc_p`. Verified on `aci/D2N001` (11 decisions): **10/10** other decisions' chosen fills are present in the assembled document. Counterfactuals therefore DO absorb other policy decisions — the excerpt a reader judges already contains every other decision's generalization. Empirical confirmation: **12.7%** of total ΔU variance is within-pair across-context (mean within-group variance 0.00074 against total 0.0058), i.e. cross-decision interaction is real but secondary to `(decision, action-pair)` identity, which explains 87.3%.

**Context re-scoring is untargeted (A1).** `prepare_utility_scoring` builds `context_work` from *every* context assertion of the document and `_call_runtime_reader` issues a reader call for each. Median 8 context assertions per document (max 34), so a probe costs ~16 context reader calls. On the verified document only **5 of 14** excerpts changed, and **all 5 contained the flipped span**; the other 9 were byte-identical. With the reader at `temperature=0.0`, a byte-identical excerpt yields an identical answer, so those calls are **provably** wasted — skipping is exact, not approximate.

**Delivered-family noise is not stochastic.** Generation is pinned at `temperature=0`, single-flight, with a content-addressed disk cache, so `out_final` is deterministic given `doc_p`. The movement is therefore greedy-decode *input sensitivity*, not sampling. Characterized over 10,459 pairs: 63% of pairs show ≥1 non-linked delivered flip, median 2 flips per affected pair (29% of a document's delivered assertions at once), p90 4, max 9. **100% of the 13,824 flips are `contains` contracts; `required_sections` flipped zero times**, so section structure is stable. And 84% are near-total losses with 0% partial paraphrase changes: the failure is **omission of unrelated content**, not rewording.

**The roundtrip uses the loose prompt.** `_template(job)` returns `SCHEMA_TEMPLATE` only when `job["template"] == "schema"`, but both the pin (`roundtrip.py:87`) and the generation call (`roundtrip.py:283`) pass `{"corpus": corpus}` with no such key. So the ranker always uses `CLINICAL_NOTE` ("using standard note sections" — no format specification), and `SCHEMA_NOTE` — which mandates sections *and* line formats like `"problem — category — status"` — is unreachable on this path. No caller in `scripts/` or `src/cloak/reward/` ever requests it.

## Revision items

**R1 — targeted context re-scoring (A2+A3). NOT BUILT — superseded by an existing mechanism (2026-08-03).** The saving this item targets is already delivered by the content-addressed LLM cache: across the campaign, 111,855 logical context-reader calls collapse to **11,459 distinct (assertion, excerpt) prompts, a 89.8% dedup rate** — more than the 78% this criterion identified as skippable. `BatchedContextReader._read_one` uses the cache by default (it passes `refresh=True` only when a caller explicitly asks). Building a second memoization layer over it would save a disk lookup, not a model call. The *finding* stands and is recorded; the code change is unnecessary. Original specification retained below for the record.

*(original)* For a probe at decision X, re-score only those context assertions whose excerpt differs between the two assembled documents; reuse the score for byte-identical excerpts. Selection is a string comparison on the already-assembled `doc_p`, requires no model, and is exact under `temperature=0`. Expected saving on the verified document: 9/14 (64%) of context reader calls. Note this subsumes the A2/A3 distinction — "span is subject/object" and "span merely occurs in the excerpt" both reduce to "the excerpt changed", which is the provable criterion.

**R2 — normalize per-decision attribution by linked mass.** Replace `Σ_linked w·Δs / W` with `Σ_linked w·Δs / W_L` where `W_L` is the decision's own linked weight, giving "what fraction of this decision's obligations the flip broke" in [−1, 1]. Measured effect: p75 |ΔU| 0.0302 → 0.2500 (×8.3), p90 0.1066 → 0.4565, fraction above the 0.044 floor 17% → 32%. The *objective* keeps `W` for cross-document comparability, and document-level utility remains the final arbiter for a composed action vector (required, because per-decision attribution deliberately excludes the measured cross-decision spillover).

**R3 — reduce delivered-family OMISSION.** Revised 2026-08-03 after measuring the flip anatomy; an earlier version of this item proposed a paraphrase-robust scorer and that proposal is **withdrawn**.

Measured over all 13,824 non-linked delivered flips: **84% are near-total loss** (a score ≥0.5 falling to ≤0.05) and **0% are partial paraphrase changes**. The term is absent from the generated note, not reworded. `invert` is already a semantic cascade (placeholder, exact/fuzzy generalization, semantic-window fallback, `"semantic": True` with a pinned semantic model), which is the most likely reason the paraphrase bucket is empty — it normalizes reworded controlled fills before scoring. Consequently **no scorer or extractor change can address this**: neither an entailment judgement nor a better `invert` can match or map content the model never produced.

The action-mode transition rates also refute a placeholder-confusion hypothesis: keep↔level 63%, level↔level 65%, level↔placeholder 49%, keep↔placeholder 54% — placeholders are the *least* disruptive transitions.

The only remaining lever is generation. Wire `SCHEMA_NOTE` into the ranker roundtrip: its `ASSESSMENT: one line per active problem` and `PLAN: one line per active problem` slots make silent omission structurally difficult. A terminology-fidelity instruction was also considered and is **not** adopted — it targets paraphrase, which is 0% of the measured problem, and it would push toward verbatim echo, which interacts with `docs/issues/remote-llm-echo-absorption.md` for no measurable gain.

**R4 — keep the assembly as-is.** Verified correct; do not switch to locally-patched originals, which would forfeit the measured 12.7% cross-decision component.

## Measurements & success criteria

1. **R1 equivalence (blocking).** On pairs already in the cache, targeted re-scoring must reproduce the untargeted ΔU **exactly** (bitwise on the skipped assertions, since their inputs are byte-identical). Any mismatch means the skip criterion is unsound and R1 is withdrawn.
2. **R1 saving.** Report reader calls avoided per probe, corpus-wide, as the concrete cost reduction available for buying evidence breadth.
3. **R2 restratification.** Re-derive the census strata under `W_L`; report the shift in exact / sub-floor / live counts and the fraction of decisions that become measurable.
4. **R3 diagnosis — DONE 2026-08-03.** Flip anatomy measured (84% omission / 0% paraphrase) and the placeholder hypothesis refuted; see R3. Concrete examples inspected on `aci/D2N001`; note that examples must be sampled by transition type, since iterating the cache in order produced two placeholder cases that were unrepresentative of the population.
5. **R3 arm, if funded.** Re-score a small fixed document set under `SCHEMA_NOTE` and measure whether the non-linked delivered *omission* rate falls from its current 63% of pairs. Report per-transition rates so the level↔level case (currently the worst at 65%) is visible separately.

## Cost and the cache warning

R1, R2, and measurement 4 are **cache-only and free** — no remote calls, no reader calls, no GPU.

R3 is not. The task prompt is part of the reward pin (`TASK_PROMPT_PIN_VERSION`, `template_hash`), and cached rewards are valid only under the pin that produced them. **Switching to `SCHEMA_NOTE` invalidates the entire utility cache** — every scored action vector in `utility-results.jsonl`, which is the accumulated output of the whole campaign and the source of all 10,459 evidence pairs. The same applies to replacing `contains` scoring. R3 therefore requires an explicit budget decision and should be scoped to a small pinned subset first, not the corpus.

## Risks & caveats

- R1's soundness rests on the reader being deterministic at `temperature=0` for identical input. Measurement 1 tests exactly this and blocks on it.
- R2 changes the units of every threshold downstream, including the 0.044 floor, which was measured at document granularity and must be re-measured under `W_L` before any gate reuses it.
- Context assertions are 66.9% of the reward denominator (median 68% per document), so a context-only probe channel measures a *majority* but not all of utility; delivered coverage cannot simply be dropped.
- The 18% systematically-signed residue remains split between dependency under-declaration and real spillover. A free triage rule now exists — does the flipped span appear in the assertion's excerpt — and should be applied before any artifact repair.

## Results (2026-08-03)

### Targeted context re-scoring — gate FAILED, conditionally salvageable

**Saving available:** of 112,918 context assertion-instances across all pairs, **78% (88,619) are provably skippable** and only 22% need re-scoring. Per-document skip fraction: median 67%, range 0–100%.

**Blocking gate: FAILED.** Of the 88,619 skippable instances where both scores were cached, **13 disagreed** — a byte-identical reader excerpt scoring differently — every one of them a full 1.0 flip. The gate was specified as zero-violation, so it fails as written.

**Decomposition, which is what matters:**
- Re-assembly is exact: for every cached vector, the locally re-assembled `doc_p` matches the cached `doc_p` **byte-for-byte** (0 drift), so the comparison itself is sound and the violations are real.
- All 9,339 cache rows share one `reader_pin`, `extractor_pin`, and `task_prompt_pin`, so these are **not** stale-pin artifacts.
- The 13 violations come from just **4 distinct (document, assertion) pairs**, one of which (`aci/D2N027`) accounts for 8 of them.
- Split by cause: **5 are `reader_refresh` mismatches** — a non-determinism knob that is part of the cache identity and can simply be held constant — and **8 are same-refresh genuine reader non-determinism** despite `temperature=0`. The `aci/D2N027` excerpt is 1,181 characters against 384–408 for the others, making a length or batch-composition effect a plausible cause.

**Interpretation, and why this is not a reason to abandon the criterion.** In all 13 cases the two scores were produced from *byte-identical reader input*, so one of them is simply wrong and the ΔU between them is a pure artifact. The current untargeted pipeline **includes** that artifact as a spurious 1.0 flip; skipping would reuse one score and eliminate it. Skipping therefore removes contamination rather than hiding signal, even on the failures.

**Proposed disposition (needs a decision, not adopted unilaterally):** adopt the criterion conditional on (a) holding `reader_refresh` constant within a probe pair, and (b) quarantining the 4 flagged assertions pending a reader-determinism investigation. Do not report the gate as passed.

### Linked-mass normalization — ADOPTED

| stratum | ÷ W (current) | ÷ W_L (proposed) |
|---|---|---|
| structurally derivable | 2,895 | 2,895 |
| exact tie | 5,069 | 5,069 |
| **sub-floor** | **1,204** | **85** |
| **live** | **1,291** | **2,410** |

The ambiguous sub-floor band — the stratum that consumed v13 through v15 — **collapses by 93%**, and live pairs nearly double. |Δ| percentiles move from 0.0000 / 0.0302 / 0.1066 to 0.0000 / 0.2500 / 0.4565.

Exact ties and derivable ties are **invariant** at 5,069 and 2,895, which is the required internal consistency check: a positive rescaling can neither create nor destroy an exact zero. That the counts match exactly confirms the implementation.

**Caveat that blocks reuse of the old threshold.** This restratification holds the numeric 0.044 fixed while changing its units, which is not principled — 0.044 was measured at document granularity. The distributional separation is the real result; the floor must be re-measured under `W_L` before any gate consumes it.

## Artifacts

Verification scripts in session scratch; reproducible from `results/ranker_v2/architecture/equivalence_critic/evidence-rows-linked.json`. Predecessors: [Gate 1](2026-08-01-RL-ranker-gate1-representation.md), [v15 screening](2026-07-31-RL-ranker-v15-equivalence-critic.md). Companion issues: [counterfactual ΔU measurement](../../docs/issues/counterfactual-delta-u-measurement.md), [QA dependency under-declaration](../../docs/issues/qa-dependency-underdeclaration.md).
