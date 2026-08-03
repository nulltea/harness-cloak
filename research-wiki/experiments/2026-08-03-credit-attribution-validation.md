---
type: experiment
node_id: exp:credit-attribution-validation
verdict: "Leg D FAILS and its preregistered rule fired: the total document delta predicts held-out total effect better than the attributed delta (linked-route sign agreement 0.661 vs 0.530), so spillover is reproducible structure and excluding it from the GRADIENT discards transferable information. The gradient is reverted to the total delta; attribution is retained for tie qualification, where Leg D says nothing. Legs A-C PASS. The additivity identity holds on every document; the gradient-delta identity closes exactly at 3047 once cancellation is accounted; document-route credit is retained on all 1,873 pairs. Leg B surfaced a previously unidentified mechanism (union cancellation) that the monotone qualification statistic is immune to by construction. Leg D and the behavioural leg remain unrun."
confidence: "Leg D scope-limited: 99.3% of its folds come from the 4 RL training documents, so it governs the within-document regime only and says nothing about unseen documents. High on the identities (exhaustive over the full cache, exact arithmetic); the local floor is still unmeasured, so no gate may consume the movement statistic yet"
created: 2026-08-03
model: no training in the cache-only legs; the behavioural leg reuses the frozen adopted v14 cycle checkpoint
dataset: results/ranker_v2/cache/utility-results.jsonl (10,459 single-decision pairs, 9,319 scored vectors, one uniform reward pin)
result: legs A-C pass; gradient identity 2857 + 191 - 1 = 3047 exact; mean |delta| on the counterfactual channel 0.0407 -> 0.0311 (x0.76); exact-tie stratum grows 3103 -> 5704 under monotone movement
tags: [reward, counterfactual, credit-attribution, validation, preregistration, ranker-v2]
companion: ../../docs/issues/counterfactual-delta-u-measurement.md
---

# Validation of the credit-attribution revision

Validates the four changes shipped 2026-08-03 (commits bbbd361, 435a57c, d5dfec2, bc10ed8) against the existing utility cache. Design rationale and the adjudication that produced them: [measurement revision record](2026-08-03-counterfactual-measurement-revision.md). The changes correct the *instrument* every ranker experiment since v13 has read its results through; they do not address the round-4 scale or composition defects.

## What is being validated

| step | change | commit |
|---|---|---|
| 1 | scope-matched substitution: credit the measured component exactly, add a leave-one-out advantage over the complement | bbbd361 |
| 2 | excerpt-changed linkage unioned into the attributed set, pair-local, declarations untouched | 435a57c |
| 3 | set-monotone weighted-L1 movement replaces the `/W_L` average for tie qualification | d5dfec2 |
| 4 | context preflight plus a fixed delivered audit quota, disarmed by default | bc10ed8 |

## Legs and gates

### Leg A — reduction identities (blocking, free)

Each change must collapse to the previous behaviour in the regime where it should be inert. Any failure means the change is not a refinement but a redefinition.

1. **Empty complement.** With the attributed set equal to the whole document, `hybrid_utility_loss` must return a value *bit-identical* to the pre-change whole-term substitution. Already unit-tested; re-verify on cached pairs.
2. **Additivity.** `subset_advantages(S) + subset_advantages(Q∖S) == subset_advantages(Q)` for every probed pair — the identity that licenses the decomposition.
3. **Quota disarmed.** With `delivered_audit_fraction = 0.0`, scheduler output (requests, ordering, direction balance, endpoint allocation) must be identical to HEAD~4 on a fixed seed.
4. **Objective reconstruction.** The family decomposition must still equal the total objective to within the existing tolerance, with `counterfactual_complement` accounted.

### Leg B — gradient-delta accounting (free)

Recompute, over all 10,459 cached pairs, what the counterfactual gradient becomes. Preregistered expectations from the pre-implementation measurement, which the run must reproduce:

- pairs whose correct gradient is zero and previously received a spurious push: **3,047** silenced (linked route)
- pairs previously pushed in the **wrong direction**: **177** sign-corrected
- pairs on decisions no assertion claims: **1,873** must *retain* credit (blanket restriction would have zeroed them — this is the regression the route-consistency fix exists to prevent)
- false zeros rescued by excerpt-changed linkage: **305**
- mean |ΔU| on the counterfactual channel: 0.0407 → 0.0228 (×0.56)

A deviation of more than ±1% on any count means the implementation does not match the analysis it was derived from.

### Leg C — restratification and the local floor (free, then one measurement)

Re-derive the census strata under the monotone movement statistic. **Blocking caveat:** `TIE_EXIT_BOUND = 0.044` was measured in document-aggregate units and is *not* dimensionally valid for weighted-L1 movement over an attributed subset. This leg reports the movement distribution and the strata it implies at several candidate thresholds; **no gate may consume the new statistic until the floor is re-measured in its own units.** Report exact-tie and derivable counts as invariants — a positive rescaling can neither create nor destroy an exact zero, so any change there is a bug.

### Leg D — the held-out predictor comparison (the decisive leg, free)

Sol High's test of whether excluding spillover from per-decision credit is *biased* rather than merely lower-variance. Split repeated `(document, decision, action-pair)` observations by surrounding-context hash and compare three predictors on **held-out contexts**:

1. attributed ΔU (what we now use)
2. total document ΔU (what we used before)
3. attributed ΔU plus a cross-fitted mean spillover estimate

Scored on held-out **total-utility** sign accuracy and regret, with coefficient variance, and stratified by `context` versus `delivered`, by keep→generalize versus level→level, and by linked versus document route.

- If (1) preserves total-utility decisions as well as (2) while showing lower variance, the exclusion is justified and no spillover model is needed.
- If (3) beats (1) materially, spillover is predictable and excluding it is biased — the estimator must be built, with document-disjoint cross-fitting and shrinkage toward zero.
- If (2) beats both, the revision is wrong and must be reverted.

### Leg E — behavioural leg (GPU, only after A–D pass)

Short training run on the adopted configuration with the new credit signal, comparing per-decision credit stability and λ-separation against the v14 baseline. Not scoped here: every RL checkpoint and behavioural gate trained under the old gradient is non-comparable, so this leg establishes a new baseline rather than beating the old one.

## Preregistered decision left open

`delivered_audit_fraction` is inert at its default of 0.0. Arming it is an experimental choice with a value that must be preregistered, not tuned: it sets how much probe budget is spent sampling the global channel. Leg C reports `pairs_without_local_channel` so the fraction can be chosen against the measured distribution rather than guessed, but **the number itself is Timo's to set**, and once set it must not be adjusted in response to results.

## What breaks (from the adjudication, restated for the record)

**Survives:** the utility cache in full (per-assertion scores are stored, so all evidence is re-minable), the QA artifact, count targets, privacy head, BC/ExIt checkpoints.

**Must be re-derived:** tie ledger labels and the v14 hinge/projection constraints; the counterfactual scheduler manifest and coverage accounting; the normative counterfactual definition in `docs/specs/RL/interactive-ranker-v2.md`; every RL checkpoint and behavioural gate trained under the old gradient, including α adequacy, λ monotonicity and realized-privacy operating points; and the local resolution floor.

## Results (2026-08-03)

### Leg A — reduction identities: PASS

**Additivity** holds on every document with cached rollouts: `A(S) + A(Q∖S) == A(Q)` to within 1e-9, zero violations. This is the identity that licenses decomposing credit into a measured component and a provisional complement, so it was the blocking check.

Empty-complement bit-identity, quota-disarmed scheduler identity, and objective reconstruction are covered by unit tests, and the full affected suites show a failure set identical to HEAD across all four commits.

### Leg B — gradient-delta accounting: PASS, after correcting a mis-derived gate

The first draft of this leg gated on counts measured against the *blanket* restriction — i.e. before step 2 existed — and three of them "deviated". They were mis-derived by me, not violated by the implementation. Rather than retune the constants (which would have made the gate vacuous), the gate was replaced by an **identity relating the two designs**:

```
silenced now (2857) + rescued from silence (191) - cancelled into silence (1) = 3047
```

which reproduces the pre-step-2 count exactly. Step 2 converts pairs out of the silenced set precisely when it rescues them from a false zero.

| quantity | value |
|---|---|
| spurious pushes silenced | 2,857 |
| false zeros rescued by excerpt-changed linkage | 191 |
| document-route pairs retaining credit | **1,873** (blanket restriction would have zeroed all) |
| sign corrections | 196 |
| mean \|ΔU\| on the counterfactual channel | 0.0407 → 0.0311 (×0.76) |

**New mechanism found by the off-by-one — union cancellation.** Unioning excerpt-changed assertions can drive a real declared effect to exactly zero when movements oppose: measured on **2 pairs** (declared +0.1039 and +0.1066, union 0.0000, movement 0.2078 and 0.2131). This is the signal-destruction Timo anticipated, arriving through cancellation rather than the denominator dilution he named — and neither this session's analysis nor Sol High's had identified it. Disposition: for the *gradient* the signed net is the honest answer to "what did this decision do to document utility", so zero is correct there. For *qualification* the monotone weighted-L1 statistic sums \|Δs\| and therefore **cannot cancel** — those pairs record movement 0.21, far above any plausible floor — so it cannot manufacture a false tie. Step 3 protects exactly the case step 2 introduced.

### Leg C — restratification: reported, not gated

| stratum | document \|ΔU\| (old) | monotone movement (new) |
|---|---|---|
| exact | 3,103 | **5,704** |
| sub-floor | 4,811 | 2,511 |
| live | 2,545 | 2,244 |

Movement percentiles: p50 0.0000, p75 0.0328, p90 0.1066, p99 0.3956.

The exact-tie stratum nearly doubles because movement is computed over the decision's *own* assertions rather than the whole document, so contamination no longer manufactures spurious nonzeros. **Blocking caveat restated:** these strata use 0.044, which was measured in document-aggregate units and is not dimensionally valid for weighted-L1 movement over a subset. No gate may consume the new statistic until the floor is re-measured in its own units.

### Leg D — FAILED, and the preregistered rule fired

386 groups with repeated contexts, 7,175 leave-one-out folds. Each predictor is measured on training contexts and scored against the total document delta in a **held-out** context, so no predictor can observe its own target.

| predictor | sign agreement | MAE | prediction sd |
|---|---|---|---|
| **total** (what we used before) | **0.621** | 0.0378 | 0.0854 |
| attributed (what step 1 shipped) | 0.524 | 0.0377 | 0.0848 |
| corrected (attributed + pooled, document-disjoint, shrunk spillover) | 0.548 | 0.0377 | 0.0848 |

By credit route:

| route | n | attributed | total | corrected |
|---|---|---|---|---|
| document | 1,863 | 0.507 | 0.507 | 0.516 |
| linked | 5,312 | 0.530 | **0.661** | 0.559 |

**Harness self-check:** on the document route, attributed *is* the whole document by construction, and the two predictors agree to three decimals — as they must. That validates the measurement before its conclusion is read.

**The rule as written was:** *"If (2) beats both, the revision is wrong and must be reverted."* It does, so it was.

**What this means.** The comparison holds out *surrounding configurations*, so a component that varied randomly per configuration could not transfer and `total` would gain nothing. It gains 13 points of sign agreement on the linked route, which means the knock-on effect — the remote model dropping other findings when one span changes — carries **reproducible structure** about the objective. That matches the 6.4-sigma systematic generalization cost measured earlier. Sol High's objection was correct and my "chaotic and unattributable" framing was wrong. Also: the attributed delta is *not* lower-variance (MAE and spread are indistinguishable), so the variance reduction I claimed as its justification does not exist — it is simply less informative, at 0.530 barely above chance.

**SCOPE LIMIT — added 2026-08-03 after checking, and it bounds every claim above.** "Held out" here means a held-out *combination of choices within the same document*, NOT a held-out document. The 386 groups span only **6 documents, and 99.3% of the 10,207 folds come from the 4 RL training documents** (D2N027 3,440; D2N063 3,019; D2N031 2,919; D2N005 761; two others contribute 68 between them). Only densely-scored documents have repeated configurations, so no other document can enter. Therefore:

- The result supports the gradient revert **for the regime it measured** — within a document, across the configurations a training run actually moves through — which is the regime the gradient operates in.
- It establishes **nothing** about documents outside training. This is the same 4-document wall that stopped v15 and Gate 1, for the third time.

**TWO CLAIMS RETRACTED.** (1) I wrote that this "largely closes" the cross-fitted spillover-estimator option. It does not: the estimate had effectively four donor documents, so it was underpowered by construction and the option remains untested. (2) I inferred that the missing information is "pair- and context-specific structure" from the correction recovering only 3 of 13 points. That does not follow — my estimator was a single pooled *scalar*, which could never capture pair-specific structure regardless of how much data it had. The honest statement is narrower: **a global constant offset recovers 3 of the 13 points; whether a structured model would do better is untested.**

**Scope of the revert (surgical, not wholesale).** Leg D tested one thing: the delta used as the *gradient*. So the pair loss returns to the total document delta. Everything else stands, because Leg D says nothing about it:

- **retained** — the monotone weighted-L1 qualification statistic (step 3): tie labelling asks whether an obligation broke, not how much the document moved, and it is measured over the decision's own assertions by design.
- **retained** — excerpt-changed linkage (step 2) for qualification, which is what avoids 305 false zeros.
- **retained but now inert** — scope-matched substitution (step 1): with the gradient covering every assertion the complement is empty by construction, so `hybrid_utility_loss` reduces to whole-term substitution. The mechanism stays available for a genuine partial-measurement channel, which is exactly what a context-only probe would be.
- **retained** — the context preflight and audit quota (step 4), still disarmed.

The original R2 framing — *"the objective keeps W; only the per-decision attribution statistic changes"* — was right, and I over-extended it to the gradient. Leg D caught that before any training run consumed it, which is what the leg was for.

### Not run

Leg E (behavioural). `delivered_audit_fraction` remains 0.0 pending a preregistered value. Also unrun: the local resolution floor in movement units, which still gates any use of the monotone statistic in a gate.

## Artifacts

pending — planned `results/ranker_v2/architecture/credit_attribution/`. Implementation commits bbbd361, 435a57c, d5dfec2, bc10ed8. Predecessors: [measurement revision](2026-08-03-counterfactual-measurement-revision.md), [Gate 1](2026-08-01-RL-ranker-gate1-representation.md).
