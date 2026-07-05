---
type: reference
status: current
created: 2026-07-05
updated: 2026-07-05
tags: [detector, gliner, requirements, spec, quasi, generality, operating-point]
companion: [docs/research/learned-PII-detection.md, research-wiki/training/2026-07-05-FT-detector-v4-base-genfirst-mix.md]
---

# Detector model — target properties & competences

What the ideal PII/QI span detector must **do** (competences) and must **be** (properties), with
measured reference points from the FT-detector runs (v1–v4). This is the spec the next detector is
selected against; it is not a plan and carries no run steps.

Every number here is a realized measurement from a committed run (cited), an outcome — never a knob tuned
to hit a target (empirical-honesty rule, CLAUDE.md). "Best-so-far" columns say which run reached it, so a
successor is judged against the actual Pareto frontier, not an aspiration.

**Gate status — this is a provisional v0 acceptance spec (Codex R2).** The **release gates** are the recall
axes: C1 (DIRECT/QUASI), C2 (gap types incl. ORG ≥ 0.90), C5 (span width), PR1 (robustness), PR3 (precision
floor). **PR7–PR11 (typed/boundary accuracy, calibration, latency, scope, reproducibility) and the
end-to-end privacy/utility gate (PR6) are, for now, MANDATORY REPORTING but not release gates** — they lack
measured baselines. The spec hardens to v1 (they become gates) once those are measured. Adopt it as the
recall-acceptance spec, not a final release contract.

## Role in the pipeline
The detector locates the spans of `doc_orig` to anonymize (feeds the substitutor/ranker). It is a
**composition**: a supervised fixed-schema core **plus** a cheap per-user-type path (gazetteer / zero-shot
label phrase / fine-tune). Missing a sensitive span is a **privacy leak**; over-flagging is
ranker-absorbable noise. So recall on the sensitive set is the axis that must not regress; precision is a
bounded diagnostic.

## Definitions
- **DIRECT / QUASI** — TAB identifier classes. DIRECT = a mention that alone re-identifies (name, SSN);
  QUASI = quasi-identifier that re-identifies in combination (age, city, dates, employer, identifying
  events). Gold = union of annotators' DIRECT+QUASI mentions.
- **TAB-8 schema** — the fixed types: PERSON, ORG, LOC, DATETIME, CODE, QUANTITY, **DEM** (demographics:
  nationality/ethnicity/religion/profession/age), **MISC** (other identifying attribute or event).
- **Gap types** — MISC, DEM, identifying QUANTITY: the guideline-defined categories **no zero-shot label
  phrase reaches** (every off-the-shelf checkpoint stalls at MISC ≤ 0.32). Their competence comes only from
  supervision.
- **Open-label generality** — zero-shot recall on entity types **outside** the schema, from a natural-language
  label phrase alone (measured on held-out MultiNERD types: animal, disease, food, …). Proxy for
  user-defined-type extensibility. Reported as any-recall + precision at a fixed threshold.
- **any / typed recall** — `any` = predicted span overlaps a gold span (type-agnostic); `typed` = also
  correct type.
- **Operating point** — the score threshold at which the detector is run; a point on the PR curve chosen
  post-training, per corpus. Not a training parameter.
- **precision (proxy)** — fraction of predicted spans overlapping any annotated mention (incl. NO_MASK);
  lenient, diagnostic. See `docs/issues/performance.md` / the gate.

## Competences (capabilities the ideal detector must have)

### C1 — Fixed-schema QUASI/DIRECT coverage (privacy-critical)
Detect DIRECT and QUASI mentions of the TAB-8 types at high recall. **This is the non-negotiable axis** — a
missed sensitive span leaves an identifier in `doc_p`.
- DIRECT any-recall ≈ **1.00** (achieved v1–v4).
- QUASI any-recall **≥ 0.95** (v2 0.979, v3 0.973, v4 0.966).

### C2 — Guideline gap types (the reason the detector must be trained, not zero-shot)
Recognize MISC (identifying events/attributes), DEM (demographics), and identifying QUANTITY — categories
unreachable from a label phrase. Supervision is mandatory: stock knowledgator MISC = 0.32 (base) / 0.21
(large) → fine-tuned 0.86–0.93.
- MISC any-recall **≥ 0.89** (best: v4 base+genfirst **0.925**; v2 0.895; v3 0.861).
- DEM **≥ 0.96**, identifying QUANTITY **≥ 0.97** (met v1–v4).

### C3 — Open-label generality / tailorability (must survive fine-tuning)
Recognize user-defined types **outside** the schema, zero-shot from a label phrase — and **retain** this
after fixed-schema fine-tuning (the failure mode is catastrophic label-narrowing: v1 eroded stock 0.925 →
0.835 — from the v1 training record, measured at thr 0.3; not in this session's re-run JSONs).
- Held-out any-recall **> 0.90**, reported **with precision** (v3 large 0.988 @ prec 0.444; v4 base+genfirst
  0.918 @ 0.435; stock-base 0.925 @ 0.421). A recall gain at collapsing precision does not count —
  compare at matched precision / firing.

### C4 — Cross-domain transfer at one schema
Hold C1/C2 recall across domains (legal, biography, clinical, social) **without type sprawl** — same 8
types, more domains. Confirmed legal↔bio (bio-test QUASI 0.989, v2/v3 records — near-ceiling, so
undiscriminating); **open gap:** MISC transfer beyond legal+bio is unproven (no clinical/social
identifying-event gold).

### C5 — Long-span emission
Emit spans as long as the annotation guideline allows (identifying events run > 12 words; the data caps at
60). The candidate-span width ceiling (`max_width`) must be **≥ 60**. base's `max_width = 12` is a
structural ceiling that silently drops long MISC; large's 100 covers it (but costs VRAM for spans > 60 that
never occur — cap ≈ 60).

### C6 — Composition / per-user-type extension
Accept new sensitive types via the cheapest sufficient path — gazetteer, zero-shot label phrase, or
targeted fine-tune — **without** retraining the core or losing the open-label interface (C3).

## Properties (qualities the detector must hold)

### PR1 — Bounded, valid output at every threshold (robustness)
No crashes, no spans outside the text, at any operating point. (The deberta-v3-large backbone emits
phantom padding-region spans at low threshold — a GLiNER decoder defect; the detector must guard against it,
as `src/cloak/detect.py` now does. Verified: `src/cloak/tests/test_detect_padding_guard.py`.)

### PR2 — Model- & domain-specific operating point, principled selection
The operating point is **per corpus** (TAB ≈ 0.02, non-TAB ≈ 0.3 — the fine-tuned model is sharp on its
domain and peaks at a low threshold there). Checkpoints/models are selected by **recall-at-matched-precision
/ AUPRC**, never recall at a fixed threshold (the fixed-0.3 rule manufactured a phantom "overfit" — v2 audit).

### PR3 — Precision floor (diagnostic, not the comparison axis)
precision(proxy) **≥ 0.716** at the operating point (v1 baseline floor); realized **0.786–0.850** on TAB test
(v4 0.786, v2 0.814, v3 0.850; v4 dev dips to ~0.777 — still above floor). Over-detection
above the floor is acceptable (ranker-absorbable); it is never traded for a secondary metric via a per-model
knob. **Warning gate (Codex R2):** 0.716 is a lenient fail-floor; since v2–v4 already run 0.786–0.850, also
flag any candidate with a *material precision regression vs the selected predecessor at matched recall* — a
drop within-floor can still cost downstream utility, so it warrants review even if it passes the floor.

### PR4 — Local, closed-box-friendly efficiency
Runs locally on one iGPU (the pipeline hides content from the *remote* model; the detector must not need a
remote call). Prefer the smallest backbone that meets C1–C5: base (deberta-v3-small, ~5.8 GB, fast) is the
default; the large backbone (~50 GB, ~10× slower) is justified **only** if an axis needs it — and on TAB it
did **not** (large's TAB recall ≤ base's). See PR2 in `2026-07-04-FT-detector-v3-large-balanced.md`.

### PR5 — Type-balanced, not type-robbing
Improving a scarce type must not starve a common one. The genfirst mix + rare-type upsampling lifted MISC
(0.925) but **regressed ORG 0.948 → 0.848** by dropping TAB share and upsampling only MISC/DEM/QUANT. The
ideal detector holds **every** QUASI type ≥ ~0.90 (ORG floor is the current casualty to fix).

### PR6 — Outcome-measured, honestly
Detector quality is a diagnostic; the quantity that decides the pipeline is privacy-vs-utility against an
**LLM re-identification attacker** on `doc_p`/`out_final`. Detector metrics are reported as outcomes with
the win *and* the regression, never a tuned secondary quantity. **This end-to-end gate is not yet run** — an
open gap (below): all detector numbers here are upstream proxies until the re-id attack is measured.

### PR7 — Boundary & typed accuracy (not just any-recall)
`any`-recall (overlap) is the headline, but anonymization needs the *right span boundaries* (a partial span
leaves identifying text) and, for the substitutor, the *right type*. Report **typed** recall alongside
`any` per type, and a boundary-exactness rate; target typed ≥ ~0.90 on DIRECT and the well-supported QUASI
types. (Current `typed` numbers are recorded by the gate but not yet gated.)

### PR8 — Calibration & threshold stability
The per-corpus operating point (PR2) is only useful if the score distribution is stable: a small domain
shift must not swing recall wildly at a fixed threshold. Target: recall change ≤ a few points for a ±0.01
threshold move around the op point; monitor score-distribution drift across domains. (Fine-tuned models are
*sharp* — low-threshold op points are sensitive; this is why selection is matched-precision, not fixed-thr.)

### PR9 — Latency & memory budget (local, interactive)
Bounded inference cost on one iGPU. base ≈ 5.8 GB / fast; large ≈ 50 GB / ~10× slower (PR4). `max_width`
inflates candidate-span compute (12→60 ≈ 5× the span set) — keep it at the minimum covering C5 (~60). The
selection sweep itself must be affordable (see `docs/issues/performance.md` — sweep-mode ~5–7×). No hard
SLA yet; set one when the detector is wired into the interactive path.

### PR10 — Scope: language, domain, label-phrasing sensitivity
- **Language:** English only for now (TAB/ECHR legal, Wikipedia-bio). Multilingual is **out of scope** until
  a multilingual eval + data exist — state it, don't assume transfer.
- **Label-phrasing sensitivity:** the zero-shot/label-phrase interface is sensitive to wording; the 8 TAB
  phrases are **fixed and held constant across all models** (honesty — no per-model label tuning). A property
  the detector must tolerate: modest rephrasing of a user's type label should not collapse recall.

### PR11 — Reproducibility & versioning
Every run is reproducible: fixed `seed`, a `run_manifest.json` (init, data, hyperparams, lib versions),
and a versioned training record (`YYYY-MM-DD-FT-detector-vN-…`). Model + dataset artifacts are pinned;
result JSONs are committed. Selection is documented (which checkpoint, which threshold, why).

## Operating targets (reference frontier, measured)

| Axis | Requirement | Best so far (run) | Notes |
|---|---|---|---|
| DIRECT any | ≈ 1.00 | 1.000 (v1–v4) | |
| QUASI any | ≥ 0.95 | 0.979 (v2 base) | v3/v4 slightly lower |
| MISC any | ≥ 0.89 | **0.925 (v4 base+genfirst)** | supervision-only competence |
| DEM any | ≥ 0.96 | 0.973 (v2) | |
| QUANTITY any | ≥ 0.97 | 0.997 (v3) | |
| ORG any | ≥ 0.90 | 0.948 (v2) | v4 regressed to 0.848 (PR5) |
| precision(proxy) | ≥ 0.716 | 0.786–0.850 | diagnostic (v4 test 0.786) |
| generality any | > 0.90 (w/ precision) | 0.988 @0.444 (v3 large) | 0.918 @0.435 (v4 base) |
| span width | max_width ≥ 60 | large 100 / base 12 | base ceiling drops long MISC |

## Measured tensions (the Pareto reality — no single config wins all)
- **Generality ↔ TAB specialization.** A model that keeps firing on arbitrary open labels (generality ↑) is
  less specialized to the 8 types (TAB recall ↓). v2 base specialized hard (TAB↑, gen 0.843); v3 large
  generalized (gen 0.988, TAB slightly ↓).
- **Rare-type balancing ↔ common-type recall (PR5).** Upsampling MISC/DEM/QUANT + a generality-first mix
  lifts MISC but starves ORG and dips precision.
- **Backbone size ↔ TAB recall & cost.** Large lifts generality (with the mix) but not TAB, at ~10× cost and
  a decoder bug (PR1). Capacity was not the TAB lever.
- **Attribution (not a clean factorial — caveat):** generality rose via *both* the diverse-label mix
  (v2→v4 on base, single-variable: +0.075) and the large backbone (v4→v3: +0.070). But v3-vs-v4 also differ
  in lr (6e-6 vs 1e-5) and batch (16 vs 8), so the "+0.070 backbone" increment is **confounded**, not a
  clean isolate. The mix increment (+0.075) is clean (v4 changed only data vs v2). Read as "both plausibly
  contribute", not a measured decomposition (`2026-07-05-FT-detector-v4-base-genfirst-mix.md`).

## Non-goals
- Not a general NER tagger — the schema is the TAB-8 sensitive set (+ user-defined extensions), not all
  entities.
- Not the privacy metric — that is the downstream re-identification attack, not detector recall (PR6).
- Not a remote/API model — must run locally (PR4).
- **Not coreference / entity-linking.** The detector emits *spans*, not resolved entity clusters. The
  pipeline's consistent-replacement need (the same person masked identically everywhere) is a **separate,
  downstream stage** (`coref_chains` in `src/cloak/detect.py`) that consumes detector spans — a named
  dependency, out of scope for the detector model itself.

## Evaluation protocol
- **Fixed schema (C1/C2/PR3):** `scripts/latticecloak_detection_gate.py` on `corpora/tab/echr_{dev,test}.json`;
  per-type any/typed recall + precision(proxy); select on dev, one test run per final config.
- **Generality (C3):** `scripts/spikes/pii_zeroshot_generality.py` on held-out MultiNERD types (reserved —
  never train on it); report any-recall + precision + typed.
- **Cross-domain (C4):** the gate on `corpora/wikipedia_bio/test.json` (real bio gold); clinical/social
  qualitative until gold exists.
- **Selection (PR2):** dev threshold sweep → recall-at-matched-precision / AUPRC, then per-corpus op point.
- **Robustness (PR1):** the padding-guard regression test; no out-of-text spans at threshold 0.02.

## Open gaps (what the ideal detector still lacks)
1. **ORG regression under balancing (PR5)** — the immediate fix: gentler generality-first mix (Pile ≈ 15%,
   don't drop TAB share so far) and/or floor ORG in the balancer.
2. **Cross-domain MISC (C4)** — no clinical/social identifying-event supervision exists; transfer unproven.
3. **Generality at matched precision (C3)** — currently compared at a fixed threshold; make the comparison
   precision-matched to rule out over-firing definitively.
4. **A config that holds v2's TAB (MISC 0.895, ORG 0.948) *and* recovers generality** — none does yet; the
   candidate next run is base + a gentler generality mix (see gap 1).
5. **End-to-end privacy/utility unmeasured (PR6).** Every number here is an upstream detector proxy; the
   deciding metric — LLM re-identification-attacker success on `doc_p`/`out_final` and downstream utility —
   has not been run against any detector version. This is the gap that most limits what the spec can claim.
6. **Typed/boundary accuracy & calibration (PR7/PR8) recorded but not gated** — `typed` recall exists in the
   gate output; boundary-exactness and threshold-stability are not yet measured or bounded.

## Sources
FT-detector runs: [v1](../../research-wiki/training/2026-07-03-FT-detector-v1-tab-quasi.md) ·
[v2](../../research-wiki/training/2026-07-04-FT-detector-v2-quasi.md) ·
[v3](../../research-wiki/training/2026-07-04-FT-detector-v3-large-balanced.md) ·
[v4](../../research-wiki/training/2026-07-05-FT-detector-v4-base-genfirst-mix.md).
Rationale/§5: [learned-PII-detection.md](../research/learned-PII-detection.md).
Perf: [performance.md](../issues/performance.md).
