---
type: handoff
status: current
created: 2026-07-05
updated: 2026-07-05
tags: [extractor, detector-pointer, semantic-window, audit, handoff]
companion: [../plans/2026-07-05-extractor-inverse-designs.md, ../plans/2026-07-05-detector-pointer-extractor.md]
---

# Handoff: extractor pointer implementation and audit results

## Current State

This session implemented two local extractor improvements in `src/cloak/extract.py`:

- **Semantic-window fallback** inside the default `invert()` cascade.
  - Runs after placeholder, exact, and fuzzy90 paths.
  - Uses fuzzy 60-90 candidate windows, MiniLM cosine, a best-vs-runner-up margin, and cheap type sanity.
  - Tuned conservatively after a false null-control match (`something` -> `anything`).
  - Current constants: `SEMANTIC_MIN=0.70`, `SEMANTIC_MARGIN=0.04`, `_GENERIC_SEMANTIC_FILLS={"something"}`.
- **Detector-pointer arm** as explicit API `invert_detector_pointer(...)`.
  - Default `invert()` is still the reward-path extractor; pointer is opt-in.
  - Rule pre-pass: placeholders, exact, fuzzy90.
  - Residue goes to detector typed candidates.
  - Detector model is explicit: `DETECTOR_POINTER_GLINER_MODEL = "data/models/pii_gliner_multidomain/checkpoint-2479"`.
  - Candidate spans are dilated by +/- 2 token boundaries.
  - Pointer scoring currently uses MiniLM via `_pointer_scores`; no learned `FT-extractor` checkpoint exists yet.
  - Stats added: `gen_pointer`, `gen_abstain`.

Supporting changes:

- `scripts/surrogate_env_diagnostics.py` now counts `gen_semantic` and `gen_pointer` in generalization firings.
- New tests in `src/cloak/tests/test_extract.py` cover semantic fallback, pointer assignment, ambiguity abstain, rule pre-pass preservation, and explicit detector checkpoint construction.
- New spike scripts:
  - `scripts/spikes/extractor_semantic_verify.py`
  - `scripts/spikes/extractor_pointer_compare.py`
  - `scripts/spikes/extractor_pointer_by_type.py`
- New experiment record:
  - `research-wiki/experiments/extractor-pointer-by-type.md`

## Verification

Local tests:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests
```

Result: `34 passed`, with only existing SWIG deprecation warnings.

Self-check:

```bash
PYTHONPATH=src .venv/bin/python src/cloak/extract.py
```

Result: self-check OK.

## Audit Results

Baseline audit rerun:

```bash
INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python -u \
  scripts/spikes/extractor_miss_audit.py \
  --env data/ranker_env.json --arms data/task_arms_tau0.02.json \
  --corpora clinical --n-docs 16 --workers 6 \
  > results/extractor_miss_audit.log 2>&1
```

Output: `results/extractor_miss_audit.json`.

Counts over 75 level fills:

- exact: 15
- fuzzy90: 12
- band60_90: 33
- absent: 15
- rule hit rate over all level fills: 27/75 = 0.360
- placeholder echo rate: 0.239 over 67 placeholders

## Echoed-Only By-Type Comparison

The user correctly noted that true absent spans are not extractor opportunities. The final comparison uses denominator = generalized spans classified by the audit echo classifier as exact, fuzzy90, or band60_90. `absent` is excluded.

Run:

```bash
INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python -u \
  scripts/spikes/extractor_pointer_by_type.py \
  --env data/ranker_env.json --arms data/task_arms_tau0.02.json \
  --corpora clinical --n-docs 16 --workers 6 --tau-det 0.3 \
  > results/extractor_pointer_by_type.log 2>&1
```

Output: `results/extractor_pointer_by_type.json`.

Totals over echoed generalized spans only:

| Extractor | Extracted | Rate |
|---|---:|---:|
| Rule baseline | 27/60 | 45.0% |
| Semantic-window | 28/60 | 46.7% |
| Detector-pointer | 28/60 | 46.7% |

By type:

| Type | Present in `out_p` | Rule | Semantic-window | Detector-pointer |
|---|---:|---:|---:|---:|
| PERSON | 0 | n/a | n/a | n/a |
| ORG | 0 | n/a | n/a | n/a |
| LOC | 1 | 0/1 | 0/1 | 0/1 |
| DATETIME | 10 | 1/10 | 1/10 | 2/10 |
| CODE | 0 | n/a | n/a | n/a |
| QUANTITY | 5 | 5/5 | 5/5 | 5/5 |
| DEM | 32 | 18/32 | 18/32 | 18/32 |
| MISC | 12 | 3/12 | 4/12 | 3/12 |

Non-rule recoveries:

- Semantic-window recovered MISC: `lasix` from fill `a drug`.
- Detector-pointer recovered DATETIME: `last weekend` from fill `at some point`.

Important caveat documented in the experiment record: `present_in_out_p` is based on the audit fuzzy echo classifier. The `band60_90` bucket means a loose aligned candidate exists; it is not a human label that the remote answer truly contains the span.

## Observations

- Detector-pointer ties semantic-window overall on this slice, but helps a different type.
- Detector-pointer improves DATETIME from 10.0% to 20.0%, a one-span absolute gain.
- Detector-pointer does not improve the largest bucket, DEM.
- QUANTITY is already saturated by the rule extractor.
- PERSON, ORG, and CODE have no echoed generalized spans in this audit slice.
- Current detector-pointer is **not** a learned pointer model. It tests the typed candidate interface plus MiniLM scoring. Do not treat this as the ceiling for `FT-extractor`.

## Recommended Next Steps

1. Run a detector-ceiling spike by type on answer prose:
   - For each audit `band60_90` candidate, check whether `Detector(gliner_model="data/models/pii_gliner_multidomain/checkpoint-2479")` proposes an overlapping same-type span.
   - Report ceiling by type, especially DEM and MISC.
2. If detector ceiling is acceptable, spec `FT-extractor v1` before any training run per the repo schema.
3. If detector ceiling is poor for DEM/MISC, do not train the pointer yet; route work to detector answer-prose fine-tuning or candidate proposal changes.
4. Any extractor default change still invalidates anchors/probes/scan verdicts/policies per re-gate discipline.

## Suggested Skills For Next Session

- `diagnose` if debugging detector/pointer failure cases.
- `improve-codebase-architecture` if refactoring extractor APIs beyond this explicit arm.
- `handoff` again before launching long training or probe rebuild work.
