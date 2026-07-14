---
type: experiment
node_id: exp:detector-finding-vs-diagnosis-separation
title: "Separating exam-findings from diagnoses in the qa-v2 clinical detector: GLiNER label schema vs confidence threshold"
idea_id: "idea:qa-v2-condition-over-capture"
verdict: partial
confidence: medium
date: "2026-07-14"
result: "GLiNER label-schema split is not worth adding: no added label cleanly separates the four exam-findings (edema/erythema/immunocompromised/acute exacerbation) from real diagnoses — `disease` rejects all four findings but also misses hyperthyroidism (0.0), and adding probe labels destabilizes existing types (kidney transplant flips health-condition->medical process). Detector `condition` confidence is the usable lever: a 0.5 health-condition admission threshold drops acute exacerbation (0.382) and immunocompromised (0.410) while keeping edema (0.677), erythema (0.634), and every real diagnosis (>=0.736). Adopted the 0.5 gate; did NOT add disease/symptom/clinical-sign labels."
hardware: "AMD Strix Halo iGPU (gfx1151), host .venv torch"
duration: "~1 session of spikes"
provenance: "scripts/spikes/detector_finding_label_split.py (superseded, exact-surface-match filter bug), clean span-overlap re-measure /tmp/clean_measure.log; frozen scores from /tmp/qa-v2-d2n002-newdet.json; model knowledgator/gliner-pii-large-v1.0 @0.35"
companion: "docs/issues/detector-misclassifications.md, docs/specs/qa-builder-v2.md"
added: 2026-07-14T00:00:00Z
tags: ["detector", "gliner", "zero-shot", "clinical", "qa-v2", "confidence-threshold", "empirical-honesty"]
---

# Separating exam-findings from diagnoses in the qa-v2 clinical detector

**verdict:** `partial` (label-schema split rejected; a 0.5 confidence admission threshold adopted as the usable-but-imperfect lever)  ·  **confidence:** `medium`  ·  tests `idea:qa-v2-condition-over-capture`

## Question

The merged qa-v2 clinical detector fix removed junk types (`demographic-other`, PERSON, CODE) but now
freezes descriptive exam-findings — `edema`, `erythema`, `immunocompromised`, `acute exacerbation` — as
controlled `health-condition` decisions (D2N002 eligible spans grew 14->18). `acute exacerbation`
(a modifier of `arthritis`, detector score 0.382) then hijacked the QA relation
`prescribed_with(arthritis -> ultram)` into `prescribed_with(acute exacerbation -> ultram)`. Can the
detector separate these findings from real diagnoses so they never become controlled conditions?

## Method

`knowledgator/gliner-pii-large-v1.0`, threshold 0.35, on `aci/D2N002`. Two discriminators:
1. **GLiNER label schema.** Prompt GLiNER with added zero-shot labels (`symptom`,
   `physical examination finding`, `clinical sign`, `disease`, `body part`) alongside `condition` and
   see whether findings move off `condition`. (A first spike reported all-zero finding scores; that was
   a measurement bug — it filtered entities by exact surface text, dropping hits on other span
   boundaries. Re-measured by span overlap.)
2. **Detector confidence.** The frozen `detector_provenance.score` for each span's `condition` label.

## Result

**The model is zero-shot responsive** — added labels fire on the right spans: `symptom`->joint
pain/fatigue/nausea, `clinical sign`->vital signs/murmur, `disease`->arthritis/hypothyroidism/polycystic
kidneys, `body part`->knees. But **no added label cleanly separates the four findings from all real
diagnoses** (clean span-overlap scores):

| surface | condition | disease | symptom | clin.sign |
|---|---:|---:|---:|---:|
| arthritis | 0.575 | 0.771 | 0 | 0 |
| hypothyroidism | 0.692 | 0.681 | 0 | 0 |
| hyperthyroidism | 0.538 | **0.000** | 0 | 0 |
| kidney transplant | 0.000 | 0.000 | 0 | 0 |
| edema | 0.347 | 0.000 | 0 | 0 |
| erythema | 0.314 | 0.000 | 0 | 0 |
| immunocompromised | 0.544 | 0.000 | 0 | 0 |
| acute exacerbation | 0.355 | 0.000 | 0 | 0 |

`disease` rejects all four findings but also misses `hyperthyroidism` (0.000) — a real diagnosis — so a
`disease`-positive rule drops legitimate conditions. `symptom`/`clinical sign` are zero on all eight
targets (they capture *other* spans). And adding probe labels destabilizes existing classifications:
`kidney transplant` flips `health-condition`->`medical process`. **Adding these labels is not worth it.**

**Confidence is the usable lever.** Frozen production `condition` scores (no probe labels):

| frozen score | surface | kept by 0.5 gate? |
|---:|---|---|
| 0.382 | acute exacerbation | dropped |
| 0.410 | immunocompromised | dropped |
| 0.634 | erythema | kept |
| 0.677 | edema | kept |
| 0.736 | hyperthyroidism | kept |
| 0.941–0.979 | kidney transplant, hypothyroidism, arthritis | kept |

A **0.5 health-condition admission threshold** drops the harmful `acute exacerbation` (0.382) and
`immunocompromised` (0.410) with a wide margin (nearest kept = erythema 0.634; nearest dropped =
immunocompromised 0.410), while keeping `edema`/`erythema` and every real diagnosis (min 0.736).

## Decision

- **Do not** add `disease`/`symptom`/`clinical sign` labels to the detector schema — partial separation,
  destabilizes existing types, not worth the complexity.
- **Adopt** a 0.5 `health-condition` admission threshold. `acute exacerbation` removal is the win;
  `edema`/`erythema` remaining as conditions is acceptable. (`immunocompromised` is also dropped at 0.5.)

## Caveats / limitations

- Single document (`aci/D2N002`); the finding/diagnosis confidence bands overlap near 0.63–0.74, so a
  real diagnosis on another document could score <0.5 and be wrongly dropped. The threshold needs a
  matched-setting recall check on a wider clinical slice before it is treated as frozen.
- This separates *findings* from *diagnoses* by confidence only; it is not a semantic finding detector.

## Sources

- `docs/issues/detector-misclassifications.md` (finding-over-capture register, corrected label evidence)
- `docs/specs/qa-builder-v2.md` (relation-eligibility contract)
