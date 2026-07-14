---
type: research
status: current
created: 2026-07-13
updated: 2026-07-14
tags: [detector, clinical, aci, misclassification, transcript-normalization, runtime-types, issue-register]
companion: [2026-07-10-detector-junk-and-noise-gate-limits.md,
            ../specs/RL/interactive-ranker-v2.md,
            ../specs/qa-builder-v2.md]
---

# Issue — clinical detector misclassifications in ACI D2N002

The pinned RL-v2 clinical detector still produces obvious false positives and wrong runtime types
on `aci/D2N002`. These errors were observed after replacing the stale coarse `DEM/MISC` ranker
environment with the intended detector configuration:

- model: `knowledgator/gliner-pii-large-v1.0`;
- threshold: `0.35`;
- label schema: `knowledgator-native-clinical-v1`;
- composition: GLiNER plus Presidio under the clinical profile.

The smoke artifact was built from `/tmp/task_arms_qa_v2_d2n002.json`; its detector pin is recorded in
`/tmp/qa_utility_d2n002_correct_detector.json`. These paths are ephemeral smoke outputs, not committed
evidence artifacts. The observations below must be reproduced into a durable diagnostic artifact
before quantitative claims.

## Observed failures

| Surface | Emitted runtime type | Failure |
|---|---|---|
| `wan na` | `PERSON` | False positive caused by split conversational orthography; the lexical item is `wanna`, not a person. The same transcript family reportedly contains `gon na` for `gonna`. |
| `white blood cell count` | `CODE` | Clinical measurement name misclassified as an identifier/code. |
| `heart rate` | `CODE` | Clinical vital-sign name misclassified as an identifier/code. |
| `two out of six` | `CODE` | Clinical measurement/value phrase misclassified as an identifier/code. |
| `andrew` (three occurrences; intended name `Andrew`) | `demographic-other` | All three occurrences of the person name were routed to the demographic fallback instead of `PERSON`. |

### `demographic-other` is an erroneous RL-v2 clinical runtime type

`demographic-other` must be removed from the RL-v2 clinical detector/runtime output path. It is a
fine-DEM residual fallback, not a clinical-PII type and not a valid destination for a person name.
Its placeholder-first policy turns a type-routing error into unnecessary document damage. The
clinical detector must emit `PERSON` for name labels and reject or diagnose any residual
demographic fallback instead of freezing it as a decision.

This is deliberately scoped to the RL-v2 clinical configuration. The residual type remains a
separate legacy/fine-DEM concern until its other consumers are audited; do not silently remove it
globally under this issue.

The current frozen occurrence provenance records the detector configuration and confidence but not
the winning detector source or overlap-resolution trace. Therefore this evidence does **not** yet
attribute each error to GLiNER, Presidio, role-phrase rewriting, or cross-source overlap resolution.
That missing attribution is itself a diagnostic gap.

## Why this matters

- False `PERSON` and `CODE` detections introduce unnecessary placeholders and can degrade clinical
  task utility.
- Wrong runtime types select the wrong substitution semantics and prevent meaningful lattice actions.
- Detector errors are frozen and shared by QA and RL, so they become common-mode environment errors;
  QA must not reinterpret them as annotation truth.
- A detector configuration is not acceptable merely because it removes legacy `DEM/MISC` decisions.
  The frozen inventory still requires qualitative and aggregate error gates.

## Required investigation

1. Preserve per-occurrence source provenance: GLiNER label, Presidio recognizer, raw scores, overlap
   candidates, winning candidate, and any post-detection retyping.
2. Reproduce the failures on the frozen ACI development slice and count split-token false positives,
   clinical-measurement-to-`CODE` errors, and person-name routing errors.
3. Test transcript-aware normalization or augmentation for split forms such as `wan na` and `gon na`
   without silently changing source offsets.
4. Test a deterministic clinical-measurement routing rule or positive type contract before adding
   string-specific deny-list patches.
5. Add pre-training gates that fail when these error classes exceed preregistered thresholds; do not
   tune them from final RL or attacker results.
6. Remove `demographic-other` from the RL-v2 clinical runtime type contract and add a gate rejecting
   any occurrence routed to it; preserve an attributable rejection record rather than a decision.

## Non-decisions

No remediation is approved by this issue. In particular, do not silently delete these spans, remap
all `CODE` detections, or normalize transcript text without offset-preserving evidence. The immediate
requirement is attributable diagnostics followed by the simplest measured fix.

## Reproduction commands

```bash
PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_arms_artifact.py \
  --n-docs 3 --corpora clinical --detector-config qa-v2-clinical \
  --out /tmp/task_arms_qa_v2_d2n002.json

jq '[.clinical["aci/D2N002"].tau_walk[1][] | {surface, type, score}]
    | group_by(.surface, .type)
    | map({surface: .[0].surface, type: .[0].type, count: length, scores: map(.score)})' \
  /tmp/task_arms_qa_v2_d2n002.json
```

## Resolution measured on 2026-07-14

The attributable detector and QA-v2 runtime contracts resolve the four preregistered failure
classes on both the representative document and the complete local ACI slice. Evidence is committed
as the [D2N002 gate artifact](../../results/clinical_detector_gate_aci_d2n002.json) and the
[67-document ACI gate artifact](../../results/clinical_detector_gate_aci.json).

Both artifacts pin `knowledgator/gliner-pii-large-v1.0` at threshold `0.35`, label schema
`knowledgator-native-clinical-v1` plus its exact `label_map`, the controlled runtime types, the
clinical profile, and GLiNER plus Presidio composition. Each document carries a SHA-256 source hash.
They were generated with one flattened detector batch per command:

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/python -u \
  scripts/clinical_detector_gate.py \
  --corpus aci \
  --doc-id aci/D2N002 \
  --out results/clinical_detector_gate_aci_d2n002.json

PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/python -u \
  scripts/clinical_detector_gate.py \
  --corpus aci \
  --out results/clinical_detector_gate_aci.json
```

Measured outcomes:

| Scope | Documents | Wall time | Split-contraction `PERSON` | Native clinical `CODE` without identifier shape | Explicit name not `PERSON` | Frozen `demographic-other` |
|---|---:|---:|---:|---:|---:|---:|
| `aci/D2N002` | 1 | 9.218 s | 0 | 0 | 0 | 0 |
| Complete local ACI slice | 67 | 86.912 s | 0 | 0 | 0 | 0 |

The D2N002 artifact contains 45 accepted candidates, 6 rejected candidates, 4 overlap losers,
8 normalization events, and 0 post-detection rejections. All three `andrew` occurrences are accepted
as `PERSON` from native GLiNER label `name`; the duplicate Presidio candidates remain visible as
same-type overlap losers. `heart rate`, `two out of six`, and `white blood cell count` are retained as
rejected native `medical code` candidates with reason
`clinical_code_without_identifier_shape`. The analysis view records `wan na`→`wanna ` and
`gon na`→`gonna ` at unchanged offsets.

Across all 67 ACI documents, the artifact contains 3,021 accepted candidates, 293 rejected
candidates, 224 overlap losers, 376 normalization events, and 42 post-detection rejections.
Representative accepted evidence includes `martha` as `PERSON/name`; rejected evidence includes
`three out of six systolic ejection murmur` and `one plus` under the positive code contract.

The 42 post-detection rejections are not frozen `demographic-other` occurrences and therefore do not
fail the preregistered gate. They are nevertheless an unresolved recall/typing audit queue: examples
include `nick`, `cushing`, and `miller`, where a lowercase Presidio `PERSON` candidate was rejected
after the legacy role-word path proposed `demographic-other`. The gate also does not establish overall
clinical entity recall, privacy, or downstream utility; those require their own matched-setting
evaluations.

The standardized `scripts/harness/perf_gate.md` prompt and codex-rescue/auto-review backend were not
available in this worktree. The runs used the approved flattened-batch design and measured GPU wall
times above; no standardized external performance review was run or implied.
