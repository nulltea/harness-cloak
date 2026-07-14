---
type: plan
status: current
created: 2026-07-14
updated: 2026-07-14
tags: [detector, clinical, aci, provenance, transcript-normalization, runtime-types, qa-v2]
companion: ../issues/detector-misclassifications.md
---

# Clinical detector attribution and runtime contract

## Goal

Fix the attributable detector and post-detection failures recorded in
[`docs/issues/detector-misclassifications.md`](../issues/detector-misclassifications.md) without
changing the pinned QA-v2 clinical model, threshold, label set, or realized-privacy operating point.
The resulting artifact must make every accepted or rejected occurrence traceable to its raw detector
candidate and must prevent residual `demographic-other` decisions in the QA-v2 clinical path.

The fixed path remains scoped to the pinned QA-v2 clinical configuration:

- model: `knowledgator/gliner-pii-large-v1.0`;
- threshold: `0.35`;
- label schema: `knowledgator-native-clinical-v1`;
- composition: GLiNER plus Presidio with the clinical profile.

The legacy fine-demographic path retains `demographic-other` until its consumers are audited
separately.

## Measured root causes

An attributable run on `aci/D2N002` separated the observed symptoms into three causes:

| Surface | Raw source | Raw label | Detector result | Fault location |
|---|---|---|---|---|
| `andrew` | GLiNER and Presidio agree | `name` / `PERSON` | `PERSON` | `substitute()` retypes the lowercase name because WordNet contains `andrew.n.01` |
| `wan na` | Presidio `SpacyRecognizer` | `PERSON` | `PERSON` | spaCy parses split conversational orthography as a name |
| `heart rate` | GLiNER | `medical code` | `CODE` | the native label lacks a positive identifier-shape contract |
| `two out of six` | GLiNER | `medical code` | `CODE` | the native label lacks a positive identifier-shape contract |
| `white blood cell count` | GLiNER | `medical code` | `CODE` | the native label lacks a positive identifier-shape contract |

The source transcript and frozen arms artifact contain three `andrew` occurrences, despite the issue
table's earlier count of two. All three are detected as `PERSON` before post-detection retyping.

Across the complete local ACI corpus, 376 `wan na` or `gon na` occurrences occur in 67 documents.
Presidio emits 43 overlapping `PERSON` hits across 24 documents. Replacing each split form with an
equal-length normalized view (`wan na` to `wanna ` and `gon na` to `gonna `) reduces those hits to zero
without changing character offsets.

The positive-code probe rejects the three measurement phrases and accepts representative patient
identifiers `E11.9`, `AB123456`, and `943 476 5919`. It deliberately rejects an alphabetic clinical
status such as `DNR code`: that is clinical content, not a patient identifier, and therefore must not
enter the direct-identifier `CODE` path.

## Design

### Attributable detection result

Add a diagnostic detection entry point alongside the compatibility `Detector.detect()` interface.
The diagnostic result contains accepted spans plus a decision record for every raw candidate. A raw
candidate records:

- original source offsets and surface;
- proposed runtime type;
- raw GLiNER label or Presidio entity type;
- Presidio recognizer name where applicable;
- unmodified source score;
- any offset-preserving input normalization that affected the candidate;
- lexical, clinical-contract, overlap-resolution, and negative-filter dispositions;
- the winning candidate for an overlap group, if one exists.

`Detector.detect()` remains a projection of this result to accepted spans, preserving existing callers.
The overlap resolver gains an internal traced form; the existing `_dedupe()` remains a compatibility
wrapper so existing unit tests and call sites keep their current interface.

Accepted `Span` values retain their winning raw label and recognizer as optional fields. Existing span
constructors remain source-compatible through defaults.

### Offset-preserving transcript view

The clinical profile runs Presidio on a normalized analysis view while GLiNER continues to receive the
original transcript. Only known split contraction spellings are compacted, and padding spaces preserve
the original string length. Presidio offsets are therefore valid against the original text without an
offset map; emitted surfaces always come from the original text.

The trace records a normalization event for every transformed region, including its original surface,
analysis-view surface, unchanged offsets, and rule. These events remain observable even when normalization
correctly prevents Presidio from emitting a candidate. Non-clinical profiles keep their current input
unchanged.

### Positive clinical `CODE` contract

For the pinned native clinical labels `medical code` and `healthcare number`, accept a GLiNER `CODE`
candidate only when its surface contains at least one ASCII digit. This is a positive identifier-shape
contract, not a list of known false-positive phrases. Presidio's typed identifier recognizers retain
their existing independent admission rules.

Candidates rejected by this contract remain in the diagnostic result with reason
`clinical_code_without_identifier_shape`. The model score is not recalibrated and the detector
threshold is unchanged.

### Name-label and runtime-type contract

An accepted GLiNER candidate whose raw native label is `name`, `first name`, or `last name` is a
`PERSON` by contract. The substitutor must not reinterpret such a winner through the lowercase
WordNet role heuristic. This fixes `andrew` at the source of the type corruption while preserving the
legacy heuristic for detector configurations that do not expose an explicit native name label.

The QA-v2 clinical arms builder validates the final substitution record before freezing it. Any
`demographic-other` occurrence fails the build and is preserved in the diagnostic rejection output;
it never becomes a controllable decision. This gate is configuration-specific and does not remove the
legacy runtime type globally.

### Frozen artifact provenance

Every substitution record carries the accepted span's per-occurrence detector provenance. The frozen
occurrence builder combines that record with the global detector pin instead of replacing it with a
model-and-score-only block. Rejected candidates and overlap losers are stored in a document-level
detector diagnostics section of the arms artifact so failures that do not produce a substitution record
remain auditable.

The persisted schema distinguishes:

- `accepted`: winner reached substitution;
- `rejected`: candidate failed a named contract or filter;
- `overlap_loser`: candidate lost to an attributable winner;
- `post_detection_rejected`: a downstream runtime contract refused to freeze the occurrence.

### Real-data pre-training gate

Add a durable clinical-detector gate under `scripts/`, not a one-time spike. It consumes the pinned
detector configuration and a named ACI slice, emits a JSON diagnostic artifact, and exits nonzero when
any preregistered error count exceeds its threshold.

The initial thresholds are zero for:

- accepted `PERSON` spans overlapping `wan na` or `gon na`;
- accepted native clinical `CODE` candidates without identifier-shaped evidence;
- explicit native name labels that finish with a runtime type other than `PERSON`;
- frozen QA-v2 clinical occurrences of type `demographic-other`.

The gate reports counts and examples for accepted, rejected, and overlap-loser families. Thresholds are
defined before the fixed end-to-end run and are not tuned from RL, attacker, or final utility results.

## Data flow

1. GLiNER reads the original transcript and emits labeled raw candidates.
2. The detector records transformed-region events, then Presidio reads the equal-length clinical analysis
   view and emits raw candidates whose offsets index the original transcript.
3. Source-specific positive contracts accept or reject raw candidates with named reasons.
4. Traced overlap resolution selects winners and records losing candidates.
5. The existing clinical negative filter accepts, drops, or retypes supported domain candidates and
   records its disposition.
6. The substitutor preserves explicit name-label winners as `PERSON` and attaches winner provenance to
   its local record.
7. The QA-v2 clinical builder rejects residual `demographic-other`, freezes accepted occurrences, and
   stores all detector diagnostics beside the document arms.

## Testing and validation

Implementation follows test-driven development:

- unit tests for equal-length normalization and unchanged offsets;
- unit tests for traced overlap winners and losers;
- unit tests for the positive native-code contract, including representative accepted identifiers and
  rejected measurements;
- a regression test proving explicit lowercase name labels bypass WordNet role retyping;
- artifact tests proving per-occurrence provenance survives freezing and rejected candidates remain
  attributable;
- a QA-v2 clinical gate test rejecting `demographic-other` without changing the legacy registry;
- CLI tests for threshold enforcement and diagnostic output families.

Real-data validation first runs `aci/D2N002` end to end, inspects the resulting artifact, and reports
counts and examples for accepted, rejected, overlap-loser, and post-detection-rejected records. The
smallest full-slice run that answers the aggregate question then evaluates the local 67-document ACI
corpus. Heavy inference uses the host `.venv`, one GPU process, unbuffered output, batched model calls,
and the project performance gate when its required prompt and reviewer backend are available.

Completion requires the exact issue surfaces to have these outcomes:

- all three `andrew` occurrences are accepted `PERSON` spans;
- split contractions are rejected before overlap resolution with attributable normalization evidence;
- the three measurement phrases are rejected as non-identifier-shaped native code candidates;
- no QA-v2 clinical frozen occurrence uses `demographic-other`;
- all required diagnostic output families are nonempty where the real case supplies examples.

## Alternatives rejected

String-specific deny lists and an `andrew` exception would be smaller but would encode observed surfaces
instead of the failure classes. A global threshold change or model-specific score calibration would move
the operating point, confound realized privacy, and still leave the post-detection type corruption and
missing provenance unresolved. Retraining is not justified until the deterministic contracts are
measured across the preregistered real-data gate.

## Risks and boundaries

- The digit-bearing `CODE` contract is deliberately scoped to native clinical GLiNER labels. It does not
  redefine Presidio identifiers or legacy detector schemas.
- Equal-length normalization may change nearby spaCy decisions, so the full ACI gate reports all
  Presidio candidate deltas, not only contraction overlaps.
- Preserving explicit native name labels favors privacy when a common noun is mislabeled as a name. Such
  false positives remain visible in diagnostics and require a separately measured person-name contract;
  they must not be silently converted into demographics.
- The current checkout lacks `scripts/harness/perf_gate.md` and the auto-review skill's required
  `codex:codex-rescue` backend. Heavy validation must report that limitation unless those resources become
  available before launch.
