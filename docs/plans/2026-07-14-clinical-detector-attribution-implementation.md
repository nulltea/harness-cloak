---
type: plan
status: current
created: 2026-07-14
updated: 2026-07-14
tags: [detector, clinical, aci, provenance, implementation, qa-v2]
companion: 2026-07-14-clinical-detector-attribution-and-contract.md
---

# Clinical Detector Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pinned QA-v2 clinical detector attributable, remove the measured ACI misclassifications through offset-preserving and positive-type contracts, and enforce the corrected behavior with a durable real-data gate.

**Architecture:** `Detector` gains a compatibility-preserving diagnostic result that records normalization events and every candidate disposition while projecting accepted winners through `detect()`. The substitutor consumes winning-label provenance to protect explicit names and exposes rejected demographic fallbacks to the QA-v2 arms artifact. A durable CLI batches the ACI slice through the pinned detector and enforces preregistered zero-error thresholds.

**Tech Stack:** Python 3.12, dataclasses, GLiNER, Presidio/spaCy, pytest, JSON artifacts, host ROCm `.venv`.

## Global Constraints

- Keep `knowledgator/gliner-pii-large-v1.0`, threshold `0.35`, `knowledgator-native-clinical-v1`, GLiNER plus Presidio, and the clinical profile unchanged.
- Do not add or normalize any model-specific score calibration; accepted/rejected status is an outcome at the pinned operating point.
- Run GPU and pytest commands through `/home/timo/repos/agent-cloak/.venv/bin/python`; use `-u` for long or GPU commands and run only one GPU process at a time.
- Preserve original source offsets and surfaces. The Presidio analysis view must have exactly the same length as the original text.
- Scope native clinical code and `demographic-other` contracts to QA-v2 clinical behavior; preserve legacy fine-demographic consumers.
- Preserve accepted, rejected, overlap-loser, normalization, and post-detection-rejection evidence in durable JSON.
- Follow strict TDD: add one failing behavioral test, run it and confirm the expected failure, implement the minimum production change, then rerun the focused suite.
- Assign each implementation task to GPT-5.6 Terra with High reasoning effort, as requested; do not use fast/priority service mode.
- Do not modify or depend on uncommitted changes in `/home/timo/repos/agent-cloak`; this worktree is the source of truth.
- One-time diagnostic helpers belong under `scripts/spikes/`; the reusable pre-training gate belongs directly under `scripts/`.

---

### Task 1: Attributable detector pipeline and native clinical contracts

**Files:**
- Modify: `src/cloak/detect.py`
- Modify: `src/cloak/tests/test_detect_dedupe.py`
- Modify: `src/cloak/tests/test_detect_profiles.py`
- Create: `src/cloak/tests/test_detect_diagnostics.py`

**Interfaces:**
- Consumes: existing `Span`, `_dedupe()`, `_apply_negative_filter()`, `_chunks()`, `Detector.detect()`.
- Produces: `QA_V2_CLINICAL_LABELS`, `NormalizationEvent`, `DetectionResult`, `Detector.detect_with_diagnostics(text)`, `Detector.detect_many_with_diagnostics(texts)`, optional winning provenance on `Span`, and the unchanged `Detector.detect(text) -> list[Span]` projection.

- [ ] **Step 1: Write failing normalization and native-code contract tests**

Create `src/cloak/tests/test_detect_diagnostics.py` with real pipeline helpers but fake GLiNER and Presidio boundaries:

```python
from types import SimpleNamespace

from cloak.detect import (
    Detector,
    QA_V2_CLINICAL_LABELS,
    _clinical_presidio_view,
    _native_clinical_code_rejection,
)


def test_clinical_presidio_view_compacts_split_contractions_without_moving_offsets():
    text = "i wan na leave and i'm gon na call Andrew"
    view, events = _clinical_presidio_view(text)

    assert len(view) == len(text)
    assert view == "i wanna  leave and i'm gonna  call Andrew"
    assert [(event.start, event.end, event.surface, event.analysis_surface)
            for event in events] == [
        (2, 8, "wan na", "wanna "),
        (23, 29, "gon na", "gonna "),
    ]
    assert all(text[event.start:event.end] == event.surface for event in events)


def test_native_clinical_code_contract_requires_identifier_shape():
    assert _native_clinical_code_rejection("heart rate", "medical code") == (
        "clinical_code_without_identifier_shape"
    )
    assert _native_clinical_code_rejection("two out of six", "medical code") == (
        "clinical_code_without_identifier_shape"
    )
    assert _native_clinical_code_rejection("E11.9", "medical code") is None
    assert _native_clinical_code_rejection("AB123456", "healthcare number") is None
    assert _native_clinical_code_rejection("943 476 5919", "healthcare number") is None
    assert _native_clinical_code_rejection("heart rate", "condition") is None


def test_qa_v2_native_labels_map_explicit_names_to_person():
    assert QA_V2_CLINICAL_LABELS["name"] == "PERSON"
    assert QA_V2_CLINICAL_LABELS["first name"] == "PERSON"
    assert QA_V2_CLINICAL_LABELS["last name"] == "PERSON"
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/python -m pytest -q \
  src/cloak/tests/test_detect_diagnostics.py
```

Expected: collection fails because the new interfaces are not defined.

- [ ] **Step 3: Implement the normalization and positive-code primitives**

In `src/cloak/detect.py`, add the pinned label map already specified by the approved design and these focused primitives:

```python
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace

QA_V2_CLINICAL_LABELS = {
    "condition": "health-condition",
    "drug": "drug",
    "medical process": "medical-procedure",
    "location city": "LOC",
    "location country": "LOC",
    "location state": "LOC",
    "name": "PERSON",
    "first name": "PERSON",
    "last name": "PERSON",
    "age": "age",
    "gender": "gender",
    "marital status": "marital-status",
    "organization medical facility": "organization-medical-facility",
    "healthcare number": "CODE",
    "medical code": "CODE",
    "dose": "QUANTITY",
}

_SPLIT_CONTRACTION = re.compile(r"\b(?:wan|gon) na\b", re.IGNORECASE)
_NATIVE_CLINICAL_CODE_LABELS = frozenset({"medical code", "healthcare number"})


@dataclass(frozen=True)
class NormalizationEvent:
    start: int
    end: int
    surface: str
    analysis_surface: str
    rule: str = "clinical_split_contraction"


def _clinical_presidio_view(text: str) -> tuple[str, list[NormalizationEvent]]:
    events: list[NormalizationEvent] = []

    def compact(match: re.Match) -> str:
        surface = match.group(0)
        analysis_surface = surface.replace(" ", "") + " "
        events.append(NormalizationEvent(
            match.start(), match.end(), surface, analysis_surface
        ))
        return analysis_surface

    view = _SPLIT_CONTRACTION.sub(compact, text)
    assert len(view) == len(text)
    return view, events


def _native_clinical_code_rejection(surface: str, raw_label: str | None) -> str | None:
    if raw_label in _NATIVE_CLINICAL_CODE_LABELS and not re.search(r"[0-9]", surface):
        return "clinical_code_without_identifier_shape"
    return None
```

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run the Step 2 command. Expected: all tests in `test_detect_diagnostics.py` pass.

- [ ] **Step 5: Write failing traced-overlap and compatibility tests**

Extend `test_detect_diagnostics.py` and `test_detect_dedupe.py`:

```python
from cloak.detect import Span, _dedupe_with_diagnostics


def test_dedupe_diagnostics_record_same_type_and_cross_type_losers():
    spans = [
        Span(0, 6, "Andrew", "PERSON", 0.91, "gliner", raw_label="name"),
        Span(0, 6, "Andrew", "PERSON", 0.85, "presidio",
             raw_label="PERSON", recognizer="SpacyRecognizer"),
        Span(0, 12, "Andrew Smith", "MISC", 0.40, "gliner", raw_label="condition"),
    ]

    winners, diagnostics = _dedupe_with_diagnostics(spans)

    assert [(span.text, span.type, span.source) for span in winners] == [
        ("Andrew", "PERSON", "gliner")
    ]
    statuses = {(row["source"], row["runtime_type"]): row for row in diagnostics}
    assert statuses[("gliner", "PERSON")]["status"] == "accepted"
    assert statuses[("presidio", "PERSON")]["status"] == "overlap_loser"
    assert statuses[("presidio", "PERSON")]["winner"]["source"] == "gliner"
    assert statuses[("gliner", "MISC")]["status"] == "overlap_loser"


def test_legacy_dedupe_projects_only_winners():
    spans = [
        Span(0, 6, "Andrew", "PERSON", 0.91, "gliner"),
        Span(0, 6, "Andrew", "PERSON", 0.85, "presidio"),
    ]
    assert _dedupe(spans) == _dedupe_with_diagnostics(spans)[0]
```

- [ ] **Step 6: Run the traced-overlap tests and confirm RED**

Run:

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/python -m pytest -q \
  src/cloak/tests/test_detect_diagnostics.py \
  src/cloak/tests/test_detect_dedupe.py
```

Expected: failure because `Span` lacks provenance fields and `_dedupe_with_diagnostics` is undefined.

- [ ] **Step 7: Implement candidate provenance and traced overlap resolution**

Append optional fields after `Span.chain` so every existing positional constructor stays valid:

```python
@dataclass
class Span:
    start: int
    end: int
    text: str
    type: str
    score: float
    source: str
    chain: int = -1
    raw_label: str | None = None
    recognizer: str | None = None
    detector_provenance: dict | None = None
```

Implement `_dedupe_with_diagnostics(spans) -> tuple[list[Span], list[dict]]` by preserving the exact two-pass ordering and effective-score rule currently in `_dedupe()`. Each input candidate receives one row with `start`, `end`, `surface`, `runtime_type`, `score`, `source`, `raw_label`, `recognizer`, `status`, `reason`, and `winner`. Same-type losses use `same_type_wider_candidate`; cross-type losses use `cross_type_higher_effective_score`. Accepted winners receive `detector_provenance` containing their source fields and every overlapping candidate row. Replace `_dedupe()` with this compatibility wrapper:

```python
def _dedupe(spans: list[Span]) -> list[Span]:
    return _dedupe_with_diagnostics(spans)[0]
```

- [ ] **Step 8: Run overlap and existing detector unit suites**

Run:

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/python -m pytest -q \
  src/cloak/tests/test_detect_diagnostics.py \
  src/cloak/tests/test_detect_dedupe.py \
  src/cloak/tests/test_detect_chunks.py \
  src/cloak/tests/test_detect_noise_filter.py \
  src/cloak/tests/test_detect_profiles.py
```

Expected: all selected tests pass.

- [ ] **Step 9: Write failing diagnostic detector API tests**

Use `Detector.__new__(Detector)` and fake `gliner`/`presidio` objects so no model loads. Assert:

```python
class FakeGLiNER:
    config = SimpleNamespace(max_len=None)

    def __init__(self, entities):
        self.entities = entities
        self.calls = []

    def batch_predict_entities(self, texts, labels, threshold, batch_size):
        self.calls.append(list(texts))
        return [list(self.entities) for _text in texts]


class FakePresidio:
    def analyze(self, *, text, language):
        return []


def fake_qa_v2_detector(*, gliner_entities):
    detector = Detector.__new__(Detector)
    detector.threshold = 0.35
    detector.batch_size = 16
    detector.fine_dem = False
    detector.label2type = dict(QA_V2_CLINICAL_LABELS)
    detector.labels = list(detector.label2type)
    detector.gliner = FakeGLiNER(gliner_entities)
    detector.max_words = None
    detector.presidio = FakePresidio()
    detector.profile = PROFILES["clinical"]
    detector.stop_words = _stop_words(detector.profile)
    return detector


def test_detect_with_diagnostics_attributes_contract_rejections_and_normalization():
    detector = fake_qa_v2_detector(
        gliner_entities=[
            {"start": 0, "end": 6, "text": "andrew", "label": "name", "score": 0.95},
            {"start": 27, "end": 37, "text": "heart rate", "label": "medical code", "score": 0.56},
        ],
        presidio_entities=[],
    )
    result = detector.detect_with_diagnostics("andrew says i wan na check heart rate")

    assert [(span.text, span.type, span.raw_label) for span in result.spans] == [
        ("andrew", "PERSON", "name")
    ]
    assert any(row["reason"] == "clinical_code_without_identifier_shape"
               for row in result.candidates)
    assert [event.surface for event in result.normalizations] == ["wan na"]
    assert detector.detect("andrew says i wan na check heart rate") == result.spans
```

Add a second test whose fake GLiNER records all supplied chunks and assert `detect_many_with_diagnostics([text1, text2])` invokes one `batch_predict_entities` call for the flattened chunks while returning results in input order.

- [ ] **Step 10: Run diagnostic API tests and confirm RED**

Run the Step 8 command. Expected: the new tests fail because `DetectionResult` and diagnostic detector methods are absent.

- [ ] **Step 11: Implement the diagnostic single- and multi-document pipeline**

Add:

```python
@dataclass
class DetectionResult:
    spans: list[Span]
    candidates: list[dict]
    normalizations: list[NormalizationEvent]

    def as_dict(self) -> dict:
        return {
            "accepted": [asdict(span) for span in self.spans],
            "candidates": self.candidates,
            "normalizations": [asdict(event) for event in self.normalizations],
        }
```

Extend `Detector.__init__` with `label2type: Mapping[str, str] | None = None`; copy the mapping and set `self.labels` from it. Implement `detect_many_with_diagnostics` by flattening all `_chunks()` outputs into one GLiNER `batch_predict_entities` call, regrouping raw candidates by document, and running Presidio once per document on the clinical view. Populate `Span.raw_label` and `Span.recognizer`, record lexical and positive-contract rejections before `_dedupe_with_diagnostics`, then compare `_apply_negative_filter` input/output to record negative-filter rejection or retype dispositions. Implement:

```python
def detect_with_diagnostics(self, text: str) -> DetectionResult:
    return self.detect_many_with_diagnostics([text])[0]

def detect(self, text: str) -> list[Span]:
    return self.detect_with_diagnostics(text).spans
```

Only apply `_native_clinical_code_rejection` when `self.profile.name == "clinical"` and the candidate's raw label is one of the pinned native clinical code labels. Presidio continues to map its own identifier entities independently.

- [ ] **Step 12: Run Task 1 tests and the non-GPU detector suite**

Run:

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/python -m pytest -q \
  src/cloak/tests/test_detect_diagnostics.py \
  src/cloak/tests/test_detect_dedupe.py \
  src/cloak/tests/test_detect_chunks.py \
  src/cloak/tests/test_detect_noise_filter.py \
  src/cloak/tests/test_detect_profiles.py \
  src/cloak/tests/test_detect_coref.py \
  src/cloak/tests/test_detect_padding_guard.py \
  src/cloak/tests/test_detect_window_guard.py
```

Expected: all selected tests pass with no new warnings.

- [ ] **Step 13: Commit Task 1**

```bash
git add src/cloak/detect.py \
  src/cloak/tests/test_detect_diagnostics.py \
  src/cloak/tests/test_detect_dedupe.py \
  src/cloak/tests/test_detect_profiles.py
git commit -m "feat(detector): preserve attributable clinical decisions"
```

### Task 2: Name preservation, demographic rejection, and artifact provenance

**Files:**
- Modify: `src/cloak/substitute.py`
- Modify: `scripts/build_arms_artifact.py`
- Modify: `src/cloak/train/qa_builder.py`
- Modify: `src/cloak/tests/test_substitute_prepass.py`
- Modify: `src/cloak/tests/test_build_arms_artifact_cli.py`
- Modify: `src/cloak/tests/test_qa_builder_v2.py`

**Interfaces:**
- Consumes: Task 1 `QA_V2_CLINICAL_LABELS`, `DetectionResult`, and `Span.detector_provenance`.
- Produces: `prepare_spans_for_substitution(text, spans, reject_demographic_other=False)`, QA-v2 clinical detector preset, per-document `detector_diagnostics`, substitution-record provenance, and frozen-occurrence provenance merging.

- [ ] **Step 1: Write failing explicit-name and residual-demographic tests**

Add to `test_substitute_prepass.py`:

```python
from cloak.detect import Span
from cloak.substitute import prepare_spans_for_substitution


def test_explicit_native_name_label_bypasses_wordnet_role_retyping():
    span = Span(0, 6, "andrew", "PERSON", 0.95, "gliner", raw_label="name")
    prepared, rejected = prepare_spans_for_substitution(
        "andrew arrived", [span], reject_demographic_other=True
    )
    assert [(row.text, row.type) for row in prepared] == [("andrew", "PERSON")]
    assert rejected == []


def test_qa_v2_runtime_contract_rejects_residual_demographic_fallback():
    span = Span(0, 7, "patient", "PERSON", 0.85, "presidio",
                raw_label="PERSON", recognizer="SpacyRecognizer")
    prepared, rejected = prepare_spans_for_substitution(
        "patient arrived", [span], reject_demographic_other=True
    )
    assert prepared == []
    assert rejected[0]["status"] == "post_detection_rejected"
    assert rejected[0]["reason"] == "qa_v2_forbidden_demographic_other"
    assert rejected[0]["proposed_runtime_type"] == "demographic-other"
```

- [ ] **Step 2: Run the focused tests and confirm RED**

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/python -m pytest -q \
  src/cloak/tests/test_substitute_prepass.py
```

Expected: import or assertion failure because the preparation interface is absent.

- [ ] **Step 3: Extract preparation and preserve explicit name-label winners**

Implement in `src/cloak/substitute.py`:

```python
_EXPLICIT_NAME_LABELS = frozenset({"name", "first name", "last name"})


def prepare_spans_for_substitution(
    text: str,
    spans: list[Span],
    *,
    reject_demographic_other: bool = False,
) -> tuple[list[Span], list[dict]]:
    prepared: list[Span] = []
    rejected: list[dict] = []
    for source_span in spans:
        span = replace(source_span)
        if (span.type == "PERSON" and span.text[0].islower()
                and span.raw_label not in _EXPLICIT_NAME_LABELS
                and _is_role_phrase(span.text)):
            proposed = relabel_dem(span.text)
            if reject_demographic_other and proposed == "demographic-other":
                rejected.append({
                    "start": span.start,
                    "end": span.end,
                    "surface": span.text,
                    "source": span.source,
                    "raw_label": span.raw_label,
                    "recognizer": span.recognizer,
                    "score": span.score,
                    "status": "post_detection_rejected",
                    "reason": "qa_v2_forbidden_demographic_other",
                    "proposed_runtime_type": proposed,
                })
                continue
            span.type = proposed
        prepared.append(span)
    return prepared, rejected
```

Import `replace`, call this helper at the start of `substitute()` with the default contract, and remove the old in-place role loop. When building each substitution record, include `detector_provenance` from the winning span; if Task 1 did not populate it, synthesize source/raw-label/recognizer/score fields from the span.

- [ ] **Step 4: Run substitution tests and confirm GREEN**

Run the Step 2 command plus:

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/python -m pytest -q \
  src/cloak/tests/test_roundtrip.py \
  src/cloak/tests/test_fine_runtime_types.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Write failing QA-v2 builder and diagnostic artifact tests**

Extend `test_build_arms_artifact_cli.py` with tests that assert:

```python
def test_qa_v2_clinical_preset_is_pinned_and_override_free(monkeypatch):
    mod = _module()
    args = mod.parse_args(["--corpora", "clinical", "--detector-config", "qa-v2-clinical"])
    seen = {}
    monkeypatch.setattr(mod, "Detector", lambda **kwargs: seen.update(kwargs) or object())
    mod.make_detector(args, "clinical")
    assert seen == {
        "gliner_model": "knowledgator/gliner-pii-large-v1.0",
        "threshold": 0.35,
        "profile": "clinical",
        "label2type": mod.QA_V2_CLINICAL_LABELS,
    }


def test_qa_v2_document_entry_persists_detector_diagnostic_families():
    detection = DetectionResult(
        spans=[Span(0, 6, "andrew", "PERSON", 0.95, "gliner", raw_label="name")],
        candidates=[{
            "start": 0, "end": 6, "surface": "andrew", "runtime_type": "PERSON",
            "score": 0.95, "source": "gliner", "raw_label": "name",
            "recognizer": None, "status": "accepted", "reason": None, "winner": None,
        }],
        normalizations=[],
    )
    entry = mod.document_entry_from_detection(
        "andrew arrived", detection, tau=0.02, qa_v2=True
    )
    assert set(entry["detector_diagnostics"]) == {
        "accepted", "candidates", "normalizations", "post_detection_rejections"
    }
    assert all(row["type"] != "demographic-other" for row in entry["tau_walk"][1])
```

Extend `test_qa_builder_v2.py` so `frozen_occurrences_from_arms()` merges the global pin with a row's source-level provenance and keeps its overlap candidates rather than replacing them.

- [ ] **Step 6: Run artifact tests and confirm RED**

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/python -m pytest -q \
  src/cloak/tests/test_build_arms_artifact_cli.py \
  src/cloak/tests/test_qa_builder_v2.py::test_frozen_occurrences_from_arms_carries_detector_provenance
```

Expected: missing CLI preset/document helper assertions fail.

- [ ] **Step 7: Implement the pinned QA-v2 preset and diagnostic artifact assembly**

Port the approved preset into `scripts/build_arms_artifact.py` without reading the dirty main worktree:

```python
QA_V2_CLINICAL_MODEL = "knowledgator/gliner-pii-large-v1.0"
QA_V2_CLINICAL_THRESHOLD = 0.35
QA_V2_CONTROLLED_TYPES = frozenset({
    "LOC", "drug", "health-condition", "medical-procedure",
})
```

Add `--detector-config {deployment,qa-v2-clinical}` and reject combinations of the QA-v2 preset with `--detector-model`, `--threshold`, or `--fine-dem`. Preserve the model, threshold, label schema, controlled types, Presidio flag, and profiles in `_meta.detector`.

Add:

```python
def document_entry_from_detection(text, detection, *, tau: float, qa_v2: bool) -> dict:
    spans, post_rejections = prepare_spans_for_substitution(
        text, detection.spans, reject_demographic_other=qa_v2
    )
    arms = build_arms(text, spans, tau)
    entry = {arm: [doc_p, records] for arm, (doc_p, records) in arms.items()}
    if qa_v2 and any(
        row.get("type") == "demographic-other"
        for _doc_p, records in entry.values()
        if isinstance(records, list)
        for row in records
    ):
        raise ValueError("qa-v2-clinical cannot freeze demographic-other")
    diagnostic = detection.as_dict()
    entry["detector_diagnostics"] = {
        **diagnostic,
        "post_detection_rejections": post_rejections,
    }
    return entry
```

Adapt the comprehension to avoid treating `action_table` or diagnostics as arm tuples. In `main()`, call `detect_with_diagnostics()` for QA-v2 documents, build the entry through this helper, and compute `action_table` from the resulting `tau_walk` record. Deployment mode may use the same diagnostic entry point because `detect()` is only a projection.

Keep the existing `action_table(controlled_types=...)` behavior from the approved QA preset: QA-v2 action menus include only `QA_V2_CONTROLLED_TYPES` and skip controlled types with no non-placeholder levels.

- [ ] **Step 8: Merge global and per-occurrence provenance when freezing**

In `frozen_occurrences_from_arms()`, replace the global overwrite with:

```python
"detector_provenance": {
    **dict(detector_provenance or {}),
    **dict(row.get("detector_provenance") or {}),
    "score": row.get("score"),
}
```

Only emit the field when either the global pin or row provenance exists.

- [ ] **Step 9: Run Task 2 focused suites**

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/python -m pytest -q \
  src/cloak/tests/test_substitute_prepass.py \
  src/cloak/tests/test_build_arms_artifact_cli.py \
  src/cloak/tests/test_qa_builder_v2.py \
  src/cloak/tests/test_fine_runtime_types.py \
  src/cloak/tests/test_roundtrip.py
```

Expected: all selected tests pass.

- [ ] **Step 10: Commit Task 2**

```bash
git add src/cloak/substitute.py \
  scripts/build_arms_artifact.py \
  src/cloak/train/qa_builder.py \
  src/cloak/tests/test_substitute_prepass.py \
  src/cloak/tests/test_build_arms_artifact_cli.py \
  src/cloak/tests/test_qa_builder_v2.py
git commit -m "fix(clinical): enforce attributable runtime contracts"
```

### Task 3: Durable clinical gate and real-data artifact validation

**Files:**
- Create: `scripts/clinical_detector_gate.py`
- Create: `src/cloak/tests/test_clinical_detector_gate.py`
- Modify: `docs/issues/detector-misclassifications.md`
- Create: `results/clinical_detector_gate_aci_d2n002.json`
- Create: `results/clinical_detector_gate_aci.json`

**Interfaces:**
- Consumes: Task 1 `Detector.detect_many_with_diagnostics`, Task 2 `prepare_spans_for_substitution`, pinned QA-v2 constants, and diagnostic schema.
- Produces: reusable `evaluate_documents(rows, detector) -> dict`, CLI JSON artifacts, zero-threshold exit behavior, and measured issue resolution.

- [ ] **Step 1: Write failing pure gate-evaluation tests**

Create `test_clinical_detector_gate.py` with synthetic `DetectionResult` fixtures that do not load models:

```python
def document_fixture_with_each_family():
    text = "andrew says i wan na check heart rate"
    detection = DetectionResult(
        spans=[
            Span(0, 6, "andrew", "LOC", 0.95, "gliner", raw_label="name"),
            Span(14, 20, "wan na", "PERSON", 0.85, "presidio",
                 raw_label="PERSON", recognizer="SpacyRecognizer"),
            Span(27, 37, "heart rate", "CODE", 0.56, "gliner",
                 raw_label="medical code"),
        ],
        candidates=[
            {"start": 0, "end": 6, "surface": "andrew", "runtime_type": "LOC",
             "source": "gliner", "raw_label": "name", "status": "accepted", "reason": None},
            {"start": 14, "end": 20, "surface": "wan na", "runtime_type": "PERSON",
             "source": "presidio", "raw_label": "PERSON", "status": "accepted", "reason": None},
            {"start": 27, "end": 37, "surface": "heart rate", "runtime_type": "CODE",
             "source": "gliner", "raw_label": "medical code", "status": "accepted", "reason": None},
            {"start": 0, "end": 6, "surface": "andrew", "runtime_type": "PERSON",
             "source": "presidio", "raw_label": "PERSON", "status": "overlap_loser",
             "reason": "cross_type_higher_effective_score"},
        ],
        normalizations=[NormalizationEvent(14, 20, "wan na", "wanna ")],
    )
    demographic = Span(0, 7, "patient", "demographic-other", 0.85, "presidio")
    return {
        "doc_id": "aci/test",
        "text": text,
        "detection": detection,
        "prepared_spans": [*detection.spans, demographic],
        "post_detection_rejections": [{
            "surface": "patient", "status": "post_detection_rejected",
            "reason": "qa_v2_forbidden_demographic_other",
        }],
    }


def clean_document_fixture():
    return {
        "doc_id": "aci/clean",
        "text": "Andrew arrived",
        "detection": DetectionResult(
            spans=[Span(0, 6, "Andrew", "PERSON", 0.95, "gliner", raw_label="name")],
            candidates=[{
                "start": 0, "end": 6, "surface": "Andrew", "runtime_type": "PERSON",
                "source": "gliner", "raw_label": "name", "status": "accepted", "reason": None,
            }],
            normalizations=[],
        ),
        "prepared_spans": [
            Span(0, 6, "Andrew", "PERSON", 0.95, "gliner", raw_label="name")
        ],
        "post_detection_rejections": [],
    }


def test_gate_counts_required_error_classes_and_output_families():
    report = gate.evaluate_results([document_fixture_with_each_family()])
    assert report["counts"] == {
        "split_contraction_person": 1,
        "clinical_code_without_identifier_shape": 1,
        "explicit_name_not_person": 1,
        "frozen_demographic_other": 1,
    }
    assert set(report["families"]) == {
        "accepted", "rejected", "overlap_loser", "normalizations",
        "post_detection_rejected",
    }


def test_gate_passes_only_when_all_preregistered_counts_are_zero():
    report = gate.evaluate_results([clean_document_fixture()])
    assert report["thresholds"] == {
        "split_contraction_person": 0,
        "clinical_code_without_identifier_shape": 0,
        "explicit_name_not_person": 0,
        "frozen_demographic_other": 0,
    }
    assert report["gate_pass"] is True
```

Add a CLI test with a monkeypatched detector and temporary corpus/output. Assert exit zero for a clean report, nonzero for an error, and that the output file is written in both cases.

- [ ] **Step 2: Run gate tests and confirm RED**

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/python -m pytest -q \
  src/cloak/tests/test_clinical_detector_gate.py
```

Expected: module import fails because the durable gate does not exist.

- [ ] **Step 3: Implement pure evaluation and the batched CLI**

`scripts/clinical_detector_gate.py` must:

- accept `--corpus` with default `aci`, `--doc-id` for an optional exact document, `--limit`, and required `--out`;
- instantiate the exact QA-v2 model, threshold, profile, and label mapping;
- call `detect_many_with_diagnostics` once for the selected texts;
- prepare each accepted span set with `reject_demographic_other=True`;
- report the four preregistered counts and threshold zero for each;
- retain examples for every diagnostic family, without truncating the D2N002 artifact;
- include source document SHA-256 hashes and detector pin;
- write JSON before exiting `0` on pass or `1` on gate failure.

The pure `evaluate_results()` accepts already-constructed rows of `{doc_id, text, detection, prepared_spans, post_detection_rejections}` so unit tests need no model. Split-contraction overlap uses the normalization-event offsets. Explicit names are raw labels `name`, `first name`, or `last name`; compare their accepted/prepared runtime type to `PERSON`. `frozen_demographic_other` counts only prepared spans or substitution records, not correctly rejected diagnostic rows.

- [ ] **Step 4: Run gate tests and confirm GREEN**

Run the Step 2 command. Expected: all gate unit and CLI tests pass.

- [ ] **Step 5: Run the representative D2N002 real-data gate**

First confirm there is no live GPU Python process:

```bash
pgrep -af '/home/timo/repos/agent-cloak/.venv/bin/python|python.*(pytest|train|build_arms)' || true
```

Then run:

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/python -u \
  scripts/clinical_detector_gate.py \
  --corpus aci \
  --doc-id aci/D2N002 \
  --out results/clinical_detector_gate_aci_d2n002.json
```

Expected: exit `0`, all four gate counts zero, all three `andrew` occurrences accepted as `PERSON`, normalization evidence for the transcript's split contractions, rejected native-code rows for `heart rate`, `two out of six`, and `white blood cell count`, and at least one overlap loser from duplicate Andrew candidates.

- [ ] **Step 6: Inspect the actual D2N002 artifact**

Run:

```bash
jq '{gate_pass, counts, thresholds, family_counts: (.families | map_values(length)),
     andrew: [.families.accepted[] | select(.surface == "andrew") | {surface, runtime_type, source, raw_label}],
     code_rejections: [.families.rejected[] | select(.reason == "clinical_code_without_identifier_shape") | .surface]}' \
  results/clinical_detector_gate_aci_d2n002.json
```

Expected: `gate_pass: true`; three Andrew `PERSON` rows; the three named measurement surfaces in `code_rejections`; nonzero accepted, rejected, normalization, and overlap-loser family counts. Report `post_detection_rejected` honestly even when zero.

- [ ] **Step 7: Run the smallest full ACI aggregate gate**

The approved performance design uses one flattened GLiNER batch call across the selected documents. With the measured single-document load/inference under 15 seconds, the 67-document local slice is expected below ten minutes. Run:

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/python -u \
  scripts/clinical_detector_gate.py \
  --corpus aci \
  --out results/clinical_detector_gate_aci.json
```

Expected: a complete measured artifact. Do not claim a pass if any preregistered count is nonzero; report the measured failure and examples instead of changing thresholds or scores.

- [ ] **Step 8: Update the issue with measured resolution and limitations**

Set `updated: 2026-07-14`. Add a resolution section linking both committed results artifacts, recording the exact commands, counts, family counts, representative accepted/rejected examples, wall time, model pin, and any regressions or unresolved findings. Correct the observed Andrew count from two to three. Record that `scripts/harness/perf_gate.md` and the auto-review backend were unavailable if they remain absent; do not imply the standardized external perf review ran.

- [ ] **Step 9: Run focused and full verification**

Focused:

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/python -m pytest -q \
  src/cloak/tests/test_detect_diagnostics.py \
  src/cloak/tests/test_detect_dedupe.py \
  src/cloak/tests/test_detect_profiles.py \
  src/cloak/tests/test_substitute_prepass.py \
  src/cloak/tests/test_build_arms_artifact_cli.py \
  src/cloak/tests/test_qa_builder_v2.py \
  src/cloak/tests/test_clinical_detector_gate.py
```

Full:

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/python -m pytest -q
```

Compare full-suite failures to the recorded baseline of `693 passed, 14 failed, 1 skipped`. Fix regressions caused by this branch. Preserve and report unrelated baseline failures rather than editing unrelated code.

- [ ] **Step 10: Commit Task 3**

```bash
git add scripts/clinical_detector_gate.py \
  src/cloak/tests/test_clinical_detector_gate.py \
  docs/issues/detector-misclassifications.md \
  results/clinical_detector_gate_aci_d2n002.json \
  results/clinical_detector_gate_aci.json
git commit -m "test(clinical): gate attributable detector outcomes"
```

---

## Final review and completion

After all three task reviews pass, generate a whole-branch review package from merge base through `HEAD`, dispatch a fresh critical reviewer, and resolve every Critical or Important finding through one fix subagent and re-review. Then run the verification commands fresh, inspect both real artifacts again, and use the finishing-a-development-branch workflow. Do not merge, push, or open a pull request without a separate user request.
