import importlib
import json
import sys
from pathlib import Path

import pytest

from cloak.detect import DetectionResult, NormalizationEvent, Span


ROOT = Path(__file__).resolve().parents[3]


def _module():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        return importlib.import_module("clinical_detector_gate")
    finally:
        sys.path.pop(0)


gate = _module()


def document_fixture_with_each_family():
    text = "andrew says i wan na check heart rate"
    detection = DetectionResult(
        spans=[
            Span(0, 6, "andrew", "LOC", 0.95, "gliner", raw_label="name"),
            Span(
                14,
                20,
                "wan na",
                "PERSON",
                0.85,
                "presidio",
                raw_label="PERSON",
                recognizer="SpacyRecognizer",
            ),
            Span(
                27,
                37,
                "heart rate",
                "CODE",
                0.56,
                "gliner",
                raw_label="medical code",
            ),
        ],
        candidates=[
            {
                "start": 0,
                "end": 6,
                "surface": "andrew",
                "runtime_type": "LOC",
                "source": "gliner",
                "raw_label": "name",
                "status": "accepted",
                "reason": None,
            },
            {
                "start": 14,
                "end": 20,
                "surface": "wan na",
                "runtime_type": "PERSON",
                "source": "presidio",
                "raw_label": "PERSON",
                "status": "accepted",
                "reason": None,
            },
            {
                "start": 27,
                "end": 37,
                "surface": "heart rate",
                "runtime_type": "CODE",
                "source": "gliner",
                "raw_label": "medical code",
                "status": "accepted",
                "reason": None,
            },
            {
                "start": 0,
                "end": 6,
                "surface": "andrew",
                "runtime_type": "PERSON",
                "source": "presidio",
                "raw_label": "PERSON",
                "status": "overlap_loser",
                "reason": "cross_type_higher_effective_score",
            },
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
            "surface": "patient",
            "status": "post_detection_rejected",
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
                "start": 0,
                "end": 6,
                "surface": "Andrew",
                "runtime_type": "PERSON",
                "source": "gliner",
                "raw_label": "name",
                "status": "accepted",
                "reason": None,
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
        "accepted",
        "rejected",
        "overlap_loser",
        "normalizations",
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


def test_gate_detector_pin_uses_exact_native_schema_and_runtime_type_contract():
    assert gate.DETECTOR_PIN["label_schema"] == "knowledgator-native-clinical-v1"
    assert gate.DETECTOR_PIN["label_map"] == gate.QA_V2_CLINICAL_LABELS
    assert gate.DETECTOR_PIN["controlled_runtime_types"] == sorted(
        gate.QA_V2_CONTROLLED_TYPES
    )
    assert "controlled_types" not in gate.DETECTOR_PIN


@pytest.mark.parametrize("corpus", ["aci", "clinical", "mts"])
def test_gate_accepts_clinical_corpus_aliases(corpus, tmp_path):
    args = gate.parse_args(["--corpus", corpus, "--out", str(tmp_path / "gate.json")])
    assert args.corpus == corpus


def test_gate_rejects_nonclinical_corpus(tmp_path):
    with pytest.raises(SystemExit):
        gate.parse_args([
            "--corpus",
            "enron",
            "--out",
            str(tmp_path / "gate.json"),
        ])


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections
        self.calls = []

    def detect_many_with_diagnostics(self, texts):
        self.calls.append(list(texts))
        return list(self.detections)


def test_cli_writes_clean_and_failing_reports_before_returning_status(tmp_path, monkeypatch):
    rows = [{"id": "aci/clean", "text": "Andrew arrived"}]
    monkeypatch.setattr(gate, "load_task_docs", lambda corpus: list(rows))

    clean_detector = FakeDetector([clean_document_fixture()["detection"]])
    monkeypatch.setattr(gate, "Detector", lambda **_kwargs: clean_detector)
    clean_out = tmp_path / "clean.json"
    assert gate.main(["--out", str(clean_out)]) == 0
    assert json.loads(clean_out.read_text())["gate_pass"] is True
    assert clean_detector.calls == [["Andrew arrived"]]

    failing = document_fixture_with_each_family()["detection"]
    failing_detector = FakeDetector([failing])
    monkeypatch.setattr(gate, "Detector", lambda **_kwargs: failing_detector)
    failing_out = tmp_path / "failing.json"
    assert gate.main(["--out", str(failing_out)]) == 1
    assert json.loads(failing_out.read_text())["gate_pass"] is False
    assert failing_detector.calls == [["Andrew arrived"]]
