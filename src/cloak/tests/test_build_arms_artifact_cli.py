import importlib
import sys
from pathlib import Path

import pytest

from cloak.detect import DetectionResult, Span


ROOT = Path(__file__).resolve().parents[3]


def _module():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        return importlib.import_module("build_arms_artifact")
    finally:
        sys.path.pop(0)


def test_build_arms_accepts_fine_detector_args(monkeypatch):
    mod = _module()
    args = mod.parse_args([
        "--detector-model", "data/models/pii_gliner_finedem/final",
        "--fine-dem",
        "--threshold", "0.22",
    ])

    seen = {}

    class FakeDetector:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(mod, "Detector", FakeDetector)
    mod.make_detector(args, "clinical")

    assert seen == {
        "gliner_model": "data/models/pii_gliner_finedem/final",
        "threshold": 0.22,
        "fine_dem": True,
        "profile": "clinical",
    }


def test_qa_v2_clinical_preset_is_pinned_and_override_free(monkeypatch):
    mod = _module()
    args = mod.parse_args([
        "--corpora",
        "clinical",
        "--detector-config",
        "qa-v2-clinical",
    ])
    seen = {}
    monkeypatch.setattr(mod, "Detector", lambda **kwargs: seen.update(kwargs) or object())

    mod.make_detector(args, "clinical")

    assert seen == {
        "gliner_model": "knowledgator/gliner-pii-large-v1.0",
        "threshold": 0.35,
        "profile": "clinical",
        "label2type": mod.QA_V2_CLINICAL_LABELS,
    }


@pytest.mark.parametrize(
    "override",
    [
        ["--detector-model", "custom/model"],
        ["--threshold", "0.2"],
        ["--fine-dem"],
    ],
)
def test_qa_v2_clinical_preset_rejects_detector_overrides(override):
    mod = _module()

    with pytest.raises(SystemExit):
        mod.parse_args(["--detector-config", "qa-v2-clinical", *override])


def test_qa_v2_document_entry_persists_detector_diagnostic_families():
    mod = _module()
    detection = DetectionResult(
        spans=[Span(0, 6, "andrew", "PERSON", 0.95, "gliner", raw_label="name")],
        candidates=[{
            "start": 0,
            "end": 6,
            "surface": "andrew",
            "runtime_type": "PERSON",
            "score": 0.95,
            "source": "gliner",
            "raw_label": "name",
            "recognizer": None,
            "status": "accepted",
            "reason": None,
            "winner": None,
        }],
        normalizations=[],
    )

    entry = mod.document_entry_from_detection(
        "andrew arrived", detection, tau=0.02, qa_v2=True
    )

    assert set(entry["detector_diagnostics"]) == {
        "accepted",
        "candidates",
        "normalizations",
        "post_detection_rejections",
    }
    assert all(row["type"] != "demographic-other" for row in entry["tau_walk"][1])


def test_qa_v2_action_table_limits_controlled_types_and_requires_levels(monkeypatch):
    mod = _module()
    monkeypatch.setattr(mod, "walk_risk", lambda *_args: 0.01)
    monkeypatch.setattr(mod, "fill_proximity", lambda *_args: 0.02)
    monkeypatch.setattr(mod, "aset_count", lambda *_args, **_kwargs: 10)
    records = [
        {
            "surface": "aspirin",
            "type": "drug",
            "start": 0,
            "end": 7,
            "lattice": ["medication", "<DRUG_1>"],
            "action": "generalize",
            "replacement": "medication",
        },
        {
            "surface": "patient",
            "type": "health-condition",
            "start": 8,
            "end": 15,
            "lattice": ["<HEALTH_CONDITION_1>"],
            "action": "placeholder",
            "replacement": "<HEALTH_CONDITION_1>",
        },
        {
            "surface": "42",
            "type": "age",
            "start": 16,
            "end": 18,
            "lattice": ["adult"],
            "action": "generalize",
            "replacement": "adult",
        },
    ]

    table = mod.action_table(
        "aspirin patient 42",
        records,
        controlled_types=mod.QA_V2_CONTROLLED_TYPES,
    )

    assert set(table) == {"aspirin"}
