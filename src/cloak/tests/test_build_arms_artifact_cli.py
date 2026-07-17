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


@pytest.mark.parametrize("corpus", ["clinical", "aci", "mts"])
def test_qa_v2_clinical_preset_accepts_only_clinical_aliases(corpus):
    mod = _module()

    args = mod.parse_args([
        "--corpora",
        corpus,
        "--detector-config",
        "qa-v2-clinical",
    ])

    assert mod.profile_for(corpus) == "clinical"
    assert args.corpora == corpus


def test_qa_v2_clinical_preset_normalizes_comma_separated_corpora_once():
    mod = _module()

    args = mod.parse_args([
        "--corpora",
        "clinical, aci",
        "--detector-config",
        "qa-v2-clinical",
    ])

    assert args.corpora == "clinical,aci"
    assert mod._detector_metadata(args, args.corpora.split(","))["profiles"] == {
        "clinical": "clinical",
        "aci": "clinical",
    }


def test_qa_v2_clinical_preset_rejects_nonclinical_corpus():
    mod = _module()

    with pytest.raises(SystemExit):
        mod.parse_args([
            "--corpora",
            "enron",
            "--detector-config",
            "qa-v2-clinical",
        ])


def test_qa_v2_make_detector_rejects_nonclinical_profile():
    mod = _module()
    args = mod.parse_args([
        "--corpora",
        "clinical",
        "--detector-config",
        "qa-v2-clinical",
    ])

    with pytest.raises(ValueError, match="requires a clinical profile"):
        mod.make_detector(args, "reddit")


def test_detector_metadata_uses_schema_id_map_and_runtime_type_contract():
    mod = _module()
    qa_args = mod.parse_args([
        "--corpora",
        "aci",
        "--detector-config",
        "qa-v2-clinical",
    ])
    qa_metadata = mod._detector_metadata(qa_args, ["aci"])

    assert qa_metadata["label_schema"] == "knowledgator-native-clinical-v1"
    assert qa_metadata["label_map"] == mod.QA_V2_CLINICAL_LABELS
    assert qa_metadata["controlled_runtime_types"] == sorted(mod.QA_V2_CONTROLLED_TYPES)
    assert "controlled_types" not in qa_metadata

    deployment_metadata = mod._detector_metadata(
        mod.parse_args(["--corpora", "enron"]), ["enron"]
    )
    assert deployment_metadata["label_schema"] == "tab-8"
    assert deployment_metadata["label_map"] == mod.GLINER_LABELS
    assert deployment_metadata["controlled_runtime_types"] is None
    assert "controlled_types" not in deployment_metadata


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
    assert "tau_walk" not in entry
    assert all(row["type"] != "demographic-other" for row in entry["v2_occurrences"])


def test_qa_v2_freeze_never_calls_legacy_tau_walk(monkeypatch):
    mod = _module()
    detection = DetectionResult(
        spans=[Span(0, 8, "aspirin", "drug", 0.95, "gliner", raw_label="drug")],
        candidates=[], normalizations=[],
    )
    monkeypatch.setattr(mod, "build_arms", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("legacy tau walk must not run for QA-v2")
    ))
    monkeypatch.setattr(mod, "freeze_policy_free_candidates", lambda _text, _spans: [{
        "surface": "aspirin", "type": "drug", "start": 0, "end": 8,
        "lattice": ["an analgesic"],
    }])

    entry = mod.document_entry_from_detection("aspirin", detection, tau=0.02, qa_v2=True)

    assert entry["v2_occurrences"][0]["lattice"] == ["an analgesic"]


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


def test_v2_action_table_has_only_legal_lattice_and_typed_placeholder(monkeypatch):
    mod = _module()
    monkeypatch.setattr(mod, "aset_count", lambda *_args, **_kwargs: 42.0)
    monkeypatch.setattr(mod, "walk_risk", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("V2 menu must not score legacy risk")
    ))
    table = mod.v2_action_table("aspirin", [{
        "surface": "aspirin", "type": "drug", "start": 0, "end": 7,
        "lattice": ["an analgesic", "<DRUG_1>"],
        "action": "placeholder", "replacement": "<DRUG_1>", "risk": 0.9,
        "exhausted": True,
    }], controlled_types=mod.QA_V2_CONTROLLED_TYPES)

    row = next(iter(table.values()))
    assert row["actions"] == [
        {"fill": "an analgesic", "mode": "level", "aset": 42.0, "legal": True},
        {"fill": None, "mode": "placeholder", "placeholder_type": "drug", "legal": True},
    ]


def test_public_detector_manifest_retains_pinned_qa_contract(monkeypatch):
    mod = _module()
    args = mod.parse_args([
        "--corpora", "clinical",
        "--detector-config", "qa-v2-clinical",
    ])
    seen = {}
    monkeypatch.setattr(mod, "Detector", lambda **kwargs: seen.update(kwargs) or object())

    mod.make_detector(args, "clinical")
    manifest = mod.detector_manifest(args, ["clinical"])

    assert seen["gliner_model"] == "knowledgator/gliner-pii-large-v1.0"
    assert seen["threshold"] == 0.35
    assert seen["profile"] == "clinical"
    assert seen["label2type"]["condition"] == "health-condition"
    assert seen["label2type"]["drug"] == "drug"
    assert seen["label2type"]["medical process"] == "medical-procedure"
    assert "DEM" not in seen["label2type"].values()
    assert "MISC" not in seen["label2type"].values()
    assert manifest["label_schema"] == "knowledgator-native-clinical-v1"
    assert manifest["label_map"] == mod.QA_V2_CLINICAL_LABELS
    assert manifest["controlled_runtime_types"] == [
        "LOC", "drug", "health-condition", "medical-procedure",
    ]


def test_action_table_accepts_set_for_qa_v2_lattice_roles(monkeypatch):
    mod = _module()
    monkeypatch.setattr(mod, "walk_risk", lambda *args, **kwargs: 0.1)
    monkeypatch.setattr(mod, "fill_proximity", lambda *args, **kwargs: 0.2)
    monkeypatch.setattr(mod, "aset_count", lambda *args, **kwargs: 10.0)
    records = [
        {
            "surface": "arthritis", "type": "health-condition", "start": 0, "end": 9,
            "action": "generalize", "replacement": "a disease",
            "lattice": ["a disease"],
        },
        {
            "surface": "62-year-old", "type": "age", "start": 14, "end": 25,
            "action": "generalize", "replacement": "an age",
            "lattice": ["an age"],
        },
    ]

    table = mod.action_table(
        "arthritis and 62-year-old",
        records,
        controlled_types={"health-condition", "drug", "medical-procedure", "LOC"},
    )

    assert list(table) == ["arthritis"]
