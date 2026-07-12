import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from build_mined_lattice_profiles import (
    DetectedSpan,
    build_mined_artifact,
    normalize_detector_label,
    _unique_spans,
)


def _row(levels):
    return {"aliases": [], "levels": levels, "source_ids": ["test:source"], "count": 1000.0}


def test_detector_labels_normalize_to_runtime_profile_types():
    assert normalize_detector_label("condition") == "health-condition"
    assert normalize_detector_label("injury") == "injury"
    assert normalize_detector_label("medical process") == "medical-procedure"
    assert normalize_detector_label("organization medical facility") == "organization-medical-facility"
    assert normalize_detector_label("drug") == "drug"


def test_unique_spans_resolves_surface_to_highest_scoring_label():
    # same surface under competing labels -> keeps only the most confident label, so it lands
    # in exactly one runtime profile instead of being double-counted across the type boundary.
    spans = [
        DetectedSpan("kidney stones", "injury", "d1", 0.60),
        DetectedSpan("kidney stones", "condition", "d2", 0.93),
        DetectedSpan("ankle sprain", "injury", "d3", 0.91),
        DetectedSpan("ankle sprain", "condition", "d4", 0.40),
    ]
    resolved = {s.surface: s.detector_label for s in _unique_spans(spans)}
    assert resolved == {"kidney stones": "condition", "ankle sprain": "injury"}


def test_common_profile_fuzzy_match_skips_detected_span():
    common = {
        "schema_version": 1,
        "profiles": {"drug": {"acetaminophen": _row(["medication"])}},
    }
    fine = {"schema_version": 1, "profiles": {"drug": {}}}

    artifact, stats = build_mined_artifact(
        [DetectedSpan("Acetaminophen tablets", "drug", "doc1", 0.9)],
        common,
        fine,
    )

    assert artifact["profiles"] == {}
    assert stats["skipped_common"] == 1


def test_fine_profile_match_copies_existing_row():
    common = {"schema_version": 1, "profiles": {"health-condition": {}}}
    fine = {
        "schema_version": 1,
        "profiles": {
            "health-condition": {
                "diabetes mellitus": {
                    "aliases": ["diabetes"],
                    "levels": ["endocrine condition"],
                    "source_ids": ["DOID:9351"],
                    "count": 1000.0,
                }
            }
        },
    }

    artifact, stats = build_mined_artifact(
        [DetectedSpan("Diabetes", "condition", "doc1", 0.95)],
        common,
        fine,
    )

    got = artifact["profiles"]["health-condition"]["diabetes mellitus"]
    assert got["levels"] == ["endocrine condition"]
    assert got["source_ids"] == ["DOID:9351"]
    assert stats["copied_fine"] == 1


def test_missing_span_appends_conservative_new_entry():
    common = {"schema_version": 1, "profiles": {"medical-procedure": {}}}
    fine = {"schema_version": 1, "profiles": {"medical-procedure": {}}}

    artifact, stats = build_mined_artifact(
        [DetectedSpan("nerve conduction study", "medical process", "doc42", 0.88)],
        common,
        fine,
    )

    got = artifact["profiles"]["medical-procedure"]["nerve conduction study"]
    assert got == {
        "aliases": [],
        "levels": ["medical procedure"],
        "source_ids": ["mined-clinical:doc42"],
        "count": 1.0,
    }
    assert stats["new_entries"] == 1
    assert json.dumps(artifact)


def test_generic_detected_type_level_terms_are_skipped():
    common = {"schema_version": 1, "profiles": {"drug": {}}}
    fine = {"schema_version": 1, "profiles": {"drug": {}}}

    artifact, stats = build_mined_artifact(
        [DetectedSpan("medication", "drug", "doc1", 0.9)],
        common,
        fine,
    )

    assert artifact["profiles"] == {}
    assert stats["generic_skipped"] == 1
