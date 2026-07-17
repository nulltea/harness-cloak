import json
from pathlib import Path

from cloak.lattice_producer.repair_queue import build_repair_queue


def _profiles(path: Path) -> None:
    path.write_text(json.dumps({
        "profiles": {
            "health-condition": {
                "diabetes": {
                    "aliases": ["diabetes mellitus"],
                    "levels": ["glucose metabolism disease", "medical condition"],
                    "level_counts": {"glucose metabolism disease": 10.0, "medical condition": 1000.0},
                },
            },
            "medical-procedure": {},
        },
    }))


def test_repair_queue_routes_unresolved_type_entailing_span_to_nonrunnable_triage(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    _profiles(profiles)
    environment = {
        "events": [
            {
                "doc_id": "aci/D1", "code": "unprofiled_profile_backed_span",
                "entity_refs": {"surface": "physical therapy", "runtime_type": "medical-procedure"},
                "evidence": {"profile_match": {"outcome": "abstained"}},
            },
            {
                "doc_id": "aci/D1", "code": "unprofiled_self_type_diagnostic",
                "entity_refs": {"surface": "physical therapy", "runtime_type": "medical-procedure"},
                "evidence": {
                    "prep_filtered": False,
                    "self_type_score": 0.9,
                    "source_excerpt": "The patient was referred to physical therapy.",
                },
            },
        ],
    }

    queue = build_repair_queue([environment], [], profiles_path=profiles)

    assert queue["producer_items"] == []
    item = queue["triage_items"][0]
    assert item["surface"] == "physical therapy"
    assert item["eligible"] is False
    assert item["skip_reason"] == "requires_profile_identity_evidence"
    assert item["reprocess_reason"] == "unprofiled_type_entailed_unresolved"
    assert "medical-procedure" in item["reprocess_hint"]
    assert "[SPAN]physical therapy[/SPAN]" in item["marked_context_sentence"]
    assert queue["manual_review"] == []


def test_repair_queue_routes_reference_backed_missing_profile_to_producer(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    _profiles(profiles)
    environment = {
        "events": [
            {
                "doc_id": "aci/D1", "code": "unprofiled_profile_backed_span",
                "entity_refs": {"surface": "physical therapy", "runtime_type": "medical-procedure"},
                "evidence": {},
            },
            {
                "doc_id": "aci/D1", "code": "unprofiled_self_type_diagnostic",
                "entity_refs": {"surface": "physical therapy", "runtime_type": "medical-procedure"},
                "evidence": {"prep_filtered": False, "self_type_score": 0.9},
            },
        ],
    }

    queue = build_repair_queue(
        [environment], [], profiles_path=profiles,
        reference_lookup=lambda item: [{"source_family": "fixture", "level": item["surface"]}],
    )

    assert len(queue["producer_items"]) == 1
    item = queue["producer_items"][0]
    assert item["force_model_proposal"] is True
    assert item["reprocess_reason"] == "missing_profile_reference_backed"
    assert queue["triage_items"] == []


def test_repair_queue_keeps_unentailed_detector_residue_out_of_producer(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    _profiles(profiles)
    environment = {
        "events": [
            {
                "doc_id": "aci/D1", "code": "unprofiled_profile_backed_span",
                "entity_refs": {"surface": "intervention", "runtime_type": "medical-procedure"},
                "evidence": {},
            },
            {
                "doc_id": "aci/D1", "code": "unprofiled_self_type_diagnostic",
                "entity_refs": {"surface": "intervention", "runtime_type": "medical-procedure"},
                "evidence": {"prep_filtered": True, "self_type_score": None,
                             "source_excerpt": "The intervention was discussed."},
            },
        ],
    }

    queue = build_repair_queue([environment], [], profiles_path=profiles)

    assert queue["producer_items"] == []
    assert queue["manual_review"][0]["reason_codes"] == ["unprofiled_self_type_unentailed"]


def test_repair_queue_does_not_treat_low_score_diagnostic_as_type_entailment(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    _profiles(profiles)
    environment = {
        "events": [
            {
                "doc_id": "aci/D1", "code": "unprofiled_profile_backed_span",
                "entity_refs": {"surface": "right knee", "runtime_type": "health-condition"},
                "evidence": {},
            },
            {
                "doc_id": "aci/D1", "code": "unprofiled_self_type_diagnostic",
                "entity_refs": {"surface": "right knee", "runtime_type": "health-condition"},
                "evidence": {"prep_filtered": False, "self_type_score": 0.43},
            },
        ],
    }

    queue = build_repair_queue([environment], [], profiles_path=profiles)

    assert queue["producer_items"] == []
    assert queue["manual_review"][0]["reason_codes"] == ["unprofiled_self_type_unentailed"]


def test_repair_queue_dedupes_ladder_evidence_by_profile_entry(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    _profiles(profiles)
    environment = {
        "events": [{
            "doc_id": "aci/D1", "code": "controlled_ladder_issue",
            "entity_refs": {"surface": "diabetes", "runtime_type": "health-condition", "entry": "diabetes"},
            "evidence": {"issues": ["nonmonotone_level_counts"]},
        }],
    }
    qa = {
        "events": [{
            "doc_id": "aci/D2", "code": "lattice_level_suspect",
            "entity_refs": {},
            "evidence": {"surface": "diabetes", "runtime_type": "health-condition", "chain": ["bad", "good"]},
        }],
    }

    queue = build_repair_queue([environment], [qa], profiles_path=profiles)

    assert len(queue["producer_items"]) == 1
    item = queue["producer_items"][0]
    assert item["canonical_value"] == "diabetes"
    assert item["repair_kind"] == "ladder_repair"
    assert item["reprocess_reason"] == "lattice_structure_or_utility_failure"
    assert "controlled_ladder_issue" in item["reprocess_hint"]
    assert set(item["repair_reason_codes"]) == {"controlled_ladder_issue", "lattice_level_suspect"}
    assert {finding["code"] for finding in item["repair_findings"]} == {
        "controlled_ladder_issue", "lattice_level_suspect",
    }
    assert {"code": "controlled_ladder_issue", "issues": ["nonmonotone_level_counts"]} in item["repair_findings"]
    assert {"code": "lattice_level_suspect", "chain": ["bad", "good"]} in item["repair_findings"]
