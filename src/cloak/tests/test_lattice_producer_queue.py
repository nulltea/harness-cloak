import json
from pathlib import Path

import pytest

from cloak.lattice_producer.coverage import (
    CategoryOutcome,
    build_category_coverage,
    registry_entry_for_label,
    registry_outcome_for_runtime_type,
)
from cloak.lattice_producer.propose import (
    QWEN36_THINKING_BUDGET_TOKENS,
    assemble_context_packet,
    ensure_local_base_url,
    extract_candidate_levels,
    propose_with_llama_swap,
)
from cloak.lattice_producer.queue import _queue_from_profile_categories, build_or_load_queue


def _write_profiles(path: Path, profiles: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created": "2026-07-07",
                "sources": {},
                "profiles": profiles,
            }
        )
    )


def test_category_registry_maps_direct_and_v7_fine_labels() -> None:
    assert registry_entry_for_label("email address").outcome == CategoryOutcome.FORCED_PLACEHOLDER
    assert registry_entry_for_label("email address").runtime_type == "CODE"
    assert registry_entry_for_label("profession").outcome == CategoryOutcome.RUNTIME_LATTICE
    assert registry_entry_for_label("profession").runtime_type == "profession"
    assert registry_entry_for_label("organization").runtime_type == "ORG"
    assert registry_entry_for_label("organization medical facility").runtime_type == "organization-medical-facility"
    assert registry_entry_for_label("medical process").outcome == CategoryOutcome.NEEDS_PROFILE


def test_registry_outcome_for_runtime_type_matches_registry_semantics() -> None:
    assert registry_outcome_for_runtime_type("drug") == CategoryOutcome.RUNTIME_LATTICE
    assert registry_outcome_for_runtime_type("medical-procedure") == CategoryOutcome.RUNTIME_LATTICE
    assert registry_outcome_for_runtime_type("MISC") == CategoryOutcome.RUNTIME_LATTICE
    assert registry_outcome_for_runtime_type("health-condition") == CategoryOutcome.RUNTIME_LATTICE
    assert registry_outcome_for_runtime_type("organization-medical-facility") == CategoryOutcome.RUNTIME_LATTICE
    assert registry_outcome_for_runtime_type("nonexistent-type") == CategoryOutcome.NEEDS_PROFILE


def test_coverage_emits_generated_universe_gap_for_unmined_runtime_lattice(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    _write_profiles(profiles, {"profession": {}})

    gaps = build_category_coverage(profiles, category="profession")

    assert gaps == [
        {
            "detector_label_family": "profession",
            "runtime_type": "profession",
            "outcome": "runtime_lattice",
            "profile_row_count": 0,
            "non_placeholder_level_count": 0,
            "dataset_backed_source_exists": False,
            "generated_universe_required": True,
        }
    ]


def test_coverage_accepts_multiple_category_filters(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    _write_profiles(profiles, {"profession": {}, "health-condition": {}})

    gaps = build_category_coverage(profiles, categories=["profession", "health-condition"])

    assert [row["runtime_type"] for row in gaps] == [
        "health-condition",
        "health-condition",
        "profession",
    ]


def test_category_seeded_items_do_not_force_model_proposal(tmp_path):
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({
        "profiles": {"drug": {"aspirin": {"aliases": ["asa"], "levels": [], "level_counts": {}}}}
    }))
    items = _queue_from_profile_categories(profiles, ["drug"])
    assert items, "expected one drug item"
    assert all("force_model_proposal" not in item for item in items)


def test_queue_accepts_multiple_category_filters_for_existing_profile_entries(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    run_dir = tmp_path / "run"
    _write_profiles(
        profiles,
        {
            "drug": {
                "aspirin": {"aliases": ["acetylsalicylic acid"], "levels": ["medication"], "source_ids": [], "count": 1},
            },
            "health-condition": {
                "asthma": {"aliases": [], "levels": ["respiratory condition"], "source_ids": [], "count": 1},
            },
            "medical-procedure": {
                "colonoscopy": {"aliases": [], "levels": ["endoscopy"], "source_ids": [], "count": 1},
            },
            "profession": {
                "doctor": {"aliases": [], "levels": ["healthcare worker"], "source_ids": [], "count": 1},
            },
        },
    )

    items = build_or_load_queue(run_dir, profiles, categories=["drug", "health-condition", "medical-procedure"])

    assert [(item["runtime_type"], item["surface"]) for item in items] == [
        ("drug", "aspirin"),
        ("health-condition", "asthma"),
        ("medical-procedure", "colonoscopy"),
    ]
    assert items[0]["aliases"] == ["acetylsalicylic acid"]
    assert all(item["task_kind"] == "level-proposal" for item in items)
    assert all("force_model_proposal" not in item for item in items)


def test_queue_skips_forced_placeholder_and_rejects_dem(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    run_dir = tmp_path / "run"
    explicit_queue = tmp_path / "queue.jsonl"
    _write_profiles(profiles, {})
    explicit_queue.write_text(
        "\n".join(
            [
                json.dumps({"item_id": "forced", "detector_label_family": "email address"}),
                json.dumps({"item_id": "bad", "runtime_type": "DEM", "surface": "Polish"}),
            ]
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="DEM"):
        build_or_load_queue(run_dir, profiles, explicit_queue=explicit_queue)

    explicit_queue.write_text(json.dumps({"item_id": "forced", "detector_label_family": "email address"}) + "\n")
    items = build_or_load_queue(run_dir, profiles, explicit_queue=explicit_queue)
    assert items[0]["eligible"] is False
    assert items[0]["skip_reason"] == "forced_placeholder"


def test_queue_treats_domain_profile_types_as_lattice_runtime_types(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    run_dir = tmp_path / "run"
    explicit_queue = tmp_path / "queue.jsonl"
    _write_profiles(profiles, {})
    explicit_queue.write_text(
        "\n".join(
            [
                json.dumps({"item_id": "drug:aspirin", "runtime_type": "drug", "surface": "aspirin"}),
                json.dumps({"item_id": "medical-procedure:xray", "runtime_type": "medical-procedure", "surface": "xray"}),
                json.dumps({"item_id": "health-condition:asthma", "runtime_type": "health-condition", "surface": "asthma"}),
                json.dumps({"item_id": "misc:thing", "runtime_type": "MISC", "surface": "thing"}),
            ]
        )
        + "\n"
    )

    items = build_or_load_queue(run_dir, profiles, explicit_queue=explicit_queue)

    assert [item["eligible"] for item in items] == [True, True, True, True]


def test_queue_from_profile_categories_makes_drug_producer_eligible(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    run_dir = tmp_path / "run"
    _write_profiles(
        profiles,
        {
            "drug": {"aspirin": {"aliases": [], "levels": ["medication"], "source_ids": [], "count": 1}},
            "medical-procedure": {"colonoscopy": {"aliases": [], "levels": ["endoscopy"], "source_ids": [], "count": 1}},
        },
    )

    items = build_or_load_queue(run_dir, profiles, categories=["drug", "medical-procedure"])

    by_type = {item["runtime_type"]: item for item in items}
    assert by_type["drug"]["eligible"] is True
    assert by_type["medical-procedure"]["eligible"] is True


def test_context_packet_uses_bounded_artifact_slices_not_logs(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "EXPERIMENT_LOG.md").write_text("raw model output that must not enter prompts")
    _write_profiles(
        profiles,
        {
            "profession": {
                "journalist": {"aliases": ["reporter"], "levels": ["media worker"], "source_ids": [], "count": 1},
                "cardiologist": {"aliases": [], "levels": ["medical specialist"], "source_ids": [], "count": 1},
            }
        },
    )
    item = {
        "item_id": "profession:cardiologist",
        "task_kind": "level-proposal",
        "runtime_type": "profession",
        "detector_label_family": "profession",
        "surface": "cardiologist",
        "marked_context_sentence": "She is a [SPAN]cardiologist[/SPAN] in Oslo.",
    }

    packet = assemble_context_packet(
        item,
        profiles_path=profiles,
        run_dir=run_dir,
        prompt_version="lattice-producer-v1",
        max_context_rows=1,
    )

    serialized = json.dumps(packet, sort_keys=True)
    assert packet["context_packet_hash"]
    assert len(packet["nearby_profile_rows"]) == 1
    assert "raw model output" not in serialized
    assert "journalist" not in serialized
    assert "model-provided numeric counts are ignored" not in serialized
    assert "Do not provide certifying counts" not in serialized
    assert "aliases" in packet["required_proposal_fields"]
    assert "proposed_count" in packet["required_level_fields"]


def test_context_hash_changes_when_relevant_artifact_slice_changes(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    item = {
        "item_id": "profession:cardiologist",
        "task_kind": "level-proposal",
        "runtime_type": "profession",
        "detector_label_family": "profession",
        "surface": "cardiologist",
    }
    _write_profiles(
        profiles,
        {"profession": {"cardiologist": {"aliases": [], "levels": ["medical specialist"], "source_ids": [], "count": 1}}},
    )
    first = assemble_context_packet(
        item,
        profiles_path=profiles,
        run_dir=run_dir,
        prompt_version="lattice-producer-v1",
        max_context_rows=2,
    )
    _write_profiles(
        profiles,
        {"profession": {"cardiologist": {"aliases": [], "levels": ["healthcare worker"], "source_ids": [], "count": 1}}},
    )
    second = assemble_context_packet(
        item,
        profiles_path=profiles,
        run_dir=run_dir,
        prompt_version="lattice-producer-v1",
        max_context_rows=2,
    )

    assert first["context_packet_hash"] != second["context_packet_hash"]


def test_context_packet_includes_rejection_feedback_for_retry(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_profiles(profiles, {"profession": {}})
    item = {
        "item_id": "profession:privacy engineer",
        "runtime_type": "profession",
        "detector_label_family": "profession",
        "surface": "privacy engineer",
        "retry_attempt": 1,
        "rejection_feedback": [
            {
                "reason": "missing_aliases",
                "level": "architecture and engineering occupation",
                "guidance": "Retry must include aliases and profession-specific evidence.",
            }
        ],
    }

    packet = assemble_context_packet(
        item,
        profiles_path=profiles,
        run_dir=run_dir,
        prompt_version="lattice-producer-v1",
        max_context_rows=1,
    )

    assert packet["retry_attempt"] == 1
    assert packet["previous_rejection_feedback"][0]["reason"] == "missing_aliases"
    assert "architecture and engineering occupation" in json.dumps(packet)


def test_context_packet_carries_repair_queue_reason_and_hint(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_profiles(profiles, {"medical-procedure": {}})
    packet = assemble_context_packet(
        {
            "runtime_type": "medical-procedure",
            "surface": "physical therapy",
            "reprocess_reason": "missing_profile_type_entailed",
            "reprocess_hint": "The marked context entails a medical procedure; establish a truthful profile.",
        },
        profiles_path=profiles,
        run_dir=run_dir,
        prompt_version="lattice-producer-v1",
        max_context_rows=1,
    )

    assert packet["reprocess_reason"] == "missing_profile_type_entailed"
    assert packet["reprocess_hint"].startswith("The marked context")
    assert "stable real-world referent" in packet["reprocess_instruction"]
    assert "current_profile" not in packet  # no existing row for a missing profile


def test_context_packet_ladder_repair_carries_current_profile_and_findings(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_profiles(
        profiles,
        {
            "health-condition": {
                "diabetes": {
                    "aliases": ["diabetes mellitus"],
                    "levels": ["glucose metabolism disease", "medical condition"],
                    "level_counts": {"glucose metabolism disease": 10.0, "medical condition": 1000.0},
                    "count": 1.0,
                    "level_grounding": {"glucose metabolism disease": {"status": "model-proposed"}},
                },
            },
        },
    )
    findings = [{
        "code": "lattice_level_suspect",
        "unreadable_level": "glucose metabolism disease",
        "readable_coarser_level": "medical condition",
        "chain": ["glucose metabolism disease", "medical condition"],
    }]
    packet = assemble_context_packet(
        {
            "runtime_type": "health-condition",
            "surface": "diabetes",
            "canonical_value": "diabetes",
            "reprocess_reason": "lattice_structure_or_utility_failure",
            "reprocess_hint": "Audit evidence: lattice_level_suspect.",
            "repair_findings": findings,
        },
        profiles_path=profiles,
        run_dir=run_dir,
        prompt_version="lattice-producer-v1",
        max_context_rows=1,
    )

    assert packet["repair_findings"] == findings
    assert packet["current_profile"]["levels"] == ["glucose metabolism disease", "medical condition"]
    assert packet["current_profile"]["level_counts"]["medical condition"] == 1000.0
    assert packet["current_profile"]["level_grounding_status"] == {"glucose metabolism disease": "model-proposed"}
    assert "do not copy current_profile" in packet["reprocess_instruction"]
    # the packet hash covers the repair context, so repair prompts stay cache-deterministic
    assert packet["context_packet_hash"]


def test_proposal_base_url_must_be_local() -> None:
    ensure_local_base_url("http://localhost:8060/v1")
    with pytest.raises(ValueError, match="local"):
        ensure_local_base_url("https://api.openai.com/v1")


def test_proposal_call_omits_thinking_budget_by_default(monkeypatch, tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_profiles(profiles, {"profession": {}})
    seen = {}

    class FakeCompletions:
        def create(self, **kwargs):
            seen.update(kwargs)

            class Message:
                content = '{"candidates":[]}'

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        def __init__(self, **kwargs):
            seen["client"] = kwargs
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("cloak.lattice_producer.propose.OpenAI", FakeClient)
    monkeypatch.setenv("CLOAK_LLM_CACHE", str(tmp_path / "cache"))

    propose_with_llama_swap(
        {
            "item_id": "profession:privacy engineer",
            "runtime_type": "profession",
            "surface": "privacy engineer",
        },
        profiles_path=profiles,
        run_dir=run_dir,
        prompt_version="lattice-producer-v1",
        max_context_rows=1,
        base_url="http://localhost:8060/v1",
        model="Qwen3.6-35B-A3B",
    )

    assert "extra_body" not in seen


def test_proposal_call_accepts_custom_thinking_budget(monkeypatch, tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_profiles(profiles, {"drug": {}})
    seen = {}

    class FakeCompletions:
        def create(self, **kwargs):
            seen.update(kwargs)

            class Message:
                content = '{"candidates":[]}'

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("cloak.lattice_producer.propose.OpenAI", FakeClient)
    monkeypatch.setenv("CLOAK_LLM_CACHE", str(tmp_path / "cache"))

    propose_with_llama_swap(
        {
            "item_id": "drug:aspirin",
            "runtime_type": "drug",
            "surface": "aspirin",
        },
        profiles_path=profiles,
        run_dir=run_dir,
        prompt_version="lattice-producer-v1",
        max_context_rows=1,
        base_url="http://localhost:8060/v1",
        model="Qwen3.6-35B-A3B",
        thinking_budget_tokens=1024,
    )

    assert seen["extra_body"] == {"thinking_budget_tokens": 1024}


def test_proposal_invalid_json_after_escalation_returns_parse_error(monkeypatch, tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_profiles(profiles, {"drug": {}})

    class FakeCompletions:
        calls = 0

        def create(self, **kwargs):
            type(self).calls += 1

            class Message:
                content = "not json"

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("cloak.lattice_producer.propose.OpenAI", FakeClient)
    monkeypatch.setenv("CLOAK_LLM_CACHE", str(tmp_path / "cache"))

    payload = propose_with_llama_swap(
        {
            "item_id": "drug:aspirin",
            "runtime_type": "drug",
            "surface": "aspirin",
        },
        profiles_path=profiles,
        run_dir=run_dir,
        prompt_version="lattice-producer-v1",
        max_context_rows=1,
        base_url="http://localhost:8060/v1",
        model="Qwen3.6-35B-A3B",
        escalation_model="Qwen3.6-35B-A3B",
        thinking_budget_tokens=1024,
    )

    assert FakeCompletions.calls == 2
    assert payload["candidates"] == []
    assert payload["parse_error"] == "invalid_json"
    assert payload["raw"] == "not json"


def test_extract_candidate_levels_accepts_model_schema_variants() -> None:
    assert extract_candidate_levels({"candidate_levels": ["professional worker", "technical worker"]}) == [
        {"level": "professional worker", "selector": "candidate_levels", "source_family": "model-proposed"},
        {"level": "technical worker", "selector": "candidate_levels", "source_family": "model-proposed"},
    ]
    assert extract_candidate_levels({"proposed_levels": [{"level": "science occupation"}]}) == [
        {
            "level": "science occupation",
            "selector": "proposed_levels",
            "source_family": "model-proposed",
        }
    ]


def test_extract_candidate_levels_preserves_aliases_counts_and_evidence() -> None:
    candidates = extract_candidate_levels(
        {
            "aliases": ["data protection engineer"],
            "candidates": [
                {
                    "level": "privacy and security software professional",
                    "proposed_count": 180,
                    "count_evidence": "Estimated candidate set includes privacy engineering, security engineering, GRC engineering, and software privacy roles.",
                    "rationale": "Preserves the privacy/security/software context without naming the exact profession.",
                    "selector": "model-domain-cluster:privacy-security-software",
                }
            ],
        }
    )

    assert candidates == [
        {
            "level": "privacy and security software professional",
            "proposed_count": 180,
            "count_evidence": "Estimated candidate set includes privacy engineering, security engineering, GRC engineering, and software privacy roles.",
            "rationale": "Preserves the privacy/security/software context without naming the exact profession.",
            "selector": "model-domain-cluster:privacy-security-software",
            "aliases": ["data protection engineer"],
            "source_family": "model-proposed",
        }
    ]
