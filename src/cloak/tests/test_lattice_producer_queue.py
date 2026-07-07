import json
from pathlib import Path

import pytest

from cloak.lattice_producer.coverage import (
    CategoryOutcome,
    build_category_coverage,
    registry_entry_for_label,
)
from cloak.lattice_producer.propose import (
    QWEN36_THINKING_BUDGET_TOKENS,
    assemble_context_packet,
    ensure_local_base_url,
    extract_candidate_levels,
    propose_with_llama_swap,
)
from cloak.lattice_producer.queue import build_or_load_queue


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
            ]
        )
        + "\n"
    )

    items = build_or_load_queue(run_dir, profiles, explicit_queue=explicit_queue)

    assert [item["eligible"] for item in items] == [True, True, True]


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


def test_proposal_base_url_must_be_local() -> None:
    ensure_local_base_url("http://localhost:8060/v1")
    with pytest.raises(ValueError, match="local"):
        ensure_local_base_url("https://api.openai.com/v1")


def test_proposal_call_sets_qwen_thinking_budget(monkeypatch, tmp_path: Path) -> None:
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
    monkeypatch.setenv("INFERDPT_LLM_CACHE", str(tmp_path / "cache"))

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

    assert seen["extra_body"] == {"thinking_budget_tokens": QWEN36_THINKING_BUDGET_TOKENS}


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
