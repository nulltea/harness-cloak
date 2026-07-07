import json
from pathlib import Path

import pytest

from cloak.lattice_producer.coverage import (
    CategoryOutcome,
    build_category_coverage,
    registry_entry_for_label,
)
from cloak.lattice_producer.propose import assemble_context_packet, ensure_local_base_url
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


def test_proposal_base_url_must_be_local() -> None:
    ensure_local_base_url("http://localhost:8060/v1")
    with pytest.raises(ValueError, match="local"):
        ensure_local_base_url("https://api.openai.com/v1")
