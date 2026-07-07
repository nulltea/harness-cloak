import json
from pathlib import Path

from cloak.lattice_producer.merge import ensure_proposed_artifact, persist_proposed_artifact, validate_proposed_artifact


def _base_profile(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created": "2026-07-07",
                "sources": {},
                "profiles": {
                    "profession": {
                        "journalist": {
                            "aliases": ["reporter"],
                            "levels": ["media worker"],
                            "source_ids": ["fixture:journalist"],
                            "count": 1000,
                        }
                    }
                },
            }
        )
    )


def test_persist_writes_separate_proposal_incrementally_and_leaves_canonical_untouched(tmp_path: Path) -> None:
    canonical = tmp_path / "fine_lattice_profiles.json"
    proposed = tmp_path / "fine_lattice_profiles.proposed.json"
    _base_profile(canonical)
    before = canonical.read_text()
    item = {
        "item_id": "profession:cardiologist",
        "runtime_type": "profession",
        "surface": "cardiologist",
        "entry_origin": "generated-universe",
        "aliases": ["heart doctor"],
    }
    accepted = [
        {
            "level": "medical specialist",
            "level_count": 42.0,
            "level_grounding": {"status": "proposal-universe", "selector": "generated_group:medical-specialist"},
        }
    ]

    persist_proposed_artifact(
        canonical,
        proposed,
        run_id="run-1",
        item=item,
        accepted=accepted,
    )
    persist_proposed_artifact(
        canonical,
        proposed,
        run_id="run-1",
        item=item,
        accepted=accepted,
    )

    assert canonical.read_text() == before
    artifact = json.loads(proposed.read_text())
    assert "journalist" not in artifact["profiles"].get("profession", {})
    row = artifact["profiles"]["profession"]["cardiologist"]
    assert artifact["artifact_role"] == "proposal"
    assert row["entry_origin"] == "generated-universe"
    assert row["levels"] == ["medical specialist"]
    assert row["level_counts"]["medical specialist"] == 42.0
    assert row["level_groundings"]["medical specialist"]["status"] == "proposal-universe"
    assert row["source_ids"] == ["producer:run-1:profession:cardiologist"]


def test_persist_includes_model_proposed_aliases_from_accepted_records(tmp_path: Path) -> None:
    canonical = tmp_path / "fine_lattice_profiles.json"
    proposed = tmp_path / "fine_lattice_profiles.proposed.json"
    _base_profile(canonical)
    item = {
        "item_id": "profession:privacy engineer",
        "runtime_type": "profession",
        "surface": "privacy engineer",
        "aliases": [],
    }
    accepted = [
        {
            "level": "privacy and security software professional",
            "aliases": ["data protection engineer", "privacy software engineer"],
            "level_count": 180.0,
            "level_grounding": {
                "status": "model-proposed",
                "source_family": "model-proposed",
                "count_evidence": "Includes privacy engineering, security engineering, and software privacy roles.",
            },
        }
    ]

    persist_proposed_artifact(canonical, proposed, run_id="run-1", item=item, accepted=accepted)

    row = json.loads(proposed.read_text())["profiles"]["profession"]["privacy engineer"]
    assert row["aliases"] == ["data protection engineer", "privacy software engineer"]
    assert row["level_groundings"]["privacy and security software professional"]["status"] == "model-proposed"


def test_persist_keeps_counts_and_groundings_in_level_order(tmp_path: Path) -> None:
    canonical = tmp_path / "fine_lattice_profiles.json"
    proposed = tmp_path / "fine_lattice_profiles.proposed.json"
    _base_profile(canonical)
    item = {
        "item_id": "profession:sustainability auditor",
        "runtime_type": "profession",
        "surface": "sustainability auditor",
    }
    accepted = [
        {"level": "professional worker", "level_count": 1000.0, "level_grounding": {"status": "certifying"}},
        {"level": "business and financial occupation", "level_count": 1000.0, "level_grounding": {"status": "certifying"}},
        {"level": "technical worker", "level_count": 1000.0, "level_grounding": {"status": "certifying"}},
    ]

    persist_proposed_artifact(canonical, proposed, run_id="run-1", item=item, accepted=accepted)

    row = json.loads(proposed.read_text())["profiles"]["profession"]["sustainability auditor"]
    assert row["levels"] == ["business and financial occupation", "technical worker", "professional worker"]
    assert list(row["level_counts"]) == row["levels"]
    assert list(row["level_groundings"]) == row["levels"]


def test_persist_entry_count_uses_most_specific_lowest_level_count(tmp_path: Path) -> None:
    canonical = tmp_path / "fine_lattice_profiles.json"
    proposed = tmp_path / "fine_lattice_profiles.proposed.json"
    _base_profile(canonical)
    item = {
        "item_id": "profession:stage rigger",
        "runtime_type": "profession",
        "surface": "stage rigger",
    }
    accepted = [
        {"level": "construction worker", "level_count": 150.0, "level_grounding": {"status": "model-proposed"}},
        {"level": "skilled trades worker", "level_count": 300.0, "level_grounding": {"status": "model-proposed"}},
    ]

    persist_proposed_artifact(canonical, proposed, run_id="run-1", item=item, accepted=accepted)

    row = json.loads(proposed.read_text())["profiles"]["profession"]["stage rigger"]
    assert row["levels"] == ["construction worker", "skilled trades worker"]
    assert row["count"] == 150.0


def test_ensure_proposed_artifact_materializes_empty_review_file(tmp_path: Path) -> None:
    canonical = tmp_path / "fine_lattice_profiles.json"
    proposed = tmp_path / "proposed" / "fine_lattice_profiles.empty.proposed.json"
    _base_profile(canonical)

    ensure_proposed_artifact(canonical, proposed, run_id="run-empty")

    artifact = json.loads(proposed.read_text())
    assert artifact["artifact_role"] == "proposal"
    assert artifact["proposal_scope"] == "producer-processed-only"
    assert artifact["producer_run_id"] == "run-empty"
    assert artifact["profiles"] == {}


def test_ensure_proposed_artifact_resets_old_copied_proposal_file(tmp_path: Path) -> None:
    canonical = tmp_path / "fine_lattice_profiles.json"
    proposed = tmp_path / "proposed" / "fine_lattice_profiles.old.proposed.json"
    _base_profile(canonical)
    proposed.parent.mkdir()
    proposed.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_role": "proposal",
                "profiles": {
                    "profession": {
                        "journalist": {"aliases": [], "levels": ["media worker"], "source_ids": [], "count": 1000}
                    }
                },
            }
        )
    )

    ensure_proposed_artifact(canonical, proposed, run_id="run-reset")

    artifact = json.loads(proposed.read_text())
    assert artifact["proposal_scope"] == "producer-processed-only"
    assert artifact["profiles"] == {}


def test_validate_proposed_artifact_rejects_dem_placeholders_and_missing_counts(tmp_path: Path) -> None:
    proposed = tmp_path / "bad.proposed.json"
    proposed.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created": "2026-07-07",
                "artifact_role": "proposal",
                "profiles": {
                    "DEM": {"polish": {"levels": ["european"], "count": 1}},
                    "profession": {"doctor": {"levels": ["<PROFESSION_1>"], "source_ids": [], "count": 1}},
                },
            }
        )
    )

    errors = validate_proposed_artifact(proposed)

    assert any("DEM" in e for e in errors)
    assert any("placeholder" in e for e in errors)
    assert any("level_counts" in e for e in errors)
