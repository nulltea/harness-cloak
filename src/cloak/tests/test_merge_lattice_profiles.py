import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from merge_lattice_profiles import merge_profile_artifacts


def _row(levels, *, aliases=None, source_ids=None, count=1.0):
    return {
        "aliases": aliases or [],
        "levels": levels,
        "source_ids": source_ids or ["test:row"],
        "count": count,
    }


def test_merge_profile_artifacts_adds_mined_categories_and_entries():
    common = {
        "schema_version": 1,
        "created": "2026-07-01",
        "sources": {"common": {"path": "comm"}},
        "profiles": {
            "drug": {"acetaminophen": _row(["medication"], count=1000.0)},
            "profession": {"teacher": _row(["education worker"], count=500.0)},
        },
    }
    mined = {
        "schema_version": 1,
        "created": "2026-07-02",
        "sources": {"clinical-mined": {"corpus": "clinical"}},
        "profiles": {
            "medical-procedure": {
                "nerve conduction study": _row(
                    ["medical procedure"],
                    source_ids=["mined-clinical:doc42"],
                )
            }
        },
    }

    artifact = merge_profile_artifacts(common, mined, created="2026-07-07")

    assert artifact["created"] == "2026-07-07"
    assert set(artifact["sources"]) == {"common", "clinical-mined"}
    assert "teacher" in artifact["profiles"]["profession"]
    assert artifact["profiles"]["medical-procedure"]["nerve conduction study"] == {
        "aliases": [],
        "levels": ["medical procedure"],
        "source_ids": ["mined-clinical:doc42"],
        "count": 1.0,
    }


def test_merge_profile_artifacts_dedupes_mined_entry_against_existing_alias():
    common = {
        "schema_version": 1,
        "created": "2026-07-01",
        "sources": {},
        "profiles": {
            "drug": {
                "acetaminophen": _row(
                    ["medication"],
                    aliases=["apap", "tylenol"],
                    source_ids=["common:acetaminophen"],
                    count=1000.0,
                )
            }
        },
    }
    mined = {
        "schema_version": 1,
        "created": "2026-07-02",
        "sources": {},
        "profiles": {
            "drug": {
                "tylenol": _row(
                    ["over the counter medication"],
                    aliases=["noisy unrelated product"],
                    source_ids=["mined-clinical:doc7"],
                    count=1.0,
                )
            }
        },
    }

    artifact = merge_profile_artifacts(common, mined, created="2026-07-07")
    drugs = artifact["profiles"]["drug"]

    assert list(drugs) == ["acetaminophen"]
    assert drugs["acetaminophen"] == {
        "aliases": ["apap", "tylenol"],
        "levels": ["medication", "over the counter medication"],
        "source_ids": ["common:acetaminophen"],
        "count": 1000.0,
    }


def test_merge_profile_artifacts_does_not_match_drugs_by_noisy_mined_aliases():
    common = {
        "schema_version": 1,
        "created": "2026-07-01",
        "sources": {},
        "profiles": {
            "drug": {
                "acetaminophen": _row(
                    ["medication"],
                    aliases=["tylenol"],
                    source_ids=["common:acetaminophen"],
                    count=1000.0,
                )
            }
        },
    }
    mined = {
        "schema_version": 1,
        "created": "2026-07-02",
        "sources": {},
        "profiles": {
            "drug": {
                ".alpha.-hexylcinnamaldehyde": _row(
                    ["medication"],
                    aliases=["acetaminophen", "random skincare product"],
                    source_ids=[f"openfda-ndc:{i:04d}" for i in range(10, 0, -1)],
                    count=1.0,
                )
            }
        },
    }

    artifact = merge_profile_artifacts(common, mined, created="2026-07-07")
    drugs = artifact["profiles"]["drug"]

    assert set(drugs) == {"acetaminophen", ".alpha.-hexylcinnamaldehyde"}
    assert drugs["acetaminophen"]["aliases"] == ["tylenol"]
    assert drugs["acetaminophen"]["source_ids"] == ["common:acetaminophen"]
    assert drugs[".alpha.-hexylcinnamaldehyde"] == {
        "aliases": [],
        "levels": ["medication"],
        "source_ids": [
            "openfda-ndc:0001",
            "openfda-ndc:0002",
            "openfda-ndc:0003",
            "openfda-ndc:0004",
            "openfda-ndc:0005",
        ],
        "count": 1.0,
    }
