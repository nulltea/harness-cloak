import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from merge_lattice_profiles import apply_curated_merges, merge_profile_artifacts


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
        "entry_origin": "observed-surface",
    }


def test_merge_profile_artifacts_carries_level_counts_into_matched_entry():
    common = {
        "schema_version": 1,
        "created": "2026-07-01",
        "sources": {},
        "profiles": {
            "drug": {"bupropion": _row(["medication"], count=386.0)},
        },
    }
    mined = {
        "schema_version": 1,
        "created": "2026-07-08",
        "sources": {},
        "profiles": {
            "drug": {
                "bupropion": {
                    "aliases": [],
                    "levels": ["aminoketone", "medication", "chemical substance"],
                    "source_ids": ["producer:x:bupropion"],
                    "count": 9.0,
                    "level_counts": {
                        "aminoketone": 9.0,
                        "medication": 10999.71,
                        "chemical substance": 249969031.57,
                    },
                    "level_grounding": {
                        "aminoketone": {"status": "certifying", "source_family": "openfda-pharm-class"},
                        "medication": {"status": "model-proposed", "source_family": "model-proposed"},
                        "chemical substance": {"status": "model-proposed", "source_family": "model-proposed"},
                    },
                }
            }
        },
    }

    artifact = merge_profile_artifacts(common, mined, created="2026-07-08")
    row = artifact["profiles"]["drug"]["bupropion"]

    # new class level from the incoming row is folded into the level chain, reordered by the
    # now-resolved counts (not existing-then-incoming concatenation order) so the chain stays
    # narrow-to-broad: aminoketone (9) < medication (10999.71) < chemical substance (249969031.57).
    assert row["levels"] == ["aminoketone", "medication", "chemical substance"]
    assert row["level_counts"] == {
        "aminoketone": 9.0,
        "medication": 10999.71,
        "chemical substance": 249969031.57,
    }
    assert row["level_grounding"]["aminoketone"]["status"] == "certifying"


def test_merge_profile_artifacts_drops_self_referential_level_from_folded_narrower_entry():
    # regression: a narrower entry ("aortic valve stenosis") folds into the broader canonical
    # ("aortic valve disease") via alias match; its most-specific level IS the canonical surface,
    # which would leak the original. The merge must drop that self-referential level (and its
    # level_counts/level_grounding) rather than emit an artifact validate_profile_artifact rejects.
    common = {
        "schema_version": 1,
        "created": "2026-07-01",
        "sources": {},
        "profiles": {
            "health-condition": {
                "aortic valve disease": _row(
                    ["heart valve disease", "heart disease"],
                    aliases=["aortic valve stenosis"],
                    count=1.0,
                ),
            },
        },
    }
    mined = {
        "schema_version": 1,
        "created": "2026-07-08",
        "sources": {},
        "profiles": {
            "health-condition": {
                "aortic valve stenosis": {
                    "aliases": [],
                    "levels": ["aortic valve disease", "heart valve disease"],
                    "source_ids": ["producer:x:aortic valve stenosis"],
                    "count": 1.0,
                    "level_counts": {"aortic valve disease": 29.0, "heart valve disease": 40.0},
                    "level_grounding": {"aortic valve disease": {"status": "model-proposed"}},
                }
            }
        },
    }

    artifact = merge_profile_artifacts(common, mined, created="2026-07-08")
    row = artifact["profiles"]["health-condition"]["aortic valve disease"]

    assert "aortic valve disease" not in row["levels"]
    assert "aortic valve disease" not in row.get("level_counts", {})
    assert "aortic valve disease" not in row.get("level_grounding", {})
    assert "aortic valve stenosis" in row["aliases"]


def test_merge_profile_artifacts_matches_level_counts_casing_to_merged_levels():
    # regression: _merge_unique_preserve_order keeps whichever side's casing it saw first for a
    # level ("infectious disease" from existing beats "Infectious disease" from incoming), so
    # level_counts keys must be remapped to that casing too, not copied from incoming verbatim.
    common = {
        "schema_version": 1,
        "created": "2026-07-01",
        "sources": {},
        "profiles": {
            "health-condition": {"chlamydia": _row(["infectious disease"], count=1000.0)},
        },
    }
    mined = {
        "schema_version": 1,
        "created": "2026-07-08",
        "sources": {},
        "profiles": {
            "health-condition": {
                "chlamydia": {
                    "aliases": [],
                    "levels": ["Sexually transmitted infection", "Infectious disease"],
                    "source_ids": ["test:row"],
                    "count": 200.0,
                    "level_counts": {"Sexually transmitted infection": 200.0, "Infectious disease": 1400.0},
                    "level_grounding": {
                        "Sexually transmitted infection": {"status": "certifying"},
                        "Infectious disease": {"status": "certifying"},
                    },
                }
            }
        },
    }

    artifact = merge_profile_artifacts(common, mined, created="2026-07-08")
    row = artifact["profiles"]["health-condition"]["chlamydia"]

    assert set(row["level_counts"]) == set(row["levels"])
    assert row["level_counts"]["infectious disease"] == 1400.0


def test_merge_profile_artifacts_keeps_existing_certifying_count_over_incoming_guess():
    common = {
        "schema_version": 1,
        "created": "2026-07-01",
        "sources": {},
        "profiles": {
            "drug": {
                "bupropion": {
                    "aliases": [],
                    "levels": ["aminoketone"],
                    "source_ids": ["test:row"],
                    "count": 9.0,
                    "level_counts": {"aminoketone": 9.0},
                    "level_grounding": {"aminoketone": {"status": "certifying"}},
                }
            },
        },
    }
    mined = {
        "schema_version": 1,
        "created": "2026-07-08",
        "sources": {},
        "profiles": {
            "drug": {
                "bupropion": {
                    "aliases": [],
                    "levels": ["aminoketone"],
                    "source_ids": ["test:row"],
                    "count": 1.0,
                    "level_counts": {"aminoketone": 500000.0},
                    "level_grounding": {"aminoketone": {"status": "model-proposed"}},
                }
            }
        },
    }

    artifact = merge_profile_artifacts(common, mined, created="2026-07-08")
    row = artifact["profiles"]["drug"]["bupropion"]

    assert row["level_counts"]["aminoketone"] == 9.0
    assert row["level_grounding"]["aminoketone"]["status"] == "certifying"


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


def test_merge_profile_artifacts_entity_dedup_unions_distinct_canonicals():
    # entity_dedup=True: no alias fold. An incoming entry whose canonical (or alias) points at an
    # existing surface is kept SEPARATE; only an exact canonical norm-equality merges via _merge_row.
    common = {
        "schema_version": 1,
        "created": "2026-07-01",
        "sources": {},
        "profiles": {
            "health-condition": {
                "zorbosis": _row(["fictional ailment"], source_ids=["common:zorbosis"], count=5.0),
            }
        },
    }
    mined = {
        "schema_version": 1,
        "created": "2026-07-02",
        "sources": {},
        "profiles": {
            "health-condition": {
                # alias points at the existing canonical -> DEFAULT path folds; dedup keeps separate
                "quibbex": _row(["fictional ailment"], aliases=["zorbosis"], source_ids=["mined:1"]),
                # exact canonical dup -> merges into the existing entry even under dedup
                "zorbosis": _row(["fictional ailment", "broad ailment"], source_ids=["mined:2"]),
            }
        },
    }

    deduped = merge_profile_artifacts(common, mined, entity_dedup=True, created="2026-07-07")
    hc = deduped["profiles"]["health-condition"]
    assert set(hc) == {"zorbosis", "quibbex"}          # quibbex NOT alias-folded
    assert "quibbex" not in hc["zorbosis"].get("aliases", [])
    assert "broad ailment" in hc["zorbosis"]["levels"]  # exact-canonical dup merged
    assert "mined:2" in hc["zorbosis"]["source_ids"]

    # contrast: the default alias-fold path collapses quibbex into zorbosis
    folded = merge_profile_artifacts(common, mined, created="2026-07-07")
    assert set(folded["profiles"]["health-condition"]) == {"zorbosis"}


def test_apply_curated_merges_folds_pair_and_skips_missing(capsys):
    artifact = {
        "schema_version": 1,
        "created": "2026-07-01",
        "sources": {},
        "profiles": {
            "drug": {
                "zalprix": {
                    "aliases": ["zx"],
                    "levels": ["antiviral medication"],
                    "source_ids": ["common:zalprix"],
                    "count": 3.0,
                    "level_counts": {"antiviral medication": 100.0},
                },
                "zalprex": {
                    "aliases": ["zpx"],
                    "levels": ["antiviral medication"],
                    "source_ids": ["mined:zalprex"],
                    "count": 7.0,
                    "level_counts": {"antiviral medication": 250.0},
                },
            }
        },
    }

    apply_curated_merges(artifact, [
        ("drug", "zalprix", "zalprex"),
        ("drug", "zalprix", "does-not-exist"),  # missing fold -> warn + skip
    ])
    drugs = artifact["profiles"]["drug"]

    assert "zalprex" not in drugs                       # folded canonical removed
    keep = drugs["zalprix"]
    assert set(keep["aliases"]) == {"zalprex", "zpx", "zx"}   # aliases + folded name union
    assert set(keep["source_ids"]) == {"common:zalprix", "mined:zalprex"}
    assert keep["count"] == 7.0                          # max count
    assert keep["level_counts"]["antiviral medication"] == 250.0  # per-shared-level max
    assert "WARN curated-merge skipped" in capsys.readouterr().out
