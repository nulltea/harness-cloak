import json

from cloak.lattice import lattice_for
from cloak.lattice_profiles import (
    load_profiles,
    lookup_count,
    lookup_levels,
    validate_profile_artifact,
)


def _clear_lattice_profile_caches():
    import cloak.lattice_profiles as lp

    lp._load_cached.cache_clear()
    lp._index_cached.cache_clear()


def _artifact():
    return {
        "schema_version": 1,
        "created": "2026-07-07",
        "sources": {"profession": ["esco"]},
        "profiles": {
            "profession": {
                "journalist": {
                    "aliases": ["reporter"],
                    "levels": ["media worker"],
                    "source_ids": ["esco:2642.7"],
                    "count": 1000.0,
                }
            },
            "health-condition": {
                "diabetes": {
                    "aliases": ["diabetes mellitus"],
                    "levels": ["endocrine condition", "chronic condition"],
                    "source_ids": ["mondo:0005015"],
                    "count": 1000.0,
                }
            },
        },
    }


def _standard_artifact():
    return {
        "schema_version": 1,
        "created": "2026-07-07",
        "sources": {"profession": ["esco-rdf"]},
        "profiles": {
            "profession": {
                "journalist": {
                    "aliases": ["reporter"],
                    "levels": ["media worker"],
                    "source_ids": ["esco:2642.7"],
                    "count": 1000.0,
                }
            }
        },
    }


def test_lookup_levels_by_surface_and_alias(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(_artifact()))

    assert lookup_levels("journalist", "profession", path) == ["media worker"]
    assert lookup_levels("Reporter", "profession", path) == ["media worker"]
    assert lookup_levels("diabetes mellitus", "health-condition", path) == [
        "endocrine condition",
        "chronic condition",
    ]


def test_lookup_entry_returns_canonical_for_canonical_and_alias(tmp_path):
    artifact = {
        "schema_version": 1, "created": "2026-07-10", "sources": {},
        "profiles": {"health-condition": {
            "blorbitis": {"aliases": ["blorb inflammation"],
                          "levels": ["organ disease", "disease"],
                          "source_ids": ["t:1"], "count": 100.0},
        }},
    }
    p = tmp_path / "profiles.json"
    p.write_text(json.dumps(artifact))
    from cloak.lattice_profiles import lookup_entry, lookup_levels
    assert lookup_entry("Blorbitis", "health-condition", p) == \
        ("blorbitis", ["organ disease", "disease"])
    assert lookup_entry("blorb  inflammation", "health-condition", p) == \
        ("blorbitis", ["organ disease", "disease"])
    assert lookup_entry("unknownitis", "health-condition", p) is None
    # wrapper unchanged behavior
    assert lookup_levels("blorb inflammation", "health-condition", p) == \
        ["organ disease", "disease"]
    assert lookup_levels("unknownitis", "health-condition", p) == []


def test_lookup_levels_accepts_standard_levels_field(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(_standard_artifact()))

    assert lookup_levels("Reporter", "profession", path) == ["media worker"]


def test_lookup_count_by_runtime_type_and_fill(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(_artifact()))

    assert lookup_count("media worker", "profession", path) == 1000.0
    assert lookup_count("chronic condition", "health-condition", path) == 1000.0
    assert lookup_count("media worker", "health-condition", path) is None


def test_validate_rejects_type_name_phrases_and_unknown_types():
    art = _artifact()
    art["profiles"]["profession"]["bad"] = {
        "aliases": [],
        "levels": ["a profession"],
        "source_ids": ["manual:bad"],
        "count": 1000.0,
    }
    art["profiles"]["nosuchtype"] = {}

    errors = validate_profile_artifact(art)
    assert any("type-name phrase" in e for e in errors)
    assert any("unknown runtime type" in e for e in errors)


def test_validate_standard_cache_schema_requires_inline_levels():
    art = _standard_artifact()

    assert validate_profile_artifact(art) == []

    bad = _standard_artifact()
    bad["schema_version"] = 2
    bad["profiles"]["profession"]["journalist"].pop("levels")

    errors = validate_profile_artifact(bad)
    assert any("schema_version must be 1" in e for e in errors)
    assert any("profession:journalist has no levels" in e for e in errors)


def test_load_missing_profile_returns_empty_artifact(tmp_path):
    got = load_profiles(tmp_path / "missing.json")
    assert got["schema_version"] == 1
    assert got["profiles"] == {}


def test_lookup_indexes_aliases_and_level_counts(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(_artifact()))

    assert lookup_levels("Reporter", "profession", path) == ["media worker"]
    assert lookup_count("media worker", "profession", path) == 1000.0


def test_lookup_count_uses_explicit_level_count_not_row_count(tmp_path):
    art = _artifact()
    art["profiles"]["drug"] = {
        "bupropion": {
            "aliases": [],
            "levels": ["aminoketone", "medication"],
            "source_ids": [],
            "count": 9.0,
            "level_counts": {"aminoketone": 9.0, "medication": 10999.71},
        },
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(art))
    _clear_lattice_profile_caches()

    assert lookup_count("medication", "drug", path) == 10999.71
    assert lookup_count("aminoketone", "drug", path) == 9.0


def test_lookup_count_uses_max_when_multiple_explicit_rows_share_level(tmp_path):
    art = _artifact()
    art["profiles"]["drug"] = {
        "aspirin": {
            "aliases": [],
            "levels": ["analgesic", "medication"],
            "source_ids": [],
            "count": 4.0,
            "level_counts": {"analgesic": 4.0, "medication": 20.0},
        },
        "ibuprofen": {
            "aliases": [],
            "levels": ["analgesic", "medication"],
            "source_ids": [],
            "count": 7.0,
            "level_counts": {"analgesic": 7.0, "medication": 35.0},
        },
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(art))
    _clear_lattice_profile_caches()

    assert lookup_count("medication", "drug", path) == 35.0


def test_lookup_count_uses_max_of_legacy_row_counts_when_level_counts_absent(tmp_path):
    art = _artifact()
    art["profiles"]["drug"] = {
        "aspirin": {"aliases": [], "levels": ["medication"], "source_ids": [], "count": 4.0},
        "ibuprofen": {"aliases": [], "levels": ["medication"], "source_ids": [], "count": 7.0},
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(art))
    _clear_lattice_profile_caches()

    assert lookup_count("medication", "drug", path) == 7.0


def test_lookup_count_prefers_explicit_level_count_over_legacy_row_count(tmp_path):
    art = _artifact()
    art["profiles"]["drug"] = {
        "aspirin": {"aliases": [], "levels": ["medication"], "source_ids": [], "count": 900.0},
        "bupropion": {
            "aliases": [],
            "levels": ["aminoketone", "medication"],
            "source_ids": [],
            "count": 9.0,
            "level_counts": {"aminoketone": 9.0, "medication": 42.0},
        },
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(art))
    _clear_lattice_profile_caches()

    assert lookup_count("medication", "drug", path) == 42.0


def test_lookup_count_prefers_explicit_level_counts_over_scalar_sum(tmp_path):
    art = _artifact()
    art["profiles"]["drug"] = {
        "aspirin": {"aliases": [], "levels": ["medication"], "source_ids": [], "count": 4.0},
        "bupropion": {
            "aliases": [],
            "levels": ["aminoketone", "medication"],
            "source_ids": [],
            "count": 9.0,
            "level_counts": {"aminoketone": 9.0, "medication": 10999.71},
            "level_grounding": {
                "medication": {"status": "certifying", "source_family": "openfda-pharm-class"},
            },
        },
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(art))

    # an explicit level_counts value on any row sharing the level wins over the legacy
    # scalar-count sum from rows that only have the old schema.
    assert lookup_count("medication", "drug", path) == 10999.71
    assert lookup_count("aminoketone", "drug", path) == 9.0


def test_validate_rejects_non_monotone_level_counts():
    art = _artifact()
    art["profiles"]["drug"] = {
        "bupropion": {
            "aliases": [],
            "levels": ["aminoketone", "medication"],
            "source_ids": [],
            "count": 9.0,
            "level_counts": {"aminoketone": 100.0, "medication": 9.0},
        },
    }

    errors = validate_profile_artifact(art)
    assert any("level_counts not monotone" in e for e in errors)


def test_validate_rejects_bad_level_counts():
    art = _artifact()
    art["profiles"]["drug"] = {
        "bupropion": {
            "aliases": [],
            "levels": ["aminoketone", "medication"],
            "source_ids": [],
            "count": 9.0,
            "level_counts": {"aminoketone": 9.0, "not-a-real-level": 5.0, "medication": 0.5},
        },
    }

    errors = validate_profile_artifact(art)
    assert any("level_counts key not in levels: not-a-real-level" in e for e in errors)
    assert any("level_counts['medication'] must be >= 1" in e for e in errors)


def test_lattice_for_uses_profile_levels(monkeypatch, tmp_path):
    import cloak.lattice as lat
    import cloak.lattice_profiles as lp

    art = _artifact()
    art["profiles"]["profession"]["editorial columnist"] = {
        "aliases": ["correspondent"],
        "levels": ["publishing worker"],
        "source_ids": ["test:profile-only"],
        "count": 321.0,
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(art))
    monkeypatch.setattr(lp, "DEFAULT_PROFILE_PATH", path)
    monkeypatch.setattr(lat, "wordnet_chain", lambda *args, **kwargs: None)
    lp._load_cached.cache_clear()

    got = lat.lattice_for("correspondent", "profession", "The correspondent called.")

    assert got == ["publishing worker", "<PROFESSION_1>"]


def test_lattice_for_uses_profile_levels_for_loc_and_org(monkeypatch, tmp_path):
    import cloak.lattice as lat
    import cloak.lattice_profiles as lp

    art = _artifact()
    art["profiles"]["LOC"] = {
        "oslo": {
            "aliases": [],
            "levels": ["a city in norway", "a city in europe"],
            "source_ids": ["geonames:3143244"],
            "count": 1082575.0,
        }
    }
    art["profiles"]["ORG"] = {
        "sberbank": {
            "aliases": [],
            "levels": ["a financial institution", "an organization"],
            "source_ids": ["legacy-teacher-cache:sberbank"],
            "count": 1.0,
        }
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(art))
    monkeypatch.setattr(lp, "DEFAULT_PROFILE_PATH", path)
    monkeypatch.setattr(lat, "geonames_chain", lambda *args, **kwargs: None)
    monkeypatch.setattr(lat, "wordnet_chain", lambda *args, **kwargs: None)
    lp._load_cached.cache_clear()
    lp._index_cached.cache_clear()

    assert lat.lattice_for("Oslo", "LOC") == ["a city in norway", "a city in europe"]
    assert lat.lattice_for("Sberbank", "ORG") == ["a financial institution", "an organization"]


def test_lattice_for_uses_profile_levels_for_drugs(monkeypatch, tmp_path):
    import cloak.lattice_profiles as lp

    art = _artifact()
    art["profiles"]["drug"] = {
        "glucophage": {
            "aliases": ["metformin hydrochloride"],
            "levels": ["medication"],
            "source_ids": ["openfda-ndc:0002-8215"],
            "count": 1000.0,
        }
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(art))
    monkeypatch.setattr(lp, "DEFAULT_PROFILE_PATH", path)
    lp._load_cached.cache_clear()
    lp._index_cached.cache_clear()

    assert lookup_levels("metformin hydrochloride", "drug", path) == ["medication"]
    assert validate_profile_artifact(art) == []
    assert lattice_for("Glucophage", "drug") == ["medication", "<DRUG_1>"]


def test_lattice_for_uses_profile_levels_for_medical_procedures(monkeypatch, tmp_path):
    import cloak.lattice_profiles as lp

    art = _artifact()
    art["profiles"]["medical-procedure"] = {
        "excision appendix open approach": {
            "aliases": [],
            "levels": ["medical and surgical procedure", "medical procedure"],
            "source_ids": ["icd10pcs:0DBJ0ZZ"],
            "count": 1000.0,
        }
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(art))
    monkeypatch.setattr(lp, "DEFAULT_PROFILE_PATH", path)
    lp._load_cached.cache_clear()
    lp._index_cached.cache_clear()

    assert validate_profile_artifact(art) == []
    assert lattice_for("excision appendix open approach", "medical-procedure") == [
        "medical and surgical procedure",
        "medical procedure",
        "<MEDICAL_PROCEDURE_1>",
    ]


def test_lattice_for_uses_profile_levels_for_medical_facilities(monkeypatch, tmp_path):
    import cloak.lattice_profiles as lp

    art = _artifact()
    art["profiles"]["organization-medical-facility"] = {
        "example general hospital": {
            "aliases": ["example hospital"],
            "levels": ["hospital", "healthcare organization"],
            "source_ids": ["nppes:1234567890"],
            "count": 1000.0,
        }
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(art))
    monkeypatch.setattr(lp, "DEFAULT_PROFILE_PATH", path)
    lp._load_cached.cache_clear()
    lp._index_cached.cache_clear()

    assert validate_profile_artifact(art) == []
    assert lattice_for("example hospital", "organization-medical-facility") == [
        "hospital",
        "healthcare organization",
        "<ORGANIZATION_MEDICAL_FACILITY_1>",
    ]


def test_substitute_uses_profile_levels(monkeypatch, tmp_path):
    import cloak.lattice_profiles as lp
    import cloak.substitute as sub
    from cloak.detect import Span

    art = _artifact()
    art["profiles"]["profession"]["software developer"] = {
        "aliases": ["application developer"],
        "levels": ["computer and mathematical occupation", "professional worker"],
        "source_ids": ["test:profile"],
        "count": 1000.0,
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(art))
    monkeypatch.setattr(lp, "DEFAULT_PROFILE_PATH", path)
    monkeypatch.setattr(sub, "walk_risk", lambda *args, **kwargs: 0.0)
    lp._load_cached.cache_clear()
    lp._index_cached.cache_clear()

    text = "Mia is an application developer."
    span = Span(
        text.index("application developer"),
        text.index("application developer") + len("application developer"),
        "application developer",
        "profession",
        0.99,
        "stub",
    )

    doc_p, record = sub.substitute(text, [span], tau=0.02)

    assert "an computer" not in doc_p
    assert "a computer and mathematical occupation" in doc_p
    assert record[0]["action"] == "generalize"
    assert record[0]["lattice"] == [
        "computer and mathematical occupation",
        "professional worker",
        "<PROFESSION_1>",
    ]


def test_aset_count_uses_profile_count(monkeypatch, tmp_path):
    import cloak.anonymity as anon
    import cloak.lattice_profiles as lp

    art = _artifact()
    art["profiles"]["profession"]["editorial columnist"] = {
        "aliases": ["correspondent"],
        "levels": ["publishing worker"],
        "source_ids": ["test:profile-only"],
        "count": 321.0,
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(art))
    monkeypatch.setattr(lp, "DEFAULT_PROFILE_PATH", path)
    monkeypatch.setattr(anon, "_wn_leaf_count", lambda *args, **kwargs: None)
    lp._load_cached.cache_clear()
    anon.aset_count.cache_clear()

    assert anon.aset_count("publishing worker", "profession", "correspondent", strict=True) == 321.0


def test_aset_count_uses_profile_level_counts_for_domain_types_and_fails_closed(monkeypatch, tmp_path):
    import cloak.anonymity as anon
    import cloak.lattice_profiles as lp

    art = _artifact()
    art["profiles"]["drug"] = {
        "bupropion": {
            "aliases": [],
            "levels": ["aminoketone", "medication"],
            "source_ids": ["test:drug"],
            "count": 9.0,
            "level_counts": {"aminoketone": 9.0, "medication": 42.0},
        },
    }
    art["profiles"]["health-condition"]["asthma"] = {
        "aliases": [],
        "levels": ["respiratory condition", "chronic condition"],
        "source_ids": ["test:health"],
        "count": 12.0,
        "level_counts": {"respiratory condition": 12.0, "chronic condition": 88.0},
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(art))
    monkeypatch.setattr(lp, "DEFAULT_PROFILE_PATH", path)
    monkeypatch.setattr(anon, "_wn_leaf_count", lambda *args, **kwargs: None)
    _clear_lattice_profile_caches()
    anon.aset_count.cache_clear()

    assert anon.aset_count("medication", "drug", "bupropion", strict=True) == 42.0
    assert anon.aset_count("chronic condition", "health-condition", "asthma", strict=True) == 88.0
    assert anon.aset_count("not in the profile", "drug", "bupropion", strict=True) == 1.0
