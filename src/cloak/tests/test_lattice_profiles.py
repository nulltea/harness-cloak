import json

from cloak.lattice_profiles import (
    load_profiles,
    lookup_count,
    lookup_levels,
    validate_profile_artifact,
)


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


def test_lookup_levels_by_surface_and_alias(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(_artifact()))

    assert lookup_levels("journalist", "profession", path) == ["media worker"]
    assert lookup_levels("Reporter", "profession", path) == ["media worker"]
    assert lookup_levels("diabetes mellitus", "health-condition", path) == [
        "endocrine condition",
        "chronic condition",
    ]


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


def test_load_missing_profile_returns_empty_artifact(tmp_path):
    got = load_profiles(tmp_path / "missing.json")
    assert got["schema_version"] == 1
    assert got["profiles"] == {}


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
