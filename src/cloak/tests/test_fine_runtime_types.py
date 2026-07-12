import json
import sys
from pathlib import Path

import pytest

from cloak.detect import Span
from cloak.extract import _type_sane, invert
from cloak.lattice import lattice_for
from cloak.anonymity import aset_count
from cloak.probe import MIN_POOL, walk_risk
from cloak.substitute import substitute
from cloak.runtime_types import (
    PLACEHOLDER_RE,
    PLACEHOLDER_ONLY_TYPES,
    RUNTIME_TYPES,
    placeholder_type_token,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import build_probe_distractors as bpd  # noqa: E402
import build_ranker_env as bre  # noqa: E402
import train_ranker as tr  # noqa: E402


def test_placeholder_token_normalization_and_regex():
    assert placeholder_type_token("health-condition") == "HEALTH_CONDITION"
    assert placeholder_type_token("PERSON") == "PERSON"
    assert PLACEHOLDER_RE.fullmatch("<HEALTH_CONDITION_1>")
    assert PLACEHOLDER_RE.fullmatch("<MARITAL_STATUS_2>")
    assert PLACEHOLDER_RE.fullmatch("<PERSON_1>")
    assert not PLACEHOLDER_RE.fullmatch("<health-condition_1>")


@pytest.mark.parametrize(
    ("surface", "typ", "expected"),
    [
        ("diabetes", "health-condition", "disease of metabolism"),
        ("journalist", "profession", "media worker"),
        ("Kurdish", "ethnicity", "of Middle Eastern ethnicity"),
    ],
)
def test_hierarchical_fine_lattices_have_grammatical_text_and_placeholder(surface, typ, expected):
    got = lattice_for(surface, typ, f"The marked span is {surface}.")
    assert expected in got
    assert got[-1] == f"<{placeholder_type_token(typ)}_1>"
    assert f"a {typ.replace('-', ' ')}" not in got


def test_placeholder_only_fine_lattice_has_no_semantic_text_by_default():
    assert lattice_for("married", "marital-status", "She is married.") == ["<MARITAL_STATUS_1>"]
    assert lattice_for("female", "gender", "The patient is female.") == ["<GENDER_1>"]


def test_teacher_cache_is_runtime_type_keyed_and_filtered(monkeypatch, tmp_path):
    import cloak.lattice as lat

    cache = tmp_path / "lattice_cache.json"
    cache.write_text(json.dumps({
        "profession::apple": {"lattice": ["technology worker", "apple", "a profession"]},
        "health-condition::apple": {"lattice": ["chronic condition"]},
        "ethnicity::unknownsurface": {"lattice": []},
    }))
    monkeypatch.setattr(lat, "CACHE", cache)
    monkeypatch.setattr(lat, "wordnet_chain", lambda *args, **kwargs: None)

    assert lat.lattice_for("apple", "profession") == ["technology worker", "<PROFESSION_1>"]
    assert lat.lattice_for("apple", "health-condition") == [
        "chronic condition", "<HEALTH_CONDITION_1>",
    ]
    assert lat.lattice_for("unknownsurface", "ethnicity") == ["<ETHNICITY_1>"]


def test_ranker_env_floors_are_inert_for_runtime_types():
    assert bre.inert_runtime_floors() == {t: 1.0 for t in RUNTIME_TYPES}


def test_fine_anonymity_fail_closed_counts():
    assert aset_count("thirty-something", "age", "34", strict=True) == 10.0
    assert aset_count("a health condition", "health-condition", "diabetes", strict=True) == 1.0
    assert aset_count("a profession", "profession", "journalist", strict=True) == 1.0
    assert aset_count("a gender", "gender", "female", strict=True) == 1.0
    assert aset_count("sector-ish", "profession", "journalist", strict=True) == 1.0


def test_fine_dem_aset_count_no_longer_uses_hand_count_fallback(monkeypatch):
    import cloak.anonymity as anon

    monkeypatch.setattr(anon, "lookup_count", lambda *args, **kwargs: None)
    monkeypatch.setattr(anon, "_wn_leaf_count", lambda *args, **kwargs: None)
    anon.aset_count.cache_clear()

    assert anon.aset_count("medical specialist", "profession", "journalist", strict=True) == 1.0


def test_substitute_preserves_fine_type_and_emits_typed_placeholder(monkeypatch):
    monkeypatch.setattr("cloak.substitute.walk_risk", lambda *args, **kwargs: 1.0)
    text = "The patient has diabetes and is married."
    spans = [
        Span(text.index("diabetes"), text.index("diabetes") + len("diabetes"),
             "diabetes", "health-condition", 0.99, "stub"),
        Span(text.index("married"), text.index("married") + len("married"),
             "married", "marital-status", 0.99, "stub"),
    ]

    doc_p, R = substitute(text, spans)

    assert "<HEALTH_CONDITION_1>" in doc_p
    assert "<MARITAL_STATUS_1>" in doc_p
    assert [r["type"] for r in R] == ["health-condition", "marital-status"]
    assert "DEM" not in {r["type"] for r in R}


def test_legacy_dem_still_substitutes(monkeypatch):
    monkeypatch.setattr("cloak.substitute.walk_risk", lambda *args, **kwargs: 1.0)
    text = "The patient is German."
    span = Span(text.index("German"), text.index("German") + len("German"),
                "German", "DEM", 0.99, "stub")
    doc_p, R = substitute(text, [span])
    assert "<DEM_1>" in doc_p
    assert R[0]["type"] == "DEM"


def test_lowercase_person_role_does_not_fallback_to_dem(monkeypatch):
    monkeypatch.setattr("cloak.substitute._is_role_phrase", lambda text: True)
    monkeypatch.setattr("cloak.substitute.walk_risk", lambda *args, **kwargs: 1.0)
    text = "The nurse called."
    span = Span(text.index("nurse"), text.index("nurse") + len("nurse"),
                "nurse", "PERSON", 0.99, "stub")
    doc_p, R = substitute(text, [span])
    assert "<PROFESSION_1>" in doc_p
    assert R[0]["type"] == "profession"


def test_fine_probe_pool_build_and_walk_risk(monkeypatch, tmp_path):
    art = {
        "clinical": {
            "d1": {"tau_walk": ["", [{"type": "health-condition", "surface": f"cond{i}"}
                                     for i in range(MIN_POOL + 1)]]},
            "d2": {"tau_walk": ["", [{"type": "profession", "surface": "journalist"}]]},
        }
    }
    monkeypatch.setattr(bpd, "load_artifact", lambda: art)
    out = tmp_path / "probe_distractors.json"
    monkeypatch.setattr(bpd, "OUT", out)
    bpd.main()
    pools = json.loads(out.read_text())
    assert "health-condition" in pools
    assert "DEM" not in pools

    monkeypatch.setattr("cloak.probe._pools", pools)
    monkeypatch.setattr("cloak.probe._logp_continuations", lambda prefix, conts: [1.0] + [0.0] * (len(conts) - 1))
    risk = walk_risk("Patient has a chronic condition.", "cond0", "a chronic condition",
                     "health-condition")
    assert 0.0 <= risk < 1.0


def test_ranker_assembles_fine_placeholder_and_seeds_existing_counter():
    text = "prior diabetes improved"
    R_walk = [{"surface": "diabetes", "type": "health-condition", "action": "generalize",
               "replacement": "chronic condition", "start": 6, "end": 14,
               "lattice": ["chronic condition"]},
              {"surface": "prior", "type": "health-condition", "action": "placeholder",
               "replacement": "<HEALTH_CONDITION_3>", "start": 0, "end": 5}]
    spans = [{"surface": "diabetes", "type": "health-condition", "start": 6,
              "actions": [{"mode": "placeholder", "fill": None, "p6": 0.0,
                           "walk_risk": 0.0}]}]
    doc_p, R = tr.assemble(text, R_walk, spans,
                           {"diabetes": spans[0]["actions"][0]})
    assert "<HEALTH_CONDITION_4>" in doc_p
    assert R[0]["type"] == "health-condition"


def test_extract_accepts_fine_placeholders_and_type_sanity():
    out, stats = invert(
        "Patient has <HEALTH_CONDITION_1>. Stray <MARITAL_STATUS_2> remains.",
        [{"action": "placeholder", "surface": "diabetes", "type": "health-condition",
          "replacement": "<HEALTH_CONDITION_1>"}],
    )
    assert out.startswith("Patient has diabetes.")
    assert stats["ph_residue"] == 1
    assert _type_sane("health-condition", "chronic condition", "chronic condition")
    assert not _type_sane("gender", "<GENDER_1>", "female")
