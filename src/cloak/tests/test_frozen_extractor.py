import copy

import cloak.frozen_extractor as fx
from cloak.extract import invert


def test_extractor_pins_include_frozen_model_ids_and_thresholds():
    assert fx.EXTRACTOR_PINS == {
        "models": {
            "encoder": "BAAI/bge-small-en-v1.5",
            "nli": "cross-encoder/nli-deberta-v3-small",
            "mlm": "roberta-base",
        },
        "thresholds": {
            "SIM_MIN": 0.55,
            "ASSIGN_MARGIN": 0.05,
            "PRIOR_WEIGHT": 0.15,
            "NLI_ENTAIL": 0.80,
            "TYPE_ENTAIL": 0.70,
            "PLL_MIN_DELTA": -6.0,
            "EPS_MARGIN": 0.02,
            "CHUNK_MAX_WORDS": 6,
        },
        "type_hypotheses": {},
        "ladder_semver": "0.1.0",
    }


def test_extract_deterministic_only_matches_invert_for_placeholder():
    R = [
        {"action": "placeholder", "surface": "Ada Lovelace", "replacement": "<PERSON_1>",
         "type": "PERSON"}
    ]
    out_p = "<PERSON_1> filed the appeal."

    expected_text, expected_stats = invert(out_p, R)
    text, stats = fx.extract(None, R, out_p)

    assert text == expected_text
    for key, value in expected_stats.items():
        assert stats[key] == value
    assert stats["entries"] == []
    assert stats["extractor_version"] == fx.extractor_version()


def test_extract_deterministic_only_matches_invert_for_exact_fill():
    R = [
        {"action": "generalize", "surface": "Hamilton County Court",
         "replacement": "a county court", "type": "ORG"}
    ]
    out_p = "The order came from a county court."

    expected_text, expected_stats = invert(out_p, R)
    text, stats = fx.extract("The order came from a county court.", R, out_p)

    assert text == expected_text
    for key, value in expected_stats.items():
        assert stats[key] == value
    assert stats["entries"] == []
    assert stats["extractor_version"] == fx.extractor_version()


def test_extract_records_residue_entries_as_no_model_abstains(monkeypatch):
    residue = [
        {"surface": "Boston", "replacement": "a city in Massachusetts", "type": "LOC"},
        {"surface": "January 13th 1982", "replacement": "the early 1980s",
         "type": "DATETIME"},
    ]

    def fake_rule_prepass(out_p, R, *, semantic):
        assert semantic is True
        return out_p, {"gen_absent": 0, "ph_residue": 0}, list(residue)

    monkeypatch.setattr(fx, "_rule_prepass", fake_rule_prepass)

    text, stats = fx.extract(None, [], "Nothing restorable here.", models=None)

    assert text == "Nothing restorable here."
    assert stats["gen_absent"] == 2
    assert stats["entries"] == [
        {"surface": "Boston", "type": "LOC", "outcome": "abstained",
         "reason": "no-models"},
        {"surface": "January 13th 1982", "type": "DATETIME", "outcome": "abstained",
         "reason": "no-models"},
    ]


def test_extractor_version_is_stable_and_changes_when_pin_changes(monkeypatch):
    first = fx.extractor_version()
    second = fx.extractor_version()
    changed = copy.deepcopy(fx.EXTRACTOR_PINS)
    changed["thresholds"]["SIM_MIN"] = 0.56

    monkeypatch.setattr(fx, "EXTRACTOR_PINS", changed)

    assert first == second
    assert first.startswith("fx-")
    assert len(first) == 15
    assert fx.extractor_version() != first
