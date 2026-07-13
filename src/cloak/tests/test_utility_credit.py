import pytest

from cloak.train.utility_credit import provisional_advantages


def _artifact():
    return {
        "documents": {"d1": {
            "utility_weight_denominator": 1.0,
            "controlled_decision_ids": ["dec1", "dec2", "dec3"],
            "occurrence_to_decision": {
                "o1": "dec1",
                "o1-repeat": "dec1",
                "o2": "dec2",
            },
        }},
        "assertions": {
            "linked1": {
                "assertion_id": "linked1", "doc_id": "d1", "scope": "linked",
                "occurrence_ids": ["o1", "o1-repeat"], "weight": 0.3,
            },
            "linked12": {
                "assertion_id": "linked12", "doc_id": "d1", "scope": "linked",
                "occurrence_ids": ["o1", "o2"], "weight": 0.2,
            },
            "global": {
                "assertion_id": "global", "doc_id": "d1", "scope": "global",
                "occurrence_ids": [], "weight": 0.5,
            },
        },
    }


def test_provisional_credit_partitions_linked_global_and_uncovered_fallback():
    vectors = [
        {"linked1": 1.0, "linked12": 0.0, "global": 1.0},
        {"linked1": 0.0, "linked12": 1.0, "global": 0.0},
    ]

    advantages = provisional_advantages(vectors, _artifact(), doc_id="d1")

    assert advantages == [
        pytest.approx({"dec1": 0.6, "dec2": 0.3, "dec3": 0.6}),
        pytest.approx({"dec1": -0.6, "dec2": -0.3, "dec3": -0.6}),
    ]


def test_repeated_occurrences_do_not_duplicate_linked_component_mass():
    artifact = _artifact()
    artifact["assertions"].pop("linked12")
    artifact["assertions"].pop("global")
    vectors = [{"linked1": 1.0}, {"linked1": 0.0}]

    advantages = provisional_advantages(vectors, artifact, doc_id="d1")

    assert advantages[0]["dec1"] == pytest.approx(0.3)
    assert advantages[0]["dec2"] == pytest.approx(0.3)
    assert advantages[0]["dec3"] == pytest.approx(0.3)


def test_missing_reserved_family_mass_uses_fixed_document_denominator():
    artifact = {
        "documents": {"d1": {
            "utility_weight_denominator": 1.0,
            "controlled_decision_ids": ["dec1"],
            "occurrence_to_decision": {"o1": "dec1"},
        }},
        "assertions": {"delivered": {
            "assertion_id": "delivered", "doc_id": "d1", "scope": "linked",
            "occurrence_ids": ["o1"], "weight": 0.4,
        }},
    }

    advantages = provisional_advantages(
        [{"delivered": 1.0}, {"delivered": 0.0}], artifact, doc_id="d1"
    )

    assert advantages == [
        pytest.approx({"dec1": 0.4}),
        pytest.approx({"dec1": -0.4}),
    ]


def test_tied_components_produce_zero_advantage():
    vectors = [
        {"linked1": 1.0, "linked12": 1.0, "global": 1.0},
        {"linked1": 1.0, "linked12": 1.0, "global": 1.0},
    ]

    advantages = provisional_advantages(vectors, _artifact(), doc_id="d1")

    assert all(value == pytest.approx(0.0)
               for row in advantages for value in row.values())
