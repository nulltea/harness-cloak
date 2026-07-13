import pytest

import cloak.train.utility_credit as utility_credit
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


def _advantages(vectors, artifact):
    return provisional_advantages(
        vectors,
        artifact,
        artifact["documents"]["d1"]["occurrence_to_decision"],
        doc_id="d1",
    )


def test_document_utility_owns_weighted_component_aggregation():
    artifact = _artifact()

    utility = utility_credit.document_utility(
        {"linked1": 1.0, "linked12": 0.5, "global": 0.2},
        artifact,
        doc_id="d1",
    )

    assert utility == pytest.approx(0.5)


def test_document_utility_rejects_missing_component_scores():
    with pytest.raises(ValueError, match="lacks assertion 'global'"):
        utility_credit.document_utility(
            {"linked1": 1.0, "linked12": 0.5},
            _artifact(),
            doc_id="d1",
        )


def test_provisional_credit_routes_linked_global_and_uncovered_decisions():
    vectors = [
        {"linked1": 1.0, "linked12": 0.0, "global": 1.0},
        {"linked1": 0.0, "linked12": 1.0, "global": 0.0},
    ]

    advantages = _advantages(vectors, _artifact())

    assert advantages == pytest.approx({
        (0, "dec1"): 0.6,
        (0, "dec2"): 0.3,
        (0, "dec3"): 0.6,
        (1, "dec1"): -0.6,
        (1, "dec2"): -0.3,
        (1, "dec3"): -0.6,
    })


def test_repeated_occurrences_route_a_linked_assertion_once_per_decision():
    artifact = _artifact()
    artifact["assertions"].pop("linked12")
    artifact["assertions"].pop("global")

    advantages = _advantages([{"linked1": 1.0}, {"linked1": 0.0}], artifact)

    assert advantages == pytest.approx({
        (0, "dec1"): 0.3,
        (0, "dec2"): 0.3,
        (0, "dec3"): 0.3,
        (1, "dec1"): -0.3,
        (1, "dec2"): -0.3,
        (1, "dec3"): -0.3,
    })


def test_fixed_denominator_preserves_missing_reserved_family_mass():
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

    advantages = _advantages([{"delivered": 1.0}, {"delivered": 0.0}], artifact)

    assert advantages == pytest.approx({(0, "dec1"): 0.4, (1, "dec1"): -0.4})


def test_tied_components_produce_zero_advantage():
    vectors = [
        {"linked1": 1.0, "linked12": 1.0, "global": 1.0},
        {"linked1": 1.0, "linked12": 1.0, "global": 1.0},
    ]

    assert all(value == pytest.approx(0.0)
               for value in _advantages(vectors, _artifact()).values())


def test_linked_decision_never_adds_complete_document_advantage():
    artifact = _artifact()
    artifact["assertions"].pop("linked12")
    artifact["assertions"]["linked1"]["weight"] = 0.3
    artifact["assertions"]["global"]["weight"] = 0.2
    vectors = [
        {"linked1": 1.0, "global": 0.0},
        {"linked1": 0.0, "global": 1.0},
    ]

    advantages = _advantages(vectors, artifact)

    assert advantages[(0, "dec1")] == pytest.approx(0.1)
    assert advantages[(1, "dec1")] == pytest.approx(-0.1)


def test_unknown_linked_occurrence_is_rejected():
    artifact = _artifact()
    artifact["assertions"]["linked1"]["occurrence_ids"] = ["missing"]

    with pytest.raises(ValueError, match="unknown occurrence"):
        _advantages([
            {"linked1": 1.0, "linked12": 0.0, "global": 1.0},
            {"linked1": 0.0, "linked12": 1.0, "global": 0.0},
        ], artifact)
