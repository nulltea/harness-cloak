"""Fixed-denominator structured utility credit for utility-assertions-v2."""
import inspect
from dataclasses import FrozenInstanceError, fields

import pytest


def _artifact():
    return {
        "artifact_version": "utility-assertions-v2",
        "documents": {"d1": {
            "assertion_ids": ["policy", "fixed", "mixed", "global"],
            "utility_weight_denominator": 1.0,
            "policy_decision_ids": ["p1", "p2", "p3"],
            "fixed_decision_ids": ["f1"],
            "uncovered_policy_decision_ids": ["p3"],
            "occurrence_to_decision": {
                "o-policy-1": "p1", "o-policy-2": "p2", "o-fixed": "f1",
            },
        }},
        "assertions": {
            "policy": {
                "assertion_id": "policy", "doc_id": "d1", "weight": 0.2,
                # Deliberately contradictory legacy fields: routing must use only v2 fields.
                "scope": "global", "occurrence_ids": ["o-policy-1"],
                "credit_routing": "linked",
                "policy_dependency_decision_ids": ["p1"],
            },
            "fixed": {
                "assertion_id": "fixed", "doc_id": "d1", "weight": 0.2,
                "scope": "linked", "occurrence_ids": ["o-fixed"],
                "credit_routing": "residual",
                "policy_dependency_decision_ids": [],
            },
            "mixed": {
                "assertion_id": "mixed", "doc_id": "d1", "weight": 0.3,
                "scope": "linked",
                "occurrence_ids": [
                    "o-policy-1", "o-policy-1", "o-fixed", "o-policy-2",
                ],
                "credit_routing": "linked",
                "policy_dependency_decision_ids": ["p1", "p2"],
            },
            "global": {
                "assertion_id": "global", "doc_id": "d1", "weight": 0.3,
                "scope": "global", "occurrence_ids": [],
                "credit_routing": "residual",
                "policy_dependency_decision_ids": [],
            },
        },
    }


def _vectors():
    return [
        {"policy": 1.0, "fixed": 1.0, "mixed": 0.0, "global": 0.0},
        {"policy": 0.0, "fixed": 0.0, "mixed": 1.0, "global": 1.0},
    ]


def test_public_credit_contract_is_exact_frozen_and_immutable():
    from cloak.reward.utility_credit import (
        DocumentUtilityCredit,
        document_utility,
        provisional_credit,
    )

    assert [field.name for field in fields(DocumentUtilityCredit)] == [
        "document_utility", "linked_utility", "residual_utility",
        "provisional_advantage", "route",
    ]
    credit = DocumentUtilityCredit(
        document_utility=(0.0, 1.0), linked_utility={},
        residual_utility=(0.0, 0.0), provisional_advantage={}, route={},
    )
    with pytest.raises(FrozenInstanceError):
        credit.document_utility = ()
    with pytest.raises(TypeError):
        credit.route["p1"] = "linked"
    with pytest.raises(TypeError):
        credit.linked_utility["p1"] = (1.0,)
    with pytest.raises(TypeError):
        credit.provisional_advantage[(0, "p1")] = 1.0
    assert list(inspect.signature(document_utility).parameters) == [
        "component_scores", "artifact", "doc_id",
    ]
    assert list(inspect.signature(provisional_credit).parameters) == [
        "component_vectors", "artifact", "doc_id",
    ]


def test_partitions_policy_fixed_mixed_global_and_uncovered_credit():
    from cloak.reward.utility_credit import provisional_credit

    credit = provisional_credit(_vectors(), _artifact(), "d1")

    assert credit.document_utility == pytest.approx((0.4, 0.6))
    assert credit.linked_utility == {
        "p1": pytest.approx((0.2, 0.3)),
        "p2": pytest.approx((0.0, 0.3)),
    }
    assert credit.residual_utility == pytest.approx((0.2, 0.3))
    assert credit.route == {"p1": "linked", "p2": "linked", "p3": "document"}
    assert credit.provisional_advantage == pytest.approx({
        (0, "p1"): -0.2, (1, "p1"): 0.2,
        (0, "p2"): -0.4, (1, "p2"): 0.4,
        (0, "p3"): -0.2, (1, "p3"): 0.2,
    })


def test_scope_occurrences_and_removed_controlled_ids_cannot_change_routes():
    from cloak.reward.utility_credit import provisional_credit

    artifact = _artifact()
    artifact["documents"]["d1"]["controlled_decision_ids"] = ["wrong"]
    for row in artifact["assertions"].values():
        row["scope"] = "global" if row["scope"] == "linked" else "linked"
        row["occurrence_ids"] = ["unknown", "unknown"]

    credit = provisional_credit(_vectors(), artifact, "d1")

    assert credit.route == {"p1": "linked", "p2": "linked", "p3": "document"}
    assert set(decision_id for _rollout, decision_id in credit.provisional_advantage) == {
        "p1", "p2", "p3",
    }


def test_mixed_hyperedge_routes_once_to_each_unique_policy_dependency():
    from cloak.reward.utility_credit import provisional_credit

    artifact = _artifact()
    artifact["assertions"]["mixed"]["policy_dependency_decision_ids"] = [
        "p1", "p1", "p2",
    ]
    credit = provisional_credit(_vectors(), artifact, "d1")

    assert credit.linked_utility["p1"] == pytest.approx((0.2, 0.3))
    assert credit.linked_utility["p2"] == pytest.approx((0.0, 0.3))


def test_missing_family_uses_real_stored_fixed_denominator():
    from cloak.reward.utility_credit import document_utility, provisional_credit

    # Migrated v16 semantics: delivered budget 0.4, absent context budget 0.6,
    # stored denominator remains 1.0 rather than being recomputed to 0.4.
    artifact = {
        "artifact_version": "utility-assertions-v2",
        "documents": {"partial": {
            "assertion_ids": ["delivered"],
            "utility_weight_denominator": 1.0,
            "policy_decision_ids": ["p1"],
            "fixed_decision_ids": [],
            "uncovered_policy_decision_ids": [],
            "missing_family_budgets": {"context": 0.6},
            "present_family_budgets": {"delivered": 0.4},
        }},
        "assertions": {"delivered": {
            "assertion_id": "delivered", "doc_id": "partial", "weight": 0.4,
            "credit_routing": "linked",
            "policy_dependency_decision_ids": ["p1"],
        }},
    }

    assert document_utility({"delivered": 1.0}, artifact, "partial") == 0.4
    credit = provisional_credit(
        [{"delivered": 1.0}, {"delivered": 0.0}], artifact, "partial",
    )
    assert credit.document_utility == pytest.approx((0.4, 0.0))
    assert credit.provisional_advantage == pytest.approx({
        (0, "p1"): 0.4, (1, "p1"): -0.4,
    })


def test_missing_assertion_scores_report_all_missing_ids():
    from cloak.reward.utility_credit import document_utility, provisional_credit

    with pytest.raises(ValueError, match=r"missing assertion scores.*fixed.*global.*mixed"):
        document_utility({"policy": 1.0}, _artifact(), "d1")
    with pytest.raises(
        ValueError, match=r"rollout 1 missing assertion scores.*fixed.*mixed",
    ):
        provisional_credit(
            [_vectors()[0], {"policy": 0.0, "global": 1.0}], _artifact(), "d1",
        )


def test_tied_components_produce_zero_for_every_rollout_decision_pair():
    from cloak.reward.utility_credit import provisional_credit

    tied = [_vectors()[0], dict(_vectors()[0]), dict(_vectors()[0])]
    credit = provisional_credit(tied, _artifact(), "d1")

    assert set(credit.provisional_advantage) == {
        (rollout, decision)
        for rollout in range(3)
        for decision in ("p1", "p2", "p3")
    }
    assert all(value == pytest.approx(0.0)
               for value in credit.provisional_advantage.values())


def test_three_rollout_loo_is_not_standard_deviation_normalized():
    from cloak.reward.utility_credit import provisional_credit

    artifact = {
        "artifact_version": "utility-assertions-v2",
        "documents": {"d": {
            "assertion_ids": ["q"], "utility_weight_denominator": 1.0,
            "policy_decision_ids": ["p"], "fixed_decision_ids": [],
            "uncovered_policy_decision_ids": [],
        }},
        "assertions": {"q": {
            "assertion_id": "q", "doc_id": "d", "weight": 0.4,
            "credit_routing": "linked", "policy_dependency_decision_ids": ["p"],
        }},
    }
    credit = provisional_credit(
        [{"q": 1.0}, {"q": 0.5}, {"q": 0.0}], artifact, "d",
    )

    assert credit.provisional_advantage == pytest.approx({
        (0, "p"): 0.3, (1, "p"): 0.0, (2, "p"): -0.3,
    })


def test_provisional_credit_requires_two_rollouts():
    from cloak.reward.utility_credit import provisional_credit

    with pytest.raises(ValueError, match="at least two rollouts"):
        provisional_credit([_vectors()[0]], _artifact(), "d1")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda artifact: artifact.update(artifact_version="utility-assertions-v1"),
         "utility-assertions-v2"),
        (lambda artifact: artifact["assertions"]["policy"].update(
            policy_dependency_decision_ids=[]), "has no policy dependency"),
        (lambda artifact: artifact["assertions"]["fixed"].update(
            policy_dependency_decision_ids=["p1"]), "has policy dependencies"),
        (lambda artifact: artifact["assertions"]["policy"].update(
            policy_dependency_decision_ids=["unknown"]), "non-policy decisions"),
    ],
)
def test_invalid_v2_routing_metadata_fails_closed(mutation, message):
    from cloak.reward.utility_credit import provisional_credit

    artifact = _artifact()
    mutation(artifact)
    with pytest.raises(ValueError, match=message):
        provisional_credit(_vectors(), artifact, "d1")
