"""Ranker-v2 routing of frozen utility component vectors to policy decisions."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence


def _loo_advantages(values: Sequence[float]) -> list[float]:
    if len(values) < 2:
        raise ValueError("provisional utility credit requires at least two rollouts")
    total = sum(values)
    return [value - (total - value) / (len(values) - 1) for value in values]


def _weighted_scores(
    vectors: Sequence[Mapping[str, float]],
    assertion_ids: set[str],
    weights: Mapping[str, float],
    denominator: float,
) -> list[float]:
    if not assertion_ids:
        return [0.0] * len(vectors)
    return [
        sum(weights[assertion_id] * float(vector[assertion_id])
            for assertion_id in assertion_ids) / denominator
        for vector in vectors
    ]


def provisional_advantages(
    component_vectors: Sequence[Mapping[str, float]],
    artifact: Mapping,
    *,
    doc_id: str,
) -> list[dict[str, float]]:
    """Compute ranker-v2 linked/global/fallback LOO advantages for one document group."""
    document = artifact["documents"][doc_id]
    denominator = float(document["utility_weight_denominator"])
    if denominator <= 0:
        raise ValueError("utility_weight_denominator must be positive")
    decisions = [str(value) for value in document["controlled_decision_ids"]]
    occurrence_to_decision = {
        str(key): str(value)
        for key, value in document.get("occurrence_to_decision", {}).items()
    }
    assertions = {
        str(assertion_id): row
        for assertion_id, row in artifact.get("assertions", {}).items()
        if row.get("doc_id") == doc_id and row.get("status", "accepted") == "accepted"
    }
    weights = {
        assertion_id: float(row["weight"])
        for assertion_id, row in assertions.items()
    }
    linked_by_decision: dict[str, set[str]] = defaultdict(set)
    global_ids = set()
    for assertion_id, row in assertions.items():
        occurrence_ids = [str(value) for value in row.get("occurrence_ids", [])]
        if row.get("scope") == "global" or not occurrence_ids:
            global_ids.add(assertion_id)
            continue
        for decision_id in {
            occurrence_to_decision[occurrence_id]
            for occurrence_id in occurrence_ids
        }:
            linked_by_decision[decision_id].add(assertion_id)

    global_advantages = _loo_advantages(_weighted_scores(
        component_vectors, global_ids, weights, denominator
    ))
    document_advantages = _loo_advantages(_weighted_scores(
        component_vectors, set(assertions), weights, denominator
    ))
    linked_advantages = {
        decision_id: _loo_advantages(_weighted_scores(
            component_vectors, assertion_ids, weights, denominator
        ))
        for decision_id, assertion_ids in linked_by_decision.items()
    }

    result = []
    for rollout_index in range(len(component_vectors)):
        row = {}
        for decision_id in decisions:
            if decision_id in linked_by_decision:
                row[decision_id] = (
                    linked_advantages[decision_id][rollout_index]
                    + global_advantages[rollout_index]
                )
            else:
                row[decision_id] = document_advantages[rollout_index]
        result.append(row)
    return result
