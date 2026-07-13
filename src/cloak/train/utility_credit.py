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
    component_vectors: Sequence[Mapping[str, float]],
    assertion_ids: set[str],
    weights: Mapping[str, float],
    denominator: float,
) -> list[float]:
    if not assertion_ids:
        return [0.0] * len(component_vectors)
    try:
        return [
            sum(weights[assertion_id] * float(vector[assertion_id])
                for assertion_id in assertion_ids) / denominator
            for vector in component_vectors
        ]
    except KeyError as exc:
        raise ValueError(f"utility component vector lacks assertion {exc.args[0]!r}") from exc


def provisional_advantages(
    component_vectors: Sequence[Mapping[str, float]],
    artifact: Mapping,
    occurrence_to_decision: Mapping[str, str],
    *,
    doc_id: str,
) -> dict[tuple[int, str], float]:
    """Route linked/global/fallback utility RLOO credit for one document group."""
    if doc_id not in artifact.get("documents", {}):
        raise ValueError(f"utility artifact lacks document {doc_id!r}")
    document = artifact["documents"][doc_id]
    denominator = float(document["utility_weight_denominator"])
    if denominator <= 0:
        raise ValueError("utility_weight_denominator must be positive")

    decisions = [str(value) for value in document["controlled_decision_ids"]]
    if len(decisions) != len(set(decisions)):
        raise ValueError("controlled decision IDs must be unique")
    decision_set = set(decisions)
    occurrence_map = {str(occurrence_id): str(decision_id)
                      for occurrence_id, decision_id in occurrence_to_decision.items()}
    unknown_decisions = sorted(set(occurrence_map.values()) - decision_set)
    if unknown_decisions:
        raise ValueError(f"occurrence map names uncontrolled decisions: {unknown_decisions}")

    assertion_ids = document.get("assertion_ids")
    accepted = {
        str(assertion_id): row
        for assertion_id, row in artifact.get("assertions", {}).items()
        if row.get("doc_id") == doc_id and row.get("status", "accepted") == "accepted"
        and (assertion_ids is None or assertion_id in assertion_ids)
    }
    weights = {assertion_id: float(row["weight"])
               for assertion_id, row in accepted.items()}

    linked_by_decision: dict[str, set[str]] = defaultdict(set)
    global_ids: set[str] = set()
    for assertion_id, row in accepted.items():
        occurrence_ids = {str(value) for value in row.get("occurrence_ids", [])}
        if row.get("scope") == "global":
            if occurrence_ids:
                raise ValueError(f"global assertion {assertion_id!r} has occurrence links")
            global_ids.add(assertion_id)
            continue
        if row.get("scope") != "linked" or not occurrence_ids:
            raise ValueError(f"linked assertion {assertion_id!r} has no occurrence links")
        unknown_occurrences = sorted(occurrence_ids - set(occurrence_map))
        if unknown_occurrences:
            raise ValueError(
                f"linked assertion {assertion_id!r} has unknown occurrence links: "
                f"{unknown_occurrences}"
            )
        for decision_id in {occurrence_map[occurrence_id] for occurrence_id in occurrence_ids}:
            linked_by_decision[decision_id].add(assertion_id)

    global_advantages = _loo_advantages(_weighted_scores(
        component_vectors, global_ids, weights, denominator
    ))
    document_advantages = _loo_advantages(_weighted_scores(
        component_vectors, set(accepted), weights, denominator
    ))
    linked_advantages = {
        decision_id: _loo_advantages(_weighted_scores(
            component_vectors, assertion_ids, weights, denominator
        ))
        for decision_id, assertion_ids in linked_by_decision.items()
    }
    return {
        (rollout_index, decision_id): (
            linked_advantages[decision_id][rollout_index] + global_advantages[rollout_index]
            if decision_id in linked_by_decision
            else document_advantages[rollout_index]
        )
        for rollout_index in range(len(component_vectors))
        for decision_id in decisions
    }
