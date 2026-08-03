"""Fixed-denominator structured utility credit for ranker-v2 rollouts."""
from __future__ import annotations

import math

from cloak.qa.scoring import assertion_reward_role
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class DocumentUtilityCredit:
    document_utility: tuple[float, ...]
    linked_utility: Mapping[str, tuple[float, ...]]
    residual_utility: tuple[float, ...]
    provisional_advantage: Mapping[tuple[int, str], float]
    route: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "linked_utility",
            MappingProxyType({
                str(decision_id): tuple(values)
                for decision_id, values in self.linked_utility.items()
            }),
        )
        object.__setattr__(
            self,
            "provisional_advantage",
            MappingProxyType(dict(self.provisional_advantage)),
        )
        object.__setattr__(self, "route", MappingProxyType(dict(self.route)))


@dataclass(frozen=True)
class _Partitions:
    denominator: float
    policy_decision_ids: tuple[str, ...]
    assertions: Mapping[str, Mapping]
    weights: Mapping[str, float]
    linked_by_decision: Mapping[str, frozenset[str]]
    residual_ids: frozenset[str]


def _partitions(artifact: Mapping, doc_id: str) -> _Partitions:
    if artifact.get("artifact_version") != "utility-assertions-v2":
        raise ValueError("structured utility credit requires utility-assertions-v2")
    documents = artifact.get("documents")
    if not isinstance(documents, Mapping) or doc_id not in documents:
        raise ValueError(f"utility artifact lacks document {doc_id!r}")
    document = documents[doc_id]
    if not isinstance(document, Mapping):
        raise ValueError(f"utility document {doc_id!r} must be a mapping")

    denominator = document.get("utility_weight_denominator")
    if (
        isinstance(denominator, bool)
        or not isinstance(denominator, int | float)
        or not math.isfinite(float(denominator))
        or float(denominator) <= 0.0
    ):
        raise ValueError("utility_weight_denominator must be finite and positive")

    raw_decisions = document.get("policy_decision_ids")
    if (
        not isinstance(raw_decisions, list | tuple)
        or any(not isinstance(value, str) or not value for value in raw_decisions)
        or len(raw_decisions) != len(set(raw_decisions))
    ):
        raise ValueError("policy_decision_ids must be unique non-empty strings")
    policy_decision_ids = tuple(raw_decisions)
    policy_decision_set = set(policy_decision_ids)

    raw_assertion_ids = document.get("assertion_ids")
    if (
        not isinstance(raw_assertion_ids, list | tuple)
        or any(not isinstance(value, str) or not value for value in raw_assertion_ids)
        or len(raw_assertion_ids) != len(set(raw_assertion_ids))
    ):
        raise ValueError("document assertion_ids must be unique non-empty strings")
    all_assertions = artifact.get("assertions")
    if not isinstance(all_assertions, Mapping):
        raise ValueError("utility artifact assertions must be a mapping")

    assertions: dict[str, Mapping] = {}
    weights: dict[str, float] = {}
    linked_by_decision: dict[str, set[str]] = {}
    for assertion_id in raw_assertion_ids:
        row = all_assertions.get(assertion_id)
        if not isinstance(row, Mapping):
            raise ValueError(f"utility artifact lacks assertion {assertion_id!r}")
        if row.get("assertion_id") != assertion_id or row.get("doc_id") != doc_id:
            raise ValueError(f"utility assertion {assertion_id!r} binding mismatch")
        if row.get("status", "accepted") != "accepted":
            raise ValueError(f"document lists non-accepted assertion {assertion_id!r}")
        weight = row.get("weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, int | float)
            or not math.isfinite(float(weight))
            or float(weight) < 0.0
        ):
            raise ValueError(f"utility assertion {assertion_id!r} has invalid weight")

        routing = row.get("credit_routing")
        raw_dependencies = row.get("policy_dependency_decision_ids")
        if not isinstance(raw_dependencies, list | tuple) or any(
            not isinstance(value, str) or not value for value in raw_dependencies
        ):
            raise ValueError(
                f"utility assertion {assertion_id!r} has invalid policy dependencies"
            )
        dependencies = set(raw_dependencies)
        unknown = sorted(dependencies - policy_decision_set)
        if unknown:
            raise ValueError(
                f"utility assertion {assertion_id!r} names non-policy decisions: {unknown}"
            )
        if routing == "linked":
            if not dependencies:
                raise ValueError(
                    f"linked utility assertion {assertion_id!r} has no policy dependency"
                )
        elif routing == "residual":
            if dependencies:
                raise ValueError(
                    f"residual utility assertion {assertion_id!r} has policy dependencies"
                )
        else:
            raise ValueError(
                f"utility assertion {assertion_id!r} has invalid credit_routing"
            )
        if assertion_reward_role(row, document) == "monitoring":
            # Monitoring mass (no policy dependency, or a gold-exactness contract)
            # leaves the reward aggregate entirely under qa-utility-runtime-v2;
            # it is reported by the scorer, never credited — and never registered
            # as a decision's linked assertion (a monitoring-only decision routes
            # as "document" credit downstream).
            continue
        if routing == "linked":
            for decision_id in dependencies:
                linked_by_decision.setdefault(decision_id, set()).add(assertion_id)
        assertions[assertion_id] = row
        weights[assertion_id] = float(weight)

    policy_weight = sum(weights.values())
    if policy_weight <= 0.0:
        raise ValueError(f"document {doc_id!r} has no policy reward mass")

    return _Partitions(
        denominator=float(policy_weight),
        policy_decision_ids=policy_decision_ids,
        assertions=MappingProxyType(assertions),
        weights=MappingProxyType(weights),
        linked_by_decision=MappingProxyType({
            decision_id: frozenset(assertion_ids)
            for decision_id, assertion_ids in linked_by_decision.items()
        }),
        residual_ids=frozenset(),
    )


def _validate_vectors(
    component_vectors: Sequence[Mapping[str, float]], assertion_ids: frozenset[str],
) -> None:
    for rollout_index, vector in enumerate(component_vectors):
        if not isinstance(vector, Mapping):
            raise ValueError(f"rollout {rollout_index} component vector must be a mapping")
        missing = sorted(assertion_ids - set(vector))
        if missing:
            raise ValueError(
                f"rollout {rollout_index} missing assertion scores: {missing}"
            )
        invalid = sorted(
            assertion_id for assertion_id in assertion_ids
            if (
                isinstance(vector[assertion_id], bool)
                or not isinstance(vector[assertion_id], int | float)
                or not math.isfinite(float(vector[assertion_id]))
                or not 0.0 <= float(vector[assertion_id]) <= 1.0
            )
        )
        if invalid:
            raise ValueError(
                f"rollout {rollout_index} has invalid assertion scores: {invalid}"
            )


def _weighted_scores(
    component_vectors: Sequence[Mapping[str, float]],
    assertion_ids: frozenset[str],
    weights: Mapping[str, float],
    denominator: float,
) -> tuple[float, ...]:
    return tuple(
        sum(
            weights[assertion_id] * float(vector[assertion_id])
            for assertion_id in assertion_ids
        ) / denominator
        for vector in component_vectors
    )


def _loo_advantages(values: Sequence[float]) -> tuple[float, ...]:
    rollout_count = len(values)
    if rollout_count < 2:
        raise ValueError("provisional utility credit requires at least two rollouts")
    total = sum(values)
    return tuple(
        value - (total - value) / (rollout_count - 1)
        for value in values
    )


def document_utility(
    component_scores: Mapping[str, float], artifact: Mapping, doc_id: str,
) -> float:
    """Aggregate policy-role assertions over the fixed policy weight mass."""
    partitions = _partitions(artifact, doc_id)
    assertion_ids = frozenset(partitions.assertions)
    try:
        _validate_vectors([component_scores], assertion_ids)
    except ValueError as error:
        message = str(error).replace("rollout 0 ", "")
        raise ValueError(message) from error
    return _weighted_scores(
        [component_scores], assertion_ids, partitions.weights, partitions.denominator,
    )[0]


def decision_delta_utility(
    selected_scores: Mapping[str, float],
    alternative_scores: Mapping[str, float],
    artifact: Mapping,
    doc_id: str,
    decision_id: str,
) -> tuple[float, float | None]:
    """Per-decision utility difference restricted to the decision's own assertions.

    Returns (document_normalized, linked_normalized). Both difference ONLY the
    assertions the artifact declares depend on this decision; assertions outside
    that set cannot be affected by it, so including them adds variance with no
    signal (measured: 60% of pairs move such assertions, median 0.019, and 65%
    of provably-tied pairs report nonzero — docs/issues/counterfactual-delta-u-measurement.md).

    The two normalizations serve different purposes and are not interchangeable:
    dividing by the document weight denominator keeps the value in the OBJECTIVE's
    units, comparable across documents; dividing by the decision's own linked mass
    gives the fraction of THIS decision's obligations the change broke, in
    [-1, 1], which is the right unit for comparing against a resolution threshold
    (a decision owning 10% of the mass otherwise cannot clear a floor expressed in
    whole-document units). linked_normalized is None when the decision has no
    linked assertions — a structurally derivable tie, where no feedback channel
    can separate its actions.
    """
    partitions = _partitions(artifact, doc_id)
    linked = partitions.linked_by_decision.get(decision_id, frozenset())
    numerator = sum(
        partitions.weights[key] * (selected_scores[key] - alternative_scores[key])
        for key in linked
        if key in selected_scores and key in alternative_scores
    )
    linked_mass = sum(partitions.weights[key] for key in linked)
    return (
        numerator / partitions.denominator,
        (numerator / linked_mass) if linked_mass > 0.0 else None,
    )


def assertion_weights(artifact: Mapping, doc_id: str) -> Mapping[str, float]:
    """Credited assertion weights for a document (monitoring mass excluded)."""
    return _partitions(artifact, doc_id).weights


def document_denominator(artifact: Mapping, doc_id: str) -> float:
    """The fixed reward denominator: the sum of credited assertion weights."""
    return _partitions(artifact, doc_id).denominator


def excerpt_changed_assertions(
    artifact: Mapping,
    doc_id: str,
    selected_doc_p: str,
    alternative_doc_p: str,
) -> frozenset[str]:
    """Context assertions whose reader input differs between two rewrites.

    A CONTEXT assertion is scored by the reader on an excerpt of `doc_p`. If the
    excerpt is byte-identical the reader (temperature 0) returns the identical
    answer, so only assertions whose excerpt CHANGED can have moved. This is
    pair-local causal support, deliberately NOT a dependency declaration:
    an assertion unchanged in this pair may move in another, and rewriting
    `policy_dependency_decision_ids` from pair-local evidence would add
    leave-one-out variance and invalidate cache identities.

    Measured motivation: declarations miss real influence on 46 (doc, decision,
    assertion) triples, leaving 305 pairs recorded as exact ties despite a
    genuine reader-score change (docs/issues/qa-dependency-underdeclaration.md).
    DELIVERED assertions have no excerpt, so no analogous local test exists;
    their global movement stays unattributed spillover.
    """
    from cloak.qa.scoring import reader_excerpt

    partitions = _partitions(artifact, doc_id)
    changed = {
        assertion_id
        for assertion_id, row in partitions.assertions.items()
        if row.get("family") == "context"
        and reader_excerpt(selected_doc_p, row.get("evidence") or {})
        != reader_excerpt(alternative_doc_p, row.get("evidence") or {})
    }
    return frozenset(changed)


def attributed_assertions(
    artifact: Mapping, doc_id: str, decision_id: str,
) -> frozenset[str]:
    """The assertion set a counterfactual at this decision is credited for.

    A decision that some assertion declares a dependency on is credited for its
    OWN declared assertions — matching `provisional_credit`, which gives such a
    decision its linked advantage rather than the whole document's. A decision
    that no assertion claims keeps whole-document credit, because that is the
    provisional fallback it would otherwise receive and `hybrid_utility_loss`
    substitutes rather than adds (route-consistency, 2026-08-03).
    """
    partitions = _partitions(artifact, doc_id)
    linked = partitions.linked_by_decision.get(decision_id, frozenset())
    return linked if linked else frozenset(partitions.weights)


def attributed_delta_utility(
    selected_scores: Mapping[str, float],
    alternative_scores: Mapping[str, float],
    artifact: Mapping,
    doc_id: str,
    decision_id: str,
    extra_assertions: Iterable[str] = (),
) -> tuple[float, frozenset[str]]:
    """Counterfactual delta over the decision's attributed set, in document units.

    Returns (delta, attributed_set). `extra_assertions` admits pair-local
    evidence of influence that the static declaration misses; it is a union, so
    it can only add movement, never dilute (the denominator is the fixed
    document weight denominator).
    """
    partitions = _partitions(artifact, doc_id)
    attributed = attributed_assertions(artifact, doc_id, decision_id) | (
        frozenset(extra_assertions) & frozenset(partitions.weights)
    )
    delta = sum(
        partitions.weights[key] * (selected_scores[key] - alternative_scores[key])
        for key in attributed
        if key in selected_scores and key in alternative_scores
    ) / partitions.denominator
    return delta, attributed


def subset_advantages(
    component_vectors: Sequence[Mapping[str, float]],
    artifact: Mapping,
    doc_id: str,
    assertion_ids: Iterable[str],
) -> tuple[float, ...]:
    """Leave-one-out advantages over an arbitrary assertion subset.

    `_weighted_scores` is a sum over assertions and `_loo_advantages` is linear,
    so advantages are ADDITIVE over disjoint assertion sets. That is what lets
    scope-matched substitution credit a measured component exactly while leaving
    the unmeasured complement to its provisional estimate, with no double count
    and no gap: A(S) + A(Q\\S) == A(Q).
    """
    partitions = _partitions(artifact, doc_id)
    ids = frozenset(assertion_ids) & frozenset(partitions.weights)
    return _loo_advantages(_weighted_scores(
        component_vectors, ids, partitions.weights, partitions.denominator,
    ))


def provisional_credit(
    component_vectors: Sequence[Mapping[str, float]], artifact: Mapping, doc_id: str,
) -> DocumentUtilityCredit:
    """Partition v2 assertions and compute unnormalized leave-one-out credit."""
    if len(component_vectors) < 2:
        raise ValueError("provisional utility credit requires at least two rollouts")
    partitions = _partitions(artifact, doc_id)
    all_ids = frozenset(partitions.assertions)
    _validate_vectors(component_vectors, all_ids)

    document_scores = _weighted_scores(
        component_vectors, all_ids, partitions.weights, partitions.denominator,
    )
    residual_scores = _weighted_scores(
        component_vectors,
        partitions.residual_ids,
        partitions.weights,
        partitions.denominator,
    )
    linked_scores = {
        decision_id: _weighted_scores(
            component_vectors,
            assertion_ids,
            partitions.weights,
            partitions.denominator,
        )
        for decision_id, assertion_ids in partitions.linked_by_decision.items()
    }

    document_advantage = _loo_advantages(document_scores)
    residual_advantage = _loo_advantages(residual_scores)
    linked_advantage = {
        decision_id: _loo_advantages(values)
        for decision_id, values in linked_scores.items()
    }
    route = {
        decision_id: (
            "linked" if decision_id in partitions.linked_by_decision else "document"
        )
        for decision_id in partitions.policy_decision_ids
    }
    provisional_advantage = {
        (rollout_index, decision_id): (
            linked_advantage[decision_id][rollout_index]
            + residual_advantage[rollout_index]
            if route[decision_id] == "linked"
            else document_advantage[rollout_index]
        )
        for rollout_index in range(len(component_vectors))
        for decision_id in partitions.policy_decision_ids
    }
    return DocumentUtilityCredit(
        document_utility=document_scores,
        linked_utility=linked_scores,
        residual_utility=residual_scores,
        provisional_advantage=provisional_advantage,
        route=route,
    )
