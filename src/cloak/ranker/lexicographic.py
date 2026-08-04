"""Exact (epsilon = 0) lexicographic selection over complete document action vectors.

Primary objective: measured document utility. Secondary: the frozen profile-count
score, optimised ONLY inside the primary's exact argmax set — the filter the
lexicographic-MORL result prescribes, not an additive bonus
(docs/plans/2026-08-04-epsilon-zero-lexicographic-gate.md §2).

Two properties are load-bearing:

* Membership in the optimal set is decided on an exact `Decimal` numerator
  reconstructed from the pinned component scores and weights. Float
  representation slack is not semantic slack: a vector that is 1e-12 worse is
  worse, and admitting it would silently reintroduce a tolerance band.
* Both selectors are totally ordered. Every tie is resolved by the stable vector
  key, never by input order, so the choice is reproducible across runs and cache
  orderings (Vamplew et al. 2024 on interference from arbitrary tie-breaking).

No policy, checkpoint, tower logit, alpha, or torch dependency: this module sees
measurements only.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from cloak.reward.utility_cache import utility_binding

VectorKey = tuple[tuple[str, str], ...]


def _validate_vector_key(vector_key: Any, context: str) -> VectorKey:  # noqa: ANN401
    if not isinstance(vector_key, tuple) or not vector_key:
        raise ValueError(f"{context} must be a non-empty tuple of (decision, action)")
    for pair in vector_key:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(not isinstance(value, str) or not value for value in pair)
        ):
            raise ValueError(f"{context} must hold non-empty (decision, action) pairs")
    decisions = [decision_id for decision_id, _action_id in vector_key]
    if len(set(decisions)) != len(decisions):
        raise ValueError(f"{context} repeats a decision id")
    return vector_key


def _validate_finite(value: Any, context: str) -> float:  # noqa: ANN401
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{context} must be finite")
    return float(value)


def _validate_key(value: Any, context: str) -> Decimal:  # noqa: ANN401
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{context} must be a finite Decimal")
    return value


@dataclass(frozen=True)
class LexicographicCandidate:
    """One complete, cached, legal document action vector with both objectives."""

    doc_id: str
    vector_key: VectorKey
    utility_key: Decimal
    utility: float
    privacy_key: Decimal
    privacy_score: float
    result_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.doc_id, str) or not self.doc_id:
            raise ValueError("candidate doc_id must be a non-empty string")
        if not isinstance(self.result_hash, str) or not self.result_hash:
            raise ValueError("candidate result_hash must be a non-empty string")
        _validate_vector_key(self.vector_key, "candidate vector_key")
        _validate_key(self.utility_key, "candidate utility_key")
        _validate_key(self.privacy_key, "candidate privacy_key")
        _validate_finite(self.utility, "candidate utility")
        _validate_finite(self.privacy_score, "candidate privacy_score")


@dataclass(frozen=True)
class LexicographicSelection:
    selected: LexicographicCandidate
    feasible_count: int
    optimal_utility_key: Decimal
    feasible_privacy_min: Decimal
    feasible_privacy_max: Decimal


def exact_document_utility_key(
    component_scores: Mapping[str, float],
    utility_artifact: Mapping,
    doc_id: str,
) -> Decimal:
    """Exact weighted-utility numerator for one document, as a Decimal.

    The denominator is constant within a document, so the numerator alone decides
    exact-set membership. Never compare this key across documents.
    """
    binding = utility_binding(utility_artifact, doc_id)
    total = Decimal(0)
    for assertion_id in sorted(binding["weights"]):
        if assertion_id not in component_scores:
            raise ValueError(
                f"component scores lack weighted assertion {assertion_id} for {doc_id}"
            )
        score = _validate_finite(
            component_scores[assertion_id], f"component score {assertion_id}"
        )
        total += Decimal(str(binding["weights"][assertion_id])) * Decimal(str(score))
    return total


def _validate_candidates(
    candidates: Sequence[LexicographicCandidate],
) -> tuple[LexicographicCandidate, ...]:
    rows = tuple(candidates)
    if not rows:
        raise ValueError("selection requires at least one candidate")
    if len({row.doc_id for row in rows}) != 1:
        raise ValueError("selection requires candidates from exactly one document")
    decisions = {tuple(decision_id for decision_id, _ in row.vector_key) for row in rows}
    if len(decisions) != 1:
        raise ValueError("selection requires one identical decision sequence")
    return rows


def _selection(
    selected: LexicographicCandidate,
    feasible: Sequence[LexicographicCandidate],
    optimum: Decimal,
) -> LexicographicSelection:
    return LexicographicSelection(
        selected=selected,
        feasible_count=len(feasible),
        optimal_utility_key=optimum,
        feasible_privacy_min=min(row.privacy_key for row in feasible),
        feasible_privacy_max=max(row.privacy_key for row in feasible),
    )


def _exact_optimal_set(
    candidates: Sequence[LexicographicCandidate],
) -> tuple[tuple[LexicographicCandidate, ...], Decimal]:
    rows = _validate_candidates(candidates)
    optimum = max(row.utility_key for row in rows)
    return tuple(row for row in rows if row.utility_key == optimum), optimum


def select_epsilon_zero(
    candidates: Sequence[LexicographicCandidate],
) -> LexicographicSelection:
    """Maximise utility exactly, then maximise the frozen count score inside that set."""
    feasible, optimum = _exact_optimal_set(candidates)
    selected = min(feasible, key=lambda row: (-row.privacy_key, row.vector_key))
    return _selection(selected, feasible, optimum)


def select_utility_only(
    candidates: Sequence[LexicographicCandidate],
    bc_vector_key: VectorKey,
) -> LexicographicSelection:
    """Privacy-blind comparator: exact utility optimum, then nearest to the BC teacher."""
    feasible, optimum = _exact_optimal_set(candidates)
    _validate_vector_key(bc_vector_key, "bc_vector_key")
    expected = tuple(decision_id for decision_id, _ in feasible[0].vector_key)
    if tuple(decision_id for decision_id, _ in bc_vector_key) != expected:
        raise ValueError("bc_vector_key decision sequence differs from the candidates")
    selected = min(
        feasible,
        key=lambda row: (_hamming(row.vector_key, bc_vector_key), row.vector_key),
    )
    return _selection(selected, feasible, optimum)


def _hamming(vector_key: VectorKey, other: VectorKey) -> int:
    return sum(
        1 for (_, action_id), (_, reference) in zip(vector_key, other, strict=True)
        if action_id != reference
    )
