"""Immutable calibration points, utility/count frontiers, and lambda menus."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from cloak.train.count_reward import CountReward
from cloak.train.interactive_ranker import legal_action_ids
from cloak.train.ranker_environment import RankerAction, RankerDocument
from cloak.train.utility_cache import UtilityResult
from cloak.train.utility_cache import stable_hash
from cloak.train.utility_credit import document_utility


CALIBRATION_POOL_VERSION = "ranker-v2-calibration-pool-v1"
LAMBDA_MENU_VERSION = "ranker-v2-lambda-menu-v1"
_NUMERIC_TOLERANCE = 1e-12


def _frozen_mapping(value: Mapping) -> Mapping:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class CalibrationTrajectory:
    doc_id: str
    corpus: str
    sources: tuple[str, ...]
    ordered_action_vector: tuple[tuple[str, str], ...]
    action_modes: tuple[str, ...]
    runtime_types: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.doc_id or not self.corpus or not self.sources:
            raise ValueError("calibration trajectory requires document, corpus, and source")
        if not self.ordered_action_vector:
            raise ValueError("calibration trajectory requires an action vector")
        if len({decision_id for decision_id, _ in self.ordered_action_vector}) != len(
            self.ordered_action_vector
        ):
            raise ValueError("calibration action vector repeats a decision")
        if (
            len(self.action_modes) != len(self.ordered_action_vector)
            or len(self.runtime_types) != len(self.ordered_action_vector)
        ):
            raise ValueError("calibration trajectory metadata length differs from vector")
        object.__setattr__(self, "sources", tuple(sorted(set(self.sources))))

    @property
    def action_vector(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.ordered_action_vector))


@dataclass(frozen=True)
class CalibrationPoint:
    doc_id: str
    corpus: str
    sources: tuple[str, ...]
    ordered_action_vector: tuple[tuple[str, str], ...]
    utility: float
    count_score: float
    component_scores: Mapping[str, float]
    count_provenance: Mapping[str, Mapping[str, Any]]
    reward_pins: Mapping[str, str]
    action_modes: tuple[str, ...]
    runtime_types: tuple[str, ...]
    result_hash: str

    def __post_init__(self) -> None:
        if not self.doc_id or not self.corpus or not self.sources or not self.result_hash:
            raise ValueError("calibration point lacks identity")
        if not self.ordered_action_vector:
            raise ValueError("calibration point requires an action vector")
        if len({decision_id for decision_id, _ in self.ordered_action_vector}) != len(
            self.ordered_action_vector
        ):
            raise ValueError("calibration action vector repeats a decision")
        if not math.isfinite(float(self.utility)) or not math.isfinite(
            float(self.count_score)
        ):
            raise ValueError("calibration point requires finite U and P")
        if not 0.0 <= float(self.count_score) <= 1.0:
            raise ValueError("calibration count score must lie in [0, 1]")
        if (
            len(self.action_modes) != len(self.ordered_action_vector)
            or len(self.runtime_types) != len(self.ordered_action_vector)
        ):
            raise ValueError("calibration point metadata length differs from vector")
        if not self.reward_pins or any(
            not isinstance(value, str) or not value for value in self.reward_pins.values()
        ):
            raise ValueError("calibration point requires reward pins")
        object.__setattr__(self, "sources", tuple(sorted(set(self.sources))))
        object.__setattr__(
            self,
            "component_scores",
            _frozen_mapping({key: float(value) for key, value in self.component_scores.items()}),
        )
        object.__setattr__(
            self,
            "count_provenance",
            _frozen_mapping({
                key: _frozen_mapping(value)
                for key, value in self.count_provenance.items()
            }),
        )
        object.__setattr__(self, "reward_pins", _frozen_mapping(self.reward_pins))


@dataclass(frozen=True)
class MergedPoint:
    doc_id: str
    utility: float
    count_score: float
    canonical_action_vector: tuple[tuple[str, str], ...]
    tie_multiplicity: int
    points: tuple[CalibrationPoint, ...]


@dataclass(frozen=True)
class SwitchPoint:
    doc_id: str
    value: float
    weight: float
    left_action_vector: tuple[tuple[str, str], ...]
    right_action_vector: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class DocumentFrontier:
    doc_id: str
    unique_point_count: int
    nondominated: tuple[MergedPoint, ...]
    envelope: tuple[MergedPoint, ...]
    switch_points: tuple[SwitchPoint, ...]
    switch_eligible: bool


def _action_by_id(document_action_ids: Mapping[str, RankerAction], action_id: str) -> RankerAction:
    try:
        return document_action_ids[action_id]
    except KeyError as error:
        raise ValueError(f"unknown anchor action: {action_id}") from error


def _reserved_fixed_fills(document: RankerDocument) -> tuple[str, ...]:
    occurrences = {
        str(row["occurrence_id"]): row for row in document.occurrences
    }
    fills = []
    for decision in document.fixed_decisions:
        action = decision.actions[0]
        if action.mode == "placeholder":
            continue
        if action.mode == "level":
            candidates = (action.fill,)
        else:
            candidates = tuple(
                str(occurrences[occurrence_id]["surface"])
                for occurrence_id in decision.occurrence_ids
            )
        for fill in candidates:
            if not fill:
                raise ValueError(f"fixed decision lacks a fill: {decision.decision_id}")
            fills.append(fill)
    return tuple(fills)


def _anchor_walk(
    document: RankerDocument,
    count_reward: CountReward,
    source: str,
    chooser,
) -> CalibrationTrajectory:
    claimed: dict[str, str] = {}
    reserved = _reserved_fixed_fills(document)
    vector = []
    modes = []
    runtime_types = []
    for decision in document.policy_decisions:
        actions = {action.action_id: action for action in decision.actions}
        menu = legal_action_ids(decision, claimed, reserved)
        selected_id = chooser(decision, menu, count_reward)
        if selected_id not in menu:
            raise ValueError(f"anchor chooser returned an illegal action: {selected_id}")
        action = _action_by_id(actions, selected_id)
        vector.append((decision.decision_id, selected_id))
        modes.append(action.mode)
        runtime_types.append(decision.runtime_type)
        if action.mode == "level":
            assert action.fill is not None
            claimed.setdefault(action.fill.casefold(), decision.decision_id)
    return CalibrationTrajectory(
        doc_id=document.doc_id,
        corpus=document.corpus,
        sources=(source,),
        ordered_action_vector=tuple(vector),
        action_modes=tuple(modes),
        runtime_types=tuple(runtime_types),
    )


def _choose_behavior_cloning(decision, menu, _count_reward) -> str:
    legal = set(menu)
    levels = [
        action for action in decision.actions
        if action.mode == "level" and action.action_id in legal
    ]
    if levels:
        if any(action.authored_level_index is None for action in levels):
            raise ValueError(f"level lacks authored order: {decision.decision_id}")
        return min(
            levels,
            key=lambda action: (int(action.authored_level_index), action.action_id),
        ).action_id
    placeholders = [
        action.action_id for action in decision.actions
        if action.mode == "placeholder" and action.action_id in legal
    ]
    if len(placeholders) != 1:
        raise ValueError(f"anchor requires one placeholder: {decision.decision_id}")
    return placeholders[0]


def _choose_mode(mode: str):
    def choose(decision, menu, _count_reward):
        matches = [
            action.action_id for action in decision.actions
            if action.mode == mode and action.action_id in menu
        ]
        if len(matches) != 1:
            raise ValueError(f"anchor requires one {mode}: {decision.decision_id}")
        return matches[0]

    return choose


def _choose_minimum_count_non_keep(decision, menu, count_reward: CountReward) -> str:
    actions = {action.action_id: action for action in decision.actions}
    candidates = tuple(
        action_id for action_id in menu if actions[action_id].mode != "keep"
    )
    if not candidates:
        raise ValueError(f"non-KEEP anchor has no legal action: {decision.decision_id}")
    scores = count_reward.action_scores(decision.decision_id, candidates)
    return min(
        zip(candidates, scores.tolist(), strict=True),
        key=lambda row: (
            float(row[1]),
            actions[row[0]].mode != "level",
            actions[row[0]].authored_level_index
            if actions[row[0]].authored_level_index is not None else math.inf,
            row[0],
        ),
    )[0]


def _choose_midpoint_level(decision, menu, _count_reward) -> str:
    legal = set(menu)
    levels = sorted(
        (
            action for action in decision.actions
            if action.mode == "level" and action.action_id in legal
        ),
        key=lambda action: (
            action.authored_level_index
            if action.authored_level_index is not None else math.inf,
            action.action_id,
        ),
    )
    if levels:
        if any(action.authored_level_index is None for action in levels):
            raise ValueError(f"level lacks authored order: {decision.decision_id}")
        return levels[len(levels) // 2].action_id
    return _choose_mode("placeholder")(decision, menu, _count_reward)


def build_anchor_trajectories(
    document: RankerDocument,
    count_reward: CountReward,
) -> tuple[CalibrationTrajectory, ...]:
    """Build and exact-deduplicate the five deterministic calibration anchors."""

    raw = (
        _anchor_walk(
            document, count_reward, "behavior_cloning", _choose_behavior_cloning,
        ),
        _anchor_walk(document, count_reward, "keep_walk", _choose_mode("keep")),
        _anchor_walk(
            document,
            count_reward,
            "minimum_count_non_keep_walk",
            _choose_minimum_count_non_keep,
        ),
        _anchor_walk(
            document, count_reward, "midpoint_level_walk", _choose_midpoint_level,
        ),
        _anchor_walk(
            document,
            count_reward,
            "all_placeholder_walk",
            _choose_mode("placeholder"),
        ),
    )
    unique: dict[tuple[tuple[str, str], ...], CalibrationTrajectory] = {}
    for trajectory in raw:
        previous = unique.get(trajectory.ordered_action_vector)
        if previous is None:
            unique[trajectory.ordered_action_vector] = trajectory
        else:
            unique[trajectory.ordered_action_vector] = replace(
                previous,
                sources=tuple(sorted(set(previous.sources) | set(trajectory.sources))),
            )
    return tuple(unique[key] for key in sorted(unique))


def calibration_point_from_result(
    trajectory: CalibrationTrajectory,
    result: UtilityResult,
    *,
    count_reward: CountReward,
    count_state: Mapping,
    utility_artifact: Mapping,
    reward_pins: Mapping[str, str],
) -> CalibrationPoint:
    if result.doc_id != trajectory.doc_id:
        raise ValueError("cached utility result document differs from calibration trajectory")
    if dict(result.action_vector) != dict(trajectory.ordered_action_vector):
        raise ValueError("cached utility result vector differs from calibration trajectory")
    scores = [
        float(count_reward.action_scores(decision_id, (action_id,))[0])
        for decision_id, action_id in trajectory.ordered_action_vector
    ]
    raw_provenance = count_state.get("action_scores", {})
    provenance = {}
    for _decision_id, action_id in trajectory.ordered_action_vector:
        row = raw_provenance.get(action_id)
        if not isinstance(row, Mapping):
            raise ValueError(f"count state lacks provenance for action {action_id}")
        provenance[action_id] = {
            key: row.get(key)
            for key in (
                "mode", "profile_id", "grounding_status", "source_family",
                "evidence_ref",
            )
        }
    return CalibrationPoint(
        doc_id=trajectory.doc_id,
        corpus=trajectory.corpus,
        sources=trajectory.sources,
        ordered_action_vector=trajectory.ordered_action_vector,
        utility=document_utility(
            result.component_scores, utility_artifact, trajectory.doc_id,
        ),
        count_score=sum(scores) / len(scores),
        component_scores=result.component_scores,
        count_provenance=provenance,
        reward_pins=reward_pins,
        action_modes=trajectory.action_modes,
        runtime_types=trajectory.runtime_types,
        result_hash=result.result_hash,
    )


def _point_payload(point: CalibrationPoint) -> dict[str, Any]:
    return {
        "sources": list(point.sources),
        "ordered_action_vector": [list(row) for row in point.ordered_action_vector],
        "utility": point.utility,
        "count_score": point.count_score,
        "component_scores": dict(point.component_scores),
        "count_provenance": {
            key: dict(value) for key, value in point.count_provenance.items()
        },
        "reward_pins": dict(point.reward_pins),
        "action_modes": list(point.action_modes),
        "runtime_types": list(point.runtime_types),
        "result_hash": point.result_hash,
    }


def freeze_calibration_pool(
    points: Sequence[CalibrationPoint],
    *,
    split_by_doc: Mapping[str, str],
    reward_pins: Mapping[str, str],
) -> dict[str, Any]:
    deduplicated = deduplicate_action_vectors(points)
    if not deduplicated:
        raise ValueError("calibration pool cannot be empty")
    doc_ids = {point.doc_id for point in deduplicated}
    if set(split_by_doc) != doc_ids or any(
        value not in {"train", "development"} for value in split_by_doc.values()
    ):
        raise ValueError("calibration split must cover every pool document exactly")
    if any(dict(point.reward_pins) != dict(reward_pins) for point in deduplicated):
        raise ValueError("calibration point reward pins differ")
    grouped: dict[str, list[CalibrationPoint]] = defaultdict(list)
    for point in deduplicated:
        grouped[point.doc_id].append(point)
    artifact: dict[str, Any] = {
        "artifact_version": CALIBRATION_POOL_VERSION,
        "reward_pins": dict(sorted(reward_pins.items())),
        "documents": {
            doc_id: {
                "corpus": rows[0].corpus,
                "split": split_by_doc[doc_id],
                "points": [_point_payload(point) for point in rows],
            }
            for doc_id, rows in sorted(grouped.items())
        },
        "summary": {
            "document_count": len(grouped),
            "trajectory_count": len(deduplicated),
            "point_count": len({
                (point.doc_id, point.utility, point.count_score)
                for point in deduplicated
            }),
        },
    }
    artifact["artifact_hash"] = stable_hash(artifact)
    return artifact


def _same_point_except_sources(left: CalibrationPoint, right: CalibrationPoint) -> bool:
    return all(
        getattr(left, name) == getattr(right, name)
        for name in (
            "doc_id", "corpus", "ordered_action_vector", "utility", "count_score",
            "component_scores", "count_provenance", "reward_pins", "action_modes",
            "runtime_types", "result_hash",
        )
    )


def deduplicate_action_vectors(
    points: Sequence[CalibrationPoint],
) -> tuple[CalibrationPoint, ...]:
    """Merge byte-equivalent ordered vectors and reject conflicting cached results."""

    unique: dict[tuple[str, tuple[tuple[str, str], ...]], CalibrationPoint] = {}
    for point in points:
        key = (point.doc_id, point.ordered_action_vector)
        previous = unique.get(key)
        if previous is None:
            unique[key] = point
            continue
        if not _same_point_except_sources(previous, point):
            raise ValueError(
                f"conflicting duplicate action vector for {point.doc_id}"
            )
        unique[key] = replace(
            previous,
            sources=tuple(sorted(set(previous.sources) | set(point.sources))),
        )
    return tuple(unique[key] for key in sorted(unique))


def merge_exact_point_ties(
    points: Sequence[CalibrationPoint],
) -> tuple[MergedPoint, ...]:
    """Merge exact U/P ties after exact action-vector deduplication."""

    groups: dict[tuple[str, float, float], list[CalibrationPoint]] = defaultdict(list)
    for point in deduplicate_action_vectors(points):
        groups[(point.doc_id, float(point.utility), float(point.count_score))].append(point)
    merged = []
    for (doc_id, utility, count_score), tied in groups.items():
        ordered = tuple(sorted(tied, key=lambda row: row.ordered_action_vector))
        merged.append(MergedPoint(
            doc_id=doc_id,
            utility=utility,
            count_score=count_score,
            canonical_action_vector=ordered[0].ordered_action_vector,
            tie_multiplicity=len(ordered),
            points=ordered,
        ))
    return tuple(sorted(
        merged,
        key=lambda row: (row.doc_id, row.count_score, -row.utility, row.canonical_action_vector),
    ))


def remove_weakly_dominated(
    points: Sequence[MergedPoint],
) -> tuple[MergedPoint, ...]:
    kept = []
    for point in points:
        dominated = any(
            other is not point
            and other.doc_id == point.doc_id
            and other.utility >= point.utility
            and other.count_score >= point.count_score
            and (
                other.utility > point.utility
                or other.count_score > point.count_score
            )
            for other in points
        )
        if not dominated:
            kept.append(point)
    return tuple(sorted(
        kept,
        key=lambda row: (row.doc_id, row.count_score, -row.utility),
    ))


def _slope(left: MergedPoint, right: MergedPoint) -> float:
    return (right.utility - left.utility) / (right.count_score - left.count_score)


def upper_convex_envelope(
    points: Sequence[MergedPoint],
) -> tuple[MergedPoint, ...]:
    """Return the concave upper hull in increasing count-score order."""

    ordered = sorted(points, key=lambda row: (row.count_score, -row.utility))
    if len({point.doc_id for point in ordered}) > 1:
        raise ValueError("upper envelope requires one document")
    hull: list[MergedPoint] = []
    for point in ordered:
        while len(hull) >= 2 and _slope(hull[-2], hull[-1]) <= _slope(
            hull[-1], point
        ) + _NUMERIC_TOLERANCE:
            hull.pop()
        hull.append(point)
    return tuple(hull)


def document_frontier(points: Sequence[CalibrationPoint]) -> DocumentFrontier:
    points = tuple(points)
    if not points:
        raise ValueError("document frontier requires calibration points")
    doc_ids = {point.doc_id for point in points}
    if len(doc_ids) != 1:
        raise ValueError("document frontier requires one document")
    doc_id = next(iter(doc_ids))
    merged = merge_exact_point_ties(points)
    nondominated = remove_weakly_dominated(merged)
    envelope = upper_convex_envelope(nondominated)
    switch_eligible = len(merged) >= 3
    raw: list[tuple[float, MergedPoint, MergedPoint]] = []
    if switch_eligible:
        for left, right in zip(envelope, envelope[1:]):
            if right.count_score <= left.count_score or left.utility <= right.utility:
                continue
            value = (left.utility - right.utility) / (
                right.count_score - left.count_score
            )
            if math.isfinite(value) and value > 0.0 and not any(
                math.isclose(value, previous, rel_tol=0.0, abs_tol=_NUMERIC_TOLERANCE)
                for previous, _, _ in raw
            ):
                raw.append((value, left, right))
    weight = 1.0 / len(raw) if raw else 0.0
    switches = tuple(SwitchPoint(
        doc_id=doc_id,
        value=value,
        weight=weight,
        left_action_vector=left.canonical_action_vector,
        right_action_vector=right.canonical_action_vector,
    ) for value, left, right in raw)
    return DocumentFrontier(
        doc_id=doc_id,
        unique_point_count=len(merged),
        nondominated=nondominated,
        envelope=envelope,
        switch_points=switches,
        switch_eligible=switch_eligible,
    )


def weighted_log_quantiles(
    switches: Sequence[SwitchPoint], quantiles: Sequence[float],
) -> tuple[float, ...]:
    rows = sorted(switches, key=lambda row: (math.log(row.value), row.doc_id))
    if not rows or any(
        not math.isfinite(row.value) or row.value <= 0.0 or row.weight <= 0.0
        for row in rows
    ):
        raise ValueError("weighted log quantiles require positive switches and weights")
    total = sum(row.weight for row in rows)
    result = []
    for quantile in quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must lie in [0, 1]")
        target = quantile * total
        cumulative = 0.0
        selected = rows[-1].value
        for row in rows:
            cumulative += row.weight
            if cumulative + _NUMERIC_TOLERANCE >= target:
                selected = row.value
                break
        result.append(float(selected))
    return tuple(result)


def snap_to_observed(value: float, observed: Sequence[float]) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("snap target must be finite and positive")
    candidates = sorted(set(float(item) for item in observed))
    if not candidates or any(not math.isfinite(item) or item <= 0.0 for item in candidates):
        raise ValueError("observed switch points must be finite and positive")
    return min(candidates, key=lambda item: (abs(math.log(item) - math.log(value)), item))


def _winner(points: Sequence[CalibrationPoint], lambda_value: float) -> CalibrationPoint:
    best_value = max(point.utility + lambda_value * point.count_score for point in points)
    tied = [
        point for point in points
        if math.isclose(
            point.utility + lambda_value * point.count_score,
            best_value,
            rel_tol=0.0,
            abs_tol=_NUMERIC_TOLERANCE,
        )
    ]
    return min(tied, key=lambda row: row.ordered_action_vector)


def _points_by_document(
    pool: Sequence[CalibrationPoint],
) -> dict[str, tuple[CalibrationPoint, ...]]:
    grouped: dict[str, list[CalibrationPoint]] = defaultdict(list)
    for point in deduplicate_action_vectors(pool):
        grouped[point.doc_id].append(point)
    return {key: tuple(value) for key, value in sorted(grouped.items())}


def replay_signature(pool: Sequence[CalibrationPoint], lambda_value: float) -> str:
    grouped = _points_by_document(pool)
    winners = {doc_id: _winner(points, lambda_value) for doc_id, points in grouped.items()}
    modes = defaultdict(int)
    total = 0
    for winner in winners.values():
        for mode in winner.action_modes:
            modes[mode] += 1
            total += 1
    payload = {
        "winners": {
            doc_id: list(winner.ordered_action_vector)
            for doc_id, winner in winners.items()
        },
        "mode_rates": {
            mode: count / total for mode, count in sorted(modes.items())
        } if total else {},
    }
    return stable_hash(payload)


def merge_equivalent_lambdas(
    values: Sequence[float], pool: Sequence[CalibrationPoint],
) -> tuple[tuple[float, ...], dict[float, str]]:
    ordered = tuple(float(value) for value in values)
    if not ordered or ordered[0] != 0.0 or any(
        not math.isfinite(value) or value < 0.0 for value in ordered
    ) or tuple(sorted(set(ordered))) != ordered:
        raise ValueError("lambda values must be unique, increasing, and start at zero")
    signatures = {value: replay_signature(pool, value) for value in ordered}
    seen = set()
    kept = []
    for value in ordered:
        signature = signatures[value]
        if signature not in seen:
            kept.append(value)
            seen.add(signature)
    return tuple(kept), signatures


def _quantiles_for_size(menu_size: int) -> tuple[float, ...]:
    values = {
        3: (0.40, 0.90),
        4: (0.25, 0.60, 0.90),
        5: (0.20, 0.45, 0.70, 0.90),
    }
    try:
        return values[menu_size]
    except KeyError as error:
        raise ValueError("lambda menu size must be three, four, or five") from error


def _replay_report(
    pool: Sequence[CalibrationPoint], values: Sequence[float],
) -> dict[str, Any]:
    grouped = _points_by_document(pool)
    winners_by_value = [
        {doc_id: _winner(points, value) for doc_id, points in grouped.items()}
        for value in values
    ]
    eligible = {
        doc_id for doc_id, points in grouped.items()
        if len(remove_weakly_dominated(merge_exact_point_ties(points))) >= 2
    }
    changes = []
    for left, right in zip(winners_by_value, winners_by_value[1:]):
        changes.append(
            sum(
                left[doc_id].ordered_action_vector
                != right[doc_id].ordered_action_vector
                for doc_id in eligible
            ) / len(eligible) if eligible else 0.0
        )
    selected_count = [
        sum(winner.count_score for winner in winners.values()) / len(winners)
        for winners in winners_by_value
    ]
    placeholder_fraction = []
    mode_rates = []
    support = []
    for winners in winners_by_value:
        modes: dict[str, int] = defaultdict(int)
        corpora: dict[str, set[str]] = defaultdict(set)
        types: dict[str, int] = defaultdict(int)
        total = 0
        for doc_id, winner in winners.items():
            corpora[winner.corpus].add(doc_id)
            for mode, runtime_type in zip(
                winner.action_modes, winner.runtime_types, strict=True
            ):
                modes[mode] += 1
                types[runtime_type] += 1
                total += 1
        placeholder_fraction.append(modes.get("placeholder", 0) / total if total else 0.0)
        mode_rates.append({
            mode: count / total for mode, count in sorted(modes.items())
        } if total else {})
        support.append({
            "documents_by_corpus": {
                corpus: len(ids) for corpus, ids in sorted(corpora.items())
            },
            "decisions_by_type": dict(sorted(types.items())),
        })
    pure_utility = {
        doc_id: min(
            (point for point in points if point.utility == max(row.utility for row in points)),
            key=lambda row: row.ordered_action_vector,
        ).ordered_action_vector
        for doc_id, points in grouped.items()
    }
    lambda_zero_identity = all(
        winners_by_value[0][doc_id].ordered_action_vector == action_vector
        for doc_id, action_vector in pure_utility.items()
    )
    return {
        "winner_signatures": [replay_signature(pool, value) for value in values],
        "winner_change": changes,
        "selected_count_score": selected_count,
        "lambda_zero_identity": lambda_zero_identity,
        "placeholder_fraction": placeholder_fraction,
        "mode_rates": mode_rates,
        "support": support,
        "eligible_document_count": len(eligible),
    }


def _failure_reasons(
    values: Sequence[float],
    report: Mapping[str, Any],
    *,
    min_adjacent_winner_change: float,
    max_placeholder_fraction: float,
    min_supported_documents_by_corpus: int,
    min_supported_decisions_by_type: int,
) -> list[str]:
    failures = []
    if len(values) < 3:
        failures.append("fewer_than_three_supported_profiles")
    if any(value < min_adjacent_winner_change for value in report["winner_change"]):
        failures.append("adjacent_winner_change_below_threshold")
    counts = report["selected_count_score"]
    if any(right + _NUMERIC_TOLERANCE < left for left, right in zip(counts, counts[1:])):
        failures.append("selected_count_score_decreased")
    if not report["lambda_zero_identity"]:
        failures.append("lambda_zero_identity_failed")
    if report["placeholder_fraction"] and (
        report["placeholder_fraction"][-1] > max_placeholder_fraction
    ):
        failures.append("all_placeholder_ceiling_exceeded")
    for row in report["support"]:
        if any(
            count < min_supported_documents_by_corpus
            for count in row["documents_by_corpus"].values()
        ):
            failures.append("corpus_support_below_threshold")
            break
        if any(
            count < min_supported_decisions_by_type
            for count in row["decisions_by_type"].values()
        ):
            failures.append("type_support_below_threshold")
            break
    return sorted(set(failures))


def select_lambda_menu(
    pool: Sequence[CalibrationPoint],
    *,
    menu_size: int,
    min_adjacent_winner_change: float,
    max_placeholder_fraction: float,
    min_supported_documents_by_corpus: int,
    min_supported_decisions_by_type: int,
    max_replacement_passes: int = 2,
) -> dict[str, Any]:
    """Select, replay, and gate a three-to-five-profile observed lambda menu."""

    if max_replacement_passes != 2:
        raise ValueError("lambda replacement procedure is frozen to two passes")
    if not 0.0 <= min_adjacent_winner_change <= 1.0:
        raise ValueError("winner-change threshold must lie in [0, 1]")
    if not 0.0 <= max_placeholder_fraction <= 1.0:
        raise ValueError("placeholder ceiling must lie in [0, 1]")
    grouped = _points_by_document(pool)
    frontiers = tuple(document_frontier(points) for points in grouped.values())
    switches = tuple(
        switch for frontier in frontiers for switch in frontier.switch_points
    )
    if switches:
        nonzero = weighted_log_quantiles(switches, _quantiles_for_size(menu_size))
        values = tuple(sorted(set((0.0, *nonzero))))
        values, _ = merge_equivalent_lambdas(values, pool)
    else:
        values = (0.0,)
    replacement_passes = 0
    report = _replay_report(pool, values)
    failures = _failure_reasons(
        values,
        report,
        min_adjacent_winner_change=min_adjacent_winner_change,
        max_placeholder_fraction=max_placeholder_fraction,
        min_supported_documents_by_corpus=min_supported_documents_by_corpus,
        min_supported_decisions_by_type=min_supported_decisions_by_type,
    )
    observed = sorted(set(switch.value for switch in switches))
    while failures and replacement_passes < max_replacement_passes and len(values) >= 3:
        replacement_passes += 1
        alternatives = []
        for index in range(1, len(values)):
            lower = values[index - 1]
            upper = values[index + 1] if index + 1 < len(values) else math.inf
            for candidate in observed:
                if lower < candidate < upper and candidate not in values:
                    proposal = tuple(sorted((*values[:index], candidate, *values[index + 1:])))
                    proposal, _ = merge_equivalent_lambdas(proposal, pool)
                    proposal_report = _replay_report(pool, proposal)
                    proposal_failures = _failure_reasons(
                        proposal,
                        proposal_report,
                        min_adjacent_winner_change=min_adjacent_winner_change,
                        max_placeholder_fraction=max_placeholder_fraction,
                        min_supported_documents_by_corpus=min_supported_documents_by_corpus,
                        min_supported_decisions_by_type=min_supported_decisions_by_type,
                    )
                    alternatives.append((
                        len(proposal_failures),
                        -min(proposal_report["winner_change"], default=0.0),
                        candidate,
                        proposal,
                        proposal_report,
                        proposal_failures,
                    ))
        if alternatives:
            _, _, _, values, report, failures = min(alternatives, key=lambda row: row[:3])
        else:
            # The deterministic pass was attempted but the observed switch set offered no
            # order-preserving replacement. A second pass records the frozen exhaustion limit.
            continue
    artifact: dict[str, Any] = {
        "artifact_version": LAMBDA_MENU_VERSION,
        "values": list(values),
        "profile_names": ["lambda-zero", *(
            f"lambda-{index}" for index in range(1, len(values))
        )],
        "switch_points": [
            {
                "doc_id": switch.doc_id,
                "value": switch.value,
                "weight": switch.weight,
                "left_action_vector": list(switch.left_action_vector),
                "right_action_vector": list(switch.right_action_vector),
            }
            for switch in switches
        ],
        "replay_report": report,
        "replacement_passes": replacement_passes,
        "failure_reasons": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    artifact["artifact_hash"] = stable_hash(artifact)
    return artifact
