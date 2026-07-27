"""Predeclared ranker-v2 diagnostics, threshold freezing, and cache admission."""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from statistics import mean
from typing import Any

from cloak.train.profile_count import ProfileCountTargets
from cloak.train.interactive_ranker import assemble_action_vector, legal_action_ids
from cloak.train.lambda_menu import (
    CalibrationPoint,
    CalibrationTrajectory,
    deduplicate_action_vectors,
    document_frontier,
)
from cloak.train.ranker_privacy import REQUIRED_BASELINES
from cloak.train.ranker_environment import RankerDocument
from cloak.train.utility_cache import UtilityCache, UtilityRequest, stable_hash
from cloak.train.utility_credit import document_utility


DIAGNOSTIC_VERSION = "ranker-v2-diagnostics-v1"
SPIKE_VERSION = "ranker-v2-diagnostic-spike-v1"
THRESHOLD_RULES_VERSION = "ranker-v2-threshold-rules-v1"
THRESHOLD_MANIFEST_VERSION = "ranker-v2-threshold-manifest-v1"
PRIVACY_DIAGNOSTIC_VERSION = "ranker-v2-semantic-privacy-diagnostic-v2"
PRIVACY_BOOTSTRAP_SEED = 1729
PRIVACY_BOOTSTRAP_SAMPLES = 2000
PRIVACY_MARGINS = {
    "candidate_calibration": 0.01,
    "candidate_ordering": 0.01,
    "candidate_regret": 0.01,
    "authored_calibration": 0.05,
    "authored_regret": 0.01,
}

_EMPIRICAL_FIELDS = (
    "feasibility_gates.min_distinct_points_per_document",
    "feasibility_gates.min_supported_documents_by_corpus",
    "feasibility_gates.min_supported_decisions_by_type",
    "feasibility_gates.min_adjacent_winner_change",
    "feasibility_gates.max_flat_menu_fraction",
    "feasibility_gates.collision_rate_trigger",
    "feasibility_gates.lost_count_opportunity_trigger",
    "feasibility_gates.min_nonzero_counterfactual_rate",
    "feasibility_gates.max_all_placeholder_fraction",
    "scheduler.call_budget",
    "scheduler.uniform_reserve_fraction",
    "scheduler.endpoint_fraction",
    "scheduler.direction_balance_tolerance",
    "acceptance.confidence_level",
    "acceptance.utility_noninferiority_margin",
    "acceptance.minimum_supported_documents",
)
_ACTIONS = frozenset({"block", "ablation", "reduce_scope", "report_only"})
_SPLITS = frozenset({"train", "development", "train+development"})


def required_empirical_threshold_fields() -> tuple[str, ...]:
    return _EMPIRICAL_FIELDS


def _rule(
    measurement_definition: str,
    candidate_rule: Mapping[str, Any],
    *,
    action: str,
    allowed_split: str = "train+development",
    support_rule: str = "all structurally eligible groups in the allowed split",
    tie_handling: str = "choose the more conservative value; exact ties choose the smaller value",
) -> dict[str, Any]:
    return {
        "measurement_definition": measurement_definition,
        "candidate_rule": dict(candidate_rule),
        "allowed_split": allowed_split,
        "support_rule": support_rule,
        "tie_handling": tie_handling,
        "action": action,
    }


def default_threshold_rules() -> dict[str, Any]:
    """Return pre-spike selection rules without inspecting any run outcome."""

    fields = {
        "feasibility_gates.min_distinct_points_per_document": _rule(
            "Distinct exact (U, P_count) points required for switch estimation.",
            {"kind": "fixed", "value": 3},
            action="reduce_scope",
        ),
        "feasibility_gates.min_supported_documents_by_corpus": _rule(
            "Documents with positive switch support in each corpus.",
            {
                "kind": "min_group_fraction",
                "path": (
                    "measurements.frontier_switch_support."
                    "positive_switch_documents_by_corpus"
                ),
                "fraction": 0.5,
                "minimum": 1,
            },
            action="reduce_scope",
        ),
        "feasibility_gates.min_supported_decisions_by_type": _rule(
            "Controlled decisions with count support in each runtime type.",
            {
                "kind": "min_group_fraction",
                "path": "measurements.support.decisions_by_type",
                "fraction": 0.5,
                "minimum": 1,
            },
            action="reduce_scope",
        ),
        "feasibility_gates.min_adjacent_winner_change": _rule(
            "Deterministic replay winner-change floor between adjacent profiles.",
            {
                "kind": "measurement",
                "path": "measurements.frontier_switch_support.winner_change_floor",
            },
            action="reduce_scope",
        ),
        "feasibility_gates.max_flat_menu_fraction": _rule(
            "Fraction of decisions whose non-placeholder level scores are flat.",
            {
                "kind": "measurement",
                "path": "measurements.count_signal.flat_menu_fraction",
            },
            action="ablation",
        ),
        "feasibility_gates.collision_rate_trigger": _rule(
            "Dynamic-mask collision events divided by eligible later decisions.",
            {
                "kind": "measurement",
                "path": "measurements.injectivity.collision_rate",
            },
            action="ablation",
        ),
        "feasibility_gates.lost_count_opportunity_trigger": _rule(
            "Maximum pre-mask minus post-mask attainable level count score.",
            {
                "kind": "measurement",
                "path": "measurements.injectivity.lost_count_opportunity_max",
            },
            action="ablation",
        ),
        "feasibility_gates.min_nonzero_counterfactual_rate": _rule(
            "Fraction of measured adjacent counterfactuals with nonzero delta_U.",
            {
                "kind": "measurement",
                "path": "measurements.counterfactual_support.nonzero_rate",
            },
            action="reduce_scope",
        ),
        "feasibility_gates.max_all_placeholder_fraction": _rule(
            "Highest supported profile's aggregate placeholder ceiling.",
            {"kind": "fixed", "value": 0.95},
            action="reduce_scope",
        ),
        "scheduler.call_budget": _rule(
            "Frozen counterfactual calls per scheduler allocation unit.",
            {"kind": "fixed", "value": 5},
            action="block",
            allowed_split="train",
        ),
        "scheduler.uniform_reserve_fraction": _rule(
            "Seeded-uniform share of the fixed counterfactual budget.",
            {"kind": "fixed", "value": 0.2},
            action="block",
            allowed_split="train",
        ),
        "scheduler.endpoint_fraction": _rule(
            "Minority share reserved for KEEP and placeholder endpoints.",
            {"kind": "fixed", "value": 0.2},
            action="block",
            allowed_split="train",
        ),
        "scheduler.direction_balance_tolerance": _rule(
            "Maximum absolute finer/coarser count difference within a profile.",
            {"kind": "fixed", "value": 1},
            action="block",
            allowed_split="train",
        ),
        "acceptance.confidence_level": _rule(
            "Paired document-level confidence level for later promotion tests.",
            {"kind": "fixed", "value": 0.95},
            action="block",
            allowed_split="development",
        ),
        "acceptance.utility_noninferiority_margin": _rule(
            "Smallest utility difference resolvable above deterministic reader jitter.",
            {
                "kind": "measurement",
                "path": (
                    "measurements.utility_resolution."
                    "reader_jitter_by_split.development.abs_max"
                ),
            },
            action="block",
            allowed_split="development",
        ),
        "acceptance.minimum_supported_documents": _rule(
            "Minimum development documents available to the paired promotion test.",
            {
                "kind": "min_group_fraction",
                "path": (
                    "measurements.support.documents_by_corpus_by_split.development"
                ),
                "fraction": 0.5,
                "minimum": 1,
            },
            action="block",
            allowed_split="development",
        ),
    }
    return {
        "artifact_version": THRESHOLD_RULES_VERSION,
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "fields": fields,
    }


def validate_threshold_rules(rules: Mapping) -> dict[str, Any]:
    if rules.get("artifact_version") != THRESHOLD_RULES_VERSION:
        raise ValueError("unsupported threshold-rules version")
    if rules.get("diagnostic_version") != DIAGNOSTIC_VERSION:
        raise ValueError("threshold rules diagnostic version mismatch")
    fields = rules.get("fields")
    if not isinstance(fields, Mapping) or set(fields) != set(_EMPIRICAL_FIELDS):
        raise ValueError("threshold rules do not cover every empirical field")
    required = {
        "measurement_definition", "candidate_rule", "allowed_split",
        "support_rule", "tie_handling", "action",
    }
    for name, row in fields.items():
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError(f"threshold rule has invalid schema: {name}")
        if row["allowed_split"] not in _SPLITS or row["action"] not in _ACTIONS:
            raise ValueError(f"threshold rule has invalid split/action: {name}")
        if row["action"] == "report_only":
            raise ValueError(f"run-relevant threshold cannot be report_only: {name}")
        if not all(
            isinstance(row[key], str) and row[key].strip()
            for key in ("measurement_definition", "support_rule", "tie_handling")
        ) or not isinstance(row["candidate_rule"], Mapping):
            raise ValueError(f"threshold rule is incomplete: {name}")
    return deepcopy(dict(rules))


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    values = tuple(float(value) for value in values)
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
    }


def _fixed_fills(document: RankerDocument) -> tuple[str, ...]:
    occurrences = {str(row["occurrence_id"]): row for row in document.occurrences}
    fills = []
    for decision in document.fixed_decisions:
        action = decision.actions[0]
        if action.mode == "placeholder":
            continue
        if action.mode == "level":
            rows = (action.fill,)
        else:
            rows = tuple(
                str(occurrences[occurrence_id]["surface"])
                for occurrence_id in decision.occurrence_ids
            )
        fills.extend(str(fill) for fill in rows if fill)
    return tuple(fills)


def _injectivity_measurements(
    points: Sequence[CalibrationPoint],
    documents: Mapping[str, RankerDocument],
    count_reward: ProfileCountTargets,
) -> dict[str, Any]:
    collision_events = 0
    eligible_decisions = 0
    losses = []
    by_type: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"collision_events": 0, "eligible_decisions": 0, "losses": []}
    )
    for point in points:
        document = documents[point.doc_id]
        selected = dict(point.ordered_action_vector)
        claimed: dict[str, str] = {}
        reserved = _fixed_fills(document)
        for decision in document.policy_decisions:
            levels = tuple(
                action.action_id for action in decision.actions if action.mode == "level"
            )
            legal = legal_action_ids(decision, claimed, reserved)
            legal_levels = tuple(action_id for action_id in levels if action_id in legal)
            if levels:
                eligible_decisions += 1
                by_type[decision.runtime_type]["eligible_decisions"] += 1
                if len(legal_levels) < len(levels):
                    collision_events += 1
                    by_type[decision.runtime_type]["collision_events"] += 1
                before = count_reward.action_scores(decision.decision_id, levels)
                after = count_reward.action_scores(decision.decision_id, legal_levels)
                best_before = float(before.max()) if len(before) else 0.0
                best_after = float(after.max()) if len(after) else 0.0
                losses.append(max(0.0, best_before - best_after))
                by_type[decision.runtime_type]["losses"].append(
                    max(0.0, best_before - best_after)
                )
            action = next(
                row for row in decision.actions
                if row.action_id == selected[decision.decision_id]
            )
            if action.mode == "level":
                assert action.fill is not None
                claimed.setdefault(action.fill.casefold(), decision.decision_id)
    return {
        "collision_events": collision_events,
        "eligible_later_decisions": eligible_decisions,
        "collision_rate": collision_events / eligible_decisions if eligible_decisions else 0.0,
        "lost_count_opportunity": _distribution(losses),
        "lost_count_opportunity_max": max(losses, default=0.0),
        "by_type": {
            runtime_type: {
                "collision_events": row["collision_events"],
                "eligible_decisions": row["eligible_decisions"],
                "collision_rate": (
                    row["collision_events"] / row["eligible_decisions"]
                    if row["eligible_decisions"] else 0.0
                ),
                "lost_count_opportunity": _distribution(row["losses"]),
                "lost_count_opportunity_max": max(row["losses"], default=0.0),
            }
            for runtime_type, row in sorted(by_type.items())
        },
    }


def _count_measurements(
    documents: Mapping[str, RankerDocument],
    count_reward: ProfileCountTargets,
    count_state: Mapping,
) -> dict[str, Any]:
    flat = 0
    decision_count = 0
    clipped = 0
    level_count = 0
    deltas = []
    by_type: dict[str, list[float]] = defaultdict(list)
    for document in documents.values():
        for decision in document.policy_decisions:
            ordered = sorted(
                (action for action in decision.actions if action.mode == "level"),
                key=lambda action: (int(action.authored_level_index), action.action_id),
            )
            if not ordered:
                continue
            ids = tuple(action.action_id for action in ordered)
            scores = [
                float(value) for value in count_reward.action_scores(
                    decision.decision_id, ids,
                ).tolist()
            ]
            decision_count += 1
            flat += len(set(scores)) <= 1
            clipped += sum(value >= 1.0 for value in scores)
            level_count += len(scores)
            adjacent = [right - left for left, right in zip(scores, scores[1:])]
            deltas.extend(adjacent)
            by_type[decision.runtime_type].extend(adjacent)
    provenance = Counter()
    provenance_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for row in count_state.get("action_targets", {}).values():
        if row.get("mode") == "level":
            label = str(row.get("grounding_status") or "provisional-null")
            provenance[label] += 1
            provenance_by_type[str(row.get("runtime_type") or "unknown")][label] += 1
    return {
        # Measured over profile-relative targets: each ladder is normalized by its own
        # maximum log-count, so a score of 1.0 is the profile ceiling (reached by
        # construction) rather than a clipped type reference, and the tag count is the
        # number of profiles carrying a normalization tag, not tagged decisions.
        "basis": "profile-targets",
        "decision_count": decision_count,
        "flat_menu_count": flat,
        "flat_menu_fraction": flat / decision_count if decision_count else 0.0,
        "clipped_level_count": clipped,
        "clipped_level_fraction": clipped / level_count if level_count else 0.0,
        "adjacent_delta_p": _distribution(deltas) | {
            "zero_count": sum(value == 0.0 for value in deltas),
        },
        "adjacent_delta_p_by_type": {
            runtime_type: _distribution(values) | {
                "zero_count": sum(value == 0.0 for value in values),
            }
            for runtime_type, values in sorted(by_type.items())
        },
        "provenance_counts": dict(sorted(provenance.items())),
        "provenance_by_type": {
            runtime_type: dict(sorted(values.items()))
            for runtime_type, values in sorted(provenance_by_type.items())
        },
        "provisional_tag_count": len(count_state.get("profile_tags", ())),
    }


def _sign_counts(values: Sequence[float]) -> dict[str, int]:
    return {
        "count": len(values),
        "zero_count": sum(value == 0.0 for value in values),
        "positive_count": sum(value > 0.0 for value in values),
        "negative_count": sum(value < 0.0 for value in values),
    }


def _credit_measurements(
    doc_ids: set[str],
    documents: Mapping[str, RankerDocument],
    utility_artifact: Mapping,
) -> dict[str, Any]:
    by_document = {}
    by_corpus: dict[str, Counter[str]] = defaultdict(Counter)
    for doc_id in sorted(doc_ids):
        document_row = utility_artifact["documents"][doc_id]
        rows = [
            utility_artifact["assertions"][assertion_id]
            for assertion_id in document_row["assertion_ids"]
            if utility_artifact["assertions"][assertion_id].get(
                "status", "accepted"
            ) == "accepted"
        ]
        linked_decisions = {
            decision_id
            for row in rows if row.get("credit_routing") == "linked"
            for decision_id in row.get("policy_dependency_decision_ids", ())
        }
        policy_decisions = set(document_row["policy_decision_ids"])
        record = {
            "linked_assertions": sum(
                row.get("credit_routing") == "linked" for row in rows
            ),
            "residual_assertions": sum(
                row.get("credit_routing") == "residual" for row in rows
            ),
            "fallback_decisions": len(policy_decisions - linked_decisions),
            "uncovered_decisions": len(
                document_row.get("uncovered_policy_decision_ids", ())
            ),
        }
        by_document[doc_id] = record
        by_corpus[documents[doc_id].corpus].update(record)
    totals = Counter()
    for record in by_document.values():
        totals.update(record)
    return {
        **{
            key: totals[key]
            for key in (
                "linked_assertions", "residual_assertions", "fallback_decisions",
                "uncovered_decisions",
            )
        },
        "by_document": by_document,
        "by_corpus": {
            corpus: dict(values) for corpus, values in sorted(by_corpus.items())
        },
    }


def build_diagnostic_spike(
    pool: Sequence[CalibrationPoint],
    *,
    documents: Mapping[str, RankerDocument],
    utility_artifact: Mapping,
    count_reward: ProfileCountTargets,
    count_state: Mapping,
    menu_artifact: Mapping,
    split_by_doc: Mapping[str, str],
    reader_jitter: Sequence[float] | Mapping[str, Sequence[float]],
    counterfactual_records: Sequence[Mapping],
) -> dict[str, Any]:
    """Emit the complete content-free preflight measurement set."""

    points = deduplicate_action_vectors(pool)
    if not points:
        raise ValueError("diagnostic spike requires a nonempty calibration pool")
    doc_ids = {point.doc_id for point in points}
    if set(split_by_doc) != doc_ids or not doc_ids <= set(documents):
        raise ValueError("diagnostic split/documents differ from calibration pool")
    grouped: dict[str, list[CalibrationPoint]] = defaultdict(list)
    for point in points:
        grouped[point.doc_id].append(point)
    frontiers = {
        doc_id: document_frontier(rows) for doc_id, rows in grouped.items()
    }
    switches = [
        switch.value for frontier in frontiers.values()
        for switch in frontier.switch_points
    ]
    utilities_by_doc = {
        doc_id: sorted(set(point.utility for point in rows))
        for doc_id, rows in grouped.items()
    }
    utility_steps = [
        right - left
        for values in utilities_by_doc.values()
        for left, right in zip(values, values[1:])
        if right > left
    ]
    delta_u = [float(row["delta_u"]) for row in counterfactual_records]
    decision_metadata = {
        decision.decision_id: (document.corpus, decision.runtime_type)
        for document in documents.values()
        for decision in document.policy_decisions
    }
    delta_by_type: dict[str, list[float]] = defaultdict(list)
    delta_by_corpus: dict[str, list[float]] = defaultdict(list)
    for row in counterfactual_records:
        decision_id = str(row["decision_id"])
        if decision_id not in decision_metadata:
            raise ValueError(f"counterfactual record has unknown decision: {decision_id}")
        corpus, runtime_type = decision_metadata[decision_id]
        delta_by_type[runtime_type].append(float(row["delta_u"]))
        delta_by_corpus[corpus].append(float(row["delta_u"]))
    corpus_docs: dict[str, set[str]] = defaultdict(set)
    type_decisions: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for point in points:
        corpus_docs[point.corpus].add(point.doc_id)
        source_counts.update(point.sources)
    for rows in grouped.values():
        type_decisions.update(rows[0].runtime_types)
    menu_report = menu_artifact.get("replay_report", {})
    winner_change = [float(value) for value in menu_report.get("winner_change", ())]
    if isinstance(reader_jitter, Mapping):
        jitter_by_split = {
            str(split): [float(value) for value in values]
            for split, values in reader_jitter.items()
        }
    else:
        observed_splits = set(split_by_doc.values())
        if reader_jitter and len(observed_splits) != 1:
            raise ValueError(
                "reader jitter must be split-keyed when calibration uses both splits"
            )
        split = next(iter(observed_splits)) if len(observed_splits) == 1 else "development"
        jitter_by_split = {split: [float(value) for value in reader_jitter]}
    if any(split not in {"train", "development"} for split in jitter_by_split):
        raise ValueError("reader jitter has an invalid split")
    jitter = [value for values in jitter_by_split.values() for value in values]
    count_measurements = _count_measurements(documents, count_reward, count_state)
    documents_by_corpus_by_split: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    decisions_by_type_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for doc_id, rows in grouped.items():
        split = split_by_doc[doc_id]
        documents_by_corpus_by_split[split][rows[0].corpus].add(doc_id)
        decisions_by_type_by_split[split].update(rows[0].runtime_types)
    measurements = {
        "trajectory_support": {
            "unique_trajectories": len(points),
            "unique_points": len({
                (point.doc_id, point.utility, point.count_score) for point in points
            }),
            "trajectories_by_document": {
                doc_id: len(rows) for doc_id, rows in sorted(grouped.items())
            },
            "distinct_points_by_document": {
                doc_id: frontier.unique_point_count
                for doc_id, frontier in sorted(frontiers.items())
            },
        },
        "frontier_switch_support": {
            "nondominated_by_document": {
                doc_id: len(frontier.nondominated)
                for doc_id, frontier in sorted(frontiers.items())
            },
            "positive_switches_by_document": {
                doc_id: len(frontier.switch_points)
                for doc_id, frontier in sorted(frontiers.items())
            },
            "switch_spread": _distribution(switches),
            "winner_change": winner_change,
            "winner_change_floor": min(winner_change, default=0.0),
            "positive_switch_documents_by_corpus": {
                corpus: sum(
                    bool(frontiers[doc_id].switch_points)
                    for doc_id in ids if doc_id in frontiers
                )
                for corpus, ids in sorted(corpus_docs.items())
            },
        },
        "winner_signatures": {
            "values": list(menu_artifact.get("values", ())),
            "signatures": list(menu_report.get("winner_signatures", ())),
        },
        "utility_resolution": {
            "quantization_step": min(utility_steps, default=0.0),
            "quantization_step_by_document": {
                doc_id: min(
                    (
                        right - left
                        for left, right in zip(values, values[1:])
                        if right > left
                    ),
                    default=0.0,
                )
                for doc_id, values in sorted(utilities_by_doc.items())
            },
            "reader_jitter": _distribution(jitter),
            "reader_jitter_abs_max": (
                max(abs(value) for value in jitter) if jitter else None
            ),
            "reader_jitter_by_split": {
                split: _distribution(values) | {
                    "abs_max": max(abs(value) for value in values) if values else None,
                }
                for split, values in sorted(jitter_by_split.items())
            },
        },
        "credit_coverage": _credit_measurements(
            doc_ids, documents, utility_artifact,
        ),
        "counterfactual_support": {
            **_sign_counts(delta_u),
            "nonzero_rate": (
                sum(value != 0.0 for value in delta_u) / len(delta_u)
                if delta_u else None
            ),
            "magnitude": _distribution([abs(value) for value in delta_u]),
            "by_type": {
                runtime_type: _sign_counts(values)
                for runtime_type, values in sorted(delta_by_type.items())
            },
            "by_corpus": {
                corpus: _sign_counts(values)
                for corpus, values in sorted(delta_by_corpus.items())
            },
        },
        "count_signal": count_measurements,
        "injectivity": _injectivity_measurements(points, documents, count_reward),
        "support": {
            "documents_by_corpus": {
                corpus: len(ids) for corpus, ids in sorted(corpus_docs.items())
            },
            "documents_by_corpus_by_split": {
                split: {
                    corpus: len(ids) for corpus, ids in sorted(corpora.items())
                }
                for split, corpora in sorted(documents_by_corpus_by_split.items())
            },
            "decisions_by_type": dict(sorted(type_decisions.items())),
            "decisions_by_type_by_split": {
                split: dict(sorted(values.items()))
                for split, values in sorted(decisions_by_type_by_split.items())
            },
            "trajectories_by_source": dict(sorted(source_counts.items())),
            "profiles": {
                str(value): len(doc_ids) for value in menu_artifact.get("values", ())
            },
            "profile_support": {
                str(value): (
                    menu_report.get("support", ())[index]
                    if index < len(menu_report.get("support", ()))
                    else {
                        "documents_by_corpus": {
                            corpus: len(ids)
                            for corpus, ids in sorted(corpus_docs.items())
                        },
                        "decisions_by_type": dict(sorted(type_decisions.items())),
                    }
                )
                for index, value in enumerate(menu_artifact.get("values", ()))
            },
            "decision_count": sum(
                len(documents[doc_id].policy_decisions) for doc_id in doc_ids
            ),
            "provenance": count_measurements["provenance_counts"],
            "provenance_by_type": count_measurements["provenance_by_type"],
            "split_documents": dict(sorted(Counter(split_by_doc.values()).items())),
        },
    }
    dataset_hash = stable_hash({
        "split_by_doc": dict(sorted(split_by_doc.items())),
        "points": [
            {
                "doc_id": point.doc_id,
                "ordered_action_vector": list(point.ordered_action_vector),
                "result_hash": point.result_hash,
            }
            for point in points
        ],
    })
    artifact: dict[str, Any] = {
        "artifact_version": SPIKE_VERSION,
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "dataset_hash": dataset_hash,
        "environment_hash": count_state.get("environment_hash"),
        "utility_artifact_hash": utility_artifact.get("artifact_hash"),
        "count_artifact_hash": count_state.get("artifact_hash"),
        "measurements": measurements,
    }
    artifact["artifact_hash"] = stable_hash(artifact)
    return artifact


def _path(value: Mapping, dotted: str) -> Any:
    current: Any = value
    for segment in dotted.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            raise ValueError(f"threshold measurement path is missing: {dotted}")
        current = current[segment]
    return current


def _resolve_candidate(rule: Mapping, spike: Mapping) -> int | float:
    kind = rule.get("kind")
    if kind == "fixed":
        value = rule.get("value")
    elif kind == "measurement":
        value = _path(spike, str(rule.get("path")))
    elif kind == "min_group_fraction":
        groups = _path(spike, str(rule.get("path")))
        if not isinstance(groups, Mapping) or not groups:
            raise ValueError("group threshold rule requires nonempty support")
        raw = min(float(value) for value in groups.values()) * float(rule["fraction"])
        value = max(int(rule.get("minimum", 0)), math.floor(raw))
    else:
        raise ValueError(f"unsupported threshold candidate rule: {kind}")
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError("threshold candidate did not produce a finite numeric value")
    return value


def freeze_threshold_manifest(
    rules: Mapping,
    spike: Mapping,
    *,
    pins: Mapping[str, str],
) -> dict[str, Any]:
    rules = validate_threshold_rules(rules)
    if spike.get("artifact_version") != SPIKE_VERSION:
        raise ValueError("threshold manifest requires a diagnostic spike")
    required_pins = {
        "reward_version", "environment_hash", "span_decision_artifact_hash",
        "utility_component_artifact_hash", "count_artifact_hash",
    }
    if set(pins) != required_pins or any(
        not isinstance(value, str) or not value for value in pins.values()
    ):
        raise ValueError("threshold manifest pins are incomplete")
    sections: dict[str, dict[str, Any]] = {
        "feasibility_gates": {}, "scheduler": {}, "acceptance": {},
    }
    applied = {}
    for field in _EMPIRICAL_FIELDS:
        section, name = field.split(".", 1)
        value = _resolve_candidate(rules["fields"][field]["candidate_rule"], spike)
        sections[section][name] = value
        applied[field] = {
            "value": value,
            "action": rules["fields"][field]["action"],
            "allowed_split": rules["fields"][field]["allowed_split"],
        }
    sections["acceptance"].update({
        "bootstrap_unit": "document",
        "utility_metrics": ["fixed_denominator_document_utility"],
    })
    manifest: dict[str, Any] = {
        "artifact_version": THRESHOLD_MANIFEST_VERSION,
        "diagnostic_version": DIAGNOSTIC_VERSION,
        **dict(pins),
        "diagnostic_dataset_hash": spike.get("dataset_hash"),
        "spike_output_hash": spike.get("artifact_hash"),
        "threshold_rules_hash": stable_hash(rules),
        "hard_gates": {
            "explicit_count_coverage": 1.0,
            "fallback_count_gradient_mass": 0.0,
            "missing_occurrence_decision_mappings": 0,
            "nonmonotone_profiles": 0,
            "lambda_zero_identity": "exact",
        },
        **sections,
        "applied_rules": applied,
    }
    manifest["artifact_hash"] = stable_hash(manifest)
    return manifest


def cache_only_missing_report(
    trajectories: Sequence[CalibrationTrajectory],
    *,
    documents: Mapping[str, RankerDocument],
    utility_artifact: Mapping,
    environment_hash: str,
    cache: UtilityCache,
) -> dict[str, Any]:
    """Resolve exact reward identities and report misses without dispatching work."""

    from cloak.train.roundtrip import _cache_identity

    unique: dict[tuple[str, tuple[tuple[str, str], ...]], CalibrationTrajectory] = {}
    for trajectory in trajectories:
        key = (trajectory.doc_id, trajectory.ordered_action_vector)
        previous = unique.get(key)
        if previous is None:
            unique[key] = trajectory
        else:
            unique[key] = CalibrationTrajectory(
                doc_id=trajectory.doc_id,
                corpus=trajectory.corpus,
                sources=tuple(sorted(set(previous.sources) | set(trajectory.sources))),
                ordered_action_vector=trajectory.ordered_action_vector,
                action_modes=trajectory.action_modes,
                runtime_types=trajectory.runtime_types,
            )
    missing = []
    hits = 0
    context_items = 0
    for key in sorted(unique):
        trajectory = unique[key]
        try:
            document = documents[trajectory.doc_id]
        except KeyError as error:
            raise ValueError(
                f"calibration trajectory document is unavailable: {trajectory.doc_id}"
            ) from error
        request = UtilityRequest(
            document=document,
            action_vector=trajectory.action_vector,
            utility_artifact=utility_artifact,
            environment_hash=environment_hash,
        )
        doc_p, _ = assemble_action_vector(document, trajectory.action_vector)
        identity = _cache_identity(request, doc_p, reader_refresh=False)
        if cache.lookup(identity) is not None:
            hits += 1
            continue
        reader_items = sum(
            row.get("doc_id") == trajectory.doc_id
            and row.get("status", "accepted") == "accepted"
            and row.get("family") == "context"
            for row in utility_artifact.get("assertions", {}).values()
        )
        context_items += reader_items
        missing.append({
            "doc_id": trajectory.doc_id,
            "sources": list(trajectory.sources),
            "ordered_action_vector": [list(row) for row in trajectory.ordered_action_vector],
            "action_vector_hash": stable_hash(list(trajectory.ordered_action_vector)),
            "remote_tasks": 1,
            "context_reader_work_items": reader_items,
        })
    return {
        "report_version": "ranker-v2-cache-only-preflight-v1",
        "required_action_vector_count": len(unique),
        "cache_hits": hits,
        "missing_action_vector_count": len(missing),
        "remote_tasks": len(missing),
        "context_reader_work_items": context_items,
        "missing": missing,
        "dispatched": False,
    }


def reader_jitter_from_cache(
    cache: UtilityCache,
    *,
    utility_artifact: Mapping,
    split_by_doc: Mapping[str, str],
) -> dict[str, tuple[float, ...]]:
    """Return paired refresh-minus-base utility differences by allowed split."""

    paired: dict[tuple[Any, ...], dict[bool, Any]] = defaultdict(dict)
    for identity, result in cache.entries.values():
        doc_id = str(identity.get("doc_id"))
        if doc_id not in split_by_doc:
            continue
        refresh = identity.get("reader_refresh")
        ordered = identity.get("ordered_action_vector")
        if not isinstance(refresh, bool) or not isinstance(ordered, list):
            continue
        key = (
            doc_id,
            tuple(tuple(str(value) for value in pair) for pair in ordered),
            identity.get("environment_hash"),
            identity.get("utility_artifact_hash"),
        )
        if refresh in paired[key]:
            raise ValueError("utility cache repeats a reader-refresh identity")
        paired[key][refresh] = result
    by_split: dict[str, list[float]] = {
        split: [] for split in sorted(set(split_by_doc.values()))
    }
    for (doc_id, *_), rows in paired.items():
        if set(rows) != {False, True}:
            continue
        base = document_utility(
            rows[False].component_scores, utility_artifact, doc_id,
        )
        refreshed = document_utility(
            rows[True].component_scores, utility_artifact, doc_id,
        )
        by_split[split_by_doc[doc_id]].append(refreshed - base)
    return {
        split: tuple(values) for split, values in sorted(by_split.items())
    }


def build_privacy_diagnostic_manifest(
    seed_reports: Sequence[Mapping],
    *,
    split_manifest_hash: str,
    metric_report_hash: str,
    counterexample_report: Mapping | None = None,
) -> dict[str, Any]:
    """Build profile-paired controller gates and a separate distributional audit."""
    if not seed_reports:
        raise ValueError("privacy diagnostics require seed reports")
    if not split_manifest_hash or not metric_report_hash:
        raise ValueError("privacy diagnostic hashes are required")

    seed_verdicts = []
    frozen_reports = []
    expected_baselines = set(REQUIRED_BASELINES)

    def finite(metric: str, values: Mapping) -> float | None:
        value = values.get(metric)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
        ):
            return None
        return float(value)

    for report in seed_reports:
        if report.get("profile_held_out") is not True:
            raise ValueError("privacy diagnostics require profile-held-out reports")
        splits = report.get("splits")
        if not isinstance(splits, Mapping) or set(splits) != {"dev", "test"}:
            raise ValueError("privacy diagnostics require dev and test splits")
        for split_name, split_report in splits.items():
            if not isinstance(split_report, Mapping):
                raise ValueError(f"privacy {split_name} report must be a mapping")
            semantic = split_report.get("semantic")
            baselines = split_report.get("baselines")
            if not isinstance(semantic, Mapping) or not isinstance(baselines, Mapping):
                raise ValueError(f"privacy {split_name} report is incomplete")
            if set(baselines) != expected_baselines:
                raise ValueError(f"privacy {split_name} baseline coverage mismatch")
            for name, metrics in {"semantic": semantic, **baselines}.items():
                if not isinstance(metrics, Mapping):
                    raise ValueError(f"privacy metrics are invalid for {name}")
                for stratum in (
                    "overall", "by_runtime_type", "by_grounding_status",
                    "by_profile",
                ):
                    if not isinstance(metrics.get(stratum), Mapping) or not metrics[stratum]:
                        raise ValueError(
                            f"privacy metrics omit {stratum} for {split_name}:{name}"
                        )

        test_report = splits["test"]
        semantic_overall = test_report["semantic"]["overall"]
        comparisons = {}

        candidate = test_report["baselines"]["candidate_only"]["overall"]
        semantic_calibration = finite(
            "profile_relative_calibration_error", semantic_overall
        )
        semantic_ordering = finite(
            "within_menu_pairwise_accuracy", semantic_overall
        )
        semantic_regret = finite("selected_action_regret", semantic_overall)
        candidate_calibration = finite(
            "profile_relative_calibration_error", candidate
        )
        candidate_ordering = finite("within_menu_pairwise_accuracy", candidate)
        candidate_regret = finite("selected_action_regret", candidate)
        candidate_metrics = {
            "profile_relative_calibration_error": (
                semantic_calibration is not None
                and candidate_calibration is not None
                and semantic_calibration
                <= candidate_calibration + PRIVACY_MARGINS["candidate_calibration"]
            ),
            "within_menu_pairwise_accuracy": (
                semantic_ordering is not None
                and candidate_ordering is not None
                and semantic_ordering
                >= candidate_ordering - PRIVACY_MARGINS["candidate_ordering"]
            ),
            "selected_action_regret": (
                semantic_regret is not None
                and candidate_regret is not None
                and semantic_regret
                <= candidate_regret + PRIVACY_MARGINS["candidate_regret"]
            ),
        }
        improves_candidate = (
            semantic_calibration is not None
            and candidate_calibration is not None
            and semantic_calibration < candidate_calibration
        ) or (
            semantic_ordering is not None
            and candidate_ordering is not None
            and semantic_ordering > candidate_ordering
        )
        comparisons["candidate_only"] = {
            "role": "competitive_baseline",
            "blocking_metrics": candidate_metrics,
            "improves_at_least_one_primary": improves_candidate,
            "verdict": "REPORT_ONLY_POINT_ESTIMATE",
        }

        authored = test_report["baselines"][
            "authored_position_mode_type"
        ]["overall"]
        authored_calibration = finite(
            "profile_relative_calibration_error", authored
        )
        authored_regret = finite("selected_action_regret", authored)
        authored_metrics = {
            "profile_relative_calibration_error": (
                semantic_calibration is not None
                and authored_calibration is not None
                and semantic_calibration
                <= authored_calibration
                + PRIVACY_MARGINS["authored_calibration"]
            ),
            "selected_action_regret": (
                semantic_regret is not None
                and authored_regret is not None
                and semantic_regret
                <= authored_regret + PRIVACY_MARGINS["authored_regret"]
            ),
        }
        comparisons["authored_position_mode_type"] = {
            "role": "oracle_ceiling",
            "noninferiority_margins": {
                "profile_relative_calibration_error": PRIVACY_MARGINS[
                    "authored_calibration"
                ],
                "selected_action_regret": PRIVACY_MARGINS["authored_regret"],
            },
            "blocking_metrics": authored_metrics,
            "ordering": "reference_only",
            "verdict": "PASS" if all(authored_metrics.values()) else "FAIL",
        }

        for baseline_name in ("mode_type_only", "train_profile_mean"):
            baseline = test_report["baselines"][baseline_name]["overall"]
            baseline_calibration = finite(
                "profile_relative_calibration_error", baseline
            )
            baseline_regret = finite("selected_action_regret", baseline)
            sanity_metrics = {
                "profile_relative_calibration_error": (
                    semantic_calibration is not None
                    and baseline_calibration is not None
                    and semantic_calibration < baseline_calibration
                ),
                "selected_action_regret": (
                    semantic_regret is not None
                    and baseline_regret is not None
                    and semantic_regret < baseline_regret
                ),
            }
            comparisons[baseline_name] = {
                "role": "sanity_floor",
                "blocking_metrics": sanity_metrics,
                "verdict": "PASS" if all(sanity_metrics.values()) else "FAIL",
            }
        seed_verdicts.append({
            "seed": report.get("seed"),
            "comparisons": comparisons,
            "verdict": "REPORT_ONLY_POINT_ESTIMATE",
        })
        frozen_reports.append(deepcopy(dict(report)))

    def profile_improvements(
        baseline_name: str, metric: str, *, higher_is_better: bool
    ) -> dict[str, float] | None:
        per_profile: dict[str, list[float]] = defaultdict(list)
        expected_profiles: set[str] | None = None
        for report in seed_reports:
            test = report["splits"]["test"]
            semantic = test["semantic"]["by_profile"]
            baseline = test["baselines"][baseline_name]["by_profile"]
            if set(semantic) != set(baseline):
                return None
            if expected_profiles is None:
                expected_profiles = set(semantic)
            elif set(semantic) != expected_profiles:
                return None
            for profile_id in semantic:
                semantic_value = finite(metric, semantic[profile_id])
                baseline_value = finite(metric, baseline[profile_id])
                if semantic_value is None or baseline_value is None:
                    return None
                improvement = (
                    semantic_value - baseline_value
                    if higher_is_better
                    else baseline_value - semantic_value
                )
                per_profile[profile_id].append(improvement)
        return {
            profile_id: sum(values) / len(values)
            for profile_id, values in sorted(per_profile.items())
        }

    def paired_bootstrap(values: Mapping[str, float]) -> dict[str, Any]:
        ordered = tuple(values[key] for key in sorted(values))
        if not ordered:
            raise ValueError("privacy bootstrap requires profile deltas")
        generator = random.Random(PRIVACY_BOOTSTRAP_SEED)
        estimates = []
        for _ in range(PRIVACY_BOOTSTRAP_SAMPLES):
            estimates.append(sum(
                ordered[generator.randrange(len(ordered))]
                for _ in range(len(ordered))
            ) / len(ordered))
        estimates.sort()
        lower = estimates[int(0.025 * (len(estimates) - 1))]
        upper = estimates[int(0.975 * (len(estimates) - 1))]
        return {
            "mean": sum(ordered) / len(ordered),
            "ci_95": [lower, upper],
        }

    paired_values = {
        "profile_relative_calibration_improvement": profile_improvements(
            "candidate_only",
            "profile_relative_calibration_error",
            higher_is_better=False,
        ),
        "within_menu_ordering_improvement": profile_improvements(
            "candidate_only",
            "within_menu_pairwise_accuracy",
            higher_is_better=True,
        ),
        "selected_action_regret_improvement": profile_improvements(
            "candidate_only",
            "selected_action_regret",
            higher_is_better=False,
        ),
        "authored_calibration_improvement": profile_improvements(
            "authored_position_mode_type",
            "profile_relative_calibration_error",
            higher_is_better=False,
        ),
        "authored_regret_improvement": profile_improvements(
            "authored_position_mode_type",
            "selected_action_regret",
            higher_is_better=False,
        ),
    }
    overall_metrics_valid = all(
        finite(metric, report["splits"]["test"]["semantic"]["overall"]) is not None
        for report in seed_reports
        for metric in (
            "profile_relative_calibration_error",
            "within_menu_pairwise_accuracy",
            "selected_action_regret",
        )
    )
    paired_valid = all(value is not None for value in paired_values.values())
    paired = (
        {
            name: paired_bootstrap(value)
            for name, value in paired_values.items()
            if value is not None
        }
        if paired_valid else {}
    )
    profile_count = (
        len(next(iter(paired_values.values())))
        if paired_valid else 0
    )
    if paired_valid:
        calibration = paired["profile_relative_calibration_improvement"]
        ordering = paired["within_menu_ordering_improvement"]
        regret = paired["selected_action_regret_improvement"]
        authored_calibration = paired["authored_calibration_improvement"]
        authored_regret = paired["authored_regret_improvement"]
        primary_pass = (
            calibration["ci_95"][0] > 0
            and ordering["ci_95"][0]
            >= -PRIVACY_MARGINS["candidate_ordering"]
        ) or (
            ordering["ci_95"][0] > 0
            and calibration["ci_95"][0]
            >= -PRIVACY_MARGINS["candidate_calibration"]
        )
        regret_pass = (
            regret["ci_95"][0] >= -PRIVACY_MARGINS["candidate_regret"]
        )
        authored_pass = (
            authored_calibration["ci_95"][0]
            >= -PRIVACY_MARGINS["authored_calibration"]
            and authored_regret["ci_95"][0]
            >= -PRIVACY_MARGINS["authored_regret"]
        )
    else:
        primary_pass = regret_pass = authored_pass = False

    sanity_pass = all(
        comparison["verdict"] == "PASS"
        for seed_row in seed_verdicts
        for name, comparison in seed_row["comparisons"].items()
        if name in {"mode_type_only", "train_profile_mean"}
    )
    controller_pass = (
        overall_metrics_valid
        and paired_valid
        and primary_pass
        and regret_pass
        and authored_pass
        and sanity_pass
    )
    if not overall_metrics_valid or not paired_valid:
        controller_verdict = "FAIL"
    elif len(seed_reports) < 3:
        controller_verdict = "NEEDS_MULTI_SEED_EVIDENCE"
    elif not controller_pass:
        controller_verdict = "FAIL"
    else:
        controller_verdict = "PASS"

    if counterexample_report is None:
        counterexample = {
            "status": "N/A",
            "reason": "counterexample_set_absent",
            "artifact_hash": None,
        }
    else:
        counterexample_hash = counterexample_report.get("artifact_hash")
        counterexample_verdict = counterexample_report.get("verdict")
        if (
            not isinstance(counterexample_hash, str)
            or not counterexample_hash
            or counterexample_verdict not in {"PASS", "FAIL"}
        ):
            raise ValueError("privacy counterexample report is invalid")
        counterexample = {
            "status": counterexample_verdict,
            "artifact_hash": counterexample_hash,
        }

    counterexample_pass = counterexample["status"] == "PASS"
    if controller_verdict == "FAIL" or counterexample["status"] == "FAIL":
        promotion_verdict = "FAIL"
    elif controller_verdict == "NEEDS_MULTI_SEED_EVIDENCE":
        promotion_verdict = (
            "NEEDS_MULTI_SEED_EVIDENCE"
            if counterexample_pass
            else "NEEDS_MULTI_SEED_AND_COUNTEREXAMPLE_SET"
        )
    elif not counterexample_pass:
        promotion_verdict = "NEEDS_COUNTEREXAMPLE_SET"
    else:
        promotion_verdict = "PROMOTE"

    manifest = {
        "artifact_version": PRIVACY_DIAGNOSTIC_VERSION,
        "profile_held_out": True,
        "aci_document_generalization_claimed": False,
        "split_manifest_hash": split_manifest_hash,
        "metric_report_hash": metric_report_hash,
        "required_baselines": list(REQUIRED_BASELINES),
        "seed_count": len(seed_reports),
        "preregistered_margins": dict(PRIVACY_MARGINS),
        "blocking_metrics": [
            "profile_relative_calibration_error",
            "within_menu_pairwise_accuracy",
            "selected_action_regret",
            "lexical_semantic_counterexamples",
        ],
        "report_only_metrics": [
            "nll",
            "interval_95_coverage",
            "sigma_fixed",
            "median_absolute_log_error",
        ],
        "seed_reports": frozen_reports,
        "policy_fitness_gate": {
            "controller_metrics_verdict": controller_verdict,
            "candidate_only_paired_bootstrap": {
                "bootstrap_seed": PRIVACY_BOOTSTRAP_SEED,
                "bootstrap_samples": PRIVACY_BOOTSTRAP_SAMPLES,
                "profile_count": profile_count,
                **paired,
            },
            "lexical_counterexamples": counterexample,
        },
        "distributional_audit_gate": {
            "metrics": ["nll", "interval_95_coverage", "sigma_fixed"],
            "verdict": "REPORT_ONLY",
            "count_and_interval_claims_allowed": False,
        },
        "relative_promotion": {
            "seed_verdicts": seed_verdicts,
            "verdict": promotion_verdict,
        },
    }
    manifest["artifact_hash"] = stable_hash(manifest)
    return manifest
