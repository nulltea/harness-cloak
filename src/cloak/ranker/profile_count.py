"""Strict own-profile count targets for the Ranker-v2 semantic privacy head."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import log
from numbers import Integral, Real

import torch

from cloak.ranker.count_reward import (
    CountGateError,
    _evidence_ref,
    _frozen_environment,
    _iter_decisions,
    _level_clause_results,
    _policy_mapping_errors,
)


ARTIFACT_VERSION = "ranker-v2-profile-count-targets-v1"
REPORT_VERSION = "ranker-v2-profile-count-gate-v1"
SINGLETON_TAG = "singleton_profile_normalization"
_MENU_GAP_REASONS = frozenset({
    "keep_endpoint", "placeholder_endpoint", "supported_action_modes", "level_actions",
})


@dataclass(frozen=True)
class ProfileActionTarget:
    decision_id: str
    action_id: str
    profile_id: str
    runtime_type: str
    mode: str
    log_count: float | None
    profile_score: float
    grounding_status: str | None
    source_family: str | None


def _canonical_hash(payload: Mapping) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _legal_actions(decision: Mapping) -> list[Mapping]:
    return [
        action for action in decision.get("actions", [])
        if action.get("legal", True)
    ]


def _authored_levels(levels: Sequence[Mapping]) -> tuple[list[Mapping], list[str]]:
    indices = [action.get("authored_level_index") for action in levels]
    if any(
        not isinstance(index, Integral) or isinstance(index, bool) or int(index) < 0
        for index in indices
    ):
        return list(levels), ["invalid_authored_level_index"]
    normalized = [int(index) for index in indices]
    if len(set(normalized)) != len(normalized):
        return list(levels), ["duplicate_authored_level_index"]
    return [
        action for _, action in sorted(zip(normalized, levels), key=lambda pair: pair[0])
    ], []


def _profile_identity(decision: Mapping) -> str:
    # One profile is intentionally reused by decisions whose canonical source text
    # is an alias (for example, Lipitor and atorvastatin). Only a cross-type reuse
    # is an identity collision; source aliases and repeated documents are not.
    return str(decision.get("runtime_type", ""))


def _gate_state(environment: Mapping, *, mode: str) -> tuple[dict, list[dict]]:
    frozen = _frozen_environment(environment)
    decision_states: list[dict] = []
    decision_ids: Counter[str] = Counter()
    action_ids: Counter[str] = Counter()
    profile_identities: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for doc_id, _document, _occurrences, decision in _iter_decisions(environment):
        decision_id = str(decision.get("decision_id"))
        decision_ids[decision_id] += 1
        profile_value = decision.get("profile_id")
        profile_id = str(profile_value) if profile_value is not None else ""
        runtime_type = str(decision.get("runtime_type", ""))
        legal_actions = _legal_actions(decision)
        for action in legal_actions:
            action_ids[str(action.get("action_id"))] += 1

        levels = [action for action in legal_actions if action.get("mode") == "level"]
        ordered_levels, authored_errors = _authored_levels(levels)
        reasons = list(authored_errors)
        if not profile_id:
            reasons.append("matched_profile")
        if profile_id:
            profile_identities[profile_id][_profile_identity(decision)].append(decision_id)

        mode_counts = Counter(str(action.get("mode", "")) for action in legal_actions)
        if mode_counts["keep"] != 1:
            reasons.append("keep_endpoint")
        if mode_counts["placeholder"] != 1:
            reasons.append("placeholder_endpoint")
        if sum(mode_counts.values()) != sum(
            mode_counts[name] for name in ("level", "keep", "placeholder")
        ):
            reasons.append("supported_action_modes")
        if not levels:
            reasons.append("level_actions")

        level_reports = []
        log_counts: dict[str, float] = {}
        numeric_counts: list[tuple[str, float]] = []
        for action in ordered_levels:
            action_id = str(action.get("action_id"))
            grounding = action.get("count_grounding")
            clause_results, gap_reasons = _level_clause_results(action)
            if not profile_id:
                gap_reasons = [*gap_reasons, "matched_profile"]
            admitted = not gap_reasons
            count = action.get("count")
            numeric_count = (
                float(count)
                if isinstance(count, Real) and not isinstance(count, bool)
                else None
            )
            if admitted and numeric_count is not None:
                log_counts[action_id] = log(max(numeric_count, 1.0))
            if numeric_count is not None:
                numeric_counts.append((action_id, numeric_count))
            level_report = {
                "doc_id": str(doc_id),
                "decision_id": decision_id,
                "action_id": action_id,
                "profile_id": profile_value,
                "runtime_type": runtime_type,
                "fill": action.get("fill"),
                "count": numeric_count,
                "grounding_status": (
                    grounding.get("status") if isinstance(grounding, Mapping) else None
                ),
                "source_family": (
                    grounding.get("source_family")
                    if isinstance(grounding, Mapping) else None
                ),
                "evidence_ref": _evidence_ref(grounding),
                "clause_results": clause_results,
                "admitted": admitted,
                "gap_reasons": sorted(set(gap_reasons)),
            }
            level_reports.append(level_report)
            reasons.extend(gap_reasons)

        decreases = [
            {
                "left_action_id": left_id,
                "left_count": left_count,
                "right_action_id": right_id,
                "right_count": right_count,
            }
            for (left_id, left_count), (right_id, right_count)
            in zip(numeric_counts, numeric_counts[1:])
            if right_count < left_count
        ]
        if decreases:
            reasons.append("nonmonotone_authored_ladder")

        denominator = max(log_counts.values()) if log_counts else None
        if len(levels) >= 2 and len(log_counts) == len(levels) and denominator == 0.0:
            reasons.append("zero_profile_denominator")

        decision_states.append({
            "doc_id": str(doc_id),
            "decision": decision,
            "decision_id": decision_id,
            "profile_id": profile_id,
            "profile_value": profile_value,
            "runtime_type": runtime_type,
            "legal_actions": legal_actions,
            "ordered_levels": ordered_levels,
            "level_reports": level_reports,
            "log_counts": log_counts,
            "denominator": denominator,
            "decreases": decreases,
            "reasons": reasons,
            "singleton": len(levels) == 1,
        })

    duplicate_profiles = []
    for profile_id, identities in sorted(profile_identities.items()):
        if len(identities) <= 1:
            continue
        affected = sorted(
            decision_id for ids in identities.values() for decision_id in ids
        )
        duplicate_profiles.append({
            "profile_id": profile_id,
            "identities": [
                {
                    "runtime_type": identity,
                    "decision_ids": sorted(ids),
                }
                for identity, ids in sorted(identities.items())
            ],
            "decision_ids": affected,
        })
        for state in decision_states:
            if state["decision_id"] in affected:
                state["reasons"].append("duplicate_profile_id")

    duplicate_decision_id_values = sorted(
        key for key, count in decision_ids.items() if count > 1
    )
    duplicate_action_id_values = sorted(
        key for key, count in action_ids.items() if count > 1
    )
    for state in decision_states:
        if state["decision_id"] in duplicate_decision_id_values:
            state["reasons"].append("duplicate_decision_id")
        if any(
            str(action.get("action_id")) in duplicate_action_id_values
            for action in state["legal_actions"]
        ):
            state["reasons"].append("duplicate_action_id")

    mapping_errors = _policy_mapping_errors(environment)
    mapping_decisions = {
        str(row["decision_id"])
        for row in mapping_errors
        if row.get("decision_id") is not None
    }
    documents_with_mapping_errors = {
        str(row["doc_id"]) for row in mapping_errors if row.get("decision_id") is None
    }
    for state in decision_states:
        if (
            state["decision_id"] in mapping_decisions
            or state["doc_id"] in documents_with_mapping_errors
        ):
            state["reasons"].append("policy_mapping")

    for state in decision_states:
        state["reasons"] = sorted(set(state["reasons"]))
        state["privacy_head_eligible"] = not state["reasons"]

    levels = [row for state in decision_states for row in state["level_reports"]]
    gaps = [row for row in levels if not row["admitted"]]
    nonmonotone = [
        {
            "doc_id": state["doc_id"],
            "decision_id": state["decision_id"],
            "profile_id": state["profile_value"],
            "decreases": state["decreases"],
        }
        for state in decision_states if state["decreases"]
    ]
    zero_denominators = [
        {
            "doc_id": state["doc_id"],
            "decision_id": state["decision_id"],
            "profile_id": state["profile_value"],
        }
        for state in decision_states
        if "zero_profile_denominator" in state["reasons"]
    ]
    eligible_count = sum(state["privacy_head_eligible"] for state in decision_states)
    ineligible_count = len(decision_states) - eligible_count
    clauses = [
        {
            "clause": "explicit_coverage",
            "result": "PASS" if not gaps else "FAIL",
            "observed": len(gaps),
            "required": 0,
        },
        {
            "clause": "matched_profiles",
            "result": "PASS" if all(state["profile_id"] for state in decision_states) else "FAIL",
            "observed": sum(not state["profile_id"] for state in decision_states),
            "required": 0,
        },
        {
            "clause": "duplicate_profile_ids",
            "result": "PASS" if not duplicate_profiles else "FAIL",
            "observed": len(duplicate_profiles),
            "required": 0,
        },
        {
            "clause": "nonmonotone_authored_ladders",
            "result": "PASS" if not nonmonotone else "FAIL",
            "observed": len(nonmonotone),
            "required": 0,
        },
        {
            "clause": "positive_profile_denominators",
            "result": "PASS" if not zero_denominators else "FAIL",
            "observed": len(zero_denominators),
            "required": 0,
        },
        {
            "clause": "unique_artifact_ids",
            "result": "PASS" if not (
                duplicate_decision_id_values or duplicate_action_id_values
            ) else "FAIL",
            "observed": len(duplicate_decision_id_values) + len(duplicate_action_id_values),
            "required": 0,
        },
        {
            "clause": "complete_action_menus",
            "result": "PASS" if all(
                not (_MENU_GAP_REASONS & set(state["reasons"]))
                for state in decision_states
            ) else "FAIL",
            "observed": sum(
                bool(_MENU_GAP_REASONS & set(state["reasons"]))
                for state in decision_states
            ),
            "required": 0,
        },
        {
            "clause": "policy_mappings",
            "result": "PASS" if not mapping_errors else "FAIL",
            "observed": len(mapping_errors),
            "required": 0,
        },
    ]
    strict_verdict = "PASS" if all(
        row["result"] == "PASS" for row in clauses
    ) else "FAIL"
    by_type = {}
    for runtime_type in sorted({state["runtime_type"] for state in decision_states}):
        rows = [state for state in decision_states if state["runtime_type"] == runtime_type]
        by_type[runtime_type] = {
            "decisions": len(rows),
            "privacy_head_eligible": sum(row["privacy_head_eligible"] for row in rows),
            "privacy_head_ineligible": sum(not row["privacy_head_eligible"] for row in rows),
            "singleton_profile_normalizations": sum(
                row["singleton"] and row["privacy_head_eligible"] for row in rows
            ),
        }

    report = {
        "report_version": REPORT_VERSION,
        "environment_hash": frozen.get("environment_hash"),
        "mode": mode,
        "verdict": strict_verdict,
        "strict_verdict": strict_verdict,
        "clauses": clauses,
        "summary": {
            "ranker_selectable_decisions": len(decision_states),
            "privacy_head_eligible_decisions": eligible_count,
            "privacy_head_ineligible_decisions": ineligible_count,
            "level_actions": len(levels),
            "admitted_level_actions": len(levels) - len(gaps),
            "gap_count": len(gaps),
            "singleton_profile_normalizations": sum(
                state["singleton"] and state["privacy_head_eligible"]
                for state in decision_states
            ),
        },
        "by_runtime_type": by_type,
        "decisions": [
            {
                "doc_id": state["doc_id"],
                "decision_id": state["decision_id"],
                "profile_id": state["profile_value"],
                "runtime_type": state["runtime_type"],
                "privacy_head_eligible": state["privacy_head_eligible"],
                "gap_reasons": state["reasons"],
                "level_action_count": len(state["ordered_levels"]),
                "profile_log_count_denominator": state["denominator"],
                "normalization_tags": (
                    [SINGLETON_TAG]
                    if state["singleton"] and state["privacy_head_eligible"] else []
                ),
            }
            for state in decision_states
        ],
        "levels": levels,
        "gaps": gaps,
        "duplicate_profile_ids": duplicate_profiles,
        "duplicate_decision_ids": duplicate_decision_id_values,
        "duplicate_action_ids": duplicate_action_id_values,
        "nonmonotone_profiles": nonmonotone,
        "zero_denominator_profiles": zero_denominators,
        "missing_policy_mappings": mapping_errors,
    }
    return report, decision_states


def build_profile_count_targets(environment: Mapping, *, strict: bool) -> dict:
    """Build profile-relative exact targets or a diagnostic-only partial artifact."""
    mode = "strict" if strict else "diagnostic"
    report, decision_states = _gate_state(environment, mode=mode)
    if strict and report["strict_verdict"] != "PASS":
        raise CountGateError(report)

    action_targets: dict[str, dict] = {}
    decision_actions: dict[str, list[str]] = {}
    decision_eligibility: dict[str, bool] = {}
    profile_tags: defaultdict[str, set[str]] = defaultdict(set)
    for state in decision_states:
        decision_id = state["decision_id"]
        profile_id = state["profile_id"]
        runtime_type = state["runtime_type"]
        eligible = state["privacy_head_eligible"]
        decision_eligibility[decision_id] = eligible
        decision_actions[decision_id] = [
            str(action["action_id"]) for action in state["legal_actions"]
        ]
        if eligible and state["singleton"]:
            profile_tags[profile_id].add(SINGLETON_TAG)

        for action in state["legal_actions"]:
            action_id = str(action["action_id"])
            mode_name = str(action.get("mode", ""))
            grounding = action.get("count_grounding")
            if mode_name == "keep":
                log_count = None
                score = 0.0
            elif mode_name == "placeholder":
                log_count = None
                score = 1.0
            elif mode_name == "level" and eligible:
                log_count = state["log_counts"][action_id]
                score = (
                    1.0
                    if state["singleton"]
                    else log_count / float(state["denominator"])
                )
            else:
                continue
            row = ProfileActionTarget(
                decision_id=decision_id,
                action_id=action_id,
                profile_id=profile_id,
                runtime_type=runtime_type,
                mode=mode_name,
                log_count=log_count,
                profile_score=float(score),
                grounding_status=(
                    grounding.get("status")
                    if mode_name == "level" and isinstance(grounding, Mapping) else None
                ),
                source_family=(
                    grounding.get("source_family")
                    if mode_name == "level" and isinstance(grounding, Mapping) else None
                ),
            )
            action_targets[action_id] = asdict(row)

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "environment_hash": report["environment_hash"],
        "gate_mode": mode,
        "decision_actions": decision_actions,
        "decision_eligibility": decision_eligibility,
        "action_targets": action_targets,
        "profile_tags": {
            profile_id: sorted(tags) for profile_id, tags in sorted(profile_tags.items())
        },
        "gate_report": report,
    }
    artifact["artifact_hash"] = _canonical_hash(artifact)
    return artifact


class ProfileCountTargets:
    """Restricted runtime view of exact profile-count supervision.

    ``action_scores`` and ``selected_document_score`` are for the exact shaping
    objective, calibration, and diagnostics only. ``target_rows`` is for privacy-head
    training, calibration, and diagnostics only. None is a policy feature interface:
    actor code must never receive raw/log counts, exact profile scores, authored
    indices, or menu sizes from this object.
    """

    def __init__(
        self,
        actions: Mapping[str, ProfileActionTarget],
        decision_actions: Mapping[str, Sequence[str]],
        decision_eligibility: Mapping[str, bool],
    ):
        self._actions = dict(actions)
        self._decision_actions = {
            str(key): tuple(str(value) for value in values)
            for key, values in decision_actions.items()
        }
        self._decision_eligibility = {
            str(key): bool(value) for key, value in decision_eligibility.items()
        }

    @classmethod
    def from_artifact(cls, payload: Mapping) -> "ProfileCountTargets":
        if payload.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("unsupported profile count target artifact version")
        if payload.get("artifact_hash"):
            unhashed = {key: value for key, value in payload.items() if key != "artifact_hash"}
            if payload["artifact_hash"] != _canonical_hash(unhashed):
                raise ValueError("profile count target artifact hash mismatch")

        raw_actions = payload.get("action_targets")
        raw_decisions = payload.get("decision_actions")
        eligibility = payload.get("decision_eligibility")
        if not isinstance(raw_actions, Mapping):
            raise ValueError("profile count artifact is missing action_targets")
        if not isinstance(raw_decisions, Mapping):
            raise ValueError("profile count artifact is missing decision_actions")
        if not isinstance(eligibility, Mapping) or set(eligibility) != set(raw_decisions):
            raise ValueError("profile count artifact has invalid decision_eligibility")

        actions = {
            str(action_id): ProfileActionTarget(**row)
            for action_id, row in raw_actions.items()
        }
        for action_id, row in actions.items():
            if row.action_id != action_id:
                raise ValueError(f"profile action key mismatch: {action_id}")

        seen = set()
        for decision_id, registered in raw_decisions.items():
            if isinstance(registered, (str, bytes)) or not isinstance(registered, Sequence):
                raise ValueError(f"invalid profile action list for {decision_id}")
            for action_id in registered:
                action_id = str(action_id)
                if action_id in seen:
                    raise ValueError(f"duplicate profile action id: {action_id}")
                seen.add(action_id)
                row = actions.get(action_id)
                if row is not None and row.decision_id != str(decision_id):
                    raise ValueError(
                        f"profile action decision mismatch: {decision_id}:{action_id}"
                    )
            if bool(eligibility[decision_id]) and any(
                str(action_id) not in actions for action_id in registered
            ):
                raise ValueError(f"eligible profile decision has missing targets: {decision_id}")
        extra = sorted(set(actions) - seen)
        if extra:
            raise ValueError(f"unregistered profile action targets: {extra}")
        return cls(actions, raw_decisions, eligibility)

    def _require_eligible(self, decision_id: str) -> tuple[str, ...]:
        if decision_id not in self._decision_actions:
            raise KeyError(f"unknown profile count decision: {decision_id}")
        if not self._decision_eligibility[decision_id]:
            raise ValueError(f"decision is not privacy-head eligible: {decision_id}")
        return self._decision_actions[decision_id]

    def action_scores(
        self, decision_id: str, action_ids: Sequence[str]
    ) -> torch.Tensor:
        """Return exact-objective scores; never pass this tensor into actor features."""
        registered = set(self._require_eligible(decision_id))
        unknown = [action_id for action_id in action_ids if action_id not in registered]
        if unknown:
            raise KeyError(f"unknown profile actions for {decision_id}: {unknown}")
        return torch.tensor(
            [self._actions[action_id].profile_score for action_id in action_ids],
            dtype=torch.float64,
        )

    def target_rows(
        self, *, eligible_only: bool = True
    ) -> tuple[ProfileActionTarget, ...]:
        """Return privacy-head training/calibration/diagnostic rows, never actor inputs."""
        rows = self._actions.values()
        if eligible_only:
            rows = (
                row for row in rows if self._decision_eligibility[row.decision_id]
            )
        return tuple(rows)

    def selected_document_score(self, action_vector: Mapping[str, str]) -> float:
        """Compute the exact diagnostic/shaping score; never expose it as actor state."""
        ineligible = sorted(
            decision_id for decision_id, eligible in self._decision_eligibility.items()
            if not eligible
        )
        if ineligible:
            raise ValueError(f"profile count artifact has ineligible decisions: {ineligible}")
        missing = sorted(set(self._decision_actions) - set(action_vector))
        if missing:
            raise ValueError(f"profile count action vector missing decisions: {missing}")
        extra = sorted(set(action_vector) - set(self._decision_actions))
        if extra:
            raise ValueError(f"profile count action vector has unknown decisions: {extra}")
        scores = []
        for decision_id, registered in self._decision_actions.items():
            action_id = str(action_vector[decision_id])
            if action_id not in registered:
                raise KeyError(f"unknown profile action for {decision_id}: {action_id}")
            scores.append(self._actions[action_id].profile_score)
        if not scores:
            raise ValueError("profile count artifact has no decisions")
        return sum(scores) / len(scores)
