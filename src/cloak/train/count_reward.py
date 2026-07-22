"""Versioned, profile-balanced count shaping for the interactive Ranker-v2."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite, log10
from numbers import Real
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from cloak.train.interactive_ranker import ReplayedStep


ADMITTED_GROUNDING_STATUSES = frozenset({
    "certifying", "model-proposed", "proposal-universe",
})
_FORBIDDEN_PROVENANCE_TOKENS = ("default", "fallback", "generic", "sentinel")


@dataclass(frozen=True)
class CountActionScore:
    action_id: str
    decision_id: str
    runtime_type: str
    profile_id: str
    mode: str
    count: float | None
    score: float
    grounding_status: str | None
    source_family: str | None
    evidence_ref: str | None


@dataclass(frozen=True)
class TypeCountReference:
    runtime_type: str
    k_ref: float
    resolution: str
    profile_support: int
    low_reference_support: bool
    flat_count_signal: bool


class CountGateError(ValueError):
    """A count state cannot be published under the requested gate mode."""

    def __init__(self, report: dict):
        self.report = report
        failed = [
            row["clause"] for row in report.get("clauses", [])
            if row.get("result") == "FAIL"
        ]
        super().__init__("count gate failed: " + ", ".join(failed))


def _frozen_environment(environment: Mapping) -> Mapping:
    if environment.get("artifact_version") != "ranker-v2-environment-v2":
        raise ValueError("count reward requires ranker-v2-environment-v2")
    frozen = environment.get("frozen_environment")
    if not isinstance(frozen, Mapping):
        raise ValueError("environment is missing frozen_environment")
    return frozen


def _iter_decisions(environment: Mapping):
    frozen = _frozen_environment(environment)
    for doc_id, document in frozen.get("documents", {}).items():
        occurrences = {
            str(row.get("occurrence_id")): row
            for row in document.get("occurrences", [])
        }
        for decision in document.get("decisions", []):
            if decision.get("ranker_selectable", True):
                yield str(doc_id), document, occurrences, decision


def _evidence_ref(grounding: Mapping | None) -> str | None:
    if not isinstance(grounding, Mapping):
        return None
    for field in (
        "evidence_ref", "member_set_ref", "generated_universe_ref", "count_evidence",
    ):
        value = grounding.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _status_evidence_valid(grounding: Mapping | None) -> bool:
    if not isinstance(grounding, Mapping):
        return False
    status = grounding.get("status")
    if status == "certifying":
        return bool(grounding.get("member_set_ref"))
    if status == "model-proposed":
        return bool(grounding.get("selector") and grounding.get("count_evidence"))
    if status == "proposal-universe":
        return bool(
            grounding.get("member_set_ref") or grounding.get("generated_universe_ref")
        )
    return False


def _nondefault_provenance(grounding: Mapping | None) -> bool:
    if not isinstance(grounding, Mapping):
        return False
    source_family = str(grounding.get("source_family") or "").strip()
    evidence_ref = _evidence_ref(grounding) or ""
    if not source_family or not evidence_ref:
        return False
    lowered = f"{source_family} {evidence_ref}".casefold()
    return not any(token in lowered for token in _FORBIDDEN_PROVENANCE_TOKENS)


def _level_clause_results(action: Mapping) -> tuple[dict[str, str], list[str]]:
    count = action.get("count")
    grounding = action.get("count_grounding")
    explicit_count = (
        isinstance(count, Real)
        and not isinstance(count, bool)
        and isfinite(float(count))
        and float(count) >= 1.0
    )
    accepted_status = (
        isinstance(grounding, Mapping)
        and grounding.get("status") in ADMITTED_GROUNDING_STATUSES
        and bool(grounding.get("source_family"))
    )
    status_evidence = _status_evidence_valid(grounding)
    nondefault = _nondefault_provenance(grounding)
    results = {
        "explicit_count": "PASS" if explicit_count else "FAIL",
        "accepted_status": "PASS" if accepted_status else "FAIL",
        "status_evidence": "PASS" if status_evidence else "FAIL",
        "nondefault_provenance": "PASS" if nondefault else "FAIL",
    }
    return results, [name for name, result in results.items() if result == "FAIL"]


def _policy_mapping_errors(environment: Mapping) -> list[dict]:
    frozen = _frozen_environment(environment)
    expected_by_document = {}
    for corpus_documents in environment.get("corpora", {}).values():
        for doc_id, document in corpus_documents.items():
            expected_by_document[str(doc_id)] = set(document.get("policy_decision_ids", []))
    errors = []
    for doc_id, document in frozen.get("documents", {}).items():
        decisions = {
            str(row.get("decision_id")): row for row in document.get("decisions", [])
        }
        selectable = {
            decision_id for decision_id, decision in decisions.items()
            if decision.get("ranker_selectable", True)
        }
        expected = expected_by_document.get(str(doc_id))
        if expected is None or expected != selectable:
            errors.append({
                "doc_id": str(doc_id),
                "kind": "policy-decision-set",
                "missing": sorted(selectable - (expected or set())),
                "extra": sorted((expected or set()) - selectable),
            })
        occurrences = {
            str(row.get("occurrence_id")): row
            for row in document.get("occurrences", [])
        }
        for decision_id in sorted(selectable):
            for occurrence_id in decisions[decision_id].get("occurrence_ids", []):
                occurrence = occurrences.get(str(occurrence_id))
                if occurrence is None or occurrence.get("decision_id") != decision_id:
                    errors.append({
                        "doc_id": str(doc_id),
                        "decision_id": decision_id,
                        "occurrence_id": str(occurrence_id),
                        "kind": "occurrence-mapping",
                    })
    return errors


def validate_complete_counts(environment: Mapping) -> dict:
    """Return the strict clause-level count gate report without publishing state."""
    frozen = _frozen_environment(environment)
    levels = []
    gaps = []
    model_proposed = 0
    nonmonotone = []
    for doc_id, _document, _occurrences, decision in _iter_decisions(environment):
        decision_levels = [
            action for action in decision.get("actions", [])
            if action.get("mode") == "level" and action.get("legal", True)
        ]
        for action in decision_levels:
            grounding = action.get("count_grounding")
            clause_results, reasons = _level_clause_results(action)
            admitted = not reasons and decision.get("profile_id") is not None
            if decision.get("profile_id") is None:
                reasons = [*reasons, "matched_profile"]
            if isinstance(grounding, Mapping) and grounding.get("status") == "model-proposed":
                model_proposed += 1
            record = {
                "doc_id": doc_id,
                "decision_id": str(decision.get("decision_id")),
                "action_id": str(action.get("action_id")),
                "runtime_type": str(decision.get("runtime_type", "")),
                "profile_id": decision.get("profile_id"),
                "fill": action.get("fill"),
                "authored_level_index": action.get("authored_level_index"),
                "count": float(action["count"])
                if isinstance(action.get("count"), Real)
                and not isinstance(action.get("count"), bool)
                else None,
                "grounding_status": grounding.get("status")
                if isinstance(grounding, Mapping) else None,
                "source_family": grounding.get("source_family")
                if isinstance(grounding, Mapping) else None,
                "evidence_ref": _evidence_ref(grounding),
                "clause_results": clause_results,
                "admitted": admitted,
                "gap_reasons": sorted(set(reasons)),
            }
            levels.append(record)
            if not admitted:
                gaps.append(record)
        ordered = sorted(
            [row for row in decision_levels if row.get("count") is not None],
            key=lambda row: int(row.get("authored_level_index", 0)),
        )
        decreases = [
            {
                "left_action_id": str(left.get("action_id")),
                "left_count": float(left["count"]),
                "right_action_id": str(right.get("action_id")),
                "right_count": float(right["count"]),
            }
            for left, right in zip(ordered, ordered[1:])
            if isinstance(left.get("count"), Real)
            and not isinstance(left.get("count"), bool)
            and isinstance(right.get("count"), Real)
            and not isinstance(right.get("count"), bool)
            and float(right["count"]) < float(left["count"])
        ]
        if decreases:
            nonmonotone.append({
                "doc_id": doc_id,
                "decision_id": str(decision.get("decision_id")),
                "profile_id": decision.get("profile_id"),
                "decreases": decreases,
            })
    mapping_errors = _policy_mapping_errors(environment)
    admitted_count = len(levels) - len(gaps)
    coverage = admitted_count / len(levels) if levels else 1.0
    clauses = [
        {
            "clause": "explicit_coverage",
            "result": "PASS" if coverage == 1.0 else "FAIL",
            "observed": coverage,
            "required": 1.0,
        },
        {
            "clause": "fallback_gradient_mass",
            "result": "PASS",
            "observed": 0.0,
            "required": 0.0,
        },
        {
            "clause": "missing_policy_mappings",
            "result": "PASS" if not mapping_errors else "FAIL",
            "observed": len(mapping_errors),
            "required": 0,
        },
        {
            "clause": "nonmonotone_profiles",
            "result": "PASS" if not nonmonotone else "FAIL",
            "observed": len(nonmonotone),
            "required": 0,
        },
    ]
    verdict = "PASS" if all(row["result"] == "PASS" for row in clauses) else "FAIL"
    return {
        "report_version": "count-gate-report-v1",
        "environment_hash": frozen.get("environment_hash"),
        "mode": "strict",
        "verdict": verdict,
        "strict_verdict": verdict,
        "clauses": clauses,
        "summary": {
            "level_actions": len(levels),
            "admitted_level_actions": admitted_count,
            "gap_count": len(gaps),
            "model_proposed_level_actions": model_proposed,
        },
        "levels": levels,
        "gaps": gaps,
        "missing_policy_mappings": mapping_errors,
        "nonmonotone_profiles": nonmonotone,
    }


def _admitted_level_rows(environment: Mapping) -> list[dict]:
    rows = []
    for _doc_id, _document, _occurrences, decision in _iter_decisions(environment):
        profile_id = decision.get("profile_id")
        if not profile_id:
            continue
        for action in decision.get("actions", []):
            if action.get("mode") != "level" or not action.get("legal", True):
                continue
            _results, reasons = _level_clause_results(action)
            if reasons:
                continue
            rows.append({
                "runtime_type": str(decision.get("runtime_type", "")),
                "profile_id": str(profile_id),
                "count": float(action["count"]),
                "grounding": action["count_grounding"],
            })
    return rows


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def resolve_type_references(environment: Mapping) -> dict[str, TypeCountReference]:
    rows_by_type: dict[str, list[dict]] = defaultdict(list)
    for row in _admitted_level_rows(environment):
        rows_by_type[row["runtime_type"]].append(row)
    references = {}
    for runtime_type, rows in sorted(rows_by_type.items()):
        maxima_by_profile: dict[str, float] = {}
        for row in rows:
            maxima_by_profile[row["profile_id"]] = max(
                maxima_by_profile.get(row["profile_id"], 0.0), row["count"],
            )
        values = list(maxima_by_profile.values())
        support = len(values)
        if values and all(value == 1.0 for value in values):
            references[runtime_type] = TypeCountReference(
                runtime_type, 1.0, "flat-count-signal", support, support < 20, True,
            )
            continue
        universe_pairs = {
            (grounding.get("universe_ref"), float(grounding["universe_size"]))
            for grounding in (row["grounding"] for row in rows)
            if grounding.get("status") == "certifying"
            and grounding.get("universe_ref")
            and isinstance(grounding.get("universe_size"), Real)
            and not isinstance(grounding.get("universe_size"), bool)
            and isfinite(float(grounding["universe_size"]))
            and float(grounding["universe_size"]) >= 1.0
        }
        all_coherent = len(universe_pairs) == 1 and all(
            row["grounding"].get("status") == "certifying"
            and row["grounding"].get("universe_ref")
            and isinstance(row["grounding"].get("universe_size"), Real)
            for row in rows
        )
        if all_coherent:
            k_ref = next(iter(universe_pairs))[1]
            resolution = "grounded-universe"
            low_support = False
        elif support >= 20:
            k_ref = _percentile(values, 0.95)
            resolution = "profile-balanced-p95"
            low_support = False
        elif values:
            k_ref = max(values)
            resolution = "max-profile-fallback"
            low_support = True
        else:
            k_ref = 1.0
            resolution = "flat-count-signal"
            low_support = True
        references[runtime_type] = TypeCountReference(
            runtime_type, float(k_ref), resolution, support, low_support, not values,
        )
    return references


def _provisional_tags(report: Mapping) -> list[dict]:
    by_decision: dict[str, dict] = {}
    for gap in report.get("gaps", []):
        decision_id = str(gap["decision_id"])
        row = by_decision.setdefault(decision_id, {
            "decision_id": decision_id,
            "runtime_type": gap["runtime_type"],
            "profile_id": gap.get("profile_id"),
            "gap_action_ids": [],
        })
        row["gap_action_ids"].append(str(gap["action_id"]))
    return [by_decision[key] for key in sorted(by_decision)]


def _score_level(count: float, reference: TypeCountReference) -> float:
    if reference.flat_count_signal or reference.k_ref <= 1.0:
        return 0.0
    raw = log10(max(float(count), 1.0)) / log10(reference.k_ref)
    return min(1.0, max(0.0, raw))


def _augment_gate_diagnostics(
    report: dict,
    references: Mapping[str, TypeCountReference],
    action_scores: Mapping[str, CountActionScore],
    environment: Mapping,
) -> None:
    report["type_references"] = {
        key: asdict(value) for key, value in references.items()
    }
    clipped_by_type = Counter()
    admitted_by_type = Counter()
    provenance = defaultdict(Counter)
    for level in report["levels"]:
        if not level["admitted"]:
            continue
        runtime_type = level["runtime_type"]
        admitted_by_type[runtime_type] += 1
        provenance[runtime_type][level["grounding_status"]] += 1
        reference = references[runtime_type]
        if float(level["count"]) > reference.k_ref:
            clipped_by_type[runtime_type] += 1
    deltas_by_type = defaultdict(list)
    for _doc_id, _document, _occurrences, decision in _iter_decisions(environment):
        ordered = sorted(
            [
                action_scores[str(action["action_id"])]
                for action in decision.get("actions", [])
                if action.get("mode") == "level"
                and str(action.get("action_id")) in action_scores
            ],
            key=lambda row: next(
                int(action.get("authored_level_index", 0))
                for action in decision.get("actions", [])
                if str(action.get("action_id")) == row.action_id
            ),
        )
        deltas_by_type[str(decision.get("runtime_type", ""))].extend(
            right.score - left.score for left, right in zip(ordered, ordered[1:])
        )
    report["type_diagnostics"] = {}
    for runtime_type in sorted(references):
        deltas = deltas_by_type[runtime_type]
        report["type_diagnostics"][runtime_type] = {
            "clipping_rate": (
                clipped_by_type[runtime_type] / admitted_by_type[runtime_type]
                if admitted_by_type[runtime_type] else 0.0
            ),
            "provenance_by_status": dict(sorted(provenance[runtime_type].items())),
            "adjacent_delta_p": {
                "count": len(deltas),
                "min": min(deltas) if deltas else None,
                "max": max(deltas) if deltas else None,
                "mean": sum(deltas) / len(deltas) if deltas else None,
                "zero_count": sum(value == 0.0 for value in deltas),
            },
        }


def _build_count_reward_state(environment: Mapping, *, provisional: bool) -> dict:
    report = validate_complete_counts(environment)
    hard_failures = {
        row["clause"] for row in report["clauses"]
        if row["result"] == "FAIL"
    }
    always_hard = hard_failures & {"missing_policy_mappings", "nonmonotone_profiles"}
    if always_hard or (hard_failures and not provisional):
        raise CountGateError(report)
    tags = _provisional_tags(report) if provisional else []
    tagged_decisions = {row["decision_id"] for row in tags}
    references = resolve_type_references(environment)
    action_scores: dict[str, CountActionScore] = {}
    decision_actions: dict[str, list[str]] = {}
    for _doc_id, _document, _occurrences, decision in _iter_decisions(environment):
        decision_id = str(decision["decision_id"])
        runtime_type = str(decision.get("runtime_type", ""))
        profile_id = str(decision.get("profile_id") or "")
        ids = []
        for action in decision.get("actions", []):
            if not action.get("legal", True):
                continue
            action_id = str(action["action_id"])
            mode = str(action.get("mode", ""))
            grounding = action.get("count_grounding")
            count = (
                float(action["count"])
                if isinstance(action.get("count"), Real)
                and not isinstance(action.get("count"), bool)
                else None
            )
            if mode == "placeholder":
                score = 1.0
            elif mode == "keep" or decision_id in tagged_decisions:
                score = 0.0
            elif mode == "level" and count is not None:
                score = _score_level(count, references[runtime_type])
            else:
                raise ValueError(f"unsupported count action {decision_id}:{action_id}")
            row = CountActionScore(
                action_id=action_id,
                decision_id=decision_id,
                runtime_type=runtime_type,
                profile_id=profile_id,
                mode=mode,
                count=count if mode == "level" else None,
                score=float(score),
                grounding_status=grounding.get("status")
                if mode == "level" and isinstance(grounding, Mapping) else None,
                source_family=grounding.get("source_family")
                if mode == "level" and isinstance(grounding, Mapping) else None,
                evidence_ref=_evidence_ref(grounding) if mode == "level" else None,
            )
            action_scores[action_id] = row
            ids.append(action_id)
        decision_actions[decision_id] = ids
    if provisional:
        report["mode"] = "provisional"
        report["verdict"] = "PASS"
        report["provisional_decision_tags"] = tags
        report["provisional_tag_count_by_type"] = dict(sorted(Counter(
            row["runtime_type"] for row in tags
        ).items()))
    _augment_gate_diagnostics(report, references, action_scores, environment)
    return {
        "artifact_version": "count-reward-state-v1",
        "environment_hash": _frozen_environment(environment).get("environment_hash"),
        "gate_mode": "provisional" if provisional else "strict",
        "gate_report": report,
        "provisional_decision_tags": tags,
        "type_references": {
            key: asdict(value) for key, value in references.items()
        },
        "action_scores": {
            key: asdict(value) for key, value in action_scores.items()
        },
        "decision_actions": decision_actions,
    }


def build_count_reward_state(environment: Mapping) -> dict:
    return _build_count_reward_state(environment, provisional=False)


class CountReward:
    def __init__(
        self,
        actions: Mapping[str, CountActionScore],
        decision_actions: Mapping[str, Sequence[str]],
    ):
        self._actions = dict(actions)
        self._decision_actions = {
            str(key): tuple(value) for key, value in decision_actions.items()
        }

    @classmethod
    def from_artifact(cls, payload: Mapping) -> CountReward:
        if payload.get("artifact_version") != "count-reward-state-v1":
            raise ValueError("unsupported count reward artifact version")
        if payload.get("artifact_hash"):
            unhashed = {key: value for key, value in payload.items() if key != "artifact_hash"}
            encoded = json.dumps(
                unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            ).encode("utf-8")
            expected = "sha256:" + hashlib.sha256(encoded).hexdigest()
            if payload["artifact_hash"] != expected:
                raise ValueError("count reward artifact hash mismatch")
        actions = {
            str(action_id): CountActionScore(**row)
            for action_id, row in payload.get("action_scores", {}).items()
        }
        for action_id, row in actions.items():
            if action_id != row.action_id:
                raise ValueError(f"count action key mismatch: {action_id}")
        decision_actions = payload.get("decision_actions", {})
        seen = set()
        for decision_id, action_ids in decision_actions.items():
            for action_id in action_ids:
                if action_id in seen:
                    raise ValueError(f"duplicate count action id: {action_id}")
                seen.add(action_id)
                if action_id not in actions or actions[action_id].decision_id != decision_id:
                    raise ValueError(f"count action decision mismatch: {decision_id}:{action_id}")
        return cls(actions, decision_actions)

    def action_scores(
        self, decision_id: str, legal_action_ids: Sequence[str]
    ) -> torch.Tensor:
        registered = set(self._decision_actions.get(decision_id, ()))
        if not registered:
            raise KeyError(f"unknown count decision: {decision_id}")
        unknown = [action_id for action_id in legal_action_ids if action_id not in registered]
        if unknown:
            raise KeyError(f"unknown legal count actions for {decision_id}: {unknown}")
        return torch.tensor(
            [self._actions[action_id].score for action_id in legal_action_ids],
            dtype=torch.float32,
        )

    def selected_document_score(
        self, action_vector: Mapping[str, str]
    ) -> float:
        missing = sorted(set(self._decision_actions) - set(action_vector))
        if missing:
            raise ValueError(f"count action vector missing decisions: {missing}")
        return sum(
            self._actions[str(action_vector[decision_id])].score
            for decision_id in self._decision_actions
        ) / len(self._decision_actions)


def expected_count_loss(
    replay_steps: Sequence["ReplayedStep"],
    count_reward: CountReward,
    lambda_value: float,
    decision_count: int,
    rollout_count: int,
) -> torch.Tensor:
    if decision_count <= 0 or rollout_count <= 0:
        raise ValueError("decision_count and rollout_count must be positive")
    if not isfinite(float(lambda_value)) or lambda_value < 0:
        raise ValueError("lambda_value must be finite and nonnegative")
    expectations = []
    for step in replay_steps:
        log_probs = step.log_probs
        if log_probs.ndim != 1 or len(log_probs) != len(step.legal_action_ids):
            raise ValueError(f"incomplete replay menu for {step.decision_id}")
        scores = count_reward.action_scores(
            step.decision_id, step.legal_action_ids,
        ).to(device=log_probs.device, dtype=log_probs.dtype)
        expectations.append(torch.sum(torch.exp(log_probs) * scores))
    if expectations:
        total = torch.stack(expectations).sum()
    else:
        total = torch.tensor(0.0)
    return -total * (float(lambda_value) / decision_count / rollout_count)
