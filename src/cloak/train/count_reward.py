"""Frozen-environment validation shared by the ranker-v2 count gates."""
from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Real


ADMITTED_GROUNDING_STATUSES = frozenset({
    "certifying", "model-proposed", "proposal-universe",
})
_FORBIDDEN_PROVENANCE_TOKENS = ("default", "fallback", "generic", "sentinel")


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
