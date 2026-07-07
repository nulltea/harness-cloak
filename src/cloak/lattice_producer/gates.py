"""Safety gates for proposed lattice levels."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from cloak.anonymity import K_FLOORS
from cloak.lattice import is_type_name_phrase
from cloak.runtime_types import FORCED_PLACEHOLDER_TYPES, PLACEHOLDER_RE


@dataclass
class GateResult:
    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


_GENERIC_PROFESSION_LEVELS = {
    "worker",
    "professional worker",
    "technical worker",
    "production worker",
    "education worker",
    "arts and media worker",
    "business and financial occupation",
    "architecture and engineering occupation",
    "science occupation",
    "construction worker",
    "installation and repair worker",
    "transportation and material moving worker",
}


def _is_model_proposed(candidate: dict[str, Any]) -> bool:
    grounding = candidate.get("level_grounding") or {}
    return candidate.get("source_family") == "model-proposed" or grounding.get("status") == "model-proposed"


def _aliases_for(item: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    values = [*item.get("aliases", []), *candidate.get("aliases", [])]
    return [str(value).strip() for value in values if str(value).strip()]


def _has_model_evidence(candidate: dict[str, Any]) -> bool:
    grounding = candidate.get("level_grounding") or {}
    count_evidence = str(candidate.get("count_evidence") or grounding.get("count_evidence") or "").strip()
    rationale = str(candidate.get("rationale") or candidate.get("level_rationale") or "").strip()
    selector = str(candidate.get("selector") or grounding.get("selector") or "").strip()
    return bool(count_evidence and rationale and selector)


def _is_generic_profession_level(level: str) -> bool:
    text = _norm(level)
    if text in _GENERIC_PROFESSION_LEVELS:
        return True
    tokens = text.split()
    return len(tokens) <= 3 and (text.endswith(" worker") or text.endswith(" occupation") or text.endswith(" professional"))


def gate_candidates(item: dict[str, Any], candidates: list[dict[str, Any]]) -> GateResult:
    runtime_type = item.get("runtime_type")
    surface = str(item.get("surface") or item.get("canonical_value") or "")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    model_candidates = [candidate for candidate in candidates if _is_model_proposed(candidate)]
    model_positions = {id(candidate): idx for idx, candidate in enumerate(model_candidates)}
    model_counts = [float(candidate.get("level_count", 1.0)) for candidate in model_candidates]
    flat_model_counts = len(model_counts) > 1 and len(set(model_counts)) == 1
    generic_only_model_chain = bool(model_candidates) and all(
        _is_generic_profession_level(str(candidate.get("level", ""))) for candidate in model_candidates
    )
    for candidate in candidates:
        level = str(candidate.get("level", "")).strip()
        record = {**candidate, "item_id": item.get("item_id"), "runtime_type": runtime_type}
        reason = None
        if runtime_type in FORCED_PLACEHOLDER_TYPES or runtime_type == "DEM":
            reason = "ineligible_runtime_type"
        elif PLACEHOLDER_RE.search(level):
            reason = "placeholder_terminal"
        elif surface and _norm(surface) in _norm(level):
            reason = "self_leak"
        elif re.search(r"\b\d{3,}\b", level):
            reason = "distinctive_number"
        elif is_type_name_phrase(level):
            reason = "type_name_phrase"
        if reason:
            rejected.append({**record, "reason": reason})
            continue
        grounding_status = (candidate.get("level_grounding") or {}).get("status")
        floor = float(K_FLOORS.get(str(runtime_type), 100.0))
        if _is_model_proposed(candidate):
            if not _aliases_for(item, candidate):
                diagnostics.append({**record, "reason": "missing_aliases"})
                continue
            if grounding_status != "model-proposed":
                diagnostics.append({**record, "reason": "missing_model_count"})
                continue
            if not _has_model_evidence(candidate):
                diagnostics.append({**record, "reason": "missing_model_evidence"})
                continue
            if flat_model_counts and (not generic_only_model_chain or model_positions.get(id(candidate), 0) == 0):
                diagnostics.append({**record, "reason": "flat_model_counts"})
                continue
            if generic_only_model_chain:
                diagnostics.append({**record, "reason": "weak_semantic_relevance"})
                continue
        if grounding_status == "fail-closed":
            diagnostics.append({**record, "reason": (candidate.get("level_grounding") or {}).get("reason", "fail_closed")})
            continue
        if grounding_status != "proposal-universe" and float(candidate.get("level_count", 1.0)) < floor:
            diagnostics.append({**record, "reason": "below_floor"})
            continue
        accepted.append(record)
    return GateResult(accepted=accepted, rejected=rejected, diagnostics=diagnostics)
