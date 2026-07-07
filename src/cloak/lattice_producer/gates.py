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


def gate_candidates(item: dict[str, Any], candidates: list[dict[str, Any]]) -> GateResult:
    runtime_type = item.get("runtime_type")
    surface = str(item.get("surface") or item.get("canonical_value") or "")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
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
        if grounding_status != "proposal-universe" and float(candidate.get("level_count", 1.0)) < floor:
            diagnostics.append({**record, "reason": "below_floor"})
            continue
        accepted.append(record)
    return GateResult(accepted=accepted, rejected=rejected, diagnostics=diagnostics)
