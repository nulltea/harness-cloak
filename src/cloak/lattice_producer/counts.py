"""Deterministic per-level count compiler for proposed lattice rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cloak.anonymity import aset_count
from cloak.lattice_producer.io import read_jsonl


def _as_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _candidate_text(candidate: dict[str, Any], key: str) -> str:
    value = candidate.get(key)
    if value is None and isinstance(candidate.get("level_grounding"), dict):
        value = candidate["level_grounding"].get(key)
    return str(value or "").strip()


def _generated_universe_count(runtime_type: str, level: str, generated_universe_path: str | Path) -> float:
    rows = read_jsonl(generated_universe_path)
    members = {
        str(row.get("canonical_value"))
        for row in rows
        if row.get("runtime_type") == runtime_type and level in row.get("proposed_levels", [])
    }
    members.discard("None")
    return float(len(members)) if members else 1.0


def compile_level_counts(
    item: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    generated_universe_path: str | Path,
) -> list[dict[str, Any]]:
    runtime_type = item.get("runtime_type", "")
    compiled = []
    for candidate in candidates:
        level = str(candidate.get("level", "")).strip()
        if not level:
            continue
        out = dict(candidate)
        if item.get("entry_origin") == "generated-universe" or candidate.get("entry_origin") == "generated-universe":
            count = _generated_universe_count(runtime_type, level, generated_universe_path)
            status = "proposal-universe"
            grounding = {
                "status": status,
                "source_family": "generated-universe",
                "selector": candidate.get("selector") or f"generated_level:{level}",
                "member_set_ref": f"generated-universe:{runtime_type}:{level}",
            }
        elif candidate.get("member_set"):
            count = float(len(set(candidate["member_set"])))
            grounding = {
                "status": "certifying",
                "source_family": candidate.get("source_family", "deterministic"),
                "selector": candidate.get("selector", level),
                "member_set_ref": candidate.get("member_set_ref", "inline-member-set"),
            }
        elif candidate.get("source_family") == "model-proposed":
            proposed_count = _as_positive_float(candidate.get("proposed_count", candidate.get("level_count")))
            count_evidence = _candidate_text(candidate, "count_evidence")
            selector = str(candidate.get("selector") or "").strip()
            if proposed_count is None or not count_evidence or not selector:
                count = 1.0
                grounding = {
                    "status": "fail-closed",
                    "source_family": "model-proposed",
                    "selector": selector or level,
                    "member_set_ref": None,
                    "reason": "missing_model_count_evidence",
                }
            else:
                count = proposed_count
                grounding = {
                    "status": "model-proposed",
                    "source_family": "model-proposed",
                    "selector": selector,
                    "member_set_ref": None,
                    "count_evidence": count_evidence,
                }
        else:
            count = aset_count(level, runtime_type, str(item.get("surface") or item.get("canonical_value") or ""), strict=True)
            if count <= 1.0:
                grounding = {
                    "status": "fail-closed",
                    "source_family": candidate.get("source_family", "unsupported"),
                    "selector": candidate.get("selector", level),
                    "member_set_ref": None,
                }
            else:
                grounding = {
                    "status": "certifying",
                    "source_family": "deterministic-aset",
                    "selector": level,
                    "member_set_ref": f"aset:{runtime_type}:{level}",
                }
        out["level"] = level
        out["level_count"] = float(count)
        out["level_grounding"] = grounding
        compiled.append(out)
    return [record for _, record in sorted(enumerate(compiled), key=lambda pair: (pair[1]["level_count"], pair[0]))]
