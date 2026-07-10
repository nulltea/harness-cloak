"""Deterministic parser and field scorer for schema-constrained task outputs."""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from cloak.train.reward import canon, fact_score

_SECTIONS = {
    "chief complaint": "chief_complaint",
    "history of present illness": "history_of_present_illness",
    "assessment": "assessment",
    "plan": "plan",
    "parties": "parties",
    "claims": "claims",
    "outcome": "outcome",
}
_HEADER_RE = re.compile(
    r"^\s*(chief complaint|history of present illness|assessment|plan|parties|claims|outcome)"
    r"\s*:?\s*(.*)$",
    re.IGNORECASE,
)
_DASH_RE = re.compile(r"\s*(?:--+|[—–]|\s-\s)\s*")
_ROW_FIELDS = {
    "assessment": ("problem", "category", "status"),
    "plan": ("problem", "action", "follow_up"),
    "claims": ("claim", "category", "status"),
    "outcome": ("claim", "remedy", "posture"),
}
_SCALAR_SECTIONS = ("chief_complaint", "history_of_present_illness", "parties")
_ROW_SECTIONS = ("assessment", "plan", "claims", "outcome")


def parse_sections(text: str) -> dict:
    """Parse supported schema sections without raising on malformed model output."""
    parsed = {section: [] for section in _ROW_SECTIONS}
    parsed.update({section: "" for section in _SCALAR_SECTIONS})
    buckets: dict[str, list[str]] = {section: [] for section in parsed}
    current: str | None = None

    try:
        lines = str(text or "").splitlines()
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            match = _HEADER_RE.match(line)
            if match:
                current = _SECTIONS[match.group(1).lower()]
                rest = match.group(2).strip()
                if rest:
                    buckets[current].append(rest)
                continue
            if current:
                buckets[current].append(line)

        for section in _SCALAR_SECTIONS:
            parsed[section] = _scalar_value(buckets[section])
        for section in _ROW_SECTIONS:
            fields = _ROW_FIELDS[section]
            parsed[section] = [_parse_row(line, fields) for line in buckets[section]
                               if not _is_none_line(line)]
            parsed[section] = [row for row in parsed[section] if row is not None]
    except Exception:
        return parsed
    return parsed


def schema_field_score(
    out_final_text: str,
    out_hi_text: str,
    acceptance_sets: Mapping[str, Iterable[str]] | None = None,
) -> float | None:
    """Score parsed schema fields against ceiling rows aligned by canonical problem name."""
    out_rows = _clinical_rows_by_problem(parse_sections(out_final_text))
    hi_rows = _clinical_rows_by_problem(parse_sections(out_hi_text))
    if not hi_rows:
        return None

    scores: list[float] = []
    for problem_key, gold_row in hi_rows.items():
        pred_row = out_rows.get(problem_key, {})
        for field in ("category", "status", "action", "follow_up"):
            gold = gold_row.get(field, "")
            if not gold:
                continue
            pred = pred_row.get(field, "")
            if field == "category":
                scores.append(_category_score(pred, gold, problem_key, acceptance_sets))
            else:
                scores.append(fact_score(pred, gold))
    if not scores:
        return None
    return max(0.0, min(1.0, sum(scores) / len(scores)))


def _scalar_value(lines: list[str]) -> str:
    value = " ".join(line.strip() for line in lines if line.strip()).strip()
    return "" if _is_none_line(value) else value


def _parse_row(line: str, fields: tuple[str, ...]) -> dict[str, str] | None:
    parts = [part.strip() for part in _DASH_RE.split(line.strip()) if part.strip()]
    if len(parts) < len(fields):
        return None
    return dict(zip(fields, parts[:len(fields)]))


def _is_none_line(text: str) -> bool:
    return bool(re.fullmatch(r"none\.?", str(text or "").strip(), flags=re.IGNORECASE))


def _clinical_rows_by_problem(parsed: dict) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for section in ("assessment", "plan"):
        for row in parsed.get(section, []):
            problem = row.get("problem", "")
            if not problem:
                continue
            key = canon(problem)
            merged = rows.setdefault(key, {"problem": problem})
            merged.update({k: v for k, v in row.items() if k != "problem"})
    return rows


def _category_score(
    pred: str,
    gold: str,
    problem_key: str,
    acceptance_sets: Mapping[str, Iterable[str]] | None,
) -> float:
    accepted = [gold]
    if acceptance_sets:
        for key, values in acceptance_sets.items():
            if canon(key) == problem_key:
                accepted.extend(str(value) for value in values)
                break
    return max(fact_score(pred, value) for value in accepted)
