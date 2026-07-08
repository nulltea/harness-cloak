"""Detector-label coverage registry for lattice production."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable


class CategoryOutcome(str, Enum):
    FORCED_PLACEHOLDER = "forced_placeholder"
    RUNTIME_LATTICE = "runtime_lattice"
    NEEDS_PROFILE = "needs_profile"


@dataclass(frozen=True)
class CategoryRegistryEntry:
    detector_label_family: str
    outcome: CategoryOutcome
    runtime_type: str | None
    notes: str = ""


def normalize_label_family(label: str) -> str:
    return re.sub(r"[\s_]+", " ", str(label).strip().lower()).replace("-", " ")


def _entry(label: str, outcome: CategoryOutcome, runtime_type: str | None, notes: str = ""):
    return CategoryRegistryEntry(label, outcome, runtime_type, notes)


def category_registry() -> dict[str, CategoryRegistryEntry]:
    rows = [
        *[
            _entry(label, CategoryOutcome.FORCED_PLACEHOLDER, "PERSON", "direct identifier")
            for label in ("name", "first name", "last name", "name medical professional")
        ],
        *[
            _entry(label, CategoryOutcome.FORCED_PLACEHOLDER, "CODE", "contact endpoint")
            for label in ("email address", "phone number", "ip address", "url")
        ],
        *[
            _entry(label, CategoryOutcome.FORCED_PLACEHOLDER, "CODE", "account or credential")
            for label in (
                "account number",
                "bank account",
                "routing number",
                "credit card",
                "cvv",
                "ssn",
                "passport number",
                "driver license",
                "username",
                "password",
                "vehicle id",
                "healthcare number",
                "medical code",
            )
        ],
        *[_entry(label, CategoryOutcome.RUNTIME_LATTICE, "DATETIME") for label in ("dob", "credit card expiration", "discharge date", "admission date")],
        *[_entry(label, CategoryOutcome.RUNTIME_LATTICE, "QUANTITY") for label in ("money", "dose")],
        _entry("age", CategoryOutcome.RUNTIME_LATTICE, "age"),
        *[
            _entry(label, CategoryOutcome.FORCED_PLACEHOLDER, label.replace(" ", "-"))
            for label in ("gender", "marital status", "sexual orientation")
        ],
        *[
            _entry(label, CategoryOutcome.RUNTIME_LATTICE, "LOC")
            for label in ("location city", "location state", "location country")
        ],
        *[
            _entry(label, CategoryOutcome.FORCED_PLACEHOLDER, "CODE", "exact address or postal code")
            for label in ("location address", "location street", "location zip")
        ],
        _entry("organization", CategoryOutcome.RUNTIME_LATTICE, "ORG"),
        _entry("organization medical facility", CategoryOutcome.RUNTIME_LATTICE, "organization-medical-facility"),
        *[_entry(label, CategoryOutcome.RUNTIME_LATTICE, "health-condition") for label in ("condition", "injury")],
        *[_entry(label, CategoryOutcome.NEEDS_PROFILE, None) for label in ("medical process", "drug", "blood type")],
        *[
            _entry(label, CategoryOutcome.RUNTIME_LATTICE, label)
            for label in ("nationality", "ethnicity", "religion", "profession", "family-role")
        ],
        _entry("demographic other", CategoryOutcome.FORCED_PLACEHOLDER, "demographic-other"),
        # These two runtime types were previously only "eligible" via a hardcoded set in
        # queue.py that duplicated (and, for "drug", contradicted) this registry. Registering
        # them here directly -- the same way "organization medical facility" already is --
        # closes that gap instead of moving it.
        _entry("medical procedure", CategoryOutcome.RUNTIME_LATTICE, "medical-procedure"),
        _entry("misc", CategoryOutcome.RUNTIME_LATTICE, "MISC"),
    ]
    return {normalize_label_family(row.detector_label_family): row for row in rows}


def registry_entry_for_label(label: str) -> CategoryRegistryEntry:
    key = normalize_label_family(label)
    registry = category_registry()
    if key not in registry:
        return CategoryRegistryEntry(key, CategoryOutcome.NEEDS_PROFILE, None, "unmapped detector label")
    return registry[key]


def registry_outcome_for_runtime_type(runtime_type: str) -> CategoryOutcome:
    """Eligibility for a runtime type that's already resolved (e.g. from an existing profile
    artifact's own top-level keys), as opposed to a raw detector label family.

    A runtime type is eligible if either (a) some detector label family routes into it as
    `runtime_lattice`, or (b) the runtime type's own name is itself a registered label (e.g.
    "drug" is both a domain runtime type and its own needs_profile-gated label). Unknown runtime
    types fail closed to needs_profile, same as an unmapped detector label.
    """
    registry = category_registry()
    for entry in registry.values():
        if entry.runtime_type == runtime_type and entry.outcome == CategoryOutcome.RUNTIME_LATTICE:
            return CategoryOutcome.RUNTIME_LATTICE
    key = normalize_label_family(runtime_type)
    if key in registry:
        return registry[key].outcome
    return CategoryOutcome.NEEDS_PROFILE


def _load_profiles(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"profiles": {}}
    return json.loads(path.read_text())


def normalize_category_filters(
    category: str | None = None,
    categories: Iterable[str] | None = None,
) -> list[str]:
    values = []
    if category:
        values.append(category)
    if categories:
        values.extend(categories)
    out = []
    seen = set()
    for value in values:
        for part in str(value).split(","):
            key = re.sub(r"\s+", "-", str(part).strip().lower())
            if key and key not in seen:
                seen.add(key)
                out.append(key)
    return out


def build_category_coverage(
    profiles_path: str | Path,
    category: str | None = None,
    categories: Iterable[str] | None = None,
) -> list[dict]:
    artifact = _load_profiles(profiles_path)
    profiles = artifact.get("profiles", {})
    rows: list[dict] = []
    category_norms = {
        normalize_label_family(value)
        for value in normalize_category_filters(category=category, categories=categories)
    }
    for key, entry in sorted(category_registry().items()):
        if category_norms and not (category_norms & {key, normalize_label_family(entry.runtime_type or "")}):
            continue
        entries = profiles.get(entry.runtime_type or "", {}) if entry.runtime_type else {}
        level_count = sum(len(row.get("levels", [])) for row in entries.values() if isinstance(row, dict))
        dataset_backed = any(
            any(not str(source_id).startswith("producer:") for source_id in row.get("source_ids", []))
            for row in entries.values()
            if isinstance(row, dict)
        )
        rows.append(
            {
                "detector_label_family": entry.detector_label_family,
                "runtime_type": entry.runtime_type,
                "outcome": entry.outcome.value,
                "profile_row_count": len(entries),
                "non_placeholder_level_count": level_count,
                "dataset_backed_source_exists": bool(dataset_backed),
                "generated_universe_required": entry.outcome == CategoryOutcome.RUNTIME_LATTICE
                and (not entries or level_count == 0),
            }
        )
    return rows


def write_category_coverage(
    profiles_path: str | Path,
    out_path: str | Path,
    category: str | None = None,
    categories: Iterable[str] | None = None,
) -> list[dict]:
    rows = build_category_coverage(profiles_path, category=category, categories=categories)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows
