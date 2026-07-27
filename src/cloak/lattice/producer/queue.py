"""Queue construction for lattice producer runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from cloak.lattice.producer.coverage import (
    CategoryOutcome,
    build_category_coverage,
    normalize_category_filters,
    normalize_label_family,
    registry_entry_for_label,
    registry_outcome_for_runtime_type,
)
from cloak.lattice.producer.io import read_jsonl
from cloak.runtime_types import FORCED_PLACEHOLDER_TYPES, RUNTIME_TYPES


def normalize_item(raw: dict[str, Any], index: int = 0) -> dict[str, Any]:
    item = dict(raw)
    label = item.get("detector_label_family")
    if label and not item.get("runtime_type"):
        entry = registry_entry_for_label(str(label))
        item["runtime_type"] = entry.runtime_type
        item["detector_label_family"] = entry.detector_label_family
        item["registry_outcome"] = entry.outcome.value
    runtime_type = item.get("runtime_type")
    if runtime_type == "DEM":
        raise ValueError("DEM is an eval rollup and cannot be queued for fine lattice production")
    if runtime_type and runtime_type not in RUNTIME_TYPES:
        raise ValueError(f"unknown runtime type: {runtime_type}")
    if not item.get("item_id"):
        surface = item.get("surface") or item.get("canonical_value") or item.get("detector_label_family") or index
        item["item_id"] = f"{runtime_type or 'unmapped'}:{surface}"
    # a caller (e.g. _queue_from_profile_categories) may set runtime_type directly from an
    # existing profile artifact's own top-level keys rather than from a raw detector label --
    # resolve eligibility from the registry either way, never from a second, locally-duplicated
    # set of "eligible" runtime types (that duplication once let "drug" bypass its own
    # needs_profile registry entry).
    if "registry_outcome" not in item and runtime_type:
        item["registry_outcome"] = registry_outcome_for_runtime_type(runtime_type).value
    if runtime_type in FORCED_PLACEHOLDER_TYPES or item.get("registry_outcome") == CategoryOutcome.FORCED_PLACEHOLDER.value:
        item["eligible"] = False
        item["skip_reason"] = "forced_placeholder"
    elif item.get("registry_outcome") == CategoryOutcome.RUNTIME_LATTICE.value:
        item.setdefault("eligible", True)
    else:
        item["eligible"] = False
        item.setdefault("skip_reason", "needs_profile")
    item.setdefault("task_kind", "generated-universe" if item.get("entry_origin") == "generated-universe" else "level-proposal")
    return item


def _queue_from_coverage(
    profiles_path: str | Path,
    category: str | None = None,
    categories: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    items = []
    for row in build_category_coverage(profiles_path, category=category, categories=categories):
        if row["outcome"] == CategoryOutcome.RUNTIME_LATTICE.value and row["generated_universe_required"]:
            items.append(
                normalize_item(
                    {
                        "item_id": f"coverage:{row['detector_label_family']}",
                        "task_kind": "generated-universe",
                        "detector_label_family": row["detector_label_family"],
                        "runtime_type": row["runtime_type"],
                        "surface": row["detector_label_family"],
                        "entry_origin": "generated-universe",
                    },
                    len(items),
                )
            )
    return items


def _load_profiles(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"profiles": {}}
    return json.loads(path.read_text())


def _runtime_types_for_categories(profiles: dict, categories: Iterable[str]) -> list[str]:
    filters = [normalize_label_family(category) for category in normalize_category_filters(categories=categories)]
    selected = []
    seen = set()
    profile_runtime_types = set(profiles.get("profiles", {}))
    for category in filters:
        for runtime_type in sorted(profile_runtime_types):
            if normalize_label_family(runtime_type) == category and runtime_type not in seen:
                selected.append(runtime_type)
                seen.add(runtime_type)
        entry = registry_entry_for_label(category)
        if entry.runtime_type in profile_runtime_types and entry.runtime_type not in seen:
            selected.append(entry.runtime_type)
            seen.add(entry.runtime_type)
    return selected


def _queue_from_profile_categories(profiles_path: str | Path, categories: Iterable[str]) -> list[dict[str, Any]]:
    artifact = _load_profiles(profiles_path)
    profiles = artifact.get("profiles", {})
    items = []
    for runtime_type in _runtime_types_for_categories(artifact, categories):
        for surface, row in sorted(profiles.get(runtime_type, {}).items()):
            items.append(
                normalize_item(
                    {
                        "item_id": f"{runtime_type}:{surface}",
                        "task_kind": "level-proposal",
                        "runtime_type": runtime_type,
                        "detector_label_family": runtime_type,
                        "surface": surface,
                        "canonical_value": surface,
                        "aliases": list(row.get("aliases", [])) if isinstance(row, dict) else [],
                    },
                    len(items),
                )
            )
    return items


def build_or_load_queue(
    run_dir: str | Path,
    profiles_path: str | Path,
    *,
    explicit_queue: str | Path | None = None,
    category: str | None = None,
    categories: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    queue_path = run_dir / "queue.jsonl"
    category_filters = normalize_category_filters(category=category, categories=categories)
    if explicit_queue:
        rows = read_jsonl(explicit_queue)
    elif queue_path.exists():
        rows = read_jsonl(queue_path)
    elif category_filters:
        rows = _queue_from_profile_categories(profiles_path, category_filters)
        if not rows:
            rows = _queue_from_coverage(profiles_path, categories=category_filters)
    else:
        rows = _queue_from_coverage(profiles_path)
    items = [normalize_item(row, idx) for idx, row in enumerate(rows)]
    if not explicit_queue:
        run_dir.mkdir(parents=True, exist_ok=True)
        queue_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in items))
    return items
