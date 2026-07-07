"""Queue construction for lattice producer runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cloak.lattice_producer.coverage import CategoryOutcome, build_category_coverage, registry_entry_for_label
from cloak.lattice_producer.io import read_jsonl
from cloak.runtime_types import FORCED_PLACEHOLDER_TYPES, RUNTIME_TYPES

LATTICE_RUNTIME_TYPES = {
    "ORG",
    "LOC",
    "DATETIME",
    "QUANTITY",
    "MISC",
    "nationality",
    "ethnicity",
    "religion",
    "profession",
    "age",
    "health-condition",
    "family-role",
    "drug",
    "medical-procedure",
    "organization-medical-facility",
}


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
    if runtime_type in FORCED_PLACEHOLDER_TYPES or item.get("registry_outcome") == CategoryOutcome.FORCED_PLACEHOLDER.value:
        item["eligible"] = False
        item["skip_reason"] = "forced_placeholder"
    elif runtime_type in LATTICE_RUNTIME_TYPES:
        item.setdefault("eligible", True)
    else:
        item["eligible"] = False
        item.setdefault("skip_reason", "needs_profile")
    item.setdefault("task_kind", "generated-universe" if item.get("entry_origin") == "generated-universe" else "level-proposal")
    return item


def _queue_from_coverage(profiles_path: str | Path, category: str | None = None) -> list[dict[str, Any]]:
    items = []
    for row in build_category_coverage(profiles_path, category=category):
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


def build_or_load_queue(
    run_dir: str | Path,
    profiles_path: str | Path,
    *,
    explicit_queue: str | Path | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    queue_path = run_dir / "queue.jsonl"
    if explicit_queue:
        rows = read_jsonl(explicit_queue)
    elif queue_path.exists():
        rows = read_jsonl(queue_path)
    else:
        rows = _queue_from_coverage(profiles_path, category=category)
    items = [normalize_item(row, idx) for idx, row in enumerate(rows)]
    if not explicit_queue:
        run_dir.mkdir(parents=True, exist_ok=True)
        queue_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in items))
    return items
