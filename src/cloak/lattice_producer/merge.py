"""Proposed artifact persistence and validation."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from cloak.lattice_producer.io import atomic_write_json
from cloak.lattice_profiles import validate_profile_artifact
from cloak.runtime_types import FORCED_PLACEHOLDER_TYPES, PLACEHOLDER_RE

PROPOSAL_SCOPE = "producer-processed-only"


def _load_artifact(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"schema_version": 1, "created": str(date.today()), "sources": {}, "profiles": {}}
    return json.loads(path.read_text())


def _hash_file(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dedupe_append(existing: list[str], additions: list[str]) -> list[str]:
    out = list(existing)
    seen = {x.lower() for x in out}
    for value in additions:
        if value.lower() not in seen:
            out.append(value)
            seen.add(value.lower())
    return out


def _accepted_aliases(records: list[dict[str, Any]]) -> list[str]:
    aliases: list[str] = []
    for record in records:
        aliases.extend(str(alias).lower() for alias in record.get("aliases", []) if str(alias).strip())
    return aliases


def _generality_score(level: str) -> int:
    text = level.strip().lower()
    exact = {
        "worker": 120,
        "professional worker": 110,
        "technical worker": 70,
        "production worker": 65,
        "arts and media worker": 55,
    }
    if text in exact:
        return exact[text]
    if text.endswith(" worker"):
        return 60
    if text.endswith(" occupation"):
        return 35
    return 20


def _counts_monotone(records: list[dict[str, Any]]) -> bool:
    counts = [float(record.get("level_count", 1.0)) for record in records]
    return all(left <= right for left, right in zip(counts, counts[1:]))


def _order_accepted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    semantic = [
        record
        for _, record in sorted(
            enumerate(records),
            key=lambda pair: (_generality_score(str(pair[1].get("level", ""))), pair[0]),
        )
    ]
    if _counts_monotone(semantic):
        return semantic
    return [
        record
        for _, record in sorted(
            enumerate(records),
            key=lambda pair: (float(pair[1].get("level_count", 1.0)), _generality_score(str(pair[1].get("level", ""))), pair[0]),
        )
    ]


def ensure_proposed_artifact(canonical_path: str | Path, proposed_path: str | Path, *, run_id: str) -> None:
    if Path(proposed_path).exists():
        artifact = _load_artifact(proposed_path)
        if artifact.get("proposal_scope") != PROPOSAL_SCOPE:
            artifact = {"schema_version": 1, "created": str(date.today()), "sources": {}, "profiles": {}}
    else:
        artifact = {"schema_version": 1, "created": str(date.today()), "sources": {}, "profiles": {}}
    artifact["schema_version"] = 1
    artifact.setdefault("created", str(date.today()))
    artifact["artifact_role"] = "proposal"
    artifact["proposal_scope"] = PROPOSAL_SCOPE
    artifact["base_profile_hash"] = artifact.get("base_profile_hash") or _hash_file(canonical_path)
    artifact["producer_run_id"] = run_id
    artifact.setdefault("sources", {})
    artifact.setdefault("profiles", {})
    atomic_write_json(proposed_path, artifact)


def persist_proposed_artifact(
    canonical_path: str | Path,
    proposed_path: str | Path,
    *,
    run_id: str,
    item: dict[str, Any],
    accepted: list[dict[str, Any]],
) -> None:
    ensure_proposed_artifact(canonical_path, proposed_path, run_id=run_id)
    artifact = _load_artifact(proposed_path)
    runtime_type = item.get("runtime_type")
    if runtime_type == "DEM" or runtime_type in FORCED_PLACEHOLDER_TYPES:
        return
    surface = str(item.get("canonical_value") or item.get("surface") or "").strip().lower()
    if not surface or not accepted:
        return
    profiles = artifact["profiles"].setdefault(runtime_type, {})
    row = profiles.setdefault(
        surface,
        {
            "aliases": [],
            "levels": [],
            "source_ids": [],
            "count": 1.0,
            "entry_origin": item.get("entry_origin", "observed-surface"),
            "level_counts": {},
            "level_groundings": {},
        },
    )
    ordered = _order_accepted_records(accepted)
    row["aliases"] = _dedupe_append(
        list(row.get("aliases", [])),
        [
            *[str(a).lower() for a in item.get("aliases", []) if a],
            *_accepted_aliases(ordered),
        ],
    )
    row["levels"] = _dedupe_append(
        [level for level in row.get("levels", []) if not PLACEHOLDER_RE.search(str(level))],
        [record["level"] for record in ordered if not PLACEHOLDER_RE.search(str(record.get("level", "")))],
    )
    row.setdefault("level_counts", {})
    row.setdefault("level_groundings", {})
    row["level_counts"] = {level: row["level_counts"][level] for level in row["levels"] if level in row["level_counts"]}
    row["level_groundings"] = {level: row["level_groundings"][level] for level in row["levels"] if level in row["level_groundings"]}
    for record in ordered:
        level = record["level"]
        row["level_counts"][level] = float(record["level_count"])
        row["level_groundings"][level] = dict(record["level_grounding"])
    row["level_counts"] = {level: row["level_counts"][level] for level in row["levels"] if level in row["level_counts"]}
    row["level_groundings"] = {level: row["level_groundings"][level] for level in row["levels"] if level in row["level_groundings"]}
    row["entry_origin"] = row.get("entry_origin") or item.get("entry_origin", "observed-surface")
    row["source_ids"] = _dedupe_append(list(row.get("source_ids", [])), [f"producer:{run_id}:{item.get('item_id')}"])
    if row["level_counts"]:
        row["count"] = max(float(v) for v in row["level_counts"].values())
    atomic_write_json(proposed_path, artifact)


def validate_proposed_artifact(path: str | Path) -> list[str]:
    artifact = _load_artifact(path)
    errors = []
    if artifact.get("artifact_role") != "proposal":
        errors.append("artifact_role must be proposal")
    if artifact.get("proposal_scope") != PROPOSAL_SCOPE:
        errors.append(f"proposal_scope must be {PROPOSAL_SCOPE}")
    profiles = artifact.get("profiles", {})
    if "DEM" in profiles:
        errors.append("DEM profiles are forbidden in proposed artifacts")
    for runtime_type, entries in profiles.items():
        if runtime_type in FORCED_PLACEHOLDER_TYPES:
            for surface, row in entries.items():
                if row.get("levels"):
                    errors.append(f"{runtime_type}:{surface} forced-placeholder type has text levels")
        for surface, row in entries.items():
            levels = row.get("levels", [])
            counts = row.get("level_counts", {})
            groundings = row.get("level_groundings", {})
            prev = 0.0
            for level in levels:
                if PLACEHOLDER_RE.search(str(level)):
                    errors.append(f"{runtime_type}:{surface} has placeholder in levels")
                if level not in counts:
                    errors.append(f"{runtime_type}:{surface}:{level} missing level_counts")
                if level not in groundings:
                    errors.append(f"{runtime_type}:{surface}:{level} missing level_groundings")
                count = float(counts.get(level, 1.0))
                if count < prev:
                    errors.append(f"{runtime_type}:{surface} level_counts are not monotone")
                prev = count
                if row.get("entry_origin") == "generated-universe" and groundings.get(level, {}).get("status") != "proposal-universe":
                    errors.append(f"{runtime_type}:{surface}:{level} generated-universe level is not proposal-universe")
    runtime_artifact = json.loads(json.dumps(artifact))
    runtime_artifact.pop("artifact_role", None)
    runtime_artifact.pop("proposal_scope", None)
    runtime_artifact.pop("base_profile_hash", None)
    runtime_artifact.pop("producer_run_id", None)
    errors.extend(validate_profile_artifact(runtime_artifact))
    return errors
