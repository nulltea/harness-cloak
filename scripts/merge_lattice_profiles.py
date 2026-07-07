"""Merge common and mined runtime lattice profile artifacts."""
from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import date
from pathlib import Path

from cloak.lattice_profiles import SCHEMA_VERSION, validate_profile_artifact

DRUG_SOURCE_ID_LIMIT = 5


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _merge_unique_preserve_order(left: list[str], right: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in [*left, *right]:
        key = _norm(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _merge_sources(left: dict, right: dict) -> dict:
    out = copy.deepcopy(left)
    for name, spec in right.items():
        if name not in out or out[name] == spec:
            out[name] = copy.deepcopy(spec)
            continue
        suffix = 2
        candidate = f"{name}-{suffix}"
        while candidate in out:
            suffix += 1
            candidate = f"{name}-{suffix}"
        out[candidate] = copy.deepcopy(spec)
    return dict(sorted(out.items()))


def _index_entries(entries: dict[str, dict]) -> dict[str, str]:
    index = {}
    for canonical, row in entries.items():
        for key in [_norm(canonical), *[_norm(alias) for alias in row.get("aliases", [])]]:
            if key:
                index.setdefault(key, canonical)
    return index


def _clean_incoming_row(runtime_type: str, row: dict) -> dict:
    out = copy.deepcopy(row)
    if runtime_type == "drug":
        out["aliases"] = []
        out["source_ids"] = sorted(source_id for source_id in out.get("source_ids", []) if source_id)[:DRUG_SOURCE_ID_LIMIT]
    return out


def _merge_row(runtime_type: str, existing_canonical: str, existing: dict, incoming_canonical: str, incoming: dict) -> dict:
    aliases = list(existing.get("aliases", []))
    if runtime_type != "drug" and _norm(incoming_canonical) != _norm(existing_canonical):
        aliases.append(incoming_canonical)
    aliases = _merge_unique_preserve_order(aliases, list(incoming.get("aliases", [])))
    aliases = [alias for alias in aliases if _norm(alias) != _norm(existing_canonical)]
    source_ids = sorted(
        {
            source_id
            for source_id in [*existing.get("source_ids", []), *incoming.get("source_ids", [])]
            if source_id
        }
    )
    if runtime_type == "drug":
        source_ids = list(existing.get("source_ids", []))[:DRUG_SOURCE_ID_LIMIT]
    return {
        "aliases": sorted(aliases, key=_norm),
        "levels": _merge_unique_preserve_order(list(existing.get("levels", [])), list(incoming.get("levels", []))),
        "source_ids": source_ids,
        "count": max(float(existing.get("count", 1.0) or 1.0), float(incoming.get("count", 1.0) or 1.0)),
    }


def merge_profile_artifacts(common_artifact: dict, mined_artifact: dict, *, created: str | None = None) -> dict:
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "created": created or str(date.today()),
        "sources": _merge_sources(common_artifact.get("sources", {}), mined_artifact.get("sources", {})),
        "profiles": copy.deepcopy(common_artifact.get("profiles", {})),
    }

    for runtime_type, mined_entries in mined_artifact.get("profiles", {}).items():
        entries = artifact["profiles"].setdefault(runtime_type, {})
        index = _index_entries(entries)
        for incoming_canonical, incoming_row in mined_entries.items():
            incoming_row = _clean_incoming_row(runtime_type, incoming_row)
            match = index.get(_norm(incoming_canonical))
            if match is None and runtime_type != "drug":
                for alias in incoming_row.get("aliases", []):
                    match = index.get(_norm(alias))
                    if match is not None:
                        break
            if match is None:
                entries[incoming_canonical] = copy.deepcopy(incoming_row)
                match = incoming_canonical
            else:
                entries[match] = _merge_row(runtime_type, match, entries[match], incoming_canonical, incoming_row)
            index = _index_entries(entries)

    artifact["profiles"] = {
        runtime_type: dict(sorted(entries.items(), key=lambda item: _norm(item[0])))
        for runtime_type, entries in sorted(artifact["profiles"].items())
    }
    errors = validate_profile_artifact(artifact)
    if errors:
        raise ValueError("invalid merged lattice profile artifact:\n" + "\n".join(errors[:50]))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common", default="data/lattice_profiles/comm_lattice_profiles.json")
    parser.add_argument("--mined", default="data/lattice_profiles/mined_lattice_profiles.json")
    parser.add_argument("--out", default="data/lattice_profiles/lattice_profiles.json")
    args = parser.parse_args()

    common_artifact = json.loads(Path(args.common).read_text())
    mined_artifact = json.loads(Path(args.mined).read_text())
    artifact = merge_profile_artifacts(common_artifact, mined_artifact)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    profile_total = sum(len(entries) for entries in artifact["profiles"].values())
    print(f"profiles={profile_total} -> {out}", flush=True)


if __name__ == "__main__":
    main()
