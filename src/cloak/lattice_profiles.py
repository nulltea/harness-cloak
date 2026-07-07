"""Dataset-backed fine-type lattice profile artifact loader."""
import json
import re
from functools import lru_cache
from pathlib import Path

from cloak.runtime_types import PLACEHOLDER_ONLY_TYPES, RUNTIME_TYPES

DEFAULT_PROFILE_PATH = Path("data/lattice_profiles/fine_lattice_profiles.json")


def _empty_artifact() -> dict:
    return {"schema_version": 1, "created": None, "sources": {}, "profiles": {}}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _is_type_name_phrase(fill: str) -> bool:
    from cloak.lattice import is_type_name_phrase

    return is_type_name_phrase(fill)


@lru_cache(maxsize=16)
def _load_cached(path_s: str) -> dict:
    path = Path(path_s)
    if not path.exists():
        return _empty_artifact()
    return json.loads(path.read_text())


def load_profiles(path: str | Path | None = None) -> dict:
    return _load_cached(str(path or DEFAULT_PROFILE_PATH))


def validate_profile_artifact(artifact: dict) -> list[str]:
    errors = []
    if artifact.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    profiles = artifact.get("profiles")
    if not isinstance(profiles, dict):
        return errors + ["profiles must be an object"]
    for runtime_type, entries in profiles.items():
        if runtime_type not in RUNTIME_TYPES:
            errors.append(f"unknown runtime type: {runtime_type}")
        if not isinstance(entries, dict):
            errors.append(f"profiles[{runtime_type}] must be an object")
            continue
        for surface, row in entries.items():
            levels = row.get("levels", [])
            if not levels and runtime_type not in PLACEHOLDER_ONLY_TYPES:
                errors.append(f"{runtime_type}:{surface} has no levels")
            for level in levels:
                if _is_type_name_phrase(level):
                    errors.append(f"{runtime_type}:{surface} has type-name phrase: {level}")
                if _norm(surface) and _norm(surface) in _norm(level):
                    errors.append(f"{runtime_type}:{surface} leaks original surface in level: {level}")
            if float(row.get("count", 0.0) or 0.0) < 1.0:
                errors.append(f"{runtime_type}:{surface} count must be >= 1")
    return errors


def _iter_rows(artifact: dict, runtime_type: str):
    for surface, row in artifact.get("profiles", {}).get(runtime_type, {}).items():
        yield surface, row


def lookup_levels(surface: str, runtime_type: str, path: str | Path | None = None) -> list[str]:
    key = _norm(surface)
    for canonical, row in _iter_rows(load_profiles(path), runtime_type):
        aliases = [_norm(canonical), *[_norm(a) for a in row.get("aliases", [])]]
        if key in aliases:
            return list(row.get("levels", []))
    return []


def lookup_count(fill: str, runtime_type: str, path: str | Path | None = None) -> float | None:
    key = _norm(fill)
    for _, row in _iter_rows(load_profiles(path), runtime_type):
        if key in {_norm(x) for x in row.get("levels", [])}:
            return float(row.get("count", 1.0))
    return None
