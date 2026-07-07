"""Dataset-backed generalization lattice cache loader."""
import json
import re
from functools import lru_cache
from pathlib import Path

from cloak.runtime_types import PLACEHOLDER_ONLY_TYPES, RUNTIME_TYPES

DEFAULT_PROFILE_PATH = Path("data/lattice_profiles/fine_lattice_profiles.json")
SCHEMA_VERSION = 1


def _empty_artifact() -> dict:
    return {"schema_version": SCHEMA_VERSION, "created": None, "sources": {}, "profiles": {}}


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
    if artifact.get("schema_version") != SCHEMA_VERSION:
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


def _build_indexes(artifact: dict) -> dict:
    by_surface = {}
    by_level = {}
    for runtime_type, entries in artifact.get("profiles", {}).items():
        surface_index = by_surface.setdefault(runtime_type, {})
        level_index = by_level.setdefault(runtime_type, {})
        for canonical, row in entries.items():
            levels = list(row.get("levels", []))
            for key in [_norm(canonical), *[_norm(a) for a in row.get("aliases", [])]]:
                if key:
                    surface_index.setdefault(key, levels)
            count = float(row.get("count", 1.0))
            for level in levels:
                key = _norm(level)
                if key:
                    level_index.setdefault(key, count)
    return {"by_surface": by_surface, "by_level": by_level}


@lru_cache(maxsize=16)
def _index_cached(path_s: str) -> dict:
    return _build_indexes(load_profiles(path_s))


def lookup_levels(surface: str, runtime_type: str, path: str | Path | None = None) -> list[str]:
    key = _norm(surface)
    idx = _index_cached(str(path or DEFAULT_PROFILE_PATH))
    return list(idx["by_surface"].get(runtime_type, {}).get(key, []))


def lookup_count(fill: str, runtime_type: str, path: str | Path | None = None) -> float | None:
    key = _norm(fill)
    idx = _index_cached(str(path or DEFAULT_PROFILE_PATH))
    return idx["by_level"].get(runtime_type, {}).get(key)
