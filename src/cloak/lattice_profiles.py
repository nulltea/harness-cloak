"""Dataset-backed generalization lattice cache loader."""
import json
import re
from functools import lru_cache
from pathlib import Path

from cloak.runtime_types import PLACEHOLDER_ONLY_TYPES, RUNTIME_TYPES

# Main lattice source (user decision 2026-07-08): the producer-merged artifact, which carries
# per-level `level_counts`. fine_lattice_profiles.json is the legacy dataset-backed cache.
DEFAULT_PROFILE_PATH = Path("data/lattice_profiles/lattice_profiles.json")
SCHEMA_VERSION = 1


def _empty_artifact() -> dict:
    return {"schema_version": SCHEMA_VERSION, "created": None, "sources": {}, "profiles": {}}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def singularize(text: str) -> str:
    """Fold a trailing plural 's' for surface-identity matching ('migraines' -> 'migraine',
    'kidney stones' -> 'kidney stone'). Single source of truth for the plural fold that was
    duplicated ad hoc as rstrip('s') in lattice.py and detect.py; keep all surface-identity
    matching (detection, lattice resolution, the QA freeze) going through here."""
    return _norm(text).rstrip("s")


def _lookup(sub_index: dict, surface: str):
    """Index lookup with a plural-fold fallback, so a plural surface resolves to its
    singular profile row when there is no exact match."""
    key = _norm(surface)
    if key in sub_index:
        return sub_index[key]
    singular = singularize(surface)
    return sub_index.get(singular) if singular != key else None


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
            level_counts = row.get("level_counts") or {}
            for level, value in level_counts.items():
                if level not in levels:
                    errors.append(f"{runtime_type}:{surface} level_counts key not in levels: {level}")
                if float(value or 0.0) < 1.0:
                    errors.append(f"{runtime_type}:{surface} level_counts['{level}'] must be >= 1")
            # k-anonymity walk invariant: counts non-decreasing from specific to broad along
            # the row's level order (docs/specs/offline-k-anonimity-risk-walk.md)
            covered = [(lvl, float(level_counts[lvl])) for lvl in levels if lvl in level_counts]
            for (prev_l, prev_c), (cur_l, cur_c) in zip(covered, covered[1:]):
                if cur_c < prev_c:
                    errors.append(f"{runtime_type}:{surface} level_counts not monotone: "
                                  f"'{cur_l}' ({cur_c}) < '{prev_l}' ({prev_c})")
    return errors


def _build_indexes(artifact: dict) -> dict:
    by_surface = {}
    by_level = {}
    by_level_explicit = {}
    by_group = {}
    for runtime_type, entries in artifact.get("profiles", {}).items():
        surface_index = by_surface.setdefault(runtime_type, {})
        level_index = by_level.setdefault(runtime_type, {})
        explicit_index = by_level_explicit.setdefault(runtime_type, {})
        group_index = by_group.setdefault(runtime_type, {})
        for canonical, row in entries.items():
            levels = list(row.get("levels", []))
            # surface-equivalent group: the canonical + every alias, as written (a reader may
            # answer with any of them). Inherits the row's alias quality (see the near-duplicate
            # issue: a mis-folded alias like hypotension-as-hypertension would over-accept).
            group = [canonical, *row.get("aliases", [])]
            for key in [_norm(canonical), *[_norm(a) for a in row.get("aliases", [])]]:
                if key:
                    surface_index.setdefault(key, (canonical, levels))
                    group_index.setdefault(key, group)
            count = float(row.get("count", 1.0))
            level_counts = row.get("level_counts") or {}
            for level in levels:
                key = _norm(level)
                if not key:
                    continue
                if level in level_counts:
                    # an explicit per-level count is an absolute estimate for that
                    # generalization tier, not a per-surface frequency to sum -- it wins over
                    # the legacy row-count max below.
                    explicit_index[key] = max(explicit_index.get(key, 0.0), float(level_counts[level]))
                else:
                    level_index[key] = max(level_index.get(key, 0.0), count)
    for runtime_type, explicit in by_level_explicit.items():
        by_level[runtime_type] = {**by_level.get(runtime_type, {}), **explicit}
    return {"by_surface": by_surface, "by_level": by_level, "by_group": by_group}


@lru_cache(maxsize=16)
def _index_cached(path_s: str) -> dict:
    return _build_indexes(load_profiles(path_s))


def lookup_entry(surface: str, runtime_type: str,
                 path: str | Path | None = None) -> tuple[str, list[str]] | None:
    """Resolve a surface to its profile row: (canonical, levels). None = no entry."""
    idx = _index_cached(str(path or DEFAULT_PROFILE_PATH))
    got = _lookup(idx["by_surface"].get(runtime_type, {}), surface)
    if got is None:
        return None
    canonical, levels = got
    return canonical, list(levels)


def lookup_levels(surface: str, runtime_type: str, path: str | Path | None = None) -> list[str]:
    got = lookup_entry(surface, runtime_type, path)
    return got[1] if got else []


def lookup_count(fill: str, runtime_type: str, path: str | Path | None = None) -> float | None:
    idx = _index_cached(str(path or DEFAULT_PROFILE_PATH))
    return _lookup(idx["by_level"].get(runtime_type, {}), fill)


def lookup_aliases(surface: str, runtime_type: str, path: str | Path | None = None) -> list[str]:
    """Surface-equivalent group for `surface`'s row: [canonical, *aliases]. These are alternative
    strings for the SAME exact value, so any of them satisfies a ladder rung the exact surface
    would (finest tier -> entails every rung). [] when there is no matching row."""
    idx = _index_cached(str(path or DEFAULT_PROFILE_PATH))
    return list(_lookup(idx["by_group"].get(runtime_type, {}), surface) or [])
