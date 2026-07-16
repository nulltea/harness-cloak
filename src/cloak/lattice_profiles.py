"""Dataset-backed generalization lattice cache loader."""
import json
import re
import zipfile
from functools import lru_cache
from pathlib import Path

from cloak.runtime_types import PLACEHOLDER_ONLY_TYPES, RUNTIME_TYPES

# Main lattice source (user decision 2026-07-08): the producer-merged artifact, which carries
# per-level `level_counts`. fine_lattice_profiles.json is the legacy dataset-backed cache.
DEFAULT_PROFILE_PATH = Path("data/lattice_profiles/lattice_profiles.json")
# openFDA NDC drug dataset (brand -> active ingredient) for safe brand-alias resolution.
DEFAULT_NDC_PATH = Path("data/lattice_sources/raw/drug/openfda_ndc.json.zip")
# Salt/hydrate suffix tokens stripped so a brand's active ingredient matches the profile's
# base entry ("pantoprazole sodium" -> "pantoprazole", "atorvastatin trihydrate" -> ...).
_DRUG_SALT_TOKENS = frozenset({
    "sodium", "calcium", "hydrochloride", "hcl", "sulfate", "magnesium", "potassium",
    "bitartrate", "besylate", "maleate", "mesylate", "succinate", "tartrate", "fumarate",
    "citrate", "phosphate", "acetate", "hydrobromide", "hydrate", "monohydrate",
    "dihydrate", "trihydrate",
})
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


def _drug_base_ingredient(name: str) -> str:
    """Strip salt/hydrate suffix tokens so an NDC ingredient or profile canonical reduces to
    its base drug name ('pantoprazole sodium' -> 'pantoprazole')."""
    return " ".join(t for t in _norm(name).split() if t not in _DRUG_SALT_TOKENS)


@lru_cache(maxsize=2)
def _ndc_brand_ingredient_index(path_s: str) -> dict:
    """Map each brand token / ingredient (from SINGLE-active-ingredient openFDA NDC records) to
    the set of base ingredients it denotes. Combination products (multi-ingredient) are excluded
    -- they are inherently ambiguous. A token resolving to >1 base ingredient is ambiguous too."""
    path = Path(path_s)
    if not path.exists():
        return {}
    with zipfile.ZipFile(path) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            records = json.load(handle).get("results", [])
    token_to_bases: dict[str, set[str]] = {}
    for record in records:
        ingredients = record.get("active_ingredients") or []
        if len(ingredients) != 1:
            continue
        base = _drug_base_ingredient(ingredients[0].get("name") or "")
        if not base:
            continue
        brand = _norm(record.get("brand_name") or "")
        for token in set(brand.split()) | {base}:
            token_to_bases.setdefault(token, set()).add(base)
    return {token: frozenset(bases) for token, bases in token_to_bases.items()}


def resolve_drug_generic(surface: str, ndc_path: str | Path | None = None) -> str | None:
    """Resolve a drug surface (usually a brand) to its single base ingredient via openFDA NDC.
    None when absent or AMBIGUOUS (combination product, or a brand reused across ingredients) --
    strict by design, so a wrong alias is never invented."""
    index = _ndc_brand_ingredient_index(str(ndc_path or DEFAULT_NDC_PATH))
    bases = index.get(_norm(surface))
    return next(iter(bases)) if bases and len(bases) == 1 else None


def resolve_missing_drug_aliases(
    surfaces,
    *,
    profile_path: str | Path | None = None,
    ndc_path: str | Path | None = None,
) -> dict[str, str]:
    """SAFE auto data-fix. For each drug surface with no profile entry, resolve its base
    ingredient via openFDA NDC; if that ingredient ALREADY EXISTS as a drug profile entry with
    levels, add the surface as an alias to that entry and persist the profile. Never invents
    entries or levels and never resolves an ambiguous brand. Returns {surface: canonical} of the
    aliases added; unresolved surfaces are left untouched (still flagged for review)."""
    profile_path = Path(profile_path or DEFAULT_PROFILE_PATH)
    artifact = json.loads(profile_path.read_text())
    drugs = artifact.get("profiles", {}).get("drug", {})
    base_to_canonical = {
        _drug_base_ingredient(canonical): canonical
        for canonical, row in drugs.items()
        if row.get("levels") or row.get("level_counts")
    }
    fixed: dict[str, str] = {}
    for surface in surfaces:
        key = _norm(surface)
        if not key or lookup_entry(surface, "drug", profile_path) is not None:
            continue  # unknown/blank or already resolvable
        base = resolve_drug_generic(surface, ndc_path)
        canonical = base_to_canonical.get(base) if base else None
        if canonical is None or _norm(canonical) == key:
            continue  # generic absent from profile, or surface already is the canonical
        aliases = drugs[canonical].setdefault("aliases", [])
        if key not in {_norm(a) for a in aliases}:
            aliases.append(key)
            fixed[surface] = canonical
    if fixed:
        # canonical profile serialization (matches scripts/build_lattice_profiles.py) so a
        # persisted alias is a minimal diff, not a whole-file reformat.
        profile_path.write_text(json.dumps(artifact, indent=2, sort_keys=True))
        _load_cached.cache_clear()
        _index_cached.cache_clear()
    return fixed
