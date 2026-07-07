---
type: plan
status: current
created: 2026-07-07
updated: 2026-07-07
tags: [substitution, lattice, fine-types, datasets, ontology, privacy, plan]
companion: [../specs/lattice-substitutor.md, ../research/datasets.md, 2026-07-07-lattice-substitutor-fine-types.md]
---

# Exhaustive Fine-Type Lattice Dataset Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible offline artifact pipeline that fills fine-type generalization lattices from
available dataset-backed taxonomies and ontologies, while preserving the runtime contract in
`docs/specs/lattice-substitutor.md`.

**Architecture:** Add a source-ingestion layer that normalizes downloaded datasets into a single
`LatticeProfile` artifact keyed by runtime type and surface. Runtime substitution reads only the approved
local artifact plus existing deterministic rules; it never downloads data, calls remote APIs, or uses `DEM`
as a hidden fallback for fine leaves.

**Tech Stack:** Python stdlib (`csv`, `json`, `zipfile`, `xml.etree.ElementTree`, `urllib` only in explicit
fetch scripts), existing `cloak` modules, pytest. Avoid new production dependencies unless explicitly
approved.

## Global Constraints

- Fine detector types are externally visible runtime substitution types in `Span.type`, `R[*].type`,
  action tables, probe pools, and placeholders.
- `DEM` remains research-eval rollup only; no new fine-mode runtime path may emit or silently inherit `DEM`.
- Generalization text levels must be grammatical replacements, never terminal type-name phrases such as
  `a nationality`, `an ethnicity`, `a profession`, or `a health condition`.
- Every non-direct fine lattice terminates in a typed placeholder such as `<HEALTH_CONDITION_1>`.
- Dataset coverage builds candidate lattices, not privacy claims. Claims require attacker-measured privacy
  at matched realized privacy.
- Deployed substitution must be local and deterministic. Source downloads, Wikidata/SPARQL queries, and
  teacher generation are explicit offline artifact-build steps only.
- Do not add production dependencies or hit external/rate-limited APIs without explicit user confirmation.
- Long/background jobs must run unbuffered with `.venv/bin/python -u ...`; any heavy/GPU run must pass the
  repo performance gate first.
- Docs in `docs/**/*.md` need frontmatter with `type`, `status`, `created`, `updated`, and `tags`.

## File Structure

- Create `src/cloak/lattice_profiles.py`
  - Owns artifact schema, profile loading, alias lookup, candidate lookup, and validation helpers.
- Modify `src/cloak/lattice.py`
  - Uses `lattice_profiles.lookup_levels(surface, runtime_type)` before strict WordNet / offline teacher
    cache for fine hierarchical leaves.
- Modify `src/cloak/anonymity.py`
  - Uses approved profile counts for fine hierarchical replacements before strict WordNet fallback.
- Create `scripts/build_lattice_profiles.py`
  - Durable artifact builder; consumes cached raw source files and writes approved local profiles.
- Create `scripts/lattice_sources/`
  - Parser modules for each source family. These are source-specific and never imported by runtime code.
- Create `scripts/download/fetch_lattice_sources.py`
  - Explicit, manually run downloader for open sources. It writes raw files under `data/lattice_sources/raw`.
    It must print each URL before fetching and must not run from tests.
- Create `data/lattice_sources/README.md`
  - Documents raw-source placement, licensing notes, and exact build commands.
- Create generated artifact path `data/lattice_profiles/fine_lattice_profiles.json`
  - Not hand-edited; produced by `scripts/build_lattice_profiles.py`.
- Create tests:
  - `src/cloak/tests/test_lattice_profiles.py`
  - `src/cloak/tests/test_lattice_profile_builders.py`

## Artifact Schema

`data/lattice_profiles/fine_lattice_profiles.json` must use this shape:

```json
{
  "schema_version": 1,
  "created": "2026-07-07",
  "sources": {
    "profession": ["esco", "onet", "isco08", "wikidata-p106"],
    "health-condition": ["mondo", "disease-ontology", "mesh", "icd11", "umls"]
  },
  "profiles": {
    "profession": {
      "journalist": {
        "aliases": ["reporter", "correspondent"],
        "levels": ["media worker"],
        "source_ids": ["esco:2642.7", "onet:27-3023.00"],
        "count": 1000.0
      }
    }
  }
}
```

Runtime helper signatures:

```python
def load_profiles(path: str | Path | None = None) -> dict: ...
def lookup_levels(surface: str, runtime_type: str, path: str | Path | None = None) -> list[str]: ...
def lookup_count(fill: str, runtime_type: str, path: str | Path | None = None) -> float | None: ...
def validate_profile_artifact(artifact: dict) -> list[str]: ...
```

`lookup_levels()` returns approved text levels only, never the typed placeholder. `lattice_for()` appends the
placeholder terminal.

## Source Coverage Targets

| Runtime type | Primary sources | Runtime output policy |
|---|---|---|
| `profession` | ESCO, O*NET, ISCO-08, Wikidata P106 | Title/alias to sector/domain levels |
| `health-condition` | Mondo, Disease Ontology, MeSH RDF, ICD-11, UMLS | Disease/diagnosis synonym to condition-family levels |
| `ethnicity` | HANCESTRO, Census race/ethnicity standards, Wikidata | Ancestry/population term to grammatical ancestry/region levels |
| `nationality` | Wikidata demonyms, CLDR territory names, GeoNames, HANCESTRO regions | Demonym/country to region/continent levels |
| `religion` | Wikidata P140, ARDA, Wikidata subclass graph | Denomination to tradition/broad affiliation levels |
| `family-role` | KIN ontology, WordNet, Wikidata kinship, schema.org Person relations | Kinship term to broad family-role levels |
| `age` | deterministic parser; optional MeSH/CDC age groups | Numeric age to bucket/life-stage levels |
| `gender` | GSSO, Wikidata sex-or-gender values | Alias normalization only; placeholder-only by default |
| `marital-status` | FHIR marital status, HL7 marital-status terminology | Alias normalization only; placeholder-only by default |
| `sexual-orientation` | GSSO, Wikidata sexual-orientation values | Alias normalization only; placeholder-only by default |
| `demographic-other` | none | Placeholder-first residual bucket |

## Task 1 - Profile Schema and Runtime Loader

**Files:**
- Create: `src/cloak/lattice_profiles.py`
- Test: `src/cloak/tests/test_lattice_profiles.py`

**Interfaces:**
- Produces:
  - `DEFAULT_PROFILE_PATH: Path`
  - `load_profiles(path: str | Path | None = None) -> dict`
  - `lookup_levels(surface: str, runtime_type: str, path: str | Path | None = None) -> list[str]`
  - `lookup_count(fill: str, runtime_type: str, path: str | Path | None = None) -> float | None`
  - `validate_profile_artifact(artifact: dict) -> list[str]`
- Consumes:
  - `cloak.runtime_types.RUNTIME_TYPES`
  - `cloak.lattice.is_type_name_phrase`

- [ ] **Step 1: Write failing loader tests**

Add `src/cloak/tests/test_lattice_profiles.py`:

```python
import json

from cloak.lattice_profiles import (
    load_profiles,
    lookup_count,
    lookup_levels,
    validate_profile_artifact,
)


def _artifact():
    return {
        "schema_version": 1,
        "created": "2026-07-07",
        "sources": {"profession": ["esco"]},
        "profiles": {
            "profession": {
                "journalist": {
                    "aliases": ["reporter"],
                    "levels": ["media worker"],
                    "source_ids": ["esco:2642.7"],
                    "count": 1000.0,
                }
            },
            "health-condition": {
                "diabetes": {
                    "aliases": ["diabetes mellitus"],
                    "levels": ["endocrine condition", "chronic condition"],
                    "source_ids": ["mondo:0005015"],
                    "count": 1000.0,
                }
            },
        },
    }


def test_lookup_levels_by_surface_and_alias(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(_artifact()))

    assert lookup_levels("journalist", "profession", path) == ["media worker"]
    assert lookup_levels("Reporter", "profession", path) == ["media worker"]
    assert lookup_levels("diabetes mellitus", "health-condition", path) == [
        "endocrine condition",
        "chronic condition",
    ]


def test_lookup_count_by_runtime_type_and_fill(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(_artifact()))

    assert lookup_count("media worker", "profession", path) == 1000.0
    assert lookup_count("chronic condition", "health-condition", path) == 1000.0
    assert lookup_count("media worker", "health-condition", path) is None


def test_validate_rejects_type_name_phrases_and_unknown_types():
    art = _artifact()
    art["profiles"]["profession"]["bad"] = {
        "aliases": [],
        "levels": ["a profession"],
        "source_ids": ["manual:bad"],
        "count": 1000.0,
    }
    art["profiles"]["nosuchtype"] = {}

    errors = validate_profile_artifact(art)
    assert any("type-name phrase" in e for e in errors)
    assert any("unknown runtime type" in e for e in errors)


def test_load_missing_profile_returns_empty_artifact(tmp_path):
    got = load_profiles(tmp_path / "missing.json")
    assert got["schema_version"] == 1
    assert got["profiles"] == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_lattice_profiles.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'cloak.lattice_profiles'`.

- [ ] **Step 3: Implement `src/cloak/lattice_profiles.py`**

Use this implementation shape:

```python
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
```

- [ ] **Step 4: Run tests to verify green**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_lattice_profiles.py -q
```

Expected: `4 passed`.

## Task 2 - Source Parsers for Cached Raw Files

**Files:**
- Create: `scripts/lattice_sources/__init__.py`
- Create: `scripts/lattice_sources/common.py`
- Create: `scripts/lattice_sources/occupation.py`
- Create: `scripts/lattice_sources/obo.py`
- Create: `scripts/lattice_sources/categorical.py`
- Test: `src/cloak/tests/test_lattice_profile_builders.py`

**Interfaces:**
- Produces:
  - `scripts.lattice_sources.common.ProfileRow`
  - `scripts.lattice_sources.occupation.rows_from_onet_titles(path: Path) -> list[ProfileRow]`
  - `scripts.lattice_sources.occupation.rows_from_isco_csv(path: Path) -> list[ProfileRow]`
  - `scripts.lattice_sources.obo.rows_from_obo(path: Path, runtime_type: str, family_roots: dict[str, str]) -> list[ProfileRow]`
  - `scripts.lattice_sources.categorical.alias_rows(runtime_type: str, rows: dict[str, list[str]]) -> list[ProfileRow]`
- Consumes:
  - stdlib only

- [ ] **Step 1: Write parser tests with tiny fixture files**

Add to `src/cloak/tests/test_lattice_profile_builders.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from lattice_sources.categorical import alias_rows
from lattice_sources.obo import rows_from_obo
from lattice_sources.occupation import rows_from_isco_csv, rows_from_onet_titles


def test_onet_titles_to_profession_rows(tmp_path):
    src = tmp_path / "Alternate Titles.txt"
    src.write_text(
        "O*NET-SOC Code\tTitle\tAlternate Title\tShort Title\tSource(s)\n"
        "27-3023.00\tNews Analysts, Reporters, and Journalists\tReporter\tN\tsample\n"
    )

    rows = rows_from_onet_titles(src)

    assert rows[0].runtime_type == "profession"
    assert rows[0].surface == "news analysts, reporters, and journalists"
    assert "reporter" in rows[0].aliases
    assert "media worker" in rows[0].levels


def test_isco_csv_to_profession_rows(tmp_path):
    src = tmp_path / "isco.csv"
    src.write_text(
        "code,title,major_group\n"
        "2211,Generalist medical practitioners,Professionals\n"
    )

    rows = rows_from_isco_csv(src)

    assert rows[0].surface == "generalist medical practitioners"
    assert rows[0].levels == ["professional worker"]


def test_obo_rows_use_synonyms_and_family_roots(tmp_path):
    src = tmp_path / "doid.obo"
    src.write_text(
        "[Term]\n"
        "id: DOID:9351\n"
        "name: diabetes mellitus\n"
        "synonym: \"diabetes\" EXACT []\n"
        "is_a: DOID:28 ! endocrine system disease\n"
        "\n"
        "[Term]\n"
        "id: DOID:28\n"
        "name: endocrine system disease\n"
    )

    rows = rows_from_obo(src, "health-condition", {"DOID:28": "endocrine condition"})

    row = next(r for r in rows if r.surface == "diabetes mellitus")
    assert "diabetes" in row.aliases
    assert row.levels == ["endocrine condition"]


def test_categorical_alias_rows_have_no_levels():
    rows = alias_rows("marital-status", {"married": ["wedded", "spouse"]})

    assert rows[0].runtime_type == "marital-status"
    assert rows[0].surface == "married"
    assert rows[0].aliases == ["wedded", "spouse"]
    assert rows[0].levels == []
```

- [ ] **Step 2: Run parser tests to verify failure**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_lattice_profile_builders.py -q
```

Expected: import failure for missing `lattice_sources`.

- [ ] **Step 3: Implement common row model**

Create `scripts/lattice_sources/common.py`:

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProfileRow:
    runtime_type: str
    surface: str
    aliases: list[str] = field(default_factory=list)
    levels: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    count: float = 1000.0


def norm(text: str) -> str:
    return " ".join(str(text).strip().lower().split())
```

Create `scripts/lattice_sources/__init__.py`:

```python
"""Offline dataset parsers for fine lattice profile artifacts."""
```

- [ ] **Step 4: Implement occupation parser**

Create `scripts/lattice_sources/occupation.py`:

```python
import csv
from pathlib import Path

from lattice_sources.common import ProfileRow, norm


def _profession_levels(title: str, major_group: str = "") -> list[str]:
    t = norm(f"{title} {major_group}")
    if any(w in t for w in ("journalist", "reporter", "news analyst", "correspondent")):
        return ["media worker"]
    if any(w in t for w in ("medical", "physician", "doctor", "nurse", "health")):
        return ["healthcare worker"]
    if any(w in t for w in ("law", "legal", "judge", "prosecutor")):
        return ["legal professional"]
    if any(w in t for w in ("teacher", "education", "professor", "school")):
        return ["education worker"]
    if "professional" in t or major_group:
        return ["professional worker"]
    return ["worker"]


def rows_from_onet_titles(path: Path) -> list[ProfileRow]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            title = norm(r.get("Title", ""))
            alt = norm(r.get("Alternate Title", ""))
            if not title or not alt:
                continue
            rows.append(ProfileRow(
                runtime_type="profession",
                surface=title,
                aliases=[alt],
                levels=_profession_levels(f"{title} {alt}"),
                source_ids=[f"onet:{r.get('O*NET-SOC Code', '').strip()}"],
            ))
    return rows


def rows_from_isco_csv(path: Path) -> list[ProfileRow]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            title = norm(r.get("title", ""))
            if not title:
                continue
            rows.append(ProfileRow(
                runtime_type="profession",
                surface=title,
                aliases=[],
                levels=_profession_levels(title, r.get("major_group", "")),
                source_ids=[f"isco:{r.get('code', '').strip()}"],
            ))
    return rows
```

- [ ] **Step 5: Implement OBO parser**

Create `scripts/lattice_sources/obo.py`:

```python
import re
from pathlib import Path

from lattice_sources.common import ProfileRow, norm


def _terms(path: Path) -> list[dict]:
    terms = []
    cur = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "[Term]":
            if cur:
                terms.append(cur)
            cur = {"synonyms": [], "parents": []}
            continue
        if cur is None or not line:
            continue
        if line.startswith("id: "):
            cur["id"] = line[4:].strip()
        elif line.startswith("name: "):
            cur["name"] = line[6:].strip()
        elif line.startswith("synonym: "):
            m = re.search(r'"([^"]+)"', line)
            if m:
                cur["synonyms"].append(m.group(1))
        elif line.startswith("is_a: "):
            cur["parents"].append(line[6:].split()[0])
    if cur:
        terms.append(cur)
    return terms


def rows_from_obo(path: Path, runtime_type: str, family_roots: dict[str, str]) -> list[ProfileRow]:
    rows = []
    for term in _terms(path):
        name = norm(term.get("name", ""))
        if not name:
            continue
        levels = [family_roots[p] for p in term.get("parents", []) if p in family_roots]
        if not levels:
            continue
        rows.append(ProfileRow(
            runtime_type=runtime_type,
            surface=name,
            aliases=[norm(s) for s in term.get("synonyms", []) if norm(s) != name],
            levels=levels,
            source_ids=[term.get("id", "")],
        ))
    return rows
```

- [ ] **Step 6: Implement categorical parser**

Create `scripts/lattice_sources/categorical.py`:

```python
from lattice_sources.common import ProfileRow, norm


def alias_rows(runtime_type: str, rows: dict[str, list[str]]) -> list[ProfileRow]:
    return [
        ProfileRow(
            runtime_type=runtime_type,
            surface=norm(surface),
            aliases=[norm(a) for a in aliases],
            levels=[],
            source_ids=[f"manual:{runtime_type}:{norm(surface)}"],
            count=1.0,
        )
        for surface, aliases in rows.items()
    ]
```

- [ ] **Step 7: Run parser tests to verify green**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_lattice_profile_builders.py -q
```

Expected: `4 passed`.

## Task 3 - Artifact Builder

**Files:**
- Create: `scripts/build_lattice_profiles.py`
- Create: `data/lattice_sources/README.md`
- Modify: `src/cloak/tests/test_lattice_profile_builders.py`

**Interfaces:**
- Produces:
  - CLI: `scripts/build_lattice_profiles.py --raw-dir data/lattice_sources/raw --out data/lattice_profiles/fine_lattice_profiles.json`
  - function: `merge_rows(rows: list[ProfileRow]) -> dict`
- Consumes:
  - parsers from Task 2
  - validator from Task 1

- [ ] **Step 1: Add builder merge test**

Append to `src/cloak/tests/test_lattice_profile_builders.py`:

```python
from build_lattice_profiles import merge_rows
from lattice_sources.common import ProfileRow


def test_merge_rows_combines_aliases_levels_and_source_ids():
    artifact = merge_rows([
        ProfileRow("profession", "journalist", ["reporter"], ["media worker"], ["esco:1"], 1000.0),
        ProfileRow("profession", "journalist", ["correspondent"], ["media worker"], ["onet:2"], 1200.0),
    ])

    row = artifact["profiles"]["profession"]["journalist"]
    assert row["aliases"] == ["correspondent", "reporter"]
    assert row["levels"] == ["media worker"]
    assert row["source_ids"] == ["esco:1", "onet:2"]
    assert row["count"] == 1200.0
```

- [ ] **Step 2: Run builder test to verify failure**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_lattice_profile_builders.py::test_merge_rows_combines_aliases_levels_and_source_ids -q
```

Expected: import failure for missing `build_lattice_profiles`.

- [ ] **Step 3: Implement builder**

Create `scripts/build_lattice_profiles.py`:

```python
"""Build fine runtime lattice profiles from cached raw datasets.

Network access is deliberately absent here. Use scripts/download/fetch_lattice_sources.py or manual downloads
to populate data/lattice_sources/raw first.
"""
import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from cloak.lattice_profiles import validate_profile_artifact
from lattice_sources.categorical import alias_rows
from lattice_sources.common import ProfileRow, norm
from lattice_sources.obo import rows_from_obo
from lattice_sources.occupation import rows_from_isco_csv, rows_from_onet_titles

HEALTH_FAMILY_ROOTS = {
    "DOID:28": "endocrine condition",
    "MONDO:0005151": "infectious disease",
    "MONDO:0005084": "respiratory condition",
    "MONDO:0002025": "mental health condition",
}

CATEGORICAL_ALIASES = {
    "gender": {"female": ["woman"], "male": ["man"]},
    "marital-status": {"married": ["wedded"], "divorced": ["formerly married"], "widowed": ["widow", "widower"]},
    "sexual-orientation": {"gay": ["homosexual"], "bisexual": ["bi"], "lesbian": []},
}


def _add_row(dst: dict, row: ProfileRow) -> None:
    rt = row.runtime_type
    surface = norm(row.surface)
    cur = dst[rt].setdefault(surface, {"aliases": set(), "levels": [], "source_ids": set(), "count": 1.0})
    cur["aliases"].update(a for a in row.aliases if norm(a) and norm(a) != surface)
    for level in row.levels:
        level = norm(level)
        if level and level not in cur["levels"]:
            cur["levels"].append(level)
    cur["source_ids"].update(s for s in row.source_ids if s)
    cur["count"] = max(float(cur["count"]), float(row.count))


def merge_rows(rows: list[ProfileRow]) -> dict:
    merged = defaultdict(dict)
    for row in rows:
        _add_row(merged, row)
    profiles = {}
    for rt, entries in sorted(merged.items()):
        profiles[rt] = {}
        for surface, row in sorted(entries.items()):
            profiles[rt][surface] = {
                "aliases": sorted(row["aliases"]),
                "levels": row["levels"],
                "source_ids": sorted(row["source_ids"]),
                "count": row["count"],
            }
    return {"schema_version": 1, "created": str(date.today()), "sources": {}, "profiles": profiles}


def collect_rows(raw_dir: Path) -> list[ProfileRow]:
    rows = []
    onet = raw_dir / "onet" / "Alternate Titles.txt"
    if onet.exists():
        rows.extend(rows_from_onet_titles(onet))
    isco = raw_dir / "isco" / "isco.csv"
    if isco.exists():
        rows.extend(rows_from_isco_csv(isco))
    for name in ("mondo.obo", "doid.obo"):
        path = raw_dir / "health" / name
        if path.exists():
            rows.extend(rows_from_obo(path, "health-condition", HEALTH_FAMILY_ROOTS))
    for runtime_type, aliases in CATEGORICAL_ALIASES.items():
        rows.extend(alias_rows(runtime_type, aliases))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/lattice_sources/raw")
    ap.add_argument("--out", default="data/lattice_profiles/fine_lattice_profiles.json")
    args = ap.parse_args()
    rows = collect_rows(Path(args.raw_dir))
    artifact = merge_rows(rows)
    errors = validate_profile_artifact(artifact)
    if errors:
        raise SystemExit("invalid lattice profile artifact:\n" + "\n".join(errors[:50]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    print(f"rows={len(rows)} profiles={sum(len(v) for v in artifact['profiles'].values())} -> {out}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Document raw source placement**

Create `data/lattice_sources/README.md`:

```markdown
# Lattice source cache

This directory holds raw files used by `scripts/build_lattice_profiles.py`.

Runtime substitution never reads this directory and never downloads data. Raw sources are fetched or placed
manually, then normalized into `data/lattice_profiles/fine_lattice_profiles.json`.

Expected paths:

- `raw/onet/Alternate Titles.txt`
- `raw/isco/isco.csv`
- `raw/health/mondo.obo`
- `raw/health/doid.obo`

Build:

```bash
PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_lattice_profiles.py \
  --raw-dir data/lattice_sources/raw \
  --out data/lattice_profiles/fine_lattice_profiles.json
```
```

- [ ] **Step 5: Run builder tests**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_lattice_profile_builders.py -q
```

Expected: all builder/parser tests pass.

## Task 4 - Runtime Integration

**Files:**
- Modify: `src/cloak/lattice.py`
- Modify: `src/cloak/anonymity.py`
- Test: `src/cloak/tests/test_lattice_profiles.py`

**Interfaces:**
- Consumes:
  - `lookup_levels(surface, runtime_type, path=None)`
  - `lookup_count(fill, runtime_type, path=None)`
- Produces:
  - `lattice_for()` uses dataset-backed levels before WordNet/teacher fallback for hierarchical fine leaves.
  - `aset_count()` accepts approved dataset-backed levels in strict mode.

- [ ] **Step 1: Add runtime integration tests**

Append to `src/cloak/tests/test_lattice_profiles.py`:

```python
def test_lattice_for_uses_profile_levels(monkeypatch, tmp_path):
    import cloak.lattice as lat
    import cloak.lattice_profiles as lp

    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(_artifact()))
    monkeypatch.setattr(lp, "DEFAULT_PROFILE_PATH", path)
    lp._load_cached.cache_clear()

    got = lat.lattice_for("reporter", "profession", "The reporter called.")

    assert got == ["media worker", "<PROFESSION_1>"]


def test_aset_count_uses_profile_count(monkeypatch, tmp_path):
    import cloak.anonymity as anon
    import cloak.lattice_profiles as lp

    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(_artifact()))
    monkeypatch.setattr(lp, "DEFAULT_PROFILE_PATH", path)
    lp._load_cached.cache_clear()

    assert anon.aset_count("media worker", "profession", "journalist", strict=True) == 1000.0
```

- [ ] **Step 2: Run integration tests to verify failure**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  src/cloak/tests/test_lattice_profiles.py::test_lattice_for_uses_profile_levels \
  src/cloak/tests/test_lattice_profiles.py::test_aset_count_uses_profile_count -q
```

Expected: assertions fail because runtime does not consult profiles yet.

- [ ] **Step 3: Update `lattice_for()`**

In `src/cloak/lattice.py`, import `lookup_levels`:

```python
from cloak.lattice_profiles import lookup_levels
```

Inside the hierarchical fine-leaf branch, before `_fine_curated_chain()`:

```python
got = lookup_levels(span_text, span_type)
deterministic = bool(got)
if not got:
    got = _fine_curated_chain(span_text, span_type)
    deterministic = got is not None
```

Keep the existing typed placeholder append path unchanged.

- [ ] **Step 4: Update `aset_count()`**

In `src/cloak/anonymity.py`, import `lookup_count`:

```python
from cloak.lattice_profiles import lookup_count
```

In the `span_type in FINE_DEM_TYPES` branch, before `_APPROVED_FINE_COUNTS`:

```python
got = lookup_count(fill, span_type)
if got is None:
    got = _APPROVED_FINE_COUNTS.get(span_type, {}).get(fill.lower().strip())
```

- [ ] **Step 5: Run integration tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_lattice_profiles.py -q
```

Expected: all lattice profile tests pass.

## Task 5 - Source Fetcher, Explicit Only

**Files:**
- Create: `scripts/download/fetch_lattice_sources.py`
- Modify: `data/lattice_sources/README.md`

**Interfaces:**
- Produces:
  - CLI with source flags: `--onet`, `--disease-ontology`, `--mondo`
- Consumes:
  - Python stdlib `urllib.request`

- [ ] **Step 1: Create fetcher with explicit flags**

Create `scripts/download/fetch_lattice_sources.py`:

```python
"""Explicit downloader for open lattice source files.

This script is never imported by runtime code and should be run only when the user explicitly wants to
refresh local source artifacts.
"""
import argparse
import urllib.request
from pathlib import Path

SOURCES = {
    "disease-ontology": (
        "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.obo",
        "health/doid.obo",
    ),
    "mondo": (
        "https://raw.githubusercontent.com/monarch-initiative/mondo/master/src/ontology/mondo.obo",
        "health/mondo.obo",
    ),
}


def fetch(name: str, raw_dir: Path) -> Path:
    url, rel = SOURCES[name]
    out = raw_dir / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetch {name}: {url} -> {out}", flush=True)
    urllib.request.urlretrieve(url, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/lattice_sources/raw")
    ap.add_argument("--source", action="append", choices=sorted(SOURCES), required=True)
    args = ap.parse_args()
    for source in args.source:
        fetch(source, Path(args.raw_dir))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add README fetch examples**

Append to `data/lattice_sources/README.md`:

```markdown
Optional explicit fetch for open OBO sources:

```bash
PYTHONPATH=src:scripts .venv/bin/python -u scripts/download/fetch_lattice_sources.py \
  --source disease-ontology \
  --source mondo
```

O*NET, ESCO, ISCO, UMLS, ICD-11, Wikidata, and ARDA may have terms, licenses, credentials, or rate limits.
Place their exported files manually under `raw/` unless the user explicitly approves an automated fetch path.
```

- [ ] **Step 3: Syntax-check scripts without fetching**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m py_compile \
  scripts/download/fetch_lattice_sources.py \
  scripts/build_lattice_profiles.py
```

Expected: exit 0. Do not run the fetcher unless the user explicitly asks.

## Task 6 - Exhaustiveness Report and Failure Modes

**Files:**
- Modify: `scripts/build_lattice_profiles.py`
- Create: `results/lattice_profile_coverage.example.json` only if the repo convention accepts checked-in
  examples; otherwise keep reports generated under `results/` untracked.
- Test: `src/cloak/tests/test_lattice_profile_builders.py`

**Interfaces:**
- Produces:
  - `coverage_report(artifact: dict) -> dict`
  - CLI option `--coverage-out results/lattice_profile_coverage.json`

- [ ] **Step 1: Add coverage report test**

Append to `src/cloak/tests/test_lattice_profile_builders.py`:

```python
from build_lattice_profiles import coverage_report


def test_coverage_report_marks_placeholder_only_types_separately():
    art = {
        "schema_version": 1,
        "created": "2026-07-07",
        "sources": {},
        "profiles": {"profession": {"journalist": {"levels": ["media worker"]}}},
    }

    report = coverage_report(art)

    assert report["profile_counts"]["profession"] == 1
    assert "gender" in report["placeholder_only_types"]
    assert "demographic-other" in report["placeholder_first_types"]
```

- [ ] **Step 2: Implement coverage report**

In `scripts/build_lattice_profiles.py`, add:

```python
from cloak.runtime_types import PLACEHOLDER_ONLY_TYPES


def coverage_report(artifact: dict) -> dict:
    profiles = artifact.get("profiles", {})
    return {
        "profile_counts": {runtime_type: len(rows) for runtime_type, rows in sorted(profiles.items())},
        "placeholder_only_types": sorted(PLACEHOLDER_ONLY_TYPES),
        "placeholder_first_types": ["demographic-other"],
        "notes": [
            "Coverage counts candidate surfaces only; they are not privacy guarantees.",
            "Every emitted text level still requires lattice filters and anonymity floors.",
        ],
    }
```

In `main()`, add argument:

```python
ap.add_argument("--coverage-out", default=None)
```

After writing the artifact:

```python
if args.coverage_out:
    cov = Path(args.coverage_out)
    cov.parent.mkdir(parents=True, exist_ok=True)
    cov.write_text(json.dumps(coverage_report(artifact), indent=2, sort_keys=True))
    print(f"coverage -> {cov}", flush=True)
```

- [ ] **Step 3: Run coverage tests**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest \
  src/cloak/tests/test_lattice_profile_builders.py::test_coverage_report_marks_placeholder_only_types_separately -q
```

Expected: pass.

## Task 7 - End-to-End Artifact Smoke

**Files:**
- Test: `src/cloak/tests/test_lattice_profile_builders.py`

**Interfaces:**
- Consumes:
  - `build_lattice_profiles.main()` through CLI subprocess
- Produces:
  - a tiny generated artifact under pytest `tmp_path`

- [ ] **Step 1: Add CLI smoke test**

Append to `src/cloak/tests/test_lattice_profile_builders.py`:

```python
import json
import subprocess


def test_build_lattice_profiles_cli_smoke(tmp_path):
    raw = tmp_path / "raw"
    (raw / "onet").mkdir(parents=True)
    (raw / "onet" / "Alternate Titles.txt").write_text(
        "O*NET-SOC Code\tTitle\tAlternate Title\tShort Title\tSource(s)\n"
        "27-3023.00\tNews Analysts, Reporters, and Journalists\tReporter\tN\tsample\n"
    )
    out = tmp_path / "profiles.json"
    cov = tmp_path / "coverage.json"

    subprocess.run(
        [
            ".venv/bin/python",
            "-u",
            "scripts/build_lattice_profiles.py",
            "--raw-dir",
            str(raw),
            "--out",
            str(out),
            "--coverage-out",
            str(cov),
        ],
        check=True,
        env={"PYTHONPATH": "src:scripts"},
    )

    art = json.loads(out.read_text())
    assert art["profiles"]["profession"]
    assert json.loads(cov.read_text())["profile_counts"]["profession"] >= 1
```

- [ ] **Step 2: Run CLI smoke**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest \
  src/cloak/tests/test_lattice_profile_builders.py::test_build_lattice_profiles_cli_smoke -q
```

Expected: pass and no network access.

## Task 8 - Documentation Update

**Files:**
- Modify: `docs/specs/lattice-substitutor.md`
- Modify: `docs/research/datasets.md`
- Modify: `data/lattice_sources/README.md`

**Interfaces:**
- Produces:
  - documented artifact path
  - documented source refresh/build commands
  - explicit empirical-honesty caveat

- [ ] **Step 1: Update spec with artifact contract**

In `docs/specs/lattice-substitutor.md`, under the fine-type source registry, add:

```markdown
The durable generated artifact is `data/lattice_profiles/fine_lattice_profiles.json`.
Runtime code may read this artifact but must not read raw source files or call source APIs.
Raw source files live under `data/lattice_sources/raw/` and are consumed only by
`scripts/build_lattice_profiles.py`.
```

- [ ] **Step 2: Update datasets survey with build status**

In `docs/research/datasets.md`, under "Top sources by fine type for lattice construction", add:

```markdown
Build status: these sources seed `data/lattice_profiles/fine_lattice_profiles.json` through the offline
builder planned in `docs/plans/2026-07-07-fine-lattice-dataset-build.md`. Placeholder-only leaves use these
sources for alias normalization only, not semantic replacement text.
```

- [ ] **Step 3: Run doc whitespace check**

Run:

```bash
git diff --check -- docs/specs/lattice-substitutor.md docs/research/datasets.md data/lattice_sources/README.md
```

Expected: no output.

## Verification Protocol

Minimum local verification before calling implementation complete:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest \
  src/cloak/tests/test_lattice_profiles.py \
  src/cloak/tests/test_lattice_profile_builders.py -q

PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_fine_runtime_types.py -q

PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests -q

PYTHONPATH=src:scripts .venv/bin/python -m py_compile \
  scripts/download/fetch_lattice_sources.py \
  scripts/build_lattice_profiles.py

git diff --check
```

If implementation changes deployed substitution behavior beyond profile lookup, also run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_extract.py -q
PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_train_roundtrip_mode.py -q
```

Do not run source fetchers, Wikidata queries, ICD/UMLS API calls, or long ontology builds without explicit
user approval. If a source build exceeds 10 minutes, run the repo performance gate first and confirm that the
smallest useful build is being run.

## Self-Review Checklist

- [ ] Every fine runtime type in `docs/specs/lattice-substitutor.md` has either a dataset-backed source path,
  deterministic rule path, or explicit placeholder-only policy.
- [ ] No task introduces `DEM` as a fine-runtime fallback.
- [ ] No source parser emits terminal type-name phrases.
- [ ] No runtime path downloads data or calls remote APIs.
- [ ] No new production dependency is required.
- [ ] Tests cover alias lookup, source parsing, artifact validation, runtime lattice lookup, anonymity counts,
  and CLI smoke.
- [ ] Coverage reports are documented as candidate coverage, not privacy evidence.
