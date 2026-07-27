---
type: plan
status: current
created: 2026-07-10
updated: 2026-07-27
tags: [lattice, lattice-profiles, deduplication, aliases, entity-resolution, qa-build, profile-matching]
companion: [docs/specs/lattice-entry-dedup-and-span-resolution.md]
---

# Lattice Entry Dedup + Span Resolution Implementation Plan

> **Post-refactor note (2026-07-27):** module paths in this doc were updated to the
> regrouped `src/cloak` layout — see the path mapping in [the cleanup plan](2026-07-27-codebase-cleanup-refactor.md).
> Still named here but **deleted, not moved** in that cleanup: `scripts/build_probes.py` (superseded by `scripts/build_arms_artifact.py` + `scripts/build_qa_utility_artifact.py`) and `train/ladder_probes.py` (probe tier retired, no successor).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detected spans resolve to a canonical lattice entry (dedup co-referent surfaces per
document), and true synonym rows in `lattice_profiles.json` merge into one canonical row +
alias union.

**Architecture:** Three layers per the spec
(`docs/specs/lattice-entry-dedup-and-span-resolution.md`): (1) the profile loader exposes
canonical identity (`lookup_entry`) and the retrieve-then-verify matcher carries it on exact
hits; (2) the QA build resolves lattice spans through the *same* `match_spans_batch` the
substitutor uses and dedups by `(runtime_type, entry)`; (3) a new
`lattice_producer/entity_merge.py` merges synonym rows — ontology (DOID) oracle auto-merges,
a precision-calibrated cross-encoder gate may auto-merge unlinked identical-level pairs,
everything else goes to a review report — wired into the producer graph and a reprocess CLI.

**Tech Stack:** Python 3.12, pytest, numpy, sentence-transformers (bge-small, already the
embindex model), HF cross-encoder (calibration + gate), the vendored
`data/lattice_sources/raw/health/doid.obo`.

## Global Constraints

- **No duplicated machinery:** the QA build resolves spans via
  `cloak.lattice.profile_match.match_spans_batch` — never a parallel matcher (user hard requirement).
- **Never merge rows on identical levels alone** — sibling entities legitimately share ladders.
- **Empirical honesty:** the cross-encoder gate threshold is calibrated once on the
  ontology-derived eval set and frozen; if no threshold reaches precision ≥ 0.999 at recall
  ≥ 0.10, the gate ships disabled (review-file only) and the numbers are the finding.
- **Term hygiene:** merged-pair/duplicate-claim listings go to JSON report files, never stdout.
  Test fixtures use synthetic invented names (e.g. `blorbitis`), not real medical terms.
- **Tests:** run with `.venv/bin/python -m pytest` from the repo root (GPU venv). Unit tests
  must not require GPU or network — heavy models are injectable (`embed_fn`, `gate_fn`,
  `nli_batch_fn` parameters, monkeypatch).
- **Artifact writes:** `json.dumps(artifact, indent=2, sort_keys=True)` (existing script
  convention), atomic (write temp + rename, or `cloak.lattice.producer.io.atomic_write_json`
  — note that helper uses indent=2 without sort_keys, so scripts write directly).
- **Commits:** path-scoped `git add <files>` only — this is a shared checkout with unrelated
  modified files. Never `git add -A`/`git commit -a`. Check `git diff --cached --name-only`
  is empty before staging.
- **GPU:** one GPU process at a time; long runs via `.venv/bin/python -u` to a log file.
- **Paid teacher calls** (the end-to-end QA-build revalidation) require explicit user OK first
  — the plan stops there.

---

### Task 1: Canonical identity in loader + matcher exact hits

**Files:**
- Modify: `src/cloak/lattice/profiles.py:81-126` (`_build_indexes`, new `lookup_entry`,
  `lookup_levels` becomes wrapper)
- Modify: `src/cloak/lattice/profile_match.py:42-49,179-186` (exact-hit `MatchResult.entry`)
- Modify: `src/cloak/detection/span_prep.py:114` (exact `match` block carries `entry`)
- Test: `src/cloak/tests/test_lattice_profiles.py`, `src/cloak/tests/test_profile_match.py`

**Interfaces:**
- Produces: `lookup_entry(surface: str, runtime_type: str, path: str | Path | None = None) ->
  tuple[str, list[str]] | None` returning `(canonical, levels)`; `MatchResult.entry` is the
  matched canonical string on **every** hit (exact and semantic), `None` only inside the
  batch-result dict where the value itself is `None` (abstain).
- Consumed by: Task 6 (`_detect_docs` dedups on `MatchResult.entry`).

- [ ] **Step 1: Write the failing tests**

Append to `src/cloak/tests/test_lattice_profiles.py` (follow the file's existing artifact-dict
fixture style — it builds artifacts inline and points the loader at a tmp JSON file):

```python
def test_lookup_entry_returns_canonical_for_canonical_and_alias(tmp_path):
    artifact = {
        "schema_version": 1, "created": "2026-07-10", "sources": {},
        "profiles": {"health-condition": {
            "blorbitis": {"aliases": ["blorb inflammation"],
                          "levels": ["organ disease", "disease"],
                          "source_ids": ["t:1"], "count": 100.0},
        }},
    }
    p = tmp_path / "profiles.json"
    p.write_text(json.dumps(artifact))
    from cloak.lattice.profiles import lookup_entry, lookup_levels
    assert lookup_entry("Blorbitis", "health-condition", p) == \
        ("blorbitis", ["organ disease", "disease"])
    assert lookup_entry("blorb  inflammation", "health-condition", p) == \
        ("blorbitis", ["organ disease", "disease"])
    assert lookup_entry("unknownitis", "health-condition", p) is None
    # wrapper unchanged behavior
    assert lookup_levels("blorb inflammation", "health-condition", p) == \
        ["organ disease", "disease"]
    assert lookup_levels("unknownitis", "health-condition", p) == []
```

Append to `src/cloak/tests/test_profile_match.py` (reuse that file's existing tmp-profile
helper if one exists; otherwise the same inline artifact pattern):

```python
def test_exact_hit_carries_canonical_entry(tmp_path):
    artifact = {
        "schema_version": 1, "created": "2026-07-10", "sources": {},
        "profiles": {"health-condition": {
            "blorbitis": {"aliases": ["blorb inflammation"],
                          "levels": ["organ disease"], "source_ids": ["t:1"], "count": 10.0},
        }},
    }
    p = tmp_path / "profiles.json"
    p.write_text(json.dumps(artifact))
    from cloak.lattice.profile_match import match_spans_batch, span_key
    got = match_spans_batch([("blorb inflammation", "health-condition", "ctx sentence")],
                            profiles_path=p)
    m = got[span_key("blorb inflammation", "health-condition")]
    assert m is not None and m.kind == "exact"
    assert m.entry == "blorbitis"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_profiles.py -k lookup_entry src/cloak/tests/test_profile_match.py -k carries_canonical -v`
Expected: FAIL (`ImportError: cannot import name 'lookup_entry'`; `m.entry is None`).

- [ ] **Step 3: Implement**

In `src/cloak/lattice/profiles.py`, `_build_indexes` — store the canonical next to levels:

```python
        for canonical, row in entries.items():
            levels = list(row.get("levels", []))
            for key in [_norm(canonical), *[_norm(a) for a in row.get("aliases", [])]]:
                if key:
                    surface_index.setdefault(key, (canonical, levels))
```

Replace `lookup_levels` with:

```python
def lookup_entry(surface: str, runtime_type: str,
                 path: str | Path | None = None) -> tuple[str, list[str]] | None:
    """Resolve a surface to its profile row: (canonical, levels). None = no entry."""
    key = _norm(surface)
    idx = _index_cached(str(path or DEFAULT_PROFILE_PATH))
    got = idx["by_surface"].get(runtime_type, {}).get(key)
    if got is None:
        return None
    canonical, levels = got
    return canonical, list(levels)


def lookup_levels(surface: str, runtime_type: str, path: str | Path | None = None) -> list[str]:
    got = lookup_entry(surface, runtime_type, path)
    return got[1] if got else []
```

In `src/cloak/lattice/profile_match.py` — exact path (currently `lp.lookup_levels(...)` around
line 180) becomes:

```python
    for key, (span_text, context) in todo.items():
        got = lp.lookup_entry(span_text, key[0], profiles_path)
        if got:
            out[key] = MatchResult(list(got[1]), "exact", True, 1.0, got[0])
        else:
            out[key] = None
            if context:
                misses.append((key, span_text, context))
```

Update the `MatchResult.entry` field comment: `# matched canonical (exact and semantic hits)`.

In `src/cloak/detection/span_prep.py:114`, the exact `match` diagnostic gains the entry:

```python
                entry["match"] = ({"kind": "exact", "entry": m.entry} if m.kind == "exact" else
```

(keep the semantic branch exactly as is).

- [ ] **Step 4: Run the focused tests, then the touched suites**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_profiles.py src/cloak/tests/test_profile_match.py src/cloak/tests/test_substitute_prepass.py src/cloak/tests/test_lattice_for_proposals.py -v`
Expected: one known failure — `src/cloak/tests/test_substitute_prepass.py:44` asserts the exact
match block equals `{"kind": "exact"}` verbatim; update that assertion to
`{"kind": "exact", "entry": "aspirin"}` (the fixture's canonical — check the fixture's profile
row and use its actual canonical key). Then all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cloak/lattice/profiles.py src/cloak/lattice/profile_match.py src/cloak/detection/span_prep.py \
        src/cloak/tests/test_lattice_profiles.py src/cloak/tests/test_profile_match.py \
        src/cloak/tests/test_substitute_prepass.py
git commit -m "feat(lattice): lookup_entry canonical identity; exact hits carry entry"
```

---

### Task 2: DOID parser gains exact synonyms + obsolete flag

**Files:**
- Modify: `src/cloak/lattice/producer/reference_sources.py:143-176` (`DoidNode`,
  `load_doid_index`)
- Test: `src/cloak/tests/test_lattice_producer_reference_sources.py`

**Interfaces:**
- Produces: `DoidNode` gains `exact_synonyms: list[str]` and `obsolete: bool`. Existing
  fields/consumers unchanged.
- Consumed by: Task 3 (`doid_surface_index`), Task 4 (calibration pair building).

- [ ] **Step 1: Write the failing test**

Append to `src/cloak/tests/test_lattice_producer_reference_sources.py`:

```python
OBO_FIXTURE = """format-version: 1.2

[Term]
id: DOID:0000001
name: blorbitis
synonym: "blorb inflammation" EXACT []
synonym: "blorby feeling" RELATED []
is_a: DOID:0000009 ! organ disease

[Term]
id: DOID:0000002
name: old blorbitis
is_obsolete: true

[Term]
id: DOID:0000009
name: organ disease
"""


def test_doid_index_parses_exact_synonyms_and_obsolete(tmp_path):
    obo = tmp_path / "mini.obo"
    obo.write_text(OBO_FIXTURE)
    from cloak.lattice.producer.reference_sources import load_doid_index
    nodes = load_doid_index(str(obo))
    assert nodes["DOID:0000001"].exact_synonyms == ["blorb inflammation"]
    assert nodes["DOID:0000001"].obsolete is False
    assert nodes["DOID:0000002"].obsolete is True
    assert nodes["DOID:0000009"].exact_synonyms == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_reference_sources.py -k exact_synonyms -v`
Expected: FAIL (`AttributeError: 'DoidNode' object has no attribute 'exact_synonyms'`).

- [ ] **Step 3: Implement**

In `reference_sources.py`:

```python
@dataclass
class DoidNode:
    id: str
    name: str
    parents: list[str] = field(default_factory=list)
    exact_synonyms: list[str] = field(default_factory=list)
    obsolete: bool = False


_TERM_SYN_EXACT_RE = re.compile(r'^synonym:\s*"(.+?)"\s+EXACT\b', re.M)
_TERM_OBSOLETE_RE = re.compile(r"^is_obsolete:\s*true", re.M)
```

and in `load_doid_index`, extend the node construction:

```python
        nodes[node_id] = DoidNode(
            id=node_id,
            name=name_match.group(1).strip(),
            parents=_TERM_ISA_RE.findall(stanza),
            exact_synonyms=_TERM_SYN_EXACT_RE.findall(stanza),
            obsolete=bool(_TERM_OBSOLETE_RE.search(stanza)),
        )
```

- [ ] **Step 4: Run the file's full test suite**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_reference_sources.py -v`
Expected: PASS (existing DOID tests untouched — additive fields only).

- [ ] **Step 5: Commit**

```bash
git add src/cloak/lattice/producer/reference_sources.py \
        src/cloak/tests/test_lattice_producer_reference_sources.py
git commit -m "feat(lattice-producer): DOID parser carries exact synonyms + obsolete flag"
```

---

### Task 3: entity_merge module

**Files:**
- Create: `src/cloak/lattice/producer/entity_merge.py`
- Test: `src/cloak/tests/test_lattice_producer_entity_merge.py`

**Interfaces:**
- Consumes: `load_doid_index` (Task 2 fields).
- Produces:
  - `doid_surface_index(obo_path: str) -> dict[str, str]` — norm surface → DOID id
    (non-obsolete; ambiguous surfaces excluded).
  - `doid_preferred_names(obo_path: str) -> dict[str, str]` — id → name.
  - `doid_synonyms(obo_path: str) -> dict[str, list[str]]` — id → exact synonyms.
  - `merge_runtime_type(entries, *, oracle_index=None, preferred_name=None,
    ontology_synonyms=None, gate_fn=None, gate_threshold=None, embed_fn=None) ->
    tuple[dict, dict]` — `(new_entries, report)`.
  - `apply_entity_merge(artifact, *, obo_paths=None, gate_fn=None, gate_threshold=None,
    embed_fn=None) -> dict` — mutates `artifact` in place (like
    `coherence.normalize_coherence`), returns report.
  - `gate_fn` contract: `gate_fn(surfaces_a: list[str], surfaces_b: list[str]) -> float` —
    max same-entity score over the surface cross-product, in [0, 1].
- Consumed by: Task 4 (producer graph node), Task 5 (reprocess CLI).

- [ ] **Step 1: Write the failing tests**

Create `src/cloak/tests/test_lattice_producer_entity_merge.py`:

```python
import json

from cloak.lattice.producer.entity_merge import (
    apply_entity_merge,
    block_pairs,
    doid_surface_index,
    merge_runtime_type,
    row_ontology_id,
)

OBO_FIXTURE = """format-version: 1.2

[Term]
id: DOID:0000001
name: blorbitis
synonym: "blorb inflammation" EXACT []
is_a: DOID:0000009 ! organ disease

[Term]
id: DOID:0000003
name: glimmerosis
synonym: "glimmer syndrome" EXACT []
is_a: DOID:0000009 ! organ disease

[Term]
id: DOID:0000004
name: shared name thing
synonym: "ambiguous surface" EXACT []

[Term]
id: DOID:0000005
name: other shared thing
synonym: "ambiguous surface" EXACT []

[Term]
id: DOID:0000009
name: organ disease
"""


def _row(levels, aliases=(), count=10.0):
    return {"aliases": list(aliases), "levels": list(levels),
            "source_ids": ["t:1"], "count": count}


def _obo(tmp_path):
    p = tmp_path / "mini.obo"
    p.write_text(OBO_FIXTURE)
    return str(p)


def test_surface_index_skips_ambiguous_and_maps_synonyms(tmp_path):
    idx = doid_surface_index(_obo(tmp_path))
    assert idx["blorbitis"] == "DOID:0000001"
    assert idx["blorb inflammation"] == "DOID:0000001"
    assert "ambiguous surface" not in idx          # claimed by two ids -> excluded


def test_row_ontology_id_requires_unanimous_surfaces(tmp_path):
    idx = doid_surface_index(_obo(tmp_path))
    assert row_ontology_id("blorbitis", _row(["organ disease"]), idx) == "DOID:0000001"
    # aliases spanning two ids -> conflicting row, no id
    mixed = _row(["organ disease"], aliases=["glimmer syndrome"])
    assert row_ontology_id("blorbitis", mixed, idx) is None
    assert row_ontology_id("unknownitis", _row(["organ disease"]), idx) is None


def test_block_pairs_identical_levels_and_embedding_neighbors(tmp_path):
    entries = {
        "a": _row(["x", "y"]), "b": _row(["x", "y"]), "c": _row(["z"]),
    }
    assert block_pairs(entries) == {("a", "b")}
    # embedding blocking adds near neighbors even with different levels
    import numpy as np
    vecs = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.99, 0.14]}
    embed = lambda texts: np.array([vecs[t.split(" ; ")[0]] for t in texts])
    got = block_pairs(entries, embed_fn=embed)
    assert ("a", "b") in got and (("a", "c") in got or ("b", "c") in got)


def test_ontology_linked_rows_merge_with_alias_union(tmp_path):
    obo = _obo(tmp_path)
    entries = {
        "blorbitis": _row(["organ disease"], aliases=["blorby"], count=50.0),
        "blorb inflammation": _row(["organ disease"], aliases=["the blorbs"], count=20.0),
        "glimmerosis": _row(["organ disease"], count=5.0),   # sibling: same levels, other id
    }
    idx = doid_surface_index(obo)
    merged, report = merge_runtime_type(
        entries, oracle_index=idx,
        preferred_name={"DOID:0000001": "blorbitis"},
        ontology_synonyms={"DOID:0000001": ["blorb inflammation"]})
    assert "blorbitis" in merged and "blorb inflammation" not in merged
    assert "glimmerosis" in merged                       # sibling never merged
    row = merged["blorbitis"]
    assert set(row["aliases"]) >= {"blorby", "the blorbs", "blorb inflammation"}
    assert row["count"] == 50.0
    assert len(report["merged"]) == 1


def test_ontology_linked_but_levels_differ_goes_to_review(tmp_path):
    obo = _obo(tmp_path)
    entries = {
        "blorbitis": _row(["organ disease"]),
        "blorb inflammation": _row(["tissue disease"]),
    }
    merged, report = merge_runtime_type(entries, oracle_index=doid_surface_index(obo))
    assert set(merged) == set(entries)
    assert len(report["review"]) == 1


def test_gate_merges_unlinked_identical_levels_above_threshold(tmp_path):
    entries = {
        "flurbitis": _row(["organ disease"], count=30.0),
        "flurb disease": _row(["organ disease"], count=10.0),
        "glimmerosis": _row(["organ disease"], count=5.0),
    }
    def gate(sa, sb):
        return 0.99 if {"flurbitis"} & set(sa + sb) and {"flurb disease"} & set(sa + sb) else 0.1
    merged, report = merge_runtime_type(entries, gate_fn=gate, gate_threshold=0.95)
    assert "flurbitis" in merged and "flurb disease" not in merged
    assert "glimmerosis" in merged
    # without a gate the same pair is review-only
    merged2, report2 = merge_runtime_type(entries)
    assert set(merged2) == set(entries)
    assert any(r["why"] == "unlinked identical levels" for r in report2["review"])


def test_merged_row_level_counts_take_per_level_max():
    entries = {
        "flurbitis": {**_row(["organ disease"]), "level_counts": {"organ disease": 100.0}},
        "flurb disease": {**_row(["organ disease"]), "level_counts": {"organ disease": 400.0}},
    }
    merged, _ = merge_runtime_type(entries, gate_fn=lambda a, b: 1.0, gate_threshold=0.9)
    (row,) = merged.values()
    assert row["level_counts"] == {"organ disease": 400.0}


def test_apply_entity_merge_reports_duplicate_surface_claims(tmp_path):
    artifact = {"schema_version": 1, "created": "2026-07-10", "sources": {}, "profiles": {
        "LOC": {"springtown": _row(["city"], aliases=["the springs"]),
                "springtown two": _row(["city"], aliases=["the springs"])},
    }}
    report = apply_entity_merge(artifact)   # no oracle, no gate -> nothing merges
    assert set(artifact["profiles"]["LOC"]) == {"springtown", "springtown two"}
    assert report["duplicate_surface_claims"]["LOC"] == [
        {"surface": "the springs", "rows": ["springtown", "springtown two"]}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_entity_merge.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'cloak.lattice.producer.entity_merge'`).

- [ ] **Step 3: Implement `src/cloak/lattice/producer/entity_merge.py`**

```python
"""Entry-level entity merge: fold true synonym canonical rows into one row + alias union.

Spec: docs/specs/lattice-entry-dedup-and-span-resolution.md (Part 3). The entry-level analog
of coherence.py (which merges synonym *level strings*). Ontology-linked pairs auto-merge; a
precision-calibrated cross-encoder gate may auto-merge unlinked identical-level pairs;
everything else lands in the review report. Never merges on identical levels alone.
"""
from __future__ import annotations

import copy
import re
from collections import defaultdict
from typing import Any, Callable

from cloak.lattice.producer.reference_sources import DEFAULT_DOID_OBO, load_doid_index

BLOCK_TOP_K = 5
BLOCK_SIM_FLOOR = 0.80
DEFAULT_OBO_PATHS = {"health-condition": str(DEFAULT_DOID_OBO)}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _row_surfaces(canonical: str, row: dict[str, Any]) -> list[str]:
    return [canonical, *[str(a) for a in row.get("aliases", [])]]


def doid_surface_index(obo_path: str = str(DEFAULT_DOID_OBO)) -> dict[str, str]:
    """norm(surface) -> ontology id over names + EXACT synonyms of non-obsolete terms.
    A surface claimed by more than one id is ambiguous and excluded (never oracle evidence)."""
    claims: dict[str, set[str]] = defaultdict(set)
    for node in load_doid_index(obo_path).values():
        if node.obsolete:
            continue
        for surface in [node.name, *node.exact_synonyms]:
            key = _norm(surface)
            if key:
                claims[key].add(node.id)
    return {s: next(iter(ids)) for s, ids in claims.items() if len(ids) == 1}


def doid_preferred_names(obo_path: str = str(DEFAULT_DOID_OBO)) -> dict[str, str]:
    return {node.id: node.name for node in load_doid_index(obo_path).values()
            if not node.obsolete}


def doid_synonyms(obo_path: str = str(DEFAULT_DOID_OBO)) -> dict[str, list[str]]:
    return {node.id: list(node.exact_synonyms) for node in load_doid_index(obo_path).values()
            if not node.obsolete}


def row_ontology_id(canonical: str, row: dict[str, Any],
                    surface_index: dict[str, str]) -> str | None:
    """Unanimous ontology id across the row's surfaces; None when unknown or conflicting."""
    ids = {surface_index[key] for key in map(_norm, _row_surfaces(canonical, row))
           if key in surface_index}
    return next(iter(ids)) if len(ids) == 1 else None


def block_pairs(entries: dict[str, dict], embed_fn: Callable | None = None) -> set[tuple[str, str]]:
    """Candidate pairs: identical level-lists ∪ embedding top-k neighbors (recall-oriented)."""
    keys = sorted(entries)
    pairs: set[tuple[str, str]] = set()
    by_levels: dict[tuple, list[str]] = defaultdict(list)
    for key in keys:
        by_levels[tuple(entries[key].get("levels", []))].append(key)
    for group in by_levels.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                pairs.add((a, b))
    if embed_fn is not None and len(keys) > 1:
        import numpy as np

        texts = [" ; ".join(_row_surfaces(k, entries[k])[:8]) for k in keys]
        vectors = np.asarray(embed_fn(texts), dtype=np.float32)
        vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)
        sims = vectors @ vectors.T
        for i, a in enumerate(keys):
            picked = 0
            for j in np.argsort(-sims[i]):
                if j == i:
                    continue
                if sims[i][j] < BLOCK_SIM_FLOOR or picked >= BLOCK_TOP_K:
                    break
                pairs.add(tuple(sorted((a, keys[int(j)]))))
                picked += 1
    return pairs


def _merge_component(members: list[str], entries: dict[str, dict], canonical: str,
                     extra_aliases: list[str]) -> dict[str, Any]:
    base_key = max(members, key=lambda k: (float(entries[k].get("count", 1.0)), k))
    row = copy.deepcopy(entries[base_key])
    aliases, seen = [], {_norm(canonical)}
    for surface in [*[s for m in members for s in _row_surfaces(m, entries[m])], *extra_aliases]:
        key = _norm(surface)
        if key and key not in seen:
            seen.add(key)
            aliases.append(key)
    row["aliases"] = aliases
    row["count"] = max(float(entries[m].get("count", 1.0)) for m in members)
    level_counts: dict[str, float] = {}
    for m in members:
        for level, value in (entries[m].get("level_counts") or {}).items():
            level_counts[level] = max(level_counts.get(level, 0.0), float(value))
    if level_counts:
        row["level_counts"] = level_counts
    source_ids = sorted({s for m in members for s in entries[m].get("source_ids", [])})
    if source_ids:
        row["source_ids"] = source_ids
    return row


def merge_runtime_type(entries: dict[str, dict], *, oracle_index: dict[str, str] | None = None,
                       preferred_name: dict[str, str] | None = None,
                       ontology_synonyms: dict[str, list[str]] | None = None,
                       gate_fn: Callable[[list[str], list[str]], float] | None = None,
                       gate_threshold: float | None = None,
                       embed_fn: Callable | None = None) -> tuple[dict, dict]:
    report: dict[str, Any] = {"merged": [], "review": [], "gate_scored": 0}
    row_id = {k: (row_ontology_id(k, entries[k], oracle_index) if oracle_index else None)
              for k in entries}

    parent = {k: k for k in entries}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in sorted(block_pairs(entries, embed_fn=embed_fn)):
        same_levels = entries[a].get("levels", []) == entries[b].get("levels", [])
        ia, ib = row_id[a], row_id[b]
        if ia and ib and ia == ib:
            if same_levels:
                parent[find(a)] = find(b)
                report["merged"].append({"a": a, "b": b, "why": f"ontology:{ia}"})
            else:
                report["review"].append({"a": a, "b": b, "why": f"ontology:{ia} levels differ"})
            continue
        if ia and ib:               # two different ontology ids -> distinct entities
            continue
        if not same_levels:
            continue                # never merge or review across different ladders
        if gate_fn is not None and gate_threshold is not None:
            score = gate_fn(_row_surfaces(a, entries[a])[:4], _row_surfaces(b, entries[b])[:4])
            report["gate_scored"] += 1
            if score >= gate_threshold:
                parent[find(a)] = find(b)
                report["merged"].append({"a": a, "b": b, "why": f"gate:{score:.3f}"})
                continue
        report["review"].append({"a": a, "b": b, "why": "unlinked identical levels"})

    components: dict[str, list[str]] = defaultdict(list)
    for k in entries:
        components[find(k)].append(k)
    out: dict[str, dict] = {}
    for members in components.values():
        if len(members) == 1:
            out[members[0]] = entries[members[0]]
            continue
        ids = {row_id[m] for m in members if row_id[m]}
        oid = next(iter(ids)) if len(ids) == 1 else None
        canonical = _norm((preferred_name or {}).get(oid, "")) if oid else ""
        if not canonical:
            canonical = max(members, key=lambda k: (float(entries[k].get("count", 1.0)),
                                                    [-ord(c) for c in k]))
        extra = list((ontology_synonyms or {}).get(oid, [])) if oid else []
        out[canonical] = _merge_component(sorted(members), entries, canonical, extra)
    return dict(sorted(out.items())), report


def _duplicate_surface_claims(entries: dict[str, dict]) -> list[dict]:
    claims: dict[str, list[str]] = defaultdict(list)
    for canonical, row in entries.items():
        for surface in _row_surfaces(canonical, row):
            key = _norm(surface)
            if key and canonical not in claims[key]:
                claims[key].append(canonical)
    return [{"surface": s, "rows": sorted(rows)}
            for s, rows in sorted(claims.items()) if len(rows) > 1]


def apply_entity_merge(artifact: dict, *, obo_paths: dict[str, str] | None = None,
                       gate_fn: Callable | None = None, gate_threshold: float | None = None,
                       embed_fn: Callable | None = None) -> dict:
    """Mutates artifact['profiles'] in place (like coherence.normalize_coherence); returns
    the merge report. Ontology oracles apply only to types in obo_paths."""
    obo_paths = DEFAULT_OBO_PATHS if obo_paths is None else obo_paths
    report: dict[str, Any] = {"types": {}, "duplicate_surface_claims": {}}
    for runtime_type, entries in artifact.get("profiles", {}).items():
        kwargs: dict[str, Any] = {"gate_fn": gate_fn, "gate_threshold": gate_threshold,
                                  "embed_fn": embed_fn}
        if runtime_type in obo_paths:
            obo = obo_paths[runtime_type]
            kwargs.update(oracle_index=doid_surface_index(obo),
                          preferred_name=doid_preferred_names(obo),
                          ontology_synonyms=doid_synonyms(obo))
        merged, type_report = merge_runtime_type(entries, **kwargs)
        artifact["profiles"][runtime_type] = merged
        if type_report["merged"] or type_report["review"] or type_report["gate_scored"]:
            report["types"][runtime_type] = type_report
        dupes = _duplicate_surface_claims(merged)
        if dupes:
            report["duplicate_surface_claims"][runtime_type] = dupes
    return report
```

Note on `test_apply_entity_merge_reports_duplicate_surface_claims`: `apply_entity_merge`
runs LOC with no oracle and no gate, so nothing merges and the two rows both claiming one
alias surface land in `duplicate_surface_claims` — a *diagnostic*, not an error (cross-row
homonyms are legitimate for LOC; measured 47 in the live artifact).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_entity_merge.py -v`
Expected: PASS. Fix any mismatch by adjusting the implementation (tests define the contract).

- [ ] **Step 5: Commit**

```bash
git add src/cloak/lattice/producer/entity_merge.py \
        src/cloak/tests/test_lattice_producer_entity_merge.py
git commit -m "feat(lattice-producer): entity merge for synonym canonical rows"
```

---

### Task 4: Producer graph wiring

**Files:**
- Modify: `src/cloak/lattice/producer/graph.py:581-596` (`normalize_coherence_node`)
- Test: `src/cloak/tests/test_lattice_producer_graph.py`

**Interfaces:**
- Consumes: `apply_entity_merge(artifact)` (Task 3).
- Produces: producer runs write `entity_merge_report.json` next to `coherence_report.json`;
  proposed artifacts have synonym rows merged before validation.

- [ ] **Step 1: Write the failing test**

Append to `src/cloak/tests/test_lattice_producer_graph.py`, following that file's existing
`normalize_coherence_node` test pattern (it builds a `state` dict with `proposed_out`,
`profiles_path`, `run_id`, and a tmp artifact file — reuse the same helpers/fixtures):

```python
def test_normalize_coherence_node_applies_entity_merge(tmp_path, monkeypatch):
    proposed = tmp_path / "proposed.json"
    proposed.write_text(json.dumps({
        "schema_version": 1, "created": "2026-07-10", "sources": {}, "profiles": {
            "health-condition": {
                "blorbitis": {"aliases": [], "levels": ["organ disease"],
                              "source_ids": ["t:1"], "count": 10.0}}},
        "artifact_role": "proposal", "proposal_scope": "producer-processed-only"}))
    calls = {}
    def fake_merge(artifact, **kwargs):
        calls["profiles"] = artifact["profiles"]
        return {"types": {"health-condition": {"merged": [], "review": [], "gate_scored": 0}},
                "duplicate_surface_claims": {}}
    monkeypatch.setattr("cloak.lattice.producer.graph.apply_entity_merge", fake_merge)
    state = {"proposed_out": str(proposed), "profiles_path": str(tmp_path / "canon.json"),
             "run_id": "test-run", "run_dir": str(tmp_path)}
    from cloak.lattice.producer.graph import normalize_coherence_node
    normalize_coherence_node(state)
    assert "health-condition" in calls["profiles"]
    assert (tmp_path / "entity_merge_report.json").exists()
```

(`_jsonl_path(state, name)` is `Path(state["run_dir"]) / name` — the `state` dict above
matches its contract.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_graph.py -k entity_merge -v`
Expected: FAIL (`AttributeError: ... no attribute 'apply_entity_merge'` on the graph module).

- [ ] **Step 3: Implement**

In `graph.py`: add `from cloak.lattice.producer.entity_merge import apply_entity_merge` next
to the `normalize_coherence` import, and extend `normalize_coherence_node` after the
coherence report write:

```python
    entity_report = apply_entity_merge(artifact)
    proposed.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    if entity_report.get("types") or entity_report.get("duplicate_surface_claims"):
        _jsonl_path(state, "entity_merge_report.json").write_text(
            json.dumps(entity_report, indent=2, sort_keys=True))
```

(Order: run `normalize_coherence(artifact)` first — level-string synonyms collapse before
entry-level identity is judged — then `apply_entity_merge(artifact)`, then a single final
`proposed.write_text(...)`; drop the earlier intermediate write so the file is written once.)

- [ ] **Step 4: Run the graph test suite**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_graph.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cloak/lattice/producer/graph.py src/cloak/tests/test_lattice_producer_graph.py
git commit -m "feat(lattice-producer): entity merge runs in producer before validation"
```

---

### Task 5: Gate calibration script + reprocess CLI

**Files:**
- Create: `scripts/calibrate_entity_merge_gate.py`
- Create: `scripts/dedupe_lattice_profile_entries.py`
- Test: `src/cloak/tests/test_entity_merge_scripts.py`

**Interfaces:**
- Consumes: Task 2 (`DoidNode.exact_synonyms`, parents), Task 3 (`apply_entity_merge`),
  `cloak.lattice.profile_match.build_embindex` / `_st_model`.
- Produces:
  - `calibrate_entity_merge_gate.py` writes an eval artifact JSON:
    `{"model_id", "template", "sample", "seed", "sweep": [{"threshold", "precision",
    "recall", "predicted_positives"}...], "chosen_threshold": float | null,
    "precision_bar": 0.999, "recall_floor": 0.10}`.
  - `dedupe_lattice_profile_entries.py` rewrites the profile artifact + writes a merge
    report JSON + rebuilds the embindex.
- Importable pure functions (tested without GPU): `build_eval_pairs(nodes, sample, seed)`,
  `choose_threshold(scored_pairs, precision_bar, recall_floor)`, and the CLI `main(argv)`
  of the dedupe script with injectable `gate_fn`/`embed_fn`.

- [ ] **Step 1: Write the failing tests**

Create `src/cloak/tests/test_entity_merge_scripts.py`:

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from calibrate_entity_merge_gate import build_eval_pairs, choose_threshold
from cloak.lattice.producer.reference_sources import DoidNode


def _nodes():
    return {
        "DOID:1": DoidNode("DOID:1", "blorbitis", parents=["DOID:9"],
                           exact_synonyms=["blorb inflammation"]),
        "DOID:2": DoidNode("DOID:2", "glimmerosis", parents=["DOID:9"]),
        "DOID:3": DoidNode("DOID:3", "flurbosis", parents=["DOID:9"]),
        "DOID:8": DoidNode("DOID:8", "old thing", parents=["DOID:9"], obsolete=True),
        "DOID:9": DoidNode("DOID:9", "organ disease"),
    }


def test_build_eval_pairs_positives_are_synonyms_negatives_are_siblings():
    pos, neg = build_eval_pairs(_nodes(), sample=100, seed=0)
    assert ("blorbitis", "blorb inflammation") in pos
    assert all(a != b for a, b in neg)
    sib_names = {frozenset(p) for p in neg}
    assert frozenset(("glimmerosis", "flurbosis")) in sib_names
    assert not any("old thing" in p for p in [*pos, *neg])   # obsolete excluded


def test_choose_threshold_requires_precision_bar_and_recall_floor():
    scored = [(0.9, True), (0.8, True), (0.7, False), (0.6, True), (0.2, False)]
    # at t=0.8: P=1.0, R=2/3 -> chosen; lower t admits the 0.7 negative
    assert choose_threshold(scored, precision_bar=0.999, recall_floor=0.10) == 0.8
    # unreachable bar -> None (gate ships disabled)
    assert choose_threshold([(0.9, False), (0.8, True)], 0.999, 0.10) is None


def test_dedupe_cli_rewrites_artifact_and_report(tmp_path):
    import dedupe_lattice_profile_entries as cli
    artifact = {"schema_version": 1, "created": "2026-07-10", "sources": {}, "profiles": {
        "health-condition": {
            "blorbitis": {"aliases": [], "levels": ["organ disease"],
                          "source_ids": ["t:1"], "count": 10.0},
            "blorb inflammation": {"aliases": [], "levels": ["organ disease"],
                                   "source_ids": ["t:2"], "count": 4.0}}}}
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps(artifact))
    obo = tmp_path / "mini.obo"
    obo.write_text('[Term]\nid: DOID:1\nname: blorbitis\n'
                   'synonym: "blorb inflammation" EXACT []\n')
    report_out = tmp_path / "report.json"
    cli.main(["--profiles", str(profiles), "--obo", f"health-condition={obo}",
              "--report-out", str(report_out), "--skip-embindex", "--no-embed-blocking"])
    got = json.loads(profiles.read_text())
    assert set(got["profiles"]["health-condition"]) == {"blorbitis"}
    report = json.loads(report_out.read_text())
    assert report["types"]["health-condition"]["merged"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_entity_merge_scripts.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'calibrate_entity_merge_gate'`).

- [ ] **Step 3: Implement `scripts/calibrate_entity_merge_gate.py`**

```python
"""Calibrate the entity-merge cross-encoder gate on ontology-derived pairs.

Positives: (name, EXACT synonym) pairs of non-obsolete DOID terms. Hard negatives:
same-parent sibling name pairs (the class that must never merge). Reports a threshold sweep
and the chosen threshold (smallest t with precision >= 0.999 and recall >= 0.10), or null —
then the gate ships disabled and the numbers are the finding (empirical-honesty rule: the
threshold is calibrated once here and frozen; never per-run tuning).

GPU job: run as
  .venv/bin/python -u scripts/calibrate_entity_merge_gate.py \
      --out results/entity_merge_gate_eval.json 2>&1 | tee results/entity_merge_gate_eval.log
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloak.lattice.producer.reference_sources import DEFAULT_DOID_OBO, load_doid_index

PRECISION_BAR = 0.999
RECALL_FLOOR = 0.10
DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"


def build_eval_pairs(nodes: dict, sample: int, seed: int):
    """(positives, negatives): synonym pairs vs same-parent sibling name pairs."""
    rng = random.Random(seed)
    live = {nid: n for nid, n in nodes.items() if not n.obsolete}
    positives = [(n.name, syn) for n in live.values() for syn in n.exact_synonyms]
    by_parent = defaultdict(list)
    for n in live.values():
        for p in n.parents:
            by_parent[p].append(n.name)
    negatives = [(sibs[i], sibs[j]) for sibs in by_parent.values()
                 for i in range(len(sibs)) for j in range(i + 1, len(sibs))]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    return positives[:sample], negatives[:sample]


def choose_threshold(scored: list[tuple[float, bool]], precision_bar: float,
                     recall_floor: float) -> float | None:
    """scored: (score, is_positive). Smallest threshold meeting both bars, else None."""
    total_pos = sum(1 for _, y in scored if y)
    best = None
    for t in sorted({s for s, _ in scored}):
        pred = [(s, y) for s, y in scored if s >= t]
        if not pred:
            continue
        tp = sum(1 for _, y in pred if y)
        precision = tp / len(pred)
        recall = tp / total_pos if total_pos else 0.0
        if precision >= precision_bar and recall >= recall_floor:
            best = t
            break
    return best


def nli_gate_scorer(model_id: str):
    """Same-entity score for a surface pair: min of both-direction entailment probs."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    entail_idx = next(i for i, name in model.config.id2label.items()
                      if name.lower().startswith("entail"))

    def score_pairs(pairs: list[tuple[str, str]], batch_size: int = 64) -> list[float]:
        out = []
        both = [(a, b) for a, b in pairs] + [(b, a) for a, b in pairs]
        probs = []
        with torch.no_grad():
            for i in range(0, len(both), batch_size):
                chunk = both[i:i + batch_size]
                enc = tok([a for a, _ in chunk], [b for _, b in chunk], padding=True,
                          truncation=True, return_tensors="pt").to(model.device)
                p = torch.softmax(model(**enc).logits, dim=-1)[:, entail_idx]
                probs.extend(p.tolist())
        n = len(pairs)
        for i in range(n):
            out.append(min(probs[i], probs[n + i]))
        return out

    return score_pairs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obo", default=str(DEFAULT_DOID_OBO))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/entity_merge_gate_eval.json")
    args = ap.parse_args(argv)

    nodes = load_doid_index(args.obo)
    positives, negatives = build_eval_pairs(nodes, args.sample, args.seed)
    print(f"eval pairs: {len(positives)} positives, {len(negatives)} negatives", flush=True)

    scorer = nli_gate_scorer(args.model)
    scored = list(zip(scorer(positives), [True] * len(positives))) + \
             list(zip(scorer(negatives), [False] * len(negatives)))

    total_pos = len(positives)
    sweep = []
    for t in [round(0.05 * i, 2) for i in range(1, 20)]:
        pred = [(s, y) for s, y in scored if s >= t]
        tp = sum(1 for _, y in pred if y)
        sweep.append({"threshold": t,
                      "precision": round(tp / len(pred), 4) if pred else None,
                      "recall": round(tp / total_pos, 4) if total_pos else 0.0,
                      "predicted_positives": len(pred)})
    chosen = choose_threshold(scored, PRECISION_BAR, RECALL_FLOOR)
    out = {"model_id": args.model, "obo": args.obo, "sample": args.sample, "seed": args.seed,
           "positives": len(positives), "negatives": len(negatives),
           "precision_bar": PRECISION_BAR, "recall_floor": RECALL_FLOOR,
           "sweep": sweep, "chosen_threshold": chosen}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"chosen_threshold={chosen} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement `scripts/dedupe_lattice_profile_entries.py`**

```python
"""Reprocess an existing lattice_profiles.json: merge synonym rows (entity merge), validate,
rewrite atomically, rebuild the embedding index.

Spec: docs/specs/lattice-entry-dedup-and-span-resolution.md (Part 3, reprocess CLI). The gate
is enabled only when --gate-eval points to a calibration artifact whose chosen_threshold is
non-null (scripts/calibrate_entity_merge_gate.py). Merged-pair listings go to the JSON report,
not stdout.

  .venv/bin/python scripts/dedupe_lattice_profile_entries.py \
      --gate-eval results/entity_merge_gate_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloak.lattice.producer.entity_merge import DEFAULT_OBO_PATHS, apply_entity_merge
from cloak.lattice.profiles import DEFAULT_PROFILE_PATH, validate_profile_artifact


def _atomic_write(path: Path, artifact: dict) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", default=str(DEFAULT_PROFILE_PATH))
    ap.add_argument("--gate-eval", default=None,
                    help="calibration JSON; gate enabled iff chosen_threshold is non-null")
    ap.add_argument("--obo", action="append", default=[],
                    help="runtime_type=obo_path override (default: health-condition=DOID)")
    ap.add_argument("--report-out", default="results/entity_merge_report.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-embindex", action="store_true")
    ap.add_argument("--no-embed-blocking", action="store_true",
                    help="block on identical levels only (no embedding model)")
    args = ap.parse_args(argv)

    profiles_path = Path(args.profiles)
    artifact = json.loads(profiles_path.read_text())

    obo_paths = dict(DEFAULT_OBO_PATHS)
    for spec in args.obo:
        runtime_type, _, obo = spec.partition("=")
        obo_paths[runtime_type] = obo

    gate_fn = gate_threshold = None
    if args.gate_eval:
        eval_art = json.loads(Path(args.gate_eval).read_text())
        gate_threshold = eval_art.get("chosen_threshold")
        if gate_threshold is not None:
            from calibrate_entity_merge_gate import nli_gate_scorer

            score_pairs = nli_gate_scorer(eval_art["model_id"])
            gate_fn = lambda sa, sb: max(score_pairs([(a, b) for a in sa for b in sb]))
            print(f"gate enabled: {eval_art['model_id']} @ {gate_threshold}", flush=True)
        else:
            print("gate eval has chosen_threshold=null -> gate disabled (review-only)",
                  flush=True)

    embed_fn = None
    if not args.no_embed_blocking:
        from cloak.lattice.profile_match import DEFAULT_MODEL_ID, _st_model

        model = _st_model(DEFAULT_MODEL_ID)
        embed_fn = lambda texts: model.encode(texts, normalize_embeddings=True)

    before = {rt: len(entries) for rt, entries in artifact.get("profiles", {}).items()}
    report = apply_entity_merge(artifact, obo_paths=obo_paths, gate_fn=gate_fn,
                                gate_threshold=gate_threshold, embed_fn=embed_fn)
    after = {rt: len(entries) for rt, entries in artifact.get("profiles", {}).items()}
    report["entry_counts"] = {rt: {"before": before[rt], "after": after[rt]}
                              for rt in before}

    errors = validate_profile_artifact(artifact)
    if errors:
        sys.exit("post-merge validation failed:\n" + "\n".join(errors[:20]))

    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, indent=2, sort_keys=True))
    merged_n = sum(len(t["merged"]) for t in report.get("types", {}).values())
    review_n = sum(len(t["review"]) for t in report.get("types", {}).values())
    print(f"merged pairs: {merged_n}; review pairs: {review_n}; "
          f"entry counts: {json.dumps(report['entry_counts'])}", flush=True)
    if args.dry_run:
        print("dry-run: artifact NOT written", flush=True)
        return
    _atomic_write(profiles_path, artifact)
    if not args.skip_embindex:
        from cloak.lattice.profile_match import build_embindex

        out = build_embindex(profiles_path)
        print(f"embindex rebuilt: {out}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_entity_merge_scripts.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/calibrate_entity_merge_gate.py scripts/dedupe_lattice_profile_entries.py \
        src/cloak/tests/test_entity_merge_scripts.py
git commit -m "feat(lattice): gate calibration + profile dedup reprocess CLI"
```

---

### Task 6: QA-build span dedup by entry

**Files:**
- Modify: `scripts/build_probes.py:358-390` (`_detect_docs`)
- Test: `src/cloak/tests/test_build_probes_detect.py` (new)

**Interfaces:**
- Consumes: `match_spans_batch`, `span_key` (Task 1's entry-carrying exact hits);
  `MatchResult.entry`.
- Produces: `_detect_docs` output spans gain an `"entry"` key on lattice-role spans; one
  lattice span per `(runtime_type, entry)` per document.

- [ ] **Step 1: Write the failing test**

Create `src/cloak/tests/test_build_probes_detect.py`. GLiNER and the matcher are both
monkeypatched — the test exercises only the dedup/resolution logic:

```python
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))


def test_detect_docs_dedups_lattice_spans_by_entry(monkeypatch):
    import build_probes
    from cloak.lattice.profile_match import MatchResult

    class FakeGliner:
        def predict_entities(self, piece, labels, threshold):
            return [
                {"text": "blorbitis", "label": "condition"},
                {"text": "blorb inflammation", "label": "condition"},
                {"text": "glimmerosis", "label": "condition"},
                {"text": "ghostitis", "label": "condition"},      # matcher abstains
                {"text": "Ann", "label": "name"},
                {"text": "Ann", "label": "name"},                  # surface-dup placeholder
            ]

    monkeypatch.setattr(build_probes, "GLINER_LOADER",
                        lambda model: FakeGliner(), raising=False)
    # if _detect_docs imports GLiNER inline, monkeypatch the gliner module instead:
    fake_mod = types.SimpleNamespace(GLiNER=types.SimpleNamespace(
        from_pretrained=lambda model: FakeGliner()))
    monkeypatch.setitem(sys.modules, "gliner", fake_mod)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)))

    def fake_match(items, **kwargs):
        entry_of = {"blorbitis": "blorbitis", "blorb inflammation": "blorbitis",
                    "glimmerosis": "glimmerosis"}
        out = {}
        for surface, rtype, ctx in items:
            from cloak.lattice.profile_match import span_key
            e = entry_of.get(surface.lower())
            out[span_key(surface, rtype)] = (
                MatchResult(["organ disease"], "exact", True, 1.0, e) if e else None)
        return out

    monkeypatch.setattr(build_probes, "match_spans_batch", fake_match, raising=False)

    docs = [{"id": "d1", "text": "Ann has blorbitis, blorb inflammation, glimmerosis, "
                                 "ghostitis. Ann rests."}]
    got = build_probes._detect_docs(docs, "any-model", 0.3)
    spans = got["d1"]
    lattice = [s for s in spans if s["role"] == "lattice"]
    # blorbitis + blorb inflammation collapse to one span; ghostitis dropped (abstain)
    assert [s["surface"] for s in lattice] == ["blorbitis", "glimmerosis"]
    assert lattice[0]["entry"] == "blorbitis"
    # placeholder spans still deduped by surface, kept without entry
    names = [s for s in spans if s["type"] == "PERSON"]
    assert len(names) == 1 and "entry" not in names[0]
```

(If monkeypatching `build_probes.match_spans_batch` fails because the import is inside the
function, move the import to module level in Step 3 — module-level import is the fix, not a
different patch target.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_build_probes_detect.py -v`
Expected: FAIL (current `_detect_docs` keeps both co-referent surfaces / has no `entry` key).

- [ ] **Step 3: Implement**

Rewrite `_detect_docs` in `scripts/build_probes.py` (imports `match_spans_batch`, `span_key`
at module level next to the other cloak imports):

```python
def _detect_docs(docs, model, threshold, max_words=320):
    """Fresh zero-shot detection -> {doc_id: [{surface, type, role, sent[, entry]}]}.
    `lattice` spans resolve through the shared retrieve-then-verify matcher
    (cloak.lattice.profile_match.match_spans_batch — same machinery as the substitutor) and are
    deduped per (runtime_type, matched canonical entry), so co-referent surfaces collapse to
    one span per document; matcher abstain drops the span (as a no-profile span is dropped
    today). `placeholder`/`quasi` spans are deduped per (surface, type). GPU (GLiNER)."""
    import torch
    from gliner import GLiNER

    from ladder_probes import sentence_of   # train/ladder_probes.py, retired 2026-07-27

    g = GLiNER.from_pretrained(model)
    if torch.cuda.is_available():
        g = g.to("cuda")
    labels = list(DETECT_LABELS)
    out = {}
    for d in docs:
        words = d["text"].split()
        seen, cands = set(), []
        for i in range(0, len(words), max_words):
            piece = " ".join(words[i:i + max_words])
            for e in g.predict_entities(piece, labels, threshold=threshold):
                surface, (rtype, role) = e["text"].strip(), DETECT_LABELS[e["label"]]
                key = (surface.lower(), rtype)
                if not surface or key in seen:
                    continue
                seen.add(key)
                cands.append({"surface": surface, "type": rtype, "role": role,
                              "sent": sentence_of(d["text"], surface)})
        lattice_cands = [c for c in cands if c["role"] == "lattice"]
        matches = match_spans_batch(
            [(c["surface"], c["type"], c["sent"]) for c in lattice_cands])
        spans, seen_entries = [], set()
        for c in cands:
            if c["role"] != "lattice":
                spans.append(c)
                continue
            m = matches.get(span_key(c["surface"], c["type"]))
            if m is None:
                continue  # abstain: not a probe span, drop (current no-profile behavior)
            entry_key = (c["type"], m.entry)
            if entry_key in seen_entries:
                continue  # co-referent duplicate within this document
            seen_entries.add(entry_key)
            spans.append({**c, "entry": m.entry})
        out[d["id"]] = spans
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_build_probes_detect.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_probes.py src/cloak/tests/test_build_probes_detect.py
git commit -m "feat(qa-build): dedup lattice spans by canonical entry via shared matcher"
```

---

### Task 7: Execution — calibrate, reprocess the live artifact, verify

This task runs the built tooling on real data. **GPU jobs — one at a time; check
`rocm-smi --showpidgpus` for a live GPU process first.**

- [ ] **Step 1: Full test suite**

Run: `.venv/bin/python -m pytest src/cloak/tests/ -x -q`
Expected: PASS (no regressions from Tasks 1–6).

- [ ] **Step 2: Gate calibration (GPU, ~minutes)**

```bash
.venv/bin/python -u scripts/calibrate_entity_merge_gate.py \
    --out results/entity_merge_gate_eval.json 2>&1 | tee results/entity_merge_gate_eval.log
```
Expected: prints pair counts and `chosen_threshold=<float or None>`. Either outcome is valid
— `None` means the gate ships disabled and the sweep numbers are the finding.

- [ ] **Step 3: Dry-run reprocess, inspect counts**

```bash
.venv/bin/python -u scripts/dedupe_lattice_profile_entries.py \
    --gate-eval results/entity_merge_gate_eval.json --dry-run
```
Expected: merged/review pair counts and per-type before/after entry counts on stdout (no
term listings). Sanity: `health-condition` merges ≥ 1 pair (the two known synonym rows);
no type loses a large fraction of entries (>20% would be a red flag — stop and inspect
`results/entity_merge_report.json`).

- [ ] **Step 4: Real reprocess + embindex rebuild**

```bash
.venv/bin/python -u scripts/dedupe_lattice_profile_entries.py \
    --gate-eval results/entity_merge_gate_eval.json
.venv/bin/python -m pytest src/cloak/tests/test_lattice_profiles.py src/cloak/tests/test_profile_match.py -q
```
Expected: artifact rewritten + `lattice_profiles.embindex.npz` rebuilt; tests still pass.

- [ ] **Step 5: Commit data artifacts**

```bash
git add data/lattice_profiles/lattice_profiles.json \
        data/lattice_profiles/lattice_profiles.embindex.npz \
        results/entity_merge_gate_eval.json results/entity_merge_report.json
git commit -m "data(lattice): entity-merge reprocess of lattice_profiles + gate calibration"
```
(If the parallel session has uncommitted changes to `lattice_profiles.json`, coordinate
before this step — do not clobber.)

- [ ] **Step 6: STOP — end-to-end revalidation needs user OK**

The 1-doc QA-build rerun (`scripts/build_probes.py --detect` on the clinical validation doc)
consumes paid teacher calls. Ask the user for explicit approval with the expected call count
before running; compare duplicate-span count and `reader_rung_reject_rate` against
`results/ladder_build_detect_1doc.log`.
