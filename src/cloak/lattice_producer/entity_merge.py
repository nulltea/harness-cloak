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

from cloak.lattice_producer.reference_sources import DEFAULT_DOID_OBO, load_doid_index

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
    component_ids: dict[str, set[str]] = {k: ({row_id[k]} if row_id[k] else set())
                                          for k in entries}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> bool:
        """Union a's and b's components unless that would put two distinct ontology ids in
        one component (an unlinked bridge row must never fuse ontology-distinct entities)."""
        ra, rb = find(a), find(b)
        if ra == rb:
            return True
        merged_ids = component_ids[ra] | component_ids[rb]
        if len(merged_ids) > 1:
            return False
        parent[ra] = rb
        component_ids[rb] = merged_ids
        return True

    pairs = block_pairs(entries, embed_fn=embed_fn)
    by_id: dict[str, list[str]] = defaultdict(list)
    for key, oid in row_id.items():
        if oid:
            by_id[oid].append(key)
    for group in by_id.values():
        for i, a in enumerate(sorted(group)):
            for b in sorted(group)[i + 1:]:
                pairs.add((a, b))

    for a, b in sorted(pairs):
        same_levels = entries[a].get("levels", []) == entries[b].get("levels", [])
        ia, ib = row_id[a], row_id[b]
        if ia and ib and ia == ib:
            if same_levels:
                union(a, b)   # same id on both sides: can never bridge distinct ids
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
                if union(a, b):
                    report["merged"].append({"a": a, "b": b, "why": f"gate:{score:.3f}"})
                else:
                    report["review"].append(
                        {"a": a, "b": b, "why": "would bridge distinct ontology ids"})
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
