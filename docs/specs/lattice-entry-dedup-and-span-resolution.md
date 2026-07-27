---
type: reference
status: current
created: 2026-07-10
updated: 2026-07-27
tags: [lattice, lattice-profiles, deduplication, aliases, entity-resolution, qa-build, profile-matching]
companion: [docs/issues/2026-07-10-near-duplicate-condition-lattice-entries.md,
            docs/specs/generalization-lattice-cache.md,
            docs/specs/substitutor-profile-match-retrieve-verify.md]
---

# Lattice entry dedup + span-to-entry resolution

> **Post-refactor note (2026-07-27):** module paths in this doc were updated to the
> regrouped `src/cloak` layout — see the path mapping in [the cleanup plan](../plans/2026-07-27-codebase-cleanup-refactor.md).
> Still named here but **deleted, not moved** in that cleanup: `scripts/build_probes.py` (superseded by `scripts/build_arms_artifact.py` + `scripts/build_qa_utility_artifact.py`).

## Purpose

Fixes the near-duplicate condition-entry issue
([issue doc](../issues/2026-07-10-near-duplicate-condition-lattice-entries.md)): co-referent
surfaces in one document become multiple lattice spans (redundant paid teacher calls, split RL
credit), and a minority of true synonym conditions exist as separate canonical rows with identical
levels. Two complementary fixes:

1. **Span resolution (build-side):** detected spans resolve to a canonical profile entry through
   the *existing* retrieve-then-verify matcher, and the QA build dedups spans per document by
   entry identity — one span per underlying entity.
2. **Entry dedup (profile-side):** true synonym rows merge into one canonical row + union alias
   set, decided by an ontology oracle with a measured cross-encoder gate for pairs the ontology
   leaves unlinked.

## Definitions

- **Canonical entry / canonical key** — one row of `lattice_profiles.json` under a runtime type;
  the key string identifies the entity.
- **Alias** — an alternative surface listed on a row; the loader's surface index resolves aliases
  to the row's levels.
- **Entry identity** — the canonical key a surface resolves to; two surfaces with the same entry
  identity denote the same entity.
- **Retrieve-then-verify matcher** — the implemented profile matcher
  (`src/cloak/lattice/profile_match.py`): exact `_norm` lookup fast path, then embedding retrieval
  proposing an entry, certified by the NLI gate in the span's sentence
  ([spec](substitutor-profile-match-retrieve-verify.md)).
- **Ontology oracle** — a curated external source asserting entity identity via preferred names +
  exact-synonym edges (for health conditions: the vendored
  `data/lattice_sources/raw/health/doid.obo`).
- **Blocking** — candidate-pair generation in entity resolution: only blocked pairs are ever
  scored/decided, keeping the decision step off the full O(n²) pair space.
- **Cross-encoder gate** — a pairwise same-entity scorer applied to blocked pairs the ontology
  leaves unlinked; auto-merges only above a threshold calibrated for near-perfect precision on an
  ontology-derived eval set.

## Part 1 — canonical identity in the shared resolver

Root cause shared by every consumer: the loader's surface index
(`src/cloak/lattice/profiles.py::_build_indexes`) maps surface → levels and **discards the
canonical key**, so even the matcher's exact hits return `entry=None`
(`src/cloak/lattice/profile_match.py`, exact-hit `MatchResult`). No current path exposes entry identity
for an exact match.

Changes:

- `_build_indexes`: the per-type surface index stores `(canonical, levels)` per normalized
  surface (first-writer-wins unchanged).
- New `lookup_entry(surface, runtime_type, path=None) -> tuple[str, list[str]] | None` returning
  `(canonical, levels)`; `lookup_levels` becomes a thin wrapper over it.
- `profile_match.py` exact hits populate `MatchResult.entry` with the canonical key. Semantic
  hits already do. `MatchResult.entry` is therefore always set on a hit; the `R` diagnostic
  `match` block gains the entry on exact hits for free.

No behavior change for existing callers of `lookup_levels`.

## Part 2 — QA-build span dedup through the existing matcher

**No parallel machinery**: the QA build resolves spans with the same `match_spans_batch` the
substitutor uses (hard requirement).

Changes in `scripts/build_probes.py::_detect_docs`:

- For lattice-role spans, replace the bare `lookup_levels` filter with `match_spans_batch` over
  `(surface, runtime_type, sent)` — the sentence is already computed and is the NLI certifier's
  input. Matcher abstain (`None`) = the span is dropped, exactly as a no-levels lattice span is
  dropped today (it does not reach the floor either — preserving current behavior; widening
  floor coverage is out of scope), while adding certified semantic recovery of variant surfaces
  missing from alias lists.
- Dedup key for lattice-role spans changes from `(surface, runtime_type)` to
  `(runtime_type, MatchResult.entry)`. The **first-occurring surface** stays the representative
  span — the reader lookup must use the surface the note actually states.
- Placeholder/quasi spans keep the current `(surface, runtime_type)` dedup — they have no entry
  identity and are only floor-hidden.

Effect on the measured symptom: co-referent alias variants in one document collapse to one span →
one teacher call → no redundant ceiling-rejects; the RL reward's per-span credit no longer splits
across duplicates of the same fact.

## Part 3 — profile-side entity merge

New module `src/cloak/lattice/producer/entity_merge.py` — the entry-level analog of
`coherence.py` (which merges synonym *level strings*). Two invocation paths:

- **Producer path:** applied to accepted rows before persistence in the producer merge flow, so
  future runs cannot reintroduce synonym rows.
- **Reprocess CLI:** durable `scripts/dedupe_lattice_profile_entries.py` — load an existing
  `lattice_profiles.json`, run the merge, validate, write the artifact plus a JSON report of
  merged pairs and review-file candidates (term-level listings stay in JSON, not stdout).

Pipeline, per runtime type:

1. **Block.** Candidate pairs = rows with byte-identical `levels` ∪ embedding top-k nearest
   neighbors over canonical+alias surfaces (same embedding model/infra as the profile embindex).
2. **Decide.**
   - *Ontology-linked:* map each row (canonical + aliases) to an ontology id via a
     name/exact-synonym index over the vendored obo. Two rows on the same id **auto-merge** —
     provided their `levels` are identical; linked pairs with differing levels go to the review
     file, never auto-merge.
   - *Cross-encoder gate (unlinked pairs):* auto-merge only if `levels` are byte-identical AND
     the same-entity score clears the calibrated threshold, AND the runtime type is covered by
     the gate's calibration eval (health-condition for the DOID-derived eval) — the precision
     guarantee does not transfer to type distributions it was never measured on (first live
     dry-run: a disease-calibrated threshold gate-merged 27.6% of profession rows). Pairs below
     threshold, outside gate scope, or any pair if the gate fails calibration, land in the
     review file.
   - The blocking + gate machinery is type-agnostic; only the health-condition ontology adapter
     ships now. Types without an oracle run blocking + gate (or review-only if the gate is
     disabled for that type).
3. **Merge semantics.**
   - Canonical key = the ontology preferred label when linked (this also repairs
     acronym-fragment canonical keys), normalized to the artifact's lowercase key convention;
     otherwise the higher-`count` row's key (tie: lexicographically smaller key, for
     determinism).
   - `aliases` = union of both rows' canonicals + aliases (minus the new canonical), enriched
     with the ontology's exact synonyms for the linked id — widening the exact fast path is the
     cheap, deterministic part of resolution coverage.
   - `count` = max; `level_counts` = per-level max (validator monotonicity re-checked);
     `source_ids` = union.
4. **Duplicate-claim diagnostic.** The merge report lists every normalized surface (canonical
   or alias) claimed by more than one row within a type. This is a report diagnostic, not a
   validator error: cross-row homonyms are legitimate (measured 2026-07-10: 47 duplicate claims
   in the live artifact, all in location/profession/religion — e.g. one place name naming two
   distinct locations), so a hard uniqueness check would reject correct data. Today a duplicate
   claim silently resolves first-wins in the loader index; the diagnostic makes it visible.

### Cross-encoder calibration (the "test how well it works" gate)

Before the gate may auto-merge anything:

- **Eval set (ontology-derived, no manual labels):** positives = exact-synonym surface pairs;
  hard negatives = same-parent sibling pairs (the exact class that must never merge).
- Report precision/recall over thresholds as a JSON artifact; pick the threshold at ~perfect
  precision (a false merge corrupts anonymity accounting; a miss costs one redundant teacher
  call — the asymmetry dictates precision-first).
- Model choice is an experiment parameter recorded in the eval artifact, not hardcoded here.
- **Empirical honesty:** if no threshold achieves the precision bar at useful recall, the gate
  ships disabled (review-file only) and the numbers are reported as the finding. Threshold is
  calibrated once on this eval set and frozen; never tuned per run.

## Testing & verification

- Unit: `lookup_entry` (canonical + alias resolution); exact-hit `MatchResult.entry`;
  `_detect_docs` dedup by entry with an injected matcher; merge semantics + ontology adapter on
  a tiny synthetic obo fixture; cross-encoder gate behind an injectable scorer.
- Calibration artifact: the precision/recall JSON from the gate eval, produced before any
  auto-merge run.
- End-to-end: rerun the 1-doc clinical validation
  (`results/ladder_build_detect_1doc.log` baseline) — co-referent condition spans must collapse
  to one lattice span / one teacher call, with the reject-rate delta reported.
- GPU jobs (embindex, gate eval, detection rerun) follow the one-GPU-process rule and the perf
  gate.

## Non-goals

- No runtime cross-document dedup and no per-occurrence re-certification (documented
  per-surface granularity of the matcher is unchanged).
- No learned canonicalizer adoption (its own graduation criterion is not met).
- No oracle adapters beyond health conditions in this change.
- No merging on identical levels alone, ever — sibling entities legitimately share ladders.
