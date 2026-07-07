---
type: reference
status: current
created: 2026-07-07
updated: 2026-07-07
tags: [privacy, lattice, anonymity-set, k-anonymity, count-floors, offline-risk]
companion: [docs/specs/generalization-lattice-cache.md, docs/specs/lattice-substitutor.md, docs/research/inference-risk-enforcement.md]
---

# Offline k-anonymity risk walk for generalization lattices

## Purpose

This spec defines how the offline lattice builder assigns deterministic anonymity-set counts to
generalization lattice levels. These counts are the only model-free privacy certificate used by
`aset >= K_FLOORS[type]` legality masks in ranker environments and future runtime policies.

The strategy is deliberately stricter than the current cache artifact: **risk counts attach to levels, not
to source rows**. A row-level source statistic such as city population, dataset frequency, or the default
`1000.0` is provenance only until it is converted into a level-level anonymity set with the rules below.

## Definitions

- **Original-granularity value** - the finest canonical value the detector span represents inside its
  runtime type: a city for a city span, a country/demonym for a nationality span, a job title or occupation
  concept for a profession span, a denomination for a religion span, and so on.
- **Type universe** - the offline set of original-granularity values considered for one runtime type under
  one source family and one normalization policy.
- **Level** - a grammatical replacement phrase in a lattice, such as `a city in norway`,
  `professional worker`, or `christian religion`.
- **Level count** - the number of original-granularity values in the same type universe that are truthfully
  consistent with that level.
- **Certifying count** - a level count that may be compared to `K_FLOORS[type]`.
- **Fail-closed count** - `1.0`, used whenever the builder cannot prove a certifying count.
- **Generic terminal** - a type-label-like phrase such as `a place` or `an organization`. It is a fallback
  text surface, not evidence that a source universe has been counted.

## Core Rule

For every emitted `(runtime_type, level)` pair:

```text
certifying_count(runtime_type, level)
  = |{v in type_universe(runtime_type) : level is truthful for v under the same lattice policy}|
```

The count is computed once offline and stored with the lattice artifact. Runtime code only performs a lookup
and a floor comparison. It must not infer counts from row defaults, source popularity fields, lexical shape,
WordNet fallback behavior, or a model prediction.

## Design Decision

I considered three strategies:

1. **Keep row-level counts** - cheap, but wrong: the same row count gets reused for both specific and broad
   levels, so `computer and mathematical occupation` and `professional worker` can receive the same count.
2. **Use per-level structural counts** - deterministic, auditable, and aligned with k-anonymity semantics.
3. **Use offline LM-calibrated risk scores** - useful as diagnostics, but not a certifying mask because the
   guarantee becomes model- and prompt-dependent.

Use strategy 2. Strategy 3 remains an offline validation/referee signal. Strategy 1 is not acceptable for
privacy certification.

## Offline Risk Walk

The builder walks each runtime type independently:

1. **Build the type universe.** Normalize canonical values and aliases, deduplicate source concepts, and
   record the source IDs that justify each value.
2. **Generate truthful levels.** For each value, emit ordered levels using the same deterministic lattice
   policy that runtime lookup will use.
3. **Invert levels to members.** Build `(runtime_type, level) -> set[canonical_value_id]`.
4. **Assign level counts.** Count set cardinality for each level. Store that as the certifying count.
5. **Check monotonicity per row.** Along one row's lattice, counts must be non-decreasing from specific to
   broad levels. A violation is a build failure unless the narrower/broader order is corrected.
6. **Fail closed on unsupported levels.** Any level without a source-backed member set gets `1.0`, not a
   heuristic estimate.
7. **Calibrate floors separately.** `K_FLOORS` are derived from attacker measurements over the emitted
   counts; they are not baked into the count assignment itself.

## Required Artifact Shape

The next cache schema should store counts by level. A compatible shape is:

```json
{
  "profiles": {
    "profession": {
      "software developer": {
        "aliases": ["application developer"],
        "levels": ["computer and mathematical occupation", "professional worker"],
        "level_counts": {
          "computer and mathematical occupation": 312,
          "professional worker": 5430
        },
        "source_ids": ["onet-job-title:15-1252.00"]
      }
    }
  }
}
```

`count` may remain temporarily for backward compatibility, but it must not be used as a certifying count for
fine-type levels unless it is equal to the selected level's `level_counts[level]`.

## Type-Specific Count Semantics

### LOC

LOC counts are structural place counts, not population.

- For `a city in <region|country|continent>`, count unique city values in the GeoNames city universe that
  belong to the named region, country, or continent.
- For `a country in <continent>`, count unique country values in the GeoNames country universe that belong
  to the named continent.
- Duplicate city surfaces resolve to the same deterministic representative used by the lattice builder, but
  counts should use canonical GeoNames value IDs, not surface strings, when possible.
- Population may be stored as `prior_weight` for diagnostics or future population-weighted calibration. It is
  not the default k-anonymity count.

### nationality

Nationality counts are counts of country/citizenship values in the nationality universe.

- A demonym alias and its country name share one canonical value.
- Regional levels such as `western european nationality` count countries mapped to that region.
- Continent levels such as `european nationality` count countries mapped to that continent.
- Population-weighted nationality risk is a separate optional metric and must not silently replace the
  structural country count.

### profession

Profession counts are counts of occupation concepts or canonical job-title values under a profession level.

- O*NET/SOC, ESCO/ISCO, and manual profession maps must first normalize to canonical value IDs.
- `computer and mathematical occupation` counts values assigned to that SOC/ISCO family.
- `professional worker` counts all profession values under the professional-worker umbrella.
- Aliases do not add count. Multiple aliases for one occupation concept count once.
- Long or filtered-out surfaces do not contribute to count unless they remain as canonical values in the
  selected type universe. The filter policy must be recorded with the artifact.

### religion

Religion counts are counts of denomination/tradition values in the religion universe.

- `christian religion` counts denominations or religion rows mapped to Christianity.
- `religious tradition` counts all supported religion rows that map to that broad level.
- Numeric ARDA codes, Wikidata IDs, and aliases deduplicate to canonical value IDs before counting.
- A level with only a hand-written label and no member set fails closed.

### health-condition

Health-condition counts are counts of disease/condition concepts under ontology families.

- OBO/MONDO/DOID IDs define canonical concept values.
- A family level such as `endocrine condition` counts descendant condition concepts mapped to that family.
- A broader level such as `chronic condition` must count a defined descendant set. If the source only asserts
  a label without a closed member set, it fails closed.

### age, DATETIME, and QUANTITY

Rule-derived numeric/time counts are computed from interval width at the original's granularity.

- `thirty-something` for age counts integer ages 30 through 39: `10`.
- A date window counts `window_size / original_granularity`, floored at `1.0`.
- A quantity range counts `(hi - lo) / original_step + 1`, floored at `1.0`.
- Unparseable numeric/time levels fail closed in certifying mode.

### ORG, MISC, OTHER, and open-vocabulary types

Open-vocabulary types require a real source universe before text levels can certify privacy.

- Legacy teacher-cache ORG rows do not define an organization universe. Their levels get `1.0` unless a
  separate organization taxonomy/source assigns a member set.
- WordNet leaf counts are diagnostics unless the build records the exact synset, sense-selection rule, and
  descendant member set used for certification.
- Generic terminals such as `an organization`, `something`, or `a personal attribute` are not substitute
  evidence for a counted universe. Prefer the typed placeholder when no counted level is available.

### Placeholder-only categorical types

`gender`, `marital-status`, and `sexual-orientation` have no text levels by default.

- Keep-original has count `1.0`.
- Placeholder terminal is always legal and carries no re-identifying surface.
- If a future user-defined lattice adds text levels, it must also define an explicit type universe and
  per-level counts before those levels can be legal above `1.0`.

## Invariants

- Counts are per runtime type; counts from different runtime types are never compared.
- Counts are per level; row-level defaults cannot certify levels.
- Counts are deterministic for a fixed source snapshot and normalization policy.
- Counts are monotone non-decreasing along every emitted lattice row.
- Count `1.0` means exact keep, unparseable, unsupported, or otherwise uncertified.
- Placeholder terminals are legal by construction and do not need anonymity-set counts.
- Attacker-evaluation remains the only basis for privacy claims. Counts define admissible actions, not final
  realized privacy.

## Validation

Every cache build that changes sources, normalization, lattice generation, or count assignment must emit:

- schema validation errors;
- count monotonicity violations;
- number of levels with fail-closed counts by runtime type;
- distribution of level counts by runtime type;
- examples where a level appears under multiple source families;
- a count-vs-attacker shootout before updating `K_FLOORS`.

No benchmark or training result should be compared across cache versions unless the count schema, source
snapshot, and floor calibration are pinned in the run artifact.
