---
type: dev-log
status: current
created: 2026-07-07
updated: 2026-07-07
tags: [lattice-cache, substitution, profession, geonames, esco, onet, cldr]
companion: ../../specs/generalization-lattice-cache.md
---

# Lattice cache population, cleanup, and runtime wiring

Built the dataset-backed generalization lattice cache and verified that the runtime substitutor
uses it through the normal `substitute()` path. The active artifact is
`data/lattice_profiles/fine_lattice_profiles.json`, built offline from already-downloaded local
sources under `data/lattice_sources/raw`, optional local GeoNames files, and the conservative
legacy teacher-cache import.

## What changed

- Added the standard lattice profile cache path and schema in `cloak.lattice_profiles`, with
  inline `levels` for reviewability and direct lookup speed.
- Added offline source parsers/build orchestration for CLDR/Wikidata nationalities, ARDA/Wikidata
  religions, GeoNames locations, O*NET/ESCO/ISCO professions, manual categorical aliases, OBO
  health families when present, and conservative legacy ORG rows.
- Wired `lattice_for()` to consult the profile cache before old fallbacks for fine types, LOC,
  and ORG. `substitute()` already calls `lattice_for()` for all non-direct spans, so runtime
  substitution now uses the new cache without an additional substitutor-specific code path.
- Rebuilt `data/lattice_profiles/fine_lattice_profiles.json`; after the profession surface cap,
  the artifact has 220,892 total profiles and 21,443 profession profiles.
- Added a substitutor-level regression test proving an alias lookup (`application developer`
  through canonical `software developer`) reaches cache-backed profession levels and records them
  in the substitution record.
- Fixed final output article agreement for lattice replacements, e.g. `an application developer`
  no longer becomes `an computer and mathematical occupation`.

## Design choices

- **Inline profile entries instead of source indirection.** The cache stores reviewable surface
  entries with `aliases`, `levels`, `source_ids`, and `count`. This keeps runtime lookup simple
  and lets a human inspect or edit a lattice row without chasing dataset identifiers.
- **Cache-first, fallback-second.** Runtime looks in the dataset-backed cache before GeoNames,
  curated fine maps, WordNet, or legacy teacher cache. This preserves deterministic offline
  substitution while keeping old behavior for uncovered spans.
- **No hidden DEM expansion.** DEM remains a legacy/research-eval rollup; fine runtime types use
  their own cache entries or typed placeholder terminals.
- **Profession rows are deliberately conservative.** Singleton `worker` rows are dropped, rows
  whose only levels self-leak are dropped, generic O*NET titles spanning multiple SOC major
  groups are skipped, and canonical profession surfaces longer than two words are filtered out.
  Aliases on retained rows are preserved as-is, including long aliases.
- **Occupation taxonomy beats title heuristics.** O*NET SOC major groups and ESCO/ISCO codes are
  used where available; title keywords are only a fallback or a more specific override for clear
  cases such as legal, health, education, and media occupations.
- **GeoNames ambiguity uses a single deterministic representative.** Duplicate city surfaces keep
  the most populous GeoNames row, matching the prior runtime fallback behavior and avoiding merged
  multi-country level bags.
- **Nationality aliases are explicit.** CLDR gives country labels, not demonyms, so common demonym
  aliases are added by a small local deterministic table rather than pretending CLDR supplied them.
- **Legacy ORG import is quarantined.** Untyped teacher-cache rows are imported only for
  high-confidence organization/institution/company language, with short/generic or weak untyped
  rows excluded. Explicit `ORG::...` keys remain allowed.

## Verification snapshot

- Commits:

  | commit | summary | role |
  |---|---|---|
  | `75ef31e` | Build exhaustive lattice profile cache | Added the dataset-backed cache pipeline, source parsers, runtime lookup integration, standard cache spec, and rebuilt artifact. |
  | pending | Profession surface cap and substitutor smoke | Filters profession canonical surfaces to at most two words, rebuilds the cache, verifies substitutor cache use, fixes indefinite articles, and adds this dev log. |

- Cache rebuild command:
  `PYTHONPATH=src:scripts .venv/bin/python -u scripts/populate_lattice_profiles.py --raw-dir data/lattice_sources/raw --geo-dir data/geonames --teacher-cache data/lattice_cache.json --out data/lattice_profiles/fine_lattice_profiles.json --coverage-out results/lattice_profile_coverage.json --report-out results/lattice_profile_population.json --require-exhaustive`
- Rebuild result: `rows=251881 profiles=220892`, `exhaustive=True`.
- Cache audit after rebuild: schema validation errors `[]`; profession surfaces over two words
  `0`; profession singleton `["worker"]` rows `0`.
- Real-cache substitutor smoke:
  `application developer` -> `computer and mathematical occupation`; `Oslo` -> `a city in norway`.
- Test commands run before commit:
  `PYTHONPATH=src:scripts .venv/bin/python -m pytest src/cloak/tests/test_lattice_profiles.py src/cloak/tests/test_lattice_profile_builders.py -q`
  and `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests -q`.

## Caveats

- Profession completeness is intentionally reduced by the two-word canonical-surface cap. This is
  a data-quality tradeoff requested after auditing the cache; long aliases on retained entries
  still allow many longer detected strings to resolve.
- ORG coverage is still sparse and conservative. A proper organization source family is preferable
  to broad legacy teacher-cache inference.
- The article fixer is a small English heuristic for final output readability, not a full grammar
  engine.
