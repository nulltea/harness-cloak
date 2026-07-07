---
type: reference
status: current
created: 2026-07-07
updated: 2026-07-07
tags: [substitution, lattice, cache, schema, datasets]
companion: [docs/specs/lattice-substitutor.md, docs/plans/2026-07-07-fine-lattice-dataset-build.md]
---

# Generalization Lattice Cache Schema

## Purpose

The generalization lattice cache is the offline artifact read by runtime substitution to map a detected
surface and runtime type to truthful replacement candidates. Runtime code must not download datasets, call
remote APIs, or run teacher generation while substituting prompts.

The standard deterministic cache path is:

```text
data/lattice_profiles/fine_lattice_profiles.json
```

The older teacher cache path is:

```text
data/lattice_cache.json
```

`data/lattice_cache.json` is a legacy teacher-result cache for coarse/WordNet fallback paths. It is not the
standard dataset-backed lattice cache.

`LOC` and fine `nationality` are intentionally separate:

- `LOC` city/country chains are computed at runtime from GeoNames files under `data/geonames/`, especially
  `countryInfo.txt`, `admin1CodesASCII.txt`, and `cities500.txt`, then fall back to WordNet when GeoNames
  has no match.
- `nationality` rows in this cache come from nationality/citizenship datasets such as CLDR and Wikidata
  seed exports. `source_ids` such as `cldr:DE` or `wikidata:Q183` are provenance only; runtime lookup uses
  the inline `profiles[nationality][surface].levels` row, not those source IDs.

## Standard Schema

```json
{
  "schema_version": 1,
  "created": "2026-07-07",
  "sources": {},
  "profiles": {
    "nationality": {
      "germany": {
        "aliases": ["federal republic of germany"],
        "levels": ["western european nationality", "european nationality"],
        "source_ids": ["cldr:DE", "wikidata:Q183"],
        "count": 1000.0
      },
      "united states": {
        "aliases": [
          "the united states",
          "the united states of america",
          "the usa",
          "us"
        ],
        "levels": ["north american nationality", "american nationality"],
        "source_ids": ["cldr:US", "wikidata:Q30"],
        "count": 1000.0
      }
    },
    "LOC": {
      "new york city": {
        "aliases": ["big apple", "city of new york", "new york", "ny", "nyc"],
        "levels": [
          "a city in new york",
          "a city in united states",
          "a city in north america"
        ],
        "source_ids": ["geonames:5128581"],
        "count": 8804190.0
      },
      "germany": {
        "aliases": ["de", "deu", "gm"],
        "levels": ["a country in europe"],
        "source_ids": ["geonames-country:DE"],
        "count": 82927922.0
      }
    },
    "ORG": {
      "starbucks": {
        "aliases": [],
        "levels": ["a coffee shop", "a retail chain", "a commercial establishment"],
        "source_ids": ["legacy-teacher-cache:starbucks"],
        "count": 1.0
      },
      "sberbank": {
        "aliases": [],
        "levels": ["a financial institution", "an organization"],
        "source_ids": ["legacy-teacher-cache:sberbank"],
        "count": 1.0
      }
    },
    "profession": {
      "journalist": {
        "aliases": ["correspondent", "reporter", "news writer"],
        "levels": ["media worker", "professional worker"],
        "source_ids": [
          "esco:a47f82d3-4cf7-4abf-bb36-2c8cbad28234",
          "onet-job-title:27-3023.00",
          "wikidata:Q1930187"
        ],
        "count": 1000.0
      }
    },
    "religion": {
      "judaism": {
        "aliases": ["jewish religion"],
        "levels": ["abrahamic religion", "religious tradition"],
        "source_ids": ["wikidata:Q9268"],
        "count": 1000.0
      },
      "catholics": {
        "aliases": [],
        "levels": ["christian religion", "religious tradition"],
        "source_ids": ["arda:1200"],
        "count": 1000.0
      }
    },
    "gender": {
      "female": {
        "aliases": ["woman"],
        "levels": [],
        "source_ids": ["manual:gender:female"],
        "count": 1.0
      }
    },
    "marital-status": {
      "married": {
        "aliases": ["wedded"],
        "levels": [],
        "source_ids": ["manual:marital-status:married"],
        "count": 1.0
      }
    }
  }
}
```

## Field Contract

- `schema_version`: integer. Version `1` is the standard cache schema.
- `created`: build date in `YYYY-MM-DD`.
- `sources`: optional source summary object. It must not be required for runtime lookup or manual lattice
  review.
- `profiles`: object keyed first by runtime type, then by normalized canonical surface.
- `aliases`: normalized alternate surfaces that should resolve to the same row.
- `levels`: ordered replacement candidates, most specific to most general. Placeholder terminals are not
  stored here; runtime appends the typed placeholder where required.
- `source_ids`: source-local provenance identifiers such as `cldr:DE` or `esco:<uuid>`.
- `count`: conservative anonymity-set count for the row's lattice fills.

Rows are intentionally self-contained. A reviewer can inspect or edit a surface's aliases and replacement
levels without dereferencing source-family metadata elsewhere in the file.

Current local builders populate this schema for every runtime type with local data:

- `LOC` from GeoNames country/city files.
- `ORG` from conservative rows in the legacy teacher cache.
- `nationality`, `profession`, `religion`, `gender`, `marital-status`, and `sexual-orientation` from the
  dataset-backed fine-type builders.

Placeholder-only categorical types have `levels: []` by design. Their runtime non-keep action is the typed
placeholder terminal, e.g. `<GENDER_1>` or `<MARITAL_STATUS_1>`.

## Runtime Indexing

The artifact is optimized in memory, not by duplicating review-sensitive data in JSON. The loader builds and
caches two process-local indexes:

- `(runtime_type, canonical_or_alias) -> levels`
- `(runtime_type, level) -> count`

These indexes are derived from the row data on load, so editing `profiles` remains the only required cache
edit.

## Invariants

- Runtime types must be members of `cloak.runtime_types.RUNTIME_TYPES`.
- Non-placeholder-only runtime types must have at least one text candidate in `levels`.
- `levels` entries must be grammatical replacement phrases, not type-name labels such as `a profession`.
- A `levels` entry must not contain the canonical surface text.
- `count` must be at least `1.0`.
- Runtime substitution reads this cache locally and deterministically.
