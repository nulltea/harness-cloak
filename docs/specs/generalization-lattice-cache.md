---
type: reference
status: current
created: 2026-07-07
updated: 2026-07-10
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
        "count": 1.0,
        "entry_origin": "observed-surface"
      },
      "sberbank": {
        "aliases": [],
        "levels": ["a financial institution", "an organization"],
        "source_ids": ["legacy-teacher-cache:sberbank"],
        "count": 1.0,
        "entry_origin": "observed-surface"
      }
    },
    "drug": {
      "bupropion": {
        "aliases": [],
        "levels": ["aminoketone", "central nervous system agent"],
        "source_ids": ["producer:drug-health-procedure-nemotron-3-super:drug:bupropion"],
        "count": 9.0,
        "entry_origin": "observed-surface",
        "level_counts": {
          "aminoketone": 9.0,
          "central nervous system agent": 400.0
        },
        "level_grounding": {
          "aminoketone": {
            "status": "certifying",
            "source_family": "openfda-pharm-class",
            "selector": "openfda_ndc.pharm_class == 'Aminoketone [EPC]'",
            "member_set_ref": "openfda-ndc:pharm_class:Aminoketone [EPC]"
          },
          "central nervous system agent": {
            "status": "model-proposed",
            "source_family": "model-proposed",
            "count_basis": "real-world-reference-estimate",
            "selector": "central nervous system agent",
            "member_set_ref": null
          }
        }
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
- `entry_origin`: how the row's surface was obtained. `observed-surface` (a real surface from a corpus or a
  seed dataset — GeoNames/CLDR/ESCO and every dataset-seeded row use this) or `generated-universe` (a synthetic
  surface produced by the lattice producer to fill out a level's membership). Runtime lookup ignores this field.
- `level_counts` (optional): per-level anonymity-set size, keyed by a member of `levels`. An explicit value is
  an absolute estimate for that generalization tier, not a per-surface frequency, and it overrides the legacy
  row-`count` fallback in the runtime `(runtime_type, level) -> count` index. Carried by the producer-built
  types (`drug`/`health-condition`/`medical-procedure`) and by `LOC`/`nationality`/`organization-medical-facility`,
  whose per-level counts are deterministically countable (GeoNames universe / CLDR M49 containment /
  corpus-membership; see `scripts/spikes/ground_deterministic_level_counts.py`). The remaining dataset-seeded types carry only the
  row-level `count`, since their per-level anonymity sizes are not grounded (fabricating them would invent
  privacy numbers). A row may carry counts for only a subset of its levels — the k-anonymity walk is enforced
  over the covered subset.
- `level_grounding` (optional): per-level provenance for each `level_counts` entry, keyed the same way. Fields:
  `status` (`certifying` · `corpus-adjusted-from-certifying-source` · `model-proposed`), `source_family` (e.g.
  `doid-is-a`, `icd10pcs-prefix`, `openfda-pharm-class`, `geonames-universe`, `cldr-m49`, `corpus-membership`,
  `model-proposed`),
  `count_basis` (e.g. `corpus-membership`, `real-world-reference-estimate`), `count_evidence`, `selector`,
  `member_set_ref`. Review/audit metadata only; runtime lookup does not read it.

Rows are intentionally self-contained. A reviewer can inspect or edit a surface's aliases and replacement
levels without dereferencing source-family metadata elsewhere in the file. `entry_origin`, `level_counts`, and
`level_grounding` are optional: a row with only `aliases`, `levels`, `source_ids`, and `count` is valid.

Current local builders populate this schema for every runtime type with local data:

- `LOC` from GeoNames country/city files.
- `ORG` from conservative rows in the legacy teacher cache.
- `nationality`, `profession`, `religion`, `gender`, `marital-status`, and `sexual-orientation` from the
  dataset-backed fine-type builders.
- `drug`, `health-condition`, and `medical-procedure` from the generalization lattice producer, merged in via
  `scripts/merge_lattice_profiles.py`. These carry `level_counts`/`level_grounding`.
- `LOC`, `nationality`, and `organization-medical-facility` also carry `level_counts`/`level_grounding`,
  grounded deterministically by `scripts/spikes/ground_deterministic_level_counts.py`: `LOC` from the GeoNames
  universe and `nationality` from CLDR M49 (both real-world, `status: certifying`); `organization-medical-facility`
  from corpus-membership over its own rows (`status: model-proposed`, `count_basis: corpus-membership`), a
  conservative undercount used because the full NPPES universe is not available locally. The remaining
  dataset-seeded types (`profession`, `religion`, `ORG`, placeholder-only categoricals) carry only `count`.

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
- A `levels` entry must not contain the canonical surface text. (Merges that fold a narrower surface into a
  broader canonical drop any pulled-in level that would violate this — see `scripts/merge_lattice_profiles.py`.)
- `count` must be at least `1.0`.
- Every `level_counts` key must be a member of `levels`, and each value must be at least `1.0`.
- `level_counts` must be non-decreasing along the row's `levels` order (specific → broad k-anonymity walk;
  see `docs/specs/offline-k-anonimity-risk-walk.md`).
- Runtime substitution reads this cache locally and deterministically.
