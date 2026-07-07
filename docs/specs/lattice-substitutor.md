---
type: reference
status: current
created: 2026-07-07
updated: 2026-07-07
tags: [substitution, lattice, detector, fine-types, privacy]
companion: [docs/specs/detector-model.md, docs/specs/RL/surrogate-ranker-infiller.md, docs/plans/2026-07-07-lattice-substitutor-fine-types.md]
---

# Lattice substitutor - fine-type runtime spec

## Purpose

The lattice substitutor rewrites `doc_orig` into `doc_p` by replacing detected sensitive spans with either
typed placeholders or truthful, privacy-legal generalizations. It consumes detector spans and produces the
client-side substitution record `R`, which later drives extraction and reconstruction.

This spec defines the next runtime contract after FT-detector v7: **fine detector types are first-class
substitution types**. `DEM` remains only a research-evaluation rollup for TAB comparability, not a runtime
substitution type for v7 fine mode.

## Definitions

- **Runtime type** - the type string written into `Span.type`, `R[*].type`, action tables, anonymity floors,
  probe pools, and typed placeholders.
- **Research rollup** - a mapping used only by detector evaluation, e.g. `nationality -> DEM`, to score
  against TAB-8 gold.
- **Lattice** - ordered replacement candidates for a span, most specific to most general. The action set
  also includes keep-original at depth 0 where a policy permits it, and a typed placeholder terminal that is
  always legal.
- **Coarsest text level** - the broadest grammatical, truthful phrase that may replace the original surface
  in context. It must be a usable replacement phrase, not a bare restatement of the type name.
- **Placeholder terminal** - the final privacy action for a runtime type, e.g. `<NATIONALITY_1>`. For fine
  demographic leaves, this is the lattice/action terminal when no grammatical privacy-legal text level
  remains.
- **Placeholder-only leaf** - a leaf whose sensitive categorical fact should not be rewritten into a semantic
  text floor. Its non-keep privacy action is a typed placeholder.
- **Anonymity floor** - the per-runtime-type minimum anonymity-set count required for a non-placeholder
  replacement to be legal.
- **Probe pool** - same-runtime-type distractor surfaces used by offline contrastive re-identification
  diagnostics (`walk_risk`). Pools are diagnostics and teacher features, not the deployment privacy mask.
- **Rule-based generalization** - a deterministic rewrite rule that maps a surface to a coarser truthful
  phrase, e.g. `34 -> thirty-something` or `120,000 dollars -> between 60,000 and 240,000 dollars`. Earlier
  notes called these "buckets"; this spec uses the more explicit term.

## Runtime Type Contract

The detector must emit these runtime types to the substitutor. The table separates three ideas that are easy
to conflate:

- **Detector/schema origin** - where the type enters the pipeline.
- **Span class** - what kind of sensitive surface the type denotes.
- **Substitution policy family** - which action/lattice rule family handles the type at runtime.

| Runtime type         | Detector/schema origin     | Span class                                              | Substitution policy family                           |
| -------------------- | -------------------------- | ------------------------------------------------------- | ---------------------------------------------------- |
| `PERSON`             | TAB / Presidio direct type | Person name or alias                                    | Forced typed placeholder                             |
| `CODE`               | TAB / Presidio direct type | Reference number, contact code, account-like identifier | Forced typed placeholder                             |
| `ORG`                | TAB-8 coarse type          | Organization, company, court, institution               | Organization generalization lattice                  |
| `LOC`                | TAB / Presidio coarse type | Location, address, city, country                        | GeoNames first, then WordNet / teacher lattice       |
| `DATETIME`           | TAB / Presidio coarse type | Date, time, duration                                    | Rule-based date/time generalization lattice          |
| `QUANTITY`           | TAB / Presidio coarse type | Amount, money, percentage, count                        | Rule-based quantity-range generalization lattice     |
| `MISC`               | TAB-8 coarse type          | Identifying residual attribute or event                 | WordNet / teacher lattice, conservative floor        |
| `nationality`        | Fine DEM leaf              | Nationality or citizenship                              | Country/region/continent/nationality lattice         |
| `ethnicity`          | Fine DEM leaf              | Ethnicity, race, ancestry group                         | Ethnicity/ancestry-region lattice                    |
| `religion`           | Fine DEM leaf              | Religion, belief, denomination, branch                  | Religious tradition/affiliation lattice              |
| `profession`         | Fine DEM leaf              | Profession, occupation, job title                       | Profession domain/sector lattice                     |
| `age`                | Fine DEM leaf              | Age expression                                          | Rule-based age-range generalization lattice          |
| `gender`             | Fine DEM leaf              | Gender value                                            | Placeholder-or-keep categorical policy               |
| `marital-status`     | Fine DEM leaf              | Marital status value                                    | Placeholder-or-keep categorical policy               |
| `health-condition`   | Fine DEM leaf              | Disease, diagnosis, health condition                    | Condition-family lattice                             |
| `sexual-orientation` | Fine DEM leaf              | Sexual orientation value                                | Placeholder-or-keep categorical policy               |
| `family-role`        | Fine DEM leaf              | Family role or relationship                             | Family-relationship lattice with conservative floors |
| `demographic-other`  | Fine DEM leaf              | Residual demographic attribute                          | Placeholder-first residual demographic policy        |

`DEM` must not be emitted by the v7 fine-mode substitutor. It may still appear when running old coarse
detectors, historical artifacts, or TAB research gates. Any new fine-mode runtime path that produces `DEM`
is a bug unless the caller explicitly requested coarse legacy mode.

## Placeholder Convention

Typed placeholder tokens use the runtime type directly, normalized for token syntax:

1. Uppercase ASCII.
2. Replace every run of non-alphanumeric characters with `_`.
3. Strip leading/trailing `_`.
4. Append a 1-based per-type counter.

Examples:

| Runtime type | Placeholder |
|---|---|
| `PERSON` | `<PERSON_1>` |
| `health-condition` | `<HEALTH_CONDITION_1>` |
| `marital-status` | `<MARITAL_STATUS_1>` |
| `sexual-orientation` | `<SEXUAL_ORIENTATION_1>` |
| `family-role` | `<FAMILY_ROLE_1>` |

`R` stores the external runtime type, not `DEM`:

```json
{
  "surface": "diabetes",
  "type": "health-condition",
  "action": "placeholder",
  "replacement": "<HEALTH_CONDITION_1>"
}
```

All placeholder consumers must accept internal underscores. The current placeholder residue and counter-seed
patterns that only match `<[A-Z]+_\d+>` are not compatible with this contract; they must become equivalent to
`<[A-Z][A-Z0-9_]*_\d+>`.

## Lattice Policy by Type

### Direct identifiers

`PERSON` and `CODE` are forced placeholders. They do not receive textual lattice generalizations in the
deployed substitutor because the surface itself is directly identifying and the typed placeholder is cleanly
invertible.

### Existing quasi types

`DATETIME`, `QUANTITY`, and `LOC` keep their current special sources:

- `DATETIME` uses rule-based date/time generalizations. Age-like text should move to the `age` runtime type
  in fine mode.
- `QUANTITY` uses rule-based numeric range generalizations.
- `LOC` uses GeoNames chains first, then WordNet or teacher fallback.

`ORG` and `MISC` continue to use WordNet / teacher lattices, but their coarsest text levels remain
conservative:

| Type | Coarsest text level |
|---|---|
| `ORG` | `an organization` |
| `MISC` | `something` |

### Hierarchical fine leaves

These leaves should expose useful semantic generalizations when the anonymity floor allows them. Their text
levels must remain grammatical as replacements for the detected span. A phrase that only names the type, such
as `a nationality`, `an ethnicity`, or `a profession`, is not a valid terminal text level for these leaves;
the terminal action is the typed placeholder.

#### `nationality`

Goal: preserve broad nationality/citizenship utility without exposing exact citizenship unless allowed.

Preferred lattice sources:

1. Demonym/country gazetteer: `Polish -> Central European -> European -> <NATIONALITY_n>`.
2. Country-to-continent fallback: `Kenyan -> East African -> African -> <NATIONALITY_n>`.

The exact demonym or country is keep-original only, never a generated floor.

#### `ethnicity`

Goal: support finer ethnicity generalization through grammatical ancestry/region abstractions.

Examples:

- `Kurdish -> of Middle Eastern ethnicity -> of West Asian ethnicity -> <ETHNICITY_n>`.
- `Roma -> of European ethnicity -> <ETHNICITY_n>`.
- `Tamil -> of South Asian ethnicity -> <ETHNICITY_n>`.

Implementation can start with a curated gazetteer for observed TAB/Nemotron terms, then fall back to teacher
lattices gated by NLI. The coarsest text level should usually be a grammatical region/ancestry phrase when a
reliable mapping exists; otherwise use the typed placeholder.

#### `religion`

Goal: generalize denominations/traditions carefully without inventing false hierarchies.

Examples:

- `Catholic -> Christian -> <RELIGION_n>`.
- `Sunni -> Muslim -> <RELIGION_n>`.
- `Muslim -> <RELIGION_n>` if no broader grammatical truthful text level is available.

Do not use `a religious affiliation` as a replacement. It names the type but does not usually substitute
cleanly for a religion surface.

#### `profession`

Goal: preserve occupational utility through domain/sector ladders instead of collapsing to a type-name phrase.

Examples:

- `cardiologist -> medical specialist -> healthcare worker -> <PROFESSION_n>`.
- `journalist -> media worker -> <PROFESSION_n>`.
- `prosecutor -> legal professional -> <PROFESSION_n>`.
- `teacher -> education worker -> <PROFESSION_n>`.

Implementation should include a small profession-domain map for high-frequency occupations, then WordNet or
teacher fallback. `a profession` is not a valid terminal text level; use `<PROFESSION_n>` when no grammatical
domain/sector replacement is legal.

#### `age`

Goal: reuse the existing rule-based age generalization behavior under the external `age` type.

Examples:

- `34 -> thirty-something`.
- `17 years old -> teenaged`.
- `72-year-old -> seventy-something`.

The coarsest text level is a parsed age-range generalization, not the text `an age range`, when a
deterministic rule can parse the surface. If parsing fails, the terminal privacy action is `<AGE_n>`.

#### `health-condition`

Goal: preserve medical/health utility at a condition-family level where possible.

Examples:

- `diabetes -> endocrine condition -> chronic condition -> <HEALTH_CONDITION_n>`.
- `depression -> mental health condition -> <HEALTH_CONDITION_n>`.
- `asthma -> respiratory condition -> <HEALTH_CONDITION_n>`.
- `HIV -> infectious disease -> <HEALTH_CONDITION_n>`.

Implementation can start with a curated condition-family map over observed detector training/eval surfaces.
WordNet may help for common diseases, but must be NLI-gated because medical hypernyms are easy to overstate.
Do not use `a health condition` as the terminal text level; use `<HEALTH_CONDITION_n>` when no grammatical
condition-family replacement is legal.

#### `family-role`

Goal: retain broad family-relation information only when useful and legal.

Examples:

- `daughter -> child -> <FAMILY_ROLE_n>`.
- `wife -> spouse -> <FAMILY_ROLE_n>`.
- `grandfather -> grandparent -> <FAMILY_ROLE_n>`.

This leaf is more privacy-sensitive than profession or health because exact relationship structure can aid
re-identification. Its anonymity floor should be conservative, and the placeholder terminal should be common
at strict operating points. Do not use `a family relationship` as a terminal text level.

### Placeholder-or-keep categorical leaves

These leaves are first-class lattice/action types but do not get semantic text generalizations by default:

- `gender`
- `marital-status`
- `sexual-orientation`

Their action set is:

1. `keep-original`, with anonymity set `1`, legal only when the user/policy explicitly sets the type floor
   to allow keep.
2. Typed placeholder terminal, always legal:
   - `<GENDER_n>`
   - `<MARITAL_STATUS_n>`
   - `<SEXUAL_ORIENTATION_n>`

Do not emit replacements like `a gender`, `a marital status`, or `a sexual orientation` in the default
runtime. Those phrases leak the presence of the exact sensitive category while carrying little task utility.
They may exist as internal labels for documentation, but not as generated `doc_p` fills.

### Residual demographic leaf

`demographic-other` is a catch-all for fine DEM spans the relabeler/model cannot place into a coherent leaf.
Default behavior is placeholder-first:

1. Keep-original only under explicit user waiver.
2. A grammatical text level only if a strict policy permits semantic residual disclosure and the replacement
   is not just a restatement of the type name.
3. `<DEMOGRAPHIC_OTHER_n>` otherwise.

The implementation should fail closed to the placeholder if it cannot certify a non-placeholder action.

## Generalization Lattice Construction

Lattices are built by runtime type, not by TAB rollup. Every lattice has the same shape:

1. Optional keep-original action, legal only under the caller's policy.
2. Zero or more grammatical text levels, ordered most specific to most general.
3. Typed placeholder terminal, always legal.

The text levels must come from the first trustworthy source that applies:

| Source | Applies to | Runtime requirement |
|---|---|---|
| Rule-based generalizer | `DATETIME`, `QUANTITY`, `age` | Deterministic parse only; parse miss means no text level. |
| Gazetteer / curated map | `nationality`, `ethnicity`, `profession`, `health-condition`, `religion`, `family-role` | Maps observed surfaces to approved grammatical levels. |
| Structured ontology | `LOC`, selected medical/profession/religion terms when available | Full-phrase match only for certifying privacy. |
| Strict WordNet | `ORG`, `MISC`, and non-covered hierarchical leaves | Full-phrase synset only; no last-word fallback for legality. |
| Offline teacher cache | Any hierarchical type with no local hit | Precomputed candidates only, never a live deployed remote call. |

Initial fine-type lattice source registry:

| Runtime type | Initial dataset-backed sources | Runtime use |
|---|---|---|
| `nationality` | Wikidata demonyms, CLDR territory names, GeoNames, HANCESTRO regions | Demonym/country to region/continent chains; exact citizenship remains keep-only |
| `ethnicity` | HANCESTRO, U.S. Census race/ethnicity standards, Wikidata | Ancestry/population to region rollups; use only grammatical replacement phrases |
| `religion` | Wikidata religion/worldview graph, ARDA religion datasets, Wikidata subclass graph | Denomination to tradition or broad affiliation; avoid type-label phrases |
| `profession` | ESCO, O*NET, ISCO-08, Wikidata occupation graph | Job-title aliases and occupation to sector/domain ladders |
| `age` | Deterministic parser; optional MeSH Age Groups and CDC age-group tables | Numeric age to age bucket/life-stage text when parsed; parse miss goes placeholder |
| `health-condition` | Mondo, Disease Ontology, MeSH RDF, ICD-11, UMLS | Disease synonym normalization and condition-family hierarchy |
| `family-role` | KIN ontology, WordNet, Wikidata kinship properties, schema.org Person relations | Kinship term to broad family-role ladders with conservative floors |
| `gender` | GSSO, Wikidata sex-or-gender values | Alias normalization only; no semantic text level by default |
| `marital-status` | FHIR marital-status value set, HL7 marital-status terminology | Alias normalization only; no semantic text level by default |
| `sexual-orientation` | GSSO, Wikidata sexual-orientation values | Alias normalization only; no semantic text level by default |
| `demographic-other` | None as an exhaustive source | Placeholder-first residual bucket; semantic text only under explicit strict policy |

The durable generated artifact is `data/lattice_profiles/fine_lattice_profiles.json`.
Runtime code may read this artifact but must not read raw source files or call source APIs.
Raw source files live under `data/lattice_sources/raw/` and are consumed only by
`scripts/build_lattice_profiles.py`.

The detailed dataset survey and links live in
[`docs/research/datasets.md`](../research/datasets.md#top-sources-by-fine-type-for-lattice-construction).

All candidate text levels, regardless of source, must pass the same filters:

- The replacement is grammatical in the original sentence.
- The replacement is truthfully entailed by the original sentence under NLI or an equivalent deterministic
  rule.
- The replacement does not contain the original surface, original-specific numbers, or proper-name tokens.
- The replacement is not merely a type-name phrase such as `a nationality`, `a profession`, or
  `a health condition`.
- `aset_count(fill, type, original, strict=True) >= K_FLOORS[type]` for legality.
- If the filter removes every text level, the lattice still contains the typed placeholder terminal.

Offline teacher-assisted lattice building is allowed only as an artifact build step:

1. Collect uncovered `(surface, type, context)` triples from fine-mode detector output or training/eval
   surfaces.
2. Prompt the teacher with the runtime type and the grammar constraint: candidates must be replacements for
   the marked span in context, not labels naming the type.
3. NLI-gate the candidates in the span sentence.
4. Run the generic candidate filters above.
5. Store approved candidates in the local lattice cache with the runtime type included in the cache key.
6. Treat empty approved candidate lists as a valid result: placeholder-only is the safe outcome.

User-defined runtime types must register a lattice profile before they can emit text levels. A profile
defines the detector label/gazetteer, placeholder token, policy family, candidate sources, `aset_count`
semantics, and initial `K_FLOORS` entry. Without a profile, a new type is placeholder-only by default.

## Anonymity Floors

Every runtime type must have an explicit floor in `K_FLOORS`. Fine leaves must not silently inherit `DEM`.

Initial floors:

| Runtime type | Initial floor | Rationale |
|---|---:|---|
| `LOC` | 100 | existing calibrated default |
| `ORG` | 100 | existing calibrated default |
| `DATETIME` | 100 | existing calibrated default |
| `QUANTITY` | 100 | existing calibrated default |
| `MISC` | 100 | existing default-deny posture |
| `nationality` | 100 | coarse region/nationality nodes only unless broad enough |
| `ethnicity` | 100 | avoid small ethnicity/region cells |
| `religion` | 100 | conservative until measured |
| `profession` | 100 | domain/sector nodes should clear or placeholder |
| `age` | 100 | most decade ranges do not clear 100 by count alone; strict policy will often placeholder |
| `health-condition` | 100 | family-level only unless countable |
| `family-role` | 100 | conservative because household structure can identify |
| `gender` | 2 | allows keep only under explicit low-floor/user-waiver settings; default policy can still force placeholder |
| `marital-status` | 2 | same placeholder-or-keep policy |
| `sexual-orientation` | 2 | same placeholder-or-keep policy, but default deployment should set force-placeholder |
| `demographic-other` | 100 | residual default-deny |
| `OTHER` | 100 | unknown runtime type default-deny |

The placeholder-or-keep leaves need a second policy bit in addition to numeric floors:
`force_placeholder_types = {"gender", "marital-status", "sexual-orientation"}` by default. Setting a floor to
`1` or `2` is not enough to emit generic semantic text; it only allows keep if the caller explicitly removes
the type from `force_placeholder_types`.

Measured floor calibration must follow the empirical-honesty rule: compare realized privacy against the same
attacker protocol, do not tune per-type floors to equalize lexical overlap or any secondary diagnostic.

## Anonymity Counts

`aset_count(fill, span_type, original, strict=True)` must understand every fine runtime type.

Required behavior:

- `fill == original` returns `1`.
- Typed placeholders are always legal outside `aset_count`; they do not need a count.
- Approved coarsest text levels for hierarchical leaves return `GENERIC` or a large conservative count only
  when the fill exactly matches a grammatical approved replacement.
- Specific-looking but unparseable fine fills fail closed to `1`.
- `gender`, `marital-status`, and `sexual-orientation` non-placeholder fills fail closed unless they are exact
  keep-original and the caller's policy permits keep.
- `age` uses the age-specific rule branch, not `DATETIME` calendar-window rules.
- `profession`, `health-condition`, `religion`, `ethnicity`, `nationality`, and `family-role` can use curated
  count tables first, then strict WordNet counts only where the phrase has a full-phrase synset. Last-word
  fallbacks are diagnostic-only, not certifying.

## Probe Pools

`data/probe_distractors.json` currently has coarse TAB-8 keys. Fine runtime types need pools under their exact
type names, or `walk_risk()` will fail closed with risk `1.0` and the legacy tau walk will exhaust to
placeholder for all fine leaves.

Pool requirements:

1. Build pools from fine-mode detected surfaces and/or fine-labeled training records.
2. Store keys for every runtime type, including the placeholder-or-keep categorical leaves.
3. Set a minimum pool size gate. If a pool has fewer than `MIN_POOL`, report it as missing and keep the
   fail-closed behavior.
4. Do not alias fine pools to `DEM`; that hides the mismatch the migration is meant to fix.

For deployment legality, probe pools remain diagnostic. The legal mask is `aset_count >= K_FLOORS[type]`
plus the typed placeholder terminal.

## Action Set Construction

For each non-direct span, the action table should be:

1. Optional keep-original action:
   - included for ranker training and explicit user-waiver policies;
   - `aset = 1`;
   - illegal by default for normal privacy operation unless the type floor/policy permits it.
2. Zero or more lattice text levels:
   - only for types with hierarchical or rule-based generalization policy;
   - each level has `fill`, `mode = "level"`, `aset`, `walk_risk` diagnostic, and proximity diagnostic.
3. Typed placeholder terminal:
   - `mode = "placeholder"`;
   - replacement assembled dynamically as `<NORMALIZED_TYPE_n>`;
   - always legal.

For placeholder-or-keep leaves, the action table contains only keep-original and placeholder by default.

## Substitution Record `R`

`R` must preserve fine runtime type identity:

```json
[
  {
    "surface": "journalist",
    "type": "profession",
    "action": "generalize",
    "replacement": "media worker",
    "lattice": ["media worker", "<PROFESSION_1>"],
    "risk": 0.0
  },
  {
    "surface": "married",
    "type": "marital-status",
    "action": "placeholder",
    "replacement": "<MARITAL_STATUS_1>",
    "risk": 0.0
  }
]
```

No new fine-mode `R` entry should use `type: "DEM"`.

## Extractor and Reconstructor Compatibility

The extractor and reconstructor consume `R.type` for typed sanity checks, pointer compatibility, prompt
linearization, and placeholder cleanup. They must be updated for fine runtime types.

Required compatibility changes:

- Placeholder token regexes must support internal underscores.
- `_type_sane()` must understand fine leaves. It should not uppercase fine type strings and lose the
  hyphenated identity before matching.
- Pointer compatibility should match fine types exactly. Research-only rollup to `DEM` is not valid for
  runtime inversion.
- Reconstructor linearization should print fine types as stored in `R`, e.g.
  `health-condition: "chronic condition" => "diabetes"`.
- Placeholder-or-keep leaves should be easy to invert because the placeholder path is exact.

Legacy `DEM` entries in old artifacts may remain supported by compatibility code, but new fine-mode tests
must assert that v7 fine substitution produces fine types.

## Research Evaluation Boundary

`rollup_type()` remains useful for TAB-8 scoring:

```text
nationality -> DEM
ethnicity -> DEM
religion -> DEM
profession -> DEM
age -> DEM
gender -> DEM
marital-status -> DEM
health-condition -> DEM
sexual-orientation -> DEM
family-role -> DEM
demographic-other -> DEM
```

That rollup must not be used in substitution, action construction, anonymity floors, probe pools,
placeholders, `R`, extractor typing, or ranker features except when explicitly loading legacy coarse
artifacts.

Implementation steps and verification live in
[2026-07-07-lattice-substitutor-fine-types.md](../plans/2026-07-07-lattice-substitutor-fine-types.md).

## Non-goals

- Do not recalibrate method comparisons with per-model or per-type fudge factors.
- Do not use `DEM` as an invisible runtime fallback for fine leaves.
- Do not add remote calls to deployed substitution.
- Do not claim improved privacy from finer types without an attacker-measured privacy result.
- Do not solve MISC decomposition here. MISC remains coarse unless a separate spec decomposes it.

## Open Questions

1. The exact curated maps for `ethnicity`, `profession`, `health-condition`, `nationality`, `religion`, and
   `family-role` should start from observed TAB/Nemotron/v7 surfaces and be expanded only as needed.
2. Whether `age` should keep decade-range generalizations under the initial floor of 100 is an empirical
   policy question: count-based legality may placeholder many ages. That is acceptable if measured honestly.
3. Placeholder-or-keep policy needs a user-facing configuration surface before users can intentionally waive
   hiding for `gender`, `marital-status`, or `sexual-orientation`.
4. End-to-end privacy and utility remain unmeasured for fine runtime types. Detector and substitutor tests are
   upstream checks, not the final privacy claim.
