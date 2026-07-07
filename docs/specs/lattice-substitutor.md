---
type: reference
status: current
created: 2026-07-07
updated: 2026-07-07
tags: [substitution, lattice, detector, fine-types, privacy, implementation-plan]
companion: [docs/specs/detector-model.md, docs/specs/RL/surrogate-ranker-infiller.md]
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
- **Floor label** - the final generic text node for leaves that can safely expose a broad semantic category.
- **Placeholder-only leaf** - a leaf whose sensitive categorical fact should not be rewritten into a semantic
  text floor. Its non-keep privacy action is a typed placeholder.
- **Anonymity floor** - the per-runtime-type minimum anonymity-set count required for a non-placeholder
  replacement to be legal.
- **Probe pool** - same-runtime-type distractor surfaces used by offline contrastive re-identification
  diagnostics (`walk_risk`). Pools are diagnostics and teacher features, not the deployment privacy mask.

## Runtime Type Contract

The detector must emit these runtime types to the substitutor:

| Runtime type | Source | Runtime role |
|---|---|---|
| `PERSON` | direct identifier | forced typed placeholder |
| `CODE` | direct identifier | forced typed placeholder |
| `ORG` | named quasi/direct context | lattice via organization generalization |
| `LOC` | location | GeoNames / WordNet lattice |
| `DATETIME` | date, time, duration | date/time bucket lattice |
| `QUANTITY` | amount, money, percent | quantity bucket lattice |
| `MISC` | identifying residual attribute/event | WordNet / teacher lattice, conservative floor |
| `nationality` | fine demographic leaf | country/region/continent/nationality lattice |
| `ethnicity` | fine demographic leaf | ethnicity/ancestry-region lattice |
| `religion` | fine demographic leaf | religious tradition/affiliation lattice |
| `profession` | fine demographic leaf | profession domain/sector lattice |
| `age` | fine demographic leaf | age bucket lattice |
| `gender` | fine demographic leaf | placeholder-or-keep categorical leaf |
| `marital-status` | fine demographic leaf | placeholder-or-keep categorical leaf |
| `health-condition` | fine demographic leaf | condition-family lattice |
| `sexual-orientation` | fine demographic leaf | placeholder-or-keep categorical leaf |
| `family-role` | fine demographic leaf | family-relationship lattice with conservative floors |
| `demographic-other` | fine demographic leaf | placeholder-first residual demographic leaf |

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

- `DATETIME` uses date/time buckets. Age-like text should move to the `age` runtime type in fine mode.
- `QUANTITY` uses numeric range buckets.
- `LOC` uses GeoNames chains first, then WordNet or teacher fallback.

`ORG` and `MISC` continue to use WordNet / teacher lattices, but their floor labels remain conservative:

| Type | Floor label |
|---|---|
| `ORG` | `an organization` |
| `MISC` | `something` |

### Hierarchical fine leaves

These leaves should expose useful semantic generalizations when the anonymity floor allows them.

#### `nationality`

Goal: preserve broad nationality/citizenship utility without exposing exact citizenship unless allowed.

Preferred lattice sources:

1. Demonym/country gazetteer: `Polish -> a Central European nationality -> a European nationality -> a nationality`.
2. Country-to-continent fallback: `Kenyan -> an East African nationality -> an African nationality -> a nationality`.
3. Floor: `a nationality`.

The exact demonym or country is keep-original only, never a generated floor.

#### `ethnicity`

Goal: support finer ethnicity generalization than `an ethnicity`, especially region/ancestry abstractions.

Examples:

- `Kurdish -> a Middle Eastern ethnicity -> a West Asian ethnicity -> an ethnicity`.
- `Roma -> a European ethnicity -> an ethnicity`.
- `Tamil -> a South Asian ethnicity -> an ethnicity`.

Implementation can start with a curated gazetteer for observed TAB/Nemotron terms, then fall back to teacher
lattices gated by NLI. The floor label is `an ethnicity`, but it should usually be reached through a region
node when a reliable mapping exists.

#### `religion`

Goal: generalize denominations/traditions carefully without inventing false hierarchies.

Examples:

- `Catholic -> a Christian denomination -> a Christian religious affiliation -> a religious affiliation`.
- `Sunni -> an Islamic branch -> a religious affiliation`.
- `Muslim -> a religious affiliation` if no stricter truthful intermediate is available.

Floor: `a religious affiliation`.

#### `profession`

Goal: preserve occupational utility through domain/sector ladders instead of collapsing directly to
`a profession`.

Examples:

- `cardiologist -> a medical specialist -> a healthcare profession -> a profession`.
- `journalist -> a media profession -> a profession`.
- `prosecutor -> a legal profession -> a profession`.
- `teacher -> an education profession -> a profession`.

Implementation should include a small profession-domain map for high-frequency occupations, then WordNet or
teacher fallback. The floor is `a profession`, but the expected useful nodes are domain and sector nodes.

#### `age`

Goal: reuse the existing age bucket behavior under the external `age` type.

Examples:

- `34 -> thirty-something`.
- `17 years old -> teenaged`.
- `72-year-old -> seventy-something`.

The floor is a bucket, not the text `an age range`, when a bucket can be parsed. If parsing fails, the
terminal privacy action is `<AGE_n>`.

#### `health-condition`

Goal: preserve medical/health utility at a condition-family level where possible.

Examples:

- `diabetes -> an endocrine condition -> a chronic condition -> a health condition`.
- `depression -> a mental health condition -> a health condition`.
- `asthma -> a respiratory condition -> a health condition`.
- `HIV -> an infectious disease -> a health condition`.

Implementation can start with a curated condition-family map over observed detector training/eval surfaces.
WordNet may help for common diseases, but must be NLI-gated because medical hypernyms are easy to overstate.
Floor: `a health condition`.

#### `family-role`

Goal: retain broad family-relation information only when useful and legal.

Examples:

- `daughter -> a child -> a family relationship`.
- `wife -> a spouse -> a family relationship`.
- `grandfather -> a grandparent -> a family relationship`.

This leaf is more privacy-sensitive than profession or health because exact relationship structure can aid
re-identification. Its anonymity floor should be conservative, and the placeholder terminal should be common
at strict operating points. Floor: `a family relationship`.

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
2. `a demographic attribute` only if a strict policy permits semantic residual disclosure.
3. `<DEMOGRAPHIC_OTHER_n>` otherwise.

The implementation should fail closed to the placeholder if it cannot certify a non-placeholder action.

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
| `age` | 100 | most decade buckets do not clear 100 by count alone; strict policy will often placeholder |
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
- Known floor labels for hierarchical leaves return `GENERIC` or a large conservative count only when the
  fill exactly matches an approved floor.
- Specific-looking but unparseable fine fills fail closed to `1`.
- `gender`, `marital-status`, and `sexual-orientation` non-placeholder fills fail closed unless they are exact
  keep-original and the caller's policy permits keep.
- `age` uses the age branch from date bucketing, not `DATETIME` calendar windows.
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
   - only for types with hierarchical or bucketable policy;
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
    "replacement": "a media profession",
    "lattice": ["a media profession", "a profession"],
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
  `health-condition: "a health condition" => "diabetes"`.
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

## Implementation Plan

### Step 1 - Add a runtime type registry

Create one shared registry module or constants block that defines:

- `FINE_DEM_TYPES`
- `RUNTIME_TYPES`
- `LEGACY_ROLLUP_TYPES`
- `PLACEHOLDER_ONLY_TYPES`
- `FORCED_PLACEHOLDER_TYPES`
- `placeholder_type_token(type_name: str) -> str`
- `placeholder_regex`

Use it from detector/substitutor/ranker/extractor rather than duplicating ad hoc type string rules.

Acceptance tests:

- `placeholder_type_token("health-condition") == "HEALTH_CONDITION"`.
- `placeholder_type_token("PERSON") == "PERSON"`.
- Placeholder regex matches `<HEALTH_CONDITION_1>` and `<PERSON_1>`.
- Placeholder regex rejects malformed tokens like `<health-condition_1>`.

### Step 2 - Add fine lattice floors and policies

Update lattice construction so known fine leaves never fall back to `"something"` unless their policy says
they are residual and placeholder-first.

For placeholder-or-keep leaves, the lattice helper may return an empty text-level list or a policy object that
marks the type as text-level-disabled. The important contract is that action construction still adds
keep-original when policy permits and always adds the typed placeholder terminal.

Acceptance tests:

- `lattice_for("diabetes", "health-condition", context)` includes `a health condition` or a stricter
  health-family node.
- `lattice_for("journalist", "profession", context)` includes `a media profession` or `a profession`.
- `lattice_for("Kurdish", "ethnicity", context)` includes a region/ethnicity node or `an ethnicity`.
- `lattice_for("married", "marital-status", context)` exposes no semantic text level by default; action
  construction still supplies placeholder.
- `lattice_for("female", "gender", context)` exposes no semantic text level by default.
- `lattice_for("gay", "sexual-orientation", context)` exposes no semantic text level by default.

### Step 3 - Add fine anonymity floors and counts

Extend `K_FLOORS` and `aset_count()` for every runtime type.

Acceptance tests:

- Every `RUNTIME_TYPES` member has a floor or an explicit direct-placeholder exemption.
- `aset_count("a health condition", "health-condition", "diabetes", strict=True) == GENERIC`.
- `aset_count("a profession", "profession", "journalist", strict=True) == GENERIC`.
- `aset_count("a gender", "gender", "female", strict=True) == 1.0`.
- `aset_count("female", "gender", "female", strict=True) == 1.0`.
- `aset_count("thirty-something", "age", "34", strict=True) == 10.0`.

### Step 4 - Update substitutor placeholder assembly

Make typed placeholders use normalized runtime type tokens and preserve fine type strings in `R`.

Acceptance tests:

- A `Span(type="health-condition", text="diabetes")` that exhausts emits `<HEALTH_CONDITION_1>` and
  `R[0]["type"] == "health-condition"`.
- A `Span(type="marital-status", text="married")` emits `<MARITAL_STATUS_1>` under default policy.
- Fine-mode substitution never writes `type: "DEM"` for fine leaves.
- Legacy coarse `DEM` still works when an old detector emits it.

### Step 5 - Rebuild or derive fine probe pools

Update the pool builder so fine-mode surfaces are stored under fine runtime keys.

Acceptance tests:

- `data/probe_distractors.json` contains keys for all fine leaves after a fine pool build.
- Missing/thin fine pools are reported, not silently aliased to `DEM`.
- `walk_risk(..., span_type="health-condition")` does not fail closed when the pool has at least
  `MIN_POOL` entries.

### Step 6 - Update ranker/action artifacts

Action tables, floor-walk, and ranker type features must accept fine runtime types.

Acceptance tests:

- `derive_spans()` uses `floors[s["type"]]` for fine leaves, never `DEM`.
- Ranker one-hot or embedding features include fine leaves or map only truly unknown types to `OTHER`.
- Placeholder assembly in ranker rollouts emits `<HEALTH_CONDITION_1>` and seeds counters from existing
  fine placeholders.

### Step 7 - Update extractor/reconstructor consumers

Update placeholder regexes, type sanity, pointer compatibility, and reconstructor prompts.

Acceptance tests:

- `invert("Patient has <HEALTH_CONDITION_1>.", R)` restores the original health condition.
- Placeholder residue stats count stray `<MARITAL_STATUS_2>`.
- `_type_sane("health-condition", "a health condition", "a chronic condition")` accepts.
- `_type_sane("gender", "<GENDER_1>", "female")` is not used to approve semantic inversion for gender.
- Reconstructor restore map preserves `health-condition` and `profession` type labels.

### Step 8 - Add an end-to-end fine-mode smoke

Use a short text containing at least:

- a person name;
- a health condition;
- a profession;
- an ethnicity;
- marital status or sexual orientation.

Run fine-mode substitution with deterministic spans or a stub detector. Assert:

- no runtime `DEM` in `R`;
- fine placeholders are externally visible;
- hierarchical leaves get lattice candidates;
- placeholder-or-keep leaves placeholder by default;
- inversion restores placeholders exactly.

## Verification Protocol

Minimum local verification before claiming the migration complete:

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests -q
PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_extract.py -q
PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_train_roundtrip_mode.py -q
```

If code paths touch detector inference or gate scripts, also run the smallest non-heavy fine-mode smoke before
any full GPU gate:

```bash
PYTHONPATH=src .venv/bin/python -u scripts/latticecloak_detection_gate.py \
  --gliner-model data/models/pii_gliner_finedem/final \
  --fine-dem --threshold 0.02 --limit 5 \
  --out results/finedem_runtime_type_smoke.json
```

Any longer gate or rebuild must follow the repo performance gate and GPU rules in `AGENTS.md`.

## Non-goals

- Do not recalibrate method comparisons with per-model or per-type fudge factors.
- Do not use `DEM` as an invisible runtime fallback for fine leaves.
- Do not add remote calls to deployed substitution.
- Do not claim improved privacy from finer types without an attacker-measured privacy result.
- Do not solve MISC decomposition here. MISC remains coarse unless a separate spec decomposes it.

## Open Questions

1. The exact curated maps for `ethnicity`, `profession`, `health-condition`, `nationality`, `religion`, and
   `family-role` should start from observed TAB/Nemotron/v7 surfaces and be expanded only as needed.
2. Whether `age` should keep decade buckets under the initial floor of 100 is an empirical policy question:
   count-based legality may placeholder many ages. That is acceptable if measured honestly.
3. Placeholder-or-keep policy needs a user-facing configuration surface before users can intentionally waive
   hiding for `gender`, `marital-status`, or `sexual-orientation`.
4. End-to-end privacy and utility remain unmeasured for fine runtime types. Detector and substitutor tests are
   upstream checks, not the final privacy claim.
