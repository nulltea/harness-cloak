---
type: plan
status: current
created: 2026-07-07
updated: 2026-07-07
tags: [substitution, lattice, fine-types, detector, privacy, plan]
companion: [../specs/lattice-substitutor.md, ../specs/detector-model.md, ../specs/RL/surrogate-ranker-infiller.md]
---

# Fine-type lattice substitutor implementation plan

## Goal

Implement the runtime contract in `docs/specs/lattice-substitutor.md`: fine detector types become externally
visible substitution types in `R`, typed placeholders, lattice/action tables, anonymity floors, probe pools,
and extractor/reconstructor paths. `DEM` remains research-eval rollup only.

## Constraints

- Do not use `DEM` as a hidden fallback for fine runtime types.
- Do not use type-name phrases such as `a nationality`, `an ethnicity`, `a profession`, or
  `a health condition` as terminal text levels.
- Every non-direct lattice ends with a typed placeholder terminal.
- Deployed substitution must not make live remote calls; teacher lattices are offline artifacts only.
- Method comparisons and floor choices must follow the empirical-honesty rules in `AGENTS.md`.

## Implementation steps

### Step 1 - Runtime type registry

Add shared runtime type constants for:

- `FINE_DEM_TYPES`
- `RUNTIME_TYPES`
- `LEGACY_ROLLUP_TYPES`
- `PLACEHOLDER_ONLY_TYPES`
- `FORCED_PLACEHOLDER_TYPES`
- `placeholder_type_token(type_name: str) -> str`
- a placeholder regex equivalent to `<[A-Z][A-Z0-9_]*_\d+>`

Use the registry from detector/substitutor/ranker/extractor code instead of duplicating placeholder/type
normalization rules.

Acceptance tests:

- `placeholder_type_token("health-condition") == "HEALTH_CONDITION"`.
- `placeholder_type_token("PERSON") == "PERSON"`.
- Placeholder regex matches `<HEALTH_CONDITION_1>`, `<MARITAL_STATUS_2>`, and `<PERSON_1>`.
- Placeholder regex rejects `<health-condition_1>`.

### Step 2 - Generalization lattice builders

Implement runtime-type lattice profiles. Each profile returns grammatical text levels plus a typed placeholder
terminal; it never returns `DEM` or a type-name phrase as a terminal text level.

Builder rules:

- `DATETIME`, `QUANTITY`, `age`: deterministic rule-based generalizers; parse miss means no text level.
- `nationality`, `ethnicity`, `profession`, `health-condition`, `religion`, `family-role`: curated map first,
  then approved ontology/strict WordNet where available, then offline teacher cache.
- `gender`, `marital-status`, `sexual-orientation`: no semantic text levels by default; keep-original only
  under explicit policy, placeholder always.
- `demographic-other`: placeholder-first; only emit a grammatical text level under explicit policy and never
  just a type-name phrase.

Candidate filters:

- grammatical replacement in the original sentence;
- truthfulness via deterministic rule or NLI gate;
- no original surface, original-specific numbers, or proper-name tokens;
- no `a <type-name>` terminal phrases;
- legal only when `aset_count(fill, type, original, strict=True) >= K_FLOORS[type]`.

Acceptance tests:

- `lattice_for("diabetes", "health-condition", context)` includes a grammatical node like
  `chronic condition` and terminates with `<HEALTH_CONDITION_n>`.
- `lattice_for("journalist", "profession", context)` includes `media worker` or an equivalent grammatical
  sector node and terminates with `<PROFESSION_n>`.
- `lattice_for("Kurdish", "ethnicity", context)` includes a grammatical ancestry/region node and terminates
  with `<ETHNICITY_n>`.
- `lattice_for("married", "marital-status", context)` exposes no semantic text level by default and still
  has `<MARITAL_STATUS_n>` as the terminal action.
- Type-name phrases such as `a profession` and `a health condition` are rejected or count as illegal.

### Step 3 - Offline teacher cache for uncovered surfaces

Extend the teacher-lattice artifact path so cache keys include runtime type as well as surface. The prompt
must tell the teacher that candidates are replacements for the marked span in context, not labels naming the
type.

Build flow:

1. Collect uncovered `(surface, type, context)` triples from fine-mode detector output or training/eval
   surfaces.
2. Generate candidate levels offline.
3. NLI-gate each candidate in the span sentence.
4. Run the generic lattice filters from Step 2.
5. Store approved candidates locally.
6. Store empty approved lists as valid placeholder-only outcomes.

Acceptance tests:

- Cached entries for the same surface under different runtime types do not collide.
- Teacher candidates containing the original surface are rejected.
- Teacher candidates that only restate the type name are rejected.
- Empty approved teacher output still allows placeholder substitution.

### Step 4 - Fine anonymity floors and counts

Extend `K_FLOORS` and `aset_count()` for every runtime type in the spec.

Acceptance tests:

- Every `RUNTIME_TYPES` member has a floor or an explicit direct-placeholder exemption.
- `aset_count("thirty-something", "age", "34", strict=True) == 10.0`.
- `aset_count("a health condition", "health-condition", "diabetes", strict=True) == 1.0`.
- `aset_count("a profession", "profession", "journalist", strict=True) == 1.0`.
- `aset_count("a gender", "gender", "female", strict=True) == 1.0`.
- Specific-looking but unparseable fine fills fail closed to `1.0`.

### Step 5 - Substitutor and `R`

Make typed placeholders use normalized runtime type tokens and preserve fine type strings in `R`.

Acceptance tests:

- A `Span(type="health-condition", text="diabetes")` that exhausts emits `<HEALTH_CONDITION_1>` and
  `R[0]["type"] == "health-condition"`.
- A `Span(type="marital-status", text="married")` emits `<MARITAL_STATUS_1>` under default policy.
- Fine-mode substitution never writes `type: "DEM"` for fine leaves.
- Legacy coarse `DEM` still works when an old detector emits it.

### Step 6 - Fine probe pools

Update the pool builder so fine-mode surfaces are stored under exact runtime type keys.

Acceptance tests:

- `data/probe_distractors.json` contains keys for all fine leaves after a fine pool build.
- Missing or thin fine pools are reported, not silently aliased to `DEM`.
- `walk_risk(..., span_type="health-condition")` does not fail closed when the pool has at least
  `MIN_POOL` entries.

### Step 7 - Ranker/action artifacts

Action tables, floor-walk, and ranker type features must accept fine runtime types.

Acceptance tests:

- `derive_spans()` uses `floors[s["type"]]` for fine leaves, never `DEM`.
- Ranker type features include fine leaves or map only truly unknown types to `OTHER`.
- Placeholder assembly in ranker rollouts emits `<HEALTH_CONDITION_1>` and seeds counters from existing
  fine placeholders.

### Step 8 - Extractor/reconstructor compatibility

Update placeholder regexes, type sanity, pointer compatibility, and reconstructor prompt linearization.

Acceptance tests:

- `invert("Patient has <HEALTH_CONDITION_1>.", R)` restores the original health condition.
- Placeholder residue stats count stray `<MARITAL_STATUS_2>`.
- `_type_sane("health-condition", "chronic condition", "chronic condition")` accepts.
- `_type_sane("gender", "<GENDER_1>", "female")` is not used to approve semantic inversion for gender.
- Reconstructor restore maps preserve `health-condition`, `profession`, and other fine type labels.

### Step 9 - End-to-end fine-mode smoke

Use a deterministic span list or stub detector over text containing:

- a person name;
- a health-condition span;
- a profession span;
- an ethnicity span;
- marital status or sexual orientation.

Assert:

- no runtime `DEM` in `R`;
- fine placeholders are externally visible;
- hierarchical leaves get grammatical lattice candidates;
- placeholder-or-keep leaves placeholder by default;
- inversion restores placeholders exactly.

## Verification protocol

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
