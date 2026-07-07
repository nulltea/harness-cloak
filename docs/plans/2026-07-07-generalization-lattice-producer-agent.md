---
type: plan
status: current
created: 2026-07-07
updated: 2026-07-07
tags:
  - substitution
  - lattice
  - langgraph
  - offline-build
  - fine-types
  - llama-swap
companion:
  - docs/specs/lattice-substitutor.md
  - docs/specs/generalization-lattice-cache.md
  - docs/specs/offline-k-anonimity-risk-walk.md
  - research-wiki/training/2026-07-05-FT-detector-v7-dem-decompose.md
---

# LangGraph Generalization Lattice Producer Agent

> For agentic workers: implement this plan task-by-task. Keep checkboxes updated. Do not replace LangGraph
> with a custom loop, ad hoc state machine, or manual Pi Agent process.

## Goal

Build a long-running offline **LangGraph** agent that expands the local lattice profile artifact with
truthful, grammatical generalization levels for fine detector runtime types. The producer uses the served
local model behind llama-swap to propose missing lattice levels, but runtime substitution remains fully
deterministic and reads only local artifacts.

The producer must not limit itself to entries already mined from local datasets. It must also generate
proposed universe entries, aliases, lattice levels, and deterministic proposed per-level counts for
Knowledgator GLiNER PII categories and FT-detector v7 fine leaves whose local dataset coverage is thin or
missing.

Primary output:

```text
data/lattice_profiles/proposed/fine_lattice_profiles.proposed.json
```

Canonical runtime artifact remains:

```text
data/lattice_profiles/fine_lattice_profiles.json
```

The proposed artifact is a separate cache-shaped file, not the real lattice cache. It is updated
incrementally as levels are accepted so crashes do not lose already verified work. It becomes eligible for
mapping into the canonical runtime cache only after validation and review approval.

Proposed artifacts under `data/lattice_profiles/proposed/` are **fresh agent-output files**. They must not
copy all rows from the canonical profile artifact. They contain only rows processed and accepted by the
producer, plus proposal metadata such as `base_profile_hash`. They must set
`proposal_scope: "producer-processed-only"` so older copied-cache proposal files can be detected and reset.

## Locked Tool Decision

**LangGraph is the required orchestration tool.** The producer must be implemented as a LangGraph graph with
typed state, named nodes, conditional edges, checkpointing, and a review interrupt before merge approval.

No fallback implementations are allowed:

- No repo-native custom agent loop.
- No hand-rolled resume cursor as the orchestration source of truth.
- No manual Pi Agent workflow for production lattice generation.
- No CrewAI, LlamaIndex Workflows, Pydantic AI durable workflows, or other orchestration substitutes.

Pi Agent may still be used by a human operator for manual review or prompt exploration outside the producer,
but Pi is not the runtime for this plan.

Rationale: this task is a long-running, resumable, inspectable workflow. LangGraph directly targets
long-running stateful agents, checkpoint persistence, fault tolerance, streaming, and human-in-the-loop
interrupts. The lattice-specific semantics stay in local `cloak` modules; LangGraph owns execution control.

## Non-Negotiable Constraints

- Runtime substitution must never call LangGraph, llama-swap, or teacher generation.
- The producer must write proposed lattice levels to a separate proposed artifact, never directly to the
  canonical cache.
- Proposed data must be persisted incrementally after each accepted item, not only at the end of the run.
- `DEM` is an eval rollup only; no new fine-mode output may create `DEM` profile rows.
- Direct identifiers and placeholder-only leaves do not receive text lattice levels.
- Below-floor candidates may be logged as diagnostics but must not enter profile `levels`.
- Per-level counts are required for proposed levels. The model may propose grounding, but certifying counts
  must be computed by a deterministic verifier/compiler following
  [offline-k-anonimity-risk-walk.md](../specs/offline-k-anonimity-risk-walk.md). If grounding cannot be
  compiled to a source-backed member set, the level fails closed with count `1.0`.
- For categories without mined dataset entries, the agent must generate proposed type-universe entries first,
  then compile proposed level counts over that generated universe. Counts over generated universes are
  proposal counts, not runtime-certifying counts, until the proposed universe and count walk pass review.
- The graph must be resumable after process death using a persistent LangGraph checkpointer.
- Node side effects must be idempotent because LangGraph checkpoints at graph super-step boundaries and a
  failed node may rerun from the start.
- LangGraph state must not become model context by default. The model sees only a fresh bounded context
  packet for the current item/category, assembled from typed state and artifact slices.
- The producer must not feed prior assistant turns, raw `EXPERIMENT_LOG.md`, full JSONL logs, full profile
  artifacts, or previous raw model outputs back into later model calls.
- Local model calls must be cached with keys that include model, base URL, prompt version, runtime type,
  surface, and context.
- No external dataset downloads, remote APIs, dependency installs, or anonymity-floor changes during a
  producer run.
- Any implementation step that adds a new dependency still requires explicit user confirmation before the
  install/change is performed.

## Runtime Type Scope

Eligible lattice-producing runtime types:

- `ORG`
- `LOC`
- `DATETIME`
- `QUANTITY`
- `MISC`
- `nationality`
- `ethnicity`
- `religion`
- `profession`
- `age`
- `health-condition`
- `family-role`

Placeholder-only or direct runtime types are queueable only for audit/alias accounting:

- `PERSON`
- `CODE`
- `gender`
- `marital-status`
- `sexual-orientation`
- `demographic-other`

Raw Knowledgator labels must be normalized before queueing. Direct labels such as `name`, `email address`,
`phone number`, account identifiers, credentials, and government IDs must map to forced-placeholder runtime
policies and must not reach the proposal node.

## Category Coverage Registry

Add a registry that maps every detector label family to one of three outcomes:

1. `forced_placeholder` - no text levels; typed placeholder only.
2. `runtime_lattice` - map to an existing runtime type and generate/compile lattice entries.
3. `needs_profile` - no existing runtime type; create a proposed profile only after explicit policy review.

Initial mapping:

| Detector label family | Runtime outcome | Notes |
|---|---|---|
| `name`, `first name`, `last name`, `name medical professional` | `forced_placeholder` -> `PERSON` | Direct identifiers |
| `email address`, `phone number`, `ip address`, `url` | `forced_placeholder` -> `CODE` | Contact endpoints are direct identifiers |
| `account number`, `bank account`, `routing number`, `credit card`, `cvv`, `ssn`, `passport number`, `driver license`, `username`, `password`, `vehicle id`, `healthcare number`, `medical code` | `forced_placeholder` -> `CODE` | Account/credential/code-like identifiers |
| `dob`, `credit card expiration`, discharge/admission dates | `runtime_lattice` -> `DATETIME` | Rule-derived time lattices |
| `money`, `dose` | `runtime_lattice` -> `QUANTITY` | Dose may need unit-aware medical quantity tests |
| `age` | `runtime_lattice` -> `age` | Rule-derived age lattices |
| `gender`, `marital status`, `sexual orientation` | `forced_placeholder` -> fine categorical leaves | Placeholder-or-keep policy |
| `location address`, `location street`, `location city`, `location state`, `location country`, `location zip` | `runtime_lattice` -> `LOC` when geographic, otherwise `CODE` for exact postal/address codes | Address fragments need conservative handling |
| `organization medical facility`, organizations | `runtime_lattice` -> `ORG` | Counts fail closed unless a source universe exists |
| `condition`, `injury` | `runtime_lattice` -> `health-condition` | Generate missing entries and health-family levels |
| `medical process`, `drug`, `blood type` | `needs_profile` by default | Do not silently fold into health-condition unless a profile defines levels/count semantics |
| v7 `nationality`, `ethnicity`, `religion`, `profession`, `family-role` | `runtime_lattice` -> same fine runtime type | Generate entries where dataset coverage is missing |
| v7 `demographic-other` | `forced_placeholder` unless explicit semantic residual policy exists | Residual bucket stays default-deny |

The queue builder must emit coverage gaps for every label family, not only for observed profile rows. A
category with no dataset-backed entries becomes a generated-universe task when it maps to `runtime_lattice`,
or a `needs_profile` report item when no runtime policy exists.

## Files

Create:

- `src/cloak/lattice_producer/__init__.py`
- `src/cloak/lattice_producer/state.py`
- `src/cloak/lattice_producer/graph.py`
- `src/cloak/lattice_producer/queue.py`
- `src/cloak/lattice_producer/propose.py`
- `src/cloak/lattice_producer/gates.py`
- `src/cloak/lattice_producer/merge.py`
- `src/cloak/lattice_producer/coverage.py`
- `src/cloak/lattice_producer/counts.py`
- `scripts/run_lattice_producer.py`
- `src/cloak/tests/test_lattice_producer_queue.py`
- `src/cloak/tests/test_lattice_producer_gates.py`
- `src/cloak/tests/test_lattice_producer_merge.py`
- `src/cloak/tests/test_lattice_producer_graph.py`

Modify:

- `requirements.txt` only after explicit approval to add `langgraph`.
- `docs/specs/generalization-lattice-cache.md` only if the accepted/proposed artifact convention needs a
  schema clarification for proposed artifacts or per-level counts.

Do not modify runtime substitution code unless tests reveal it already cannot consume valid profile artifacts.

## Run Directory Contract

Each run lives under:

```text
data/lattice_producer/runs/<run_id>/
```

Required files:

| File | Owner | Purpose |
|---|---|---|
| `EXPERIMENT_BRIEF.md` | graph initialization | Frozen run brief: goal, constraints, success metric, model pin |
| `EXPERIMENT_LOG.md` | graph reporting nodes | Append-only per-entry processing log plus final status |
| `queue.jsonl` | queue builder | Normalized work items |
| `coverage_gaps.json` | coverage node | Missing/weak category coverage by detector label family and runtime type |
| `generated_universe.jsonl` | generation node | Proposed canonical entries and aliases for categories without mined datasets |
| `proposals.jsonl` | proposal node | Raw model proposals plus parse diagnostics |
| `accepted.jsonl` | gate node | Mergeable levels that passed every gate |
| `rejected.jsonl` | gate node | Rejections with machine-readable reasons |
| `diagnostics.jsonl` | gate node | Below-floor or otherwise informative non-mergeable candidates |
| `fine_lattice_profiles.proposed.json` | persist node | Incrementally updated cache-shaped proposed artifact under `data/lattice_profiles/proposed/` |
| `coverage.json` | report node | Per-type queue/accepted/rejected/missing counts |
| `checkpoints.sqlite` | LangGraph checkpointer | Persistent graph checkpoints |

There must be no separate `run_state.json` cursor. The graph checkpoint plus append-only artifact logs are
the state. If a summary file is useful for humans, generate it from LangGraph state and logs; do not make it
authoritative.

## Context Management and Context-Rot Control

LangGraph provides durable execution state; it does not require using accumulated conversational history as
LLM context. This producer must separate **workflow state** from **model prompt context**.

Statefulness comes from:

- LangGraph checkpoint state for control fields: queue index, current item, counts, review status, and error
  status.
- Append-only artifacts for durable knowledge: `generated_universe.jsonl`, `accepted.jsonl`,
  `diagnostics.jsonl`, and `data/lattice_profiles/proposed/fine_lattice_profiles.proposed.json`.
- Static run configuration: model names, local base URL, artifact paths, prompt version, and bounded CLI
  caps.

LLM context comes only from a freshly assembled **context packet** for the current node invocation. The
packet must be small, typed, and reproducible:

```json
{
  "prompt_version": "lattice-producer-v1",
  "task_kind": "generated-universe|level-proposal",
  "runtime_type": "profession",
  "detector_label_family": "profession",
  "surface_or_entry": "cardiologist",
  "marked_context_sentence": "She is a [SPAN]cardiologist[/SPAN] in Oslo.",
  "type_policy": "...short policy excerpt...",
  "allowed_outputs": "...short schema excerpt...",
  "nearby_profile_rows": [],
  "category_slice": [],
  "forbidden_outputs": ["type-name phrases", "original surface leaks", "direct identifiers"]
}
```

Rules:

- Do not pass the full conversation transcript to the model.
- Do not pass full run logs, full proposed artifacts, or full source files to the model.
- Do not use `EXPERIMENT_LOG.md` as model memory. It is for human monitoring only.
- Do not summarize prior model outputs into future prompts unless the summary is generated from accepted
  structured artifacts, not prose history.
- Limit `nearby_profile_rows` and `category_slice` by deterministic retrieval: same runtime type, same
  detector label family, same generated grouping, alias overlap, or exact source grounding. Use caps such as
  `--max-context-rows`.
- If the category slice exceeds the cap, include aggregate counts plus top relevant rows, never an LLM-written
  freeform summary as the only source of truth.
- Prompt reproducibility is required: the context packet hash is part of the model cache key and proposal
  provenance.

This design gives the producer coherence through stable policy, schema, and artifacts while avoiding context
rot from long-running narrative memory.

## Proposed Artifact Schema

The proposed artifact must be structurally close to the canonical profile cache so it can be promoted after
verification. It must not be read by runtime substitution unless a caller explicitly asks for a staged
artifact.

Required shape:

```json
{
  "schema_version": 1,
  "created": "2026-07-07",
  "artifact_role": "proposal",
  "proposal_scope": "producer-processed-only",
  "base_profile_hash": "...",
  "producer_run_id": "...",
  "sources": {},
  "profiles": {
    "profession": {
      "cardiologist": {
        "aliases": [],
        "entry_origin": "generated-universe",
        "levels": ["medical specialist", "healthcare worker"],
        "level_counts": {
          "medical specialist": 42,
          "healthcare worker": 310
        },
        "level_groundings": {
          "medical specialist": {
            "status": "proposal-universe",
            "source_family": "generated-universe",
            "selector": "generated_group:medical-specialist",
            "member_set_ref": "producer:<run_id>:profession:medical-specialist"
          },
          "healthcare worker": {
            "status": "proposal-universe",
            "source_family": "generated-universe",
            "selector": "generated_group:healthcare-worker",
            "member_set_ref": "producer:<run_id>:profession:healthcare-worker"
          }
        },
        "source_ids": ["producer:<run_id>:<item_id>"]
      }
    }
  }
}
```

Differences from the current canonical cache are intentional:

- `artifact_role: "proposal"` prevents confusing this file with the runtime cache.
- `level_counts` stores deterministic per-level counts. Row-level `count` is not a certifying field for
  proposed fine-type levels.
- `level_groundings` records how the deterministic compiler produced or failed to produce the count.
- `entry_origin` distinguishes `dataset-mined`, `observed-surface`, `legacy-cache`, and
  `generated-universe` rows.
- Promotion to the canonical cache is a separate approval-gated step that may either preserve `level_counts`
  if the canonical schema has been upgraded or compile them into the current runtime count lookup path.

Generated-universe rows have stricter status rules:

- Their `level_counts` are computed by deterministic inversion over the generated proposed universe.
- Their `level_groundings[*].status` must be `proposal-universe` until review accepts the generated universe
  as a source snapshot.
- They may pass into the proposed artifact for review, but they must not be treated as runtime-certifying
  unless the approval step explicitly promotes the generated universe and count walk.

The producer must rewrite this proposed artifact atomically after each accepted item: write a temporary file
in the same directory, `fsync` where practical, then rename over the previous proposed artifact.

## LangGraph State

Define a compact typed state in `src/cloak/lattice_producer/state.py`.

Required fields:

```python
class ProducerState(TypedDict):
    run_id: str
    run_dir: str
    profiles_path: str
    proposed_out: str
    queue_path: str
    current_item: dict | None
    queue_index: int
    prompt_version: str
    model: str
    escalation_model: str | None
    base_url: str
    max_items: int | None
    max_context_rows: int
    processed: int
    accepted: int
    rejected: int
    diagnostics: int
    proposed_persisted: int
    coverage_gaps: int
    generated_entries: int
    errors: list[dict]
    needs_review: bool
    review_decision: str | None
```

Do not store full profile artifacts, large corpora, or full proposal histories in graph state. Store paths,
counts, current item, and checkpointable control state. Append detailed records to JSONL files.

Use a stable `thread_id` derived from `run_id`. Keep it under 255 characters for checkpointer compatibility:

```text
lattice-producer:<run_id_hash>
```

## Graph Shape

Implement a LangGraph `StateGraph[ProducerState]` with these nodes:

```text
START
  -> initialize_run
  -> build_category_coverage
  -> build_or_load_queue
  -> select_next_item
       -> generate_universe_entries
       -> deterministic_lookup
       -> validate_proposed_artifact
  -> propose_with_llama_swap
  -> compile_level_counts
  -> gate_candidates
  -> persist_proposed_artifact
  -> record_item_result
  -> should_continue
       -> select_next_item
       -> validate_proposed_artifact
  -> review_interrupt
  -> finalize_run
  -> END
```

Conditional edges:

- `select_next_item -> validate_proposed_artifact` when queue is exhausted or `max_items` is reached.
- `select_next_item -> generate_universe_entries` for generated-universe tasks.
- `select_next_item -> deterministic_lookup` for observed-surface/profile/legacy-cache tasks.
- `generate_universe_entries -> propose_with_llama_swap` to generate levels for accepted proposed entries.
- `deterministic_lookup -> compile_level_counts` when deterministic sources produce candidates.
- `deterministic_lookup -> propose_with_llama_swap` when no deterministic candidate exists and context is
  available.
- `deterministic_lookup -> record_item_result` when the item is ineligible or contextless with no local hit.
- `propose_with_llama_swap -> compile_level_counts` on parseable response.
- `propose_with_llama_swap -> record_item_result` on unavailable llama-swap, parse failure, or empty proposal.
- `compile_level_counts -> gate_candidates` after every candidate has a certifying count or fail-closed
  count.
- `gate_candidates -> persist_proposed_artifact` when at least one level is accepted.
- `gate_candidates -> record_item_result` when no level is accepted.
- `persist_proposed_artifact -> record_item_result` after an atomic proposed-artifact write.
- `validate_proposed_artifact -> review_interrupt` only when validation passes.
- `validate_proposed_artifact -> finalize_run` with failure status when validation fails.

`review_interrupt` must use LangGraph interrupt semantics to pause with a JSON-serializable summary:

```json
{
  "run_id": "...",
  "proposed_out": "...",
  "coverage": ".../coverage.json",
  "accepted": 123,
  "rejected": 45,
  "diagnostics": 12
}
```

Resume accepts one of:

- `approve`
- `reject`
- `approve-proposed-only`

The graph must not overwrite the canonical `fine_lattice_profiles.json` without approval.

## Node Contracts

### `initialize_run`

- Create run directory.
- Create `EXPERIMENT_BRIEF.md` if missing.
- Create empty JSONL log files if missing.
- Initialize LangGraph state fields.
- Validate that the current command uses local llama-swap unless `--offline-only` is set.

Idempotency: safe to rerun; never truncates existing logs.

### `build_or_load_queue`

- If `queue.jsonl` exists, load it and preserve order.
- Otherwise build it from configured sources:
  - category coverage gaps from all known Knowledgator label families and v7 fine leaves;
  - missing observed fine-mode detector surfaces;
  - profile rows with missing levels for eligible lattice-producing types;
  - legacy teacher cache rows that can be re-keyed by runtime type;
  - explicit `--queue` input.
- Normalize every item to runtime type.
- Reject `DEM` in fine mode.
- Mark direct/placeholder-only types as ineligible, not as proposal work.

Idempotency: same inputs produce same queue order and same item IDs.

### `build_category_coverage`

- Load the category coverage registry.
- Compare every `runtime_lattice` category against the current profile artifact and known mined source rows.
- Emit `coverage_gaps.json` with:
  - detector label family;
  - mapped runtime type;
  - current profile row count;
  - current non-placeholder level count;
  - whether a dataset-backed source exists;
  - whether a generated-universe task is required.
- Do not mark direct/placeholder-only categories as missing lattice coverage; report them as intentionally
  placeholder-only.

Coverage gaps are first-class queue sources. The producer must cover categories with no mined entries by
generating proposed universe entries, not by leaving them absent from the plan.

### `select_next_item`

- Advance from `queue_index` to the next unprocessed item.
- Skip items already present in accepted/rejected/diagnostic logs by item ID.
- Stop when queue is exhausted or `max_items` is reached.

### `deterministic_lookup`

Try local sources before model calls:

- existing `lookup_levels(surface, runtime_type)`;
- rule generalizers for `DATETIME`, `QUANTITY`, and `age`;
- GeoNames/profile sources for `LOC`;
- curated/profile rows for fine leaves;
- strict WordNet full-phrase matches only where already allowed by lattice policy.

Outputs candidate levels with provenance `deterministic:<source>`.

### `generate_universe_entries`

- For a generated-universe task, ask the local model to propose a bounded set of canonical entries for the
  runtime type or detector label family, with aliases and broad grouping hints.
- Each proposed entry must include:
  - `runtime_type`;
  - `detector_label_family`;
  - `canonical_value`;
  - `aliases`;
  - `proposed_levels`;
  - `proposed_groundings`;
  - `entry_origin: "generated-universe"`;
  - `generation_rationale`, treated as non-certifying.
- Persist accepted proposed entries immediately to `generated_universe.jsonl` before level proposal/count
  work continues.
- Keep the generation bounded by CLI controls such as `--max-generated-entries-per-category`.
- Do not generate direct identifiers, credentials, account numbers, real names, phone numbers, addresses, or
  other concrete PII examples for placeholder-only categories.

The generated universe is a proposed source snapshot. It can support proposed counts inside the staged
artifact, but it does not certify runtime privacy until reviewed and promoted.

### `assemble_context_packet`

- Build the exact bounded JSON context packet for `generate_universe_entries` or `propose_with_llama_swap`.
- Select only relevant artifact slices:
  - same runtime type;
  - same detector label family;
  - same generated grouping;
  - alias overlap;
  - exact source grounding.
- Enforce `--max-context-rows`.
- Include hashes of every artifact slice used.
- Exclude full transcripts, full logs, full artifacts, and raw historical model outputs.

This can be a helper called by proposal nodes rather than a separate graph node, but it must have unit tests.

### `propose_with_llama_swap`

- Call the existing OpenAI-compatible local client pattern against `base_url`.
- Use temperature `0.0`.
- Require strict JSON output.
- For observed-surface tasks, require the model to output candidate levels plus proposed grounding selectors,
  not certifying numeric counts. Numeric counts in model output are ignored and recorded as non-certifying
  rationale only.
- For generated-universe tasks, require the model to output proposed entries, levels, grouping selectors, and
  aliases. Counts are still computed by `compile_level_counts`, not trusted from model text.
- Build prompts only from `assemble_context_packet()`. Do not use accumulated LangGraph state or messages as
  freeform prompt memory.
- Cache every model response under `INFERDPT_LLM_CACHE` or an equivalent repo-local cache keyed by prompt
  version, request identity, and context packet hash.
- Record raw output in `proposals.jsonl`.
- If the first model returns invalid JSON or empty levels, optionally call `escalation_model`.

This node must not call remote endpoints. If `base_url` is not localhost or an explicitly approved local
endpoint, fail the run.

### `compile_level_counts`

- For each candidate level, compile the proposed grounding into a deterministic member set according to
  [offline-k-anonimity-risk-walk.md](../specs/offline-k-anonimity-risk-walk.md).
- Store `level_count = len(member_set)` when the grounding is source-backed and certifying.
- For generated-universe rows, build a deterministic proposed type universe from `generated_universe.jsonl`
  and compute `level_count` by inverting levels over that proposed universe. Mark grounding status
  `proposal-universe`, not `certifying`, until review promotes it.
- Store count `1.0` with `status: "fail-closed"` when the grounding is missing, unsupported, open-vocabulary
  without a source universe, unparseable, or non-monotone.
- Check monotonicity across each row's proposed levels. Counts must be non-decreasing from specific to broad;
  otherwise reorder if the semantic order is wrong, or reject the offending level.
- Emit `level_groundings` metadata for every proposed level.

The generative model is not allowed to provide certifying counts. The LangGraph producer may propose
per-level counts only through this deterministic compiler node.

### `gate_candidates`

Apply the gates uniformly to deterministic and teacher-generated candidates:

| Gate | Merge requirement |
|---|---|
| Runtime type | Eligible lattice-producing type only |
| Grammar | Candidate can replace the marked span in context, or source is deterministic and context-independent |
| Truth | Replacement sentence is entailed by original sentence under NLI or deterministic source rule |
| Leak | No original surface, alias, exact distinctive number, code, or proper-name token |
| Type-name | Not a bare runtime label phrase |
| Ordering | Later levels are no more specific than earlier levels |
| Count | Dataset-backed levels need certifying `level_counts[level] >= K_FLOORS[runtime_type]`; generated-universe levels may enter the proposed artifact with `proposal-universe` counts but are not runtime-certifying |
| Schema | Candidate can be represented in the profile schema |

Below-floor candidates go to `diagnostics.jsonl`, not `accepted.jsonl`.

### `record_item_result`

- Append exactly one result record per item ID.
- Update counts in graph state.
- Update `EXPERIMENT_LOG.md` with a short rolling summary at a configurable interval.

### `persist_proposed_artifact`

- Read canonical profiles.
- Apply the current item's accepted rows into the proposed artifact path immediately.
- Preserve deterministic dataset-backed levels before producer levels.
- Deduplicate case-insensitively.
- Store `level_counts` and `level_groundings` for every accepted proposed level.
- Store generated-universe entries in the proposed artifact even when they came from no mined dataset, with
  `entry_origin: "generated-universe"` and non-certifying `proposal-universe` count status.
- Never store placeholder terminals in `levels`.
- Never create `DEM` rows.
- Store provenance in `source_ids` as `producer:<run_id>:<item_id>`.
- Write atomically through a same-directory temporary file and rename.
- Make repeated writes idempotent by item ID and level text.

### `validate_proposed_artifact`

- Run `validate_profile_artifact()`.
- Run producer-specific invariants:
  - no `DEM` profiles;
  - no direct/placeholder-only type with text levels from producer;
  - no below-floor diagnostics in accepted rows;
  - every accepted level has `level_counts[level]` and `level_groundings[level]`;
  - certifying counts are deterministic and monotone along each row;
  - generated-universe counts are labeled `proposal-universe` until approval;
  - every `runtime_lattice` coverage gap is either filled, queued, or explicitly reported as blocked;
  - no placeholder token in `levels`.

### `review_interrupt`

- Pause for human approval after validation.
- The interrupt payload must include proposed artifact path and coverage summary.
- Approval may copy proposed artifact to canonical path only when explicitly requested by CLI option
  `--allow-canonical-overwrite`; otherwise approval leaves the proposed artifact staged.

### `finalize_run`

- Write final coverage report.
- Write final `EXPERIMENT_LOG.md` status.
- Exit nonzero on validation failure or rejected review.

## CLI

Create:

```text
scripts/run_lattice_producer.py
```

Required command shape:

```bash
PYTHONPATH=src .venv/bin/python -u scripts/run_lattice_producer.py \
  --run-dir data/lattice_producer/runs/2026-07-07-fine-leaves-v1 \
  --profiles data/lattice_profiles/fine_lattice_profiles.json \
  --out data/lattice_profiles/proposed/fine_lattice_profiles.proposed.json \
  --base-url http://localhost:8060/v1 \
  --model "gemma 4 (E4B)" \
  --escalation-model "Qwen3.6-35B-A3B" \
  --max-items 200 \
  --workers 1
```

Additional flags:

- `--queue <path>`: use a prebuilt queue.
- `--offline-only`: skip llama-swap and use deterministic sources only.
- `--resume`: resume an existing LangGraph thread.
- `--review-decision approve|reject|approve-proposed-only`: resume from review interrupt.
- `--allow-canonical-overwrite`: permit approved run to replace canonical profile artifact.
- `--max-generated-entries-per-category <n>`: cap generated-universe expansion per uncovered category.
- `--max-context-rows <n>`: cap profile/generated-universe rows included in each model context packet.
- `--category <label-or-runtime-type>`: restrict a run to one detector label family or runtime type for
  smoke tests.

Default `--workers` is `1`. Raise it only after a tiny saturation probe shows the pinned llama-swap endpoint
benefits from concurrency.

## Implementation Tasks

- [x] Confirm dependency policy for adding `langgraph`.
- [x] Add LangGraph dependency only after approval.
- [x] Add `src/cloak/lattice_producer/state.py` with `ProducerState` and typed helper records.
- [x] Add context-packet tests: no transcript/log/full-artifact leakage; deterministic slicing; context hash
      changes when relevant artifact slices change; cap enforcement with `--max-context-rows`.
- [x] Add category coverage registry tests for Knowledgator personal/contact/financial/healthcare/ID labels
      plus v7 fine leaves.
- [x] Implement `src/cloak/lattice_producer/coverage.py`.
- [x] Add queue builder tests for runtime type normalization, `DEM` rejection, and forced-placeholder skips.
- [x] Implement `src/cloak/lattice_producer/queue.py`.
- [x] Add generated-universe tests: uncovered runtime-lattice category creates proposed entries/aliases;
      placeholder-only category does not generate concrete PII examples.
- [x] Add proposed-artifact schema tests for `artifact_role`, `level_counts`, `level_groundings`, and
      canonical-cache separation.
- [x] Add deterministic count-compiler tests from `docs/specs/offline-k-anonimity-risk-walk.md`: member-set
      count, generated-universe proposed count, fail-closed `1.0`, and monotone count ordering.
- [x] Add gate tests for self-leaks, type-name phrases, distinctive-number leaks, below-floor diagnostics,
      and placeholder terminals.
- [x] Implement `src/cloak/lattice_producer/gates.py`.
- [ ] Add incremental persistence tests: accepted item writes the proposed artifact before the next queue
      item, crash after write resumes without duplicate levels, and canonical cache is untouched.
- [x] Add merge/persist tests for schema-valid proposed artifacts, source provenance, no `DEM`, no
      placeholders in levels, per-level counts, and deterministic-level ordering.
- [x] Implement `src/cloak/lattice_producer/merge.py`.
- [x] Implement `src/cloak/lattice_producer/counts.py`.
- [ ] Add mocked proposal tests for strict JSON parsing, cache keys, local-base-url enforcement, and
      generated-universe expansion behavior, using only bounded context packets.
- [x] Implement `src/cloak/lattice_producer/propose.py`.
- [ ] Add graph tests with a persistent temporary SQLite checkpointer: run, interrupt, resume, and crash
      recovery.
- [x] Implement `src/cloak/lattice_producer/graph.py` with `StateGraph`, named nodes, conditional edges,
      deterministic count compilation, incremental proposed-artifact persistence, and review interrupt.
- [x] Add `scripts/run_lattice_producer.py`.
- [x] Run unit tests.
- [x] Run a deterministic offline-only smoke with a tiny fixture queue.
- [ ] Run a live llama-swap smoke only if the local endpoint is already up and no other GPU job is active.
- [ ] Record exact commands and outputs in `EXPERIMENT_LOG.md`.

## Verification Commands

Unit tests:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  src/cloak/tests/test_lattice_producer_queue.py \
  src/cloak/tests/test_lattice_producer_gates.py \
  src/cloak/tests/test_lattice_producer_merge.py \
  src/cloak/tests/test_lattice_producer_graph.py -q
```

Existing profile regression tests:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  src/cloak/tests/test_lattice_profiles.py \
  src/cloak/tests/test_lattice_profile_builders.py -q
```

Offline smoke:

```bash
PYTHONPATH=src .venv/bin/python -u scripts/run_lattice_producer.py \
  --run-dir data/lattice_producer/runs/offline-smoke \
  --queue tests/fixtures/lattice_producer_queue.jsonl \
  --profiles data/lattice_profiles/fine_lattice_profiles.json \
  --out data/lattice_profiles/proposed/fine_lattice_profiles.offline-smoke.json \
  --offline-only \
  --max-items 5
```

Live local-model smoke:

```bash
INFERDPT_LLM_CACHE=data/llm_cache \
PYTHONPATH=src \
.venv/bin/python -u scripts/run_lattice_producer.py \
  --run-dir data/lattice_producer/runs/live-smoke \
  --queue tests/fixtures/lattice_producer_queue.jsonl \
  --profiles data/lattice_profiles/fine_lattice_profiles.json \
  --out data/lattice_profiles/proposed/fine_lattice_profiles.live-smoke.json \
  --base-url http://localhost:8060/v1 \
  --model "gemma 4 (E4B)" \
  --max-items 5 \
  --workers 1
```

Do not claim completion unless every command run is reported with its exact command line.

## Failure Handling

- LangGraph checkpoint unavailable: fail before processing queue items.
- llama-swap unavailable: record run failure unless `--offline-only` is set.
- Invalid model JSON: log proposal rejection; escalate if configured; otherwise continue.
- Validation failure: do not emit canonical artifact; leave proposed artifact and logs for inspection.
- Review rejection: leave proposed artifact staged; final status is rejected.
- Generated-universe category produces unsafe or direct examples: reject the entries, preserve raw proposal in
  `proposals.jsonl`, and continue.
- Process crash: resume with the same `run_id`/thread ID and continue from the latest checkpoint; JSONL writes
  remain idempotent by item ID.
- Process crash after an accepted item: the proposed artifact must already contain that item, and resume must
  not require replaying the model call to recover it.

## Open Questions

1. Which persistent checkpointer package/version should be used locally: SQLite for single-machine runs or
   Postgres if later used as a service.
2. Whether `approve` should ever overwrite canonical profiles automatically, or whether this repo should
   always require a separate explicit promotion command.
3. Whether the live teacher prompt should use only `gemma 4 (E4B)` plus deterministic gates, or also allow
   Qwen escalation for malformed/empty first-pass proposals.
4. Whether generated-universe rows should ever be promoted to runtime-certifying sources, or should remain
   review-only scaffolding until replaced by an external dataset.

## Sources

- Runtime contract: [lattice-substitutor.md](../specs/lattice-substitutor.md).
- Profile schema: [generalization-lattice-cache.md](../specs/generalization-lattice-cache.md).
- Per-level count policy: [offline-k-anonimity-risk-walk.md](../specs/offline-k-anonimity-risk-walk.md).
- Count-policy handoff: `/tmp/agent-cloak-generative-lattice-producer-handoff.md`.
- Fine detector record: [2026-07-05-FT-detector-v7-dem-decompose.md](../../research-wiki/training/2026-07-05-FT-detector-v7-dem-decompose.md).
- Knowledgator GLiNER PII model card: [Hugging Face](https://huggingface.co/knowledgator/gliner-pii-base-v1.0).
- LangGraph overview: [LangChain docs](https://docs.langchain.com/oss/python/langgraph/overview).
- LangGraph persistence: [LangChain docs](https://docs.langchain.com/oss/python/langgraph/persistence).
- LangGraph checkpointers: [LangChain docs](https://docs.langchain.com/oss/python/langgraph/checkpointers).
- LangGraph interrupts: [LangChain docs](https://docs.langchain.com/oss/python/langgraph/interrupts).
- LangGraph Graph API idempotency: [LangChain docs](https://docs.langchain.com/oss/python/langgraph/graph-api).
