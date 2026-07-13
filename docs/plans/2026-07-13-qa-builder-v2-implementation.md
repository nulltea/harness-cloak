---
type: plan
status: current
created: 2026-07-13
updated: 2026-07-13
tags: [qa, rl, implementation, utility-components, ranker-v2]
companion: [docs/specs/qa-builder-v2.md,
            docs/specs/RL/qa-builder-v2-decision-log.md,
            docs/specs/RL/interactive-ranker-v2.md]
---

# QA Builder v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the frozen QA-builder v2 artifact, batched two-channel scoring, and ranker-v2 structured utility-credit seam without external model calls.

**Architecture:** `cloak.train.qa_builder` owns frozen identities, deterministic assertion compilation, optional relation proposals, validation, weighting, and runtime component scoring. `roundtrip.py` returns the stable component vector. `train_ranker.py` consumes that vector through linked/global/fallback routing; it does not collapse v2 artifacts into legacy scalar trajectory credit.

**Tech Stack:** Python 3, PyTorch, pytest, existing `cloak` ranker/reward/round-trip modules.

## Global Constraints

- QA and RL consume the same frozen occurrence/decision artifact; QA never detects spans.
- Context assertions score `doc_p`; delivered assertions score `out_final`.
- Every rewarded context assertion passes `doc_orig`, one pinned joint legal generalization anchor, and fails the all-placeholder anchor.
- Family weights use a fixed denominator; missing families do not renormalize surviving weights.
- Occurrence links are routing hints, not causal labels.
- Ranker v2 exclusively owns linked/global/fallback advantages and tested-pair substitution.
- Missing linked QA never removes a decision or implies zero utility relevance.
- Runtime context assertions use one batched reader request per rollout.
- Optional teacher escalation is at most one cached call per under-supported document; no paid or external call is run without explicit approval.
- No new production dependency.

---

### Task 1: Adopt and Harden the Existing QA Draft

**Files:**
- Modify: `src/cloak/train/qa_builder.py`
- Create: `scripts/build_qa_utility_artifact.py`
- Modify: `src/cloak/train/roundtrip.py`
- Modify: `scripts/train_ranker.py`
- Test: `src/cloak/tests/test_qa_builder_v2.py`
- Test: `src/cloak/tests/test_build_qa_utility_artifact_cli.py`
- Test: `src/cloak/tests/test_train_roundtrip_mode.py`
- Modify: `docs/specs/qa-builder-v2.md`
- Modify: `docs/specs/RL/qa-builder-v2-decision-log.md`

**Interfaces:**
- Produces: `freeze_ranker_environment(...) -> dict`
- Produces: `package_utility_artifact(..., family_budgets, pins) -> dict`
- Produces: `assign_static_weights(assertions, family_budgets) -> (assertions, document_state)`
- Produces: `build_utility_artifact(...) -> dict`
- Produces: `score_utility(artifact, doc_id, doc_p, out_final, reader) -> dict`
- Produces from `roundtrip_batch`: stable `component_scores` for v2 jobs.

- [ ] **Step 1: Audit the uncommitted draft against the normative spec**

Treat all current uncommitted QA-builder files as a draft, not accepted implementation. Verify repeated occurrences, stable IDs, scope/link integrity, fixed missing-family denominator, authoritative truth, joint anchors, optional one-call teacher escalation, and one-batch runtime context scoring.

- [ ] **Step 2: Add or correct tests for every uncovered requirement**

Tests must cover invalid links/scopes, duplicate IDs, teacher abstention, protected locator leakage, joint multi-decision anchors, original/generalized/placeholder validation, fixed denominator, deterministic delivered scoring, and round-trip vector output.

- [ ] **Step 3: Complete the minimal deep module and CLI**

The artifact must contain `artifact_version`, `artifact_hash`, `environment_hash`, pins, document measurement state, assertion IDs, uncovered decision IDs, fixed `utility_weight_denominator`, and stable assertion rows. The default CLI path must perform zero external calls. Preserve legacy round-trip behavior when no v2 artifact is supplied.

- [ ] **Step 4: Run the complete draft-slice tests**

Run: `PYTHONPATH=src:scripts .venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_build_qa_utility_artifact_cli.py src/cloak/tests/test_train_roundtrip_mode.py src/cloak/tests/test_roundtrip.py`
Expected: all tests pass.

- [ ] **Step 5: Commit the reviewed slice**

Commit task files with message `feat(qa): add frozen two-channel utility artifacts`.

### Task 2: Route Structured Utility Credit in Ranker Training

**Files:**
- Modify: `scripts/train_ranker.py`
- Modify or create: `src/cloak/train/utility_credit.py`
- Test: `src/cloak/tests/test_train_roundtrip_mode.py`
- Test or create: `src/cloak/tests/test_utility_credit.py`

**Interfaces:**
- Consumes: per-rollout `{assertion_id: score}` vectors and frozen artifact links/weights.
- Produces: `provisional_advantages(component_vectors, artifact, occurrence_to_decision) -> dict[(rollout, decision), float]`
- Produces: linked `A_link + A_global`, uncovered `A_document`, and no duplicate complete-document term.

- [ ] **Step 1: Write synthetic failing credit tests**

Cover linked-only routing, global routing to every decision, multi-decision links, repeated occurrences counted once, uncovered fallback, fixed denominator, tied components, and no linked/document double counting.

- [ ] **Step 2: Implement a deep utility-credit function**

Keep weighting and RLOO calculations in one module. Return one provisional advantage per rollout-decision pair. Do not apply QA coverage as a reward multiplier.

- [ ] **Step 3: Integrate the v2 path into `train_roundtrip`**

When a utility artifact is present, retain per-decision policy log-probabilities and multiply them by their structured provisional advantages. Preserve the legacy scalar path only when no v2 artifact is supplied.

- [ ] **Step 4: Verify counterfactual substitution**

For tested pairs, replace the provisional term in-place with the existing pairwise term using the same `1/G` document-group normalization; never append a separately averaged counterfactual loss.

- [ ] **Step 5: Run focused training tests**

Run: `PYTHONPATH=src:scripts .venv/bin/pytest -q src/cloak/tests/test_utility_credit.py src/cloak/tests/test_train_roundtrip_mode.py`
Expected: all tests pass.

- [ ] **Step 6: Commit the reviewed slice**

Commit task files with message `feat(rl): route QA utility credit by decision`.

### Task 3: Gate, Cache, and Smoke the Integrated Path

**Files:**
- Modify: `scripts/train_ranker.py`
- Modify: `scripts/build_qa_utility_artifact.py`
- Test: `src/cloak/tests/test_train_roundtrip_mode.py`
- Test: `src/cloak/tests/test_build_qa_utility_artifact_cli.py`
- Create: `research-wiki/training/2026-07-13-RL-ranker-v2-qa-utility-smoke.md`

**Interfaces:**
- Produces: artifact/environment identity gate and QA-specific preflight report.
- Produces: deterministic document/action-vector cache identity.

- [ ] **Step 1: Add failing gate and cache tests**

Cover pin mismatches, dangling assertions, invalid denominator, unsupported/build-failed documents, zero accepted assertions, cache-key action-vector sensitivity, and no legacy probe-count filtering.

- [ ] **Step 2: Implement the minimal gates**

Reference ranker-v2 for shared routing/loss gates; keep only QA-specific artifact checks locally. Report base versus counterfactual call budgets without launching remote calls.

- [ ] **Step 3: Run the complete model-free suite**

Run: `PYTHONPATH=src:scripts .venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_build_qa_utility_artifact_cli.py src/cloak/tests/test_utility_credit.py src/cloak/tests/test_train_roundtrip_mode.py src/cloak/tests/test_roundtrip.py`
Expected: all tests pass.

- [ ] **Step 4: Run a tiny deterministic artifact smoke**

Run the CLI on `aci/D2N002` with teacher escalation disabled and a checked-in or temporary manifest. Record artifact counts, missing families, uncovered decisions, runtime, and exact command in the training record. Do not make external calls.

- [ ] **Step 5: Commit the reviewed slice**

Commit task files with message `test(qa): gate and smoke utility artifacts`.

### Task 4: Compile Authoritative Delivered Assertions

**Files:**
- Modify: `src/cloak/train/qa_builder.py`
- Test: `src/cloak/tests/test_qa_builder_v2.py`

**Interfaces:**
- Produces: `AciTaskAdapter.delivered_candidates(doc_id, document, reference, environment_document) -> list[dict]`
- Produces scoring contracts for `required_sections`, `field_value`, `contains`, and `exact_relation`.
- Extends: `score_utility(...)` with deterministic evaluation of every delivered contract.

- [ ] **Step 1: Add failing delivered-component tests**

Use a compact ACI note fixture containing required headings, demographic fields, assessment/status rows, and a condition-treatment-test relation. Assert that the compiler emits separate `content`, `field`, `structure`, and `exact_relation` groups; structural assertions must have a capped share of the delivered budget and must not aggregate semantic fields a second time.

- [ ] **Step 2: Implement a deterministic ACI note parser**

Parse uppercase section headings and assessment-plan rows from the authoritative human reference. Emit only facts supported by the source/reference/task schema. Treat the reference as truth for delivered assertions; never use a ceiling output as truth. Abstain on ambiguous rows rather than inventing clinical normalization.

- [ ] **Step 3: Implement deterministic delivered scorers**

Score required-section presence, parseability, field/value agreement, explicit omission/content checks, and exact symbolic relations on `out_final`. Keep each fact/relation in one group so schema aggregates cannot double-count it. Structural compliance is low-weight through the frozen structural cap.

- [ ] **Step 4: Run delivered-component tests**

Run: `PYTHONPATH=src:scripts .venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py -k 'delivered or schema or exact_relation'`
Expected: all selected tests pass.

- [ ] **Step 5: Commit the reviewed slice**

Commit task files with message `feat(qa): compile deterministic ACI utility assertions`.

### Task 5: Compile and Validate Context Probes

**Files:**
- Modify: `src/cloak/train/qa_builder.py`
- Test: `src/cloak/tests/test_qa_builder_v2.py`

**Interfaces:**
- Produces: `AciTaskAdapter.semantic_property_candidates(doc_id, document, environment_document) -> list[dict]`
- Extends: the single batched teacher proposal schema with `contextual_relation` candidates only.
- Produces: complete accepted/rejected candidate records with stable rejection IDs and evidence.

- [ ] **Step 1: Add failing semantic-property tests**

For every controlled decision with at least one legal non-KEEP, non-placeholder action, assert that deterministic candidates expose the legal property support band, link all repeated occurrences, and pin one joint representative anchor. Do not require one probe per rung; deduplicate equivalent property levels.

- [ ] **Step 2: Implement template-first semantic-property candidates**

Derive category/function questions from the frozen runtime type and legal lattice action semantics. Questions must test meaning unavailable from `<TYPE_N>` without containing the protected surface or accepted answer. If no safe task-native template can identify the role from surviving context, record an explicit `not_generated` rejection rather than fabricating a probe.

- [ ] **Step 3: Harden teacher contextual relations**

Keep at most one cached batched Nemotron proposal call per under-supported document. Compile only exact-evidence, legal-property relations. Preserve each rejected proposal with reason, proposal hash, and non-sensitive evidence metadata; aggregate summaries remain derived.

- [ ] **Step 4: Validate both context subtypes through the three-point gate**

Every rewarded context assertion must pass `doc_orig`, pass its joint coarsest legal representative-generalization anchor, fail the all-placeholder anchor, and satisfy the frozen stability gate. Store the full validation evidence and anchor vector/hash.

- [ ] **Step 5: Run context-component tests**

Run: `PYTHONPATH=src:scripts .venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py -k 'semantic_property or contextual_relation or context or rejection or anchor'`
Expected: all selected tests pass.

- [ ] **Step 6: Commit the reviewed slice**

Commit task files with message `feat(qa): compile context-preservation probes`.

### Task 6: Export Inspectable Artifacts and D2N002 Acceptance

**Files:**
- Modify: `scripts/build_qa_utility_artifact.py`
- Modify: `src/cloak/train/qa_builder.py`
- Modify: `src/cloak/tests/test_build_qa_utility_artifact_cli.py`
- Modify: `src/cloak/tests/test_qa_builder_v2.py`

**Interfaces:**
- Produces: one normative JSON artifact plus derived `<stem>.assertions.json` and `<stem>.qa-pairs.json` views.
- Produces: `artifact_views(artifact) -> tuple[dict, dict]` without a second source of truth.

- [ ] **Step 1: Add failing derived-view tests**

Assert that the assertions view groups structural, field/content, exact-relation, and contextual records with evidence and scoring contracts. Assert that the QA-pairs view groups semantic-property and contextual-relation questions by decision, exposing legal action/property support, occurrence links, accepted values, validation evidence, and rejection states.

- [ ] **Step 2: Implement deterministic view projection**

Generate both views solely from the normative artifact. Preserve stable component, occurrence, decision, group, and action IDs. Never maintain separate caches or regenerate questions while exporting.

- [ ] **Step 3: Add a substantive D2N002 acceptance test**

Using corrected frozen detector fixtures and stubbed teacher/reader responses, require nonzero delivered structure/field/exact-relation assertions, nonzero accepted semantic-property and contextual-relation assertions, complete rejection rows, both family budgets present, and both derived files. The test must fail if the artifact regresses to surface-containment-only output.

- [ ] **Step 4: Run the complete QA-builder suite**

Run: `PYTHONPATH=src:scripts .venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_build_qa_utility_artifact_cli.py src/cloak/tests/test_train_roundtrip_mode.py src/cloak/tests/test_roundtrip.py`
Expected: all tests pass.

- [ ] **Step 5: Run a local D2N002 smoke without external calls**

Build from `/tmp/ranker_env_qa_v2_d2n002.json` and `/tmp/task_arms_qa_v2_d2n002.json` with a deterministic stub/cached fixture. Inspect assertion subtype counts and both derived outputs. Do not call OpenRouter or any paid/external model.

- [ ] **Step 6: Commit the reviewed slice**

Commit task files with message `feat(qa): export complete QA artifacts`.
