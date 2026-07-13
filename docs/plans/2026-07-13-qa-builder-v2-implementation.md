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

**Delivered scope:** QA artifact construction/scoring/gates, provisional structured credit,
complete-result caching, and the tested-pair substitution hook. No artifact-mode counterfactual
scheduler/executor, lambda conditioning or selection, or count-objective training is delivered by
this plan's implementation commits.

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
- Create: `research-wiki/training/2026-07-13-RL-ranker-v4-qa-utility-smoke.md`

**Interfaces:**
- Produces: artifact/environment identity gate and QA-specific preflight report.
- Produces: deterministic document/action-vector cache identity.

- [ ] **Step 1: Add failing gate and cache tests**

Cover pin mismatches, dangling assertions, invalid denominator, unsupported/build-failed documents, zero accepted assertions, cache-key action-vector sensitivity, and no legacy probe-count filtering.

- [ ] **Step 2: Implement the minimal gates**

Reference ranker-v2 for shared routing/loss gates; keep only QA-specific artifact checks locally. Report the planned base versus counterfactual call surface without launching remote calls.

- [ ] **Step 3: Run the complete model-free suite**

Run: `PYTHONPATH=src:scripts .venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_build_qa_utility_artifact_cli.py src/cloak/tests/test_utility_credit.py src/cloak/tests/test_train_roundtrip_mode.py src/cloak/tests/test_roundtrip.py`
Expected: all tests pass.

- [ ] **Step 4: Run a tiny deterministic artifact smoke**

Run the CLI on `aci/D2N002` with teacher escalation disabled and a checked-in or temporary manifest. Record artifact counts, missing families, uncovered decisions, runtime, and exact command in the training record. Do not make external calls.

- [ ] **Step 5: Commit the reviewed slice**

Commit task files with message `test(qa): gate and smoke utility artifacts`.
