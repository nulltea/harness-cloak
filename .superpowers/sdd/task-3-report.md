# Task 3 Report — QA Gate, Cache, and Smoke

## Status

Completed on `HEAD` with `test(qa): gate and smoke utility artifacts`.

## Result

- Enforced exact artifact/environment hash identity and rejected `unsupported`, `build_failed`,
  and zero-accepted-assertion documents before training.
- Added a deterministic QA rollout cache identity covering document ID, complete action vector,
  `doc_p`, `out_final`, artifact hash, and scorer pin.
- Added a QA-only preflight report for accepted-family counts, missing families, uncovered
  decisions, and base versus one-decision-counterfactual call surfaces. It performs no calls and
  does not duplicate ranker-v2 routing, loss, or scheduler gates.
- Corrected the implementation plan's training-record version from ranker v2 to ranker v4.

## TDD Evidence

The new tests were written before production changes. Red phase:

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_train_roundtrip_mode.py -k 'subset_environment_hash_mismatch or unmeasured_documents or without_accepted_assertions or rollout_cache_identity' src/cloak/tests/test_build_qa_utility_artifact_cli.py
```

```text
5 failed, 41 deselected in 0.83s
```

The CLI preflight test then failed separately with the expected missing helper:

```text
AttributeError: module 'train_ranker' has no attribute 'qa_utility_preflight_report'
1 failed in 1.75s
```

Green focused check:

```bash
PYTHONPATH=src:scripts .venv/bin/pytest -q src/cloak/tests/test_train_roundtrip_mode.py -k 'subset_environment_hash_mismatch or unmeasured_documents or without_accepted_assertions or rollout_cache_identity' src/cloak/tests/test_build_qa_utility_artifact_cli.py
```

```text
5 passed, 41 deselected in 0.78s
```

## Verification

Temporary symlinks to `/home/timo/repos/agent-cloak/.venv` and the read-only clinical corpus
were removed after each command. No external or paid calls were made.

```bash
PYTHONPATH=src:scripts .venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_build_qa_utility_artifact_cli.py src/cloak/tests/test_utility_credit.py src/cloak/tests/test_train_roundtrip_mode.py src/cloak/tests/test_roundtrip.py
.venv/bin/python -m py_compile scripts/train_ranker.py scripts/build_qa_utility_artifact.py
git diff --check
```

```text
95 passed in 2.66s
```

The deterministic smoke command and measured output are recorded in
`research-wiki/training/2026-07-13-RL-ranker-v4-qa-utility-smoke.md`.

## Commit

`HEAD` — `test(qa): gate and smoke utility artifacts`.

## Concerns

- The smoke artifact is partial: 13 delivered assertions, zero context assertions, and one
  uncovered decision. It is not evidence of RL, privacy, or empirical utility success.
- Exact environment identity intentionally rejects a subset smoke artifact for a full training
  environment; a full run must rebuild its QA artifact under the exact frozen environment.
- The preflight exposes per-rollout/per-selected-pair call surfaces only. The ranker-v2 scheduler
  still owns total counterfactual allocation and cache-hit measurements.

---

## Review Fixes — Reward Cache and Coverage Gate

### Result

- Added the single artifact-backed QA reward cache requested by the specification. It loads one
  content-addressed JSON file per training process, validates stored full-result identities, and
  atomically persists base-rollout misses.
- The key covers document ID, the complete stable decision/action vector, `doc_p`, artifact hash,
  scorer/reader/remote/extractor pins, and `out_final` when the result is stored and revalidated.
  Exact duplicate rollouts in one batch are coalesced before `roundtrip_batch` and count as cache
  hits; only the base artifact loop uses this cache. No counterfactual cache-hit claim is made.
- Artifact gates now compare document ID sets exactly against the frozen training subset; missing,
  empty, or extra coverage fails. Attachment also fails fast on missing, unsupported, or
  build-failed document state.
- Artifact-mode CLI runs require `--utility-reward-cache PATH`; legacy no-artifact behavior is
  unchanged. CLI preflight testing now parses the emitted JSON rather than recomputing it.

### TDD Evidence

Red command:

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_train_roundtrip_mode.py -k 'document_coverage or missing_document_state or unusable_document_state or utility_reward_cache or identical_artifact_rollout_cache'
```

```text
9 failed, 45 deselected in 1.60s
```

The first green pass exposed duplicate misses inside a single rollout batch (`batch_sizes == [2]`
instead of `[1]`); the cache now coalesces in-flight identities before dispatch. Green command:

```text
9 passed, 45 deselected in 1.52s
```

### Verification

```bash
PYTHONPATH=src:scripts .venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_build_qa_utility_artifact_cli.py src/cloak/tests/test_utility_credit.py src/cloak/tests/test_train_roundtrip_mode.py src/cloak/tests/test_roundtrip.py
.venv/bin/python -m py_compile scripts/train_ranker.py scripts/build_qa_utility_artifact.py
git diff --check
```

```text
104 passed in 2.64s
```

### Commit

`HEAD` — `fix(qa): wire reward cache and coverage gate`.

### Concerns

- The cache stores completed base artifact rollouts only. Ranker-v2 still owns future
  counterfactual scheduling and no counterfactual measurement was added here.
- The `aci/D2N002` smoke remains partial (no context assertions); its command and measured output
  did not change, so its training record was intentionally not rewritten.
