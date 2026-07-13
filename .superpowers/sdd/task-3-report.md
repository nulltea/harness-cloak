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
