# Task 3 report — semantic log-count head pretraining machinery

## Status

Implementation complete and stub-only verified. No real pretraining, encoder loading, GPU use,
network access, or commit occurred. Synthetic smoke outputs validate orchestration and artifact
contracts only; they are not experimental findings or a promotion result.

## Loss and normalization semantics

- `SemanticPrivacyHead` owns a separate privacy projection and two-output MLP.
- Predicted means use `softplus`; predicted scales use `1e-4 + softplus`.
- Training consumes admitted level rows only. The loss returns `nll`, `pairwise_rank`,
  `profile_huber`, and `total`; tied target pairs are excluded from ranking.
- Profile-score supervision normalizes predicted level means over each complete decision menu.
  Singleton levels receive `1.0`; multi-level scores use the maximum predicted level mean and are
  clipped to `[0, 1]`.
- Controller-facing normalization requires one KEEP and one placeholder, then overwrites their
  scores to exact `0` and `1` respectively.
- The optional count-basis/source-family input is a frozen train-split categorical vocabulary with
  an explicit unknown category. It is concatenated only inside the privacy branch.
- Gradient tests establish that privacy loss reaches the privacy projection/head while an unrelated
  utility projection receives no gradient.

## Split, baseline, and metric design

- Splits are deterministic, seeded, content-addressed, stratified by runtime type, and grouped by
  complete `profile_id`. Every observed runtime type must have enough profiles to appear in
  train/dev/test; otherwise split construction fails.
- Neural comparisons use the same projection/head dimensions for the semantic model,
  authored-position+mode/type MLP, mode/type-only MLP, and candidate-only frozen-vector baseline.
- The train-profile mean is stable and type-conditioned, with a global fallback; it stores no
  held-out profile identity table.
- Held-out metrics contain NLL, median absolute log error, median multiplicative error, 95% interval
  coverage, within-menu unequal-pair accuracy, per-menu Spearman, profile-relative calibration
  error, and selected-level profile-relative regret.
- Metrics are emitted overall and by runtime type, grounding status, and source family. KEEP and
  placeholder are excluded from learned metrics.
- The frozen diagnostic manifest requires profile-held-out dev/test reports and every baseline.
  Unsupported comparison metrics produce a fail verdict. It explicitly declines ACI/document
  generalization claims.

## Checkpoint and CLI contract

- Checkpoint version: `ranker-v2-semantic-privacy-v1`.
- The contract binds environment, profile-target artifact, representation-manifest file, encoder
  revision, split-manifest, model dimensions, optional categorical vocabulary, loss weights, all
  three seeds, the checkpoint's training seed, and aggregate metric-report hash.
- Save rejects model/contract dimension disagreement. Load validates version, full contract, model
  dimensions, state keys, tensor shapes, and dtypes before mutating the supplied model.
- The CLI requires three distinct explicit seeds, supports `--max-steps`, loads frozen
  representations without loading an encoder, and defaults to
  `results/ranker_v2/architecture/privacy/`.
- The inspected synthetic CLI output contains a split manifest, aggregate and per-seed metric
  reports, three checkpoints, and a diagnostic manifest with all required baseline families.

## TDD and verification

Initial command:

```text
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
  src/cloak/tests/test_ranker_privacy.py
```

Initial result: collection failed because `cloak.train.ranker_privacy` did not exist.

Final command:

```text
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
  src/cloak/tests/test_ranker_privacy.py \
  src/cloak/tests/test_ranker_diagnostics.py
```

Output tail:

```text
...............................                                          [100%]
31 passed, 2 warnings in 52.12s
```

The warnings are pre-existing SWIG import deprecations. Python bytecode compilation,
`git diff --check`, synthetic artifact inspection, and no-GPU/no-network source audits also pass.

## Concerns

- Real profile-held-out quality, runtime-type behavior, grounding-stratum behavior, calibration,
  baseline comparisons, and relative promotion remain unvalidated until Task 9 runs the frozen
  real-data slice.
- The CLI deliberately performs CPU training only in this task's synthetic smoke. Full-scale memory,
  runtime, and optimizer behavior are not claimed here.
