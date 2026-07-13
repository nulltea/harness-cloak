# QA-builder v2 final-review fix report

## Commit

- Implementation commit: `8552eb02f3e005b1d0959d9ba3a19d5a61c2dab3`

## Changed files

- `src/cloak/train/qa_builder.py`
- `scripts/build_qa_utility_artifact.py`
- `scripts/train_ranker.py`
- `src/cloak/tests/test_qa_builder_v2.py`
- `src/cloak/tests/test_build_qa_utility_artifact_cli.py`
- `src/cloak/tests/test_train_roundtrip_mode.py`
- `research-wiki/training/2026-07-13-RL-ranker-v4-qa-utility-smoke.md`
- `.superpowers/sdd/final-review-fix-report.md`

## Implemented fixes

- Runtime context scoring now uses the canonical permutation-0 option renderer used by validation.
- Repeated reader-stability trials request independent reads with `refresh=True`, with compatibility fallback for simple injected readers.
- Frozen action legality and environment identity now bind to effective per-type count floors, including CLI overrides and `OTHER` fallback.
- Artifact pins are authoritative and transitive across task, builder, teacher state, reader, scorer, and the canonical normalized threshold manifest; training requires and checks an explicit expected manifest hash.
- Relational leakage lint covers partial meaningful tokens, answer aliases, every frozen protected occurrence surface/alias, questions, and options without one-character matching.
- Utility reward caching is append-only fail-closed JSONL with complete-result validation, new-record-only writes, and conflicting-duplicate/truncation detection.
- Optional relation-teacher prompts include authoritative reference evidence while compilation still requires exact source evidence.
- The smoke record now states that runtime scoring and cache paths were not exercised and that zero context assertions are not utility/privacy evidence.

The append-only JSONL design was chosen over SQLite to preserve the existing content-addressed in-memory interface while removing whole-history rewrites with no migration layer or dependency. Pin and floor rules are centralized in shared helpers so builder and training cannot silently diverge.

## RED evidence

The worktree has no local `.venv`, so the first literal verification attempt failed before collection:

```text
PYTHONPATH=src:scripts .venv/bin/pytest ...
/bin/bash: .venv/bin/pytest: No such file or directory
```

All TDD runs below therefore used the repository host environment at `/home/timo/repos/agent-cloak/.venv/bin/pytest`.

1. Runtime rendering and independent refresh:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py -k 'context_validation_refreshes_repeated_reader_trials or runtime_renders_option_questions_like_validation_permutation_zero'
2 failed, 45 deselected
```

The refresh sequence contained only ordinary calls, and runtime questions omitted validated options.

2. Builder floor legality and authoritative pins:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py -k 'freeze_binds_action_legality_and_identity_to_effective_floors or builder_rejects_under_floor_representative_action or builder_emits_authoritative_transitive_pins_and_manifest_identity'
3 failed, 47 deselected
```

`freeze_ranker_environment` did not accept floor overrides, under-floor anchors reached reader validation, and caller-forged pins survived packaging.

3. Builder CLI floor override:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_build_qa_utility_artifact_cli.py -k 'build_qa_utility_artifact_cli or floor_override_changes_frozen_legality_and_identity'
2 failed, 6 passed
```

The relevant product failure was an unrecognized `--floors` argument; the other failure exposed the test subprocess's invalid assumption that this worktree contains `.venv`, which was corrected by using the host environment and explicit worktree `PYTHONPATH`.

4. Training expected-manifest and floor gate:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_train_roundtrip_mode.py -k 'requires_expected_manifest_hash_before_training_initialization or recomputes_manifest_hash_and_checks_expected_identity or training_gate_rejects_under_floor_representative_action_from_frozen_environment'
3 failed, 89 deselected
```

Training did not require the expected hash, the gate had no expected-hash parameter, and under-floor representative anchors were not rejected.

5. Leakage lint:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py -k 'partial_answer_clue or frozen_answer_alias or unlinked_protected_surface_or_alias or lints_protected_leakage_in_options or preserves_uncontrolled_frozen_occurrence'
5 failed, 49 deselected
```

Every leaking proposal was accepted, and frozen occurrence aliases were not preserved.

6. Append-only cache:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_train_roundtrip_mode.py -k 'appends_one_jsonl_record or second_write_excludes_prior_history or rejects_truncated_jsonl_record or rejects_conflicting_duplicate_identity'
4 failed, 92 deselected
```

The cache still rewrote one whole JSON object, had no new-batch persistence interface, and did not provide specific truncation/conflicting-duplicate handling.

7. Authoritative reference prompt:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py::test_relation_prompt_exposes_only_closed_ids_properties_and_source
1 failed
```

`relation_teacher_prompt` rejected the new `authoritative_reference` argument.

## GREEN evidence

Required focused suite:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_build_qa_utility_artifact_cli.py src/cloak/tests/test_train_roundtrip_mode.py src/cloak/tests/test_utility_credit.py src/cloak/tests/test_roundtrip.py
171 passed in 2.82s
```

Changed Python files compiled successfully:

```text
/home/timo/repos/agent-cloak/.venv/bin/python -m py_compile scripts/build_qa_utility_artifact.py scripts/train_ranker.py src/cloak/train/qa_builder.py src/cloak/tests/test_build_qa_utility_artifact_cli.py src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_train_roundtrip_mode.py
exit 0, no output
```

Patch hygiene:

```text
git diff --check
exit 0, no output
```



## Concerns

- No external relation-teacher run was authorized or executed. The context channel remains empirically undemonstrated, and the smoke still has zero context assertions.
- Existing version-2 whole-file reward caches are intentionally rejected by the fail-closed version-3 loader and must be rebuilt as JSONL.
- Verification used the host `.venv` because this worktree has no `.venv` entry.

---

## Final-review follow-up fix wave

### Commit

- Implementation commit: `91f9e6f6b96d37b9b0bb5fa233ef69f09d7501ad`

### Changed files

- `src/cloak/train/qa_builder.py`
- `src/cloak/train/utility_credit.py`
- `src/cloak/train/roundtrip.py`
- `src/cloak/tests/test_qa_builder_v2.py`
- `src/cloak/tests/test_utility_credit.py`
- `src/cloak/tests/test_roundtrip.py`
- `.superpowers/sdd/final-review-fix-report.md`

### RED evidence

1. Validation/runtime option ordering after packaging:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py::test_packaged_runtime_option_order_matches_prepackaging_validation
1 failed in 0.07s
```

Validation rendered `musculoskeletal | respiratory | endocrine`; after final ID assignment runtime rendered `endocrine | respiratory | musculoskeletal`.

2. Exact short protected phrase leakage:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py::test_relational_compiler_rejects_exact_short_protected_phrase
1 failed in 0.04s
```

The compiler accepted a question containing the exact protected phrase `HIV`.

3. Builder/ranker weighted-aggregation ownership:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py::test_delivered_only_scoring_is_deterministic_without_reader_call src/cloak/tests/test_utility_credit.py::test_document_utility_owns_weighted_component_aggregation src/cloak/tests/test_utility_credit.py::test_document_utility_rejects_missing_component_scores src/cloak/tests/test_roundtrip.py::test_utility_artifact_roundtrip_aggregates_builder_components
4 failed in 0.06s
```

`score_utility` still returned `utility`, `utility_credit` had no document aggregator, and roundtrip required the builder-owned scalar.

4. Strict reader count types:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py::test_threshold_manifest_requires_integer_reader_counts
6 failed in 0.08s
```

Both reader count fields silently accepted `True`, `1.0`, and `"1"`.

### GREEN evidence

All new and directly updated regressions:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py::test_packaged_runtime_option_order_matches_prepackaging_validation src/cloak/tests/test_qa_builder_v2.py::test_relational_compiler_rejects_exact_short_protected_phrase src/cloak/tests/test_qa_builder_v2.py::test_threshold_manifest_requires_integer_reader_counts src/cloak/tests/test_qa_builder_v2.py::test_delivered_only_scoring_is_deterministic_without_reader_call src/cloak/tests/test_utility_credit.py::test_document_utility_owns_weighted_component_aggregation src/cloak/tests/test_utility_credit.py::test_document_utility_rejects_missing_component_scores src/cloak/tests/test_roundtrip.py::test_utility_artifact_roundtrip_aggregates_builder_components src/cloak/tests/test_qa_builder_v2.py::test_roundtrip_utility_artifact_scores_doc_p_and_out_final
13 passed in 0.06s
```

Required focused five-file suite:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_build_qa_utility_artifact_cli.py src/cloak/tests/test_train_roundtrip_mode.py src/cloak/tests/test_utility_credit.py src/cloak/tests/test_roundtrip.py
182 passed in 2.78s
```

Changed Python files compiled successfully:

```text
/home/timo/repos/agent-cloak/.venv/bin/python -m py_compile src/cloak/train/qa_builder.py src/cloak/train/utility_credit.py src/cloak/train/roundtrip.py src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_utility_credit.py src/cloak/tests/test_roundtrip.py
exit 0, no output
```

Patch hygiene:

```text
git diff --check
exit 0, no output
```

### Concerns

- No new concern from this follow-up wave. The prior no-teacher-run and cache-v2 migration concerns remain unchanged.

---

## Semantic pin bump follow-up

### Commit

- Implementation commit: `55949362da5e812dd3e1cb3e3ac039d4200a40cb`

### Changed files

- `src/cloak/train/qa_builder.py`
- `src/cloak/tests/test_train_roundtrip_mode.py`
- `.superpowers/sdd/final-review-fix-report.md`

### RED evidence

Previous builder and runtime-scorer pins were still accepted:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_train_roundtrip_mode.py::test_utility_artifact_gate_rejects_previous_semantic_pins
2 failed in 0.85s
```

Both parameterized cases failed with `DID NOT RAISE SystemExit`: `assertion-compiler-v2` matched the live builder pin, and `qa-utility-scorer-v1` matched the live scorer pin.

### GREEN evidence

Pin regression after bumping to `assertion-compiler-v3` and `qa-utility-scorer-v2`:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_train_roundtrip_mode.py::test_utility_artifact_gate_rejects_previous_semantic_pins
2 passed in 0.81s
```

Required focused five-file suite:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_build_qa_utility_artifact_cli.py src/cloak/tests/test_train_roundtrip_mode.py src/cloak/tests/test_utility_credit.py src/cloak/tests/test_roundtrip.py
184 passed in 3.02s
```

Changed Python files compiled successfully:

```text
/home/timo/repos/agent-cloak/.venv/bin/python -m py_compile src/cloak/train/qa_builder.py src/cloak/tests/test_train_roundtrip_mode.py
exit 0, no output
```

Patch hygiene:

```text
git diff --check
exit 0, no output
```

### Concerns

- Artifacts sealed with `assertion-compiler-v2` or `qa-utility-scorer-v1` now intentionally fail the live gate and must be rebuilt.

---

## Final broad-review fix wave

### Commits

- Implementation commit: `6605deb08219c0d87b2cd0a98bd52c8f190afb6e`
- Immediate-prior reward-pin regression commit: `fb2a91b78af1995324cf786c6f189579998c1780`

### Changed files

- `src/cloak/train/qa_builder.py`
- `src/cloak/train/roundtrip.py`
- `scripts/build_qa_utility_artifact.py`
- `scripts/train_ranker.py`
- `src/cloak/tests/test_qa_builder_v2.py`
- `src/cloak/tests/test_roundtrip.py`
- `src/cloak/tests/test_train_roundtrip_mode.py`
- `src/cloak/tests/test_build_qa_utility_artifact_cli.py`
- `research-wiki/training/2026-07-13-RL-ranker-v4-qa-utility-smoke.md`
- `.superpowers/sdd/final-review-fix-report.md`

### RED evidence

1. Semantic deduplication, refresh capability, and dependency identity:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py::test_context_validation_repeated_trials_require_refresh_capability src/cloak/tests/test_qa_builder_v2.py::test_builder_prefers_linked_aci_age_fact_over_global_duplicate src/cloak/tests/test_qa_builder_v2.py::test_high_level_builder_calls_teacher_once_then_compiles_and_validates src/cloak/tests/test_qa_builder_v2.py::test_builder_emits_authoritative_transitive_pins_and_manifest_identity src/cloak/tests/test_qa_builder_v2.py::test_builder_stamps_exact_production_reader_and_teacher_dependencies src/cloak/tests/test_train_roundtrip_mode.py::test_utility_artifact_gate_rejects_injected_builder_dependencies src/cloak/tests/test_train_roundtrip_mode.py::test_utility_artifact_gate_rejects_forged_cross_scope_semantic_duplicate
6 failed, 1 passed in 0.97s
```

The builder silently replayed refresh trials through readers without refresh support, retained both
linked and global ACI age facts, stamped injected dependencies as production, and let the forged
cross-scope semantic duplicate reach weight validation. The exact-production dependency case was
the one passing test.

2. Frozen source/reference identity, complete reward identity/cache use, and semantic pin bumps:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py::test_freeze_binds_source_and_authoritative_reference_identity src/cloak/tests/test_train_roundtrip_mode.py::test_utility_artifact_gate_rejects_source_or_reference_mismatch src/cloak/tests/test_train_roundtrip_mode.py::test_frozen_training_environment_binds_loaded_source_and_gold_reference src/cloak/tests/test_roundtrip.py::test_roundtrip_reward_pin_tracks_actual_task_prompt_and_invert src/cloak/tests/test_train_roundtrip_mode.py::test_utility_reward_cache_invalidates_task_prompt_or_invert_identity src/cloak/tests/test_train_roundtrip_mode.py::test_greedy_artifact_readout_reuses_persisted_reward_cache src/cloak/tests/test_train_roundtrip_mode.py::test_utility_artifact_gate_rejects_previous_semantic_pins
9 failed in 1.05s
```

Freeze accepted no source/reference inputs; two mismatch cases did not raise; training did not bind
loaded source/reference hashes; the centralized reward-pin helper and greedy cached readout did not
exist; prompt/invert cache identity could not be formed; and both immediately previous builder and
scorer pins were accepted.

### GREEN evidence

Required focused five-file suite, including the explicit previous reward-v1 cache regression:

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_utility_credit.py src/cloak/tests/test_roundtrip.py src/cloak/tests/test_train_roundtrip_mode.py src/cloak/tests/test_build_qa_utility_artifact_cli.py
197 passed in 2.82s
```

Changed Python files compiled successfully:

```text
/home/timo/repos/agent-cloak/.venv/bin/python -m py_compile scripts/build_qa_utility_artifact.py scripts/train_ranker.py src/cloak/train/qa_builder.py src/cloak/train/roundtrip.py src/cloak/tests/test_build_qa_utility_artifact_cli.py src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_roundtrip.py src/cloak/tests/test_train_roundtrip_mode.py
exit 0, no output
```

Patch hygiene:

```text
git diff --check
exit 0, no output
```

### Deterministic no-call smoke

The smoke ran from `/home/timo/repos/agent-cloak`, importing and executing the requested worktree:

```text
PYTHONPATH=/home/timo/repos/agent-cloak/.worktrees/qa-builder-v2-clean/src:/home/timo/repos/agent-cloak/.worktrees/qa-builder-v2-clean/scripts /usr/bin/time -f 'wall_s=%e' -o /tmp/qa-utility-broad-review.alPlZV/time.txt /home/timo/repos/agent-cloak/.venv/bin/python -u /home/timo/repos/agent-cloak/.worktrees/qa-builder-v2-clean/scripts/build_qa_utility_artifact.py --env data/ranker_env.json --arms data/task_arms_tau0.02.json --corpus clinical --doc-id aci/D2N002 --threshold-manifest /tmp/qa-utility-broad-review.alPlZV/manifest.json --out /tmp/qa-utility-broad-review.alPlZV/aci-D2N002.utility.json
wrote /tmp/qa-utility-broad-review.alPlZV/aci-D2N002.utility.json: docs=1 assertions=12 rejections=0
qa preflight: {"artifact_hash":"sha256:dfab9263ac7fe9d75e64ea9d42787a0a67418c5a07846a344729ff7312c9836b","call_budget":{"base":{"context_reader_batches_per_rollout":{"aci/D2N002":0},"remote_round_trips_per_rollout":1},"counterfactual":{"context_reader_batches_per_selected_pair":{"aci/D2N002":0},"remote_round_trips_per_selected_pair":1}},"cost_budgets":{"base":{"context_reader_batches_per_rollout":1,"remote_round_trips_per_rollout":1},"counterfactual":{"context_reader_batches_per_selected_pair":1,"remote_round_trips_per_selected_pair":1}},"documents":{"aci/D2N002":{"accepted_assertion_count":12,"context_assertion_count":0,"context_reader_batches_per_rollout":0,"delivered_assertion_count":12,"measurement_state":"partial","missing_family_budgets":["context"],"uncovered_decision_count":1,"uncovered_decision_ids":["sha256:35865283793e22780435488e33bfd0acc154cb517854b046479153013c72288f"]}},"environment_hash":"sha256:89d55c3747386687abfd7c4cc044a5a90b5231fee4d346bd6dd82b2b7065b544","executed_remote_calls":0,"totals":{"accepted_assertions":12,"context_assertions":0,"delivered_assertions":12,"documents":1,"uncovered_decisions":1}}
wall_s=0.98
```

- Builder/scorer pins: `qa-builder-v2-assertion-compiler-v4` and `qa-utility-scorer-v3`.
- Reward pin: `qa-builder-v2-roundtrip-reward-v2`; the smoke did not execute runtime rewards.
- Source hash: `sha256:c7e96f34a4878d6e33a02b5bc4de2becce9d58208dc74ac80800a99a123b1fb5`.
- Authoritative-reference hash: `sha256:c377953855c5df61ebf712fdf004ccb1b67172fc8f727454b6d0eacb721c48cc`.
- Relation teacher remained explicitly disabled. No relation-teacher, remote model, context-reader,
  or paid call was made; `executed_remote_calls=0`.

### Concerns

- Artifacts and cached rewards sealed with the immediately previous builder v3, scorer v2, or
  reward v1 identities are intentionally invalid and must be rebuilt.
- The transient smoke remains partial: zero context assertions, one uncovered decision, no runtime
  reward/cache execution, and no utility or privacy evidence.

## Follow-up final QA-builder review fix (2026-07-13)

### Implemented fixes

- The round-trip reward identity now pins the complete `cloak.extract` source, the directly used
  `cloak.runtime_types` source, and installed `rapidfuzz` and `sentence-transformers` versions.
  The semantic reward pin is `qa-builder-v2-roundtrip-reward-v3`, invalidating reward-v2 cache rows.
- `enforce_utility_artifact_gate` now rejects every delivered assertion whose scoring contract is
  absent, empty, unsupported, malformed, or has an empty expected value before training can issue
  a remote round trip.

The full-module source digest was chosen over a curated helper list because `invert` relies on
private cascade helpers and module constants; source hashing avoids silently omitting future local
behavior changes. It deliberately invalidates caches for unrelated edits in `cloak.extract`.

### RED evidence

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_train_roundtrip_mode.py::test_utility_artifact_gate_rejects_invalid_delivered_scoring_contract
5 failed in 0.94s
```

Before the gate change, absent, empty, unsupported, and empty-value delivered contracts all
passed preflight.

### GREEN evidence

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_utility_credit.py src/cloak/tests/test_roundtrip.py src/cloak/tests/test_train_roundtrip_mode.py src/cloak/tests/test_build_qa_utility_artifact_cli.py
202 passed in 2.83s
```

```text
/home/timo/repos/agent-cloak/.venv/bin/python -m py_compile src/cloak/extract.py src/cloak/train/roundtrip.py scripts/train_ranker.py src/cloak/tests/test_roundtrip.py src/cloak/tests/test_train_roundtrip_mode.py
exit 0, no output

git diff --check
exit 0, no output
```

## Transitive extractor dependency pin fix (2026-07-13)

### Implemented fixes

- Semantic extraction now loads `sentence-transformers/all-MiniLM-L6-v2` at exact Hugging Face
  revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- The extractor implementation identity records that model provenance and installed versions of
  `rapidfuzz`, `sentence-transformers`, `torch`, `transformers`, `tokenizers`, `numpy`, and
  `huggingface-hub`.
- The round-trip reward pin is now `qa-builder-v2-roundtrip-reward-v4`; reward-v3 cache rows are
  rejected.

The extractor pin is the correct transitive identity boundary: it is included verbatim in the
round-trip reward pin, which is included in the utility reward cache key.

### Regression coverage

- `test_semantic_model_loads_pinned_hf_revision` verifies the exact model ID and revision passed to
  `SentenceTransformer` at load time.
- `test_roundtrip_reward_pin_tracks_actual_task_prompt_and_invert` verifies all required package
  keys and model provenance appear in the reward identity.
- `test_utility_reward_cache_invalidates_task_prompt_or_invert_provenance` verifies a changed model
  revision or a changed `torch` version changes the scorer pin and misses the persisted cache.
- `test_utility_reward_cache_rejects_previous_roundtrip_reward_pin` verifies reward-v3 misses after
  the reward-v4 bump.

### Verification

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_utility_credit.py src/cloak/tests/test_roundtrip.py src/cloak/tests/test_train_roundtrip_mode.py src/cloak/tests/test_build_qa_utility_artifact_cli.py
202 passed in 2.86s

PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_extract.py
12 passed in 0.02s

/home/timo/repos/agent-cloak/.venv/bin/python -m py_compile src/cloak/extract.py src/cloak/train/roundtrip.py src/cloak/tests/test_extract.py src/cloak/tests/test_roundtrip.py src/cloak/tests/test_train_roundtrip_mode.py
exit 0, no output

git diff --check
exit 0, no output
```

### Concerns

- Existing reward-v3 cache entries intentionally miss and require recomputation.

## Final whole-change review fixes (2026-07-13)

### Implemented fixes

- KEEP and placeholder are unconditional action endpoints; floor filtering now applies only to
  non-KEEP lattice levels through the shared legality helper used by environment freezing and
  ranker span derivation.
- ACI now derives bounded, masked, lattice-backed semantic-property context candidates before any
  optional relation-teacher escalation. Delivered facts use human-reference evidence when present,
  otherwise explicit source evidence with authority metadata.
- Per-document teacher/compiler/reader infrastructure failures seal the artifact document as
  `build_failed`; training preflight rejects it. Builder/scorer pins include a complete
  QA-builder source digest.
- Utility reward cache rows require exact assertion IDs, weights, denominator, and artifact binding;
  score sets and recall are checked before store, load, or reuse. CLI artifact builds have a
  fail-closed content-addressed cache and teacher escalation requires that cache.
- Threshold manifests and preflight reports now carry optional explicit wall-time budgets and
  deterministic build measurements; runtime round trips report measured wall seconds only.

### RED evidence

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py::test_threshold_manifest_requires_non_boolean_nonnegative_min_context_assertions src/cloak/tests/test_qa_builder_v2.py::test_threshold_manifest_freezes_wall_time_budgets src/cloak/tests/test_qa_builder_v2.py::test_aci_d2n002_joint_anchor_keeps_unrelated_decisions_above_floor src/cloak/tests/test_qa_builder_v2.py::test_aci_deterministic_semantic_candidate_masks_all_protected_locators src/cloak/tests/test_qa_builder_v2.py::test_deterministic_semantic_candidate_accepts_through_fake_reader src/cloak/tests/test_qa_builder_v2.py::test_aci_delivered_facts_fall_back_to_explicit_source_with_authority_metadata src/cloak/tests/test_qa_builder_v2.py::test_context_reader_failure_marks_only_that_document_build_failed src/cloak/tests/test_build_qa_utility_artifact_cli.py::test_build_cache_reuses_complete_artifact_and_rejects_corruption src/cloak/tests/test_build_qa_utility_artifact_cli.py::test_teacher_escalation_requires_artifact_build_cache src/cloak/tests/test_train_roundtrip_mode.py::test_utility_reward_cache_binds_assertion_weights_denominator_and_recall
11 failed, 2 passed in 1.10s
```

### GREEN evidence

```text
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_build_qa_utility_artifact_cli.py src/cloak/tests/test_train_roundtrip_mode.py src/cloak/tests/test_utility_credit.py src/cloak/tests/test_roundtrip.py src/cloak/tests/test_extract.py
229 passed in 3.86s

PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/python -m py_compile src/cloak/train/qa_builder.py scripts/build_qa_utility_artifact.py scripts/train_ranker.py src/cloak/train/roundtrip.py src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_build_qa_utility_artifact_cli.py src/cloak/tests/test_train_roundtrip_mode.py src/cloak/tests/test_roundtrip.py src/cloak/tests/test_extract.py
exit 0, no output

git diff --check
exit 0, no output
```

### Deterministic no-call smoke

```text
PYTHONPATH=/home/timo/repos/agent-cloak/.worktrees/qa-builder-v2-clean/src:/home/timo/repos/agent-cloak/.worktrees/qa-builder-v2-clean/scripts /usr/bin/time -f 'wall_s=%e' -o /tmp/qa-builder-final-smoke.PuYx74/time.txt /home/timo/repos/agent-cloak/.venv/bin/python -u /home/timo/repos/agent-cloak/.worktrees/qa-builder-v2-clean/scripts/build_qa_utility_artifact.py --env data/ranker_env.json --arms data/task_arms_tau0.02.json --corpus clinical --doc-id aci/D2N002 --threshold-manifest /tmp/qa-builder-final-smoke.PuYx74/manifest.json --build-cache /tmp/qa-builder-final-smoke.PuYx74/cache --out /tmp/qa-builder-final-smoke.PuYx74/aci-D2N002.utility.json
wrote /tmp/qa-builder-final-smoke.PuYx74/aci-D2N002.utility.json: docs=1 assertions=13 rejections=0
wall_s=0.95
```

- Artifact: `sha256:54089cb264147fcbbfea8b033ab3025565c65a44e14916a3499f1441dd8c156c`;
  environment: `sha256:3218b2b377721ad33bde0ad61654a7b327f795c68fe52d52d6f427d6c1c6dcb8`.
- Preflight recorded zero context assertions, 13 delivered assertions, no uncovered decisions,
  `executed_remote_calls=0`, and a 0.0040-second build measurement under its explicit five-second ceiling.

### Concerns

- The smoke remains a partial, no-context artifact. It provides no context-utility, training,
  privacy, attacker, or remote-runtime evidence.
- Existing artifact and utility-reward cache entries lack the new builder/scorer digest or exact
  cache binding and intentionally fail closed; rebuild them before reuse.
