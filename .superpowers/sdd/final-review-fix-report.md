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
