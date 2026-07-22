# QA Builder v2 Final Fix Report

## Status

Complete. All Critical, Important, and Minor final-review findings were fixed in one change wave on `codex/qa-builder-v2-clean`. No external calls were made.

## Design Decisions

- Defined one canonical live context-reader pin from the actual endpoint, model, prompt hash/version, response schema/version, and decoding settings. Artifacts persist this pin, the gate compares it exactly, and reward-cache request identity includes it.
- Chose fail-closed cache integrity rather than silently treating corruption as a miss. A corrupt cache file, request record, result hash, result schema, or invalid score now raises an explicit validation error.
- Preserved atomic JSON persistence while moving it to one `store_many` call per dispatched miss batch. Duplicate requests remain coalesced and hit/miss accounting remains request-accurate.
- Derived lattice entailment from canonical frozen action order and semantics rather than trusting optional input `entails` metadata.
- Kept all numeric gate values manifest-driven. The gate validates manifest-declared call ceilings and recomputes metadata from assertions and the live environment.

## Findings Resolved

1. **Reader pin and refresh**: added the canonical pin, persisted and gated it, included it in cache identity, and propagated `reader_refresh=True` through `_read_batch` to `LLMClient.generate` for ExIt verification.
2. **Cache result integrity**: canonical-hashed the complete stored result (`out_p`, `out_final`, component scores, recall, status, and version); validated required keys, types, finiteness, ranges, and consistency on load and lookup.
3. **Cache persistence scaling**: batched validated inserts from each dispatched job batch and persisted atomically once.
4. **Lattice closure**: frozen finer actions entail their own property and every coarser property; coarser actions do not entail finer properties. Representative-anchor integration tests verify exact broad and narrow anchors after freezing.
5. **Artifact counterfactual fraction**: artifact mode rejects every nonzero `--cf-frac` before torch initialization, logging, or training. Legacy mode is unchanged.
6. **Strict gate metadata**: required cost budgets and weight groups; rejected unknown measurement states; recomputed uncovered decisions, family state, counts, controlled-decision ordering, and call-surface preflight from authoritative inputs.
7. **Reverse companion link**: linked `docs/research/adverserial-RL.md` back to the RL-ranker v4 smoke record.
8. **Scope clarity**: documented implemented QA artifacts, provisional structured credit, and the substitution hook only. Counterfactual scheduling/execution, lambda conditioning, and count training are explicitly not implemented.

## TDD Evidence

### Red

Command:

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_train_roundtrip_mode.py -k 'fails_closed_on_result_tampering or rejects_invalid_scores or persists_once_per_dispatched_batch or unknown_measurement_state or requires_frozen_cost_budgets or requires_weight_groups or recomputes_uncovered_decisions or live_reader_pin_mismatch or qa_preflight_recomputes or artifact_cf_frac'
```

Result: `20 failed, 98 deselected in 1.11s`. Failures covered missing cache integrity, batch persistence, reader-pin enforcement, artifact counterfactual rejection, and strict gate recomputation.

Command:

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py -k 'freeze_preserves_lattice_entailment_closure or context_reader_pin_covers or batched_context_reader_refresh or roundtrip_utility_artifact_scores'
```

Result: `4 failed, 36 deselected in 0.06s`. Failures covered missing lattice closure, reader pin, and refresh propagation.

Command:

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_train_roundtrip_mode.py -k authoritative_controlled_decision_order
```

Result: `1 failed, 77 deselected in 0.80s`. The gate trusted reported controlled-decision order.

### Green

Targeted commands produced:

- Reader/lattice regression set: `4 passed, 36 deselected in 0.02s`.
- Cache/gate/config regression set: `20 passed, 57 deselected in 0.82s`.
- Authoritative-order regression set: `4 passed, 74 deselected in 0.79s`.
- Complete focused model-free suite before documentation updates: `129 passed in 1.75s`.

## Deterministic Local Smoke

The manifest schema changed, so the no-teacher smoke was rerun. Temporary worktree symlinks exposed the host `.venv` and original read-only clinical corpus, then were removed. No teacher, reader, or other remote call was made.

Command:

```bash
PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_qa_utility_artifact.py --env data/ranker_env.json --arms data/task_arms_tau0.02.json --corpus clinical --doc-id aci/D2N002 --threshold-manifest /tmp/qa-utility-final-fix.FNHSoM/manifest.json --out /tmp/qa-utility-final-fix.FNHSoM/aci-D2N002.utility.json
```

Output:

```text
wrote /tmp/qa-utility-final-fix.FNHSoM/aci-D2N002.utility.json: docs=1 assertions=13 rejections=0
```

Measured preflight:

- Artifact hash: `sha256:57b6934ff83310d3a5710150d55c0e566289fb109c1e4e6e275ee649182daba3`.
- Environment hash: `sha256:f09c36723681014c7e6f94725b2c3f61bfe3ab98bb074f585a4c870943fa900f`.
- Reader prompt hash: `sha256:ec1c90a6487f8365b41259618f01f4eb876e4f7251d062fa29296cd07f71e274`.
- Measurement state: `partial`; accepted assertions: `13` (`0` context, `13` delivered); missing family: `context`.
- Uncovered decisions: `1` (`sha256:35865283793e22780435488e33bfd0acc154cb517854b046479153013c72288f`).
- Planned/observed base call surface: `1` remote round trip, `0` context-reader batches per rollout.
- Planned/observed counterfactual call surface: `1` remote round trip, `0` context-reader batches per selected pair.
- Executed remote calls: `0`; wall time: `0.96s`.

## Final Verification

Command:

```bash
set -e
cleanup() { rm -f .venv corpora/clinical; rmdir corpora 2>/dev/null || true; }
trap cleanup EXIT
ln -s /home/timo/repos/agent-cloak/.venv .venv
mkdir -p corpora
ln -s /home/timo/repos/agent-cloak/corpora/clinical corpora/clinical
PYTHONPATH=src:scripts .venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_build_qa_utility_artifact_cli.py src/cloak/tests/test_utility_credit.py src/cloak/tests/test_train_roundtrip_mode.py src/cloak/tests/test_roundtrip.py
.venv/bin/python -m py_compile scripts/train_ranker.py scripts/build_qa_utility_artifact.py src/cloak/train/qa_builder.py src/cloak/train/roundtrip.py src/cloak/train/utility_credit.py
git diff --check
```

Result:

```text
131 passed in 2.67s
```

`py_compile` and `git diff --check` both exited `0` with no output.

## Files

- `src/cloak/train/qa_builder.py`
- `src/cloak/train/roundtrip.py`
- `scripts/build_qa_utility_artifact.py`
- `scripts/train_ranker.py`
- `src/cloak/tests/test_qa_builder_v2.py`
- `src/cloak/tests/test_build_qa_utility_artifact_cli.py`
- `src/cloak/tests/test_train_roundtrip_mode.py`
- `docs/research/adverserial-RL.md`
- `docs/specs/qa-builder-v2.md`
- `docs/specs/RL/qa-builder-v2-decision-log.md`
- `docs/plans/2026-07-13-qa-builder-v2-implementation.md`
- `research-wiki/training/2026-07-13-RL-ranker-v4-qa-utility-smoke.md`

## Commit

`HEAD` — `fix(qa): harden reader and reward cache contracts`

## Concerns

- The deterministic smoke remains intentionally partial: no context-family assertion, 13 delivered assertions, and one uncovered decision. It supports contract validation only, not a utility, privacy, or RL result claim.
- Artifact-mode counterfactual scheduling/execution, lambda conditioning, and count-objective training remain unimplemented. Nonzero artifact `--cf-frac` now fails explicitly.
- Cache schema v1 files are intentionally rejected by the fail-closed v2 loader and must be regenerated.
- Smoke artifacts remain in `/tmp/qa-utility-final-fix.FNHSoM` for inspection.

---

# Localized Context-Preflight Fix Wave

## Status

Complete on 2026-07-13. No external calls were made.

## Root Causes and Fixes

- `build_qa_utility_artifact.py` discarded the full frozen environment before preflight and supplied only its hash. `build_from_files` can now return the exact environment it used, and `main` passes that full value to `qa_utility_preflight_report`.
- Artifact mode had no reward-mode invariant before torch seeding and data/model setup. `--utility-artifact` now requires `--reward roundtrip` immediately after argument parsing.
- Family-budget validation checked names inconsistently and allowed zero or non-finite values at some boundaries. One `normalize_family_budgets` contract now requires exactly `context` and `delivered`, each positive and finite, in packaging/building, CLI manifest loading, artifact gating, and preflight.
- Preflight values previously described as frozen budgets were actually the planned/observed call surface. The report and committed documentation now use the correct terminology.

## TDD Red Evidence

Command:

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/pytest -q src/cloak/tests/test_build_qa_utility_artifact_cli.py src/cloak/tests/test_train_roundtrip_mode.py src/cloak/tests/test_qa_builder_v2.py -k 'context_build_cli_preflights or invalid_family_budgets or utility_artifact_requires_roundtrip or exact_positive_family_budgets or builder_requires_exact_positive'
```

Result: `16 failed, 6 passed, 119 deselected in 1.10s`. The context CLI rejected injected dependencies, invalid budgets reached source/weight logic, surrogate artifact mode reached `torch.manual_seed`, and package validation accepted invalid budgets.

## TDD Green Evidence

The same command passed after the minimal boundary fixes:

```text
22 passed, 119 deselected in 0.84s
```

The context integration was then strengthened to use the real relation compiler with fixture-generated frozen occurrence IDs. Its isolated rerun produced:

```text
1 passed in 0.80s
```

The integration executes argument parsing, file loading, environment freezing, relation compilation, legal joint-anchor construction, reader validation, artifact packaging, strict gate validation, preflight reporting, and output writing. The fixture teacher and reader are local deterministic callables.

## Focused Suite

Command:

```bash
set -e
cleanup() { rm -f .venv corpora/clinical; rmdir corpora 2>/dev/null || true; }
trap cleanup EXIT
ln -s /home/timo/repos/agent-cloak/.venv .venv
mkdir -p corpora
ln -s /home/timo/repos/agent-cloak/corpora/clinical corpora/clinical
PYTHONPATH=src:scripts .venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_build_qa_utility_artifact_cli.py src/cloak/tests/test_utility_credit.py src/cloak/tests/test_train_roundtrip_mode.py src/cloak/tests/test_roundtrip.py
```

Result: `153 passed in 2.71s`.

## Deterministic Local Smoke

Preflight behavior changed, so the no-teacher smoke was rerun with the prior manifest and no external calls:

```bash
PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_qa_utility_artifact.py --env data/ranker_env.json --arms data/task_arms_tau0.02.json --corpus clinical --doc-id aci/D2N002 --threshold-manifest /tmp/qa-utility-localized-fix.LJXZmd/manifest.json --out /tmp/qa-utility-localized-fix.LJXZmd/aci-D2N002.utility.json
```

Measured output was unchanged:

- Artifact hash: `sha256:57b6934ff83310d3a5710150d55c0e566289fb109c1e4e6e275ee649182daba3`.
- Environment hash: `sha256:f09c36723681014c7e6f94725b2c3f61bfe3ab98bb074f585a4c870943fa900f`.
- Assertions: `13` accepted (`0` context, `13` delivered); one uncovered decision.
- Planned/observed call surface: base `1` remote round trip and `0` context-reader batches; counterfactual `1` remote round trip and `0` context-reader batches.
- Executed remote calls: `0`; wall time: `0.96s`.

The training record's measured values therefore did not change; only its call-surface wording was corrected.

## Final Verification

Command:

```bash
set -e
cleanup() { rm -f .venv corpora/clinical; rmdir corpora 2>/dev/null || true; }
trap cleanup EXIT
ln -s /home/timo/repos/agent-cloak/.venv .venv
mkdir -p corpora
ln -s /home/timo/repos/agent-cloak/corpora/clinical corpora/clinical
PYTHONPATH=src:scripts .venv/bin/pytest -q src/cloak/tests/test_qa_builder_v2.py src/cloak/tests/test_build_qa_utility_artifact_cli.py src/cloak/tests/test_utility_credit.py src/cloak/tests/test_train_roundtrip_mode.py src/cloak/tests/test_roundtrip.py
.venv/bin/python -m py_compile scripts/train_ranker.py scripts/build_qa_utility_artifact.py src/cloak/train/qa_builder.py src/cloak/train/roundtrip.py src/cloak/train/utility_credit.py
git diff --check
```

Result: `153 passed in 2.72s`. `py_compile` and `git diff --check` exited `0` with no output. Temporary dependency links were removed by the cleanup trap.

## Files

- `scripts/build_qa_utility_artifact.py`
- `scripts/train_ranker.py`
- `src/cloak/train/qa_builder.py`
- `src/cloak/tests/test_build_qa_utility_artifact_cli.py`
- `src/cloak/tests/test_train_roundtrip_mode.py`
- `src/cloak/tests/test_qa_builder_v2.py`
- `docs/specs/qa-builder-v2.md`
- `docs/plans/2026-07-13-qa-builder-v2-implementation.md`
- `research-wiki/training/2026-07-13-RL-ranker-v4-qa-utility-smoke.md`
- `.superpowers/sdd/final-fix-report.md`

## Commit

`HEAD` — `fix(qa): validate context preflight and artifact mode`

## Concerns

- The deterministic smoke remains partial and supports contract validation only, not utility, privacy, or RL claims.
- The public preflight JSON field remains named `call_budget` for schema compatibility, but its values are documented as the planned/observed call surface; `cost_budgets` remains the manifest-declared ceiling.
- Artifact-mode counterfactual scheduling/execution, lambda conditioning, and count training remain out of scope.
