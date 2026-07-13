---
type: training-experiment
status: done
created: 2026-07-13
model: RankerPolicy unchanged; QA-builder v2 deterministic ACI artifact/preflight smoke only
dataset: ACI clinical `aci/D2N002` only (1 document; no train/validation split and no optimizer updates)
result: "Offline QA artifact smoke passed its local gate: 13 delivered assertions, no context assertions, one uncovered decision, and zero external calls; this is not an RL, privacy, or empirical-utility result."
tags: [ranker, qa-builder-v2, utility-artifact, preflight, deterministic-smoke, offline]
companion: ../../docs/research/adverserial-RL.md
---

# RL-ranker v4 QA utility smoke

## Objective & hypothesis

Validate the integrated QA-specific artifact gate, strict environment identity, deterministic
complete-reward cache identity, and no-call preflight reporting on one ACI document. The narrow
hypothesis was that the deterministic builder can emit a gate-valid partial artifact without a
relation teacher or any remote request. This is a plumbing smoke, not an RL run and not evidence
of privacy or utility improvement.

## Training data

- **Sources:** `corpora/clinical/aci.jsonl` through `load_task_docs("clinical")`, with frozen
  `data/ranker_env.json` and `data/task_arms_tau0.02.json`.
- **Ratio:** ACI `aci/D2N002` is 1/1 documents (100%); no other corpus, split, or held-out data
  participates.
- **Type mapping:** no remapping or detection runs. The builder consumes the frozen
  occurrence/decision identities from the arms artifact; all decision/action IDs remain their
  frozen values.

## Training config

No optimizer, policy sampling, remote generation, extraction, reader inference, relation teacher,
or counterfactual was launched. Relation escalation remained disabled because
`--relation-teacher` was omitted.

The implemented scope exercised here is QA artifact construction/scoring/gating. Elsewhere in the
same code slice, ranker training has provisional structured credit and a tested-pair substitution
hook only. There is no artifact-mode counterfactual scheduler/executor, lambda conditioning or
selection, or count-objective training implementation or claim.

The exact smoke invocation was:

```bash
PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_qa_utility_artifact.py --env data/ranker_env.json --arms data/task_arms_tau0.02.json --corpus clinical --doc-id aci/D2N002 --threshold-manifest /tmp/qa-utility-final-fix.FNHSoM/manifest.json --out /tmp/qa-utility-final-fix.FNHSoM/aci-D2N002.utility.json
```

The temporary manifest contained `family_budgets={context: 0.6, delivered: 0.4}`,
`min_context_assertions=0`, `task_pin=aci-v1`, and a planned maximum call surface of one remote
round trip plus at most one context-reader batch per rollout or selected pair. The builder
persisted the canonical live reader pin rather than accepting a manifest-supplied alias. The
worktree used temporary symlinks to the host `.venv` and read-only clinical corpus, both removed
immediately after the command.

## Selection & operating point

There is no policy-selection operating point. The artifact is a single deterministic build under
the frozen environment and manifest above. The preflight reports the fixed QA call surface only:
one base remote round trip per rollout and one extra remote round trip per selected
counterfactual pair; ranker-v2 retains ownership of scheduler totals and allocation.

## Evaluation & success criteria

Pass means the artifact passes `enforce_utility_artifact_gate`, the CLI emits an offline
preflight report, and the report records artifact counts, missing families, uncovered decisions,
and planned base/counterfactual calls without launching a remote call. This does not meet any
criterion for full RL, realized privacy, attacker resistance, or utility variation.

## Results

The smoke completed in **0.96 s** and printed:

```text
wrote /tmp/qa-utility-final-fix.FNHSoM/aci-D2N002.utility.json: docs=1 assertions=13 rejections=0
```

- Artifact hash: `sha256:57b6934ff83310d3a5710150d55c0e566289fb109c1e4e6e275ee649182daba3`.
- Environment hash: `sha256:f09c36723681014c7e6f94725b2c3f61bfe3ab98bb074f585a4c870943fa900f`.
- Reader pin: `Qwen3.5-0.8B` at `http://localhost:8060/v1`, batched prompt
  `qa-context-batch-prompt-v1`, response schema `qa-context-answers-v1`, temperature `0`,
  `max_tokens=512`, non-thinking JSON-object decoding.
- Measurement state: `partial`; accepted assertions: **13** total (**0 context**, **13 delivered**).
- Missing family: `context`; rejection summary: `{}`. The smoke therefore does not establish
  context-assertion support or context/delivered utility variation.
- Uncovered decisions: **1** —
  `sha256:35865283793e22780435488e33bfd0acc154cb517854b046479153013c72288f`.
- Preflight planned calls: base `1` remote round trip/rollout and `0` context-reader batches for
  this document; counterfactual `1` extra remote round trip/selected pair and `0` context-reader
  batches. `executed_remote_calls=0`.

The gate and cache tests also reject exact environment-hash or live-reader-pin mismatches,
unknown/unsupported/build-failed measurement states, missing cost/group metadata, forged family
or uncovered-decision summaries, empty accepted assertion sets, invalid fixed denominators, and
dangling links. Cache v2 fail-closes on complete-result hash or field tampering and validates
finite range-bounded component scores and recall; one dispatched miss batch persists atomically
once.

## Ablations

None. Relation teacher escalation was deliberately disabled rather than compared. No full
training, counterfactual scheduler, privacy attack, or utility ablation ran.

## Cost

Wall time was **0.96 s**. The artifact build made **zero external or paid calls**. The reported
base/counterfactual figures are pinned call-surface counts, not measured remote throughput or a
ranker-v2 scheduler allocation.

## Risks & caveats

- This partial artifact has no accepted context family, so it cannot support any claim requiring
  contextual-generalization measurement or context/delivered variation.
- One controlled decision remains uncovered; ranker-v2 fallback/counterfactual behavior was not
  exercised here.
- Strict environment-hash equality means a subset smoke artifact is intentionally not a
  certificate for a full-environment RL run; rebuild under the exact training environment.
- No RL optimization, remote round trip, privacy attack, or empirical comparison was run.

## Artifacts

- Code: `scripts/train_ranker.py`, `scripts/build_qa_utility_artifact.py`.
- Tests: `src/cloak/tests/test_train_roundtrip_mode.py`,
  `src/cloak/tests/test_build_qa_utility_artifact_cli.py`.
- Frozen inputs: `data/ranker_env.json`, `data/task_arms_tau0.02.json`.
- Transient smoke artifact, manifest, stdout, and timing: `/tmp/qa-utility-final-fix.FNHSoM/`
  (not a durable experiment artifact; left for local review and eventual system cleanup).
- Predecessor: [RL-ranker v3 round-trip pilot](2026-07-06-RL-ranker-v3-roundtrip-pilot.md).
- Successor: none recorded; any full ranker-v4 run must receive a new record rather than
  overwriting this smoke.

## Sources

- [QA builder v2 specification](../../docs/specs/qa-builder-v2.md).
- [Interactive ranker v2 specification](../../docs/specs/RL/interactive-ranker-v2.md).
- [QA builder v2 implementation plan](../../docs/plans/2026-07-13-qa-builder-v2-implementation.md).
- [RL background companion](../../docs/research/adverserial-RL.md).
