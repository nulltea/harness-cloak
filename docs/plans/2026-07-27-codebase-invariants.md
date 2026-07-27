---
type: reference
status: current
created: 2026-07-27
updated: 2026-07-27
tags: [invariants, architecture, refactor]
companion: [2026-07-27-codebase-cleanup-refactor.md]
---

# Codebase invariants — reference card

What any change to `src/cloak`, `scripts/`, or the artifacts must preserve. Distilled from
§Invariants of [the cleanup plan](2026-07-27-codebase-cleanup-refactor.md) into the durable form;
the plan keeps the phase history and the path mapping.

## The two live workflows

Everything in `src/cloak` is reachable from one of these. Code that isn't is dead by definition.

**Data production** — corpora → detection → lattice → QA → frozen environment → caches:

```
build_task_corpora.py → build_arms_artifact.py (clinical detection + inline aset)
  → build_lattice_profiles.py | run_lattice_producer.py (+ populate_/merge_/dedupe_ profiles)
  → build_qa_utility_artifact.py → build_ranker_env.py
  → build_ranker_representation_cache.py → build_profile_count_targets.py
```

**Training** — `scripts/train_interactive_ranker.py {bc, exit-collect, train}`, semantic-v1 policy
with direct counts, plus `scripts/run_ranker_preflight.py` (support scan) and
`scripts/train_ranker_privacy_head.py`. Reward = `score_roundtrip_batch`
(`src/cloak/reward/roundtrip.py`).

Gate for any change touching either: full `pytest src/` green **plus** the 3-doc BC+ExIt smoke
reproducing its current numbers from cache.

`scripts/` is flat by decision — no stage subdirectories.

## Artifact and cache identity

Every `*_VERSION` constant, pin-dict shape, and hash input is part of an artifact's identity.
Byte-identical inputs must keep producing byte-identical identities.

**The trap:** `invert_implementation_pin()` in `src/cloak/reward/extract.py` hashes the **source
text and module `__name__`** of `cloak.reward.extract` and `cloak.runtime_types` into every
utility-cache identity. Editing, renaming, or moving either module invalidates the whole reward
cache — a docstring fix is enough. Both module keys are derived from `__name__`, so a move is
self-consistent but still re-baselines.

Consequence: land any edit to those two modules **before** a preflight/calibration run populates
the cache. Cheap when the cache is a handful of rows; hours of recompute after.

**Re-baseline procedure** (when a pin change is unavoidable):

1. `mv results/ranker_v2/cache/utility-results.jsonl …/utility-results.pre-<reason>.jsonl`
2. Repopulate with the 3-doc BC smoke, `CLOAK_LLM_CACHE=data/llm_cache` — all remote calls must
   be disk-cache hits (verify: zero new files under `data/llm_cache`).
3. Prove identity-only: per-doc `utility` / `count_score` and the BC epoch losses must be
   byte-identical to the archived rows. If a number moved, it was not a pin change.
4. Re-run the hash gate; note the drop in re-derivable identities (archived rows carry the old
   pin and are not re-derivable by design).

## Single-source pins

Exactly one definition each; a move keeps it at one.

| pin | value | home |
|---|---|---|
| `RT_MODEL` (reward round-trip) | `medgemma-4b-it` | `src/cloak/reward/roundtrip.py` |
| `QA_MODEL` (context reader) | `medgemma-4b-it` | `src/cloak/qa/scoring.py` |
| `ENCODER_ID` + `ENCODER_REVISION` | BioClinical-ModernBERT-base @ `c3648aa8…` | `src/cloak/ranker/representation.py` |
| detector model + threshold + labels | `checkpoint-2479`, 0.3, `QA_V2_CLINICAL_LABELS` | `src/cloak/detection/detect.py` |
| `NLI_MODEL` | DeBERTa-v3-base-mnli-fever-anli | `src/cloak/lattice/core.py` |

The reader is medgemma, in one place. Never a second reader in a spike.

## Determinism

Temp-0 generation plus `CLOAK_LLM_CACHE` memoization; seeded splits/folds; `sort_keys=True` on
every hash and artifact write. No `set` iteration order in anything that reaches an artifact,
a sampled band, or a test assertion.

## Privacy boundary

Only `doc_p` renders leave the host. Uncontrolled-span leakage is tracked in `docs/issues/` and
never silently widened. Empirical-honesty machinery (promotion-gate semantics, matched-privacy
comparison, no per-model calibration knobs) is intact — see CLAUDE.md and `docs/specs/RL/`.

## Tests are the contract

Tests move with their code. A test is deleted only when the behavior it pins dies in the same
commit.

**Executable form of the structure — `src/cloak/tests/test_architecture.py`** (runs in the default
`pytest src/` sweep): `cloak` never imports `inferdpt`; `cloak` never imports `scripts`;
`qa/scoring` is the bottom of the QA stack (never imports builder/teacher/freeze/review); `reward/**`
imports nothing under `cloak.ranker` except `cloak.ranker.environment`; `ranker/environment.py`
stays policy- and torch-free (only `cloak.corpora` / `cloak.runtime_types`); the retired training
package stays gone; only the five core modules sit loose at the `cloak` root and all five stage
packages exist.

**Executable form of identity — `scripts/refactor_hash_gate.py`**: recomputes the frozen
environment hash, the profile-count target hash, and every stored utility-cache identity.

```bash
CLOAK_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python scripts/refactor_hash_gate.py
```

Run both after any structural change. Green suite + `HASH GATE PASS` is the whole bar.

## Docs stay navigable

Every path move lands in the plan's path-mapping table, and referencing docs with
`status: current`/`partial` are updated in the same commit. A doc whose subject was deleted gets
`status: stale` + `archive_reason`, not a rewrite.
