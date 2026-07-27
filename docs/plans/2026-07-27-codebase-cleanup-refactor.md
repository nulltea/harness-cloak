---
type: plan
status: current
created: 2026-07-27
updated: 2026-07-27
tags: [refactor, cleanup, codebase-structure, invariants, tech-debt]
companion: [../specs/RL/ranker-v2-architecture.md]
---

# Codebase cleanup, reorganization, and refactor plan

Audit date 2026-07-27. Evidence: import-reachability analysis from live entry points,
per-script classification (50 scripts), spike/inferdpt/bench deletion scan (102 files),
cloak-internal legacy map, and a full section map of `qa_builder.py` (10,158 lines).
Repo at audit time: `src/` = 74,645 lines across 170 py files; 50 scripts + 76 spikes;
87 test files.

## Invariants (what every phase must preserve)

- **I1 — The two live workflows run end-to-end.** (a) Data production: corpora →
  clinical detection (arms artifact) → lattice profiles/producer → QA utility artifact →
  frozen environment → representation cache → profile-count targets. (b) Training:
  `bc` / `exit-collect` / `train` (semantic-v1, direct counts) + preflight + privacy
  diagnostics, reward via `score_roundtrip_batch`. Gate: full pytest suite green plus the
  3-doc BC+ExIt smoke reproducing its current numbers from cache.
- **I2 — Artifact and cache identities are stable.** Every `*_VERSION` constant, pin dict
  shape, and hash input stays byte-identical. Trap: `invert_implementation_pin()` embeds
  **source hashes and module names of `cloak.extract` and `cloak.runtime_types`** into
  every utility-cache identity. Consequence: any edit/move/rename of those two modules
  must land **before** the calibration preflight populates the reward cache (currently
  ~17 rows — free; post-preflight — hours of recompute). The refactor is therefore
  sequenced entirely before the preflight.
- **I3 — Single-source pins.** `RT_MODEL` (medgemma), `QA_MODEL`, the encoder
  revision, detector model+threshold+labels, `NLI_MODEL`: exactly one definition each;
  moves keep one definition.
- **I4 — Empirical-honesty machinery intact**: promotion gate semantics, matched-privacy
  comparison rules, no calibration knobs (CLAUDE.md hard rule; specs in docs/specs/RL).
- **I5 — Determinism**: temp-0 + `CLOAK_LLM_CACHE` memoization, seeded splits/folds,
  `sort_keys` stable hashing.
- **I6 — Privacy boundary**: only `doc_p` renders leave the host; uncontrolled-span
  leakage is tracked in docs/issues and never silently widened.
- **I7 — Tests are the contract**: tests move with their code; a test is deleted only
  when the behavior it pins dies in the same commit.
- **I8 — Docs stay navigable**: every path move lands in the path-mapping table
  (Phase 4) and referencing docs with `status: current` are updated in the same commit.

## Deletion list (ranked, evidence-backed)

### D1 — dead by evidence, no live dependents (~12,100 lines)

| what | lines | evidence |
|---|---:|---|
| 61 of 76 `scripts/spikes/*` | ~9,440 | findings extracted to research-wiki/docs (extraction receipts verified per family); only 4 spikes have live code deps; 5 doc-referenced spikes were already deleted previously with no breakage. KEEP list (15): `privacy_block_attribution`, `roundtrip_support_scan`, `build_common_lattice_profiles`, `identity_attack`, `privacy_probe_shootout`, `probe_shootout_rescore`, `probe_flip_scan`, `ground_deterministic_level_counts`, `reconstructor_issue_probes`, `reconstructor_residue_fresh_probe`, `validate_profile_match(+.out.txt)`, `pii_zeroshot_generality`, `lattice_count_shootout`, `check_ranker_m1_dom_layout.mjs` |
| `src/inferdpt` RANTEXT core (9 files + its test) | ~971 | CLAUDE.md marks InferDPT+RANTEXT abandoned; `src/cloak` imports none of it (a test asserts exactly that); migration-debt issue is `status: stale` |
| `scripts/{dp_sweep,build_phi_subvocab,build_cnndm_corpus}.py` | ~700 | only importers of the RANTEXT core; ε-sweep era |
| `scripts/latticecloak_{engine,inventory,task_eval,tau_sweep,task_tau_sweep}.py`, `surrogate_env_diagnostics.py` | ~1,600 | SynthPAI/latticecloak era; superseded by `run_roundtrip_benchmark.py`; **keep `latticecloak_detection_gate.py`** (only TAB gold-span detector gate, named by current spec) |
| `cloak/{score,synthpai}.py` | 74 | die with the latticecloak scripts (bench.metrics import of `score` moves inline or `score.py` relocates — 45 lines, decide at implementation) |
| `qa_builder.py` §22 dormant NLI/cue gates | 245 | both feature flags hardcoded disabled; file itself says "retained pending removal" |

### D2 — legacy ranker-v1 retirement (~5,000 lines; requires the conversions below)

| what | lines | evidence |
|---|---:|---|
| `scripts/{train_ranker,reward_gate,build_probes,annotate_lattice_counts}.py` | ~1,500 | superseded by `train_interactive_ranker` / `run_ranker_preflight` / `build_ranker_env` / inline aset in arms build |
| `cloak/train/ranker.py` minus `LambdaProfile` | ~647 | `ConditionalRankerPolicy` reachable only via `--policy-architecture legacy-film-gru`; `LambdaProfile` moves to `ranker_environment.py` |
| `legacy-film-gru` branches in `train_interactive_ranker.py` + `interactive_ranker.py` | ~200 | initial-RL scope is semantic-v1 only; the XOR `(count_reward is None) == (profile_targets is None)` paths simplify to profile-targets-only |
| `cloak/train/count_reward.py` legacy half (185–646) | ~478 | live core = lines 1–184 (env validation) which `profile_count.py` imports; **prerequisite:** convert `lambda_menu.py` + `ranker_diagnostics.py` off `CountReward` |
| `cloak/train/{probes,ladder_probes,schema_task}.py` + `reward.py` minus ~60 live lines | ~1,730 | probe-era; live surface of `reward.py` is 4 names (`QA_MODEL`, `QA_BASE_URL`, `canon`, `fact_score` + helpers) — fold into the QA scoring module |
| `roundtrip.py` legacy `roundtrip_batch` cluster + `__main__` | ~180 | no `src/cloak` caller; all callers are D1/D2 scripts/spikes |
| `extract.invert_detector_pointer` | 56 | spike-only |

### D-open — dormant strategic tracks (decision required, not auto-delete)

- **Reconstructor/frozen-extractor track** (`cloak/{frozen_extractor,reconstruct}.py`,
  `scripts/{build_reconstructor_data,train_reconstructor}.py`, 2 spikes; ~2,900 lines):
  currently unreachable from live paths, but it *is* CLAUDE.md direction #2 (the tailored
  extraction model). Recommendation: **archive to a branch** (`archive/reconstructor-track`),
  delete from main, revive when the extraction phase starts. Keeps main readable without
  losing the work.
- **Detector fine-tune track** (`scripts/{build_pii_span_dataset,train_pii_gliner}.py`,
  `latticecloak_detection_gate.py`): produces the detector checkpoint live code loads and
  is the per-user-type extensibility path (CLAUDE.md positioning). Recommendation: **keep**,
  relocate under `scripts/detector/`.
- **UNCLEAR leftovers**: `detector_pipeline_red_tests.py` (keep with detector track),
  `build_count_reward_state.py` + `migrate_qa_utility_artifact.py` (migration one-shots —
  delete after D2 since count-state consumers retire).

## Target structure

```
src/cloak/
  llm.py  concurrent.py  runtime_types.py  corpora.py  tasks.py      # shared core
  detection/   detect.py  span_gate.py  span_prep.py (from substitute.py)  probe.py
  lattice/     lattice.py  profiles.py  profile_match.py  anonymity.py  producer/ (=lattice_producer)
  qa/          builder.py (compile core)  teacher.py  scoring.py (runtime slice)
               freeze.py  audit.py  relation_support_gate.py  review.py
  reward/      roundtrip.py  extract.py  utility_cache.py  utility_credit.py
  ranker/      environment.py  representation.py  semantic.py  privacy.py
               interactive.py  profile_count.py  lambda_menu.py  diagnostics.py
               counterfactuals.py
  tests/       mirrors the packages
src/bench/     unchanged (current spec owns it; in-flight plan extends it)
scripts/       live workflows only, grouped: ranker/  lattice/  qa/  detector/  spikes/
```

Naming: packages by pipeline stage; no doc-internal identifiers; `qa/scoring.py` is the
~800-line runtime slice (`prepare/finalize_utility_scoring`, readers, answer scoring,
`_parse_aci_note`, delivered contracts) — the only part of QA the training loop executes.

## Phases (each = commit series, gated)

- **Phase 0 — guardrails** (before anything): `tests/test_architecture.py` with import
  contracts (no `cloak` → `inferdpt`; `qa/scoring` never imports teacher/freeze code;
  `reward/` never imports `ranker/`); a hash-stability gate script asserting the frozen
  env hash + profile-target hash + one utility-cache identity recompute unchanged; record
  baseline test count. Fix the two CLAUDE.md phantoms (nonexistent `scripts/harness/perf_gate.md`,
  naming example pointing at a to-be-deleted script).
- **Phase 1 — D1 deletions.** Pure `git rm` + dangling-reference touch-ups in docs
  (mark referencing docs `status: stale` where the subject died). Gate: suite + hash gate.
- **Phase 2 — D2 retirement.** Order: convert `lambda_menu`/`ranker_diagnostics` off
  `CountReward` → fold `reward.py` live names into QA scoring → delete legacy modules and
  the `legacy-film-gru` CLI branch → move `LambdaProfile`. Gate: suite + hash gate +
  cached 3-doc smoke.
- **Phase 3 — qa_builder split** (10,158 → 4 modules + compile core): extract
  `scoring.py` first (7 public names, one importer), then `teacher.py`, `freeze.py`,
  `review.py`; promote the underscore names that already leak across the boundary.
  No behavior change; `UTILITY_SCORER_VERSION` and reader pins untouched. Gate: suite +
  hash gate (the reader pin and scorer version prove the runtime slice unchanged).
- **Phase 4 — package regroup** per the target structure, in one mechanical commit per
  package; includes the `extract.py`/`runtime_types.py` moves (pin identity changes —
  **must precede the preflight**, cache is trivially small); ends with the path-mapping
  table appended to this doc and doc references updated.
- **Phase 5 — invariants doc**: distill §Invariants into `docs/plans/` reference doc
  (type: reference) + wire the architecture test into the default pytest run.
  **Done:** [`2026-07-27-codebase-invariants.md`](2026-07-27-codebase-invariants.md);
  `src/cloak/tests/test_architecture.py` needs no wiring — it is collected by `pytest src/`.

Execution: SDD (codex implements per phase task, Fable reviews/commits path-scoped);
phases 1–2 are also safe as direct implementation. Estimated result: **~17,000 lines
removed (~20% of the repo)**, `qa_builder` down to a coherent compile core, every
remaining file reachable from a named live workflow.

## Open decisions

1. Reconstructor track: archive-branch (recommended) vs keep-in-main vs delete.
   **Resolved:** archived to branch `archive/reconstructor-track`, deleted from main.
2. `legacy-film-gru` policy architecture: delete (recommended; scope decision already
   excludes it) vs keep behind the flag. **Resolved:** deleted.
3. Grouping depth for `scripts/`: subdirectories (recommended) vs flat with prefixes.
   **Resolved: `scripts/` stays FLAT** — deliberate deviation from the target structure above.
   34 live scripts with self-descriptive prefixes (`build_*`, `run_*`, `train_*`) navigate fine;
   subdirectories would have rewritten every `PYTHONPATH=src:scripts` invocation in the docs,
   the training records, and the runbooks for no navigability gain. The `scripts/ranker/ lattice/
   qa/ detector/` line in §Target structure is therefore **not** implemented; `scripts/spikes/`
   (which predates the plan) stays.

## Path mapping (Phase 4a, executed)

Executed 2026-07-27; every move used `git mv`, so history follows. Old paths below no longer
exist — there are **no compatibility shims**. Report: `.superpowers/sdd/refactor-phase4a-report.md`.

### Stays at the `cloak` root (shared core)

`llm.py` · `concurrent.py` · `runtime_types.py` · `corpora.py` · `tasks.py`

### `cloak/detection/`

| old | new | module path |
|---|---|---|
| `src/cloak/detect.py` | `src/cloak/detection/detect.py` | `cloak.detect` → `cloak.detection.detect` |
| `src/cloak/span_gate.py` | `src/cloak/detection/span_gate.py` | `cloak.span_gate` → `cloak.detection.span_gate` |
| `src/cloak/probe.py` | `src/cloak/detection/probe.py` | `cloak.probe` → `cloak.detection.probe` |
| `src/cloak/substitute.py` | `src/cloak/detection/span_prep.py` | `cloak.substitute` → `cloak.detection.span_prep` |

### `cloak/lattice/`

| old | new | module path |
|---|---|---|
| `src/cloak/lattice.py` | `src/cloak/lattice/core.py` | `cloak.lattice` → `cloak.lattice.core` |
| `src/cloak/lattice_profiles.py` | `src/cloak/lattice/profiles.py` | `cloak.lattice_profiles` → `cloak.lattice.profiles` |
| `src/cloak/profile_match.py` | `src/cloak/lattice/profile_match.py` | `cloak.profile_match` → `cloak.lattice.profile_match` |
| `src/cloak/anonymity.py` | `src/cloak/lattice/anonymity.py` | `cloak.anonymity` → `cloak.lattice.anonymity` |
| `src/cloak/lattice_producer/*` (15 files) | `src/cloak/lattice/producer/*` | `cloak.lattice_producer.X` → `cloak.lattice.producer.X` |

Producer submodules moved wholesale: `coherence` `counts` `coverage` `entity_merge` `gates`
`graph` `io` `merge` `propose` `queue` `reference_sources` `repair_queue` `state` `vocabulary`.

### `cloak/qa/` (all out of the retired training subpackage)

| old | new | module path |
|---|---|---|
| `train/qa_builder.py` | `qa/builder.py` | → `cloak.qa.builder` |
| `train/qa_scoring.py` | `qa/scoring.py` | → `cloak.qa.scoring` |
| `train/qa_teacher.py` | `qa/teacher.py` | → `cloak.qa.teacher` |
| `train/qa_freeze.py` | `qa/freeze.py` | → `cloak.qa.freeze` |
| `train/qa_review.py` | `qa/review.py` | → `cloak.qa.review` |
| `train/qa_audit.py` | `qa/audit.py` | → `cloak.qa.audit` |
| `train/relation_support_gate.py` | `qa/relation_support_gate.py` | → `cloak.qa.relation_support_gate` |

### `cloak/reward/`

| old | new | module path |
|---|---|---|
| `train/roundtrip.py` | `reward/roundtrip.py` | → `cloak.reward.roundtrip` |
| `src/cloak/extract.py` | `reward/extract.py` | → `cloak.reward.extract` |
| `train/utility_cache.py` | `reward/utility_cache.py` | → `cloak.reward.utility_cache` |
| `train/utility_credit.py` | `reward/utility_credit.py` | → `cloak.reward.utility_credit` |

### `cloak/ranker/`

| old | new | module path |
|---|---|---|
| `train/ranker_environment.py` | `ranker/environment.py` | → `cloak.ranker.environment` |
| `train/ranker_representation.py` | `ranker/representation.py` | → `cloak.ranker.representation` |
| `train/semantic_ranker.py` | `ranker/semantic.py` | → `cloak.ranker.semantic` |
| `train/ranker_privacy.py` | `ranker/privacy.py` | → `cloak.ranker.privacy` |
| `train/interactive_ranker.py` | `ranker/interactive.py` | → `cloak.ranker.interactive` |
| `train/profile_count.py` | `ranker/profile_count.py` | → `cloak.ranker.profile_count` |
| `train/lambda_menu.py` | `ranker/lambda_menu.py` | → `cloak.ranker.lambda_menu` |
| `train/ranker_diagnostics.py` | `ranker/diagnostics.py` | → `cloak.ranker.diagnostics` |
| `train/counterfactuals.py` | `ranker/counterfactuals.py` | → `cloak.ranker.counterfactuals` |
| `train/count_reward.py` (live env-validation core) | `ranker/count_reward.py` | → `cloak.ranker.count_reward` |
| `train/ranker_architecture_diagnostics.py` | `ranker/architecture_diagnostics.py` | → `cloak.ranker.architecture_diagnostics` |

`assemble_action_vector` (+ `legal_action_ids` and their helpers) moved out of the trainer
module into `ranker/environment.py` — the renderer now lives with the data it renders.

### `src/inferdpt` deleted

| old | new |
|---|---|
| `src/inferdpt/probes/` | `src/cloak/probes/` |
| `src/inferdpt/attacks/` | `src/cloak/attacks/` |
| `src/inferdpt/llm.py`, `pipeline.py` | already superseded by `src/cloak/llm.py`, `src/cloak/concurrent.py` |
| `src/inferdpt/__init__.py`, `src/inferdpt/tests/`, RANTEXT core | deleted (recover with `git show`) |

### Deleted, not moved

The retired training subpackage's `ranker.py` (minus `LambdaProfile`), `reward.py` (its four
live names folded into `qa/scoring.py`), `probes.py`, `ladder_probes.py`, `schema_task.py`, the
legacy half of `count_reward.py`; `cloak/{score,synthpai,frozen_extractor,reconstruct}.py`;
`extract.invert_detector_pointer`; `scripts/{train_ranker,reward_gate,build_probes,`
`annotate_lattice_counts,dp_sweep,build_phi_subvocab,build_cnndm_corpus}.py`; the
`latticecloak_*` sweep scripts (except `latticecloak_detection_gate.py`); 61 spikes.
Reconstructor/frozen-extractor track preserved on branch `archive/reconstructor-track`.

### Pin re-baseline (one-time, identity-only)

`invert_implementation_pin()` hashes the source and **module names** of the extractor and
`cloak.runtime_types` into every utility-cache identity, so the extractor's move (plus the
`invert_detector_pointer` deletion) changed the pin exactly once. Module keys are now *derived*
from `__name__` instead of hardcoded, so future moves are self-maintaining. The pre-move cache
was archived to `results/ranker_v2/cache/utility-results.pre-phase4-repin.jsonl` (26 rows) and
repopulated from the 3-doc BC smoke off the LLM disk cache (zero live calls). **Identity moved,
behavior did not:** per-doc `utility` and `count_score` were byte-identical before and after, as
were the BC epoch losses. Cost of the archive: the hash gate now re-derives 3 cache identities
instead of 21 — the archived rows carry the old pin and are not re-derivable by design.
