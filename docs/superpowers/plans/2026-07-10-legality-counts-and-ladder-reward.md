# Legality-mask per-level counts + ladder-reward wiring — implementation plan

Companion track to `docs/superpowers/plans/2026-07-10-reward-determinism.md` (same loop, same
final review). Covers issue register §2 residual (per-level counts → legality mask) and the
approved training-task redesign build work (`docs/specs/RL/training-task-env.md` §Probe
generation, §Reward assembly; dev-log `docs/dev/logs/2026-07-09-training-task-env-decision.md`).

## State (verified 2026-07-10, do not re-derive)

- `src/cloak/lattice_profiles.py::_build_indexes` ALREADY routes explicit `level_counts` into
  `lookup_count` with a max-merge across surfaces (landed in `9f1d0da`). The register's
  "uncommitted row-sum change" criticism is stale — row-sum survives only as the **legacy
  fallback** for levels with no explicit count (`level_index[key] += count`), which still
  double-counts across surfaces sharing a level.
- `aset_count` (`src/cloak/anonymity.py`) consumes `lookup_count` for FINE_DEM/domain types;
  fail-closed to 1.0 on None.
- Runtime artifact `data/lattice_profiles/lattice_profiles.json` coverage: drug 538/539 rows
  carry `level_counts`, health-condition 201/771, medical-procedure 0/488, all others 0.
  Coverage expansion = the pending proposed-artifact merge decision (user-owned, NOT here).
- Ladder machinery (`src/cloak/train/ladder_probes.py`) is standalone: generators, prompts,
  `entail_score`, `lint_rung`, `_empty_gold`, caches `data/ladder_probes.json` /
  `data/decision_probes.json` (files not yet produced). `scripts/build_probes.py` builds only
  the flat validated set with ceiling/floor anchors at TH=0.5. `src/cloak/tasks.py` has
  free-form prompts only (`TASK_TEMPLATE`).

## Global constraints

Same as the determinism plan: no GPU/proxy/network in tests (mock all LLM clients; tmp cache
dirs); `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests -q`; minimal diffs; no new
dependencies; **path-scoped commits only** (shared checkout — `git add <files> && git commit
-- <files>`, check `git diff --cached --name-only` first, never bare commit; retry on
index.lock). Producing the real probe artifacts / anchors is user-gated GPU work — code and
tests only.

Reward weights (pre-registered here, flagged for user confirmation before any pilot):
`w_exact = 0.5`, `w_sem = 0.5` per span; document-level combine = unweighted mean over the
available components (span parts mean, decision mean, schema score). Constants live in code,
never tuned per model/corpus (empirical-honesty rule).

## Concurrency map (which tasks may run in parallel)

- **legality-counts** (this file) — touches only `src/cloak/lattice_profiles.py` + its tests:
  parallel-safe with everything.
- **schema-task** — new module + `src/cloak/tasks.py` additions: parallel-safe with the
  determinism track (which never edits tasks.py).
- **probe-build wiring** — `scripts/build_probes.py`, `src/cloak/train/ladder_probes.py`:
  parallel-safe with the above.
- **two-channel reward** — edits `src/cloak/train/reward.py`, `src/cloak/train/roundtrip.py`,
  `scripts/train_ranker.py`: MUST wait for determinism Task 2 (reader refresh threading) and
  Task 3 (winner re-verification) to land, plus schema-task and probe-build (it consumes their
  interfaces).

## Task legality-counts — per-level semantics: kill the legacy row-sum

File: `src/cloak/lattice_profiles.py`. Tests: extend/new under `src/cloak/tests/`.

- In `_build_indexes`, change the legacy fallback aggregation for levels without an explicit
  `level_counts` entry from row-sum (`level_index[key] += count`) to **max**
  (`max(existing, count)`), matching the explicit branch's member-set union/max semantics
  (issue register §2: "member-set union/max, never row-sum"). Update the adjacent comment.
- Tests pinning the full read path (monkeypatch `DEFAULT_PROFILE_PATH` or pass `path=`; clear
  `_load_cached`/`_index_cached` lru caches per test):
  1. explicit `level_counts` value reaches `lookup_count` (not the row `count`);
  2. two surfaces sharing a level, both explicit → max wins;
  3. two surfaces sharing a level, no explicit counts → max of row counts (NOT the sum);
  4. explicit beats legacy when one surface has `level_counts` and another doesn't;
  5. end-to-end: `aset_count(fill, "drug"|"health-condition", orig, strict=True)` returns the
     per-level count for a profile-backed level (monkeypatched artifact), and 1.0 fail-closed
     for an unknown fill.

## Task schema-task — schema prompts + deterministic section parser + field grader

Files: `src/cloak/tasks.py` (prompts only), new `src/cloak/train/schema_task.py`, new test
file. Spec: `docs/specs/RL/training-task-env.md` §C (schema-constrained clinical task),
§"Schema task — no generated questions".

- `tasks.py`: add `SCHEMA_NOTE` (clinical) exactly per the spec's prompt (CHIEF COMPLAINT /
  HISTORY OF PRESENT ILLNESS / ASSESSMENT "problem — category — status" / PLAN "problem —
  action — follow-up"; "none" for missing sections) and `SCHEMA_CASE` (lexsum analogue:
  PARTIES / CLAIMS one line per claim "claim — category — status" / OUTCOME "claim — remedy —
  posture"). Add `SCHEMA_TEMPLATE = {"aci": SCHEMA_NOTE, "mts": SCHEMA_NOTE, "clinical":
  SCHEMA_NOTE, "lexsum": SCHEMA_CASE}` and `SCHEMA_CORPORA = frozenset(SCHEMA_TEMPLATE)`.
  Do NOT modify `TASK_TEMPLATE` — the carrier chooses at wiring time (two-channel task).
- `schema_task.py`:
  - `parse_sections(text) -> dict`: deterministic parser; section headers case-insensitive,
    tolerant of `:`/newline variants; ASSESSMENT/PLAN rows split on the em/en/hyphen dash
    into named fields (`problem`, `category`, `status` | `problem`, `action`, `follow_up`);
    missing/`none` sections → empty; never raises on malformed text (returns what it parsed).
  - `schema_field_score(out_final_text, out_hi_text, acceptance_sets=None) -> float | None`:
    parse both; align ASSESSMENT/PLAN rows on `canon`'d problem names (reuse
    `cloak.train.reward.canon`); per aligned row score fields with
    `cloak.train.reward.fact_score`; when `acceptance_sets` supplies a per-problem rung
    acceptance list, score the `category` field as max fact_score over that list (the
    entail_score rule); mean over scored fields; None when the ceiling parse has no rows
    (excluded, mirroring probe-less docs).
- Tests: parser on well-formed, missing-section, malformed, and "none" outputs; field grader
  alignment (row order permuted; extra/missing problems), category acceptance-set scoring,
  None on empty ceiling.

## Task probe-build — wire ladder + decision generation into the probe build

Files: `scripts/build_probes.py`, `src/cloak/train/ladder_probes.py`, tests. Spec:
`training-task-env.md` §Probe generation (LADDER_PROMPT/DECISION_PROMPT pv 1 already in code).

- `ladder_probes.py` additions:
  - `locator_lint(q, span_surface, other_surfaces) -> bool`: reject rung ≥ 1 questions whose
    canon tokens contain another detected span's surface (they score on `out_p`, where other
    spans are anonymized — dev-log "cross-span locator lint"). Applied at generation time.
  - `validate_ladder(entries, reader_hi, reader_lo, th) -> (kept, report_rows)`: per-rung
    anchor validation — keep rung ℓ iff `entail_score(reader_hi(q), rungs, ℓ) >= th` AND
    `entail_score(reader_lo(q), rungs, ℓ) < th`; reader callables injected (the build script
    binds them to the pinned reader over `out_hi`/`out_lo`; tests use fakes).
  - `validate_decisions(entries, reader_mc_hi, reader_mc_lo) -> (kept, report_rows)`: keep
    iff the hi pick equals gold and the lo pick differs or abstains; `mc_shuffle(options,
    seed_key)` helper for per-call seeded option shuffling (used by validation now and by the
    reward later); `depends_on` matched to detected spans by canon substring → `span_ids`
    tag, span-free probes kept but tagged.
- `build_probes.py`: `--ladder` mode — for each doc with cached anchors (`out_hi`, `out_lo`,
  reusing the existing anchor machinery), run generation (cached teacher calls) + lints +
  per-rung validation; write `data/ladder_probes.json` + `data/decision_probes.json` (the
  generators' own cache format, validation verdicts added per entry) and extend the
  probe-health report with per-corpus `reader_rung_reject_rate`, tiers/span kept, decisions
  kept/doc. Rung-0 reuse: where `data/probes_validated.json` has a validated probe for the
  fact, mark the rung-0 entry as sourced from it (spec: teacher rung-0 used only as fallback).
- Tests (all readers/teachers faked): rung validation keep/drop matrix (ceiling-fail,
  floor-pass, both-pass), locator lint drops the cross-span question, mc_shuffle determinism
  per seed and variation across seeds, decision span-tagging, health-report numbers.
- DO NOT run the build (teacher + anchors = proxy/GPU; user-gated).

## Task two-channel-reward — R_carrier (echo + semantic + decisions + schema)

Files: `src/cloak/train/reward.py`, `src/cloak/train/roundtrip.py`, `scripts/train_ranker.py`.
BLOCKED BY: determinism Task 2 + Task 3, schema-task, probe-build. Spec: `training-task-env.md`
§Reward assembly (R_carrier pseudocode) — implement it verbatim modulo names:

- `roundtrip_batch` job schema gains optional `ladder` (per-span rung entries), `decisions`,
  `out_hi` (ceiling text for schema scoring), `schema` (bool). Per job:
  - echo channel: rung-0 questions via `fact_f1s(out_final, rung0_probes)` (existing path —
    unchanged semantics for jobs without ladders);
  - semantic channel: rung ≥ 1 questions read against **out_p before inversion**
    (`_read_batch(qs, out_p)`), scored `entail_score(answer, rungs, rung)`;
  - per-span part = `w_exact * exact + w_sem * sem` (constants from this plan's Global
    constraints); doc reward = mean over parts, then unweighted mean with decision score
    (`mc_score`: reader picks from `mc_shuffle`'d options against out_final; score 1/0) and
    `schema_field_score(out_final, out_hi)` when the job's `schema` flag is set.
  - Jobs lacking ladder/decisions degrade to today's recall (baseline arm A stays runnable —
    spec §A "keep as the measured baseline arm").
- `train_ranker.py`: env/probe loading accepts the two new artifacts when present
  (`--ladder-probes`, `--decision-probes`), threads them into jobs; absent → legacy behavior
  bit-identical.
- Tests: two-channel arithmetic on faked reader outputs (placeholder earns echo-only;
  generalized fill earns semantic tier; exact tier on out_final not out_p; semantic tier on
  out_p not out_final), degradation path equals legacy recall, decision mc scoring with
  shuffled options, combine weights as pre-registered.

## Sequencing

`legality-counts` ∥ `schema-task` ∥ `probe-build` ∥ (determinism Task 2 → 3 → 4); then
`two-channel-reward`; then the loop's single final whole-branch review (both plans, one
verdict). GPU-gated afterwork (user-owned): probe artifacts build, determinism probe run,
fresh-cache support-scan re-run on the ladder reward.
