---
type: handoff
status: current
created: 2026-07-06
updated: 2026-07-06
tags: [extractor, reconstructor, design3, invert, survived-recovery, subagent-driven]
companion: [../plans/2026-07-06-reconstructor-design3-plan.md,
            ../plans/2026-07-05-survived-recovery-extractor.md,
            ../../research-wiki/experiments/extractor-pointer-by-type.md]
---

# Handoff: implement the Design 3 reconstructor plan (subagent-driven)

Next session's job: **execute `docs/plans/2026-07-06-reconstructor-design3-plan.md`
task-by-task** using `superpowers:subagent-driven-development` (fresh subagent per task,
two-stage review between tasks; subagents `model: "opus"` per user rule). The plan is
Codex-reviewed and execute-ready (see below). Do NOT re-plan or re-derive the numbers here.

## Read first
- **Plan (the spec to execute):** `docs/plans/2026-07-06-reconstructor-design3-plan.md` — 6
  bite-sized TDD tasks, complete code in every step, global constraints at top.
- **Why this design / the measured ground truth:**
  `research-wiki/experiments/extractor-pointer-by-type.md` (151-doc survival + recovery) and
  `docs/plans/2026-07-05-survived-recovery-extractor.md` (deterministic proposers, companion).
- **Review trail:** `review-stage/AUTO_REVIEW-reconstructor-design3.md` (gitignored, local
  only) — 4 Codex rounds, 6→8, every fix folded into the plan. Read it before touching the
  guard/admission/eval code so you don't undo a fix.

## What this session established (don't re-derive)
- **Corrected denominator:** of 1059 generalized spans (151 clinical+lexsum docs), **293
  have their substituted content reach out_p** ("survived"); 74 are "leaked-only" (the
  original leaked via an undetected doc_p duplicate — a *privacy* leak, out of extractor
  scope). The current cascade recovers ~239/293 ≈ **82%**. Residue = 54 "D-reworded" spans.
- **Reframe (load-bearing):** extraction is client-side and R holds every original, so the
  residue is a **localization** problem, not lost information. Ceiling is ~100% of survived
  spans, bounded only by localization precision + false-match avoidance. D-3 lossy
  (date→decade) IS recoverable; D-4 (model re-derived a *different* specific) is the
  abstain/false-match boundary.
- **`invert()` default changed** (commit `97c26af`): semantic-window matcher (Design 1) is
  now the committed default cascade (was exact+fuzzy only). **This is a re-gate event** —
  cached anchors/probes/scan verdicts/policies are stale against the new extractor; rebuild
  before any durable RL run. The Design 3 plan builds on this new `invert()`/`_rule_prepass`.

## Execution notes / gotchas
- **GPU tasks serialize with the parallel detector session.** Plan Tasks 3 (data build), 4
  (train), 6 (eval) are GPU/proxy; the detector work runs in parallel sessions. Each GPU
  task MUST `pgrep -af train_pii` first and wait — one GPU process at a time. Tasks 1, 2, 5
  are pure-code (no GPU) and can proceed anytime.
- **Working tree has uncommitted parallel-session WIP** (`src/cloak/corpora.py`,
  `tasks.py`, `train/reward.py`, several docs, many untracked spikes). **Do not commit
  those** — commit only files your task creates/edits (`src/cloak/reconstruct.py`,
  `tests/test_reconstruct.py`, `scripts/build_reconstructor_data.py`,
  `scripts/train_reconstructor.py`, `scripts/spikes/reconstructor_eval.py`, the training
  record).
- **Verified deps this session:** `transformers==5.12.1`, `peft==0.19.1`, torch ROCm
  (CUDA-available), `sentence-transformers 5.6.0`, rapidfuzz. Install-free.
- Proxy at `http://localhost:8060/v1` serves the gemma out_p pin and Qwen judge (`-np 1`,
  serial). out_p roundtrips are cached (`INFERDPT_LLM_CACHE=data/llm_cache`).
- Codex reviews on this box: MCP can't read local files (paste content); `codex exec` needs
  `--sandbox danger-full-access` + `< /dev/null`. The plan-review thread already closed.

## Key correctness points the reviewer forced (keep these invariants when implementing)
- **Admission gate** (`restorable`): NLI direction is `fill ⊨ quote` (NOT the reverse);
  scalar gate `_value_compatible` is **fail-closed** (never returns True). These reject D-4;
  a bug in either silently trains false restorations.
- **`edit_guard`**: fill-occurrence anchored + tight surface match + ambiguous-anchor bail.
  Every accepted edit must be an in-place replace of a located fill mention with THAT
  entry's surface. This is the do-no-harm boundary.
- **`classify_recovery`**: window-local at the relocated quote; plus doc-level `harm_rate`
  hard-fail. Success requires no hard-fail on EVERY eval stratum, never averaged.

## Suggested skills for next session
- `superpowers:subagent-driven-development` (primary — execute the plan)
- `superpowers:verification-before-completion` (before claiming any task/the pilot passes)
- `superpowers:test-driven-development` (each task is already TDD-structured)
- After the training run: write/complete `research-wiki/training/2026-07-06-FT-reconstructor-v1-residue-edit.md` (spec-then-results), and re-run `/auto-review-loop` on the *results* (not the plan) once eval numbers exist.
