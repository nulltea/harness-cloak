---
type: dev-log
status: current
created: 2026-07-09
updated: 2026-07-09
tags: [rl, reward-design, training-task, ladder-probes, decision-probes, schema-task, dev-log]
companion: [docs/specs/RL/training-task-env.md,
            docs/issues/2026-07-06-placeholder-gaming-reward-qa-necessity.md]
---

# Dev-log — training-task environment decision (ladder carrier + schema + decision top-tier)

Records the design decision captured in [`docs/specs/RL/training-task-env.md`](../../specs/RL/training-task-env.md)
and its clinical validation. The spec has the full alternatives (approaches A–G), prompts, and
scoring pseudocode; this log is the *why we chose what we chose* + measured validation + the
commit trail. Does not duplicate the spec.

## Decision

Round-trip reward redesigned so truthful generalization beats placeholder **where the task needs
the fact's semantics**, not everywhere. Chosen shape:

- **Carrier = B (granularity-ladder probes, two-channel scoring)** on the unchanged generation
  task. Per span, one probe per lattice rung; the exact tier scores on `out_final` (echo is
  legitimate utility for copy-facts — user decision 2026-07-08), coarser tiers score on `out_p`
  *before inversion*, where a placeholder structurally cannot earn. Reward becomes smooth and
  monotone in generalization depth, per-span credit stays exact, probe density ~2–3×.
- **+ C's schema prompt for clinical/lexsum** — the note/summary format is forced to emit
  category-level fields (`problem — category — status`), so the output itself requires the
  semantics that make generalization pay; deterministic field parsing offloads the reader; the
  given template absorbs gemma's collapse-cliff failure mode.
- **+ D (downstream-decision probes) as the ladder's top tier** where the teacher finds a
  ceiling-stable decision — the purest task-necessity signal, `depends_on` quotes tag supporting
  spans so per-span credit survives.

Rejected as headline: **E (QA-as-task)** — train/deploy skew + dissolves the task-execution
novelty; kept only as a small diagnostic arm. **A (status quo)** stays as the baseline arm.

**Root problem it fixes** (see [placeholder-gaming issue](../../issues/2026-07-06-placeholder-gaming-reward-qa-necessity.md)):
existing probes put the category in the *question* and demanded the surface as the *answer*
("Which endocrine disorder…?" → "hypothyroidism") — exactly the shape echo+invert games. The
ladder flips it: surface-neutral question, category-level gold ("What class of condition…?" →
"an endocrine condition"), with the lattice supplying both the gold and the acceptance set
(`entail_score` — deterministic, no NLI).

## Validation (clinical samples, 3 ACI docs, 15 lattice-bearing spans)

Teacher-model comparison for probe generation (`scripts/spikes/ladder_probe_gen_test.py`):

| teacher | rung survival | spans covered | decisions/doc | verdict |
|---|---|---|---|---|
| Qwen3.6-35B-A3B (pinned) | 84% | 15/15 | 4.0 | use it |
| claude-haiku-4-5 (paid proxy) | 78% | 15/15 | 4.0 | works, no gain over free local |
| gemma 4 (E4B) | 89% pre-filter | 15/15 | 4.0 | barred (reward-model family) |
| LFM2.5-8B-A1B | 0% (unparseable) | 0 | 0 | unusable, confirms 2026-07-05 demotion |

Question quality matched the design intent — ladder golds are category-level, decision probes are
genuinely clinical with correct dialogue-quoted `depends_on`. One fix landed mid-validation: an
`_empty_gold` filter drops rungs whose coarsest fill is vacuous ("something", "a disorder") — such
fills should earn no semantic-tier credit over a placeholder (84%→ from 95% pre-filter; all spans
still covered).

**Connection to the context ablation** (`research-wiki/experiments/context-injection-surface-ablation.md`):
that re-run showed surface-recall own-recall is flat at *both* baseline extremes (~30%
action-dependent either way) — which is precisely the weakness the ladder's graded semantic tiers
are built to remove. The two threads converge: the reward signal is the leverage, not the
marginal baseline or the context encoder.

## Commits

| commit | what |
|---|---|
| `9f1eaae` | task-necessity probe design spec (`training-task-env.md`) + ladder/decision generator (`ladder_probes.py`) + validation spike (`ladder_probe_gen_test.py`) |
| `0884825` | refactor: LLM helpers moved into the `cloak` namespace (touches `ladder_probes.py`) |

(Adjacent, same reward-env cleanup but tracked separately: `e326de1` ranker features 30→24;
`b2a5085` ceiling-baseline ablation re-run.)

## Status / next

- Spec + generator + teacher validation: **done**.
- **Not yet wired** into `scripts/build_probes.py` — the anchor validation (per-tier
  ceiling/floor with `entail_score`), shuffled-options MC scoring for decision probes, and a
  cross-span locator lint (reject rung ≥1 questions naming another detected surface, since those
  are scored on `out_p` where other spans are anonymized) are the remaining build work.
- Blocked-adjacent: any pilot run inherits the reward non-determinism finding
  ([handoff](../../handoffs/2026-07-09-reward-nondeterminism-parallel-roundtrips.md)) — resolve
  that before trusting a ladder-reward support scan.
