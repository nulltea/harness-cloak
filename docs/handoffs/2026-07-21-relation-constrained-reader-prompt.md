---
type: handoff
status: current
created: 2026-07-21
updated: 2026-07-27
tags: [qa, reader, prompt, context-reader, reader-pin, relation-qa, finer-level, re-gate]
companion: docs/specs/qa-builder-v2.md
---

# Handoff — relation-constrained context-reader prompt

## Objective

Implement, carefully and across ALL relation-QA shapes, a **relation-constrained context reader**:
the reader prompt gains one line that restates the QA as the relation the ANSWER must satisfy,
naming the **locator** argument by its **exact rendered fill**. Empirically this makes the small
local reader extract the right span without reasoning — it "runs the reader for us" by restating
the question as a constraint. Validated as sound on the diagnostic cases (below); the task is a
correct, general, regression-safe implementation + the pinned re-gate it entails.

This is a **pinned reader change** (`qa-context-reader-v3` → next revision) → invalidates
`reader_pin` / `gate_manifest_hash` → requires a re-gate. Pins are transitive; treat as such.

## The finding (settled) and why it works

The finer-level readability check (opt-in `--reader-finer-level-check`, added this session) surfaced
kept relation QAs the production reader mis-reads: it grabs a lexically-resonant DISTRACTOR span
instead of the answer. Root example (E1, D2N002, `tests_for`): question "Which test was ordered to
evaluate the [primary immunodeficiency disorder]?"; at the answer fill "autoimmune serology testing"
the reader returned a white-blood-cell clause near the subject (0.0), because "autoimmune" resonates
with the "immunodeficiency" in the question. Same doc, same question, only the answer fill differs
between the passing (supported) and failing (finer) render.

A/B over 4 prompt variants (spike `scripts/spikes/reader_prompt_ab.py`, fresh no-cache reads with
the REAL production reader `read_context_batch`):

- **V0 (current)** — "complete the REQUEST … Copy the answer span exactly". Baseline.
- **V1** — REQUEST→QUESTION only. **No effect** (byte-identical answers to V0). The user's initial
  "REQUEST is the culprit" hypothesis is refuted.
- **V2** — adds "shortest exact span". **Regresses** — shortens but still picks the wrong sub-span,
  and on adjacent-entity renders ("[heart disease] with diastolic dysfunction") it grabs the tail.
  Do NOT use "shortest".
- **V3/V4** — adds a relation-constraint line naming the locator. **Fixes E1** and is the only lever
  that works.

Correct V4 wording (keep V0's "copy the answer span exactly", QUESTION not REQUEST, NO "shortest"):

```
Read the DOCUMENT and answer the QUESTION. Copy the answer span exactly from the DOCUMENT —
do not rephrase, summarize, or add words.
If the DOCUMENT does not answer it, reply with exactly NONE.
Your ANSWER must satisfy the relation: ANSWER <clause naming the LOCATOR by exact rendered fill>.

DOCUMENT:
{context}

QUESTION: {question}
```

**Regression audit (spike `reader_relconstraint_regression.py` + `_detail.py`):** over 56 kept 2-arg
relation QAs at supported render, V4 showed 7 "regressions" (V0-pass→V4-fail), 2 gains, 47 unchanged.
Per-case inspection (`results/qa_v2_stage_ab/relconstraint_detail.log`) shows **none is a clean
regression**:
- ~3 are **V0 circular/bogus passes** — reverse QAs render the answer at its gold level INSIDE the
  context, so V0 "passes" by echoing that sentence without reasoning; V4 declines. Most striking:
  the **diuretic ↔ colon-cancer false relation** that V0 rubber-stamps and V4 correctly rejects (0.0).
- 1 is a **junk QA** (a `tests_for` whose locator literal is a whole verb phrase "start you on some
  lasix 40 milligrams once a day") — malformed by construction, not a fair test.
- ~3 are **mis-attributed / ambiguous relations** (e.g. blood work that is actually for anemia paired
  with heart disease; multiple procedures for one condition) where V4's divergent answer is defensible
  and V0's pass is again a circular echo.

Conclusion: implemented correctly, V4 is a **precision improvement, not a regressor**; the raw
"regression count" was measuring V0's circular passes. It works as intended.

## Implementation traps (do all of these)

1. **Locator = the NON-answer argument, orientation-aware.** Forward (`answer_role=object`): locator
   = subject. Reverse (`answer_role=subject`): locator = object. Never name the answer argument
   (naming it leaks the answer). Reverse swaps subject↔object — the single most error-prone spot.
2. **Locator string = EXACT rendered fill, not `support_property`.** The question's locator level and
   the doc_p-rendered level can differ; if the clause names a string not verbatim in the context the
   reader hunts for something absent. Use the locator decision's pinned-level `action.fill` (the
   string the render substitutes), or the context literal verbatim. See `rendered_fill()` in
   `scripts/spikes/reader_relconstraint_regression.py`.
3. **Clause map for all 5 relations × both orientations** (`prescribed_with`, `procedure_for`,
   `tests_for`, `contraindicated_because_of`, `causes_or_explains`). Draft map in the spike's
   `CLAUSE` dict — refine wording (author by hand, opus; NOT Codex, per repo rule on prompt work).
4. **Compound rows are NOT yet designed.** Set-valued (`linked_decision_set`), compound-span-reverse,
   and multi-literal rows have MULTIPLE locators; the clause must list them (or the constraint be
   restated per member). Scope this explicitly — the spikes only cover 2-arg cases.
5. **Garbage literal locators** (junk QAs) must degrade gracefully — a malformed clause shouldn't
   crash; it's acceptable for such a QA to fail.
6. **Both reader entry points must change consistently:** the gate reader
   (`validate_context_assertions` via `BatchedContextReader._read_one`, `qa_builder.py` ~L629) AND the
   runtime scorer (`score_utility`, and `roundtrip.py`). Same prompt, or gate/runtime disagree.
   The clause needs the assertion's relation + arguments + rendered locator at read time — the reader
   currently only gets `(question, context)`, so plumbing the relation/locator into the read call is
   part of the work.
7. **Set-valued reader** (`read_context_set_batch` / `_read_set_one`) is a separate prompt — decide
   whether the relation constraint applies there too.
8. **Pin + re-gate:** bump `CONTEXT_READER_PROMPT_VERSION`, update `DEFAULT_CONTEXT_READER_PIN` and
   the threshold manifest `reader_pin`; a full re-gate follows (invalidates trained policies).

## Deeper confound to weigh (bigger than the prompt)

Reverse QAs render the answer at its gold level IN the context, so ANY reader can pass circularly by
echoing that sentence — V0 does, V4 resists. This means the gate itself is partly gameable by the
reader copying the rendered fill. The relation constraint mitigates it, but the clean fix may be in
how reverse QAs are scored/rendered (don't render the answer decision at its gold level in the
reader excerpt, or score against a held-out rendering). Decide whether to address alongside the
prompt or track separately — it affects the validity of the whole context-reader gate, not just
these cases.

## Validation plan

- Re-A/B V0 vs the final V4 across the **full 67-doc kept relation set** (not just 5 docs) with the
  regression harness; require zero *clean* regressions (adjudicate any V0-pass→V4-fail as circular /
  junk / mis-attributed, as done here).
- Confirm the finer-level worklist's reader-artifact entries (E1-class) flip to pass under V4.
- Confirm false relations stay rejected (E7 diuretic↔colon-cancer must remain 0.0 — V4 already does).
- All local reader, 0 paid. GPU: one process at a time; `rocm-smi --showpidgpus` + ask before runs.

## Related work / state

- **Chain-order fix is uncommitted and should land first or alongside.** `_frozen_semantic_chain`
  now orders the entailment ladder by the decision's AUTHORED profile ladder (load-validated monotone
  in the profile's own `level_counts`) instead of re-sorting by the GLOBAL `coarseness_rank`/`aset`
  (which is miscalibrated cross-profile — global "heart disease"=400 > "thoracic disease"=390,
  inverting organ vs region). Fixes a reward-gradient inversion; validated: 5/5 cardiovascular/GI
  finer-level worklist entries fixed, **77 decisions'** chains corrected, kept set unchanged, 922
  tests pass. It re-freezes at ARMS build, so full propagation needs an arms+env rebuild (0 paid —
  teacher prompts are chain-independent, stay cached). File: `src/cloak/qa/builder.py`
  `_frozen_semantic_chain` + regression test `test_semantic_chain_follows_authored_ladder_not_global_aset`
  in `test_qa_builder_v2.py`. A corrected-chain env exists at
  `results/qa_v2_stage_ab/ranker-env-chainfix2.json`.
- **Last commit:** `87cbcc3` (causes_or_explains premise gate + finer-level soft/hard modes).
- **`aset_count` is live, not legacy** — feeds ranker policy feature, BC target selection, hiding /
  representative-anchor selection. Do NOT change `aset_count` for this work; the chain-order fix
  deliberately stops the CHAIN from using it, nothing else.

## Key lessons (carry forward)

- **Validate with the EXACT production reader.** The context reader is `BatchedContextReader`
  (`read_context_batch`, "Read the DOCUMENT / QUESTION" prompt). `reward._read_batch` (`QA_PROMPT`,
  "shortest exact answer … a name/phrase") is a DIFFERENT reader — using it by mistake produced a
  wrong intermediate conclusion (E1 "passes") this session. Never A/B a lookalike.
- Judge by scores/counts; delegate any raw clinical-output reading to an Opus subagent (dual-use
  classifier). The reports in `results/qa_v2_stage_ab/*.md` hold the concrete specifics.

## Artifacts

- Spikes (`scripts/spikes/`): `reader_prompt_ab.py`, `reader_relconstraint_regression.py`,
  `relconstraint_regression_detail.py`, `finer_level_reader_answers.py`, `e1_ladder_reads.py`,
  `remaining_worklist_trace.py`, `finer_level_reader_answers.py`.
- Analysis (`results/qa_v2_stage_ab/`): `finer_level_failures_analysis.md`,
  `finer_level_failures_for_fable.md`, `relconstraint_detail.log`, `reader_prompt_ab.log`,
  `relconstraint_regression.log`; corrected-chain env `ranker-env-chainfix2.json`.
