---
type: research
status: stale
created: 2026-07-11
updated: 2026-07-27
tags: [rl, qa-build, ladder-probes, decision-probes, teacher-prompt, question-design, issue]
companion: [docs/specs/RL/training-task-env.md,
            docs/issues/2026-07-06-placeholder-gaming-reward-qa-necessity.md,
            docs/issues/2026-07-10-near-duplicate-condition-lattice-entries.md]
archive_reason: subject retired in 2026-07-27 cleanup (see docs/plans/2026-07-27-codebase-cleanup-refactor.md); the ladder/decision probe tier and its
  build script were deleted — the live QA build is docs/specs/qa-builder-v2.md
---

# Issue register — ladder/decision QA-build: teacher-question-design failures

The ladder QA-build (`scripts/build_probes.py --detect`) is validated and mechanically clean —
it produces semantically-grounded probes with **zero parse failures** on both teacher models
(super/ultra sweep, 2026-07-12). What remains is a set of **question-design quality issues**:
the generated probes do not reliably *require the protected fact*, so the reward can be satisfied
without preserving it (placeholder-gaming resurfacing at the question-quality level). The
decision tier is the worst case (world-knowledge trivia). Issues are logged below, most important
first; Issues 3–4 are the next-session teacher-prompt redesign.

## Context — validated build state (2026-07-12)

- **QA source of truth is `data/lattice_profiles/lattice_profiles.json`**, not the stale
  `ranker_env_*.json`. `span_levels` reads levels from the profile; `build_probes --detect`
  freshly detects spans (`knowledgator/gliner-pii-large-v1.0`, native PII labels →
  runtime_type, lattice/placeholder/quasi roles), levels + counts from the profile.
- **Teacher = `nvidia/nemotron-3-super-120b-a12b:free`** (OpenRouter). Sweep vs
  `…ultra-550b…:free` showed **super is the better teacher**: more kept spans (8 vs 6), zero
  gold-leak lint rejects (ultra 13), and the only surviving decision. ultra's extra size does
  not help; it leaks golds and produces no floor-discriminative decision.
- **Fix stack landed** (all committed): cache-scoping (`_reusable`, stale `dragon` leak),
  clinical-role grounding constraint (no positional/enumeration), negation/screening detection
  filter, `reasoning=exclude` + `response_format=json_object` + **no `max_tokens` cap** (the
  reasoning-in-content truncation fix — this is what took ultra from 9/15 unparseable to 0),
  alias acceptance set, decision retarget. `LADDER_PV=3`, `DECISION_PV=3`.
- **Latest validated sweep**: `results/ladder_generations.{super,ultra}.json`,
  `data/probes_ladder_validated.{super,ultra}.json`, `results/ladder_gen_rejects.{super,ultra}.json`.
  super: 15 lattice spans → 50 rung candidates → 15 kept (6 semantic-tier spans), 0 parse fails,
  1 decision kept. Reject is ceiling-dominated (reader-capability / duplicate-surface).

---

## Open issues (most important first)

### 1. Decision probes are trivia or plan-readbacks — the tier is not doing its job

**What it is.** The decision tier (`DECISION_PROMPT`, spec's D tier) is meant to test
*task-necessity*: a care decision whose correct answer needs the fact's meaning **as documented
for this patient**, so a placeholder breaks it and a truthful generalization preserves it.

**What's actually happening.** Two failure shapes, and the validation selects the wrong one:

- **World-knowledge trivia (kept, but useless).** The one decision that survived pv3 is
  ```
  Q: Which body system does a mammogram primarily evaluate?   gold: Reproductive   depends_on: ['mammogram']
  ```
  It **names the fact** ("a mammogram") and asks a **generic property of it**, so a competent
  reader answers it *without the note*. It tests "does the reader know what a mammogram is,"
  not "did the output preserve this patient's information." (Also the gold is wrong — breast is
  not cleanly reproductive-system — but that's secondary.) This is the failure the
  2026-07-11 `DECISION_PROMPT` retarget *introduced*: pushing toward "which body-system
  category / which specialist does [named fact] belong to" collapsed decisions into
  decontextualized lookups that also duplicate the ladder's semantic tier.

- **Plan-readbacks (floor-rejected).** The pre-retarget shape ("what change to lisinopril is
  recommended?" → "increase to 40mg") reads the answer off the plan line; it survives
  anonymization (the dose isn't sensitive), so the floor answers it too → dropped. Correct to
  drop, but it means the tier yields ~1/doc at best on clinical.

- **Perverse validation dynamic (root of the selection problem).** Genuinely note-necessary
  decisions *fail* because the pinned MC reader (`Qwen3.5-0.8B`) can't reason them. Measured:
  ```
  Q: Given the patient's kidney transplant and stable kidney function, is Ultram an appropriate
     analgesic for the acute arthritis exacerbation?   → ceiling-rejected (reader mis-pick)
  ```
  This is a *good* probe (needs two documented facts about THIS patient; a placeholder for the
  transplant breaks it) — and it's exactly what validation throws away, while it keeps the
  mammogram trivia. So the tier is currently anti-selecting quality.

**Requirements a decision probe must meet (that the current prompt does not enforce).**
1. **Not answerable from world knowledge** — the answer must depend on what the note documents
   about *this* patient; replacing that content with a placeholder must make it unanswerable.
2. **Do not name the target fact and ask its generic property** — that is trivia. Identify the
   situation by the patient's documented circumstances; the decision is a care action *for this
   patient*.
3. **Not a verbatim plan/dose/interval readback** (the original failure) — but the fix is *not*
   "ask a category"; it is "ask a patient-specific care decision the note supports."

**Same root as Issue 3** — decisions are the extreme case of "questions are trivially answerable
without needing the fact"; resolve the decision-prompt rethink as part of that redesign.

**Open design decision (needs a human call — do NOT just patch the prompt again):**
- (a) Rewrite `DECISION_PROMPT` toward patient-specific, note-necessary, non-named decisions and
  accept that many will ceiling-reject under the small reader.
- (b) Pair the decision tier with a **stronger reader** for validation/scoring (the small MC
  reader is the binding constraint on good decisions).
- (c) **Shelve the decision tier** as low-yield on clinical (the spec already treats it as a
  minority top-tier) and proceed with the ladder's semantic tiers as the validated signal.

Relevant code: `DECISION_PROMPT`, `DECISION_KINDS`, `validate_decisions` in
`src/cloak/train/ladder_probes.py`; reader `_reader_mc_for_context` in `scripts/build_probes.py`.

---

### 2. Per-rung ladder questions are redundant on nested lattices — collapse to ONE semantic question at the finest rung (decided; not yet implemented)

**What it is.** The teacher writes one question per rung, but for nested taxonomic lattices
(artery disease ⊂ vascular disease ⊂ cardiovascular system disease ⊂ disease of anatomical
entity) **no natural-language question uniquely selects one level** — every level is a true
answer to "what kind of disease is this?". So the teacher repeats one generic question across
rungs, and the graded acceptance sets do the leveling. Measured (super pv3):
```
hypertension r1: "What is the specific medical condition label ... blood pressure medication?" gold=artery disease
hypertension r2:  (identical question)                                                          gold=vascular disease
hypertension r3:  (identical question)                                                          gold=cardiovascular system disease
hypertension r4:  (identical question)                                                          gold=disease of anatomical entity
```
Same for `osteoporosis` r0–r4, `kidney stones` r0–r3, `arthritis` r0–r4. So r1–r4 are **one
measurement scored at four thresholds**, not four probes — "4 kept rungs" overstates.

**Why it matters (the degeneracy).** Channels: rung 0 scores on `out_final` (echo — a placeholder
echoes+inverts, so it IS credited here, by design for copy-facts); rungs ≥ 1 score on `out_p`
(semantic — a placeholder scores 0). With per-rung golds + **monotone acceptance** (finer entails
coarser), a *coarse* fill still satisfies the coarse rungs and earns partial credit — so
coarsening is only weakly penalized and the policy can "pick the first floor-legal rung and be
done." There is no strong pull toward the **finest legal** rung, which is the actual utility
objective.

**Decided resolution.** One semantic question per span, pinned at **rung 1** (the finest
generalization); drop the r2…rL questions.
- Gold/acceptance = `{surface, rung-1}`: keeping the surface OR the finest generalization scores
  full; a **coarser** fill's recovered answer no longer matches the specific gold, so it scores
  less; placeholder scores 0. This makes coarsening actually cost reward → gradient toward the
  finest legal rung.
- Keep rung 0 (exact/echo) on `out_final`. Validate: ceiling-answerable AND floor-not.

**Dependency — sharpen the scorer.** The "answerable at r1 but not r3" cutoff is only as sharp as
`fact_score`. Token-F1 gives "vascular disease" vs gold "artery disease" ≈ 0.5 (shared word
"disease"), so the cutoff is currently **soft**. Sharpen the semantic-tier match (head-noun /
full-phrase, not bag-of-words overlap) in the same change, or the specificity gradient stays
blurred.

**Code touchpoints:** `LADDER_PROMPT` (ask rung 0 + one finest-generalization question),
`ladder_probes_for_docs` (emit 2 questions, not L+1), `_score_ladder`/`validate_ladder` (semantic
channel = the single rung-1 probe), `fact_score`/`entail_score` (scorer sharpening), `LADDER_PV`
bump. In `src/cloak/train/ladder_probes.py` + `src/cloak/train/roundtrip.py` + `scripts/build_probes.py`.

---

### 3. Rejection patterns show questions are placeholder/trivially answerable OR reader-unanswerable — teacher prompt needs a redesign (OPEN, for next session)

**The pattern.** Looking across the full kept/rejected dump (super + ultra), the rejects split
into two families, and both mean the question isn't doing its job — while the questions that
*would* do the job tend to fail for the wrong reason:

- **FLOOR family — answerable even at the all-placeholder floor (the fact isn't needed).**
  - *Echo* (rung 0, `out_final`): a placeholder echoes through the remote model and `invert()`
    restores the surface, so the exact-tier question is answered with the fact hidden — measured:
    `depression r0`, `hypertension r0`, `echocardiogram r0/r1`, `allergic rhinitis r0`,
    `osteoarthritis r0` all floor `lo≈1.0`.
  - *Context-inferable* (semantic tier): the category survives anonymization because the answer
    is recoverable from surrounding surviving text — `osteoarthritis r2/r4` (`lo` 0.57/0.80),
    `echocardiogram r1`.
  - These probes **do not penalize placeholdering** — a placeholder-everything policy still
    passes them. In reward terms they *promote* placeholder, which is the original
    placeholder-gaming failure resurfacing at the question-quality level
    (`docs/issues/2026-07-06-placeholder-gaming-reward-qa-necessity.md`).

- **CEILING family — unanswerable even at the ceiling (reader can't recover it).**
  - Multi-condition-list disambiguation: `osteoporosis`, `kidney stones` ("what condition did
    Doctor Kumar follow up on?" when the note lists six) — the small reader picks a wrong
    neighbour.
  - Absent/denied or reworded facts: `nasal congestion` (denied in ROS), `arthritis` (note
    phrases it differently).
  - Crucially this family **includes the genuinely discriminative questions we want** — the ones
    that truly require the protected fact — but they're lost to the pinned `Qwen3.5-0.8B` reader.

**The bind.** Questions that require the protected fact tend to ceiling-fail (reader limit);
questions that pass tend to be echo/context/world answerable (don't need the fact). So the kept
set skews toward **non-discriminative** probes, and the reward can be satisfied *without
preserving the fact* — exactly what the ladder was built to prevent.

**Task for next session (open — think before coding).** Redesign the teacher prompt(s) (ladder +
decision together) so a question is answerable **only when the fact is preserved at the needed
granularity** — never via surface echo, never from world knowledge, never from surviving
surrounding context. Directions to explore, not decisions:
- make the question's answer the fact's protected attribute that is absent from every other part
  of the note (so co-surviving context can't leak it);
- construct questions whose correct answer *changes with the fill's granularity* (so a coarser
  fill gives a measurably different/worse answer) rather than yes/no or single-token echoes;
- forbid grounding on facts that themselves survive anonymization.
- This likely must be paired with the reader-capability question (Issue: the small MC/QA reader
  is the binding constraint) — a stronger reader may be needed so discriminative questions stop
  ceiling-failing. Otherwise prompt fixes trade floor-answerable probes for ceiling-failing ones.

**Decisions are the extreme case of this.** Every decision probe in the sweep is world-knowledge
trivia that *names the fact* and is answerable without the note, so essentially all of them are
floor-answerable (`lo_pick == gold`):
```
Which body-system category does the condition 'Hypertension' belong to?   -> Cardiovascular
Which medical specialist should primarily manage congestive heart failure? -> Cardiologist
What is the purpose of ordering a thyroid panel?                           -> evaluate thyroid status
```
The retargeted `DECISION_PROMPT` produces ~100% trivia (one spurious survivor, the `mammogram`
probe in Issue 1). So the decision teacher prompt must be rethought in the **same** redesign, to
the same bar: a decision must be answerable only from what the note documents about *this patient*
and must break when the fact is placeholdered — not from naming the fact + asking a generic
property. Issue 1's open (a)/(b)/(c) choice is the decision facet of this one; resolve them
together.

Cross-refs: Issue 1 (decisions — same root), Issue 2 (single rung-1 question), placeholder-gaming
issue, `docs/specs/RL/training-task-env.md` (probe-necessity intent).

---

### 4. Question design is too narrow: condition-only targets + factoid "what/which" only, no reasoning ("why"/relational) questions (OPEN, for next session — part of the teacher-prompt redesign)

Two related narrownesses in what the teacher generates:

**(a) Targets are conditions; medications are only used as identifying context, rarely probed
themselves.** Almost every probe is *about a condition*, and drugs appear only as anchors ("the
issue treated with the blood pressure medication", "treated with synthroid"). The few
drug-targeted probes are thin (`lisinopril` class). Partly a detection/coverage artifact
(clinical yield is ~16 conditions : 3 drugs : 1 procedure, and many drug surfaces don't resolve
to a profile lattice), but the prompt also never *asks about the medication as the fact* (its
class, its indication, why it was chosen). Drug spans deserve first-class probes, not just a
role as context for condition questions.

**(b) All questions are factoid "what/which"; none are "why"/"how"/relational.** The prompt's
rung-answer framing ("the rung's phrase is the best answer") + "short-phrase answer" constraint
forces category factoids and structurally excludes reasoning questions. We lose exactly the
genuinely good, hard-to-game questions — the ones that require integrating multiple documented
facts and context, e.g.:
```
Why did the clinician prescribe <drug> for this patient given <comorbidity>?
What does <finding> imply for managing <condition> in a patient with <other fact>?
```
Not all questions should be "why" — but we need those, plus other **context-aware reasoning**
shapes. These are inherently note-dependent and resist echo/world-knowledge answering (the
Issue-3 failure), which is why they matter.

**Structural note (where these belong).** The granularity **ladder is intrinsically "what/which"**
— it measures how specifically a *category* survives, so its answer is a phrase at some rung; that
is correct for the ladder and should stay. **Reasoning/"why"/relational questions belong in the
decision (or a new reasoning) tier**, not the ladder. This sharpens Issue 1/Issue 3: the decision
tier's problem isn't only "trivia" — it's that decisions are *category lookups* when they should
be *reasoning questions* that require the documented relationships among facts (condition ↔ drug ↔
comorbidity ↔ finding). Redesign the decision/reasoning prompt toward those, and give drugs (and
procedures) first-class targets.

Cross-refs: Issue 1 (decisions), Issue 3 (require-the-fact redesign), `training-task-env.md`.

---

### 5. Screening/abstain span deletion leaked surfaces into the floor anchor (deletion FIXED; polarity-annotation redesign OPEN)

**What happened (found 2026-07-12, investigating the pv4 sweep's OUT_LO_P for aci/D2N002).**
Raw GLiNER recall was fine (synthroid 0.989, tylenol 0.938, kidney transplant 0.952 under the
exact build settings) — but `_detect_docs`' post-filters *deleted* the spans: synthroid via the
negation/screening filter (its first mention is the dialogue question "how are you doing with
the synthroid?"), tylenol + kidney transplant via matcher abstain (stale embindex → exact-only;
plus a label→type routing mismatch: GLiNER says `medical process`, the profile entry lives
under `health-condition`). Deletion removed them from the **floor anonymization**, so the
all-placeholder anchor sent those surfaces to the remote verbatim — corrupting floor verdicts
(some "context-leak" floor-rejects were actually this) and letting the remote re-infer hidden
facts (it reconstructed a thyroid condition from the surviving drug — wrongly, as
HYPERthyroidism).

**Fixed (commit 30e9564): demote, don't delete.** Screened/abstained lattice detections keep
placeholder role — no probes, but hidden at the floor and present in the env span set. This
also decouples the two things deletion had conflated: probe *eligibility* vs floor *hiding*.

**OPEN — screening filter becomes polarity annotation (next teacher-prompt redesign session,
with Issues 3–4).** Demotion leaves denied/screened facts probe-less → utility-silent in RL →
a count-rewarding privacy term pushes them unopposed to placeholder, destroying pertinent
negatives ("denies chest pain" is clinically load-bearing). Direction decided in the
2026-07-12 coverage analysis: keep the interrogative/denial detection but attach it to the
span as a status flag (`active / denied / screened`) and pass it to the teacher, which writes
status-faithful questions ("Which symptom did the patient deny on review of systems?" gold:
the symptom). This removes the fabricated-premise failure the filter was built for (the
teacher is no longer misinformed) AND gives pertinent negatives real utility signal. Belongs
to the same redesign as attribute-absent grounding: both are "stop hiding ground truth about
the note from the teacher." Companion analysis: admission≠probing decoupling, class-stratified
probe sampling under the per-rollout reader budget, per-span coverage metadata in the artifact
(reward-side notes: `docs/dev/logs/2026-07-12-qa-reward-codesign-notes.md`).

Related repairs still pending: rebuild the stale `lattice_profiles.embindex.npz` (exact-only
matching amplified abstains in the pv4 sweep), and the `medical process`↔`health-condition`
type-routing mismatch for facts like transplant status.

---

_Further issues to be appended below as we work through them._
