---
type: reference
status: current
created: 2026-07-12
updated: 2026-07-21
tags: [qa, reward-design, utility-components, context-preservation, credit-routing,
       interactive-ranker, spec, deterministic-relations, gleaning]
companion: [docs/specs/RL/interactive-ranker-v2.md,
            docs/specs/RL/interactive-ranker-v2-decision-log.md,
            docs/specs/RL/interactive-ranker-v2-diagnostics.md,
            docs/specs/RL/qa-builder-v2-decision-log.md,
            docs/specs/RL/training-task-env.md]
---

# QA builder v2 — context-preservation utility for interactive ranker v2

**Status: implemented and validated (67-doc ACI corpus, 2026-07-21). This document describes the
pipeline as built — `src/cloak/train/qa_builder.py` plus the builder scripts — not an aspirational
design. The historical normative design and its migration narrative live in the decision log.**

QA builder v2 produces a frozen set of utility assertions that reward truthful generalization when
it preserves useful context that a generic placeholder destroys, plus delivered-output assertions
on the final round-trip result. QA is a utility instrument, never a privacy measurement. Privacy
remains held-out LLM re-identification success on `doc_p` and `out_final` at matched realized
privacy and identical settings.

## Definitions

- **occurrence / decision** — a detected source span / the stable per-value policy decision it
  links to (many occurrences → one decision). Frozen before QA; the builder never re-detects.
- **level (generalization level / support property)** — a legal non-KEEP, non-placeholder lattice
  meaning of a decision (e.g. a drug's class), ordered most-specific → coarsest.
- **linked argument** — a relation argument that is a controlled occurrence (carries `span_label`
  `S#`, `occurrence_id`, and one level as `support_property`).
- **context literal** — a relation argument that is exact, uncontrolled source text (never a
  detector decision; survives every render verbatim).
- **fact key** — the decision-level directional identity of a relation:
  `(relation, subject-identity, object-identity)`, where a linked identity is its decision and a
  literal identity is its canonical text. Levels do not enter the key; forward and reverse QA of
  the same pair share one key. Compound rows decompose into subject × object pair keys.
- **three-point reader gate** — a QA is kept iff the pinned reader answers it on `doc_orig`
  (score ≥ t), on the representative generalized render (≥ t), and NOT on the all-placeholder
  render (< t), with t = the manifest `reader_threshold` (1.0 in production).
- **opportunity** — a mined, source-supported candidate pair (subject, object, relation) from the
  structural miner; the recall ledger that escalation, the deterministic stage, and gleaning
  targeting all consume.

## Pipeline overview

Five artifacts, built in order; each step pins the previous one:

```text
lattice_profiles.json                      (curated generalization lattices; owned upstream)
  └─ scripts/build_profile_embindex.py     → lattice_profiles.embindex.npz
       └─ scripts/build_arms_artifact.py   → arms.json (+ environment-audit sidecars)
            └─ scripts/build_ranker_env.py → ranker-env.json (frozen v2 environment)
                 └─ scripts/build_qa_utility_artifact.py → *.utility.json (+ views, qa-audit)
```

### Embedding index (`build_profile_embindex.py`)

Builds the embedding-index sidecar over `lattice_profiles.json` used by profile matching
(`cloak.profile_match`, default encoder `BAAI/bge-small-en-v1.5`). Must be rebuilt whenever the
lattice profiles change; a stale index silently degrades lattice resolution in the arms build.

### Arms (`build_arms_artifact.py`)

Detection runs ONCE here and is frozen. The QA-v2 clinical preset is pinned and may not be
overridden per flag (`--detector-config qa-v2-clinical`): GLiNER
`knowledgator/gliner-pii-large-v1.0`, threshold 0.35, label schema
`knowledgator-native-clinical-v1`, health-condition admission floor 0.5 (drops exam-finding noise
below real diagnoses), controlled runtime types `{LOC, drug, health-condition,
medical-procedure}`. The artifact embeds the v2 frozen input per document (occurrences, decisions,
action menus) and writes environment-audit sidecars. Rationale for freezing: detection is
nondeterministic across processes on long docs, so recomputing per consumer breaks caches and
reproducibility.

The full ACI corpus is `--corpora aci --n-docs 67`.

### Ranker environment (`build_ranker_env.py`)

Freezes the ranker-v2 environment from the arms artifact (`--skip-probes` in the QA-v2 flow: the
retired teacher QA-probe path is not used). Output carries the `environment_hash` that every
downstream artifact and cache key pins.

### QA utility artifact (`build_qa_utility_artifact.py`)

The QA build proper. Production flags:

- `--relation-teacher` — primary relation-teacher pass (paid, cached);
- `--relation-support-escalation` (default on) — MedGemma opportunity-miner escalation judge;
- `--informative-context-judge` (default on) — semantic-property probe admission judge (the probe
  family itself is currently disabled; machinery dormant);
- `--relation-support-prefilter` — LLM set-call proposer of context-literal candidates
  (augment-only; requires escalation);
- `--relation-deterministic-stage` — template relation QAs between primary and gleaning (below);
- `--relation-teacher-gleaning` — the paid repair+glean second pass (requires the manifest
  `relation_escalation_policy`).

Every enabled option is recorded in the artifact pins; disabled options leave the build
byte-identical to the option-free path. The threshold manifest
(`data/qa_v2/relation_gate_manifest.json`) pins the reader (medgemma-4b-it, prompt
`qa-context-reader-v3`, single-span response schema, revision `qa-reader-r4`), reader threshold
1.0, stability repetitions/permutations, and the family budgets (context 0.6 / delivered 0.4,
structural cap 0.1).

## QA model

### Relation inventory

Five directed relations; the arrow is the complete argument-direction contract
(`CLINICAL_RELATION_INVENTORY` / `ACI_RELATION_CONTRACT`):

```text
prescribed_with              condition → drug
procedure_for                condition → medical procedure (incl. past treatment)
tests_for                    condition → diagnostic/monitoring test, lab, imaging, exam
contraindicated_because_of   drug or procedure → condition
causes_or_explains           condition → condition or symptom
```

PERSON/holder relations and generic has_status/has_category remain excluded (see the decision
log); disconnection from this inventory never removes a ranker decision or implies zero utility
relevance.

### Arguments, orientation, answers

A relation has exactly one subject and one-or-more objects. Argument forms are disjoint: linked
(controlled span, S-label + level) or context literal (exact uncontrolled text). Every relation
needs ≥ 1 linked argument; a context literal can never be a gated answer (it survives the
placeholder render, so a literal answer is placeholder-answerable by construction). Consequently
QA orientation is chosen so that **the answered argument is always a controlled span**:

- **forward** (`answer_role: object`) — subject named in the question at a level, object answers;
- **reverse** (`answer_role: subject`) — object named in the question (level or literal), subject
  answers. For one linked + one literal argument, the literal is always the question locator and
  the linked argument's role forces `answer_role`.

Answer targets, scored by `_context_answer_score`:

- `linked_decision {decision_id, required_property}` — the reader's free-form answer resolves
  against the decision's frozen semantic chain (`answer_aliases`), then binary credit iff the
  resolved node entails the required property. No token-overlap partial credit.
- `linked_decision_set {members: [...]}` — set-valued QA: the reader answers with a JSON array
  (dedicated set-read prompt, `read_context_set_batch`); score = one-to-one per-member recall.
  At threshold 1.0 the gate demands every member readable on orig and rep and ≥ 1 member hidden
  on placeholder. Extra predictions (e.g. literals) are ignored — literals are pre-excluded from
  privacy scoring by construction.
- `literal {expected_values}` — lexical `fact_score` against the exact grounded text.

Compound rows (one subject + N objects: set-valued forward, compound span locator, multi-literal
reverse) carry all arguments in evidence and decompose into pair fact keys everywhere identity
matters (dedup, gleaning targeting, coverage).

### QA modalities

Six question shapes are live (examples illustrative, off-corpus entities):

| shape | question form | answer | producer |
|---|---|---|---|
| span → span (forward) | "Which medication was prescribed or used to treat the *endocrine condition*?" | object level, e.g. *thyroid hormone replacement* | primary/gleaning teacher (freely authored); stage forward template |
| span → span (reverse) | "For what medical condition was the *thyroid hormone replacement* prescribed?" | subject level, e.g. *endocrine condition* | reverse framing (ambiguity flips); stage reverse + singleton fallback |
| literal → span | "What single medical condition or diagnosis were *thyroid labs* ordered to evaluate or treat?" | subject level | teacher literal probes (compile forces the linked role to answer); stage literal-reverse (single literal) |
| {literals} → span | "What single medical condition were *bmp, lipid panel, and a1c* ordered to evaluate?" | subject level — the compound locator pins one condition where a single generic literal cannot | stage literal-reverse (≥ 2-literal group) |
| {spans} → span | "For what single medical condition were *loop diuretic* and *beta blocker* all prescribed?" | subject level | stage compound span locator (ambiguous-group fallback) |
| span → {spans} | "List EVERY distinct medication that the document says was prescribed to treat the patient's *endocrine condition*." | JSON array; `linked_decision_set`, one-to-one member recall over the controlled object set | stage set-valued forward (ambiguous-group fallback) |

For ambiguous groups the stage tries the shapes in a fixed order — per-object reverse flips, then
set-valued forward, then compound span reverse — stopping as soon as the group's pairs are
covered. `span → literal` (a literal as the ANSWER) is deliberately not a live modality: an
uncontrolled literal survives the placeholder render, so such a QA is placeholder-answerable by
construction; the `literal` answer target survives only for cached legacy artifacts, and the
compiler forces the linked argument into answer position for every new literal-bearing relation.

### Locator and answer level pinning

Terminology, precisely: a **span** is a detected CONTROLLED occurrence — it carries a decision and
a lattice profile, and the ranker acts on it. A **literal** is an UNCONTROLLED exact source span —
no decision, never rewritten, so it appears verbatim in every render.

A span-locator question never contains the span's surface; it names the span by a generalization
level. The two pinned levels of a span↔span QA are chosen differently:

- **Answer span: `required_property` = the SUPPORTED level — the coarsest that passes the reader
  gate.** Deterministic rows search the answer decision's levels coarsest → finest and keep the
  first three-point-gate pass; teacher rows pin the teacher's supported level (same semantics).
  Example (reverse): "For what medical condition was the *loop diuretic* prescribed?" with
  `required_property: heart disease` — the coarsest subject level the reader confirmed on both
  `doc_orig` and the representative render.
  **The reward band is "at or finer than the supported level."** Scoring is chain entailment:
  the reader's answer resolves to a chain node by its aliases, and credit requires that node to
  entail `required_property`. The chain is linear specific → coarse, so every node from
  KEEP/surface up through the supported level entails it and scores 1.0; anything coarser scores
  0. QA therefore rewards ANY ranking inside the band, and pressure toward the coarse end of the
  band comes from the count/privacy objective, not from QA — that division of labor is
  intentional.
- **Question locator span.** Deterministic rows pin the locator at the FINEST legal level
  (escalated past a level that equals/contains a FOREIGN protected surface): the locator is the
  given premise, not the measured fact, and the most specific wording maximizes the reader's
  ability to ground it in the rendered document. Teacher rows are teacher-authored — the prompt
  directs "the most specific level that still conveys the relation", a raw-surface echo is
  mechanically substituted with the teacher's selected level, and the leakage repair may recolor
  it — so finest is a tendency there, not a guarantee. Reverse flips inherit the locator level of
  their seed (finest for opportunity seeds, the teacher's object level for teacher-attempt
  seeds).

**Documented gaps (RL-scoring assumptions, not yet gate-certified):**

1. **Finer-band READABILITY is untested.** The scoring side of the reward band is guaranteed by
   entailment (above), but the gate certifies readability at exactly one (locator level, answer
   level) combination — the supported level. Whether the reader still answers the question when
   `doc_p` renders the answer decision FINER, and whether the finer node's `answer_aliases`
   actually resolve, is assumed, never gated. A finer level that is unreadable in practice scores
   0 at RL time and under-rewards finer-than-required rankings for that decision.
2. **The frozen locator level can steer the policy.** The question text freezes the locator at one
   level; at RL time the policy renders that decision at whatever level it ranks, and the reader
   must bridge the question wording to the rendered wording. That bridge is untested across
   levels, so reward may be maximized by ranking the locator decision at exactly the frozen level
   — a spurious preference the gate never checked for.

**Literal locators are policy-invariant** — a literal is never rewritten in any render, so the
question locator grounds identically under every action vector and cannot steer the policy. This
makes `literal/s → span` the RL-safest locator form; gap 1 (the answer side) still applies to it
unchanged. Closing both gaps would mean certifying additional (locator level × answer level)
combinations at gate time (a bounded sweep, at reader-call cost); that is future work, recorded
here so the assumption is explicit rather than silent.

## Opportunity mining and escalation

`relation_support_opportunities` enumerates typed subject × object pairs (linked spans from the
teacher span inventory × linked spans + context literals), grounds each pair through the same
anchor derivation the compiler uses, then applies a lexical cue gate. The miner's cue survives
(unlike the compiler's, below) because it is the only precision filter on a combinatorial
enumeration.

**Judge mining (escalation).** Opt-in, default on: cue-MISSES go to a MedGemma
(`medgemma-4b-it`, local llama-swap endpoint) accept-only relation judge
(`RelationSupportCascade`), so the judge can only widen the cue-matched set. Mechanics:

- **Structural junk prefilter** before any judge call: a pending pair is skipped only when the
  surface structure PROVES it non-assertive — coordinated list siblings (only list punctuation and
  and/or between the arguments, e.g. a history enumeration) or an argument under an explicit
  negation cue. Absence-of-signal rules (no cue word, large distance) are deliberately NOT used —
  they drop real relations. This keeps the accept-biased judge from converting list co-occurrence
  into junk seeds.
- **Recall-oriented premise**: the judge reads ALL occurrence clauses of both arguments (every
  mention of a linked argument's decision, every occurrence of a literal), not just the anchor
  quote — the supporting evidence often sits at a different mention than the grounded one.
  Verdicts are batched (`judge_batch`) into one model round per document.
- **Judge-gated relation**: for `causes_or_explains` the cue is necessary but NOT sufficient — the
  block-level cue accepts every condition × condition pair sharing a causal word — so cue-PASSES
  are also judged; for that relation the judge is a precision filter and the escalated set is not
  a cue-only superset. For every other relation it is.
- Accepted opportunities record `recovered_by_escalation`, and the manifest
  `relation_escalation_policy` (per-scope minimum opportunity counts, coverage fractions, caps)
  drives the escalation accounting and gleaning trigger.

No-regression invariant: with the escalator off, the result is exactly the cue gate.

The **context-literal prefilter** (opt-in) widens the object space: the gazetteer proposer only
emits test/procedure/status/category literals, so drug/symptom/condition literal objects are
structurally unreachable without it. One MedGemma set-call per (controlled condition, relation)
enumerates related literals, verbatim-located in the source and typed by the relation slot.
Augment-only (union with the gazetteer) so no currently-accepted pair can be dropped.

Opportunities are the shared recall ledger: escalation accounting, the deterministic stage's seed,
reverse-framing's flip candidates, and gleaning's "missed" targets all read it.

## Deterministic relation stage

Opt-in (`--relation-deterministic-stage`); runs BETWEEN the primary teacher pass and gleaning so
the paid repair pass only sees what free generation could not keep. Seeded from ALL accepted
opportunities, deduplicated against primary keeps by fact key.

Generation plans, per (relation, subject decision) group of span pairs:

1. **Singleton** (one object): forward template QA (subject locator at a level, object answers);
   on gate failure at every level, retry as the reverse flip before giving up.
2. **Ambiguous group** (≥ 2 objects): per-object reverse flips first, then — while any pair is
   still unkept — a **set-valued forward** QA (answer = the full controlled object set at finest
   levels, single trial), then a **compound span-locator reverse** (all object levels in one
   question pinning a single subject, answer-level searched).
3. **Literal pairs**: the literal-reverse builder — the literal object(s) become the question
   locator, the controlled condition answers; ≥ 2 literals form a compound locator that
   disambiguates where one generic literal cannot. Stage seed = all accepted opportunities (the
   standalone post-gleaning pass keeps the narrower judge-recovered seed when the stage is off).

**Answer-level search (the teacher-prior replacement).** Deterministic rows have no
teacher-proposed supported level, so trials walk the answer decision's levels coarsest → finest
and the FIRST three-point-gate pass wins — the coarsest supported level, the same semantics as the
teacher's prior. A static skip list drops bare type-word levels ("medication", "medical
condition", …) before spending reader calls. Only a plan's FINAL failed trial leaves a rejection
record, so speculative trials cannot flood the rejection channel or distort gleaning targets.

**Question locators** sit at the finest legal level, escalated past any level that equals or
contains a FOREIGN protected surface (a level echoing another decision's raw surface reads as a
locator for that span and would trip the leak gate).

Span-pair plans compile through the normal `compile_relations` path (every teacher-proposal guard
applies); compound/set/literal rows are built directly (the compiler is two-argument) and carry
the full evidence contract. All stage keeps run `run_id: deterministic_stage`,
`teacher_id: deterministic`.

Measured (5-doc A/B at identical settings): +65% kept relation QAs at zero teacher cost, primary
pass byte-identical; deterministic-only coverage of teacher-primary facts is ~35% (bounded by
miner recall, 13/20 mined), so the stage complements the primary teacher rather than replacing it.

## Quality gates

### Compile-time (deterministic, per proposal)

- argument resolution against frozen IDs; sibling remap when a repeated value's label grounds at a
  different mention; argument-type contract per relation; self-pair rejection;
- anchor derivation: clause → adjacent clauses → plan-section/problem-block/speaker-turn, never
  across problem switches; exact grounding of every argument in the anchor;
- polarity consistency and source-contradiction checks; hedge/modality is a non-blocking
  diagnostic (routes back to repair only if the reader then fails the row);
- `literal_will_be_substituted`: a context literal that names a detected controlled entity is
  rejected (or promoted to a linked argument when lattice-resolvable);
- **leakage**: answer-token overlap with the question (with answer-type words and, for reverse
  rows, locator tokens exempt) triggers deterministic level recoloring, else rejection;
  protected-locator and protected-answer checks run the question/answers against every controlled
  decision's surfaces with per-term legal-level exemptions. A protected-locator collision caused
  solely by an unrecolorable context literal is tagged (`leak_source: context_literal`) and is
  dead weight — no author can fix it, so it is never repair-targeted;
- duplicate fact-group dedup within a pass;
- the compiler's own lexical cue gate is DISABLED (`RELATION_CUE_GATES_DISABLED`): for an authored
  proposal the reader gate is the semantic acceptance test; the maintained cue lexicon was not
  sustainable on informal clinical speech. The miner keeps its cue (see above).

### Reader gate (three-point, per candidate)

`validate_context_assertions` renders three contexts — `doc_orig`, the joint representative
anchor (every linked decision at its required level, greedy-injective fills, co-referent
protected-term hiding matched to deployment), and the all-placeholder floor — excerpts only the
assertion's own transcript turns, and reads each with the pinned reader. Keep iff orig ≥ t and
rep ≥ t and placeholder < t, with optional stability repetitions and option permutations
(manifest-pinned; production 1/1). Set rows route to the JSON-array set reader; scoring is the
per-member recall above. A gate failure with orig ≥ t and rep < t triggers a coarser-readable
lattice probe recorded as diagnostic evidence.

Gate rejections carry `teacher_id`/`run_id` for attribution.

## Teacher

### Primary call

One pinned, cached, batched relation-proposal call per document with eligible controlled spans:
`openai/gpt-oss-120b` via OpenRouter, routed provider `deepinfra/turbo`, no fallbacks, strict JSON
response schema (S-label-bound linked arguments, `linked`/`context` kinds, ≤ 12 relations,
one candidate-accounting row per displayed label). The teacher authors questions, accepted
answers, and scoring contracts; code owns IDs, vocabulary, validation, and never treats proposals
as gold. Abstention is recorded, never retried for coverage. Prompt, model, provider, and schema
are cache pins.

### Repair + glean (secondary pass)

Opt-in paid second pass over ONLY the facts worth a teacher call. Target selection
(`_gleaning_targets`) builds a deduplicated, prioritized set (ambiguous > fixable > missed):

- **ambiguous** — a rejected relation with a co-valid same-type answer in scope (re-author with a
  distinguishing source detail);
- **fixable** — a rejection whose reason is in the fix-hint taxonomy (protected locator/answer,
  answer leakage, invalid property/question/evidence, placeholder-answerable, hedged, mispaired
  literal);
- **missed** — an accepted opportunity nothing attempted.

Three guards keep the paid channel honest:

1. **Kept-fact guard** — a pair fact covered by ANY kept row (compound rows decomposed) is never
   re-targeted, and compound attempts (kept or rejected) count as proposed so their pairs cannot
   resurrect as missed.
2. **Reader-outcome routing** (`_reader_outcome_route`) — a rejection whose stored three-point
   scores show orig/rep verdict disagreement or placeholder-only readability is a
   **lattice suspect** (a data defect; surfaced per doc in
   `relation_coverage.reader_routed_out`, never gleaned); a deterministic-authored relation the
   reader confirmed on NO render is **reader-verified no-relation** (miner co-occurrence junk;
   dropped, never re-authored). Teacher-authored rejections are exempt from the no-relation route.
   This is deliberately post-hoc filtering by reader outcome instead of tightening the
   recall-oriented miner/judge.
3. Dead-weight literal-collision leaks (above) are excluded from the fixable taxonomy.

The repair prompt restricts DETECTED SPANS and source clauses to the targets' regions, groups
targets by shared source region, states each distinct fix hint ONCE in a FIX GUIDE keyed by tag,
and batches ≤ 20 targets per call. Same privacy/response contract as the primary; the primary
prompt is untouched (cache-safe).

After gleaning, kept rows are rebuilt as `pre_teacher + stage keeps + merge(primary, secondary)`:
merge prefers the primary formulation by fact key; secondary rows duplicating a stage-covered pair
are dropped before the merge; stage keeps rejoin directly (compound rows cannot ride the two-arg
merge).

**Post-gleaning deterministic passes.** Reverse framing flips every object of an ambiguous
(relation, subject) group across ALL forward attempts (primary + gleaning + judge-accepted
opportunities) in an isolated doc-global pass, deduped against everything kept; the standalone
literal-reverse pass runs only when the stage is off (the stage supersedes it with the wider
seed).

**Measured economics (67-doc v15 build, new lattice env):** 616 kept relation QAs
(149 primary / 391 stage / 65 reverse framing / 11 gleaning), mean 9.2 per doc (median 7); the
reader-outcome router excluded 1,985 no-relation and 392 lattice-suspect rejections from repair;
gleaning returned 14 keeps from 49 paid batches. Gleaning is a safety net, not a load-bearing
stage.

## Utility assertions and scoring

Two measurement families: **context assertions on `doc_p`** and **delivered assertions on
`out_final`**. Weights: family budgets split equally across fact groups, then across a group's
assertions; missing families keep the fixed denominator without renormalization.

### Context probes

- `contextual_relation` — the relation QAs above; the live context family.
- `semantic_property` — deterministic category/function probes ("what kind of thing is this
  span"), admitted by role-cue regexes with an optional MedGemma informativeness judge
  (`--informative-context-judge`) for cue misses (free when the entity already carries mined
  relation evidence). The family is currently DISABLED after measuring 1 kept assertion against
  114 role-cue rejections on a development slice; generation and judge machinery are retained
  dormant.

### Delivered / schema probes

Deterministic, adapter-owned (`AciTaskAdapter.deterministic_candidates`); truth comes from the
human reference or the fixed task schema, never from ceiling output or teacher interpretation:

- `content` (linked, `contains`) — a reference-backed controlled value must survive into
  `out_final` (lexical `fact_score`); the omission channel.
- `field` (global, `field_value`) — exact agreement of DEMOGRAPHIC fields and per-condition
  ASSESSMENT row fields (e.g. status/category) between the parsed output and the reference.
- `structure` (global, `required_sections`) — required note sections present and parseable, with
  expected row shapes/counts for assessment and plan; structural compliance is capped by the
  manifest's structural budget so it can never substitute for semantic retention.
- `exact_relation` (`exact_relation`, PLAN section) — symbolic relation preservation: the
  plan row for a condition still carries its expected treatment/test values after the round trip.
  A placeholder-heavy pipeline can pass this while failing the context assertion for the same
  fact — the two families are deliberately not collapsible.

Runtime scoring (`score_utility`) replays each context assertion against the rollout's `doc_p`
with the same reader, same per-assertion turn excerpts, and the same answer-target scorers (set
rows via the set reader); delivered contracts are deterministic. Cache keys are
document/action-vector level.

## Invariants (unchanged and load-bearing)

1. Detect once; QA and RL consume the same frozen environment.
2. Assertions precede questions; teacher proposals are not gold.
3. Measurement (family) and routing (scope) are orthogonal; missing coverage never removes a
   ranker decision or implies zero relevance.
4. Pins are transitive: detector, lattice profiles/embindex, environment, teacher
   (model+provider+prompt+schema), reader, manifest, and builder flags all key caches; a pin
   mismatch invalidates dependents.
5. Thresholds are preregistered; attacker results cannot tune QA gates.
6. Failures are findings: no per-model calibration, post-hoc weights, relaxed gates, or selective
   document removal. Method comparisons only at matched realized privacy and identical settings.

## Artifact

`build_utility_artifact` emits one artifact: assertions (with evidence: arguments, argument
spans, reader turns, validation scores, run/teacher attribution), the full rejection ledger
(stable reason taxonomy + detail reasons + attribution), `relation_generation` attempts,
`relation_support_opportunities`, `relation_escalation` (per-doc primary/stage/gleaning
accounting, prompt hashes, batch counts), `relation_coverage` (unresolved targets by kind +
`reader_routed_out`), candidate accounting, and pins. Sidecar views: assertions, qa-pairs, and
the qa-audit trio. Infrastructure failure is an explicit state, never a silent zero score.

## Sources

- [Interactive ranker v2](RL/interactive-ranker-v2.md)
- [QA builder v2 decision log](RL/qa-builder-v2-decision-log.md) — historical normative design,
  PERSON-relation removal, migration narrative
- [Training-task environment](RL/training-task-env.md)
- Gleaning+repair taxonomy plan: `docs/plans/qa-relation-gleaning-repair.md`
- Validation artifacts: `results/qa_v2_stage_ab/` (5-doc A/B arms, deterministic-only coverage,
  gate-failure classification report), `results/qa_v2_aci_full_v15/` (67-doc build)
