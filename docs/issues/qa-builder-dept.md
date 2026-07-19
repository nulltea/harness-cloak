---
type: reference
status: current
created: 2026-07-18
updated: 2026-07-19
tags: [qa-v2, debt, cue-gate, relation-teacher, hedge-guard]
---

# QA-builder debt log

Deliberate shortcuts and dormant code in the QA-v2 builder, so they get tracked instead of
rotting. One section per item; remove the section when the debt is paid.

## Compiler relation cue gates: disabled, not yet removed (2026-07-18)

**State.** Relation cue gating (fixed lexical-cue lexicon + NLI rescue) is disabled **in the
compiler** via `RELATION_CUE_GATES_DISABLED = True` in `src/cloak/train/qa_builder.py`. For a
teacher-proposed relation the three-point reader gate is the sole semantic acceptance check.
Structural safeguards are unchanged and still mandatory: anchor derivation and locality caps,
exact literal/argument grounding, no problem-switch crossing, hedge guard, leak checks.

**Why.** The maintained cue lexicon is not sustainable on informal clinical speech (unlimited
paraphrase/anaphora → recurring false-negatives) and would need re-tweaking per new domain.
Commit `2609ae5` already dropped it for literal→linked probes on this rationale; the
2026-07-18 e2e diagnostics run (D2N001–D2N007, gpt-oss/deepinfra-bf16) motivated extending the
drop to span→span at the compiler.

**Disabled call sites** (short-circuit on the flag; cue code retained dormant):

- `compile_relational_assertions` — span→span `invalid_evidence` reject via
  `_relation_quote_has_direct_support`.
- `_remap_to_groundable_siblings.grounds` — compiler sibling remap, span→span direct-support
  conjunct (already returned early for literal probes since `2609ae5`; now also flag-gated).

**Deliberately NOT disabled — the opportunity miner keeps its cue gate.**
`relation_support_opportunities` and its remap `_remap_to_lexically_groundable_siblings` still
apply `_relation_quote_has_lexical_cue_support` unconditionally. There the cue is the *only*
precision filter on a full combinatorial pair enumeration; disabling it inflated span_span
opportunities ~10x (D2N002 30→301, D2N001 9→84) and floods gleaning's uncapped `missed`-target
set with junk. The miner is a conservative lower-bound signal by design, so it retains the cue.
If the cue lexicon is eventually removed, the miner needs a *replacement* precision bound first
(e.g. clause-distance / same-problem-block), not a bare removal.

**Dormant-but-untouched (legacy, not part of the active compiler gate):**

- `relation_evidence_windows` / `_window_pair_has_relation_shape` /
  `_RELATION_WINDOW_CUE_PATTERN` — legacy evidence-window builder, now only consulted to
  resolve old `evidence_window_id` proposals at compile; its cue filters gate nothing in the
  v21-prompt path. Candidate for deletion together with the cue functions.

**Removal plan (when confident):** delete the flag and the compiler-side cue functions
(`_relation_quote_has_direct_support`, `_relation_quote_has_semantic_support`, and — only after
the miner has a replacement precision bound — `_relation_quote_has_lexical_cue_support`), the
per-relation cue tables, the legacy window path above, and their unit tests; keep the structural
checks. Before removal, confirm on real data that reader-gate-only compiler acceptance does not
admit cross-problem junk the cue gate was silently catching (watch `three_point_gate_failed`
volume and false-keep audits on a multi-doc run).

**Pin.** `builder_pin` bumped `qa-builder-v2-assertion-compiler-v11` → `v12` with this change;
artifacts produced with the compiler cue gate disabled are identifiable by the pin.

## Opportunity-miner cue-miss escalation — additive, opt-in (2026-07-18)

`relation_support_opportunities` takes an optional `escalator` callback (default `None`). It is
consulted **only on a cue-miss** and may only return accept, so the accepted-opportunity set is
always a **superset** of the cue-only set — a no-regression invariant, proven by
`test_relation_support_escalation_is_additive_superset` and confirmed on real docs (`escalator=None`
reproduces the exact cue-on counts D2N001–007; dialogue docs D2N005/D2N007 recover from 0). The
escalator is `RelationSupportCascade` (`src/cloak/train/relation_support_gate.py`): recall-first,
accept-biased, MedGemma-4b judge on the anchor's surgical stitch, with an optional accept-only
MedNLI cost tier. Wired via `build_qa_utility_artifact.py --relation-support-escalation`
(default ON; `--no-relation-support-escalation` = miner byte-identical to the cue gate). Empirical
basis: the MedGemma judge held 0.96 recall / 0.80+ mis-pair rejection on a held-out doc set where
clinical NLI rejected 0/7.

Open follow-ups: (a) premise widening if the anchor stitch truncates evidence (the callback receives
`document`); (b) capping/ranking the gleaning `missed`-target set, since escalation admits more
targets and that set is currently uncapped; (c) enabling the MedNLI accept-only tier if per-candidate
judge latency becomes the bottleneck at NP-chunk scale.

### causes_or_explains: judge-gated cue-ok + directionality fix (2026-07-18)

The opportunity miner flooded D2N002's repair set with junk `causes_or_explains` pairs (fever→arthritis,
headaches→arthritis, both directions). Diagnosis (validated locally on D2N002, 0 paid):
- 70 were **cue-miss** pairs the accept-biased escalator recovered. Fixed by tightening the MedGemma
  judge: a causation rule (rule 6 in `_JUDGE_SYSTEM`) requiring explicit directional attribution and
  rejecting co-mention / negated findings, plus `reject`-on-error for causes_or_explains
  (`_REJECT_ON_ERROR_RELATIONS`). Result: 0/70 accepted.
- 14 were **cue-ok** pairs bypassing the escalator via block-level cue matching (`allow_plan_section`
  fires for any condition pair sharing a causal word). Fixed by `_JUDGE_GATED_RELATIONS`: for
  causes_or_explains the cue is necessary-but-not-sufficient, so cue-ok pairs also require judge
  confirmation. Result: 1/14 kept (the real `arthritis→knee`, correct direction).
- Also fixed a **directionality bug** in `relation_support_gate.RELATION_CLAIM`: causes_or_explains was
  the passive `"{s} is caused or explained by {o}"` (inverted vs the contract's `subject causes object`);
  now active `"{s} causes or explains {o}"`, so the judge assesses the correct proposition.

Net on D2N002: causal junk 84→1. The judge-gate breaks the strict additive invariant for this one
relation (escalator is a precision filter, not only recovery); unchanged for all other relations and
for cue-only mode (escalator=None). Also added a self-pair guard in `_gleaning_targets` (a rejected
teacher proposal pairing a decision with itself, e.g. knees→knees, is no longer handed back to repair).

## Context-probe role-cue lexicon superseded by the informative-context judge (2026-07-18)

The `semantic_property` context-probe gate admitted an entity only if a sentence matched the
per-type role-cue regex lexicon (`AciTaskAdapter.semantic_type_contract["role_patterns"]`) — same
maintained-lexicon anti-pattern as the relation cue gate; on D2N001–007 it rejected 46 entities as
`no_task_role_cue`, half of which carried relation evidence. Now, on a regex miss,
`_task_role_context_locator` escalates (admit-only, same no-regression shape as the miner
escalator — a cue match is honored unchanged and never consults the judge):

1. **Relation-reuse (free):** the decision appears as a linked argument in a mined relation
   opportunity → admit; its clinical role in context is already established. `role_cue`
   recorded as `relation_evidence`.
2. **MedGemma informativeness judge:** one call per candidate locator sentence (≤3/decision) on
   the *redacted* sentence (`[target item]` mask, so the judge sees what the reader question
   quotes) — "does the remaining context establish the masked item's clinical role?".
   `build_informative_context_judge` in `src/cloak/train/relation_support_gate.py`; accept-biased
   on infra/parse error (the three-point reader gate is the real acceptance check). `role_cue`
   recorded as `semantic_judge`; a judged rejection gets `detail_reason:
   uninformative_context_judged` (regex-only misses keep `no_task_role_cue`).

Wired via `build_qa_utility_artifact.py --informative-context-judge` (default ON;
`--no-informative-context-judge` = pure regex lexicon). One flag governs both tiers, so off is
byte-identical to the old gate. Pin: `builder_pin` v12 → v13. The regex lexicon stays as the fast
first tier; it becomes removable once the judge path is validated on real multi-doc runs (watch
placeholder-answerable and reader-gate volumes among `semantic_judge`/`relation_evidence` admits).

**Superseded same day — semantic_property probes disabled entirely.** Before the judge's first
real run, `SEMANTIC_PROPERTY_PROBES_DISABLED = True` (`qa_builder.py`) short-circuits
`semantic_property_candidates` to `[]`: on D2N001–007 the probe family yielded 1 kept assertion
against 114 `no_task_role_cue` rejections, and the QA focus is teacher relations. The generation
code and the judge escalation above stay dormant behind the flag (the judge remains live
machinery-wise and validated by unit tests); re-enable by flipping the flag if category probes
return to scope, or delete both together if they don't.

## Open: span_literal opportunity recall is gazetteer-bound (2026-07-18)

Not fixed — logged so it is not lost. The opportunity miner's context-literal supply comes from
`relation_context_candidates`, a 4-pattern regex gazetteer (`order/check X labs|panel|test`;
`refer(red) to X therapy|surgery|procedure`; `status …`; `category …`). It does not match common
test/procedure literals — "finger splint", "follow-up x-ray", "lumbar spine x-ray",
"echocardiogram" — so those never become `span_literal` opportunities, are never flagged
`missed`, and gleaning cannot recover them when a teacher draw omits them. This (not the cue gate)
is the binding constraint behind the D2N005 (4→2) and D2N007 (4→1) kept-relation regressions
observed on 2026-07-18. Fix = broaden context-literal recall without a hand-maintained
gazetteer (same brittleness class as the cue gate). Scope TBD; needs a precision/recall check so
it does not reintroduce the miner explosion above.

## Hedge guard: regex modality gate demoted to diagnostic; semantic replacement planned (2026-07-19)

**Why the current guard is limited.** The compile-time hedge gate
(`_relation_window_is_hedged` + `_question_is_conditional`, `qa_builder.py`) rejects a relation as
`hedged_relation` when the SOURCE window between the two arguments matches `_HEDGE_PATTERN`
(may/might/would/if/consider…) AND the teacher QUESTION is not itself conditional. The intent is
correct — a conditional/planned source ("if it persists we'll order an MRI") paired with a definite
question ("which test was ordered?") is a false assertion that corrupts the QA/reward signal, and
the reader gate does NOT catch it (the reader tests *answerability of the privacy-rewritten doc*,
not *whether the source actually asserts the relation at the stated modality*). But both halves are
brittle regex:

- `_relation_window_is_hedged` scans the whole min→max argument window, so an unrelated "might" or
  "consider" anywhere in the window rejects a valid relation (the code comment itself calls this
  "whack-a-mole"); and it misses hedges the pattern does not enumerate (false negatives).
- `_question_is_conditional` is the same brittleness on the question side.
- `tests_for` is blanket-exempt (finding-hedge is pervasive there), so a genuinely *planned* test
  with a "was ordered" question is never caught for that relation.

This is the same maintained-lexicon anti-pattern already retired for the relation cue gate
(`RELATION_CUE_GATES_DISABLED`) and the semantic_property role-cue lexicon.

**Interim change (this commit): demoted to a non-blocking diagnostic.** The gate no longer
rejects. When the (regex) hedge condition fires it records `evidence.modality_diagnostics =
["hedged_source_definite_question"]` on the relation and lets it proceed to the reader gate;
`compute_review_flags` surfaces it as an `info` `hedged_source_definite_question` flag
(fix_class `ontology_review`). Rationale: the regex is too blunt to *block* on (net false-negatives),
but the observation is still worth surfacing for review until the semantic replacement lands. The
`hedged_relation` reject reason is no longer emitted as a blocker.
Empirically the relations this gate blocked were mostly conditional-procedure/PRN cases (D2N004/D2N007
`procedure_for`/`prescribed_with`) that are better judged semantically than by window regex.

**Reader-failed hedge-flagged relations still route to repair.** Demoting the blocker must not lose
the old repair path, so: `reject_context_candidate` carries `evidence.modality_diagnostics` onto the
reader-gate rejection, and `_gleaning_targets` routes a `three_point_gate_failed` rejection carrying
`hedged_source_definite_question` back as a `fixable` target with the conditional-phrasing hint
(reusing the `hedged_relation` `_GLEANING_FIX_HINTS` entry). Net: hedge-flagged → reaches the reader →
kept if confirmed (no needless redraw), routed to repair only if the reader also fails it — strictly
better-targeted than the pre-demotion path, which repaired every hedged relation. Note this only
fires for relations that pass all compile gates then fail the reader; a hedge-flagged relation caught
by an earlier compile gate (e.g. a context-literal `procedure_for` hitting `answer_leakage`) never
reaches the reader and is that gate's concern, not the hedge's.

**Proposed replacement: a strict semantic modality judge (compiler-side).** Replace both regexes
with one MedGemma call per teacher proposal, reusing the existing `_claim`/`RELATION_CLAIM` templates
and MedGemma client/cache/pin from `relation_support_gate.py`, but a NEW wrapper (not
`build_medgemma_judge`, which never sees the question):

- Signature: `build_relation_modality_judge(client, *, accept_on_error=False)` returning
  `judge(*, premise, claim, question, relation) -> ModalityVerdict(relation_asserted,
  question_ok, why)`.
- Prompt: `_JUDGE_PREAMBLE` grounding rules + a MODALITY rule + the teacher QUESTION; JSON
  `{"relation_asserted": bool, "question_not_stronger_than_source": bool, "why": "<=12w"}`. Worked
  examples OFF-CORPUS (never an ACI entity — see the burned-example rule).
- **Strict, not accept-biased** (opposite of the miner's `RelationSupportCascade`): parse/infra
  error → reject + log, because a false keep here is an RL-reward error. This is the compiler-side
  discipline: the compiler protects source-truth/modality before reward; the miner is accept-biased
  for recall; the reader only checks that the rewrite preserved an already-source-valid question.
- Wiring: at the current hedge site, `question_not_stronger_than_source == false` rejects
  (`question_overstates_source`); `relation_asserted == false` optionally rejects
  (`source_relation_not_asserted`) — the latter broadens the gate to a source-truth check for every
  proposal and overlaps the problem-switch gate, so ship modality-only first.
- Cached + pinned by (premise, claim, question, relation, judge-prompt-version, judge-pin);
  ~1 call/proposal (cap-bounded), batch across a doc only if latency bites.
- Rollout: opt-in flag (`--relation-modality-judge`, default off = byte-identical to today), same
  pattern as `--relation-support-escalation`/`--informative-context-judge`; validate on real data,
  THEN delete `_HEDGE_PATTERN`, `_relation_window_is_hedged`, `_CONDITIONAL_QUESTION_PATTERN`,
  `_question_is_conditional`, and the `tests_for` exemption. Do not delete the regex in the same
  commit that adds the judge.

Origin: reviewed with GPT-5.6 (Sol, high) 2026-07-19 — the gate-audit conclusion was that the
reader is an *answerability* gate not a *source-truth* gate, so this cannot be dropped onto the
reader, only semanticized.
