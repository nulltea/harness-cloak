---
type: plan
status: current
created: 2026-07-16
updated: 2026-07-16
tags: [qa-builder-v2, relation-teacher, gleaning, repair, escalation, gpt-oss]
companion: [docs/specs/qa-builder-v2.md]
---

# Relation gleaning + repair (replaces the Nemotron escalation)

## Goal
Replace the blind Nemotron secondary teacher with a **targeted second gpt-oss call** that
gleans *missed* relations and *repairs* fixable-rejected ones, then re-gates and merge-dedups
its returns. Max **1** gleaning+repair attempt per doc.

## Trigger (after the primary gpt-oss build + all gates)
Fire iff the target set (below) is non-empty. Targets, by kind:
- **ambiguous** — any kept-or-rejected relation carrying `answer_competing` (the ambiguity
  monitor). Hint: name the specific one (disambiguate).
- **missed** — a `relation_support_opportunities` ledger fact whose `_relation_fact_key` is in
  **neither** the kept set **nor** any primary proposal (kept or rejected). The teacher never
  addressed it. Hint: evidence card only.
- **fixable-rejected** — a rejection whose `detail_reason` is in the FIXABLE set. Hint per reason.

Conservative rule (avoid false-negatives): marginal reasons default to **fixable-INCLUDE**.

### Rejection taxonomy (frozen, approved 2026-07-16)
FIXABLE (include, with hint):
`invalid_evidence`, `invalid_evidence_occurrence` (cross-section co-reference → re-anchor);
`three_point_gate_failed` **only** when `answer_competing` present (disambiguate);
`protected_locator`, `protected_answer`, `answer_leakage` (recolor to levels);
`hedged_relation` (phrase question conditionally);
`literal_will_be_substituted` (reference by S-label);
`placeholder_answerable`, `floor_answerable` (pick a discriminative level);
`invalid_question`, `invalid_property` (re-author question / fix property).

EXCLUDE (100% legitimate / not teacher-fixable):
`no_task_role_cue`/`not_generated` **without** a ledger opportunity, `source_contradiction`,
`invalid_polarity`, `invalid_argument_types`, `invalid_relation`, `duplicate_fact_group`,
`not_authoritative_for_delivery`, `placeholder_type_only`, `unknown_context_literal`,
`unknown_context_candidate`, `relation_cap_exceeded`;
data-owned `three_point_gate_failed → lattice_level_suspect`;
reader-owned `three_point_gate_failed → representative_unreadable`, `reader_unstable`;
infra/malformed `context_reader_failed`, `generation_failed`, `infrastructure_failed`,
`invalid_subtype`, `invalid_scoring_contract`, `unsafe_template_leakage`,
`representative_protected_identity_survived`.

`three_point_gate_failed` is split by the existing `compute_review_flags` classification:
ambiguous→fix, lattice_level_suspect→data(exclude), representative_unreadable→reader(exclude).

## Second-call prompt (`relation_repair_prompt`)
Same base scaffold as `relation_teacher_prompt`, but:
- evidence cards **filtered** to only the target relations' clauses (drop cards for kept +
  legitimately-rejected relations);
- **DETECTED SPANS filtered** to only S-labels present in the leftover cards (exclude spans not
  relevant to any target);
- a REPAIR section listing each target: its argument S-labels/literals, `rejection_reason`,
  and the per-reason `fix_hint`; missed targets carry only the evidence card.
- same response schema; same gpt-oss teacher config as the primary (NOT nemotron).
- the primary `relation_teacher_prompt` output stays **byte-identical** (teacher cache key); shared
  card logic is factored into `_relation_evidence_cards` without changing the primary string.

### Deferred (A/B, not now)
Per-reason **fix examples** (worked repair demonstrations) added to the REPAIR section, but only for
the reasons that actually apply to the targets in this doc — not all reasons. Evaluate against the
plain hint-only prompt before adopting.

## Merge + re-gate
Returns run through the identical compile + 3-point gate. Merge into the kept set with
`_relation_fact_key` **dedup** (drop a return that duplicates a kept relation even though its
evidence wasn't sent — the user's dedup hedge). Reuse `merge_kept_relation_rows`.

## Units
- U1 `_gleaning_targets(...)` — taxonomy + missed-ledger + ambiguous → target list. Unit-tested.
- U2 `relation_repair_prompt(...)` — filtered cards + repair/hint section.
- U3 wire into `build_utility_artifact` (replace the escalation branch); gpt-oss repair teacher; max 1.
- U4 CLI/manifest: `--relation-teacher-gleaning` (retire the nemotron secondary construction).
- U5 provenance: `relation_gleaning` record (targets, hints, returns, merge disposition).
- U6 tests: units + e2e on aci/D2N006 (should glean the reflux↔ultrasound ambiguity + any missed).

## Risks
- False-negatives in the taxonomy → mitigated by conservative default (marginal→include).
- Extra paid gpt-oss call per triggering doc (was a free nemotron call). Operationally material;
  keep opt-in + max 1.
- The repair call still hits the same compiler/reader gates; a genuinely-unfixable target simply
  fails again (no infinite loop; max 1).

## Sources
Reuses the escalation ledger/merge from `feat(qa): conditional teacher escalation` (a260520) and
the ambiguity monitor (`relation_answer_ambiguous`).
