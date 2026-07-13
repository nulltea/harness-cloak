---
type: reference
status: current
created: 2026-07-12
updated: 2026-07-13
tags: [qa, reward-design, utility-components, decision-log, schema, assertions, rl]
companion: [docs/specs/qa-builder-v2.md,
            docs/specs/RL/interactive-ranker-v2.md,
            docs/specs/RL/interactive-ranker-v2-decision-log.md,
            docs/specs/RL/interactive-ranker-v2-diagnostics.md]
---

# QA builder v2 — design decision log

This log records consequential design forks for QA builder v2. The companion specification is
normative; this document preserves accepted options, alternatives, and the reasoning that fixes
each choice. Undecided forks remain outside this log until the user selects an option.

## Per-sample schema assertions use a teacher-assisted compiler

**Decision.** Define the ACI output schema once per versioned task adapter. For each training
sample, compile a structured assertion manifest using deterministic extraction for explicit facts
and authoritative fields, one bounded relation-teacher request when an eligible controlled span
exists, and deterministic evidence validation.

The teacher is a bounded proposal mechanism, not the annotation authority. At most one batched,
cached call is allowed per document. It receives the full source document, authoritative reference
evidence when available, source-ordered eligible `S#` labels, exact quoted context literals, and
the ACI adapter's closed relation ontology. The harness assigns stable IDs, values, links,
channels, and interpretation classes after the teacher responds.

Each adapter declares one authoritative truth source: human reference where it contains the
required fact or relation, otherwise exact `doc_orig` evidence. Ceiling output is feasibility
evidence, never truth. Teacher interpretation alone is insufficient.

`out_final` assertions measure delivered content, fields, omissions, and exact or symbolic
relations. `doc_p` assertions measure semantic properties and contextual reasoning unavailable
from generic placeholders. A source fact omitted by the ceiling is not anonymization damage.

**Accepted implementation shape.** Fixed task schema plus deterministic extraction for structural
facts, one bounded teacher request for contextual relation semantics, and evidence-based compiler
acceptance. Deterministic code does not template relation questions or accepted answers.
“Relational compiler” names this complete teacher proposal-and-validation operation. The later
local-only ID/weight/hash step is artifact packaging, not compilation.

**Rejected alternatives.** Parse only `out_hi`; let the teacher write the complete manifest;
use deterministic extraction only. Ceiling-only assertions inherit task-model omissions and
mistakes. Teacher-authored manifests recreate the circular annotation problem and permit world
knowledge to become gold. Deterministic-only extraction is authoritative but misses clinically
important relations such as treatment constraints and condition-to-monitoring links.

**D2N002 example.** Deterministic candidates include age, sex, kidney transplant,
hypothyroidism, arthritis, Synthroid, thyroid panel, and physical therapy. Bounded teacher
proposals may connect hypothyroidism to Synthroid and thyroid monitoring, or arthritis treatment
constraints to kidney-transplant history. Validation preserves the distinction between
placeholder-friendly symbolic relations on `out_final` and category/function reasoning on
`doc_p`.

**Required tests.** The teacher cannot override authoritative values or IDs; every accepted
relation resolves to exact evidence; unsupported relations remain rejected attempts; source,
and authoritative reference evidence resolve under the adapter contract; ceiling evidence never
overrides truth; identical pinned inputs compile identical assertion IDs and manifests.

## Semantic and contextual components are defined on doc_orig and scored on doc_p

**Decision.** Generate semantic-property and contextual/downstream-decision components only from
accepted assertion-manifest entries. Define and reader-validate each measurement against
`doc_orig`, validate anti-placeholder discrimination against the frozen all-placeholder rewrite,
and score each rollout directly on `doc_p`. Do not use remote generated `out_p` as the reader
context for these component families.

Generated `out_p` is too narrow a measurement surface: the remote task can omit, compress,
reorganize, or miscategorize facts even when the transformed input retained enough context.
Direct `doc_p` scoring isolates whether the rewrite preserved semantic properties and contextual
relations. Separate `out_final` components continue to measure delivered task utility and remote
generation degradation.

Semantic-property gold comes from frozen lattice or adapter ontology state. For contextual
relations, the bounded teacher authors the semantic question, accepted answer(s), and scoring
contract; the compiler owns source grounding, typed ordered roles, occurrence links, legal action
support, leakage checks, duplicate suppression, and anchors. This replaces deterministic
relation-QA templates, which cannot express document-specific relational semantics. Questions are
static and cannot contain protected-value locators; no rollout-dependent question rewriting is
used.

**Relational assertion compilation.** Use a closed task-adapter relation vocabulary. A bounded
teacher receives `doc_orig`, source-ordered eligible `S#` labels, and non-gating evidence cards;
it proposes typed relations, teacher-authored QA, and label or exact-literal arguments. It does
not choose an evidence-window ID or author the authoritative evidence quote. Deterministic
compilation resolves labels, derives the minimal clause-level evidence anchor, accepts only legal
canonical classes and direct source connection, and rejects wrong polarity or contradiction.
Medical or domain knowledge absent from the document cannot become an assertion.

**Retained machinery.** Lattice-owned answers for fixed semantic-property assertions; the local
pinned reader; original versus all-placeholder validation; deterministic leakage and option lint;
and stable occurrence links. Contextual relation questions, accepted answers, and scoring
contracts are teacher-authored and validator-gated.

**Retired machinery.** Generated `out_p` as semantic/contextual reader context; exactly two
probes per span; deterministic relation-QA templates; teacher selection of policy decisions;
free-text `depends_on`; forced generation when no legal locator exists; and pair/window gates
that narrow semantic search. Repetition and option permutation are separately deferred rather
than retired or newly enabled.

**Rejected alternatives.** Continue scoring on generated `out_p`; generate independently from
raw documents instead of compiled assertions; expose the all-placeholder document to the
generator; rewrite questions per rollout using the replacement map. Generated-output scoring
confounds rewrite preservation with task-model omissions. Independent discovery recreates
teacher-selected targets and dependencies. Floor-visible generation overfits one placeholder
anchor. Runtime question rewriting changes the measurement together with the intervention.

## Static cross-family weights prevent density bias; ranker v2 owns utility credit

**Decision.** Freeze context and delivered family budgets in the QA threshold manifest. Divide
each family budget equally across unique family-local fact/relation groups, then divide a group
budget equally across its accepted assertions. Context and delivered assertions about the same
fact remain separate groups, preserving both required channels without arbitrary doubled mass.
Structural-only assertions are diagnostic or consume a capped delivered-family share.

If a family is absent, use a fixed denominator equal to the sum of the reserved context and
delivered budgets. The absent family contributes zero numerator and its reserved mass remains
absent; surviving weights are not renormalized. Store the denominator and missing-family state
without fabricating zero-score assertions. The document cannot support a claim about the absent
family, but its decisions remain eligible for fallback.

QA builder emits the frozen weights and document denominator. Ranker-v2's `weighted_mean` uses
that denominator and exclusively owns `Q_j`, `Q_global`, `A_link`, `A_global`, `A_document`,
fallback, counterfactual substitution, and loss composition.

**Rejected alternatives.** Flat unweighted assertions; hierarchical family advantages inside
QA builder; per-document or per-corpus normalization. Flat weighting turns builder yield into a
hidden reward weight. Builder-owned advantages conflict with ranker-v2's normative partition.
Coverage-relative normalization changes the objective.

**Rejected missing-family alternatives.** Renormalize surviving weights to full mass; mark every
missing-family document unsupported for all utility training. Renormalization gives delivered-
only documents full-strength reward and breaks cross-document scale. Full exclusion would make
QA coverage control the training population despite ranker-v2's fallback design.

## QA builder is one deep module with two measurement families

**Decision.** Expose one build interface and one runtime scoring interface. Keep detection,
assertion compilation, optional question wording, validation, coverage accounting, and cache
identity inside the builder implementation. The task adapter is the only public variation seam.
Use only context and delivered measurement families. Semantic-property and contextual decision
are context subtypes; content, schema field, and exact relation are delivered subtypes. Routing
scope is orthogonal: either family may be linked, while an occurrence-empty complete-task
assertion has global scope. There is no global measurement family.

Do not expose separate public ladder, decision, schema, validator, compiler, or cache modules.
Do not add an adapter seam until a second task genuinely requires it.

**Rejected alternatives.** Preserve the current independent builder/scorer surfaces or create a
public module per assertion subtype. Both make callers coordinate invariants and multiply cache,
pin, testing, and performance failure surfaces.

## Builder-time counterfactual validation is off by default

**Decision.** Use structural, leakage, anchor, and stability validation during artifact
construction. Do not add model counterfactual calls to the default builder. Ranker-time tested
action pairs substitute bounded local contextual evidence for provisional routing on actual
rollout states; they are not unbiased causal attribution. A small report-only builder audit may
be added only if measured link disagreement justifies it.

**Rejected alternatives.** Validate every singleton or multi-decision link with builder-time
interventions; maintain a fixed builder intervention budget from the first run. Both duplicate
the ranker's pairwise measurement machinery, add remote calls, and create another cache surface
before evidence shows it is needed.

## Generation and caching use one bounded semantic teacher request

**Decision.** Use deterministic extraction for structural facts and safety/representability
prefiltering only. For relation QA, make at most one batched, cached teacher proposal call per
document; deterministic code must not generate the question or accepted answers from templates.
Permit explicit abstention rather than retries. Use one build cache and one document/action-vector
score cache. Retire separate ladder, decision, and schema caches after migration.

The escalation teacher is pinned to `nvidia/nemotron-3-super-120b-a12b:free` through OpenRouter
and uses `OPENROUTER_API_KEY`. A model, provider, prompt, or response-schema change creates a new
builder pin and invalidates cached proposals. The pin also records a bounded generation
configuration: 8,192 maximum completion tokens and 1,024 maximum reasoning tokens with the
reasoning trace excluded. Nemotron's selected route has reasoning enabled by default, so merely
excluding its trace is not a reasoning-budget control. The request uses OpenRouter strict JSON
Schema with required `relations` and `candidate_accounting`; JSON-object mode alone accepted an
observed `{}` reply and is therefore rejected for this contract. Each request binds its schema to
the displayed `S#` inventory (valid relation role/kind enums and `S#` ledger cardinality), but
does not bind teacher-authored answers to finite candidate values. It permits an empty relation
list without an empty enum. Deterministic compilation verifies exact once-only ledger coverage and
every cross-field relation contract. Evidence cards are non-gating navigation aids; the response
contains no evidence-window ID. The compiler derives a minimal same-clause, or explicitly
cue-linked adjacent-clause, anchor from resolved arguments and validates it. For a recognized
clinical assessment/plan structure it may use one explicit condition heading plus its following
plan bullets as a bounded section anchor, but only when every resolved argument and the direct
relation cue are inside that one section. This prevents both duplicate-span confusion and
arbitrary proximity links without forcing semantic search through a deterministic pair/window
filter.

**Rejected alternatives.** Teacher generation for every assertion; forced retries until every
decision has a probe; separate cache families for every subtype. These optimize nominal coverage
at the cost of latency, circular failure loops, and operational complexity. Missing linked QA is
handled by complete-document fallback and counterfactual priority instead.

## Relation ontology separates prescription from procedure treatment

**Decision.** Relation signatures remain directional, condition/diagnosis first.
`prescribed_with` is condition -> drug prescribed or explicitly used for that condition.
`treated_with` is condition -> medical procedure only. A proposal with a drug in
`treated_with` is rejected; it is not silently accepted with reversed arguments. Historical v1
occurrence-only proposals may be migrated to `prescribed_with` only at artifact-read/compile
compatibility boundaries, with the new relation stored in the artifact.

**Decision.** A relation may mix linked decision arguments and source-grounded context/literal
arguments. Only linked arguments require lattice properties and participate in routing/anchors;
all arguments require exact source evidence and their declared runtime type. This prevents useful
labs, therapy, status, and category context from being invented as controlled detector decisions.

The deterministic prefilter is limited to safety and representability: include only accepted
frozen controlled occurrences with an exact source span, a legal non-KEEP/non-placeholder level,
and a canonical class compatible with some ontology role. It does not choose relation pairs,
evidence windows, or semantic facts. Every exposed controlled runtime type has a canonical map;
`medical-procedure` maps to `procedure`. Missing a map is a configuration error. The prompt shows
each retained occurrence once in source order as a single line, for example
`[S7: arthritis | condition | levels: joint disease; inflammatory condition]`; it never shows
internal IDs, hashes, offsets, aliases, detector type, or polarity.

**Decision.** A context argument is an exact quoted literal. It requires at least one linked `S#`
argument. The compiler first derives a candidate anchor from resolved linked spans, then resolves
the literal uniquely inside it and derives its class using the adapter's closed lexical/structural
literal contract plus the relation-slot class. It rejects an ambiguous, duplicate, untyped,
slot-incompatible, or protected controlled-span resolution. There is no global context-candidate
inventory and no context-literal ledger obligation.

**Decision.** The teacher request has a maximum of 12 relations per document, seeks diverse valid
coverage, and accounts for every displayed `S#` exactly once as `emitted`,
`exhausted_no_relation`, or `unsupported`. `emitted` records a proposal attempt, not acceptance;
`exhausted_no_relation` records a considered label with no explicit supported relation; and
`unsupported` records insufficient source role/connection for any ontology relation, never zero
utility relevance. The compiler records omitted prefilter candidates separately as
`ineligible_prefilter`, sanitizes ledger reasons so they do not repeat protected text, and
cross-checks emitted labels against proposals. A missing, duplicate, or inconsistent row yields a
`ledger_inconsistent` diagnostic without rejecting an otherwise valid compiled relation.
Diversity is a bounded coverage diagnostic, never a quota or hallucination incentive; the
artifact stores attempted, accepted, and rejected counts by relation type.

**Evidence and leakage decision.** Evidence cards are source-navigation aids, not semantic gates.
The teacher searches the full document and returns labels/literals, question, accepted answers,
and scoring contract; it does not return an evidence-window ID or authoritative quote. The
compiler derives a candidate same-clause anchor from linked spans, or an adjacent-clause anchor
only with an unambiguous word-boundary cue. A recognized condition-headed assessment/plan block
is a third, bounded anchor form when all arguments and the direct cue stay inside that block; it
does not permit cross-turn or generic proximity linking. The compiler resolves a quoted literal
within the selected anchor and then validates direct connection, polarity, and contradiction.
Protected-term lint covers every controlled decision and deterministic aliases, whether or not
displayed to the teacher; raw detector-only non-decisions remain outside the protected set. Answer-to-question leakage remains a
separate check. Teacher-authored accepted answers are nonempty free strings and must not be bound
to a finite schema enum.

**Deferred.** Reader stability repetitions and answer-option permutation remain at their current
single-pass gates until smoke results justify and preregister a changed protocol. No
repetition/permutation behavior or thresholds are changed by this relation-QA work.

**Required migration.** Implement this contract as a new pinned relation-teacher prompt and
response-schema/artifact revision (v4 or a later explicitly named successor). The migration must
invalidate old proposal caches and test label-only grounding, exact-literal ambiguity, medical
procedure mapping, leakage scope, ledger invariants, cross-turn/proximity rejection, relation
directions, joint anchors, reader rejection, and non-forced diversity diagnostics.

## Context assertions use one joint representative generalization anchor

**Decision.** A rewarded context assertion must pass on `doc_orig`, pass on one joint legal
generalization anchor, and fail on the all-placeholder anchor. In the joint anchor every linked
decision takes its coarsest legal non-placeholder action whose lattice semantics still entail
the assertion; unrelated decisions remain KEEP. Reject the assertion if no such vector exists or
the reader fails it. Store the complete vector/hash and supported property level.

Do not create one question or call per rung. The gate establishes generalization over
placeholder for utility. QA does not reward generalization over KEEP inside the supported band;
ranker-v2's exact count objective supplies pressure toward the coarsest semantically viable
action.

**Rejected alternatives.** Validate only original versus placeholder; validate every rung;
require sampled rollout actions during build. The first proves only KEEP-over-placeholder
utility. Per-rung validation multiplies cost without distinct measurements. Rollout sampling is
not needed to establish semantic support.

## Runtime scoring is document-batched

**Decision.** Score all context assertions for one `(rollout doc_p, reader/scorer pin)` in one
batched reader request. Score delivered assertions deterministically where possible and batch
the remainder. Cache the complete score vector at document/action-vector granularity. Gate
reader and remote calls per rollout as well as wall time.

**Rejected alternatives.** One reader call per assertion; assertion-level runtime caches; a
wall-time-only cost gate. These reproduce the reader bottleneck, fragment cache reuse, and hide
excessive call counts behind hardware-dependent timing.

## Direct doc_p scoring remains inside complete counterfactual evaluation

**Decision.** A one-decision counterfactual regenerates changed `doc_p`, runs the complete pinned
remote task and extraction path to produce `out_final`, and rescores the full weighted vector of
context and delivered assertions. Batched direct-`doc_p` scoring is only an efficiency
optimization and cannot bypass the remote round trip. The measured pair is bounded local
contextual evidence, not independent causal attribution.

**Rejected alternative.** Recompute only directly linked `doc_p` assertions. That would miss
changes to delivered output and other assertions and would not measure the complete utility
difference required by ranker v2.

## QA thresholds are frozen experiment state

**Decision.** A QA threshold manifest fixes, before artifact build and full RL, the eligibility
prefilter and no-eligible-span state; joint representative-action selection and ties; the deferred
reader-stability/permutation policy and threshold; family budgets, group construction, structural
cap, and missing-family state; and call/wall-time budgets. Train/development preflight may
instantiate values once under those declared protocols. Full RL and attacker results cannot alter
them.

**Rejected alternatives.** Leave triggers qualitative; hard-code unsupported constants before
measurement; choose values after observing full training or attacker outcomes.

## Delivered truth never comes from the ceiling output

**Decision.** Delivered assertion truth and gold come from the adapter's authoritative source,
human reference, or fixed task schema. Ceiling output may establish task-model feasibility or an
explicit model-relative diagnostic only. An omission means an authoritative required fact is
absent from `out_final`; ceiling omission is reported separately.

**Rejected alternative.** Use ceiling agreement as delivered truth. That inherits task-model
omissions and errors and confuses model-relative feasibility with task correctness.
