---
type: reference
status: current
created: 2026-07-12
updated: 2026-07-14
tags: [qa, reward-design, utility-components, context-preservation, credit-routing,
       interactive-ranker, spec]
companion: [docs/specs/RL/interactive-ranker-v2.md,
            docs/specs/RL/interactive-ranker-v2-decision-log.md,
            docs/specs/RL/interactive-ranker-v2-diagnostics.md,
            docs/specs/RL/qa-builder-v2-decision-log.md,
            docs/specs/RL/training-task-env.md]
---

# QA builder v2 — context-preservation utility for interactive ranker v2

**Status: normative design with an implemented ACI builder/scorer; empirical gates remain
uncertified until the preregistered smoke and support runs complete.**

QA builder v2 produces a small, frozen set of utility assertions that reward truthful
generalization when it preserves useful context that a generic placeholder destroys. It also
measures the quality of the final delivered output. QA is a utility instrument, never a privacy
measurement. Privacy remains held-out LLM re-identification success on `doc_p` and `out_final`
at matched realized privacy and identical settings.

## Core design

The design has two complementary measurement channels:

1. **Context preservation on `doc_p`.** Semantic-property and contextual-decision assertions
   test whether the transformed input still carries category, function, and relational meaning.
   These are the primary signal that a legal truthful generalization can preserve useful context
   better than a generic placeholder.
2. **Delivered utility on `out_final`.** Content and schema assertions test omissions, required
   structure, field/category/status agreement, exact recovery, and symbolic relation
   preservation after the complete round trip.

Neither channel substitutes for the other. A placeholder may preserve symbolic utility through
schema slots and local inversion without preserving semantics. Conversely, `doc_p` may retain
useful context that the remote task later omits. The reward must observe both effects.

The builder optimizes reliable measurements, not accepted-pair count. Missing QA coverage never
removes a policy decision or implies zero utility relevance.

Every rewarded context assertion must distinguish three points: it succeeds on `doc_orig`,
succeeds on at least one legal non-KEEP, non-placeholder transformed anchor whose lattice
semantics support the asserted property, and fails on the all-placeholder anchor. This is the
minimum evidence that the assertion rewards usable generalization rather than merely KEEP.

## Invariants

1. **Detect once.** QA and RL consume the same frozen occurrence/decision artifact. The builder
   never runs detection.
2. **Stable many-to-one decisions.** Occurrences link to stable IDs; equivalent occurrences map
   to one policy decision.
3. **Assertions precede questions.** The builder first compiles grounded assertions. Questions,
   when needed, are only scoring forms of accepted assertions.
4. **Teacher proposals are not gold.** Code owns IDs, canonical links, relation vocabulary,
   channels, and validation. The teacher authors contextual relation questions and accepted
   answers; code validates rather than deterministically inventing that semantic content.
5. **Probe links are routing hints.** They route provisional credit but do not certify causality.
6. **Measurement and routing are orthogonal.** Assertion family records what is measured;
   `scope: linked | global` records how ranker v2 routes it.
7. **Counterfactuals provide bounded local evidence.** For a tested action pair, ranker v2
   substitutes measured local contextual evidence for provisional routing. This is not unbiased
   causal attribution and does not establish independent decision causality.
8. **Pins are transitive.** Detector, task, model, extractor, reader, scorer, assertion, or gate
   changes invalidate dependent artifacts and caches.
9. **Thresholds are preregistered.** Final RL or attacker results cannot tune QA gates, weights,
   generation, or validation.
10. **Repeated-context leakage remains an audit assumption.** One decision may control repeated
    occurrences, but repetition can still increase inference risk.

## One deep module

QA builder v2 exposes one external interface:

```text
build_utility_artifact(
    frozen_environment,
    task_adapter,
    source_documents,
    references,
    optional_ceiling_diagnostics,
    builder_pin,
) -> UtilityArtifact
```

Callers do not orchestrate detection, relation extraction, question generation, validation,
coverage accounting, or cache keys. Those are implementation details behind this interface.

The task adapter is the only public variation seam:

```text
TaskAdapter:
  fixed_schema
  assertion_ontology
  deterministic_candidates(document, environment)
  score(assertion, context)
```

Add an adapter only when a second task genuinely requires different schema or assertion logic.
Do not expose separate public ladder, decision, schema, validator, or compiler modules.

**Compiler terminology is normative.** Relational-assertion compilation is the complete
deterministic-first operation that may invoke the one-call teacher escalation and then validates
the proposal against frozen IDs, legal properties, and exact evidence. Artifact packaging is the
subsequent local-only operation that assigns IDs, links, weights, coverage states, hashes, and
pins. A packaging-only command must not be described as the assertion compiler.

Runtime scoring has one interface:

```text
score_utility(artifact, doc_id, doc_p, out_final) -> {assertion_id: ScoreRecord}
```

Generated `out_p` may remain in the round-trip artifact for diagnostics, but semantic and
contextual QA does not read it.

Runtime scoring batches all context assertions for one `(doc_p, reader/scorer pin)` into one
reader request. Delivered assertions are deterministic where possible and otherwise share one
batched request. Cache keys are document/action-vector level, never assertion-call level.

## Frozen environment contract

The builder consumes one immutable artifact containing:

```yaml
environment_hash: sha256:...
documents:
  aci/D2N002:
    source_hash: sha256:...
    occurrences:
      - occurrence_id: occ:...
        start: 120
        end: 134
        surface: hypothyroidism
        runtime_type: health-condition
        polarity: active
        detector_provenance: {...}
        overlap_disposition: accepted
        decision_id: dec:...
    decisions:
      - decision_id: dec:...
        runtime_type: health-condition
        occurrence_ids: [occ:...]
        controlled: true
        action_menu_hash: sha256:...
```

Every controlled occurrence maps to exactly one decision. IDs derive from canonical versioned
fields and hashes, not list position or `surface.lower()`. Canonical substring matching is
migration-only and produces explicitly low-confidence links.

## Building per-sample assertions

The task schema is fixed once per adapter. Each document receives an assertion manifest through
one simple pipeline.

### Deterministic extraction

Code derives what it can authoritatively:

- frozen occurrence IDs, offsets, types, surfaces, polarity, aliases, and lattice levels;
- explicit age, sex, condition, drug, procedure, and other task facts;
- exact source and human-reference matches;
- fixed schema sections and row keys;
- exact ceiling matches.

### One bounded relation-teacher request

For every document with at least one eligible controlled `S#` span, make one pinned, batched,
cached relation-proposal call. Deterministic extraction supplies structural facts and the
safety/representability prefilter; it does not create contextual relation QA templates. A document
with no eligible controlled span makes no relation-teacher call and records that explicit
no-candidate state. The teacher may abstain; do not retry to chase coverage.

The teacher receives `doc_orig`, authoritative reference evidence when available, and the compact
relation prompt described in [Teacher prompt specification](#teacher-prompt-specification). It
selects from the ACI adapter's closed relation vocabulary. Abstain rather than retrying to chase
coverage.

The relation teacher is pinned to
`nvidia/nemotron-3-super-120b-a12b:free` through
`https://openrouter.ai/api/v1`, authenticated by `OPENROUTER_API_KEY`. Changing model, provider,
prompt, response schema, or generation configuration invalidates the build cache and artifact
pin. The request sets no completion or reasoning token cap: the model's reasoning is mandatory
on this route, and observed caps repeatedly produced empty replies or a truncated source scan
(the r16 smoke's reasoning cut off mid-document, leaving explicit relations unproposed). The
configuration only excludes the reasoning trace from the returned reply, keeping it out of the
cache and artifact. The OpenRouter wire request uses strict JSON Schema,
with `relations` and `candidate_accounting` required at the top level; basic JSON-object mode is
insufficient because `{}` is syntactically valid but semantically unusable. Per document, the
wire schema constrains relation roles to `subject`/`object` fixed by argument position, argument
kinds to `linked`/`context`, linked short labels to the document's displayed `S#` labels, and
ledger cardinality to the number of displayed `S#` labels. Because the compiler unconditionally
rejects a relation with no linked argument, the argument pair is bound to the three shapes
`linked+linked`, `linked+context`, and `context+linked`; a zero-linked pair is unrepresentable
on the wire (the r16 smoke's only proposal was exactly that always-rejected shape). It must
permit a clean empty relation list and must not create an empty enumeration for any optional
field. It must not bind teacher-authored accepted answers to a finite answer enum. The compiler
remains the authority for cross-field semantics and exactly-once candidate coverage.

### Teacher prompt specification

This section is normative for the relation teacher's human-facing prompt. It deliberately
separates semantic search from deterministic validation.

**Responsibilities.** The teacher searches `doc_orig` for as many source-grounded,
task-relevant, non-duplicate relations as the preregistered cap permits, with relation diversity
where evidence supports it. It authors the contextual question, accepted answer(s), scoring
contract, and semantic generalization requirements. It must not fabricate a relation merely to
cover a type. Deterministic code maps the teacher's short labels back to frozen records and
validates source grounding, argument types, legal generalization support, protected-term leakage,
duplicate fact groups, reader gates, and representative anchors.

**Prompt presentation.** The human-facing instruction is a short fixed outline, not a narrative
wall of text. It must appear in this order:

```text
TASK
Find as many explicit, source-grounded, non-duplicate relations as the cap permits.
Prefer diversity only when the source supports it. Abstain rather than inventing a fact.

HOW TO INSPECT THE SOURCE
Read the full document. Use evidence cards to find and name source spans.
Consider every displayed span. A relation may use any ontology-compatible spans in the source;
cards do not limit semantic search.

PRIVACY-SAFE QA
Write a question and accepted answer(s) that test the relation's meaning.
Never copy a displayed controlled source span or alias into either; use its displayed
generalization level. An exact uncontrolled literal is allowed only when it is the measured fact.

RELATION INVENTORY
<the concise plain-text inventory below>

WORKED EXAMPLES
<the three examples below>

EVIDENCE CARDS
<compact cards for this document>

RESPONSE
State the required record contents: relation; subject then object argument; question; accepted
answers; the fixed scoring contract. A displayed span is a linked argument carrying its S-label
as span_label and exactly one of that label's listed levels, copied verbatim, as
support_property. Every relation needs at least one linked S-label argument; never quote a
displayed span as a context literal; an uncontrolled argument is a context literal with its
exact source text. Emit each distinct fact once, at the S-label inside the sentence that states
the relation; other labels of that value are duplicate_mention rows naming the emitting label.
Include compact record-shaped examples (one all-linked, one with a context literal). Then
request one concise accounting record for every displayed label, with a short label-referential
reason on every row.
```

The instruction must not contain implementation/wire prose such as “the constrained wire
schema,” JSON-null instructions, hash rules, offset rules, or internal validation gates. Those
belong solely to the provider response schema and deterministic compiler. The only
response-level language visible to the teacher is the semantic task and the short list of
required record contents above. Naming the record fields (`span_label`, `support_property`,
`literal`, argument kind) is required, not optional: the r16 smoke's reasoning trace shows the
teacher planning a free-text format (“format? Not fully specified”) and then falling into the
degenerate all-context wire shape the compiler always rejects. The record-contents list aligns
the teacher's plan with what the constrained decoder will demand.

The prompt includes these compact worked examples before document-specific evidence cards:

```text
Example A — drug, not procedure
Source: “… [S1: arthritis | condition | levels: joint disease; inflammatory condition] …
prescribe [S2: Ultram | drug | levels: opioid analgesic] …”
Relation: prescribed_with(S1, S2)
Safe QA: “Which medication category was prescribed for the joint condition?”
Accepted answer: “opioid analgesic”
Never call this treated_with.

Example B — explicit monitoring
Source: “… To follow the [S3: thyroid condition | condition | levels: endocrine condition],
order thyroid labs …”
Relation: monitored_by(S3, "thyroid labs")
Safe QA: “What follow-up testing was ordered for the endocrine condition?”
Accepted answer: “thyroid labs”

Example C — do not infer a relation
Source: “… autoimmune panel …” with no explicit linked condition or purpose.
Action: do not emit monitored_by merely because the test is present.
```

**Teacher input.** The prompt contains the full source document plus compact, readable
relation-focused evidence cards. A card contains a source excerpt and inline short labels, for
example:

```text
E12
... acute exacerbation of [S7: arthritis | condition | levels: joint disease; inflammatory condition]
... prescribe [S8: Ultram | drug | levels: opioid analgesic] ...
```

**Detected-span presentation.** The teacher sees source occurrences, not internal detector
records. Before prompt construction, code selects a detector occurrence only when all of the
following hold:

1. it is an accepted frozen occurrence with an exact span in `doc_orig`;
2. it is linked to a controlled decision;
3. that decision has at least one legal non-KEEP, non-placeholder generalization level; and
4. its adapter-mapped coarse class can fill at least one relation-inventory role (`condition`,
   `symptom`, `drug`, `procedure`, or `provider`).

Thus age, demographic/person labels, unsupported detector types, and any occurrence without a
usable lattice profile are absent from the teacher prompt. This is eligibility filtering, not a
semantic judgement about whether two retained spans have a relation.

Every retained *source occurrence* receives one document-local label in source order: `S1`, `S2`,
…. Repeated mentions receive separate labels because they are separate source anchors, even when
they map internally to one policy decision. The visible form is exactly one line:

```text
[S7: arthritis | condition | levels: joint disease; inflammatory condition]
```

`arthritis` is the exact source text, `condition` is the adapter's human-readable relation class,
and the listed levels are the decision's legal non-KEEP/non-placeholder lattice meanings. Code
retains the opaque occurrence ID, decision ID, offsets, detector runtime type, aliases, polarity,
and the `S7` mapping; none of those fields is shown to the teacher. The teacher refers to `S7`
in its relation record, and the compiler resolves that label to the exact occurrence before any
validation.

An uncontrolled source-grounded literal is an exact quoted literal authored by the teacher. It is
permitted only in a relation with at least one linked `S#` argument. The compiler first resolves
the linked labels and derives candidate evidence anchors from them; it then resolves the literal
as one exact, unique span inside an anchor. It derives the literal's canonical class from the
adapter's closed literal contract: explicit lexical/structural rules for `test`, `procedure`,
`provider`, `status`, and `category`, plus the relation object's permitted slot class. It rejects
an ambiguous, duplicate, untyped, or slot-incompatible resolution. The compiler privately retains
the exact source spans and mappings needed for that literal.

The prompt must not expose unrelated detector types (for example age or `demographic-other`),
opaque occurrence/context hashes, character offsets, aliases, the ambiguous word `surface`, a
global context-candidate inventory, or a global evidence-window/pair matrix. Use `text` or
`source span` for the displayed source text, and `allowed generalization levels` for the lattice
levels; do not use `legal_support_properties` in teacher-facing prose.

Evidence cards are navigation and grounding aids, not a semantic prefilter. In particular, code
must not generate or expose `eligible_pairs`, and must not use relation-specific cue/order
heuristics to decide which pairs the teacher may propose. Cards may exclude policy-ineligible
spans, but must not decide that an ontology-compatible source pair lacks a relation. The teacher
may propose any source-grounded ontology-compatible relation it finds in the supplied document and
cards; the compiler performs the subsequent deterministic evidence check.

**Evidence grounding.** The teacher does not select an evidence-window ID or manufacture an
evidence quote. The compiler resolves linked arguments first and derives the smallest candidate
source anchor containing them: normally one clause, or two adjacent clauses only when an
unambiguous, word-boundary relation cue links them. For a recognized clinical assessment/plan
block, it may instead use the single explicit condition heading and its following plan bullets as
one bounded source section. This fallback applies only when every resolved argument is in that
same section and the relation's direct cue occurs there; it is not a cross-turn or free-form
proximity bridge. It resolves a quoted literal only within the chosen anchor, then verifies that
the completed relation is directly supported. The compiler rejects proximity links, cross-turn
links, cue substrings, contradictory polarity, and a relation whose arguments cannot be directly
connected in that anchor. Two clause conventions are normative for transcript sources: the cue is
searched within the clause or two adjacent clauses holding the arguments (an explicit cue such as
"ca n't take" legitimately precedes the subject), and a period inside a run ("if ... and
prescribe") is a spoken hesitation marker, never a clause boundary.

**Problem-block anchor (multi-turn span pairs).** A spoken assessment/plan discusses one problem
across several turns; a true relation can straddle the patient acknowledgments inside it (a
condition named when the problem is introduced, a test ordered for it a sentence later —
`monitored_by(arthritis -> autoimmune panel)` in D2N002, spike-confirmed 2026-07-14). After the
clause and plan-section anchors fail, the compiler grounds the relation in the one problem block
containing every argument, bounded by the assessment opener and each "for your <next> problem"
switch; it never bridges a problem switch or unrelated small talk, and the cue check still runs
over the argument window. **Hedge guard:** because these broad anchors span conditional talk, a
relation whose argument window is conditional/hypothetical ("if your symptoms continue",
"possibly", "we can consider") is rejected as `hedged_relation` at every anchor scope — the
authoritative reference states such plans conditionally, so asserting them as fact would violate
the truth-source rule. The hedge match is tightened so the spoken "if ... and prescribe"
disfluency and dosing "as needed" do not block a real prescription.

Two closed extensions cover the clinical indication form. A detector-typed condition whose
surface names a performed procedure (closed lexicon: transplant, surgery, -ectomy, -plasty,
bypass, graft, replacement, repair) may also fill procedure slots, displayed to the teacher as
`condition/procedure`; an ordinary condition surface may not. And `treated_with` accepts the
reversed indication connector "<procedure> for <condition>" within one clause ("had the kidney
transplant a few years ago for some polycystic kidneys" — stated verbatim in the D2N002
reference). Class gating keeps the generic word "for" inert for every other argument pairing. Cards help the teacher navigate; they neither limit search nor
constitute accepted evidence. There is no `evidence_window_id` in the teacher response or relation
artifact.

**Leakage scope.** Protected-term lint applies to every controlled decision and its deterministic
aliases, whether or not that decision was eligible for teacher presentation. Display eligibility
only limits semantic search and candidate accounting; it never removes privacy protection. Raw
detector-only occurrences that are not controlled decisions are outside this protected set. A
context literal may appear in an accepted answer only when it resolves to an uncontrolled literal;
if it also resolves to a protected controlled span, the proposal is rejected as ambiguous. The
compiler separately checks answer-to-question leakage.

Three lint refinements are normative, each with a measured motivation. First, tokens of a
decision's declared legal generalization levels are exempt from that decision's token-overlap
lint: QA is directed to use those levels verbatim, and without the exemption "solid organ
transplant" could never appear in a question about a kidney transplant. Full-term containment of
the protected surface still rejects. Second, before the leakage gates run, the compiler
substitutes each linked argument's protected surface/alias in the question and accepted answers
with that argument's teacher-selected support property, recording `sanitized_qa` in the
assertion evidence. This is mechanical substitution of teacher-chosen content, not authorship:
three consecutive live smokes wrote the source surface into otherwise valid QA despite
escalating prompt guidance, and the leakage gates rerun on the substituted text. Third,
placeholder-label tokens of the linked argument types (for example "medication", "condition")
are information-free for level-based QA and are exempt from answer-to-question overlap; the
answer's distinguishing tokens remain linted.

**Candidate accounting.** The response includes exactly one concise accounting row for every
displayed `S#` label: `emitted`, `duplicate_mention`, `exhausted_no_relation`, or `unsupported`.
`emitted` means the teacher attempted a relation using that label; it does not claim compiler
acceptance. `duplicate_mention` means another label of the same underlying value already carries
the fact, named in the reason; without this state a live smoke fabricated one relation per
repeated label to make every label `emitted`, crowding out real relation types under the cap.
`exhausted_no_relation` means the label was considered but no explicit, ontology-compatible
relation is supported. `unsupported` means the source does not establish enough semantic role or
connection for that label to support any ontology relation; it is not a claim that the policy
decision lacks utility. The wire schema requires a nonempty bounded reason on every row. The compiler records prefilter exclusions separately as
`ineligible_prefilter`; those records are not teacher ledger members. Quoted literals are not
ledger members. Reasons must be concise, sanitized, and label-referential: they
must not repeat protected source text. This is a bounded coverage diagnostic, never a requirement
to invent a relation or a reason to remove a ranker decision. The compiler cross-checks the ledger
against proposals and preserves attempted, accepted, and rejected counts by relation type. A
missing, duplicate, or proposal-inconsistent ledger row records `ledger_inconsistent` as a
diagnostic; it must not reject an otherwise valid relation or erase its compilation outcome.

**Relation inventory.** Present this inventory as structured instructional text, never JSON.
The arrow is the complete argument-direction contract; do not separately repeat `ordered`,
`ordered_roles`, or `argument_classes` in teacher-facing material.

```text
RELATION INVENTORY

Use only these directed relations. The arrow gives argument direction.
Do not reverse it. Emit a relation only when the source explicitly supports it.

1. prescribed_with
   condition or diagnosis → drug
   Use when a drug is prescribed, continued, taken, or used for that condition.
   Example: “arthritis … prescribe Ultram” → arthritis prescribed_with Ultram.
   Do not use for a procedure.

2. treated_with
   condition or diagnosis → medical procedure
   Use when a procedure is used to treat the condition.
   Example: “stone treated with lithotripsy” → kidney stone treated_with lithotripsy.
   Never use for a drug.

3. monitored_by
   condition or diagnosis → monitoring test, monitoring procedure, or provider
   Use when the source says the condition is monitored, evaluated, checked, or followed by it.
   Example: “to follow the thyroid condition, order thyroid labs” → condition monitored_by
   thyroid labs. Do not infer monitoring from proximity or a test that appears elsewhere.

4. contraindicated_because_of
   drug or procedure → condition or diagnosis
   Use only when the source explicitly states the treatment/procedure cannot be used because of
   the condition.

5. causes_or_explains
   condition or diagnosis → condition or symptom
   Use only for an explicit causal or explanatory statement.

6. referred_to
   condition or diagnosis → provider or procedure
   Use when the source explicitly refers the patient for that provider/procedure.

7. has_status
   clinical concept → status
   Use an explicitly stated status only. An adapter must preregister an explicit source form that
   connects the concept to the status; otherwise this type is expected to have zero coverage.

8. has_category
   clinical concept → category
   Use an explicitly stated category/classification only. An adapter must preregister an explicit
   source form that connects the concept to the category; otherwise this type is expected to have
   zero coverage.

9. has_condition
   person → condition or diagnosis
   Use when the source states the person has, presents with, or was diagnosed with the condition.

10. takes_medication
    person → drug
    Use when the source states the person takes, is on, continues, or was prescribed the drug.

11. underwent_procedure
    person → medical procedure
    Use when the source states the person had, underwent, or received the procedure.
```

Relations 9–11 anchor a generalizable clinical span to a **person**. The person is the subject
and the clinical span is always the object and the answer; the person is never the answer (it has
no generalization level). These relations exist for coverage: a clinical span that participates in
no condition↔drug↔procedure relation still earns a context assertion by anchoring to the person.
`age` is deliberately excluded — it is a demographic attribute, not a per-fact relation subject,
and would only duplicate person-anchored facts with a weaker, non-disambiguating anchor.

The prompt additionally requires semantic QA: questions and accepted answers must not copy a
displayed controlled source span or alias. They should use the selected allowed generalization
level where needed; an uncontrolled context literal may remain an answer when that is what the
relation measures. Accepted answers are teacher-authored nonempty strings, not deterministic
templates or a schema enum. The compiler validates them for answer leakage, protected-term
leakage, grounding, and reader support. The worked examples above are mandatory prompt content;
add only one short contraindication example if the task adapter supports that relation.

The provider's strict response schema remains a machine-level constraint. It may bind short labels
and cardinality, but its field-level machinery must not be reproduced as prose instructions that
compete with the semantic task.

The teacher cannot invent relation types, span labels, or unsupported source facts. It does author
the contextual QA question, accepted semantic answer(s), and scoring contract for an otherwise
compiled relation; deterministic code validates rather than templates that semantic content.

Relation arguments have three disjoint forms. A **linked decision argument** is a controlled,
frozen occurrence ID and carries a legal lattice-support property; it receives routing links and
the joint representative-anchor check. A **context/literal argument** is an exact, typed,
source-grounded span (for example a lab, physical therapy, status, or category) that is not a
detector decision and never requires a lattice action. A **placeholder-anchor argument** is a
controlled *identity* occurrence (PERSON, and any type that is placeholder-by-rule) that has no
generalization level: it is never the answer and carries no `support_property`; it exists only to
anchor the answer to a specific individual. All three forms require exact source evidence. Only
linked and placeholder-anchor arguments enter `occurrence_ids`; only linked arguments carry a
`decision_requirements` lattice action.

**Identity-token anchoring (relations 9–11).** A person's surface differs across renders — the
real name in the clear document, `<PERSON_2>` after substitution — and it has no level to
reference. To give the reader a *stable* anchor, identity types are pre-substituted to their
frozen placeholder tokens in **two** views: the teacher's source view and the gate's `original`
reader context. The teacher therefore never sees the real name (a privacy benefit) and anchors
its question on `<PERSON_2>`, which then appears identically in the original, representative,
placeholder, and runtime `doc_p` contexts. This is faithful, not a hack: identity types are
placeholder-by-rule, so the pipeline never emits the real name — the honest utility baseline is
*clinical-clear, identity-anonymized*. The clinical object remains the only span that varies
across the three gate renders, so the three-point gate still discriminates on the object
(original ✓, representative ✓, all-placeholder ✗). The person→token assignment must be identical
across the teacher view, all three gate contexts, and runtime, reusing the frozen
occurrence→placeholder fills so the anchor never drifts. Because the anchor is a per-person token,
multi-person documents disambiguate for free (each person is a distinct token); no patient-vs-
provider role classification is required.

Placeholder-anchor arguments are exempt from the `literal_will_be_substituted` guard (they are
anchors by design, referenced by their placeholder token, not doomed literals), and the compiler
must reject any relation that makes a placeholder-anchor argument the answer.

Every adapter must map each controlled runtime type that it exposes to one canonical relation
class before prefiltering. In particular, `medical-procedure` maps to `procedure`; a missing map
is a configuration error, not grounds to silently remove a valid controlled procedure.

**Versioning and deferred reader checks.** This redesign requires a new relation-teacher prompt
revision and response-schema/artifact revision (v4 or a later explicitly named successor); both
are cache pins. Reader repetition and answer-option permutation remain deliberately deferred:
the current single-pass contextual reader gates remain in force, but no repeated/permuted protocol
is introduced or claimed until a smoke result motivates and preregisters it.

### Deterministic compilation

A relation is accepted only when:

- every linked `S#` label exists and every quoted literal resolves uniquely in the derived anchor;
- the relation permits the arguments' canonical classes;
- the compiler-derived evidence anchor directly connects the arguments;
- polarity is consistent;
- no source contradiction exists.

The one bounded request uses the compact span labels, evidence cards, relation inventory, and a
preregistered maximum of **12** relations per document. Context arguments may reference a local
card label or an exact quoted literal under the resolution rule above. Coverage and relation
diversity are diagnostics, not a reward for fabricating one relation of every type. The compiler
rejects reversed roles, duplicate fact groups, unsupported literals, answer/protected-term
leakage, and invalid scoring contracts. It records per-document and corpus attempted, accepted,
and rejected counts by relation type. Missing coverage never removes a ranker decision or implies
zero relevance.

General domain knowledge absent from the document cannot become an assertion. Teacher failure or
abstention produces an explicit missing/rejected state, not a fabricated fallback.

Each task adapter declares one authoritative truth source: the human reference where it contains
the required fact or relation, otherwise explicit `doc_orig` evidence. Ceiling output is only a
feasibility context for delivered assertions; it is never truth. Teacher interpretation alone is
never sufficient.

## Utility assertions

Use only two measurement families. Routing scope is a separate field.

### Context assertions on `doc_p`

These combine the old ladder and downstream-decision surfaces into one family. Subtypes are
metadata, not separate pipelines:

- `semantic_property`: category or function retained by truthful generalization;
- `contextual_relation`: a relation or bounded decision requiring preserved context.

Each assertion is defined and reader-validated on `doc_orig`, fails the all-placeholder anchor,
passes its pinned joint representative generalization anchor, and is scored on each rollout's
`doc_p`.

Use one joint representative anchor per assertion. Every linked decision takes its coarsest legal
non-placeholder action whose lattice semantics still entail the assertion; every unrelated
decision remains KEEP. If no such joint action vector exists or the reader fails it, reject the
assertion. Store the complete action vector and hash plus the supported property level. This
avoids combinatorial anchors and requires no question or model call per rung.

This gate establishes that truthful generalization retains more measured utility than
placeholder. It does not make QA reward generalization over KEEP inside the supported semantic
band. Ranker-v2's exact local count objective supplies pressure away from KEEP and toward the
coarsest semantically viable action. This division of labor is intentional.

Semantic accepted answers for fixed structural assertions come from the frozen lattice or adapter
ontology. Contextual gold and dependencies come from accepted compiled relations. Do not generate
one question per lattice rung and do not force exactly two questions per span. Generate one
measurement only when an accepted task-relevant assertion exists.

Relation QA is teacher-authored in the same bounded relation proposal: the teacher supplies the
question, accepted semantic answer(s), and scoring contract. Deterministic code must never
invent that relation semantics from templates; it validates grounding, answer/protected-term
leakage, duplicate fact groups, reader support, and the joint anchor. Structural and delivered
assertions may use deterministic contracts because their semantics are already fixed by the task
adapter. Questions are static and may not contain protected-value locators; they are not rewritten
per rollout.

The generator does not see the all-placeholder document. Validation remains independent and
prevents wording from overfitting one floor realization.

### Delivered-output assertions on `out_final`

These measure what the user receives:

- authoritative source/reference-backed content coverage and omissions;
- required sections and parseability;
- field/category/status agreement;
- exact recovery and symbolic relation preservation;
- one non-overlapping global task-quality criterion when justified by the adapter.

Structural compliance alone is diagnostic or receives a small preregistered weight. It is not
semantic retention. Field assertions are scored separately; a schema aggregate cannot recompute
them.

Delivered assertion truth and gold come only from the adapter's authoritative source, human
reference, or fixed task schema. Ceiling output may establish task-model feasibility or define an
explicitly model-relative diagnostic; it never defines truth. An omission assertion means an
authoritative required fact is absent from `out_final`, regardless of whether the ceiling also
omits it; ceiling omission is reported separately as a feasibility limitation.

For `aci/D2N002`, local inversion may preserve
`hypothyroidism -> Synthroid -> thyroid labs` or `arthritis -> physical therapy`. Those are
legitimate symbolic delivered-utility successes. They do not prove that the transformed prompt
preserved thyroid/endocrine or treatment-constraint meaning.

The same document also shows why final-output assertions are necessary: placeholder-heavy
generation can omit age, sex, or kidney-transplant history, compress medical history, reorganize
assessment rows, and shift category/status values.

Either measurement family may have `scope: linked` with occurrence IDs or `scope: global` with
an empty occurrence list. A complete-task delivered assertion is therefore
`family: delivered, scope: global`; there is no third global measurement family.

## Minimal artifact

The builder emits one artifact containing accepted assertions, rejection summaries, coverage,
and pins:

```yaml
artifact_version: utility-assertions-v1
artifact_hash: sha256:...
environment_hash: sha256:...
task_pin: {...}
builder_pin: {...}
teacher_pin: {...}
reader_pin: {...}
gate_manifest_hash: sha256:...
documents:
  aci/D2N002:
    measurement_state: measured | partial | unsupported | build_failed
    utility_weight_denominator: <context budget + delivered budget>
    present_family_budgets: [context, delivered]
    missing_family_budgets: []
    assertion_ids: [ast:...]
    controlled_decision_ids: [dec:...]
    uncovered_decision_ids: [dec:...]
assertions:
  ast:...:
    doc_id: aci/D2N002
    family: context | delivered
    scope: linked | global
    subtype: semantic_property | contextual_relation | content | field | exact_relation
    occurrence_ids: [occ:...]
    relation: monitored_by
    expected_values: [...]
    group_id: fact-or-relation:...
    weight: <derived from frozen family/group budgets>
    expected_action_support:
      joint_anchor_action_vector: {dec:...: action:...}
      joint_anchor_hash: sha256:...
      property_level: endocrine-system-disease
    question: ...
    scoring_contract: {...}
    evidence: {...}
    status: accepted
rejections:
  summary_by_reason: {...}
```

Stable assertion IDs hash document, assertion semantics, evidence offsets, scoring contract, and
definition version. Rollout scores are separate:

```yaml
assertion_id: ast:...
rollout_id: ...
status: scored | abstained | failed
score: 0.0
raw_observation: ...
scorer_pin: {...}
```

Infrastructure failure never silently becomes task score zero.

## Validation

Use one bounded validation path with only load-bearing gates:

1. **Identity and evidence:** IDs resolve, evidence matches the adapter's authoritative source,
   and assertion semantics are internally valid.
2. **Leakage:** the question or options do not reveal the answer.
3. **Context support:** the assertion passes `doc_orig`, passes its pinned joint representative
   generalization anchor, and fails the all-placeholder anchor.
4. **Stability:** until a later preregistered change, the single-pass reader gate is used. Any
   repeated-read or option-permutation bound is a deferred future protocol, not a current gate.

Other checks remain adapter-specific diagnostics until a measured failure justifies promoting
them to a gate.

Do not run builder-time model counterfactuals by default. Ranker-time one-decision pairs provide
bounded local contextual evidence on actual rollout states. A small report-only audit sample may
be added only if link disagreement becomes a measured problem.

Use a compact rejection taxonomy:

```text
not_generated
generation_failed
invalid
leakage
unsupported
floor_answerable
unstable
infrastructure_failed
```

Keep detailed evidence internally, but callers and gates depend only on these stable classes.

## Static anti-density weights

The builder does not define a utility aggregation or advantage algorithm. The frozen QA
threshold manifest declares fixed `context` and `delivered` family budgets. Within each family,
divide its budget equally across unique fact/relation `group_id`s, then divide each group budget
equally across its accepted assertions. Context and delivered assertions about the same fact use
separate family-local groups: they do not collapse into one score, and their total mass remains
bounded by their family budgets. Structural-only assertions are diagnostic or consume a capped
portion of the delivered budget.

If a document lacks one family, use a fixed denominator equal to the sum of the reserved context
and delivered family budgets. The absent family contributes zero numerator and its reserved mass
remains absent; surviving components are not renormalized to full strength. Do not fabricate
zero-score assertions. Record the fixed denominator plus present and missing family budgets in
the document artifact. Such a document cannot support a claim about the missing family, but its
decisions remain eligible for ranker-v2 fallback credit.

The weighted component vector and frozen document denominator are the interface to ranker v2.
Ranker-v2's `weighted_mean` uses that denominator rather than the sum of present weights. The
ranker specification still exclusively owns `Q_j`, `Q_global`, `A_link`, `A_global`,
`A_document`, fallback, counterfactual substitution, and loss. QA builder v2 defines only frozen
weight state; it does not define parallel advantages or loss composition.

## Credit routing

For decision `j`, linked assertions are those whose occurrence IDs map to `j`. An assertion
appears once per decision even when several mapped occurrences repeat it. Multi-decision
relations route provisionally to each linked decision.

Ranker v2 maps linked occurrence IDs to decisions and exclusively owns routing, fallback,
advantages, pairwise substitution, and loss. The builder supplies only stable IDs, scope, links,
group IDs, weights, and scores. See
[interactive ranker v2](RL/interactive-ranker-v2.md#contextual-counterfactual-credit) for the
normative one-decision intervention. Its complete reward pin includes both direct `doc_p`
context scores and round-trip `out_final` delivered scores; batching and caching do not weaken
that intervention.

Missing coverage never removes a decision or sets its utility relevance to zero.

## Cache and pins

Use one content-addressed build cache and one document/action-vector rollout cache.

The build key includes authoritative source/reference hashes, frozen environment, task adapter,
teacher/reader identities, prompts, assertion ontology, joint representative anchor vector/hash,
and threshold manifest. Model-relative ceiling diagnostics are pinned only when used. The rollout
key adds the complete action vector, `doc_p`, `out_final`, artifact hash, and scorer pin. It
stores the complete batched score vector.

A pin mismatch invalidates the dependent cache and all downstream gates. Do not preserve separate
public ladder, decision, or schema caches after migration.

## QA threshold manifest and pre-training gate

Before building the artifact, freeze a QA threshold manifest defining:

- the eligibility prefilter, no-eligible-span state, and one-call-per-eligible-document rule;
- joint representative-action selection and deterministic tie handling;
- the deferred reader-stability/permutation policy state and acceptance threshold;
- context/delivered family budgets, group construction, structural cap, and missing-family state;
- build and runtime cost budgets in calls and wall time.

No numeric constants are invented here. The selection protocols are fixed before artifact build,
full RL, and held-out evaluation; train/development preflight may instantiate their values once.

Before lambda selection, ExIt reuse, or RL, require:

- exact environment-hash identity between QA and ranker;
- zero dangling occurrence or decision links;
- zero duplicate assertion IDs with different semantics;
- zero scope/link inconsistencies or linked/global duplication;
- valid pins and no infrastructure failure represented as a score;
- every rewarded context assertion passes its pinned joint representative anchor and records the
  complete action vector/hash and property support;
- measurable variation in both context and delivered utility for any claim requiring both;
- explicit counts of missing families and uncovered decisions;
- reader jitter below the frozen measurement threshold;
- a frozen static weight/group manifest;
- reader and remote calls per rollout within the preregistered cost budget.

Shared environment, routing, fallback, counterfactual scheduler, and loss gates remain owned by
interactive ranker v2 and its diagnostics specification; QA does not duplicate them here.

Report assertion support by corpus, type, family, linked/uncovered decision status, and rejection
reason. Coverage is diagnostic; there is no minimum accepted-probe count. Final attacker results
cannot tune this gate.

The cost report separates base rollout round trips from extra one-decision counterfactual round
trips and reports counterfactual cache-hit rate plus total remote generation, extraction, and
reader calls per training document and epoch. Counterfactual calls remain governed by ranker-v2's
frozen scheduler budget, but QA support reporting must expose when sparse routing makes them the
dominant expected cost.

## Migration plan

1. **Freeze identities.** Build the shared occurrence/decision artifact and prove stable
   many-to-one mappings.
2. **Wrap current scores.** Emit per-assertion records from current exact, ladder, decision, and
   schema measurements; use the legacy scalar only as an equivalence check.
3. **Build the ACI adapter.** Add deterministic structural extraction and eligibility filtering,
   one batched teacher relation request for each eligible document, representative generalization
   anchors, and context assertions scored on `doc_p`.
4. **Switch credit.** Remove probe-count document filtering, activate linked/global/fallback
   routing, and prioritize uncovered decisions for counterfactuals.
5. **Retire legacy surfaces.** Remove independent detection, generated-`out_p` QA, exactly-two
   ladder probes, whole-document decision discovery, free-text dependencies, and separate probe
   artifacts after migration fixtures pass.

## Required tests

- identical pinned inputs produce identical environment, assertion, and artifact IDs;
- frozen family budgets deterministically produce group/assertion weights, keep context and
  delivered groups separate for the same fact, and obey the declared missing-family state;
- missing-family fixtures retain the full reserved denominator, contribute zero absent-family
  numerator, and do not renormalize surviving weights;
- teacher output cannot create or override IDs, gold, relation types, polarity, or aliases;
- every accepted relation resolves to exact source evidence and legal argument types;
- assertions resolve to the adapter's authoritative source, while ceiling evidence is used only
  for feasibility diagnostics;
- context assertions pass `doc_orig`, pass a compatible legal non-KEEP/non-placeholder anchor,
  fail the placeholder anchor, record the joint action vector/hash and property support, and
  score rollout `doc_p`;
- multi-decision context assertions use one joint anchor with linked decisions at their coarsest
  entailing legal actions and unrelated decisions at KEEP;
- delivered assertions score `out_final`, with structural checks unable to satisfy semantic ones;
- a placeholder-restored symbolic relation can pass while its context assertion fails;
- field assertions cannot re-enter through a schema aggregate;
- repeated occurrences map one assertion once to one decision;
- global credit reaches all decisions without duplicating linked assertions;
- uncovered decisions receive complete-document fallback and counterfactual priority;
- tested-pair evidence substitutes for provisional credit and the complete vector is rescored;
- ranker-v2 alone computes weighted means, linked/global/document advantages, fallback, and loss;
- all context assertions for one rollout use one batched reader request;
- deterministic delivered assertions use no reader call and remaining delivered assertions batch;
- each eligible document makes one cached relation-teacher call, while a no-eligible-span document
  records its no-call state;
- parser, reader, or infrastructure failure follows declared semantics rather than implicit zero;
- changing each pin invalidates the correct cache and downstream gate.

## Required preflight spikes

Only three bounded spikes are required before implementation claims:

1. **ACI assertion support:** on a tiny development slice, measure deterministic structural
   extraction, relation-teacher acceptance and diversity diagnostics, representative-generalization support,
   context/delivered score variation, and uncovered decisions.
2. **Reader stability:** measure repeated-read and option-order disagreement on the proposed
   context assertions.
3. **Cost:** report base rollout round trips separately from extra one-decision counterfactual
   round trips, counterfactual cache-hit rate, and total remote generation, extraction, and
   reader calls per training document and epoch, plus wall time under the pinned concurrency
   regime.

Use local or existing cached machinery where possible. Any paid or external model call requires
explicit user approval. These spikes set feasibility thresholds before full RL; they make no
privacy claim.

## Stop conditions

- context assertions do not distinguish truthful generalization from placeholders;
- count pressure pushes actions beyond the supported semantic boundary or creates a utility
  cliff at unsupported lattice levels;
- accepted count rises without context or delivered-utility variation;
- schema/structural scores dominate context scores;
- symbolic restoration is reported as remote semantic retention;
- reader instability is comparable to rollout score differences;
- teacher proposals require open-ended relation types or unsupported world knowledge;
- policy movement concentrates on linked decisions while fallback/counterfactual paths are inert;
- count score improves while held-out realized privacy worsens.

Failures are findings under the pinned design. Do not repair them with per-model calibration,
post-hoc weights, relaxed gates, or selective document removal.

## Sources

- [Interactive ranker v2](RL/interactive-ranker-v2.md)
- [Interactive ranker v2 decision log](RL/interactive-ranker-v2-decision-log.md)
- [Interactive ranker v2 diagnostics](RL/interactive-ranker-v2-diagnostics.md)
- [QA builder v2 decision log](RL/qa-builder-v2-decision-log.md)
- [Training-task environment](RL/training-task-env.md)
- [RL environment issue register](../issues/2026-07-08-rl-env-and-lattice-count-issue-register.md)
- [Detector noise-gate limits](../issues/2026-07-10-detector-junk-and-noise-gate-limits.md)
- `results/qa_pairs_pv4_super.txt` and the 2026-07-12 failure analyses are auxiliary evidence
  about the retired pipeline, not requirements.
