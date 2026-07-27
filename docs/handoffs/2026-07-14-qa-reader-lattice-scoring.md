---
type: handoff
status: current
created: 2026-07-14
updated: 2026-07-27
tags: [qa-builder-v2, context-qa, reader, scorer, lattice-entailment, rl-ranker-v2, handoff]
companion: [docs/specs/qa-builder-v2.md,
            docs/specs/RL/interactive-ranker-v2.md,
            docs/specs/RL/qa-builder-v2-decision-log.md,
            docs/issues/qa-relation-scoring-lexical-not-entailment.md]
---

# Handoff — make QA context scoring lattice-aware

## Focus for the next session

Amend QA-builder v2 so context probes reward the complete utility-safe semantic band:
KEEP and truthful generalizations that preserve the probe's required property receive full
utility; coarser meanings and placeholders do not. Preserve the existing cheap runtime shape:
one free-form reader batch per rollout, deterministic local scoring, and no reader call per
lattice rung.

This handoff records the recommended correction. No implementation has been made for it yet.
The motivating diagnosis and evidence are in
`docs/issues/qa-relation-scoring-lexical-not-entailment.md`; do not re-derive that issue from
scratch.

## Executive recommendation

Split context-answer scoring into two explicit contracts:

1. **Literal answer:** for a relation with one linked decision and one safe uncontrolled
   literal, normally ask for the literal. Keep lexical matching against the exact grounded
   literal. The linked decision's selected `support_property` is the privacy-safe locator in
   the question and the semantic requirement on the rollout.
2. **Linked answer:** when the answer denotes a controlled decision, do not use lexical
   `accepted_values` as reward truth. Resolve the free-form answer to a node in that decision's
   frozen local lattice, then award binary credit iff the resolved node entails the assertion's
   required property.

Do not show lattice levels as closed-set reader options. The reader receives only `doc_p` and a
natural question. Do not run NLI at RL time. Do not change gold answers per rollout. Count values
and count order never determine semantic entailment.

## Terminology that must remain distinct

- **`support_property`** — the minimum semantic property that one linked argument must retain
  for this relation/probe. It defines a semantic boundary for the intervention.
- **`accepted_values`** — current string golds against which every reader answer is lexically
  scored. In the proposed design, only literal-answer contracts retain compiler-derived
  `expected_values`; linked-answer contracts have no accepted string gold.
- **`required_property`** — the linked-answer scoring boundary, copied from the answer target's
  validated `support_property`.
- **Semantic band** — KEEP plus every legal lattice action whose declared semantics entail the
  required property.
- **Representative anchor** — the joint action vector with each linked decision at its coarsest
  legal non-placeholder action that still entails its `support_property`; unrelated decisions
  remain KEEP.

The semantic boundary is not a count floor. It may happen to be the highest-count utility-safe
action when counts increase with coarsening, but semantic sufficiency selects the boundary and
ranker-v2's count reward chooses among utility-equivalent actions inside it.

## Current defect

The teacher currently supplies `accepted_answers`; compilation normalizes and stores them as
`accepted_values` in `src/cloak/qa/builder.py`. Both builder validation and runtime utility
then call:

```python
max(fact_score(reader_answer, value) for value in accepted_values)
```

`fact_score` is lexical. For required property `solid organ transplant`, a reader answer of
`kidney transplant` receives less credit than the exact teacher phrase, although the source
identity is strictly more informative. The result is a lexical bump around one rung rather than
a semantic band.

There is a second missing prerequisite: `freeze_ranker_environment` currently records only
self-entailment. KEEP entails only its source string, and a level action entails only its own
fill. The scorer therefore cannot currently look up that `kidney transplant` entails
`solid organ transplant`.

### What `three_point_gate_failed` means

Every otherwise compiled context assertion is tested on three documents with the same fixed
question and answer contract:

```text
pass = score(original) >= threshold
   and score(representative_generalization) >= threshold
   and score(all_placeholder) < threshold
```

The current threshold is `1.0`. If the placeholder passes, the builder emits the more specific
`placeholder_answerable` rejection. `three_point_gate_failed` is the coarse fallback used when the
placeholder correctly fails but the original, representative, or both fail. It therefore does
not identify which positive point failed, and current validation evidence stores scores but not
the raw reader answers needed to diagnose why.

Measured in the D2N002 r24 artifact:

- all 19 gate-rejected context candidates scored `original = 0` and `placeholder = 0`;
- 8 scored `representative = 1`; 11 also scored `representative = 0`;
- among the 5 contextual-relation candidates, all 5 scored `original = 0`, only 1 scored
  `representative = 1`, and all 5 correctly scored `placeholder = 0`.

The all-zero original side has two different causes that must not be conflated:

1. **Malformed linked-answer scoring:** questions/golds use a generalization such as
   `solid organ transplant`, while the original reader can validly answer `kidney transplant`.
   Lexical `fact_score` gives that finer answer insufficient credit. The lattice-entailment scorer
   in this handoff directly fixes this case.
2. **Actual reader/probe failure:** for a literal-answer question such as "What medication was
   continued for the thyroid gland disease?", the original document says `hypothyroidism`. If the
   reader cannot connect that source term to the question's generalized locator, it may answer
   `NONE` even though the expected literal is `synthroid`. Changing the answer scorer cannot repair
   a missing reader answer; the probe should remain rejected unless question design or a validated
   reader change makes both original and representative pass.

Some representative failures were also exposed to underscore-vs-space lexical mismatch
(`solid_organ_transplant` versus `solid organ transplant`). The latest code normalizes teacher
underscores, but no post-change representative real-data run has yet established how many failures
remain. Do not claim that normalization solved the gate epidemic without that rerun.

## Teacher contract

### Selecting `support_property`

The redesigned teacher must **not choose a separate `accepted_answers` field**. That field is
redundant and invites the teacher to create a second, inconsistent semantic target. The teacher
chooses only:

- one `support_property` for every linked argument;
- one natural question; and
- one `answer_role` (`subject` or `object`) identifying which argument answers the question.

The compiler derives the scoring target from the selected argument:

- context/literal answer target -> `expected_values = [exact grounded literal]`;
- linked answer target -> `required_property = selected argument.support_property` and
  `answer_decision_id = selected argument.decision_id`.

Replace the current instruction to choose the "most specific" listed level with this rule:

> For each linked argument, copy exactly one listed level into `support_property`. Choose the
> coarsest listed level that still preserves the specific relation and keeps the question
> meaningfully answerable. Do not choose a generic detector-type level that could describe almost
> any entity of that type. Do not optimize for count, copy the protected source value, or use a
> placeholder.

Operational reading for the teacher:

1. Start from the broadest listed level.
2. Ask whether that level preserves why these two arguments have this particular relation.
3. If it makes the relation vacuous or loses the needed category/function, move one level more
   specific.
4. Select the first relation-sufficient, non-vacuous level.

Example:

```text
source: kidney transplant
levels: solid organ transplant -> medical condition
relation: medications are contraindicated because of this history
```

Choose `solid organ transplant`. `medical condition` is true but does not preserve the
transplant-specific treatment constraint.

The teacher proposes this semantic boundary. Deterministic code must verify that it is exactly
one legal listed property, reject known generic/root-only properties when the adapter says they
are vacuous for that relation, and retain the original/representative/placeholder reader gate.
Do not add an exhaustive reader search over every rung. If exact boundary minimality remains
uncertain, measure immediate-coarser behavior on a report-only audit sample rather than adding a
production model call per rung.

### Selecting a context `literal`

A `literal` is not a teacher-generated paraphrase. After finding an explicit allowed relation,
the teacher uses a linked S-label for every controlled argument and copies a context `literal`
only for an argument that is not represented by a controlled decision.

The literal-selection instruction is:

> Copy the shortest exact contiguous source span that names the uncontrolled relation argument
> and preserves every modifier needed for this specific relation. Drop unnecessary determiners,
> quantities, and deictic wrappers when the remaining phrase is still an exact source substring.
> Do not paraphrase, normalize to outside terminology, quote a displayed controlled span, or copy
> the surrounding clause. If no concise unambiguous exact span exists, abstain from this probe.

Examples:

| Source wording | Selected literal | Why |
|---|---|---|
| `order some thyroid labs` | `thyroid labs` | minimal answer-shaped exact span |
| `some of those anti-inflammatory medications` | `anti-inflammatory medications` | removes deictic wrapper but remains exact |
| `right knee x-ray showed no fracture` | `right knee x-ray` | keeps the modifier needed to identify the test |
| `possibly refer to physical therapy` | none | hypothetical relation; reject rather than create a probe |
| displayed controlled span `Synthroid` | none | use its S-label and lattice levels, not a literal |

The compiler resolves the literal inside the derived relation evidence anchor, records exact
offsets, verifies the argument role and relation support, and rejects missing, ambiguous,
controlled/protected, or unsupported spans. When `answer_role` selects this argument, the compiler
sets `expected_values` to this same minimal literal; there is no second teacher-authored answer
phrase. When `answer_role` selects the other argument, the literal remains a grounded relation
dependency or question locator but is not an expected answer.

### One linked argument plus one literal

Default to asking for the uncontrolled literal:

> Refer to the linked argument only through its selected `support_property`. Ask for the exact
> uncontrolled literal, and set `answer_role` to that literal argument. Do not emit a separate
> accepted answer; deterministic compilation already has the exact grounded literal.

Example:

```yaml
relation: monitored_by
arguments:
  - kind: linked
    source: hypothyroidism
    support_property: endocrine condition
  - kind: context
    literal: thyroid labs
question: What test monitors the endocrine condition?
answer_role: object
```

This is the simplest robust path: the answer is safe and exact, while answering requires the
reader to connect the retained generalized condition to the literal test. If the literal is
protected, long, ambiguous, or not a usable answer, reject the probe or use the linked-answer
contract below; do not silently accept a weak question.

### Linked answer

For all-linked relations, or the bounded fallback where the question must ask for a linked
argument, the proposal must identify exactly one answer argument by role (`subject` or `object`).
The teacher must not emit decision IDs: deterministic compilation resolves the selected argument
to its frozen occurrence and decision. The question uses only safe generalizations or safe context
literals as locators. Linked-answer contracts contain no accepted string gold; the local lattice
contract is the reward authority.

Example:

```yaml
relation: contraindicated_because_of
question: What history contraindicates those medications?
answer_role: object
scoring_contract:
  kind: lattice_entailment
```

Every other linked argument still contributes a `decision_requirement`; identifying one answer
target does not remove the relation's other dependencies.

Keep the teacher wire contract simple and deterministic:

- `answer_role` selects exactly one of the two proposed arguments;
- when that argument is `context`, the compiler derives the exact grounded literal and optional
  frozen adapter-approved aliases;
- when that argument is `linked`, the compiler derives its decision ID and copies its selected
  `support_property` into `required_property`;
- the teacher response schema no longer contains `accepted_answers`.

## Frozen semantic data

The local lattice profile is an ordered chain from specific to coarse. For example:

```yaml
canonical_surface: kidney transplant
surface_aliases: [renal transplant, kidney transplantation, transplanted kidney]
levels: [solid organ transplant, medical condition]
```

At artifact build time, freeze explicit semantic closure into each decision. Do not recover it
later from mutable profile files, count values, or action ordering.

```yaml
semantic_chain:
  - node: keep
    answer_aliases: [kidney transplant, renal transplant,
                     kidney transplantation, transplanted kidney]
    entailed_properties: [solid organ transplant, medical condition]
  - node: solid-organ-transplant
    answer_aliases: [solid organ transplant]
    entailed_properties: [solid organ transplant, medical condition]
  - node: medical-condition
    answer_aliases: [medical condition]
    entailed_properties: [medical condition]
  - node: placeholder
    answer_aliases: []
    entailed_properties: []
```

For a chain `L[0], ..., L[m]` ordered specific to coarse:

```text
entails(KEEP) = {L[0], ..., L[m]}
entails(L[i]) = {L[i], ..., L[m]}
entails(PLACEHOLDER) = {}
```

Store source aliases only in the local frozen decision environment. Never copy protected source
strings into teacher-visible prompts, remote requests, or assertion `accepted_values`. Pin the
profile/lattice identity transitively in the artifact hash and bump all affected schema, prompt,
builder, scorer, and cache revisions.

The closure is a declaration made by the accepted lattice profile, not independent proof of
natural-language entailment. Preserve provenance and audit model-proposed edges before RL; do not
paper over a bad lattice with a scorer-side NLI call.

## Reader and scorer flow

### Reader

Keep the current free-form reader interface:

```text
inputs:  one transformed document + all context questions for that document
output:  one concise free-form answer per question, or NONE
```

The reader never receives `accepted_values`, `required_property`, action IDs, lattice levels as
options, or the rollout's replacement map. This preserves its role as a context-use measurement
rather than a taxonomy multiple-choice classifier.

### Literal-answer score

For `answer_kind: literal`, retain a bounded lexical contract against exact source-grounded safe
literals:

```text
score_literal(y) = max(fact_score(y, v) for v in expected_values)
```

Do not let the teacher invent broad synonym sets. Additional aliases, if needed, must come from a
frozen adapter ontology. Continue to record the raw answer and match diagnostics.

### Linked-answer score

For `answer_kind: linked_decision`:

1. Select the frozen resolver for `answer_target.decision_id`.
2. Resolve the reader's free-form answer to exactly one local node using that decision's canonical
   labels and aliases. A protected source value or source alias resolves to KEEP locally.
3. Treat `NONE`, a placeholder label, ambiguous resolution, and unresolved text as no semantic
   node. Record the reason; infrastructure failure remains separate from task score zero.
4. Award binary credit iff `required_property` is in the resolved node's frozen
   `entailed_properties`.

```text
node = resolve(answer_target.decision_id, reader_answer)
score = 1 if required_property in entails(node) else 0
```

Do not use token-F1 partial credit for linked answers. Do not use generic NLI as the primary
resolver. If real-data evidence later shows frequent valid unresolved paraphrases, add a pinned,
cached fallback only after measuring its calibration and cost.

### Builder gate and runtime reuse

Use the same scoring contract in both places:

- **Build time:** original must pass, the pinned representative generalization must pass, and the
  all-placeholder anchor must fail.
- **RL runtime:** run one batched reader request for the rollout's `doc_p`, score every returned
  answer locally, and return the full utility component vector to ranker-v2.

Questions and answer contracts remain fixed across rollouts. Ranker-v2 continues to own weighting,
advantages, fallback, counterfactual attribution, and total loss. A one-decision counterfactual
regenerates `doc_p`, runs the complete round trip, and rescores the same full component vector.

Replace the coarse positive-side rejection with diagnostic detail while retaining the gate:

```text
original_not_answerable
representative_not_answerable
original_and_representative_not_answerable
placeholder_answerable
linked_answer_unresolved
```

Store the raw local reader answers, resolved node or literal match, and per-point score in protected
build diagnostics. Redact protected source strings from shareable reports.

## Worked examples

### Span to literal: monitored test

```text
Original:       hypothyroidism ... order thyroid labs
Representative: endocrine condition ... order thyroid labs
Placeholder:    <HEALTH_CONDITION_1> ... order thyroid labs
Question:       What test monitors the endocrine condition?
Expected value: thyroid labs
```

Expected build behavior:

| Context | Reader answer | Score |
|---|---|---:|
| original | `thyroid labs` | 1 |
| representative | `thyroid labs` | 1 |
| placeholder | `NONE` | 0 |

If the placeholder reader still answers `thyroid labs`, reject the probe. It does not require the
generalized context, even though the source relation itself is real.

### Linked answer: transplant constraint

```text
Required property: solid organ transplant
Frozen chain: kidney transplant -> solid organ transplant -> medical condition
Question: What history contraindicates those medications?
```

| Reader answer | Resolved node | Required property entailed? | Score |
|---|---|---:|---:|
| `kidney transplant` | KEEP | yes | 1 |
| `renal transplant` | KEEP alias | yes | 1 |
| `solid organ transplant` | selected level | yes | 1 |
| `medical condition` | coarser level | no | 0 |
| placeholder or `NONE` | none | no | 0 |

This is the desired reward shape: QA is indifferent between KEEP and the truthful supported
generalization; the count objective supplies pressure toward the coarser utility-safe action.

## Recommended artifact contract

Use an explicit tagged answer target rather than overloading `accepted_values`:

```yaml
answer_target:
  kind: literal
  expected_values: [thyroid labs]
scoring_contract:
  kind: literal_fact
```

or:

```yaml
answer_target:
  kind: linked_decision
  decision_id: dec:...
  required_property: solid organ transplant
scoring_contract:
  kind: lattice_entailment
```

During migration, legacy `accepted_values` may be read only as `literal.expected_values` for an
explicit legacy contract. Do not infer a linked semantic boundary from arbitrary legacy strings.
Reject or rebuild ambiguous cached assertions.

## Implementation sequence

1. Amend `docs/specs/qa-builder-v2.md`, the QA-builder decision log, and the issue status so the
   normative docs no longer prescribe one lexical path for all context answers.
2. Add failing unit tests for closure construction, decision-local answer resolution, literal
   scoring, linked scoring, and original/representative/placeholder gates.
3. Bump the relation-teacher prompt/schema, builder, artifact, scorer, and cache pins. Remove
   teacher-authored `accepted_answers`, add explicit `answer_role`, and deterministically compile
   it into the tagged artifact `answer_target` contract.
4. Freeze `semantic_chain`, aliases, and transitive `entailed_properties` from the exact profile
   used to construct each decision. Reject missing, duplicate, cyclic, or action/profile-mismatched
   chains.
5. Revise the teacher instruction: coarsest relation-sufficient non-vacuous `support_property`;
   literal-as-answer by default for linked/literal relations.
6. Split `_answer_score` into literal and linked scorers. Pass the frozen decision environment to
   both builder-time validation and runtime `score_utility` without reading mutable global state.
7. Update reports to show answer kind, required property, resolved node, resolution status, and
   score. Split three-point failures by failed point and retain protected raw answers only in local
   diagnostics. Never print protected source strings in shareable reports.
8. Run focused tests, then the smallest representative real-data E2E before claiming completion.

## Required tests

At minimum, cover:

- chain closure: KEEP entails every level; each level entails itself and coarser levels only;
- source alias resolution to KEEP without copying the source into `accepted_values`;
- same phrase in another decision cannot resolve through the wrong decision;
- ambiguous/unresolved/placeholder/NONE linked answers score zero with diagnostics;
- literal answers retain expected lexical behavior;
- `kidney transplant`, `renal transplant`, and `solid organ transplant` all receive full linked
  credit for required property `solid organ transplant`;
- `medical condition` and placeholder do not;
- a span/literal probe passes original and representative but fails placeholder;
- infrastructure failure is not converted into score zero;
- cache/artifact hashes change when profile order, aliases, closure, prompt, or scorer pin changes;
- ranker-v2 consumes the corrected component score without changing aggregation or counterfactual
  normalization.

Suggested focused verification after implementation:

```bash
PYTHONPATH=src:scripts .venv/bin/pytest -q \
  src/cloak/tests/test_qa_builder_v2.py \
  src/cloak/tests/test_relation_qa_v2.py \
  src/cloak/tests/test_build_qa_utility_artifact_cli.py \
  src/cloak/tests/test_utility_credit.py \
  src/cloak/tests/test_train_roundtrip_mode.py
```

## Real-data validation gate

Synthetic tests are insufficient because this change defines reward semantics. Before claiming
completion, run the smallest real-data build and scoring slice containing both answer kinds.
Use `aci/D2N002` if it still contains the required grounded cases.

Pre-register these falsifiable outcomes:

1. At least one linked/literal relation probe is accepted and produces a usable literal answer.
2. At least one linked-answer relation probe is accepted.
3. On the transplant case, original/KEEP and `solid organ transplant` receive equal full utility;
   `medical condition` and placeholder do not.
4. On the thyroid-labs case, original and `endocrine condition` answer `thyroid labs`; placeholder
   fails. If placeholder still answers it, the probe is rejected rather than counted as success.
5. Report accepted, rejected, unresolved, ambiguous, and infrastructure-failed counts plus raw
   examples for both answer kinds.

The existing debug smoke `scripts/spikes/relation_teacher_v5_debug_smoke.py` calls an external
rate-limited API and therefore requires explicit user approval before use. Prefer a cached replay
or the production builder against existing cached teacher responses first. If a fresh teacher
call is required, ask before running it. Inspect the generated artifact rather than reporting only
test status.

## Important boundaries

- Do not present lattice levels as reader options.
- Do not add source surfaces to `accepted_values`.
- Do not rewrite questions or gold answers per rollout.
- Do not infer entailment from counts, `coarseness_rank`, or lexical similarity.
- Do not add per-rung reader calls to the production builder.
- Do not let missing QA probes imply zero relevance for uncovered ranker decisions.
- Do not modify ranker-v2 weighting, counterfactual loss, or fallback while fixing this scorer.
- Do not claim externally proven entailment for model-proposed lattice edges; preserve provenance
  and report the assumption.

## Current workspace caution

At handoff creation, the worktree already contains unrelated modifications to `AGENTS.md`,
`data/lattice_profiles/lattice_profiles.json`, and
`src/cloak/tests/test_build_arms_artifact_cli.py`, plus an untracked lattice backup. Do not discard,
overwrite, stage, or attribute those changes to this work without inspecting their owner and
purpose.

## Suggested skills

- **superpowers:brainstorming** — only if implementation uncovers a genuine unresolved design fork;
  the main scorer architecture is pinned here.
- **superpowers:writing-plans** — turn the implementation sequence into a test-first plan before
  editing shared QA/reward code.
- **superpowers:test-driven-development** — establish linked/literal scorer behavior before changing
  `_answer_score` or artifact schemas.
- **diagnose / systematic-debugging** — use for real-data reader failures or unexpected
  placeholder answerability; do not patch prompts without localizing the failure stage.
- **experiment-audit** — audit the representative real-data reward result before making a design
  validity claim.
