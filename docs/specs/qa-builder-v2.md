---
type: reference
status: current
created: 2026-07-12
updated: 2026-07-12
tags: [qa, reward-design, utility-components, context-preservation, credit-routing,
       interactive-ranker, spec]
companion: [docs/specs/RL/interactive-ranker-v2.md,
            docs/specs/RL/interactive-ranker-v2-decision-log.md,
            docs/specs/RL/interactive-ranker-v2-diagnostics.md,
            docs/specs/RL/qa-builder-v2-decision-log.md,
            docs/specs/RL/training-task-env.md]
---

# QA builder v2 — context-preservation utility for interactive ranker v2

**Status: normative design, not implemented.**

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
4. **Teacher proposals are not gold.** Code owns IDs, canonical values, relation vocabulary,
   accepted answers, channels, and validation.
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

### Optional relation escalation

A pinned teacher is optional, not the backbone. Start with deterministic candidates from the
frozen detector, lattice, task schema, and templates. If those cannot produce enough high-value
contextual relations for a document under the preregistered support rule, make at most one
batched, cached relation-proposal call for that document. Otherwise make no teacher call.

The teacher receives `doc_orig`, authoritative reference evidence when available, and the
numbered occurrence inventory. It selects from the ACI adapter's closed relation vocabulary and
quotes exact evidence. Abstain rather than retrying to chase coverage.

The optional relation teacher is pinned to
`nvidia/nemotron-3-super-120b-a12b:free` through
`https://openrouter.ai/api/v1`, authenticated by `OPENROUTER_API_KEY`. Changing model, provider,
prompt, or response schema invalidates the build cache and artifact pin.

The initial ACI vocabulary is deliberately small:

```text
treated_with
monitored_by
contraindicated_because_of
causes_or_explains
referred_to
has_status
has_category
```

The teacher cannot create entities, IDs, gold values, or relation types.

### Deterministic compilation

A relation is accepted only when:

- every argument ID exists;
- the relation permits the argument runtime types;
- its evidence quote resolves exactly in `doc_orig`;
- the evidence directly connects the arguments;
- polarity is consistent;
- no source contradiction exists.

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

Semantic accepted answers come from the frozen lattice or adapter ontology. Contextual gold and
dependencies come from accepted compiled relations. Do not generate one question per lattice
rung and do not force exactly two questions per span. Generate one measurement only when an
accepted task-relevant assertion exists.

Question generation is template-first. A teacher is used only when no safe template exists. It
receives the accepted assertion, `doc_orig`, frozen occurrence inventory, protected aliases, and
expected answer type. It writes wording and evidence quotes only. Questions are static and may
not contain protected-value locators; they are not rewritten per rollout.

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
threshold_manifest:
  family_budgets: {context: 0.6, delivered: 0.4}
  reader_threshold: 1.0
  reader_stability_repetitions: 1
  reader_option_permutations: 1
  reader_stability_threshold: 1.0
family_budgets: {context: 0.6, delivered: 0.4}
documents:
  aci/D2N002:
    measurement_state: measured | partial | unsupported | build_failed
    utility_weight_denominator: <context budget + delivered budget>
    present_family_budgets: [context, delivered]
    missing_family_budgets: []
    weight_groups:
      context:
        "fact-or-relation:...":
          assertion_ids: [ast:...]
          weight: <derived family/group budget>
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
4. **Stability:** repeated reads and option permutations stay within the preregistered bound.

Before ranker training, the artifact gate recomputes the accepted context verdict from every
persisted original/representative/placeholder trial under the frozen reader thresholds. It also
recomputes each joint anchor against the frozen legal action menus: the vector must cover every
controlled decision, linked decisions must use an entailing non-KEEP/non-placeholder action, and
unrelated decisions must use KEEP. The gate independently checks the canonical vector hash.

Family weights remain fixed even when a family is absent. The gate requires the document
denominator to equal the frozen family-budget sum, present/missing families to partition the
manifest families, accepted weights to exhaust each present family budget (and no missing budget),
and emitted group allocations to agree with the derived family/group split.

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

- the deterministic rule for under-supported documents and optional teacher escalation;
- joint representative-action selection and deterministic tie handling;
- reader stability repetitions, permutations, and acceptance threshold;
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
3. **Build the ACI adapter.** Add deterministic candidates and templates first, one optional
   batched teacher escalation per under-supported document, representative generalization
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
- optional teacher escalation makes at most one cached call per under-supported document;
- parser, reader, or infrastructure failure follows declared semantics rather than implicit zero;
- changing each pin invalidates the correct cache and downstream gate.

## Required preflight spikes

Only three bounded spikes are required before implementation claims:

1. **ACI assertion support:** on a tiny development slice, measure deterministic extraction,
   optional teacher escalation rate, representative-generalization support,
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
