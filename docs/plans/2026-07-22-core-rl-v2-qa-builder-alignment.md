---
type: reference
status: current
created: 2026-07-22
updated: 2026-07-22
tags: [rl, ranker, qa-builder, utility-reward, credit-routing, implementation-audit]
companion: [docs/specs/qa-builder-v2.md,
            docs/specs/RL/interactive-ranker-v2.md,
            docs/specs/RL/interactive-ranker-v2-decision-log.md,
            docs/specs/RL/interactive-ranker-v2-diagnostics.md]
---

# RL-v2 and QA-builder-v2 alignment audit

## Executive finding

The central RL-v2 architecture remains valid: one policy decision per repeated value, a complete
round-trip utility vector, linked/global/fallback provisional credit, sparse one-decision
counterfactual correction, and a separate exact count objective. The utility contract beneath that
architecture has changed materially, however. The live QA-builder-v2 output is no longer the old
ladder/decision/schema carrier described by `training-task-env.md`; it is a weighted assertion
artifact with two measurement families, explicit routing scope, fixed missing-family semantics,
and subtype-specific scoring on `doc_p` and `out_final`.

Neither existing branch is an implementation base that can train RL-v2 on the production v16
artifact without repair:

- `codex/qa-builder-v2-implementation` can validate and score the current artifact, but reduces its
  component vector back to one document scalar and trains the predecessor's trajectory-level RLOO.
- `codex/qa-builder-v2-clean` contains the stronger structured-credit and fixed-`1/G` loss skeleton,
  but it targets an older QA artifact and reader/compiler contract and rejects v16 before training.

There are also two pre-RL data blockers in the current artifacts: the QA artifact publishes the
wrong policy-decision set, and the embedded action menus do not pass the normative complete-count
gate. These are not optimizer details and must be corrected before lambda calibration or RL.

## Scope and inspected state

This audit cross-checked:

- current QA specification: `docs/specs/qa-builder-v2.md` at the active branch tip;
- RL design and diagnostics: `docs/specs/RL/interactive-ranker-v2.md`, its decision log, and its
  diagnostics specification;
- production QA output: `results/qa_v2_aci_full_v16/aci_full.utility` and sidecars;
- current implementation branch: `codex/qa-builder-v2-implementation` at `131d868` in
  `/home/timo/repos/agent-cloak`;
- earlier reviewed integration branch: `codex/qa-builder-v2-clean` at `15e2e64` in
  `/home/timo/repos/agent-cloak/.worktrees/qa-builder-v2-clean`;
- the shared ACI ranker environment: `results/qa_v2_aci_full/ranker-env.json`.

The v16 artifact contains 67 ACI documents and 1,357 accepted assertions:

| family / subtype | count | routing scope |
|---|---:|---|
| context / `contextual_relation` | 634 | linked |
| delivered / `content` | 606 | linked |
| delivered / `field` | 94 | global |
| delivered / `structure` | 14 | global |
| delivered / `exact_relation` | 9 | linked |

The family budgets are context `0.6` and delivered `0.4`, with structure capped inside the
delivered budget. Sixty-three documents contain both families and have total available weight
`1.0`; four lack context assertions and retain only `0.4`. The missing context mass is not
renormalized.

## The live QA utility contract

### Measurement families and surfaces

The scorer returns one stable score per assertion ID:

```text
component_scores: {assertion_id: score}
```

Context assertions are `contextual_relation` questions scored directly on the rollout's `doc_p`.
They use the frozen v4 relation-constrained reader, per-assertion source-turn excerpts, and either
single linked-decision chain entailment or linked-decision-set recall. Delivered assertions are
scored on `out_final`: reference-backed content retention, exact field agreement, required
structure, and exact plan relations.

This separation is intentional. Context assertions test whether a truthful generalization retains
relation semantics that a placeholder destroys. Delivered assertions test whether the complete
remote-generation and extraction path delivered the required task result. Neither family may be
replaced by the other.

### Builder-owned aggregation state

QA-builder-v2 owns and freezes:

- each assertion's family, subtype, scope, occurrence links, group, weight, and scoring contract;
- the context/delivered family budgets and structural cap;
- each document's assertion IDs, available/missing family budgets, and
  `utility_weight_denominator`;
- reader, compiler, detector, teacher, threshold-manifest, environment, and artifact identities.

RL owns credit assignment and loss construction, but it must aggregate every subset with the
builder's fixed document denominator:

```text
U(S) = sum(q in S) weight[q] * score[q] / utility_weight_denominator
```

This is a weighted sum over a fixed denominator, not a subset-renormalized `weighted_mean`.

### Dependency topology

The 634 context assertions link to 427 of the 701 actually ranker-selectable decisions. Of the
context assertions, 284 link to one decision and 350 link to two or more decisions. The latter are
genuine relation hyperedges: both answer and controlled locator decisions can alter readability.
The current artifact therefore makes provisional many-to-many routing the normal case, not an edge
case.

Across all linked assertion subtypes, 603 of 701 ranker-selectable decisions have at least one
link; 98 have none. The artifact's published `uncovered_decision_ids` currently cannot be trusted,
because it is computed against a larger, incorrect decision set described below.

### Supported-level reward bands

Each context assertion pins a supported property for every linked decision. The answered decision
earns the assertion whenever the reader resolves an answer at that property or a finer chain node.
The hard v16 finer-level check certified all stored answer-side finer levels for kept assertions;
43 failed candidates were rejected and emitted as lattice work items.

This does not certify locator invariance. In the 350 multi-decision assertions, changing a locator
decision can make the frozen question wording easier or harder to bridge to the rendered level.
The QA specification explicitly accepts this as an unresolved measurement risk. RL must route the
assertion to every linked policy decision and use counterfactuals to measure local effects; it must
not interpret `required_property` as a legality mask or as exact causal attribution.

## Stale RL-v2 specification assumptions

### Retired carrier vocabulary

`interactive-ranker-v2.md` still says that existing carrier weights are pinned by
`training-task-env.md`, and describes components as probes, decisions, schema IDs, and globals.
`training-task-env.md` is a historical ladder/decision/schema design with `w_exact`, `w_sem`,
multiple-choice decisions, and schema aggregation. Those are not the live weighted utility
contract. `semantic_property` is also retired from weighted utility.

Required correction:

- make `docs/specs/qa-builder-v2.md` and the frozen utility artifact authoritative for utility
  meaning, weights, denominator, scoring surfaces, and pins;
- define the vector as `{assertion_id: score}` with `family`, `subtype`, and orthogonal `scope`;
- mark `training-task-env.md` historical/superseded for RL-v2 reward assembly rather than importing
  its carrier weights.

### Ambiguous aggregation formula

The RL spec writes linked, global, and document scores as `weighted_mean`. That can be read as
renormalizing by the weight present in each subset. The implemented QA contract explicitly forbids
that. Context-only, global-only, linked-only, and complete-document numerators all divide by the
same document denominator stored in the artifact.

Required correction: replace `weighted_mean` with the exact fixed-denominator equation and state
that missing family mass remains missing in ExIt, provisional advantages, counterfactual
`delta_U`, lambda calibration, and evaluation.

### Dependency-set representation

The RL spec calls the routing object `O_q` or `dependency_sets`; the artifact field is
`occurrence_ids`, with `scope` independently declaring `linked` or `global`. Current context rows
also carry `decision_requirements`, but these express supported semantic properties, not a second
credit-routing source.

Required correction: route from `scope` plus `occurrence_ids`, validate every linked occurrence,
and map only to ranker-selectable decisions. Use `decision_requirements` for scoring/support-band
validation only. A context relation's locator and answer occurrences both belong in the routing
hyperedge.

### Linked components with no policy-controllable dependency

The RL partition assumes that every linked occurrence maps into `D_d`. V16 violates that
assumption for delivered content about forced-placeholder PERSON/CODE decisions. Such assertions
currently have occurrence links but no policy action can be assigned to those decisions. If RL
simply drops non-policy links, these components are neither linked to a trainable decision nor
included in `Q_global`, so their utility signal disappears for covered decisions.

Required correction: first fix the QA artifact's policy-decision set. Then define a residual rule:
a linked assertion whose mapped dependency set contains no ranker-selectable decision participates
in shared global/residual credit while retaining its occurrence links as evidence. Mixed
fixed/policy hyperedges route only to their policy-selectable members.

### Missing-family and coverage semantics

The RL spec correctly preserves complete-document fallback, but it does not state that four v16
documents have a maximum possible utility of `0.4`. ExIt and RLOO remain valid because comparisons
are within document. Cross-document diagnostics and lambda calibration must use the frozen scale
as emitted and must not renormalize partial documents.

Required correction: carry `measurement_state`, `present_family_budgets`, and
`missing_family_budgets` into diagnostics and calibration artifacts. Report context-linked and
any-linked coverage separately: 427/701 decisions have context links, while 603/701 have any
linked assertion.

### Scheduler inputs no longer exist as written

The scheduler prioritizes “low-confidence dependencies,” but v16 stores no scalar dependency
confidence. It stores deterministic occurrence links, relation evidence, producer attribution,
and hyperedge cardinality. Inventing confidence from teacher provenance would add an unvalidated
reward-allocation heuristic.

Required correction: retain the approved priority for no linked components and multi-decision
dependencies, but either remove low-confidence from the normative scheduler or define and validate
a concrete artifact field before using it. Separately report no-context-link and no-any-link
decisions so the scheduler does not mistake placeholder-friendly delivered content for contextual
relation coverage.

### QA-specific failure modes are absent

RL-v2 does not yet include the live reward's two important QA risks:

- locator-level steering in multi-decision context assertions;
- answer-side band certification applies only to kept assertion levels and does not establish
  joint locator-by-answer invariance.

Required correction: add both to reward diagnostics. Counterfactual pair reports should be split
by answer decision versus locator decision and by literal-locator versus controlled-locator
assertions. These are diagnostic slices, not extra reward weights.

### Reward identity is underspecified

The utility artifact pins the builder and reader state, but RL runtime additionally depends on the
remote task model, task prompt, extractor, reader execution mode, and complete-vector cache. Those
must form one transitive reward pin. The current v16 artifact does not expose the older clean
branch's `scorer_pin`, and the active trainer does not maintain a complete component-vector cache.

Required correction: define one combined RL reward identity over artifact hash, environment hash,
remote task pin, extractor pin, runtime scorer implementation pin, reader pin, and execution/cache
contract. Counterfactual and ordinary rollout cache keys must use the same identity.

## Production artifact blockers

### Published controlled decisions are not policy decisions

The ranker-v2 environment correctly publishes `policy_decision_ids` and marks each decision with
`ranker_selectable`. There are 701 selectable decisions across the 67 ACI documents:

| runtime type | ranker-selectable decisions |
|---|---:|
| health-condition | 354 |
| drug | 219 |
| medical-procedure | 111 |
| LOC | 17 |

The QA packager instead builds `controlled_decision_ids` from `controlled`. In this environment,
`controlled` means locally rewritten and includes forced-placeholder PERSON/CODE values. V16
therefore publishes 909 “controlled” decisions: the 701 policy decisions plus 124 PERSON and 84
CODE decisions with exactly one forced-placeholder action.

Consequences:

- structured trainer alignment expects policy log-probabilities for 208 nonexistent choices;
- 102 forced decisions are incorrectly reported as uncovered;
- 106 forced decisions receive linked delivered-content assertions even though no policy can act
  on them;
- count averaging over the published list would dilute lambda by fixed actions and violate the
  RL-v2 definition of `D_d`.

Fix before RL: package `policy_decision_ids` from `ranker_selectable == true`; publish fixed
rewrite decisions separately; compute uncovered policy decisions against the selectable set; and
define residual routing for linked assertions whose dependencies are all fixed. This is a local
repackaging/schema migration and should not require regenerating teacher proposals or reader
validation, but it changes the artifact hash and invalidates downstream caches.

### Count state does not pass the RL gate

All 2,262 selectable level actions carry a finite `aset >= 1`, but the embedded action rows carry
no profile identifier, grounding record, source family, or count provenance. The scalar alone is
insufficient for the complete-count gate or for constructing versioned type references.

Using action order as the authored specific-to-coarse ladder, the v16 menus contain 76 adjacent
count decreases across 76 decisions: 54 health-condition, 16 drug, and 6 LOC. Examples include
`heart disease: 400 -> thoracic disease: 390`, `therapeutic agent: 100000 -> pharmaceutical
compound: 17000`, and a mismatched Vermont location ladder. Fourteen additional decisions have a
flat level-count menu.

The normative RL-v2 count gate must therefore fail on this snapshot. Do not sort, clip, repair, or
drop these actions inside the reward. The upstream profile/count artifacts must be corrected and
the ranker environment rebuilt with explicit count provenance. The QA utility assertions may be
reusable only after their environment/action IDs and support vectors are proven unchanged; in
practice an action-menu change should be treated as invalidating the utility artifact unless a
content-addressed migration proves exact compatibility.

## Implementation audit

### Active QA implementation branch

`codex/qa-builder-v2-implementation` contains the current scorer and can intrinsically validate the
v16 artifact. `roundtrip_batch` calls `score_utility` and returns both the complete
`component_scores` vector and its builder-weighted document scalar.

The trainer then discards the vector. `train_roundtrip` computes one scalar RLOO advantage from
`result["recall"]` and multiplies it by the sum of every action log-probability. Consequently:

- linked/global/fallback routing is not active;
- the existing `utility_credit.py` helper is not imported by the trainer;
- every action receives the same document advantage;
- QA coverage links have no effect on credit assignment.

The branch also retains the predecessor's floor-conditioned action derivation. Its CLI expects
`k_floors`, per-document `spans`, and legacy probes. The actual ACI
`ranker-v2-environment-v1` exposes `frozen_environment`, `decisions`, `occurrences`, and
`policy_decision_ids`; it has no `k_floors` or legacy `spans`. The current trainer therefore cannot
directly consume the environment that produced v16.

Its counterfactual path is also the predecessor design: independently sample level actions, flip
only to placeholder, compute a scalar reward difference, and apply a second optimizer update after
the provisional update. That violates the approved adjacent-plus-endpoint scheduler, full-menu
pair probability, and in-place loss substitution.

### Reviewed clean integration branch

`codex/qa-builder-v2-clean` adds useful partial RL-v2 work:

- fixed-denominator document utility;
- per-decision linked/global/fallback LOO advantages;
- a structured utility loss with counterfactual terms substituted at the same `1/G` weight;
- stable decision-key binding;
- complete round-trip result caching and stronger reward-pin checks.

It is not compatible with v16:

- the gate expects old `scorer_pin` and `threshold_manifest_pin` fields absent from v16 and exits
  first with `utility artifact is missing scorer_pin`;
- it expects assertion-compiler v5 and context-reader v1, while v16 is compiler v15 and reader v4
  with scorer schema v3;
- it converts the full occurrence map to strings and rejects the legitimate null decision entries
  for uncontrolled detections as `occurrence map names uncontrolled decisions: ['None']`;
- it requires sampled policy spans to equal the artifact's incorrect 909-entry
  `controlled_decision_ids` set;
- it still lacks lambda conditioning, exact count shaping, lambda-menu selection, and the approved
  counterfactual scheduler.

The clean branch should be mined for the structured-credit, fixed-`1/G`, cache, and validation
ideas, not merged wholesale over the current QA implementation.

### Runtime reader-cost mismatch

The QA decision log says runtime context scoring uses one batched reader request per rollout. The
live `score_utility` loops over context assertions and calls the reader once per assertion, and
`BatchedContextReader` explicitly performs one model request per question. V16 contains 634
context assertions over 63 documents, approximately ten reader generations per context-covered
rollout before counterfactuals.

This does not change score semantics, but it invalidates the planned cost model and can dominate
RL wall time. The implementation plan must choose and measure a truthful execution contract:
batch/fan out per-assertion prompts while preserving each assertion's excerpt and reader clause, or
preregister the actual per-question call budget. Calling the current loop “one batched request” is
incorrect.

## RL-v2 design that remains current

The following decisions survive the QA-builder changes and should not be reopened merely because
the artifact schema changed:

- one frozen detection and occurrence/decision environment shared by QA and RL;
- one policy action per ranker-selectable repeated value, applied to every mapped occurrence;
- utility and count use separate credit paths;
- linked occurrence sets are provisional routing hints, not causal labels;
- decisions without links receive complete-document fallback rather than zero utility;
- global/residual components remain active without double-counting linked components;
- one-decision counterfactuals run the complete remote, extraction, context, and delivered scoring
  path and replace provisional credit in place;
- ExIt is utility-only initialization and accepts passenger actions only as a warm start;
- lambda is fixed per document group and conditions one deployed policy;
- count shaping is experimental and realized privacy remains an external attacker outcome.

## Required alignment before the remainder plan

Before producing the implementation plan for the remaining RL-v2 work:

1. Correct and locally repackage the QA utility artifact around
   `policy_decision_ids`/`ranker_selectable`, fixed rewrite decisions, and residual routing.
2. Repair and rebuild count/action artifacts until the complete-count gate can pass with explicit
   provenance and monotone authored ladders; determine whether that invalidates v16 assertion
   support vectors.
3. Amend the RL-v2 spec to the assertion-vector, two-family, fixed-denominator contract and retire
   the old carrier dependency.
4. Choose one implementation base: current QA/scorer code plus selectively ported structured
   credit/cache pieces from the clean branch.
5. Resolve and benchmark the actual reader execution contract before estimating RL and
   counterfactual cost.

These are prerequisites for the later implementation plan, not optional cleanup. Lambda menu
selection, conditional optimization, and counterfactual scheduling should not be implemented
against the current inconsistent decision and count state.

## Amendments (2026-07-22, post-review — Timo + Fable)

The audit above describes the state at audit time. The following supersede or sharpen its
recommendations after Timo's review:

1. **Branch state:** `codex/qa-builder-v2-clean` was merged into `main` at `ca7f990` with a
   main-wins resolution on every evolved QA surface (builder, scorer, artifact CLI, trainer,
   specs, tests). Only the additive `extract.py` `invert_implementation_pin()` seam and its
   tests were taken live; the clean branch's structured-credit trainer, reward cache, and
   stronger gate remain reference code retrievable via
   `git show codex/qa-builder-v2-clean:<path>`. The "selective port, no wholesale merge"
   recommendation is thereby discharged: porting happens against the merged history.
2. **Count repair must not re-run detection.** Item 2 of the prerequisite list is amended: the
   environment is rebuilt by a deterministic migration over the existing frozen arms artifact
   (`results/qa_v2_aci_full/arms.json`), re-deriving only action menus/counts/provenance from
   the repaired profiles. Fresh detection is nondeterministic across processes and would force
   a semantic-change classification and a paid full QA rebuild.
3. **Profile repairs require Timo's explicit confirmation before promotion** (standing project
   rule), with each proposed edit classified count-only versus order/fill-changing and the
   latter's QA-rebuild cost stated.
4. **Scheduler:** the "low-confidence dependencies" input is removed from the normative
   scheduler rather than defined — matching this audit's own required correction.
5. **Scope:** the remainder plan executes Tasks 1–13 (through the real-data vertical slice)
   under the current green light; benchmark integration and matched-privacy evaluation are a
   separately approved phase.
