---
type: dev-log
status: current
created: 2026-07-22
updated: 2026-07-22
tags: [rl, ranker, qa-builder, implementation-inventory, structured-credit, roundtrip]
companion: [docs/specs/RL/interactive-ranker-v2.md,
            docs/specs/qa-builder-v2.md,
            docs/plans/2026-07-22-core-rl-v2-qa-builder-alignment.md]
---

# RL-v2 implementation inventory

> **Post-merge note (2026-07-22, later the same day):** `codex/qa-builder-v2-clean` has since
> been merged into `main` at `ca7f990` (main-wins on all evolved QA surfaces; only the additive
> `extract.py` implementation-pin seam taken live). "Divergent" below therefore now means
> "present in merged history as reference code, not active on main."

## Purpose and status vocabulary

This log records what exists in code for interactive ranker v2 as of 2026-07-22. It separates
four states that earlier status reports conflated:

- **Active** — present on `main` and compatible with the current production QA-builder-v2 code or
  artifact format.
- **Divergent** — implemented and tested on `codex/qa-builder-v2-clean`, but not merged and not
  compatible with the production v16 artifact without migration.
- **Predecessor** — working round-trip ranker machinery that predates the interactive RL-v2 spec
  and is potentially reusable, but does not implement the final objective or interfaces.
- **Absent** — required by `interactive-ranker-v2.md` and not implemented in either branch.

“Implemented” in this log means executable code with focused tests. It does not imply a successful
RL-v2 training run. No lambda-conditioned ranker has been trained, and no current v16 utility
artifact has completed an end-to-end policy update through structured credit.

## Repository snapshot

| role | branch / worktree | tip | status |
|---|---|---|---|
| production QA and current repository | `main` in `/home/timo/repos/agent-cloak` | `131d868` | active |
| compatibility branch name at same tip | `codex/qa-builder-v2-implementation` | `131d868` | same commit as `main` |
| reviewed earlier integration | `codex/qa-builder-v2-clean` in `.worktrees/qa-builder-v2-clean` | `15e2e64` | divergent |

The clean branch and main diverged from `96b7bfd`. The clean branch added 27 commits and about
9,610 inserted lines around the first QA-builder-v2 implementation and trainer integration. Main
subsequently evolved the QA compiler, artifact, detector, reader, relation generation, scorer,
and full-corpus evidence through a separate history. Neither branch is a strict superset of the
other.

## Executive implementation matrix

| RL-v2 area | active `main` | divergent clean branch | final status |
|---|---|---|---|
| Frozen occurrence/decision environment | implemented | older freezer implemented | active, with packaging defect |
| Policy-free ranker-v2 environment export | implemented | not the current exporter | active |
| QA assertion compiler and full ACI artifact | implemented and validated on 67 docs | early one-doc/no-call smoke | active |
| Per-rollout utility component vector | implemented | implemented | active |
| Fixed builder weights and denominator | implemented | implemented | active |
| Linked/global/fallback credit calculation | helper only, not trainer-wired | implemented and trainer-wired | divergent partial |
| Structured `1/G` utility loss | absent | implemented | divergent partial |
| In-place tested-pair substitution hook | absent | injectable hook implemented | divergent hook only |
| Complete rollout-result cache | absent | implemented | divergent |
| Behavior cloning | predecessor implementation | predecessor implementation | reusable, not final v2 interface |
| Utility-only ExIt | predecessor implementation | artifact-cache-aware variant | reusable, not lambda-replicated |
| Scalar RLOO and tie filtering | predecessor implementation | legacy fallback retained | reusable only |
| One-decision counterfactual execution | legacy placeholder flip | deliberately disabled for artifact mode | final design absent |
| Lambda-conditioned policy | absent | absent | absent |
| Exact type-normalized count objective | absent | absent | absent |
| Complete-count gate and type references | absent | absent | absent |
| Lambda-menu calibration and replay gate | absent | absent | absent |
| Balanced lambda training schedule | absent | absent | absent |
| Counterfactual scheduler | absent | absent | absent |
| Full hybrid loss | absent | absent | absent |
| Per-lambda evaluation and attacker frontier | absent | absent | absent |

## Active implementation on main

### Frozen environment and stable identities

The active environment path implements the shared QA/RL identity foundation:

- `scripts/build_arms_artifact.py` runs detection once and embeds a `v2_frozen_input` per
  document.
- `freeze_ranker_environment` in `src/cloak/train/qa_builder.py` freezes occurrences, repeated-
  value decisions, offsets, detector provenance, action menus, action IDs, decision IDs,
  occurrence-to-decision links, and per-document hashes.
- Repeated occurrences of one canonical value map to one decision. Missing KEEP is synthesized as
  a source-identity action where the runtime type permits it. Forced PERSON/CODE placeholder
  decisions remain represented but are marked `ranker_selectable: false`.
- `scripts/build_ranker_env.py` exports `ranker-v2-environment-v1` with the complete frozen
  environment and per-document `policy_decision_ids`. It deliberately omits legacy policy
  parameters, probe splits, and behavior-clone labels from the default v2 output. The old
  `k_floors`/`spans` format is available only behind `--legacy-v1`.

The policy-free exporter is tested in `test_build_ranker_env_v2.py`, including the absence of
legacy tau, floor, probe, and behavior-clone state.

**Known defect:** QA artifact packaging currently derives `controlled_decision_ids` from the
broader `controlled` flag rather than `ranker_selectable`/`policy_decision_ids`. This publishes
909 decisions instead of the 701 policy decisions in v16. The environment itself contains the
correct distinction; the utility package does not.

### Production QA utility artifact

The active QA-builder-v2 implementation is a deep artifact compiler and runtime scorer in
`src/cloak/train/qa_builder.py`, driven by `scripts/build_qa_utility_artifact.py`. Implemented
surfaces include:

- task-adapter-owned deterministic delivered assertions;
- source-grounded contextual relation proposals and compilation;
- one frozen artifact containing documents, occurrences, decisions, assertions, rejections,
  relation-generation traces, opportunity ledgers, coverage, escalation accounting, QA audit,
  and transitive pins;
- deterministic derived assertion and QA-pair views from the normative artifact;
- content-addressed artifact identity and per-document environment identity;
- explicit `measured`, `partial`, `unsupported`, and build-failure states;
- fixed context/delivered family budgets and fixed missing-family denominator;
- explicit linked/global routing scope and occurrence links;
- stable assertion, group, occurrence, decision, action, and environment IDs.

The current ACI task adapter emits and scores:

- context `contextual_relation` assertions on `doc_p`;
- delivered `content` assertions on `out_final`;
- delivered `field` assertions for demographic and assessment fields;
- delivered `structure` assertions with a capped structural share;
- delivered `exact_relation` assertions over plan condition/treatment/test rows.

The compiler and build pipeline additionally implement:

- the closed five-relation clinical inventory;
- typed linked arguments and exact context literals;
- forward, reverse, literal-locator, compound-locator, and set-valued question shapes;
- source-clause grounding, argument resolution, polarity checks, protected-locator checks, and
  answer-leak checks;
- one primary relation-teacher pass, deterministic relation generation, reverse framing,
  relation-support escalation, optional literal prefiltering, and repair/gleaning accounting;
- treating/contraindication conflict checks and causal/contraindication quote-grounded defenses;
- original/representative/placeholder three-point validation;
- supported answer-property bands and hard/soft finer-level readability checks;
- relation-constrained reader clauses persisted from gate time through runtime scoring;
- rejection ledgers, reader-outcome routing, lattice worklists, and QA audit sidecars.

This is the implementation referred to when QA-builder-v2 was reported complete. It is the reward
measurement environment for RL-v2, not the RL optimizer.

### Real production artifact evidence

`results/qa_v2_aci_full_v16/aci_full.utility` is a completed real-data build over all 67 ACI
documents. It contains:

- 1,357 accepted assertions;
- 634 context relations;
- 606 delivered content assertions;
- 94 delivered field assertions;
- 14 structure assertions;
- 9 exact delivered relations;
- 63 documents with both context and delivered families;
- 4 partial documents with delivered-only weight `0.4`;
- 2,977 recorded rejections and 43 hard finer-level failures in the emitted worklist.

The artifact demonstrates that the builder and scorer surfaces produce substantive real outputs.
It does not demonstrate that the current ranker learns from them. The alignment audit records the
policy-decision packaging and count-monotonicity blockers discovered in this artifact.

### Runtime component scoring

`score_utility` returns:

```text
{
  component_scores: {assertion_id: score},
  utility: sum(weight * score) / utility_weight_denominator
}
```

Context assertions are scored on the rollout's `doc_p`; delivered contracts are scored on
`out_final`. The scorer validates the reader pin before model use, uses frozen reader clauses and
source-turn excerpts, supports single and set-valued answers, parses the ACI output once for
delivered checks, and preserves the builder's fixed denominator.

`roundtrip_batch` detects an attached utility artifact, runs remote generation and local
un-perturbation, calls `score_utility`, and returns both `component_scores` and the scalar
`utility` as `recall`. This is the active bridge from the complete round trip to RL-compatible
measurement data.

**Performance reality:** the active reader issues one generation per context assertion, and
`score_utility` invokes it once per assertion because each row has its own excerpt and clause.
Despite the legacy “batched” name, this is not one model request per rollout.

### Active artifact gate

`enforce_utility_artifact_gate` in `scripts/train_ranker.py` validates before attachment:

- artifact version and required pins;
- recomputed artifact hash;
- environment or per-document environment hashes;
- positive document denominators;
- document-to-assertion ownership and complete assertion assignment;
- stable assertion IDs;
- linked/global scope consistency;
- occurrence existence and no dangling decision link;
- accepted context rows;
- joint representative action-vector/hash shape;
- supported-property shape;
- three-point validation and stability evidence.

The active gate intrinsically accepts the v16 artifact. It does not detect the distinction between
`controlled` and `ranker_selectable`, does not run the complete-count gate, and does not validate
the full runtime reward identity or complete-vector cache because those pieces are absent.

### Active but unwired utility-credit helper

`src/cloak/train/utility_credit.py` implements a model-free provisional credit calculation:

- fixed-denominator linked component scores;
- fixed-denominator global component scores;
- fixed-denominator complete-document fallback;
- leave-one-out advantages without standard-deviation normalization;
- repeated-occurrence deduplication at the decision/component set;
- linked-plus-global credit for covered decisions;
- document fallback for uncovered decisions;
- no duplicate complete-document term for covered decisions.

Its synthetic tests pass. The active trainer does not import or call it. It also reads the
artifact's incorrect `controlled_decision_ids` and stringifies null occurrence mappings, so it
requires schema hardening before production use.

### Predecessor trainer machinery still active

The current `scripts/train_ranker.py` retains several tested mechanisms that can be adapted:

- deterministic sequential action sampling in first-occurrence order;
- dynamic injectivity masking of level fills already claimed by earlier decisions;
- repeated-surface assembly into one `doc_p` and replacement map;
- feature-MLP and frozen-ModernBERT context policy variants;
- behavior cloning from the old floor-walk teacher;
- utility-only ExIt sampling, strict improvement over the reference, serial refreshed
  reverification, and winner cloning;
- group RLOO without standard-deviation normalization;
- group tie filtering;
- entropy over the dynamic masks actually sampled;
- optional KL to a frozen reference;
- round-trip generation, extraction, reader scoring, and greedy readout.

These mechanisms were developed for the predecessor floor/scalar reward design. Their current
interfaces still use legacy span rows, `p6`, `aset`, active floors, and unconditioned policy calls.
They are implementation material, not evidence that interactive RL-v2 is complete.

### Active scalar utility training behavior

When a QA utility artifact is attached, active `train_roundtrip` reads only `result["recall"]`,
computes one document RLOO advantage, and multiplies it by the sum of every selected action's
log-probability. It discards `component_scores`. The test
`test_train_roundtrip_uses_scalar_utility_when_artifact_components_vary` explicitly verifies this
current behavior.

Therefore the active policy receives complete-document scalar credit exactly as in the
predecessor. Linked/global/fallback routing is not active even though all required component data
is returned by `roundtrip_batch`.

### Active legacy counterfactual path

`counterfactual_terms` provides a predecessor counterfactual mechanism:

- choose a random fraction of selected level actions;
- change each selected action only to placeholder;
- regenerate the changed `doc_p` and run the round trip;
- compute scalar `base_utility - counterfactual_utility`;
- apply a separate policy-gradient update using the selected action log-probability.

It is tested for basic sign/routing behavior and exclusion of span-free decisions. It does not
implement adjacent finer/coarser alternatives, endpoint balancing, pair-restricted probability,
fixed-budget scheduling, full-vector diagnostics, or in-place substitution. It must not be
renamed or reused as the final RL-v2 counterfactual implementation without redesign.

## Divergent implementation on qa-builder-v2-clean

### Structured utility credit integrated into training

The clean branch implements the missing trainer seam against its older artifact contract:

- `document_utility` aggregates one component vector with frozen weights and denominator;
- `provisional_advantages` returns one advantage per rollout and stable decision ID;
- linked assertions route to every mapped decision exactly once;
- global assertions broadcast to every policy decision;
- uncovered decisions receive complete-document fallback;
- tied components produce zero advantage;
- covered decisions do not receive a duplicate document advantage.

`train_roundtrip` preserves per-decision log-probabilities, obtains component vectors from each
rollout, calls `provisional_advantages`, and invokes `structured_utility_loss`. A focused test
shows that the policy updates when scalar document utility is tied but per-decision component
credit differs. This is the strongest implemented evidence that structured routing can drive the
trainer mechanically.

### Stable decision binding

The clean branch adds decision-key binding by `(runtime_type, canonical_key)` instead of relying
on artifact order or span position. It validates uniqueness, exact decision-set coverage, and
gradient attachment by stable decision ID. This is directionally correct and should be ported to
the current environment's explicit `policy_decision_ids`.

It currently requires sampled spans to equal the obsolete artifact
`controlled_decision_ids`. On v16 that means 909 entries, including 208 fixed decisions, and so
the binding fails before optimization.

### Structured loss and counterfactual substitution hook

`structured_utility_loss` builds exactly one term for every rollout-decision pair:

- provisional `-A * log pi(a)` when no tested-pair term exists;
- caller-provided counterfactual loss when a tested pair exists;
- one shared division by rollout-group size `G`.

Tests verify that a counterfactual term replaces rather than supplements the provisional term and
that gradients attach by decision ID. The actual RL-v2 pairwise loss, alternative selector,
scheduler, and artifact-mode counterfactual executor were not implemented. Nonzero artifact
`--cf-frac` intentionally exits with an explicit unsupported message.

### Complete utility-result cache

The clean branch implements an append-only content-addressed `UtilityRewardCache` for complete
round-trip results. Its identity binds:

- document and full action vector;
- `doc_p`;
- utility artifact and assertion binding;
- assertion weights and denominator;
- runtime scorer identity;
- task prompt;
- remote generation pin;
- extractor and transitive semantic-model pins;
- complete reward version.

It stores `out_p`, `out_final`, component scores, and scalar utility together; validates score
types and result hashes; rejects corruption, truncation, conflicting duplicate identities, and
prior reward versions; coalesces duplicate misses; appends only new records; and reuses cached
results in training, ExIt candidate evaluation, and greedy readout. Serial ExIt reverification
deliberately bypasses the cache.

This cache is not present on main. Its pin schema targets the old QA reader/compiler and must be
rebuilt around the current v16 scorer and final RL reward identity rather than copied verbatim.

### Stronger gate and preflight

The clean branch gate adds checks absent from main:

- explicit expected threshold-manifest hash;
- exact live reader, builder, teacher, scorer, task, extractor, and source/reference identities;
- document coverage and measurement states;
- exact controlled-decision keys and order-independent binding;
- family budgets, weight groups, cost budgets, and wall-time diagnostics;
- recomputed uncovered decisions;
- supported representative action legality and coarsest support;
- delivered scoring-contract validity;
- complete reward-cache identity.

`qa_utility_preflight_report` reports assertion/decision coverage and planned base versus
counterfactual call surfaces without launching model calls.

The stronger gate rejects the current v16 artifact immediately because v16 no longer carries the
clean branch's `scorer_pin` and `threshold_manifest_pin`, and its compiler/reader versions differ.
Even if those pin checks were migrated, the full occurrence map contains null decision values that
the clean utility router stringifies to `"None"` and rejects.

### Clean-branch smoke evidence

The recorded 2026-07-13 no-call smoke built one ACI D2N002 artifact in 0.95 seconds with:

- 13 delivered assertions;
- zero context assertions;
- zero external calls;
- no optimizer updates;
- no runtime scoring or reward-cache exercise;
- old compiler v5 and old reader/scorer pins.

It validates old artifact plumbing only. It is not a current QA result and not an RL training
result. The current 67-document v16 artifact supersedes it as QA-builder evidence, while the clean
tests remain useful evidence for the structured-credit and cache mechanics.

## Normative RL-v2 coverage by section

### Episode and policy

**Implemented predecessor pieces:** sequential walk, dynamic injectivity mask, consistent repeated-
surface action application, MLP and frozen-encoder policies.

**Not implemented:** the deployed `rank(document, frozen_occurrences_and_decisions,
lambda_profile)` interface; direct consumption of `ranker-v2-environment-v1`; action state over all
mapped occurrence contexts and previous decisions; replacement of floor terminology.

### Lambda conditioning

No implementation exists. Neither policy accepts `lambda_profile`; neither has magnitude plus
profile identity, FiLM/cross-features, identity initialization, responsiveness tests, or profile-
conditioned deployment.

### Count shaping

Only predecessor data plumbing exists: action rows carry legacy `aset` values and the policy uses
`log10_aset` plus an active-floor feature. There is no versioned count artifact, complete-count
gate, type reference calculation, normalized `p_j(a)`, dense expected count loss, count
provenance report, clipping report, or collision opportunity accounting.

The v16 action menus fail the future count gate as currently stored: 76 selectable decisions have
an adjacent count decrease, and action rows do not carry the required provenance.

### Utility components and provisional credit

**Active:** full assertion vector, weights, fixed denominator, linked/global scope, occurrence
links, runtime scoring, synthetic credit helper.

**Divergent:** trainer-wired linked/global/fallback advantages and per-decision log-probabilities.

**Not production-ready:** correct policy-decision filtering, residual routing for assertions linked
only to fixed decisions, and current-artifact integration.

### Contextual counterfactuals

**Predecessor only:** scalar level-to-placeholder interventions with separate updates.

**Divergent hook:** in-place replacement slot for externally supplied tested-pair losses.

**Absent:** adjacent alternatives, balanced directions/endpoints, original-state legality,
injectivity-conflict handling under one-decision intervention, pair probability, bounded pairwise
loss, full-vector `delta_U`, current reward cache, and pair diagnostics.

### Counterfactual scheduler

No implementation exists for fixed budgets, uniform reserve, coverage/uncertainty priorities,
pair age, no-starvation selection probability, or scheduler reports.

### Hybrid loss

The clean branch implements only the provisional structured utility term and substitution shape.
The final loss combining that utility term with exact count shaping, lambda conditioning, entropy,
and collapse-triggered KL does not exist.

### Training sequence

**Predecessor:** behavior cloning, utility-only ExIt, RLOO, entropy, KL, and greedy readout.

**Divergent improvement:** component-aware provisional optimization and artifact-aware ExIt cache.

**Absent:** lambda-zero identity initialization, cloning verified ExIt targets under every supported
profile, balanced Latin-cycle lambda schedule, count-health gates, and current reward cache.

### Lambda-menu selection

No calibration pool builder, `(U,P)` frontier construction, convex-envelope calculation, switch-
point extraction, quantile snapping, replay signatures, replacement passes, supported-menu
manifest, or acceptance gate is implemented.

### Gates, diagnostics, and evaluation

**Active:** QA artifact integrity gate, production QA audit, real artifact counts, reader/lattice
diagnostics.

**Divergent:** stronger old-contract preflight and reward-cache integrity checks.

**Absent:** complete-count health report, lambda responsiveness, scheduler report, diagnostic spike
manifest, per-profile held-out utility, fixed lambda-zero control, realized privacy attacker
evaluation, and matched-realized-privacy comparison.

## Verification performed for this inventory

### Active main suite

Command:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
  src/cloak/tests/test_build_ranker_env_v2.py \
  src/cloak/tests/test_build_qa_utility_artifact_cli.py \
  src/cloak/tests/test_qa_builder_v2.py \
  src/cloak/tests/test_utility_credit.py \
  src/cloak/tests/test_train_roundtrip_mode.py \
  src/cloak/tests/test_counterfactual_credit.py
```

Result: **188 passed in 4.03 seconds**.

This verifies active artifact/environment/scorer behavior and predecessor training mechanics. It
also verifies the current scalar artifact-training behavior; it does not certify structured
trainer integration.

### Divergent clean suite

Command:

```bash
PYTHONPATH=src:scripts /home/timo/repos/agent-cloak/.venv/bin/python -m pytest -q \
  src/cloak/tests/test_build_qa_utility_artifact_cli.py \
  src/cloak/tests/test_qa_builder_v2.py \
  src/cloak/tests/test_utility_credit.py \
  src/cloak/tests/test_train_roundtrip_mode.py \
  src/cloak/tests/test_counterfactual_credit.py \
  src/cloak/tests/test_roundtrip.py
```

Run from `/home/timo/repos/agent-cloak/.worktrees/qa-builder-v2-clean`.

Result: **224 passed in 4.03 seconds**.

The first attempted clean command also named `test_build_ranker_env_v2.py`, which does not exist on
that older branch; pytest exited before collection. The corrected branch-appropriate suite above
is the relevant result.

### Real-data validation boundary

The completed v16 QA artifact was inspected and its family/subtype/decision/count statistics were
recomputed. No live remote rollout, reader scoring pass, policy optimization, count objective, or
privacy attack was launched for this inventory. Therefore:

- QA-builder-v2 has real 67-document output evidence;
- active and clean mechanics have model-free test evidence;
- RL-v2 structured training remains unvalidated on real data;
- the complete interactive RL-v2 policy remains unimplemented.

## Reusable implementation assets for the remainder

The remainder plan should preserve or selectively port:

- main's current frozen environment and `policy_decision_ids`;
- main's v16 assertion compiler, scorer, full artifact, and per-assertion component vector;
- main's dynamic injectivity assembly and utility-only ExIt logic after adapting them to v2
  decisions;
- main's tested no-standardization LOO primitive;
- clean's fixed-denominator `document_utility` behavior;
- clean's per-decision linked/global/fallback credit structure;
- clean's stable decision-ID binding concept;
- clean's fixed-`1/G` structured loss and in-place substitution shape;
- clean's complete-result cache integrity and transitive reward-pin lessons;
- clean's fail-closed artifact/preflight testing style.

Do not merge the clean trainer wholesale. Reimplement these pieces against the current
`ranker-v2-environment-v1`, corrected policy-decision artifact, v16 assertion schema, current
reader/scorer, and final lambda/count/counterfactual interfaces.

## Bottom line

Substantial RL-v2 environment and reward-measurement infrastructure exists, and the earlier clean
branch proved the core structured-credit arithmetic and cache mechanics in tests. The actual
interactive RL-v2 optimizer does not exist. The missing work is not a thin final integration: it
includes correcting artifact policy identity and count state, porting structured credit to the
current schema, replacing the predecessor policy interface, implementing lambda conditioning and
count shaping, implementing real counterfactual selection/execution, composing the hybrid loss,
calibrating the lambda menu, and validating the complete loop on real data.
