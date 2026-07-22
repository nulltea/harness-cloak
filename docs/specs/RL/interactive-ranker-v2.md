---
type: reference
status: current
created: 2026-07-12
updated: 2026-07-22
tags: [rl, ranker, interactive-policy, reward-design, credit-assignment, counterfactual,
       anonymity-counts, lambda-conditioning, pareto, spec]
supersedes: docs/specs/RL/count-privacy-reward.md
companion: [docs/specs/RL/interactive-ranker-v2-decision-log.md,
            docs/specs/RL/interactive-ranker-v2-diagnostics.md,
            docs/specs/qa-builder-v2.md,
            docs/specs/RL/training-task-env.md,
            docs/specs/RL/leakage-probe-reward.md,
            docs/issues/2026-07-08-rl-env-and-lattice-count-issue-register.md]
---

# Interactive ranker v2 — conditional privacy–utility policy with structured credit

**Status: normative design, not implemented.** This specification replaces the ranker reward,
credit-assignment, and operating-point design in the round-trip ranker spec. The infiller and
the pinned round-trip generation/reader machinery remain governed by their existing specs
unless this document explicitly changes their interface.

The ranker is one lambda-conditioned sequential policy. For every detected, ranker-controlled
distinct detected value it chooses KEEP, a lattice generalization, or placeholder. It learns document task utility
from the full remote round trip, uses assertion dependencies as cheap provisional credit
routing, corrects that routing with sparse one-decision counterfactuals, and receives exact local
count shaping. A finite menu of lambda profiles lets one deployed checkpoint change its privacy–utility
preference per document or session.

Count shaping is an experimental leakage approximation. It is never a privacy measurement or
guarantee. All method comparisons remain utility versus held-out LLM re-identification success
at matched realized privacy and identical settings.

## Design invariants

1. **One frozen occurrence/decision artifact.** The composed detector runs once. QA construction
   and RL consume the same stable occurrence IDs, offsets, types, surfaces, overlap decisions,
   detector provenance, and detector version. The artifact also freezes the many-to-one mapping
   from occurrences to policy decisions.
2. **Reward scope follows causal scope.** Utility is a document outcome; count score is an
   analytic per-action quantity. They use different credit paths rather than one shared scalar
   advantage.
3. **Assertion dependencies route; they do not certify causality.** Linked assertions provide
   cheap provisional credit. Residual assertions remain active. Sparse counterfactuals supply
   contextual causal comparisons.
4. **One declared preference per document.** Lambda is selected from a finite supported menu and
   held fixed across the complete sequential action trajectory and all rollouts in its RLOO
   group.
5. **Realized privacy remains external.** Count score, lambda, KEEP rate, and lattice depth are
   diagnostics. The held-out attacker on `doc_p` and `out_final` supplies the privacy axis.

## Definitions

- **Occurrence `s`** — one detector output at one document offset with a stable `span_id`.
- **Policy decision set `D_d`** — one ranker-controlled decision per `(document, runtime type,
  normalized canonical profile/surface)`. All equivalent occurrences map to the same stable
  `decision_id`. Rule-masked PERSON/CODE values and entries with no policy choice are excluded.
- **Fixed decision** — a rewrite decision present in the frozen environment but outside the
  ranker's action space. It never receives a policy gradient.
- **Action state `h_j`** — document context aggregated around decision `j`'s occurrences,
  previous selected decisions, dynamic legal mask, action features, and selected lambda profile.
- **Utility assertion `q`** — one accepted context or delivered assertion scored from the complete
  round trip.
- **Policy dependency set** — the unique policy decisions in assertion `q`'s
  `policy_dependency_decision_ids`. It is derived from frozen occurrence routing, never from
  measurement family or scope.
- **Count score `p_j(a)`** — type-normalized bounded score for action `a` at decision `j`.
- **Document count score `P_count`** — equal mean of selected action scores over `D_d`.
- **Supported lambda menu `Lambda`** — three to five pre-registered operating profiles,
  including zero, served by one conditional checkpoint.
- **Utility counterfactual** — complete round-trip evaluation after changing one decision,
  rewriting all mapped occurrences together, and holding every other decision fixed.
- **Passenger action** — an action cloned or reinforced because its complete trajectory won,
  without evidence that the action contributed to the win.

The shared QA-to-RL artifact exposes these routing fields:

```yaml
document:
  policy_decision_ids: [decision_id]
  fixed_decision_ids: [decision_id]
  uncovered_policy_decision_ids: [decision_id]
  occurrence_to_decision: {occurrence_id: decision_id_or_null}
assertion:
  occurrence_ids: [occurrence_id]
  policy_dependency_decision_ids: [decision_id]
  credit_routing: linked | residual
```

`credit_routing=linked` exactly when the assertion has at least one policy dependency; otherwise
it is `residual`, including globally scoped assertions and assertions linked only to fixed rewrite
decisions. A mixed fixed/policy hyperedge routes once to each unique policy dependency and never
to a fixed decision. `uncovered_policy_decision_ids` is the set of policy decisions absent from
every assertion's `policy_dependency_decision_ids`.

## Episode and policy

The policy factorizes over policy decisions in deterministic first-occurrence walk order:

```text
pi(a | d, lambda) = product_j pi(a_j | h_j, lambda)
```

At each decision, the dynamic injectivity mask removes a level fill already claimed by an
earlier decision. KEEP, legal lattice levels, and placeholder otherwise remain available. The
selected action is applied consistently to every occurrence mapped to that decision. Lambda is
not a legality constraint: it changes preference over the legal menu.

The deployed interface is:

```text
rank(document, frozen_occurrences_and_decisions, lambda_profile) -> action_per_decision
```

The caller selects one supported profile before ranking the document. Changing profiles within
one document is unsupported because it changes the objective mid-trajectory.

### Lambda conditioning

The policy receives both:

- a normalized ordered magnitude, `log1p(lambda) / log1p(max(Lambda))`; and
- a learned embedding or one-hot identity for the supported profile.

A scalar added uniformly to every action logit would cancel under softmax. Conditioning must
interact with action features. The preferred head uses FiLM or explicit cross-features:

```text
z_s(a) = head(FiLM([ctx_s ; action_features_s(a)], lambda_embedding))
```

At minimum, tests must show that changing lambda changes relative logits for a fixed span and
action menu. The retired `log10_active_floor` feature slot is replaced and renamed; floor
terminology must not survive in the v2 interface.

Initialize conditioning as an identity transformation: FiLM scale heads emit one, FiLM bias
heads emit zero, profile embeddings start at zero, and any explicit lambda cross-feature branch
is zero-initialized. Before conditional optimization, every supported profile must therefore
reproduce the same unconditioned logits for identical state and legal menu. This is a tested
initialization invariant, not an expectation left to random initialization.

## Count shaping

### Type-normalized action score

For a non-placeholder action with count `K_j(a)` and runtime type `T_j`:

```text
p_j(a) = clip(log10(max(K_j(a), 1)) / log10(K_ref[T_j]), 0, 1)
p_j(KEEP) = 0
p_j(placeholder) = 1
```

`K_ref` is versioned reward state and is frozen before lambda selection. Resolve it per type in
this order:

1. Use a grounded universe size when all rewarded counts for the type are defined against that
   coherent universe.
2. Otherwise collect finite, positive, non-placeholder level counts from the frozen training
   artifact. If at least 20 distinct profiles contribute, use the profile-balanced 95th
   percentile: first take the maximum valid count within each profile, then take the percentile
   across profiles. This prevents deep profiles from receiving more reference weight merely
   because they contain more levels.
3. With fewer than 20 contributing profiles, use the maximum profile-level value and tag the
   type `low_reference_support`.
4. If every contributing value is one, mark the type `flat_count_signal`: its non-placeholder
   levels score zero and only the placeholder endpoint supplies count pressure.

Counts above `K_ref` clip to one. Clipping rate, reference provenance, profile support, and the
distribution of adjacent `delta p` values are mandatory gate outputs.

Only counts admitted by the complete-count gate below enter this calculation. Fail-closed
`1.0`, legacy-default `1000.0`, missing `level_counts`, row-level fallback counts, and the
`GENERIC` sentinel are not estimates and cannot enter `K_ref` or lattice-level shaping. Reject
by provenance/schema, not numeric value: an explicit evidenced estimate equal to `1` or `1000`
remains valid.

Type normalization is provisional. The evidence that triggers a profile-relative or
source-family-relative ablation is pinned in the
[decision log](interactive-ranker-v2-decision-log.md); a trigger causes diagnosis and a new
reward version, never retroactive renormalization.

### Document score and exact local objective

```text
P_count(d, a) = (1 / |D_d|) sum_j p_j(a_j)
```

Every policy decision has equal weight. Repeated occurrences of the same normalized value
do not multiply its count contribution: an anonymity count describes one distinct attribute
value, not the number of times it appears. Report contribution by type/profile/provenance and
report occurrence multiplicity separately.

**FORMAL-AUDIT FLAG — repeated-context leakage is real. Repeating the same value in different
contexts can increase inference or linkage risk even though v2 counts that value once. The
unique-decision mean deliberately assumes that occurrence multiplicity should not alter the
count-shaping weight. This is an accepted approximation for the current design and must be
revisited in a future formal privacy audit against occurrence-aware attackers; it is not a claim
that repetition is harmless.**

Do not pass sampled `P_count` through document RLOO. At every sampled state, optimize the
expected immediate count score over the complete legal menu:

```text
L_count = -(1/G) sum_g,k [lambda_d / |D_d|]
                         sum_a pi(a | h_gk, lambda_d) p_k(a)
```

This gives every legal action a dense, exact count gradient without remote calls. It captures
the immediate count value but not the effect of an early action on later injectivity masks.
The gate reports collision frequency and count opportunity lost through collisions. A material
effect triggers a privacy-return-to-go ablation; v2 does not add that variance preemptively.

## Utility assertions and structured credit

The complete round trip produces an assertion vector rather than only a scalar:

```text
u_g = {assertion_id: score}
U(g, Q) = sum_{q in Q} w_q u_g[q] / Z_d
Z_d = utility_weight_denominator stored for document d
```

`w_q` is the assertion weight stored by the QA artifact. `Z_d` is fixed even when a measurement
family is missing; a subset is never renormalized by its own weight sum. `U(g, empty) = 0`.

One scorer submission may flatten every context assertion from a rollout batch into one bounded
work queue. Each assertion retains its own pinned question, reader clause, turn excerpt, answer
kind, and cache identity. This scheduling does not imply one wire/model generation per rollout:
the pinned reader may issue multiple bounded transport calls. One generation is permitted only
if a separately validated multi-question reader provides that property; the live implementation
retains the existing per-assertion reader semantics.

### Linked assertion credit

For policy decision `j`, let `Q_j` be the assertions whose
`policy_dependency_decision_ids` contain `j`. Its linked score in rollout `g` is:

```text
U_link[g,j] = U(g, {q : j in policy_dependency_decision_ids[q]})
```

If an assertion depends on occurrences mapping to several policy decisions, it enters every
unique policy dependency's score once. Multiple occurrences mapping to one policy decision do
not duplicate the assertion or policy log-probability. Fixed decisions never receive credit.
This is provisional many-to-many routing, not a claim that each policy decision independently
caused the outcome.

Within one document/lambda group, compute leave-one-out advantages independently per span:

```text
A_link[g,j] = U_link[g,j] - loo_mean(U_link[-g,j])
```

No standard-deviation normalization is allowed. A linked assertion that is constant across the
group supplies no policy gradient.

### Residual assertion credit

Residual means not linked to a policy decision; it is independent of measurement family and
assertion scope. Compute:

```text
U_residual[g] = U(g, {q : credit_routing[q] = residual})
A_residual[g] = U_residual[g] - loo_mean(U_residual[-g])
```

Every policy decision with linked assertions receives `A_residual` in addition to its linked
advantage. This captures globally scoped assertions, fixed-only assertions, and other effects
that cannot route to a policy decision.

An assertion with `credit_routing=linked` is prohibited from entering `U_residual`; the partition
prevents double counting. If a document has no residual assertions, `A_residual = 0`.

For a policy decision with no linked assertions, residual-only credit may still be empty or too
thin. Define the complete document score over every accepted assertion:

```text
U_document[g] = U(g, all accepted assertions for d)
A_document[g] = U_document[g] - loo_mean(U_document[-g])
```

Use this coarse complete-document advantage only as the fallback for uncovered decisions. It is
prohibited from being added to a decision that already receives linked credit.

The provisional utility advantage is:

```text
A_provisional[g,j] = A_link[g,j] + A_residual[g]  when j has linked assertions
A_provisional[g,j] = A_document[g]                otherwise
```

Therefore, absence of an accepted QA link never implies zero utility relevance. It falls back
to the predecessor's valid but coarse trajectory credit until a counterfactual supplies causal
evidence.

### Contextual counterfactual credit

For a selected rollout and decision, choose one alternative `b_j` that was legal in the original
state. Construct `a_cf` by changing only `a_j` across every mapped occurrence and preserving
every other decision. If the alternative conflicts with a later selected fill, choose another
eligible alternative; never silently repair or resample the suffix, because that would cease to be a one-decision
intervention.

Run the complete pinned remote task, extraction, and reader pipeline:

```text
delta_U[g,j] = U_total(d, a_g) - U_total(d, a_cf)
```

`U_total` is `U(g, all accepted assertions for d)` with the fixed `Z_d`. This allows a decision
with no linked assertion to receive causal evidence through effects on other assertions.

Convert the policy's full-menu probabilities into a distribution restricted to the evaluated
pair:

```text
q_pair[g,j] = pi(a_gj | h_gj, lambda_d) /
              [pi(a_gj | h_gj, lambda_d) + pi(b_j | h_gj, lambda_d)]

L_cf[g,j] = -delta_U[g,j] * [q_pair[g,j] - 1/2]
```

The adjacent alternative depends on the sampled action, so it is not treated as an ordinary
action-independent REINFORCE baseline. This bounded loss has the same gradient as maximizing
the measured local two-action expected utility
`q_pair * U(a) + (1-q_pair) * U(b)`: positive `delta_U` favors the selected action, negative
`delta_U` favors the alternative, and the gradient saturates as the pairwise policy becomes
confident. The `1/2` centers the logged loss and changes no gradient. This is a contextual local
auxiliary objective, not an unbiased estimator over unmeasured alternatives.

Alternative selection is:

1. Prefer an adjacent finer or coarser lattice action.
2. Balance finer and coarser tests over each eligible profile.
3. Reserve a pre-registered minority of tests for KEEP and placeholder endpoints.
4. Skip duplicate-text, equal-action, illegal, and injectivity-conflicting alternatives.

Endpoint fraction and direction-balancing tolerance are run configuration, frozen before the
support gate.

### Counterfactual budget scheduler

Counterfactuals consume a fixed number of additional round trips per epoch or training run.
The default scheduler reserves 20% of calls for seeded uniform sampling over all eligible
decision-rollout pairs. The remaining 80% is allocated lexicographically:

1. decisions with no linked assertion;
2. decisions with multi-decision (hyperedge) links;
3. high policy entropy at the current supported lambda;
4. unseen adjacent action pairs;
5. oldest measured pair, preventing priority-tier starvation.

Within one priority tier, sample uniformly and deterministically from the run seed. Priority
controls measurement allocation only; it never multiplies `delta_U` or the loss. Cache keys
include the complete reward pin and action vector, so repeated pairs reuse prior round trips.

## Hybrid ranker loss

Define exactly one utility term for every rollout-decision pair:

```text
ell[g,j] = -A_provisional[g,j] log pi(a_gj | h_gj, lambda_d)
           if (g,j) is not counterfactually tested

ell[g,j] = L_cf[g,j]
           otherwise
```

Counterfactual credit substitutes in place; it is never added as a separately averaged loss.
For one document group:

```text
L_utility(d) = (1/G) sum_g sum_j ell[g,j]
```

The `1/G` weight is shared by provisional and counterfactual terms. It does not depend on the
number of counterfactual calls, so changing the measurement budget does not silently rescale
each causal correction. Utility is not divided by `|D_d|`: it is the return of the joint policy,
whose trajectory log-probability is the sum over decisions. The complete minibatch loss is:

```text
L = mean_d L_utility(d) + L_count - beta * entropy + eta * KL(pi || pi_ref)
```

RLOO tie filtering is applied at the assertion-credit level: a span with tied linked and residual
advantages contributes no provisional utility term, but its count objective remains active.
Counterfactual pairs with `delta_U = 0` are retained in diagnostics and supply no pairwise
utility gradient.

Entropy and KL coefficients are fixed per run and are not rescaled by lambda. This is why the
additive objective is used rather than `(1-lambda)U + lambda P`. KL begins off after ExIt and is
enabled only by a pre-registered collapse rule; it is never tuned separately per lambda profile.

## Training sequence

### Frozen environment and reward artifacts

Before optimization:

- freeze one detector span artifact shared by QA and RL;
- freeze action menus, counts, count provenance, and type references;
- freeze accepted utility assertions and routing fields;
- freeze remote model, task prompt, extractor, reader, scorer, concurrency regime, and caches;
- run reward-support, count-health, lambda-menu, and determinism gates.

Changing any item invalidates cached utility, switch-point calibration, and trained-policy
comparisons together.

### Complete-count pre-training gate

This gate runs before lambda calibration, ExIt pool reuse for menu selection, or conditional
RL. For every ranker-controlled profile and every non-KEEP, non-placeholder generalization
level, require:

- an explicit `level_counts[level]` entry with a finite value `>= 1`;
- a matching `level_grounding[level]` record;
- grounding status in the pre-registered accepted set (`certifying`, `model-proposed`, or
  `proposal-universe` for the current experiment);
- status-appropriate evidence: member-set reference for certifying counts; selector and count
  evidence for model-proposed counts; generated-universe reference for proposal-universe counts;
- non-decreasing counts from finer to coarser levels within the profile;
- no row-level `count`, `aset_count` fallback, missing-action `1e9`, or generic sentinel used to
  fill a level count.

The report lists every profile and level with value, status, source family, and evidence
reference, plus aggregate coverage by type and provenance. **Pass requires 100% explicit
coverage of all trainable generalization levels.** Failure blocks the whole run: do not drop a
profile, prune a level, substitute a default, or continue under a warning, because those choices
would silently change the action space or reward definition.

KEEP remains the explicit score-zero endpoint and placeholder the explicit score-one endpoint;
neither is required to carry a lattice count.

### Behavior-cloning initialization

Retain the existing deterministic behavior-cloning teacher as a support-preserving
initialization. It is lambda-independent. Record its action distribution and utility/count
point at every supported lambda, but do not represent it as an operating frontier.

### Utility-only expert iteration

ExIt is a coarse utility initializer:

```text
for each document:
    sample G complete trajectories
    score pure U_total only
    select the best trajectory strictly beating the BC reference
    serially reverify winner and reference with reader refresh
    clone every action in the verified winner
```

Lambda and count score do not affect ExIt selection. Complete winner cloning may include
passenger actions; this is accepted only as initialization. The hybrid stage supplies the
structured and causal correction.

ExIt samples and verifies trajectories under the lambda-zero input. After a winner is verified,
clone its identical action targets once under every supported profile input. This adds no remote
calls and preserves the utility warm start across all profile embeddings before count-conditioned
optimization begins.

### Lambda-conditioned hybrid optimization

All rollouts in one document group use the same lambda. Use a seeded balanced Latin-cycle
schedule over documents, epochs, and supported profiles:

```text
lambda_index = (permutation_epoch[doc_index mod |Lambda|] + epoch) mod |Lambda|
```

Any equivalent scheduler is acceptable only if, over every block of `|Lambda|` epochs, each
document is trained once at every supported profile and corpus/type exposure per profile is
reported. Random independent sampling without balance is not allowed.

## Selecting the supported lambda menu

The menu is chosen once from train/development artifacts before conditional RL. Final held-out
attacker results never select or alter lambda values.

### Calibration pool

Build a frozen per-document pool containing:

- the behavior-cloning trajectory;
- the verified utility-only ExIt winner when one exists;
- KEEP-walk, minimum-count non-KEEP walk, midpoint-level walk, and all-placeholder anchors;
- support-scan and ExIt sampled trajectories with valid cached utility;
- adjacent single-span counterfactual trajectories already measured by the support scan.

Deduplicate exact action vectors. Store pure `U_total`, `P_count`, assertion scores, count
provenance, and the complete reward/environment version. Require at least three distinct
`(U,P)` points for a document to contribute switch points; other documents remain in replay
validation but not threshold estimation.

### Per-document frontier and switch points

Within each document:

1. Merge exact `(U,P)` ties, retaining one canonical action vector and tie multiplicity.
2. Remove weakly dominated points: discard a trajectory if another has no lower utility and no
   lower count score, with at least one strict inequality.
3. Construct the upper convex envelope in `(P,U)` space. Points below a chord cannot win for
   any additive lambda and do not define settings.
4. Order envelope vertices by increasing `P`. For each adjacent pair where utility decreases
   and count score increases, compute:

```text
lambda_star = (U_left - U_right) / (P_right - P_left)
```

5. Discard non-finite, non-positive, and numerically duplicate thresholds. Each contributing
   document receives total weight one, divided equally among its retained thresholds, so a
   document with many sampled trajectories cannot dominate menu selection.

### Candidate profiles

Start with `lambda = 0`. For the nonzero settings:

1. Form the weighted empirical distribution of all retained switch points in log-lambda space.
2. For the default four-profile menu, take weighted quantiles 0.25, 0.60, and 0.90. A
   three-profile menu uses 0.40 and 0.90; a five-profile menu uses 0.20, 0.45, 0.70, and 0.90.
3. Snap each quantile to the nearest observed switch point; never invent precision unsupported
   by the pool.
4. Merge duplicates and values whose replay signatures are equivalent. A replay signature is
   the per-document winning trajectory under `U + lambda P`, plus aggregate KEEP/level/
   placeholder rates.
5. If fewer than three distinct profiles remain, the count objective lacks a supported
   controllable range: stop rather than padding the menu with arbitrary values.

### Replay acceptance gate

Replay every calibration trajectory under each candidate lambda. Accept the menu only if:

- every adjacent pair changes the winning trajectory on at least 10% of documents that have
  two or more nondominated points;
- aggregate `P_count` of selected winners is non-decreasing across profiles; this is a
  scalarization code-correctness assertion, not evidence that the menu is useful or private;
- no adjacent profiles have identical action-mode rates and document winners;
- `lambda = 0` selects exactly by pure utility;
- the highest nonzero profile does not select all-placeholder for more than 95% of controlled
  documents unless it is explicitly retained as a diagnostic-only endpoint;
- each setting has support across every training corpus and every runtime type with controlled
  actions.

If a pair is redundant, replace the less central value with the observed switch point that
maximizes newly changed document winners while preserving order. Run at most two deterministic
replacement passes. Failure after two passes means the supported menu is smaller; it is not a
license to alter rewards or counts.

Freeze the accepted values, human-facing profile names, calibration-pool hash, switch-point
artifact, replay report, and menu-selection code version in the training record before RL.

## Gates

All unresolved support, materiality, trigger, and promotion language in this section is
operationalized by the frozen threshold manifest defined in
[interactive ranker v2 diagnostics](interactive-ranker-v2-diagnostics.md). Exact invariants
remain fixed by this document; empirical boundaries are selected by the preflight protocol
before full RL and held-out evaluation.

### Reward and credit support

- Round-trip determinism and reader-jitter gates from the predecessor spec pass under the live
  pinned environment.
- Utility assertions are returned separately with stable assertion IDs and routing fields.
- Every corpus reports linked and residual assertions, decisions with no links, occurrence
  multiplicity, and rejection reasons.
- Synthetic tests verify that occurrence links map to the correct policy decisions, linked credit
  reaches only policy dependencies, residual credit reaches every linked policy decision and no
  fixed decision, uncovered
  decisions receive complete-document fallback credit, and linked decisions do not receive a
  duplicate complete-document term.
- Counterfactual tests verify one-decision intervention across every mapped occurrence,
  pairwise sign, collision handling, endpoint scheduling, and replacement of provisional credit.

### Count health

- The complete-count pre-training gate passes at 100% level coverage before lambda selection.
- Report `K_ref`, provenance, profile support, clipping rate, flat-menu rate, and adjacent
  `delta p` distributions per type.
- Report expected count-gradient mass by type, profile, and provenance.
- Assert that fallback/default provenance contributes exactly zero lattice-level gradient mass.
- Count score monotonicity follows the stored counts; any lattice/count non-monotonicity blocks
  the run rather than being silently repaired or excluded at reward time.

### Conditional-policy responsiveness

- Fixed-document logits differ across supported profiles for action menus with non-flat count
  scores.
- Greedy `P_count` is non-decreasing across supported profiles on development documents.
- KEEP, level, and placeholder rates are reported by profile and type.
- Every supported profile receives balanced document/corpus/type exposure during training.
- Lambda zero is compared against a utility-only fixed-condition control to price conditional
  interference.

### Counterfactual scheduler

- Exact call budget, uniform reserve, priority counts, cache-hit rate, and measured-pair age are
  reported.
- No eligible decision has zero selection probability.
- Selection priority never enters reward or loss magnitude.
- Decisions without linked assertions receive complete-document fallback credit and highest
  counterfactual priority. Multi-decision links are the next priority; dependency confidence is
  neither stored nor derived from provenance.

## Evaluation and verdict

Evaluate every supported profile of the single checkpoint independently. For each profile:

- measure utility on `out_final` with task metrics and whole-task regression metrics;
- measure re-identification/inference attack success on `doc_p` and leak-through on `out_final`;
- report count score, action modes, per-type behavior, assertion coverage, and count provenance as
  diagnostics;
- compare against rule baselines and external methods only at matched realized privacy and
  identical settings.

Train a fixed lambda-zero utility control in every run family. The conditioned checkpoint's
lambda-zero profile must pass the pre-registered paired non-inferiority test against it; failure
rejects the conditioned checkpoint. Additional fixed nonzero-lambda policies are optional
certification ablations and are not deployment artifacts. If they are absent, no claim is made
that the conditioned checkpoint matches the separate-policy frontier. If they are present, any
frontier-equivalence or non-domination claim uses the frozen document-bootstrap procedure from
the diagnostics manifest.

Higher lambda is not required to produce monotonically better realized attacker privacy.
Non-monotonicity is evidence that the count approximation fails in that region and is reported,
never calibrated away.

## Failure modes and stop conditions

- **QA coverage capture:** policy movement concentrates on decisions with linked assertions while
  uncovered decisions remain static. Check complete-document fallback credit and scheduler
  allocation; do not drop the uncovered decisions.
- **Passenger persistence:** ExIt-cloned actions survive despite negative adjacent
  counterfactual utility. Price with measured pairwise disagreement and increase only the
  pre-registered counterfactual budget in a new run.
- **Count coverage regression:** a profile rebuild introduces a missing, fallback, or
  non-monotone level count. Invalidate lambda calibration and block training until the artifact
  passes again.
- **Conditional collapse:** all profiles produce one policy or high profiles collapse to
  placeholder everywhere. Fail the responsiveness/menu gate.
- **Type-normalization distortion:** any revisit trigger in the decision log fires. Run the
  declared normalization ablation; do not rewrite the completed run's score.
- **Interaction miss:** one-decision counterfactuals disagree systematically with multi-decision
  assertion behavior. Add a multi-decision intervention ablation rather than pretending independent
  credit.
- **Proxy/attacker inversion:** count score rises while realized privacy worsens. This is the
  central falsification of count-driven shaping and must terminate claims based on it.

## Implementation seams

The implementation should deepen these modules rather than scatter reward arithmetic through
the trainer:

- **Conditional policy interface:** `log_probs(state, legal, lambda_profile)` and
  `sample(state, legal, lambda_profile)` own conditioning and dynamic-mask semantics.
- **Count objective interface:** versioned type references plus
  `action_scores(decision, legal) -> scores, provenance` hide normalization and diagnostics.
- **Utility credit interface:** `credit(assertion_vector, occurrence_to_decision,
  policy_dependency_decision_ids, credit_routing, rollout_group)` returns per-policy-decision
  provisional advantages without knowing policy internals.
- **Counterfactual scheduler interface:** consumes linked-assertion coverage, hyperedge status,
  entropy, pair history, and a fixed budget; returns intervention requests. It never sees or
  scales rewards.
- **Lambda-menu selector:** consumes a frozen trajectory pool and emits values, switch points,
  replay report, and hashes. It is offline and cannot inspect final attacker results.

Expected code destinations are `src/cloak/train/ranker.py` for policy conditioning,
`src/cloak/train/reward.py` or a dedicated deep reward module for count/credit calculation,
`src/cloak/train/roundtrip.py` for assertion-vector output, and `scripts/train_ranker.py` for
orchestration only. Exact paths may change during implementation planning, but the interfaces
and separation of responsibilities are normative.

## Artifacts

- frozen detector/span artifact shared by QA and RL;
- versioned utility-assertion artifact with routing fields and fixed per-document denominators;
- type count-reference artifact and count-health report;
- lambda calibration pool, switch points, replay report, and supported-menu manifest;
- counterfactual pair cache and scheduler report;
- conditional-policy training record under `research-wiki/training/` written before the run;
- per-profile held-out utility and attacker results.

## Sources and predecessor specifications

- [Round-trip ranker and infiller](roundtrip-ranker-infiller.md) — pinned remote reward,
  determinism, ExIt/RLOO history, and infiller stages.
- [Training-task environment](training-task-env.md) — historical ladder, decision, schema, and
  carrier inputs; not the live ranker-v2 reward contract.
- [Leakage-probe reward options](leakage-probe-reward.md) — count, span-recovery, and
  profile-matching alternatives and honesty boundaries.
- [Interactive ranker v2 decision log](interactive-ranker-v2-decision-log.md) — chosen forks and
  rejected alternatives.
- [QA builder v2](../qa-builder-v2.md) — implemented shared assertion artifact and scoring
  contract.
