---
type: reference
status: current
created: 2026-07-12
updated: 2026-07-22
tags: [rl, ranker, interactive-policy, reward-design, credit-assignment, counterfactual,
       anonymity-counts, lambda-conditioning, semantic-privacy, pareto, spec]
supersedes: docs/specs/RL/count-privacy-reward.md
companion: [docs/specs/RL/interactive-ranker-v2-decision-log.md,
            docs/specs/RL/interactive-ranker-v2-diagnostics.md,
            docs/specs/RL/ranker-v2-architecture.md,
            docs/specs/RL/ranker-v2-architecture-decision-log.md,
            docs/specs/qa-builder-v2.md,
            docs/specs/RL/training-task-env.md,
            docs/specs/RL/leakage-probe-reward.md,
            docs/issues/2026-07-08-rl-env-and-lattice-count-issue-register.md]
---

# Interactive ranker v2 — conditional privacy–utility policy with structured credit

**Status: normative design; reward-flow implementation exists, selected architecture remains to
be implemented and validated.** This specification replaces the ranker reward, credit-assignment,
and operating-point design in the round-trip ranker spec. The companion
`ranker-v2-architecture.md` is normative for model inputs, representations, and controller
factorization. The infiller and pinned round-trip generation/reader machinery remain governed by
their existing specs unless this document explicitly changes their interface.

The ranker is one sequential policy with an explicit lambda controller. For every distinct
ranker-controlled detected value it chooses KEEP, a lattice generalization, or placeholder. Its
utility tower learns document-task preservation from the full remote round trip. A separately
pretrained semantic privacy head predicts each source-to-candidate abstraction's log-count
distribution, while exact own-profile counts provide the local shaping target. Assertion
dependencies route cheap provisional utility credit and sparse one-decision counterfactuals
correct that routing. A finite lambda menu lets one checkpoint change its privacy--utility
preference per document or session without changing either semantic representation.

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
6. **Semantic and controller gradients are separated.** Exact count shaping cannot update the
   utility tower or selected-action memory. Privacy-head supervision cannot update utility
   parameters. During hybrid RL, utility and exact count objectives may both update the one global
   controller scale because their opposition determines its finite operating range.

## Definitions

- **Occurrence `s`** — one detector output at one document offset with a stable `span_id`.
- **Policy decision set `D_d`** — one ranker-controlled decision per `(document, runtime type,
  normalized canonical profile/surface)`. All equivalent occurrences map to the same stable
  `decision_id`. Rule-masked PERSON/CODE values and entries with no policy choice are excluded.
- **Fixed decision** — a rewrite decision present in the frozen environment but outside the
  ranker's action space. It never receives a policy gradient.
- **Semantic action state `h_j`** — candidate-conditioned document context around decision `j`'s
  occurrences, source-to-candidate utility relation, selected-action utility memory, and dynamic
  legal mask. It excludes lambda, counts, predicted privacy, authored level position, and menu
  size.
- **Utility assertion `q`** — one accepted context or delivered assertion scored from the complete
  round trip.
- **Policy dependency set** — the unique policy decisions in assertion `q`'s
  `policy_dependency_decision_ids`. It is derived from frozen occurrence routing, never from
  measurement family or scope.
- **Exact count target `p_j(a)`** — frozen own-profile-relative score in `[0,1]` derived from the
  admitted `level_counts` for action `a` at decision `j`; used by shaping and diagnostics, never as
  an actor feature.
- **Predicted privacy score `p_hat_j(a)`** — profile-menu normalization of the frozen semantic
  privacy head's predicted log-count mean; used by the deployed controller.
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

First-occurrence order is the selected prototype's canonical factorization. Selected-action
memory contains only earlier decisions in this order and has no selection-step positional
embedding. Order sensitivity is measured by replaying development documents under deterministic
reverse-order and seeded-order diagnostic walks while preserving legal-mask semantics. Material
utility or action changes reopen a two-pass draft-and-refine policy; they do not authorize silent
training-time order randomization.

### Lambda conditioning

Lambda does not enter the utility tower, semantic privacy head, document attention, relation
features, or selected-action memory. The separately produced scores combine only at the action
logit:

```text
z_j(a, lambda) = u_theta(h_j, a) + alpha * g(lambda) * p_hat_j(a)

alpha = softplus(alpha_raw)
g(lambda) = log1p(lambda) / log1p(max(Lambda))
```

`g` is fixed by the supported numeric menu. There is no profile embedding, one-hot identity,
FiLM, per-profile slope, or profile-specific head. `alpha` is one globally shared nonnegative
scalar. At lambda zero the controller is identically zero for every parameter value, so combined
logits equal utility logits exactly, not merely at initialization.

The semantic privacy head is frozen before hybrid RL. Utility losses and exact count shaping both
may update `alpha`; that shared scalar is where utility and privacy pressure balance. Exact count
shaping cannot update `u_theta`, and utility losses cannot update the privacy head. `alpha` is
frozen with the checkpoint before held-out evaluation and is never calibrated separately by
profile, type, corpus, method, or evaluation set.

## Count shaping

### Profile-relative exact target

For a lattice-level action with admitted own-profile count `K_j(a)`:

```text
ell_j(a) = log(max(K_j(a), 1))
denom_j = max_b_in_profile_levels ell_j(b)
p_j(a) = clip(ell_j(a) / denom_j, 0, 1)

p_j(KEEP) = 0
p_j(placeholder) = 1
```

The denominator is computed only across that matched profile's lattice levels; KEEP and
placeholder do not enter it. A one-level profile assigns its sole level score one and is tagged
`singleton_profile_normalization`, including when its count is one. A profile with two or more
levels whose admitted level log counts are all zero is flat and fails the validated
privacy-head/count gate. Equal profile-relative scores across different profiles do not claim equal
anonymity-set size.

Only counts admitted by the complete-count gate enter this target. Fail-closed `1.0`,
legacy-default `1000.0`, missing `level_counts`, row-level fallback counts, and generic sentinels
are not estimates. Reject by provenance/schema, not numeric value: explicit evidenced estimates
equal to `1` or `1000` remain valid. Model-proposed counts are permitted only as separately
reported experimental shaping supervision and never become formal privacy labels.

The retained direct-count fallback in `ranker-v2-architecture.md` uses strict type normalization.
It is a separate architecture and reward version, not a silent runtime substitution.

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
expected immediate exact count target over the complete legal menu. Construct a count-gradient
policy view that detaches semantic outputs but preserves the controller scale:

```text
pi_count(a | h, lambda) = softmax(mask(
    stop_gradient(u_theta(h, a))
    + alpha * g(lambda) * stop_gradient(p_hat(a))))

L_count = -(1/G) sum_g,k [lambda_d / |D_d|]
                         sum_a pi_count(a | h_gk, lambda_d) p_k(a)
```

This gives every legal action a dense exact-target gradient without remote calls, but only
`alpha` receives that gradient. Utility and counterfactual losses use the ordinary policy and may
also update `alpha`; without this opposing utility gradient, privacy-only optimization would drive
the scale toward saturated privacy pressure. The objective captures immediate count value but not
the effect of an early action on later injectivity masks. The gate reports collision frequency and
count opportunity lost through collisions. A material effect triggers a privacy-return-to-go
ablation; v2 does not add that variance preemptively.

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

Gradient ownership is:

```text
L_utility, entropy, KL --> utility tower, selected-action memory, alpha
L_count                --> alpha only
L_privacy              --> privacy projection and semantic privacy head only
```

The semantic privacy head remains frozen throughout hybrid RL. The exact target in `L_count`
therefore cannot turn the predicted privacy score into an opaque policy preference, while the
utility gradient prevents the globally shared controller scale from optimizing privacy alone.

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
- freeze action menus, own-profile counts, count provenance, and normalization tags;
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

### Semantic privacy-head pretraining

Before policy optimization, train the privacy projection and log-count distribution head from the
ordered source-to-candidate relation encoding. Split by complete profile so no source surface,
candidate menu, or count trajectory crosses train and validation. Optimize log-count likelihood,
within-menu ordering, and profile-relative calibration. KEEP and placeholder have fixed endpoint
scores and do not enter learned-head calibration metrics.

The head must beat authored-position, action-mode/type, profile-memorization, and candidate-only
baselines on profile-held-out data. Freeze the validated head before behavior cloning, ExIt, lambda
selection, and hybrid RL. Failure blocks the semantic prototype and triggers the separately logged
direct-count fallback evaluation; it does not permit true count or authored position to enter the
semantic actor.

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

ExIt samples and verifies trajectories under lambda zero. Clone each verified winner once into
the shared utility tower and selected-action memory. No profile replication is needed because the
semantic policy has no lambda identity input and the controller is exactly zero at lambda zero.

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
- Report own-profile denominators, provenance, singleton-profile rate, flat-menu rate, and
  adjacent exact-target `delta p` distributions per type and provenance.
- Report expected count-gradient mass by type, profile, and provenance.
- Assert that fallback/default provenance contributes exactly zero lattice-level gradient mass.
- Count score monotonicity follows the stored counts; any lattice/count non-monotonicity blocks
  the run rather than being silently repaired or excluded at reward time.
- Report profile-held-out semantic-head log-count likelihood, multiplicative error, interval
  coverage, within-menu ordering, profile-relative calibration, paraphrase stability, and lexical
  counterexamples separately for grounded and experimental count provenance.

### Conditional-policy responsiveness

- Fixed-state logits differ across supported lambda values only through
  `alpha * g(lambda) * p_hat`; utility and privacy predictions remain lambda-invariant.
- For a fixed state and legal menu, expected predicted privacy is non-decreasing with lambda.
- Greedy predicted and exact `P_count` are reported across supported profiles on development
  documents; whole-document non-monotonicity is diagnosed because earlier actions can change later
  legal menus and prediction error can invert exact count ordering.
- KEEP, level, and placeholder rates are reported by profile and type.
- Every supported profile receives balanced document/corpus/type exposure during training.
- Lambda zero is compared against a utility-only fixed-condition control to price conditional
  interference.
- Deterministic reverse-order and seeded-order replays quantify first-occurrence-order sensitivity.

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
- **Semantic privacy transfer failure:** the head memorizes profiles, misorders unseen menus, or
  fails calibration/paraphrase/lexical-counterexample gates. Reject the semantic prototype and
  evaluate the strict direct-count fallback; do not expose true counts to the utility tower.
- **Profile-normalization distortion:** profile-relative progress produces misleading cross-profile
  privacy pressure. Diagnose against raw-count and realized-attacker outcomes and run the logged
  strict type-normalized fallback as a new reward version; do not rewrite a completed run.
- **Order sensitivity:** reverse or seeded decision walks materially change utility or actions.
  Reopen two-pass draft-and-refine inference; do not silently randomize training order.
- **Interaction miss:** one-decision counterfactuals disagree systematically with multi-decision
  assertion behavior. Add a multi-decision intervention ablation rather than pretending independent
  credit.
- **Proxy/attacker inversion:** count score rises while realized privacy worsens. This is the
  central falsification of count-driven shaping and must terminate claims based on it.

## Implementation seams

The implementation should deepen these modules rather than scatter reward arithmetic through
the trainer:

- **Semantic policy interface:** `semantic_scores(state, legal) -> utility_logits,
  predicted_logcount_distribution` owns candidate-conditioned context and selected-action memory;
  neither output depends on lambda or true counts.
- **Controller interface:** `combine(utility_logits, predicted_privacy, lambda) -> logits` owns the
  fixed `g`, one nonnegative `alpha`, and exact lambda-zero identity.
- **Exact count-target interface:** versioned own-profile counts plus
  `action_targets(decision, legal) -> scores, provenance` hide profile normalization and
  diagnostics; targets never enter actor features.
- **Utility credit interface:** `credit(assertion_vector, occurrence_to_decision,
  policy_dependency_decision_ids, credit_routing, rollout_group)` returns per-policy-decision
  provisional advantages without knowing policy internals.
- **Counterfactual scheduler interface:** consumes linked-assertion coverage, hyperedge status,
  entropy, pair history, and a fixed budget; returns intervention requests. It never sees or
  scales rewards.
- **Lambda-menu selector:** consumes a frozen trajectory pool and emits values, switch points,
  replay report, and hashes. It is offline and cannot inspect final attacker results.

Expected code destinations are focused model modules under `src/cloak/train/` for frozen encoding,
semantic privacy, document context, selected-action memory, and the additive controller;
`src/cloak/train/interactive_ranker.py` for trajectory/loss assembly;
`src/cloak/train/roundtrip.py` for assertion-vector output; and
`scripts/train_interactive_ranker.py` for orchestration only. Exact paths are fixed by the
implementation plan, but these responsibility boundaries are normative.

## Artifacts

- frozen detector/span artifact shared by QA and RL;
- versioned utility-assertion artifact with routing fields and fixed per-document denominators;
- own-profile count-target artifact and count-health report;
- frozen semantic privacy-head checkpoint with profile-held-out calibration report;
- frozen document/relation encoder identity and content-addressed token/relation caches;
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
- [Ranker v2 architecture](ranker-v2-architecture.md) — normative model inputs, semantic privacy
  head, context readout, selected-action memory, and additive controller.
- [Ranker v2 architecture decision log](ranker-v2-architecture-decision-log.md) — architecture
  alternatives and escalation evidence.
- [QA builder v2](../qa-builder-v2.md) — implemented shared assertion artifact and scoring
  contract.
