---
type: reference
status: current
created: 2026-07-12
updated: 2026-08-05
tags: [rl, ranker, interactive-policy, reward-design, credit-assignment, counterfactual,
       anonymity-counts, lexicographic-rl, primal-dual, actor-critic, low-rank-adapters,
       freeze-policy, spec]
supersedes: docs/specs/RL/count-privacy-reward.md
companion: [docs/specs/RL/interactive-ranker-v2-decision-log.md,
            docs/specs/RL/interactive-ranker-v2-diagnostics.md,
            docs/specs/RL/ranker-v2-architecture.md,
            docs/specs/RL/ranker-v2-architecture-decision-log.md,
            docs/specs/qa-builder-v2.md,
            docs/specs/RL/training-task-env.md,
            docs/specs/RL/leakage-probe-reward.md,
            docs/research/ranker-v2-trainable-multi-objective-rl-review.md,
            docs/issues/2026-07-08-rl-env-and-lattice-count-issue-register.md]
---

# Interactive ranker v2 — lexicographic policy with structured credit

**Status: normative redesign; implementation pending.** The semantic-v1 additive-controller stack
remains implemented for historical experiments but is superseded as the selected training path.
This specification retains the frozen environment, exact count targets, QA assertion routing,
counterfactual scheduler, and sequential policy factorization. It replaces additive lambda
scalarization with policy-based lexicographic optimization while retaining a user-facing discrete
lambda setting whose semantics are explicit utility slack, not reward weight. The companion
`ranker-v2-architecture.md` is normative for the frozen encoder substrate, frozen utility branch,
private lexicographic semantic side path, objective critics, and training-only dual state.

For every distinct ranker-controlled detected value, the ranker chooses KEEP, a lattice
generalization, or placeholder. One checkpoint serves a learned utility-first/count-second policy
conditioned on one of three to five frozen user settings $\lambda_k$.
Its frozen utility branch remains an internal reference and audit path, not a second report mode.
Count and utility actor surrogates train a private post-encoder semantic side path plus residual
head while a document-level utility constraint determines how much utility gradient is required.
No deterministic selector, additive count bonus, or inference-time dual variable participates in
the served action distribution.

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
4. **One served objective, multiple explicit tolerances.** The checkpoint always implements
   utility-first/count-second. $\lambda_0$ is the exact utility identity; positive settings share
   one conditioned Tier-3 actor and differ by pre-registered document-level utility slack.
5. **Realized privacy remains external.** Count score, KEEP rate, and lattice depth are
   diagnostics. The held-out attacker on `doc_p` and `out_final` supplies the privacy axis.
6. **Objective semantics are separated; behavioral gradients are unified.** Count cannot update the
   frozen encoder substrate, frozen utility branch, or utility critic. Utility cannot update the
   count critic. Both utility and count actor surrogates update the private Tier-3 semantic adapters
   and residual head; the document dual controls their relative force from measured utility
   violation.
7. **Training state is not inference calibration.** Document-setting dual variables, critics, utility
   references, and tolerance schedules exist only during training. The served Tier-3 actor receives
   no document-specific scalar; it receives only the user-selected finite-menu setting identity.

## Definitions

- **Occurrence `s`** — one detector output at one document offset with a stable `span_id`.
- **Policy decision set `D_d`** — one ranker-controlled decision per `(document, runtime type,
  normalized canonical profile/surface)`. All equivalent occurrences map to the same stable
  `decision_id`. Rule-masked PERSON/CODE values and entries with no policy choice are excluded.
- **Fixed decision** — a rewrite decision present in the frozen environment but outside the
  ranker's action space. It never receives a policy gradient.
- **Frozen substrate `B_j(a)`** — detached clinical token/relation banks, occurrence/chunk masks,
  frozen action metadata, and canonical prior selected-action records available to both semantic
  branches.
- **Utility action state `x_U,j(a)`** — Tier-2 candidate-conditioned document context,
  source-to-candidate relation, selected-action utility memory, and interaction features. It is
  trained before lexicographic RL and then frozen.
- **Lexicographic action state `x_lex,j(a)`** — Tier-3 semantic features computed from the same
  substrate through frozen Tier-2 maps plus private rank-4 deltas. It excludes counts, authored
  level position, dual state, and menu size.
- **Lambda menu $\Lambda$** — a checkpoint-pinned ordered set
  $\{\lambda_0,\ldots,\lambda_{K-1}\}$ with $3\le K\le5$. It is a finite set of user choices, not a
  continuous scalar domain.
- **Lambda setting $\lambda_k$** — one stable menu ID selected for a complete task/session and held
  fixed for every decision and rollout in one document episode. $\lambda_0$ is the utility-only
  identity; $k>0$ activates the conditioned lexicographic actor.
- **Utility slack $\tau(\lambda_k)$** — the maximum pre-registered document-level utility reduction
  allowed relative to $b_d$ for setting $\lambda_k$. It is not a reader-noise estimate and never
  scales count reward.
- **Utility assertion `q`** — one accepted context or delivered assertion scored from the complete
  round trip.
- **Policy dependency set** — the unique policy decisions in assertion `q`'s
  `policy_dependency_decision_ids`. It is derived from frozen occurrence routing, never from
  measurement family or scope.
- **Exact count target `p_j(a)`** — frozen own-profile-relative score in `[0,1]` derived from the
  admitted `level_counts` for action `a` at decision `j`; used by shaping and diagnostics, never as
  an actor feature.
- **Document count score `P_count`** — equal mean of selected action scores over `D_d`.
- **Utility reference `b_d`** — the frozen utility-only policy's achieved expected return on
  document `d` under a version-pinned rollout manifest.
- **Document-setting dual `mu_d,k`** — nonnegative training-only multiplier that increases when the
  served policy at $\lambda_k$ violates $b_d-\tau(\lambda_k)$ and is absent at inference.
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

The served policy factorizes over policy decisions in deterministic first-occurrence walk order:

```text
pi_lex(a | d, lambda_k) = product_j pi_lex(a_j | B_j, lambda_k)
```

At each decision, the dynamic injectivity mask removes a level fill already claimed by an
earlier decision. KEEP, legal lattice levels, and placeholder otherwise remain available. The
selected action is applied consistently to every occurrence mapped to that decision.

The deployed interface is:

```text
rank(document, frozen_occurrences_and_decisions, lambda_setting_id) -> action_per_decision
```

The caller selects one registered `lambda_setting_id`; arbitrary numeric values and within-document
setting changes are rejected. The same setting is used for every sequential decision, old-policy
replay, counterfactual evaluation, and report record associated with that document.

First-occurrence order is the selected prototype's canonical factorization. Selected-action
memory contains only earlier decisions in this order and has no selection-step positional
embedding. Order sensitivity is measured by replaying development documents under deterministic
reverse-order and seeded-order diagnostic walks while preserving legal-mask semantics. Material
utility or action changes reopen a two-pass draft-and-refine policy; they do not authorize silent
training-time order randomization.

### Three-tier policy branching

The utility branch is frozen after BC, verified ExIt, and utility-only structured RL. The served
path adds a zero-initialized action-conditioned residual produced by a private semantic side path:

$$
x_{U,j}(a)=F_\theta(B_j(a)),
\qquad
u_j(a)=f_\theta(x_{U,j}(a)),
$$

$$
x_{\mathrm{lex},j}(a)=F_{\theta,\Delta W_\psi}(B_j(a)),
\qquad
z_j(a;\lambda_k)=
\begin{cases}
u_j(a), & k=0,\\
u_j(a)+r_\psi(x_{\mathrm{lex},j}(a),e_{\lambda_k}), & k>0.
\end{cases}
$$

Tier 1, $F_\theta$, and $f_\theta$ are frozen during lexicographic RL. $\Delta W_\psi$ contains
private rank-4 deltas on the allowlisted post-encoder semantic maps; $r_\psi$ is the normalized
GELU-16 residual head with a learned finite-menu setting embedding added at its 16-dimensional
pre-activation. Both actor surrogates update $\Delta W_\psi$, $r_\psi$, and the active positive
setting row. Neither path receives count, numeric slack, or dual values as input, and no
lexicographic actor gradient may reach the frozen utility reference. $\lambda_0$ bypasses Tier 3,
so its logits remain bit-identical to utility logits after every update.

### User-facing lambda menu

The deployable checkpoint pins a three-to-five-entry
`ranker-v2-lexicographic-lambda-menu-v1` manifest. The abbreviated schema below shows the two
mandatory entries; deployment adds one to three higher-slack entries:

```yaml
artifact_version: ranker-v2-lexicographic-lambda-menu-v1
scope: deployment
settings:
  - lambda_setting_id: lambda-0
    ordinal: 0
    display_label: utility
    utility_slack: 0.0
    mode: utility-identity
  - lambda_setting_id: lambda-1
    ordinal: 1
    display_label: strict-count
    utility_slack: 0.0
    mode: lexicographic
```

The historical additive `ranker-v2-lambda-menu-v1` manifest is not compatible and must fail
closed; its numeric switch points cannot be reinterpreted as utility slacks.

Additional positive settings use `mode: lexicographic` and strictly increasing explicit slacks:

$$
\tau(\lambda_0)=\tau(\lambda_1)=0,
\qquad
\tau(\lambda_1)<\tau(\lambda_2)<\cdots<\tau(\lambda_{K-1}).
$$

`lambda-0` means pure utility. `lambda-1` means maximize count only where the learned policy can
retain the exact frozen utility reference. Higher settings permit count optimization under larger
user-approved document-level utility budgets. The display labels are product copy, not empirical
privacy claims; reports always include the stable setting ID and exact slack.

The caller chooses a setting by the maximum document-level task-utility loss it is willing to
permit relative to the frozen utility reference. The interface must explain `lambda-0` as exact
utility identity, `lambda-1` as strict count-second with zero slack, and every higher setting by its
explicit pinned slack. It must not translate an ordinal into a privacy percentage or low/medium/high
privacy label before held-out attacker evaluation establishes realized privacy for that checkpoint.

The menu is frozen before training. Its order, size, labels, slacks, and hash are checkpoint state.
Changing any of them creates a new experiment and invalidates previous policy, utility-reference,
and operating-point comparisons. Historical additive lambdas, additive switch points, and the
reader's historical `0.044` resolution are prohibited as slack values unless separately justified
as user utility budgets.

The bounded mechanism run is the only exception to the three-to-five-entry product rule. It uses a
two-entry `scope: mechanism` manifest with `lambda-0` and strict `lambda-1`, is marked
non-deployable, and cannot support a user-facing frontier claim. A promoted checkpoint must carry a
new `scope: deployment` menu and be trained/evaluated against all of its entries.

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
levels whose admitted level log counts are all zero is flat and fails the validated count-health
gate. Equal profile-relative scores across different profiles do not claim equal
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

Count is an ordinary secondary reward for the served lexicographic policy, not a detached
controller target.

The current exact-count artifact is an experimental training shortcut. The intended deployment
reward source is a separately trained, frozen k-anonymity estimator behind the same count-score
interface. Replacing the shortcut changes the reward pin and requires retraining; it does not change
the lambda menu or make count a policy input.

At inference, neither exact counts nor estimator outputs are required by the actor. The user setting
selects a learned conditioned policy behavior; it does not request a fresh count lookup or alter the
reward function online.
For rollout $g$, define per-step reward and return-to-go:

$$
r^P_{g,j}=\frac{p_j(a_{g,j})}{|D_d|},
$$

$$
G^P_{g,j}=\sum_{k=j}^{|D_d|}r^P_{g,k}.
$$

The count critic supplies a baseline only:

$$
\widehat A^P_{g,j}=G^P_{g,j}-V_P(s_{g,j}).
$$

Exact counts therefore produce no reward-model noise, while sampled trajectories preserve the
effect of early actions on later injectivity masks. Count advantages are detached and enter the
lexicographic actor surrogate. They update the Tier-3 semantic adapters and residual head but not
Tier 1, Tier 2, the utility critic, or the document dual. Tier-3 selected-action memory projections
are actor parameters; canonical selected-action records and frozen Tier-2 memory projections are
not.

## Utility assertions and structured credit

The complete round trip produces an assertion vector rather than only a scalar:

```text
u_g = {assertion_id: score}
U(g, Q) = sum_{q in Q} w_q u_g[q] / Z_d
Z_d = utility_weight_denominator stored for document d
```

`w_q` is the assertion weight stored by the QA artifact. `Z_d` is fixed even when a measurement
family is missing; a subset is never renormalized by its own weight sum. `U(g, empty) = 0`.

Under the live scorer (`qa-utility-runtime-v2`), every assertion carries a reward role: it is
**policy** exactly when its dependency set intersects the document's policy decisions and its
contract kind is not a gold-exactness kind (currently `exact_relation`); otherwise it is
**monitoring**. `Z_d` is the weight mass of policy-role assertions only. Monitoring assertions —
document-wide demographic probes, exact-relation contracts, and anything the ranker's actions
cannot causally affect — are scored and reported alongside the reward but contribute no
denominator mass, no credit, and no gradient. This keeps the utility signal undiluted by
outcomes the policy cannot move.

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

Within one document/mode group, compute leave-one-out advantages independently per span:

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
q_pair[g,j] = pi(a_gj | h_gj, mode_d) /
              [pi(a_gj | h_gj, mode_d) + pi(b_j | h_gj, mode_d)]

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
3. high served-policy entropy;
4. unseen adjacent action pairs;
5. oldest measured pair, preventing priority-tier starvation.

Within one priority tier, sample uniformly and deterministically from the run seed. Priority
controls measurement allocation only; it never multiplies `delta_U` or the loss. Cache keys
include the complete reward pin and action vector, so repeated pairs reuse prior round trips.

## Hybrid ranker loss

Freeze an old served-policy snapshot for every actor update and define
$\rho_{g,j}=\pi_\psi(a_{g,j}\mid s_{g,j})/
\pi_{\mathrm{old}}(a_{g,j}\mid s_{g,j})$. For an untested rollout-decision pair, use the clipped
utility term

$$
\ell^U_{g,j}=-\min\left(
\rho_{g,j}A^U_{g,j},
\operatorname{clip}(\rho_{g,j},1-\epsilon_{\mathrm{PPO}},1+\epsilon_{\mathrm{PPO}})A^U_{g,j}
\right).
$$

For a counterfactually tested pair, substitute the bounded contextual pair loss already defined in
this specification. The counterfactual is never added as a separately averaged loss. Therefore

$$
L_U(d)=\frac{1}{G}\sum_g\sum_j\ell^U_{g,j}.
$$

Use the same old-policy ratio and clipping rule with exact count return-to-go advantage
$A^P_{g,j}$:

$$
L_P(d)=-\frac{1}{G}\sum_g\sum_j\min\left(
\rho_{g,j}A^P_{g,j},
\operatorname{clip}(\rho_{g,j},1-\epsilon_{\mathrm{PPO}},1+\epsilon_{\mathrm{PPO}})A^P_{g,j}
\right).
$$

Advantages are baseline-centered but never batch-standardized. Utility remains decision-summed;
count is decision-averaged through its reward definition. For one positive setting $\lambda_k$
held fixed over the document group, the Tier-3 actor loss is

$$
L_{\mathrm{actor}}(d,k)=L_P(d,k)+\operatorname{stopgrad}(\mu_{d,k})L_U(d,k)
-\beta H(\pi_\psi)+\eta\operatorname{KL}(\pi_\psi\Vert\pi_{\mathrm{old}}).
$$

Let $b_d$ be the pinned expected utility of the frozen utility-only policy and
$\widehat J_U(d,k)$ the served lexicographic policy's group-mean document utility at $\lambda_k$.
Define

$$
v_{d,k}=b_d-\tau(\lambda_k)-\widehat J_U(d,k),
$$

$$
L_{\mathrm{dual}}(d,k)=-\mu_{d,k}\operatorname{stopgrad}(v_{d,k}).
$$

After the dual optimizer step, project $\mu_{d,k}$ onto $[0,\infty)$. Positive violation raises the
utility weight; slack lowers it. The dual loss cannot update actor or critic parameters. Actor,
utility critic, count critic, and dual use separate optimizer parameter groups and separately
reported gradient norms.

Neither $L_P$ nor its exact count return is multiplied by $\lambda_k$ or
$\tau(\lambda_k)$. Lambda changes the actor through its setting embedding and changes permitted
utility loss through the document-setting constraint. This keeps count as a reward signal rather
than a deployed policy input or a disguised weighted-sum coefficient. $\lambda_0$ bypasses this
actor/dual update and executes the immutable utility branch.

Critic losses are Huber regressions on routed utility returns and exact count return-to-go. The
utility critic may reduce variance, but existing linked/residual RLOO and counterfactual
substitution remain authoritative in the first mechanism experiment so critic error cannot silently
change utility credit semantics.

Counterfactual pairs with `delta_U = 0` remain in diagnostics and supply no utility gradient. Their
count advantage remains active, which is the intended lexicographic tie behavior. KL is a
previous-policy trust region, not a BC/reference anchor, and must be applied identically to utility
and lexicographic rollouts. No fixed-reference KL, tie hinge, gain penalty, profile-sensitivity
regularizer, additive `alpha`, or utility-logit softcap participates in this objective.

## Training sequence

### Frozen environment and reward artifacts

Before optimization:

- freeze one detector span artifact shared by QA and RL;
- freeze action menus, own-profile counts, count provenance, and normalization tags;
- freeze accepted utility assertions and routing fields;
- freeze remote model, task prompt, extractor, reader, scorer, concurrency regime, and caches;
- run reward-support, count-health, utility-reference, and determinism gates.

Changing any item invalidates cached utility, utility references, and trained-policy
comparisons together.

### Complete-count pre-training gate

This gate runs before utility-reference construction, ExIt reuse, or lexicographic
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
initialization. It is preference-independent and trains only the future utility base.

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

Count score does not affect ExIt selection. Complete winner cloning may include
passenger actions; this is accepted only as initialization. The hybrid stage supplies the
structured and causal correction.

ExIt samples and verifies trajectories through the utility branch before Tier 3 exists. Clone each
verified winner once into the utility feature stack, base actor, and selected-action memory. Continue with utility-only
structured RL until the frozen selection rule chooses the utility-base checkpoint. Then freeze all
Tier-1 and Tier-2 parameters before creating the private Tier-3 adapters, residual head, and
critics. Verify the frozen utility-reference hash and exact Tier-3 initialization parity before
continuing.

### Utility reference and critic warm start

For every training document, replay the frozen utility-only checkpoint under a pinned set of seeds
and cache-only action vectors. Store the exact vectors, utility keys, component scores, rollout
seeds, mean expected utility $b_d$, estimator spread, and every environment/reward hash in the
utility-reference manifest. A missing cache entry blocks a cache-only mechanism run.

Fit utility and count critics from cached trajectories with actor parameters frozen. The utility
critic target follows the same linked/residual/fallback routing as actor credit; the count critic
target is exact count return-to-go. Report document-held-out error against train-mean baselines.
Critic failure cannot be hidden by actor performance and does not authorize shared trainable trunks.

### Lexicographic optimization

Each positive-setting document update samples served lexicographic trajectories, scores structured
utility and exact count, replays old-policy probabilities under the same $\lambda_k$, updates
lambda-conditioned critics, updates the Tier-3 adapters, residual head, and active setting row, then
updates $\mu_{d,k}$. Frozen utility-reference replay is paired at every
synchronous snapshot for audit but never updates Tier 1 or Tier 2. Critic, actor, and dual
optimizers use fastest, middle, and slowest timescales respectively. The first mechanism run uses
`lambda-0` as the immutable control and strict `lambda-1` with $\tau(\lambda_1)=0$ as the only
trainable setting.

Once a nonzero-slack menu is authorized, training uses one lambda setting per document rollout
group. A deterministic balanced Latin cycle assigns every training document to every positive
setting exactly once per cycle, rotating the setting order by document and seeded cycle. All
rollouts, counterfactuals, old-policy probabilities, critics, and dual updates within one group use
that same setting. $\lambda_0$ is evaluated each cycle for exact identity but supplies no Tier-3
gradient. Sampling settings independently per decision or changing settings inside a trajectory is
forbidden.

The four cache-rich campaign documents form the first bounded mechanism run. One seed runs first.
It compares the faithfully reconstructed additive controller against the lexicographic residual
actor only to determine whether the new gradient architecture produces stable count movement under
exact document utility retention. Passing authorizes two more seeds on the same documents, not a
full-corpus or deployment run.

## Historical additive lambda-menu selection (superseded 2026-08-05)

The following section preserves the previous additive-controller calibration protocol for
reproducing historical checkpoints only. It is non-normative for lexicographic training. Future
multi-setting support uses the explicit utility-slack menu defined above and cannot reuse these
switch-point lambdas.

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

- The complete-count pre-training gate passes at 100% level coverage before lexicographic training.
- Report own-profile denominators, provenance, singleton-profile rate, flat-menu rate, and
  adjacent exact-target `delta p` distributions per type and provenance.
- Report expected count-gradient mass by type, profile, and provenance.
- Assert that fallback/default provenance contributes exactly zero lattice-level gradient mass.
- Count score monotonicity follows the stored counts; any lattice/count non-monotonicity blocks
  the run rather than being silently repaired or excluded at reward time.
- Report selected count return, return-to-go critic error, and residual-actor count-gradient mass
  by document, runtime type, provenance, action mode, and decision depth.

### Lexicographic-policy responsiveness

- Frozen utility-reference logits remain bit-identical to the selected utility checkpoint after
  every update.
- Tier-3 semantic deltas and residual logits are exactly zero at initialization; they become
  non-uniform only through utility/count actor gradients.
- Actor gradient tests prove staged reachability: residual head first, adapter output factors after
  the residual map moves, and adapter input factors after their output factors move, with exact zero
  gradients in Tier 1 and Tier 2 throughout.
- Greedy exact `P_count`, complete-document utility keys, and action-mode rates are reported for
  the frozen reference and served lexicographic policy by document and type.
- Every document reports utility violation, dual value, actor-gradient norms by objective, and
  final-three-snapshot count separation.
- A temporary positive separation that decays to zero fails; sampled softness cannot substitute
  for greedy-path behavior.
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

Evaluate the served lexicographic checkpoint and its pinned frozen utility reference independently:

- measure utility on `out_final` with task metrics and whole-task regression metrics;
- measure re-identification/inference attack success on `doc_p` and leak-through on `out_final`;
- report count score, action modes, per-type behavior, assertion coverage, and count provenance as
  diagnostics;
- compare against rule baselines and external methods only at matched realized privacy and
  identical settings.

The internal utility-reference path must be bit-identical to its frozen source checkpoint; a
non-inferiority test is insufficient for this architectural invariant. This does not make it a
second product mode. The served lexicographic policy must retain each document's exact utility key
in the mechanism gate. Later held-out evaluation may use a
pre-registered paired non-inferiority test only after this strict mechanism succeeds.

Count-score improvement is not privacy improvement. Any promoted method is compared with baselines
only at matched realized attacker privacy. A count/attacker inversion terminates claims based on
count shaping and is never calibrated away.

## Failure modes and stop conditions

- **QA coverage capture:** policy movement concentrates on decisions with linked assertions while
  uncovered decisions remain static. Check complete-document fallback credit and scheduler
  allocation; do not drop the uncovered decisions.
- **Passenger persistence:** ExIt-cloned actions survive despite negative adjacent
  counterfactual utility. Price with measured pairwise disagreement and increase only the
  pre-registered counterfactual budget in a new run.
- **Count coverage regression:** a profile rebuild introduces a missing, fallback, or
  non-monotone level count. Invalidate lexicographic training and block the run until the artifact
  passes again.
- **Constraint failure:** the served policy improves count while any document falls below its
  utility reference. Reject the run; do not increase a global count coefficient.
- **No secondary movement:** utility constraints hold but Tier 3 remains inert or count separation
  decays after a mid-run peak. Inspect count advantages, adapter/head gradients, and optimizer
  timescales; do not revive additive authority controls.
- **Uniform or inert dual dynamics:** violations do not increase the corresponding document dual,
  or all documents share an accidental scalar. Fail the mechanism check.
- **Critic corruption:** critic loss updates the actor/base or count targets alter the utility
  critic. Fail closed on the gradient-isolation tests.
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

- **Frozen substrate interface:** `semantic_substrate(state, legal) -> FrozenSemanticSubstrate`
  owns token/relation banks, occurrence masks, frozen action metadata, and canonical prior
  selected-action records; it has no count dependency.
- **Utility-stack interface:** `utility_action_stack(substrate) -> x_U, utility_logits` owns the
  Tier-2 candidate-conditioned context and selected-action memory and is frozen during
  lexicographic RL.
- **Lexicographic actor interface:** `lexicographic_distribution(substrate, utility_logits,
  lambda_setting_id) -> ActionDistribution` validates the frozen menu entry and owns the private
  Tier-3 rank-4 adapters, setting-conditioned residual head, legal gather, and served policy.
- **Objective interface:** consumes the active positive setting, utility/count advantages,
  same-setting old-policy probabilities, entropy, KL, utility reference, configured slack, and
  document-setting dual; returns separately auditable actor, critic, and dual losses.
- **Exact count-target interface:** versioned own-profile counts plus
  `action_targets(decision, legal) -> scores, provenance` hide profile normalization and
  diagnostics; targets never enter actor features.
- **Utility credit interface:** `credit(assertion_vector, occurrence_to_decision,
  policy_dependency_decision_ids, credit_routing, rollout_group)` returns per-policy-decision
  provisional advantages without knowing policy internals.
- **Counterfactual scheduler interface:** consumes linked-assertion coverage, hyperedge status,
  entropy, pair history, and a fixed budget; returns intervention requests. It never sees or
  scales rewards.
- **Utility-reference builder:** consumes the frozen utility policy and pinned rollout seeds and
  emits per-document reference returns, exact utility keys, vectors, and hashes.

Expected code destinations are focused model modules under `src/cloak/ranker/` for the frozen
substrate, utility stack, Tier-3 adapters and residual head, objective critics, and primal-dual losses;
`src/cloak/ranker/interactive.py` for trajectory, credit, and counterfactual assembly;
`src/cloak/reward/roundtrip.py` for assertion-vector output; and
`scripts/train_interactive_ranker.py` for orchestration only. Exact paths are fixed by the
implementation plan, but these responsibility boundaries are normative.

## Artifacts

- frozen detector/span artifact shared by QA and RL;
- versioned utility-assertion artifact with routing fields and fixed per-document denominators;
- own-profile count-target artifact and count-health report;
- frozen document/relation encoder identity and content-addressed token/relation caches;
- frozen utility-base checkpoint and per-document utility-reference manifest;
- frozen finite lambda-menu manifest and hash;
- Tier-3 adapter target manifest, zero-init setting-conditioned residual-head schema, critic
  checkpoint, and document-setting dual-state audit;
- counterfactual pair cache and scheduler report;
- lexicographic-policy training record under `research-wiki/training/` written before the run;
- frozen-reference parity plus served-policy held-out utility and attacker results.

## Sources and predecessor specifications

- [Round-trip ranker and infiller](roundtrip-ranker-infiller.md) — pinned remote reward,
  determinism, ExIt/RLOO history, and infiller stages.
- [Training-task environment](training-task-env.md) — historical ladder, decision, schema, and
  carrier inputs; not the live ranker-v2 reward contract.
- [Leakage-probe reward options](leakage-probe-reward.md) — count, span-recovery, and
  profile-matching alternatives and honesty boundaries.
- [Interactive ranker v2 decision log](interactive-ranker-v2-decision-log.md) — chosen forks and
  rejected alternatives.
- [Ranker v2 architecture](ranker-v2-architecture.md) — normative three-tier model inputs, frozen
  utility branch, private lexicographic semantic side path, objective critics, and dual state.
- [Lexicographic actor-critic implementation plan](../../plans/2026-08-05-ranker-v2-lexicographic-actor-critic.md)
  — implementation order, interfaces, gradient tests, and bounded real-data gate.
- [Ranker v2 architecture decision log](ranker-v2-architecture-decision-log.md) — architecture
  alternatives and escalation evidence.
- [QA builder v2](../qa-builder-v2.md) — implemented shared assertion artifact and scoring
  contract.
- [Trainable multi-objective RL review](../../research/ranker-v2-trainable-multi-objective-rl-review.md)
  — issue classification, literature comparison, and selected policy-based lexicographic design.
- [Lexicographic Multi-Objective Reinforcement Learning](../../../research-wiki/papers/skalse2022_lexicographic_morl.md)
  ([arXiv 2212.13769](https://arxiv.org/abs/2212.13769)).
- [Reward Constrained Policy Optimization](../../../research-wiki/papers/tessler2018_reward_constrained_policy.md)
  ([arXiv 1805.11074](https://arxiv.org/abs/1805.11074)).
- [Constrained Policy Optimization](../../../research-wiki/papers/achiam2017_constrained_policy_optimization.md)
  ([arXiv 1705.10528](https://arxiv.org/abs/1705.10528)).

## Historical objective normalization (superseded 2026-08-05)

The hybrid group objective keeps its mixed normalization: utility, entropy, and KL
decision-summed; the count term decision-averaged (×λ/decisions/rollouts). A
preregistered spike (decision log: "objective normalization mix") found the proposed
alternatives empirically unidentifiable and behaviorally inert at reachable scales;
the practically binding controller question is strength/initialization of α, tracked
as its own fork. The decision-averaged alpha-routing variant remains available as
default-off infrastructure (`train --alpha-utility-routing per-decision`).

## Historical additive implementation status (superseded 2026-08-05)

Enumerated deviations of the live implementation from the normative text above. Each is either
an interim measure with a named exit condition or a candidate pending its preregistered gate;
nothing here silently amends the normative sections.

- **Interim privacy signal: direct grounded counts, not the semantic privacy head.** The learned
  head failed its transfer gates (decision basis in the architecture decision log); production
  runs use `DirectCountPrivacyProvider` — the frozen own-profile exact targets injected as the
  controller's privacy score, provenance-tagged `direct-count`. This is a temporary measure while
  a proper k-anonymity estimator model is trained; the privacy-head path (`--privacy-checkpoint`)
  remains first-class in the trainer and the normative controller text is unchanged. Documentation
  figures deliberately keep the privacy head as the depicted component.
- **Controller strength (candidate, gate open).** Switch-calibrated alpha initialization plus
  gap-scaled controller (decision log: "controller strength" fork) revives lambda responsiveness
  on every spike seed and is the adopted candidate, wired default-off into the trainer
  (`train --controller-gap-scaling utility-gap --alpha-init switch-calibrated`). Gap scaling
  retags `controller_transform`, so architecture pins diverge exactly when controller semantics
  do, and the KL reference is captured after calibration so KL anchors to the calibrated init.
  Final adoption is gated on re-running the frontier-regret criterion under the production
  trainer (counterfactual credit + KL enabled).
- **Initial-RL controlled-type scope.** The first production runs control only `drug`,
  `health-condition`, and `medical-procedure`. Out-of-scope or count-uncovered policy decisions
  are demoted at load time to fixed KEEP (no action, no gradient, no count contribution);
  PERSON/CODE remain rule-substituted placeholders and are tracked as monitoring assertions only.
  Full-schema control re-enters per type once its count coverage and utility probes pass the
  existing gates.
- **Zero-signal document filter.** Documents whose policy assertions provide no reward mass under
  the policy-role denominator are dropped from training at load time and logged; they remain in
  evaluation corpora.
- **Pinned remote and reader models.** Round-trip remote model and QA reader are both
  `medgemma-4b-it` (single reader definition in the QA scorer); the utility-cache identity embeds
  the extractor and runtime-type source hashes, so any change to either invalidates cached
  utilities rather than silently reusing them.
