---
type: research
status: current
created: 2026-08-05
updated: 2026-08-05
tags: [rl, ranker-v2, lexicographic, constrained-rl, morl, policy-gradient, architecture]
companion:
  - docs/specs/RL/ties-by-design.md
  - docs/research/tie-ownership-root-cause-and-solution-space.md
  - docs/research/ranker-v2-loss-function-audit.md
---

# Trainable multi-objective RL for ranker v2

## Executive conclusion

The deterministic epsilon-zero selector is not an acceptable deployment architecture, and the cache-only gate did not establish enough standardized-slate support to justify one. That rejection does **not** invalidate the lexicographic classification. The product preference that motivated the work — maximize task utility first, then use count score to distinguish utility-equivalent policies — is a standard lexicographic multi-objective RL (LMORL) objective. The literature includes trainable value-based and policy-gradient algorithms for this objective; a hard inference selector is only one possible implementation and is not required.

The best-fit next architecture is a **policy-based lexicographic actor–critic**, adapted from Skalse et al.'s policy-based LRL/LPPO family: one learned actor, separate utility and count critics, and multi-timescale constrained actor updates. Utility is optimized first. Count then trains the **same actor** subject to preserving the achieved utility return. Count never trains the utility critic, but it must train the actor directly; the current global-`alpha` bottleneck is removed. This is the principled gradient-path unification that ranker v2 has not tested.

This recommendation is narrower than “lexicographic is solved.” The literature's strongest guarantees are for expected returns under convergence, adequate exploration, separated timescales, and feasible represented policies. Ranker v2 requires document-conditional behavior, has a quantized expensive utility reward, and currently reports only about 59% pairwise agreement between actor logits and measured utility. A four-document cached training run can validate the optimization mechanism, not held-out generalization or realized privacy.

## Definitions

- **Actor.** The parameterized policy that selects ranker actions. Its logits are policy parameters, not utility estimates.
- **Critic.** A learned estimator of expected return for one objective, used to reduce policy-gradient variance or construct advantages.
- **Strict lexicographic objective.** Optimize the primary objective; among policies that are primary-optimal, optimize the secondary objective.
- **Slack lexicographic objective.** Optimize the secondary objective subject to the primary objective remaining within a declared loss budget of its optimum.
- **CMDP.** Constrained Markov decision process: maximize one expected return subject to constraints on other expected returns or costs.
- **MORL.** Multi-objective reinforcement learning: preserve the reward vector and choose a solution concept describing how objectives are ordered or traded.
- **Dual variable.** A learned Lagrange multiplier driven by measured constraint violation, representing the current shadow price of the constraint.
- **Count score.** The frozen profile-relative count shaping target. It is not a measured privacy outcome.
- **Realized privacy.** Re-identification or inference-attack performance measured on produced artifacts; the only privacy quantity that supports product comparison.

## 1. Formal issue classification

### 1.1 The solution concept is lexicographic, not additive

For document policy $\pi$, let $J_U(\pi)$ be expected task utility and $J_P(\pi)$ the expected frozen count score. The strict stated preference is

$$
\Pi_U^* = \arg\max_{\pi} J_U(\pi),
\qquad
\pi^* \in \arg\max_{\pi\in\Pi_U^*} J_P(\pi).
$$

This is exactly the lexicographic MORL solution concept formalized by [Skalse et al.](../../research-wiki/papers/skalse2022_lexicographic_morl.md) ([arXiv 2212.13769](https://arxiv.org/abs/2212.13769)). The broader MORL literature warns that the scalarization function is a model of user preference, not an implementation convenience; weighted sums are appropriate only when the objectives are genuinely commensurable and the user's preference is linear ([Hayes et al.](../../research-wiki/papers/hayes2022_practical_guide_morl.md), [arXiv 2103.09568](https://arxiv.org/abs/2103.09568), [DOI 10.1007/s10458-022-09552-y](https://doi.org/10.1007/s10458-022-09552-y)).

The current controller implements $u(a)+\alpha g(\lambda)\hat p(a)$, which is linear scalarization in policy-logit space. Repeated `alpha`, gain, softcap, and gap-scaling failures are therefore not merely poor tuning; they are evidence that the chosen solution family does not reliably implement the declared priority order. Conditional/slack lexicographic MDPs can also contain optimal policies unreachable by any fixed linear weighting ([Wray et al.](../../research-wiki/papers/wray2015_lexicographic_mdp_slack.md), [DOI 10.1609/aaai.v29i1.9647](https://doi.org/10.1609/aaai.v29i1.9647)).

### 1.2 The current actor is underspecified on utility ties

When two trajectories have equal utility, the utility policy gradient supplies no ordering force. Many actor parameterizations therefore realize the same primary return while assigning arbitrary tied margins. Shared updates from other documents can move those margins and flip greedy actions without changing the measured objective. This is reward underspecification plus policy churn, not missing counterfactual credit: counterfactuals correctly measure zero utility difference and therefore should not invent a utility order. The observed behavior matches the broad phenomena documented by [D'Amour et al.](../../research-wiki/papers/damour2020_underspecification_ml.md) ([arXiv 2011.03395](https://arxiv.org/abs/2011.03395)) and [Schaul et al.](../../research-wiki/papers/schaul2022_policy_churn.md) ([arXiv 2206.00730](https://arxiv.org/abs/2206.00730)).

Lexicographic training resolves this structurally: the secondary objective supplies an actor gradient exactly where preserving the primary objective permits it. It does not ask the primary utility head to order ties.

### 1.3 Actor logits and cardinal utility were conflated

Policy-gradient training optimizes action probabilities; it does not regress logits into utility units. Reward/preference data also leave important transformations unidentified unless the learning signal anchors cardinal scale ([Skalse et al.](../../research-wiki/papers/skalse2023_partial_identifiability_reward.md), [arXiv 2203.07475](https://arxiv.org/abs/2203.07475)). The measured 41% live-pair sign disagreement shows an additional ordering problem: the current actor logits are not a dependable utility ranker.

This defect made additive control especially fragile because `alpha * count` was compared against arbitrary actor margins. Policy-based lexicographic optimization does not require that comparison. It combines objective **return gradients** through an explicit constraint. Separate critics are still required for variance reduction and diagnostics, but the actor's raw logit scale no longer defines the utility/count exchange rate.

### 1.4 Gradient isolation became gradient starvation

The current design protects utility semantics by routing utility, global controller strength, and per-decision gain through mostly separate optimization paths. The protection goal was valid: count must not change a quantity labelled “utility.” The implementation made the wrong object pure. The utility **critic** must remain utility-only; the **actor** must receive every objective that should influence behavior.

Constrained and lexicographic policy-gradient methods use this division explicitly. The actor receives the secondary-objective gradient plus primary-constraint gradients; objective critics remain separate. Multi-timescale primal–dual updates are standard in policy-based LRL and constrained RL ([Skalse et al.](../../research-wiki/papers/skalse2022_lexicographic_morl.md), [arXiv 2212.13769](https://arxiv.org/abs/2212.13769); [Tessler et al.](../../research-wiki/papers/tessler2018_reward_constrained_policy.md), [arXiv 1805.11074](https://arxiv.org/abs/1805.11074)).

### 1.5 The product constraint is more local than a standard CMDP constraint

Standard CMDP methods constrain an expected cumulative cost. A corpus-average utility constraint can sacrifice one document and compensate on another, which is unacceptable for a per-document ranker. This is a conditional/statewise constraint problem. State-augmented constrained RL shows that fixed global scalarization can be insufficient and incorporates multipliers into policy state ([Calvo-Fullana et al.](../../research-wiki/papers/calvofullana2021_state_augmented_constrained_rl.md), [arXiv 2102.11941](https://arxiv.org/abs/2102.11941)).

That literature supports document-conditioned dual information, but it does not directly prove that a multiplier network trained on 63 documents will generalize. The first experiment must therefore report both corpus-average and worst-document constraint violations; a full design must use a state-conditioned multiplier or an equivalent group-robust constraint rather than one global scalar.

### 1.6 Count remains a proxy objective

Every method in this report optimizes `profile_score`, not privacy. A successful count-aware policy only establishes controllability of the shaping objective. It must later be compared at matched realized privacy using the registered attacker. No MORL algorithm repairs a weak privacy proxy.

## 2. Literature-backed lexicographic solutions

### 2.1 Policy-based lexicographic RL — recommended

[Skalse et al.](../../research-wiki/papers/skalse2022_lexicographic_morl.md) ([arXiv 2212.13769](https://arxiv.org/abs/2212.13769)) give a general policy-based LRL scheme and A2C/PPO instantiations. First optimize objective $K_U$. Once its achieved value $k_U$ is established, optimize $K_P$ subject to

$$
K_U(\theta) \ge k_U - \tau_t,
$$

using a Lagrangian and separated learning timescales. For two objectives, the actor update has the form

$$
\theta \leftarrow \theta + \beta_t
\left(\widehat{\nabla J_P} + \mu\widehat{\nabla J_U}\right),
$$

and the dual update is

$$
\mu \leftarrow
\left[
\mu + \eta_t\left(k_U-\tau_t-\widehat{J_U}\right)
\right]_+.
$$

`tau_t` is an optimization continuation parameter that decays toward zero to maintain a feasible interior during learning; it is not the old `0.044` measurement floor and not a product utility budget.

**Architectural translation for ranker v2:**

```text
document + decision + lambda
            |
       shared encoder
            |
      lambda-conditioned actor ----------------------> action distribution
            |
      +-----+----------------+
      |                      |
utility critic          count critic / exact counts
(utility targets only)  (count targets only)
      |                      |
 utility advantage      count advantage
      +---------- lexicographic primal-dual actor update
```

- `lambda = 0`: optimize utility only.
- strict positive setting: optimize count subject to retaining the achieved utility optimum.
- count gradients enter the actor directly, never the utility critic.
- the global `alpha`, learned gain field, tie hinge, cycle projection, gap scaling, and utility-logit softcap are unnecessary in this architecture.

**Advantages:** direct fit to the declared preference; no deterministic selector; no comparison between count scores and arbitrary logits; supports ordinary neural actors and PPO/A2C-style objectives; convergence result exists for the abstract algorithm.

**Tradeoffs:** needs objective critics or reliable objective advantages; relies on convergence and timescale separation; strict equality is difficult under function approximation, so training uses decaying tolerance while evaluation remains exact; standard guarantees are expected-return guarantees; exact lexicographic semantics provide only two regimes unless positive settings are assigned explicit primary slack budgets.

### 2.2 Lexicographic gradient projection — viable fallback, not first choice

[Tercan and Prabhu](../../research-wiki/papers/tercan2024_thresholded_lexicographic.md) ([arXiv 2408.13493](https://arxiv.org/abs/2408.13493)) propose projecting a lower-priority policy gradient into a cone that preserves higher-priority objectives. [Qiu et al.](../../research-wiki/papers/qiu2025_lppgrl_lexicographically_projected.md) ([arXiv 2511.08339](https://arxiv.org/abs/2511.08339)) develop a newer projected policy-gradient framework with Dykstra projection and subproblem exploration.

**Advantages:** one trainable actor; explicit priority order; no hard inference selector; lower-priority gradients can act directly on utility ties.

**Tradeoffs:** preservation is local in gradient geometry, not a measured document-return guarantee; noisy primary gradients can project in the wrong direction; zero primary gradients leave no local boundary normal; the strongest recent evidence is still concentrated in small continuous-control tasks. This family is useful if dual oscillation is the binding failure, but ranker v2's delayed quantized utility and weak actor ordering make it a riskier first transfer.

### 2.3 Value-based lexicographic RL — rejected for deployment

Lexicographic Q-learning filters actions through successively lower-priority learned Q-functions. It has established roots and convergence results in restricted settings, but it reintroduces an explicit action-set filter and depends on accurate action-value functions. Thresholded variants also have Bellman and non-Markovianity failure modes under common formulations ([Tercan and Prabhu](../../research-wiki/papers/tercan2024_thresholded_lexicographic.md), [arXiv 2408.13493](https://arxiv.org/abs/2408.13493)). It conflicts with the decision to reject deterministic selector-style deployment and is a poor fit for the current utility-model evidence.

## 3. Alternative literature-backed solutions

### 3.1 Constrained policy optimization

Reframe the task as

$$
\max_\pi J_P(\pi)
\quad\text{subject to}\quad
J_U(\pi) \ge b_U,
$$

where $b_U$ is the utility-only policy's achieved return, optionally minus a declared slack. CPO performs trust-region policy updates with approximate per-iteration constraint guarantees ([Achiam et al.](../../research-wiki/papers/achiam2017_constrained_policy_optimization.md), [arXiv 1705.10528](https://arxiv.org/abs/1705.10528)); RCPO uses multi-timescale primal–dual learning ([Tessler et al.](../../research-wiki/papers/tessler2018_reward_constrained_policy.md), [arXiv 1805.11074](https://arxiv.org/abs/1805.11074)). Under suitable policy spaces, constrained RL can have zero duality gap ([Paternain et al.](../../research-wiki/papers/paternain2019_zero_duality_gap.md), [arXiv 1910.13393](https://arxiv.org/abs/1910.13393)).

This becomes equivalent to strict lexicographic optimization only if $b_U$ is the true maximum attainable utility and slack is zero. With a learned baseline it is baseline-constrained privacy optimization, not formally exact lexicographic optimization.

**Advantages:** mature constrained-RL formulation; measured utility has a direct constraint role; CPO supplies bounded-update machinery; easier to expose nonzero utility budgets later.

**Tradeoffs:** corpus-average constraints can hide document losses; exact-zero slack is brittle under noisy estimates; CPO's second-order update is expensive; learned duals can oscillate, overshoot, or diverge; a suboptimal baseline changes the solution concept.

### 3.2 Scale-invariant multi-objective policy optimization

MO-MPO learns one improved action distribution per objective and distills their combination into one policy; preferences are specified by per-objective KL budgets rather than raw reward weights ([Abdolmaleki et al.](../../research-wiki/papers/abdolmaleki2020_distributional_view_multiobjective.md), [arXiv 2005.07513](https://arxiv.org/abs/2005.07513)). LP3 separates learning the policy for a preference from learning which preferences satisfy constraints ([Huang et al.](../../research-wiki/papers/huang2022_constrained_multiobjective_reinforcement.md), [PMLR paper](https://proceedings.mlr.press/v164/huang22a.html)).

**Advantages:** explicitly addresses cross-objective scale; supports one preference-conditioned policy and multiple operating points; avoids one global learned scalar dominating all decisions.

**Tradeoffs:** solves a tradeoff/Pareto problem rather than strict utility priority; KL budgets are still operational choices; substantially more policy-improvement machinery; no per-document utility guarantee; preference-conditioned policies can be behaviorally insensitive despite good aggregate MORL metrics.

This is the best alternative if the product requirement is revised from “utility first” to “smooth controllable Pareto frontier.” It is not the first choice under the current requirement.

### 3.3 Ordinary preference-conditioned scalarized policy

Condition one actor on a finite lambda menu and train it across weighted utility/count rewards. This is the cheapest architecture and already resembles ranker v2. It does not repair the measured problem: scalarization still assumes comparable scales, tied behavior still races actor margins, and conditioning does not guarantee controllability. It should remain a baseline, not the next design.

### 3.4 Anchored utility critic plus derived policy

Train utility in measured units using a ranking-plus-cardinal objective, then derive policy updates from the critic. This directly targets the 41% utility-ordering error and is compatible with every constrained or lexicographic method above. It is not a complete multi-objective solution: a calibrated utility critic still needs a correct policy-composition rule.

The earlier four-document gate showed that cardinal anchoring improved scale but reduced ordering. Therefore a future critic must compose ranking and value learning rather than replace ranking with regression. This work becomes mandatory if policy-based LRL fails because utility advantages or constraints are inaccurate; it should not be used to revive an additive `alpha` race.

### 3.5 Maximum entropy, KL anchoring, and generic gradient surgery

Entropy floors and KL trust regions can prevent premature collapse; generic multi-task gradient surgery can reduce destructive interference. None specifies the required priority order or guarantees utility retention. They are optimization aids inside a correctly formulated algorithm, not solution concepts. The previous uniform-KL experiment already showed that an SFT/BC reference can preserve the wrong operating point.

## 4. Comparison

| Family | Trainable policy | Strict utility priority | Avoids raw-logit scale race | Multiple settings | Per-document guarantee | Fit now |
|---|---|---|---|---|---|---|
| Policy-based LRL / LPPO | yes | yes, asymptotically under assumptions | yes | only through declared slack menu | no, not without conditional constraints | **best** |
| CPO / RCPO | yes | only if baseline is utility-optimal and slack zero | yes | yes, via utility budgets | expected constraint by default | strong alternative |
| Projected lexicographic PG | yes | local first-order preservation | yes | thresholds/slacks possible | no | experimental fallback |
| MO-MPO / LP3 | yes | no; learns tradeoffs | yes, uses KL budgets | yes | no | best if product semantics change |
| Preference-conditioned weighted sum | yes | no | no | yes | no | current-style baseline only |
| Lexicographic Q/filter | yes plus hard action filter | yes in restricted settings | yes | thresholds possible | no | rejected deployment form |

## 5. Recommendation

### 5.1 Adopt the solution concept, not the selector

Retain the formal statement “utility is primary; count is secondary.” Retire the deterministic selector as a candidate deployment architecture. Preserve it only as an offline oracle/test utility if useful; no training or deployment decision should depend on enumerating a candidate slate.

### 5.2 Prototype a two-objective policy-based LRL actor–critic

Build one experimental architecture:

1. **Actor:** the existing sequential semantic actor, with lambda conditioning retained. Lambda zero uses the utility branch exactly; positive lambda activates a trainable residual branch.
2. **Utility critic:** estimates utility return/advantage and receives no count gradient.
3. **Count critic:** use exact menu count values where available; otherwise estimate the trajectory count return separately.
4. **Actor update:** utility-only during the primary phase; then count policy gradient plus a utility-retention dual term during the secondary phase.
5. **Constraint mechanism:** a state/document-conditioned dual input or multiplier head driven by utility violation, not by count reward. Start with one strict positive setting and a decaying training tolerance; do not introduce a lambda menu yet.

This deliberately unifies behavioral gradients while preserving semantic separation:

```text
utility reward ------> utility critic only ----+
                                              +--> shared actor update
count reward --------> count critic only ------+
utility violation ---> dual dynamics ----------+
```

The old rule “count must not enter `u`” becomes the precise rule “count must not enter the utility critic.” Count must enter the actor, or the actor cannot learn privacy-sensitive behavior.

### 5.3 First experiment: mechanism validation, not larger training

Use the four cache-rich campaign documents and the existing frozen utility cache. Run one seed first; expand only after the mechanism passes.

**Question:** Can policy-based lexicographic training produce stable count-score separation without reducing each document's measured utility, when both gradients reach the same actor through a constrained update?

**Arms:**

- current additive controller, reconstructed faithfully;
- two-objective policy-based LRL, one utility-only regime and one strict positive regime.

**Frozen:** encoder inputs, environment, reward pin, count targets, candidate menus, rollout schedule, cache, optimizer budget, and initialization.

**Disable in the LRL arm:** additive `alpha`, gain head, tie hinge, cycle projection, utility-logit softcap, gap scaling, and profile-sensitivity regularizer. A previous-policy trust region may remain only if applied identically to both regimes and reported separately from the lexicographic utility constraint.

**Pass conditions:**

1. Count gradient reaches actor parameters and remains exactly zero in the utility critic.
2. Lambda-zero action distributions are exactly unchanged by enabling the positive branch before training.
3. Every final-three-snapshot positive-regime greedy document has utility key no lower than its corresponding lambda-zero document.
4. At least one opportunity-bearing document has stable positive count-score separation across the final three snapshots.
5. The learned constraint multiplier varies across document states or violations; a uniform scalar field fails the mechanism check.
6. No late-cycle decay back to zero separation after an earlier pass.

**Cost:** approximately 2–4 engineering days because this changes the optimization structure, then about 1–2 GPU hours for one cached seed plus its lambda-zero control. It requires zero new remote reward calls if every sampled vector remains in cache; cache misses must stop the run rather than silently change scope.

**Interpretation:**

- **Pass:** repeat three seeds on the same documents, then design a document-held-out run. It authorizes retiring controller-strength work, not claiming generalization.
- **Utility violation:** the constrained update or utility estimator is inadequate. Move to CPO-style trust-region constraint enforcement or the anchored ranking-plus-value critic.
- **No count movement with valid utility constraint:** investigate policy-gradient saturation and count-advantage routing, not `alpha` calibration.
- **Uniform dual field:** state conditioning or its optimization is ineffective; do not scale the run.
- **Pass only on training documents:** buy document breadth before any deployment claim.

### 5.4 Lambda menu comes later and changes meaning

Strict lexicographic optimization yields two semantic regimes: utility-only and utility-first/count-second. A three-to-five-level user menu requires explicit utility slack budgets

$$
\pi_j \in \arg\max_\pi J_P(\pi)
\quad\text{subject to}\quad
J_U(\pi) \ge J_U^* - \delta_j.
$$

The menu values are therefore document-level utility budgets `delta_j`, not additive reward weights. They must be selected and validated after the strict mechanism works. The old `0.044` reader diagnostic is not a budget and must not supply these values.

## 6. What this research changes

1. **Deterministic selector:** rejected as deployment architecture; no richer-slate selector run is the immediate next step.
2. **Lexicographic classification:** retained and strengthened; it is supported by trainable policy-gradient literature, not merely set filtering.
3. **Gradient separation:** revised. Objective critics stay separate; objective gradients meet in one actor under an explicit constraint.
4. **Controller campaign:** further global-`alpha`, gain-bound, softcap, hinge, and gap-scaling experiments are dominated by the policy-based LRL test.
5. **Full-corpus training:** still premature. The next run validates the learning mechanism on the cache-rich documents; held-out generalization is a later gate.

## 7. Sources

- [Skalse et al., Lexicographic Multi-Objective Reinforcement Learning](../../research-wiki/papers/skalse2022_lexicographic_morl.md) ([arXiv 2212.13769](https://arxiv.org/abs/2212.13769))
- [Hayes et al., A Practical Guide to Multi-Objective Reinforcement Learning and Planning](../../research-wiki/papers/hayes2022_practical_guide_morl.md) ([arXiv 2103.09568](https://arxiv.org/abs/2103.09568), [DOI 10.1007/s10458-022-09552-y](https://doi.org/10.1007/s10458-022-09552-y))
- [Wray et al., Multi-Objective MDPs with Conditional Lexicographic Reward Preferences](../../research-wiki/papers/wray2015_lexicographic_mdp_slack.md) ([DOI 10.1609/aaai.v29i1.9647](https://doi.org/10.1609/aaai.v29i1.9647))
- [Tercan and Prabhu, Thresholded Lexicographic Ordered Multiobjective Reinforcement Learning](../../research-wiki/papers/tercan2024_thresholded_lexicographic.md) ([arXiv 2408.13493](https://arxiv.org/abs/2408.13493))
- [Qiu et al., LPPG-RL](../../research-wiki/papers/qiu2025_lppgrl_lexicographically_projected.md) ([arXiv 2511.08339](https://arxiv.org/abs/2511.08339))
- [Achiam et al., Constrained Policy Optimization](../../research-wiki/papers/achiam2017_constrained_policy_optimization.md) ([arXiv 1705.10528](https://arxiv.org/abs/1705.10528))
- [Tessler et al., Reward Constrained Policy Optimization](../../research-wiki/papers/tessler2018_reward_constrained_policy.md) ([arXiv 1805.11074](https://arxiv.org/abs/1805.11074))
- [Paternain et al., Constrained Reinforcement Learning Has Zero Duality Gap](../../research-wiki/papers/paternain2019_zero_duality_gap.md) ([arXiv 1910.13393](https://arxiv.org/abs/1910.13393))
- [Calvo-Fullana et al., State Augmented Constrained Reinforcement Learning](../../research-wiki/papers/calvofullana2021_state_augmented_constrained_rl.md) ([arXiv 2102.11941](https://arxiv.org/abs/2102.11941))
- [Abdolmaleki et al., A Distributional View on Multi-Objective Policy Optimization](../../research-wiki/papers/abdolmaleki2020_distributional_view_multiobjective.md) ([arXiv 2005.07513](https://arxiv.org/abs/2005.07513))
- [Huang et al., A Constrained Multi-Objective Reinforcement Learning Framework](../../research-wiki/papers/huang2022_constrained_multiobjective_reinforcement.md) ([PMLR 164](https://proceedings.mlr.press/v164/huang22a.html))
- [Skalse et al., Invariance in Policy Optimisation and Partial Identifiability in Reward Learning](../../research-wiki/papers/skalse2023_partial_identifiability_reward.md) ([arXiv 2203.07475](https://arxiv.org/abs/2203.07475))
- [Schaul et al., The Phenomenon of Policy Churn](../../research-wiki/papers/schaul2022_policy_churn.md) ([arXiv 2206.00730](https://arxiv.org/abs/2206.00730))
- [D'Amour et al., Underspecification Presents Challenges for Credibility in Modern Machine Learning](../../research-wiki/papers/damour2020_underspecification_ml.md) ([arXiv 2011.03395](https://arxiv.org/abs/2011.03395))
