---
type: research
status: current
created: 2026-08-04
updated: 2026-08-04
tags: [rl, ranker-v2, loss-functions, policy-gradient, cross-entropy, regression,
       counterfactual-credit, softmax-saturation, behavior-cloning, literature]
companion: [docs/specs/RL/interactive-ranker-v2.md,
            docs/specs/RL/ranker-v2-architecture.md,
            docs/specs/RL/ties-by-design.md,
            docs/research/tie-ownership-root-cause-and-solution-space.md]
---

# Loss-function audit for ranker v2: cross-entropy, regression, and policy gradients

## Executive conclusion

Ranker-v2 hybrid RL is **not trained by mean squared error**. Its primary utility objective is a leave-one-out REINFORCE estimator; its privacy/count objective differentiates an exact finite-menu expectation; counterfactual credit differentiates an exact local two-action expectation; and its remaining terms are hinge, entropy, and KL regularizers. MSE appears only in an architecture expressivity spike and in an optional profile-sensitivity regularizer that is disabled by default. Behavior cloning and ExIt do use categorical cross-entropy, written directly as negative selected-action log probability rather than through a `cross_entropy` API.

The audit therefore rejects a wholesale “replace MSE with CE” change. The implemented loss families mostly match the quantities they optimize. The load-bearing concern is instead **gradient geometry**: deterministic BC/ExIt can install sharp logits, after which both the counterfactual and exact-count expected-reward gradients shrink toward zero under a softmax policy. This can leave decisive measured utility or count evidence unable to reverse a confidently wrong action. It is consistent with the campaign's observed BC-sharpened utility gaps and inert count/controller gradients, but it has not yet been isolated by a loss-only ablation.

The highest-value immediate test is therefore a cached loss-geometry audit followed, only if saturation is measured, by a gradient-matched comparison between the current counterfactual expected-reward loss and a magnitude-weighted pairwise logistic loss. The longer-term structural direction remains the anchored-value redesign already specified in [ties-by-design](../specs/RL/ties-by-design.md): train utility in measured units, preferably with cross-entropy over a fixed value support, then derive the policy at a fixed temperature.

## Definitions

- **Cross-entropy (CE).** A loss between a target categorical distribution and a predicted categorical distribution. With a one-hot target action, it is exactly `-log pi(a_target)`.
- **Mean squared error (MSE).** Squared regression error between a predicted scalar and a target scalar.
- **Huber / SmoothL1.** A robust scalar regression loss that is quadratic near zero and linear for large residuals.
- **Policy gradient.** A gradient of expected return obtained through `grad log pi(a|s)` weighted by a sampled return or advantage. Its implementation contains a log probability but is not ordinary supervised classification when advantages may be negative.
- **REINFORCE leave-one-out (RLOO).** A Monte Carlo policy-gradient estimator whose baseline for one rollout is the mean reward of the other rollouts in its group.
- **Exact expected-reward objective.** Direct differentiation of `sum_a pi(a|s) R(s,a)` when rewards for every legal action are known. It removes action-sampling variance but retains the geometry of the policy parameterization.
- **Counterfactual pair.** A selected action and one legal alternative evaluated in the same decision context by a full one-decision intervention.
- **Pair logit gap.** `d = log pi(a_selected) - log pi(a_alternative)`. The pair-restricted selected-action probability is `q = sigmoid(d)`.
- **Softmax saturation.** The regime in which one action probability approaches zero or one, causing probability-space expected-reward gradients to become very small.
- **Anchored utility.** A utility estimate denominated in measured utility units, rather than policy logits identified only by action ordering and an arbitrary scale.

## Scope and method

The audit traces every loss that can reach the semantic ranker or its controller in the current implementation:

- warm starts in `src/cloak/ranker/interactive.py`;
- hybrid utility, exact count, tie, entropy, KL, and sensitivity terms in `src/cloak/ranker/interactive.py`;
- counterfactual substitution in `src/cloak/ranker/counterfactuals.py`;
- semantic privacy-head pretraining in `src/cloak/ranker/privacy.py`; and
- the gain-head architecture preflight in `scripts/spikes/gain_head_preflight.py`.

The literature comparison distinguishes policy optimization, value estimation, imitation, pairwise ranking, and hard constraints. These are different statistical problems and do not have one universally optimal loss family.

## Implemented loss inventory

| Stage | Implemented objective | Loss family | Audit verdict |
|---|---|---|---|
| Gain-head preflight | MSE against distinct random scalar targets | Regression capacity test | Correct as a rejection-only architecture screen; not a production objective |
| Behavior cloning | `-log pi(a_teacher)` | One-hot categorical CE/NLL | Standard imitation loss; can make a deterministic teacher excessively sharp |
| ExIt cloning | `-log pi(a_winner)` | One-hot categorical CE/NLL | Standard cloning loss with the same sharpness risk |
| Provisional utility RL | `-A_LOO log pi(a)` | REINFORCE policy gradient | Standard and appropriate for sampled document-level returns |
| Counterfactual utility | `-delta_U (q_pair - 1/2)` | Exact local two-action expected reward | Correct objective; its gradient vanishes at confident probabilities |
| Count shaping | `-lambda sum_a pi_count(a) P(a)` | Exact finite-menu expected reward | Correct and variance-free; also weakens as the count distribution saturates |
| Tie ownership | `max(0, margin - z_gap)` | Hinge constraint | Appropriate for verified inequalities; constant gradient while violated |
| Policy regularization | `-beta H(pi) + eta KL(pi || pi_ref)` | Entropy and trust region | Standard RL machinery; coefficients and activation schedule remain design choices |
| Profile sensitivity | `(KL_adjacent - target)^2` | MSE-like auxiliary regularizer | Disabled by default; prior experiments do not support making it load-bearing |
| Privacy-head pretraining | SmoothL1 on standardized log counts, differences, and normalized scores | Robust scalar regression | Appropriate because the targets are continuous cardinal quantities |

Two terminology corrections follow from this inventory.

First, “there is no cross-entropy anywhere” is false for the full training pipeline. Both BC and ExIt explicitly minimize one-hot categorical cross-entropy. They implement it as selected-action negative log likelihood rather than calling `torch.nn.functional.cross_entropy`.

Second, the hybrid utility term is not MSE. `-A log pi(a)` is an advantage-weighted log-likelihood surrogate derived from the policy-gradient theorem. When `A > 0` it resembles positively weighted CE on the sampled action; when `A < 0` it deliberately decreases that action's likelihood and is no longer a proper cross-entropy against a probability target.

## Why MSE is correct in the gain-head preflight

The gain residual is a real scalar `r_phi(x_d)`. The preflight asks whether distinct decision features can produce distinct continuous outputs:

```text
r_phi(x_d) ~= y_d,    y_d sampled independently from Normal(0, 1)
```

This is a regression-capacity question even though the targets have no semantic meaning. Random targets are deliberate: they prevent the head from exploiting an easy shared structure and test whether the architecture can differentiate the observed decision inputs at all. MSE is a direct loss for that isolated question.

Passing is only a necessary condition. It does not show that sparse tie hinges or the dense count objective will choose to differentiate decisions. It also favors capacity when misused as a ranking metric. The preflight should therefore remain a rejection gate for collapse or suppression, while document-held-out training under the real objectives chooses between architectures.

## Literature mapping

### Policy optimization is not value regression

The ranker is currently actor-only: it has no learned value critic. Leave-one-out rollout rewards provide the baseline. Consequently, there is no production value-function MSE whose replacement by CE could fix the present actor.

The RLOO utility term is the conventional score-function form: sampled advantage multiplied by selected-action log probability. This is the right family when the environment supplies trajectory returns but not complete action values. Exact count shaping uses a stronger instrument where available: because every legal action's count target is known, it sums over the complete action menu rather than sampling one count target. Expected-action gradients are lower variance than sampled-action gradients when the action values are available.

### Value classification applies to the planned anchored tower

[Farebrother et al. 2024](../../research-wiki/papers/farebrother2024_stop_regressing_classification.md) ([arXiv 2403.03950](https://arxiv.org/abs/2403.03950)) show that categorical cross-entropy over a fixed value support can outperform scalar MSE for large, noisy, non-stationary value functions. The result does **not** imply that an actor's policy-gradient loss should become CE. It applies if ranker v2 implements the planned anchored utility/value estimator.

That redesign is a good fit for this result: utility is bounded, its measurement resolution is known, and a fixed-support distribution can retain cardinal utility units. The paper's caveat also matters: much of its advantage appears under bootstrapped and non-stationary targets, while this project's cached counterfactual deltas are directly measured and more stationary. HL-Gauss is therefore the preferred first arm, not a guaranteed winner over Huber regression.

### Policy-logit scale remains unanchored under ranking objectives

[Skalse et al. 2023](../../research-wiki/papers/skalse2023_partial_identifiability_reward.md) ([arXiv 2203.07475](https://arxiv.org/abs/2203.07475)) establish that ordinal comparisons identify rewards only up to positive linear scaling. The current counterfactual objective uses measured magnitudes as weights, but it still optimizes action probabilities rather than regressing `u(a_i)-u(a_j)` onto `delta_U`. It therefore does not give tower margins physical utility units.

[Schulman et al. 2017](../../research-wiki/papers/schulman2017_pg_soft_q_equivalence.md) ([arXiv 1704.06440](https://arxiv.org/abs/1704.06440)) give the relevant escape condition from the policy side: at an entropy-regularized fixed point with known temperature, policy logits correspond to soft action values up to a per-state shift. The current small entropy bonus is an exploration pressure, not such a fixed-temperature identification guarantee.

### CE warm starts and actor optimization have different goals

BC and ExIt ask a categorical question: imitate one selected action. CE is therefore appropriate. Hybrid RL asks a different question: change action probabilities to improve expected utility and privacy. Converting exact count targets into a one-hot max-count CE target would discard count magnitudes, force the privacy endpoint independently of utility, and give the controller a classification objective with no finite reason to stop increasing its authority.

The architecture and initialization also matter independently of the objective. [Andrychowicz et al. 2020](../../research-wiki/papers/andrychowicz2020_what_matters_onpolicy_rl.md) ([arXiv 2006.05990](https://arxiv.org/abs/2006.05990)) find large performance effects from policy-head initialization scale and recommend a much smaller final-layer initialization in their on-policy study. Their continuous-control setting does not transfer mechanically, but it supports treating warm-start margin scale as a first-class diagnostic rather than assuming a correct CE label makes arbitrary confidence harmless.

## Load-bearing loss risks

### 1. Counterfactual expected-reward gradients can be too weak to correct confident mistakes

For a counterfactual pair, define:

```text
d = log pi(a_selected) - log pi(a_alternative)
q = sigmoid(d)
L_cf = -delta_U * (q - 1/2)
```

The derivative with respect to the pair gap is:

```text
dL_cf / dd = -delta_U * q * (1 - q)
```

This is exactly the derivative of the measured local two-action expected utility, as the current spec states. It is not a mathematically incorrect surrogate. Its optimization weakness is equally explicit: `q(1-q)` approaches zero when the pair becomes confident. A selected action known to be better can receive almost no corrective gradient when the policy currently assigns it near-zero probability.

This failure geometry is especially relevant here because counterfactual terms **substitute** for provisional REINFORCE terms. A measured correction can therefore be more accurate than the provisional estimate while supplying less gradient near saturation. The recent campaign's collapsed decisions and BC-sharpened logit gaps make this a plausible load-bearing issue, but the repository has not yet reported the signed pair-gap and gradient-mass distribution needed to establish it.

### 2. Exact count expectation has the same probability-space saturation

The count loss is stronger statistically than sampled REINFORCE because it has no action-sampling variance. It is still differentiated through a softmax distribution. Its gradient is a policy covariance between count score and action log probability; as the distribution becomes deterministic, that covariance approaches zero.

Replacing it with CE toward the maximum-count action is not recommended. Such a target would erase the intended utility/privacy trade-off and would push the count-controlled parameters toward unbounded privacy authority. If saturation remains binding after the gain-head architecture is repaired, natural-gradient, mirror-descent, or anchored-value-derived policy updates are more principled directions because they improve policy-space geometry without changing the reward semantics.

### 3. Deterministic BC and ExIt can over-sharpen the initial policy

One-hot CE is the correct imitation loss, but zero training error does not bound confidence. Continuing to minimize it can keep enlarging the teacher-versus-runner-up gap. In ranker v2 this has a downstream cost: RL needs multiple distinct rollout vectors, while the controller and counterfactual losses need non-negligible probability on alternatives to obtain useful gradients.

This is a likely contributor, not an isolated causal result. The appropriate audit is to record BC/ExIt action accuracy together with entropy, teacher-runner-up logit gaps, expected unique trajectories under the production rollout count, and the controller shift required to cross those gaps.

### 4. MSE-like profile sensitivity is not a core justification problem

The optional profile-sensitivity regularizer regresses adjacent-profile KL onto a measured target with squared error. Squared loss is defensible for that scalar target, but prior experiments showed that the regularizer constrained profile shape while scale continued to move. It is disabled by default and should remain evaluation-only unless new evidence shows incremental value. Changing it to CE would not address the ranker's primary failure modes because there is no categorical target to classify.

## Recommendations

### 1. Keep the gain-head MSE preflight unchanged

Use it only to reject architectures that cannot emit distinct residuals or that suppress their own gradients. Do not use random-target fit MSE to choose the largest head; the document-held-out real-objective gate decides capacity.

### 2. Run a cached loss-geometry audit before changing training

For every cached counterfactual pair at each available checkpoint, record:

- signed pair gap `sign(delta_U) * d`;
- current corrective-gradient mass `abs(delta_U) * q * (1-q)`;
- confidently wrong, uncertain, and confidently correct mass as descriptive bins rather than a new gate;
- the same quantities by document, decision count, action mode, and epoch; and
- BC/ExIt teacher-runner-up gaps, entropy, and expected unique-vector rate.

The decision is simple: if measured utility disagreements remain concentrated where `q(1-q)` makes their gradients negligible, the expected-reward loss has an optimization problem despite representing the correct local objective. If disagreements occur at unsaturated probabilities, changing to logistic CE targets the wrong cause.

### 3. If saturation is confirmed, run one gradient-matched counterfactual ablation

Compare the current loss against:

```text
L_logistic = 0.5 * abs(delta_U)
             * softplus(-sign(delta_U) * d)
```

The factor `0.5` matches the two losses' derivative magnitude at `d=0`. This isolates tail geometry:

- both give zero utility-ordering gradient for exact ties because `abs(delta_U)=0`;
- both weaken once confidently correct; and
- logistic retains a strong correction when confidently wrong.

The logistic arm is a classification-calibrated ranking surrogate, not the exact expected local utility. Adoption therefore requires behavioral evidence, not an argument that CE is inherently superior. Pre-register correction of confidently wrong pairs, lambda-zero utility non-inferiority, greedy privacy separation, policy entropy, and churn. The utility cache remains valid because this is a post-hoc loss-assembly change.

### 4. Bound warm-start confidence separately

Retain BC/ExIt CE, but select their stopping point using both imitation and support diagnostics. A valid checkpoint must preserve teacher action accuracy without pushing alternative-action probability below what the configured rollout count can exercise. The exact operating threshold should be preregistered from rollout-diversity requirements, not tuned per model after observing privacy outcomes.

Do not change warm-start confidence and the counterfactual loss in the same experiment. Their interaction is the hypothesis, but changing both removes attribution.

### 5. Preserve the anchored-value redesign as the structural solution

The current actor logits encode policy preference, not measured utility. Even a successful logistic counterfactual arm would improve recovery from confident mistakes without anchoring margin units. The longer-term architecture should:

1. train a utility/action-value estimate in measured utility units;
2. compare HL-Gauss cross-entropy over a fixed support against robust scalar regression;
3. derive the action policy at a fixed, explicit temperature; and
4. compose privacy through anchored additive control or an epsilon-lexicographic rule.

This is where CE-versus-regression becomes a load-bearing architecture fork. It is not a reason to relabel the current policy-gradient actor as a classifier.

## Final assessment

| Question | Conclusion |
|---|---|
| Is all RL learning done through MSE? | No. Core RL uses RLOO policy gradients and exact expected rewards. |
| Is CE absent? | No. BC and ExIt are one-hot CE/NLL; hybrid policy gradients also use log probabilities but are not ordinary supervised CE. |
| Is the Phase-0 MSE justified? | Yes, as a necessary expressivity and saturation screen only. |
| Is the current hybrid loss theoretically valid? | Mostly yes. Each major term matches its intended quantity. |
| Is it optimal for this environment? | Not established. Counterfactual and count gradients may become ineffective under sharp softmax policies. |
| Should exact count shaping become CE? | No. That would change the privacy/utility objective and encourage unbounded privacy authority. |
| Where should CE be considered next? | In the anchored utility/value estimator, and as a measured counterfactual ranking ablation if saturation is confirmed. |

The audit's central conclusion is therefore narrower than “use CE”: **retain objective-correct expected rewards, measure where their softmax gradients disappear, use a matched logistic surrogate only if that disappearance is binding, and move cardinal utility learning into an explicitly anchored value objective rather than asking policy logits to acquire units indirectly.**

