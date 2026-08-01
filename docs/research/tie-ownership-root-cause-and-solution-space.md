---
type: research
status: current
created: 2026-08-01
updated: 2026-08-01
tags: [rl, ranker-v2, reward-ties, identifiability, measurement-design, lexicographic,
       constrained-rl, equivalence, deployment-generalization, root-cause, literature]
supersedes: docs/research/reward-ties-and-controller-authority.md
companion: [docs/specs/RL/ties-by-design.md,
            docs/specs/RL/interactive-ranker-v2-decision-log.md,
            docs/issues/qa-dependency-underdeclaration.md,
            research-wiki/experiments/2026-07-31-RL-ranker-v15-equivalence-critic.md]
---

# Why the privacy dial does not transfer: three defects behind the ranker's reward ties

This report consolidates the round-4 root-cause reassessment (2026-08-01) of the interactive ranker's tie-ownership problem, assembled from four independent passes — a codebase analysis, an independent Codex Sol High pass, and three literature sweeps (reward-scale identifiability; objective composition; equivalence generalization and measurement-resolution limits) — plus one new cache-only measurement executed as Gate 0. It supersedes the *solution space* of the [earlier failure-taxonomy report](reward-ties-and-controller-authority.md); that report's five-step causal chain still stands and is not repeated here.

Every quantitative claim below is a measurement from this repo's artifacts or a cited external result. Where a claim is inference rather than measurement, it is labelled as such.

## Executive summary

We spent five experiment rounds trying to make a controller win a race it could not win, because we misdiagnosed one problem as three separate symptoms. There are in fact three genuine defects, stacked, and only one of them is about modelling.

**The measurement defect (new, measured).** We difference *document-level* utility to score a single-decision counterfactual, even though the utility artifact already declares which assertions each decision can affect. On 2,895 cached pairs whose decision has zero linked assertions — pairs that are provably equivalent because no feedback channel can separate them — our statistic reports a nonzero difference for **1,873 of them (65%)**, median magnitude 0.019. Restricting the statistic to linked assertions collapses the "sub-noise" band by about three quarters. A large share of the tie problem was manufactured by our own instrument.

**The scale defect (formalised).** Ordinal comparison data identifies a reward only up to positive linear scaling. Our tower's margins therefore carry no physical units *by construction*, in the infinite-data limit, at every hyperparameter setting. Every scale-control arm we ran (soft-cap, KL anchor, sensitivity regulariser, learned gain) was addressing a symptom of an unidentifiability theorem.

**The composition defect.** We compose utility and privacy by additive scalarisation. Where the primary objective's gradient vanishes — which is exactly the tie stratum — no gradient-space composition can localise the decision; only a set-level operation can.

**The uncomfortable finding.** The refutation was already in our own research wiki. We cited a paper that prescribes a *filter* and built a *bonus*; we cited an impossibility result for fixed weightings and used it to motivate a learned gain instead of retiring the fixed weighting; and the soft-cap we added to preserve controller authority is, on re-reading the constrained-RL result we had already registered, the thing that guaranteed the controller would lose.

**What follows.** Delete five mechanisms, repurpose four, add two. The recommended direction is an ε-lexicographic filter over margins anchored in measured utility units, shipping the exact-tie core first because it is provably free — and, after the measurement correction, larger and cleaner than we believed.

## Definitions

- **Decision / action menu.** One sensitive span the policy must act on; its menu contains `keep`, one action per authored **generalization level** of the span's lattice, and a `placeholder`.
- **Utility.** Whether the round-tripped document still preserves its QA-probed answers. Measured by a local **reader** scoring per-**assertion** component scores, aggregated over a fixed weight denominator into **document utility**.
- **Assertion (QA probe).** One question-answer obligation attached to a document. Declares `credit_routing`: `linked` (with an explicit `policy_dependency_decision_ids` list) or `residual` (declared to depend on no decision).
- **Linked assertions of a decision.** The assertions that declare a dependency on it (`linked_by_decision` in `src/cloak/reward/utility_credit.py`). Its complement is, by the artifact's own declaration, unable to depend on that decision.
- **Privacy / count score (p̂).** The frozen per-action anonymity count from the lattice. Never learned; computable on any document.
- **Utility tower (u).** The λ-blind per-action logits produced by the policy's utility head.
- **Controller.** The additive combination z(a) = u(a) + α·g(λ)·p̂(a) that orders actions at positive λ, with an exact identity branch at λ=0.
- **λ (lambda) profile.** The user-facing privacy dial; g(λ) is a fixed ramp.
- **Counterfactual probe.** A scored pair of full action vectors differing in exactly one decision; its measured utility difference is **ΔU**.
- **Exact tie / sub-noise pair / live pair.** |ΔU| ≤ 1e-9 / 0 < |ΔU| ≤ 0.044 / |ΔU| > 0.044, where 0.044 is the measured reader-resolution floor.
- **Evidence ledger.** The per-(document, decision, action-pair) record of measured ΔU keyed by surrounding-context hash. Per-document, and empty at inference.
- **Structurally derivable tie.** A decision with zero linked assertions: no feedback channel can distinguish its actions, so its whole menu is one equivalence class by declaration rather than by measurement.
- **Anchoring (scale anchoring).** Training a head so its outputs are denominated in measured utility units, rather than being a ranking defined only up to monotone/affine transformation.
- **ε-lexicographic filter.** Take the set of actions within ε of the primary optimum, then maximise the secondary objective inside that set. Contrast with **additive scalarisation**, which adds a weighted secondary term to the primary score.
- **Verifiable vs unverifiable regime.** Membership tests against an *absolute* threshold admit finite-sample certificates; membership tests against a band around an *estimated* maximum do not.

## Part 1 — Where the ties come from

The lattice and QA artifact were deliberately built so several generalization levels of a span can all preserve every probed answer. When they do, the reader scores them identically and the utility reward is silent between them; only the count score distinguishes them. This is correct design, not a defect: levels that preserve every task-relevant answer *are* utility-equivalent, and grading partial credit by lattice depth would inject the privacy preference into utility, double-count it at positive λ, and corrupt matched-realized-privacy comparisons across methods. The full argument is in the [ties-by-design spec](../specs/RL/ties-by-design.md).

The consequence is structural. On an exactly tied pair the tower receives **no ordering gradient from any channel** — leave-one-out advantages are zero and the counterfactual pair loss −ΔU·(q−0.5) is identically zero. Ordering ties is therefore deliberately the controller's job, because the desired order is λ-dependent while the tower is λ-blind. That division of labour is sound. What broke is everything about *how* the controller was given authority.

Prevalence, full corpus: 67 documents with policy decisions, 652 decisions, of which 86 (13%) have zero linked assertions. The tiny stratum (≤6 decisions) is 22 documents (33%). Ties are a first-class regime, not a corner case.

## Part 2 — The three defects

### 2.1 Measurement: we difference the wrong quantity

A counterfactual probe flips one decision and takes the difference of **document-level** utility. But document utility aggregates *all* assertions over a fixed denominator, and the artifact already tells us which ones the flipped decision can affect. Every other assertion contributes measurement noise and no signal — two separate reader runs, differenced.

The arithmetic alone is alarming. Across 603 decisions with linked mass, the median decision's *entire* linked assertion mass is only **10.6%** of its document's utility denominator; for **25% of decisions the total linked mass is ≤ 0.044**, meaning that even if a wrong action destroyed every assertion that decision is responsible for, the document-level change would sit at or below the reader's resolution floor.

Gate 0 measured the effect directly over all 10,459 cached single-decision pairs:

| stratum (certification split) | document-level statistic | linked-restricted statistic |
|---|---|---|
| structurally derivable | 0 (invisible) | 45 |
| exact tie | 60 | 52 |
| sub-noise | 43 | 14 |
| live | 44 | 36 |

The training split moves the same way and harder: the sub-noise band falls from 4,735 rows to 1,182 while the exact-tie core grows from 3,010 to 4,974, with a further 2,820 rows reclassified as structurally derivable. The low-range separation in the |ΔU| distribution sharpens from a 4.6× to a **38.5×** multiplicative jump.

The decisive check is on pairs where a *proof* exists. For the 2,895 pairs whose decision has zero linked assertions, no feedback channel can separate the actions — this is the duplicate-action construction of [Bartók et al. 2014](../../research-wiki/papers/bartok2014_partial_monitoring_classification.md) ([DOI 10.1287/moor.2014.0663](https://doi.org/10.1287/moor.2014.0663)), where a loss difference that is not a function of any obtainable feedback makes the distinction unidentifiable in principle. Our document-level statistic nonetheless reports a nonzero difference on **1,873 of them (65%), median magnitude 0.019** — roughly half the "noise floor" we have treated as a physical constant.

**Is the correction safe?** Not automatically, and the direction matters: if the artifact *under-declares* dependencies, the linked statistic under-measures and manufactures false ties, which is the direction that spends real utility. A sign-consistency test on the removed residue (295 pair-groups measured in ≥3 distinct contexts; consistency = |mean| / mean|·|, where 0 is zero-mean noise and 1 is systematic) gives median **0.35**, with **62% noise-like** and **18% systematic**. So the majority of what the correction removes is genuine noise, and the systematic minority is a QA-artifact defect — filed as [qa-dependency-underdeclaration](../issues/qa-dependency-underdeclaration.md), non-blocking, with the conservative interim rule that a pair is labelled a tie only when **both** statistics agree it is below the floor.

**Reading.** The 0.044 figure is not an intrinsic reader property. It is largely the standard deviation of cross-run reader jitter summed over assertions irrelevant to the decision under test, against which we then compare a signal diluted roughly tenfold. Every experiment from v13 onward was partly fighting fog of our own making. *(Inference, not measurement: how much of the earlier failures this explains is unknown until the corrected statistic is used in a training run.)*

### 2.2 Scale: the tower's margins have no units, provably

Ordinal comparison data identifies a reward function only **up to positive linear scaling** — [Skalse et al. 2023](../../research-wiki/papers/skalse2023_partial_identifiability_reward.md) ([arXiv 2203.07475](https://arxiv.org/abs/2203.07475)), Theorem 3.10. Our counterfactual pair loss is ordinal-with-magnitude-weighting: it moves an action's probability but never asserts that u(a_i) − u(a_j) equals ΔU_ij. So the tower's scale is unidentified by construction, in the infinite-data limit, at every hyperparameter setting. No amount of capping, anchoring or gain-learning can supply information the objective never contained.

Theorem 3.8 names the escape: comparisons carry cardinal information when the comparator's temperature is *known and fixed*. [Schulman et al. 2017](../../research-wiki/papers/schulman2017_pg_soft_q_equivalence.md) ([arXiv 1704.06440](https://arxiv.org/abs/1704.06440)) states the same thing from the policy side — logits equal Q/τ up to a per-state constant only at an entropy-regularised fixed point with fixed τ, which our small entropy bonus does not deliver.

One important correction to the naive version of this fix: anchoring can fix the **scale**, not the within-menu **shift**. The additive shift freedom is a genuine invariance of any softmax policy and no regression head removes it. That is harmless, because the controller's own shift is per-menu shift-invariant too; only scale ever competed with it.

Head-to-head evidence that anchoring is the right lever: [Wang et al. 2024](../../research-wiki/papers/wang2024_helpsteer2_preference.md) ([arXiv 2410.01257](https://arxiv.org/abs/2410.01257)) compare, on matched data, regression (93.0) against a magnitude-scaled Bradley-Terry loss (92.7) against plain Bradley-Terry (91.5) — our current pair loss is the middle one — and their best model *composes* regression first with ranking on top. The implementation detail that matters: regress with a cross-entropy-over-support objective rather than MSE ([Farebrother et al. 2024](../../research-wiki/papers/farebrother2024_stop_regressing_classification.md), [arXiv 2403.03950](https://arxiv.org/abs/2403.03950)), whose one hyperparameter has a *measured* value for us — pin σ to the resolution floor, which keeps the design clear of the no-calibration-knob rule.

### 2.3 Composition: additive scalarisation cannot localise a flat region

Where the primary gradient vanishes, no gradient-space composition can localise the decision; only a set-level operation can. This generalises the argument this repo already used to reject gradient projection ([Peri et al. 2025](../../research-wiki/papers/peri2025_nonconflicting_energy_minimization.md), [arXiv 2509.01765](https://arxiv.org/abs/2509.01765)): projection needs a nonzero primary gradient to project against, and on an exact tie the utility gradient is identically zero.

Linear scalarisation also recovers only the convex hull of the Pareto front and is a substantive modelling assumption rather than a neutral default ([Hayes et al. 2022](../../research-wiki/papers/hayes2022_practical_guide_morl.md), [arXiv 2103.09568](https://arxiv.org/abs/2103.09568)). More sharply, no *fixed* weighting induces the optimal constrained policy in general ([Calvo-Fullana et al. 2021](../../research-wiki/papers/calvofullana2021_state_augmented_constrained_rl.md), [arXiv 2102.11941](https://arxiv.org/abs/2102.11941)), and the optimal lexicographic policy may not exist anywhere in the scalarised solution space ([Wray et al. 2015](../../research-wiki/papers/wray2015_lexicographic_mdp_slack.md), [DOI 10.1609/aaai.v29i1.9647](https://doi.org/10.1609/aaai.v29i1.9647), Proposition 6).

The structural insight that reconciles the two designs: **an additive controller with an anchored scale and an α derived from the measured margin gap is a *soft* ε-lexicographic filter, and the hard filter is its limit.** They are not rivals; they differ in hard versus soft thresholding. Which one is admissible is a measurable question — a single α exists iff the anchored margin distribution separates the noise band from the smallest live effect.

### 2.4 How the three compound

The measurement defect inflates the apparent tie region and blurs its boundary. The scale defect means the tower's margins inside that region are arbitrary and drifting, moved by gradients from unrelated documents ([Schaul et al. 2022](../../research-wiki/papers/schaul2022_policy_churn.md), [arXiv 2206.00730](https://arxiv.org/abs/2206.00730)) — textbook underspecification ([D'Amour et al. 2020](../../research-wiki/papers/damour2020_underspecification_ml.md), [arXiv 2011.03395](https://arxiv.org/abs/2011.03395)). The composition defect means the only tool we gave the controller is a bounded additive shift that must out-race those arbitrary margins. The observable result is a privacy dial that flips coins, most visibly on small documents where one decision moves the document average a lot.

## Part 3 — What we ran, and what it actually showed

**v13 (learned controller gain).** A state-conditioned gain head was expected to give per-decision authority. It degenerated to a bound-pinned global constant; the gain field never differentiated. *Correct reading in hindsight:* this is the predicted failure of implementing a state-augmented architecture without the dual dynamics that drive a multiplier, and the bound itself was self-defeating (see 4.2).

**v14 (evidence-supervised tie ownership).** With ties *measured* per document into a ledger and the controller forced to own them by a hinge plus cycle-boundary projection, the policy tracked the hard lexicographic oracle at 43 of 48 document-epochs with zero constraint violations and global α frozen (5.350 → 5.352, with effective tie-decision α reaching 13.58). **The mechanism works when the ties are known.** Residual oscillation moved upstream into the oracle itself, from tower churn among utility-equivalent actions and evidence-coverage gaps.

**v15 (equivalence critic).** A supervised predictor was trained to substitute prediction for measurement on unseen documents. It failed decisively: held-out gate AUC 0.599, 7 of 147 certification pairs accepted with 4 violations (precision 0.43 against a finite-sample-corrected 0.98 bar), and a Balanced-MSE baseline at chance (AUC 0.494). Conservative abstention held — no false-tie flood — so the mechanism was safe and useless. 97.6% of the evidence came from four documents.

**The honest reading of v15.** It did not establish that utility-equivalence is unlearnable. It established that frozen features plus linear probes trained almost entirely on four documents cannot certify transfer. Moreover the failure was *forecastable*: the one literature line that genuinely learns an equivalence relation and generalises it — bisimulation metrics — has a documented collapse mode under exactly a flat or sparse reward field ([Kemertas & Aumentado-Armstrong 2021](../../research-wiki/papers/kemertas2021_robust_bisimulation_metric.md), [arXiv 2110.14096](https://arxiv.org/abs/2110.14096)).

## Part 4 — Why we missed it

### 4.1 Design decisions, with their legitimate reasons

| decision | why it was reasonable | what it cost |
|---|---|---|
| Difference document-level utility for single-decision probes | document utility is the correct *objective*; the dependency data was built for credit routing, not measurement | 65% false-nonzero on provably tied pairs; an inflated, blurred tie region |
| Train the tower purely by policy gradient | it is the actor; per-action utility looked unidentifiable | margins with no units; the controller later treated them as if they had units |
| Additive scalarisation | one smooth dial, exact differentiable count credit, clean λ=0 identity | a threshold race the controller cannot win on flat regions |
| Four-document campaign | fast iteration under expensive remote reward and one GPU | architectural defect undetected for five rounds; all generalisation claims untestable |
| Accept 0.044 as a fixed floor | it *was* honestly measured | it conflated reader jitter, probe incompleteness, and true ties; and was then misused as a utility budget |
| Probe depth over breadth | correct for online credit assignment | wrong for representation learning; caused the v15 support failure |

### 4.2 The citation autopsy

Several already-registered papers contained the refutation and were read as motivation for an increment.

[Skalse et al. 2022](../../research-wiki/papers/skalse2022_lexicographic_morl.md) ([arXiv 2212.13769](https://arxiv.org/abs/2212.13769)) prescribes a **filter** — restrict to the slack-optimal set, then optimise the secondary inside it. We cited it and built a **bonus**. A filter and a shift are different objects.

[Tercan & Prabhu 2024](../../research-wiki/papers/tercan2024_thresholded_lexicographic.md) ([arXiv 2408.13493](https://arxiv.org/abs/2408.13493)) argues a single global threshold is wrong because the indifference value is state-dependent. We used that argument to justify keeping a global α — but it condemns a global exchange rate at least as hard as a global tolerance.

[Vamplew et al. 2024](../../research-wiki/papers/vamplew2024_value_function_interference.md) ([arXiv 2402.06266](https://arxiv.org/abs/2402.06266)) diagnoses tie-set interference and prescribes deterministic, stated tie-breaking. We adopted the diagnosis and dropped the prescription — which *is* a filter.

[Calvo-Fullana et al. 2021](../../research-wiki/papers/calvofullana2021_state_augmented_constrained_rl.md) proves no fixed weighting suffices. We used it to motivate a learned gain rather than to retire fixed-α, then built the state-augmented architecture *without* the dual dynamics that make a multiplier move.

Most sharply, [Roy et al. 2021](../../research-wiki/papers/roy2021_direct_behavior_specification_crl.md) ([arXiv 2112.12228](https://arxiv.org/abs/2112.12228)) re-read says a multiplier diverges upward against a constraint that resists indefinitely. On an exact tie the utility constraint is *maximally slack*, so by complementary slackness the required α is unbounded. **The soft-cap we adopted to keep α in authority is precisely what guaranteed it would lose the tie race.** The mitigation was self-defeating in exactly the regime that motivated it.

And [Zhang et al. 2026](../../research-wiki/papers/zhang2026_alam_multiplier_network.md) ([arXiv 2605.00667](https://arxiv.org/abs/2605.00667)): we kept its warning (per-state multipliers oscillate) and discarded its mechanism (supervised regression to a dual target). v13's failure was the *opposite* of oscillation — a flat, undifferentiated field.

## Part 5 — The solution space

### 5.1 The critic, reassessed

A learned per-instance filter that generalises to unseen states is **not** a workaround; it is the published standard ([Zahavy et al. 2018](../../research-wiki/papers/zahavy2018_action_elimination.md), [arXiv 1809.02121](https://arxiv.org/abs/1809.02121)), and the v15 design matched it structurally: a supervised predictor of a training-only signal, a set rule, an abstention path. Two specifics were wrong. The set rule must be built on a **confidence bound, never a point estimate**, and the guarantee must be **one-sided** — never eliminate an action that could belong to an optimal policy; abstain otherwise. A third, methodological: v15 certified an absolute-threshold object (verifiable regime) while deploying a distance-from-*predicted*-max set (unverifiable regime).

What is genuinely objectionable about the v15 critic is its *packaging*: a second utility model beside an existing utility model, trained on the same measurements, to answer a question the first model should be able to answer. The lazy and principled fix is the same one — anchor the head we already have.

The pattern the equivalence literature actually supports: **derive what is derivable, enforce it as a constraint, predict only the residue, and stratify evaluation so the derivable part never inflates the hard part's numbers.** Frameworks that achieve unseen-instance equivalence get it from declared structure — equivariance constraints in the hypothesis class ([van der Pol et al. 2020](../../research-wiki/papers/vanderpol2020_mdp_homomorphic_networks.md), [arXiv 2006.16908](https://arxiv.org/abs/2006.16908)) or duplicate-action deletion from a declared loss structure (Bartók) — not from a free-form predictor.

### 5.2 Ranked options

**1. ε-lexicographic filter over anchored margins — recommended.** Ship the **exact-tie core first**: optimising a secondary objective over the primary's *exact* argmax set costs the primary nothing (Skalse 2022), and after the Gate-0 correction that set is larger and cleaner than we believed. Extend to the sub-noise band only under a metered document-level budget. *Requires:* scale anchoring (2.2). *Risk:* a filter needs a per-instance notion of which actions are ε-optimal, which is the prediction problem again — but now posed over a calibrated quantity with a one-sided guarantee.

**2. Anchored additive, as the soft relaxation of (1).** With u denominated in ΔU units, α becomes a genuine exchange rate (utility per unit count). Admissible iff the anchored margin distribution separates the noise band from the smallest live effect — feasible α must satisfy both a lower bound (to own ties) and an upper bound (to never buy count with real answer loss). *This is a cheap measurable test and should be run before committing to either option.* Regardless of its outcome, anchored-additive remains correct for the live-margin stratum, so the likely end state is a composition: additive on live margins, filter on the ε-band.

**3. Structural derivation as a constraint and a label source, never a deployment oracle.** Zero-linked decisions are provable ties; their real value is **2,895 free, document-diverse labels across 23 documents**, attacking v15's diagnosed support failure at zero probe cost. *Hard limit:* the QA artifact is training-time supervision built by a teacher; a real user document has no assertions, so this cannot run at deployment.

**4. Full state-augmented Lagrangian.** Principled ([Paternain et al. 2019](../../research-wiki/papers/paternain2019_zero_duality_gap.md), [arXiv 1910.13393](https://arxiv.org/abs/1910.13393) gives strong duality; Calvo-Fullana gives the state-augmented form), but strictly more machinery than (1) for the same result, and complementary slackness demands a multiplier that diverges exactly where we need it. Only if (1) fails.

**5. Deployment-time probing — reopened, not rejected.** The [ties-by-design spec](../specs/RL/ties-by-design.md) rejected this partly on remote cross-query exposure. That premise is faulty: **the reader is local**, so a locally regenerated probe set never touches the remote model. The real blockers are local compute and trust in self-generated probes. This deserves a cost estimate before another prediction round is funded.

### 5.3 Deletions, repurposings, additions

**Delete** — each is scale control applied to a misspecified composition: the evidence hinge and cycle projection (a margin constraint inside the likelihood measured *zero* gain over plain ranking in Wang et al.'s Margin-BT variant, the closest published relative), the utility-logit soft-cap ([Gemma Team 2024](../../research-wiki/papers/gemmateam2024_gemma2_logit_softcap.md), [arXiv 2408.00118](https://arxiv.org/abs/2408.00118) is sound as a stability device but load-bearing only for an unanchored scale — and see 4.2), the capped learned gain, gap-scaling, and the KL anchor and profile-sensitivity regulariser as *training* pressures.

**Repurpose:** the tower becomes a utility-advantage estimator anchored in ΔU units; counterfactuals become regression targets computed on linked assertions; leave-one-out returns become value supervision; λ profiles become document-level slack budgets; the sensitivity check becomes evaluation-only; ExIt becomes a source of high-utility value targets rather than policy clones.

**Add:** exactly two — pairwise utility-difference regression on the existing head, and a constrained selector. That is a smaller system than the controller stack it replaces.

One caveat against over-deleting: separate policy and value networks outperformed shared ones on 4 of 5 environments in a 250k-agent study ([Andrychowicz et al. 2020](../../research-wiki/papers/andrychowicz2020_what_matters_onpolicy_rl.md), [arXiv 2006.05990](https://arxiv.org/abs/2006.05990)), while the cross-entropy-value line reports the opposite for representation quality. Shared-versus-separate trunk is therefore an *arm*, not an assumption. That study also supplies a free precaution that predates every mechanism we built: initialising the policy head's last layer 100× smaller was its single highest-impact finding, and our failure mode is margins growing to +225.

## Part 6 — Traps and open questions

**Trap 1 — the floor is a per-pair instrument resolution, not a per-decision spending allowance.** Spending ε independently at D decisions bounds document utility loss at ε·D; at D=25 that is 1.1, larger than the entire utility scale. The correct construction fixes a document-level budget δ and spends δ/D per decision (Wray et al., Proposition 1). Exact ties cost zero and are exempt.

**Trap 2 — the reader's resolution is not the user's tolerance.** 0.044 is what the instrument can distinguish; the utility loss a user will accept is a different quantity and may be smaller. Declaring sub-floor differences equivalent is principled — targeting within distortion D costs at most D of primary regret while replacing the learning cost with a much smaller rate-distortion term ([Russo & Van Roy 2018](../../research-wiki/papers/russo2018_satisficing_bandit.md), [arXiv 1803.02855](https://arxiv.org/abs/1803.02855)) — but conflating instrument resolution with user tolerance would be a calibration trick wearing a measurement's clothes.

**Trap 3 — an anchored margin is not guaranteed to sit near zero on tie pairs.** Anchoring makes "margin below the floor implies measured tie" definitionally true, but whether anchored margins actually land there under competing gradients is an optimisation claim, unverified anywhere in the literature.

**Open questions.** Does regression interfere with policy gradient on a shared trunk (no paper reports this combination)? Does anchoring hold on a few hundred documents, when every supporting result is at Atari or 70B scale? Is the 18% systematic residue a dependency-declaration bug, and does repairing it change the strata? What does local QA regeneration at deployment actually cost?

**Genuinely unpublished in our setting** — do not go looking for these papers: equivalence defined by an *external measurement instrument* rather than by environment dynamics; transfer of a measured equivalence relation to instances where the instrument is *absent entirely*; routing a secondary objective through utility-equivalence classes; an ε derived from a measured instrument resolution rather than chosen as a preference; and certified-conservative equivalence gating with a document-level risk bound.

## Part 7 — The gate plan

**Gate 0 — free, cache-only. Executed 2026-08-01.** Recompute all evidence under the linked-restricted statistic, re-derive the strata, harvest structurally derivable labels. Result: sub-noise band down ~75%, 2,895 free labels across 23 documents, certification tie-band 111 rows against a minimum of 45 (PASS). Artifact: `results/ranker_v2/architecture/equivalence_critic/evidence-rows-linked.json` (both statistics per row).

**Gate 1 — executed 2026-08-01. FAILED, cause not identifiable.** Probes regressing linked-restricted ΔU from the tower's own features, scored against the actor margin on the same held-out rows. Train fit is strong (tie-AUC 0.85, MAE 0.053 against a 0.873 noise ceiling), so this is not a capacity failure — but held-out discrimination does not beat the baseline (best anchored arm 0.628 versus actor 0.592 at certification, and 0.606 versus 0.733 at calibration; anchored wins on 2–3 of 15 documents). Two findings survive: anchoring **does** deliver calibrated scale on held-out data (slope ≈1.0, MAE ≈0.12, versus the actor's slope 0.010 and MAE 10.57 — the identifiability prediction confirmed empirically), and anchoring **loses ordering** (live sign agreement 0.75 versus the actor's 0.86), which settles that regression must *compose* with ranking rather than replace it. The learning-curve diagnostic (1/2/3/4 training documents → 0.518/0.655/0.616/0.525) is **underpowered**: at 66 tie / 36 live certification rows the minimum resolvable AUC difference is 0.159 against an observed range of 0.138, so it can support neither "breadth helps" nor "the representation is dead". Full record: [Gate 1 experiment](../../research-wiki/experiments/2026-08-01-RL-ranker-gate1-representation.md).

**Consequence for the plan.** The binding constraint is now evidence volume — in training (4 documents) and, newly and more sharply, in *evaluation* (102 judged held-out rows). The preregistered unfrozen-trunk escalation is not worth running, because it would be scored on the same rows and inherit the same 0.159 resolution floor. Buying evaluation breadth, structural derivation, or deployment-time regeneration all rank above further modelling arms.

**Gate 2 — cache-only. Not yet run.** The selector gate: exact-tie-only filter first (provably free), cache-only held-out greedy rollouts against both the current actor-controller and the evidence oracle; then extend to the sub-noise band under a document-level budget with the *measured* aggregate utility drop reported against the bound.

**No RL run is justified before Gates 1 and 2 pass.**

## Sources

Registered wiki pages, grouped by the question they bear on. Scale identifiability and anchoring: [skalse2023_partial_identifiability_reward](../../research-wiki/papers/skalse2023_partial_identifiability_reward.md) ([arXiv 2203.07475](https://arxiv.org/abs/2203.07475)), [schulman2017_pg_soft_q_equivalence](../../research-wiki/papers/schulman2017_pg_soft_q_equivalence.md) ([arXiv 1704.06440](https://arxiv.org/abs/1704.06440)), [wang2024_helpsteer2_preference](../../research-wiki/papers/wang2024_helpsteer2_preference.md) ([arXiv 2410.01257](https://arxiv.org/abs/2410.01257)), [farebrother2024_stop_regressing_classification](../../research-wiki/papers/farebrother2024_stop_regressing_classification.md) ([arXiv 2403.03950](https://arxiv.org/abs/2403.03950)), [andrychowicz2020_what_matters_onpolicy_rl](../../research-wiki/papers/andrychowicz2020_what_matters_onpolicy_rl.md) ([arXiv 2006.05990](https://arxiv.org/abs/2006.05990)). Composition: [skalse2022_lexicographic_morl](../../research-wiki/papers/skalse2022_lexicographic_morl.md) ([arXiv 2212.13769](https://arxiv.org/abs/2212.13769)), [tercan2024_thresholded_lexicographic](../../research-wiki/papers/tercan2024_thresholded_lexicographic.md) ([arXiv 2408.13493](https://arxiv.org/abs/2408.13493)), [wray2015_lexicographic_mdp_slack](../../research-wiki/papers/wray2015_lexicographic_mdp_slack.md) ([DOI 10.1609/aaai.v29i1.9647](https://doi.org/10.1609/aaai.v29i1.9647)), [paternain2019_zero_duality_gap](../../research-wiki/papers/paternain2019_zero_duality_gap.md) ([arXiv 1910.13393](https://arxiv.org/abs/1910.13393)), [calvofullana2021_state_augmented_constrained_rl](../../research-wiki/papers/calvofullana2021_state_augmented_constrained_rl.md) ([arXiv 2102.11941](https://arxiv.org/abs/2102.11941)), [roy2021_direct_behavior_specification_crl](../../research-wiki/papers/roy2021_direct_behavior_specification_crl.md) ([arXiv 2112.12228](https://arxiv.org/abs/2112.12228)), [zhang2026_alam_multiplier_network](../../research-wiki/papers/zhang2026_alam_multiplier_network.md) ([arXiv 2605.00667](https://arxiv.org/abs/2605.00667)), [hayes2022_practical_guide_morl](../../research-wiki/papers/hayes2022_practical_guide_morl.md) ([arXiv 2103.09568](https://arxiv.org/abs/2103.09568)), [vamplew2024_value_function_interference](../../research-wiki/papers/vamplew2024_value_function_interference.md) ([arXiv 2402.06266](https://arxiv.org/abs/2402.06266)), [peri2025_nonconflicting_energy_minimization](../../research-wiki/papers/peri2025_nonconflicting_energy_minimization.md) ([arXiv 2509.01765](https://arxiv.org/abs/2509.01765)). Equivalence, filters, and measurement limits: [zahavy2018_action_elimination](../../research-wiki/papers/zahavy2018_action_elimination.md) ([arXiv 1809.02121](https://arxiv.org/abs/1809.02121)), [vanderpol2020_mdp_homomorphic_networks](../../research-wiki/papers/vanderpol2020_mdp_homomorphic_networks.md) ([arXiv 2006.16908](https://arxiv.org/abs/2006.16908)), [kemertas2021_robust_bisimulation_metric](../../research-wiki/papers/kemertas2021_robust_bisimulation_metric.md) ([arXiv 2110.14096](https://arxiv.org/abs/2110.14096)), [bartok2014_partial_monitoring_classification](../../research-wiki/papers/bartok2014_partial_monitoring_classification.md) ([DOI 10.1287/moor.2014.0663](https://doi.org/10.1287/moor.2014.0663)), [russo2018_satisficing_bandit](../../research-wiki/papers/russo2018_satisficing_bandit.md) ([arXiv 1803.02855](https://arxiv.org/abs/1803.02855)), [asadi2019_state_action_equivalence](../../research-wiki/papers/asadi2019_state_action_equivalence.md) ([arXiv 1910.04077](https://arxiv.org/abs/1910.04077)), [baram2021_action_redundancy](../../research-wiki/papers/baram2021_action_redundancy.md) ([arXiv 2102.11329](https://arxiv.org/abs/2102.11329)). Drift and underspecification: [schaul2022_policy_churn](../../research-wiki/papers/schaul2022_policy_churn.md) ([arXiv 2206.00730](https://arxiv.org/abs/2206.00730)), [damour2020_underspecification_ml](../../research-wiki/papers/damour2020_underspecification_ml.md) ([arXiv 2011.03395](https://arxiv.org/abs/2011.03395)), [mandal2023_performative_rl](../../research-wiki/papers/mandal2023_performative_rl.md) ([arXiv 2207.00046](https://arxiv.org/abs/2207.00046)), [gemmateam2024_gemma2_logit_softcap](../../research-wiki/papers/gemmateam2024_gemma2_logit_softcap.md) ([arXiv 2408.00118](https://arxiv.org/abs/2408.00118)).

Repo artifacts: [ties-by-design spec §7](../specs/RL/ties-by-design.md), [decision log round-4 entry](../specs/RL/interactive-ranker-v2-decision-log.md), [v15 experiment record](../../research-wiki/experiments/2026-07-31-RL-ranker-v15-equivalence-critic.md), [v14 experiment record](../../research-wiki/experiments/2026-07-31-RL-ranker-v14-evidence-tie-ownership.md), [QA dependency issue](../issues/qa-dependency-underdeclaration.md), miner `scripts/spikes/equivalence_critic_screening.py`, census `scripts/spikes/tie_structure_census.py`.
