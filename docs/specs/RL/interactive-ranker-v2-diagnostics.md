---
type: reference
status: current
created: 2026-07-12
updated: 2026-07-22
tags: [rl, ranker, diagnostics, gates, thresholds, preregistration, calibration, spec]
companion: [docs/specs/RL/interactive-ranker-v2.md,
            docs/specs/RL/interactive-ranker-v2-decision-log.md,
            docs/specs/qa-builder-v2.md,
            docs/issues/2026-07-08-rl-env-and-lattice-count-issue-register.md]
---

# Interactive ranker v2 — diagnostics, thresholds, and pre-training gates

**Status: normative diagnostic design; empirical threshold values remain to be frozen by the
preflight spike.** This specification defines what is measured, how measurements become
decisions, when thresholds are chosen, and which data may influence them. The companion ranker
spec defines the policy and loss.

A numeric threshold is a precommitted boundary that converts a diagnostic into an action:
continue, block, reduce scope, trigger an ablation, reject promotion, or record a future audit
obligation. Thresholds are not reward weights and do not enter gradients unless the companion
spec explicitly defines them as training coefficients.

## Why thresholds are part of the experiment

Terms such as *material*, *sufficiently supported*, and *meaningfully different* are incomplete
requirements. Without an operational boundary, the same measured result can be called harmless
when the run succeeds and causal when it fails. Every consequential term must therefore resolve
to one of:

- an exact invariant;
- a frozen numeric threshold;
- a frozen statistical test;
- a report-only diagnostic that cannot change the completed run's verdict.

Thresholds are versioned reward/environment state. Changing one after the full run creates a
new run and training record; it never reinterprets the old result.

## Threshold classes

### Correctness invariants

These values follow from the design and do not wait for a diagnostic spike:

| Invariant | Required value | Failure action |
|---|---:|---|
| Explicit count coverage over every trainable generalization level | 100% | Block lambda selection and training |
| Fallback/default lattice-level count gradient mass | 0% | Block |
| Missing occurrence-to-decision mappings | 0 | Block |
| Non-monotone accepted level-count profiles | 0 | Block |
| Lambda-zero scalar reward selection versus pure utility | Exact identity | Block |
| Unsupported lambda profile used at training or inference | 0 | Block |
| Reward-cache pin mismatch | 0 | Invalidate cache and re-gate |

Numeric equality is not used to identify count defaults. Count admission is provenance- and
schema-based, so an explicit evidenced count numerically equal to `1` or `1000` remains valid.

### Feasibility gates

These decide whether the frozen environment contains enough variation to justify training:

- distinct `(U, P_count)` points per document;
- supported switch points for selecting lambda profiles;
- adjacent-profile winner-change rate;
- nonzero counterfactual utility differences;
- linked, residual, and complete-document fallback utility coverage;
- flat count menus and adjacent count-score separation;
- corpus/type/decision support at each lambda profile;
- injectivity collisions and lost future count opportunity.

Their boundaries depend on observed support and measurement resolution. The preflight spike
sets them before full RL.

### Statistical acceptance thresholds

These judge trained policies rather than environment feasibility:

- conditional-policy utility non-inferiority at lambda zero;
- conditioned-frontier comparison against fixed-condition controls;
- whole-task utility regression at matched realized privacy;
- document-bootstrap dominance or non-inferiority decisions;
- supported-profile responsiveness and collapse.

Every acceptance rule declares metric, paired sampling unit, confidence level, effect-size or
non-inferiority margin, minimum support, multiple-comparison treatment when applicable, and
failure action.

### Report-only diagnostics

These may motivate a future run but cannot alter the completed run's verdict:

- relationship between count score and held-out attacker success;
- type-normalization revisit signals;
- repeated-context leakage and occurrence multiplicity;
- count-gradient concentration by provenance/profile/type;
- proxy/attacker inversion;
- per-profile entropy and action-mode trajectories.

Held-out attacker results never tune count normalization, lambda values, or pre-training gates.

## Preflight diagnostic spike

### Purpose

The spike prices measurement resolution and support before expensive RL. It prevents invented
thresholds that are impossible, vacuous, or below the environment's noise floor. It does not
optimize a method or make privacy claims.

### Inputs

Use only frozen train/development artifacts:

- behavior-cloning and verified utility-only ExIt trajectories;
- rule-policy anchors;
- support-scan and cached rollout trajectories;
- cached adjacent-decision counterfactuals;
- frozen utility assertion vectors and occurrence-to-decision mappings;
- admitted explicit counts, count provenance, and type references;
- deterministic reader-refresh replicates used to estimate residual score jitter.

Final held-out documents and final attacker outcomes are forbidden inputs.

### Required measurements

For every document, corpus, runtime type, and supported decision class where applicable, emit:

- number of unique trajectories and unique `(U, P_count)` points;
- number and spread of nondominated points and positive lambda switch points;
- winner signatures across candidate lambda values;
- utility quantization step and reader-jitter distribution;
- linked/residual/fallback utility-assertion counts;
- policy decisions with no linked assertion;
- counterfactual `delta_U` zero rate, sign balance, and magnitude distribution;
- flat-menu rate, clipped count-score rate, and adjacent `delta p` distribution;
- collision event rate and lost count opportunity;
- support counts by corpus, type, profile, decision, and count provenance.

The spike output is immutable and content-addressed. The threshold manifest records its hash.

### Threshold-selection discipline

Before running the spike, predeclare for each unresolved threshold:

- measurement definition;
- candidate selection rule;
- allowed data split;
- minimum support calculation;
- deterministic tie handling;
- whether the result blocks, triggers an ablation, reduces scope, or only reports.

After the spike, instantiate the numeric values exactly once, record them in the training
record and threshold manifest, and freeze both before full RL. If the predeclared selection rule
cannot produce a value, stop and revise the design under a new diagnostic version.

## Diagnostic definitions and actions

### Lambda-menu feasibility

For document `d`, let `F_d` be its nondominated calibration trajectories. Report:

```text
n_points[d]      = number of unique (U, P_count) points
n_switches[d]    = number of retained positive adjacent-envelope switch points
winner_change[i] = fraction of eligible documents whose replay winner changes
                   between adjacent lambda profiles i and i+1
```

The existing `10%` adjacent winner-change rule is provisional until the spike prices the
deterministic reward-jitter floor and available switch-point spread. Its frozen replacement
must exceed the measured false-change floor and retain enough documents for each supported
corpus. The all-placeholder ceiling is likewise frozen after replay; an endpoint retained only
for diagnosis is labeled non-deployable.

If fewer than three distinct supported profiles remain after the declared replay and
replacement procedure, reduce the menu or stop. Never pad it with arbitrary values.

### Corpus and type support

“Support across every corpus and type” resolves to explicit minimum counts in the threshold
manifest:

- documents with at least two nondominated trajectories;
- controlled decisions exposed to the lambda profile;
- admitted profiles contributing count gradients;
- positive switch points;
- nonzero counterfactual comparisons where causal correction is claimed.

Types or corpora below minimum support remain in descriptive reporting but cannot support a
stratified training or generalization claim. They are not silently removed from aggregate
held-out evaluation.

### Injectivity interaction trigger

Measure both frequency and consequence:

```text
collision_rate = decisions whose legal menu changed because an earlier decision claimed a fill
                 / eligible later decisions

lost_count_opportunity = best p_j before dynamic masking - best p_j after dynamic masking
```

A privacy-return-to-go ablation triggers only under the frozen joint rule over collision rate
and lost opportunity. Frequent collisions with no score loss and rare severe collisions are
reported separately rather than collapsed into one rate.

### Count-signal health

Measure:

- fraction of decisions with no adjacent count-score separation;
- distribution of adjacent `delta p` by type;
- clipping rate at score one;
- expected count-gradient mass by type, profile, and provenance;
- concentration of gradient mass in the highest-contributing profiles;
- share of decisions carrying explicit model-proposed versus certifying counts.

Complete explicit count coverage is a hard gate. Compression, saturation, and concentration
thresholds are future-normalization triggers: they cause a declared profile-relative or
source-family-relative ablation in a new run, not retroactive score repair.

### Utility-credit mixture

At every epoch, report detached gradient norm and absolute weighted advantage mass from:

- linked utility credit;
- residual utility credit;
- complete-document fallback credit;
- bounded counterfactual pair credit;
- analytic count credit;
- entropy and KL regularization.

Stratify by corpus, runtime type, lambda profile, and linked-versus-uncovered policy-decision
status.
This report detects one estimator or coverage class dominating the heterogeneous credit mixture.
It is diagnostic unless a dominance threshold and response are frozen before the run.

### Counterfactual support

Report scheduler allocation, uniform-reserve coverage, cache-hit rate, eligible decisions never
measured, endpoint/direction balance, and `delta_U` distribution. Thresholds must distinguish:

- environment quantization producing true zero differences;
- insufficient measurement budget;
- a policy already confident on utility-equivalent actions;
- missing utility assertions for uncovered decisions.

The scheduler's priority score never scales reward or pair-loss magnitude.

### Conditional-policy responsiveness

For every supported lambda profile, report:

- fixed-document relative logits over identical legal menus;
- greedy count score;
- KEEP/generalization/placeholder rates by type;
- entropy trajectory;
- document/corpus/type exposure during the balanced schedule;
- utility and whole-task metrics.

Lambda-zero fixed-condition control is mandatory. The conditional model must satisfy a frozen
paired non-inferiority test against it. Additional fixed-lambda controls are certification
ablations; if absent, no claim is made that conditioning matches the separate-policy frontier.

### Frontier and utility acceptance

Every promotion rule records:

```text
sampling unit: document
comparison: paired at identical settings and matched realized privacy
confidence procedure: frozen bootstrap/test
confidence level: frozen before evaluation
utility metrics: frozen
non-inferiority or dominance margin: frozen
minimum supported documents: frozen
failure action: reject profile, reject checkpoint, or report unsupported
```

“Materially dominated” is forbidden in final adjudication unless replaced by this complete test
definition. Margins express the smallest operationally meaningful regression, not the observed
effect and not a per-model calibration knob.

## Threshold manifest

The pre-training run writes a content-addressed manifest with at least:

```yaml
diagnostic_version:
reward_version:
environment_hash:
span_decision_artifact_hash:
utility_component_artifact_hash:
count_artifact_hash:
diagnostic_dataset_hash:
spike_output_hash:

hard_gates:
  explicit_count_coverage: 1.0
  fallback_count_gradient_mass: 0.0
  missing_occurrence_decision_mappings: 0
  nonmonotone_profiles: 0
  lambda_zero_identity: exact

feasibility_gates:
  min_distinct_points_per_document:
  min_supported_documents_by_corpus:
  min_supported_decisions_by_type:
  min_adjacent_winner_change:
  max_flat_menu_fraction:
  collision_rate_trigger:
  lost_count_opportunity_trigger:
  min_nonzero_counterfactual_rate:

scheduler:
  call_budget:
  uniform_reserve_fraction:
  endpoint_fraction:
  direction_balance_tolerance:

acceptance:
  bootstrap_unit: document
  confidence_level:
  utility_metrics:
  utility_noninferiority_margin:
  minimum_supported_documents:
```

Empty fields are invalid for a run that depends on the corresponding decision. A manifest may
mark a field `report_only` only when the diagnostic cannot change that run's scope or verdict.

## Anti-calibration rules

The diagnostic process must not:

- inspect final held-out attacker results before thresholds and lambda profiles are frozen;
- choose different thresholds per competing model to make each pass;
- tune thresholds after full training results are visible;
- recalibrate count scores to force realized-privacy monotonicity;
- remove difficult corpora, types, profiles, or documents from aggregate evaluation;
- convert a failed hard gate into a warning without a new spec and training record;
- use a proxy threshold to claim privacy.

The preflight spike calibrates measurement and feasibility. It does not normalize methods or
equalize privacy.

## Required artifacts

- spike configuration and predeclared threshold-selection rules;
- immutable diagnostic output;
- completed threshold manifest;
- gate report with clause-level PASS/FAIL and support counts;
- per-epoch credit-mixture and conditional-responsiveness reports;
- final statistical adjudication report at matched realized privacy;
- companion training record linking every artifact hash.
