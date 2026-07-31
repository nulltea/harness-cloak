---
type: reference
status: current
created: 2026-07-12
updated: 2026-07-28
tags: [rl, ranker, reward-design, credit-assignment, decision-log, privacy, utility]
companion: [docs/specs/RL/interactive-ranker-v2.md,
            docs/specs/RL/interactive-ranker-v2-diagnostics.md,
            docs/specs/qa-builder-v2.md,
            docs/specs/RL/training-task-env.md]
---

# Interactive ranker v2 — design decision log

This log records consequential design forks for the interactive ranker v2. The companion
spec is normative once written; this document preserves rejected alternatives and the
reasoning behind each choice.

## Diagnostic thresholds are pre-registered experiment state

**Decision.** Every consequential term such as *material*, *supported*, or *dominated* resolves
to an exact invariant, frozen numeric threshold, frozen statistical test, or report-only
diagnostic. Exact correctness values are fixed directly. Empirical feasibility and acceptance
boundaries are instantiated once from a train/development preflight spike under predeclared
selection rules, then frozen before full RL and held-out evaluation. See
[interactive ranker v2 diagnostics](interactive-ranker-v2-diagnostics.md).

**Rejected alternatives.** Invent all empirical constants before measuring the environment;
decide materiality after seeing training or attacker results. The first can produce vacuous or
impossible gates; the second creates post-hoc discretion and invalidates adjudication.

## Count signal is experimental shaping, not a privacy measurement

**Decision.** Permit explicit model-proposed counts as an experimental training signal, but
remove fail-closed, legacy-default, missing-field, and generic-sentinel fallbacks from every
trainable lattice level. Count-derived reward must never be described as certified anonymity
or realized privacy. Held-out LLM re-identification remains the privacy verdict.

**Rejected alternatives.** Block all count-shaped training until certifying member-set counts
exist; permit fallback defaults as shaping values. Certifying-only counts would block the
current experiment rather than test explicit model-proposed ordering. Fallback defaults encode
missing-data conventions, not estimates, and let the policy learn artifact provenance.

**Required gate.** Before training or lambda-menu selection, every generalization level in every
ranker-controlled profile must have an explicit count and accepted provenance. Numeric value
alone is insufficient: an evidenced count of `1` or `1000` is valid, while an unprovenanced
fallback with the same value is not. Missing or rejected counts block the whole run rather than
silently dropping profiles or levels.

**Required reporting.** Results stratify count contribution by grounded versus explicit
model-proposed provenance. Generic placeholder remains a separate policy endpoint, not a
measured lattice count.

## Utility and count privacy use different credit paths

**Decision.** Optimize document task utility through sampled episodic credit, while optimizing
the analytically known count score through an exact local expected-action objective. Do not
collapse both terms into one document scalar before assigning gradient.

For rollout group member `g` and span `s`, the intended loss shape is:

```text
L = -(1/G) sum_g,j A^U[g,j] log pi(a[g,j] | h[g,j])
    -(lambda/G) sum_g,j (1/|D_d|) sum_a pi(a | h[g,j]) p_j(a)
    - beta H(pi) + eta KL(pi || pi_ref)
```

The count term supplies exact local credit over every currently legal action. Utility remains
episodic because the remote task consumes the complete rewritten document and span actions
interact through semantics, generation behavior, and the sequential injectivity mask.

**Rejected alternatives.** Broadcast a combined document reward to all selected actions;
train independent per-span utility rewards. The first wastes known count attribution and
raises variance. The second optimizes the wrong objective by deleting cross-span context.

## Additive privacy weighting

**Decision.** Express the objective as utility plus an additive privacy-shaping weight. The
equivalent leakage-penalty form may be used in explanations, but the implementation keeps the
utility scale fixed:

```text
J = U + lambda P_count
L_count = 1 - P_count
J = U - lambda L_count + constant
```

`lambda` is a fixed, pre-registered operating-point parameter per run. It is swept to produce
candidate policies; realized privacy is measured independently and comparisons are made only
at matched realized privacy.

**Rejected alternative.** `(1-lambda) U + lambda P`. For `lambda < 1` this is the additive
objective scaled by `(1-lambda)`, with effective privacy weight `lambda/(1-lambda)`. It adds no
frontier points, compresses the useful range near one, and changes the relative scale of
entropy, KL, and other trainer terms unless every term is rescaled consistently.

## One frozen span artifact serves QA construction and RL

**Decision.** Run the composed detector once, freeze the resulting spans with stable IDs, and
make both QA construction and ranker training consume that exact artifact. The intended
detector is Presidio plus `knowledgator/gliner-pii-large-v1.0`, subject to the detector spec's
version and overlap-resolution pins.

Using the same detector configuration in two independent runs is insufficient: preprocessing,
overlap resolution, model revisions, or ID assignment could drift. Detection errors remain
common-mode limitations and must be reported rather than interpreted as annotation truth.

**Rejected alternative.** Independently re-detect spans during QA construction and ranker
environment construction. This can produce probes linked to decisions absent from the policy
action space.

## Probe links are routing hints, corrected by counterfactuals

**Decision.** Treat each probe's `span_ids` as a credit-routing hyperedge, not causal ground
truth. Singleton ladder probes route cheaply to one span; decision probes may route to several;
unlinked and document-global criteria retain trajectory-level credit. Sparse contextual
counterfactuals replace provisional linked credit for the tested span.

The QA builder should eventually emit explicit stable IDs plus supporting quotes. Canonical
substring matching remains a migration fallback, not the target interface.

**Rejected alternatives.** Treat `span_ids` as hard attribution; discard probe links and rely
only on document RLOO plus counterfactual calls. Hard attribution converts teacher and detector
errors directly into gradient errors. Counterfactual-only attribution is honest but too costly
for dense training.

## Decisions without accepted probes retain utility credit

**Decision.** A span with no accepted linked probe must not receive zero utility credit. QA
coverage is currently incomplete and rejection-heavy; absence of a probe is not evidence that
the span is irrelevant to the task.

Linked probe components route focused credit. Unlinked probes, schema/coherence criteria, and
other document-global components retain shared trajectory credit. The reward assembly must
partition components so linked probe scores are not counted again through a duplicate complete
document reward.

**Rejected alternative.** Update only spans with accepted linked probes. This would train the
ranker around QA-builder coverage artifacts and systematically neglect difficult or rejected
spans.

## ExIt is a utility-only initialization stage

**Decision.** Retain expert iteration as a coarse warm start selected only by document task
utility. For each document, sample complete trajectories, verify that the best candidate
strictly beats the behavior-cloning reference under serial cache-bypassed scoring, and clone
the complete verified winner. Count shaping does not participate in ExIt selection.

ExIt labels are intentionally approximate: a winning trajectory may contain passenger
actions. The subsequent hybrid RL stage is responsible for correcting them through linked
probe credit, document-global residual credit, and sparse contextual counterfactuals. ExIt is
not presented as action-level causal supervision.

**Rejected alternatives.** Counterfactual-filter every ExIt winner before cloning; remove ExIt
and begin hybrid RLOO directly from behavior cloning. Filtering would improve labels but adds
one or more round trips per tested span and complicates partial winner reconstruction. Direct
RLOO is cleaner but reopens the quantized-reward and tied-group failure that motivated ExIt.

## Counterfactuals compare adjacent actions and periodically test endpoints

**Decision.** For a selected span action, the default counterfactual changes only that span to
one adjacent lattice action. When both directions exist, the scheduler alternates or balances
finer and coarser comparisons. A fixed minority of tests compare against KEEP or placeholder
so the learner remains anchored to the action menu's semantic-retention and deletion
endpoints.

The resulting difference reward measures the local contextual utility slope while holding all
other span actions fixed. It can distinguish KEEP from fine generalization and locate where
additional count-shaped privacy begins to cost task utility.

**Rejected alternatives.** Placeholder-only counterfactuals; enumeration of every legal
alternative. Placeholder-only credit tests semantic preservation versus deletion but cannot
resolve neighboring lattice levels. Full enumeration gives the complete contextual landscape
but multiplies round-trip cost by the action-menu width.

## Counterfactual utility uses bounded in-place pair substitution

**Decision.** A tested rollout-decision pair replaces its provisional REINFORCE term at the same
outer `1/G` weight. It is not collected into a separately averaged counterfactual loss, because
that would make each correction's strength depend on the number of calls in the scheduler
budget.

For selected action `a` and measured alternative `b`, restrict the policy to the pair,
`q = pi(a) / [pi(a) + pi(b)]`, and optimize the measured local expected utility with the
centered loss `-delta_U * (q - 1/2)`. This is bounded and saturates with confidence.

**Rejected alternatives.** `mean(L_cf_tested)` after the provisional loss; the unbounded
weighted log-ratio `-delta_U [log pi(a) - log pi(b)]`. The first creates budget-dependent
gradient scaling. The second pushes pairwise logits without saturation and relies on entropy or
KL to contain a noisy comparison.

## Counterfactual budget follows coverage and uncertainty

**Decision.** Allocate a fixed counterfactual-call budget with a priority scheduler. Highest
priority goes to ranker-controlled decisions with no accepted linked probes across their
occurrences, followed by uncertain or multi-decision links, high policy entropy, and unresolved adjacent-action utility slopes. Keep a
fixed uniform exploration reserve so every eligible span retains nonzero inspection
probability.

Priority changes only which spans receive expensive causal measurement. It must never scale
the resulting utility difference or pairwise-loss magnitude; QA coverage, link confidence, and
policy entropy are scheduling signals, not hidden reward weights.

**Rejected alternatives.** Uniform-only sampling and a fixed fraction per document. Uniform
sampling wastes calls on well-understood decisions and may rarely inspect uncovered spans. A
per-document fraction equalizes documents rather than information value and still requires a
within-document selection rule.

## One policy supports a finite menu of lambda profiles

**Decision.** Train one lambda-conditioned ranker checkpoint on a fixed, pre-registered menu
of three to five supported lambda values. Lambda is fixed for a complete document rollout
group. Training uses a seeded balanced schedule so every document is exposed to every profile
across epochs and every profile receives comparable document coverage. No behavior is promised
between supported values.

The menu always contains `lambda = 0` as the utility-only anchor. Remaining values are selected
before training from a frozen calibration pool of trajectories with separately stored utility
and analytic count scores. Dominated trajectories are removed; utility-versus-count switch
points are computed within documents; active switch regions are weighted by how many decisions
they change; representative settings are selected and replayed. Redundant settings and settings
that cause unintended all-placeholder collapse are rejected before the training record is
frozen. The normative spec must define this protocol algorithmically, including provenance,
tie handling, clustering, minimum-support gates, and freeze rules.

**Architecture amendment (2026-07-22).** The finite numeric menu and balanced exposure remain
approved. The later architecture decision removes the supported-profile identity embedding:
lambda enters only through the explicit additive controller in
`ranker-v2-architecture.md`. No behavior is promised between supported numeric values.

**Rejected alternatives.** Ship one checkpoint per lambda; expose a continuous lambda slider.
Separate checkpoints are expensive to train and operationally awkward. A continuous slider
requires reliable interpolation at values that were not directly trained or validated.

## Conditional initialization preserves the utility warm start

**Superseded architecture detail.** The original decision initialized FiLM and profile embeddings
to preserve the warm start. The selected additive controller removes both. Lambda-zero equality is
now an exact structural identity, and ExIt clones a winner once into the shared utility policy.
The retained decision is utility-only initialization, not the obsolete conditioning mechanism.

**Rejected alternative.** Train BC/ExIt through only one randomly initialized profile path and
leave other profile embeddings untouched until hybrid RL. That would confound conditional
interference with initialization damage and discard the support-preserving warm start.

## Count scores are normalized within runtime type

**Superseded for the selected semantic prototype; retained for the direct-count fallback.** The
original decision converted an action's count into a bounded score using a fixed reference count for
its runtime type:

```text
p_j(a) = clip(log(max(K_j(a), 1)) / log(K_ref[type(j)]), 0, 1)
```

KEEP scores zero and placeholder scores one. `K_ref` is frozen before lambda-grid selection
and training. Prefer a grounded type-universe size when one exists; otherwise use a
pre-registered robust statistic over valid non-placeholder counts for that type. The exact
reference statistic, minimum support, clipping policy, and flat-type behavior belong in the
normative spec. Runtime lookup cannot manufacture a shaping value: the pre-training artifact
gate proves complete explicit coverage first.

Type normalization prevents naturally larger count universes such as drugs from dominating
smaller universes such as medical procedures while preserving count magnitude across profiles
within one detector/runtime class.

**Rejected alternatives.** One global count denominator; profile-relative normalization.
Global normalization compares incompatible type universes. Profile-relative normalization
insulates inconsistent profiles but automatically assigns the coarsest action in every shallow
profile the maximum score and renormalizes old actions whenever that profile gains a broader
level.

### Revisit triggers

Type normalization is provisional and must be revisited if any of these signals appears:

- **Within-type compression or saturation.** A material fraction of trainable profiles has no
  useful adjacent score separation because all actions cluster near zero or one. The gate must
  report per-type adjacent `delta p` distributions and the fraction of flat action menus.
- **Profile-scale domination.** A small subset of profiles supplies most of the count-gradient
  magnitude or consistently wins despite weak realized privacy. Report count-gradient mass by
  type, profile, and count provenance.
- **Within-type attacker mismatch.** At fixed documents/settings, higher type-normalized scores
  fail to associate with lower held-out attacker success, or the association is negative on a
  sufficiently supported type. This is a diagnostic trigger, never a basis for claiming
  privacy from the count score.
- **Type-specific policy collapse.** Increasing lambda drives one type to placeholder/coarsest
  actions while other types remain unresponsive, without a corresponding realized-privacy
  improvement. Report action-mode and selected-score curves separately by type.
- **Count-source divergence within a type.** A type acquires multiple source families whose
  count distributions or attacker correlations differ materially, invalidating the assumption
  that one type reference is coherent.
- **Alternative replay wins.** On the same frozen trajectory pool, profile-relative or
  source-family-relative normalization yields materially better supported ordering against the
  held-out attacker and avoids the degeneracies above. Any switch requires a new reward version,
  lambda selection, gate run, and training record; it is never applied retroactively.

The first response to a trigger is diagnosis and an ablation, not automatic renormalization.
Degenerate behavior under the type-normalized fallback remains a reportable result. The selected
semantic prototype instead uses own-profile log-count normalization for both privacy-head
supervision and the exact local target, as recorded in the architecture decision log.

## Document count score is an equal mean over controlled decisions

**Decision.** Aggregate type-normalized action scores with equal weight over every unique
ranker-controlled decision in the document:

```text
P_count(d, a) = (1 / |D_d|) sum_j p_j(a_j)
```

Rule-masked PERSON/CODE decisions and any other actions outside the ranker's control are
excluded so constants do not dilute the gradient. Distinct controlled values of the same type
contribute separately; repeated occurrences of one normalized value do not. Report contribution
by type and decision, and report repeated/overlapping occurrences separately.

**Rejected alternatives.** Equal weight per represented type; occurrence-weighted decisions.
Type balancing understates multiple distinct attributes of one type. Occurrence weighting makes
textual repetition and detector fragmentation alter effective lambda.

## Detector occurrences map many-to-one onto policy decisions

**Decision.** Detector outputs remain occurrence-level records with stable span IDs and offsets.
The frozen artifact additionally maps them to one policy decision per
`(document, runtime type, normalized canonical profile/surface)`. The ranker samples one action,
stores one log-probability, and applies one fill consistently across every occurrence mapped to
that decision. Occurrence-linked QA components are unioned at the decision; a counterfactual
changes all mapped occurrences together.

`P_count` is an equal mean over unique controlled decisions, not occurrences. Repeating the same
normalized value therefore does not duplicate its count score or count gradient. The stable
decision ID must include runtime type; the current `surface.lower()` key is insufficient for
ambiguous strings detected under different types.

**FORMAL-AUDIT FLAG — repeated-context leakage is real. Repeated mentions of the same value in
different contexts can increase inference and linkage risk. The unique-decision count mean
deliberately assumes that occurrence multiplicity should not change count-shaping weight. This
is an accepted approximation for the current design, not a claim that repetition is harmless;
future formal privacy audit must test occurrence-aware attacks and compare unique-decision,
occurrence-weighted, and contextual leakage objectives.**

**Rejected alternatives.** One independent action per occurrence; occurrence-multiplicity
weighting of the shared decision. Independent actions require a major rewrite of assembly,
injectivity, and reconstruction and can produce inconsistent substitutions for one value.
Multiplicity weighting makes transcript verbosity and detector fragmentation silently change
effective lambda even though the count describes one distinct attribute value.

## Linked and global utility components are partitioned without duplication

**Decision.** For a span with accepted links, linked probes contribute span-routed utility
credit and genuinely unlinked/document-level criteria contribute shared global credit; the
complete document score is not added again. A span with no accepted linked probe falls back to
the complete-document utility advantage, so missing QA coverage never means zero utility
credit. When a contextual counterfactual is evaluated, its causal pairwise term replaces the
approximate provisional policy-gradient term for that tested span and rollout.

**Rejected alternatives.** Blend linked credit with a weighted copy of the complete document
advantage; broadcast only the complete document advantage. Blending double-counts linked
probes and introduces another sensitive coefficient. Complete-document-only credit discards
the approved probe-routing information and restores the current passenger-action problem.

## The v16 assertion artifact is the live QA-to-RL contract

**Decision.** Selectively port the contract-alignment changes from the clean branch: round-trip
utility is an assertion-ID score vector, and every assertion subset is divided by the document's
fixed stored `utility_weight_denominator`. Missing families do not change that denominator. The
artifact partitions policy decisions from fixed rewrite decisions, maps occurrences to either
kind or null, and stores each assertion's unique policy dependencies plus `linked` or `residual`
routing. Linked means at least one policy dependency. Global assertions and fixed-only links are
residual; mixed fixed/policy hyperedges route once to each unique policy dependency and never
give a gradient to a fixed decision. This entry supersedes this log's earlier probe-component
and linked/global wording for the live ranker-v2 contract.

Runtime scoring preserves per-assertion question, clause, excerpt, answer-kind, and cache
identity while allowing one scorer submission to schedule a bounded work queue across a rollout
batch. That batching does not promise one reader transport call or model generation per rollout;
only a separately validated multi-question reader could establish that property.

The counterfactual scheduler uses, in order after its uniform reserve: no linked assertion,
multi-decision (hyperedge) links, high policy entropy, an unseen adjacent pair, and the oldest
measured pair. The v16 artifact stores no dependency-confidence scalar, and none is derived from
provenance. This scheduler rule supersedes the earlier link-confidence wording in this log.

**Rejected alternatives.** Renormalize only the families present in a document; give gradients
to fixed decisions; treat fixed-only links as linked policy credit; invent dependency confidence
from provenance; merge the divergent branch wholesale. The first three change the reward or
policy routing, the fourth fabricates absent artifact state, and the last imports unrelated drift
instead of the reviewed contract corrections.

## OPEN FORK — objective normalization mix across document sizes (2026-07-28)

**The fork.** The hybrid group objective mixes normalizations: utility, entropy, and KL
are decision-summed; the count term is decision-averaged (×λ/D/rollouts). Analysis of
gradient routing sharpens the issue: all decision-summed terms feed the utility tower
AND alpha, while the count term feeds ONLY alpha — so the tower's internal term ratios
are already composition-invariant, and the sole pathology is that **alpha's two opposing
pressures scale differently with document decision count D** (utility pull ∝ D, count
pull D-invariant; D spans 4–24, a ~6× ratio drift). Consequences: lambda's realized
per-decision pressure depends on document composition, and alpha's equilibrium
over-weights long documents' utility side. Adam absorbs across-step scale, not
within-step term ratios on the shared scalar.

**Candidate designs (spike arms).**
1. *Current mix* (baseline / possible no-op if measured D-dependence is negligible).
2. *Average-all*: divide utility/entropy/KL by D too. Fixes the alpha ratio; also
   rescales tower gradients per document and perturbs λ=0 dynamics.
3. *Sum-all*: drop the count term's /D. Fixes the alpha ratio with both pulls ∝ D
   (long documents vote proportionally on alpha); λ=0 pathways byte-identical.
4. *Alpha-channel routing* (Codex Sol recommendation, 2026-07-28): numerically
   identical forward pass and logged losses via a stop-gradient identity
   z̃ = u + sg(c) + (c − sg(c))/D for the utility/entropy/KL logits, count logits
   unchanged — alpha's utility-side gradient becomes decision-averaged, matching the
   count side; the tower keeps full document-level gradients; λ=0 identity and the
   frozen calibration replay are untouched. Lambda's meaning becomes a
   document-composition-stable per-decision privacy preference.

**Preregistered decision protocol (user-selected primary criterion: alpha-pressure
length-invariance; stability battery as hard gate).** Spike: 4 arms × 8 cached
documents stratified over D (2 each in 4–7, 8–12, 13–18, 19–24, mixed task families)
× 12 epochs (three full 4-profile Latin cycles) × 3 seeds, identical initial
checkpoints and seeded document ordering; fully cached reward. Per group, record
signed/absolute alpha gradients per term family, alpha trajectory by λ and D bin,
per-profile mode rates, mean profile-relative privacy, document utility, and exact
λ0-control divergence. Primary diagnostic: regress
log(|g_α,U + g_α,H + g_α,KL| + ε) − log(|g_α,P| + ε) on log D over nonzero-λ groups;
the desired normalization has slope b_D ≈ 0. Decision rule: adopt arm 4 iff it
(i) cuts |b_D| ≥ 50% vs current and lands |b_D| ≤ 0.25, (ii) same-λ mean privacy
differs ≤ 0.05 between short and long bins, (iii) mean document utility non-inferior
within the measured 0.044 reader-noise floor, (iv) λ ordering stays monotonic in
realized privacy, (v) λ0 divergence exactly zero, (vi) no seed shows collapse or
non-finite gradients. Fall back to average-all only if routing passes controller
checks but leaves a reproducible tower-side length pathology; retain the current mix
only if its measured D-dependence is already negligible; sum-all is a negative
control unless it uniquely avoids long-document privacy inflation.

**Status: CLOSED 2026-07-28 — current mix retained (null adjudication).** Two
measurement rounds (frozen-state, then the approved training-dynamics revision:
2 arms × 4 length-binned docs × 8 epochs × 8 rollouts, identical seeds): the b_D
diagnostic is unmeasurable — current-mix slope flipped sign across rounds
(+0.45 → −0.71) at R² ≤ 0.04 with 9/16 groups advantage-degenerate, and the
routing arm measured −1.70 (outside ±0.25). Both arms were behaviorally identical
to four decimals (α moved ~1e-3 over 8 epochs; routing's relative count-pressure
gain visible only as α_end 1.0012 vs 1.0008). Preregistered fallback applied:
retain the current mix. The alpha-routing implementation stays as tested,
default-off infrastructure (`--alpha-utility-routing per-decision`).

**Discoveries superseding the fork.** (1) The α utility-pull is a noise-dominated
covariance (diffusive ~√D), so no fixed-exponent renormalization is identifiable —
the theoretical D-scaling describes only the signal component, which is negligible
at reachable scales. (2) The binding constraint is CONTROLLER STRENGTH, not
normalization: realized privacy is ~0.14 at every λ including λ3 (whose menu
replay shows 88% placeholder), because α·g(λ)·privacy at α≈1 cannot move
BC-sharpened utility logits, and α's learning timescale (~1e-3 / 8 epochs) is
hundreds of epochs. Same-λ short-vs-long privacy gaps (0.07–0.15) are
composition-driven (identical across arms). SUCCESSOR FORK to open before
full-corpus runs: α initialization / parametrization / count-loss scale.
Artifacts: results/ranker_v2/architecture/objective-normalization-*.json;
spike scripts/spikes/objective_normalization_spike.py.

## OPEN FORK — controller strength: switch-calibrated, gap-scaled alpha (2026-07-28)

**Problem (measured, normalization spike).** At alpha=1 the controller shift
alpha·g(lambda)·p_hat cannot move BC-sharpened utility logits: mean selected count
score is ~0.135–0.15 at every profile including lambda-3; alpha drifts ~1e-3 per
8 epochs. Lambda is behaviorally disconnected at trainable timescales.

**Preregistered design candidates (Codex Sol consultation, session 019f8fa3).**
Arms: (1) current — raw controller, alpha=1; (2) init-only — raw controller,
alpha initialized to the weighted median of raw switch thresholds
t_j = min over more-private actions of (u(a*)−u(a))/(p(a)−p(a*)) over BC menus,
each document contributing total weight 1; (3) gap-scaled — controller multiplied
by the detached per-menu utility-logit range s_j (scale-aware: the tower cannot
defeat the controller by sharpening logits), alpha initialized to the weighted
median of s_j-normalized thresholds. g(lambda-3)=1 (lambda-3 is max), so no
g-correction is needed in either init.

**Spike.** 4 composition-diverse docs (D=4/12/18/22) × 12 epochs (3 Latin cycles)
× 8 rollouts; screening seed 17, then current + best responsive arm on 2
confirmation seeds. Metrics per lambda and document: selected count score P;
paired within-document Delta P_d(lambda) = P_d(lambda) − P_d(lambda-0) per cycle;
mode rates; document utility; utility regret vs the cached (U,P) upper frontier;
utility-logit range and controller-to-gap ratio; alpha trajectory; entropy;
lambda-zero divergence; duplicate-rollout/degenerate-advantage rates.

**Preregistered passing rule (all across confirmation seeds).** (1) lambda-zero
exactly identical; (2) mean selected P non-decreasing across profiles; (3) at
least two adjacent profile gaps ≥ 0.05; (4) P(lambda-3) − P(lambda-0) ≥ 0.20;
(5) ≥75% of document-seed pairs show nonnegative paired lambda-3 movement;
(6) lambda-3 placeholder rate < 95%; (7) median utility regret vs the cached
frontier ≤ 0.044 (the measured reader-noise floor); (8) no non-finite values,
uncontrolled alpha growth, or cycle-to-cycle mode oscillation. If init-only and
gap-scaled both pass, prefer gap-scaled unless raw responsiveness is stable
across BC-logit-range strata.

**Artifact consequences (preregistered).** Lambda reward values stay frozen for
the spike; behavioral menu claims require revalidation; a new threshold-manifest
version precedes production RL; the KL reference must be regenerated under any
adopted controller (an alpha=1 reference would pull a calibrated controller back
to the weak regime); BC checkpoints remain valid with alpha deterministically
reset after import. Terminology: the readout is the SELECTED COUNT SCORE — a
shaping proxy; realized privacy remains attacker success and no privacy claim
follows from this spike.

**Adjudication 2026-07-29.** Screening + 3-seed confirmation + approved 24-epoch
extension. RESPONSIVENESS SOLVED: switch-calibrated initialization (measured raw
threshold median 7.13; gap-normalized 0.73) revives the controller on every seed —
selected count score monotone across profiles, Delta P(lambda-3) = +0.22..+0.41,
placeholder rate <= 7% (privacy from generalization levels, not collapse), alpha
stable, lambda-zero utility preserved; the current alpha=1 arm stays flat on every
seed. Preregistered preference selected gap-scaled; items 1-6 and 8 pass on all
seeds at 12 and 24 epochs. ITEM 7 FAILS DEFINITIVELY: median utility regret vs the
cached (U,P) frontier plateaus at 0.063-0.084 (floor 0.044), flat from cycle ~4 —
not under-training. Interpretations, in decreasing likelihood: (a) the spike loop
omits counterfactual credit and KL — the production mechanisms built for
frontier-tracking — so item 7 was evaluated under a weaker optimizer than the one
that would ship; (b) the frontier is exploration-enriched (every arm's samples,
~2.3k vectors) — an in-hindsight bar; (c) a genuine limit of one global scalar
alpha. Artifacts: results/ranker_v2/architecture/controller-strength-*.json
(12ep snapshots + 24ep extension).

**Status.** OPEN — responsiveness component (switch-calibrated init + gap scaling)
is the adopted CANDIDATE; final adoption gated on re-evaluating item 7 under the
production trainer (counterfactual credit + KL enabled) rather than the simplified
spike loop. If it fails there too, iterate on per-decision credit, not on alpha.

## OPEN FORK — small-document credit support: rollout collapse kills LOO utility credit (2026-07-29)

**Problem (measured).** Seed-to-seed spread of the controller's per-document
lambda response localizes entirely to small documents: final-cycle
Delta P(lambda-3) across the three 12-epoch gap-scaled spike seeds is
+0.13/+0.61/+0.35 on D2N005 (D=4) and +0.05/+0.31/+0.25 on D2N027 (D=12), vs
seed-invariant +-0.03 bands on D2N031 (D=25) and D2N063 (D=19). The production
seed-17 run shows the same behavior (D2N005 lambda-3 collapses to utility 1.00 /
count score ~0.10 by cycle 1). Direct diagnostics from the production epoch
reports: D2N005 groups average 2.58 unique action vectors of 8 rollouts with
5/12 groups FULLY degenerate (all 8 rollouts identical); D2N027 3.67 unique,
4/12 fully degenerate; large docs 5.4-6.0 unique, 0 fully degenerate. A fully
degenerate group has every leave-one-out advantage exactly zero — the
provisional utility gradient is dead, the exact count gradient reaches only
global alpha, entropy (beta=0.01) is negligible, and the KL collapse trigger is
aggregate-level and blind to per-document collapse.

**Causal chain (joint adjudication, this session + Codex Sol High session
019f8fa3).** Small D -> low trajectory support under the BC-sharpened policy
(implied per-rollout dominant-vector probability ~0.90 on D2N005) -> duplicate
rollouts -> dead or noisy LOO credit -> the surviving rare vectors make early
utility updates a coin flip -> self-reinforcing sharpening -> seed-specific mode
lock. This is mode SELECTION by early luck, not a small-document bias. Ranked
contributors: (1) trajectory collapse (dominant, measured); (2) reward-effective
diversity below action diversity — distinct vectors with identical assertion
scores or differences under the 0.044 reader-noise floor; (3) estimator support
~sqrt(D) fewer credit-bearing terms per update; (4) linked-mass/sensitivity
heterogeneity (D2N005 linked assertions per decision [1,2,2,11]); (5) global KL
trigger misses local collapse (amplifier). REFUTED as a cause: global-alpha
starvation — the switchable-decision fraction at the calibrated alpha is ~50% on
every doc (2/4, 7/12, 10/21, 9/18). NOT reopened: objective normalization
(closed fork; this is an estimator-support problem, not a normalization one).

**Preregistered spike (staged, production trainer, 4 spike docs).** Arms are
isolated interventions over the completed 3-seed 12-epoch production baseline:
- Arm R — support-scaled rollouts (formula corrected pre-implementation:
  the initially logged ceil(64/D) was arithmetically inconsistent with its own
  worked example): per group, compute the dominant-trajectory probability
  p_hat exactly (product of per-decision max probabilities along the greedy
  legal walk under the current policy and profile) and set
  R = clamp(ceil(log(0.05)/log(p_hat)), 8, 32) — i.e., enough rollouts to hold
  the fully-degenerate probability under 5%, capped. Self-tuning per document
  and epoch; measured p_hat~0.90 gives D2N005 R=29-32, diverse large docs stay
  at 8. Duplicate vectors are cache-identical, so extra rollouts cost almost no
  new scoring.
- Arm C — counterfactual dedup + broadcast + degeneracy-triggered coverage:
  budget counts unique (vector, decision, alternative) interventions; a measured
  Delta U broadcasts its substituted pair loss to every rollout with the
  identical complete vector (exact, zero extra calls); when unique vectors <= 3,
  cover every eligible decision of the dominant vector; budget 5*ceil(D/5)
  capped at 15 via a spike-labeled threshold-manifest version (base stays 5).
- Staging: both arms 8 epochs on seed 17 (mechanism screening) -> the arm(s)
  passing mechanism checks run 12 epochs on seeds 29/47 (+ seed-17 12-epoch
  completion) for the behavioral gates.
- Mechanism checks: Arm R — small-doc fully-degenerate rate <= 10%, median
  unique vectors >= 4, reward-distinct vectors >= 2x baseline. Arm C — raw
  degeneracy unchanged, utility-dead groups after substitution <= 10%, >= 75% of
  small-doc decisions receive a counterfactual per Latin cycle.
- Behavioral gates (3 seeds): small-doc Delta P(lambda-3) sample SD <= 0.07 and
  cross-seed range <= 0.15; count-responsiveness items of the controller fork
  still pass; conditional lambda-zero utility within 0.044 of the fixed
  control; no placeholder collapse or non-finite values. Frontier regret
  (item 7, <= 0.044) is reported per arm and adjudicated for the controller
  fork, not this one.
- Readouts recorded per group: unique action vectors; unique component-score
  vectors (reward-distinct); utility-dead groups before/after counterfactual
  substitution; per-decision probe coverage and |Delta U| vs the 0.044 floor;
  gradient norms by term family; paired Delta P; regret; lambda-zero identity.

**Interactions.** The item-7 24-epoch extension of the controller fork runs on
whichever credit configuration this fork adopts — a dead-credit trainer
under-uses added epochs. Production-readiness note independent of arms: the KL
collapse trigger should gain a per-document condition (tracked, not an arm).

**Mechanism screening adjudicated 2026-07-29 (seed 17, 8 epochs, matched
baseline window).** Arm R FAILS: it eliminates fully-degenerate groups (25% ->
0% on both small docs, mean 14 rollouts on D2N005) but the extra rollouts are
reward-identical — reward-distinct vectors 1.06x/1.14x baseline against the
preregistered >= 2x bar, and D2N005 median unique vectors 3.0 < 4. The
preregistered diagnostic branch fires: the bottleneck is reward-effective
diversity (assertion sensitivity), not rollout count — sampled-diversity
approaches cannot fix small documents. Arm C PASSES: raw degeneracy comparable
(12% vs 25%), utility-dead groups after substitution 0/16 (bar <= 10%),
per-decision probe coverage 100% on both small docs at steady state (cycle 1;
cycle 0 is the empty-history ramp: 50%/25%), 293 broadcast pair losses + 674
deduplicated duplicate probes over 8 epochs. Probe signal is live where it
matters: measured |Delta U| on D2N005 reaches 0.273 with 36% nonzero probes
(large docs invert: 90% nonzero but small magnitudes — per-decision utility
share scales inversely with D).

**Stage 2 adjudicated 2026-07-30 (Arm C, 12 epochs x seeds 17/29/47).**
Mechanically clean on all seeds: TRAIN PASS, lambda-zero identity exact,
monotone profile response, conditional lambda-zero utility within the 0.044
floor of the fixed control (gaps 0.003/0.041/0.040). D2N027 CLOSES: final-cycle
Delta P(lambda-3) = +0.10/+0.14/+0.07, SD 0.036 (bar 0.07), range 0.072 (bar
0.15) — versus baseline SD ~0.14. Frontier regret tightens to 0.064-0.069
across seeds (baseline 0.067/0.072) but stays above the 0.044 item-7 floor.
D2N005 FAILS the primary gate: +0.42/+0.29/+0.05, SD 0.190, range 0.374 — the
seed lottery on the smallest doc survives exact per-decision credit.

**Post-hoc sensitivity measurement (preregistered next probe).** From cached
single-decision pairs: D2N005's reward surface is LIVE on 3 of 4 decisions
(74-90% nonzero utility spans, medians 0.07-0.27, well above reader noise) and
PERFECTLY FLAT on one (38/38 contexts span exactly 0.000 — the utility
artifact never distinguishes its actions). D2N027 shows a gradient of
sensitivity (medians 0.000-0.175). Residual D2N005 variance is therefore NOT
dead credit: with D=4, each decision's lambda-3 switch is a discrete ~0.25
quantum of P, and whether the calibrated GLOBAL alpha crosses a given
decision's switch threshold drifts with per-seed logit sharpening — a few
discrete coin flips dominate the doc-level Delta P. Large D averages these
flips out (D2N027/31/63 stable); D=4 cannot.

**Root-cause investigation 2026-07-30 (systematic-debugging, this session +
independent Codex Sol High pass; full convergence).** The lambda-3 coin flip
is a deterministic property of the learned logits, not sampling (fixed-
checkpoint replay, 128 trajectories: final E[P|lambda-3] = 0.498/0.271/0.084
for s17/s29/s47; 8-rollout SE 0.003-0.017). Causal chain, each link measured:
(1) D2N005's reward surface is tie-dominated at the top of every menu —
U(L0) == U(keep) EXACTLY in 113/113 cached single-decision contexts, and one
decision is flat across its whole menu — so utility credit cannot order the
lambda-3-relevant choices (level-0 generalization is utility-FREE privacy the
policy leaves on the table). (2) The tie is filled by cross-document
generalization through shared parameters: at BC init keep-L0 margins are
NEGATIVE (-5.6..-7.6, BC clones levels); after RL they are +41..+225 with the
same direction on every seed — the corpus-wide truth "keep is utility-safe"
leaks into pairs where this document's reward is silent. (3) Unbounded softmax
sharpening (driven by the real signal punishing deep levels; no gradient
clipping — max_grad_norm=null; entropy 0.01 negligible) amplifies the
inherited margin without limit: seed-47 logit ranges exploded 9 -> 277-327,
saturating the distribution and killing entropy/count/forward-KL gradients.
(4) The controller's bounded shift alpha*dp*range (<= ~43% of range for
keep->L1) races the amplified margin fraction — crossings = cycle flips,
runaway = seed-47 all-keep collapse. (5) Nothing can push back: the count
gradient reaches ONLY global alpha by design, and the KL trigger is
aggregate-level (never fired) with a forward direction whose gradient
vanishes exactly when needed. Codex additionally showed the calibrated KL
REFERENCES are healthy anchors (E[P|lambda-3] ~0.66-0.67, ranges 9-15) that
the trigger never uses, and flagged a measurement confound: Latin-cycle
Delta P mixes lambda-conditioning with between-epoch policy evolution —
synchronous fixed-checkpoint evaluation is the correct instrument. REFUTED:
profile count targets (monotone, well-separated, no inversion — audited),
alpha-calibration staleness as primary (calibration is correct; margins are
re-shaped after it), and sampling noise. Verdict: SYSTEMIC (tie-breaking is
undefined and drifts keep-ward; sharpening unbounded; count bottlenecked
through one scalar) exposed maximally by D2N005's tie/saturation structure.

**Proposed next spike (awaiting approval).** Always-on low-weight KL to the
calibrated reference from epoch 0 (two arms: current forward direction
KL(pi||ref) vs reverse KL(ref||pi), eta=0.01 — reverse keeps a nonzero
~(pi-ref) gradient under saturation), seeds 17+47, 12 epochs, Arm C
broadcast active, no clipping in the same runs (separate safety ablation
later). Synchronous per-cycle 4-profile evaluation of the SAME checkpoint as
primary readout. Preregistered pass: D2N005 synchronous
E[P|l3]-E[P|l0] >= 0.20 every completed cycle, lambda-3 exact-P cycle range
<= 0.10, cross-seed final difference <= 0.10, reward-flat decision expected
p >= 0.50, regret <= 0.044 unchanged as the item-7 gate, lambda-zero within
0.044 of control, logit range growth <= 3x the reference. Escalation if KL
stabilizes but pins suboptimal behavior: learned context-conditioned
controller gain alpha_j = alpha_global + delta_alpha_phi(decision state),
trained by count+utility gradients, initialized to zero — no stored
per-decision constants, training stays meaningful (addresses the rejected
pre-baked-threshold objection). Explicitly NOT recommended: routing count
gradient into the utility tower (shortcut risk, breaks the utility semantics
of u and lambda-zero comparability).

## Fork continuation — tie-ownership and margin control: three-way analysis + literature taxonomy (2026-07-30)

**Inputs.** (1) This session's brainstorm; (2) independent Codex Sol High pass
(session 019f8fa3); (3) a literature sweep (27 verified sources, 9 registered
in the wiki). All three start from the v11 adjudication: fixed-reference KL
stabilizes sharpening and holds lambda-3 but pins lambda-zero — "reference too
restrictive" for a profile whose job is to leave the reference.

**Failure-mode taxonomy (each root-cause link is a named phenomenon).**
Link 1 (tie silence) = advantage collapse / value-function interference from a
many-to-one utility — [vamplew2024_value_function_interference](../../research-wiki/papers/vamplew2024_value_function_interference.md)
([arXiv 2402.06266](https://arxiv.org/abs/2402.06266)) is the closest published
account of links 1->2->4 as one chain and prescribes DETERMINISTIC, stated
tie-breaking; [yu2025_dapo_open_source_llm_rl](../../research-wiki/papers/yu2025_dapo_open_source_llm_rl.md)
([arXiv 2503.14476](https://arxiv.org/abs/2503.14476)) treats reward-homogeneous
groups as gradient-dead and filters them at runtime. Link 2 (cross-doc drift) =
underspecification — [damour2020_underspecification_ml](../../research-wiki/papers/damour2020_underspecification_ml.md)
([arXiv 2011.03395](https://arxiv.org/abs/2011.03395)) — plus policy churn —
[schaul2022_policy_churn](../../research-wiki/papers/schaul2022_policy_churn.md)
([arXiv 2206.00730](https://arxiv.org/abs/2206.00730)). Link 3 (unbounded
sharpening) = entropy collapse with a derived covariance law explaining WHY a
fixed entropy bonus is outrun rather than mis-tuned —
[cui2025_entropy_mechanism_rl](../../research-wiki/papers/cui2025_entropy_mechanism_rl.md)
([arXiv 2505.22617](https://arxiv.org/abs/2505.22617)). Link 4 (bounded shift
loses the race) = preference-conditioning controllability failure —
[delasheras2026_controllability_preference_morl](../../research-wiki/papers/delasheras2026_controllability_preference_morl.md)
([arXiv 2605.10585](https://arxiv.org/abs/2605.10585)): conditioned agents can
ace aggregate metrics while the conditioning input is behaviorally inert;
controllability must be measured first-class (our synchronous snapshot IS that
metric). Link 5 (fixed-ref KL pins lambda-zero) = the fixed-reference vs
current-policy regularization tension, independently published as a premise —
[he2026_unifying_stable_optimization_reference_regularization](../../research-wiki/papers/he2026_unifying_stable_optimization_reference_regularization.md)
([arXiv 2602.11523](https://arxiv.org/abs/2602.11523)). Unpublished-synthesis
gaps the sweep could NOT find sources for: (a) "bounded additive controller
authority vs unbounded logit scale" as a named design quantity — our link 4
appears novel; (b) regularizing specifically ON the reward-indifference set
(closest precedent: runtime-statistic-gated KL,
[lin2026_tepo_token_level_policy_optimization](../../research-wiki/papers/lin2026_tepo_token_level_policy_optimization.md)
([arXiv 2604.12736](https://arxiv.org/abs/2604.12736))); (c) exact ties from a
graded scorer as a distinct phenomenon.

**Three-way solution ranking (disagreement made explicit).**
- This session: entropy FLOOR first — bounded margins make the existing
  calibrated controller a deterministic tie-owner (satisfying Vamplew's
  prescription with zero new capacity); SAC-style auto-tuned target entropy
  ([haarnoja2018_sac_algorithms_applications](../../research-wiki/papers/haarnoja2018_sac_algorithms_applications.md),
  [arXiv 1812.05905](https://arxiv.org/abs/1812.05905)) replaces the dead
  beta=0.01 bonus with a dual variable. Learned gain = escalation.
- Codex Sol High: learned state-conditioned monotone gain alpha_j =
  softplus(alpha_raw + delta_phi(stopgrad(h_j))) FIRST — names conditional
  negative transfer (four lambda-tasks share one lambda-blind tower; only the
  controller can separate them, so controller CAPACITY is load-bearing);
  entropy floor second ("entropy says remain uncertain, not choose more
  private"); gated KL demoted to adjunct (high-lambda KL still updates shared
  u — indirect pinning). Second escalation: epsilon-lexicographic constrained
  controller (max E[P] s.t. utility loss <= 0.044) — the faithful
  formalization of free privacy, gated on counterfactual attribution quality.
- Literature: entropy-floor/covariance-clamp family is "the best structural
  fit" for a bounded additive shift on shared logits; lexicographic family is
  the theory of tie-ownership; anchor-selectivity (KL-on-ties) is promising
  but unprecedented; fixed-vs-current anchor tension is structural, so eta
  coefficient search is a dead end.

**Proposed decisive spike (awaiting approval).** Two arms x seeds 17+47,
8-epoch screening s47-first with synchronous evaluation and early kill, v11
runs as controls: Arm E = auto-tuned target-entropy floor (target normalized
entropy 0.10, dual update, no fixed KL) — tests "bounded margins + existing
calibrated shift suffice as deterministic tie-owner". Arm G = learned
monotone controller gain (zero-init residual over frozen features, count +
high-lambda utility trainable, u untouched, lambda-zero exact identity) —
tests "controller capacity is load-bearing". Gates (adapted from the joint
lists): lambda-zero per-doc utility loss <= 0.044 vs control; D2N005
synchronous Delta P(lambda-3) >= 0.20 by final cycle with cycle range <= 0.10;
cross-seed final difference <= 0.10; no menu-logit range > 50 and no
persistent action probability > 0.999; frontier regret reported. Decision
rule: E alone passes -> adopt E (simpler, no capacity risk; G stays available
for item-7 escalation); G alone passes -> adopt G; both pass -> adopt E and
record G as the item-7 candidate; both fail -> combine (Codex's Arm-2) or
escalate to epsilon-lexicographic.

## Preregistered ablation — remove gap-scaling under the logit soft-cap (2026-07-30)

**Hypothesis (audit finding).** Gap-scaling (`controller shift = alpha*g(lambda)*range*p_hat`) is a ratio workaround for logit-scale divergence whose structural fix is the utility-logit soft-cap. Under the cap it is redundant and actively harmful: it transmits the tower's scale wandering into controller-authority wandering, with two measured failure exits — upward (sensitivity-s47: ranges 54->170 act as an inverse temperature, freezing the whole lambda family and killing every corrective gradient, since `z = range*(u/range + alpha*g*p_hat)`) and downward (softcap-s47: ranges drifted to ~2, shrinking the shift to ~0.7 logits — controller powerless, greedy lambda-3 all-keep while sampled P read 0.22-0.50 from softness). The range statistic is additionally decision-irrelevant (set by the crater of never-selected actions, e.g. placeholder at u~-440). A fixed ABSOLUTE shift over capped logits severs the authority-wandering channel on both sides, and controller dominance over a genuinely indifferent (soft) utility tower is the intended lambda-3 semantics, not a failure. The cap inherits gap's secondary function (stable calibration units): alpha is recalibrated from raw (absolute) switch-threshold medians measured on capped warm-start logits.

**Arms (screening harness of RL-ranker v12: 8 epochs, seed 47 first, synchronous snapshots, greedy-path stability among the gates, early kill).** A1 = composed-with-gap (`--utility-logit-softcap 25 --profile-sensitivity-reg 0.1 --controller-gap-scaling utility-gap`, alpha-init switch-calibrated normalized median) — currently running. A2 = composed-no-gap (same, minus gap-scaling; alpha-init from the RAW switch-threshold median measured on capped logits; sensitivity regularizer's analytic profile reconstruction follows the no-gap controller formula; new controller_transform tag). Gate on running A2: A1 passes its v12 gates. If A1 fails, A2 still runs (the hypothesis predicts gap contributes to A1's failure), but adoption then requires A2 to pass the full gate set alone.

**Preregistered decision rule.** Adopt no-gap iff A2 passes all v12 gates (lambda-zero utility loss <= 0.044 vs control; D2N005 synchronous Delta P(lambda-3) >= 0.20 final with cycle range <= 0.10 on BOTH sampled and greedy paths; no menu-logit range > 50; no persistent action probability > 0.999; flat-decision expected privacy >= 0.50) AND is not worse than A1 on any gate A1 passes. Tie -> prefer no-gap (simpler mechanism, one fewer coupled statistic, constant absolute authority). Secondary prediction to check, not gate: no-gap shows flatter controller-authority (shift magnitude constant by construction) across epochs; residual instability, if any, isolates to the utility shape. Artifact consequences on adoption: controller_transform retag, alpha recalibration in absolute units, KL-reference regeneration, threshold-manifest version bump before production (already preregistered by the controller-strength fork).

**Adjudicated 2026-07-31.** A1 (composed-with-gap) FAILED: best-yet epochs 0-4 (Delta P to +0.42, real greedy separation) then epoch-cadence collapse-recover oscillation — with scale controlled, per-epoch utility credit on the shared tower swings keep-margins across the switch line faster than the sensitivity term damps. A2 (no-gap) also fails the gate set (final Delta P +0.12 < 0.20; early transient) BUT uniquely reaches a STABLE equilibrium — three consecutive greedy-stable lambda-ordered epochs, confirming the authority-wandering mechanism: constant absolute shift admits an equilibrium; gap-scaled shift does not. Residual deficiency is separation magnitude: raw switch thresholds scale with the (grown) logit range while the fixed shift does not, so the switchable fraction decays from its calibration value. Both preregistered secondary predictions held (flatter authority; residual instability isolated to utility shape). CONSEQUENCE: the v12 screening is complete with all capacity-free arms failed; the tie-ownership fork's preregistered escalation — the learned state-conditioned controller gain (ceiling-bounded, gradient-trained not dual-ascent, gain-field smoothness gate, random-gain and fixed-rule-gain controls per the registered literature) — is now the sole open candidate. Cap (25) and no-gap are ADOPTED as infrastructure for that arm (they solve scale and lambda-zero freedom and provide the equilibrium the gain modulates).


## Tie-ownership fork — learned-gain escalation adjudicated (2026-07-31)

The preregistered escalation was implemented (zero-init bounded state-conditioned gain head, gradient-trained, per-decision controller_alpha recorded every epoch) and screened in three stages on seed 47 over the adopted infrastructure (softcap 25 + no-gap + sensitivity 0.1): shared-lr (mechanism null — field flat), gain-lr 1e-2 (mechanism alive but GLOBAL-ONLY; large docs reach best-ever separation +0.42..+0.55 while D2N005 goes inert), and a 16-epoch extension (differentiation NEVER emerges: cross-decision alpha spread 0.00 everywhere; the field saturates at exactly the bound ceiling 6.84, where tanh saturation kills the gradient — the anti-divergence guardrail and expressivity are in direct tension). The count gradient's common component dominates the weak per-decision utility opposition at every tested lr and horizon; pooled decision features carry insufficient discriminative signal. Full data in research-wiki/experiments/2026-07-31-RL-ranker-v13-learned-gain-screening.md.

**Status.** The learned gain is REJECTED as designed. Adopted and retained: softcap 25, no-gap, sensitivity regularizer, and (observation, not adoption) a higher global alpha (~6.8 vs calibrated 5.35) materially helps D2N005 at the cost of large-document late decay — the seesaw is conserved under any global dial, which is the fork's central measured claim. Remaining preregistered exits, decision required: (1) epsilon-lexicographic constrained controller (max count subject to measured utility loss <= 0.044 — the faithful formalization of free privacy; leans on counterfactual attribution quality; a real design+implementation effort), or (2) accept and document a tiny-document variance bound (ship with the stable no-gap equilibrium; declare per-document Delta P spread expected at D<=6; cheapest, ships nondeterminism on tiny docs). Also available to either exit: input-feature redesign for the gain head (runtime-measured per-decision tie statistics, e.g. counterfactual Delta-U sensitivity, as discriminative features) — a new design iteration, not a rerun.
