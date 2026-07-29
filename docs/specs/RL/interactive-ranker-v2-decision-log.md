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
- Arm R — support-scaled rollouts: per-document rollout count
  R_d = clamp(ceil(64/D), 8, 32) (D2N005 32, D2N027 8->16-24 band, large docs
  8). Duplicate vectors are cache-identical, so extra rollouts cost almost no
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

**Status.** OPEN — evidence and design recorded; implementation and runs
awaiting approval.
