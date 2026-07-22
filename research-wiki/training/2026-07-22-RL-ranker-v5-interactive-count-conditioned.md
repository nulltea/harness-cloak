---
type: training-experiment
status: planned
created: 2026-07-22
model: ConditionalRankerPolicy — frozen ModernBERT occurrence/action embeddings, sequential GRU state, and identity-initialized FiLM lambda conditioning
dataset: ACI ranker-v2 environment — 67 documents split deterministically into 53 train and 14 development documents; final held-out attacker data excluded
result: pending
tags: [ranker, rl, interactive, count-conditioned, lambda-menu, structured-credit, preflight]
companion: ../../docs/specs/RL/interactive-ranker-v2.md
---

# RL-ranker v5 interactive count-conditioned policy

> **Superseded before running (2026-07-22, same day):** the FiLM/GRU conditional architecture this
> record planned was superseded by the semantic prototype
> (`docs/specs/RL/ranker-v2-architecture.md`) before any optimizer run. No run occurred; status
> stays `planned` permanently. Successor record: RL-ranker v6
> (`2026-07-22-RL-ranker-v6-semantic-privacy-context.md`). The preregistered smoke documents and
> deterministic split carry forward to v6.

## Objective & hypothesis

Train one sequential policy over the frozen ranker-v2 action menus, conditioned on a supported
lambda profile, using fixed-denominator QA utility credit and the versioned analytic count reward.
The hypothesis is that a single identity-initialized conditional policy can expose a supported
utility/count frontier without changing the lambda-zero initialization or calibrating against final
attacker outcomes. Failure to retain at least three supported profiles, preserve lambda-zero
identity, or satisfy the frozen feasibility gates stops the run.

## Training data

- **Source:** the 67 ACI documents in `results/ranker_v2/environment/ranker-env.json`, with assertion
  routing from `results/ranker_v2/qa/aci-full.utility` and count scores only through
  `CountReward.from_artifact(results/ranker_v2/reward/count-reward-state.json)`.
- **Split:** compute `sha256("ranker-v2-calibration-split-v1\0" + doc_id) mod 5`; bucket `0` is
  development and buckets `1`–`4` are train. This produces 53 train and 14 development documents.
  The content-free ordered split manifest hash is
  `sha256:214b942301815433df34c0e521d8b66cdc1dcb98a49643b5938a2632974f6bbb`.
- **Calibration scope:** behavior-cloning trajectories, verified ExIt winners, deterministic rule
  anchors, support trajectories, measured adjacent counterfactuals, and cached complete component
  vectors from the train/development split only.
- **Attacker exclusion:** final held-out documents, attacker outputs, realized attacker success, and
  leak-through outcomes are forbidden inputs to threshold selection, lambda selection, scope
  reduction, or count normalization.
- **Runtime types:** use the frozen environment mappings unchanged. Provisional flat decisions are
  consumed through their published `CountReward` action scores; no type, decision, or profile is
  special-cased or silently removed.

The three pre-registered ACI smoke documents are:

- `aci/D2N001` — both accepted context and delivered assertion families;
- `aci/D2N048` — no accepted context assertion and at least one accepted delivered assertion;
- `aci/D2N002` — at least one policy decision mapped to repeated occurrences.

All three are in the deterministic train split. Selection used IDs and structural artifact fields
only; document content was not inspected for selection.

## Training config

The planned policy is the Task 6 `ConditionalRankerPolicy`, initialized by deterministic behavior
cloning and verified utility-only ExIt. Each document group uses one supported lambda profile and a
balanced Latin-cycle schedule. Utility uses fixed-denominator structured credit with measured
one-decision pair losses substituting for provisional terms. Count loss is the exact expectation
over each complete legal menu. Entropy and any collapse-triggered KL term are lambda-independent.

The calibration preflight is offline and cache-first. It freezes the exact trajectory pool, reward
pins, threshold rules, spike output, empirical thresholds, switch points, replay report, and the
accepted three-to-five-profile menu before optimizer updates. Missing cache entries stop cleanly and
must not be fabricated or dispatched without approval.

## Selection & operating point

The candidate menu starts at lambda zero. For the default four-profile menu, nonzero candidates are
the weighted `0.25`, `0.60`, and `0.90` quantiles of positive switch points in log-lambda space,
snapped to observed switch points. Equivalent replay signatures merge. Redundant adjacent profiles
receive at most two deterministic replacement passes. Fewer than three supported profiles stops
selection rather than padding the menu.

Menu acceptance requires the frozen adjacent winner-change threshold, nondecreasing selected
`P_count`, exact lambda-zero pure-utility identity, the frozen all-placeholder ceiling, distinct
adjacent replay signatures, and frozen corpus/type support. Priority, threshold, and support values
do not scale rewards.

## Evaluation & success criteria

Preflight and training use the following registered criteria:

- Hard invariants: explicit count coverage `1.0`, fallback/default count-gradient mass `0.0`,
  missing occurrence-to-decision mappings `0`, nonmonotone accepted profiles `0`, and exact
  lambda-zero scalar selection identity. Unsupported profiles and reward-cache pin mismatches are
  also zero-tolerance failures.
- Feasibility: freeze numeric minimums for distinct points, supported documents by corpus,
  supported decisions by type, adjacent winner change, nonzero counterfactuals, count-menu
  separation, and corpus/type/profile/provenance support from the predeclared train/development
  diagnostic rules only.
- Interaction: report collision rate and lost count opportunity separately; trigger an ablation only
  under the predeclared joint rule.
- Menu: retain three to five profiles, preserve nondecreasing selected `P_count`, exact lambda-zero
  utility identity, and a highest-profile placeholder rate below the frozen ceiling unless explicitly
  labeled diagnostic-only.
- Conditional training: report linked, residual, fallback, counterfactual, count, entropy, and KL
  contributions; balanced exposure by document/corpus/type/profile; and fixed-document response on
  non-flat menus.
- Final promotion: compare the conditioned lambda-zero profile with a fixed lambda-zero utility
  control using the frozen paired document-level non-inferiority procedure. Any realized-privacy
  comparison is at matched realized privacy and identical settings.

Stop if any hard invariant fails; a predeclared threshold rule cannot produce a value; the cache
lacks required trajectories; fewer than three supported profiles survive; replacement still leaves
redundant profiles after two passes; lambda-zero identity or selected-count monotonicity fails; a
required corpus/type lacks frozen support; conditional profiles collapse; or the lambda-zero control
fails its frozen non-inferiority rule. Proxy/attacker inversion terminates count-driven privacy
claims and is never calibrated away.

## Results

Pending. No preflight spike, menu selection, optimizer update, remote generation, or attacker
evaluation has run under this record.

## Ablations

Predeclared triggers may require a privacy-return-to-go collision ablation, a count-normalization
ablation in a new run, or fixed-condition certification policies. These are not silently enabled.
Absence of nonzero fixed-condition controls means no claim that conditioning matches a
separate-policy frontier. Multi-decision interventions are a future ablation if one-decision effects
systematically disagree with hyperedge behavior.

## Cost

Task 11 is local, deterministic, model-free, and cache-only. Its expected pass condition on the
currently empty utility cache is a clean stop with exact missing trajectory/action-vector and work
counts. Any later remote/extractor/reader dispatch requires explicit approval and a separate cost
estimate. GPU work is not required for preflight.

## Risks & caveats

- The published count state is provisional: 42 decisions are flat-tagged. Their published scores
  supply level score zero and placeholder score one through `CountReward`; this can concentrate
  pressure at the placeholder endpoint and is reported rather than repaired here.
- Calibration support may be insufficient because the complete utility cache is currently empty.
- Reward quantization and reader jitter can erase apparent switch support; thresholds are frozen
  from the predeclared diagnostic rules before training, never from favorable full-run outcomes.
- A supported count-score frontier is not evidence of realized privacy. Held-out attacker behavior
  can falsify the proxy after training.

## Artifacts

- Environment hash: `sha256:4cc754a7143252613d2ef0160d7778580621fd973a32e5a0388da510170ddc8a`
  (file hash `sha256:07f568af1c63d4dff007d95ea58a3540e585e974a3ab26590b927fd6aec42583`).
- Utility artifact hash: `sha256:633250a2ecc22bf09df779eaf6e65354bac0a144621c05a5dfcb408c7f5e9b18`
  (file hash `sha256:276aa0cc6ca2b0994cf791ab23c35e02dad5ddbe4e70a293af5c34242299412f`).
- Count-state artifact hash: `sha256:2b135396ac15fedc191bb6ed55c61d8c9437646ca6b532e81b3b1f2251d22706`
  (file hash `sha256:914bcface4102b8ec3f88e650160bf1639d4869f5c8b384ae44a30a9085ef306`).
- Planned preflight outputs: `results/ranker_v2/preflight/threshold-rules.json`,
  `calibration-pool.json`, `diagnostic-spike.json`, `threshold-manifest.json`,
  `switch-points.json`, `replay-report.json`, `lambda-menu.json`, and `gate-report.json`.
- Planned utility cache: `results/ranker_v2/cache/utility-results.jsonl`.
- Predecessor: [RL-ranker v4 QA utility smoke](2026-07-13-RL-ranker-v4-qa-utility-smoke.md).

## Sources

- [Interactive ranker v2 specification](../../docs/specs/RL/interactive-ranker-v2.md).
- [Interactive ranker v2 diagnostics](../../docs/specs/RL/interactive-ranker-v2-diagnostics.md).
- [QA builder v2 specification](../../docs/specs/qa-builder-v2.md).
- [RL-ranker v4 QA utility smoke](2026-07-13-RL-ranker-v4-qa-utility-smoke.md).
