---
type: research
status: current
created: 2026-07-23
updated: 2026-07-23
tags: [issue, privacy-controller, direct-counts, rl-ranker, data-quality]
companion: [../specs/RL/ranker-v2-architecture.md]
---

# Issue: 42 policy decisions are privacy-ineligible for the direct-count controller

## Symptom

The profile-count target artifact (`results/ranker_v2/reward/profile-count-targets.json`,
`decision_eligibility`) marks 42 of 705 environment policy decisions ineligible; their
level-action rows are intentionally absent from `action_targets` (gate design — see the
artifact's `gate_report`). The direct-count privacy provider is strict/fail-closed, so any
policy forward over these menus raises `ValueError`. They spread across 34 of 67 documents,
so document-level exclusion is not viable.

## Provisional handling (in place)

`scripts/train_interactive_ranker.py` (`_demote_privacy_ineligible_decisions`) demotes each
uncovered menu to a fixed keep decision on the semantic direct-count path: the policy does
not control it, the original surface stays, and the demoted count is logged at startup
(42 expected on the current artifacts). The learned-checkpoint path is unaffected.

## Proper fix

Either extend count grounding so these decisions get level targets (preferred — they are
ordinary menus whose levels failed the count gate), or make ineligibility an explicit
environment-level annotation so eligibility is decided at QA-build time rather than by
provider lookup failure at training time.

## Impact

RL smoke and hybrid training on the direct-count path optimize over 663 of 705 decisions.
Utility/privacy metrics are unaffected for the retained menus; the 42 keep-fixed decisions
leave their entities un-rewritten, which slightly understates achievable privacy for
affected documents.
