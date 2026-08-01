---
type: research
status: current
created: 2026-08-01
updated: 2026-08-01
tags: [qa, utility-artifact, credit-routing, policy-dependency, measurement, ranker-v2, empirical-honesty]
companion: [../specs/RL/ties-by-design.md, ../specs/qa-builder-v2.md,
            qa-relation-scoring-lexical-not-entailment.md]
---

# Issue — QA assertions appear to under-declare their policy dependencies

Each utility assertion declares `credit_routing` (`linked` with an explicit `policy_dependency_decision_ids` list, or `residual` with none). `src/cloak/reward/utility_credit.py` resolves these into `linked_by_decision`, and the round-4 zoom-out (2026-08-01) proposes using them to restrict the counterfactual ΔU statistic to the assertions a flipped decision can actually affect — which removes a large, pure-noise contamination from every measurement (see `docs/specs/RL/ties-by-design.md` §7.1).

That correction is only as trustworthy as the declarations. A cache-only test says they are mostly right and sometimes wrong.

## Measurement

For each single-decision counterfactual pair, the **residue** is the document-level ΔU minus the linked-restricted ΔU — i.e. movement in assertions the artifact says cannot depend on the flipped decision. Over pair-groups measured in ≥3 distinct surrounding contexts (n=295 groups with nonzero residue), sign-consistency `|mean(residue)| / mean(|residue|)` (0 = zero-mean noise, 1 = perfectly systematic):

- median **0.35**, mean 0.45
- **62%** noise-like (≤ 0.5) — consistent with reader jitter on unrelated assertions
- **18%** highly consistent (≥ 0.9) — consistent with a real dependency the artifact does not declare
- of the 23 groups whose residue exceeds the 0.044 floor, 13% are highly consistent

**Revision (2026-08-01, same day):** the systematic minority is *not* necessarily under-declaration. Orienting residues by generalization depth shows a genuine cross-decision spillover — generalizing a span degrades assertions elsewhere at 6.4 sigma, concentrated in the first `keep → generalize` step. See [reader-jitter-contaminates-delta-u](reader-jitter-contaminates-delta-u.md). Some unknown share of the 18% is that real effect rather than a declaration bug, and the split is not yet measured. The disposition below is unchanged (both explanations argue for the conservative both-statistics tie rule), but the "candidate dependency repairs" in step 2 must be triaged against spillover before any artifact edit.

Separately, on the 2,895 cached pairs whose decision has **zero** linked assertions, the document-level statistic reports a nonzero difference for 1,873 (65%), median magnitude 0.019.

## Why it matters

Under-declaration biases the linked-restricted statistic **downward**, so it would manufacture false ties — the dangerous direction, since a false equivalence spends real utility to buy privacy. The noise majority argues for adopting the corrected statistic; the systematic minority argues for not trusting it alone.

## Disposition (does not block)

Per the standing rule that data defects are filed rather than allowed to block RL work:

1. **Adopt the linked-restricted statistic** for evidence mining and regression targets, but label a pair a tie only when **both** statistics agree it is below the floor. That keeps the noise reduction while refusing the direction that costs utility.
2. **Flag the ~18% systematic groups** for QA-builder review; they are candidate dependency repairs, not reader noise.
3. **Adjudicating measurement not yet run:** score one identical action vector twice through the reader. Movement in any assertion is then pure jitter by definition, which separates the two explanations directly. Costs reader calls, hence deferred.

## Artifacts

Test script: `/tmp` scratch (round-4 zoom-out); reproducible from `results/ranker_v2/architecture/equivalence_critic/evidence-rows-linked.json`, which carries both statistics per row plus `linked_assertions` and `derivable_tie` flags. Miner: `scripts/spikes/equivalence_critic_screening.py mine`.
