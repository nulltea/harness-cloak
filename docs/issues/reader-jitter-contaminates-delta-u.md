---
type: research
status: current
created: 2026-08-01
updated: 2026-08-01
tags: [reward, reader, measurement-noise, utility, counterfactual, delta-u, ranker-v2, empirical-honesty]
companion: [qa-dependency-underdeclaration.md,
            ../specs/RL/ties-by-design.md,
            ../research/tie-ownership-root-cause-and-solution-space.md]
---

# Issue — reader jitter contaminates every measured ΔU, and is most of the "0.044 floor"

Utility is scored by a local reader over per-assertion QA probes on a full round-tripped document. Two scorings of two different action vectors are two independent reader passes, so **every** measured ΔU carries the jitter of both. Because we historically differenced *document-level* utility for a single-decision counterfactual, that jitter was summed over the whole assertion set while the signal was diluted by the fraction of weight mass the flipped decision actually owns (median 10.6%).

This issue records the jitter itself as a standing measurement property. The statistic-level fix is tracked in the [ties-by-design spec §7.1](../specs/RL/ties-by-design.md); the dependency-declaration question is [qa-dependency-underdeclaration](qa-dependency-underdeclaration.md).

## Measurements (cache-only, 10,459 single-decision pairs)

Movement on assertions the artifact declares **cannot** depend on the flipped decision — pure contamination by construction:

- nonzero in **60%** of pairs; median magnitude **0.0188**, p90 **0.0572**, max **0.50**
- oriented residue (more-general action minus less-general) has standard deviation **0.0375**
- on the 2,895 pairs whose decision has zero linked assertions — provably equivalent — the document-level statistic still reports nonzero for **1,873 (65%)**, median **0.019**

**Consequence for the floor.** The 0.044 reader-resolution figure is largely the standard deviation of this aggregate contamination, not a statement about the reader's ability to judge whether a single question was answered. It manufactured a large share of the "sub-noise" stratum (21% of measured decisions) that v13–v15 then designed machinery around.

## Jitter is not the only component: a real spillover exists

Orienting each pair by generalization depth separates zero-mean noise from systematic effect:

| comparison | n | mean | mean / standard error |
|---|---|---|---|
| all pairs | 10,459 | −0.00236 | **−6.4** |
| keep → level | 5,118 | −0.00395 | **−7.5** |
| level → level | 4,822 | −0.00057 | −1.1 |
| level → placeholder | 519 | −0.00334 | −1.7 |

Generalizing a span **systematically degrades assertions elsewhere in the document** at 6.4 sigma, concentrated in the first `keep → generalize` step rather than in generalization depth — consistent with the remote model performing marginally worse on any anonymized document. This is a real utility cost, not a measurement artifact.

Magnitude keeps it in perspective: −0.0024 against a jitter standard deviation of 0.0375, so the systematic part is roughly **6% of the noise**. Jitter dominates by an order of magnitude.

**This revises [qa-dependency-underdeclaration](qa-dependency-underdeclaration.md):** the 18% systematically-signed residue recorded there was attributed to dependency under-declaration, but at least part of it is this genuine cross-decision spillover. The two have different fixes and the split between them is not yet measured.

## Why a bias term is not the answer

A bias absorbs a systematic offset; jitter is zero-mean, so a fitted constant converges to zero. Structurally, any menu-shared constant also cancels in every margin `u(a_i) − u(a_j)` and inside the softmax, so it is unidentifiable in the controller. Expressing "generalizing costs something" requires a term that is a function of the action (which the action-mode and runtime-type embeddings already provide), not an intercept.

## Live mitigations and open work

Adopted: **restrict the ΔU statistic to linked assertions** (removes most jitter *and* the spillover from per-decision attribution), and the **ε-insensitive deadband at 0.044** in the magnitude loss, which is the principled response to noise that cannot be removed — refuse to fit inside the instrument's resolution rather than trying to correct for it.

Not yet done, in rough cost order:

1. **The adjudicating test:** score one *identical* action vector twice through the reader. Any movement is jitter by definition, which cleanly separates it from spillover and from dependency mis-declaration. Costs reader calls; deferred.
2. **Re-measure the resolution floor at linked granularity.** The current 0.044 was measured at document level and is the wrong instrument for a local question; the corrected floor is unknown and is an input to every threshold downstream.
3. **Account for spillover at the document level.** Per-decision attribution now excludes it by design, so composed cost is invisible: at ~0.004 per generalized decision over 10–25 decisions, composed spillover could reach 0.04–0.10 of document utility. Document-level utility must remain the final arbiter for a composed action vector.
4. **Precision-weighted (heteroscedastic) loss** if repeat-scoring ever yields per-measurement variance estimates.

## Artifacts

Reproducible from `results/ranker_v2/architecture/equivalence_critic/evidence-rows-linked.json` (carries both statistics, `linked_assertions`, and `derivable_tie` per row); miner `scripts/spikes/equivalence_critic_screening.py mine`.
