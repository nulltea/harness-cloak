---
type: research
status: current
created: 2026-08-01
updated: 2026-08-03
tags: [reward, remote-generation-variance, measurement-noise, utility, counterfactual, delta-u, ranker-v2, empirical-honesty]
companion: [qa-dependency-underdeclaration.md,
            ../specs/RL/ties-by-design.md,
            ../research/tie-ownership-root-cause-and-solution-space.md]
---

# Issue — counterfactual ΔU is contaminated by remote-generation variance, which is most of the "0.044 floor"

**Correction (2026-08-03):** this issue was originally filed as "reader jitter". That diagnosis was wrong. Decomposing the contamination by assertion family shows it is overwhelmingly **remote-LLM output variance**, not reader sampling — see "Which family carries the contamination" below. The measurements are unchanged; the attributed cause and the implied fix are not.

Because we differenced *document-level* utility for a single-decision counterfactual, contamination was summed over the whole assertion set while the signal was diluted relative to a threshold fixed in whole-document units (the flipped decision owns a median of 10.6% of the weight mass).

This issue records the contamination as a standing measurement property. The statistic-level fix is tracked in the [ties-by-design spec §7.1](../specs/RL/ties-by-design.md); the dependency-declaration question is [qa-dependency-underdeclaration](qa-dependency-underdeclaration.md).

## Measurements (cache-only, 10,459 single-decision pairs)

Movement on assertions the artifact declares **cannot** depend on the flipped decision — pure contamination by construction:

- nonzero in **60%** of pairs; median magnitude **0.0188**, p90 **0.0572**, max **0.50**
- oriented residue (more-general action minus less-general) has standard deviation **0.0375**
- on the 2,895 pairs whose decision has zero linked assertions — provably equivalent — the document-level statistic still reports nonzero for **1,873 (65%)**, median **0.019**

**Consequence for the floor.** The 0.044 reader-resolution figure is largely the standard deviation of this aggregate contamination, not a statement about the reader's ability to judge whether a single question was answered. It manufactured a large share of the "sub-noise" stratum (21% of measured decisions) that v13–v15 then designed machinery around.

## Noise is not the only component: a real spillover exists

Orienting each pair by generalization depth separates zero-mean noise from systematic effect:

| comparison | n | mean | mean / standard error |
|---|---|---|---|
| all pairs | 10,459 | −0.00236 | **−6.4** |
| keep → level | 5,118 | −0.00395 | **−7.5** |
| level → level | 4,822 | −0.00057 | −1.1 |
| level → placeholder | 519 | −0.00334 | −1.7 |

Generalizing a span **systematically degrades assertions elsewhere in the document** at 6.4 sigma, concentrated in the first `keep → generalize` step rather than in generalization depth — consistent with the remote model performing marginally worse on any anonymized document. This is a real utility cost, not a measurement artifact.

Magnitude keeps it in perspective: −0.0024 against a contamination standard deviation of 0.0375, so the systematic part is roughly **6% of the noise**. The stochastic component dominates by an order of magnitude.

**This revises [qa-dependency-underdeclaration](qa-dependency-underdeclaration.md):** the 18% systematically-signed residue recorded there was attributed to dependency under-declaration, but at least part of it is this genuine cross-decision spillover. The two have different fixes and the split between them is not yet measured.

## Which family carries the contamination (decisive, 2026-08-03)

The utility artifact has two assertion families, scored by completely different paths (`src/cloak/qa/scoring.py:908`): **`context`** (634 assertions) is reader-scored against `reader_excerpt(doc_p, evidence)` — an excerpt of `doc_p`, no remote call; **`delivered`** (723 assertions) is scored *deterministically* against `out_final` by a parsing contract, with no reader involved.

Non-linked (provably irrelevant) per-assertion contributions, by family:

| family | scored against | nonzero contributions | standard deviation |
|---|---|---|---|
| `context` | `doc_p` excerpt (reader) | **0.4%** | 0.0051 |
| `delivered` | `out_final` (deterministic parse) | **11.3%** | 0.0081 |

So the contamination is ~30× more prevalent in the family that depends on the **remote model's regenerated output**, and nearly absent in the family scored on `doc_p`. When the flipped span is outside a context assertion's excerpt, the reader's input is byte-identical and its score does not move. The dominant noise source is therefore the remote LLM producing different text elsewhere in the note, not reader sampling.

This also reframes the 6.4-sigma spillover recorded below: it is most plausibly the remote model performing marginally worse on any anonymized document, which is a real behavioural effect of the pipeline rather than measurement error.

**Consequence — a much cheaper and cleaner probe channel exists.** Counterfactual ΔU restricted to `context` assertions requires **no remote call and no extraction**: assemble `doc_p`, score a few excerpts with the reader. That is both ~30× less contaminated and dramatically cheaper than the current two-full-roundtrips-per-probe design, which bears directly on the evidence-volume constraint that blocked Gate 1. Targeting further — to only those relations in which the flipped span is subject, object, or occurrence — is the sharpest form and is Timo's proposal (2026-08-03).

## Normalization: divide by linked mass, not document mass

An earlier framing in this issue ("signal divided by ~10 while noise is summed") conflated two things and was partly wrong. Dividing by the document denominator `W` scales signal and contamination *identically*, so it does not affect signal-to-noise; the damage from including non-linked assertions is purely additive variance. Dilution bites only against a **threshold fixed in whole-document units**.

The fix is therefore to normalize the *attribution* statistic by the decision's own linked mass `W_L`, giving "what fraction of this decision's obligations did the flip break" in [−1, 1]:

| normalization | p75 |ΔU| | p90 |ΔU| | fraction above 0.044 |
|---|---|---|---|
| ÷ `W` (document mass, current) | 0.0302 | 0.1066 | 17% |
| ÷ `W_L` (linked mass, proposed) | 0.2500 | 0.4565 | 32% |

8.3× amplification at p75 and nearly double the fraction of measurable decisions. The objective keeps `W` (cross-document comparability, and document utility remains the final arbiter for a composed action vector); only the per-decision attribution statistic changes.

## Signal does exist (2026-08-03)

Sign-consistency over pair-groups measured in ≥3 distinct contexts: the **linked** ΔU has median consistency **1.00** with 71% of groups ≥0.9, against **0.37** and 20% for the non-linked residue. 35% of groups show a mean linked effect above two standard errors. Effect magnitudes (document-normalized): median 0.0032, p75 0.0296, p90 0.0703, max 0.617. The signal is real and its direction is highly reproducible; it was buried by additive contamination and by a threshold in the wrong units, not absent.

## Why a bias term is not the answer

A bias absorbs a systematic offset; the stochastic component is zero-mean, so a fitted constant converges to zero. Structurally, any menu-shared constant also cancels in every margin `u(a_i) − u(a_j)` and inside the softmax, so it is unidentifiable in the controller. Expressing "generalizing costs something" requires a term that is a function of the action (which the action-mode and runtime-type embeddings already provide), not an intercept.

## Live mitigations and open work

Adopted: **restrict the ΔU statistic to linked assertions** (removes most contamination *and* the spillover from per-decision attribution), and the **ε-insensitive deadband at 0.044** in the magnitude loss, which is the principled response to noise that cannot be removed — refuse to fit inside the instrument's resolution rather than trying to correct for it.

Not yet done, in rough cost order:

1. **The adjudicating test:** score one *identical* action vector twice through the FULL roundtrip. Any movement is stochastic by definition; per the family decomposition, expect it to be concentrated in the `delivered` family and to reflect remote-generation sampling rather than the reader. Costs remote + reader calls; deferred.
2. **Re-measure the resolution floor at linked granularity.** The current 0.044 was measured at document level and is the wrong instrument for a local question; the corrected floor is unknown and is an input to every threshold downstream.
3. **Account for spillover at the document level.** Per-decision attribution now excludes it by design, so composed cost is invisible: at ~0.004 per generalized decision over 10–25 decisions, composed spillover could reach 0.04–0.10 of document utility. Document-level utility must remain the final arbiter for a composed action vector.
4. **Normalize per-decision attribution by linked mass `W_L`** (see above) so the threshold is expressed in the units the effect actually lives in.
5. **Prototype a context-only probe channel** (no remote call, targeted to the flipped span's relations) and compare its measured ΔU against the full-roundtrip statistic on pairs where both exist.
6. **Precision-weighted (heteroscedastic) loss** if repeat-scoring ever yields per-measurement variance estimates.

## Artifacts

Reproducible from `results/ranker_v2/architecture/equivalence_critic/evidence-rows-linked.json` (carries both statistics, `linked_assertions`, and `derivable_tie` per row); miner `scripts/spikes/equivalence_critic_screening.py mine`.

## Reader non-determinism at temperature 0 (found 2026-08-03)

Testing the targeted-re-scoring criterion produced an unexpected side finding: **a byte-identical reader excerpt sometimes scores differently.** Over 88,619 checks where the excerpt was byte-identical between two cached scorings, 13 disagreed, every one a full 1.0 flip. Re-assembly was verified exact (0 drift against the cached `doc_p`) and all cache rows share one reader/extractor/task-prompt pin, so these are neither reconstruction nor stale-pin artifacts.

The 13 come from only **4 distinct (document, assertion) pairs** — `aci/D2N027` accounts for 8 — and split into **5 `reader_refresh` mismatches** (a knob in the cache identity, controllable by holding it constant) and **8 genuine same-refresh non-determinism** events. The `aci/D2N027` excerpt is 1,181 characters against 384–408 for the others, so a length or `BatchedContextReader` batch-composition effect is the leading hypothesis.

Rate is 0.015%, and for these cases the disagreement is itself the artifact — two scores from identical input means one is wrong and their difference is spurious. But it contradicts the determinism the reward pin is documented to rely on ("the determinism is load-bearing"), so it is filed rather than absorbed. Open: identify whether long excerpts or batch composition is responsible, and quarantine the 4 assertions meanwhile.
