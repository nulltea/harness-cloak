---
type: research
status: current
created: 2026-08-03
updated: 2026-08-03
tags: [reward, delivered-assertions, roundtrip, task-prompt, omission, out-final, ranker-v2, empirical-honesty]
companion: [counterfactual-delta-u-measurement.md, remote-llm-echo-absorption.md,
            ../../research-wiki/experiments/2026-08-03-counterfactual-measurement-revision.md]
---

# Issue — delivered assertions drop unrelated content when any one span changes

`delivered`-family assertions (723 of 1,357; 33.1% of the reward denominator) are scored against `out_final` by a parsing contract, with no reader involved. Measured over all 10,459 cached single-decision counterfactual pairs, **63% of pairs show at least one delivered assertion moving that the artifact declares cannot depend on the flipped decision** — median 2 such assertions per affected pair, which is 29% of a document's delivered mass at once, p90 4, max 9.

Concretely (`aci/D2N001`): flipping a procedure span to a placeholder moved a `contains 'mammogram'` assertion from 1.0 to 0.0, where that assertion's declared dependency is an entirely different decision. Flipping a medication span — on a decision with *zero* linked assertions, so with no utility signal of its own — moved `contains 'nasal congestion'` 1.0 → 0.0 and `contains 'pitting edema'` 1.0 → 0.0046.

## What kind of failure this is

**Not stochastic.** Generation is pinned at `temperature=0`, single-flight, behind a content-addressed disk cache, so `out_final` is deterministic given `doc_p`. The same flip reproduces every time. This cannot be averaged away by repeat measurement, and a repeat-scoring test would return exactly zero by construction.

**Not format collapse.** All 13,824 observed flips are `contains` contracts. `required_sections` flipped **zero** times, so the note's section structure is stable even under the loose prompt.

**Not paraphrase — omission.** Of the 13,824 flips, **84% are near-total losses** (a score ≥0.5 falling to ≤0.05) and **0%** are partial changes with both ends above 0.05. The term is absent from the generated note, not reworded.

**Not placeholder confusion.** Flip rate by action-mode transition: keep↔level 63%, **level↔level 65%**, level↔placeholder 49%, keep↔placeholder 54%. Placeholders are the *least* disruptive transition; swapping one generalization level for another is the worst. (An earlier reading of two hand-picked examples suggested the opposite; those examples were the first two cache hits, not a representative sample.)

So the mechanism is: greedy decoding is deterministic but chaotic, one changed input token flips an early choice, and the model re-plans what it includes — dropping a clinical finding elsewhere in the note.

## Two fixes considered and withdrawn

**A paraphrase-robust scorer (entailment or embedding instead of `contains`) — withdrawn.** 606 of 723 delivered assertions score by substring containment, which is the lexical-matcher pattern this project rules out elsewhere, so this looked like the obvious fix. It is not: the paraphrase bucket measures **0%**. `invert` is already a semantic cascade (placeholder, exact/fuzzy generalization, semantic-window fallback; `"semantic": True` with a pinned semantic model), which is the most likely reason no paraphrase survives to scoring — it normalizes reworded controlled fills first. No scorer can match, and no extractor can invert, content the model never produced.

**A terminology-fidelity prompt instruction — withdrawn.** A sentence such as "name every condition, medication, and procedure exactly as the dialogue names it" targets paraphrase, i.e. 0% of the measured problem, and pushes toward verbatim echo, which interacts with [remote-llm-echo-absorption](remote-llm-echo-absorption.md) for no measurable gain.

## Proposed fix — constrain generation

Generation is the only remaining lever. `SCHEMA_NOTE` already exists in `src/cloak/tasks.py` and is **unreachable on the ranker path**: `_template(job)` returns it only when `job["template"] == "schema"`, but both the reward pin (`roundtrip.py:87`) and the generation call (`roundtrip.py:283`) pass `{"corpus": corpus}` with no such key, and no caller in `scripts/` or `src/cloak/reward/` ever requests it. The ranker therefore always uses `CLINICAL_NOTE`, whose only structural guidance is "using standard note sections".

`SCHEMA_NOTE` mandates four sections plus `ASSESSMENT: one line per active problem, formatted "problem — category — status"` and the same shape for `PLAN`. A per-problem slot the model must fill makes silent omission structurally difficult, which is exactly the measured failure. It also narrows the decode's freedom to reorganize.

**Success criterion, if funded:** the non-linked delivered omission rate falls from its current 63% of pairs, with per-transition rates reported separately so the worst case (level↔level, 65%) stays visible.

## Cost, and why this is deferred

The task prompt is part of the reward pin (`TASK_PROMPT_PIN_VERSION`, `template_hash`), and cached rewards are valid only under the pin that produced them. **Switching prompts invalidates the entire utility cache** — every scored action vector in `results/ranker_v2/cache/utility-results.jsonl`, which is the accumulated output of the whole campaign and the source of all 10,459 evidence pairs. It therefore needs an explicit budget decision and should be scoped to a small pinned document subset first, never the corpus in one step.

Deferred behind the two free measurement fixes (targeted context re-scoring and linked-mass normalization) in the [measurement revision record](../../research-wiki/experiments/2026-08-03-counterfactual-measurement-revision.md).

## Wider implication, worth stating plainly

This is a property of the **reward environment**, not only of counterfactual measurement. Document utility itself moves when an unrelated span changes, in 63% of single-span perturbations. That bounds how precisely *any* method can be evaluated on this corpus, and it is a plausible contributor to the seed-to-seed and checkpoint-to-checkpoint instability the tie-ownership campaign spent five rounds chasing. Constraining generation may therefore be a prerequisite for the privacy-utility comparison rather than a measurement nicety — but that claim is untested until the arm above runs.
