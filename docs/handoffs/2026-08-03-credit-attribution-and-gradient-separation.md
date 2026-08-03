---
type: handoff
status: current
created: 2026-08-03
updated: 2026-08-03
tags: [rl, ranker-v2, counterfactual, credit-attribution, gradient-routing, handoff]
companion: [docs/research/tie-ownership-root-cause-and-solution-space.md,
            docs/issues/counterfactual-delta-u-measurement.md,
            docs/issues/delivered-assertion-omission.md,
            research-wiki/experiments/2026-08-03-credit-attribution-validation.md]
---

# Handoff — credit-attribution revision done; gradient separation under re-evaluation

## Open question being adjudicated right now

Codex Sol High (session `019f8fa3-5e69-7c72-bd5f-1f1ea12eb7b5`) was dispatched 2026-08-03 with a round-6 brief: **is the deliberate three-way gradient separation the systemic defect?** Prompt: `scratchpad/codex-round6-gradients.md`; output: `scratchpad/codex-round6-out.log`. Timo's framing — he wants a next step that moves the *training* problem, not another offline generalization probe, and he suspects the separated gradient paths are the root issue.

The three paths, each isolated on purpose:

1. **Utility tower** `u` — leave-one-out advantages plus counterfactual pair losses `-ΔU·(q−0.5)`.
2. **Global controller strength** `alpha_raw` — the expected-count objective only; the count path uses `softplus(alpha_raw + residual.detach())`.
3. **Per-decision gain residual** — the tie-margin hinge only; `tie_margin_loss` detaches both the utility logits and `alpha_raw`.

The pattern that motivates the suspicion: every intervention touched one path and another absorbed it. v13's learned gain degenerated to a global constant (count gradient reaches only `alpha_raw`). v14's hinge made the controller own measured ties but the oscillation moved upstream into the tower, which the hinge may not touch. Gate 1's scale anchoring failed to beat the tower's own margin. Leg D showed changing the tower's credit signal made prediction worse.

## What shipped, and its measured effect

Commits: `bbbd361` scope-matched substitution, `435a57c` excerpt-changed linkage, `d5dfec2` monotone tie qualification, `bc10ed8` preflight + audit quota, `dbf1b31` Leg D revert, `8d5ac3c` scope limit and retractions. Earlier: `fde08a0` reader dedup, `a1fb5ef` attribution statistic.

**Net effect on training, measured — this is the number that matters:**

- **Tower gradient: unchanged.** `delta_u` is back to the total document delta; the scope-matched mechanism is present, tested, and **inert** because the complement is empty by construction.
- **Controller gradient: 44 → 42 qualifying tie pairs, 12 → 11 decisions.**
- **Reader calls: 78% fewer** context calls per batch, independent of cache state.
- `delivered_audit_fraction = 0.0` — the quota is **disarmed**.

A training smoke run would therefore measure noise. Do not run one until a change to the gradient or the controller lands.

## Live defect

`TIE_EXIT_BOUND = 0.044` is a document-aggregate figure being compared against the weighted-L1 movement statistic, which is a different unit. It errs conservative (movement ≥ |attributed delta|, so it over-disqualifies) but it is **uncalibrated**, and the local floor is unmeasured. No gate may consume the movement statistic until it is re-measured, and the 2-pair label delta above cannot be fully trusted meanwhile.

## Findings worth not rediscovering

- The old ΔU statistic reported nonzero on **65%** of provably-tied pairs (1,873 of 2,895), median 0.019. The "0.044 reader floor" is largely aggregate contamination, not reader resolution.
- The contamination is **the remote model, not the reader**: context assertions (scored on a `doc_p` excerpt) move on 0.4% of irrelevant contributions, delivered assertions (scored on `out_final`) on 11.3%. Generation is `temperature=0` with a content-addressed cache, so this is greedy-decode input sensitivity, not sampling.
- The failure is **omission (84%), not paraphrase (0%)**. `invert` is already a semantic cascade, which likely explains the empty paraphrase bucket — so no scorer or extractor change can address it.
- **Placeholders are the least disruptive transition** (49–54%); level↔level is the worst (65%).
- `SCHEMA_NOTE` in `src/cloak/tasks.py` mandates one line per active problem and is **unreachable** on the ranker path — neither the pin nor the generation call passes a `schema` flag. Wiring it invalidates the whole utility cache (the prompt is in the reward pin).
- Leg D: the coupling between decisions is **reproducible structure, not noise** (total delta beats attributed 0.661 vs 0.530 on the linked route across held-out configurations).
- Anchoring delivers scale (slope ≈1.0 vs the actor's 0.010) but **loses ordering** (0.75 vs 0.86), so regression must compose with ranking, not replace it.
- 13% of decisions (86 of 652) have zero linked assertions and are **provable ties** needing no measurement.

## The wall

**v15, Gate 1 and Leg D all reduce to the same 4 documents.** Leg D's "held-out" meant a held-out *configuration within* a document — 99.3% of its 10,207 folds come from the 4 RL training documents. Gate 1's learning curve had a minimum resolvable AUC difference of 0.159 against an observed range of 0.138. Every cross-document question returns "insufficient evidence", and buying breadth costs reader and remote calls — a budget decision Timo has not yet made.

## Next steps (pending Sol High's round-6 answer)

1. Re-measure the local tie floor in movement units — cheap, and it unblocks the qualification statistic.
2. Sol High's recommendation on unifying the gradient paths.
3. `SCHEMA_NOTE` on a small pinned subset — biggest lever on the environment defect, invalidates the cache.
4. ε-lexicographic filter over the *measured* exact-tie core — provably free per Skalse, measurable within documents.
5. Evidence breadth — the actual blocker, needs Timo's budget call.

## Process note

This session shipped three instrument changes and one was wrong; the preregistered revert rule in the validation record caught it before a training run consumed it. Several other errors were caught by Sol High's adversarial passes. Keep the preregister-a-revert-rule pattern for anything touching the reward, and keep sending reward-side designs to Sol High before implementing.
