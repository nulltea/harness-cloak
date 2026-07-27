---
type: research
status: current
created: 2026-07-27
updated: 2026-07-27
tags: [issue, pre-rl-audit, reward, dead-reward, schema-probes, extractor, stability]
companion: [2026-07-23-ungrounded-delivered-assertions.md, remote-llm-echo-absorption.md]
---

# Pre-RL audit: reward-signal and stability risks (2026-07-27)

Evidence: all 19 re-derivable cached (vector, out_p, out_final) rows; static audit of
all 723 delivered + 634 context assertions over 67 docs; direct-count targets over all
652 in-scope menus; code read of the hybrid objective and the alpha controller.

## R1 — Zero-signal documents (HIGH, dead reward)

4 docs have **zero policy-linked assertions** (`aci/D2N048, D2N054, D2N055, D2N060`):
utility is constant for every action vector (verified: D2N048 = 0.1000 across 5
vectors), so utility advantage is identically zero forever; they burn rollouts and
remote calls for noise. 4 more docs are <20% linked (`D2N030` 1/13, `D2N022` 2/19,
`D2N038` 2/10, `D2N066` 3/16). Mitigation: exclude zero-linked docs from the RL doc
set at load (same mechanism as scope demotion); weight-aware sampling for the low-link
tail. NOT a blocker for smoke-scale runs.

## R2 — Generalization echo absence caps restorable utility (HIGH, reward ceiling)

**Survey 2026-07-27** (`results/ranker_v2/architecture/echo-absence-survey.json`,
67 docs × BC vector, 1,182 generalized mentions): echoed 71.4%; absent 28.6%
(dropped 15.5%, partial-paraphrase 13.0%, pure-acronym 0.1% — though the acronym
detector misses compound forms like "ACE inhibitor", so some paraphrase mass is
really abbreviation). Worst runtime type: medical-procedure (38% absent); drugs 23%.
Overall source-restoration rate 63.9%. Scope correction: the context family reads
excerpts of doc_p, not out_final — echo absence affects ONLY delivered-family credit.
Absence is also action-dependent (fill choice changes echo odds), so part of it is
signal the policy can learn, not pure ceiling.

Across 19 vectors: 187 generalized mentions restored (174 exact / 10 fuzzy /
3 semantic) but **65 (26%) `gen_absent`** — the remote note never echoes the fill, so
inversion has nothing to splice and `contains`-type delivered credit dies. Root-cause
trace (D2N001): fill "angiotensin converting enzyme inhibitor" → medgemma writes the
abbreviation ("ACE inhibitor") → exact/fuzzy/semantic passes all miss → out_final lacks
lisinopril → assertion 0. This is the known echo-absorption channel
([remote-llm-echo-absorption.md]) now quantified; the extractor itself is NOT at fault
(0/19 determinism mismatches, 0 placeholder residue). Mitigation options: abbreviation/
alias expansion in the fuzzy pass (extractor-pin change → plan a re-baseline), or
prompt-template instruction to echo substituted terms verbatim (task-prompt-pin change,
re-gates). Either changes a reward pin — do it BEFORE the preflight or not at all.

## R3 — Schema probes are template-fragile (MEDIUM, systematic zeros)

`parse_aci_note` on real medgemma notes finds a `demographic` section in 1 of 3;
`DEMOGRAPHIC field_value` contracts (age/sex) score 0 on the other two regardless of
content. Combined with 50/723 delivered assertions being source-ungrounded
(≤2 per doc, spread over ~25 docs), the delivered family carries a systematic
policy-invariant zero mass. Cancels in leave-one-out advantage; distorts ceilings,
cross-doc aggregates, and any threshold calibrated on absolute utility (the preflight!).
Mitigation: classify delivered contracts policy-linked vs invariant (already filed) and
report utilities decomposed; consider dropping DEMOGRAPHIC field_value from the
policy-reward denominator.

## R4 — Objective normalization mix (MEDIUM, cross-doc gradient scale)

`total = utility + count − β·entropy + η·kl`: count is decision-AVERAGED
(×λ/decisions/rollouts) while utility and entropy are decision-SUMMED (/rollouts only).
Per-doc gradient scale for utility/entropy grows with menu count (5–25 decisions ⇒ ~5×
spread vs the count term). Not a correctness bug; a stability/λ-calibration hazard —
the effective utility:count ratio differs per document. Decide: align normalization
(behavior change, re-gates nothing cached) or accept and calibrate λ per the current mix.

## R5 — LOO advantage degeneracy at low entropy (LOW at current settings)

Leave-one-out advantage over N rollouts is exactly 0 for duplicated action vectors.
BC-initialized policies sample near-deterministically; smoke evidence is healthy
(10–11 unique of 12 rollouts) but entropy collapses as training sharpens — β·entropy
is the only counterpressure. Watch `adjacent_winner_change`/exposure reports; consider
rollouts ≥4 per group.

## Verified healthy (no action)

- Count-shaping signal: 0 flat menus of 652; median within-menu level-score range 0.596
  (p10 0.385); only 6 single-level menus. Strong gradient everywhere.
- Alpha controller: `count_combined = utility.detach() + α·g(λ)·privacy.detach()` —
  count loss reaches ONLY α; utility REINFORCE reaches α through log_probs. Opposing
  gradients exactly as specified.
- Extractor: deterministic (19/19 byte-identical), zero placeholder residue,
  retain-ambiguous rule active (15 retained).
- Utility responsiveness on linked docs: D2N001 spans [0.714, 0.886] over 10 vectors,
  D2N002 [0.646, 0.833] over 9 — every vector distinct.
- Hard training gates + non-finite guards present throughout the objective.
