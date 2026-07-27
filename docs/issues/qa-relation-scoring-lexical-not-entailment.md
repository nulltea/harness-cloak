---
type: research
status: current
created: 2026-07-14
updated: 2026-07-27
tags: [qa, relation-qa, scoring, entailment, fact-score, three-point-gate, reward-design, empirical-honesty]
companion: [../specs/qa-builder-v2.md, detector-misclassifications.md,
            2026-07-11-ladder-decision-qa-question-design.md]
---

# Issue — context-relation QA scoring is lexical, not lattice-entailment aware

The QA-builder-v2 context assertions are scored by `_answer_score` =
`max(fact_score(reader_answer, v) for v in accepted_values)`. `fact_score`
(`src/cloak/qa/scoring.py`) is **lexical**: number-gate → token containment
(gold tokens ⊆ answer → 1.0) → acronym → token-F1 fallback. It has no model of
the generalization lattice. The accepted answer for a relation is a single
teacher-authored generalization level. The combination means the scorer does
**not** implement the "truthful generalization" semantics the spec assumes, and
this is a prime suspect for the `three_point_gate_failed` epidemic where every
compiled relation dies at the reader gate.

## Intended design (spec) vs actual behavior

**Intended** (`docs/specs/qa-builder-v2.md`): the gold level is a *utility
floor*. A rewarded context assertion "succeeds on `doc_orig`, succeeds on the
coarsest legal non-placeholder anchor that still entails the property, fails on
all-placeholder." The explicit division of labor: "QA does not reward
generalization over KEEP inside the supported band; ranker-v2's exact count
objective supplies pressure toward the coarsest semantically viable action." So
KEEP and any action at least as specific as the floor should receive **full**
utility credit; only coarsening past the meaning-carrying level (or placeholder)
should lose it; and the *privacy/count* objective — not the QA — moves the
ranker toward coarser actions.

**Actual**: with gold = `solid organ transplant` (from the D2N002 gated build,
`contraindicated_because_of(certain medications -> kidney transplant)`):

| ranker action on the decision | `doc_p` text | reader answer | fact_score vs gold |
|---|---|---|---|
| pick the gold level | "solid organ transplant" | "solid organ transplant" | ~1.0 |
| KEEP | "kidney transplant" | "kidney transplant" | ~0.4 (token-F1 on "transplant") |
| coarser ("medical condition") | "medical condition" | "medical condition" | 0 |

So the reward **peaks at the teacher's exact level phrasing and under-credits
KEEP** — a bump around one level, not a floor. A KEEP rollout (which preserves
strictly more information) is penalized relative to emitting exactly the
teacher's level. This contradicts the intended "floor" semantics and biases the
utility signal toward one specific granularity.

Build-time consequence: the three-point gate requires the reader on `doc_orig`
("kidney transplant") to reproduce the gold level ("solid organ transplant").
Under a lexical scorer that mapping fails, so the assertion is rejected
(`three_point_gate_failed`) even though the relation is genuine and grounded.
This is consistent with every relation in r18–r24 dying at that gate.

## Rejected fix — per-rollout gold

Store the original span as the answer and swap in whatever level the ranker's
substitution produced, so the gold tracks `doc_p`. **Rejected**: (a) it breaks
anti-placeholder discrimination — a placeholder rollout gets a
placeholder-shaped gold and trivially passes, destroying the signal that
separates generalization from deletion (the spec already rejects
"rewrite questions/answers per rollout using the replacement map … changes the
measurement together with the intervention"); (b) it freezes the protected
surface into the artifact as gold, a leak channel.

## Candidate fixes (not yet chosen)

1. **Entailment/lattice-aware scoring.** Credit an answer when it is the gold
   level *or finer* by the lattice relationship (kidney transplant ⊨ solid
   organ transplant ⊨ medical condition), rather than by token overlap. Needs a
   lattice-relationship or NLI check in the context scorer.
2. **`accepted_values` = the lattice chain.** Expand the accepted set to the
   floor level plus all finer levels, so any at-or-finer action matches *some*
   accepted value under the existing lexical scorer. Cheaper; approximates (1).

**Shared tradeoff:** fully crediting KEEP means accepting the *original surface*
("kidney transplant") as a gold answer, which is the protected term the artifact
is built to keep out. So KEEP credit must route through a surface→level mapping
(NLI/lattice), not by putting the surface in `accepted_values`.

## Related smaller item — vacuous level selection

Independently, the relation teacher sometimes picks the **coarsest/root** legal
level as the answer (`medical condition` for kidney transplant, when
`solid organ transplant` was available and would also pass leakage). That puts
the floor too *low* — every condition entails "medical condition", so the
assertion discriminates nothing. Fix: prefer the most specific meaning-carrying
level (prompt guidance, or a compiler check rejecting a root-level answer when a
more specific legal level exists). This is orthogonal to the scoring gap above
but compounds it.

## Evidence

- `/tmp/qa-v2-d2n002-gated.json` — `contraindicated_because_of(certain
  medications -> S7 kidney transplant)`, `support_property`/answer =
  `medical condition`; S7 levels = `[solid organ transplant, medical condition]`.
- `_answer_score` / `fact_score` (`src/cloak/qa/scoring.py`).
- Reader gate outcomes across r18–r24: relations reach `three_point_gate_failed`.

## Status

Diagnosis only; no fix applied. Issues 1 (vacuous level) and 2 (treated_with
indication-connector bug, separate) are being fixed independently. The
entailment-scoring decision is a reward-design change and needs an explicit
choice between fix (1) and (2) plus the KEEP-credit/leak tradeoff before
implementation.
