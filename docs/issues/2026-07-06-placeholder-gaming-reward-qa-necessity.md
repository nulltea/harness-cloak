---
type: research
status: current
created: 2026-07-06
updated: 2026-07-06
tags: [rl, round-trip-reward, placeholder-gaming, floor-rejection, qa-construction,
       goodhart, reward-design, issue]
companion: [../specs/RL/roundtrip-ranker-infiller.md, ../dev/logs/2026-07-06-qa-build-corpus-expansion.md]
---

# Issue: placeholder-gaming — recall-only reward + surface-recall QA pairs don't reward informative generalization

## Summary

The round-trip reward is **recall-only**. Placeholdering a span can achieve *high recall
with full privacy* (the placeholder token echoes through the remote model and `invert()`
restores it), so a policy has a standing incentive toward **placeholder-everything** — which
preserves probed-fact recall while degrading actual task utility. The only structural defense
in the current design is **floor-rejection**, and it is shaky: it *withholds training data*
rather than *rewarding the right behavior*, and it rests on a QA-construction assumption we
have not verified.

## Background

- Reward `R_rt(doc, actions)` = mean token-F1 of a frozen QA reader answering the doc's
  train probes against `out_final` (spec [roundtrip-ranker-infiller.md](../specs/RL/roundtrip-ranker-infiller.md)
  Phase 1). Privacy is **not** in the reward — it lives in the legality mask (floors).
- Placeholder is always a legal action. Measured this session (`scripts/spikes/qwen_floor_leak_audit.py`):
  when a span is placeholdered, the remote model frequently **echoes the placeholder token**
  verbatim and `invert()` restores the original — so the fact is fully recovered at zero
  remote leakage (70% of floor "leaks" were this echo; e.g. `andrew` → `<DEM_4>` → note
  writes `<DEM_4>` → invert → `andrew`).
- **Floor-rejection** (Phase 0 step 4, implemented `scripts/build_probes.py`): a probe is kept
  only if answerable at the ceiling (`doc_orig`) AND *not* at the all-placeholder floor. Echo-
  recoverable probes are dropped, so the surviving probes are "absorb" facts where placeholder
  *loses* the fact — on those the reward penalizes placeholder.

## The issue (precise)

1. **The defense is subtractive, not corrective.** Floor-rejection prevents placeholder-gaming
   only by *deleting* the probes where placeholder wins, leaving a probe set biased toward
   facts where placeholder loses. It never *rewards* choosing a lattice generalization over a
   placeholder on the merits. Measured cost (this session, full build): **2300 of 3606
   candidate probes (64%) are floor-rejected**, and the dropped set is enriched in echo-prone
   quasi types (DATETIME 1.5×, LOC 1.5×, ORG 1.3× over-represented vs the kept set).

2. **Root cause — QA pairs test surface recall, not task-necessity of the PII.** A probe asks
   "what is span X?" and scores recovery of X's *surface*. For a lattice generalization to
   legitimately beat a placeholder, the task must **require** the PII to make a correct
   downstream inference — e.g. a span `London` matters because the answer depends on the fact
   that *something is happening in England*; a placeholder breaks that inference, a
   generalization (`a city in England`) preserves it. Our probes do **not** encode this — they
   test whether the exact span reappears, not whether the task needs it. **We have no evidence
   our data/probes have the task-necessity structure** that would make informative
   generalization measurably better than placeholder.

3. **Consequence.** The reward cannot distinguish "placeholder is fine here" (fact the task
   doesn't need) from "generalization preserves needed context, placeholder destroys it." So it
   cannot *teach* lattice-over-placeholder; it can only *withhold* the cases where placeholder
   would win. Utility degradation from over-placeholdering unprobed content is caught only at
   Phase-5 (whole-task-quality regression gate, Risk 7) — detection at eval, not a training
   signal.

## Related finding (separate but adjacent)

Direct identifiers **PERSON and CODE are absent from the environment entirely**: the arms
builder (`scripts/build_arms_artifact.py:53`, `action_table`) keeps only spans with a
generalization lattice (`if not e.get("lattice"): continue`), and proper names / IDs have no
lattice. Confirmed by the env span-type distribution (present: DATETIME, DEM, ORG, MISC, LOC,
QUANTITY; absent: PERSON, CODE). Names are still placeholdered in `doc_p` via the τ-walk R, but
they are **not ranker decisions and not probed** — so the highest-risk direct identifiers are
outside the learned/rewarded loop. (`andrew` only appears because the detector misclassified it
as DEM, a lattice-bearing type.)

## What a real fix requires

Not a patch to floor-rejection. It requires **rethinking QA-pair construction** so probes test
a downstream inference that a lattice generalization preserves but a placeholder breaks — i.e.
context-dependent questions where the PII carries task-relevant signal. Candidate directions
(unresolved, for post-pilot design):
- Generate probes that require the PII's *category/relation* for a correct answer, not its
  surface (task-necessity probes).
- A reward term that positively credits informative generalization (utility signal), accepting
  the Goodhart risk the spec currently avoids by keeping utility eval-only.
- Verify whether the current corpora even contain task-necessity structure before trusting the
  round-trip reward to reward generalization.

## Scope decision (2026-07-06)

No mechanism besides **floor-rejection** (and the eval-side Phase-5 regression gate) currently
addresses this — the reward is recall-only and the probes are surface-recall. **The first RL
pilot proceeds with the current implementation**; this is a logged known-limitation to revisit
after the pilot. The separate open question for the pilot is the **reader model** choice.

## Sources

Session audits: `scripts/spikes/qwen_floor_leak_audit.py` (echo/survivor/hallucination
classification), `scripts/spikes/reader_miss_audit.py`, the floor-reject-by-type breakdown.
Spec: [roundtrip-ranker-infiller.md](../specs/RL/roundtrip-ranker-infiller.md) Phase 0 step 4,
Phase 1, Risks 7–8, Phase 5 regression gate.
