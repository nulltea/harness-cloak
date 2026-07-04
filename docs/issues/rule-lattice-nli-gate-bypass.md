---
type: research
status: current
created: 2026-07-04
updated: 2026-07-04
tags: [issue, lattice, nli-gate, wordnet, geonames, truthfulness, substitution]
companion: [../specs/RL/surrogate-ranker-infiller.md, ../plans/2026-07-03-surrogate-rl-gaps-fixes.md]
---

# Issue: rule-sourced lattices bypass the NLI truthfulness gate

**Measured (2026-07-03, probe-shootout examples):** context-wrong fills ship in doc_p —
"dragon" (the clinician's **Nuance Dragon dictation assistant**, addressed as "hey, dragon" in
ACI-Bench dialogues) → "a mythical monster"; "washington" (the state, hiking context) → "a city
in District of Columbia"; "vermont" → "a city in Australia" (GeoNames most-populous-match picked
an Australian town).

**Root cause:** `lattice_for` (`src/cloak/lattice.py`) routes DATETIME/QUANTITY through rule
buckets, LOC through GeoNames/WordNet, and other types through WordNet **directly** — only
teacher-generated lattices pass through `nli_gate` (context-entailment check). The truthfulness
gate exists; rule-sourced candidates never see it, so wrong word-senses and wrong geo-matches
ship unchecked.

**Impact:** utility (nonsense fills confuse the remote task; "a mythical monster" in a clinical
note prompt), and occasionally privacy-neutral over-generalization ("vermont"→Australia leaked
nothing but is factually false — fails the project's strictly-truthful-generalization
requirement). Secondary observation: "dragon" is a software product name — arguably not
sensitive at all; fixed-schema over-detection is the tailorability argument (user-specified
types), tracked separately.

**Fix (scheduled — gaps plan Phase 2 batch):** route every lattice, regardless of source,
through `nli_gate(entity, context_sentence, candidates)` before use; candidates failing
entailment are dropped (empty lattice ⇒ generic typed placeholder via the exhaustion rule).
Cacheable per (entity, sentence); NLI model already loaded for the gate. Acceptance: the three
measured examples produce either a truthful level or a placeholder.

## Status after the Phase-2 fix: PARTIALLY resolved — the verifier has a known ceiling

The NLI gate measures **sentence-level entailment**: keep the candidate iff the fill-substituted
sentence is a logically weaker claim the original sentence supports (P(entailment) ≥ 0.6). That
reliably catches *unsupported specifics* ("vermont" → "a city in **Australia**": the premise
lends Australia no support). It is **not a semantics oracle**: pure word-sense errors can pass
as lexical hypernymy — an NLI model may accept "dragon → a mythical monster" without knowing
this "dragon" is dictation software, because dragon-is-a-mythical-monster is true in the
dominant sense. Wherever the gate passes a wrong-sense fill, the residual root cause is often
the *typing* of the span (detector assigned DEM via WordNet — itself the sense error), which no
lattice-side gate can repair.

## Proposed solution (future, decided to consider 2026-07-04): truthfulness in the RL reward

Extend the stage-2 (infiller) reward with a **truthfulness term** so the infiller is trained to
be *context-aware truthful*, rather than relying on the verifier alone:

```
r = α·(1−A) + (1−α)·U + β·T(doc_p, doc_orig)     # T = truthfulness score
```

- **T candidates** (same probe-validation discipline as the privacy probes — promote by
  measured correlation with a ground truth, e.g. teacher-LLM truthfulness judgments on a
  one-time labeled set): mean NLI entailment of fill-substituted sentences (the gate's score
  used gradedly rather than thresholded), or an entailment model conditioned on wider context
  than one sentence.
- **Division of labor**: the gate stays as the hard floor (constraint — reward cannot
  *guarantee*); the reward term supplies the gradient that teaches the infiller to *propose*
  truthful fills in the first place, reducing resample/fallback rates and covering the graded
  region the binary gate cannot express.
- **Cost**: T reuses the already-loaded NLI encoder per candidate sentence — batched, no new
  model; β joins α in the sweep space (one more knob → justify via ablation, per the
  no-calibration rule).
- **Trigger to implement**: E1 smoke measurement shows a material rate of gate-rejections /
  fallbacks driven by untruthful proposals, or eval shows wrong-sense fills passing the gate at
  a rate that damages utility.

Until then this issue stays open with the Phase-2 gate as the mitigation.
