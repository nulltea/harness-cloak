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
