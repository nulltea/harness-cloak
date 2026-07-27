---
type: research
status: current
created: 2026-07-10
updated: 2026-07-27
tags: [detector, gliner, noise-gate, mining, injury, health-condition, data-quality, issue-register]
companion: [../handoffs/2026-07-09-detector-noise-investigation.md]
---

# Issue — mined-span junk and the deny-list noise gate's precision ceiling

The clinical span miner (`scripts/build_mined_lattice_profiles.py`, GLiNER-large
`knowledgator/gliner-pii-large-v1.0`) emits a substantial fraction of non-entity and wrong-type
surfaces. The negative filter that is supposed to remove them (`is_noise_span`, `src/cloak/detection/detect.py`)
is a **rule-based deny-list that fails open**, so it only removes the noise categories someone has
explicitly enumerated. It cannot recognize noise it hasn't been told about. This document records the
observed junk, the gate's mechanism and why it structurally can't close the gap, and the fix options.

Measured on the `injury` label of the large re-mine
(`results/mined_lattice_profile_spans_large.jsonl`) while splitting injury into its own runtime type
(see `2026-07-09-detector-noise-investigation.md` companion). The injury bucket is the worst case, but
the same failure mode applies to every mined type — the gate has no positive entity model.

## 1. What the gate is (and is not)

`is_noise_span(surface, runtime_type)` is **purely rule-based** — no embeddings, no model, no semantic
similarity. It performs exactly three operations:

- **Normalization** — lowercase, strip non-alphanumerics, collapse whitespace (`_noise_norm`), plus a
  spaces-removed `compact` form for abbreviations.
- **Set membership** — `in` checks against hand-written `frozenset`s: `_NOISE_LAB_TESTS`,
  `_NOISE_IMAGING_DIAGNOSTICS`, `_NOISE_ANATOMY`, `_NOISE_DEVICE_SUPPLIES`, `_NOISE_KEEP_TOKENS`.
- **Regex** — `_NOISE_DEVICE_PATTERN`, `_NOISE_LAB_PATTERN`, `_NOISE_LEGAL_ADMIN_PATTERN`,
  `_NOISE_JUNK_NUMERIC`, `_NOISE_FRAGRANCE_EXCIPIENT`, plus a catch-all for single tokens ≤2 chars.

Order of evaluation: KEEP allowlist wins first → lab → imaging → device → anatomy → legal/admin →
numeric/fragrance → ≤2-char single token. **Anything not matched fails open** (returns `False` = "not
noise"), by explicit design — the gate errs toward keeping real entities and never drops on a guess. It
only runs for `runtime_type in _NOISE_FILTER_TYPES` (`drug`, `health-condition`, `medical-procedure`,
and now `injury`).

It does **not** use `data/lattice_profiles/lattice_profiles.embindex.npz` (the repo's embedding index)
or any nearest-neighbor / similarity signal. That asset exists for substitution/retrieval, not filtering.

## 2. Observed junk (injury label, large re-mine)

After `injury→injury` mapping, best-label dedup, generic-surface skip, and the noise gate, **200
injury entries** remained, of which roughly half are noise the gate cannot catch:

- **Anatomy the gate misses** — `appendix`, `colon`, `kidneys`, `liver`, `throat`, `tonsils`,
  `uterus`, `ovaries`, `diaphragm`, `fallopian tubes`, `parathyroids`, `nerve roots`. The gate caught
  the 14 limb/torso terms it *does* enumerate (`ankle`, `arm`, `back`, `foot`, `knee`, `hip`,
  `shoulder`, `elbow`, `lower back`, `bun`, …) but `_NOISE_ANATOMY` lists **limbs and torso regions
  only — no internal organs**.
- **Arbitrary nouns / true junk** — `brick`, `job`, `pandemic`, `asbestos`, `saps`, `salty food`,
  `ant bait`, `drunk driver`, `dawn knots legs`, `dements`, `metal issues`. In no blocklist; the ≤2-char
  catch-all doesn't fire (`brick` is 5 chars, one token).
- **Wrong-type but real entities (cross-type misroute, not junk)** — conditions `brain cancer`,
  `headache`, `ulcer`, `cataract`, `nausea`; procedures `cervical spine mri`, `kidney transplant`,
  `rotator cuff repair`, `gall bladder removed`. The gate correctly leaves these (they *are* entities);
  they need type-routing, not noise-dropping.

Confident model errors persist regardless of label wording: probing showed GLiNER tags `stroke` (0.87)
and `bruising` (0.73) as injury even with the most specific label string
(`"physical injury from trauma or accident"`). Some of these are arguably not even wrong (a contusion
*is* a soft-tissue injury), i.e. the injury/condition boundary genuinely overlaps.

## 3. Root cause

The gate is a **precision tool by construction**: it removes only what it can positively confirm is a
known non-entity category, and fails open on everything else to avoid dropping real entities. Its
coverage is therefore exactly the strings enumerated in its sets and patterns — nothing generalizes.
A synonym, a morphological variant, an organ not on the list, or a random noun all pass. This is not a
bug in the rules; it is the ceiling of a deny-list with no positive entity model. Enlarging the
enumerations (e.g. adding organs) handles specific families but never closes the long tail
(`brick`, `pandemic`).

Upstream, the miner's `threshold=0.3` admits a lot of low-confidence spans (~48% of injury detections
scored <0.5), and GLiNER's zero-shot label buckets are inherently low-precision on broad labels like
`injury`. So junk enters cheaply and the gate can only remove the fraction it recognizes.

## 4. Fix options (not yet decided)

Ordered cheapest-first; these are alternatives, not a sequence.

1. **Enlarge the deny-list enumerations** — add an organ/anatomy set, extend junk patterns. Cheap,
   deterministic, auditable. Closes the anatomy family; does nothing for the arbitrary-noun tail.
2. **Keyword positive lexicon per type** — e.g. keep an injury surface only if it matches an injury
   lexicon (`fracture|sprain|strain|tear|wound|laceration|contusion|dislocation|rupture|injury|trauma|
   burn|herniation`). High precision, cheap, no model; misses idioms (`broke my ankle`,
   `twisted my knee`).
3. **Semantic gate** — embed each mined surface (reuse the embedding index) and reject those whose
   nearest real-entity neighbors are all anatomy/junk, or that fall below a similarity floor to any
   real type exemplar. Generalizes beyond enumerated strings. Cost: needs a labeled anchor set and a
   **calibrated threshold** — and per the empirical-honesty rule that threshold must be fixed, not a
   per-run fudge knob.
4. **Model triage** — classify each survivor (injury/condition/procedure/noise) with the producer's
   own LLM. Highest precision, also fixes cross-type misroute (#2 in section 2); ~O(surviving-spans)
   cheap classification calls. This is what the injury cleanup is currently weighing.
5. **Raise the miner threshold** — orthogonal upstream lever; cuts low-confidence junk at the cost of
   recall on real low-score entities. Does not touch confident errors (`stroke`, `bruising`).

Note options 3–4 add a semantic/positive-entity signal the gate structurally lacks; 1–2 stay within
the deny-list paradigm and only extend its reach.

## Artifacts / pointers

- Gate: `src/cloak/detection/detect.py` — `is_noise_span`, `_NOISE_*` sets/patterns, `_NOISE_FILTER_TYPES`.
- Miner: `scripts/build_mined_lattice_profiles.py` — `DETECTOR_LABELS`, `LABEL_TO_RUNTIME_TYPE`,
  `_unique_spans` (best-label dedup), `is_noise_span` call site.
- Raw spans measured: `results/mined_lattice_profile_spans_large.jsonl`.
- Companion investigation (stock vs fine-tuned detector, injury split):
  `docs/handoffs/2026-07-09-detector-noise-investigation.md`.
