---
type: plan
status: current
created: 2026-07-05
updated: 2026-07-05
tags: [extractor, invert, survived-recovery, original-surface-proposer, acronym, verification, abstain]
companion: [2026-07-05-detector-pointer-extractor-v2.md, ../../research-wiki/experiments/extractor-pointer-by-type.md]
---

# Survived-recovery extractor: recover ~all spans whose content reaches out_p

## Target, measured honestly

The denominator is spans whose **substituted (generalized) content** reached out_p — the
fill, or a rewording of it, is present. Measured on **151 docs (clinical + lexsum), 1059
generalizations** (`results/survival_by_type.json`, `results/survival_recovery_pilot.json`):

| Population | n | out_p contains | substituted-content? |
|---|--:|---|---|
| A. Leaked + fill echoed | 21 | fill AND leaked original | yes |
| B. Fill verbatim | 181 | the fill exactly | yes |
| C. Fill fuzzy | 37 | the fill at ≥90 fuzzy | yes |
| D. Reworded | 54 | neither exact fill nor exact original | yes |
| **Substituted content survived** | **293** | | |
| A. Leaked-only | 74 | ONLY the original (fill absent) | **no — leak, out of scope** |

The 74 leaked-only spans are the privacy leak (undetected duplicate of the original leaked
through doc_p), not a substituted span surviving — out of extraction scope; out_final already
carries the original there, so the extractor's only duty is do-no-harm (don't overwrite it).

The current cascade recovers **~239/293 ≈ 82%** (A-both + B + C, via fill matching). The
residue is the **54 D-reworded** spans, which are FOUR phenomena — only two recoverable:

Extraction is client-side and R holds every original, so none of these lack the information
to recover — the specifics were removed only from the *remote* model. What varies is
localization difficulty and false-match risk:

| D sub-class | example | limit |
|---|---|---|
| 1. Fill reworded below fuzzy-90 | `an organization`→"the organization" | easy — semantic/low-threshold fill match |
| 2. Acronym / alias of original | `coronary artery bypass grafting`→"CABG surgery" | easy — original-surface alias match |
| 3. Lossy generalization | `january 13th 1982`→"Early 1980s"; `Minneapolis`→"in Minnesota" | **localization** — locate the reworded fill, swap the exact original from R |
| 4. Model re-derived a different specific | `the last four years`→"three years ago" | **false-match** — restore from R only if the mention really is that fill; else abstain |

**The true ceiling is ~all 293 survived spans**, bounded by localization precision and
false-match avoidance, not by lost information. String/fuzzy/acronym matching (this doc's
deterministic proposers) gets ~82% + the D-1/2 aliases; the paraphrase-level residue (D-1
semantic, D-3) needs a **learned reconstructor (Design 3)**, and D-4 needs the verification/
abstain gate. This supersedes v2's DEM/MISC-miss premise (those were leaked originals + fuzzy
noise).

## Root mechanism gap

`invert()` searches out_p for the **fill** only. That covers the fill-echo bulk (A-both + B
+ C ≈ 239/293) but misses the two recoverable D sub-classes:
- **D-1 (fill reworded below fuzzy-90)** — the fill echoed with a determiner/morphology
  change ("an organization"→"the organization"). Needs semantic or lower-threshold matching
  on the *fill*.
- **D-2 (acronym/alias of the original)** — the model reconstructs a recognizable variant of
  the ORIGINAL ("coronary artery bypass grafting"→"CABG surgery", "O'Reilly Auto Parts"→
  "O'Reilly Automotive"). Neither exact fill nor exact original is present, but the original's
  initialism/alias is. The original surface is known from R.

So two additions: a **relaxed/semantic fill matcher** (D-1) and a **proposer that searches
for the original surface and its variants** (D-2). The latter also cheaply confirms
leaked-only originals as a do-no-harm guard (not counted as recovery). D-classes 3–4 get no
proposer — they are unrecoverable/abstain.

## Design

Extend the cascade with one proposer and one gate; existing exact/fuzzy/semantic paths keep
their behavior.

### Proposer: original-surface-variant match (new)

For each residue entry (surface `s`, fill `f`, type `t`), search out_p for a mention of `s`
using, in order:
1. **Exact `s`** (word-boundary) — do-no-harm confirm of leaked-only originals (already
   correct in out_final; a no-op, not counted as recovery), and locks those mentions so no
   later step overwrites them.
2. **Acronym / initialism** of `s` — generate the initialism ("coronary artery bypass
   grafting"→"CABG") and common medical/legal alias forms; match against out_p. Deterministic,
   no model.
3. **Morphological / case variants** — stem/lemma match ("grafting"↔"graft").
4. **Embedding fallback** — MiniLM cosine of `s` against detector-typed spans of type `t`
   (reuses the v2 detector proposer + the existing semantic scorer), for reworded synonyms.

A hit means the original concept is present; place `s` at that mention. This proposer runs
**before** the fill-fuzzy path when `s`-signal is strong (exact/acronym), because matching
the known original is higher-precision than fuzzy-matching a generic fill.

### Verification / abstain gate (do-no-harm)

Before any *non-exact* substitution (fill-fuzzy, semantic, embedding, acronym below a hard
match), require the located window to be consistent with the original concept:
- **NLI entailment** window ⊨ `s` (or `s` ⊨ window) with `nli-deberta-v3-small` (from v2), OR
- deterministic pass for exact-`s` / exact-fill / exact-acronym (no model needed).

If the window contradicts or is unrelated (the "a couple of weeks ago" drift: future vs
past), **abstain** — leave the model's text untouched. Abstaining is correct: asserting the
original where the model changed the meaning injects a false fact into out_final, worse than
a miss. Stat `gen_abstain`; report it separately.

### Do-no-harm on A_leaked (explicit)

When `s` is already present in out_p, never let a fill-based or embedding guess overwrite it.
Exact-`s` confirmation runs first and locks those mentions.

## Metric fix (measure recovery honestly)

Replace the `surface ∈ out_final` proxy (inflates on common-word surfaces like "nurse",
"daughter") with **mention-anchored recovery**: align the judge's grounded quote through the
inversion and check that *that* mention became `s`. This is the number to report per type;
the loose proxy is a diagnostic only.

## Base models / cost

No new heavy model. Acronym/alias + morphology = deterministic; embedding = MiniLM (in use);
NLI = `nli-deberta-v3-small` (v2, ~0.14B, one forward per non-exact candidate, local). The
learned pointer (detector-pointer v1/v2) stays optional for the reworded residue and is now
better seeded: the A/B/C/D labels + verification decisions are clean training triples.

## Evaluation & success

- Per-type mention-anchored recovery of survived spans vs the current cascade, at a
  **precision floor**: zero wrong-surface substitutions (a false recovery corrupts out_final).
- `gen_abstain` rate and the specific spans abstained (should be the semantic-drift class).
- Re-run on lexsum + a larger clinical sample (14 docs is small; expect more D-cases and the
  first PERSON/ORG/CODE survived spans, which this slice lacked).
- Shared gaming guard from v2 (exact-vs-recovered gap; abstain trajectory under RL).

## Risks

- **Acronym generation false positives** (a 4-letter initialism collides with an unrelated
  token) — gate acronym hits through the NLI/type check, don't accept bare initialism.
- **Over-recovery of drift** — the abstain gate is a hard constraint, not a diagnostic; a
  wrong substitution is worse than leaving the model's text.
- **Small-sample ceiling** — 36/37 is on 14 clinical docs; the mechanism ranking (original-
  surface anchor dominant) should be re-confirmed on lexsum/placeholders before it is fixed.
