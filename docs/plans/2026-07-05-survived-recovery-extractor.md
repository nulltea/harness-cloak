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

The denominator must be spans whose **substituted (generalized) content** reached out_p —
i.e. the fill, or a rewording of it, is present. The first LLM-judge pass reported 37
"survived", but that conflated two phenomena (caught on review): a leaked-original mention
is present in out_p not because the substituted span survived, but because an **undetected
duplicate of the original leaked through doc_p** (see the survival investigation). Splitting
the 14-doc clinical slice by *what actually appears in out_p*:

| Population | n | out_p contains | is it substituted-content? | recover by |
|---|--:|---|---|---|
| B. Fill verbatim | 14 | the fill exactly | yes | exact swap (have) |
| C. Fill fuzzy | 7 | the fill at ≥90 fuzzy | yes | fuzzy swap (have) |
| D. Reworded | 2 | neither surface nor fill | yes (reworded fill) | see below |
| A. Leaked + fill echoed | 2 | fill AND leaked original | yes | exact swap (have) |
| **Substituted content survived** | **25** | | | |
| A. Leaked-only | 12 | ONLY the original (fill absent) | **no — leak, not survival** | out of scope |

**The real denominator is 25, not 37.** The 12 leaked-only spans (11/12 have the original
un-replaced elsewhere in doc_p) are the privacy leak, not a substituted span surviving — the
fill never reached out_p. They are out of extraction scope: out_final already carries the
original at that mention, so the extractor's only duty is do-no-harm (don't overwrite it).

Of the 25 substituted-content-survived spans, the current cascade recovers **B+C+A-both =
23/25** via fill matching. The residue is 2 D-cases:
- `coronary artery bypass grafting` → "CABG surgery" — abbreviation of the ORIGINAL,
  reconstructed by the model. **Recoverable** with acronym/alias matching.
- `the next couple weeks` → "a couple of weeks ago" — semantic drift (future→past).
  Substituting the original asserts a wrong fact. **Abstain, do not recover.**

So the honest ceiling is ~24/25 *safe* recovery: current 23 + the recoverable CABG case,
with the drift case abstained. The design closes the recoverable D-case and hardens do-no-
harm, rather than chasing the DEM/MISC "misses" v2 targeted — those turned out to be leaked
originals (A leaked-only, out of scope) and fuzzy noise, not recoverable fills (this
supersedes v2's premise).

## Root mechanism gap

`invert()` searches out_p for the **fill** only. That is sufficient for the fill-echo bulk
(B+C+A-both = 23/25). The one recoverable miss is where the model reconstructs an
**abbreviation of the original** (D-CABG: "coronary artery bypass grafting" → "CABG
surgery") — neither fill nor exact original is present, but the original's initialism is.
The original surface is known from R, so **add a proposer that searches for the original
surface and its variants** to catch this class. (This proposer also cheaply confirms
leaked-only originals as a do-no-harm guard, but those are already correct and not counted
as recovery.)

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
