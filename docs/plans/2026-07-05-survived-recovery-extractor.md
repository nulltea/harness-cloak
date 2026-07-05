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

The denominator is **survived spans** (content present in out_p; see
`results/survival_by_type.json`, LLM-alignment judge). On the 14-doc clinical slice, 37
generalizations survived, decomposing by what the remote model actually emitted:

| Population | n | out_p contains | recover by |
|---|--:|---|---|
| A. Leaked original | 14 | the ORIGINAL surface (undetected duplicate leaked) | confirm & leave |
| B. Fill verbatim | 14 | the fill exactly | exact swap (have) |
| C. Fill fuzzy | 7 | the fill at ≥90 fuzzy | fuzzy swap (have) |
| D. Reworded | 2 | neither surface nor fill | see below |

The current cascade (`invert()`: placeholder + exact + fuzzy + semantic) already recovers
**A+B+C = 35/37**. The residue is 2 D-cases:
- `coronary artery bypass grafting` → "CABG surgery" — abbreviation of the ORIGINAL,
  reconstructed by the model. **Recoverable** with acronym/alias matching.
- `the next couple weeks` → "a couple of weeks ago" — semantic drift (future→past).
  Substituting the original asserts a wrong fact. **Abstain, do not recover.**

So the honest ceiling is ~36/37 *safe* recovery, not a blind 37/37. The design closes the
recoverable D-case and hardens the do-no-harm behavior, rather than chasing the DEM/MISC
"misses" v2 targeted — those turned out to be leaked originals (A) and fuzzy noise, not
recoverable fills (this supersedes v2's premise).

## Root mechanism gap

`invert()` searches out_p for the **fill** only. But the model frequently produces
something closer to the **original**: it leaks the original verbatim (A, 38% of survived),
or reconstructs an abbreviation of it (D-CABG). The original surface is known from R and is
a stronger anchor than the fill for exactly these cases. **Add a proposer that searches for
the original surface and its variants.**

## Design

Extend the cascade with one proposer and one gate; existing exact/fuzzy/semantic paths keep
their behavior.

### Proposer: original-surface-variant match (new)

For each residue entry (surface `s`, fill `f`, type `t`), search out_p for a mention of `s`
using, in order:
1. **Exact `s`** (word-boundary) — actively confirms A_leaked instead of relying on the
   fill being absent. Idempotent: if `s` is already there, recovery is a no-op (correct).
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
