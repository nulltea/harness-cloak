---
type: experiment
node_id: exp:extractor-pointer-by-type
title: "Survived-span recovery by type (corrected denominator)"
idea_id: ""
verdict: partial
confidence: medium
date: 2026-07-05
hardware: "host .venv ROCm GPU; Qwen3.6-35B alignment judge + GLiNER detector + MiniLM"
duration: "~few min judge (Qwen -np 1, cached out_p) + deterministic recovery pass"
provenance: "scripts/spikes/survival_by_type.py; results/survival_by_type.json; scripts/spikes/extractor_pointer_by_type.py; results/extractor_pointer_by_type.json; scripts/spikes/extractor_miss_audit.py; results/extractor_miss_audit.json"
tags: ["extractor", "survived-span", "detector-pointer", "semantic-window", "llm-judge", "denominator", "tab-types"]
---

# Survived-span recovery by type (corrected denominator)

**verdict:** `partial`  .  **confidence:** `medium`

> Supersedes this record's first version, which measured extractor recovery against a
> `present_in_out_p` denominator defined as rapidfuzz `partial_ratio >= 60` of the fill
> against `out_p`. That denominator was wrong twice over (below); its headline "rule 27/60,
> semantic-window 28/60, detector-pointer 28/60 — a tie" is a **denominator artifact**, not
> a measure of extractor quality.

## Question

Of the spans the substitutor actually generalized, how many have their **substituted
(generalized) content reach `out_p`** (a "survived span" — the only thing an extractor can
recover), and how much of that does the extractor recover, **per TAB type**?

Whether a span was detected/substituted at all is out of scope (detector/substitutor
concern, or a utility-preserving choice). This measures survival and recovery only.

## Two denominator errors in the first version

1. **Fuzzy noise counted as present.** `partial_ratio >= 60` of a generic fill
   (`"at some point"`, `"something"`, `"a city in United States"`) aligns somewhere in a
   long note by chance. On the miss-audit band, ~87% of `band60_90` "present" fills were
   noise (cos 0.01–0.15 to their aligned window), not echoes.
2. **Leaked originals miscredited as survival.** An LLM judge (below) then over-counted
   the other way: it credited a span as "survived" when the **original surface** appeared in
   `out_p`, even when that surface reached `out_p` via an **undetected duplicate leaking
   through `doc_p`** (offset-scoped substitution replaces only the detected occurrence;
   `substitute.py:114`), while the substituted *fill* never appeared. That is a privacy
   leak, not the substituted span surviving.

## Method (corrected)

- Inputs: `data/ranker_env.json`, `data/task_arms_tau0.02.json`; corpus `clinical`;
  16 docs requested, 14 with spans; 75 generalization entries in `R`.
- `out_p = Remote(task_prompt(doc_p))` with the pinned reward env (gemma 4 E4B, temp 0,
  non-thinking), cached.
- **Survival judge** (`survival_by_type.py`): `Qwen3.6-35B-A3B`, temp 0, non-thinking — a
  different family from gemma, so no self-preference on its own output. One call per doc,
  all spans batched, structured JSON per span `{label ∈ SURVIVED|REWORDED|TEMPLATED|ABSENT,
  quote}`. Robustness anchors: (a) exact-fill matches auto-labeled SURVIVED (also the
  calibration ground truth); (b) **quote grounding** — a SURVIVED/REWORDED claim must quote
  a verbatim `out_p` substring, else downgraded to ABSENT (kills hallucinated positives).
- **Leaked-only guard** (deterministic): a judged-survived span is excluded from the
  substituted-content count when the fill is *not* present in `out_p` (exact or fuzzy≥90)
  but the original surface *is* — i.e. only the leaked original is present.
- **Recovery decomposition** (deterministic): each survived span classified by what `out_p`
  actually contains, which fixes the recovering mechanism exactly.

## Results — survival by type (corrected)

Denominator = generalizations whose substituted content reached `out_p`.

| Type | Substituted | Judge-survived | Leaked-only (excl.) | **Subst. survived** | Rate |
|---|---:|---:|---:|---:|---:|
| PERSON | 0 | 0 | 0 | 0 | — |
| ORG | 0 | 0 | 0 | 0 | — |
| LOC | 2 | 0 | 0 | 0 | 0.0 |
| DATETIME | 10 | 2 | 0 | 2 | 0.20 |
| CODE | 0 | 0 | 0 | 0 | — |
| QUANTITY | 7 | 5 | 0 | 5 | 0.71 |
| DEM | 43 | 21 | 7 | 14 | 0.33 |
| MISC | 13 | 9 | 5 | 4 | 0.31 |
| **TOTAL** | **75** | **37** | **12** | **25** | **0.33** |

The real survived denominator is **25**, not the judge's 37 and not the first version's 60.
PERSON/ORG/CODE had no surviving generalizations in this slice, so it says nothing about
those types.

## Results — population decomposition (37 judge-survived)

| Population | n | `out_p` contains | substituted-content? |
|---|---:|---|---|
| A. Leaked, fill also echoed | 2 | fill **and** leaked original | yes |
| A. Leaked-only | 12 | ONLY the original (fill absent; 11/12 un-replaced in `doc_p`) | **no — leak** |
| B. Fill verbatim | 14 | the fill exactly | yes |
| C. Fill fuzzy | 7 | the fill at ≥90 fuzzy | yes |
| D. Reworded | 2 | neither fill nor exact original | yes (reworded fill) |

Substituted-content survived = B(14) + C(7) + D(2) + A-both(2) = **25**. Leaked-only 12 are
the privacy leak.

## Results — recovery of the 25

The current rule cascade (`invert()`: placeholder + exact + fuzzy-90 + semantic-window)
recovers **23/25** by construction: it inverts every fill that is present exactly or at
fuzzy≥90 (B + C + A-both = 23). The residue is the 2 D-cases:

| Original surface | Fill | `out_p` mention | Verdict |
|---|---|---|---|
| coronary artery bypass grafting | something | "CABG surgery" | **recoverable** — abbreviation of the original; needs acronym/alias match |
| the next couple weeks | at some point | "a couple of weeks ago" | **abstain** — semantic drift (future→past); substituting the original asserts a false fact |

So the safe recovery ceiling is **~24/25** (23 + CABG), with the drift case correctly left
untouched. Detector-pointer and the MiniLM semantic-window fallback add nothing here: the
one recoverable miss needs abbreviation matching against the *original surface*, not fill
similarity, and the other needs an abstain, not a match. Their first-version "one extra
recovery each" landed inside the fuzzy-noise band and does not survive the corrected
denominator.

## Calibration & honesty caveats

- Judge vs exact-match ground truth: **11/15 = 0.73 agreement** — the judge misses ~27% of
  certain fill echoes, so judge-survival is a **lower bound**; true survival is likely
  slightly above 25. Grounding downgrades were clean (1/43 = 2%), so it does not over-claim.
- The leaked-only exclusion is deterministic (fill-present vs surface-present), not judge-
  dependent, so the 25 vs 37 split is robust even where the judge is noisy.
- Single clinical slice, 14 docs; lexsum, placeholders, and a larger sample are unrun and
  will surface the first PERSON/ORG/CODE survivals and more D-cases.

## Interpretation

The extractor was never "far from best" on survivable content: on the corrected denominator
it already recovers ~23/25, and the honest ceiling is ~24/25 once a semantic-drift case is
abstained. The apparent large gap in the first version was denominator pollution (fuzzy
noise + miscredited leaks). The remaining design work is narrow: an **original-surface +
acronym/alias proposer** plus an **NLI abstain gate** (see the design doc), not the DEM/MISC
"miss" chase the detector-pointer v2 premise assumed. The larger issue this surfaced —
undetected duplicate mentions leaking originals into `doc_p` (12/75 here) — is a
detector/substitutor coverage matter, out of the extractor's scope but worth its own record.

## Artifacts

- `scripts/spikes/survival_by_type.py`, `results/survival_by_type.json` (corrected survival)
- `results/extractor_miss_audit.json` (band echo classes, cos)
- `results/extractor_pointer_by_type.json` (first-version comparison, polluted denominator)

## Connections

Informs and corrects the extractor design in
`docs/plans/2026-07-05-survived-recovery-extractor.md`
(supersedes the premise of `docs/plans/2026-07-05-detector-pointer-extractor-v2.md`).
Edges not yet recorded in `graph/edges.jsonl`.
