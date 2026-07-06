---
type: experiment
node_id: exp:extractor-pointer-by-type
title: "Survived-span recovery by type (corrected denominator)"
idea_id: ""
verdict: partial
confidence: medium
date: 2026-07-05
hardware: "host .venv ROCm GPU; Qwen3.6-35B alignment judge + GLiNER detector + MiniLM"
duration: "151-doc pilot run (clinical+lexsum): judge Qwen -np 1 + deterministic recovery pass"
provenance: "scripts/spikes/survival_by_type.py; results/survival_by_type.json; results/survival_recovery_pilot.json; scripts/spikes/extractor_pointer_by_type.py; scripts/spikes/extractor_miss_audit.py; results/extractor_miss_audit.json"
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

- Inputs: `data/ranker_env_pilot.json`, `data/task_arms_pilot.json`; corpora
  `clinical,lexsum`; 80/80 docs requested, **151 with spans**; **1059 generalization
  entries** in `R`. (A first pass used the 16-doc frozen clinical env; the pilot run below
  supersedes it and is the reported result.)
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

## Results — survival by type (151 docs, clinical+lexsum)

Denominator = generalizations whose substituted content reached `out_p`.

| Type | Substituted | Judge-survived | Leaked-only (excl.) | **Subst. survived** | Rate |
|---|---:|---:|---:|---:|---:|
| ORG | 144 | 68 | 12 | 56 | 0.39 |
| LOC | 172 | 68 | 6 | 62 | 0.36 |
| DATETIME | 265 | 41 | 0 | 41 | 0.16 |
| QUANTITY | 89 | 15 | 1 | 14 | 0.16 |
| DEM | 272 | 122 | 29 | 93 | 0.34 |
| MISC | 117 | 53 | 26 | 27 | 0.23 |
| **TOTAL** | **1059** | **367** | **74** | **293** | **0.28** |

**~28% of generalizations survive** into `out_p` as recoverable content; 74 more are
leaked-only (privacy leak, excluded). PERSON and CODE never appear — those types are
placeholdered (`<PERSON_1>`/`<CODE_1>`), not generalized, so they are outside this measure
(placeholder-token echo is a separate, trivially-invertible channel). DATETIME/QUANTITY
survive least (0.16) — exact dates/amounts generalize to lossy forms (decade, range) that
rarely echo; ORG/LOC/DEM survive most.

## Results — population decomposition (367 judge-survived)

| Population | n | `out_p` contains | substituted-content? |
|---|---:|---|---|
| A. Leaked + fill echoed | 21 | fill **and** leaked original | yes |
| A. Leaked-only | 74 | ONLY the original (fill absent) | **no — leak** |
| B. Fill verbatim | 181 | the fill exactly | yes |
| C. Fill fuzzy | 37 | the fill at ≥90 fuzzy | yes |
| D. Reworded | 54 | neither fill nor exact original | yes (reworded/lossy) |

Substituted-content survived = A-both(21) + B(181) + C(37) + D(54) = **293**.

## Results — recovery of the 293

Current rule cascade (`invert()`: placeholder + exact + fuzzy-90 + semantic-window) recovers
**~239/293 ≈ 82%** (surface-in-`out_final` proxy; A-both 21, B ~174/181, C ~34/37 — the few
B/C misses are proxy artifacts from multi-word surfaces and walk-order placeholder
collisions, not real misses). The gap is the **54 D-reworded** spans, and inspecting them
shows D is four distinct phenomena — only two are recoverable:

Extraction is **client-side**, and R holds every original surface. So none of these are
"unrecoverable" for lack of information — the specifics were removed only from the *remote*
model's view; the client legitimately restores them from R. Every D sub-class is recoverable
*in principle*; what varies is the **localization difficulty** (finding the reworded/lossy
fill mention in `out_p`) and the **false-match risk** (matching a mention that is not that
fill).

| D sub-class | example | limit on recovery |
|---|---|---|
| 1. Fill reworded below fuzzy-90 | `an organization`→"the organization"; `a person of color`→"color employees" | easy — semantic / lower-threshold fill match; original from R |
| 2. Acronym / alias of original | `coronary artery bypass grafting`→"CABG surgery"; `O'Reilly Auto Parts`→"O'Reilly Automotive" | easy — original-surface alias/acronym match |
| 3. Lossy generalization | `january 13th 1982`→"Early 1980s"; `Minneapolis`→"in Minnesota"; `the United States`→"a state" | **localization** — the fill echoed reworded; locate it, then swap the exact original from R (NOT fabrication — R has it) |
| 4. Model re-derived a different specific | `the last four years`→"three years ago"; `the District of Kansas`→"Wichita U.S. District Court" | **false-match** — restoring the true original from R corrects the model's guess, but only if the mention really is that fill; else abstain |

So the true ceiling is **~all 293 survived spans**, bounded not by lost information but by
**localization precision on reworded mentions and false-match avoidance**. The rule cascade
gets the string-matchable ~82% (A/B/C + D-1/2 aliases); the reworded/lossy residue (D-1
semantic, D-3) needs paraphrase-level localization, which is what a learned reconstructor
(Design 3) is for. D-4 is the precision boundary: a verification/abstain gate, not a wall.

## Calibration & honesty caveats

- Judge vs exact-match ground truth: **150/195 = 0.77 agreement** — the judge misses ~23% of
  certain fill echoes, so judge-survival is a **lower bound**; true survival is somewhat
  above 293. Grounding downgrades clean (3/384 = 0.8%), so it does not over-claim.
- The leaked-only exclusion is deterministic (fill-present vs surface-present), not judge-
  dependent, so the 293-vs-367 split is robust even where the judge is noisy.
- The 82% recovery is a `surface ∈ out_final` proxy; mention-anchored recovery (design doc)
  is needed to pin B/C exactly and to score D-classes 1–2 precisely.
- Two corpora (clinical + lexsum). D-class ratios differ by corpus (lexsum contributes the
  legal-entity aliases and lossy court/location cases); a per-corpus D breakdown is unrun.

## Interpretation

The extractor already recovers ~82% of survived content by string matching; the remaining
~18% is the D-reworded residue. Because extraction is client-side and R holds every original,
**the true ceiling is ~100% of survived spans** — the residue is a localization problem
(finding the reworded/lossy fill mention), not lost information, and a false-match precision
boundary (D-4), not a wall. This is exactly the case for a **learned reconstructor (Design
3)**: a local seq2seq that reads `out_p + R` and rewrites, resolving paraphrase-level
localization that string/fuzzy/acronym matching cannot, with a verification gate bounding
false substitutions. The larger issue this surfaced — undetected duplicate mentions leaking
originals into `doc_p` (74/1059 = 7% here) — is a detector/substitutor coverage matter, out
of the extractor's scope but worth its own record.

## Artifacts

- `scripts/spikes/survival_by_type.py`, `results/survival_by_type.json` (151-doc survival)
- `results/survival_recovery_pilot.json` (A/B/C/D populations + D-case listing)
- `results/extractor_miss_audit.json` (band echo classes, cos)
- `results/extractor_pointer_by_type.json` (first-version comparison, polluted denominator)

## Connections

Informs and corrects the extractor design in
`docs/plans/2026-07-05-survived-recovery-extractor.md`
(supersedes the premise of `docs/plans/2026-07-05-detector-pointer-extractor-v2.md`).
Edges not yet recorded in `graph/edges.jsonl`.
