---
type: paper
node_id: paper:zaratiana2026_gliner2_pii
title: "GLiNER2-PII: A Multilingual Model for Personally Identifiable Information Extraction"
authors: ["Urchade Zaratiana", "Ash Lewis", "George Hurn-Maloney"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2605.09973"
  doi: null
  s2: null
tags: ["pii-detection", "span-detection", "benchmark", "gliner", "ood-evaluation", "recall"]
added: 2026-07-03T00:00:00Z
---

# GLiNER2-PII: A Multilingual Model for Personally Identifiable Information Extraction

## One-line thesis
A 0.3B GLiNER2 adaptation for 42-type PII span extraction, trained on 4,910 constraint-generated
multilingual synthetic texts — but the durable contribution here is the **SPY benchmark**, a
naturally-occurring OOD test set on which every off-the-shelf PII detector, including OpenAI's
Privacy Filter, degrades sharply.

## Problem / Gap
PII spans are "heterogeneous, locale-dependent, context-sensitive, and often embedded in noisy or
semi-structured documents." Training data is scarce (real PII can't be shared), so detectors are
trained on synthetic corpora — and the paper's evaluation shows in-distribution synthetic F1 does
not transfer to real legal/medical text.

## Method
- Model: 0.3B params, adapted from GLiNER2 (arXiv 2507.18546, span–label matching head), 42 PII
  entity types at character-span resolution, multilingual.
- Data: constraint-driven generation pipeline → 4,910 annotated texts across languages, domains,
  formats, and entity distributions.
- Released on HF (`fastino/gliner2-privacy-filter-PII-multi`).

## Key Results — the SPY benchmark head-to-head (the reason this page exists)
**SPY benchmark** = two naturally-occurring OOD subsets: **Legal Questions** (100 legal Q&A docs)
and **Medical Consultations** (100 medical transcripts), chosen precisely to test generalization
away from synthetic training distributions. Five systems compared; GLiNER2-PII wins span-F1.
Table 2 extract for the two most relevant baselines:

| System | Subset | Precision | Recall | F1 |
|---|---|---|---|---|
| OpenAI Privacy Filter | Legal | 0.250 | 0.640 | 0.360 |
| OpenAI Privacy Filter | Medical | 0.271 | 0.671 | 0.386 |
| urchade/gliner_multi_pii-v1 | Legal | 0.522 | 0.308 | 0.388 |
| urchade/gliner_multi_pii-v1 | Medical | 0.483 | 0.314 | 0.381 |

The two baselines sit at opposite ends of the same tradeoff: OpenAI Privacy Filter is **high-recall,
low-precision** (catches ~0.64–0.67 of spans but a majority of flags are false positives);
`gliner_multi_pii-v1` is **high-precision, low-recall** (few false positives, misses ~0.69 of
spans). The paper argues that for redaction — where a missed span is more costly than
over-redaction — the recall-favoring profile is preferable, which is why GLiNER2-PII targets high
recall while lifting precision.

## Relevance to This Project
Two direct bearings on the span detector (`docs/research/learned-PII-detection.md`):
1. **Independent corroboration of the benchmark-honesty problem.** Our detection gate is measured
   on TAB (real ECHR court text), not synthetic PII corpora. SPY shows exactly why that matters:
   detectors scoring F1 ~0.96 on synthetic PII-Masking collapse to F1 0.36–0.39 on real
   legal/medical text. This is the detector-level instance of the project's empirical-honesty rule
   — measure against the real distribution, not the surface benchmark the model was fit to. It
   independently supports our expectation that off-the-shelf PII fine-tunes will underperform on
   the TAB gate.
2. **Recall-bias is the right error asymmetry — and even safer here than in plain redaction.** For
   LatticeCloak a missed span survives verbatim in `doc_p` (a hard privacy leak), while an
   over-detected span is only a utility cost that the lattice/RL utility term can recover from. So
   the recall-favoring conclusion holds a fortiori. The precision/recall split of the two baselines
   is the design axis: OpenAI Privacy Filter's bias direction is right but its precision collapse is
   severe; `gliner_multi_pii-v1`'s precision-bias is the wrong direction for us.

## Limitations / Failure Modes
- SPY subsets are small (100 docs each); still synthetic-free but low-n.
- Trained on 42 **formal-PII** types; no TAB-style quasi-identifiers (identifying events,
  demographics, quantities) — same label-space gap as every other public PII fine-tune.
- Preprint.

## Reusable Ingredients
The SPY benchmark as an OOD sanity check; the constraint-driven synthetic-generation recipe as an
auxiliary-data option; the head-to-head numbers as the comparison bar for real-domain PII detection.

## Open Questions
Does the recall-favoring training generalize to quasi-identifier spans, or only to the 42 formal
types? (Not tested — SPY annotates formal PII.)
