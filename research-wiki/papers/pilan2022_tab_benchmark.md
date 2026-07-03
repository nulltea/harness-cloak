---
type: paper
node_id: paper:pilan2022_tab_benchmark
title: "The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization"
authors: ["Ildikó Pilán", "Pierre Lison", "Lilja Øvrelid", "Anthi Papadopoulou", "David Sánchez", "Montserrat Batet"]
year: 2022
venue: "Computational Linguistics 48(4)"
external_ids:
  arxiv: "2202.00443"
  doi: "10.1162/coli_a_00458"
  s2: null
tags: ["pii-detection", "anonymization", "benchmark", "quasi-identifiers", "corpus"]
added: 2026-07-03T00:00:00Z
---

# The Text Anonymization Benchmark (TAB)

## One-line thesis
An annotated corpus of 1,268 ECHR court cases (1,014 train / 127 dev / 127 test) with span-level
annotations of direct and quasi-identifiers across 8 entity types, plus privacy-weighted evaluation
metrics for text anonymization.

## Problem / Gap
NER corpora don't capture anonymization: what must be masked is not "named entities" but any
direct or quasi-identifying information, including demographics, quantities, and free-form
circumstantial descriptions. No prior corpus annotated identifier *type* (DIRECT vs QUASI),
confidential attributes, and coreference for anonymization evaluation.

## Method
Multi-annotator span annotation of ECHR judgments with 8 entity types (PERSON, ORG, LOC, DATETIME,
CODE, QUANTITY, DEM, MISC), each mention tagged DIRECT (identifies alone) or QUASI (identifies in
combination), with coreference chains. Evaluation framework weights recall over precision
(privacy-first, F2-style) and scores against the union/intersection of annotators. Baseline:
Longformer (RoBERTa-based long-document encoder) fine-tuned for token classification on the train
split; an early RoBERTa variant scored 2–4 F1 points lower, motivating the long-context encoder.

## Key Results
- Fine-tuned Longformer baseline achieves high recall on direct identifiers and good recall on
  quasi-identifiers on the test split — supervised in-domain token classification is a strong
  recipe for this corpus.
- Rule/dictionary and off-the-shelf NER baselines under-recall quasi-identifiers badly.

## Relevance to This Project
TAB is our detection-gate corpus (`corpora/tab/`, `scripts/latticecloak_detection_gate.py`) and the
source of the project's universal entity schema. Its train/dev splits are already local and unused —
they are the primary supervised signal for the planned fine-tuned span detector. The paper's own
Longformer result is direct evidence that fine-tuning a compact encoder on TAB train fixes exactly
the QUASI-recall gap our zero-shot GLiNER∪Presidio detector shows (MISC 0.21, QUANTITY 0.25,
DEM 0.56 any-recall).

## Limitations / Failure Modes
- Single domain (legal, ECHR English); models fine-tuned on it inherit court-writing style.
- Annotator disagreement on QUASI boundaries; union-gold contains noisy/broad positives.

## Reusable Ingredients
Train/dev/test splits (local), `evaluation.py` privacy-weighted metrics, `longformer_experiments/`
fine-tuning code as reference.

## Open Questions
How well does a TAB-fine-tuned detector transfer to our non-legal corpora (Enron, clinical,
SynthPAI)?
