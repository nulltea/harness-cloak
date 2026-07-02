---
type: paper
node_id: paper:yim2023_acibench_visit_note_generation
title: "ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation"
authors: ["Wen-wai Yim", "Yujuan Fu", "Asma Ben Abacha", "Neal Snider", "Thomas Lin", "Meliha Yetisgen"]
year: 2023
venue: "Scientific Data (Nature)"
external_ids:
  arxiv: "2306.02022"
  doi: "10.1038/s41597-023-02487-3"
  s2: null
tags: ["benchmark", "clinical-note-generation", "task-utility", "pii-rich", "dialogue-summarization", "eval-corpus"]
added: 2026-07-02T00:00:00Z
---

# ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation

## One-line thesis
The largest dataset for AI-assisted clinical note generation from doctor–patient dialogue: full encounter
transcripts paired with complete, section-structured SOAP notes for end-to-end note-generation benchmarking.

## Method / Contents
207 role-played doctor–patient encounters, each with the full dialogue transcript and a full clinical note
covering all SOAP sections (Subjective / Objective / Assessment / Plan). Encounters are longer than MTS-Dialog
and the reference notes are complete rather than single-section. Ships patient metadata (age, name, gender).
Basis for the MEDIQA-Chat 2023 (ACL ClinicalNLP) TaskB full-note-generation and MEDIQA-Sum shared tasks.
Baselines benchmarked include fine-tuned and few-shot generative models; utility scored with ROUGE, BERTScore,
and fact-based metrics against the reference notes.

## Key Results
- Establishes reference notes + standardized splits so note generation can be scored against gold text
  (n-gram, contextual-embedding, and fact-extraction metrics), not just human preference.
- Small (207) but complete-SOAP, complementary to MTS-Dialog's larger, shorter, single-section corpus.

## Limitations / Failure Modes
- Small corpus (207); role-played, not real clinical encounters. Notes are lightly de-identified on direct
  names but retain the quasi-identifying clinical content (age, sex, conditions, medications, dosages, dates).

## Relevance to This Project
**Why surfaced:** primary task-oriented, PII-rich utility benchmark adopted to replace the prefix-continuation /
summarization eval flagged as insufficient in §5.4 of the LatticeCloak report
([`docs/html/LatticeCloak.html`](../../docs/html/LatticeCloak.html)). **Fit:** unlike SynthPAI summarization/QA
(which paraphrases *around* substituted spans, leaving rung-A inversion unexercised and utility insensitive to
τ), a SOAP note **must restate** the coarsened quasi-identifiers — age, sex, conditions, meds, dosages, dates —
so LatticeCloak's inversion actually fires and coarsening carries a measurable, reference-scored utility cost.
This is where the τ→utility Pareto and the rung-A→rung-B hypothesis become testable. Paired with the email
domain ([`zhang2019_aeslc_subject_line_generation`](zhang2019_aeslc_subject_line_generation.md)) as the two
adopted evaluation domains; spec in [`docs/specs/benchmarks.md`](../../docs/specs/benchmarks.md).

## Connections
_Edges recorded in `graph/edges.jsonl`._ Companion clinical corpus:
[`benabacha2023_mtsdialog_clinical_note`](benabacha2023_mtsdialog_clinical_note.md).
