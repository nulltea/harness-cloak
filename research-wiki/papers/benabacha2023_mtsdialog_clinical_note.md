---
type: paper
node_id: paper:benabacha2023_mtsdialog_clinical_note
title: "An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters (MTS-Dialog)"
authors: ["Asma Ben Abacha", "Wen-wai Yim", "Yadan Fan", "Thomas Lin"]
year: 2023
venue: "EACL 2023"
external_ids:
  arxiv: null
  doi: null
  acl: "2023.eacl-main.168"
tags: ["benchmark", "clinical-note-generation", "task-utility", "pii-rich", "dialogue-summarization", "eval-corpus"]
added: 2026-07-02T00:00:00Z
---

# An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters (MTS-Dialog)

## One-line thesis
A 1.7k-pair corpus of short doctor–patient conversations with corresponding section-structured clinical notes,
plus an empirical study of models, data augmentation, and guided summarization for note generation.

## Method / Contents
MTS-Dialog: 1,700 doctor–patient dialogues (~16k turns) with clinical-note summaries (train 1,201 / valid 100 /
plus test). Conversations are relatively short; reference notes are concise, each carrying a section header
specifying the note category (few words to one paragraph). The paper studies task feasibility and existing
LMs, data augmentation, and guided summarization, comparing n-gram (ROUGE), contextual-embedding (BERTScore),
and fact-extraction metrics alongside expert NLG evaluation.

## Key Results
- Larger and shorter-form complement to ACI-Bench; section-header conditioning improves note generation.
- Provides reference summaries + splits for reference-scored utility on note generation.

## Limitations / Failure Modes
- Short single-section notes (less comprehensive than ACI-Bench's full SOAP). Reference notes are concise, so
  some dialogue detail is legitimately dropped — utility scoring must not penalize licensed omission.

## Relevance to This Project
**Why surfaced:** the larger clinical corpus in the task-oriented eval adopted for the LatticeCloak next stage
(§5.4, [`docs/html/LatticeCloak.html`](../../docs/html/LatticeCloak.html)); paired with ACI-Bench for scale +
full-SOAP coverage. **Fit:** the note restates the quasi-identifiers LatticeCloak coarsens (age, conditions,
meds, dates), so it exercises rung-A inversion and makes utility sensitive to τ — the gap SynthPAI
summarization/QA could not surface. Spec: [`docs/specs/benchmarks.md`](../../docs/specs/benchmarks.md).

## Connections
_Edges recorded in `graph/edges.jsonl`._ Companion clinical corpus:
[`yim2023_acibench_visit_note_generation`](yim2023_acibench_visit_note_generation.md).
