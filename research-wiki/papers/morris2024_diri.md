---
type: paper
node_id: paper:morris2024_diri
title: "DIRI: Adversarial Patient Reidentification with Large Language Models for Evaluating Clinical Text Anonymization"
authors: ["John X. Morris", "Thomas R. Campion", "Sri Laasya Nutheti", "Yifan Peng", "Akhil Raj", "Ramin Zabih", "Curtis L. Cole"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2410.17035"
  doi: null
  s2: null
tags: ["re-identification", "adversarial-evaluation", "llm-attacker", "clinical", "rd2", "rd4"]
added: 2026-07-02T00:00:00Z
---

# DIRI: adversarial patient re-identification for evaluating anonymization

## One-line thesis
An LLM-based adversary that re-identifies the patient behind a redacted clinical note — an evaluation
instrument showing current de-identifiers still leak.

## Key Results
- Against Philter (rule), BiLSTM-CRF, and ClinicalBERT deidentifiers on Weill Cornell data, DIRI still
  re-identified **9% of notes even after ClinicalBERT masking**.

## Relevance to This Project
**Why surfaced:** round-2 (Semantic Scholar) discovery — the *attacker/evaluator* side of RD4/RD2 in
[`docs/research/learned-substitution.md`](../../docs/research/learned-substitution.md). **Fit:** the
concrete "evaluate a substitutor against an LLM re-identifier" instrument the taxonomy demands (F4/A1);
by John Morris (embedding-inversion lineage), it complements
[Staab](staab2024_llm_anonymizers.md)/[Pang](pang2024_reconstruction_dp_text_llm.md) on the attack side.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
