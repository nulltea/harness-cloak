---
type: paper
node_id: paper:staab2024_llm_anonymizers
title: "Large Language Models are Advanced Anonymizers"
authors: ["Robin Staab", "Mark Vero", "Mislav Balunović", "Martin Vechev"]
year: 2024
venue: "ICLR 2025"
external_ids:
  arxiv: "2402.13846"
  doi: null
  s2: null
tags: ["llm-anonymization", "adversarial", "re-identification", "rd2", "rd4"]
added: 2026-07-02T00:00:00Z
---

# Large Language Models are Advanced Anonymizers

## One-line thesis
Turn the LLM's near-human attribute-inference ability into a defense: iteratively infer what an adversary
could deduce, then rewrite to remove it — adversarial anonymization.

## Method
An evaluation framework for anonymization against adversarial LLM inference, plus an
infer-then-rewrite loop leveraging the LLM's own inferential capability.

## Key Results
- Across 13 models on real + synthetic text, beats commercial anonymizers; 50-participant human
  evaluation strongly prefers the LLM-anonymized text.

## Relevance to This Project
**Why surfaced:** anchor for both the RD4 *LLM-adversarial* substitution branch and the RD2 combination/
QI-reasoning problem — see [`docs/research/learned-substitution.md`](../../docs/research/learned-substitution.md)
and [`beyond-rantext.md`](../../docs/research/beyond-rantext.md). **Fit:** the clearest embodiment of the
reconstruction duality — the same LLM that re-identifies is the best anonymizer — and the reason RD2/RD4
must be evaluated against an LLM adversary, not surface metrics.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
