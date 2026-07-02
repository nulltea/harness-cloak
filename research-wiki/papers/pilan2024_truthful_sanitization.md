---
type: paper
node_id: paper:pilan2024_truthful_sanitization
title: "Truthful Text Sanitization Guided by Inference Attacks"
authors: ["Ildikó Pilán", "Benet Manzanares-Salor", "David Sánchez", "Pierre Lison"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2412.12928"
  doi: null
  s2: null
tags: ["text-anonymization", "attack-guided", "generalization", "llm", "rd2", "rd4"]
added: 2026-07-02T00:00:00Z
---

# Truthful Text Sanitization Guided by Inference Attacks

## One-line thesis
Replace PII with *truthful generalizations* (broader terms that subsume the original), choosing among
LLM-generated candidates by how well they resist an inference-attack evaluator.

## Method
Two stages with instruction-tuned LLMs (Mistral 7B Instruct): (1) generate + rank generalization
candidates for each PII span; (2) score each by an inference attack, selecting the privacy/utility
optimum. Evaluated on the Text Anonymization Benchmark.

## Key Results
- Higher utility with minimal re-identification increase vs. full suppression; better truthfulness than
  Presidio.

## Relevance to This Project
**Why surfaced:** round-2 (arXiv) discovery; a strong **RD2∩RD4** node in
[`docs/research/learned-substitution.md`](../../docs/research/learned-substitution.md). **Fit:** realizes
[Lison 2021](lison2021_anonymisation_models_text.md)'s "model disclosure risk, not spans" by *guiding
substitution with an inference attacker* — the attack-guided branch of learned substitution; the
"generalization" surrogate is a middle ground between RD3 typed placeholders and RD4 free rewriting.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Abstract (original summary)

> Text sanitization via *generalizations* — broader but still informative terms that subsume the semantic
> content of the original spans. Instruction-tuned LLMs are used in two stages: generate and rank
> replacement candidates for PII, then evaluate their privacy protection via inference attacks, selecting
> replacements that balance privacy and utility. With Mistral 7B Instruct on the Text Anonymization
> Benchmark, the approach achieves enhanced utility with minimal increase in re-identification risk vs.
> full suppression, and superior truthfulness preservation relative to tools like Microsoft Presidio.
