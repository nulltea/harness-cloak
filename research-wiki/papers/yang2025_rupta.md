---
type: paper
node_id: paper:yang2025_rupta
title: "Robust Utility-Preserving Text Anonymization Based on Large Language Models"
authors: ["Tianyu Yang", "Xiaodan Zhu", "Iryna Gurevych"]
year: 2025
venue: "ACL 2025"
external_ids:
  arxiv: "2407.11770"
  doi: null
  s2: null
tags: ["llm-anonymization", "adversarial", "distillation", "re-identification", "rd4", "rd2"]
added: 2026-07-02T00:00:00Z
---

# RUPTA: robust utility-preserving text anonymization with LLMs

## One-line thesis
Anonymize against LLM re-identification with three collaborating LLM components — a privacy evaluator, a
utility evaluator, and an optimizer — then distill the capability into a lightweight local model.

## Method
Iterative LLM-driven anonymization guided by privacy + utility evaluators; distillation of the pipeline
into small models for scale/real-time.

## Key Results
- Outperforms baselines on reducing re-identification risk while preserving downstream utility.
- Demonstrates distillation into lightweight models — the "make the private stage local" move.

## Relevance to This Project
**Why surfaced:** a **2025 SOTA** on the *non-DP LLM-adversarial* branch of RD4 in
[`docs/research/learned-substitution.md`](../../docs/research/learned-substitution.md). **Fit:** the
strongest empirical privacy-utility trade-off (no formal DP), directly confronting the LLM QI-reasoning
re-identifier (RD2/[Staab](staab2024_llm_anonymizers.md)); distillation-to-local is the practical route
to an on-device RD4 substitutor.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Abstract (original)

> Anonymizing text that contains sensitive information is crucial for a wide range of applications.
> Existing techniques face the emerging challenges of the re-identification ability of large language
> models (LLMs), which have shown advanced capability in memorizing detailed information and reasoning
> over dispersed pieces of patterns to draw conclusions. When defending against LLM-based
> re-identification, anonymization could jeopardize the utility of the resulting anonymized data in
> downstream tasks. In general, the interaction between anonymization and data utility requires a deeper
> understanding within the context of LLMs. In this paper, we propose a framework composed of three key
> LLM-based components: a privacy evaluator, a utility evaluator, and an optimization component, which
> work collaboratively to perform anonymization. Extensive experiments demonstrate that the proposed
> model outperforms existing baselines, showing robustness in reducing the risk of re-identification
> while preserving greater data utility in downstream tasks.
