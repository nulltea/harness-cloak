---
type: paper
node_id: paper:chen2023_hide_seek_has
title: "Hide and Seek (HaS): A Lightweight Framework for Prompt Privacy Protection"
authors: ["Yu Chen", "Tingxin Li", "Huiming Liu", "Yang Yu"]
year: 2023
venue: "arXiv"
external_ids:
  arxiv: "2309.03057"
  doi: null
  s2: null
tags: ["has", "entity-anonymization", "black-box-inference", "framework-analog"]
added: 2026-06-29T12:14:23Z
---

# Hide and Seek (HaS): A Lightweight Framework for Prompt Privacy Protection

## One-line thesis
Hide-and-Seek: a local model anonymizes private entities before the black-box LLM and de-anonymizes the response — same protect->remote->restore architecture as InferDPT, entity-level not DP.

## Problem / Gap
_TODO._

## Method
_TODO._

## Key Results
_TODO._

## Assumptions
_TODO._

## Limitations / Failure Modes
_TODO._

## Reusable Ingredients
_TODO._

## Open Questions
_TODO._

## Claims
_TODO._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
_TODO._

## Abstract (original)

> Numerous companies have started offering services based on large language models (LLM), such as ChatGPT, which inevitably raises privacy concerns as users' prompts are exposed to the model provider. Previous research on secure reasoning using multi-party computation (MPC) has proven to be impractical for LLM applications due to its time-consuming and communication-intensive nature. While lightweight anonymization techniques can protect private information in prompts through substitution or masking, they fail to recover sensitive data replaced in the LLM-generated results. In this paper, we expand the application scenarios of anonymization techniques by training a small local model to de-anonymize the LLM's returned results with minimal computational overhead. We introduce the HaS framework, where "H(ide)" and "S(eek)" represent its two core processes: hiding private entities for anonymization and seeking private entities for de-anonymization, respectively. To quantitatively assess HaS's privacy protection performance, we propose both black-box and white-box adversarial models. Furthermore, we conduct experiments to evaluate HaS's usability in translation and classification tasks. The experimental findings demonstrate that the HaS framework achieves an optimal balance between privacy protection and utility.

