---
type: paper
node_id: paper:utpala2023_locally_differentially_private
title: "Locally Differentially Private Document Generation Using Zero Shot Prompting"
authors: ["Saiteja Utpala", "Sara Hooker", "Pin Yu Chen"]
year: 2023
venue: "arXiv"
external_ids:
  arxiv: "2310.16111"
  doi: null
  s2: null
tags: ["dp-prompt", "paraphrase", "ldp", "framework-analog"]
added: 2026-06-29T12:14:23Z
---

# Locally Differentially Private Document Generation Using Zero Shot Prompting

## One-line thesis
Locally differentially private document generation by zero-shot LLM paraphrasing under the exponential mechanism — a DP-paraphrase perturbation alternative to token swapping.

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

> Numerous studies have highlighted the privacy risks associated with pretrained large language models. In contrast, our research offers a unique perspective by demonstrating that pretrained large language models can effectively contribute to privacy preservation. We propose a locally differentially private mechanism called DP-Prompt, which leverages the power of pretrained large language models and zero-shot prompting to counter author de-anonymization attacks while minimizing the impact on downstream utility. When DP-Prompt is used with a powerful language model like ChatGPT (gpt-3.5), we observe a notable reduction in the success rate of de-anonymization attacks, showing that it surpasses existing approaches by a considerable margin despite its simpler design. For instance, in the case of the IMDB dataset, DP-Prompt (with ChatGPT) perfectly recovers the clean sentiment F1 score while achieving a 46\% reduction in author identification F1 score against static attackers and a 26\% reduction against adaptive attackers. We conduct extensive experiments across six open-source large language models, ranging up to 7 billion parameters, to analyze various effects of the privacy-utility tradeoff.

