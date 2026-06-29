---
type: paper
node_id: paper:yue2021_differential_privacy_text
title: "Differential Privacy for Text Analytics via Natural Text Sanitization"
authors: ["Xiang Yue", "Minxin Du", "Tianhao Wang", "Yaliang Li", "Huan Sun", "Sherman S. M. Chow"]
year: 2021
venue: "arXiv"
external_ids:
  arxiv: "2106.01221"
  doi: null
  s2: null
tags: ["santext", "metric-dp", "ldp", "baseline", "rantext-analog"]
added: 2026-06-29T12:14:21Z
---

# Differential Privacy for Text Analytics via Natural Text Sanitization

## One-line thesis
Token-wise metric-LDP text sanitization replacing words with Euclidean-near embeddings (SanText/SanText+); predecessor baseline.

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

> Texts convey sophisticated knowledge. However, texts also convey sensitive information. Despite the success of general-purpose language models and domain-specific mechanisms with differential privacy (DP), existing text sanitization mechanisms still provide low utility, as cursed by the high-dimensional text representation. The companion issue of utilizing sanitized texts for downstream analytics is also under-explored. This paper takes a direct approach to text sanitization. Our insight is to consider both sensitivity and similarity via our new local DP notion. The sanitized texts also contribute to our sanitization-aware pretraining and fine-tuning, enabling privacy-preserving natural language processing over the BERT language model with promising utility. Surprisingly, the high utility does not boost up the success rate of inference attacks.

