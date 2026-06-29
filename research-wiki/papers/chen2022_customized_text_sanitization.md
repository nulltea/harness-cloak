---
type: paper
node_id: paper:chen2022_customized_text_sanitization
title: "A Customized Text Sanitization Mechanism with Differential Privacy"
authors: ["Huimin Chen", "Fengran Mo", "Yanhao Wang", "Cen Chen", "Jian-Yun Nie", "Chengyu Wang", "Jamie Cui"]
year: 2022
venue: "https://aclanthology.org/2023.findings-acl.355/"
external_ids:
  arxiv: "2207.01193"
  doi: null
  s2: null
tags: ["custext", "exponential-mechanism", "counter-fitted", "ldp", "rantext-analog"]
added: 2026-06-29T12:14:22Z
---

# A Customized Text Sanitization Mechanism with Differential Privacy

## One-line thesis
Token-level sanitization via the exponential mechanism over a small customized candidate set, compatible with cosine/counter-fitted embeddings (CusText/CusText+).

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

> As privacy issues are receiving increasing attention within the Natural Language Processing (NLP) community, numerous methods have been proposed to sanitize texts subject to differential privacy. However, the state-of-the-art text sanitization mechanisms based on metric local differential privacy (MLDP) do not apply to non-metric semantic similarity measures and cannot achieve good trade-offs between privacy and utility. To address the above limitations, we propose a novel Customized Text (CusText) sanitization mechanism based on the original $ε$-differential privacy (DP) definition, which is compatible with any similarity measure. Furthermore, CusText assigns each input token a customized output set of tokens to provide more advanced privacy protection at the token level. Extensive experiments on several benchmark datasets show that CusText achieves a better trade-off between privacy and utility than existing mechanisms. The code is available at https://github.com/sai4july/CusText.

