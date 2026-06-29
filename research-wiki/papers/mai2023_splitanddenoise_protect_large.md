---
type: paper
node_id: paper:mai2023_splitanddenoise_protect_large
title: "Split-and-Denoise: Protect large language model inference with local differential privacy"
authors: ["Peihua Mai", "Ran Yan", "Zhe Huang", "Youjia Yang", "Yan Pang"]
year: 2023
venue: "arXiv"
external_ids:
  arxiv: "2310.09130"
  doi: null
  s2: null
tags: ["split-and-denoise", "embedding-ldp", "denoise", "white-box", "framework-analog"]
added: 2026-06-29T12:14:24Z
---

# Split-and-Denoise: Protect large language model inference with local differential privacy

## One-line thesis
Client adds LDP noise to token embeddings before the cloud LLM and locally denoises — embedding-level analog that needs model internals (not black-box text).

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

> Large Language Models (LLMs) excel in natural language understanding by capturing hidden semantics in vector space. This process enriches the value of text embeddings for various downstream tasks, thereby fostering the Embedding-as-a-Service (EaaS) business model. However, the risk of privacy leakage due to direct text transmission to servers remains a critical concern. To address this, we introduce Split-N-Denoise (SnD), an private inference framework that splits the model to execute the token embedding layer on the client side at minimal computational cost. This allows the client to introduce noise prior to transmitting the embeddings to the server, and subsequently receive and denoise the perturbed output embeddings for downstream tasks. Our approach is designed for the inference stage of LLMs and requires no modifications to the model parameters. Extensive experiments demonstrate SnD's effectiveness in optimizing the privacy-utility tradeoff across various LLM architectures and diverse downstream tasks. The results reveal an improvement in performance under the same privacy budget compared to the baselines by over 10\% on average, offering clients a privacy-preserving solution for local privacy protection.

