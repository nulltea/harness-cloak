---
type: paper
node_id: paper:tong2023_inferdpt_privacypreserving_inference
title: "InferDPT: Privacy-Preserving Inference for Closed-box Large Language Model"
authors: ["Meng Tong", "Kejiang Chen", "Jie Zhang", "Yuang Qi", "Weiming Zhang", "Nenghai Yu", "Tianwei Zhang", "Zhikun Zhang"]
year: 2023
venue: "arXiv"
external_ids:
  arxiv: "2310.12214"
  doi: null
  s2: null
tags: ["inferdpt", "rantext", "ldp", "black-box-inference", "anchor"]
added: 2026-06-29T12:14:19Z
---

# InferDPT: Privacy-Preserving Inference for Closed-box Large Language Model

## One-line thesis
Privacy-preserving black-box LLM inference: locally perturb the prompt with token-level LDP (RANTEXT), query the remote LLM, then locally reconstruct an aligned output.

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

> Large language models (LLMs), like ChatGPT, have greatly simplified text generation tasks. However, they have also raised concerns about privacy risks such as data leakage and unauthorized data collection. Existing solutions for privacy-preserving inference face practical challenges related to computation time and communication costs. In this paper, we propose InferDPT, the first practical framework for the privacy-preserving Inference of black-box LLMs, implementing Differential Privacy in Text generation. InferDPT comprises two key modules: the "perturbation module" utilizes the exponential mechanism to generate a perturbed prompt, facilitating privacy-preserving inference with black-box LLMs, and the "extraction module", inspired by knowledge distillation and retrieval-augmented generation, extracts coherent and consistent text from the perturbed generation result, ensuring successful text generation completion. To address privacy concerns related to previous exponential mechanisms' susceptibility to embedding revision attacks, we introduce RANTEXT, a novel differential privacy mechanism integrated into the perturbation module of InferDPT, which introduces the concept of "RANdom adjacency" for TEXT perturbation within the prompt. Experimental results across three datasets demonstrate that the text generation quality of InferDPT is comparable to that of non-private GPT-4, and RANTEXT surpasses existing state-of-the-art mechanisms, namely, SANTEXT+ and CUSTEXT+ in the trade-off between privacy and utility. Even with an privacy parameter epsilon value of 6.0, RANTEXT achieves an average privacy protection rate exceeding 90% against embedding revision attacks, which is 0.58 times higher than that of SANTEXT+ and 3.35 times higher than that of CUSTEXT+.

