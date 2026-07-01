---
type: paper
node_id: paper:pang2024_reconstruction_dp_text_llm
title: "Reconstruction of Differentially Private Text Sanitization via Large Language Models"
authors: ["Shuchao Pang", "Zhigang Lu", "Haichen Wang", "Peng Fu", "Yongbin Zhou", "Minhui Xue"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2410.12443"
  doi: null
  s2: null
tags: ["reconstruction-attack", "llm-attacker", "rantext-family", "word-level-dp", "sentence-level-dp"]
added: 2026-07-01T00:00:00Z
---

# Reconstruction of Differentially Private Text Sanitization via Large Language Models

## One-line thesis
LLMs can reconstruct the altered/removed private content from DP-sanitized prompts at high recovery
rates, showing the per-token DP guarantee does not bind document-level semantics.

## Problem / Gap
DP is the de facto standard against text leakage, but no prior work tested whether a capable LLM
can simply re-infer what DP removed.

## Method
Two attacks by LLM accessibility — black-box (sample text pairs as in-context instructions) and
white-box (fine-tuning on pairs) — evaluated across many modern LLMs and datasets, vs. both
word-level and sentence-level DP.

## Key Results
- Black-box word-level DP recovery on WikiMIA: **72.18% LLaMA-2 (70B), 82.39% LLaMA-3 (70B),
  75.35% Gemma-2, 91.2% ChatGPT-4o, 94.01% Claude-3.5 Sonnet**.
- Concludes that well-known LLMs are a new security risk for existing DP text sanitization.

## Assumptions
_TODO._

## Limitations / Failure Modes (of the RANTEXT-family it reports on)
- The protected asset is correlated semantic content; a strong LLM re-infers it from residual
  context → **A1** (token independence fails empirically), **F4** (surface guarantee ≠ semantic
  privacy). Directly instantiates the **C3/E1 reconstruction duality**: the same LLM used to
  restore utility is the reconstruction adversary.

## Reusable Ingredients
Black/white-box LLM reconstruction protocol as a semantic-privacy attacker for evaluation.

## Open Questions
_TODO._

## Claims
_TODO._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Primary external evidence for **A1 / F4** and the **C3/E1 duality** in
`docs/research/rantext-limitations.md`: the honest-but-curious remote LLM in InferDPT's own threat
model is exactly this reconstruction adversary.

## Abstract (original)

> Differential privacy (DP) is the de facto privacy standard against privacy leakage attacks,
> including many recently discovered ones against large language models (LLMs). However, we
> discovered that LLMs could reconstruct the altered/removed privacy from given DP-sanitized
> prompts. We propose two attacks (black-box and white-box) based on the accessibility to LLMs and
> show that LLMs could connect the pair of DP-sanitized text and the corresponding private training
> data of LLMs by giving sample text pairs as instructions (in the black-box attacks) or
> fine-tuning data (in the white-box attacks). To illustrate our findings, we conduct comprehensive
> experiments on modern LLMs (e.g., LLaMA-2, LLaMA-3, ChatGPT-3.5, ChatGPT-4, ChatGPT-4o, Claude-3,
> Claude-3.5, OPT, GPT-Neo, GPT-J, Gemma-2, and Pythia) using commonly used datasets (such as
> WikiMIA, Pile-CC, and Pile-Wiki) against both word-level and sentence-level DP. The experimental
> results show promising recovery rates, e.g., the black-box attacks against the word-level DP over
> WikiMIA dataset gave 72.18% on LLaMA-2 (70B), 82.39% on LLaMA-3 (70B), 75.35% on Gemma-2, 91.2% on
> ChatGPT-4o, and 94.01% on Claude-3.5 (Sonnet). More urgently, this study indicates that these
> well-known LLMs have emerged as a new security risk for existing DP text sanitization approaches
> in the current environment.
