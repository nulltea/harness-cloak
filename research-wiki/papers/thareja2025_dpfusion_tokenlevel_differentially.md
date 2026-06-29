---
type: paper
node_id: paper:thareja2025_dpfusion_tokenlevel_differentially
title: "DP-Fusion: Token-Level Differentially Private Inference for Large Language Models"
authors: ["Rushil Thareja", "Preslav Nakov", "Praneeth Vepakomma", "Nils Lukas"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2507.04531"
  doi: null
  s2: null
tags: ["dp-fusion", "token-level-dp", "white-box", "inference", "framework-analog"]
added: 2026-06-29T12:14:25Z
---

# DP-Fusion: Token-Level Differentially Private Inference for Large Language Models

## One-line thesis
Token-level DP inference bounding output dependence on private tokens by mixing original vs redacted next-token distributions — needs logits (white-box), not API-only.

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

> Large language models (LLMs) do not preserve privacy at inference-time. The LLM's outputs can inadvertently reveal information about the model's context, which presents a privacy challenge when the LLM is augmented via tools or databases containing sensitive information. Existing privacy-preserving methods at inference-time have significant limitations since they (i) lack provable guarantees or (ii) have a poor utility/privacy trade-off. We propose DP-Fusion, a Differentially Private Inference (DPI) mechanism for LLMs that provably bounds the influence a set of tokens in the context can have on the LLM's output. DP-Fusion works as follows: (1) label a subset of sensitive tokens, (2) infer the LLM without any sensitive tokens to obtain a baseline, (3) infer the LLM with the sensitive tokens, and (4) blend distributions so that the final output remains within a bounded distance of the baseline distribution. While this per-token influence bound also mitigates jailbreak-style prompt injection, we focus on \emph{document privatization}, where the goal is to paraphrase a document containing sensitive tokens, e.g., personally identifiable information, so that no attacker can reliably infer them from the paraphrased document while preserving high text quality. The privacy/utility trade-off is controlled by $ε$, where $ε=0$ hides sensitive tokens entirely, while higher values trade off privacy for improved text quality. We show that our method creates token-level provably privatized documents with substantially improved theoretical and empirical privacy, achieving $6\times$ lower perplexity than related DPI methods.

