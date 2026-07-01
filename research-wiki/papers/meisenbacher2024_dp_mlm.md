---
type: paper
node_id: paper:meisenbacher2024_dp_mlm
title: "DP-MLM: Differentially Private Text Rewriting Using Masked Language Models"
authors: ["Stephen Meisenbacher", "Maulik Chevli", "Juraj Vladika", "Florian Matthes"]
year: 2024
venue: "ACL 2024 Findings"
external_ids:
  arxiv: "2407.00637"
  doi: null
  s2: null
tags: ["dp-text-rewriting", "masked-language-model", "learned-substitution", "local-dp", "rd4"]
added: 2026-07-01T00:00:00Z
---

# DP-MLM: Differentially Private Text Rewriting Using Masked Language Models

## One-line thesis
Rewrite text under DP one token at a time by temperature-sampling substitutions from an encoder-only
MLM's contextualized logits; MLMs give better utility at lower ε than decoder-based rewriters.

## Method
Uses a masked language model to produce a contextualized replacement per token via the exponential
mechanism over MLM logits, preserving semantic similarity while giving formal DP.

## Key Results
- Encoder-only MLMs achieve **superior utility preservation at lower ε** than decoder-based DP rewriting.
- More customization flexibility than paraphrase-style decoder methods; implementation released.

## Limitations / Failure Modes
Per-token contextual rewriting still operates within the metric/exponential-mechanism DP frame; the
same MLM is a known reconstruction/MI attacker (cut-both-ways).

## Relevance to This Project
**Why surfaced:** the most on-point anchor for **RD4** in
[`docs/research/beyond-rantext.md`](../../docs/research/beyond-rantext.md): substitution *is* the MLM
pretraining objective, and this shows an encoder-only MLM finetune beats decoder paraphrasing at fixed
ε — evidence for RD4's "finetune existing MLM/infilling models, don't train from scratch" thesis and
for RD5's "denoising is the right architecture" (E2). **Relevance:** by the same group as the
double-edged-reconstruction work, so it also grounds the RD4 "cut-both-ways" warning.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Abstract (original)

> The task of text privatization using Differential Privacy has recently taken the form of text
> rewriting, in which an input text is obfuscated via the use of generative (large) language models.
> While these methods have shown promising results in the ability to preserve privacy, these methods
> rely on autoregressive models which lack a mechanism to contextualize the private rewriting process.
> In response to this, we propose DP-MLM, a new method for differentially private text rewriting based
> on leveraging masked language models (MLMs) to rewrite text in a semantically similar and obfuscated
> manner. We accomplish this with a simple contextualization technique whereby we rewrite a text one
> token at a time. We find that utilizing encoder-only MLMs provides better utility preservation at
> lower epsilon levels, as compared to previous methods relying on larger models with a decoder. In
> addition, MLMs allow for greater customization of the rewriting mechanism.
