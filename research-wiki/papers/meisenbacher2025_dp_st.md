---
type: paper
node_id: paper:meisenbacher2025_dp_st
title: "Leveraging Semantic Triples for Private Document Generation with Local Differential Privacy Guarantees"
authors: ["Stephen Meisenbacher", "Maulik Chevli", "Florian Matthes"]
year: 2025
venue: "EMNLP 2025"
external_ids:
  arxiv: "2508.20736"
  doi: null
  s2: null
tags: ["dp-text-rewriting", "semantic-triples", "llm-reconstruction", "local-dp", "rd4", "rd5"]
added: 2026-07-02T00:00:00Z
---

# DP-ST: semantic-triple private document generation under local DP

## One-line thesis
Decompose a document into semantic triples, privatize each within a *privatization neighborhood* under
local DP, then use LLM post-processing to reconstruct coherent text — buying coherence at lower ε via
divide-and-conquer.

## Method
Extract semantic triples → exponential mechanism over an auxiliary corpus per triple (neighborhood-scoped
LDP) → LLM post-processing synthesizes a fluent document from the privatized triples.

## Key Results
- Divide-and-conquer + a *neighborhood* DP notion + LLM reconstruction yields coherent output at lower ε
  than whole-document rewriting, while balancing privacy and utility.
- Reinforces that coherence is the binding constraint for reasonable-ε privatization.

## Relevance to This Project
**Why surfaced:** a **2025 SOTA** RD4 method that *also* implements the RD5 reverse step — see
[`docs/research/learned-substitution.md`](../../docs/research/learned-substitution.md). **Fit:** the
"privatize a structured intermediate, then LLM-reconstruct" pattern is the same shape as InferDPT's
perturb→extract; it keeps a (relaxed, neighborhood) guarantee while getting coherence — one of the two
current frontiers for RD4.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Abstract (original)

> Many works at the intersection of Differential Privacy (DP) in Natural Language Processing aim to
> protect privacy by transforming texts under DP guarantees. This can be performed in a variety of ways,
> from word perturbations to full document rewriting, and most often under local DP. Here, an input text
> must be made indistinguishable from any other potential text, within some bound governed by the privacy
> parameter ε. Such a guarantee is quite demanding, and recent works show that privatizing texts under
> local DP can only be done reasonably under very high ε values. Addressing this challenge, we introduce
> DP-ST, which leverages semantic triples for neighborhood-aware private document generation under local
> DP guarantees. Through the evaluation of our method, we demonstrate the effectiveness of the
> divide-and-conquer paradigm, particularly when limiting the DP notion (and privacy guarantees) to that
> of a privatization neighborhood. When combined with LLM post-processing, our method allows for coherent
> text generation even at lower ε values, while still balancing privacy and utility. These findings
> highlight the importance of coherence in achieving balanced privatization outputs at reasonable ε levels.
