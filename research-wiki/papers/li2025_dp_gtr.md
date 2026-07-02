---
type: paper
node_id: paper:li2025_dp_gtr
title: "DP-GTR: Differentially Private Prompt Protection via Group Text Rewriting"
authors: ["Mingchen Li", "Heng Fan", "Song Fu", "Junhua Ding", "Yunhe Feng"]
year: 2025
venue: "EMNLP 2025 Findings"
external_ids:
  arxiv: "2503.04990"
  doi: null
  s2: null
tags: ["dp-text-rewriting", "prompt-privacy", "multi-granular", "in-context-learning", "rd4"]
added: 2026-07-02T00:00:00Z
---

# DP-GTR: differentially private prompt protection via group text rewriting

## One-line thesis
A three-stage plug-in that protects online-LLM *prompts* by combining local DP with in-context learning,
rewriting at both document and word granularity ("group text rewriting").

## Method
Multi-granular (document + word) rewriting under LDP, aggregated via in-context learning to improve the
privacy-utility trade-off; composes with existing rewriting mechanisms.

## Key Results
- Improves privacy and utility over single-granularity rewriting; functions as a plug-in.
- Does **not** explicitly recover the answer from the remote LLM — privatization is the deliverable.

## Relevance to This Project
**Why surfaced:** the *multi-granular / prompt-protection* branch of RD4 in
[`docs/research/learned-substitution.md`](../../docs/research/learned-substitution.md). **Fit:** targets
the online-LLM prompt setting (like InferDPT) and mixes document- and word-level DP — relevant to how a
role-aware policy (RD1) might operate across granularities; notably lacks the RD5 reverse step.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
