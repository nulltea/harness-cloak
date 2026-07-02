---
type: paper
node_id: paper:meisenbacher2024_just_rewrite_again
title: "Just Rewrite It Again: A Post-Processing Method for Enhanced Semantic Similarity and Privacy Preservation of Differentially Private Rewritten Text"
authors: ["Stephen Meisenbacher", "Florian Matthes"]
year: 2024
venue: "ARES 2024 (IWAPS)"
external_ids:
  arxiv: "2405.19831"
  doi: "10.1145/3664476.3669926"
  s2: null
tags: ["dp-text-rewriting", "post-processing", "reconstruction", "rd4", "rd5"]
added: 2026-07-02T00:00:00Z
---

# Just Rewrite It Again — post-processing for DP rewritten text

## One-line thesis
A simple post-processing step — *rewrite the DP-rewritten text again* to realign it with the original —
improves both semantic similarity and empirical privacy at once.

## Key Results
- Re-rewriting DP output yields text closer to the original in meaning **and** scoring better in
  empirical (adversarial inference) privacy tests.

## Relevance to This Project
**Why surfaced:** the RD4→RD5 bridge / reconstruction bonus in
[`docs/research/learned-substitution.md`](../../docs/research/learned-substitution.md). **Fit:** the
productive form of the reconstruction duality — a *local post-processing* pass (free under DP
post-processing immunity) that improves utility and privacy simultaneously; the same principle as the
extraction module and [double-edged reconstruction](meisenbacher2025_double_edged_reconstruction.md).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Abstract (original)

> The study of Differential Privacy (DP) in Natural Language Processing often views the task of text
> privatization as a rewriting task, in which sensitive input texts are rewritten to hide explicit or
> implicit private information. In order to evaluate the privacy-preserving capabilities of a DP text
> rewriting mechanism, empirical privacy tests are frequently employed. In these tests, an adversary is
> modeled, who aims to infer sensitive information (e.g., gender) about the author behind a (privatized)
> text. Looking to improve the empirical protections provided by DP rewriting methods, we propose a
> simple post-processing method based on the goal of aligning rewritten texts with their original
> counterparts, where DP rewritten texts are rewritten again. Our results show that such an approach not
> only produces outputs that are more semantically reminiscent of the original inputs, but also texts
> which score on average better in empirical privacy evaluations.
