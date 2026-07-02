---
type: paper
node_id: paper:igamberdiev2022_dp_rewrite
title: "DP-Rewrite: Towards Reproducibility and Transparency in Differentially Private Text Rewriting"
authors: ["Timour Igamberdiev", "Thomas Arnold", "Ivan Habernal"]
year: 2022
venue: "COLING 2022"
external_ids:
  arxiv: "2208.10400"
  doi: null
  s2: null
tags: ["dp-text-rewriting", "reproducibility", "tooling", "rd4"]
added: 2026-07-02T00:00:00Z
---

# DP-Rewrite: reproducibility & transparency in DP text rewriting

## One-line thesis
An open-source, modular framework for DP text rewriting whose case study on ADePT *detected a privacy
leak in its pre-training* — the transparency node that precedes DP-BART.

## Key Results
- Modular framework (datasets, models, pretraining, metrics) to validate DP-rewriting claims.
- Its ADePT case study found a privacy leak in ADePT's pre-training approach — empirically foreshadowing
  [Habernal 2021](habernal2021_dp_nlp_devil.md)'s formal proof.

## Relevance to This Project
**Why surfaced:** round-2 (arXiv) discovery; the reproducibility/tooling node of the RD4 lineage in
[`docs/research/learned-substitution.md`](../../docs/research/learned-substitution.md). **Fit:** same
authors as DP-BART; embodies this project's empirical-honesty rule — a framework built specifically to
catch the kind of silent DP failure (ADePT) that motivates RD4's "formal guarantee" open problem.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Abstract (original)

> Text rewriting with differential privacy (DP) provides concrete theoretical guarantees for protecting
> the privacy of individuals in textual documents. In practice, existing systems may lack the means to
> validate their privacy-preserving claims, leading to problems of transparency and reproducibility. We
> introduce DP-Rewrite, an open-source framework for differentially private text rewriting which aims to
> solve these problems by being modular, extensible, and highly customizable. Our system incorporates a
> variety of downstream datasets, models, pre-training procedures, and evaluation metrics to provide a
> flexible way to lead and validate private text rewriting research. To demonstrate our software in
> practice, we provide a set of experiments as a case study on the ADePT DP text rewriting system,
> detecting a privacy leak in its pre-training approach. Our system is publicly available, and we hope
> that it will help the community to make DP text rewriting research more accessible and transparent.
