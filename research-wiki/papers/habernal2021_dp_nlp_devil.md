---
type: paper
node_id: paper:habernal2021_dp_nlp_devil
title: "When differential privacy meets NLP: The devil is in the detail"
authors: ["Ivan Habernal"]
year: 2021
venue: "EMNLP 2021"
external_ids:
  arxiv: "2109.03175"
  doi: null
  s2: null
tags: ["dp-theory", "critique", "sensitivity", "rd4", "cautionary"]
added: 2026-07-02T00:00:00Z
---

# When differential privacy meets NLP: The devil is in the detail

## One-line thesis
Formal analysis proving [ADePT](krishna2021_adept.md) is *not* differentially private — its sensitivity
was under-computed, voiding the guarantee and results.

## Key Results
- The true sensitivity is higher by **at least a factor of 6** (optimistic, tiny-encoder case).
- Under the flawed calibration, the fraction of utterances left effectively un-privatized could reach
  ~100% of the dataset.
- Thesis: DP-in-NLP claims must survive line-by-line scrutiny of the sensitivity/noise calibration.

## Relevance to This Project
**Why surfaced:** the discipline check on the entire RD4 line in
[`docs/research/learned-substitution.md`](../../docs/research/learned-substitution.md). **Fit:** it is
the concrete reason RD4's open problem ("a formal guarantee for a learned/contextual swap") is hard —
learned continuous-latent mechanisms are exactly where DP proofs break. Mirrors this project's own
empirical-honesty rule and the realized-ε caveats in [`rantext-limitations.md`](../../docs/research/rantext-limitations.md).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
