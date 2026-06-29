---
type: paper
node_id: paper:feyisetan2019_privacy_utilitypreserving_textual
title: "Privacy- and Utility-Preserving Textual Analysis via Calibrated Multivariate Perturbations"
authors: ["Oluwaseyi Feyisetan", "Borja Balle", "Thomas Drake", "Tom Diethe"]
year: 2019
venue: "arXiv"
external_ids:
  arxiv: "1910.08902"
  doi: null
  s2: null
tags: ["metric-dp", "calibrated-perturbation", "word-embedding", "foundational"]
added: 2026-06-29T12:14:19Z
---

# Privacy- and Utility-Preserving Textual Analysis via Calibrated Multivariate Perturbations

## One-line thesis
Calibrated multivariate (Laplace) perturbations in word-embedding space for d_X-privacy text — the foundational metric-DP text mechanism.

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

> Accurately learning from user data while providing quantifiable privacy guarantees provides an opportunity to build better ML models while maintaining user trust. This paper presents a formal approach to carrying out privacy preserving text perturbation using the notion of dx-privacy designed to achieve geo-indistinguishability in location data. Our approach applies carefully calibrated noise to vector representation of words in a high dimension space as defined by word embedding models. We present a privacy proof that satisfies dx-privacy where the privacy parameter epsilon provides guarantees with respect to a distance metric defined by the word embedding space. We demonstrate how epsilon can be selected by analyzing plausible deniability statistics backed up by large scale analysis on GloVe and fastText embeddings. We conduct privacy audit experiments against 2 baseline models and utility experiments on 3 datasets to demonstrate the tradeoff between privacy and utility for varying values of epsilon on different task types. Our results demonstrate practical utility (< 2% utility loss for training binary classifiers) while providing better privacy guarantees than baseline models.

