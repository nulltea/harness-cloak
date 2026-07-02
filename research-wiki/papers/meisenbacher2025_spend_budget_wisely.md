---
type: paper
node_id: paper:meisenbacher2025_spend_budget_wisely
title: "Spend Your Budget Wisely: Towards an Intelligent Distribution of the Privacy Budget in Differentially Private Text Rewriting"
authors: ["Stephen Meisenbacher", "Chaeeun Joy Lee", "Florian Matthes"]
year: 2025
venue: "CODASPY 2025"
external_ids:
  arxiv: "2503.22379"
  doi: null
  s2: null
tags: ["dp-text-rewriting", "budget-allocation", "salience", "rd1", "rd4"]
added: 2026-07-02T00:00:00Z
---

# Spend Your Budget Wisely — intelligent ε distribution in DP text rewriting

## One-line thesis
Not all tokens are equally sensitive, so distribute the privacy budget ε per-token by
linguistic/NLP salience rather than uniformly — the first work to do so for DP text rewriting.

## Method
A toolkit of linguistics- and NLP-based scorers assigns each token a share of the document's ε; compared
against naive uniform allocation on privacy + utility.

## Key Results
- At the same total budget, **intelligent per-token allocation gives higher privacy and better
  privacy-utility trade-offs** than uniform ε.

## Relevance to This Project
**Why surfaced:** this is **RD1 (role/salience budgeting) realized inside RD4** — see
[`docs/research/beyond-rantext.md`](../../docs/research/beyond-rantext.md) and
[`learned-substitution.md`](../../docs/research/learned-substitution.md). **Fit:** direct empirical
support for the taxonomy's core prescription — decompose the scalar ε by token salience (fixes F1b) —
demonstrated within a learned-rewriting system.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Abstract (original)

> The task of Differentially Private Text Rewriting is a class of text privatization techniques in which
> (sensitive) input textual documents are rewritten under Differential Privacy (DP) guarantees. The
> motivation behind such methods is to hide both explicit and implicit identifiers that could be
> contained in text, while still retaining the semantic meaning of the original text, thus preserving
> utility. Recent years have seen an uptick in research output in this field, offering a diverse array of
> word-, sentence-, and document-level DP rewriting methods. Common to these methods is the selection of a
> privacy budget (i.e., the ε parameter), which governs the degree to which a text is privatized. One
> major limitation of previous works, stemming directly from the unique structure of language itself, is
> the lack of consideration of where the privacy budget should be allocated, as not all aspects of
> language, and therefore text, are equally sensitive or personal. In this work, we are the first to
> address this shortcoming, asking the question of how a given privacy budget can be intelligently and
> sensibly distributed amongst a target document. We construct and evaluate a toolkit of linguistics- and
> NLP-based methods used to allocate a privacy budget to constituent tokens in a text document. In a
> series of privacy and utility experiments, we empirically demonstrate that given the same privacy
> budget, intelligent distribution leads to higher privacy levels and more positive trade-offs than a
> naive distribution of ε.
