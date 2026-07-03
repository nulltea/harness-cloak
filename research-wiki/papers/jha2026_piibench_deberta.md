---
type: paper
node_id: paper:jha2026_piibench_deberta
title: "Fine-Tuning Over Architectural Complexity: Broad-Coverage PII Detection on PIIBench with DeBERTa"
authors: ["Pritesh Jha"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2605.25816"
  doi: null
  s2: null
tags: ["pii-detection", "token-classification", "deberta", "fine-tuning", "architecture"]
added: 2026-07-03T00:00:00Z
---

# Fine-Tuning Over Architectural Complexity: Broad-Coverage PII Detection on PIIBench with DeBERTa

## One-line thesis
On a corrected multi-source PII benchmark (82 entity types, 10 source datasets), directly fine-tuned
DeBERTa token classification with weighted cross-entropy beats hierarchical and curriculum
architectural variants — data and a simple objective matter more than architecture.

## Problem / Gap
Broad-coverage PII detection (many entity types, many domains) tempts architectural complexity
(source conditioning, hierarchy, curricula). No controlled comparison of these against plain
fine-tuning existed on a broad multi-source benchmark.

## Method
Three DeBERTa-based systems trained on the same PIIBench preparation: (1) direct token-classification
fine-tuning with weighted cross-entropy; (2) source-conditioned hierarchical model (SC+H);
(3) SC+H plus a three-phase curriculum. Evaluated span-level F1 on a 5,000-record held-out subset
and a 100,002-record full split.

## Key Results
- Direct fine-tune: 0.6476 span F1 (test_5k), 0.6455 (full) — wins on 54 of 82 entity types.
- SC+H: 0.5899; SC+H+Curriculum: 0.2772; strongest published comparator: 0.1723.
- Claim: "diverse task-specific training data and a simple weighted cross-entropy objective
  contribute more to broad-coverage PII detection than the tested architectural and curriculum
  complexity."

## Relevance to This Project
Directly supports the size-optimal design choice for our improved span detector: a plain compact
encoder + BIO token-classification head, fine-tuned on in-domain + multi-domain PII data, rather
than a novel architecture. Weighted cross-entropy for rare types (our MISC/DEM/QUANTITY gap) is the
one training-objective ingredient worth copying.

## Limitations / Failure Modes
- Single-author arXiv preprint, not peer-reviewed; PIIBench label taxonomy (82 types) is much finer
  than TAB's 8, so absolute F1 values don't transfer.
- No zero-shot/open-type comparison (GLiNER-style) at matched data.

## Reusable Ingredients
Weighted-CE recipe for class imbalance; evidence hierarchy for the "fine-tune > architecture" claim.

## Open Questions
Does the conclusion hold at TAB's coarser 8-type schema where QUASI boundary ambiguity, not type
count, is the hard part?
