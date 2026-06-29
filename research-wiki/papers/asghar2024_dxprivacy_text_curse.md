---
type: paper
node_id: paper:asghar2024_dxprivacy_text_curse
title: "$d_X$-Privacy for Text and the Curse of Dimensionality"
authors: ["Hassan Jameel Asghar", "Robin Carpentier", "Benjamin Zi Hao Zhao", "Dali Kaafar"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2411.13784"
  doi: null
  s2: null
tags: ["metric-dp", "curse-of-dimensionality", "d_x-privacy"]
added: 2026-06-29T12:14:21Z
---

# $d_X$-Privacy for Text and the Curse of Dimensionality

## One-line thesis
Shows d_X-privacy (metric-LDP) text mechanisms degrade in high-dimensional embedding spaces — the curse of dimensionality.

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

> A widely used method to ensure privacy of unstructured text data is the multidimensional Laplace mechanism for $d_X$-privacy, which is a relaxation of differential privacy for metric spaces. We identify an intriguing peculiarity of this mechanism. When applied on a word-by-word basis, the mechanism either outputs the original word, or completely dissimilar words, and very rarely outputs semantically similar words. We investigate this observation in detail, and tie it to the fact that the distance of the nearest neighbor of a word in any word embedding model (which are high-dimensional) is much larger than the relative difference in distances to any of its two consecutive neighbors. We also show that the dot product of the multidimensional Laplace noise vector with any word embedding plays a crucial role in designating the nearest neighbor. We derive the distribution, moments and tail bounds of this dot product. We further propose a fix as a post-processing step, which satisfactorily removes the above-mentioned issue.

