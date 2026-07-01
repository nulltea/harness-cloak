---
type: paper
node_id: paper:mattern2022_limits_word_level_dp
title: "The Limits of Word Level Differential Privacy"
authors: ["Justus Mattern", "Benjamin Weggenmann", "Florian Kerschbaum"]
year: 2022
venue: "NAACL 2022 Findings"
external_ids:
  arxiv: "2205.02130"
  doi: "10.18653/v1/2022.findings-naacl.65"
  s2: null
tags: ["word-level-dp", "rantext-family", "critique", "metric-dp", "deanonymization"]
added: 2026-07-01T00:00:00Z
---

# The Limits of Word Level Differential Privacy

## One-line thesis
The foundational critique of word-embedding-perturbation DP: the theoretical privacy guarantee is
mathematically weaker than claimed, and in practice these mechanisms fail on deanonymization,
content preservation, and language quality.

## Problem / Gap
_TODO._

## Method
_TODO._

## Key Results
- Identifies **mathematical constraints that diminish the theoretical privacy guarantee** of
  word-level DP embedding perturbation.
- Demonstrates **practical failure** on three fronts: protection against deanonymization,
  preservation of original content, and quality of language output.
- Proposes a paraphrasing-LM alternative with a formal guarantee that avoids most weaknesses.

## Assumptions
_TODO._

## Limitations / Failure Modes (of the RANTEXT-family it critiques)
- Word-by-word perturbation cannot preserve coherence → **F1a** (context-blindness).
- Indiscriminate per-word treatment damages content/meaning → **F1b** (salience-blindness).
- The formal per-word guarantee overstates real protection; deanonymization still succeeds
  → **F4** (guarantee/evaluation scope mismatch) and **A1** (token-independence).

## Reusable Ingredients
_TODO._

## Open Questions
_TODO._

## Claims
_TODO._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Anchors the taxonomy in `docs/research/rantext-limitations.md`: this is the earliest formal
statement of **F1a / F1b / F4 / A1**. Cited there as the foundational critique of the whole
word-level MLDP family that RANTEXT belongs to.

## Abstract (original)

> As the issues of privacy and trust are receiving increasing attention within the research
> community, various attempts have been made to anonymize textual data. A significant subset of
> these approaches incorporate differentially private mechanisms to perturb word embeddings, thus
> replacing individual words in a sentence. While these methods represent very important
> contributions, have various advantages over other techniques and do show anonymization
> capabilities, they have several shortcomings. In this paper, we investigate these weaknesses and
> demonstrate significant mathematical constraints diminishing the theoretical privacy guarantee as
> well as major practical shortcomings with regard to the protection against deanonymization
> attacks, the preservation of content of the original sentences as well as the quality of the
> language output. Finally, we propose a new method for text anonymization based on transformer
> based language models fine-tuned for paraphrasing that circumvents most of the identified
> weaknesses and also offers a formal privacy guarantee. We evaluate the performance of our method
> via thorough experimentation and demonstrate superior performance over the discussed mechanisms.
