---
type: paper
node_id: paper:ambadkar2026_d3po_diversity_regularizer
title: "D3PO: Decomposed, Diversity-Driven Policy Optimization for Preference-Conditioned MORL"
authors: ["Ambadkar", "Panda", "Kale", "Dodge", "Verma"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2602.07764"
  doi: null
  s2: null
tags: ["morl", "preference-conditioning", "diversity-regularizer", "mode-collapse"]
added: 2026-07-30T00:00:00Z
---

# D3PO: Decomposed, Diversity-Driven Policy Optimization for Preference-Conditioned MORL

## One-line thesis
Preference-conditioned policies suffer representational mode collapse across the conditioning space (one behavior served for many preference values); fixed by a Scaled Diversity Regularizer penalizing the gap between the ACTUAL KL between action distributions at nearby conditioning values and a TARGET KL proportional to conditioning distance.

## Key Results
- Names cross-condition mode collapse; regularizes input->output sensitivity directly.
- Conditioning folded in after trust-region stabilization; diversity term scaled by preference distance.

## Relevance to This Project
A third fix family that may dissolve our entropy-vs-learned-gain fork: it constrains d(pi)/d(lambda) directly — the property the controller exists to buy — with no learned gain and no entropy constant, and the regularizer value IS the controllability measurement. Candidate arm for the tie-ownership spike.
