---
type: paper
node_id: paper:damour2020_underspecification_ml
title: "Underspecification Presents Challenges for Credibility in Modern Machine Learning"
authors: ["Alexander DAmour", "et al."]
year: 2020
venue: "JMLR 23:226 (arXiv 2020)"
external_ids:
  arxiv: "2011.03395"
  doi: null
  s2: null
tags: ["underspecification", "seed-variance", "identifiability"]
added: 2026-07-30T00:00:00Z
---

# Underspecification Presents Challenges for Credibility in Modern Machine Learning

## One-line thesis
When the training objective does not pin down the predictor, many solutions score identically and seed-level choices decide behavior on exactly the axes the loss is indifferent to.

## Key Results
- Underspecification demonstrated across ML pipelines: equal training performance, divergent deployment behavior, selected arbitrarily by seed.

## Relevance to This Project
Surfaced for root-cause link 2: our seed lottery on reward-tied action pairs is the textbook RL instance - the utility loss is indifferent between keep and L0, so shared-parameter generalization (not the objective) decides, differently per training trajectory. Justifies treating tie-set behavior as requiring an explicit owner.
