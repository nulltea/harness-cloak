---
type: paper
node_id: paper:gemmateam2024_gemma2_logit_softcap
title: "Gemma 2: Improving Open Language Models at a Practical Size"
authors: ["Gemma Team, Google DeepMind"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2408.00118"
  doi: null
  s2: null
tags: ["logit-capping", "softcap", "training-stability", "production"]
added: 2026-07-30T00:00:00Z
---

# Gemma 2: Improving Open Language Models at a Practical Size

## One-line thesis
Caps logits with logits <- cap * tanh(logits/cap) (attention 50.0, final logits 30.0) — an order-preserving, differentiable bound on logit scale used in production-scale training.

## Key Results
- Production precedent for direct logit soft-capping as a stability primitive.

## Relevance to This Project
The better instantiation of the bound-the-margins fix family: one auditable constant, stateless, order-preserving, and it bounds exactly the quantity our additive controller shift competes against — without the target-entropy constant's task sensitivity or its saturation side effects. Candidate Arm for the tie-ownership spike.
