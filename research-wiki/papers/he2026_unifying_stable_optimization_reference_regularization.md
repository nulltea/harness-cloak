---
type: paper
node_id: paper:he2026_unifying_stable_optimization_reference_regularization
title: "Unifying Stable Optimization and Reference Regularization in RLHF"
authors: ["Li He", "Qiang Qu", "He Zhao", "Stephen Wan"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2602.11523"
  doi: null
  s2: null
tags: ["rlhf", "kl-regularization", "reference-anchor", "trust-region"]
added: 2026-07-30T00:00:00Z
---

# Unifying Stable Optimization and Reference Regularization in RLHF

## One-line thesis
RLHF simultaneously regularizes toward a fixed reference (anti-drift) and toward the current policy (stability); the implicit trade-off between the two is underexplored, and the paper replaces the pair with one unified regularizer.

## Key Results
- Names and formalizes the fixed-reference vs current-policy regularization tension.
- Unified regularizer subsuming both roles.

## Relevance to This Project
Surfaced after the v11 KL-anchor spike failed by over-anchoring: our finding (fixed-reference KL bounds drift but pins the profile that must leave the reference) is this paper premise, independently reached. Confirms the dilemma is structural, not an eta-tuning issue, and points at anchor-selectivity or unified formulations rather than coefficient search.
