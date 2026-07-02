---
type: paper
node_id: paper:krishna2021_adept
title: "ADePT: Auto-encoder based Differentially Private Text Transformation"
authors: ["Satyapriya Krishna", "Rahul Gupta", "Christophe Dupuy"]
year: 2021
venue: "EACL 2021"
external_ids:
  arxiv: "2102.01502"
  doi: null
  s2: null
tags: ["dp-text-rewriting", "autoencoder", "local-dp", "learned-substitution", "rd4", "cautionary"]
added: 2026-07-02T00:00:00Z
---

# ADePT: Auto-encoder based Differentially Private Text Transformation

## One-line thesis
Transform text privately by encoding it, adding Laplace noise to the latent vector, and decoding a
rewrite — the origin point of the learned DP text-rewriting line.

## Method
Auto-encoder over utterances; Laplace noise on the bottleneck latent for a claimed ε-DP transformation;
evaluated on downstream NLP tasks and against membership-inference.

## Key Results
- Reported strong utility + MIA resistance vs. baselines.
- **⚠️ The privacy claim is void:** [Habernal 2021](habernal2021_dp_nlp_devil.md) proved ADePT's
  sensitivity was under-computed (by ≥6×), so it is *not* DP.

## Relevance to This Project
**Why surfaced:** the root of the RD4 lineage (→ DP-BART → DP-MLM → DP-ST) in
[`docs/research/learned-substitution.md`](../../docs/research/learned-substitution.md) and the field's
cautionary tale. **Fit:** shows the exact hazard of a learned continuous-latent DP mechanism — the DP
proof silently breaks at the sensitivity computation. Any RD4 latent-noise design must avoid this.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Abstract (original)

> Privacy is an important concern when building statistical models on data containing personal
> information. Differential privacy offers a strong definition of privacy and can be used to solve
> several privacy concerns (Dwork et al., 2014). Multiple solutions have been proposed for the
> differentially-private transformation of datasets containing sensitive information. However, such
> transformation algorithms offer poor utility in Natural Language Processing (NLP) tasks due to noise
> added in the process. In this paper, we address this issue by providing a utility-preserving
> differentially private text transformation algorithm using auto-encoders. Our algorithm transforms
> text to offer robustness against attacks and produces transformations with high semantic quality that
> perform well on downstream NLP tasks. We prove the theoretical privacy guarantee of our algorithm and
> assess its privacy leakage under Membership Inference Attacks (MIA) (Shokri et al., 2017) on models
> trained with transformed data. Our results show that the proposed model performs better against MIA
> attacks while offering lower to no degradation in the utility of the underlying transformation process
> compared to existing baselines.
