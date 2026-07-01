---
type: paper
node_id: paper:igamberdiev2023_dp_bart
title: "DP-BART for Privatized Text Rewriting under Local Differential Privacy"
authors: ["Timour Igamberdiev", "Ivan Habernal"]
year: 2023
venue: "ACL 2023 Findings"
external_ids:
  arxiv: "2302.07636"
  doi: null
  s2: null
tags: ["dp-text-rewriting", "seq2seq", "local-dp", "learned-substitution", "rd4"]
added: 2026-07-01T00:00:00Z
---

# DP-BART for Privatized Text Rewriting under Local Differential Privacy

## One-line thesis
Privatize text by injecting DP noise into a BART encoder's latent representation and decoding a
rewrite; novel clipping + iterative pruning drastically cut the noise needed for a given ε.

## Method
Adds calibrated noise to internal BART representations between encoder and decoder; a clipping method,
iterative pruning, and further training of the latent space reduce the noise budget required for LDP.

## Key Results
- Outperforms prior LDP text-rewriting systems on five datasets across privacy levels (downstream
  text-classification utility).
- Explicitly diagnoses the **strict text-adjacency constraint of the LDP paradigm** as the source of
  the high noise requirement — a mechanism-level limitation, not a tuning issue.

## Limitations / Failure Modes
Latent-noise LDP still inherits the adjacency-constraint noise blow-up; the paper discusses formal
flaws and unrealistic guarantees in the prior word-level systems it improves on.

## Relevance to This Project
**Why surfaced:** anchor for **RD4 (learned context-aware substitution)** in
[`docs/research/beyond-rantext.md`](../../docs/research/beyond-rantext.md) — a concrete seq2seq LM
rewriter under LDP, i.e. substitution as the model's own objective rather than embedding-distance
sampling. **Relevance:** demonstrates both the promise (whole-sequence contextual rewrite) and the
limit (adjacency constraint → noise) of learned DP rewriting, directly informing the RD3-vs-RD4
tradeoff and the "formal guarantee for a contextual swap" open question.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Abstract (original)

> Privatized text rewriting with local differential privacy (LDP) is a recent approach that enables
> sharing of sensitive textual documents while formally guaranteeing privacy protection to
> individuals. However, existing systems face several issues, such as formal mathematical flaws,
> unrealistic privacy guarantees, privatization of only individual words, as well as a lack of
> transparency and reproducibility. In this paper, we propose a new system 'DP-BART' that largely
> outperforms existing LDP systems. Our approach uses a novel clipping method, iterative pruning, and
> further training of internal representations which drastically reduces the amount of noise required
> for DP guarantees. We run experiments on five textual datasets of varying sizes, rewriting them at
> different privacy guarantees and evaluating the rewritten texts on downstream text classification
> tasks. Finally, we thoroughly discuss the privatized text rewriting approach and its limitations,
> including the problem of the strict text adjacency constraint in the LDP paradigm that leads to the
> high noise requirement.
