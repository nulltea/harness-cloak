---
type: handoff
status: current
created: 2026-06-29
updated: 2026-06-29
tags: [lever2, phi, anisotropy, phrase-bert, short-span, embeddings]
companion: docs/perturbation-sota.md
---

# Lever 2 follow-up: short-span embedding models for φ (Phrase-BERT et al.)

Deferred from the probe-scorer work. Lever 2 (the embedding map φ) is the binding
constraint on RANTEXT: qwen3-embedding single-token vectors and the Qwen3-1.7B
`embed_tokens` matrix both fail (anisotropy / distance concentration), and standard
sentence/LLM embedders are documented to be poor on short spans (they never see short
phrases in pretraining and fall back to lexical overlap; anisotropy floor ~0.64–0.72 on
bare entities, measured here).

## What to evaluate
Purpose-built short-span / phrase models as **better-conditioned φ candidates** and as a
**short-span similarity scorer**:

- **Phrase-BERT** (Wang et al. 2021, [arXiv:2109.06304](https://arxiv.org/abs/2109.06304);
  HF `whaleloops/phrase-bert`) — BERT contrastively fine-tuned on phrases; better phrase
  geometry, increased lexical diversity among neighbours. ~110M, CPU-runnable.
- **fastText** (char-n-gram → OOV/short-span coverage) and **counter-fitted vectors**
  (synonym-aware; CUSTEXT+'s utility came from these) — lightweight static alternatives.

## Two uses
1. **φ for the perturbation (Lever 2).** Build a vocab-embedding cache from Phrase-BERT
   (and/or fastText / counter-fitted), then re-run `diagnostics.py` (anisotropy, rel-spread,
   mechanism retention) and the φ A/B in `eval.py`, comparing against qwen3-embedding and
   the Qwen3-matrix. Hypothesis: better spread + synonym structure → higher retention at a
   given candidate-set size. Pairs with the documented fixes in
   `docs/perturbation-sota.md` (counter-fitted / whitening / rank-based candidate set).
2. **PII paraphrase tail.** PII containment now uses rapidfuzz (verbatim/fuzzy). Phrase-BERT
   cosine (or an LLM judge) is for the *non-verbatim* entities rapidfuzz misses (entity
   referred to but reworded). Lower priority than (1).

## Status
Deferred. The PII probes were switched to rapidfuzz + Presidio-on-target (this is the
containment fix); Phrase-BERT is the semantic short-span follow-up, scoped to Lever 2.
