---
type: paper
node_id: paper:meisenbacher2025_double_edged_reconstruction
title: "The Double-edged Sword of LLM-based Data Reconstruction: Understanding and Mitigating Contextual Vulnerability in Word-level Differential Privacy Text Sanitization"
authors: ["Stephen Meisenbacher", "Alexandra Klymenko", "Andreea-Elena Bodea", "Florian Matthes"]
year: 2025
venue: "WPES 2025"
external_ids:
  arxiv: "2508.18976"
  doi: "10.1145/3733802.3764058"
  s2: null
tags: ["reconstruction-attack", "contextual-vulnerability", "rantext-family", "word-level-dp", "post-processing"]
added: 2026-07-01T00:00:00Z
---

# The Double-edged Sword of LLM-based Data Reconstruction

## One-line thesis
Word-level DP sanitization leaves *contextual clues* (contextual vulnerability); LLMs can exploit
them to reconstruct semantics — but the same capability can be turned into a post-processing step
that improves both privacy and utility.

## Problem / Gap
Single-word perturbations are done in isolation, yet a whole sanitized document leaves cross-token
context an attacker (or a helper) can exploit; prior work under-tested this across mechanisms/levels.

## Method
Uses advanced LLMs to reconstruct DP-sanitized text across a broader range of mechanisms and
privacy levels than prior work; then repurposes reconstruction as an adversarial post-processing step.

## Key Results
- Confirms the **double-edged sword**: LLMs infer original semantics and can degrade empirical
  privacy, yet reconstruction-as-post-processing can *increase* privacy and quality.
- Recommends "thinking adversarially" — run reconstruction to harden DP-sanitized output.

## Assumptions
_TODO._

## Limitations / Failure Modes (of the RANTEXT-family it reports on)
- **F1a** (context-blindness) is the *source* of contextual vulnerability; per-token randomization
  ignores document-level correlations → **F4 / A1**.
- Most direct external statement of the **E1↔C3 duality**: reconstruction is simultaneously the
  attack and the utility/privacy repair.

## Reusable Ingredients
LLM reconstruction as an adversarial post-processing / privacy-hardening step.

## Open Questions
_TODO._

## Claims
_TODO._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Cited in `docs/research/rantext-limitations.md` as the field's clearest framing of the **E1↔C3
reconstruction duality** and of **F1a → F4** (contextual vulnerability from token independence).

## Abstract (original)

> Differentially private text sanitization refers to the process of privatizing texts under the
> framework of Differential Privacy (DP), providing provable privacy guarantees while also
> empirically defending against adversaries seeking to harm privacy. Despite their simplicity, DP
> text sanitization methods operating at the word level exhibit a number of shortcomings, among them
> the tendency to leave contextual clues from the original texts due to randomization during
> sanitization – this we refer to as contextual vulnerability. Given the powerful contextual
> understanding and inference capabilities of Large Language Models (LLMs), we explore to what
> extent LLMs can be leveraged to exploit the contextual vulnerability of DP-sanitized texts. We
> expand on previous work not only in the use of advanced LLMs, but also in testing a broader range
> of sanitization mechanisms at various privacy levels. Our experiments uncover a double-edged sword
> effect of LLM-based data reconstruction attacks on privacy and utility: while LLMs can indeed
> infer original semantics and sometimes degrade empirical privacy protections, they can also be
> used for good, to improve the quality and privacy of DP-sanitized texts. Based on our findings, we
> propose recommendations for using LLM data reconstruction as a post-processing step, serving to
> increase privacy protection by thinking adversarially.
