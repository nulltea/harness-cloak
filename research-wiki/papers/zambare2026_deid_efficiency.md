---
type: paper
node_id: paper:zambare2026_deid_efficiency
title: "Towards Fair and Efficient De-identification: Quantifying the Efficiency and Generalizability of De-identification Approaches"
authors: ["Noopur Zambare", "Kiana Aghakasiri", "Carissa Lin", "Carrie Ye", "J. Ross Mitchell", "Mohamed Abdalla"]
year: 2026
venue: "EACL 2026 Findings"
external_ids:
  arxiv: "2602.15869"
  doi: null
  s2: null
tags: ["pii-detection", "de-identification", "efficiency", "model-size", "clinical"]
added: 2026-07-03T00:00:00Z
---

# Towards Fair and Efficient De-identification

## One-line thesis
Fine-tuned compact encoders (BERT, ClinicalBERT, ModernBERT) match 70B-class LLMs on
de-identification at a fraction of the inference cost, and *generalize better* across languages,
cultures, and name distributions.

## Problem / Gap
LLMs are increasingly used as de-identification hammers; nobody had quantified the
efficiency–generalizability trade-off against fine-tuned compact encoders fairly.

## Method
Compare three system classes on clinical de-identification: fine-tuned transformers (BERT,
ClinicalBERT, ModernBERT), small LLMs (Llama-8B, Qwen-7B), large LLMs (Llama-70B, Qwen-72B).
Evaluate accuracy, inference cost, and generalization across Mandarin, Hindi, Spanish, French,
Bengali identifiers, regional English variants, and gendered names. Release
BERT-MultiCulture-DEID models fine-tuned on MIMIC with multilingual identifiers.

## Key Results
- Compact fine-tuned models achieve comparable de-identification performance while substantially
  reducing inference cost.
- Smaller fine-tuned models *outperform* larger LLM systems on cross-cultural identifier
  generalization — the "bigger generalizes better" intuition fails here.

## Relevance to This Project
Kills the "use a local LLM as the PII detector" option on both axes we care about: our detector
must run locally per document (cost matters on one iGPU) and must not miss identifiers from any
name distribution (recall is the privacy ceiling). Confirms the compact-encoder fine-tuning path
and suggests ModernBERT as a long-context alternative backbone if chunking proves harmful.

## Limitations / Failure Modes
- Clinical domain (MIMIC) — direct identifiers dominate; says little about TAB-style free-form
  quasi-identifiers (MISC events, demographics).

## Reusable Ingredients
Efficiency/generalizability evaluation framing; ModernBERT as candidate backbone;
multicultural-name augmentation idea for robustness.

## Open Questions
Does the compact-model advantage persist for QUASI spans that require discourse-level judgment?
