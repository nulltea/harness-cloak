---
type: paper
node_id: paper:zaratiana2023_gliner
title: "GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer"
authors: ["Urchade Zaratiana", "Nadi Tomeh", "Pierre Holat", "Thierry Charnois"]
year: 2023
venue: "NAACL 2024"
external_ids:
  arxiv: "2311.08526"
  doi: null
  s2: null
tags: ["ner", "zero-shot", "span-detection", "pii-detection", "architecture"]
added: 2026-07-03T00:00:00Z
---

# GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer

## One-line thesis
Zero-shot NER as span–label matching: a compact bidirectional encoder (DeBERTa-v3 backbone) embeds
candidate spans and natural-language entity-type prompts into a shared space and scores matches with
a sigmoid dot product, beating ChatGPT and fine-tuned LLMs on zero-shot NER at a fraction of the size.

## Problem / Gap
Open-type NER was handled by autoregressive LLMs (expensive, slow, weak span grounding). Fixed-label
token classifiers can't accept new entity types at inference.

## Method
DeBERTa-v3 backbone (chosen for empirical performance) with input
`[ENT] type_0 [ENT] type_1 … [SEP] word_0 …`: entity representations from the `[ENT]` token
outputs via a two-layer FFN; span embedding `S_ij = FFN(h_i ⊕ h_j)` over all spans up to max width
K=12 (linear complexity); match score `φ(i,j,t) = σ(S_ij · q_t)` trained with binary cross-entropy,
in-batch negative entity types. Greedy non-overlapping span decoding (nested allowed, partial
overlap forbidden), score cutoff 0.5. Non-pretrained layers width 768, dropout 0.4; AdamW,
LR 1e-5 (backbone) / 5e-5 (heads), ≤30k steps, warmup + cosine. Sizes: small 50M / medium 90M /
large 0.3B. Training data: Pile-NER — 44,889 Pile passages, 240k spans, 13k unique entity types,
ChatGPT-annotated; v2.x checkpoints retrained on NuNER (~+3% F1 cross-domain).

## Key Results
- GLiNER-L (0.3B): 60.9 avg zero-shot F1 on CrossNER, beating GoLLIE-7B (58.0), UniNER-13B (55.6),
  ChatGPT (47.5) at 23× smaller than the best LLM baseline.
- Label phrasing acts as a soft prompt: recall depends heavily on how the type is verbalized.
- Fine-tunable; ecosystem of PII fine-tunes exists (urchade/gliner_multi_pii-v1, Knowledgator
  GLiNER-PII small/base/large/edge with 60+ types, NVIDIA GLiNER-PII with 55+ types).

## Relevance to This Project
GLiNER-small-v2.1 zero-shot is our current detector (`src/cloak/detect.py`), with label phrases
hand-mapped to TAB types. The detection gate shows its zero-shot ceiling: DIRECT any-recall 0.998
but QUASI MISC 0.21 / QUANTITY 0.25 / DEM 0.56. Our span-length diagnostic (2026-07-03) refuted the
max-span-width explanation (only 4% of MISC gold spans exceed 12 words), so the gap is semantic —
zero-shot label phrases cannot express TAB's quasi-identifier notions. GLiNER remains a candidate
*fine-tuning* architecture (retains open-label interface for non-TAB corpora) to compare against a
plain token classifier at matched training data.

## Limitations / Failure Modes
- Max span width truncates long free-form spans (minor for TAB).
- Inference cost grows with the number of label prompts per call.
- Zero-shot recall is prompt-phrasing-sensitive and weak on abstract/relational categories.

## Reusable Ingredients
`gliner` library training code for fine-tuning; existing `Detector` integration and chunking.

## Open Questions
Does fine-tuning on TAB erase the zero-shot generality that makes GLiNER attractive
(catastrophic forgetting), or can mixed training retain both?
