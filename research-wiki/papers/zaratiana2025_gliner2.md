---
type: paper
node_id: paper:zaratiana2025_gliner2
title: "GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface"
authors: ["Urchade Zaratiana", "Gil Pasternak", "Oliver Boyd", "George Hurn-Maloney", "Ash Lewis"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2507.18546"
  doi: null
  s2: null
tags: ["ner", "zero-shot", "multi-task", "span-detection", "efficiency"]
added: 2026-07-03T00:00:00Z
---

# GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface

## One-line thesis
Successor to GLiNER: one sub-500M encoder handles NER, text classification, and hierarchical
structured extraction through a schema interface, CPU-deployable, pip-installable
(github.com/fastino-ai/GLiNER2).

## Problem / Gap
GLiNER covers only flat entity extraction; multi-task IE (entities + document attributes +
structures) still required LLMs or multiple specialist models.

## Method
Schema-driven prompting of a single bidirectional encoder (DeBERTa-v3 family, 205M params,
context extended 512→2048 tokens): input `[Task Prompt] ⊕ [SEP] ⊕ [Input Text]`; span-scoring
heads for entity/structure tasks, classification heads for label tasks; multiple tasks compose in
one forward pass. Training: 254,334 examples — 135,698 real documents (news, Wikipedia, legal,
PubMed, arXiv) annotated by GPT-4o, plus 118,636 GPT-4o synthetic business/personal scenarios with
full multi-task annotations. Checkpoints: gliner2-base/multi/large-v1 (Fastino HF).

## Key Results
- CrossNER zero-shot: 0.590 avg F1 — near GPT-4o (0.599), a few points under the NER-specialist
  GLiNER-M (0.615), while being a general multi-task model.
- Zero-shot text classification: best open-source average (0.72) vs GLiClass 0.63, DeBERTa-NLI 0.69.
- All labels in one pass regardless of label count (DeBERTa-NLI is 6.8× slower at 20 labels);
  ~2.6× GPT-4o speedup on CPU.

## Relevance to This Project
The relevant frontier for the open-label detector path: if we fine-tune a GLiNER-family model for
TAB categories, GLiNER2's schema interface could later carry DIRECT/QUASI as a per-span attribute
in the same pass (identifier-type classification is a schema field, not a second model). Also the
natural upgrade path if the substitutor later needs joint span + attribute extraction. Noted, not
adopted: for the current detection-gate gap, plain fine-tuning evidence
([[jha2026_piibench_deberta]]) says the simpler head wins.

## Limitations / Failure Modes
- Preprint; span-detection accuracy at matched fine-tuning data vs a plain token classifier is not
  isolated in the paper.

## Reusable Ingredients
Schema interface for joint span+attribute prediction; pip library.

## Open Questions
Does schema multi-tasking cost span recall relative to a single-task fine-tune of the same encoder?
