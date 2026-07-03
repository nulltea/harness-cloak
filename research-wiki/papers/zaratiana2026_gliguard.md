---
type: paper
node_id: paper:zaratiana2026_gliguard
title: "GLiGuard: Schema-Conditioned Classification for LLM Safeguard"
authors: ["Urchade Zaratiana", "Mary Newhauser", "George Hurn-Maloney", "Ash Lewis"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2605.07982"
  doi: null
  s2: null
tags: ["safety", "moderation", "gliner", "encoder", "schema"]
added: 2026-07-03T00:00:00Z
---

# GLiGuard: Schema-Conditioned Classification for LLM Safeguard (Fastino)

## One-line thesis
Safety moderation reframed as schema-conditioned multi-label classification on the GLiNER2
encoder: one 0.3B single-pass model covers prompt/response safety, refusal detection, 14 harm
categories, and 11 jailbreak strategies, competitive with 7B–27B decoder guards.

## Problem / Gap
LLM guardrails use autoregressive 7B+ models — high latency/cost per moderation call.

## Method
Full fine-tune of GLiNER2-base-v1 (300M) for 20 epochs (AdamW). Task definitions, label names, and
descriptions are encoded into the input with `[P]` (task delimiter) and `[L]` (label prefix)
markers, so task/label blocks compose at inference; hard decision rules combine the heads into a
safety verdict. Training data: 87k human-annotated WildGuardTrain examples + GPT-4.1 synthetic
edge cases for fine-grained harm categories.

## Key Results
- 23–90× smaller than compared guards, 16× throughput, 17× lower latency; within 1.7 F1 of the
  strongest prompt-classification baseline (PolyGuard-Qwen 89.4) across nine safety benchmarks.

## Relevance to This Project
One of two concurrent "GLiNER-as-guardrail" works (the other: HiveTrace's GLiNER Guard,
[[minko2026_gliner_guard]]). Surfaced while evaluating GLiNER-family checkpoints as initialization
for our TAB span-detector fine-tune. Not a candidate initialization: it is a safety-moderation
specialist (classification heads, no PII span supervision), so its fine-tuning moved the encoder
*away* from span extraction. Its value here is as evidence that the GLiNER2 encoder fine-tunes
cleanly to new task framings at 0.3B.

## Limitations / Failure Modes
- No span-level PII task at all; moderation taxonomies only. Vendor preprint (Fastino).

## Reusable Ingredients
Schema-conditioned label composition (`[P]`/`[L]`) as a pattern for attaching per-span attributes
(e.g. DIRECT/QUASI) without a second model.

## Open Questions
Whether safety fine-tuning degrades the underlying GLiNER2 span-extraction ability
(catastrophic forgetting) — relevant to any multi-task detector we might build later.
