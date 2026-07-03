---
type: paper
node_id: paper:minko2026_gliner_guard
title: "GLiNER Guard: Unified Encoder Family for Production LLM Safety and Privacy"
authors: ["Bogdan Minko", "Sabrina Sadiekh", "Evgeniy Kokuykin"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2605.05277"
  doi: null
  s2: null
tags: ["pii-detection", "safety", "encoder", "production", "benchmark"]
added: 2026-07-03T00:00:00Z
---

# GLiNER Guard: Unified Encoder Family for Production LLM Safety and Privacy

## One-line thesis
A 145–209M-parameter encoder family doing safety classification and span-level PII detection in a
single forward pass, with production throughput (193 req/s, P99 < 1s on one A100; 1.6× GLiNER2
throughput), plus PII-Bench, a span-level PII evaluation benchmark.

## Problem / Gap
Production LLM guardrails run separate models for safety classification and PII detection, paying
the speed-vs-accuracy trade-off twice.

## Method
Unified compact encoder variants trained multi-task on safety + PII span data; dynamic batching
for serving. Releases PII-Bench for end-to-end span-level PII evaluation. **Backbone of the
compact variants: mmBERT-small** — a multilingual ModernBERT adaptation (22 layers, 384-dim hidden
states, rotary positional embeddings, 100+ languages) — *not* the DeBERTa-v3 spine of the original
GLiNER family. Three variants: uni-encoder (147M, GLiNER2-style cross-encoding, re-encodes schema
per request); shared-weight bi-encoder (145M, one backbone for both branches, label embeddings
precomputable/cacheable for fixed schemas); Omni (209M, initialized from GLiNER2 Multi, keeps more
general-domain transfer). Training: 467,273 multi-task examples (95/5 split) with up to six
supervision signals per sample (span extraction, safety classification, adversarial attack
detection, harmful-content categorization, intent recognition, tone classification); 108,702
examples carry span labels over 32 PII entity types.

## Key Results
- Compact variants reach ~84 F1 average on their PII/safety suite while holding production latency.
- Demonstrates the GLiNER-family encoder scales down to ~150M for PII spans without collapse.

## Relevance to This Project
Calibrates the size-optimal operating point: span-level PII detection is served in production at
~150M params, i.e. our DeBERTa-v3-base / GLiNER-small scale is the right weight class — no need for
a larger backbone before the data-side gap (TAB fine-tuning) is closed. PII-Bench is a candidate
out-of-domain check for our fine-tuned detector, complementing the in-domain TAB gate.

## Limitations / Failure Modes
- Vendor-adjacent preprint (HiveTrace); PII taxonomy is formal-PII-centric (emails, IDs, names),
  not TAB-style quasi-identifiers — their F1 does not predict our QUASI recall.

## Reusable Ingredients
PII-Bench as OOD evaluation; evidence for ~150M weight class.

## Open Questions
Whether PII-Bench includes free-form quasi-identifier spans or only formal PII types.
