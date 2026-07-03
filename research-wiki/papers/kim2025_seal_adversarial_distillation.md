---
type: paper
node_id: paper:kim2025_seal_adversarial_distillation
title: "Self-Refining Language Model Anonymizers via Adversarial Distillation"
authors: ["Kyuyoung Kim", "Hyunjun Jeon", "Jinwoo Shin"]
year: 2025
venue: "NeurIPS 2025"
external_ids:
  arxiv: "2506.01420"
  doi: null
  s2: null
tags: ["llm-anonymization", "distillation", "adversarial", "local-model", "self-refinement", "synthpai", "d2"]
added: 2026-07-02T19:01:24Z
---

# SEAL: self-refining SLM anonymizers via adversarial distillation

## One-line thesis

Distill both *anonymization* and *self-critique* from adversarial LLM-anonymizer-vs-inference-attacker
trajectories into small (≤8B) local models via SFT + preference learning, so anonymization needs no
external model at inference — 8B matches the GPT-4 anonymizer's privacy-utility trade-off and, with
iterative self-refinement, surpasses it on privacy.

## Problem / Gap

LLM-based anonymizers (Staab-style adversarial anonymization) rely on proprietary frontier models
at inference — cost, and the sensitive text is exposed to exactly the kind of untrusted external
system anonymization is supposed to protect against. Gap: get frontier-grade anonymization out of a
local SLM.

## Method

1. **Adversarial trajectory collection:** an LLM anonymizer and an LLM inference (attribute-guessing)
   model play the Staab adversarial loop over SynthPAI; the trajectories record anonymized text
   versions plus the attacker's inferred attributes at each round.
2. **Adversarial distillation:** SFT + preference learning (preference pairs from
   trajectory improvement steps) trains SLMs on two capabilities at once — *anonymize* and
   *critique* (judge whether an anonymization still leaks and what to fix).
3. **Self-refinement at inference:** the distilled model iteratively critiques and revises its own
   output — inference-time compute buys additional privacy without any external model.

## Key Results

- 8B models trained with SEAL reach a privacy-utility trade-off **comparable to the GPT-4
  anonymizer** on SynthPAI (attribute-inference attacker).
- With self-refinement iterations, the 8B model **surpasses GPT-4 on privacy protection**.
- Code released (11/2025): <https://github.com/kykim0/SEAL>. **No model checkpoints on HF**
  (author org checked 2026-07-02) — the critique capability lives inside the anonymizer, not as a
  standalone attacker model.

## Assumptions

- Text-release setting: the anonymized text is the end product; utility is measured on the
  anonymized text itself, not on a downstream task answer mapped back.
- SynthPAI attribute inference is the privacy measure (same benchmark family as AgentStealth/Staab).
- A frontier LLM is available offline for trajectory collection (teacher-time only, like our
  teacher-cascade pattern).

## Limitations / Failure Modes

- **No round trip:** nothing maps a remote model's *output* back; no reverse record, no extractor.
  The co-design space (substitution record, extraction) is untouched.
- **Monolithic 8B policy:** an order of magnitude above our deployed-local-compute budget
  (sub-1B encoder cascade); no decomposition into detection/lookup/ranking/infill subtasks.
- **Utility axis:** text-similarity utility on the released text — subject to the same
  under-anonymization-rewarding concern as AgentStealth's surface metric (exact metrics to verify
  against the full paper before quoting numbers).
- Preference learning (offline) rather than online RL — no live reward on a measured downstream
  outcome.

## Reusable Ingredients

- **Released adversarial trajectories / collection code** over SynthPAI — candidate SFT/preference
  data, and a template for our Way-2-search → DPO-pairs pipeline.
- **Critique distillation** — a distilled local critique head is a candidate *trained* attack head
  `A` (our fork-1 escalation option b), buildable from their trajectory format even though they
  ship no standalone attacker weights.
- **Self-refinement loop** = inference-time search guided by a learned critic — conceptually the
  bridge between our Way 2 (training-free search) and Way 1 (trained policy).
- SFT + preference-learning recipe as the no-policy-gradient alternative to hand-rolled GRPO.

## Open Questions

- Are the collected trajectories themselves in the repo (usable data) or only the collection code?
- What exact utility metric does the paper use, and does it reward under-anonymization?
- How many self-refinement iterations before privacy gains plateau (inference-time cost curve)?

## Claims

_None registered._

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

**Why surfaced:** flagged in the D1 prototype plan (2026-07-02 dataset audit) as "register in the
wiki and assess overlap before D2 planning" — the closest published neighbor to the D2 round-trip
co-optimization plan ([`2026-07-02-roundtrip-grpo-training.md`](../../docs/plans/2026-07-02-roundtrip-grpo-training.md)).

**Overlap assessment (the mandated pre-D2 check):** SEAL shares our corpus (SynthPAI), privacy
measure (attribute-inference attacker), teacher-time-only-LLM discipline, and the goal of a local
small-model anonymizer. It does **not** touch D2's claimed contributions: no round trip (utility
never measured on a remote output mapped back), no substitution record / extractor interface, no
online RL against a measured downstream reward, and its policy is a monolithic 8B rather than a
tailored sub-1B cascade. **Verdict: no contribution collision; strong methodological adjacency.**
Consequences already absorbed into the D2 plan: (i) their trajectories/recipe are candidate
training data and the template for the preference-learning fallback to GRPO; (ii) their critique
distillation is the buildable version of our escalation attack head (no weights released — the
2026-07-02 head survey confirmed nothing off-the-shelf exists); (iii) their 8B-matches-GPT-4
result is the field's benchmark for what "local anonymizer quality" means — our sub-1B cascade
must be compared against it at matched realized privacy on `doc_p` (the `out_final` axis remains
ours alone).

## Abstract (original)

> Large language models (LLMs) are increasingly used in sensitive domains, where their ability to
> infer personal data from seemingly benign text introduces emerging privacy risks. While recent
> LLM-based anonymization methods help mitigate such risks, they often rely on proprietary models
> (e.g., GPT-4), raising concerns about cost and the potential exposure of sensitive data to
> untrusted external systems. To address this, we introduce SElf-refining Anonymization with
> Language model (SEAL), a novel distillation framework for training small language models (SLMs)
> to perform effective anonymization without relying on external models at inference time. SEAL
> leverages adversarial interactions between an LLM anonymizer and an inference model to collect
> trajectories of anonymized texts and inferred attributes, which are then used to distill
> anonymization and critique capabilities into SLMs through supervised fine-tuning and preference
> learning. The resulting models learn both to anonymize text and to evaluate their outputs,
> enabling iterative improvement of anonymization quality via self-refinement. Experiments on
> SynthPAI, a dataset of synthetic personal profiles and text comments, demonstrate that SLMs
> trained with SEAL achieve substantial improvements in anonymization capabilities. Notably, 8B
> models attain a privacy-utility trade-off comparable to that of the GPT-4 anonymizer and, with
> self-refinement, even surpass it in terms of privacy protection. These results highlight the
> effectiveness of our adversarial distillation framework for training SLMs as efficient
> anonymizers.
