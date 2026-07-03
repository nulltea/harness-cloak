---
type: paper
node_id: paper:chen2025_mmoa_rag
title: "Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning"
authors: ["Yiqun Chen", "Lingyong Yan", "Weiwei Sun", "Xinyu Ma", "Yi Zhang", "Shuaiqiang Wang", "Dawei Yin", "Yiming Yang", "Jiaxin Mao"]
year: 2025
venue: "NeurIPS 2025"
external_ids:
  arxiv: "2501.15228"
  doi: null
  s2: null
tags: ["pipeline-rl", "multi-agent-rl", "shared-reward", "credit-assignment", "rag", "rl-background", "d2"]
added: 2026-07-02T19:40:00Z
---

# MMOA-RAG: a multi-component pipeline trained as cooperative multi-agent RL

## One-line thesis

Treat every module of a RAG pipeline (query rewriting, retrieval, filtering, answer generation) as
a cooperative RL agent and jointly optimize them all with multi-agent PPO toward one shared
final-answer reward (answer F1) — aligning per-module objectives with the end task beats
per-module supervised finetuning.

## Problem / Gap

Pipeline components optimized separately (SFT per module) misalign with the end objective; prior
pipeline-RL handles only two-component pipelines or ignores inter-module interdependencies.

## Method

Multi-agent cooperative formulation: each module is an agent; MAPPO (centralized advantage,
decentralized policies) with the **same shared reward** — final-answer F1 — propagated to every
agent's policy-gradient loss. Ablations isolate per-module contributions.

## Key Results

- Outperforms separate-SFT and prior RL baselines across QA benchmarks (NeurIPS 2025).
- Ablations show each jointly-trained module contributes; the framework adapts to different
  pipeline compositions and benchmarks.
- Demonstrates that **shared-scalar-reward multi-component training works in practice** for
  3–4-component LLM pipelines without per-component reward shaping.

## Assumptions

- Common-payoff setting (all agents share one reward) — no competing objectives between modules.
- The shared reward (answer F1) is cheap and objective per episode.

## Limitations / Failure Modes

- MAPPO needs a centralized critic (value model) — machinery our GRPO variant replaces with
  group-relative advantages.
- Credit assignment across agents is implicit (shared advantage); per-module signal quality
  degrades as module count grows — fine at 2–4 components.

## Reusable Ingredients

- **Shared group-advantage justification:** the direct precedent for applying one scalar advantage
  to multiple components' policy-gradient losses in a common-payoff pipeline (our stage-2 joint
  update of ranker + infiller).
- Per-module ablation protocol for attributing a jointly-trained gain.

## Open Questions

- Does the shared-reward approach hold when one component's action space is discrete/classification
  (our ranker) and another's is free generation (our infiller)? Their modules are all
  generation-shaped; ours are mixed.

## Claims

_None registered._

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

**Why surfaced:** background for the round-trip GRPO plan
([`2026-07-02-roundtrip-grpo-training.md`](../../docs/plans/2026-07-02-roundtrip-grpo-training.md)):
MMOA-RAG is the published existence proof for the plan's stage-2 design — multiple heterogeneous
pipeline components trained jointly on **one shared downstream scalar reward**, no per-component
reward engineering, no counterfactual machinery. The 2026-07-02 joint-training survey ranked this
shared-advantage pattern #1 (zero extra rollouts, no critic beyond what the algorithm needs; our
GRPO variant drops MAPPO's critic for group-relative advantages). Its per-module ablation protocol
is the template for attributing stage-2 gains between ranker and infiller.

## Abstract (original)

> Retrieval-augmented generation (RAG) is widely utilized to incorporate external knowledge into
> large language models, thereby enhancing factuality and reducing hallucinations in
> question-answering (QA) tasks. A standard RAG pipeline consists of several components, such as
> query rewriting, document retrieval, document filtering, and answer generation. However, these
> components are typically optimized separately through supervised fine-tuning, which can lead to
> misalignments between the objectives of individual components and the overarching aim of
> generating accurate answers. Although recent efforts have explored using reinforcement learning
> (RL) to optimize specific RAG components, these approaches often focus on simple pipelines with
> only two components or do not adequately address the complex interdependencies and collaborative
> interactions among the modules. To overcome these limitations, we propose treating the complex
> RAG pipeline with multiple components as a multi-agent cooperative task, in which each component
> can be regarded as an RL agent. Specifically, we present MMOA-RAG, Multi-Module joint
> Optimization Algorithm for RAG, which employs multi-agent reinforcement learning to harmonize
> all agents' goals toward a unified reward, such as the F1 score of the final answer. Experiments
> conducted on various QA benchmarks demonstrate that MMOA-RAG effectively boosts the overall
> performance of the pipeline and outperforms existing baselines.
