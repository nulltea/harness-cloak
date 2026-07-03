---
type: paper
node_id: paper:jiang2025_s3_search_agent
title: "s3: You Don't Need That Much Data to Train a Search Agent via RL"
authors: ["Pengcheng Jiang", "Xueqiang Xu", "Jiacheng Lin", "Jinfeng Xiao", "Zifeng Wang", "Jimeng Sun", "Jiawei Han"]
year: 2025
venue: "EMNLP 2025"
external_ids:
  arxiv: "2505.14146"
  doi: null
  s2: null
tags: ["pipeline-rl", "frozen-partner", "rag", "downstream-reward", "rl-background", "d2"]
added: 2026-07-02T19:40:00Z
---

# s3: RL-train the searcher, freeze the generator

## One-line thesis

Decouple a RAG pipeline's searcher from its generator, freeze the generator entirely, and RL-train
only the searcher on a *downstream-outcome* reward ("Gain Beyond RAG" = generation-accuracy
improvement over naive RAG) — 2.4k samples beat baselines trained on 70× more data, with the
trained searcher transferring across frozen/proprietary generators.

## Problem / Gap

Search-agent RL either optimizes retrieval-only metrics (NDCG — ignores downstream utility) or
finetunes the whole LLM to retrieve-and-reason jointly (entangles retrieval with generation,
incompatible with frozen/proprietary generators).

## Method

Searcher = small trainable policy issuing multi-turn queries; generator = frozen LLM. Reward per
episode: **Gain Beyond RAG** — the frozen generator's answer accuracy with the searcher's context
minus its accuracy with naive-RAG context. PPO-style RL on the searcher only.

## Key Results

- 2.4k training samples outperform baselines trained on >70× more data, across six general-QA and
  five medical-QA benchmarks.
- Model-agnostic by construction: the trained searcher works with generators it never trained
  against (the paper's stated compatibility motivation with frozen/proprietary models).
- Downstream-outcome reward ≫ search-only metric reward for end-task quality.

## Assumptions

- The frozen generator is a *reasonable* consumer of context — the searcher's learned behavior is
  only as transferable as generators are alike in how they exploit context.
- Answer accuracy is scriptable (QA with gold answers) so the reward is cheap and objective.

## Limitations / Failure Modes

- Transfer across generators is empirical, not guaranteed — a searcher tuned to one generator's
  context-use quirks can Goodhart it (cf. reward overoptimization).
- Differential reward (gain over a baseline) needs two generator calls per episode.

## Reusable Ingredients

- **Frozen-partner training pattern:** train the cheap component against a frozen expensive one,
  on a downstream-outcome reward — structurally identical to our substitutor-vs-frozen-remote-LLM.
- **Differential reward** ("gain beyond baseline"): reward = outcome(candidate) − outcome(default);
  our analog would be `U(out_final at candidate doc_p) − U(out_final at τ-walk doc_p)` — a
  variance-reduction trick that also centers the reward.
- Sample-efficiency datum: pipeline-RL with a dense downstream reward can work at 10³-sample scale
  — our 60-doc v0.1 corpus is not obviously too small.

## Open Questions

- How far does searcher transfer degrade across generator families? (The measured analog of our
  "does the trained substitutor generalize beyond Qwen3.6" question.)

## Claims

_None registered._

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

**Why surfaced:** background for the round-trip GRPO plan
([`2026-07-02-roundtrip-grpo-training.md`](../../docs/plans/2026-07-02-roundtrip-grpo-training.md)):
s3 is the cleanest published instance of the exact pattern the plan proposes — a small local
policy trained against a **frozen** LLM environment on a **measured downstream outcome**, not a
surface metric. It answers "is training against a frozen model common?" affirmatively with a
strong result, supplies the staged-training precedent (2026-07-02 joint-training survey ranked its
partner-at-greedy pattern #2), demonstrates cross-generator transfer as an achievable-but-measured
property, and its differential "gain beyond baseline" reward is a candidate refinement for our
reward centering.

## Abstract (original)

> Retrieval-augmented generation (RAG) systems empower large language models (LLMs) to access
> external knowledge during inference. Recent advances have enabled LLMs to act as search agents
> via reinforcement learning (RL), improving information acquisition through multi-turn
> interactions with retrieval engines. However, existing approaches either optimize retrieval
> using search-only metrics (e.g., NDCG) that ignore downstream utility or fine-tune the entire
> LLM to jointly reason and retrieve—entangling retrieval with generation and limiting the real
> search utility and compatibility with frozen or proprietary models. In this work, we propose s3,
> a lightweight, model-agnostic framework that decouples the searcher from the generator and
> trains the searcher using a Gain Beyond RAG reward: the improvement in generation accuracy over
> naive RAG. s3 requires only 2.4k training samples to outperform baselines trained on over 70x
> more data, consistently delivering stronger downstream performance across six general QA and
> five medical QA benchmarks.
