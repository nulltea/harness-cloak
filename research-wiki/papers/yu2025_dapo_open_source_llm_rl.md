---
type: paper
node_id: paper:yu2025_dapo_open_source_llm_rl
title: "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"
authors: ["Qiying Yu", "et al."]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2503.14476"
  doi: null
  s2: null
tags: ["group-relative-rl", "advantage-collapse", "dynamic-sampling"]
added: 2026-07-30T00:00:00Z
---

# DAPO: An Open-Source LLM Reinforcement Learning System at Scale

## One-line thesis
Group-relative RL with homogeneous rewards yields exactly-zero advantages; DAPO dynamic sampling over-samples and discards reward-homogeneous prompts so every retained batch element carries gradient.

## Key Results
- Dynamic sampling: filter degenerate (all-same-reward) groups at runtime.
- Large measured fraction of batches otherwise gradient-dead.

## Relevance to This Project
Surfaced for root-cause link 1 (dead LOO groups from identical rewards - our fully-degenerate-group phenomenon in group-relative form). Validates runtime tie/degeneracy detection as standard practice and frames our counterfactual broadcast as the complementary move (inject exact signal instead of discarding); their Advantage-Collapse-Rate-style monitoring matches our dead-group diagnostics.
