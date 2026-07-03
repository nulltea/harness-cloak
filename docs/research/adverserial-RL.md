---
type: research
status: current
created: 2026-07-02
updated: 2026-07-02
tags: [rl, grpo, reward-modeling, frozen-model, goodhart, surrogate-reward, background, d2]
companion: [../plans/2026-07-02-roundtrip-grpo-training.md, ../plans/2026-07-02-codesign-next-stage.md]
---

# RL background: training against frozen models, and what the round trip is called

Background research for the round-trip GRPO plan
([`2026-07-02-roundtrip-grpo-training.md`](../plans/2026-07-02-roundtrip-grpo-training.md)), which
trains the substitutor cascade against a reward computed through a frozen remote LLM
(`Qwen3.6-35B-A3B`). Three questions raised in the 2026-07-02 design session: is training against
a frozen model standard RL practice; does the trained policy generalize to *other* frozen models;
and what is the round-trip step called in RL vocabulary — i.e. what, formally, are we replacing
when we look for a cheaper training signal.

## Definitions

- **Frozen model:** a model whose weights are fixed during training; it shapes the reward or the
  environment but receives no gradients.
- **Environment step (transition):** in RL terms, the world's response to the policy's action.
  Here: `out_p = RemoteLLM(doc_p, task)` — the remote model *is* the environment.
- **Reward function / reward oracle:** the mapping from a completed rollout to the scalar `r`.
  Here: the full `doc_p → out_p → out_final → α·(1−A) + (1−α)·U` pipeline.
- **Reward model (RM):** a learned network standing in for the true reward (the RLHF sense).
- **Surrogate (proxy) reward:** any cheaper stand-in for the true reward oracle — learned or
  rule-based. Replacing the environment itself with a simulator is **model-based RL**.
- **Reward overoptimization / Goodharting:** optimizing a proxy so hard that true performance
  degrades while the proxy score keeps improving.
- **KL leash:** a `KL(policy ‖ reference)` penalty bounding how far optimization can push the
  policy from its initialization — the standard brake on Goodharting.
- **Frozen-partner pipeline RL:** training one component of a multi-model pipeline while the
  others stay frozen.
- Round trip, `doc_p`, `out_p`, `out_final`, R, τ-walk, MTI, α, `A`, `U`: see the
  [plan's definitions](../plans/2026-07-02-roundtrip-grpo-training.md#definitions).

## Is training against a frozen model common?

Yes — it is the default pattern in two established guises:

1. **RLHF itself.** The reward model is a frozen network; the policy optimizes against it for the
   entire run. Every RLHF-trained assistant is a policy trained against a frozen judge.
2. **Frozen-partner pipeline RL.**
   [s3](../../research-wiki/papers/jiang2025_s3_search_agent.md)
   ([arXiv 2505.14146](https://arxiv.org/abs/2505.14146)) trains a small searcher against a fully
   frozen generator LLM on a downstream-outcome reward (generation-accuracy gain), explicitly to
   stay compatible with frozen/proprietary generators — 2.4k samples beat baselines trained on
   70× more data. [MMOA-RAG](../../research-wiki/papers/chen2025_mmoa_rag.md)
   ([arXiv 2501.15228](https://arxiv.org/abs/2501.15228)) jointly trains several pipeline modules
   against one shared final-answer reward. In the anonymization literature specifically,
   [AgentStealth](../../research-wiki/papers/shao2025_agentstealth.md)
   ([arXiv 2506.22508](https://arxiv.org/abs/2506.22508)) runs GRPO against frozen
   attacker/judge models, and [SEAL](../../research-wiki/papers/kim2025_seal_adversarial_distillation.md)
   ([arXiv 2506.01420](https://arxiv.org/abs/2506.01420)) collects its distillation trajectories
   from a frozen adversary.

Our setup — substitutor trained while `Qwen3.6-35B-A3B` (environment), the MTI probe (`A`), and
the utility metric (`U`) all stay frozen — is structurally standard, not exotic.

## Does it generalize to other frozen models?

This is a real, named concern: **reward overoptimization**
([Gao et al.](../../research-wiki/papers/gao2022_reward_overoptimization.md),
[arXiv 2210.10760](https://arxiv.org/abs/2210.10760)). Measured on frozen proxy reward models:
the proxy score climbs monotonically while the *true* reward rises, peaks, and then degrades as
optimization pushes the policy further (in KL) from its initialization — bigger/better-trained
proxies delay the peak but never remove it. Translated to our setting, "the policy exploits the
local attack head's blind spots", "the policy overfits Qwen3.6's context-use quirks", and "the
policy games the utility proxy" are one phenomenon with one playbook:

- **Held-out evaluation models from a different family** — the plan already mandates this for the
  attacker (train against MTI, evaluate against a frontier LLM); the same logic extends to the
  task model (the D1 plan's second-remote-model eval arm).
- **KL leash** — stage 2's `KL(p_φ ‖ p_SFT)` term; Gao et al. study exactly this knob.
- **Ensembles / rotation** of the frozen partner inside the reward — spreads the optimization
  pressure so no single model's quirks are worth learning.
- **Low-pressure optimizers** — best-of-n/search extracts reward with less KL movement and
  Goodharts more slowly per unit of proxy gain (also Gao et al.), which is an independent argument
  for the training-free search probe (Way 2) preceding GRPO.

Positive transfer evidence exists — s3's frozen-generator-trained searcher is presented as
model-agnostic across generators — but transfer is an **empirical, measured property, never an
assumption**. For us the measurement is explicit: evaluate the trained substitutor's Pareto points
under a second remote task model and report the gap.

## What the round-trip step is called

In RL vocabulary the pieces of the plan's "round trip" decompose as:

| Plan term | RL term |
|---|---|
| `RemoteLLM(doc_p, task) → out_p` | **environment step / transition** (the frozen LLM *is* the environment) |
| `doc_p → out_p → out_final → r` pipeline | **reward function / reward oracle** |
| a cheap local stand-in for that pipeline | **surrogate (proxy) reward**; if learned, a **reward model** |
| replacing the remote LLM with a local simulator of it | **model-based RL** |

So "devise a cheaper round-trip function" translates precisely to: *design a surrogate reward
oracle for training, keeping the true oracle for evaluation.*

## The caveat that governs any surrogate

The round-trip reward is hypothesis H2's claimed novelty — "the substitutor learns corruptions the
extractor can invert but the attacker cannot" only has teeth if the reward sees the actual round
trip. Training on a surrogate is therefore a **ladder, not a replacement**: v0 trains on the
surrogate; the true round-trip oracle remains the evaluation; and the measured gap between
surrogate-trained and round-trip-trained policies is itself a first-class result (surrogate
matches → the expensive oracle was never needed, a publishable finding; surrogate falls short →
H2 earned its cost). Two constraints any candidate surrogate must satisfy:

1. **Task-grounded, not surface-similar.** A utility term scored as `doc_p`'s similarity to
   `doc_orig` re-imports the under-anonymization pathology
   ([AgentStealth](../../research-wiki/papers/shao2025_agentstealth.md)'s Standard-Prompt utility
   0.92 — similarity rewards *not anonymizing*). The surrogate must score whether the *task* can
   still be served (answerability, premise entailment), not whether the text looks unchanged.
2. **No laundered single-model dependence.** A reward model distilled from one frozen model's
   round-trip outputs inherits that model's quirks through the data; if model-independence is the
   goal, the surrogate must be model-free (rule/encoder-based) or built from an ensemble.

The candidate surrogates themselves (model-free QA-answerability + NLI premise retention; learned
reward model from D1 tuples; local ensemble round trip; hybrid curriculum) are a design fork of
the plan, not settled background — see the design-session record in
[`2026-07-02-roundtrip-grpo-training.md`](../plans/2026-07-02-roundtrip-grpo-training.md).

## Sources

[Gao et al. 2022](../../research-wiki/papers/gao2022_reward_overoptimization.md)
([arXiv 2210.10760](https://arxiv.org/abs/2210.10760)) — reward-model overoptimization scaling laws;
[s3](../../research-wiki/papers/jiang2025_s3_search_agent.md)
([arXiv 2505.14146](https://arxiv.org/abs/2505.14146)) — frozen-generator searcher RL, downstream
reward, cross-generator transfer;
[MMOA-RAG](../../research-wiki/papers/chen2025_mmoa_rag.md)
([arXiv 2501.15228](https://arxiv.org/abs/2501.15228)) — shared-reward multi-component pipeline RL;
[AgentStealth](../../research-wiki/papers/shao2025_agentstealth.md)
([arXiv 2506.22508](https://arxiv.org/abs/2506.22508)) — GRPO vs frozen judges in anonymization,
surface-utility pathology;
[SEAL](../../research-wiki/papers/kim2025_seal_adversarial_distillation.md)
([arXiv 2506.01420](https://arxiv.org/abs/2506.01420)) — frozen-adversary trajectory distillation;
[RUPTA](../../research-wiki/papers/yang2025_rupta.md)
([arXiv 2407.11770](https://arxiv.org/abs/2407.11770)) — evaluator=attacker circularity.
