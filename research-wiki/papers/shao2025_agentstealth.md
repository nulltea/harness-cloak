---
type: paper
node_id: paper:shao2025_agentstealth
title: "AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text"
authors: ["Chenyang Shao", "Tianxing Li", "Chenhao Pu", "Fengli Xu", "Yong Li"]
year: 2025
venue: "NeurIPS 2025 (under review)"
external_ids:
  arxiv: "2506.22508"
  doi: null
  s2: null
tags: ["llm-anonymization", "reinforcement-learning", "adversarial", "local-model", "rd4"]
added: 2026-07-02T00:00:00Z
---

# AgentStealth: RL-reinforced LLM anonymization on a local model

## One-line thesis
Anonymize user text with a *locally deployed small* LLM (Llama-3.1-8B-Instruct) trained by a
three-stage recipe — adversarial-workflow data collection → joint SFT (defender + attacker roles) →
online GRPO with self-generated adversarial rewards.

## Method

Objective: minimize attribute-inference attack accuracy `Acc_attack` on anonymized text `t̃` while
maximizing surface utility `U` (mean of BLEU, ROUGE-1, ROUGE-L); composite
`J = λ(1−Acc_attack) + (1−λ)U`. Three sequential stages:

**Stage 1 — adversarial anonymization workflow (data collection).** A DeepSeek-V3-driven
Staab-style adversarial loop (anonymizer vs attribute-inference attacker, N iterated rounds) over
batched text–attribute pairs, extended with:
- *In-context Contrastive Learning:* when a sample flips protection status between rounds
  (attack succeeds on `t̃_fail`, fails on `t̃_success`), the pair is fed to an LLM prompt that
  distills a generalizable insight ("why the second anonymization succeeded"). Insights live in a
  memory module capped at `M_max`, updated via UPVOTE/DOWNVOTE/EDIT/ADD operations (≤4 ops per
  update, insights <100 words). Memory is frozen after training; inference uses only stored insights.
- *Adaptive Utility-Aware Control:* after each round, if utility drop
  `U(t̃^(k−1)) − U(t̃^(k)) > τ_U`, a utility-warning message is concatenated to the next
  anonymization prompt; otherwise the base prompt is used. Anonymizer prompt instructs "change as
  little as possible."

**Stage 2 — joint SFT.** LoRA fine-tuning (rank 8, 3 epochs, lr 1e-4, AdamW, max len 8192) of
Llama-3.1-8B-Instruct on workflow-collected data with **two roles in one model**: anonymization
pairs `(t, t̃*)` (including attack-analysis-hardened cases) and attack pairs `(t′, a)` (infer
attribute from original/partially/fully anonymized text). One model `M'_dual` acts as both
defender and attacker.

**Stage 3 — online RL (GRPO).** Policy initialized from θ_SFT; reward
`R = λ_RL·R_anonymity + (1−λ_RL)·R_utility` with λ_RL = 0.5.
`R_anonymity = 1 − 𝟙[M'_attack(t̃; θ_SFT) = a]` — the adversary is the model's **own frozen SFT
attack head** (no external attacker). `R_utility = U(t̃, t)` (BLEU/ROUGE mean). Standard GRPO with
PPO-style clipping and KL penalty to the reference policy; 2 generations/group, batch size 1,
200 global steps, lr 1e-4, 2×A800-80G.

**Attacker/evaluation prompts** are taken verbatim from Staab et al. (top-3 guesses + 1–5
certainty per attribute; entity-consistency judged by an LLM yes/no/less-precise prompt).

Code: <https://github.com/tsinghua-fib-lab/AgentStealth>.

## Key Results

**Datasets:** two *synthetic* corpora merged — SynthPAI (Yukhymenko et al., NeurIPS 2024)
Reddit-style comments + Staab et al.'s 525 synthetic Q&A pairs; 8 attributes (age, gender,
location, occupation, education, relationship status, income, birthplace). Test = first 100
samples of each (200 total); rest for training. No real-user data.

**Metrics:** *Anonymity* = fraction of attributes where the attacker's top-1 guess on the 5th-round
anonymized text diverges from ground truth; *Progress* = fraction improved vs unprotected;
*ScoreUtility* = [BLEU + ROUGE-1 + ROUGE-L + (Meaning−1)/9]/4 with Meaning/Readability scored 1–10
by DeepSeek-V3.

Main table (trained 8B model; higher = better):

| Method | Anonymity | Progress | ScoreUtility | Readability | Meaning |
|---|---|---|---|---|---|
| Azure entity recognizer | 39.1% | 20.4% | 0.87 | 4.89 | 7.61 |
| Adversarial Anonymization (AA, Staab) | 56.7% | 37.4% | 0.74 | 9.49 | 8.30 |
| Standard Prompt | 50.4% | 29.2% | 0.92 | 9.83 | 9.36 |
| Workflow only | 52.5% | 31.5% | 0.80 | 9.48 | 8.46 |
| Workflow+SFT | 62.6% | 43.5% | 0.78 | 9.88 | 8.55 |
| Workflow+RL (no SFT) | 53.2% | 30.3% | 0.86 | 9.95 | 9.03 |
| **AgentStealth (Workflow+SFT+RL)** | **63.7%** | 43.3% | 0.79 | 9.89 | 8.62 |

- Headline "+12.3% anonymization" = 56.7%→63.7% vs AA (relative); "+6.8% utility" = 0.74→0.79.
- Inference-only workflow on DeepSeek-V3 (no training): AgentStealth 66.4% anonymity / 0.69
  utility vs AA 65.7% / 0.65 — workflow alone adds little anonymity (+1.1%), mostly utility.
- SFT lifts anonymization by 19.2% over base Llama-3.1-8B; SFT-trained attack head reaches ~65%
  attack accuracy vs ~50% base, comparable to DeepSeek-V3.
- RL without SFT is unstable (declining rewards, lower ceiling); RL on top of SFT adds only
  +1.1 pt anonymity (62.6→63.7).
- Trade-off curve: at fixed 50% anonymity, +8.6% relative ScoreUtility vs AA; at fixed 0.85
  utility, +16.9% anonymity.

## Limitations / Failure Modes

Paper's own admission (A.8.1): evaluated on only two synthetic datasets; no evidence it holds on
real-world text; no other suitable open datasets found.

Additional analysis (ours):
- **Self-adversarial circularity / reward-hacking surface:** the RL adversary is the model's own
  *frozen* SFT attack head. The policy can learn to beat one static attacker rather than the
  attacker distribution; the paper does not evaluate against a held-out stronger attacker at the
  final operating point (Figure 6 compares attack accuracy, but the eval attacker for the main
  Anonymity metric is not clearly stated to be independent of training).
- **Utility is surface similarity, not task output:** BLEU/ROUGE(+LLM Meaning) against the
  original text. Standard Prompt "wins" utility (0.92) precisely by barely changing the text —
  the metric rewards under-anonymization and says nothing about downstream task performance.
- **Marginal RL gain:** +1.1 pt over SFT alone at 200 GRPO steps, group size 2 — the expensive
  stage contributes least; most of the lift is the workflow data + SFT.
- **Binary per-attribute anonymity** (top-1 diverges or not) at a fixed 5th round; no
  rank/calibration view of the attacker, no matched-privacy Pareto comparison across methods.
- Tiny test set (200 merged samples); λ_RL fixed at 0.5, no sensitivity reported.

## Co-design fitness (doc_orig→doc_p ↔ out_p→out_final)

- **(a) Conditions on / emits:** anonymizer conditions on the full comment set, the attacker's
  explicit inferences about it, the frozen insight memory, and (adaptively) utility scores from
  the previous round; emits a free-form rewrite of the whole text ("change as little as
  possible", no invented information — a soft minimal-edit prior, not a span-level substitution).
- **(b) Client-side record:** none per-document. No span map or edit trace is retained; the
  insight memory is a set of *generic* rules, useless as a reverse map. Intermediate rounds exist
  during training only.
- **(c) Reverse/reconstruction step:** none. The anonymized text is the end product (social-media
  release setting); there is no out_p and no un-perturbation anywhere in the paper.
- **(d) Round-trip behavior:** untested and structurally at risk. The learned policy abstracts
  premises (case study: "coding and debugging" → "work", "Durban" → "my hometown"); a remote task
  needing those premises (advice, drafting, QA) computes on the abstraction, and nothing
  guarantees the client can re-specialize the answer — recoverability is not an optimization
  target, and the reward actively encourages destroying exactly the attributes a task may need.
- **(e) Adversary:** yes — a Staab-style LLM attribute-inference attacker (8 attributes, top-1
  match after entity-consistency judgment). During RL it is the model's own SFT attack head; at
  evaluation, attack prompts are Staab's (evaluator model identity for the main table not fully
  explicit — caveat).
- **(f) Verdict:** the strongest available template for *training* our local substitutor: the
  three-stage recipe (adversarial workflow → dual-role SFT → GRPO with
  `R = λ·attacker-failure + (1−λ)·utility`) transfers directly, and the co-design move is to swap
  `R_utility` from BLEU/ROUGE-vs-doc_orig to **round-trip utility of out_final given our
  extractor** — the reward plumbing already exists. It also shows the attacker can live inside
  the same 8B model, cheap for local training. What it cannot provide: any reconstruction
  mechanism, any per-span record to condition an extractor on, formal guarantees, or evidence
  that its anonymity survives an attacker not descended from its own training loop.

## Relevance to This Project
**Why surfaced:** round-2 (Semantic Scholar + arXiv) discovery — a **new RD4 sub-family (RL-guided)** in
[`docs/research/learned-substitution.md`](../../docs/research/learned-substitution.md). **Fit:** the
self-improving, on-device version of the [Staab](staab2024_llm_anonymizers.md)/[RUPTA](yang2025_rupta.md)
adversarial loop — arguably the strongest local-substitutor direction; trades formal DP for empirical
gains and adds reward-hacking risk.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
