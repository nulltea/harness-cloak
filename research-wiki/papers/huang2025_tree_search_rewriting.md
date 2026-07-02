---
type: paper
node_id: paper:huang2025_tree_search_rewriting
title: "Zero-Shot Privacy-Aware Text Rewriting via Iterative Tree Search"
authors: ["Shuo Huang", "Xingliang Yuan", "Gholamreza Haffari", "Lizhen Qu"]
year: 2025
venue: "EMNLP 2025"
external_ids:
  arxiv: "2509.20838"
  doi: null
  s2: null
tags: ["text-rewriting", "tree-search", "reward-model", "zero-shot", "rd4"]
added: 2026-07-02T00:00:00Z
---

# Zero-Shot Privacy-Aware Text Rewriting via Iterative Tree Search

## One-line thesis
Obfuscate/delete private segments by a zero-shot, reward-model-guided *tree search* over rewrites,
preserving coherence and naturalness — inference-time search instead of finetuning. Method name:
**NaPaRe**.

## Method
MCTS-style iterative tree search over sentence rewrites, no task-specific training anywhere:

- **Node** = a modified version of the sentence (an intermediate rewriting state).
- **Actions** per privacy segment, two discrete strategies: **delete** (remove the segment) or
  **obscure** (replace it with a less specific / more general term).
- **Expansion**: a one-step rewriter — stochastic sampling from **Llama 3.1 8B** with a privacy-aware
  prompt — generates N candidate rewrites per node. Candidates are screened by a leakage score
  LS(y, p_seg) quantifying "residual presence of private attributes" (exact computation not spelled out
  in the HTML text); failing that, the highest-utility candidate is kept.
- **Search loop** (Algorithm 1): UCT selection → expand via Rewrite() → score with reward model →
  backpropagate; terminate when reward r ≥ γ (γ = 0.10, set empirically) or the budget of **5
  iterations** is exhausted, in which case the best leaf so far is returned.
- **Sequential per-segment processing**: once the best rewrite for the first privacy segment is found,
  it is frozen and the search moves to the next segment.
- **Reward model**: **ArmoRM** (mixture-of-experts multi-objective preference model), used off-the-shelf.
  It scores general generation quality — coherence, relevance, instruction-following — *not* privacy;
  it was never trained on privacy tasks. Privacy enters the search only through the prompt + leakage
  screening, not the reward.
- **Privacy segment detection** ("Privacy Segment Alignment"): maps input spans to persona attributes
  via cosine similarity or a fine-tuned LM; a fine-tuned Llama2-7B detector reaches 89.38% ROUGE-1 /
  88.88% overlap coefficient on span detection (Table 5).

Note: the two roles of γ (leakage screen LS ≤ γ vs reward termination r ≥ γ) read ambiguously in the
HTML; the explicit statement is γ=0.10 "to filter the rewrite quality" with termination at r ≥ γ.

## Evaluation setup
- **Datasets:** **NaP2** (Naturalness and Privacy-Preserving Rewriting Corpus, built on PERSONA-CHAT;
  persona attributes are the privacy constraints; 140 examples) and **ECHR** (298 sampled legal-judgment
  texts; downstream task = binary legal judgment prediction).
- **Privacy evaluation is post-hoc, non-adversarial:** Privacy_NLI (RoBERTa/MNLI classifier: does the
  rewrite still entail the persona attribute?) and PII-extraction F1 via FLAIR tagging; plus a
  token-level reconstruction-attack ASR. No LLM re-identification attacker.
- **Baselines:** GPT-4 prompting, T5-Base fine-tuned on NaP2, DP-Prompt, DP-MLM, FLAIR-SCRUBBING
  (rule-based redaction).

## Key Results
NaP2 (persona obfuscation):

| Metric | NaPaRe | T5-NaP2 | GPT-4 | DP-Prompt | DP-MLM |
|---|---|---|---|---|---|
| Privacy_NLI ↑ | 93.02% | 93.81% | 82.24% | 77.65% | 79.16% |
| ROUGE-1 (content) ↑ | 73.68% | 73.01% | 33.33% | 85.71% | 45.05% |
| PPL ↓ | 151.83 | 279.35 | 83.35 | 788.99 | 1108.28 |
| LLM naturalness (1–5) ↑ | 4.0 | 3.0 | 4.18 | 1.00 | 1.39 |

(Human-rewrite reference PPL: 118.1. DP-Prompt's high ROUGE-1 comes with near-random privacy and PPL
789 — the DP baselines degenerate on fluency.)

- **ECHR privacy:** PII-extraction F1 on rewrites = **5.18%** for NaPaRe (lowest leakage) vs DP-Prompt
  15.68%, DP-MLM 16.46%.
- **ECHR utility:** legal-judgment accuracy 68.00% (NaPaRe, 5-sentence setting) vs 82.00% on original
  text, 58.19% DP-Prompt, 29.76% FLAIR-SCRUBBING — a 14 pp utility cost vs original, but far above
  redaction.
- **Reconstruction attack:** token-level ASR 3.07%.
- **Cost:** "Processing 100 examples takes 42.5 minutes (≈ 51 s/sentence)" on a single A40 (46GB);
  estimated $0.332 per 100 examples at $0.47/h (vs $0.42 for GPT-4 API); full ECHR run ~23 h.

## Limitations / Failure Modes
Paper's own admissions:
- Rewrites "may still retain implicit privacy cues or introduce inconsistencies" from zero-shot
  prompting variability; open-source models degrade on sentences needing multiple private edits.
- Scope limited to general text (not email/chat/structured docs); small datasets (140 + 298 examples);
  evaluation "primarily relies on automatic metrics", little human annotation.

Our analysis (not in paper):
- **The reward model optimizes only fluency/coherence — privacy is not in the objective.** The search
  maximizes ArmoRM reward subject to a leakage screen; privacy protection thus rests on the prompt, the
  segment detector, and the screen, none of which is adversarially validated during search.
- Privacy metrics are surface/classifier-based (NLI entailment of persona attributes, FLAIR PII F1,
  token-level ASR) — no LLM re-identification adversary, precisely the overstatement channel our
  project treats as disqualifying for privacy claims.
- Sequential per-segment freezing is greedy: early deletions can strand later segments in incoherent
  contexts, and cross-segment joint leakage (combinations of quasi-identifiers) is never scored.
- ~51 s/sentence for an 8B model + reward scoring is heavy for an interactive local privacy layer.

## Co-design fitness (doc_orig→doc_p ↔ out_p→out_final)
- **(a) Conditions on / emits:** conditions on the sentence, the detected privacy segments (persona
  attributes / PII spans from its alignment detector), and a privacy-aware prompt; emits a fluent
  rewritten sentence in which each segment is either deleted or generalized. Whole-sentence rewriting,
  so edits are not guaranteed to be span-local.
- **(b) Client-side record:** weaker than INTACT's. The search tree transiently contains per-segment
  (original → delete/obscure → chosen surface form) decisions, so an **edit trace is recoverable in
  principle**, but the paper discards it, and "obscure" produces generalizations whose
  narrowing-inversion would work like INTACT's lattice — while "delete" produces *nothing to invert*:
  a deleted segment leaves no anchor in `doc_p` for a local extractor to re-attach the original content
  to in `out_p`. Whole-sentence rewrites also blur span alignment, making the reverse map fuzzier.
- **(c) Reverse/reconstruction step:** none. Text-release setting; the rewrite is the product. (Its
  "reconstruction attack" is an adversary metric, not a client capability.)
- **(d) Round-trip stress:** the delete action is the main breakage — a remote task whose answer depends
  on a deleted segment silently loses the premise, and unlike generalization the loss is unrecoverable
  downstream. The naturalness objective (ArmoRM) helps the round trip indirectly: fluent, coherent
  `doc_p` keeps the remote LLM on-task (the DP baselines at PPL 789–1108 would wreck remote-task
  quality). But nothing in the search optimizes *task-answer preservation*; ECHR judgment accuracy
  (−14 pp) is the only downstream evidence, on one binary task.
- **(e) Privacy vs adversary:** no. Privacy_NLI classifier + FLAIR PII extraction + token-level
  reconstruction ASR — automatic surface/classifier metrics only; no re-identification attacker in the
  loop or in evaluation.
- **(f) Verdict:** contributes an inference-time *search scaffold* for the substitutor — candidate
  generation + screening + reward-guided selection with no training, and evidence that a
  fluency/coherence reward keeps rewrites natural where DP mechanisms degenerate. For our co-design its
  reward slot is the interesting part: swap ArmoRM for a reward combining (i) attacker resistance and
  (ii) *local-extractor reconstructability*, and the same search optimizes the round trip directly. It
  cannot provide privacy evidence (no adversarial attacker), an extractor (no reverse step, and its
  delete action is anti-extractor), or speed (~51 s/sentence).

## Relevance to This Project
**Why surfaced:** round-2 (arXiv + Semantic Scholar) discovery — the **search-based** branch of RD4 in
[`docs/research/learned-substitution.md`](../../docs/research/learned-substitution.md). **Fit:** a
reward-guided, training-free alternative to the finetuned (DP-MLM) and RL ([AgentStealth](shao2025_agentstealth.md))
substitutors; complements [RUPTA](yang2025_rupta.md)'s evaluator-guided optimization at inference time.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
