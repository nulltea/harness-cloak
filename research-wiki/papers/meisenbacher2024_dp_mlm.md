---
type: paper
node_id: paper:meisenbacher2024_dp_mlm
title: "DP-MLM: Differentially Private Text Rewriting Using Masked Language Models"
authors: ["Stephen Meisenbacher", "Maulik Chevli", "Juraj Vladika", "Florian Matthes"]
year: 2024
venue: "ACL 2024 Findings"
external_ids:
  arxiv: "2407.00637"
  doi: null
  s2: null
tags: ["dp-text-rewriting", "masked-language-model", "learned-substitution", "local-dp", "rd4"]
added: 2026-07-01T00:00:00Z
---

# DP-MLM: Differentially Private Text Rewriting Using Masked Language Models

## One-line thesis
Rewrite text under DP one token at a time by temperature-sampling substitutions from an encoder-only
MLM's contextualized logits; MLMs give better utility at lower ε than decoder-based rewriters.

## Method
Per-token exponential mechanism over MLM logits, realized as temperature sampling
(verified against the released code, `sjmeis/dpmlm`, `src/dpmlm/core.py`):

- **Conditioning.** For token *w_l*, the MLM input is the **full original sentence, `[SEP]`, and a copy
  with *w_l* replaced by `<mask>`** (the `CONCAT=True` contextualization, borrowed from Qiang et al.'s
  lexical simplification). A sliding window caps context at half the model's max length. In the default
  `dpmlm_rewrite` loop (`REPLACE=False`) every token is privatized against the *original* context, not
  the partially-privatized one; a `REPLACE=True` variant conditions on the privatized prefix.
- **Sampling = exponential mechanism.** Mask-position logits are clipped to `(C_min, C_max) = (μ, μ+4σ)`,
  estimated from 1000 SST2 examples (code defaults for roberta-base: `clip_min=-3.209`,
  `clip_max=16.305`, so sensitivity Δu = |C_max−C_min| ≈ 19.51). Sampling with temperature
  **T = 2Δu/ε** — in code, `logits / (2·sensitivity/ε)` then softmax + categorical draw — is exactly the
  exponential mechanism with utility = logit and per-token budget ε.
- **Composition.** One token costs ε; an *n*-token text costs **n·ε** by sequential composition. Output is
  strictly length-preserving and 1:1 token-aligned (capitalization of the original is re-applied).
- **Knobs.** ε per token (the code accepts a **per-token ε list** — a direct hook for budget-allocation
  schemes); `CONCAT`; `STOP` (by default English stopwords and punctuation are passed through *in the
  clear*); `REPLACE`; base MLM (paper: roberta-base, 125M). The `dpmlm_rewrite_plus` variant adds random
  insertion/deletion (`ADD_PROB=0.15`, `DEL_PROB=0.05`) for variable-length output at worst-case **2nε**.
  The repo (newer than the paper) additionally offers PII/IPI entity masks (Presidio / NER pipeline), a
  `hybrid` mode giving detected entities a separate `hybrid_budget`, and a batched GPU path
  (`privatize_batch`, all masks scored in parallel since conditioning is on the original context).

## Key Results
- **Setup:** ε ∈ {10, 25, 50, 100, 250} *per token*; utility on 9 GLUE tasks; baselines DP-Paraphrase
  (fine-tuned GPT-2) and DP-Prompt (flan-t5-base); utility/attacker classifier deberta-v3-base.
- **Utility:** SST2 accuracy 68.50 at ε=10 → 86.05 at ε=250; CoLA 69.13 → 70.18; MRPC 71.32 → 76.39.
  At ε=10 on SST2: DP-MLM 68.50 vs DP-Paraphrase 58.60 vs DP-Prompt 50.92. DP-MLM has the highest
  accuracy in **14 of 20** comparative settings.
- **Empirical privacy** (adversarial attribute inference, F1, lower better): Trustpilot gender (36k
  reviews), adaptive attacker at ε=10: 58.50 vs 69.60 baseline. Yelp authorship (2500 reviews), adaptive,
  ε=10: 62.40 vs 87.20 baseline — the adaptive attacker recovers a lot.
- **Efficiency:** ~797 tokens/min over 1,048,231 tokens (1316 min), on par with DP-Paraphrase, faster
  than DP-Prompt (535 tokens/min).

## Limitations / Failure Modes
Paper's own admissions: results tied to one representative model per mechanism (other BERT-family models
out of scope); the base mechanism cannot change output length without extra budget (variable-length
variant costs up to 2nε); empirical privacy "is a proxy" — protection is claimed "empirically and by
proxy," not against a stronger adversary; the adaptive-attacker results expose the utility↔privacy
trade-off directly.

Our analysis: per-token ε of 10–250 composes to an enormous, effectively vacuous document-level ε (nε
over hundreds of tokens); stopwords/punctuation pass through unperturbed, leaking structure and some
content; and per-token contextual rewriting stays inside the exponential-mechanism frame — the same MLM
family is a known reconstruction/MI attacker (cuts both ways).

## Co-design fitness (doc_orig→doc_p ↔ out_p→out_final)
- **(a) Conditions/emits:** conditions on the full original sentence plus a masked copy (window-capped);
  emits exactly one vocab token per input word — 1:1, position-aligned, length-preserving.
- **(b) Client-side record:** no explicit reverse map is stored, but the 1:1 alignment makes the span map
  trivial — the client holds `(position, w_orig, w_sub)` for free. That is the strongest possible
  conditioning signal an extractor could ask for from a substitutor.
- **(c) Reverse step:** none. Text release is the end product; the paper never un-perturbs anything.
- **(d) Round trip:** a remote LLM computing a task on `doc_p` sees token-level swaps that break entity
  consistency and coreference (each occurrence of a name is independently resampled); and `out_p` is free
  text with no positional correspondence to `doc_p`, so the trivial alignment that helps a *text-release*
  reading does not transfer to output extraction — the extractor would need the `(w_orig, w_sub)` table
  as a dictionary, not the positions.
- **(e) Adversarial privacy eval:** yes, empirical — trained static and adaptive attribute-inference
  classifiers — plus the formal per-token guarantee; but no LLM re-identification attacker.
- **(f) Verdict:** the best-understood learned-substitution baseline for our substitutor arm: local,
  small (125M), batched, with a per-token-ε hook that composes directly with salience budgeting
  (see [meisenbacher2025_spend_budget_wisely](meisenbacher2025_spend_budget_wisely.md)). It contributes a
  mechanism and a reusable codebase, not an extractor: at usable per-token ε the document-level guarantee
  is vacuous, entity-inconsistent swaps degrade remote-task utility, and it offers no path from `out_p`
  back to `out_final` beyond the client's own substitution table.

## Relevance to This Project
**Why surfaced:** the most on-point anchor for **RD4** in
[`docs/research/beyond-rantext.md`](../../docs/research/beyond-rantext.md): substitution *is* the MLM
pretraining objective, and this shows an encoder-only MLM finetune beats decoder paraphrasing at fixed
ε — evidence for RD4's "finetune existing MLM/infilling models, don't train from scratch" thesis and
for RD5's "denoising is the right architecture" (E2). **Relevance:** by the same group as the
double-edged-reconstruction work, so it also grounds the RD4 "cut-both-ways" warning.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
- Same author's [DP-ST](meisenbacher2025_dp_st.md) uses DP-MLM as a baseline (and beats it on coherence
  at document-level ε); [Spend Your Budget Wisely](meisenbacher2025_spend_budget_wisely.md) supplies the
  per-token ε vectors DP-MLM's code accepts.

## Abstract (original)

> The task of text privatization using Differential Privacy has recently taken the form of text
> rewriting, in which an input text is obfuscated via the use of generative (large) language models.
> While these methods have shown promising results in the ability to preserve privacy, these methods
> rely on autoregressive models which lack a mechanism to contextualize the private rewriting process.
> In response to this, we propose DP-MLM, a new method for differentially private text rewriting based
> on leveraging masked language models (MLMs) to rewrite text in a semantically similar and obfuscated
> manner. We accomplish this with a simple contextualization technique whereby we rewrite a text one
> token at a time. We find that utilizing encoder-only MLMs provides better utility preservation at
> lower epsilon levels, as compared to previous methods relying on larger models with a decoder. In
> addition, MLMs allow for greater customization of the rewriting mechanism.
