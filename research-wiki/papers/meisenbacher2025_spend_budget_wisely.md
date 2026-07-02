---
type: paper
node_id: paper:meisenbacher2025_spend_budget_wisely
title: "Spend Your Budget Wisely: Towards an Intelligent Distribution of the Privacy Budget in Differentially Private Text Rewriting"
authors: ["Stephen Meisenbacher", "Chaeeun Joy Lee", "Florian Matthes"]
year: 2025
venue: "CODASPY 2025"
external_ids:
  arxiv: "2503.22379"
  doi: null
  s2: null
tags: ["dp-text-rewriting", "budget-allocation", "salience", "rd1", "rd4"]
added: 2026-07-02T00:00:00Z
---

# Spend Your Budget Wisely — intelligent ε distribution in DP text rewriting

## One-line thesis
Not all tokens are equally sensitive, so distribute the privacy budget ε per-token by
linguistic/NLP salience rather than uniformly — the first work to do so for DP text rewriting.

## Method
A **budget allocator, not a mechanism**: five token-sensitivity scorers, each normalized to [0,1] and
averaged into one sensitivity s_i per token; per-token budgets are set **inversely proportional to
sensitivity** (more sensitive → less ε → more perturbation), rescaled so Σε_i equals the total document
budget (composition preserved).

Scorers:
- **Information Content (IC):** corpus IC from nltk (semcor/brown/bnc/shaks/treebank); nouns/verbs only,
  everything else scored 1.
- **POS weights:** {NN:14, PR:7, VB:15, CD:2, JJ:5, RB:5}, other tags 0.1.
- **NER (spaCy):** binary — named-entity tokens 1, rest 0.
- **Word Importance (WI):** embedding-similarity drop when the word is removed (gte-small).
- **Sentence Difference (SD):** original vs word-deleted sentence embedding difference (gte-small).

Applied to two carrier mechanisms: **1-Diffractor** (word-level metric DP, ε ∈ {0.1, 0.5, 1.0} per word)
and **DP-BART** (document-level (ε,δ)-DP, ε ∈ {500, 1000, 1500} per document, budget distributed over
sentences). All scorers are training-free and local.

## Key Results
Numbers from arXiv HTML v1 — the picture is **mixed**, not a uniform win:

- **Datasets:** privacy — Yelp (17,295 reviews, 10-author ID), Trustpilot (29,490, gender), Blog
  (15,070, 10-author ID); utility — GLUE (SST-2, MRPC, MNLI), BBC News (3147, 5-class), DocNLI (9136),
  IMDb (50k).
- **Privacy eval:** (i) empirical attribute inference, static + adaptive attackers (F1, lower better),
  with a relative-gain score γ; (ii) membership-style inference: Masked Token Inference (MTI_seq /
  MTI_bow — an MLM predicting original tokens from privatized context) and Nearest-Neighbor rank of the
  privatized doc against the original corpus.
- **Where distribution helps:** Blog + DP-BART ε=500, adaptive attacker F1 13.61→**8.95** (~34% better);
  NN rank on Yelp + DP-BART ε=500: 816→**964** (~18% better).
- **Where it doesn't:** Yelp + 1-Diffractor ε=0.5 adaptive F1 unchanged (88.44±1.4 both); MTI_bow on the
  same setting 0.122→0.120 (marginal).
- **Utility cost:** SST-2 + 1-Diffractor ε=0.5: 87.21→85.31 F1; BBC + DP-BART ε=500: 40.53→**32.17**
  (−20.5%). The paper concedes distribution "nearly always leads to lower utility and lower text
  coherence."

## Limitations / Failure Modes
Paper's own admissions: budget distribution nearly always costs utility and coherence; effectiveness
differs sharply between word-level (1-Diffractor: modest empirical-privacy gains, better MIA resistance)
and document-level (DP-BART: clearer privacy gains, severe utility loss on long texts); "budget
distribution is not as clear-cut with document-level DP mechanisms"; authors call for smarter allocation
than the equal-split default they improve on.

Our analysis: allocation shifts distortion onto exactly the salient tokens (entities, content nouns/verbs)
that downstream tasks — and any extractor — most need; the averaged five-scorer sensitivity is a fixed
heuristic, not learned, and its scores are computed on the raw text (fine locally, but the score vector
itself is sensitive metadata). Privacy remains empirical classifier-based, no LLM re-identification.

## Co-design fitness (doc_orig→doc_p ↔ out_p→out_final)
- **(a) Conditions/emits:** conditions on per-token salience (IC/POS/NER/WI/SD averaged); emits nothing
  itself — it hands a per-token ε vector to an underlying mechanism (word swaps for 1-Diffractor,
  latent-noise rewriting for DP-BART).
- **(b) Client-side record:** yes — the salience scores and the per-token ε vector are client-side
  artifacts that say *where* distortion was concentrated. Fed to an extractor, that is a per-position
  noise-level prior; combined with a 1:1 mechanism like DP-MLM it becomes a weighted span map.
- **(c) Reverse step:** none, and none in the carrier mechanisms; text release is the product.
- **(d) Round trip:** double-edged. Concentrating perturbation on identifying tokens is exactly what a
  round trip wants (protect identity, keep task-carrying context) — but the measured utility cost lands
  on task performance too (BBC −20.5%), and for a remote *task* the heavily-perturbed salient entities are
  often the very things the answer must mention, pushing the recovery burden onto the extractor.
- **(e) Adversarial privacy eval:** yes, empirical — static/adaptive attribute-inference plus MTI/NN
  membership-style attacks (MTI is notable: it *is* an MLM-reconstruction attacker, a weak cousin of our
  intended re-identification adversary). No LLM re-identification.
- **(f) Verdict:** the empirical anchor for RD1-inside-RD4 — decomposed, salience-aware ε is feasible and
  measurably better on privacy in some settings — and its scorer toolkit is cheap, local, training-free,
  and plugs directly into DP-MLM's per-token ε API. What it cannot provide: the gains are
  mechanism-dependent and bought with utility, which supports our thesis that budget shaping alone does
  not fix the trade-off — the substitution itself must be learned, and the un-perturb step must exist.

## Relevance to This Project
**Why surfaced:** this is **RD1 (role/salience budgeting) realized inside RD4** — see
[`docs/research/beyond-rantext.md`](../../docs/research/beyond-rantext.md) and
[`learned-substitution.md`](../../docs/research/learned-substitution.md). **Fit:** direct empirical
support for the taxonomy's core prescription — decompose the scalar ε by token salience (fixes F1b) —
demonstrated within a learned-rewriting system. (Post-read caveat: support is qualified — privacy gains
are mechanism-dependent and cost utility; see Key Results.)

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
- Composes with [DP-MLM](meisenbacher2024_dp_mlm.md), whose released code accepts per-token ε lists;
  challenges the equal-per-triple split used by [DP-ST](meisenbacher2025_dp_st.md).

## Abstract (original)

> The task of Differentially Private Text Rewriting is a class of text privatization techniques in which
> (sensitive) input textual documents are rewritten under Differential Privacy (DP) guarantees. The
> motivation behind such methods is to hide both explicit and implicit identifiers that could be
> contained in text, while still retaining the semantic meaning of the original text, thus preserving
> utility. Recent years have seen an uptick in research output in this field, offering a diverse array of
> word-, sentence-, and document-level DP rewriting methods. Common to these methods is the selection of a
> privacy budget (i.e., the ε parameter), which governs the degree to which a text is privatized. One
> major limitation of previous works, stemming directly from the unique structure of language itself, is
> the lack of consideration of where the privacy budget should be allocated, as not all aspects of
> language, and therefore text, are equally sensitive or personal. In this work, we are the first to
> address this shortcoming, asking the question of how a given privacy budget can be intelligently and
> sensibly distributed amongst a target document. We construct and evaluate a toolkit of linguistics- and
> NLP-based methods used to allocate a privacy budget to constituent tokens in a text document. In a
> series of privacy and utility experiments, we empirically demonstrate that given the same privacy
> budget, intelligent distribution leads to higher privacy levels and more positive trade-offs than a
> naive distribution of ε.
