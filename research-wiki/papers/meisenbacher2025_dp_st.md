---
type: paper
node_id: paper:meisenbacher2025_dp_st
title: "Leveraging Semantic Triples for Private Document Generation with Local Differential Privacy Guarantees"
authors: ["Stephen Meisenbacher", "Maulik Chevli", "Florian Matthes"]
year: 2025
venue: "EMNLP 2025"
external_ids:
  arxiv: "2508.20736"
  doi: null
  s2: null
tags: ["dp-text-rewriting", "semantic-triples", "llm-reconstruction", "local-dp", "rd4", "rd5"]
added: 2026-07-02T00:00:00Z
---

# DP-ST: semantic-triple private document generation under local DP

## One-line thesis
Decompose a document into semantic triples, privatize each within a *privatization neighborhood* under
local DP, then use LLM post-processing to reconstruct coherent text — buying coherence at lower ε via
divide-and-conquer.

## Method
Four-stage pipeline (details from arXiv HTML v1):

1. **Triple extraction:** Stanford OpenIE on the input document; redundant extractions deduplicated with
   MinHash LSH (threshold 0.4, 128 permutations), then per-bucket the lowest-GPT-2-perplexity triple wins.
2. **Privatization neighborhood:** input triples embedded with jina-embeddings-v3 truncated to
   **32-d Matryoshka embeddings**; nearest-centroid lookup against k ∈ {50k, 100k, 200k} clusters built
   over a **public corpus of ~15M triples**. The matched cluster *is* the neighborhood — the DP notion is
   relaxed to indistinguishability *within that neighborhood*, not over all texts.
3. **Exponential mechanism per triple:** utility score = cosine similarity between the input-triple
   embedding and each public triple in the neighborhood; sensitivity 1 (cosine bounded in [0,1]); the
   document budget is split **equally across extracted triples**. Output triples are drawn from the
   *public* corpus, never from the input.
4. **LLM post-processing (local, DP-free):** Llama-3.2-**1B**/**3B**-Instruct generates "a concise text
   for the given set of triples" — coherence via post-processing immunity, zero extra ε.

Knobs: k (cluster count), per-word base ε ∈ {0.1, 0.5, 1.0} (scaled by document length to a
document-level budget), LLM size (1B/3B), LSH threshold/permutations, embedding dim.

## Key Results
- **Datasets:** Reuters (2500 docs, avg 575 words), Spooky Authors (19,579), Reddit Mental Health (2393),
  Trustpilot (29,490), Yelp (17,295). Per-word ε=0.5 corresponds to document-level ε ≈ 287.5 on Reuters,
  29.5 on Trustpilot, etc.
- **Privacy eval:** trained adversarial classifier (deberta-v3-base) inferring protected attributes
  (author ID, gender), static and adaptive; utility: G-Eval coherence (0–1), cosine similarity,
  sentiment F1 (Trustpilot/Yelp); "relative gain" balances utility loss vs privacy gain.
- **Representative row (Reuters, ε=0.5):** DP-ST (3B, 100k): G-Eval 0.360, cosine 0.578, static
  adversary F1 4.78 (unprivatized baseline 12.35), relative gain 0.129. Baselines: TEM G-Eval 0.103,
  DP-BART-Large 0.074, **DP-MLM 0.056** (word-salad regime), DP-Prompt-Large 0.431 / gain 0.457. So
  DP-ST dominates the token/document DP rewriters on coherence at this ε, while DP-Prompt (a much
  weaker/looser mechanism) still leads on relative gain in this row.
- Headline claim: divide-and-conquer + neighborhood DP + LLM post-processing yields **coherent output at
  ε values where whole-document rewriting collapses**; coherence is the binding constraint.

## Limitations / Failure Modes
Paper's own admissions: triple extraction "discards nuanced information about the writing style,
important attributes, or modifiers outside of the triples" — output is a distilled text; OpenIE "does not
represent the SOTA" in extraction; the 15M-triple public corpus caps expressiveness; and **when no triples
are extracted the input is returned unmodified** — a privacy hole on short texts.

Our analysis: the "neighborhood" DP notion is a substantive relaxation — the guarantee is only against
distinguishing inputs that map to the same cluster, and the cluster choice itself depends on the
(unprivatized) input embedding; comparisons to full-LDP baselines at the "same ε" are therefore not
guarantees of the same strength. Privacy is empirical attribute inference, not LLM re-identification.

## Co-design fitness (doc_orig→doc_p ↔ out_p→out_final)
- **(a) Conditions/emits:** conditions on per-triple 32-d embeddings and a corpus-derived neighborhood;
  emits *public-corpus* triples, then a fluent LLM-written document. Not length-preserving, not
  token-aligned; everything outside triples is dropped.
- **(b) Client-side record:** rich. The client holds the original triples, the chosen neighborhood, and
  the **triple→privatized-triple substitution table**. That table is effectively a reverse dictionary
  (which public entity/relation replaced which original one) — a natural conditioning signal for an
  `out_p→out_final` extractor doing entity re-substitution.
- **(c) Reverse step:** the LLM step is **local**, but it is *forward* generation (privatized triples →
  released text), not un-perturbation toward the original. Nothing maps output text back to `doc_orig`.
- **(d) Round trip:** structurally the closest published match to our perturb→remote→extract shape —
  a remote LLM could compute a task on the triple-derived `doc_p`, and the client's triple↔triple map
  could drive re-substitution in `out_p`. But any task depending on content outside the triples (style,
  modifiers, quantities OpenIE misses) is unrecoverable by design, and the no-triples fallback ships the
  raw input. The 15M-corpus vocabulary also constrains what the remote model can be told.
- **(e) Adversarial privacy eval:** yes, empirical — static + adaptive deberta-v3 attribute-inference —
  no LLM re-identification attacker.
- **(f) Verdict:** contributes the *pattern* we want — privatize a structured intermediate locally, let a
  local model restore fluency for free under post-processing, keep a client-side substitution table — and
  demonstrates it beats token-level DP rewriting on coherence at fixed ε. It cannot provide the extractor:
  its local LLM never inverts anything, its guarantee is neighborhood-relaxed, and its lossy
  triple bottleneck caps round-trip utility for tasks that need more than propositional content.

## Relevance to This Project
**Why surfaced:** a **2025 SOTA** RD4 method that *also* implements the RD5 reverse step — see
[`docs/research/learned-substitution.md`](../../docs/research/learned-substitution.md). **Fit:** the
"privatize a structured intermediate, then LLM-reconstruct" pattern is the same shape as InferDPT's
perturb→extract; it keeps a (relaxed, neighborhood) guarantee while getting coherence — one of the two
current frontiers for RD4. (Caveat sharpened after full read: the LLM step is forward coherence
generation, not a true RD5-style reverse/extract step — see Co-design fitness.)

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
- Benchmarks against and outperforms [DP-MLM](meisenbacher2024_dp_mlm.md) (same group) on coherence at
  document-level ε; complements [Spend Your Budget Wisely](meisenbacher2025_spend_budget_wisely.md) —
  DP-ST splits budget equally per triple, exactly the naive allocation that paper challenges.

## Abstract (original)

> Many works at the intersection of Differential Privacy (DP) in Natural Language Processing aim to
> protect privacy by transforming texts under DP guarantees. This can be performed in a variety of ways,
> from word perturbations to full document rewriting, and most often under local DP. Here, an input text
> must be made indistinguishable from any other potential text, within some bound governed by the privacy
> parameter ε. Such a guarantee is quite demanding, and recent works show that privatizing texts under
> local DP can only be done reasonably under very high ε values. Addressing this challenge, we introduce
> DP-ST, which leverages semantic triples for neighborhood-aware private document generation under local
> DP guarantees. Through the evaluation of our method, we demonstrate the effectiveness of the
> divide-and-conquer paradigm, particularly when limiting the DP notion (and privacy guarantees) to that
> of a privatization neighborhood. When combined with LLM post-processing, our method allows for coherent
> text generation even at lower ε values, while still balancing privacy and utility. These findings
> highlight the importance of coherence in achieving balanced privatization outputs at reasonable ε levels.
