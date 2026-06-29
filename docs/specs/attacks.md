---
type: reference
status: current
created: 2026-06-29
updated: 2026-06-29
tags: [attacks, reconstruction, embedding-inversion, masked-lm, rantext, spec]
companion: docs/specs/probes.md
---

# Attack specifications

Attacks attempt to **recover the raw document from Doc_p** (the perturbed prompt the remote
model and adversary see). The privacy protection level reported in the literature is
`1 − attack-success`. An attack's success is a **lower** bound on leakage (a probe upper-bounds
it; see `docs/specs/probes.md`): a weak attack overstates privacy, so attacks are reported with
the attacker named. Module: `src/inferdpt/attacks/` (`embedding_inversion.py`, `mask_bert.py`).

Threat model (InferDPT): honest-but-curious remote model; the adversary knows the mechanism,
the vocabulary V, the embedding map φ, and the budget ε, and only sees text (Doc_p), not vectors.

## 1. Embedding inversion — `attacks/embedding_inversion.py` (lower recovery = more private)

| attack | interface | metric (range) | read |
|---|---|---|---|
| token-level inversion | `invert(pairs, ve, ks=(1,5,10,20)) -> {"recovery@K": float}` | fraction of tokens whose raw form is among the K nearest, [0,1] | ↓ better |

### how it works
`pairs` are `(raw_word, perturbed_word)` for content positions, both in V. For each perturbed
token the attack embeds it in φ (the same cache used to perturb, per the threat model where the
adversary knows φ) and returns the top-K nearest vocabulary tokens by Euclidean distance; a hit
means the raw token is among them. It exploits that RANTEXT replaces a token with an
embedding-near token, so the original is likely in the perturbed token's neighbourhood.
`privacy@K = 1 − recovery@K`. Introduced for text sanitization in **InferDPT §VI** (embedding
inversion attack) and grounded in **Song & Raghunathan 2020** (Information Leakage in Embedding
Models).

> **vec2text is out of scope.** Sentence-level vector inversion (vec2text, Morris/Kuleshov 2023;
> transferable inversion, [arXiv:2406.10280](https://arxiv.org/abs/2406.10280)) inverts a dense
> embedding *vector*; RANTEXT releases *text*, so there is no vector on the wire. The applicable
> attack is token-level nearest-neighbour recovery above.

## 2. Masked-LM (BERT) inference — `attacks/mask_bert.py` (lower recovery = more private)

| attack | interface | metric (range) | read |
|---|---|---|---|
| top-1 recovery | `reconstruct(perturbed_words, raw_words, is_content)["top1_recovery"]` | fraction recovered exactly, [0,1] | ↓ better |
| top-5 recovery | `reconstruct(...)["top5_recovery"]` | raw token in BERT's top-5, [0,1] | ↓ better |
| posterior on truth | `reconstruct(...)["mean_posterior_true"]` | mean p(raw token) at the mask, [0,1] | ↓ better |
| rank of truth | `reconstruct(...)["mean_rank_true"]` | mean rank of the raw token in BERT's distribution, [1, \|vocab\|] | ↑ better |

### how it works
Slides `[MASK]` across each perturbed content-word position (selected by `is_content`), runs a
pretrained masked language model (`bert-base-uncased`) to predict that position from the
perturbed context, and compares against the known raw word. Binary: top-1 / top-5 recovery.
Graded (a soft leakage signal, more informative than a binary hit and connected to mutual
information): the posterior mass on the true token and its rank, computed only where the raw
word is a single BERT wordpiece. Word-level single-`[MASK]`: a raw word BERT would split into
multiple wordpieces cannot be matched by one prediction and counts as not recovered
(conservative, over-estimates privacy). Method from **Yue et al. 2021** (SanText) and
**InferDPT §VI-B**; model **Devlin et al. 2019** (BERT). The paper notes BERT under-recovers
tokens from a foreign tokenizer's vocabulary, which motivated the stronger LLM attack below.

## 3. LLM reconstruction (deferred, documented)

Not yet implemented. Planned: a strong model (Qwen3.6-35B-A3B) is prompted to reconstruct the
raw text from Doc_p; scored as content-word recovery (general) and PII entity recall (targeted).
It is the realistic worst case (context-aware, exploits a language prior) and the substrate for
the **V-information** contextual-leakage probe (with a no-context control to subtract the
attacker's parametric knowledge). References: **InferDPT** GPT inference attack;
**On the Vulnerability of Text Sanitization** ([arXiv:2410.17052](https://arxiv.org/abs/2410.17052));
context-influence control ([arXiv:2410.03026](https://arxiv.org/abs/2410.03026)). See
`docs/research/mi-probes.md`.
