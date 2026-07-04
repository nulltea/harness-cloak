---
type: reference
status: current
created: 2026-06-29
updated: 2026-07-03
tags: [attacks, reconstruction, embedding-inversion, masked-lm, rantext, guess-back,
       span-inversion, probe-shootout, spec]
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

## 3. Span-inversion guess-back probes (substitution mechanism) — shootout 2026-07-03

Candidate-level probes for the lattice substitutor: given a doc_p sentence with a fill in the
slot, how likely does an attacker recover the original span? Used as the τ-walk gate and the RL
reward's privacy term — **never as reported privacy** (that stays the LLM attacker; a probe is
promoted only by measured correlation with one — the fork-1 correlation rule).

**Shootout protocol** (`scripts/spikes/privacy_probe_shootout.py`, results
`results/privacy_probe_shootout.json`): 150 (span, lattice-level) items from the arms artifact
(clinical + enron); referee = LLM guess-back on the deployed sentence (mechanism disclosed,
5 guesses, fuzzy hit ≥ 85). Referee available so far: **local Qwen3.6-35B-A3B** (hit@1 0.127,
hit@5 0.200; caveat: it is also the pipeline's generation model, and only ~30 positive labels —
an independent frontier referee redo is pending authorization; probe scores are
referee-independent and re-scorable via `scripts/spikes/probe_shootout_rescore.py`).

| probe | mechanism | AUC hit@1 | AUC hit@5 | per-span level-ordering | verdict |
|---|---|---|---|---|---|
| MTI mask-away (`cloak/probe.py` legacy) | mask the fill, predict original from context alone | — | — | — | **candidate-INVARIANT** (identical score for every level of a span; degenerate τ-walk, zero RL gradient) — disqualified before the shootout |
| P2 appositive MLM | roberta-base; slot masked, fill kept visible as appositive; max top-50 prob over distinctive tokens | 0.477 | 0.571 | 0.50 | **failed** — chance-level vs the attacker; single-token limit + unnatural syntax |
| P3 multi-mask PLL | roberta-base; k masks in the slot next to the visible fill; mean P of original tokens | 0.691 | 0.683 | 0.786 | viable |
| P4 contrastive re-id | pythia-410m (fp16, local); softmax over {original} ∪ ≤15 same-type corpus distractors of length-normalized logP(candidate \| sentence + disclosure suffix) | 0.713 | 0.637 | **0.857** | best level-ordering; anonymity-set semantics; ~50 ms/item, precomputable per (span, level) |
| P6 embedding sim | MiniLM cos(fill, original) | **0.822** | **0.827** | 0.786 | best AUC; context-blind, and gameable by a trainable infiller (stage 2) — reward-internal use only, never a reported measure |

**Combination (measured, same labels):** rank-mean(P4, P6) AUC 0.801, leave-one-out logistic
0.823 — no gain over P6 alone (0.827); level-ordering of the blend 0.714 — *worse than either*.
Naive blending averages away each probe's strength; if both jobs matter, assign probes to jobs
(P4-style for per-level walk ordering, P6-style for doc-level discrimination) rather than mixing
scores.

**Second referee (frontier): gemini-3.1-pro-preview, 2026-07-03** — hit@1 0.200 / hit@5 0.313
overall; **hit@5 0.490 among the 96 parsed replies** (54/150 empty after 429/503 retry
exhaustion; scored as misses — parsed-only AUCs shift ≤ 0.03, ranking unchanged). AUC@5
all/parsed-only: P2 0.571/0.569, P3 0.663/0.675, P4 0.601/0.631, P6 **0.760/0.768**;
level-ordering: P2 0.333, P3 0.571, P4 **0.714**, P6 0.643; combo AUC 0.727/0.747 (again below
P6 alone). **Cross-referee verdict (stable under both):** P2 disqualified; P6 best
discrimination; P4 best within-span level-ordering; blending never wins. Attacker-severity
note: hit@5 ≈ 0.5 on engaged fills at τ-walk operating points — the dominant recovery channel
in the examples is *retained context* (undetected sibling mentions, e.g. "gastroenterologist"
left cleartext next to a substituted "gastroenterology"), i.e. detection/coref recall, upstream
of level selection.

## 4. LLM reconstruction (deferred, documented)

Not yet implemented. Planned: a strong model (Qwen3.6-35B-A3B) is prompted to reconstruct the
raw text from Doc_p; scored as content-word recovery (general) and PII entity recall (targeted).
It is the realistic worst case (context-aware, exploits a language prior) and the substrate for
the **V-information** contextual-leakage probe (with a no-context control to subtract the
attacker's parametric knowledge). References: **InferDPT** GPT inference attack;
**On the Vulnerability of Text Sanitization** ([arXiv:2410.17052](https://arxiv.org/abs/2410.17052));
context-influence control ([arXiv:2410.03026](https://arxiv.org/abs/2410.03026)). See
`docs/research/mi-probes.md`.
