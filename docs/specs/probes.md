---
type: reference
status: current
created: 2026-06-29
updated: 2026-06-29
tags: [probes, leakage, mutual-information, utility, rantext, spec]
companion: docs/research/mi-probes.md
---

# Probe specifications

Probes are **attack-independent** measurements on the perturbation or on the pipeline
outputs. They never run an attack, so a correlation between a probe and an attack is not
circular. Utility cosine probes use **SimCSE** (`princeton-nlp/sup-simcse-roberta-large`,
STS-calibrated, paper-faithful, usable dynamic range); a **cross-encoder reranker**
(`qwen3-reranker-0.6b` via `/v1/rerank`) gives a second, more discriminative utility score;
**PII probes use rapidfuzz fuzzy containment** (not cosine — cosine floors ~0.7 on bare
entities). The qwen3-embedding scorer remains available (`_common.embed`) for the token-level
geometry work. Module: `src/inferdpt/probes/` (`leakage.py`, `mi.py`, `utility.py`; shared
helpers in `_common.py`). LLM responses are disk-cached when `$INFERDPT_LLM_CACHE` is set, so
re-runs over identical prompts are instant.

Information-theoretically, a probe is meant to **upper-bound** leakage (what any adversary
could extract), whereas an attack (see `docs/specs/attacks.md`) gives a **lower** bound.

> **Not probes (mechanism diagnostics).** `candidate set |C_r|/V`, `replacement similarity
> cos(o,r)`, `anisotropy`, `relative spread`, and `sampling entropy` live in
> `src/inferdpt/diagnostics.py`. They describe the mechanism/geometry, not leakage of a
> released artifact. `cos(o,r)` in particular is dual (higher = more utility, less privacy),
> so it is a diagnostic on the tradeoff axis, not a one-directional probe.

## 1. Leakage probes — `probes/leakage.py` (lower = more private)

| probe | interface | metric (range) | read |
|---|---|---|---|
| content-word overlap | `overlap(doc, doc_p) -> float` | fraction of Doc content words appearing verbatim in Doc_p, [0,1] | ↓ better |
| self-substitution S_w | `s_w_n_w(perturber, words, ε, runs=100)["S_w"]` | P[M(w)=w], [0,1] | ↓ better |
| output support N_w | `s_w_n_w(...)["N_w"]` | distinct outputs over `runs`, [1, runs] | ↑ better |
| PII leakage (containment) | `pii_leakage(doc, doc_p, threshold=85) -> {degree, recall, n}` | mean / fraction≥threshold of rapidfuzz best-ratio of each raw PII span in Doc_p, [0,1] | ↓ better |

### content-word overlap
Tokenises Doc and Doc_p to lowercase alphabetic content words (length > 2, minus a small
stopword list) and reports `|cw(Doc) ∩ cw(Doc_p)| / |cw(Doc)|`. A verbatim-survival measure;
the cheapest leakage signal. Paper analogue: InferDPT's n-gram token-leakage check (Table VI).

### self-substitution S_w / output support N_w
The plausible-deniability statistics. For each word, run the mechanism `runs` times (fresh
randomness): `S_w` = fraction of runs where the output equals the input (probability a word is
left unchanged); `N_w` = number of distinct outputs observed. Lower S_w and higher N_w mean an
adversary cannot assume the word was preserved or narrow it to few options. Introduced by
**Feyisetan et al. 2020** (Privacy- and Utility-Preserving Textual Analysis via Calibrated
Multivariate Perturbations); standardised in the word-level metric-DP benchmark
([arXiv:2404.03324](https://arxiv.org/abs/2404.03324)).

### PII leakage (containment)
Presidio detects PII spans in Doc (PERSON, LOCATION, ORGANIZATION, NRP, DATE_TIME,
EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, IBAN_CODE, US_SSN). For each entity, the probe
takes the best **rapidfuzz** ratio (normalized exact substring, else max of `partial_ratio`
and `token_set_ratio`) against Doc_p, and reports the mean over entities (`degree`) and the
fraction at or above `threshold`=85 (`recall`). This is the record-linkage containment test;
it avoids the embedding cosine floor (~0.7 on bare entities) and handles multi-word/numeric
spans. Lower means the entity does not survive into the perturbed prompt. Detector: **Microsoft
Presidio**; fuzzy match: **rapidfuzz** (Jaro/Levenshtein family). (`pii_semantic_leakage` is a
back-compat alias.) The paraphrase tail (entity reworded, not verbatim) is out of fuzzy scope
and deferred to a short-span semantic scorer (Phrase-BERT; see the Lever-2 handoff).
PII-leakage framing: **Lukas et al., IEEE S&P 2023** (Analyzing Leakage of PII in LMs).

## 2. Mutual-information probes — `probes/mi.py` (lower = more private, bits)

| probe | interface | metric (range) | read |
|---|---|---|---|
| per-token channel MI | `token_channel_mi(perturber, words, ε, runs=200)["mi_bits"]` | I(token;replacement), [0, log₂\|V\|] bits | ↓ better |
| per-token leakage (KL) | `token_channel_mi(...)["mean_token_leakage_bits" / "max_token_leakage_bits"]` | KL(p(y\|x) ‖ p(y)) per token, bits | ↓ better |
| n-gram MI | `ngram_mi(perturber, docs, ε, n=1)["mi_bits"]` | empirical I(raw n-gram; perturbed n-gram), bits | ↓ better |

### per-token channel MI
The exact leakage of the per-token channel. The conditional `p(y|x)` is known analytically
(the exponential-mechanism distribution marginalised over the Laplace radius); it is estimated
by soft-averaging that conditional over `runs` noise draws (low variance, no hard sampling).
With a prior `p(x)` (uniform by default), `I(X;Y) = H(Y) − H(Y|X)` in bits. By Fano's
inequality this upper-bounds any adversary's per-token recovery. Foundations: mutual
information as a leakage measure, **Cuff & Yu, CCS 2016** (Differential Privacy as a Mutual
Information Constraint); plug-in MI of a known discrete channel is textbook (**Cover & Thomas**).
The exponential mechanism is **McSherry & Talwar 2007**.

### per-token leakage (pointwise KL)
The per-token contribution `KL(p(y|x) ‖ p(y))`; identifies which tokens leak most (and can be
correlated with PII). `mean_token_leakage_bits` equals the MI under a uniform prior;
`max_token_leakage_bits` flags the worst token.

### n-gram MI
Empirical plug-in MI (bits) with **Miller–Madow** bias correction (**Miller 1955**;
**Paninski 2003**) between aligned raw and perturbed n-grams over a corpus. `n=1` ≈ empirical
token MI; `n≥2` is a local-context proxy. Requires many tokens for `n≥2` (sparsity); inflated
on small corpora. Related: pairwise MI in masked sequence models
([arXiv:2605.20187](https://arxiv.org/abs/2605.20187)).

> **Context-aware MI is deferred.** The per-token channel MI misses cross-token leakage
> (`I(xᵢ; Y₁:ₙ) ≥ I(xᵢ; Yᵢ)`). The realistic measure is usable / V-information via an LLM
> reconstruction attacker with a no-context control; documented in `docs/research/mi-probes.md`,
> not yet implemented.

## 3. Utility probes — `probes/utility.py` (higher = more faithful)

| probe | interface | metric (range) | read |
|---|---|---|---|
| utility | `utility(doc, output) -> float` | SimCSE cos(Doc, output), [−1,1] | ↑ better |
| utility_control | `utility_control(control_out, output) -> float` | SimCSE cos(non-private gen, output), [−1,1] | ↑ better |
| utility (reranker) | `utility_rerank(doc, output) -> float` | cross-encoder relevance(Doc, output), [0,1] | ↑ better |
| utility_control (reranker) | `utility_control_rerank(control_out, output) -> float` | relevance(non-private gen, output), [0,1] | ↑ better |
| PII reconstruction recall | `pii_reconstruction_recall(doc, output, threshold=85) -> {degree, recall, n}` | rapidfuzz recovery of raw PII spans in the output | ↑ better |

### utility / utility_control (SimCSE)
SimCSE cosine between the reference (Doc, or the non-private generation) and the final output.
SimCSE is contrastively trained for STS, giving cosine a usable dynamic range (it fixes the
anisotropy that compresses generic embedders). Analogue of InferDPT's **coherence** metric, a
SimCSE cosine (**Gao et al. 2021**, SimCSE; InferDPT Table VII). Both are weakly discriminating
across ε because the local extraction model re-grounds on the true Doc; `utility_control`
(fidelity to the non-private generation) is the cleaner of the two.

### utility / utility_control (reranker)
Cross-encoder relevance score (`qwen3-reranker-0.6b`) of the output to the reference — jointly
attends to the pair, more discriminative than bi-encoder cosine on the present/absent margin,
but **saturates near 1.0** on the pipeline outputs (they are faithful to Doc at every ε), which
is itself the evidence that output utility is set by extraction, not by ε.

`utility_control` is the analogue of InferDPT **Table X** (cosine between the final generation
and the non-private GPT-4 output).

### PII reconstruction recall
The benign twin of PII leakage: the same Presidio + **rapidfuzz containment** primitive, applied
to the **final (local, trusted) output** instead of Doc_p. Higher means the legitimate pipeline
reproduced the document's true entities (desirable, since the output never leaves the client).
On our pipeline it sits at ~1.0 across ε (extraction re-grounds on Doc), confirming it is a
fidelity metric, not an ε signal. Same detector/refs as PII leakage.
