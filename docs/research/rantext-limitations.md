---
type: research
status: current
created: 2026-06-30
updated: 2026-06-30
tags: [rantext, inferdpt, privacy, taxonomy, embedding-map, extraction, threat-model]
companion: [rantext.md, embedding-map.md]
---

# RANTEXT / InferDPT — failure taxonomy

A formalized categorization of where RANTEXT (InferDPT's per-token ε-LDP perturbation) leaks,
breaks utility, or is measured wrong. Each failure carries an **empirical receipt** from this
repo's own runs (see `embedding-map.md`, `rantext.md`, `research-wiki/`) or the external
literature. This is a research-spike artifact, not a fix plan — design implications are listed
last and deliberately not pursued here.

## Definitions

- **φ (embedding map):** token → ℝ^N vector used by the mechanism to measure replacement distance.
- **ε-LDP (local differential privacy):** per-token randomization bound; here realized by the
  exponential mechanism's ε/2 score factor, *not* by the noise radius.
- **V:** the perturbation vocabulary (candidate replacements). Either cl100k BPE sub-word pieces
  (default `tokeniser.py`) or a whole-word sub-vocab (experiments) — the distinction matters for F3b.
- **C_r (random adjacency list):** candidate set `{v ∈ V : ‖φ(t) − φ(v)‖ < r}` for a drawn radius r.
- **Z(ε):** scale that sets the noise radius and hence `|C_r|`; an operating-point knob, not the ε-LDP source.
- **Doc / Doc_p:** raw document / perturbed document sent to the remote black-box LLM.
- **Gen_p / Gen:** remote completion of Doc_p / final locally-extracted output.
- **Extraction module (g):** local model that takes raw `Doc` + `Gen_p` → `Gen`.
- **inv@K:** fraction of raw tokens recovered by nearest-neighbour search in φ (an attack metric).
- **S_w:** self-substitution probability (token left unchanged). **syn_prec@K:** fraction of φ
  k-NN that are WordNet synonyms/hypernyms (replacement-quality proxy).
- **Anisotropy:** mean pairwise cosine of φ; high anisotropy concentrates distances.
- **Referential vs lexical identity:** whether the *referent* (the person/fact) survives a swap
  (Sam→Samuel keeps the referent) vs whether the surface word is a dictionary synonym.

## Frame

The root assumptions **A1–A3** are realized by concrete mechanism properties; the failures are
those assumptions breaking against measured outcomes.

| Root assumption | Mechanism property that encodes it | Breaks as |
|---|---|---|
| **A1** token independence — privacy(doc) = ∏ per-token perturbations | `Doc_p` = concatenation of i.i.d. per-token outputs | F1, F4 |
| **A2** geometric adequacy — Euclidean distance in φ ≈ "safe & meaning-preserving to swap" | replacement = Euclidean proximity in φ | F1b, F2 |
| **A3** lexical surface = the privacy unit | V is a surface-form set; privacy measured at token/n-gram | F3, F4 |

## Perturbation failures

### F1a — Context-blindness (issue 1.1)
The mechanism is `M(xᵢ)`, never `M(xᵢ | x_{<i}, x_{>i})`; replacement is chosen by embedding
geometry alone, so coherence is incidental. General property of the word-level MLDP family, not a
tuning issue.
**Receipt:** `mi.py` computes a per-token channel `I(Xᵢ;Yᵢ)` *because* the mechanism cannot
express cross-token conditioning. See [The Limits of Word-Level DP](https://www.researchgate.net/publication/362257345).

### F1b — Salience-blindness (issue 1.2)
ε is a single global budget applied uniformly, with no notion of a token's leverage over the rest
of the generation. The disease→disease swap is canonical: two condition names sit close in φ, so
the exponential mechanism exchanges them — premise preserved, clinical fact corrupted.
**Receipt:** the most consequential design fact; it is the F2↔E1 tension hub (see unifying flaw).

### F2 — Identity-preserving replacement (issue 1.3)
Proximity sampling preserves *referential identity*. RANTEXT's defence is Theorem 1 (any `t′∈V`
*can* appear in `C_r`), but mass concentrates near φ(t). Its own "synonym proportion" metric
measures *lexical* synonymy, so Sam→Samuel scores as success while leaking the person.
**Receipt — two opposite A2 failure modes by φ geometry:**
- *Concentration (ada-002 regime):* `S_w` ∈ 0.033–0.122 on the sub-vocab; identity survives.
- *Saturation (high-dim φ, repo G1):* naïve qwen3-embedding (1024-d) → `|C_r| → 100%` of V at
  every ε → uniform sampling, **ε inert, word salad**. Theorem 1 degenerates to "everything equiprobable."

So A2 fails from both sides — too concentrated leaks identity, too flat destroys meaning. No single
φ sits between for all tokens.

### F3a — Case/whitespace fragility (issue 1.4)
`rantext.py` does an exact-string `index.get(surface)`; `sarah` / ` Sarah` / `Sarah` are distinct
rows with distinct neighbourhoods, so the same entity perturbs inconsistently.

### F3b — OOV handling (issue 1.5) — ⚠️ implementation diverges from the paper
- *Paper + whole-word sub-vocab experiments:* tokens not in V are **discarded** ("discards proper
  nouns"). This is privacy-by-deletion: a utility hole *and* a leak, because the deletion pattern
  marks exactly where the sensitive spans were.
- *Default `tokeniser.py` path:* V is cl100k **BPE sub-word pieces** (`tokeniser.py:33`), so
  `Sarah Johnson` tokenizes into in-V pieces and is **perturbed, not dropped** — trading the
  deletion-hole leak for **partial sub-word PII survival**.

Both are bad; the typed-placeholder fix dominates either, but which leak is live depends on the
tokenizer path.

### F4 — Guarantee/evaluation scope mismatch
Per-token, candidate-set-conditional ε does not bound document-level *semantic* disclosure, and the
paper's privacy metrics are lexical.
**Receipt — surface metrics provably do not track privacy:**
- `inv@10` is **dominated by anisotropy, not privacy**: lower anisotropy → *higher* inversion
  (Pythia aniso 0.016 → inv@10 0.428; Gemma aniso 0.151 → 0.139). Low inv@10 = narrow embedding
  cone, not protection.
- `pii_leak` flat 0.512–0.561 and vocab `overlap` flat 0.041–0.118 across 8 φ — no separation.
- A1 fails *empirically*: [Reconstruction of DP Text via LLMs](https://arxiv.org/html/2410.12443v1)
  and [The Double-edged Sword](https://arxiv.org/html/2508.18976) show LLM reconstruction recaptures
  the semantics — and the honest-but-curious GPT-4 in this threat model *is* such an adversary.

## Extraction failures

### E1 — Utility attribution (issue 2.1) — the load-bearing result
g is fed the raw `Doc` locally and uses `Gen_p` only as a reference signal; when `Gen_p` is
incoherent, the remote LLM's marginal contribution collapses → "a local model with extra steps."
**Receipt — measured directly:** end-to-end utility is **flat at 0.78–0.83 across all 8 φ**, and
counter-fitted's real geometry edge (syn_prec **+0.061**, the largest of any φ) **washes out e2e**
(utility Δ −0.008 [−.036,+.019], straddles 0). The wiki conclusion: the extractor "re-grounds on
the **true prefix**." Utility is produced by g seeing raw `Doc` — not by the perturbation.
**Corollary:** fixing φ is the wrong layer; the binding lever is the extractor.

### E2 — Architecture mismatch (issue 2.2a)
The extraction task is conditional denoising/infilling (align `Gen_p` to `Doc`, repair F3b holes,
reconcile F1a incoherence) — a seq2seq/MLM objective, not open-ended decoding.
**Receipt:** the repo uses BERT/MLM **only as an attacker** (`attacks/mask_bert.py`), never as the
reconstructor — the denoising-capable head is on the wrong side.

### E3 — Task-distribution narrowness (issue 2.2b)
The "shared-token overlap between `Gen_p` and `Gen`" premise is a property of *continuation*; no
reason to hold for instruction-following / QA / structured output, which is the dominant real use case.
**Receipt:** corpora are CNN/DM + wikitext + arXiv — no instruction/QA coverage.

## Dependency structure

Everything compounds on one node — **the single scalar ε** — through three edge families.

```
                       A1 token-indep ─────────────▶ F4 (per-token ε ⇏ doc-level semantic bound)
                       A3 surface-unit ──┬─▶ F3a ─gray─▶ F3b ─gray─▶ E1 (holes/markers starve extractor)
                                         └─▶ F4 (lexical metrics ⇏ privacy)

              ┌──────────────────── ε (ONE knob) ────────────────────┐
  purple  F2 identity-removal ◀─tension─▶ F1b salience ─purple─▶ F1a incoherence ─teal─▶ E1
              │ (low ε → premise drifts, F1↑)        (uniform ε)                  (Gen_p salad → lift collapses)
              └ (high ε → near-synonym/self leak, F2↑)
                                                                       E1 ◀─teal─▶ E2 (wrong arch worsens attribution)
```

- **gray** = tokenization-surface (A3): F3a → F3b → E1
- **purple** = semantic-perturbation (A2): F2 ⇄ F1b → F1a
- **teal** = extraction (P5): F1a → E1 ⇄ E2

## The unifying flaw

**One scalar ε governs three quantities that must be controlled independently** —
identity-removal (F2), premise-preservation (F1), extraction-signal (E1) — so RANTEXT cannot sit
anywhere good: lowering ε to destroy identity worsens premise drift and starves Gen_p; raising it
to help coherence/extraction reintroduces near-synonym/self leakage.

The *only other* knob — `|C_r|` size via `Z(ε)` — is **mis-specified across embeddings and
forbidden to tune**: the paper's `Z(ε)` constants encode ada-002's geometry around the token
"happy"; reused elsewhere they give degenerate `|C_r|` (13–39%, or 100%), and per-model
recalibration places models at *different realized privacy* (the repo's hard rule). So there is no
second axis to recover the missing degrees of freedom.

And A2's geometry, even fixed, does not rescue it: syn_prec (P2) is the **only** improvable φ
property and it **does not reach e2e** (E1 dominates). Improving φ is the wrong layer.

The deepest assumption is **A1 (token independence)**: the privacy claim is per-token and lexical,
but the protected asset is a semantic, correlated object. LLM reconstruction recaptures it, so F4
is not a measurement nitpick — it is why the per-token guarantee does not bind the thing that matters.

## Design implications (not pursued here)

Where this points if building past InferDPT. Ordering is set by the receipts: E1's e2e-flatness
means budget-allocation and the extractor are load-bearing; "better φ" provably is not.

- **Split the budget by token role, not one global ε** — a salience/PII pass (NER + leakage-influence
  score) decides whether and how far to perturb each span. Breaks the F2→F1b edge: scrub identity
  spans hard, leave premise-bearing tokens intact.
- **Replace deletion with type-preserving placeholders** (`<PERSON_1>`, consistent per entity) —
  preserves syntax + referential structure for both the remote model and g; removes the F3b
  span-marker leak regardless of tokenizer path.
- **Normalize before tokenizing** — casefold + entity-resolve before BPE so F3a/F3b collapse.
- **Targeted extractor** — conditional denoising/infilling (seq2seq/MLM), and evaluate on
  instruction/QA (E3), not just continuation.
- **Evaluate privacy semantically** — add a re-identification / embedding-reconstruction attacker on
  top of the n-gram metric, or surface-level leakage gives a false sense of privacy (F4 receipt).

## Sources

### Corroborating papers (registered in research-wiki, with the failures each reports)

- [`paper:mattern2022_limits_word_level_dp`](../../research-wiki/papers/mattern2022_limits_word_level_dp.md)
  — The Limits of Word Level Differential Privacy (NAACL Findings 2022). **F1a, F1b, F4, A1** —
  foundational critique of the family.
- [`paper:tong2025_vulnerability_text_sanitization`](../../research-wiki/papers/tong2025_vulnerability_text_sanitization.md)
  — On the Vulnerability of Text Sanitization (NAACL 2025; *by the InferDPT authors*). **F2, F4** —
  optimal reconstruction, +46.4% ASR at ε=4.0.
- [`paper:pang2024_reconstruction_dp_text_llm`](../../research-wiki/papers/pang2024_reconstruction_dp_text_llm.md)
  — Reconstruction of DP Text Sanitization via LLMs (2024). **A1, F4, C3/E1** — 72–94% LLM recovery
  from word-level DP.
- [`paper:meisenbacher2025_double_edged_reconstruction`](../../research-wiki/papers/meisenbacher2025_double_edged_reconstruction.md)
  — The Double-edged Sword of LLM-based Data Reconstruction (WPES 2025). **F1a, F4, E1↔C3** —
  contextual vulnerability; reconstruction is both attack and repair.

### Other
- InferDPT / RANTEXT: https://arxiv.org/html/2310.12214v8 (`paper:tong2023_inferdpt_privacypreserving_inference`)
- Empirical Privacy Loss Calibration for LDP Text Rewriting: https://arxiv.org/html/2603.22968
- Repo: `docs/research/embedding-map.md`, `docs/research/rantext.md`, `docs/research/mi-probes.md`,
  `research-wiki/claims/anisotropy-bad-but-insufficient.md`, `research-wiki/gap_map.md`
