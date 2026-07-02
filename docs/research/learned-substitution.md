---
type: research
status: current
created: 2026-07-02
updated: 2026-07-02
tags: [rantext, inferdpt, privacy, learned-substitution, dp-text-rewriting, anonymization, rd4]
companion: [beyond-rantext.md, rantext-limitations.md]
supersedes: []
---

# Learned context-aware substitution — deep dive (RD4)

Focused follow-up on RD4 from [`beyond-rantext.md`](beyond-rantext.md): replace static
embedding-distance token swaps with a **model that reads context and emits a substitution**. This
traces the DP text-rewriting lineage to current SOTA, maps the architecture families, and flags which
methods also do the **reverse step** (reconstructing a final output from the privatized text /
`RemoteLLM(perturbed)`). All papers verified against their arXiv/venue records.

## Definitions

- **Learned substitution:** a trained model (MLM / seq2seq / decoder LLM) produces the replacement,
  conditioned on context — vs. RANTEXT's distance-in-φ sampling.
- **Local DP (LDP) for text:** every input text must be indistinguishable from any other within ε; very
  demanding, so realized coherence needs high ε (the constraint DP-ST relaxes).
- **Exponential mechanism:** DP selection where output probability ∝ exp(ε·score/2); temperature-sampling
  an LM (DP-Prompt) and sampling MLM logits (DP-MLM) are instances.
- **Sensitivity:** the max change in the mechanism's input to the noise calibration; miscomputing it
  breaks DP (the ADePT/Habernal lesson).
- **Neighborhood DP:** a *relaxed* notion where indistinguishability holds within a bounded neighborhood
  rather than all texts — DP-ST's move to buy coherence at lower ε.
- **Reverse / reconstruction step:** re-expressing coherent output from a privatized intermediate — LLM
  post-processing (DP-ST), re-rewriting (Just Rewrite It Again), or InferDPT-style extraction from the
  remote generation.
- Failure labels **F1a/F1b/F2/F4**, **E1/E2**, **RD0–RD5** are defined in
  [`rantext-limitations.md`](rantext-limitations.md) / [`beyond-rantext.md`](beyond-rantext.md).

## Lineage

```
ADePT (2021) ──[proven NOT DP]──▶ Habernal "devil in the detail" (2021)
   │ autoencoder + Laplace on latent                     │ sensitivity ≥6× too small
   ▼
Mattern paraphrase-anon (2022) ─ DP-BART (2023) ─ DP-Prompt (2023) ─ DP-MLM (2024)
   finetuned paraphraser         seq2seq latent noise  decoder-LLM paraphrase  encoder-MLM per-token
                                         │
              ┌──────────────────────────┼───────────────────────────┬─────────────────┐
              ▼                           ▼                           ▼                 ▼
   Just Rewrite It Again (2024)  Spend Budget Wisely (2025)     DP-ST (2025)      1-Diffractor (2024)
   post-process re-rewrite       per-token ε allocation (RD1∩RD4) triples+LLM recon   fast word-level MDP
```

- **[ADePT](../../research-wiki/papers/krishna2021_adept.md)** (Krishna et al. 2021, EACL, [arXiv 2102.01502](https://arxiv.org/abs/2102.01502))
  — autoencoder + Laplace on the latent. The origin, and the cautionary node: **not actually DP.**
- **[Habernal 2021](../../research-wiki/papers/habernal2021_dp_nlp_devil.md)** (EMNLP, [arXiv 2109.03175](https://arxiv.org/abs/2109.03175))
  — proves ADePT's sensitivity was ≥6× too small → guarantee void. RD4's discipline check.
- **[Paraphrase-anonymization](../../research-wiki/papers/mattern2022_limits_word_level_dp.md)** (Mattern et al. 2022, [arXiv 2205.02130](https://arxiv.org/abs/2205.02130))
  — finetuned paraphraser with a formal guarantee; first "learned beats word-level DP on all axes."
- **[DP-BART](../../research-wiki/papers/igamberdiev2023_dp_bart.md)** (Igamberdiev & Habernal 2023, [arXiv 2302.07636](https://arxiv.org/abs/2302.07636))
  — seq2seq latent-noise + clipping + iterative pruning; diagnoses the LDP adjacency-constraint noise blow-up.
- **[DP-Prompt](../../research-wiki/papers/utpala2023_locally_differentially_private.md)** (Utpala et al. 2023, [arXiv 2310.16111](https://arxiv.org/abs/2310.16111))
  — zero-shot FLAN-T5 paraphrase; temperature sampling = exponential mechanism.
- **[DP-MLM](../../research-wiki/papers/meisenbacher2024_dp_mlm.md)** (Meisenbacher et al. 2024, [arXiv 2407.00637](https://arxiv.org/abs/2407.00637))
  — encoder-only MLM (RoBERTa), contextual per-token exp-mech; 2025 update adds a sliding window for
  arbitrary-length documents. Best utility at low ε among the guaranteed methods.

## Architecture families

| Family | Exemplar | Mechanism | Strength | Weakness |
|---|---|---|---|---|
| Latent-noise autoencoder/seq2seq | ADePT → DP-BART | noise in continuous latent | document-level rewrite | adjacency-constraint noise blow-up; DP proof fragile |
| Decoder-LLM paraphrase | DP-Prompt | temperature sampling = exp-mech | zero-shot, fluent | coarse, one-shot whole-output DP |
| Encoder-only MLM, per-token | DP-MLM | exp-mech over masked logits | best utility at low ε; long-doc via sliding window | still per-token |
| Decompositional / structured | **DP-ST** | privatize semantic triples → LLM reconstruct | coherence at low ε (divide-and-conquer) | relies on triple extraction + relaxed neighborhood-DP |
| Group / multi-granular | **DP-GTR** | doc+word rewriting + in-context aggregation | multi-granularity; prompt plug-in | no answer-recovery step |
| Word-level metric-DP (efficient) | 1-Diffractor | geometric "diffraction" over 1-D lists | ~15× faster, low memory | word-level only (F1a persists) |
| Non-DP LLM-adversarial | **Staab**; **RUPTA** | infer→rewrite loop, evaluator-guided | best empirical trade-off; RUPTA distills to a local model | no formal DP guarantee |

The trajectory: continuous-latent noise (fragile) → per-token exp-mech (DP-MLM, solid) → decompose-and-
reconstruct (DP-ST) and LLM-adversarial (Staab/RUPTA).

## Current SOTA (2025)

- **[DP-ST](../../research-wiki/papers/meisenbacher2025_dp_st.md)** (Meisenbacher et al., EMNLP 2025, [arXiv 2508.20736](https://arxiv.org/abs/2508.20736))
  — semantic triples + neighborhood-DP + LLM reconstruction; coherent output at *lower* ε. Keeps a
  (relaxed) guarantee. One frontier.
- **[RUPTA](../../research-wiki/papers/yang2025_rupta.md)** (Yang, Zhu, Gurevych, ACL 2025, [arXiv 2407.11770](https://arxiv.org/abs/2407.11770))
  — privacy-evaluator + utility-evaluator + optimizer LLMs; distilled to a lightweight local model. Best
  empirical trade-off, no formal DP. The other frontier.
- **[Staab et al.](../../research-wiki/papers/staab2024_llm_anonymizers.md)** (ICLR 2025, [arXiv 2402.13846](https://arxiv.org/abs/2402.13846))
  — adversarial anonymization: infer-then-rewrite; the duality embodied (best anonymizer = best re-identifier).

## Crossovers with the rest of the taxonomy

- **RD1 inside RD4:** [Spend Your Budget Wisely](../../research-wiki/papers/meisenbacher2025_spend_budget_wisely.md)
  (CODASPY 2025, [arXiv 2503.22379](https://arxiv.org/abs/2503.22379)) allocates ε per token by linguistic
  salience and beats uniform ε — direct empirical support for decomposing the scalar budget (fixes F1b).
- **Realized ε:** *On the Impact of Noise in DP Text Rewriting* ([arXiv 2501.19022](https://arxiv.org/abs/2501.19022))
  and *Empirical Privacy Loss Calibration* ([arXiv 2603.22968](https://arxiv.org/abs/2603.22968)) confirm
  nominal ε ≠ realized privacy here (the taxonomy's D1).
- **Survey:** *A Survey on Current Trends and Recent Advances in Text Anonymization* ([arXiv 2508.21587](https://arxiv.org/abs/2508.21587)).

## The reverse process (reconstruction from the privatized text)

Most DP-rewriting papers stop at "produce a private text" and never do the round-trip. The ones that
specify a reconstruction/extraction step:

- **[InferDPT](../../research-wiki/papers/tong2023_inferdpt_privacypreserving_inference.md)** — canonical:
  extraction module (RAG + distillation-inspired) rebuilds an aligned output from `RemoteLLM(perturbed)`.
- **[DP-ST](../../research-wiki/papers/meisenbacher2025_dp_st.md)** — LLM post-processing reconstructs
  coherent text from privatized triples (local reconstruction; same "privatize a representation → LLM
  re-express" shape as InferDPT).
- **[HaS](../../research-wiki/papers/chen2023_hide_seek_has.md)** — trains a local "Seek" model to
  de-anonymize the remote output.
- **[Just Rewrite It Again](../../research-wiki/papers/meisenbacher2024_just_rewrite_again.md)** (ARES 2024, [arXiv 2405.19831](https://arxiv.org/abs/2405.19831))
  — a *post-processing re-rewrite* that improves both semantic similarity and empirical privacy; the
  productive form of the reconstruction duality, free under post-processing immunity.
- **[RUPTA](../../research-wiki/papers/yang2025_rupta.md)** — distills the anonymizer into a local model
  (the "make the private stage local" move).
- **[DP-GTR](../../research-wiki/papers/li2025_dp_gtr.md)** (EMNLP 2025, [arXiv 2503.04990](https://arxiv.org/abs/2503.04990))
  — protects the *prompt* to an online LLM, but does **not** recover the answer from the remote response.

## Conclusions

Two viable RD4 spines, and they are different bets:
- **Keep a formal guarantee:** DP-MLM-style contextual per-token exp-mech **+** Spend-Your-Budget salience
  allocation (RD1∩RD4) **+** DP-ST-style decompose-and-LLM-reconstruct for coherence.
- **Maximize empirical trade-off:** LLM-adversarial anonymization (Staab / RUPTA) distilled to a local
  model, evaluated against an LLM re-identifier — no DP guarantee, best numbers.

Encoder-only MLMs (DP-MLM) are the low-ε sweet spot among guaranteed methods; the reconstruction step is
best treated as a **local, zero-budget** stage (post-processing immunity), which DP-ST, Just-Rewrite, and
InferDPT all exploit.

## Open problems

- **The core gap:** no method gives a *formal* guarantee for a genuinely context-conditioned swap — DP-ST
  buys it only by relaxing to neighborhood-DP; Staab/RUPTA drop it entirely. Closing this is the RD4 prize.
- Getting sensitivity right for learned/latent mechanisms (the ADePT/Habernal hazard).
- On-device cost of the LLM-adversarial branch (distillation quality vs. the teacher).
- Instruction/QA-task evaluation (E3) — this whole line is validated on classification/paraphrase, not the
  dominant real use case.

## Sources

Registered anchors (wiki, with arXiv):
[ADePT](../../research-wiki/papers/krishna2021_adept.md) ([arXiv 2102.01502](https://arxiv.org/abs/2102.01502)),
[Habernal critique](../../research-wiki/papers/habernal2021_dp_nlp_devil.md) ([arXiv 2109.03175](https://arxiv.org/abs/2109.03175)),
[Paraphrase-anon / Limits](../../research-wiki/papers/mattern2022_limits_word_level_dp.md) ([arXiv 2205.02130](https://arxiv.org/abs/2205.02130)),
[DP-BART](../../research-wiki/papers/igamberdiev2023_dp_bart.md) ([arXiv 2302.07636](https://arxiv.org/abs/2302.07636)),
[DP-Prompt](../../research-wiki/papers/utpala2023_locally_differentially_private.md) ([arXiv 2310.16111](https://arxiv.org/abs/2310.16111)),
[DP-MLM](../../research-wiki/papers/meisenbacher2024_dp_mlm.md) ([arXiv 2407.00637](https://arxiv.org/abs/2407.00637)),
[1-Diffractor](../../research-wiki/papers/meisenbacher2024_1diffractor.md) ([arXiv 2405.01678](https://arxiv.org/abs/2405.01678)),
[Just Rewrite It Again](../../research-wiki/papers/meisenbacher2024_just_rewrite_again.md) ([arXiv 2405.19831](https://arxiv.org/abs/2405.19831)),
[Spend Your Budget Wisely](../../research-wiki/papers/meisenbacher2025_spend_budget_wisely.md) ([arXiv 2503.22379](https://arxiv.org/abs/2503.22379)),
[DP-ST](../../research-wiki/papers/meisenbacher2025_dp_st.md) ([arXiv 2508.20736](https://arxiv.org/abs/2508.20736)),
[DP-GTR](../../research-wiki/papers/li2025_dp_gtr.md) ([arXiv 2503.04990](https://arxiv.org/abs/2503.04990)),
[RUPTA](../../research-wiki/papers/yang2025_rupta.md) ([arXiv 2407.11770](https://arxiv.org/abs/2407.11770)),
[Staab LLM anonymizers](../../research-wiki/papers/staab2024_llm_anonymizers.md) ([arXiv 2402.13846](https://arxiv.org/abs/2402.13846)),
[InferDPT](../../research-wiki/papers/tong2023_inferdpt_privacypreserving_inference.md) ([arXiv 2310.12214](https://arxiv.org/abs/2310.12214)),
[HaS](../../research-wiki/papers/chen2023_hide_seek_has.md) ([arXiv 2309.03057](https://arxiv.org/abs/2309.03057)).
Other references cited inline with arXiv IDs.
