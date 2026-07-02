---
type: plan
status: current
created: 2026-07-02
updated: 2026-07-02
tags: [co-design, learned-substitution, extraction, reconstruction, round-trip, rd4, rd5, plan]
companion: [../research/learned-substitution.md, ../research/beyond-rantext.md, ../research/rantext-limitations.md]
---

# Next stage — substitutor↔extractor co-design

Plan for the next research+prototyping stage: co-design of the **learned substitutor**
(`doc_orig → doc_p`) and the **efficient local extractor** (`out_p → out_final`). Grounded in the
seven-paper analysis registered in `research-wiki/papers/` (2026-07-02 pass): DP-MLM, DP-ST,
Spend-Your-Budget-Wisely, RUPTA, AgentStealth, INTACT, NaPaRe.

## Definitions

- **Round trip:** `doc_orig → doc_p → RemoteLLM(doc_p, task) = out_p → out_final`; utility is scored on
  `out_final`, privacy on `doc_p` *and* `out_final` against an LLM re-identification attacker.
- **Substitution record R:** the client-held byproduct of substitution — per-span
  `(span, type, replacement, rejected candidates / ε / alignment)` — never sent remotely.
- **Truthful generalization:** replacing a span with a broader term that still subsumes it
  ("Norway" → "a Scandinavian country"); *invertible by narrowing* for the client who holds the original.
- **Matched realized privacy:** methods compared only at equal measured attacker success, moved along each
  method's own legitimate knob (see CLAUDE.md empirical-honesty rules).
- **Leak-through:** sensitive content re-entering `out_final` via the extractor (which sees `doc_orig`)
  or via the substitutor's own memorization.
- Failure labels (F1–F4, E1–E3) and directions (RD0–RD5): see
  [`rantext-limitations.md`](../research/rantext-limitations.md) /
  [`beyond-rantext.md`](../research/beyond-rantext.md).

## What the seven-paper analysis established

1. **Nobody does the round trip.** All seven papers stop at "release the sanitized text"; none maps a
   remote output back, and none evaluates against a frontier-LLM re-identification attacker. Both gaps
   are exactly this project's contribution space.
2. **Free reverse maps exist and are discarded.**
   [DP-MLM](../../research-wiki/papers/meisenbacher2024_dp_mlm.md) ([arXiv 2407.00637](https://arxiv.org/abs/2407.00637))
   emits a position-aligned `(pos, w_orig, w_sub)` map;
   [DP-ST](../../research-wiki/papers/meisenbacher2025_dp_st.md) ([arXiv 2508.20736](https://arxiv.org/abs/2508.20736))
   a private↔public triple table;
   [INTACT](../../research-wiki/papers/pilan2024_truthful_sanitization.md) ([arXiv 2412.12928](https://arxiv.org/abs/2412.12928))
   a span→generalization lattice that is invertible by narrowing. Each is an extractor conditioning
   signal the source paper throws away. **The substitution record R is the co-design interface.**
3. **Truthful generalization is the right substitution family for a round trip.** The remote model
   computes on true-but-coarser premises instead of Presidio-style falsehoods (INTACT: 93.2% vs 19.7%
   truthful), and narrowing is deterministic for the client. Conversely, deletion is anti-extractor:
   [NaPaRe](../../research-wiki/papers/huang2025_tree_search_rewriting.md) ([arXiv 2509.20838](https://arxiv.org/abs/2509.20838))'s
   `delete` action leaves no anchor in `doc_p` for any local extractor to recover.
4. **The optimization plumbing transfers, with cautions.**
   [AgentStealth](../../research-wiki/papers/shao2025_agentstealth.md) ([arXiv 2506.22508](https://arxiv.org/abs/2506.22508)):
   GRPO reward `0.5·(1−attack) + 0.5·(BLEU/ROUGE)` — swap the utility term for round-trip `out_final`
   utility and the recipe transfers; but RL adds only **+1.1 pt over SFT** (62.6→63.7), and its utility
   metric (surface similarity to `doc_orig`) rewards under-anonymization.
   [RUPTA](../../research-wiki/papers/yang2025_rupta.md) ([arXiv 2407.11770](https://arxiv.org/abs/2407.11770)):
   the lexicographic schedule "push re-id rank out of top-K, then repair utility" is a ready-made
   matched-realized-privacy operating procedure, and DPO on intermediate-vs-final traces (not SFT) is
   what closes the distillation privacy gap; but the teacher pipeline **sends `doc_orig` to GPT-4** —
   only the distilled local student fits our threat model — and evaluator=attacker circularity must be
   broken with a held-out attacker.
5. **Budget shaping alone does not fix the trade-off.**
   [Spend Your Budget Wisely](../../research-wiki/papers/meisenbacher2025_spend_budget_wisely.md)
   ([arXiv 2503.22379](https://arxiv.org/abs/2503.22379)) gets mixed privacy results and "nearly always
   lower utility", concentrating distortion on exactly the salient tokens a task answer must mention —
   independent confirmation of E1: the extractor side is load-bearing, the perturbation side alone is not.
6. **The DP branch supplies a baseline, not a bet.** DP-MLM has a per-token-ε API (composes with the
   budget allocator; same authors) and a batched GPU path, but collapses at strict budgets
   (G-Eval 0.056 at DP-ST's ε=0.5/word regime); DP-ST's neighborhood notion is a real relaxation and its
   no-triples fallback returns the input **unmodified** (a privacy hole). Report these as outcomes on the
   shared Pareto plot; don't build on them.

## Proposed directions

### D1 — Substitution-record interface + matched local extractor (start here)

**Hypothesis H1:** a small local extractor conditioned on `(doc_orig, R, out_p)` and trained on the
*exact induced corruption* beats (i) deterministic re-substitution rules and (ii) InferDPT-style
general-LLM extraction, on `out_final` utility at matched realized privacy.

**Local-compute discipline (applies to D1 and D2).** The deployed local path uses no autoregressive
instruct LLM. The substitutor's job decomposes into detection, lookup, ranking, and span-infilling
subtasks, each with a cheaper tailored architecture; DP-MLM's own headline result (encoder-only beats
decoder rewriting at fixed ε) is the field's evidence that encoders are the sweet spot for substitution.
A large LLM appears in exactly two places, neither at runtime: **offline** as distillation teacher for
the residual infiller, and **remote** as the evaluation attacker.

**Prototype:**
1. **Substitutor v0 (tailored cascade, no instruct LLM):**
   - *Detection + typing:* encoder token-classifier (DeBERTa-v3-class, ~90–180M) + Presidio patterns;
     TAB gives gold spans for the controlled condition. *Coreference* for a consistent entity→surrogate
     map: fastcoref-class encoder (~140M).
   - *Direct identifiers* → typed placeholders; *dates/numbers/codes* → rule buckets. No model.
   - *Quasi-identifier generalization candidates* → **KB lattice first** (WordNet hypernyms, Wikidata
     instance-of/subclass-of chains, GeoNames containment): deterministic, auditable, and subsumption
     is guaranteed — stronger truthfulness than INTACT's LLM generation, which degrades to
     definitions/paraphrases on DEM/MISC spans (its admitted failure mode).
   - *Residue spans with no KB entry* → small **encoder-decoder span infilling** (flan-T5/BART-base
     class, 250–780M), LoRA-distilled offline from a large teacher; multi-word spans need the
     span-infilling objective, which is exactly T5/BART pretraining — per-token MLM swap alone has
     single-token bias. An NLI encoder (~180M) verifies the output still subsumes the original.
   - *Attack-guided selection* → replace INTACT's 15–20×-cost LLM guess-back with a batched **MLM
     token-inference probe** (mask the surrogate, check whether the original ranks high — Budget-Wisely's
     MTI attack repurposed, ~125M). Receipt that this suffices at selection time: INTACT's *most
     specific* candidate already survives its own LLM filter for 53% of spans — the expensive filter
     mostly rubber-stamps. The frontier-LLM attacker remains, but only at evaluation time.
   - Emit `doc_p` **and** R (span, type, surrogate, lattice position, coref chain id).
2. **Data engine:** run the substitutor + a local remote-proxy LLM over instruction/QA/summarization
   tasks (E3 coverage, not continuation) to produce `(doc_orig, task, doc_p, R, out_p, out_ref)` tuples.
   (The proxy LLM is scaffolding for data generation and stands in for the remote model — it is not
   part of the deployed local path.)
3. **Extractor ladder:** (a) rule baseline — deterministic placeholder inversion + mention alignment
   into R (exact → fuzzy → small-embedding match) and re-substitution — extraction conditioned on R is
   mostly dictionary-shaped, so rules carry the copy-heavy bulk (the spirit of RD5's "tag, don't
   rewrite" without an edit-tagger: the 2026-07-02 tooling pass found GECToR/LaserTagger-family
   codebases frozen or dead, and R-conditioned edits are lookups a fixed tag vocabulary can't express);
   (b) small enc-dec (flan-T5-base class, ~250M) LoRA-finetuned on synthesized inversions for the
   residue rules can't align. The gap (b)−(a) is the measured value of learning the extractor.
   Detailed spec: [`2026-07-02-d1-prototype-implementation.md`](2026-07-02-d1-prototype-implementation.md).

**Justification:** attacks E1/E2 — the taxonomy's load-bearing layer — and exploits finding 2 (the
discarded reverse maps). The whole deployed stack is sub-1B of encoders/enc-decs that batch trivially
and fit the iGPU together; nothing here needs formal-DP machinery.

**Tradeoffs / risks:** no formal guarantee (privacy is empirical-adversarial by design); inherits the
RD2 detection ceiling outside gold-span datasets; precision genuinely lost where the task needed the
suppressed specificity (report, don't engineer around). Tailored-stack-specific risks: KB coverage gaps
(nicknames, novel entities, cross-lingual surface forms) fall to the distilled infiller — its quality vs
the teacher is a measured quantity, not an assumption; and document-level **QI-combination reasoning**
(RD2's hard case) is the one subtask where small encoders may genuinely trail an LLM reasoner — a local
encoder re-identification probe approximates it, and the evaluation attacker's success on `doc_p` is the
ground truth for whether that sufficed. If it doesn't, that is a finding about the minimum local model,
not a license to ship an 8B substitutor by default. **Kill criterion:** if the learned extractor
doesn't beat the rule baseline at matched realized privacy, the finding is "R + rules suffice" — publish
that and skip extractor training in D2.

### D2 — Round-trip reward co-optimization (the actual co-design loop; gated on D1)

**Hypothesis H2:** optimizing the substitutor against the joint round-trip reward
`α·(1 − attacker success on doc_p) + (1−α)·U(out_final after the D1 extractor)` shifts the Pareto
frontier vs. optimizing doc_p-surface utility (the AgentStealth/RUPTA objective) — because the
substitutor learns corruptions *the extractor can invert but the attacker cannot*, and because
doc_p-similarity provably rewards under-anonymization (AgentStealth's Standard-Prompt utility 0.92).

**Prototype:** the policy is not a monolithic LLM but the cascade's two learnable components — the
residual infiller (250–780M enc-dec) and the candidate-selection ranker — which keeps GRPO cheap on the
iGPU. SFT-first on D1 traces, then GRPO with the round-trip reward (AgentStealth recipe). Adversary in
the reward = a local attack head (the MLM probe or an encoder re-id classifier); **evaluation attacker =
a different, held-out model family** (breaks RUPTA's circularity). Cheap variant first: NaPaRe's
training-free tree search with our reward swapped into its reward slot — same objective, zero training —
as the go/no-go probe before spending RL compute.

**Tradeoffs / risks:** each reward evaluation is a full round trip (proxy call + extractor pass) —
rollouts are expensive, so batch/cache the proxy; reward hacking (the substitutor colluding with the
extractor through R — audit R's information content as a leak channel); the AgentStealth data point
says RL headroom over SFT may be ~1 pt — gate GRPO on the search-variant showing a frontier shift first.
**Kill criterion:** no shift over D1 at matched realized privacy in the search variant → co-optimization
is not where the value is; D1's interface was the contribution.

### D3 — DP-anchored baseline spine (parallel, cheap, comparison only)

Run DP-MLM (existing code, per-token-ε API, batched path) + the Budget-Wisely allocator as the
formally-anchored reference on the shared Pareto plot, with a matched extractor conditioned on its free
`(pos, w_orig, w_sub)` map. Purpose: (i) quantify the E1 lever on a guaranteed mechanism — does a
matched extractor rescue DP-MLM's known collapse at strict ε?; (ii) give every D1/D2 claim a
guaranteed-method comparison point at matched *realized* privacy (nominal per-token ε is vacuous under
sequential composition — report attacker success only).

**Tradeoffs:** known-weak utility (expected; a reportable outcome per the honesty rules); the word-pair
map only partially transfers to free-text `out_p`. Effort is small — mechanism code exists; the work is
plugging it into the D1 data engine and eval harness.

## Shared evaluation protocol (all directions)

- **Privacy = attacker success**, measured on `doc_p` **and** `out_final` (leak-through channel): a
  frontier-LLM re-identification / attribute-inference attacker (Staab-style, via the configured
  OpenAI-compatible endpoint), fixed and identical across all methods. DIRI-style re-id and
  Budget-Wisely's MLM token-inference attack kept as cheap local probes, never as the headline metric.
- **Utility on `out_final` only**, on instruction/QA/summarization tasks (E3), never continuation.
- **Comparison only at matched realized privacy**, moved along each method's own knob (α for D1/D2,
  ε for D3); no per-model calibration of any secondary quantity (hard rule).
- **Corpora:** SynthPAI + Staab synthetic Q&A (8 ground-truth attributes → attribute-inference attacker
  comes for free) and TAB (gold PII spans + background-knowledge re-id), with tasks constructed over
  both. Both are the corpora the strongest baselines (AgentStealth, INTACT/RUPTA) already use — direct
  comparability.
- **Perf gate** per CLAUDE.md before any heavy run; one GPU process at a time; `-u` logging.

## Sequencing

1. **D1 now** — data engine + substitutor v0 + extractor ladder; it is the prerequisite for D2 and the
   harness for D3.
2. **D3 in parallel** — small effort, reuses the D1 harness, and every later claim needs its baseline.
3. **D2 after D1's extractor exists** — search-variant probe first; GRPO only if the probe shows a
   frontier shift.

## Sources

Seven analysis pages (this pass, all with co-design-fitness sections):
[DP-MLM](../../research-wiki/papers/meisenbacher2024_dp_mlm.md) ([arXiv 2407.00637](https://arxiv.org/abs/2407.00637)),
[DP-ST](../../research-wiki/papers/meisenbacher2025_dp_st.md) ([arXiv 2508.20736](https://arxiv.org/abs/2508.20736)),
[Spend Your Budget Wisely](../../research-wiki/papers/meisenbacher2025_spend_budget_wisely.md) ([arXiv 2503.22379](https://arxiv.org/abs/2503.22379)),
[RUPTA](../../research-wiki/papers/yang2025_rupta.md) ([arXiv 2407.11770](https://arxiv.org/abs/2407.11770)),
[AgentStealth](../../research-wiki/papers/shao2025_agentstealth.md) ([arXiv 2506.22508](https://arxiv.org/abs/2506.22508)),
[INTACT / Truthful Sanitization](../../research-wiki/papers/pilan2024_truthful_sanitization.md) ([arXiv 2412.12928](https://arxiv.org/abs/2412.12928)),
[NaPaRe / Iterative Tree Search](../../research-wiki/papers/huang2025_tree_search_rewriting.md) ([arXiv 2509.20838](https://arxiv.org/abs/2509.20838)).
Context: [Staab LLM anonymizers](../../research-wiki/papers/staab2024_llm_anonymizers.md) ([arXiv 2402.13846](https://arxiv.org/abs/2402.13846)),
[DIRI](../../research-wiki/papers/morris2024_diri.md) ([arXiv 2410.17035](https://arxiv.org/abs/2410.17035)),
[InferDPT](../../research-wiki/papers/tong2023_inferdpt_privacypreserving_inference.md) ([arXiv 2310.12214](https://arxiv.org/abs/2310.12214)),
[HaS](../../research-wiki/papers/chen2023_hide_seek_has.md) ([arXiv 2309.03057](https://arxiv.org/abs/2309.03057)).
