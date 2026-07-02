---
type: plan
status: current
created: 2026-07-02
updated: 2026-07-02
tags: [d1, implementation, substitutor, extractor, eval, inferdpt-comparison, prototype]
companion: [2026-07-02-codesign-next-stage.md, ../research/learned-substitution.md]
---

# D1 minimal prototype — implementation plan

Implements D1 from [`2026-07-02-codesign-next-stage.md`](2026-07-02-codesign-next-stage.md): a tailored
substitutor emitting `doc_p` + substitution record R, a data engine over the remote proxy, an extractor
ladder, and an eval harness comparing against the repo's existing InferDPT implementation at matched
realized privacy. All component choices below were verified against live GitHub/HF/dump pages
(2026-07-02 research pass); flagged items are marked.

## Definitions

- **R (substitution record):** client-held JSON per document: for each span
  `{id, char_span, surface, type, id_class(direct|quasi), chain_id, action, replacement,
  lattice[most→least specific], probe_risk}`.
- **Lattice:** the ordered generalization candidates for a span ("Oslo" → "a Norwegian city" →
  "a Scandinavian capital" → "a European city").
- **τ (probe threshold):** D1's privacy knob — the substitutor walks each quasi-identifier's lattice
  most-specific-first and takes the first level whose MLM guess-back risk < τ. Lower τ = coarser output.
- **MTI probe:** masked-token-inference — mask the surrogate in context, score whether the original
  ranks in an MLM's top-k completions (selection-time attack, local, cheap).
- **out_ref:** `RemoteLLM(doc_orig, task)` — the no-privacy anchor answer; utility of any method =
  similarity of its `out_final` to `out_ref` (plus gold-grounded QA scoring).

## Component stack (verified)

| Role | Choice | Size | License | Rejected alternatives (reason) |
|---|---|---|---|---|
| Detection shell | Presidio (`presidio-analyzer`, installed) | — | MIT | — (regex/checksum recognizers cover CODE-like types NER misses) |
| Span detector | **GLiNER** `urchade/gliner_small-v2.1`; ablation `gliner_multi_pii-v1` | 166M / 209M | Apache-2.0 | piiranha (CC-NC-ND); dslim/bert-NER (no DEM/QUANTITY/CODE labels). GLiNER takes TAB's 8 categories as free-text zero-shot labels — the only detector covering them all |
| Coreference | **fastcoref** (FCoref distilled) | ~140M | MIT | maverick-coref (CC-NC + DeBERTa-large size); coreferee (unmaintained) |
| Lattice: common nouns | WordNet via nltk (installed; also covers famous geo via `instance_hypernyms`: verified "Norway"→scandinavian_country) | 0 | — | — |
| Lattice: places | GeoNames `cities500` + admin codes (~16 MB) | 0 | CC-BY 4.0 | full allCountries.zip (400 MB, unneeded) |
| Lattice: dates/numbers | hand-rolled buckets (~50 lines, `dateutil`) | 0 | — | ARX (Java); nothing standard in Python exists (verified) |
| Lattice: named entities | **Teacher-only (decision 2026-07-02, supersedes hybrid):** every entity gets a teacher-written lattice — `Qwen3.6-35B-A3B` on the **local llama-swap** (`localhost:8060`), cached offline; no entity linking at all, so no linking failure modes; NLI gate is the sole truthfulness check | 0 runtime | — | YAGO 4.5 tiny (demoted to optional chain-*validator* ablation, not in v0); Probase (dropped after 2026-07-02 audit: obtainable but 2016-vintage + noisy; typicality-as-leak-prior kept as open idea); spaCy-entity-linker (unneeded) |
| Selection probe | MTI port from `sjmeis/EpsilonDistributor` `eval.py` (`MaskedTokenInference`, ~60 lines, roberta-base) | 125M | MIT | deberta probe (stronger but breaks comparability with published MTI numbers) |
| Extractor rules | exact match → rapidfuzz → `all-MiniLM-L6-v2` cosine for paraphrased mentions | 22M | MIT/Apache | awesome-align (dead, wrong task) |
| Extractor residue | **flan-t5-base** LoRA (peft, `SEQ_2_SEQ_LM`) | 248M | Apache-2.0 | bart-base 139M (cheap ablation, native infilling objective); edit-taggers (GECToR frozen/AllenNLP, LaserTagger TF1-dead; R-conditioned edits are dictionary lookups a tag vocabulary can't express) |
| Eval attacker | adapt **`eth-sri/llm-anonymization`** inference-eval half via existing `LLMClient` | remote | MIT | DIRI (no public code, verified); TAB `evaluation.py` (span metrics, not an attack) |
| Utility metrics | ROUGE-L + BERTScore (`distilbert`/`deberta-base` backend) + SQuAD-style EM/F1 for QA | ≤180M | MIT | G-Eval judge (bias caveats; optional tie-breaker only, fixed judge id) |
| Corpora | **TAB** (1,268 ECtHR docs, gold spans: 8 types × direct/quasi × `entity_id` coref chains) + **SynthPAI** (300 profiles, 7.8k comments, 8 gold attributes) | — | MIT / data CC-BY-NC-SA | Staab 525 synthetic (SynthPAI is its bigger successor, same group) |

Existing repo pieces reused as-is: `LLMClient` + content-addressed disk cache (`llm.py`),
`pmap` threaded proxy calls, Presidio probes (`probes/_common.py`), mask-BERT attack as template
(`attacks/mask_bert.py`), incremental-JSON results convention, `__main__` self-check convention.
InferDPT baseline = the repo's own `pipeline.py`/`extraction.py` (faithful: uses the paper's verbatim
extraction instruction; upstream repo has **no license file**, ours is already clean-room from the paper).

## Module layout

New package `src/cloak/` (baseline stays in `src/inferdpt/`), same `PYTHONPATH=src` convention:

```
src/cloak/
  detect.py      # Presidio ∪ GLiNER (TAB labels) ∪ fastcoref → typed spans + chains
  lattice.py     # WordNet / GeoNames / buckets / teacher-cache → candidate lattices
  probe.py       # MTI guess-back (roberta-base), token- and span-level
  substitute.py  # routing + lattice walk at τ → (doc_p, R); placeholder numbering per chain
  extract.py     # rung A rules (exact→rapidfuzz→MiniLM) ; rung B flan-t5 LoRA inference
  tasks.py       # task construction (summarization, QA) + out_ref anchors
  attacker.py    # LLM attribute-inference / masked-entity recovery via LLMClient
scripts/
  d1_build_assets.py   # download GeoNames, build teacher lattice cache (local llama-swap), QA gen (cached)
  d1_run.py            # end-to-end sweep → results/d1_*.json (incremental)
  d1_train_extractor.py# LoRA training for rung B
```

## Phases

**v0.1 scope cut (decision 2026-07-02) — the first end-to-end loop, ~1 week, no overnight jobs:**
60 SynthPAI docs only; **lazy incremental lattice cache** (build only entities appearing in the docs
being run — low hundreds of teacher calls ≈ 5–10 min at 6-way parallel E4B; the 8-doc smoke is
near-instant); rationale-free lattice prompt by default (CoT reserved for the Qwen escalation);
3 lattice levels;
**extractor = rung A rules only** — rung B LoRA is trained only after the loop produces its first
Pareto points, to measure the B−A gap (it is D1's hypothesis test, not its plumbing). TAB round trip,
pseudonym ablation, YAGO-validator, CoT arm: all second pass. WordNet/GeoNames/rule buckets are the
zero-call bootstrap — SynthPAI quasi-identifiers are dominated by occupations/education/locations/
values they already cover, so the teacher sees only the named-entity residue.

### P0 — environment, data, detection gate (~0.5–1 day)
1. `pip install gliner fastcoref peft evaluate rouge_score bert_score` (never touch torch — ROCm rule).
2. Fetch TAB (git, standoff JSON → `corpora/tab/`), SynthPAI (HF `RobinSta/SynthPAI` →
   `corpora/synthpai/`), GeoNames cities500+admin (→ `data/geonames/`).
3. **Detection gate (the privacy ceiling, measured first):** Presidio∪GLiNER span recall/precision on
   TAB test gold spans, per entity type × direct/quasi. Decision rule: if DIRECT-identifier recall
   < 0.95, the *measured* condition still runs on detected spans but the *controlled* condition uses
   TAB gold spans, and the detector gap is reported as a finding (not patched over).

### P1 — substitutor v0 (~2–3 days)
1. `detect.py`: merge the three sources; dedupe overlapping spans (widest wins); direct-vs-quasi rule
   (PERSON full names, CODE, contact → direct; rest quasi), calibrated against TAB's gold
   `identifier_type`.
2. `lattice.py`: dates/numbers buckets; GeoNames place chains; WordNet hypernym paths (common nouns +
   `instance_hypernyms` for famous geo); **all remaining named entities via teacher-written lattices**
   (decision 2026-07-02, teacher-only — supersedes the earlier YAGO-hybrid): one call per unique
   (entity, sense-context) to the **teacher cascade** — `gemma 4 (E4B)` primary (6 llama-swap slots,
   `pmap` workers=6, thinking disabled), rationale-free prompt, 3 levels most→least specific; **NLI
   rejections regenerate once via `Qwen3.6-35B-A3B`** (CoT prompt) — cached to
   `data/lattice_cache.json`, built lazily per scale rung so runtime stays pure lookup. No entity
   linking exists in this path — the silent wrong-link failure mode is gone by construction; the NLI
   gate (original sentence must entail the generalized one) is the sole quality check, and both its
   rejection rate and the E4B→Qwen escalation rate are reported per corpus. Cache build at v0.1 scale
   (60 docs, low hundreds of entities, 6-way parallel): **~5–10 min**; full corpus ≈ under an hour;
   $0 remote (llama-swap shares the iGPU — nothing else on the GPU during builds).
3. `probe.py`: port `MaskedTokenInference` (roberta-base, top-k, batched); add span-level variant
   (mask the whole surrogate, check original lemmas in top-k infills).
4. `substitute.py`: direct → `<PERSON_1>`-style typed placeholders, numbered per coref chain
   (realistic-pseudonym variant deferred to a P4 ablation arm — decision 2026-07-02); quasi → lattice
   walk at τ; emit `doc_p` + R. `__main__` self-check on `corpora/dev.txt`.

### P2 — data engine (~1–2 days)
1. `tasks.py`: two families — (a) summarization; (b) QA: questions + gold answers generated once from
   `doc_orig` by the teacher, cached (gold answers grounded in `doc_orig`, so EM/F1 needs no model).
   **Corpus order (decision 2026-07-02): SynthPAI round trip first** (per-profile comment threads;
   8 gold attributes make the headline attacker work immediately); TAB serves v0 only via the P0
   detection gate, its round trip lands in a second pass.
2. Engine: `doc → substitute(τ) → RemoteLLM(doc_p, task) → out_p`; persist
   `{doc, task, doc_p, R, out_p, out_ref}` to `data/d1_tuples/*.jsonl` via `LLMClient` cache + `pmap`.
3. Scale ladder: 8-doc smoke → 60 docs (dp_sweep convention) → ~300 docs once metrics are stable.

### P3 — extractor ladder (~2–3 days)
1. **Rung A (rules):** for each R entry, locate the replacement's mentions in `out_p`
   (exact → rapidfuzz partial-ratio → MiniLM cosine over candidate windows), substitute the original
   surface, fix determiners/inflection by rule; placeholders invert trivially.
2. **Rung B (learned residue):** flan-t5-base LoRA; input = `out_p` ⊕ linearized R pairs, target =
   clean text. Training pairs are **synthesized** (no labels needed): apply the substitutor to reference
   texts and invert — train on `(substitute(out_ref), R, out_ref)` plus identity pairs to punish
   over-editing (~10–20k pairs). bart-base ablation if time allows.
3. Measure the rung gap (B−A) on held-out tuples; **leak-through check:** attacker on `out_final`.

### P4 — eval vs InferDPT (~2–3 days)
1. `attacker.py`, adapted from `eth-sri/llm-anonymization` prompts:
   - **SynthPAI:** 8-attribute inference, top-1 accuracy vs gold — on `doc_p` and `out_final`.
   - **TAB:** masked-entity recovery (attacker guesses the original masked identifiers; scored against
     gold surfaces — the infilling-re-id formulation, ground truth available by construction).
   - Attacker models via proxy, **held out from every pipeline role** (see Role assignment).
2. **Baselines on identical docs/tasks:** (i) InferDPT (`pipeline.default`) at ε ∈ {1,3,6,10,14};
   (ii) no-privacy anchor (`doc_orig` round trip); (iii) Presidio-placeholder-only (HaS-style);
   (iv) suppression. D1 sweeps τ (3–4 operating points). Ablation arms: realistic pseudonyms in place
   of typed placeholders (marker-awkwardness cost); optional YAGO-as-*validator* check (cross-check
   teacher chains against YAGO class chains where an unambiguous alias match exists — measures the
   teacher's hallucinated-hypernym rate on the KB-covered head, complementing the NLI gate; build only
   if the NLI rejection rate looks suspicious).
3. **InferDPT task adaptation (confound, handled explicitly):** its extraction instruction is
   continuation-specific. Dual eval: (a) *native ground* — continuation on `corpora/cnndm.jsonl`,
   InferDPT unmodified; (b) *target ground* — QA/summarization, with the minimal task-generalized
   extraction instruction, quoted verbatim in the results doc. Both reported; no tuning of their
   prompt beyond that.
4. **Pareto report:** x = attacker success, y = utility (ROUGE-L/BERTScore vs `out_ref`; EM/F1 vs gold),
   one curve per method over its own knob (τ vs ε). Comparisons stated only at matched realized privacy;
   MTI probe reported as a cheap secondary on both. Results → `results/d1_eval.json` + a findings doc.

## Role assignment (models via proxy, fixed up front)

- **Teacher (lattice cache, QA generation, student training data): cascade on the local llama-swap**
  (`localhost:8060`; decision 2026-07-02) — primary `gemma 4 (E4B)`, **6 parallel slots × 32k ctx**
  (call with `chat_template_kwargs.enable_thinking=False` — it is a thinking model that otherwise
  returns empty content); NLI-rejected lattices escalate once to `Qwen3.6-35B-A3B`. The escalation
  rate is the measured "is E4B teacher-grade" number. Fully on-box — no teacher role leaves the
  machine. Role-reuse notes: E4B is also the InferDPT *baseline's* extraction model (cross-method,
  harmless); Qwen3.6 is also the remote task model — escalated lattices phrased by the model that
  consumes `doc_p` could flatter utility, checked by the second-remote-model eval arm.
- **Remote task LLM** (the "untrusted" round-trip model): `Qwen3.6-35B-A3B` (matches InferDPT baseline's
  gen model — same remote model for both methods, required for a fair comparison).
- **Evaluation attackers** (held out from all other roles): `gpt-5.5` primary, `gemini-3.1-pro-preview`
  secondary. No model plays both attacker and teacher/judge.

## Cost estimate

- **Disk:** models ~2.6 GB (GLiNER 0.6 + FCoref ~0.5 + roberta-base 0.5 + flan-t5-base 1.0 + MiniLM
  0.09) + assets ~0.05 GB (GeoNames subset, lattice cache).
- **GPU (gfx1151, one process at a time, `-u` logging):** encoder passes batch to minutes per 300 docs;
  LoRA flan-t5-base on ~20k pairs ≈ 2–4 h (perf-gate first: estimate wall-time, confirm saturation,
  bf16, max batch). Everything else is CPU or remote.
- **Remote calls (cached, threaded):** ≈300 docs × 2 tasks × (2 round trips + attacker ≈3 calls) ≈ low
  thousands total; `INFERDPT_LLM_CACHE` makes reruns free. **Teacher calls are local** (llama-swap on
  the same iGPU, E4B 6-way parallel): v0.1 lattice cache ≈ 5–10 min, full corpus + QA generation under
  ~1–2 h total, $0, respecting the one-GPU-process rule.
- **Calendar:** ~1.5–2 weeks of part-time work across P0–P4.

## Limitations and known risks

- **Detection recall is the privacy ceiling** — measured at P0, not assumed; the controlled (gold-span)
  condition isolates substitutor quality from detector quality.
- **Teacher-only lattices (decision 2026-07-02, supersedes the YAGO-hybrid):** no entity linking → no
  linking failure modes; with the teacher on the **local llama-swap**, no teacher role leaves the
  machine even at cache-build time, so the research scaffold and the deployment trust story coincide.
  What replaces the linking risk: (i) **teacher hallucination** — a fluent but false hypernym; the NLI
  gate is the mitigation and its rejection rate is a reported outcome (optional YAGO-validator arm
  cross-checks the KB-covered head if that rate looks suspicious); (ii) **teacher-model dependence** —
  lattice quality is bound to one mid-size model; the distilled-student eval later measures how much
  of it survives 35B→250M. The cache pins model id + prompt for reproducibility. Deployment inherits
  the [Dou et al. 2024](../../research-wiki/papers/dou2024_self_disclosure_abstraction.md)
  ([arXiv 2311.09538](https://arxiv.org/abs/2311.09538)) teacher→local pattern (cf. Symbolic Knowledge
  Distillation, [arXiv 2110.07178](https://arxiv.org/abs/2110.07178)). Probase/MS Concept Graph was
  audited (2026-07-02) and dropped: the 85M-pair IsA file with counts is obtainable (Internet Archive
  capture; academic-use-only, no redistribution) but 2016-vintage, lemmatized, and noisy. Open idea
  kept from the audit: typicality P(concept|instance) as an attacker-guess prior for ordering lattice
  walks — no prior anonymization work uses it.
- **Attacker traffic:** evaluation sends `doc_p`/`out_final` of *public benchmark data* to remote
  attacker models; fine here, never with real user data.
- **Licenses:** SynthPAI data CC-BY-NC-SA (research use only — no commercial artifacts derived from it);
  upstream InferDPT repo unlicensed (we use our own implementation).
- **trl GRPOTrainer is causal-only (verified)** — a D2 constraint, flagged now: RL on the seq2seq
  infiller needs either a small causal policy (Qwen-class ≤0.5B) or best-of-n/rejection-sampling
  distillation instead of GRPO.
- **Comparison fairness:** the InferDPT task adaptation is a documented confound handled by the dual
  (native + target) evaluation; TAB and SynthPAI attackers measure different things (entity recovery vs
  attribute inference) and are never merged into one number.
- **Irrecoverable precision:** tasks whose answers require the suppressed specificity lose it by design;
  reported per-task, not engineered around.
- **Scope decisions (2026-07-02, demo-prompt review):** health conditions ("diabetes") are *kept* in
  `doc_p` — they are premise, not identifier, on our corpora (no HEALTH category; revisit if medical
  text enters scope). Context-dependent sense ambiguity (Cambridge UK-vs-MA via the GeoNames
  popularity prior) is *accepted* as a rule-based ceiling: enumerating contextual exceptions is
  unbounded, and resolving them is precisely the learned substitutor's advantage — measured in D2 /
  the distilled-student eval, not patched with more rules.

## Distilled student (deferred; deployment path + D2)

The teacher cache doubles as the task-specific finetune set for the on-device lattice generator
(flan-t5-base class) that replaces the teacher cascade (E4B → Qwen3.6-35B-A3B, llama-swap) at
deployment on hardware that can't serve it — where llama-swap-class hardware exists, E4B itself is
already a small on-device teacher, so distillation below it is only warranted for genuinely
encoder-class deployment budgets. Dataset audit (2026-07-02): **no existing
dataset replaces the cache** (Dou et al.'s abstraction data: ~700 pairs, request-gated, GPT-generated;
CANDLE/AbsPyramid: right shape, common nouns/events not entities; SemEval-2018 T9: gold entity
hypernyms but context-free, ~10³ scale). **Augmentation menu for the student**, priority order:
AbsPyramid + AbsInstruct (221K in-context instance→concept, MIT/Apache,
[HF](https://huggingface.co/datasets/ZhaoweiWang/AbsPyramid), [arXiv 2311.09174](https://arxiv.org/abs/2311.09174),
[arXiv 2402.10646](https://arxiv.org/abs/2402.10646)); SemEval-2018 T9 entity splits (license unclear —
check); CANDLE (6.18M, MIT, [arXiv 2401.07286](https://arxiv.org/abs/2401.07286)); RUPTA DB-bio GPT-4
outputs. Related for D2: **SEAL** ([arXiv 2506.01420](https://arxiv.org/abs/2506.01420)) releases
adversarial anonymize→infer trajectories over SynthPAI distilled into 8B models — register in the wiki
and assess overlap before D2 planning.

## Kill criteria (from the D1 direction, made operational)

- Rung B ≤ rung A on `out_final` utility at every matched-privacy point → finding: "R + rules suffice";
  skip extractor training in D2 and report.
- Detection-union DIRECT recall so low that detected-span privacy is dominated by detector misses →
  gold-span condition becomes the headline, detector gap becomes its own result (RD2 evidence).
- D1's best τ point strictly inside the InferDPT Pareto curve (worse on both axes at matched privacy) →
  the substitution-record thesis fails on this benchmark suite; that outcome is reportable as-is.

## Sources

Plan basis: [`2026-07-02-codesign-next-stage.md`](2026-07-02-codesign-next-stage.md) and the seven-paper
analysis listed there. New registration this pass:
[Dou et al. 2024](../../research-wiki/papers/dou2024_self_disclosure_abstraction.md)
([arXiv 2311.09538](https://arxiv.org/abs/2311.09538)).
Key code artifacts (verified live 2026-07-02): [GLiNER](https://github.com/urchade/GLiNER),
[fastcoref](https://github.com/shon-otmazgin/fastcoref),
[TAB](https://github.com/NorskRegnesentral/text-anonymization-benchmark),
[SynthPAI](https://github.com/eth-sri/SynthPAI),
[eth-sri/llm-anonymization](https://github.com/eth-sri/llm-anonymization),
[sjmeis/EpsilonDistributor](https://github.com/sjmeis/EpsilonDistributor) (MTI probe),
[sjmeis/DPMLM](https://github.com/sjmeis/dpmlm),
[GeoNames dump](https://download.geonames.org/export/dump/),
[InferDPT upstream](https://github.com/mengtong0110/InferDPT) (reference only; unlicensed).
