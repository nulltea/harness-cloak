---
type: research
status: current
created: 2026-07-03
updated: 2026-07-03
tags: [pii-detection, span-detection, detector, fine-tuning, tab, deberta, gliner, privacy-filter, benchmark-honesty, architecture]
companion: docs/plans/2026-07-03-pii-span-detector-model.md
---

# Model-based PII span detection for the cloak pipeline: requirements, the architecture landscape, and the benchmark-honesty problem

The span detector is the **privacy ceiling** of the cloak pipeline: any identifier it misses
survives verbatim in `doc_p` and is handed to an LLM re-identification attacker. This doc states
what the pipeline actually requires of a detector, maps the 2026 architecture landscape against
those requirements, and records the central caution — PII detection is a data-and-evaluation
problem far more than an architecture problem, and the standard benchmarks systematically overstate
real-world detection quality. The chosen approach (two matched fine-tuning arms on TAB) and its
execution phases live in the companion plan.

**Positioning.** A durable design goal shapes the requirements below: the detector must be
**tailorable to specific user needs** — user-specified sensitive types and their generalization
lattices (a firm's project codenames, an internal role title, a domain-specific identifier), not
only the fixed TAB schema. This makes the detector a *composition* (a supervised fixed-schema core
plus a cheap per-user-type adaptation path), and it is why zero-shot and fine-tune extensibility
appear as first-class properties rather than afterthoughts.

## Definitions

- **PII** — personally identifiable information; any text span that identifies a person directly
  or in combination with other spans.
- **Direct identifier (DIRECT)** — identifies a person on its own (name, case number).
- **Quasi-identifier (QUASI)** — identifies only in combination (date, profession, location, an
  identifying event); the aggregation target of a re-identification attacker.
- **Formal PII** — the fixed catalogue every public PII model is trained on: names, emails, phone
  numbers, account/ID codes, addresses, dates. **Not** the same as QUASI: TAB's identifying events,
  demographics, and identifying quantities are absent from every formal-PII taxonomy.
- **TAB** — Text Anonymization Benchmark: 1,268 ECHR court cases with span-level DIRECT/QUASI
  annotations over 8 entity types (PERSON, ORG, LOC, DATETIME, CODE, QUANTITY, DEM, MISC). Our
  detection-gate corpus and universal entity schema — and, being real court text, our honest
  evaluation distribution.
- **Span detection** — locating character-offset spans in text and assigning each a type.
- **Token classification / BIO** — per-token label prediction (Begin/Inside/Outside) decoded into
  spans; fixed label set, arbitrary span lengths.
- **Span-enumeration / span–label matching (GLiNER family)** — score candidate spans up to a max
  width against natural-language type phrases; open-label at inference, span length capped by width.
- **BIOES + Viterbi** — a 5-tag span scheme (Begin/Inside/Outside/End/Single) decoded by a
  constrained dynamic-programming pass; OpenAI Privacy Filter's decoder.
- **Zero-shot NER** — recognizing a type from a label phrase alone, no training examples.
- **any / typed recall** — gate metrics: a gold mention counts as detected on any character overlap
  ("any"); "typed" additionally requires the predicted TAB type to match.
- **Benchmark-honesty problem** — headline PII-detector F1 (~0.96) is measured on the same synthetic
  corpora the models were fit to; on naturally-occurring text the same models score 0.18–0.65. The
  gap is the finding, not the headline.

## 1. What the cloak pipeline requires of a span detector

The detector does not stand alone. It feeds LatticeCloak's per-span generalization lattices (and,
next, the round-trip-GRPO ranker/infiller — `docs/plans/2026-07-02-roundtrip-grpo-training.md`),
and the extractor un-perturbs `out_p` from the record `R` of `(span → replacement)`. That pipeline
role fixes the properties that matter, in priority order:

1. **QUASI recall is the objective — not aggregate F1.** The re-identification attacker aggregates
   quasi-identifiers, and a missed QI span survives verbatim in `doc_p`. DIRECT recall is already
   at ceiling (§2); the entire gap is QUASI. The detector is judged on **QUASI any-recall at a
   precision floor**, never on a headline F1 that DIRECT and easy types dominate.
2. **Recall-biased error asymmetry is preferred here — safer than in plain redaction.** At a fixed
   minimum precision floor, bias toward recall: a miss is a hard privacy leak, whereas an
   over-detection is primarily a *utility* risk — measured downstream — that the lattice walk (and
   the GRPO utility term) can often absorb by choosing a shallower generalization or keeping the
   span. Not costless (a false positive can still trigger a bad substitution), so it is bounded by
   the precision floor, not licensed without limit. The redaction literature reaches the same
   conclusion for over- vs under-redaction ([zaratiana2026_gliner2_pii](../../research-wiki/papers/zaratiana2026_gliner2_pii.md)
   ([arXiv 2605.09973](https://arxiv.org/abs/2605.09973))); it holds *a fortiori* for a reversible
   substitution pipeline. **Design commitment:** the round-trip-GRPO ranker (which replaces the
   τ-walk) decides per-span whether and how far to generalize, so an over-detected non-PII span is
   ranked *keep* at zero privacy cost — a higher-recall detector is therefore strictly better *as
   input to the ranker*, more candidate spans to learn keep/generalize on. So the detector should
   maximize recall of *possible* PII and leave the keep/generalize call to the ranker (caveat in
   §5.1d: the *current* τ-walk is a leakage gate, not a false-positive filter, so until the ranker
   lands over-detection still costs utility).
3. **Lattice-compatible typing + exact character boundaries.** Each detected span is substituted by
   walking its per-type lattice, and `R` keys reconstruction on the span's boundaries. So the type
   must map onto a lattice (our 8-type TAB schema) and boundaries must be right. Coarse 8-category
   schemas that collapse IBAN/credit-card/routing into one `account_number` bucket (OpenAI Privacy
   Filter, §3) are too coarse to select a fine lattice, and sloppy boundaries corrupt `R`.
4. **Honest evaluation on the real distribution.** In-distribution synthetic F1 does not transfer
   (§4). We gate on TAB — real ECHR court text — so our number is already the honest one, and it
   predicts that off-the-shelf synthetic-trained detectors underperform the gate.
5. **Compact, local, single-forward-pass, one-GPU.** Deployment is a local per-document pipeline on
   one iGPU (one GPU process at a time). The right weight class is ~100–300M. This rules out leaning
   on a 1.5B closed model as the pipeline detector, and long court documents need windowing since
   any compact encoder's context is shorter than a full case.
6. **Tailorable to user-specified types and lattices (composable detector).** The positioning goal:
   a user brings their own notion of what is sensitive — a list of project codenames, an internal
   role title treated as a QI — *plus* the generalization lattice for it
   (`codename → "an internal project" → "a project"`). Detection and lattice are decoupled: the
   detector need only emit a type tag that keys into a lattice registry the user (or an on-the-fly
   LLM) populates. So the requirement is not "one open-vocab model" but a **composition with a cheap
   per-user-type path** — a supervised fixed-schema core for the stable TAB-8, plus a user layer
   handled by the right tier below. The pipeline is already this shape (GLiNER ∪ Presidio).
7. **Zero-shot extensibility (enables the nameable-type tier of 6).** Recognizing a user type from
   its label phrase alone, no examples — the mechanism is an open-label (span–label matching) head
   whose label set is inference-time input, not a compiled classification head. Necessary for
   pattern-free extension, but *not sufficient*: it works for world-knowledge types (employer,
   diagnosis) and fails for guideline-defined ones (our MISC 0.21) by the same label-space logic as
   §4. The sole standing argument for keeping a GLiNER-head arm.
8. **Fine-tune extensibility (enables the hard-type tier of 6).** Adding a type cheaply from a few
   user-supplied examples when zero-shot fails. Both heads support it (BIO grows a head; a GLiNER
   head fine-tunes on the new phrase and *keeps* the open interface); OpenAI Privacy Filter's
   54%→96%-from-10%-data is a fine-tune-extensibility result, not zero-shot. The fallback tier when a
   user type is idiosyncratic; literal strings need neither — a gazetteer/regex (Presidio) suffices.
9. **Attacker-grounded, not surface-grounded, as the ultimate criterion.** Span recall is the
   ceiling and a diagnostic; the reported privacy number is the LLM re-identification attacker's
   success on `doc_p` / `out_final`. Detector recall is necessary but the pipeline is judged
   downstream (project empirical-honesty rule).

Everything below is measured against this list.

## 2. The measured gap: QUASI recall, and a refuted architecture suspect

Current detector (`src/cloak/detect.py`): zero-shot GLiNER-small-v2.1 ∪ Presidio patterns. Gate on
TAB ECHR test (`results/latticecloak_detection_gate.json`, threshold 0.3):

| Metric | any-recall | typed-recall |
|---|---|---|
| DIRECT (n=407) | **0.998** | 0.980 |
| QUASI (n=6,524) | **0.857** | 0.783 |
| QUASI MISC (n=411) | 0.214 | 0.000 |
| QUASI QUANTITY (n=287) | 0.254 | 0.226 |
| QUASI DEM (n=474) | 0.563 | 0.316 |
| QUASI CODE (n=206) | 0.757 | 0.680 |

Precision proxy 0.716. DIRECT passes; the loss is concentrated in three QUASI types — exactly the
quasi-identifiers the attacker aggregates (property 1).

**Refuted hypothesis (measured 2026-07-03).** GLiNER's span-enumeration head caps spans at 12
words and TAB MISC spans are free-form clauses, so span truncation was the natural suspect. The
gold length distribution refutes it: MISC median 2 words, p90 = 7, only 4% exceed 12 words. The
failure is **semantic** — no zero-shot label phrase expresses TAB's notions of "identifying event"
(MISC), demographic quasi-identifier (DEM), or identifying quantity. These categories are defined
by TAB's annotation guidelines, not general world knowledge, so the fix is supervision on TAB's own
train split — not a different zero-shot model or prompt. This is a data problem, not an architecture
problem, and §4 shows that is the rule in this field, not the exception.

## 3. The architecture landscape (≤~1.5B), against the requirements

Four classes are live in 2026. Autoregressive LLMs as span detectors are the fifth and are
dominated: fine-tuned BERT-class encoders match 70B-class LLMs on de-identification at a fraction of
inference cost and generalize *better* across name distributions and languages
([zambare2026_deid_efficiency](../../research-wiki/papers/zambare2026_deid_efficiency.md) ([arXiv 2602.15869](https://arxiv.org/abs/2602.15869)),
EACL 2026 Findings). Rejected on properties 4 and 5.

### 3.1 Encoder + BIO token classification
A pretrained bidirectional encoder (DeBERTa-v3, RoBERTa, ModernBERT) with a per-token softmax head.
Fixed label set, arbitrary span lengths (property 3 ✓), cost independent of type count. TAB's own
baseline — Longformer fine-tuned on TAB train — reaches high DIRECT recall and good QUASI recall on
the exact corpus and metrics of our gate ([pilan2022_tab_benchmark](../../research-wiki/papers/pilan2022_tab_benchmark.md)
([arXiv 2202.00443](https://arxiv.org/abs/2202.00443))); their plain-RoBERTa ablation lost only
2–4 F1, so long-context attention is a nicety, not the mechanism. **Fixed-label PII fine-tunes** in
this class: Piiranha (`iiiorg/piiranha-v1`, **mdeberta-v3-base**, 86M backbone / ~279M total with
the multilingual embedding table, 17 formal-PII types incl Date-of-Birth but no generic DATE,
256-token context, trained on ai4privacy PII-Masking-400k), and the Kaggle PII Data Detection
winners (DeBERTa-v3 ensembles on 22k student essays).

### 3.2 GLiNER family — encoder + span–label matching
The same encoder class, but candidate spans and natural-language label phrases are embedded into a
shared space and matched by dot product — open-label, zero-shot inference (property 7 ✓, at the cost
of label-phrase sensitivity and a span-width cap). Lineage:

| | GLiNER | GLiNER2 | GLiNER2-PII | GLiGuard (Fastino) | GLiNER Guard (HiveTrace) |
|---|---|---|---|---|---|
| Ref | [arXiv 2311.08526](https://arxiv.org/abs/2311.08526) | [arXiv 2507.18546](https://arxiv.org/abs/2507.18546) | [arXiv 2605.09973](https://arxiv.org/abs/2605.09973) | [arXiv 2605.07982](https://arxiv.org/abs/2605.07982) | [arXiv 2605.05277](https://arxiv.org/abs/2605.05277) |
| Backbone | DeBERTa-v3 | DeBERTa-v3 family | GLiNER2 (adapted) | GLiNER2-base | mmBERT-small (ModernBERT) |
| Params | 50/90/300M | 205M | 0.3B | 300M | 145–209M |
| Context | 512 | 2048 | 2048 | 2048 | ~512–2048 |
| Task | NER | NER+cls+extraction | PII spans (42 types) | safety cls (no PII span) | safety + PII (joint) |
| Data | Pile-NER (240k spans, 13k types) | 254k (GPT-4o) | 4,910 synthetic multilingual | 87k WildGuard + synth | 467k multi-task |

Off-the-shelf PII fine-tunes in this family: `urchade/gliner_multi_pii-v1`, Knowledgator GLiNER-PII
(small/base/large/edge, 60+ types), and **NVIDIA GLiNER-PII** (`nvidia/gliner-PII`, a GLiNER-large backbone — sources disagree on
`knowledgator/gliner-bi-large-v1.0` vs `urchade/gliner_large-v2.1`, ~5.7×10⁸ params, trained on
Nemotron-PII: 100k US-Census-grounded synthetic records, 50+ industries, 55+ categories). All target
**formal PII**.

### 3.3 OpenAI Privacy Filter — the outlier architecture
Architecturally the most unusual public PII model
([model card](https://huggingface.co/openai/privacy-filter),
[release](https://openai.com/index/introducing-openai-privacy-filter/)): a pre-norm transformer
encoder of 8 blocks, `d_model`=640, GQA (14 query / 2 KV heads), **Sparse Mixture-of-Experts** with
128 experts and top-4 routing — **1.5B total / ~50M active**. Its lineage is the novelty: pretrained
autoregressively (gpt-oss family), then the causal mask is converted to a bidirectional banded
pattern and the LM head is replaced with a token-classification head, post-trained with supervised
loss; spans are decoded by **BIOES + constrained Viterbi** (33 labels = O + 8×4). Context window
**128K tokens** (property 5: no chunking on long docs — genuinely useful). But only **8 coarse
categories** (`account_number`, `private_address`, `private_email`, `private_person`,
`private_phone`, `private_url`, `private_date`, `secret`) — collapses IBAN/CC/routing into
`account_number` (fails property 3's granularity), and at 1.5B it is not the local single-pass
detector property 5 wants. Reported in-distribution F1 96% (94.04% P / 98.04% R) on PII-Masking-300k
(97.43% on a corrected variant); domain-adaptation F1 54% → 96% from 10% of a target set.

### 3.4 Guardrail encoders
- **HiveTrace GLiNER Guard** — a 145–209M family doing safety classification and PII spans in one
  pass, engineered for serving throughput (193 req/s, P99 < 1 s on A100; 1.6× GLiNER2)
  ([minko2026_gliner_guard](../../research-wiki/papers/minko2026_gliner_guard.md) ([arXiv 2605.05277](https://arxiv.org/abs/2605.05277))); compact
  variants run on mmBERT-small.
- **Fastino GLiGuard** — GLiNER2-base fine-tuned into a schema-conditioned safety classifier, **no
  PII span task** ([zaratiana2026_gliguard](../../research-wiki/papers/zaratiana2026_gliguard.md) ([arXiv 2605.07982](https://arxiv.org/abs/2605.07982))).

## 4. The benchmark-honesty problem (the through-line)

Every headline above is on the same synthetic corpora the models were fit to. Three of the four
common PII benchmarks (AI4Privacy, Gretel, Nemotron) are machine-generated; the only naturally
occurring one in wide use is CoNLL-2002 (Dutch newspapers, 2000). So an in-distribution F1 of ~0.96
measures how well a model generalizes from one sample of a generator to another — a weak signal.
The gap between synthetic financial text and real financial documents can exceed the gap between two
synthetic datasets. **PII detection is a high-quality-data-and-evaluation problem more than an
architecture problem.**

Two independent real-domain evaluations make the collapse concrete:

- **Tonic.ai** ([blog](https://www.tonic.ai/blog/benchmarking-openai-privacy-filter-pii-detection))
  ran OpenAI Privacy Filter over 500+ real docs (web crawl, EHR notes, legal documents, ASR
  transcripts): **F1 0.18–0.65**, almost entirely a recall failure (default recall 10% on web crawl,
  38% on EHR), against 0.92–0.99 for a production system on the same data.
- **SPY benchmark** ([zaratiana2026_gliner2_pii](../../research-wiki/papers/zaratiana2026_gliner2_pii.md)
  ([arXiv 2605.09973](https://arxiv.org/abs/2605.09973)), naturally-occurring legal + medical text):

  | System | Legal (P/R/F1) | Medical (P/R/F1) |
  |---|---|---|
  | OpenAI Privacy Filter | 0.250 / 0.640 / 0.360 | 0.271 / 0.671 / 0.386 |
  | urchade/gliner_multi_pii-v1 | 0.522 / 0.308 / 0.388 | 0.483 / 0.314 / 0.381 |

  The two baselines sit at opposite ends of one tradeoff: Privacy Filter is **high-recall,
  low-precision** (catches spans, flags many false positives — bias direction right for us,
  property 2, but precision collapse severe); `gliner_multi_pii-v1` is **high-precision, low-recall**
  (wrong direction — it misses ~70% of spans). NVIDIA GLiNER-PII shows the same distribution
  sensitivity within synthetic evals: strict-F1 Nemotron-PII 0.87 (its own data) → AI4Privacy 0.64
  → Argilla 0.70.

**Two consequences for us.** (i) Our TAB gate is already the honest evaluation the field mostly
lacks — real court text, span-level, DIRECT/QUASI. (ii) The **label-space argument** is now
independently corroborated: our gap is QUASI MISC/DEM/QUANTITY, categories absent from every
taxonomy in §3 (formal PII only; Fastino GLiGuard has no PII span task at all). Using any off-the-shelf
checkpoint means feeding TAB label phrases through a zero-shot interface — the same move as today's
detector, whose measured MISC any-recall is 0.21. PII-oriented fine-tuning sharpened these
checkpoints *toward* formal PII, away from free-form quasi-identifier clauses. Closing the gap
requires supervision on TAB train regardless of the starting checkpoint. Off-the-shelf checkpoints
still enter the Phase-0 dev sweep, so this is tested, not assumed.

## 5. The decision, given the properties: two matched fine-tuning arms

### 5.1 Scorecard against the requirements

Legend — **P1** trained on QUASI (identifying events / demographics / quantities); **P3** schema
maps onto the TAB-8 lattice at character-span resolution; **P4** holds on real (non-synthetic)
text; **P5** ~100–300M, local single-GPU, ROCm-safe; **P7** zero-shot extensible (open-label head);
**P8** fine-tune extensible. ✓ yes · ~ partial · ✗ no.

| Option | P1 QUASI | P3 →lattice | P4 real-domain | P5 local | P7 zero-shot | P8 fine-tune | Verdict |
|---|---|---|---|---|---|---|---|
| Piiranha (mDeBERTa BIO, 17) | ✗ | ~ formal→partial | ✗ synth-trained | ✓ 279M | ✗ fixed head | ✓ | reference only |
| ai4privacy DeBERTa (BIO, 54+) | ✗ | ~ formal | ✗ synth-trained | ✓ | ✗ fixed head | ✓ | reference only |
| urchade gliner_multi_pii-v1 | ✗ | ~ formal | ✗ SPY F1 .38, P≫R (wrong bias) | ✓ | ✓ | ✓ | Phase-0 rung |
| **knowledgator/gliner-pii-base-v1.0** | ✗ | ~ 60+ formal | ~ best off-shelf on TAB dev (QUASI .898) | ✓ ~base | ✓ | ✓ | **best off-shelf; recommended Arm-B init (§5.1b)** |
| NVIDIA GLiNER-PII (GLiNER-large) | ✗ | ~ 55 formal | ✗ drops across evals | ~ 570M | ✓ | ✓ | rung (bigger) |
| OpenAI Privacy Filter (MoE) | ✗ | ✗ 8 coarse | ✗ .18–.65 (recall-bias ✓, P collapse) | ✗ 1.5B MoE | ✗ fixed BIOES | ✓ 54→96 | high-recall reference only |
| GLiNER2-PII (0.3B, open) | ✗ | ~ 42 formal | ~ best-on-SPY, recall-favoring | ✓ 0.3B | ✓ | ✓ | strong open-label init candidate |
| HiveTrace GLiNER Guard | ✗ | ~ formal | ? untested here | ~ ModernBERT ROCm risk | ✓ | ✓ | no (ROCm + safety baggage) |
| **Arm A — DeBERTa-v3-base + BIO on TAB** | ✓ trained | ✓✓ TAB-8 native | ✓ TAB is real | ✓ ~184M, ROCm-safe | ✗ fixed head | ✓ retrain | **TEST — fixed-schema core** |
| **Arm B — gliner_small-v2.1 FT on TAB** | ✓ trained | ✓ TAB-8 | ✓ TAB is real | ✓ 50–90M | ✓ open iface retained | ✓ | **TEST — user-extensible core** |

**Conclusion — the 1–3 options to test.** The scorecard collapses to one fact: **nothing
off-the-shelf clears P1** — none is trained on QUASI, so every off-the-shelf row is a reference or a
rung, not a solution, exactly as the label-space argument (§4) predicts. The real candidates are the
two supervised arms, and the tailorability requirement (P6–P8) makes Arm B non-optional rather than
a flourish:

1. **Arm A — DeBERTa-v3-base + BIO on TAB** *(test first)*: the strongest single bet for QUASI
   recall (P1), TAB-8 native (P3), ROCm-safe (P5), simplest loss. It misses P7 (fixed head) and
   satisfies P6 only as the fixed-schema *core* — user-defined types ride the separate cheap path
   (gazetteer, or a unioned open-label pass), which the pipeline already provides.
2. **Arm B — gliner_small-v2.1 fine-tuned on TAB** *(test in parallel)*: the matched arm; the only
   candidate that *also* satisfies P7 and therefore covers the composable/user-tailorable goal (P6)
   in a single model. If it matches Arm A on QUASI recall at the precision floor, it wins outright on
   tailorability.
3. **Off-the-shelf reference rung** *(Phase 0, dev-only sweep)*: run the listed checkpoints
   (`gliner_multi_pii-v1`, Knowledgator GLiNER-PII, Piiranha, a GLiNER Guard variant) through the
   gate on **dev** to *measure* the P1 gap rather than assert it, and carry **one** high-recall
   checkpoint (OpenAI Privacy Filter, the recall-biased extreme, or `gliner_multi_pii-v1`) through
   the test gate as the single reported reference.

Follow-up only if Arm B wins: a stronger open-label init — **GLiNER2-base** (broad open-generality)
or **GLiNER2-PII** (recall-favoring PII training, SPY-validated) — as a single additional arm.

### 5.1b Phase-0 dev sweep — measured (2026-07-03)

Ran the off-the-shelf rung on **TAB dev** (127 docs, threshold 0.3, fixed TAB label phrases held
constant, ∪ Presidio), via the existing `Detector` (`--gliner-model` sweep;
`results/pii_dev_sweep_*.json`). GLiNER Guard excluded (needs a forked `gliner2` bi-encoder loader
on an mmBERT/ModernBERT backbone — the ROCm risk); Piiranha and OpenAI Privacy Filter excluded (not
loadable by this detector, and cannot emit MISC/DEM/QUANTITY).

| Model | DIRECT any | QUASI any | MISC | DEM | QUANTITY | prec. proxy |
|---|---|---|---|---|---|---|
| `gliner_small-v2.1` (control) | 0.993 | 0.865 | 0.202 | 0.528 | 0.319 | 0.713 |
| `gliner_multi_pii-v1` | 0.970 | 0.850 | 0.265 | 0.532 | 0.439 | 0.804 |
| **`knowledgator/gliner-pii-base-v1.0`** | 0.998 | **0.898** | **0.316** | **0.585** | **0.684** | 0.746 |
| `nvidia/gliner-PII` | 1.000 | 0.813 | 0.141 | 0.495 | 0.199 | **0.958** |

What the data says (some of it against the pre-registration above):
- **P1 gap confirmed and now measured** — every checkpoint leaves MISC ≤ 0.32 and DEM ≤ 0.59, far
  below the QUASI-any ≥ 0.95 goal. No off-the-shelf model clears P1; TAB supervision remains required.
- **The "formal-PII fine-tuning adds nothing on the gap" claim is falsified.** Knowledgator
  gliner-pii-base lifts the *baseline general-NER* checkpoint on exactly the gap types — QUANTITY
  0.319 → 0.684, MISC 0.202 → 0.316, DEM 0.528 → 0.585, overall QUASI 0.865 → 0.898 — at comparable
  precision (0.746 vs 0.713). A formal-PII fine-tune materially helped the quasi-identifier types.
- **The "conservative-recall risk" is per-model, not a property of PII fine-tuning.** It appears for
  `gliner_multi_pii-v1` (DIRECT 0.97, ORG 0.84) and extremely for `nvidia/gliner-PII` (precision
  0.958 but MISC 0.141, worst QUASI) — but not for Knowledgator.
- **Benchmark-honesty, refined.** These GLiNER checkpoints do *not* collapse on TAB (real court text)
  the way OpenAI PF collapsed on Tonic; Knowledgator reaches QUASI 0.898. The collapse is
  concentrated in the guideline-defined types (MISC above all), not across the board — a more precise
  statement of §4 than "real text ⇒ collapse."

**Consequence for the arms (confirmed; Arm B trained — §5.3).** `knowledgator/gliner-pii-base-v1.0` dominates
the planned Arm-B init (`gliner_small-v2.1`) on QUASI recall at comparable precision, keeps the
open-label interface (P6/P7), and its base size fits P5 — so it is the recommended Arm-B init, with
`gliner_small-v2.1` demoted to the clean-span-pretraining control if a strict ablation is wanted.
Trade-off: this makes the Arm-A-vs-B contrast "BIO fixed-schema vs best open-label init" rather than
a span-pretraining-only isolation.

### 5.1c P7 zero-shot generality — pre-fine-tuning baseline (measured 2026-07-03)

To test P7 (does the open-label head find user-defined, out-of-schema types from a label phrase
alone?), measured zero-shot recall of the two candidate inits on **MultiNERD-en held-out types
outside TAB-8** — animal, disease, plant, food, media, vehicle, celestial, instrument, mythological,
biological — as a labeled proxy for "user-defined nameable types" (fixed label phrases;
`scripts/pii_zeroshot_generality.py`, `results/pii_zeroshot_generality_*.json`; n = 4,406 held-out
gold mentions over 2,896 sentences). This is the **pre-fine-tuning half** of the retention experiment.

| Init | overall any | typed | ANIM | DIS | PLANT | FOOD | MEDIA |
|---|---|---|---|---|---|---|---|
| `gliner_small-v2.1` (general NER) | 0.845 | 0.732 | 0.92 | 0.86 | 0.86 | 0.77 | 0.61 |
| **`knowledgator/gliner-pii-base-v1.0`** | **0.941** | **0.821** | 0.95 | 0.94 | 0.94 | 0.92 | 0.92 |

- **Open-label zero-shot generality is strong pre-FT** (any-recall 0.85–0.94) — the mechanism works
  for nameable world-knowledge user types before any TAB tuning.
- **Knowledgator beats gliner_small on every held-out type** (overall 0.941 vs 0.845; MEDIA 0.92 vs
  0.61). PII fine-tuning did *not* erode open-label generality — it improved it. Two-for-two with the
  Phase-0 QUASI result: `knowledgator/gliner-pii-base-v1.0` is the recommended Arm-B init on both axes.
- Precision is not cleanly measurable here (sentences carry non-held-out gold types we don't query),
  so recall is the reported signal.

**Pending — the retention delta (needs Arm B trained, Phase 2).** Re-run this probe on the
TAB-fine-tuned Arm B: a large drop means TAB-8 fine-tuning narrowed the model and P7/P6 is lost (Arm
B collapses toward Arm A); a small drop confirms tailorability survives fine-tuning. That delta is
the decisive P7 number — this baseline only establishes the "before."

### 5.1d Over-detection is the ranker's job, not the detector's (2026-07-03)

Verbatim `doc_orig → doc_p` spot-check via the full Substitutor (`scripts/spikes/pii_docp_examples.py`,
`results/pii_docp_examples.txt`), knowledgator vs `gliner_small`, on one clinical / email / social doc:
- **clinical** — knowledgator keeps the `[doctor]`/`[patient]` turn-markers that `gliner_small`
  mangles into `[a medical practitioner]`/`[a sick person]`, at the same real-PII coverage (name →
  `<PERSON_2>`, `50-year-old` → `fifty-something`). Utility win.
- **email** — knowledgator generalizes a `Commission` (ORG) that `gliner_small` leaves verbatim.
  Privacy win.
- **social (synthpai)** — on a poetic post with no PII, knowledgator over-generalizes generic nouns:
  `Staircases → "a way"`, `skies → "an atmosphere"`, `labyrinths → "a system"`; `gliner_small`
  (which detects nothing here) leaves it intact. Utility loss.

Root cause of the over-generalization: MISC fires just over threshold (0.318–0.353 at 0.3);
`substitute()` then generalizes via WordNet hypernyms; and the τ guess-back gate returns **0.0 risk**
for a non-PII span (nothing to re-identify), so it cannot reject it — **τ is a leakage guard, not a
false-positive filter.** Once a span is detected, the current pipeline always rewrites it.

**Decision:** this is *not* a detector defect to clamp. The round-trip-GRPO ranker will learn to keep
such spans (P2), so maximizing detector recall gives the ranker more to learn from — the better the
model is at surfacing *possible* PII, the more the ranker can generalize. Recall-bias is vindicated.
A per-type MISC threshold or dropping MISC on non-legal corpora stays a *stopgap* only if
over-redaction bites before the ranker lands.

### 5.2 Head vs initialization

Properties 1–5 point at a compact encoder fine-tuned on TAB. The one open question is a genuine
empirical fork that no published work resolves at matched data on a quasi-identifier corpus — and it
splits into two independent choices, not the "DeBERTa vs GLiNER" framing an earlier draft used
(GLiNER, GLiNER2, GLiNER2-PII, and Fastino GLiGuard all run on the *same DeBERTa-v3 spine* we would
fine-tune):

1. **Head** — BIO token classification (fixed 8 TAB labels) vs span–label matching (open-label,
   property 7).
2. **Initialization** — raw MLM backbone vs a GLiNER checkpoint whose backbone is already trained
   for open-type span extraction (Pile-NER / NuNER: 240k spans, 13k types).

**The case for GLiNER init:** zero-shot it already reaches 0.857 QUASI any-recall — it largely
*finds* spans; fine-tuning mostly teaches TAB's label semantics and boundaries, so with only 1,014
training docs the head start plausibly buys sample efficiency. **The case for clean DeBERTa-v3-base:**
TAB's own baseline trained RoBERTa/Longformer (no span-task pretraining) on the same 1,014 docs and
reached good QUASI recall, so raw init is *proven adequate* on exactly our corpus; TAB is densely
annotated (tens of thousands of mention spans), ample for 8 types; and no inherited LLM-annotated
boundary priors (which can help or fight TAB's guideline-specific conventions).

Neither side has decisive evidence, and a run costs ~1–2.5 h, so the plan runs **both as matched
arms** (identical TAB windows, identical dev-based selection, no per-model knobs). Protocol: the
detector threshold is set by one predeclared rule (dev QUASI any-recall at the precision floor)
applied identically to both arms; the Arm A vs Arm B comparison is read **at matched precision** (a
precision–recall curve), and threshold sweeps are diagnostic only — never a per-model fudge to
equalize precision (the CLAUDE.md calibration prohibition).

- **Arm A** — `microsoft/deberta-v3-base` + fresh BIO head. Proven-adequate recipe, unbounded span
  lengths, plain PyTorch (ROCm-safe), simplest loss. Supported by direct evidence that plain
  fine-tuning beats architectural complexity: on an 82-type PII benchmark a directly fine-tuned
  DeBERTa token classifier with weighted cross-entropy (span F1 0.648) beat source-conditioned
  (0.590) and curriculum (0.277) variants ([jha2026_piibench_deberta](../../research-wiki/papers/jha2026_piibench_deberta.md)
  ([arXiv 2605.25816](https://arxiv.org/abs/2605.25816))).
- **Arm B** — `urchade/gliner_small-v2.1` (NuNER-trained plain NER checkpoint) fine-tuned via the
  gliner library. Span-pretrained init; keeps the open-label interface (property 7). If it matches
  Arm A on the gate, it wins on flexibility.

Honest caveat: none of the cited evidence compares these two *heads* at matched data on a
quasi-identifier corpus. PIIBench/Kaggle show plain fine-tuning beats added complexity, not that BIO
beats a GLiNER head.

**Why neither Guard nor OpenAI Privacy Filter is the initialization/detector:**
- *HiveTrace GLiNER Guard*: mmBERT/ModernBERT backbone (the flash-attention/unpadding path is the
  known ROCm (gfx1151) risk here); joint-safety supervision is dead weight and an interference
  source; PII training is formal-PII; its serving innovations target throughput, not our bottleneck.
- *Fastino GLiGuard*: safety specialist, no span task.
- *OpenAI Privacy Filter*: open-weight (Apache 2.0) and fine-tunable in principle, but 1.5B MoE —
  heavier than P5's ~100–300M local target — with an 8-coarse-category schema (fails P3); its value
  (128K context, high-recall bias) is real, but the real-domain precision collapse (§4) and coarse
  schema make it a downstream-filter/reference candidate, not the core pipeline detector.
- *Off-the-shelf PII fine-tunes as init* (`gliner_multi_pii-v1`, NVIDIA GLiNER-PII, Knowledgator
  GLiNER-PII): **superseded by the Phase-0 measurement (§5.1b).** The pre-registered worry that
  formal-PII fine-tuning "adds nothing on the gap" was falsified — `knowledgator/gliner-pii-base-v1.0`
  is the best off-the-shelf QUASI start and is now the recommended Arm-B init. Two caveats survive:
  (a) a PII-fine-tuned init confounds span-pretraining with PII-fine-tuning, so it is not a clean
  span-pretraining-only ablation — keep `gliner_small-v2.1` as the control if that isolation is
  wanted; (b) `gliner_multi_pii-v1` and `nvidia/gliner-PII` show a conservative high-precision /
  low-recall bias (wrong for P2), so *not every* PII fine-tune is a good init — Knowledgator is.
- If a stronger GLiNER init is wanted, **GLiNER2-base** (2048-token context, newer data) is the
  follow-up arm — only if Arm B beats Arm A, to avoid a three-arm first experiment.

### 5.3 Arm B — measured (2026-07-03)

Trained `knowledgator/gliner-pii-base-v1.0` on TAB train (11,022 windows, 3 epochs, ~14.5 min on the
gfx1151 iGPU; `scripts/train_pii_gliner.py`). Epoch-2 (`checkpoint-2756`) selected on the dev gate;
operating threshold **0.02** fixed on the dev Pareto (max QUASI any-recall at precision ≥ 0.716,
`results/arm_b_dev_thr_*.json`). Single test-set run (`results/latticecloak_detection_gate_arm_b.json`):

| | DIRECT any | QUASI any | prec | MISC | DEM | QUANT | CODE |
|---|---|---|---|---|---|---|---|
| baseline `gliner_small` zero-shot | 0.998 | 0.857 | 0.716 | 0.214 | 0.563 | 0.254 | 0.757 |
| **Arm B `knowledgator`+TAB** | **1.000** | **0.971** | **0.861** | **0.856** | **0.951** | **0.972** | **0.985** |

**All success criteria met on test:** DIRECT any ≥ 0.99 (1.000), QUASI any ≥ 0.95 goal (**0.971**),
precision ≥ 0.716 (0.861). The gap types are transformed — MISC 0.21→0.86, DEM 0.56→0.95, QUANT
0.25→0.97 — and dev→test transfer is clean (dev QUASI 0.958 → test 0.971, no overfit). This is the
result the whole plan targeted: supervision on TAB closes the QUASI gap no off-the-shelf checkpoint could.

**P7 retention delta — the tailorability cost (measured).** Zero-shot recall on out-of-schema held-out
types (the §5.1c MultiNERD-en probe, `results/pii_zeroshot_generality_arm_b.json`) dropped from
**0.941 (pre-FT) to 0.835 (post-FT)** — back to the general-NER baseline level (`gliner_small` 0.845).
So TAB-8 fine-tuning **erodes** knowledgator's pre-FT open-label generality edge: the open interface
survives and user-defined nameable types are still found zero-shot at *general-NER* quality, but the
pre-FT advantage that made knowledgator two-for-two (§5.1c) is spent. **Tailorability is preserved, not
enhanced** — the P6/P7 case for the GLiNER head over a plain BIO head is weakened by this result
(fine-tuned, it is no longer better-than-general at zero-shot). Fine-tune extensibility (P8) remains the
reliable path for new user types. This is the honest trade: a decisive QUASI win bought at the pre-FT
generality edge. Arm A (DeBERTa-BIO) has not been run, so the head-vs-head comparison is still open.

### 5.4 Choosing the detection threshold — tradeoffs and guidelines

The GLiNER span-confidence threshold is the detector's operating-point knob. It is *not* a privacy
budget and it is *not* a per-method fudge (honesty rule) — it is chosen once, per **(model, corpus)**
pair, on a dev/held-out slice, then fixed. The tradeoff is monotone (measured on the Arm-B dev
Pareto, §5.3): lower threshold → more spans.

| Lower threshold (≈0.01–0.05) | Higher threshold (≈0.3+) |
|---|---|
| ↑ **recall** — fewer missed QUASI, i.e. fewer *hard privacy leaks* | ↓ recall — more leaks |
| ↓ **precision** — over-detection | ↑ precision — cleaner spans |
| more candidate spans for the RL ranker to learn `keep`/generalize on | less for the ranker, but less over-redaction under the current τ-walk |
| **floods** on uncalibrated / out-of-domain text | robust across domains |

Evidence: Arm-B on TAB dev — thr 0.01→0.3 moved QUASI any 0.969→0.867 and precision 0.826→0.947
(MISC 0.87→0.40). Out-of-domain (§5.3 non-TAB probe) — TAB's 0.02 floods clinical/social text, while
0.3 is clean *and still transfers* the QUASI wins; so the flooding is a threshold artifact, not lost
knowledge.

**Choose the threshold based on:**
1. **Model calibration.** A TAB-fine-tuned model is *sharp and confident* → needs a **low** threshold
   to reach recall (Arm B: 0.02 for QUASI 0.958). A zero-shot GLiNER is diffuse → operates ~**0.3**.
   Never reuse a fine-tuned model's threshold on a zero-shot model or vice versa.
2. **Domain match.** In-domain (trained distribution) tolerates the model's low threshold; **out-of-domain
   needs a higher one** (a per-corpus operating point — TAB 0.02 ≠ clinical/social, use ~0.3 there).
3. **Downstream absorber.** With the RL ranker (which learns `keep`, §5.1d), bias **lower** — over-detection
   is recovered at zero privacy cost, so maximize recall. With only the τ-walk (no ranker yet),
   respect a **precision floor** — over-detection costs utility today (the synthpai over-redaction).
4. **Privacy vs utility priority.** Privacy-ceiling framing (a miss is a hard leak) → lean **lower**;
   utility-sensitive / over-redaction-averse → lean **higher**.

**Selection rule (predeclared):** on dev, pick the **lowest** threshold whose precision ≥ the 0.716
baseline floor — this maximizes QUASI any-recall subject to the floor. Fix it; reuse only within the
same (model, corpus). Threshold sweeps are diagnostic — never tuned per method-comparison.

**Measured defaults:** Arm-B (`knowledgator`+TAB) on TAB → **0.02** (QUASI 0.971 test); zero-shot
GLiNER (stock, or any non-TAB corpus) → **0.3**.

## 6. Training data

| Dataset | Size / domain | Labels | Access | Role |
|---|---|---|---|---|
| TAB train/dev ([pilan2022_tab_benchmark](../../research-wiki/papers/pilan2022_tab_benchmark.md), [arXiv 2202.00443](https://arxiv.org/abs/2202.00443)) | 1,014 / 127 ECHR cases | 8 TAB types, DIRECT/QUASI, multi-annotator | local (`corpora/tab/`) | primary supervision + model/threshold selection |
| [ai4privacy pii-masking-400k](https://huggingface.co/datasets/ai4privacy/pii-masking-400k) | 407k synthetic, 6 langs | 17 formal-PII types | HF (academic use) | auxiliary (English slice, 17→8 map, capped ≈1:1 by tokens) |
| Kaggle PII / [PIILO](https://the-learning-agency-lab.com/learning-exchange/piilo-dataset/) | ~22k student essays | 7 formal-PII types | Kaggle | optional auxiliary |
| i2b2/n2c2 2014 de-id | clinical notes | PHI types | DUA-gated (weeks) | skipped for now |

Gold construction matches the gate: union of annotators' DIRECT/QUASI mentions (strictest privacy
ceiling). Auxiliary types map onto TAB's coarse 8; unmappable types are dropped, never invented. The
auxiliary sets are **formal-PII only** — they add robustness for PERSON/CODE/LOC but contribute
nothing for MISC/DEM/QUANTITY; TAB train is the only supervision for the actual gap.

## 7. Status and next step

Decision recorded: two matched fine-tuning arms on TAB train (Arm A: DeBERTa-v3-base + fresh BIO
head; Arm B: gliner_small-v2.1 fine-tuned) — identical data windows, identical dev-based selection,
no per-model knobs. Presidio union retained; zero-shot GLiNER retained for non-TAB corpora until
transfer is measured. Execution phases, success criteria (DIRECT any ≥ 0.99 held, QUASI any ≥ 0.95
goal, precision ≥ 0.716 baseline), and test-set discipline (iterate on dev, test once per final
config) are in the companion plan (`docs/plans/2026-07-03-pii-span-detector-model.md`). The Phase-0
off-the-shelf dev sweep **has run** (§5.1b, measured), as has the **P7 zero-shot generality baseline**
(§5.1c); together they confirmed the P1 gap and put `knowledgator/gliner-pii-base-v1.0` ahead on both
QUASI recall and out-of-schema generality — the chosen Arm-B init. **Arm B has now trained and been
evaluated (§5.3, measured):** test QUASI any **0.971** (≥0.95 goal met), DIRECT any 1.000, precision
0.861, at threshold 0.02 fixed on the dev Pareto — the QUASI gap is closed. The P7 retention delta is
also measured: zero-shot generality fell 0.941→0.835 (fine-tuning spent knowledgator's pre-FT edge;
tailorability preserved at general-NER quality, not enhanced). Arm A (DeBERTa-BIO) has not been run —
the head-vs-head comparison remains open.

_Off-the-shelf model/architecture/benchmark specifics (OpenAI Privacy Filter, the GLiNER family,
NVIDIA GLiNER-PII, Piiranha, and the SPY / Tonic.ai numbers) were verified against model cards and
papers on 2026-07-03._
