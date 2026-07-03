---
type: plan
status: current
created: 2026-07-03
updated: 2026-07-03
tags: [pii-detection, span-detection, detector, fine-tuning, tab, deberta, gliner]
companion: docs/plans/2026-07-02-d1-prototype-implementation.md
---

# Model-based PII span detector: fine-tuning plan

The span detector is the privacy ceiling of the whole cloak pipeline: any identifier it misses
survives verbatim in `doc_p`. This plan replaces the zero-shot detection stage with a supervised,
size-optimal model, based on the literature review of 2026-07-03 (papers registered in
`research-wiki/papers/`).

## 1. Baseline and gap

Current detector (`src/cloak/detect.py`): zero-shot GLiNER-small-v2.1 ∪ Presidio patterns, mapped
to TAB types. Gate result (`results/latticecloak_detection_gate.json`, TAB ECHR test, 127 docs,
threshold 0.3):

| Metric | any-recall | typed-recall |
|---|---|---|
| DIRECT (n=407) | **0.998** | 0.980 |
| QUASI (n=6,524) | **0.857** | 0.783 |
| QUASI MISC (n=411) | 0.214 | 0.000 |
| QUASI QUANTITY (n=287) | 0.254 | 0.226 |
| QUASI DEM (n=474) | 0.563 | 0.316 |
| QUASI CODE (n=206) | 0.757 | 0.680 |

Precision proxy 0.716. DIRECT passes the gate; the gap is QUASI recall — exactly the
quasi-identifiers an LLM re-identification attacker aggregates.

**Diagnostic (2026-07-03, refuted hypothesis):** GLiNER's max span width (12 words) is *not* the
cause — only 4% of test-set MISC gold spans exceed 12 words (median 2, p90 7). The failure is
semantic: zero-shot label phrases cannot express TAB's notions of "identifying event" (MISC),
demographic quasi-identifier (DEM), or identifying quantity. This is a training-data problem, not
an architecture problem.

## 2. Research summary — architectures

Full paper notes in `research-wiki/papers/`; key conclusions:

- **Fine-tuning a compact encoder beats architectural complexity.** On an 82-type multi-source PII
  benchmark, directly fine-tuned DeBERTa token classification with weighted cross-entropy beats
  source-conditioned hierarchical and curriculum variants by 6–37 F1 points
  ([jha2026_piibench_deberta](../../research-wiki/papers/jha2026_piibench_deberta.md)
  ([arXiv 2605.25816](https://arxiv.org/abs/2605.25816))). The Kaggle PII Data Detection
  competition (22k student essays) was likewise won by fine-tuned DeBERTa-v3 token classifiers.
- **TAB's own baseline proves the recipe in-domain.** The TAB paper fine-tunes Longformer for token
  classification on the train split and reaches high recall on direct identifiers and good recall
  on quasi-identifiers — on the same corpus and schema as our gate
  ([pilan2022_tab_benchmark](../../research-wiki/papers/pilan2022_tab_benchmark.md)
  ([arXiv 2202.00443](https://arxiv.org/abs/2202.00443))).
- **Compact encoders beat LLM detectors on cost *and* generalization.** Fine-tuned BERT-class
  models match 70B-class LLMs on de-identification at a fraction of inference cost and generalize
  better across name distributions and languages
  ([zambare2026_deid_efficiency](../../research-wiki/papers/zambare2026_deid_efficiency.md)
  ([arXiv 2602.15869](https://arxiv.org/abs/2602.15869)), EACL 2026 Findings). LLM-as-detector is
  rejected.
- **The right weight class is ~100–200M.** Production span-level PII detection runs at 145–209M
  params (GLiNER Guard family,
  [minko2026_gliner_guard](../../research-wiki/papers/minko2026_gliner_guard.md)
  ([arXiv 2605.05277](https://arxiv.org/abs/2605.05277))); GLiNER2 stays under 500M for multi-task
  extraction ([zaratiana2025_gliner2](../../research-wiki/papers/zaratiana2025_gliner2.md)
  ([arXiv 2507.18546](https://arxiv.org/abs/2507.18546))). Nothing larger is justified before the
  data-side gap is closed.
- **GLiNER-family = same encoder + span–label matching head.** Retains an open-label interface and
  has off-the-shelf PII fine-tunes (urchade/gliner_multi_pii-v1, Knowledgator GLiNER-PII,
  NVIDIA GLiNER-PII)
  ([zaratiana2023_gliner](../../research-wiki/papers/zaratiana2023_gliner.md)
  ([arXiv 2311.08526](https://arxiv.org/abs/2311.08526))). But existing PII fine-tunes target
  formal PII (emails, phones, IDs), not TAB-style quasi-identifiers, so they are a cheap
  first rung, not the expected fix.

**Decision (revised 2026-07-03 after review — full reasoning in
`docs/research/learned-PII-detection.md` §3):** the choice splits into head (BIO token
classification vs GLiNER span–label matching) and initialization (raw MLM backbone vs
span-task-pretrained GLiNER checkpoint); no published evidence separates them at matched data on a
quasi-identifier corpus, and a training run costs ~1–2.5 h, so run **two matched arms**:
- **Arm A**: `microsoft/deberta-v3-base` + fresh BIO head over the 8 TAB types — proven-adequate
  recipe (TAB's own baseline), unbounded span lengths, plain PyTorch (ROCm-safe), simplest loss.
- **Arm B**: `urchade/gliner_small-v2.1` fine-tuned via the gliner library — span-pretrained
  initialization (Pile-NER/NuNER), keeps the open-label interface for non-TAB corpora; if it
  matches Arm A on the gate it wins on flexibility.
Identical training windows, identical dev-based selection rule, no per-model knobs. Neither Guard
is an initialization candidate (HiveTrace: mmBERT/ModernBERT ROCm risk + safety multi-task
baggage; Fastino GLiGuard: safety specialist, no span task). GLiNER2-base (2048-token context) is
a follow-up arm only if Arm B wins. ModernBERT-base (8k context) remains the long-context fallback
only if window-boundary effects measurably hurt.

## 3. Research summary — datasets

| Dataset | Size / domain | Labels | Access | Role |
|---|---|---|---|---|
| TAB train/dev ([arXiv 2202.00443](https://arxiv.org/abs/2202.00443)) | 1,014 / 127 ECHR court cases | 8 TAB types, DIRECT/QUASI, multi-annotator | **already local** (`corpora/tab/`) | primary supervision + selection |
| [ai4privacy pii-masking-400k](https://huggingface.co/datasets/ai4privacy/pii-masking-400k) | 407k synthetic entries, 6 langs, multi-domain | 17 PII types (span-annotated) | HF; academic use with citation | auxiliary multi-domain mix (English slice, 17→8 type map) |
| Kaggle PII Detection / [PIILO](https://the-learning-agency-lab.com/learning-exchange/piilo-dataset/) | ~22k student essays | 7 formal PII types | Kaggle download | optional aux (formal PII only) |
| i2b2/n2c2 2014 de-id | clinical notes | PHI types | DUA-gated, weeks of lead time | skipped for now |

Gold construction matches the gate: union of annotators' DIRECT/QUASI mentions (strictest recall).
ai4privacy/PIILO types map onto TAB's coarse 8 (names→PERSON, emails/phones/IDs→CODE, …);
unmappable types are dropped, never invented.

## 4. Implementation plan

Artifacts named by method (no plan identifiers). All runs in the host `.venv`
(GPU direct, one GPU process at a time, `python -u` for long jobs).

**Phase 0 — off-the-shelf rung + honest split hygiene** — **DONE 2026-07-03**
- Gate gained a `--gliner-model` arg; dev iteration via `--corpus corpora/tab/echr_dev.json` (test
  run once per final config). Split hygiene fixed.
- Swept 4 GLiNER-format checkpoints on dev (`results/pii_dev_sweep_*.json`; fixed TAB label phrases,
  ∪ Presidio): `gliner_small-v2.1` (control), `gliner_multi_pii-v1`, `knowledgator/gliner-pii-base-v1.0`,
  `nvidia/gliner-PII`. Piiranha / OpenAI PF excluded (not loadable by this detector, can't emit
  MISC/DEM/QUANTITY); GLiNER Guard excluded (forked `gliner2` bi-encoder loader + mmBERT ROCm risk).
- **Result (measured, full table in report §5.1b):** P1 gap confirmed — no checkpoint clears MISC
  (≤0.32) / DEM (≤0.59); TAB supervision still required. **But the "formal-PII fine-tuning adds
  nothing on the gap" expectation was falsified**: `knowledgator/gliner-pii-base-v1.0` beats the
  general-NER baseline on QUASI (0.898 vs 0.865) and on the gap types (QUANTITY 0.319→0.684, MISC
  0.202→0.316) at comparable precision. **Pending decision:** adopt `knowledgator/gliner-pii-base-v1.0`
  as the Arm-B init in place of `gliner_small-v2.1` (which becomes the clean-ablation control).
- **P7 zero-shot generality baseline** (`scripts/pii_zeroshot_generality.py`,
  `results/pii_zeroshot_generality_*.json`): zero-shot recall on MultiNERD-en held-out types outside
  TAB-8 (proxy for user-defined types). knowledgator-pii-base 0.941 vs gliner_small 0.845 overall any
  — PII fine-tuning did *not* erode open-label generality (falsified the erosion hypothesis), so
  knowledgator is best on both P1 and P7. The decisive **retention delta** (re-run this probe on the
  TAB-fine-tuned Arm B) is pending Phase 2 — a large drop = P7/P6 lost; small drop = tailorability survives.

**Phase 1 — data prep: `scripts/build_pii_span_dataset.py`** (~0.5 day)
- TAB train/dev → tokenizer windows (512 tokens, stride 128) → BIO tags over the 8 TAB types from
  union-of-annotators char offsets; window-boundary spans assigned to the window containing their
  start, overlap resolved at decode time by score.
- Optional `--mix ai4privacy` flag emitting the mapped English slice (capped so TAB : aux ≈ 1 : 1
  by token count — TAB stays the dominant signal).
- Output: JSONL under `data/pii_span_dataset/`; one self-check assert (round-trip
  offsets→BIO→spans reproduces gold on a sample doc).

**Phase 2 — training, two matched arms** (~1 day + runs)
- **Arm A** `scripts/train_pii_token_classifier.py`: `microsoft/deberta-v3-base` +
  token-classification head (17 BIO labels), weighted cross-entropy (weight ∝ inverse type
  frequency; the one ingredient PIIBench evidence supports), bf16, batch as large as VRAM allows,
  HF `Trainer`. Artifact: `data/models/pii_token_classifier/`.
- **Arm B** `scripts/train_pii_gliner.py`: `urchade/gliner_small-v2.1` fine-tuned via the gliner
  library on the *identical* windows (spans as gliner-format records, TAB label phrases fixed once
  before training). Artifact: `data/models/pii_gliner/`.
- Shared rules: identical train/dev windows; checkpoint selection on **dev** by QUASI any-recall
  at precision ≥ 0.70 (fallback: span-F2); one GPU process at a time — run arms sequentially.
- Wall-time estimate per arm: ~10k windows/epoch × 3–5 epochs on gfx1151 ≈ 1–2.5 h → exceeds the
  10-min perf-gate line, so confirm GPU saturation on a 200-step probe before each full run
  (`scripts/harness/perf_gate.md` review).

**Phase 3 — integration: `src/cloak/detect.py`** (~0.5 day)
- Arm A: new `TokenClassifierDetector` producing the same `Span` list (start/end/text/type/score),
  windowed inference with overlap-max decoding; union with Presidio (pattern types are free
  precision for CODE) and the existing `_dedupe`.
- Arm B: no new code — the existing `Detector` takes the fine-tuned checkpoint path as
  `gliner_model`.
- Keep the zero-shot GLiNER `Detector` untouched as the non-TAB-corpora fallback until transfer is
  measured.

**Phase 4 — evaluation** (~0.5 day)
- Dev: threshold Pareto (precision vs DIRECT/QUASI any-recall) for the selected checkpoint.
- Test: **one** run of `scripts/latticecloak_detection_gate.py` per final config (each arm solo,
  each arm ∪ Presidio) — the Arm A vs Arm B head-to-head is reported from these single runs.
  Report per-type recall deltas vs baseline table above; state degeneracies plainly if any type
  regresses.
- Success criteria: DIRECT any ≥ 0.99 held; QUASI any ≥ 0.95 (goal); precision proxy not below the
  0.716 baseline. If QUASI lands short, that is the reported finding — no threshold fishing on test.

**Phase 5 — ablations / robustness (optional, after Phase 4 lands)**
- ai4privacy-mixed vs TAB-only training — does multi-domain mixing cost in-domain QUASI recall?
- GLiNER2-base as a stronger span-pretrained initialization — only if Arm B beats Arm A.
- Out-of-domain sanity: held-out ai4privacy slice + spot-check on `corpora/enron`/`clinical`
  (no gold there — qualitative only, reported as such).

## 5. Risks and honesty constraints

- **Test-set discipline:** all iteration on dev; test once per final config (Phase 0 fixes the
  current practice).
- **Union gold is noisy-broad:** union of annotators maximizes the privacy ceiling but inflates
  apparent misses of debatable spans; report any/typed separately, as the gate already does.
- **Domain overfit:** a TAB-fine-tuned detector is legal-domain-shaped. The cloak pipeline's other
  corpora keep the zero-shot detector until Phase 5 measures transfer; switching detectors per
  corpus is a pipeline setting, never a per-method calibration.
- **ROCm:** DeBERTa-v3 is plain-PyTorch-safe; ModernBERT/flash-attention path is the known risk and
  stays a fallback.
- **No new privacy knobs:** detector threshold is chosen once on dev and then fixed across all
  downstream method comparisons.
