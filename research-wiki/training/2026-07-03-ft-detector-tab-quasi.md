---
type: training-experiment
status: done
created: 2026-07-03
model: knowledgator/gliner-pii-base-v1.0
dataset: TAB train (legal, 8-type, DIRECT+QUASI union)
result: "test QUASI any 0.971 (≥0.95 goal met); open-label generality eroded 0.941→0.835"
tags: [detector, gliner, fine-tune, tab, quasi, generality-retention]
companion: docs/research/learned-PII-detection.md
---

# TAB-only QUASI span detector — fine-tune v1

## Objective & hypothesis
Close the QUASI-recall gap on TAB that no off-the-shelf checkpoint could (the guideline-defined types
MISC/DEM/QUANTITY) by fine-tuning the best zero-shot open-label detector on TAB's own train split.
**Hypothesis (confirmed):** supervision on TAB train lifts QUASI any-recall past the 0.95 goal while
holding DIRECT and precision.

## Training data
Single source — no mix.

| Source | Role | Windows | Spans | Label regime |
|---|---|---|---|---|
| TAB train (`corpora/tab/echr_train.json`, 1,014 ECHR docs) | sole supervision | 11,022 | 59,737 | 8 TAB label phrases |

Gold = union of annotators' DIRECT+QUASI mentions over the 8 types (NO_MASK excluded). Windowed by
`build_pii_span_dataset.py` (150-word entity-safe windows, subword preflight, drop spans >60 words).

## Training config
`train_pii_gliner.py`: init `knowledgator/gliner-pii-base-v1.0` (uni-encoder, span-mode markerV0);
3 epochs; batch 8; lr 1e-5 (backbone) / others-lr 5e-5; cosine + 0.1 warmup; seed 42; bf16;
gfx1151 iGPU; **~14.5 min**, loss 49→30.

## Selection & operating point
Epoch-2 (`checkpoint-2756`) selected on TAB dev (epoch-3 overfit — QUASI/MISC dropped). Operating
threshold **0.02** fixed on the dev Pareto: lowest threshold with precision ≥ 0.716 → max QUASI recall
(dev QUASI 0.958 at 0.02; see `results/arm_b_dev_thr_*.json`).

## Evaluation & success criteria
Criteria: DIRECT any ≥ 0.99 · QUASI any ≥ 0.95 (goal) · precision ≥ 0.716. Test-set discipline:
iterate on dev, one test run per final config.

## Results (measured)

### QUASI detection — knowledgator STOCK vs FINE-TUNED (critical), echr_test
Each model at its operating point (stock zero-shot @0.3; fine-tuned @0.02). **The critical result is the
STOCK→FINE-TUNED jump** — fine-tuning, not the checkpoint choice, closes the gap (gliner_small is the
nice-to-have baseline).

| Model (thr) | DIRECT | **QUASI** | prec | **MISC** | **DEM** | **QUANT** | CODE |
|---|---|---|---|---|---|---|---|
| gliner_small — baseline, zero-shot (0.3) | 0.998 | 0.857 | 0.716 | 0.214 | 0.563 | 0.254 | 0.757 |
| **knowledgator STOCK** — zero-shot (0.3) | 0.998 | 0.888 | 0.752 | 0.324 | 0.586 | 0.596 | 0.791 |
| **knowledgator FINE-TUNED** — v1 (0.02) | **1.000** | **0.971** | **0.861** | **0.856** | **0.951** | **0.972** | **0.985** |

Stock knowledgator already leads gliner_small (QUASI 0.888 vs 0.857) but still fails the gap types
(MISC 0.32, DEM 0.59). **Fine-tuning is the transformation:** MISC 0.32→0.86, DEM 0.59→0.95,
QUANT 0.60→0.97, QUASI 0.888→0.971 — clearing the ≥0.95 goal. Clean dev→test transfer (0.958→0.971).

### Open-label generality retention — the cost (MultiNERD-en, held-out types outside TAB-8)
Zero-shot any-recall on out-of-schema types (animal/disease/plant/food/media; n = 4,406 gold, thr 0.3)
— the tailorability probe. **Fine-tuning erodes it back to general-NER level:**

| Model | overall any | ANIM | DIS | PLANT | FOOD | MEDIA |
|---|---|---|---|---|---|---|
| gliner_small (reference) | 0.845 | 0.92 | 0.86 | 0.86 | 0.77 | 0.61 |
| **knowledgator STOCK** | **0.941** | 0.95 | 0.94 | 0.94 | 0.92 | 0.92 |
| **knowledgator FINE-TUNED** (v1) | **0.835** | 0.87 | 0.83 | 0.82 | 0.71 | 0.89 |

**Retention delta: 0.941 → 0.835 (−0.106)** — the fine-tuned model's out-of-schema generality ≈ the
plain general-NER baseline (0.845).

### Out-of-domain transfer (non-TAB, no gold — qualitative)
QUASI *sensitivity* transfers (ages, medical conditions, quantities on clinical text), but the 0.02 op
point floods off-domain; a per-corpus threshold (~0.3) is clean **and still transfers** the wins — the
flooding is a threshold artifact, not lost knowledge (`scripts/spikes/pii_quasi_showcase*.py`).

## Observations

- **The QUASI win comes from supervision, not the checkpoint.** gliner_small→stock-knowledgator is a
  small lift (QUASI 0.857→0.888); stock→fine-tuned is the transformation (0.888→0.971). What TAB
  supervision buys is the *guideline-defined* notion of MISC (identifying events), DEM (demographics),
  and identifying QUANTITY — categories no natural-language label phrase can express zero-shot, which
  is exactly why every off-the-shelf checkpoint stalled at MISC ≤ 0.32.

- **Why open-label generality dropped — the conclusion: catastrophic forgetting of the open-label
  matching.** Training presents *only* the 8 fixed TAB label phrases as positives, and the gliner loop
  draws its in-batch negatives from the *same 8*. So the shared uni-encoder (knowledgator's label and
  text sides are one transformer) re-specializes to discriminate TAB's 8 types and **stops practicing
  arbitrary label phrases** (animal, disease, food). The broad open-label calibration that PII
  pretraining had given the stock model (0.941) is overwritten — post-fine-tune generality (0.835) lands
  at the plain general-NER baseline (0.845), i.e. the model "forgot back" to generic NER. This is *real
  forgetting, not a threshold artifact*: stock and fine-tuned are compared at the same 0.3 threshold,
  and the drop persists. It is the standard label-set-narrowing failure of single-schema fine-tuning.

- **Operating point is model- and domain-specific.** The fine-tuned model is sharp and confident on
  TAB → its recall peaks at a *low* threshold (0.02); the stock model is diffuse → ~0.3. And 0.02 is
  TAB-specific — it floods on other domains, which need their own (higher) operating point.

- **Implication for v2:** to keep the QUASI win *and* recover generality, re-introduce label-phrase
  diversity during training (a general-NER slice with varied labels + a multi-domain mix), TAB-dominant
  so the QUASI supervision is not diluted. See the successor experiment.

## Ablations
The off-the-shelf dev sweep selected the init: knowledgator beat gliner_small / multi_pii / nvidia on
both QUASI (0.898 dev) and open-label generality (0.941). The DeBERTa-BIO arm was not run — the
fixed-head-vs-open-label head comparison is still open.

## Cost
Data prep + hardening (reviewed) ≈ 1 day; training ~14.5 min; selection + test + generality probe ~10 min.

## Risks & caveats
Legal-domain overfit; open-label generality erosion; type-shift on out-of-schema domains (clinical
conditions → DEM/MISC); union gold is noisy-broad (some debatable spans). Motivates v2 (multi-domain).

## Artifacts
Model `data/models/pii_gliner/checkpoint-2756` @0.02 · `results/latticecloak_detection_gate_arm_b.json`
· `results/arm_b_dev_thr_*.json` · `results/pii_zeroshot_generality_arm_b.json` ·
`data/models/pii_gliner/run_manifest.json`.

## Sources
Report: [`learned-PII-detection.md`](../../docs/research/learned-PII-detection.md) §5.3, §5.4.
Datasets: [`datasets.md`](../../docs/research/datasets.md). Scripts: `scripts/build_pii_span_dataset.py`,
`scripts/train_pii_gliner.py`. Successor: [`2026-07-04-ft-detector-quasi.md`](2026-07-04-ft-detector-quasi.md).
