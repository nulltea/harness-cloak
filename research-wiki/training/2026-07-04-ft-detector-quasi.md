---
type: training-experiment
status: planned
created: 2026-07-04
model: knowledgator/gliner-pii-base-v1.0
dataset: TAB + Wikipedia-bio + Nemotron-PII (mapped) + Pile-NER slice — 8-type schema
result: pending
tags: [detector, gliner, fine-tune, multi-domain, quasi, generality-retention]
companion: docs/research/learned-PII-detection.md
---

# Multi-domain QUASI span detector — fine-tune v2

## Objective & hypothesis
A QUASI-aware detector that holds TAB (legal) **and** generalizes across domains, at the *same* 8-type
schema (no type sprawl). **Hypothesis:** TAB-dominant multi-domain mixing lifts out-of-domain quasi-type
recall and recovers open-label generality (v1: 0.835 → target >0.90) without regressing TAB QUASI below 0.95.
Principle: **more domains, not more types or more data.**

## Training data (mix + ratios)
Target by windows, TAB-anchored (ratio **5 : 2.5 : 1.5 : 1**, ≈22k windows, ~2× v1):

| Source | Role | Share | ~windows | Label regime |
|---|---|---|---|---|
| **TAB train** | anchor — true QUASI incl. **MISC** (legal) | 50% | 11,022 (all) | TAB-8 |
| **Nemotron-PII** (mapped, subsampled EN) | domain breadth for DEM/DATETIME/LOC/QUANTITY (50+ industries) | 25% | ~5,500 | TAB-8 (mapped) |
| **Wikipedia-bio** ([arXiv 2205.06895](https://arxiv.org/abs/2205.06895); train split, oversample ≤×2) | 2nd *real* QUASI domain incl. **MISC** (bios) | 15% | ~2,300* | TAB-8 |
| **Pile-NER / NuNER slice** | open-label generality — diverse label phrases | 10% | ~2,200 | own diverse labels (NOT remapped) |

\*Fetched from `github.com/anthipapa/textanonymization` (Papadopoulou et al.), vendored + split (seed 42)
to `corpora/wikipedia_bio/{train.json (453 docs), test.json (100, held out for cross-domain eval)}`;
train yields ~1,142 windows → ~2,300 at ≤×2 oversample. It is the **only auxiliary source carrying MISC
(identifying events)** — so cross-domain MISC now has real supervision, not just TAB.
**MISC comes only from TAB + bio** — no other source annotates identifying events (stated limit).
**MultiNERD is reserved for the held-out generality eval — do NOT train on it** (contamination).

Nemotron→TAB-8 map: person→PERSON · org→ORG · location/city/country→LOC · date/time/DOB→DATETIME ·
SSN/account/ID/email/phone/card/MRN→CODE · money/percent→QUANTITY ·
**occupation/nationality/ethnicity/religion/age/gender→DEM**. Drop medical conditions + anything not
cleanly in the 8 (avoids the v1 DEM/MISC type-shift); drop-not-invent.

## Training config
Reuse `train_pii_gliner.py` unchanged: `--init knowledgator/gliner-pii-base-v1.0
--data-dir data/pii_span_dataset_multidomain --epochs 3 --lr 1e-5 --others-lr 5e-5 --seed 42
--out data/models/pii_gliner_multidomain`. The diverse-label slice in-batch is the
generality-preservation lever (a uni-encoder can't cleanly freeze its label side); keep 3 epochs /
modest LR (v1 overfit past epoch 2).

## Selection & operating point
Select on **TAB dev** (don't regress the anchor): lowest threshold with precision ≥ 0.716 → max QUASI
recall (§5.4). Record **per-corpus** thresholds (TAB ~0.02; bio/clinical/social recalibrated, expect ~0.3).

## Evaluation & success criteria
- **TAB test QUASI any ≥ 0.95** — held (primary gate; v1 = 0.971).
- **Wikipedia-bio test QUASI recall** — real cross-domain gold; must beat v1.
- **Open-label generality retention** (held-out MultiNERD probe) — target **> 0.90** (recover from 0.835).
- Cross-domain qualitative on clinical/enron (no gold); per-corpus threshold Pareto.
- **Pass = TAB QUASI ≥ 0.95 held AND (bio-test QUASI ↑ vs v1 OR generality > 0.90).** If mixing costs TAB
  QUASI, that is the reported finding.

## Results
_Pending — not yet run._

## Ablations (isolate each lever)
v1 TAB-only (control) → +bio → +bio+Nemotron → +Pile-NER slice. The +slice arm measures the
generality-recovery value.

## Cost
Data prep ~0.5–1 day (Nemotron map + bio loader + Pile-NER slice + per-source self-checks);
training ~20–40 min GPU; eval ~20 min.

## Risks & caveats
Synthetic Nemotron may transfer weakly to real prose; **MISC supervision spans only TAB (legal) +
Wikipedia-bio (biographies)** — no clinical/social identifying-event source, so cross-domain MISC beyond
those two domains stays untested;
type-mapping is lossy (drop-not-invent); dilution may shave TAB QUASI (token cap manages); bio
oversampling ≤×2 to avoid memorizing 453 docs.

## Artifacts (to be produced)
`data/pii_span_dataset_multidomain/{train,dev}.jsonl` · `data/models/pii_gliner_multidomain/` ·
gate + generality-probe + per-corpus-threshold result JSONs.

## Sources
Predecessor + measured baseline: [`2026-07-03-ft-detector-tab-quasi.md`](2026-07-03-ft-detector-tab-quasi.md).
Report: [`learned-PII-detection.md`](../../docs/research/learned-PII-detection.md) §5.3/§5.4.
Datasets + QUASI-usability analysis: [`datasets.md`](../../docs/research/datasets.md).
