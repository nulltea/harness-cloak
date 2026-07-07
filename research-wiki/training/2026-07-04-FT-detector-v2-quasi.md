---
type: training-experiment
status: done
created: 2026-07-04
model: knowledgator/gliner-pii-base-v1.0
dataset: TAB + Wikipedia-bio + Nemotron-PII (mapped) + Pile-NER slice — 8-type schema
result: "PASS — TAB QUASI held 0.971→0.979; open-label generality recovered 0.835→0.872; bio-test 0.989"
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
train yields ~571 unique windows → ~1,142 at the ≤×2 anti-memorization cap (≈6% of the mix, below the
15% target — bio is availability-capped; see Results). It is the **only auxiliary source carrying MISC
(identifying events)** — so cross-domain MISC now has real supervision, not just TAB.
**MISC comesSwit only from TAB + bio** — no other source annotates identifying events (stated limit).
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
modest LR (v1 dev QUASI peaked at epoch 2 at fixed thr 0.3 — overfit vs recalibration not isolated).

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

## Results (measured 2026-07-04)

**Realized mix** (windows; my spec's per-source estimates were off — recorded honestly): TAB 10,997
(55%) · Nemotron 5,495 (28%; 1,157 bad-offset spans dropped by validation) · Pile-NER 2,197 (11%) ·
**Wikipedia-bio 1,142 (6% — 571 unique bios ×2**; the ≤×2 anti-memorization cap limits it, since only
453 short bios exist, so bio can't reach the 15% target). Total 19,831. Epoch-1 (`checkpoint-2479`)
selected on TAB dev; operating threshold **0.02** (dev Pareto, precision ≥ 0.716; = v1's op point).

**TAB test (thr 0.02) — dilution did NOT cost TAB; it slightly helped:**

| | DIRECT | QUASI | prec | MISC | DEM | QUANT |
|---|---|---|---|---|---|---|
| v1 (TAB-only) | 1.000 | 0.971 | 0.861 | 0.856 | 0.951 | 0.972 |
| **v2 (multi-domain)** | 1.000 | **0.979** | 0.814 | **0.895** | **0.973** | 0.972 |

TAB QUASI held/improved (0.971→0.979); MISC 0.856→0.895, DEM 0.951→0.973. Precision slipped 0.861→0.814
(more aux ⇒ more TAB over-detection) but stays above the 0.716 floor.

**Open-label generality (MultiNERD-en held-out, thr 0.3) — partially recovered:**
stock 0.941 → v1 0.835 → **v2 0.872** (DIS fully back to 0.94; ANIM 0.87, FOOD 0.84, MEDIA 0.85). The
diverse Pile-NER slice + multi-domain mix recovered ~⅓ of the lost generality — real, but short of the
> 0.90 target.

**Wikipedia-bio test (cross-domain, 100 held-out, thr 0.02):**

| | QUASI | MISC | DEM | prec |
|---|---|---|---|---|
| v1 | 0.983 | 0.949 | 0.979 | 0.919 |
| **v2** | **0.989** | 0.949 | **1.000** | 0.904 |

v2 marginally better, **but v1 already transfers to bio strongly (0.983)** — so bio-test is *not*
discriminating (TAB→bio is easy: same DIRECT/QUASI schema, clean well-formed text). It confirms no
regression, not a large v2 gain.

**Verdict: PASS.** TAB QUASI ≥ 0.95 held (0.979) AND bio-test QUASI ↑ vs v1 (0.989 > 0.983). Multi-domain
mixing held/slightly-improved TAB, recovered generality 0.835→0.872, and marginally improved bio.
**Honest limits:** bio-test undiscriminating (v1 already strong); generality recovered but < 0.90; TAB
precision slipped 0.05; the *real* cross-domain question (clinical/social) stays unmeasured (no gold).

Artifacts: `data/models/pii_gliner_multidomain/checkpoint-2479` @0.02 ·
`results/latticecloak_detection_gate_arm_b_v2.json` · `results/arm_b_v2_bio_test.json` (+ `arm_b_v1_bio_test.json`)
· `results/pii_zeroshot_generality_arm_b_v2.json` · `results/arm_b_v2_dev_thr_*.json`.

## Observations

- **Multi-domain mixing is safe — the dilution worry is disproven.** 45% cross-domain aux did not
  lower TAB QUASI; it slightly *raised* it (0.971→0.979), with MISC 0.856→0.895 and DEM 0.951→0.973.
  Why: TAB stays the dominant single source (55%) and the aux *reinforces* the quasi-prone types rather
  than competing — Nemotron/bio contribute demographic and age spans that sharpen DEM. The only cost is
  a precision dip (0.861→0.814): broader label exposure (Pile-NER's diverse phrases + synthetic
  Nemotron) makes the model fire a little more on TAB — recoverable via threshold, or absorbed by the
  downstream ranker (`learned-PII-detection.md` §5.1d).

- **The generality-recovery lever works, partially.** Open-label generality rose 0.835→0.872. The
  diverse-label Pile-NER slice (kept as its own labels, so in-batch negatives span many phrases) stopped
  the label encoder from narrowing as hard as v1's single-schema fine-tune. It did **not** fully recover
  (0.872 < stock 0.941, < 0.90 target) because the slice is only 10% and 3 epochs still specialize toward
  TAB-8 — more diverse-label weight or a lower label-side LR would recover more, at some TAB cost.

- **Bio-test was the wrong instrument.** v1 (0.983) and v2 (0.989) both ace Wikipedia-bio because TAB→bio
  transfers trivially (shared DIRECT/QUASI schema, clean well-formed prose). So the bio result shows *no
  regression*, not a cross-domain *gain*. The real cross-domain question — noisy, out-of-schema clinical
  dialogue and social text — needs gold there; MISC/identifying-event transfer beyond legal+bio is still
  unproven.

- **MISC now has two real domains (legal + bio), not one** — bio adds 571 real bios with identifying-event
  MISC. But no clinical/social MISC source exists, so that gap persists (stated limit).

- **Net:** v2 is the better deployment detector (TAB held/up, generality up, no regression) — a modest,
  honest improvement that validates (a) multi-domain mixing is net-positive and safe, and (b) the
  generality-recovery mechanism is real. The decisive open question — cross-domain QUASI on noisy real
  text — remains unmeasured for lack of gold, and is the natural next experiment.

## Claim audit (/result-to-claim, Codex, 2026-07-04)

- **C1 — multi-domain didn't dilute TAB:** *supported for recall* (QUASI 0.971→0.979, MISC/DEM up),
  *medium for net quality* (precision 0.861→0.814). Revised wording: "did not dilute TAB recall;
  precision decreased." Missing: PR curves / recall-at-matched-precision, bootstrap CI.
- **C2 — generality recovered:** *partial* — 0.835→0.872 but still well below stock 0.941, and measured
  only at thr 0.3 (not v2's op point 0.02). Wording already hedged.
- **C3 — cross-domain bio improved:** *partial / weak* — v1 already near-ceiling (0.983) on a 100-doc,
  near-saturated test; precision fell 0.919→0.904. Wording already flags "undiscriminating."
- **C4 — "overfits after ~1 epoch": NOT supported (overclaimed) — corrected.** The per-epoch comparison
  is at a **fixed thr 0.3**, where precision *rises* across epochs (0.941→0.946→0.945) while recall
  falls — later checkpoints are more *conservative/differently-calibrated*, not demonstrably overfit
  (epoch-3 QUASI = epoch-1 = 0.871 at 0.3, so even "epochs 2–3 worse" is false). Thresholds were not
  re-swept per epoch. **Disentangling experiment:** re-evaluate ckpt-2479/4958/7437 with full dev
  threshold sweeps → AUPRC + recall-at-matched-precision (0.94/0.90/0.85); if epoch-1 still dominates
  the PR curve while train loss falls, overfit is real; if 2–3 recover at their own op point, it was
  operating-point drift, not overfitting.
  - **RESOLVED (disentangler run, `results/v2_disent_*.json`): NOT overfit — operating-point drift.**
    At *matched precision*, the three epochs' dev PR curves overlap (recall ~0.92 @prec 0.90 and
    ~0.875 @prec 0.94 for all three; epoch-3 marginally ahead at high precision). The "epoch-1 best"
    ranking was purely the fixed-thr-0.3 artifact (precision rises 0.941→0.946→0.945 across epochs →
    recall-at-0.3 falls). **Memorize probe** (`results/v2_train_gate.json`): epoch-1 TAB-train recall
    0.994 vs held-out test 0.979 (MISC 0.926 vs 0.895) — a ~1.5-pt gap ⇒ **generalizing, not
    memorizing** (even scarce MISC transfers). Conclusion: within-schema fine-tuning at this scale
    generalizes and does not overfit; the only cost is out-of-schema narrowing (the generality-retention
    axis). **Methodology fix:** select checkpoints by PR/AUPRC or recall-at-matched-precision, not
    recall at a fixed threshold (which picks the least-calibrated epoch).

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
Predecessor + measured baseline: [`2026-07-03-FT-detector-v1-tab-quasi.md`](2026-07-03-FT-detector-v1-tab-quasi.md).
Report: [`learned-PII-detection.md`](../../docs/research/learned-PII-detection.md) §5.3/§5.4.
Datasets + QUASI-usability analysis: [`datasets.md`](../../docs/research/datasets.md).
