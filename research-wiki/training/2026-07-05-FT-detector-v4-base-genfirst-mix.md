---
type: training-experiment
status: done
created: 2026-07-05
model: knowledgator/gliner-pii-base-v1.0
dataset: v3 generality-first balanced mix (data/pii_span_dataset_large_balanced) — reused as-is
result: "mix+backbone BOTH drive generality (each ~+0.07; 'large is dead weight' refuted). base+mix = best MISC (0.925) but ORG regressed 0.948→0.848; generality 0.843→0.918 (< v3 0.988). No config dominates."
tags: [detector, gliner, fine-tune, base-backbone, generality-first, type-balancing, ablation, mix-isolation]
companion: docs/research/learned-PII-detection.md
---

# Base backbone + generality-first mix — mix-vs-backbone isolation

## Objective & hypothesis
v3 (large + 25% Pile-NER + balancing) hit generality 0.988 but **regressed slightly on TAB** vs v2 base
(MISC 0.895→0.861) and its confound (backbone + mix + balancing all changed together) hides *what* drove the
generality jump. The stock numbers point at the **mix, not the backbone**: stock-large (0.894 generality) is
*worse* than stock-base (0.925), yet fine-tuned v3-large reached 0.988 — a weaker-starting backbone can't be
the cause, so the 25% Pile-NER slice is the likely driver.

**Hypothesis:** the generality-first mix, applied to the **base** backbone, recovers most of v3's generality
gain **while keeping v2 base's superior TAB recall** (esp. MISC 0.895). If so, `base + v3-mix` **dominates
v3-large** — better privacy recall, comparable tailorability, ~3× smaller/faster, and no large-backbone
padding-span bug — retiring the large backbone.

## Training data
**Reuse `data/pii_span_dataset_large_balanced/` unchanged** (v3's build: 30,255 train / 927 dev windows;
TAB 36% · Nemotron 18% · bio 4% · +5,068 rare-type upsample copies · Pile-NER 25%, post-balance). It is
backbone-agnostic — deberta-v3 small/base/large share the identical 128k tokenizer, and the trainer
re-preflights the 480-subword budget against the base init. No rebuild.

## Training config — **match v2 exactly except the data** (single-variable isolation)
`train_pii_gliner.py --init knowledgator/gliner-pii-base-v1.0
--data-dir data/pii_span_dataset_large_balanced --epochs 3 --batch-size 8 --lr 1e-5 --others-lr 5e-5
--seed 42 --out data/models/pii_gliner_base_genfirst`.
- lr 1e-5 / others-lr 5e-5 / batch 8 = v2's recipe (NOT v3's 6e-6/16 — base is stable at 1e-5; matching v2
  means `new − v2` = the mix effect alone). ~30k windows at batch 8 ≈ 3,780 steps/epoch × 3 ≈ 11,300 steps;
  base ~15–25 min GPU (v2 was 15 min at 19.8k windows). Perf gate: base+batch-8 is the characterized v1/v2
  operating point; >10 min but saturation known, no re-probe needed.

## Selection & operating point
Same as v3: TAB-dev threshold sweep, select by recall-at-matched-precision (NOT fixed-0.3); TAB op point
~0.02. Sweep the 3 epoch checkpoints × {0.02,0.05,0.1,0.2,0.3}.

## Evaluation & success criteria
Compare at matched code against **v2 (base, 10% Pile)** and **v3 (large, 25% Pile)** — all re-run this
session: `results/{v2_rerun,large_balanced}_{test,generality}.json`.
- **TAB test QUASI/MISC** — must hold ≥ v2 (target: keep MISC ≈ 0.895, unlike v3's 0.861).
- **Generality (MultiNERD held-out)** — how much of v3's 0.988 does base+mix recover from v2's 0.843?
- **Decision rule:** if TAB MISC ≥ ~0.89 held AND generality ≫ v2's 0.843 (toward v3's 0.988) → base+mix
  dominates v3-large → the backbone was dead weight. If generality stays near v2 → the *backbone* (not the
  mix) drove v3's generality, and large is justified.
- **Generality precision** is already recorded by `pii_zeroshot_generality.py` (`precision_proxy`) — no probe
  change needed. Judge base+mix generality on recall **and** precision vs v2 (0.843/0.431) and v3 (0.988/0.444),
  so a recall lift isn't confounded with over-firing.

## Results (measured 2026-07-05)

Selected **epoch 1 (`checkpoint-3782`) @ 0.02** on TAB dev (best rare-type recall at op point: dev MISC
0.886, DEM 0.967; sweep in `results/base_genfirst_dev_*.json`). All rows below at their op point, matched
gate this session.

**TAB test @0.02 — 3-way (v2 / base+mix / v3):**

| model | QUASI | MISC | DEM | QUANT | ORG | CODE | prec |
|---|---|---|---|---|---|---|---|
| v2 (base, 10% Pile, no balance) | **0.979** | 0.895 | **0.973** | 0.972 | **0.948** | – | 0.814 |
| **base+mix (base, 25% Pile, balance)** | 0.966 | **0.925** | 0.968 | 0.983 | 0.848 | 0.961 | 0.786 |
| v3 (large, 25% Pile, balance) | 0.973 | 0.861 | 0.966 | **0.997** | 0.912 | 0.981 | **0.850** |

**Generality (MultiNERD held-out @0.3) — recall / precision / typed:**

| model | recall | prec | typed |
|---|---|---|---|
| v2 (base, 10% Pile) | 0.843 | 0.431 | 0.707 |
| **base+mix (base, 25% Pile)** | **0.918** | 0.435 | 0.789 |
| v3 (large, 25% Pile) | 0.988 | 0.444 | 0.897 |

### Verdict — mix AND backbone both contribute; "large is dead weight" is REFUTED
Decompose the v2→v3 generality gain (recall, at matched precision ~0.43):
- **Mix effect** (v2 → base+mix, single-variable, only the data changes): 0.843 → **0.918 (+0.075)**, typed
  0.707 → 0.789. The 25%-Pile + balancing mix genuinely lifts base generality at matched precision — **not
  over-firing** (precision 0.431 → 0.435 flat, n_pred 832 → 896).
- **Backbone effect** (base+mix → v3, same mix; confounded with lr/batch): 0.918 → **0.988 (+0.070)**, typed
  0.789 → 0.897. The large backbone adds a **second, comparable** increment.
- ⇒ mix and backbone each supply ~half of the +0.145 total. **The hypothesis that the backbone was dead
  weight is wrong** — it earns a real generality increment beyond the mix. Neither alone reaches 0.988.

**TAB — the mix is a per-type trade, not a free win:**
- **MISC: base+mix 0.925 is the best of all three** (v2 0.895, v3 0.861). The rare-type upsampling works —
  and works *better on base than on large* (base+mix MISC 0.925 ≫ v3 0.861).
- **ORG regressed hard: 0.948 (v2) → 0.848 (base+mix).** The generality-first mix drops TAB share to 36% and
  the balancing upsamples only MISC/DEM/QUANT — so **ORG (common, non-upsampled) loses relative
  representation** and recall falls. Precision also dips (0.814 → 0.786). Overall QUASI 0.979 → 0.966.
- So the mix **helps the upsampled rare types (MISC) at the cost of a non-upsampled common type (ORG) and
  precision** — a balancing side effect, not anticipated.

**No config dominates:**
- Best **MISC** (privacy-critical, supervision-only): **base+mix (0.925)**.
- Best **balanced TAB** (overall QUASI, ORG, precision): **v2 base**.
- Best **generality/tailorability**: **v3 large (0.988)** — and the backbone genuinely earns it.
- base+mix does **not** retire the large backbone (v3 generality 0.988 ≫ base+mix 0.918); nor does it beat v2
  on balanced TAB. It is the **best MISC detector** and a Pareto-distinct point, not a dominator.

## Ablations
This experiment IS the mix-vs-backbone isolation. Companion runs (matched, this session): v2 base
(`results/v2_rerun_*`), v3 large (`results/large_balanced_*`), stock-base/large (`results/stock_*`).

## Cost
Reuse dataset (0 build); base train ~15–25 min GPU; selection sweep + eval ~30 min.

## Risks & caveats
- **lr/batch matched to v2, not v3** — so `new vs v3` differs in backbone AND lr/batch (not single-variable);
  `new vs v2` is the clean single-variable (mix) comparison. Stated, not hidden.
- **Generality precision is available** (`precision_proxy` in the probe) — judge on recall + precision, not
  recall alone (avoids the fixed-threshold over-firing trap).
- Base's `max_width=12` cannot emit long MISC spans (>12 words) — a structural ceiling large lacks; if base
  MISC lags on long spans specifically, that's the one place large's backbone genuinely helps.

## Artifacts (to be produced)
`data/models/pii_gliner_base_genfirst/` · `results/base_genfirst_dev_*.json` ·
`results/base_genfirst_{test,generality}.json`.

## Sources
Predecessors (matched re-runs this session): v2 [`2026-07-04-FT-detector-v2-quasi.md`](2026-07-04-FT-detector-v2-quasi.md),
v3 [`2026-07-04-FT-detector-v3-large-balanced.md`](2026-07-04-FT-detector-v3-large-balanced.md).
Report: [`learned-PII-detection.md`](../../docs/research/learned-PII-detection.md).
