---
type: training-experiment
status: done
created: 2026-07-04
model: knowledgator/gliner-pii-large-v1.0
dataset: TAB + Wikipedia-bio + Nemotron-PII (mapped) + Pile-NER slice — 8-type schema, per-type balanced, generality-first mix
result: "PASS — generality 0.872→0.988 (>0.90 target, beats stock-base 0.941); TAB QUASI 0.973 held but ≤ v2 (MISC 0.861 vs 0.895); large-backbone padding-span bug found + guarded"
tags: [detector, gliner, fine-tune, large-backbone, multi-domain, quasi, generality-retention, type-balancing]
companion: docs/research/learned-PII-detection.md
---

# Large-backbone, type-balanced QUASI span detector — fine-tune v3

## Objective & hypothesis
v2 closed the in-schema QUASI gap (TAB QUASI 0.979; MISC 0.895, DEM 0.973, QUANT 0.972) and recovered
open-label generality only **partially** (0.835 → 0.872, short of the > 0.90 target). The disentangler run
proved the detector **generalizes, not overfits** — the *only* open deficit is **out-of-schema open-label
generality retention**. v3 attacks that deficit with two changes on top of v2:

1. **Larger backbone** — `knowledgator/gliner-pii-large-v1.0` (deberta-v3-large, 24L/1024h, ~435M) vs v2's
   base (deberta-v3-**small**, 6L/768h, ~140M). **Hypothesis:** more capacity forgets the open-label
   matching less under single-schema-dominant fine-tuning, so generality retention rises at equal TAB recall.
   Secondary: large's `max_width=100` (vs base's 12) makes long MISC/identifying-event spans (>12 words)
   *structurally learnable* for the first time — base could never emit them.
2. **Generality-first mix + per-type balancing** — shift the diverse-label Pile-NER slice (the only lever
   v2 proved moves generality) to ~25% of the mix, and add **by-type upsampling** of scarce TAB-8 gap types
   (MISC/DEM/QUANT) so the raised Pile share does not cost rare-type recall.

**Accepted confound (deliberate):** backbone, mix, balancing, and selection method all change at once
(requester chose the full-v3 path over isolating the backbone). A generality gain therefore *cannot* be
attributed to any single lever. This is recorded, not hidden — v3 is a deployment-model push, not a lever
ablation. The backbone-isolation ablation stays open (see Ablations).

## Training data (mix + ratios + type-mapping)

Schema sources (TAB-8) are assembled and **balanced first**; the diverse-label Pile-NER slice is sized
**after** balancing, as a fixed fraction of the final total — so per-type upsampling (which touches only
TAB-8 windows) never dilutes the generality signal. The two levers act on disjoint window sets.

**Build order:**
1. Assemble schema pool: TAB (anchor, ~10,997 windows, fixed) + Nemotron (≈0.40×TAB) + Wikipedia-bio
   (all available, ~1,142 = 571 unique ×2 anti-memorization cap).
2. **By-type balancing** — upsample-with-replacement every schema-pool window containing a **rare gap type
   {MISC, DEM, QUANTITY}**, bounded to **≤ ×2 per window** and a **global ≤ ×2 duplication ceiling** (a
   window already ×2 from the wikibio oversample is not duplicated again). This is a *nudge*, not full
   balance: DATETIME (~27k) vs MISC (~2.4k) can't be equalized under ≤×2 (would need ~×10) — that would
   memorize the scarce MISC windows. Bounded upsampling raises rare-type mass without the memorization the
   v2 probe warned against (MISC train 0.926 vs test 0.895 — a ~×2 nudge stays inside the safe gap).
3. Size Pile-NER so it is **~25% of the final total** (post-balance), kept at its **own diverse labels**
   (NOT remapped to TAB-8) — the open-label-retention lever.

| Source | Role | Target share (final) | Label regime |
|---|---|---|---|
| **TAB train** | anchor — true QUASI incl. MISC (legal) | ~40% | TAB-8 |
| **Nemotron-PII** (mapped, subsampled EN) | domain breadth for DEM/DATETIME/LOC/QUANT | ~16% | TAB-8 (mapped) |
| **Wikipedia-bio** ([arXiv 2205.06895](https://arxiv.org/abs/2205.06895)) | 2nd real QUASI domain incl. MISC | ~4% (availability-capped) | TAB-8 |
| **rare-type upsample** (MISC/DEM/QUANT, ≤×2) | protect scarce gap types under the raised Pile share | folded into the three above | TAB-8 |
| **Pile-NER slice** | open-label generality — diverse label phrases | **~25%** | own diverse labels (NOT remapped) |

Realized shares/counts are **measured at build, filled in Results** (v2's per-source estimates were off —
recorded honestly then, same here). Expect total ~26–28k windows (~1.4× v2).

**Nemotron→TAB-8 map** (unchanged from v2, `NEMOTRON_MAP` in `build_pii_span_dataset.py`): person→PERSON ·
org→ORG · location/city/country→LOC · date/time/DOB→DATETIME · SSN/account/ID/email/phone/card/MRN→CODE ·
money/percent→QUANTITY · occupation/nationality/ethnicity/religion/age/gender→DEM. Drop-not-invent;
unmapped labels dropped. **MISC (identifying events) comes only from TAB + Wikipedia-bio** — no other source
annotates it (stated limit; unchanged from v2). **MultiNERD-en is reserved for the held-out generality eval
— NEVER train on it** (contamination).

## Training config
`train_pii_gliner.py` (model-agnostic, unchanged): `--init knowledgator/gliner-pii-large-v1.0
--data-dir data/pii_span_dataset_large_balanced --epochs 3 --lr 6e-6 --others-lr 3e-5 --seed 42
--out data/models/pii_gliner_large_balanced`.
- **LR lowered** to `6e-6` backbone (from v2's `1e-5`) — deberta-v3-large is less stable at 1e-5; `others-lr`
  3e-5. If generality still lags at the selected checkpoint, a lower `others-lr` (label-side) is the next
  lever — held in reserve, not stacked here to keep the LR change interpretable.
- **Dropout:** large's config default is 0.1 (base was 0.3) — left as the model default.
- **GPU (perf gate — PASSED):** one iGPU (gfx1151), one process at a time. Saturation probe (`--max-steps 3`)
  on the built dataset: batch 8 = 7.47 samples/s, **batch 16 = 9.93** (+33%), batch 32 = 10.78 (+8.5% only —
  near-saturated at 16); all fit VRAM (no OOM). **Chosen: `--batch-size 16` @ `--lr 6e-6`** — ~92% of peak
  throughput, only a 2× effective-batch bump from v2's 8 (safe at the lowered LR), keeps ~5,670 update steps.
  batch 32 rejected (thins updates to ~2,835, would need LR rescaling, not worth 8.5%). **Realized wall-time
  estimate ~2.5–2.8 hrs** (152 min train at 9.93 samples/s + 3× dev eval + 1.7 GB checkpoint saves), vs v2
  base ~15 min — the ~10× is the large backbone (24 vs 6 layers, 1024 vs 768 hidden).

## Selection & operating point
**Methodology fix (from the v2 claim audit — do NOT select at a fixed threshold):** pick the checkpoint by
**dev AUPRC / recall-at-matched-precision** (0.94 / 0.90 / 0.85 precision points), NOT recall at fixed 0.3.
The fixed-0.3 rule picked the least-calibrated epoch and manufactured the phantom "overfit." Per-checkpoint
dev threshold sweep, then per-corpus operating point (TAB ~0.02, non-TAB ~0.3; §5.4).

## Evaluation & success criteria
- **TAB test QUASI any ≥ 0.95** — held (primary gate; v2 = 0.979). If the raised aux share costs TAB QUASI,
  that is the reported finding.
- **Open-label generality retention** (held-out MultiNERD-en probe) — target **> 0.90** (beat v2's 0.872).
  **This is the v3 target.**
- **Wikipedia-bio test QUASI ≥ v2** (0.989) — no-regression check (undiscriminating; v1 already 0.983).
- Long-MISC recall: report MISC recall split by gold span length (≤12 vs >12 words) — the axis where large's
  `max_width=100` should beat base structurally.
- **Pass = TAB QUASI ≥ 0.95 held AND generality > 0.90.** (Generality is the point of v3; bio is a guardrail.)

## Results (measured)

**Realized build mix** (measured 2026-07-04, `results/build_large_balanced.log`; total **30,255** windows,
~1.5× v2): TAB 10,997 (36.3%) · Nemotron 5,495 (18.2%; 3 over-budget + 1 bad-offset dropped) · Wikipedia-bio
1,142 (3.8%; 571 unique ×2) · **by-type balancing +5,068 MISC/DEM/QUANTITY copies** (schema pool 17,634 →
22,702, global ≤×2 held) · Pile-NER 7,553 (**25.0%**, post-balance, own diverse labels). dev = 927 TAB
windows. Confirms the imbalance the balancing targets: schema DATETIME ~34k spans vs MISC/DEM/QUANT ~2k each
(~10:1) — the ≤×2 nudge lifts rare-type window mass without full equalization (which would memorize scarce MISC).

**Training (measured):** 3 epochs, batch 16, lr 6e-6, 5,673 steps, **7,432 s (~2.1 hr)** at 12.2 samples/s
(faster than the 2.5–2.8 hr estimate); epoch-3 eval_loss 539 (sum-reduction — scales with span count).
Three epoch checkpoints saved (1891 / 3782 / 5673) + `final`.

**Checkpoint selection (TAB dev threshold sweep, `results/large_balanced_dev_*.json`)** — by
recall-at-matched-precision, NOT fixed-0.3 (the v2-audit fix). At the TAB op point (0.02, lowest thr with
precision ≥ 0.716):

| ckpt (epoch) | QUASI any @0.02 | prec | MISC | DEM | QUANT |
|---|---|---|---|---|---|
| **1891 (epoch 1)** ✅ | **0.963** | 0.836 | **0.780** | **0.949** | **0.994** |
| 3782 (epoch 2) | 0.963 | 0.833 | 0.764 | 0.910 | 0.985 |
| 5673 (epoch 3) | 0.943 | 0.845 | 0.732 | 0.859 | 0.963 |

**Selected epoch 1 (`checkpoint-1891`) @ 0.02** — ties best QUASI and leads on every rare type; later epochs
grow more conservative (recall falls, precision rises) — the same PR-drift pattern the v2 disentangler found,
so this is operating-point drift, not overfit. Full PR grid (5 thresholds × 3 epochs) in the result JSONs.

**⚠️ Honesty flag (TAB dev vs v2 base):** the large backbone + MISC upsampling did **not** beat v2 base on
TAB dev. v2 base @0.02 = QUASI 0.970 / MISC 0.857 / DEM 0.969 (prec 0.807); v3 large @0.02 = QUASI 0.963 /
MISC 0.780 / DEM 0.949 (prec 0.836). v3 sits at higher precision but **lower recall**, and MISC is notably
worse (0.780 vs 0.857). More capacity did not lift TAB rare-type recall here. Held-out **test** + generality
(the actual v3 target) pending below before any verdict.

**TAB test @0.02 (`checkpoint-1891`, `results/large_balanced_test.json`) vs prior versions:**

| | DIRECT | QUASI | prec | MISC | DEM | QUANT | CODE | LOC | ORG | PERSON | DATETIME |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (TAB-only, base) | 1.000 | 0.971 | 0.861 | 0.856 | 0.951 | 0.972 | 0.985 | – | – | – | – |
| v2 (multi-domain, base) | 1.000 | **0.979** | 0.814 | **0.895** | **0.973** | 0.972 | – | – | – | – | – |
| **v3 (large, balanced)** | 1.000 | 0.973 | **0.850** | 0.861 | 0.966 | **0.997** | 0.981 | 0.998 | 0.912 | 0.998 | 1.000 |

TAB QUASI 0.973 (≥ 0.95 gate held) but **below v2's 0.979**; MISC 0.861 **below v2's 0.895**; DEM 0.966 ≈ v2.
Precision rose 0.814 → 0.850, QUANT 0.972 → 0.997. Net on TAB: the large backbone + MISC-upsampling did
**not** lift TAB rare-type recall — it traded a little recall for precision. The `max_width=100` structural
advantage did not show up as higher overall MISC recall (long-MISC split below is the deferred check).

**Open-label generality (MultiNERD-en held-out, @0.3, `results/large_balanced_generality.json`) — the v3
target, decisively met:**

|                                  | overall any | typed |
| -------------------------------- | ----------- | ----- |
| stock knowledgator **base**      | 0.941       | –     |
| v1 (base, TAB-only)              | 0.835       | –     |
| v2 (base, multi-domain)          | 0.872       | –     |
| **v3 (large, generality-first)** | **0.988**   | 0.897 |

**stock 0.941 → v1 0.835 → v2 0.872 → v3 0.988** (any-recall). v3 nearly saturates open-label any-recall and
**exceeds even the stock base model** — the target was > 0.90.

**The gain is real, not over-firing (precision checked, matched re-runs @0.3):** v3 recall 0.988 at
**precision 0.444** and **typed 0.897**, vs v2 recall 0.843 / prec 0.431 / typed 0.707 — v3 lifts recall +0.145
*at comparable precision* (+0.013) with only ~14% more predictions (948 vs 832), and typed recall +0.19.
Over-firing would raise recall while dropping precision; instead precision held and typed recall jumped, so
this is a genuine open-label-quality gain. (Full generality precision table: stock-base 0.925/0.421,
stock-large 0.894/0.528, v2 0.843/0.431, v3 0.988/0.444 — recall/precision.)

**Attribution (RESOLVED by the base+mix run, `2026-07-05-FT-detector-v4-base-genfirst-mix.md`):** **both levers
contribute, ~equally.** Decomposing the +0.145 generality-recall gain: mix effect (v2 base → base+mix,
single-variable) = 0.843 → 0.918 (+0.075); backbone effect (base+mix → v3, same mix) = 0.918 → 0.988 (+0.070).
The earlier guess "it's the mix, not the backbone" (from stock-large 0.894 < stock-base 0.925) was **wrong** —
the large backbone earns a real generality increment beyond the mix; neither alone reaches 0.988.

**Wikipedia-bio test (@0.02, `results/large_balanced_bio_test.json`):** QUASI 0.992 (v2 0.989), MISC 0.966
(v2 0.949), DEM 0.990 (v2 1.000), prec 0.900. Marginally above v2 but still near-ceiling / undiscriminating
(as v2 flagged — TAB→bio transfers trivially).

**Verdict: PASS.** TAB QUASI ≥ 0.95 held (0.973) **AND** generality > 0.90 (0.988, vs v2 0.872). v3 is the
better deployment detector **when open-label generality / tailorability matters** (a first-class requirement):
it recovers nearly all open-label breadth (0.988) at a marginal TAB cost (QUASI −0.006, MISC −0.034, offset by
+0.036 precision). **If TAB rare-type recall (esp. MISC) is paramount, v2 base is marginally better** — the
honest trade. **Long-MISC-by-length split (≤12 vs >12 words) not yet run** — the one remaining explanatory
analysis (whether large's `max_width` helps the long spans base can't emit, even though overall MISC is lower).

### Inference bug found + fixed: phantom padding-region spans at low threshold
The v3 **large** model, at threshold < ~0.1, emits low-confidence (0.04–0.08) **MISC** ("other identifying
attribute or event") spans whose token indices land **past the real sequence, in the padding region** (e.g.
`start=225` into a 203-token map; contiguous runs). GLiNER's `_map_entities_to_original` indexes the
token→char map unguarded → `IndexError` (crashed 11/12 dev batches @0.02). **Base/v2 never does this (0/12).**
Diagnosed via systematic-debugging: not `max_width` (crashes at every width incl. 12), not a threshold
artifact of real spans — genuinely phantom predictions in padding that only clear a low threshold.
**Root-cause fix at the shared boundary** (`src/cloak/detect.py::_guarded_map_entities_to_original`, installed
in `Detector.__init__`): drop spans whose indices exceed the token→char map — those map to no real text, so
this is a no-op for base and **does not move the operating point** (empirical-honesty: not a threshold dodge).
Regression test: `src/cloak/tests/test_detect_padding_guard.py`.
**Attribution (measured):** we did **not** induce it — **stock** `gliner-pii-large-v1.0` emits *more* phantom
padding spans than our fine-tune (**8,363 vs 7,690** over 177 dev chunks @0.02). It is a stock large-backbone /
GLiNER-decoder property; the MISC-upsampling hypothesis is **wrong** (our fine-tune fires slightly fewer). v2
base = 0/12 batches, so this surfaces with the large backbone specifically. Real detection is unaffected — only
non-text padding spans are dropped.

## Observations
- **The v3 target (generality) is decisively met — 0.872 → 0.988, beating even stock-base 0.941.** The
  large backbone + 25% diverse-label Pile-NER retained (recovered past) open-label breadth that base
  fine-tunes eroded. This is the headline: for the tailorability requirement (zero-shot / user-defined
  types), v3 is materially better than every prior detector.
- **But the large backbone did NOT help TAB — it slightly hurt.** TAB QUASI 0.979 → 0.973, MISC 0.895 →
  0.861 (both below v2 base), offset by precision 0.814 → 0.850. More capacity + MISC-upsampling bought
  *precision*, not TAB rare-type recall. The intuition "bigger model + balanced data lifts MISC" was
  **not** borne out. The `max_width=100` structural advantage didn't surface as higher overall MISC recall
  (long-MISC-by-length split still needed to see if it helps the >12-word tail specifically).
- **The generality win and the TAB dip are the same coin.** A model that keeps firing on arbitrary
  open-label phrases (generality 0.988) is inherently less specialized to TAB's 8 types — hence the small
  TAB recall/precision retrade. v2 base specialized harder (TAB up, generality down); v3 large generalizes
  (generality up, TAB slightly down). Which is "better" depends on the deployment axis — stated as a trade,
  not a strict win.
- **A real inference bug surfaced (see below) — but it's stock-large's, not ours.** The large backbone emits
  phantom padding-region MISC spans at low threshold; stock does it *more* than our fine-tune. Fixed at the
  shared `detect.py` boundary; real detection unaffected.
- **Confound stands:** backbone + mix + balancing changed together, so the 0.872 → 0.988 jump can't be split
  between "large capacity" and "25% Pile-NER." The backbone-isolation ablation (below) would separate them.

## Ablations
- **Stock-large vs v3 fine-tuned** (measured — `results/stock_large_{test,generality}.json`; stock @0.3
  its natural op point, v3 ft @0.02, per the v1-doc methodology):

  |                  | TAB QUASI | MISC | DEM | QUANT | prec | generality (any) |
  |------------------|-----------|------|-----|-------|------|------------------|
  | stock-large @0.3 | 0.840     | 0.207 | 0.561 | 0.390 | 0.747 | 0.894 |
  | **v3 ft @0.02**  | **0.973** | **0.861** | **0.966** | **0.997** | **0.850** | **0.988** |
  | Δ                | +0.133    | **+0.654** | +0.405 | +0.607 | +0.103 | +0.094 |

  Two findings: (a) **fine-tuning transforms the large model on TAB** — the guideline gap types MISC/DEM/QUANT
  are unreachable zero-shot (MISC 0.207) and only supervision fixes them (0.861), same story as base (v1-doc:
  stock-base MISC 0.324 → 0.856). Stock-large is even a *worse* zero-shot TAB detector than stock-base. (b)
  **fine-tuning RAISED generality (0.894 → 0.988) instead of eroding it** — the opposite of base fine-tunes
  (v1: 0.941 → 0.835). The large backbone + 25% diverse-label Pile-NER resisted the label-narrowing that hit
  base. _Checked for over-firing: v3 generality recall 0.988 at precision 0.444 (vs stock-large 0.894/0.528) —
  fine-tuning raised recall and typed-recall (0.824→0.897) with only a mild precision dip, so it's a real
  open-label gain, not flooding._
- **Open (deliberately not run in v3):** backbone-isolation — large on the *exact* v2 mix + builder, to
  attribute the generality gain to capacity vs the mix/balancing changes. This is now the priority follow-up
  (v3 cleared > 0.90, so "why" — capacity or Pile-NER weight — is the open question).
- **Deferred:** MISC recall by span length (≤12 vs >12 words) isolates the `max_width` structural effect.

## Cost (actual)
Builder changes + self-check ≈ 0.5 day; data build ~10 min (CPU); **training 7,432 s ≈ 2.1 hr GPU**
(batch 16, ~10× v2 base's 15 min — the large backbone); selection sweep (15 dev gate runs) + test +
generality + bio + attribution ≈ 40 min GPU. Perf-gate probes (batch 8/16/32 saturation) ~2 min.

## Risks & caveats
- **Confounded attribution** (accepted, above) — backbone + mix + balancing + selection all move together.
- **TAB-share drop to ~40%** could shave TAB QUASI; the by-type upsample is the mitigation for the scarce
  types, but PERSON/DATETIME dominance also drops — watch TAB QUASI/precision. v2 held TAB at 45% aux, so
  ~52% aux is an extrapolation, not a repeat.
- **≤×2 upsample memorization** — bounded to stay inside the v2-probed safe gap; if v3 train-vs-test gap on
  MISC widens beyond ~1.5 pt, back the cap off.
- **Large stability** — lower LR mitigates; watch loss curve. bf16 on gfx1151 as in v2.
- **Synthetic Nemotron** transfers weakly to real prose; **cross-domain MISC beyond legal+bio still
  untested** (no clinical/social identifying-event gold) — unchanged limit from v2.

## Builder changes required (review before the run — `/auto-review-loop` per handoff)
`scripts/build_pii_span_dataset.py`:
1. **Per-type balancing** — new step after schema-source assembly: upsample-with-replacement windows
   containing {MISC, DEM, QUANTITY}, ≤×2 per window, global ≤×2 duplication ceiling (coordinate with the
   existing wikibio ≤×2 oversample so nothing goes ×4). One runnable self-check: assert no window appears
   > 2× and rare-type span mass strictly increases.
2. **Pile-NER post-balance sizing** — Pile-NER is no longer capped by a fixed pre-ratio; size it to hit its
   target fraction of the **final** (post-balance) total. `MIX_RATIO` semantics change accordingly — document
   the new meaning in the module docstring (the current `nemotron=N`/`pilener=N` numbers only set HF stream
   depth, not kept-window count — clarify this too, it misled the handoff).
`train_pii_gliner.py` needs **no change** (model-agnostic; just pass `--init …large… --lr 6e-6`).

## Artifacts (produced)
- Dataset: `data/pii_span_dataset_large_balanced/{train,dev}.jsonl` (30,255 / 927 windows);
  build log `results/build_large_balanced.log`.
- Model: `data/models/pii_gliner_large_balanced/checkpoint-{1891,3782,5673}` + `final` +
  `run_manifest.json`. **Selected: `checkpoint-1891` @ threshold 0.02** (TAB op point).
- Dev selection sweep: `results/large_balanced_dev_{1891,3782,5673}_{0.02,0.05,0.1,0.2,0.3}.json`.
- TAB test: `results/large_balanced_test.json` · generality: `results/large_balanced_generality.json` ·
  bio: `results/large_balanced_bio_test.json` · stock-large baselines: `results/stock_large_test.json`,
  `results/stock_large_generality.json`.
- **Bug fix:** `src/cloak/detect.py::_guarded_map_entities_to_original` + `_install_gliner_bounds_guard`;
  regression test `src/cloak/tests/test_detect_padding_guard.py`.
- **Not yet wired into `src/cloak/detect.py` as the default model** (still defaults to v2
  `pii_gliner_multidomain/checkpoint-2479` @0.3) — deployment wiring is a separate decision given the
  v3-vs-v2 trade (generality↑ vs TAB-MISC↓).

## Sources
Predecessor (measured baseline + claim audit + disentangler resolution):
[`2026-07-04-FT-detector-v2-quasi.md`](2026-07-04-FT-detector-v2-quasi.md). v1:
[`2026-07-03-FT-detector-v1-tab-quasi.md`](2026-07-03-FT-detector-v1-tab-quasi.md).
Report: [`learned-PII-detection.md`](../../docs/research/learned-PII-detection.md) §5.1–§5.4.
Datasets: [`datasets.md`](../../docs/research/datasets.md).
Backbone configs compared: knowledgator/gliner-pii-{base,large}-v1.0 `gliner_config.json` (base =
deberta-v3-small 6L/768h `max_width` 12; large = deberta-v3-large 24L/1024h `max_width` 100; both encoders
`max_position_embeddings` 512).
