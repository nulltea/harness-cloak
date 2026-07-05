---
type: training-experiment
status: planned
created: 2026-07-05
model: knowledgator/gliner-pii-base-v1.0
dataset: TAB-dominant multi-domain + diverse Pile-NER (Pile 18%, ORG-safe balance) — 8-type schema
result: pending
tags: [detector, gliner, fine-tune, base-backbone, generality, org-floor, max-width, minimal-epochs]
companion: [docs/specs/detector-model.md, research-wiki/training/2026-07-05-FT-detector-v4-base-genfirst-mix.md]
---

# FT-detector v5 — base, ORG-safe generality mix, max_width fixed

## Objective & hypothesis
Land a single **base** detector that meets the spec (`docs/specs/detector-model.md`) without the v3/v4
trade-offs: hold v2's TAB strengths (QUASI 0.979, **ORG 0.948**), keep v4's **MISC 0.925**, recover
generality toward **> 0.90**, and remove base's long-span ceiling — all on the small backbone.

Diagnosis driving the design (from the sufficiency matrix + v1–v4):
- Stock already gives DIRECT, open-label **generality (base 0.925)**, and zero-shot extensibility — the FT
  must **preserve** generality, not chase it; and must **supply** QUASI + the gap types MISC/DEM/QUANT.
- **Large backbone is not worth it** (worse QUASI, padding-span bug, ~10× cost) → base.
- **ORG-regression HYPOTHESIS (not established):** v4's ORG drop (0.948 → 0.848) *correlates* with TAB's
  relative share falling 55% → 36% (Pile 25% + upsample copies), suggesting per-batch ORG dilution. But v4
  changed several things at once (Pile share, rare-upsample, label mix, ordering), so this is a hypothesis
  to **test**, not a diagnosis. The required Pile/share sweep + realized-per-type-share logging (below) is
  what confirms or refutes it — the design must not treat the correlation as proven.
- **base `max_width=12`** structurally cannot emit MISC spans > 12 words (data caps at 60) → set it to 60.

**Hypothesis:** base + a *gentler* generality mix (Pile ~18%, TAB ~48%) + `max_width=60`, at v2's exact
recipe/schedule, holds ORG ≥ 0.90 and MISC ≥ 0.89 while lifting generality to **~0.88–0.90** (interpolating
v2 10%→0.843 and v4 25%→0.918 gives ~0.88 at 18%). **This is a strict improvement over v2 (ORG held, MISC↑,
generality↑) — NOT necessarily Pareto-dominant over v4** (v4's 0.918 generality came at 25% Pile; matching
it at 18% would require base to be more Pile-efficient than the interpolation, which is not expected).
**Honest risk:** generality on base likely tops out < v4's 0.918 at ORG-safe Pile; if the goal is > 0.918,
that needs more Pile (re-incurring the ORG risk) or the large backbone (v3) — a documented trade, and the
Pile sweep maps exactly where that frontier sits.

## Training data (sources + ratios + type-mapping)
Reuse the builder + sources; **change only the mix knobs** to be ORG-safe. Build order unchanged
(schema pool → rare-type balance → Pile sized post-balance), with:

| Source | Role | Target final share | Label regime |
|---|---|---|---|
| **TAB train** (`corpora/tab/echr_train.json`, 11k windows) | anchor: QUASI incl. MISC, **and ORG supervision** | **~48%** | TAB-8 |
| **Nemotron-PII** (mapped, EN) | DEM/DATETIME/LOC/QUANT breadth | ~18% | TAB-8 (mapped) |
| **Wikipedia-bio** ([arXiv 2205.06895](https://arxiv.org/abs/2205.06895)) | 2nd real QUASI+MISC domain | ~4% (avail-capped) | TAB-8 |
| **rare-type upsample** (MISC/DEM/QUANT, ≤×2) | keep v4's MISC lift | folded in | TAB-8 |
| **Pile-NER slice** | preserve open-label generality | **~18%** (was v4 25%) | own diverse labels |

Nemotron→TAB-8 map unchanged (`NEMOTRON_MAP`). MultiNERD reserved (eval-only). Expected total ~23k windows.

**The two knob changes vs v4** (`build_pii_span_dataset.py`, `--balance-rare` path):
1. `PILE_FRAC = 0.18` (from 0.25) — gentler generality slice → TAB relative share rises from 36% → ~48%,
   restoring ORG's per-batch signal.
2. **ORG-safe guard (new) — precisely specified.** After building, the builder logs, and the run records,
   these realized fields (per split): `n_windows_total`, `n_windows_tab`, and per TAB-8 type the
   `{window_count, mention_count, token_count}` — plus the derived **TAB window share** = `n_windows_tab /
   n_windows_total` and **ORG window share** = `org_window_count / n_windows_total`. **Fail condition:** build
   aborts if TAB window share < 0.45 OR ORG window share < its v2-build value (recomputed from the v2 build,
   not hard-coded). Enforcement = cap the rare-upsample copy count (reduce copies until TAB share ≥ 0.45)
   rather than adding ORG copies (ORG is not scarce — the lever is holding TAB share, not upsampling ORG).
   **Note:** window share ≠ ORG *recall*; the guard controls the training-signal proportion (the hypothesised
   cause), and ORG recall is verified downstream. Additionally, **dev ORG recall becomes a model-selection
   constraint** (see Selection): reject any checkpoint with dev ORG < 0.90 even if its aggregate QUASI is
   higher.

## Training config — v2's EXACT recipe/schedule (single-variable-from-v2; no epoch confound)
`train_pii_gliner.py --init knowledgator/gliner-pii-base-v1.0
--data-dir data/pii_span_dataset_base_orgsafe --epochs 3 --batch-size 8 --lr 1e-5 --others-lr 5e-5
--seed 42 --out data/models/pii_gliner_base_orgsafe`.
- **Epochs = 3, matched to v2 (reverted from an earlier "2 epochs minimal" — that was wrong).** The cosine
  schedule is defined over the epoch budget, so a 2-epoch run has a *different* LR trajectory: its "epoch 1"
  checkpoint ≠ v2/v4's epoch-1 checkpoint (which sat mid-cosine of a 3-epoch decay). To keep v5 a clean
  single-variable-from-v2 comparison (only the data mix + `max_width` differ), the schedule must match v2:
  3 epochs, cosine, warmup_ratio 0.1. Select epoch 1 (or whichever the dev sweep picks) as before. The ~⅓
  compute "saving" is not worth introducing a schedule confound into the one experiment meant to isolate the
  mix.
- **lr 1e-5 backbone / 5e-5 others** — v2's proven base recipe (base is stable at 1e-5).
- **`max_width = 60` — held CONSTANT across all runs in this experiment** (the config edit is applied
  identically to every Pile-sweep point, so it is a controlled constant, not a co-varying variable within
  the sweep). It only changes the confound *vs v2* (which used 12); to attribute the mix effect cleanly, the
  ORG/generality comparison is **v5-points against each other** (all at width 60), and the width effect is
  isolated separately by the long-MISC-by-length split (below). Set on the model config before training;
  train/eval/deploy MUST all use width 60 (a mismatch silently changes candidate spans).
- Precision/dtype: bf16 on gfx1151; batch 8 (base fits easily; no grad-accum).

**Confound control (Codex R1):** v5 changes three things vs v2 — mix (Pile 25%→18% + ORG guard), `max_width`
(12→60), and nothing else (schedule now matched). To keep results interpretable: (1) `max_width=60` is held
constant across the Pile mini-sweep so within-sweep comparisons isolate the mix; (2) the long-MISC-by-length
split isolates the width effect; (3) realized per-type shares are logged so an ORG change can be tied to the
share it was meant to control. v5 is therefore a *controlled mix sweep at fixed width/schedule*, not a
one-shot multi-variable run.

## Regularization (rely on what's already there — no new machinery)
- **Dropout 0.3** — base's config default (large was 0.1); a strong regularizer well-suited to the ~23k set.
- **weight_decay 0.01** (backbone + others) — unchanged.
- **Diverse Pile-NER slice (18%)** — the *functional* regularizer against label-narrowing (keeps the
  open-label encoder practicing arbitrary phrases) — this is why generality survives FT.
- **≤×2 upsample cap** — anti-memorization on scarce MISC (v2 probe: train 0.926 vs test 0.895 stays in the
  safe gap).
- No dropout increase, no label smoothing, no LR warmup change — YAGNI unless a train/test gap appears.

## Selection & operating point
Per-checkpoint TAB-dev threshold sweep → select by **recall-at-matched-precision / AUPRC** (not fixed-0.3),
**subject to a hard dev-ORG constraint: reject any checkpoint whose dev QUASI/ORG < 0.90** (PR5 — don't let a
high aggregate QUASI hide an ORG hole). Judge ORG by its **dev PR curve / AUPRC**, not recall at one
threshold (Codex R2 — share preservation controls exposure, not learning quality; ORG can still slip via
label competition/thresholding). Prefer the sweep-mode gate (`docs/issues/performance.md`) if built. TAB op
point ≈ 0.02.

**Frontier selection rule (across the Pile sweep):** pick the **lowest `PILE_FRAC`** whose selected
checkpoint satisfies **all** gates (QUASI ≥ 0.95, MISC ≥ 0.89, ORG ≥ 0.90, generality > 0.90 @ prec ≥ 0.431)
— lowest Pile = most ORG/TAB-safe. If **no** sweep point satisfies all gates, the finding is "**no feasible
base frontier point**": base cannot hold TAB (esp. ORG) *and* clear 0.90 generality at once → generality
> 0.90 needs the large backbone (v3), a documented trade — not a v5 failure.

## Evaluation & success criteria (against the spec)
- **C1:** DIRECT ≈ 1.00; **QUASI ≥ 0.95** (hold v2's 0.979 as the bar).
- **C2:** **MISC ≥ 0.89** (keep v4's 0.925 if possible); DEM ≥ 0.96; QUANT ≥ 0.97.
- **PR5 (the fix):** **ORG ≥ 0.90** — the primary thing v5 must correct vs v4 (0.848).
- **C3 generality — two-tier bar** (fixes the R1-flagged inconsistency: 18% Pile cannot be *required* to beat
  v4's 25%-Pile 0.918):
  - **Primary (pass):** generality any-recall **> 0.90 at precision ≥ v2's 0.431**, reported with precision +
    typed (v2 0.843/0.431/0.707; v4 0.918/0.435/0.789). This is the achievable ORG-safe target.
  - **Stretch (not required):** > v4's 0.918. Reached only if the Pile sweep shows base clears it at an
    ORG-safe share — expected NOT to, per the interpolation; that non-result is itself the finding.
- **C5 / width:** MISC recall split by span length (≤12 vs >12 words) — isolates whether `max_width=60`
  recovers the long tail base v1–v4 structurally could not emit. **Also report short-span (≤3 word)
  precision** (width 60 enlarges the candidate/false-positive surface — confirm it doesn't cost short-span
  precision) and the inference latency/VRAM delta vs width 12.
- **Pass = QUASI ≥ 0.95 AND MISC ≥ 0.89 AND ORG ≥ 0.90 AND generality > 0.90 @ precision ≥ 0.431.** A run
  that clears this is a strict improvement over v2 (ORG held, MISC↑, generality↑). Beating v4's generality
  (0.918) is a stretch, not a gate.

## Ablations — STAGED, not a blanket 5-run sweep (cost-correct)
**v2 and v4 are already two points on the ORG↔Pile-share curve** (Pile 10% → ORG 0.948, gen 0.843; Pile 25%
→ ORG 0.848, gen 0.918). So the interpretability Codex asked for does **not** need four fresh runs — it
needs *one* interior point plus the width control, read against the two we have.

**Stage 1 (run first, ~35–40 min):**
- **Pile 18% run** (the point of interest) — gives a 3rd curve point (10/18/25%); with per-type share
  logging, this tests the ORG-dilution hypothesis directly against v2/v4 without new runs for 10/25%.
- **width-60 control** = v2's exact mix (Pile 10%) at `max_width=60`, v2 schedule — separates "span-candidate
  universe 12→60" from "data mix" for every v5-vs-v2 comparison.

**Stage 2 (only if Stage 1 is ambiguous — conditional, ~+1 h):** add `PILE_FRAC ∈ {0.15, 0.22}` (and 0.25 is
≈ v4) **only if** the 18% point lands near a gate boundary (ORG ≈ 0.90 or generality ≈ 0.90) or the
10/18/25% trend is non-monotonic. If 18% cleanly passes or cleanly fails all gates, the extra points are
YAGNI — the frontier is already located.

- **Long-MISC split** — isolates the `max_width=60` effect (held constant across all runs).
- Realized per-type window/mention/token shares logged per build (ties any ORG change to its cause).

## Cost (staged)
- **Stage 1 (the common case): ~35–40 min** — two base runs (Pile 18% + width-60 control), 3 epochs each
  (~15–20 min build+train+eval), serial. Read against the already-run v2/v4 points.
- **Stage 2 (only if ambiguous): +~1 h** — the 0.15/0.22 refinement points.
- Eval halves with the sweep-mode gate (`docs/issues/performance.md`). No large-backbone risk.

The earlier "~2–2.5 h / 4-point sweep required" framing was over-scoped — v2/v4 already anchor the curve, so
Stage 1 answers the question in ~35 min; the full sweep is a conditional fallback, not the plan.

## Risks & caveats
- **Base generality ceiling** — base may not clear 0.90 without the large backbone; that would confirm
  generality needs mix *and* backbone (v3), and v5's win would be "strict improvement on v2 + ORG-safe",
  not the generality crown.
- **max_width=60 side effects (not free).** Enlarges the candidate-span set ~5× (12→60 widths) →
  higher compute/VRAM, larger false-positive surface, and possible short-span precision / calibration
  shifts — not only the intended long-MISC gain. Mitigation: measure short-span (≤3 word) precision + the
  latency/VRAM delta (C5 eval); ensure train/eval/deploy all use width 60 (a mismatch silently changes
  candidate enumeration); confirm the padding-guard still holds (base was clean).
- **ORG-dilution is a hypothesis, not a diagnosis** — the required Pile/share sweep + per-type-share logging
  is what tests it; a single 18% point would leave the ORG cause unproven (Codex R1).
- **Base generality likely < v4's 0.918 at ORG-safe Pile** — the interpolation predicts ~0.88 at 18%;
  the sweep maps where the ORG↔generality frontier actually sits on base.
- Cross-domain MISC beyond legal+bio still unmeasured (no gold) — unchanged limit.

## Artifacts (to be produced)
`data/pii_span_dataset_base_orgsafe/` · `data/models/pii_gliner_base_orgsafe/` ·
`results/base_orgsafe_dev_*.json` · `results/base_orgsafe_{test,generality}.json`.

## Sources
Spec: [detector-model.md](../../docs/specs/detector-model.md). Predecessors:
[v2](2026-07-04-FT-detector-v2-quasi.md) (TAB + ORG bar, recipe),
[v4](2026-07-05-FT-detector-v4-base-genfirst-mix.md) (mix, ORG regression, attribution).
Stock cards: knowledgator/gliner-pii-{base,large}-v1.0 (HF).
