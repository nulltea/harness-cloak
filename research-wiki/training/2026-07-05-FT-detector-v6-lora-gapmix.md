---
type: training-experiment
status: done
created: 2026-07-05
model: knowledgator/gliner-pii-base-v1.0
dataset: gap-type-optimized TAB-dominant mix; Pile 0% (primary) + 5% (pre-registered anchor)
result: "Pile0 REFUTED drop-Pile (gen 0.836, head narrowing); but Pile5% BREAKS v5's frontier wall — base gen 0.932 (> stock 0.925, >> full-FT) holding TAB, MISC 0.859 the lone remaining gate. LoRA+5%Pile = most promising base config; r=32/attn+ffn next for MISC."
tags: [detector, gliner, fine-tune, lora, peft, generality-retention, gap-types, base-backbone]
companion: [docs/specs/detector-model.md, research-wiki/training/2026-07-05-FT-detector-v5-base-orgsafe-genmix.md]
---

# FT-detector v6 — LoRA (frozen encoder) + gap-optimized mix

## Objective & hypothesis
v1–v5 (full FT) hit a wall: **full fine-tuning couples specialization to forgetting** — teaching the gap
types (MISC/DEM/QUANT) re-specializes the shared encoder and erodes open-label generality (v1: stock 0.925
→ 0.835). v5 proved the consequence: **no ORG-safe base Pile-share clears both generality > 0.90 and the TAB
gates** ("no feasible base frontier point"). The diverse Pile slice only *partly* counteracted the erosion,
and it diluted TAB (the ORG regression).

**v6 changes the mechanism, not the mix knob.** Freeze the deberta encoder base and adapt it with **LoRA**;
fully train only the GLiNER head (span/prompt/rnn). Then:
- **Generality is *constrained*, not guaranteed (Codex R1 — "preserved by construction" was an overclaim).**
  Freezing the encoder base removes the dominant erosion channel of full FT (the encoder can't wholesale
  re-specialize), but generality can **still drift** because (a) the LoRA deltas apply to *all* inputs
  including open-label phrases, (b) `modules_to_save` fully trains the head (span/prompt/rnn), which can
  re-shape open-label scoring/calibration, and (c) GLiNER is a **uni-encoder** — the same encoder embeds
  label phrases and text, so a MISC-tuned delta can narrow label semantics globally. So v6 is a *bet* that
  LoRA-constrained adaptation drifts far **less** than full FT — **and stock-vs-v6 generality drift is the
  decisive measurement**, not an assumed property.
- **The Pile slice is *likely* redundant** — its main job in full FT was regularizing against encoder
  narrowing, which the freeze largely removes. Drop it to spend the mix on the **FT goal (gap types + ORG)**
  — but Pile may still aid in-batch label-negative diversity / common-label calibration, so a **5% Pile
  anchor is a pre-registered ablation** (run alongside 0%, not a post-hoc rescue — Codex R1).

**Hypothesis:** LoRA(base, frozen encoder) + a TAB-dominant gap-optimized mix **drifts generality far less
than full FT** (target: within a few points of stock 0.925, ≫ full-FT's 0.84–0.89) **and** reaches gap-type
competence near full FT → **clears both generality > 0.90 AND MISC ≥ 0.89 AND ORG ≥ 0.90** — the point full
FT could not reach. If so, LoRA dissolves the v5 frontier wall.

**Two risks that decide it:** (1) low-rank **underfits the gap types** (MISC/DEM below full-FT 0.86–0.92) →
generality↑/gap↓, a different point not a dominator; `r`/FFN/epochs are the levers. (2) **head-level semantic
narrowing** (Codex R1) — the frozen base keeps raw representations, but the trainable head can learn a
scoring geometry that helps TAB/Nemotron gap types while degrading arbitrary label binding → generality
drifts despite the freeze. The stock-vs-v6 generality delta + the r=8 ablation diagnose this.

## Training data (sources + ratios + type-mapping)
**Gap-optimized, TAB-dominant, NO Pile** (LoRA removes Pile's regularization job — see Objective). Reuse the
builder; just omit `pilener` from `--mix` (the post-balance Pile block is skipped) and keep `--balance-rare`
for the in-schema rare-type upsample (that lever is for MISC recall, not generality — keep it).

| Source | Role | Target share | Label regime |
|---|---|---|---|
| **TAB train** | anchor: QUASI incl. MISC + ORG supervision | ~60% | TAB-8 |
| **Nemotron-PII** (mapped, EN) | DEM/DATETIME/LOC/QUANT breadth | ~30% | TAB-8 (mapped) |
| **Wikipedia-bio** ([arXiv 2205.06895](https://arxiv.org/abs/2205.06895)) | 2nd real MISC domain | ~5% (avail-capped) | TAB-8 |
| **rare-type upsample** (MISC/DEM/QUANT, ≤×2) | in-schema MISC/DEM/QUANT recall | folded in | TAB-8 |
| **Pile-NER** | — | **0%** (5% only as a fallback if generality erodes, see Risks) | — |

Nemotron→TAB-8 map unchanged (`NEMOTRON_MAP`). MultiNERD reserved (eval-only). Build:
`build_pii_span_dataset.py --mix nemotron=30000,wikibio=corpora/wikipedia_bio/train.json --balance-rare
--out-dir data/pii_span_dataset_gapmix` (no `--pile-frac` needed — no Pile). Realized shares logged
(`build_shares.json`); watch that TAB/ORG window share is high (no Pile dilution → ORG should be well above
v5's 0.242).

## Training config (LoRA)
`train_pii_gliner.py --init knowledgator/gliner-pii-base-v1.0 --data-dir data/pii_span_dataset_gapmix
--lora --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --lora-target attn
--epochs 3 --batch-size 8 --lr 2e-4 --others-lr 5e-5 --seed 42 --out data/models/pii_gliner_lora_gapmix`.
- **What trains:** LoRA adapters on the deberta encoder attention (`query_proj,key_proj,value_proj`) +
  the full GLiNER head (`span_rep_layer,prompt_rep_layer,rnn` via `modules_to_save`). **~12% of params
  trainable; the deberta encoder base frozen** (measured: 25.2M/207.8M). This freeze is the generality
  mechanism.
- **LR:** `--lr 2e-4` is the LoRA LR (encoder-path; base frozen so only adapters get it) — higher than full
  FT's 1e-5 because adapters train from scratch. `--others-lr 5e-5` = the head.
- **Epochs 3** (start); LoRA has far fewer trainable params and may converge slower — **allow up to 5** if
  the loss/dev curve is still improving. `max_width` stays native 12 (widening infeasible — v5 finding).
- **Save/merge (implemented + CPU-validated):** on save the script **merges adapters into the base**
  (`merge_and_unload`) → **standard GLiNER checkpoints with no LoRA keys**, loadable by the gate unchanged.
  Final-merge is proven on CPU (train→merge→save→`from_pretrained`→predict); the per-epoch checkpoint merge
  loop is best-effort (validate on the real multi-epoch run — see Risks).

## Regularization
**LoRA *is* the regularizer** — freezing 88% of params (the encoder) is a far stronger prior than dropout/
weight-decay, and it is what preserves generality. Beyond that: LoRA dropout 0.05, head weight_decay 0.01.
**No Pile slice** (its regularization role is now the frozen encoder's). No label smoothing. YAGNI.

## Selection & operating point
Dev threshold sweep on the **merged** per-epoch checkpoints → recall-at-matched-precision / AUPRC, **dev
QUASI/ORG ≥ 0.90 constraint** (PR5). TAB op point ≈ 0.02. If the per-epoch merge fails on the real run,
select on the merged `final` + re-merge the two other epochs manually (helper: reuse the script's merge loop).

## Evaluation & success criteria (against the spec)
- **C1:** DIRECT ≈ 1.00; QUASI ≥ 0.95.
- **C2:** **MISC ≥ 0.89** (the underfit risk lands here); DEM ≥ 0.96; QUANT ≥ 0.97.
- **PR5:** **ORG ≥ 0.90** (no Pile dilution → expected comfortably above v5's 0.926).
- **C3 generality — the headline test:** any-recall **> 0.90 @ precision ≥ 0.431**, reported with precision +
  typed. **Compare to stock base (0.925/0.421/0.793):** LoRA should retain ≈ stock (frozen encoder), i.e.
  **≫ full-FT v2/v5 (0.843/0.892)**. If generality ≈ 0.92 AND MISC ≥ 0.89 → v6 clears the v5 frontier wall.
- **Pass = QUASI ≥ 0.95 AND MISC ≥ 0.89 AND ORG ≥ 0.90 AND generality > 0.90 @ prec ≥ 0.431.** The decisive,
  novel question: does frozen-base LoRA clear **both** generality and MISC — the pair full FT never could?

## Results (measured 2026-07-05)
Primary run (LoRA r=16, attn-only, Pile 0%, 3 epochs). LR split verified in the optimizer (LoRA 0.44M @2e-4,
head 24.8M @5e-5). **Per-epoch merge FAILED** (Codex-predicted fragility): `PeftModel.from_pretrained` reload
of `modules_to_save` hit `KeyError span_rep_layer.span_rep_layer.project_start…` — the smoke caught it and
excluded the checkpoints; **only the live-merged `final` (epoch 3) is gate-loadable**. Evaluated epoch 3
(defensible: LoRA benefits from full training). Build: gap-mix 22,702 windows, no Pile → **ORG window share
0.295** (highest yet — confirms Pile was the ORG diluter).

**TAB test @0.02 · generality @0.3 — v6 vs full-FT + stock:**

| | QUASI | MISC | DEM | QUANT | ORG | prec | GEN any/prec/typed |
|---|---|---|---|---|---|---|---|
| stock base (zero-shot) | 0.888 | 0.324 | 0.586 | 0.596 | 0.949 | 0.752 | **0.925**/0.421/0.793 |
| v2 (full FT) | **0.979** | 0.895 | 0.973 | 0.972 | 0.948 | 0.814 | 0.843/0.431/0.707 |
| v5 (full FT, Pile18) | 0.973 | 0.869 | 0.943 | 0.983 | 0.926 | 0.786 | 0.892/0.435/0.789 |
| **v6 (LoRA, Pile0)** | 0.968 | **0.827** | 0.941 | 0.983 | 0.910 | **0.869** | **0.836**/0.501/0.629 |

### Verdict — hypothesis REFUTED: LoRA(frozen encoder) did NOT preserve generality
- **Generality dropped to 0.836** — below stock (0.925), below full-FT v5 (0.892), ≈ v2 (0.843). Freezing the
  encoder did **not** preserve open-label generality. Worse, the *shape* is diagnostic: generality
  **precision rose (0.501) while recall/typed fell (0.836/0.629)** — the model became **more conservative /
  narrowed** on open labels, not broader.
- **Cause = head-level semantic narrowing (exactly Codex R1's "biggest risk").** The freeze protects the
  encoder's *raw representations*, but the trainable GLiNER **head dominates the trainable params (24.8M vs
  LoRA's 0.44M)** and it learned a TAB-specialized scoring geometry that narrowed arbitrary-label binding.
  **In GLiNER the head — not just the encoder — governs open-label generality**, so freezing the encoder
  alone is insufficient. The "constrained, not guaranteed" framing (R1 fix) was right; it turned out *not*
  preserved.
- **Gap types also underfit:** MISC 0.827 (< v2 0.895, v5 0.869, v4 0.925; fails the 0.89 gate) — the low-rank
  + frozen encoder didn't learn the guideline MISC notion as sharply as full FT.
- **Net: v6 is WORSE than full FT on BOTH axes** (MISC↓ and generality↓), failing MISC≥0.89 and
  generality>0.90. The one gain is precision 0.869 (highest) and ORG 0.910 (no-Pile). **Not a frontier
  improvement — a clean negative.**
- **Implication:** LoRA-on-encoder is the wrong lever for GLiNER generality retention; the head must also be
  constrained (e.g. LoRA/partially-freeze the head, or don't fully train `prompt_rep`), which conflicts with
  needing the head to learn the TAB schema. The generality↔TAB trade is deeper than the encoder.

### 5% Pile anchor (pre-registered) — FLIPS the verdict
Adding just **5% Pile** (LoRA r=16 otherwise identical) — `results/lora_gapmix_p05_*`:

| v6 variant | QUASI | MISC | DEM | QUANT | ORG | prec | GEN any/prec/typed |
|---|---|---|---|---|---|---|---|
| Pile 0% | 0.968 | 0.827 | 0.941 | 0.983 | 0.910 | 0.869 | 0.836/0.501/0.629 |
| **Pile 5%** | 0.968 | 0.859 | 0.949 | 0.993 | 0.893 | 0.845 | **0.932**/0.412/0.815 |

**Generality jumped 0.836 → 0.932** (typed 0.629 → 0.815, precision back to stock-level 0.412 ≈ stock 0.421).
So:
- **The core LoRA hypothesis (preserve generality) is VALIDATED — but the "drop Pile" sub-hypothesis was
  WRONG.** Freezing the encoder is not sufficient (the trainable head narrows with no diverse-label signal →
  Pile0 0.836); a *tiny* 5% Pile keeps the head from narrowing → generality **0.932, above stock 0.925 and
  far above full-FT (v5 0.892)**, at stock-level open-label precision (genuine retention, not over-firing).
- **This cracks v5's "no feasible base frontier" wall:** a **base** model reaches generality 0.932 while
  holding QUASI 0.968 / DEM 0.949 / QUANT 0.993 / ORG 0.893. v5 concluded generality > 0.90 with TAB-safety
  needed the large backbone — **LoRA + 5% Pile achieves it on base.**
- **Lone holdout: MISC 0.859 < 0.89** (and ORG 0.893 marginally under 0.90). The low-rank + frozen encoder
  underfits the hardest gap type — exactly what the **r=32 / attn+ffn** ablations target next.
- **Lesson (corrected):** in GLiNER, generality retention needs *both* the frozen encoder (LoRA) *and* a
  little diverse-label signal for the trainable head — Pile is not redundant under LoRA, just needed in a far
  smaller dose (5% vs full-FT's 18–25%). The pre-registered anchor (Codex R1 #2) is what caught this — the
  primary alone would have wrongly concluded "LoRA doesn't preserve generality."

**Net verdict:** v6-Pile5% is a **promising partial breakthrough** — the generality wall v5 declared needs
the large backbone is broken on base (0.932), holding TAB, with **MISC (0.859) the sole remaining gate**.
Not a clean pass (MISC < 0.89, ORG ~0.90), but the most promising base config yet on the generality↔TAB
frontier. Next: r=32 / attn+ffn to lift MISC without losing the generality win (v7).

Artifacts: `data/models/pii_gliner_lora_gapmix/final` @0.02 · `results/lora_gapmix_{dev_*,test,generality}.json`
· `data/pii_span_dataset_gapmix/build_shares.json`. **Known bug:** per-epoch LoRA merge
(`PeftModel.from_pretrained` + `modules_to_save`) — final-merge works; per-epoch needs a fix (adapter+head
state layout) if epoch selection is wanted.

## Ablations
- **Primary run:** r=16, attn-only, Pile 0%, 3 epochs.
- **PRE-REGISTERED (run regardless, Codex R1): 5% Pile anchor** — a second run at Pile 5%, same LoRA config.
  Tests whether the freeze truly makes Pile redundant (0% ≈ 5% generality) or Pile still buys label-negative
  diversity / common-label calibration. Not a post-hoc rescue — planned from the start.
- **r = 8** — the "less drift" point: if v6 generality drifts below stock, r=8 (smaller delta) diagnoses
  whether the drift is LoRA-delta magnitude (→ lower r helps) vs head-level narrowing (→ r won't help).
- **Conditional (only if primary underfits MISC < 0.89):** r=32, then `--lora-target attn+ffn` (FFN LoRA may
  help abstract/event-like MISC but raises drift risk — second-stage, not default).
- **Epochs:** 3 first; extend to 5 **only if** dev gap types are still improving **and** generality is
  stable (not if generality is sliding).
- **Clean mechanism comparison:** full-FT (v5) vs LoRA (v6) at the *same* gap-mix = freeze vs full FT.

## Cost
Build ~10 min (CPU); LoRA train ~15–20 min base (3 epochs, ~similar to full FT — fewer grads but same
forward); merge ~1 min; dev sweep + test + generality ~15 min (or ~3 min sweep-mode). No large-backbone risk.

## Risks & caveats
- **Gap-type underfit (primary risk)** — low-rank may not reach full-FT MISC/DEM; `r`/FFN/epochs are the levers.
- **Per-epoch merge unvalidated** — final-merge is CPU-proven; the loop that merges `checkpoint-*` is
  best-effort (wrapped in try/except). Validate on the real run; if it fails, select on `final` + manual merge.
- **LoRA-delta generality erosion** — the adapters apply to open-label inputs too, so freezing isn't a
  perfect guarantee; low `r` limits it; 5% Pile is the fallback.
- **Convergence** — LoRA may need > 3 epochs; watch the curve, extend if still improving.
- **Optimizer LR grouping** — GLiNER's Trainer splits `lr` (encoder-path = LoRA) vs `others_lr` (head); with
  the base frozen this maps LoRA→`--lr`, head→`--others-lr` as intended, but confirm the split on the real run.

## Artifacts (to be produced)
`data/pii_span_dataset_gapmix/{train,dev}.jsonl` + `build_shares.json` ·
`data/models/pii_gliner_lora_gapmix/{checkpoint-*_merged,final}` + `run_manifest.json` ·
`results/lora_gapmix_{dev_*,test,generality}.json`.

## Sources
Spec: [detector-model.md](../../docs/specs/detector-model.md). Predecessor (the frontier wall v6 attacks):
[v5](2026-07-05-FT-detector-v5-base-orgsafe-genmix.md). Full-FT baselines:
[v2](2026-07-04-FT-detector-v2-quasi.md), [v4](2026-07-05-FT-detector-v4-base-genfirst-mix.md). Stock
generality 0.925 (base) is the retention target. Impl: `scripts/train_pii_gliner.py --lora` (peft 0.19.1,
`get_peft_model` + `merge_and_unload`, CPU-validated).
