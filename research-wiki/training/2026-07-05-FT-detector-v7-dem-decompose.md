---
type: training-experiment
status: planned
created: 2026-07-05
model: knowledgator/gliner-pii-base-v1.0
dataset: fine-primary DEM (9 leaves) + un-collapsed Nemotron + auto-relabeled TAB DEM + 5% Pile; TAB-8 as eval rollup
result: pending
tags: [detector, gliner, fine-grained, dem-decompose, rollup, lora, lattice, tailorability]
companion: [docs/specs/detector-model.md, research-wiki/training/2026-07-05-FT-detector-v6-lora-gapmix.md]
---

# FT-detector v7 — fine-primary DEM decomposition, TAB-8 as eval rollup

## Objective & hypothesis
The coarse **DEM** type is a grab-bag (nationality/ethnicity/religion/profession/age/gender/marital/
orientation/**health**) — vague, overlapping, and the worst-typed category (v2 DEM typed 0.757, v6 0.68).
It causes real mislabels: `andrew`→DEM (name), `diabetes`→DEM (health condition), `refugee resettlement
agencies`→DEM (an ORG). Root cause: DEM is a coarse rollup over heterogeneous fine concepts that the base
model + Nemotron already represent finely — we *discarded* that granularity by mapping everything to DEM.

**v7 makes the detector fine-primary:** it predicts **fine leaf types**, and **TAB-8 (incl. DEM) is computed
only at eval by rolling the leaves up**. Per CLAUDE.md, the schema is a *generalization lattice*, not a flat
fixed set; DEM stops being a trained blob and becomes a rollup node. TAB-8 survives purely as the
**measurement anchor** (the only real annotated corpus) and for v1–v6 comparability — not as the target.

**Scope: decompose DEM only.** PERSON/ORG/LOC/DATETIME/CODE/QUANTITY stay atomic; **MISC stays coarse** (the
other grab-bag, a harder residual — separate follow-up). v7 builds the fine+rollup machinery on DEM.

**Hypothesis:** fine leaves (less confusable) + un-collapsed fine supervision + more LoRA capacity resolve the
DEM typing errors and **hold or beat the DEM-rollup recall** (v6 0.949 / v2 0.973) while **retaining v6's
generality (0.932)**. Finer types should lift *typed* recall (the model types "religion"/"health-condition",
not the amorphous DEM). **Risk:** the auto-relabeler is noisy (real-but-imperfect fine gold on TAB), and the
rollup could hide fine errors — so fine recall is reported *with* the relabeler-quality caveat, and the
DEM-rollup number stays the clean anchor.

## DEM leaf schema + rollup
Ten leaves (label phrases fed to GLiNER), all rolling up to DEM for TAB scoring:

| leaf | label phrase | roll-up |
|---|---|---|
| nationality | "nationality or citizenship" | DEM |
| ethnicity | "ethnicity or race" | DEM |
| religion | "religion or religious belief" | DEM |
| profession | "profession, occupation or job title" | DEM |
| age | "age" | DEM |
| gender | "gender" | DEM |
| marital-status | "marital status" | DEM |
| **health-condition** | "health condition, disease or medical diagnosis" | DEM* |
| sexual-orientation | "sexual orientation" | DEM |
| **family-role** | "family role or relationship" | DEM | (added during impl — widow/father/daughter/… are frequent DEM spans in TAB) |

\*TAB gold folds health into DEM, so health-condition rolls into DEM for the TAB number — but is **reported
separately** (it is semantically a sibling family, and base has a native `condition` type). Un-mappable TAB
DEM spans fall back to a coarse `demographic-other` leaf (also → DEM) so no span is lost.

## Training data (sources + ratios + type-mapping)
Reuse the builder; add a `--fine-dem` path. Base recipe = v6's gap-mix + 5% Pile (best generality), DEM
supervised at the fine level:

| Source | Role | ~share | Label regime |
|---|---|---|---|
| **TAB train** | 7 TAB types as-is + **DEM spans auto-relabeled to fine leaves** | anchor | 7 TAB phrases + 9 fine DEM phrases |
| **Nemotron-PII** (mapped, **fine DEM un-collapsed**) | fine demographic supervision | ~30% | fine leaves (+ TAB-8 for non-DEM) |
| **Wikipedia-bio** | real MISC/QUASI domain | ~5% | TAB phrases + fine DEM |
| **Pile-NER** | generality (LoRA-head anti-narrowing) | **5%** | own diverse labels |
| **rare-leaf upsample** (≤×2) | scarce fine leaves (religion/orientation/health) + MISC/QUANT | folded | — |

**Nemotron → fine leaf (un-collapse, replaces the old `→DEM`):** `nationality→nationality`,
`race_ethnicity→ethnicity`, `religious_belief→religion`, `occupation/job_title→profession`, `age→age`,
`gender→gender`, `marital_status→marital-status`, `sexuality→sexual-orientation`. (Drop the ambiguous
`language/education_level/employment_status/blood_type` — drop-not-invent.) Non-DEM Nemotron mapping unchanged.

**TAB DEM span → fine leaf (auto-relabeler, the new component):** a gazetteer/rule mapper on the DEM span
surface — nationality/ethnicity lexicon (german, polish, kurdish, gypsy, sami…) → nationality|ethnicity;
religion terms → religion; condition terms (depression, HIV, cancer, disorder, syndrome, -itis…) →
health-condition; profession lexicon/suffixes (journalist, lawyer, officer, -ist) → profession; age patterns
→ age; (homosexual…) → sexual-orientation; else → `demographic-other`. Verified against the DEM surface
distribution already extracted (top surfaces are nationalities; 122 condition surfaces). **Measured coverage
(implemented): 60.8% of TAB-dev DEM spans land on a fine leaf** (nationality 178, family-role 55, health 42,
marital 24, profession 14, …), 39% → `demographic-other` (mostly profession titles + rare nationalities;
rolls to DEM, no fine signal). Gazetteer is a first cut; a model-based relabeler (GLiNER zero-shot on the
span) could lift it if 60.8% proves insufficient. Nemotron adds high-coverage *synthetic* fine supervision.

MISC (coarse) + the other 6 TAB types: unchanged. MultiNERD reserved (eval-only).

## Training config (LoRA, higher rank for fine capacity)
`train_pii_gliner.py --init knowledgator/gliner-pii-base-v1.0 --data-dir data/pii_span_dataset_finedem
--lora --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 --lora-target attn
--epochs 3 --batch-size 8 --lr 2e-4 --others-lr 5e-5 --seed 42 --out data/models/pii_gliner_finedem`.
- **LoRA r=32 / α=64** (up from v6's 16/32) — the fine leaves need more adapter capacity than the coarse
  DEM; v6 underfit gap types at r=16. Keeps v6's frozen-encoder generality mechanism + 5% Pile (the
  head-anti-narrowing dose). If MISC/fine still underfit, `--lora-target attn+ffn` is the next lever.
- Merge-on-save → standard GLiNER checkpoints (per v6). **Fix the per-epoch merge bug first** (v6:
  `PeftModel.from_pretrained` + `modules_to_save` KeyError) or select on the merged `final`.
- Native `max_width=12` (widening infeasible, v5). base backbone (PR4).

## Selection & operating point
Dev sweep on merged checkpoints, recall-at-matched-precision. **Dev-ORG ≥ 0.90 constraint** still applies
(via rollup). TAB op point ≈ 0.02. Two selection views: **DEM-rollup** recall (the anchor) and **per-leaf**
recall.

## Evaluation & success criteria
- **DEM-rollup recall** (fine leaves → DEM, scored vs TAB coarse gold): **≥ v6 0.949**, target ≥ v2 0.973 —
  don't regress the coarse number. This is the backward-compatible anchor.
- **Per-leaf recall** on (a) Nemotron fine gold (synthetic) and (b) auto-relabeled TAB dev (real, noisy) —
  the new capability; report with the relabeler-coverage caveat. Health-condition reported separately.
- **Typed recall on DEM-rollup** — should beat v2 0.757 / v6 0.68 (finer types = clearer typing).
- **Generality (MultiNERD @0.3)** — hold ≈ v6 0.932 (fine decomposition must not cost the LoRA generality win).
- **Other TAB types** — no regression (QUASI ≥ 0.95, MISC/ORG held).
- **Qualitative recheck of the three cases:** `andrew`→PERSON, `diabetes`→health-condition, `refugee
  resettlement agencies`→ORG (not DEM). Pass = errors resolved.
- **Pass = DEM-rollup ≥ 0.949 AND DEM-rollup typed > v6 AND generality ≈ 0.932 AND the three cases fixed.**

## Results (measured)
`pending`.

## Ablations
- Fine (v7) vs coarse-DEM (v6) at the same LoRA recipe — the clean "does decomposition help typed recall" test.
- r=32 vs r=16 (v6) — capacity effect on fine leaves.
- Nemotron-only fine gold vs +TAB-auto-relabel — how much the (noisy) real relabel adds over synthetic.
- HEALTH as DEM-child (rollup) vs sibling (its own reported type) — measure both.

## Cost
Auto-relabeler build + gazetteers ~0.5 day (the main new code); dataset build ~10 min; LoRA train ~15–20 min
(base, r=32); eval (rollup + per-leaf + generality) ~20 min. No large-backbone risk.

## Risks & caveats
- **Auto-relabeler noise** — TAB fine gold is only as good as the gazetteer/rules; the DEM-rollup number is
  the clean anchor, per-leaf-on-TAB is caveated. Report relabeler coverage + spot-check.
- **Rollup hides fine errors** — a leaf mislabel that stays within DEM (e.g. religion↔ethnicity) is invisible
  to the rollup metric; the per-leaf metric catches it (on Nemotron/relabeled-TAB).
- **Fine leaves are scarcer** than coarse DEM → more upsampling / potential underfit of rare leaves
  (religion, orientation); rare-leaf upsample mitigates, watch per-leaf recall.
- **LoRA gap-type underfit may persist** at r=32 → attn+ffn fallback.
- **Per-epoch merge bug** (v6) — fix or select on final.
- MISC still coarse (out of scope) — its vagueness is unaddressed here (follow-up).

## Builder / eval changes required (review before the run)
1. **`build_pii_span_dataset.py` `--fine-dem`:** un-collapse Nemotron DEM (new fine map); auto-relabel TAB
   DEM spans (the gazetteer/rule mapper, `demographic-other` fallback); emit fine leaf phrases; extend
   `RARE_TYPES` to the fine leaves. Self-check: relabeler coverage + no span dropped.
2. **`src/cloak/detect.py`:** add the 9 fine-leaf label phrases to the inference label set (keep the 7 non-DEM
   TAB phrases + MISC).
3. **Rollup in the gate (`latticecloak_detection_gate.py`):** a `FINE→TAB8` map (all DEM leaves → DEM) applied
   to predictions before matching TAB coarse gold — so DEM-rollup recall stays comparable to v1–v6.
4. **Fine eval:** score predicted leaves vs Nemotron fine gold + auto-relabeled TAB dev (per-leaf recall).

## Artifacts (to be produced)
`data/pii_span_dataset_finedem/{train,dev}.jsonl` + `build_shares.json` + relabeler coverage report ·
`data/models/pii_gliner_finedem/{checkpoint-*_merged,final}` · `results/finedem_{dev_*,test,rollup,perleaf,generality}.json`.

## Sources
Spec: [detector-model.md](../../docs/specs/detector-model.md) (DEM decomposition / lattice). Predecessor
recipe: [v6](2026-07-05-FT-detector-v6-lora-gapmix.md) (LoRA + 5% Pile, generality 0.932). Coarse-DEM
baselines: [v2](2026-07-04-FT-detector-v2-quasi.md) (DEM 0.973/0.757), v6 (DEM 0.949/0.68). TAB DEM gold
scope (nationalities + 122 condition surfaces) established this session. Fine label sources:
knowledgator/gliner-pii-base native health/personal labels + Nemotron fine demographic labels.
