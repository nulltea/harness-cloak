---
type: handoff
status: current
created: 2026-07-04
updated: 2026-07-04
tags: [handoff, detector, gliner, fine-tune, v3, quasi, generality-retention]
companion: [research-wiki/training/2026-07-04-FT-detector-v2-quasi.md, docs/research/learned-PII-detection.md, docs/research/datasets.md]
---

# Handoff — PII/QUASI span detector v3: more + balanced data, large backbone

**Next focus (from the requester):** train **v3** = more data + **more balanced** data, on the **large
backbone `knowledgator/gliner-pii-large-v1.0`** (init). Goal: push the one thing prior rounds left open —
**out-of-schema open-label generality retention** — and test whether a bigger model absorbs multi-domain
data without forgetting.

## Where things stand (don't re-derive — read these)
- Full detector rationale, properties, threshold guidance: `docs/research/learned-PII-detection.md`
  (§5.1–§5.4). Dataset taxonomy by DIRECT/QUASI: `docs/research/datasets.md`.
- **v1** (TAB-only, done): `research-wiki/training/2026-07-03-FT-detector-v1-tab-quasi.md` — TAB QUASI
  0.857→0.971; open-label generality eroded 0.941→0.835.
- **v2** (multi-domain, done): `research-wiki/training/2026-07-04-FT-detector-v2-quasi.md` — TAB QUASI 0.979
  (dilution didn't hurt), generality recovered 0.835→**0.872**, bio-test 0.989. Full Results +
  Observations + **Claim audit** there.
- **Settled this session (disentangler + memorize probe, `results/v2_disent_*.json`, `v2_train_gate.json`):**
  the earlier "overfit after 1 epoch" was a **fixed-threshold-0.3 artifact** — at matched precision the
  epoch PR curves overlap; train-vs-test gap ~1.5 pt ⇒ **generalizing, not memorizing/overfitting**. The
  *only* real deficit is out-of-schema forgetting (generality 0.941→0.872). This is what v3 targets.

## Pipeline (reuse as-is)
- Data build: `scripts/build_pii_span_dataset.py` — `--mix nemotron=N,pilener=N,wikibio=PATH`, maps aux
  to the 8 TAB types, entity-safe 150-word windows, subword preflight (drops over-budget), TAB-dominant
  caps (`MIX_RATIO` 50/25/15/10), empty-aux drop, dev=TAB-only. Default (no `--mix`) = v1 TAB-only set.
- Train: `scripts/train_pii_gliner.py` — `--init … --data-dir … --out …`; bf16 guard, seed, resume,
  saves `{out}/checkpoint-*` + `{out}/final` + `run_manifest.json`. **Model-agnostic** — just pass
  `--init knowledgator/gliner-pii-large-v1.0`.
- Eval: `scripts/latticecloak_detection_gate.py --corpus … --threshold … --gliner-model …`; generality
  probe: `scripts/spikes/pii_zeroshot_generality.py` (MultiNERD-en, **reserved for eval — never train on it**).

## Datasets on hand
- TAB: `corpora/tab/echr_{train,dev,test}.json` (legal, DIRECT/QUASI, 8 types).
- Wikipedia-bio: `corpora/wikipedia_bio/{train.json(453),test.json(100)}` (fetched from
  github.com/anthipapa/textanonymization; TAB schema; **only aux source with MISC/identifying-events**).
- Nemotron-PII (HF `nvidia/Nemotron-PII`, synthetic, mapped in `NEMOTRON_MAP`), Pile-NER (HF
  `Universal-NER/Pile-NER-type`, diverse labels for generality). MultiNERD-en (HF, eval-only).

## v3 concrete plan (write the spec first: `research-wiki/training/2026-07-05-ft-detector-<slug>.md`)
1. **Large backbone:** `--init knowledgator/gliner-pii-large-v1.0`. **GPU risk:** larger than base — one
   iGPU (gfx1151), one process at a time; likely reduce `--batch-size` + add `--grad-accum`; confirm it
   fits VRAM on a short probe before the full run (perf gate).
2. **More data:** raise Nemotron/Pile-NER caps (e.g. `nemotron=20000,pilener=8000`); consider i2b2-2014
   (real clinical, quasi-typed — DUA-gated, weeks) if clinical is a target.
3. **More balanced data (key ask):** current mix is DATETIME-heavy (~27k) vs scarce MISC (~2.4k) / QUANT
   (~2k). Add **per-type balancing** to `build_pii_span_dataset.py` — cap over-represented types and/or
   upsample windows containing rare gap types (MISC/DEM/QUANT). This is a new builder feature.
4. **Selection methodology fix (do this):** pick the checkpoint by **PR/AUPRC or recall-at-matched-precision**,
   NOT recall at fixed 0.3 (the fixed-0.3 rule picked the least-calibrated epoch and manufactured the phantom
   "overfit" — see v2 claim audit). Per-corpus operating point still applies (TAB ~0.02, non-TAB ~0.3, §5.4).
5. **Success bar:** TAB QUASI ≥ 0.95 held; **generality > 0.90** (the real v3 target — beat v2's 0.872);
   bio-test ≥ v2. Hypothesis: large backbone retains more open-label breadth at equal TAB recall.
6. After the run: `/result-to-claim` on the new doc; update it planned→done with results + Observations.

## Uncommitted right now (commit before/at start of next session)
- Doc corrections: overfit wording in both `research-wiki/training/*.md` + the v2 Claim-audit resolution.
- Disentangler artifacts: `results/v2_disent_*.json`, `results/v2_train_gate.json`.
- (`.log`/`.err`/`.stdout` intentionally uncommitted; `data/` gitignored.)

## Do NOT touch (concurrent workstream)
The RL ranker files are someone else's active work: `src/cloak/train/{reward,ranker}.py`,
`scripts/train_ranker.py`, `research-wiki/training/2026-07-04-RL-ranker-v1-stage1-bandit.md`,
`results/ranker_train_*.json`, `docs/specs/RL/surrogate-ranker-infiller.md`, `review-stage/AUTO_REVIEW.md`.

## Model artifacts (local, gitignored — not portable)
- v2 deployment model: `data/models/pii_gliner_multidomain/checkpoint-2479` @ thr 0.02 (dev-selected).
- v1: `data/models/pii_gliner/checkpoint-2756` @ 0.02. **Neither is wired into `src/cloak/detect.py`**
  (still defaults to zero-shot `gliner_small` @0.3). Wiring v2/v3 as the TAB-corpus detector is a pending
  task (path + per-corpus threshold).

## Conventions (CLAUDE.md)
Training experiments → `research-wiki/training/YYYY-MM-DD-ft-<slug>.md`, spec-before / results-after,
report the win *and* the regression. No plan/property/arm identifiers ("P7", "Arm B") outside their
defining doc — use self-descriptive names. One-off scripts → `scripts/spikes/`. Heavy runs unbuffered
(`python -u`), one GPU process at a time.

## Suggested skills next session
`research-wiki`/plan the v3 spec doc → `experiment-plan` (optional) → build+train → `auto-review-loop`
on any new/changed script (esp. the balancing feature) → `result-to-claim` on the v3 doc →
`systematic-debugging` if results surprise.
