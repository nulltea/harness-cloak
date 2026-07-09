---
type: handoff
status: current
created: 2026-07-09
updated: 2026-07-09
tags: [detector, gliner, false-positives, type-confusion, mining, pre-rl-blocker, learned-pii-detection]
companion: docs/research/learned-PII-detection.md
---

# Handoff — investigate detector performance (non-drug / cross-type false positives) before RL

## Why this matters now (pre-RL blocker)

While preparing the lattice-producer full-run queue we found the mined surface inventory is
polluted with **detector false positives and type confusion** — non-drugs labeled `drug`, non-
procedures labeled `medical-procedure`, etc. We pruned 39 non-drug surfaces from the drug queue as a
**downstream band-aid** (`scripts/spikes/prune_nondrug_noise.py`), but the detector still emits these,
so any re-mine re-introduces them. The RL ranker and the whole round-trip pipeline consume detector
spans; garbage-in → noisy lattices and a noisy RL reward signal. **Resolve the detector before RL.**

This handoff is an *investigation* brief, not an implementation plan — root-cause first.

## Evidence (concrete, from the current run/queue)

Detector emits clearly-wrong spans, grouped by failure mode:

- **Type confusion — labeled `drug` but aren't drugs:**
  - lab tests: `a1c`, `cbc`, `bmp`, `psa`, `hcg`, `afp`, `ldh`, `b u n` (BUN), `pcr`, `hcv antibody test`
  - imaging/diagnostics: `mri`, `ecg`, `ekg`, `emg`
  - devices/supplies: `ace wrap`, `ace wraps`, `i u d` (IUD)
  - clinical abbreviations/findings: `cva`, `cad`, `lad`, `jvd`, `g c s` (GCS)
  - excipient/fragrance & junk: `.alpha.-hexylcinnamaldehyde`, `10 10 20 20`
  - **non-medical entirely**: `durable power of attorney` (a legal document, labeled `drug`)
- **Labeled `medical-procedure` but aren't procedures:** `activities of daily living`, `annual exam`,
  `blood test`, `autoimmune panel`, `range of motion` assessments, generic `... exam`.
- **Labeled `health-condition` but aren't conditions:** `arm`, `range of motion`,
  `full active and passive range of motion`, `0 7 millimeter lesion` (OCR/paraphrase noise).
- **Abbreviation over-capture:** 2–3 letter tokens (`ap`, `bl`, `cda`, `ep`, `gt`, `sh`, `ski`, `dax`)
  captured as entities with no disambiguating context.

The full pruned-from-drug list (39) is reproducible via
`PYTHONPATH=src python scripts/spikes/prune_nondrug_noise.py` (dry run). Cross-type noise is broader
than drug — health-condition and medical-procedure have their own (~16 and ~27 suspicious surfaces
by a crude heuristic; the true rate needs a proper audit, see below).

## Where the detector lives

- **`src/cloak/detect.py`** — GLiNER (zero-shot, TAB categories) ∪ Presidio (patterns). Model:
  `data/models/pii_gliner_multidomain/checkpoint-2479` (fine-tuned GLiNER; stock fallback
  `urchade/gliner_small-v2.1`). Label-phrase→TAB-type map is at the top of the file ("Phrasing
  matters for GLiNER; tune only here").
- **`scripts/build_mined_lattice_profiles.py`** — the miner. `DETECTOR_LABELS` includes `drug`,
  `medical process`(→`medical-procedure`), `medical condition`(→`health-condition`); it runs GLiNER
  over clinical task documents and emits mined surfaces. This is where the polluted surfaces came from.
- Detector design/rationale: **`docs/research/learned-PII-detection.md`** (the composition:
  supervised fixed-schema core + per-user-type gazetteer/zero-shot/fine-tune paths).
- Fine-tune history: **`research-wiki/training/*FT-detector*`** (v1→v4 records) — check what the
  current checkpoint-2479 was trained on and whether these failure classes were in eval.

## Investigation plan (diagnose, don't patch blindly)

Use `superpowers:systematic-debugging` / the `diagnose` skill. Reproduce → quantify → categorize →
root-cause → decide.

1. **Reproduce in isolation.** Run `detect.py`'s detector directly on a handful of the failing
   surfaces/sentences (e.g. a note containing "ordered an MRI", "gave an Ace wrap", "durable power of
   attorney") and confirm GLiNER (not Presidio) is the source and which label it assigns. Separate
   GLiNER FPs from Presidio pattern FPs.
2. **Quantify per type.** On the mined corpus (or a labeled clinical sample), measure precision per
   detector label (`drug`/`medical process`/`medical condition`), not just recall. The lattice work so
   far optimized/observed recall; these FPs say precision is the failure surface. Report a precision
   number per type with a confusion breakdown (what the FP *actually* is: lab/imaging/device/abbrev/
   non-medical).
3. **Categorize the failure modes** (from evidence above): (a) type confusion within medical
   (lab/imaging/device → drug/procedure), (b) abbreviation over-capture without context, (c) fully
   non-medical spans (`durable power of attorney`), (d) OCR/paraphrase junk (`0 7 millimeter lesion`,
   `10 10 20 20`).
4. **Root-cause each.** Candidate causes to test, not assume:
   - GLiNER label *phrasing* (detect.py top map) too broad — e.g. "drug" catches anything drug-adjacent
     (tests ordered alongside drugs). Try sharper label phrases and measure.
   - The fine-tuned checkpoint (checkpoint-2479) over-fires on medical abbreviations — check its train
     mix (FT-detector records) for negatives/hard-negatives of tests/devices/abbrevs.
   - Missing a **post-filter / negative gazetteer** (lab-test/imaging/device/legal term lists) between
     detection and mining. This is likely the cheapest high-leverage fix, but confirm it's needed vs a
     detector-level fix.
   - Presidio patterns mislabeling (less likely for these, but rule it out).
5. **Decide the fix tier** and write it up (own plan): (a) label-phrase tuning, (b) a negative/type
   post-filter at detection or mining time, (c) re-fine-tune with hard negatives (new FT-detector vN),
   or (d) a confidence/length threshold for short abbreviation spans. Pick per the measured
   contribution of each cause — don't stack fixes blindly.

## Constraints / watch-outs

- **Empirical honesty (CLAUDE.md):** measure precision as an outcome on a real labeled sample; don't
  hand-tune a threshold to make one example pass. Report the FP rate before/after any fix.
- The prune spike is a **band-aid for the current run only** — it does not fix the detector. Don't
  mistake "pruned the queue" for "fixed detection".
- GPU: detector inference runs in the host `.venv` on the iGPU (see CLAUDE.md GPU section); one GPU
  process at a time.
- Don't regress recall while chasing precision — measure both (the round-trip utility depends on
  recall; the RL reward depends on both).

## Suggested skills

- `superpowers:systematic-debugging` (or the `diagnose` skill) — reproduce/quantify/root-cause loop.
- `superpowers:brainstorming` then `superpowers:writing-plans` — once the root cause is known, before
  building the fix (label-tuning vs post-filter vs re-fine-tune is a real design fork).
- `research-wiki` — read the `FT-detector` training records for the current checkpoint's train mix;
  register a new record if a re-fine-tune is chosen.
- `experiment-audit` / `auto-review-loop` — validate any precision/recall claim before acting on it.

## References (by path, not duplicated here)

- Detector: `src/cloak/detect.py`; miner: `scripts/build_mined_lattice_profiles.py`
- Design: `docs/research/learned-PII-detection.md`; FT history: `research-wiki/training/*FT-detector*`
- Band-aid evidence: `scripts/spikes/prune_nondrug_noise.py` (39 pruned drug surfaces),
  `scripts/spikes/fix_cache_fallback_entries.py`
- Lattice-producer overhaul context (merged to `main`): `docs/superpowers/plans/2026-07-09-lattice-producer-overhaul.md`,
  `docs/issues/2026-07-09-lattice-producer-generation-quality-issue-register.md`,
  companion run/hybrid handoff: `docs/handoffs/2026-07-09-drug-hybrid-lattice-and-full-run.md`
