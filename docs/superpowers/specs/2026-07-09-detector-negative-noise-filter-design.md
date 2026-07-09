---
type: reference
status: current
created: 2026-07-09
updated: 2026-07-09
tags: [detector, gliner, negative-filter, precision, mining, lattice-producer, learned-pii-detection]
companion: docs/research/learned-PII-detection.md
supersedes: scripts/spikes/prune_nondrug_noise.py
---

# Design — detector negative noise filter (cross-type precision post-filter)

## Problem

The detector (GLiNER ∪ Presidio) emits cross-type false positives: lab tests, imaging/diagnostics,
devices/supplies, anatomy, legal-admin documents, and numeric junk get labeled `drug` /
`health-condition` / `medical-procedure`. Confirmed by the red-test harness
(`scripts/detector_red_tests.py`): stock base fires 45/54, fine-tuned checkpoint-2479 33/54, stock
`gliner-pii-large` 22/54 — with all 16 positive controls passing on every model, i.e. this is a
**precision** failure, not recall. Re-mining with a better model does not remove it; the residual
(labs, devices, anatomy) survives model choice and needs a schema-level backstop. The drug-only
`scripts/spikes/prune_nondrug_noise.py` band-aid does not generalize to the other two families, and a
re-mine re-introduces the noise into `data/lattice_runs/full-run/queue.jsonl` (concretely: a
`gliner-pii-large` re-mine proposed adding `cbc`/`ct`/`ecg`/`i u d`/`pcr` as medical-procedure and
`lad`/`va` as health-condition — see `results/requeue_from_large_remine.json`).

## Goal

A **cross-type negative filter**: a pure predicate that decides whether a detected span is a
confirmed noise-category surface, dropped regardless of the type the detector assigned. **Drop noise
only** — real entities in the wrong type (a vaccine `mmr` filed as procedure, condition `copd` filed
as drug) are NOT this filter's concern and pass through untouched. **Fail-open**: an unmatched
surface always passes (recall is preserved; empirical-honesty rule: don't regress recall chasing
precision).

## Definitions

- **surface** — the normalized detected span text (miner `_norm`: lowercased, non-alphanumeric →
  single spaces, stripped). e.g. `"CBC"`→`"cbc"`, `"B.U.N."`→`"b u n"`.
- **runtime_type** — the queue/profile type the span was assigned: `drug`, `health-condition`,
  `medical-procedure` (the three families the full-run queue covers). Other TAB types
  (PERSON/ORG/LOC/…) are out of scope for this filter.
- **noise category** — one of {lab-test, imaging, device, anatomy, legal-admin, junk}: surfaces that
  are not a drug/condition/procedure entity in ANY type.
- **KEEP allowlist** — surfaces reviewed as real entities that pattern-match a noise category but
  must never be dropped (real drugs/vaccines and legitimate condition abbreviations).

## Architecture

One definition, two consumers (mirrors the Task-5 `_encoder_max_words` pattern: a helper in
`detect.py` imported by the miner).

### The predicate — `src/cloak/detect.py`

```python
def is_noise_span(surface: str, runtime_type: str) -> bool:
    """True iff `surface` is a confirmed noise-category surface (lab-test/imaging/device/anatomy/
    legal-admin/junk) that is not a drug/condition/procedure entity in any type, and so should be
    dropped from detection/mining regardless of the runtime_type assigned. KEEP allowlist wins
    first; unmatched surfaces return False (fail-open)."""
```

- Input `surface` is expected already normalized by the caller's `_norm`; the predicate applies its
  own defensive normalization so it is correct on raw input too (idempotent).
- Category gazetteers/patterns are module-level constants next to the predicate.
- Evaluation order (first hit wins): **KEEP allowlist → any category match → False**.

### Category definitions (from domain knowledge, NOT the red-test surface strings)

Sourced from category *definitions* so validation on the red-test/additions sets is not circular.
Combine compiled regex patterns (for productive suffixes/heads) with modest exact-token gazetteers.

- **lab-test** — pattern `\b(panel|assay|titer|titre|screen|culture|antibody test|serology)\b`;
  gazetteer {cbc, bmp, cmp, bnp, psa, hcg, afp, ldh, bun, pcr, tsh, esr, inr, ua, hba1c, a1c, pt,
  ptt, crp, troponin, ferritin} PLUS the explicit enzyme assays {amylase, lipase, transaminase,
  aminotransferase, phosphatase}.
  NOTE `pt` is ambiguous (prothrombin time vs physical therapy) — include as lab; document it.
  **DO NOT use a `\w+ase\b` suffix pattern** — it matches "dise**ase**" and would drop every
  `…disease` condition (coronary artery disease, Parkinson's disease, COPD, …); measured 26/646
  real queued conditions falsely dropped. Enzyme assays go in the gazetteer, explicit tokens only.
- **imaging/diagnostics** — gazetteer {mri, ct, ct scan, cat scan, ecg, ekg, eeg, emg, ncs, x ray,
  xray, chest x ray, ultrasound, echo, echocardiogram, angiogram, mammogram, dexa, pet, pet scan}.
- **device/supply** — reuse the existing `_DEVICE` pattern from `prune_nondrug_noise.py`
  (`wrap|bandage|gauze|tape|swab|kit|dressing|brace|splint|sponge|applicator|cloth|wipe`) plus
  `pump|catheter|stent|pacemaker|nebulizer` and gazetteer {iud, ace wrap}.
- **anatomy** — gazetteer {arm, leg, knee, hip, shoulder, elbow, wrist, ankle, back, lower back,
  neck, chest, abdomen, foot, hand, lad (left anterior descending artery)}. Multiword body regions
  matched as whole phrases.
- **legal-admin** — pattern `\b(power of attorney|living will|driver'?s? licen[sc]e|insurance
  policy|employment contract|advance directive)\b`.
- **junk** — numeric-only (`^[\d][\d .]*$`), fragrance/excipient (`aldehyde$|cinnamaldehyde|
  limonene|linalool`), and single tokens ≤2 non-space chars.

### KEEP allowlist (wins before every category)

- The reviewed reals from `prune_nondrug_noise._KEEP` / `_KEEP_TOKENS` (tocopherol, ascorbic,
  lipoic, cholecalciferol, niacin, riboflavin, thiamine, folic, pantothen, biotin, retino,
  pyridoxine, cobalamin, …; tokens azo, mmr, pcp, cla, pop).
- **Legitimate condition abbreviations** that collide with the abbreviation-noise intuition but are
  real diagnoses: {cad (coronary artery disease), cva (cerebrovascular accident), gerd, copd, cf
  (cystic fibrosis), chf, dvt, uti, tia, ckd, mi}. These must survive even though they are short.

## Detector wiring — profile-gated

Add `negative_filter: bool` to `DetectorProfile`. `PROFILES`:
- `reddit` (default): `negative_filter=False` — **the merged default must stay bit-identical**.
- `legal`: `negative_filter=False`.
- `clinical`: `negative_filter=True`.

In `Detector.detect()`, after the existing pronoun/symbol filter and `_dedupe`, when
`self.profile.negative_filter` is true, drop any span whose `(text, type)` maps to a queue family and
`is_noise_span(_norm(text), runtime_type)` is true. Only spans whose TAB type corresponds to a queue
family (drug/health-condition/medical-procedure) are candidates; PERSON/ORG/LOC/etc. are never
filtered (the categories are orthogonal, but gate explicitly to be safe). The TAB→family mapping:
GLiNER health-condition leaf and the miner families already align on these three names; use the span
`type` directly and only consider it when it is one of the three queue families.

## Miner + requeue wiring

- `scripts/build_mined_lattice_profiles.py` `build_mined_artifact`: after `_is_generic_surface` and
  before the common/fine lookups, skip a span when `is_noise_span(surface, runtime_type)` — add a
  `noise_skipped` stat counter. Import `is_noise_span` from `cloak.detect`.
- `scripts/spikes/requeue_from_large_remine.py`: apply `is_noise_span` to additions (drop matches)
  and as a guard so a removal is never re-added; drop the local `prune_nondrug_noise.is_noise`
  drug-only guard in favor of the cross-type predicate.

## Validation (empirical honesty — measure, don't tune-to-pass)

Report all numbers; do not adjust gazetteers to make a specific example pass.

1. **Red-test held-out check** — apply `is_noise_span` to every fired `(surface, type)` in
   `results/detector_red_tests/*.json` for base, checkpoint-2479, large. Report per category: how
   many fired cases the filter now drops, and confirm **zero** of the 16 controls (control-drug/
   condition/procedure + contrast pairs) are dropped. A dropped control is a blocking defect.
2. **Additions held-out check** — apply to the 130 live additions in
   `results/requeue_from_large_remine.json`; report count dropped per type and write the dropped +
   kept surface lists to a file for manual eyeball (keep dense lists out of chat per the Fable
   safeguard note).
3. **KEEP regression** — assert every KEEP surface returns False.

## Testing — `src/cloak/tests/test_detect_noise_filter.py` (model-free)

- KEEP wins first: `is_noise_span("mmr", "drug")` False, `is_noise_span("cad", "health-condition")`
  False, `is_noise_span("cva", "health-condition")` False.
- Each category catches a representative and spares a real: `is_noise_span("cbc", "medical-procedure")`
  True but `is_noise_span("appendectomy", "medical-procedure")` False; `is_noise_span("mri", "drug")`
  True but `is_noise_span("metformin", "drug")` False; `is_noise_span("ace wrap", "drug")` True;
  `is_noise_span("arm", "health-condition")` True but `is_noise_span("asthma", "health-condition")`
  False; `is_noise_span("power of attorney", "drug")` True.
- Fail-open: an unmatched real surface returns False.
- Profile gating: `PROFILES["clinical"].negative_filter is True`,
  `PROFILES["reddit"].negative_filter is False`.
- Bit-identical default is already covered by the existing profile tests; add an assertion that the
  reddit profile leaves noise-category spans in (filter off).

Run only the named file(s): `PYTHONPATH=src .venv/bin/python -m pytest
src/cloak/tests/test_detect_noise_filter.py src/cloak/tests/test_detect_profiles.py -v`. Never the
full suite (loads models onto the shared 1-GPU box).

## Constraints

- Empirical honesty: gazetteers derived from category definitions, not from the red-test/additions
  surfaces; report catch-rate and any false-drop as outcomes.
- `reddit`/`legal` default behavior bit-identical (filter off) — do not move any eval operating point.
- Do not delete `prune_nondrug_noise.py` (documents the superseded band-aid); this design's
  frontmatter records the supersession.
- Naming rule: no plan/phase identifiers in code; name after the category/behavior.
- Commit hygiene on this shared checkout: a live producer writes untracked data files; stage only
  named files and verify `git diff --cached --name-only` before every commit.

## Out of scope

- Type-error correction (mmr→drug, copd→health-condition) — a detector-classification concern.
- Short-token confidence gate for un-categorized abbreviations (red-test fix tier e) — separate.
- Switching the miner's default model to `gliner-pii-large` — separate decision (needs a
  corpus-level recall check); this filter is model-agnostic and layers under either model.
