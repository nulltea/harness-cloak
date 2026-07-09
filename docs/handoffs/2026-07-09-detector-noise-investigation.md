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

**Which detector produced these spans (important):** the polluted surfaces were mined by
`scripts/build_mined_lattice_profiles.py` using its default **stock `knowledgator/gliner-pii-base-v1.0`**
(off-the-shelf, tagged `detector_model: knowledgator/gliner-pii-base-v1.0` on the `mined-clinical`
rows). That is **not** the production detector in `detect.py`, whose default is the fine-tuned
`data/models/pii_gliner_multidomain/checkpoint-2479` (FT-detector v4, *initialized from* the same
knowledgator base). So the FPs are a property of the **stock miner model**, and the first question is
whether the fine-tuned detector already avoids them — the fix may be as simple as re-mining with the
supervised checkpoint rather than repairing the stock model.

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

## Where the detector lives — two distinct models, don't conflate them

- **Miner that produced the polluted spans: `scripts/build_mined_lattice_profiles.py`** — default
  `--model knowledgator/gliner-pii-base-v1.0` (STOCK, off-the-shelf; see lines ~143/216/350). It runs
  GLiNER with `DETECTOR_LABELS` (`drug`, `medical process`→`medical-procedure`, `medical condition`→
  `health-condition`) over clinical task documents. **This stock model is the source of the FPs.** Its
  weakness is documented: `docs/specs/detector-model.md` — "stock knowledgator MISC = 0.32 … Supervision
  is mandatory." Using it for mining was a known-weak choice.
- **Production detector: `src/cloak/detect.py`** — GLiNER (zero-shot, TAB categories) ∪ Presidio,
  default `data/models/pii_gliner_multidomain/checkpoint-2479` (fine-tuned = FT-detector v4,
  *initialized from* `knowledgator/gliner-pii-base-v1.0`). This is a DIFFERENT model and its behavior
  on the failing surfaces is unknown — likely better (the fine-tune exists precisely to fix stock
  noise). The label-phrase→TAB map is at the top of `detect.py` ("Phrasing matters; tune only here").
- Detector design/rationale: **`docs/research/learned-PII-detection.md`** and
  **`docs/specs/detector-model.md`**.
- Fine-tune history: **`research-wiki/training/*FT-detector*`** (v1→v4; v4 record:
  `2026-07-05-FT-detector-v4-base-genfirst-mix.md`, init = knowledgator base). Check the train mix and
  whether these failure classes (lab/imaging/device/abbrev) had hard negatives in eval.
- Prior detector bake-off (why knowledgator base was chosen at all):
  `docs/archive/plans/2026-07-03-pii-span-detector-model.md`.

## Investigation plan (diagnose, don't patch blindly)

Use `superpowers:systematic-debugging` / the `diagnose` skill. Reproduce → quantify → categorize →
root-cause → decide.

1. **Stock vs fine-tuned, first.** Run BOTH the stock miner model (`knowledgator/gliner-pii-base-v1.0`,
   as `build_mined_lattice_profiles.py` uses it) AND the fine-tuned production detector
   (`detect.py` → `checkpoint-2479`) on the failing surfaces/sentences ("ordered an MRI", "gave an Ace
   wrap", "durable power of attorney", short abbrevs). If the fine-tuned detector already avoids these
   FPs, the fix is largely **re-mine with the supervised checkpoint** (or add its config to the miner),
   not repair the stock model. If BOTH fire, it's a deeper label/schema problem. Also separate GLiNER
   FPs from Presidio pattern FPs.
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
   - **Stock model is just weak here** (most likely) — `knowledgator/gliner-pii-base-v1.0` is
     off-the-shelf general-PII, not tuned to this drug/procedure/condition schema; `detector-model.md`
     already flags it needs supervision. If step 1 shows the fine-tuned detector is clean, this is the
     answer and the rest is moot.
   - GLiNER label *phrasing* (the miner's `DETECTOR_LABELS`) too broad — e.g. "drug" catches anything
     drug-adjacent (tests ordered alongside drugs). Try sharper label phrases and measure.
   - Missing a **post-filter / negative gazetteer** (lab-test/imaging/device/legal term lists) between
     detection and mining — cheapest high-leverage backstop; confirm it's needed vs just re-mining.
   - The fine-tune (FT-detector v4) may also lack hard negatives for tests/devices/abbrevs — check its
     train mix if BOTH models fire.
   - Presidio patterns mislabeling (less likely for these, but rule it out).
5. **Decide the fix tier** and write it up (own plan), cheapest-first: (a) **re-mine with the fine-tuned
   `checkpoint-2479`** instead of stock knowledgator (if step 1 shows it's clean); (b) sharpen the
   miner's label phrasing; (c) a negative/type post-filter at mining time; (d) re-fine-tune with hard
   negatives (new FT-detector vN) if even the supervised model fires; (e) a confidence/length threshold
   for short abbreviation spans. Pick per the measured contribution of each cause — don't stack blindly.

## Red-test results (2026-07-09) — both models fail; answers step 1

Red-test harness: `scripts/detector_red_tests.py` — 70 adversarial + control cases grouped by
hypothesized failure mode, run with the **exact miner invocation** (`DETECTOR_LABELS`, threshold 0.3,
`batch_predict_entities`). Each case runs twice: full sentence and bare surface (to separate
context-independent from context-induced failures). A red case "fires" when a forbidden
(surface, runtime-type) is emitted; controls pass when the right type is found. Results JSON:
`results/detector_red_tests/*.json`. Re-run after any fix to compare.

**Key finding — the "just re-mine with the fine-tuned checkpoint" hope is WRONG.** Both the stock
miner model and the fine-tuned production detector (checkpoint-2479, FT-detector v4) fire on these
cases. Fine-tuning helps but does not solve it:

| model | red cases fired (of 54) | controls recall |
|-------|--------------------------|------------------|
| stock `knowledgator/gliner-pii-base-v1.0` | 45 | 16/16 |
| fine-tuned `checkpoint-2479` (FT-detector v4) | 33 | 16/16 |
| stock `knowledgator/gliner-pii-large-v1.0` | 22 | 16/16 |

**`gliner-pii-large` (stock) beats both base-size models.** Whole categories go clean in-sentence:
imaging, non-medical, assessments, clinical abbreviations, and OCR-junk all 0 fired. What remains on
large: lab-tests 6/12, devices 5/6, body-part-as-condition 3/3 (max score 0.96 — worst everywhere),
short tokens 2/8 but at 0.96–0.98 confidence (a threshold gate won't catch those). Controls stay
16/16; the one soft spot is that only one of the two condition-abbreviation red cases resolves to
its correct `health-condition` type (1/2, vs 2/2 on the fine-tune). Implication for fix tier (a): **re-mine with stock large** is a cheaper, stronger first move
than re-mining with the fine-tuned base checkpoint — but it still needs the post-filter for the
persistent lab/device/anatomy confusion, and corpus-level recall vs base must be checked before
switching the miner (red-test controls are not a recall benchmark).

Controls (real drugs/conditions/procedures + contrast pairs `ace inhibitor`, `pcp` drug-sense) pass
16/16 on **both** — recall on true entities is intact; this is a **precision-only** failure, confirming
the handoff's hypothesis.

**Per-mode, on the fine-tuned detector (the one that matters for production):**
- **lab-test-as-drug** (12): still 9/12 fired in-sentence. FT flipped many from `drug`→`health-condition`
  (a1c, afp, ldh, bun, psa, tsh) — that's *still wrong*, just a different wrong type. `cbc`, `bmp`, `hcg`,
  `emg` still land on `drug`. Bare-mode fires only 2/12, so context drives most lab-test capture.
- **body-part-as-condition** (3): 3/3 fired, and FT made it *worse* (arm/left knee/lower back at higher
  scores, 0.46–0.76). Robust context-independent failure.
- **imaging-as-drug** (6): 4/6; FT correctly routes mri/ecg/ekg off `drug` but onto `health-condition`
  (still wrong); `emg` stays `drug`@0.86.
- **ocr-junk** (4): 2–3/4; excipient chemistry (`.alpha.-hexylcinnamaldehyde`@0.72, `limonene`@0.94)
  is confidently mislabeled `drug` by BOTH — highest-confidence errors in the whole suite.
- **short-token-overcapture** (8): 5/8 sentence, 3/8 bare — context both induces (`dax`, `ep`, `sh`)
  and suppresses. `ep`→`organization-medical-facility`@0.74 fires on both models.
- **device-as-drug** (6): FT cut this from 6→3 (nebulizer still `drug`, iud→`health-condition`, ace
  wrap→`medical-procedure`).
- **clinical-abbrev-as-drug** (5): FT nearly fixed (0/5 sentence vs stock 3/5) — the one clear FT win.
- **non-medical-capture** (5): FT cut 2→1 (`living will`→`health-condition` still fires; durable power
  of attorney now clean).

**Failure-mode taxonomy confirmed by the data** (ranked by FT-detector severity):
1. **Persistent type-confusion within medical** — lab/imaging/device tokens land on *some* medical type
   regardless of fine-tuning; FT mostly relocates the error (drug→condition) rather than removing it.
   Root cause is schema/label semantics, not model weights — no label in `DETECTOR_LABELS` means "not an
   entity", and lab/imaging/device are semantically adjacent to drug/procedure.
2. **Body-part-as-condition** — context-independent, FT-worsened. The label set has no "anatomy" sink.
3. **OCR/excipient junk** — highest-confidence errors, both models. Chemistry surface strings read as
   drug names.
4. **Short-token over-capture** — partly context-induced; a length/confidence gate would catch the
   context-independent subset but not `ep`/`dax` (fire at 0.55–0.80).
5. **Non-medical capture** — mostly fixed by FT; least severe residual.

**Decision implication (updates step 5):** re-mining with checkpoint-2479 (fix tier a) is *not*
sufficient — it leaves 33 fired cases and relocates lab/imaging errors rather than removing them. The
leverage is in the schema-level fixes: a **negative/type post-filter** (lab/imaging/device/anatomy/legal
gazetteers, tier c) catches the persistent type-confusion and OCR-junk classes that neither model
avoids, and is cheap. Label-phrase sharpening (tier b) and a re-fine-tune with these hard negatives
(tier d) are the deeper options for the type-confusion root cause. A length/confidence gate (tier e)
only helps the context-independent short-token subset.

## Config-surface audit (2026-07-09) — issues beyond model choice

> **STATUS: all six confirmed defects below are FIXED** on branch `detector-config-surface-fixes`
> (b3888e1..6c9759e, plan `docs/superpowers/plans/2026-07-09-detector-config-surface-fixes.md`,
> final review READY TO MERGE, 26 model-free tests). Design decisions taken: stricter coref aliasing
> (containment both-sides-multi-token or first-token match; bare mention joins most recent chain),
> type-aware dedupe (same-type widest, cross-type higher score with 0.9 floor for pattern-based
> Presidio), per-corpus `DetectorProfile` (reddit=status-quo default / legal / clinical), encoder-window
> word cap derived from the model's `max_len`. The list below documents the pre-fix state.

Probe: `scripts/spikes/detector_config_surface_probes.py` (model-independent parts of `detect.py`,
all confirmed by running, not speculated). Found live defects and risk surfaces independent of which
GLiNER checkpoint is used:

**Confirmed defects (probe output):**
1. **Chunk-boundary span loss** — `_chunks(1200)` hard-cuts mid-word when no `\n`/`. ` falls in the
   window's second half (probe: a name split `...Sara` | `h Johnson...`); chunks don't overlap, so a
   straddling entity is truncated or lost. Silent recall/span-marking hole on long unpunctuated text.
2. **`_dedupe` widest-wins swallows precise spans** — overlap resolution keeps the *widest* span
   regardless of score/source: a 0.31 wide GLiNER MISC span eats a 0.99 Presidio SSN-style CODE span.
   Type and boundary of the union are then wrong. Also compares Presidio's fixed pattern scores
   (0.4/0.6/0.85) with GLiNER probabilities as if commensurable.
3. **`_PRONOUNS` filter drops clinical "RN"** — the Reddit-era stop set includes `rn` (and `ngl`);
   any detected span whose surface lowercases to `rn` (registered nurse) is silently deleted in
   clinical text.
4. **`coref_chains` collapses distinct identities** — same-type token-overlap aliasing puts
   "Anna Smith" and "Peter Smith" in ONE chain (shared surname) → same placeholder → family members
   merged in the anonymized output. Round-trip correctness issue, not just noise.
5. **Custom Presidio patterns misfire on vitals/ranges** — `REF_CODE` (`\d{3,6}/\d{2,4}`) matches
   blood pressure `120/80` and year ranges `2021/22` → labeled CODE; `MONEY` bare-k matches `5k`
   (run distance) and `10M` → QUANTITY.
6. **Encoder window differs per model** — `max_len`: base/fine-tune 2048 but **gliner-pii-large 768**;
   with 1200-char chunks, heavily spaced OCR/ASR text can exceed the large model's window → the chunk
   tail is silently unscanned. Relevant if the miner switches to large (see red-test section).

**Code-evident risk surfaces (not separately probed):**
- **Label phrasing is a schema fork**: production `GLINER_LABELS`/`FINE_DEM_LABELS` vs the miner's
  `DETECTOR_LABELS` are different vocabularies over different type systems; zero-shot behavior is
  phrasing-sensitive by design ("tune only here").
- **Presidio ∋ spaCy statistical NER**: `AnalyzerEngine()` default loads `en_core_web_lg`, so
  PERSON/LOC/NRP/DATE_TIME arrive from spaCy NER too (DATE_TIME notoriously broad), not just regexes;
  "Presidio FP" ≠ "regex FP".
- **`PRESIDIO_MAP` silently drops unmapped types** (URL deliberate; CRYPTO/bank/ITIN etc. implicit).
- **`fine_dem=True` drops Presidio NRP→DEM entirely** (deliberate, but a recall dependency on the
  fine-tune's demographic leaves).
- **`relabel_dem` gazetteers** are order-sensitive first-cut lexicons (train/eval only, not inference).
- **Threshold 0.3** is a cross-domain operating point; red tests show wrong-hit scores up to 0.98, so
  it cannot separate the confirmed FP classes.
- **`batch_predict_entities` is deprecated** (gliner FutureWarning → `GLiNER.inference`) — upgrade
  hazard for both miner and production paths.
- **No unicode/whitespace normalization** before detection; miner `_norm` collapses distinct surfaces
  (punctuation stripped) during dedupe.

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

- Miner (STOCK `knowledgator/gliner-pii-base-v1.0`, source of the FPs): `scripts/build_mined_lattice_profiles.py`
  (default `--model` at ~line 350; `detector_model` tag at ~line 143)
- Production detector (fine-tuned `checkpoint-2479`, a different model): `src/cloak/detect.py`
- Design/rationale: `docs/research/learned-PII-detection.md`, `docs/specs/detector-model.md`
- FT history: `research-wiki/training/*FT-detector*` (v4: `2026-07-05-FT-detector-v4-base-genfirst-mix.md`);
  detector bake-off that chose the knowledgator base: `docs/archive/plans/2026-07-03-pii-span-detector-model.md`
- Band-aid evidence: `scripts/spikes/prune_nondrug_noise.py` (39 pruned drug surfaces),
  `scripts/spikes/fix_cache_fallback_entries.py`
- Lattice-producer overhaul context (merged to `main`): `docs/superpowers/plans/2026-07-09-lattice-producer-overhaul.md`,
  `docs/issues/2026-07-09-lattice-producer-generation-quality-issue-register.md`,
  companion run/hybrid handoff: `docs/handoffs/2026-07-09-drug-hybrid-lattice-and-full-run.md`
