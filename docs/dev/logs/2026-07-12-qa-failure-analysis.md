---
type: dev-log
status: current
created: 2026-07-12
updated: 2026-07-12
tags: [qa-build, ladder-probes, decision-probes, failure-analysis, pv4]
companion: [docs/issues/2026-07-11-ladder-decision-qa-question-design.md,
            docs/dev/logs/2026-07-12-qa-fix-design-by-surface.md]
---

# QA-probe build failure analysis — pv4 super sweep (2026-07-12)

> Per-entry failure attribution (opus pass). Independent second pass:
> `2026-07-12-qa-failure-analysis-codex.md`. Surface-classified fixes:
> `2026-07-12-qa-fix-design-by-surface.md`.

Source: `results/qa_pairs_pv4_super.txt` (10 clinical docs).
Artifacts: `data/probes_ladder_validated.super.json`, `results/ladder_gen_rejects.super.json`.
Code: `src/cloak/train/ladder_probes.py`, `src/cloak/train/reward.py`.
Prior taxonomy: `docs/issues/2026-07-11-ladder-decision-qa-question-design.md`.

Scope: 57 ladder questions (validated) + 24 decision probes (validated) + 19 generation-stage
rejects = 100 entries. Every rejected entry attributed to one primary failure mode; kept entries
audited for wrong-reason survival.

---

## 1. Verdict counts

| Class | kept | floor | ceiling | unlinked | gen-reject | total |
|-------|------|-------|---------|----------|-----------|-------|
| Ladder | 21 | 13 | 23 | — | 19 (locator 13 / lint 5 / bad_rung 1) | 76 |
| Decision | 1 | 11 | 2 | 10 | — | 24 |

Ladder validated = 57 (rung0: 38 validated + 1 gen-killed; rung1: 19 validated + 18 gen-killed).
Kept probes total: 21 ladder + 1 decision = **22**. Of the 21 ladder keeps, **3 are wrong-reason
(echo present at floor, survived only on reader mis-pick)** and 1 is weak — see §5.

---

## 2. Failure-mode taxonomy with counts

### LADDER — floor family (13): probe answerable at the all-placeholder floor → fact not required

| Mode | n | Fix surface |
|------|---|-------------|
| **Echo** (rung0 reads `out_final`; placeholder echoed by remote then `invert()` restores the surface) | 9 | working-as-intended detection of a non-discriminative probe; the real fix is the reader/echo interaction (§5) |
| **Surface survives anonymization** (the surface text is generic and was never placeholdered, or leaks as a treatment/condition name) | 3 | span detection / anonymization coverage |
| **Question telegraphs the answer** (a defining clue in the question lets the reader name the fact from world knowledge) | 1 | teacher prompt |

- Echo (9): `depression r0`, `echocardiogram r0` (D2N001); `hypertension r0`, `osteoarthritis r0`,
  `allergic rhinitis r0` (mts/0); `Morphine r0` (mts/1); `imitrex r0` (D2N003); `meloxicam r0`
  (D2N004); `arthritis r0` (D2N006).
- Surface-survives (3): `blood pressure medications r0` (D2N001 — the phrase "blood pressure
  medication" is not a placeholder and stays verbatim in the floor); `migraine r0` (mts/1 — leaks via
  the un-anonymized treatment name "migraine cocktail"); `afib r0` (D2N005 — "atrial fibrillation" is
  never placeholdered in either output).
- Telegraph (1): `osteoporosis r1` (mts/0) — Q "…associated with decreased bone density?"; the floor
  reader answers "osteoporosis" from the placeholder note because *decreased bone density* is the
  definition. Floor `lo 1.0`.

All 13 are correct rejects (the probe genuinely fails to require the hidden fact). Register named
echo and context-inference; "surface survives anonymization" is a detection-coverage sub-mode.

### LADDER — ceiling family (23): probe unanswerable even at the un-anonymized ceiling

| Mode | n | Fix surface |
|------|---|-------------|
| **Reader failure** (answer is present in `OUT_HI`; the 0.8B extractive reader grabbed a plan action / med list / dose / wrong enumerated neighbour, or a rung-1 class the extractive prompt cannot copy) | 9 | reader model + QA prompt |
| **Question ambiguous** (question does not uniquely identify the target among sibling facts) | 5 | teacher prompt |
| **Absent from ceiling output** (the summarizer dropped the fact entirely, so no reader can recover it) | 6 | span selection (don't probe facts the ceiling omits) / summariser |
| **Question mis-grounded** (grounds on a relationship not stated in `OUT_HI`) | 2 | teacher prompt |
| **Scorer miss** (correct answer scored 0 on a morphology variant) | 1 | scorer |

- Reader failure (9): `hypertension r0`(D2N001, gave plan action), `kidney stones r0`(D2N003, gave
  "Ultram"), `ibuprofen r0`+`ibuprofen r1`(D2N004, gave "meloxicam"), `chronic back pain r0`(D2N007,
  gave "Refer to physical therapy"), `chf r0`(D2N008, gave the med list), `anemia r0`(D2N008, gave
  the referral line), `seasonal allergies r1`+`anemia r1`(D2N008, gave "Anemia"/"Gastroenterology").
- Ambiguous (5): `osteoporosis r0`, `hypothyroidism r0`, `hypothyroidism r1` (mts/0 — six conditions
  co-listed under "reviewed by Doctor Kumar"; reader picks a neighbour); `diabetes r0`, `lisinopril
  r0` (D2N004 — "what condition/medication does the patient have" with several present).
- Absent-from-ceiling (6): `nasal congestion r0`+`r1` (D2N001 — OUT_HI omits it); `thyroid
  enlargement r0`+`r1` (D2N007 — a **negated** exam finding, "no thyroid enlargement", never in the
  summary); `echocardiogram r0`+`r1` (D2N008 — summary drops the imaging).
- Mis-grounded (2): `arthritis r0`+`r1` (D2N002 — "the condition managed by avoiding certain
  medications because of kidney transplant"; that link is not stated in OUT_HI).
- Scorer (1): `migraine r0` (D2N003) — hi answer "Migraines" scored **0.0** vs gold "migraine"
  (plural ≠ singular token; no containment, no F1 overlap). A valid ceiling answer wrongly rejected.

### LADDER — generation rejects (19)

| Gate | n | Verdict | Fix surface |
|------|---|---------|-------------|
| **locator** | 13 | mostly working-as-intended | teacher prompt (anchor problem) |
| **lint** | 5 | correct rejects | none (unrecoverable) |
| **bad_rung** | 1 | recoverable gate artifact | gate code |

- **locator (13)**: the rung-1 (finest-generalization) question grounds the fact through **another
  detected sensitive span**, which the gate correctly refuses (the anchor is anonymized in `out_p`).
  Verified triggers: `hypertension r1` anchors on "lisinopril" (a span); `blood pressure medications
  r1` on "hypertension"; `congestive heart failure r1` on "Martha" (name span); `imitrex r1`/`migraine
  r1` on each other's context; D2N004/D2N008 chf & lisinopril r1 on "lisinopril". This is the **core
  semantic-tier bottleneck**: the teacher cannot find a *non-sensitive* anchor for a generalization
  question, so it reaches for a neighbouring span. The gate is right; the prompt is the problem.
- **lint (5)**: the question **names the fact itself** or a finer-rung token — `osteoarthritis r1`
  ("…that osteoarthritis belongs to"), `kidney stones r1` (leaks "stones"), `diabetes r1`, `meloxicam
  r1`, `lisinopril r1` (D2N004, all name the value). Correct, unrecoverable rejects.
- **bad_rung (1)**: `echocardiogram` emitted with `"rung": 1.0` (float) — `isinstance(rung, int)`
  fails. A one-line coercion (`int(rung)` when integral) would admit it. Recoverable.

### DECISION probes (24)

| Mode | n | Fix surface |
|------|---|-------------|
| **unlinked — out-of-lattice target** (decision turns on a fact that is not a detected protected span: a lab, procedure, or drug not in the profile) | ~9 | decision teacher targeting / detection coverage |
| **unlinked — matching false-negative** (the fact IS a detected span, but `depends_on` phrasing ≠ the surface string, so the substring test misses) | 1 | linking code (`_decision_span_ids`) |
| **floor — plan / column readback** (the answer is stated verbatim in a surviving plan line or in the ASSESSMENT category column) | 7 | decision teacher prompt |
| **floor — world-knowledge via surviving anchor** (answerable from a non-anonymized anchor: Synthroid, pilates, "blood pressure management") | 3 | decision teacher prompt |
| **floor — option-structure leak** (options give it away: 3 opioids + 1 non-opioid) | 1 | decision teacher prompt |
| **ceiling — reader mis-pick** | 2 | reader |
| **kept (genuine)** | 1 | working-as-intended |

- unlinked out-of-lattice (9): lipid panel (D2N001), CT scan (mts/1, D2N003), Protonix (D2N003),
  aspirin/CABG (D2N007), low hemoglobin + endoscopy (D2N008), gallbladder + reflux (D2N006). The
  decision teacher writes about the most *salient* fact in the summary (a lab result, an imaging
  study, an organ), which is often not in the protected lattice → no training signal.
- unlinked matching-artifact (1): D2N005 "cardiac glycoside" decision, `depends_on: ['Atrial
  fibrillation']`, but the detected surface is **"afib"** — `"afib" in "atrial fibrillation"` is
  False, so it drops as unlinked despite afib being a real span (and having "atrial fibrillation" as
  an alias). A gate false-negative.
- floor readback (7): D2N002 PT, D2N004 PT, D2N005 Musculoskeletal, D2N005 X-ray, D2N007
  Psychiatrist, D2N007 Musculoskeletal, D2N008 Cardiovascular — each answer ("physical therapy",
  "Musculoskeletal", "x-ray", "psychiatric", "Heart Failure") is a literal token in the surviving
  plan line or the assessment category column of the floor output.
- floor world/anchor (3): mts/0 Hypertension (via "blood pressure management/refill"), D2N002
  Endocrinology (via un-anonymized "Synthroid"), D2N006 Exercise therapy (via "pilates").
- floor option-leak (1): mts/1 "which non-opioid medication…" — three of four options are opioids.
- ceiling (2): D2N002 autoimmune-panel body-system (reader picked Endocrine), D2N008 kidney-stone
  preventive measure (reader picked "Limit fluid intake").
- kept (1): mts/1 "Which opioid analgesic was administered?" — Morphine is placeholdered at the floor
  (`DRUG_1`), so `lo_pick=None`; ceiling picks Morphine. Genuinely requires the hidden identity.

---

## 3. Representative examples (verbatim)

**Echo floor (rung0 survives inversion):**
```
[floor] rung 0 | surface: depression   Q: What condition is the patient receiving weekly therapy for, without medication?
gold: depression | hi_answer: Depression (1.0) | lo_answer: depression (1.0)   # out_final restored "depression"
```

**Reader failure (answer present, reader grabbed plan text):**
```
[ceiling] rung 0 | surface: anemia (D2N008)   Q: What condition prompted the doctor to refer the patient to a gastroenterologist?
gold: anemia | hi_answer: Refer to gastroenterology for endoscopy and colonoscopy (0.0)   # OUT_HI assessment reads "Anemia — Hematologic — Low"
```

**Scorer miss (plural):**
```
[ceiling] rung 0 | surface: migraine (D2N003)   hi_answer: Migraines (score 0.0)   # "Migraines" vs gold "migraine": no containment/F1 overlap
```

**Locator gen-reject (anchors on a sibling span):**
```
[locator] surface: hypertension | rung: 1   Q: What broader disease category does the condition that requires lisinopril and daily blood pressure monitoring belong to?
# "lisinopril" is itself a detected span -> unreadable in out_p -> correctly rejected
```

**Decision floor (column readback):**
```
[floor] Q: Which body system is associated with the condition described as stable with fluid retention?  gold: Cardiovascular
# floor out_lo_p line: "<HEALTH_CONDITION_1> — Heart Failure — Stable"  -> the category column leaks "Heart Failure"
```

**Decision unlinked matching-artifact:**
```
[unlinked] Q: Which class of medication is the patient currently taking … ?  gold: Cardiac glycoside  depends_on: ['Atrial fibrillation']  span_ids: []
# detected surface is "afib"; substring test misses the expanded form
```

---

## 4. Modes NOT named by the 2026-07-11 register

1. **Scorer morphology miss (singular/plural).** The register flagged `fact_score`/`entail_score`
   *softness* (F1 blurring sibling rungs) but not that a plural surface ("Migraines" vs "migraine")
   scores a hard **0.0** and ceiling-rejects an otherwise-valid probe. New residual alongside the
   documented HTN/renal ones in `fact_score`'s docstring.
2. **Decision-linking matching false-negative (abbreviation vs expansion).** `_decision_span_ids`
   uses raw substring (`surface in dep_text`); when `depends_on` quotes "Atrial fibrillation" but the
   span surface is "afib" (or vice-versa), a genuinely linkable decision drops as *unlinked*. The
   register described "unlinked" only as decisions turning on non-span facts — not this surface-form
   mismatch. Aliases already exist for exactly this pair; the linker doesn't consult them.
3. **Structured-output column leak.** The clinical summary's ASSESSMENT triple
   `<problem> — <category> — <status>` preserves the **category column** ("Heart Failure",
   "psychiatric", "musculoskeletal", "Anemia") even when the problem name is placeholdered. Both the
   decision floor (body-system questions) and some rung-1 category reads (`anemia r1`) are answerable
   from this surviving column. More specific than the register's generic "context-inferable".
4. **Wrong-reason rung-0 keeps.** The register worried the kept set skews non-discriminative; the
   sharper mechanism here is that **echo makes a rung-0 floor answerable, yet reader noise (picking a
   wrong enumerated neighbour / a dose / a med list) lets it survive as KEPT.** So the kept-count is
   inflated: ~3 of 21 ladder keeps flip to floor under a competent reader (§5). Kept-quantity is not
   kept-quality.
5. **Negation filter leak.** The "negation/screening detection filter" in the landed fix stack still
   admitted `thyroid enlargement` ("no thyroid enlargement") as a probe span → 2 dead ceiling probes.

---

## 5. Kept-entry audit — wrong-reason survivals

rung-1 semantic keeps (10) are almost all genuine: the placeholder reads as a literal `<TYPE_N>` or
a generic category at the floor, so `lo` scores 0 reliably (`htn r1`, `depression r1`, `arthritis r1`
D2N006, `Morphine r1`, `digoxin r1`, `dpf r1`, `chronic back pain r1`, `kidney stones r1`,
`allergic rhinitis r1`, `hypothyroidism r1` D2N002).

rung-0 keeps (11) are fragile. Three survive **only because the surface is echoed into `out_final`
but the floor reader mis-picked**, and would become correct floor-rejects under a better reader:

| Entry | Floor has the surface? | Why it "kept" |
|-------|------------------------|---------------|
| `congestive heart failure r0` (D2N001) | yes (echo, restored) | `lo_answer`="hypertension" — reader picked the wrong co-listed condition |
| `lisinopril r0` (D2N001) | yes (echo, restored) | `lo_answer`="40 milligrams a day" — reader returned the dose |
| `congestive heart failure r0` (D2N004) | yes (echo, restored) | `lo_answer`="continue current medications (lisinopril and Lasix)" — reader returned the plan |
| `kidney stones r0` (mts/0) — weak | yes (echo, restored) | `lo_answer`="Flank pain and hematuria" — reader echoed the question's clue |

Genuine rung-0 keeps: `hypothyroidism r0` (D2N002, floor hallucinated "Hyperthyroidism"), `dpf r0`
(floor coarsened to "Fracture of the middle finger"), `digoxin r0`, `kidney stones r0`+`seasonal
allergies r0`+`lisinopril r0` (D2N008, all placeholdered at floor), `depression r0` (D2N007,
placeholdered). **Net: ~18 of 22 keeps are trustworthy; the rung-0 echo-keeps are the liability.**

---

## 6. Ranked recommendations (max kept-probe yield gain)

Baseline kept = 22 (of which ~3 wrong-reason). Estimates are net new *trustworthy* keeps.

**1. Reader: swap the pinned 0.8B extractive reader for a stronger reader AND fix the QA prompt so
it returns a diagnosis/medication NAME, never a plan action / dose / med list.** Highest leverage.
- Fixes the 9 reader-failure ceiling rejects and 5 ambiguous ones once disambiguated.
- Net new keeps: rung-1 semantic recoveries where OUT_HI states the fact but not the category and the
  reader can answer with the finer surface (acceptance set credits it): `arthritis r1` (D2N002),
  `ibuprofen r1` (D2N004), `seasonal allergies r1` (D2N008), `hypothyroidism r1` (mts/0) ≈ **+4**;
  plus rung-0 where the floor is genuinely hidden (`chf r0` D2N008 — only "Heart Failure" column
  leaks, doesn't match "congestive heart failure"; `chronic back pain r0` D2N007) ≈ **+2**.
- Also corrects the 3 wrong-reason rung-0 keeps (they flip to floor — a quality gain, slight count
  loss). **Expected: +5 to +7 net keeps, and the kept set becomes trustworthy.** This is the
  register's Issue 1(b)/3 "reader is the binding constraint" made concrete.

**2. Scorer robustness (cheap, correctness-first).** Add a lemmatiser/plural-normaliser to
`fact_score`/`canon` (migraine↔migraines), and fold the profile aliases the linker/scorer already
have. Directly recovers `migraine r0` from a mis-score (reclassifies to floor — correct, not a keep),
and prevents silent future ceiling-rejects. **Direct keep gain ~0–1; removes a class of phantom
ceiling-rejects.** Do it alongside #1 so recovered ceiling answers aren't lost to morphology.

**3. Decision linking + de-brittle `depends_on` matching.** Match `depends_on` against detected-span
**aliases and fuzzy** (rapidfuzz, the same rule `_match_gold_sentence` already uses), not raw
substring; consider matching against *all* detected surfaces, not only the decision's own list.
Recovers the afib/"atrial fibrillation" decision and reduces spurious unlinked. **Expected: +1–2
decisions.** Low ceiling — the decision tier's deeper problem (7 plan/column readbacks + 3 world-
knowledge + option leaks) needs the register's Issue-1 prompt redesign, not a linking patch; per the
register's option (c), decisions remain ~1 good probe/doc and could be deprioritised.

**Also worth doing (small, cheap):**
- **Filter negated / exam-transient spans** at detection (thyroid enlargement, nasal congestion): the
  ceiling summary reliably omits them → 5 dead ceiling probes avoided (not keeps, but stop wasting
  teacher calls and reader budget).
- **Coerce integral-float rung** in the generation gate (`bad_rung`) → admits the 1 mis-typed entry.
- **Teacher-prompt anchor guidance (rung-1):** instruct the teacher to reuse the rung-0 grounding
  clause (a non-sensitive presenting complaint / documented course), never a sibling drug/condition
  or the patient name. This is the fix for the 13 locator gen-rejects, but the recovered questions
  will still be echo/ambiguity-limited, so pair it with #1. Speculative yield; do it as part of the
  register's open teacher-prompt redesign, not standalone.

**Do NOT** try to "recover" the echo floor-rejects or the lint gen-rejects — both are working as
intended (the probe genuinely doesn't require the fact / the question names the value).
