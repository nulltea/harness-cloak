---
type: dev-log
status: current
created: 2026-07-12
updated: 2026-07-12
tags: [qa-build, ladder-probes, decision-probes, failure-analysis, pv4, codex]
companion: [docs/issues/2026-07-11-ladder-decision-qa-question-design.md,
            docs/dev/logs/2026-07-12-qa-fix-design-by-surface.md]
---

# QA-probe pv4 super failure analysis (codex pass)

> Independent per-entry failure attribution (codex/gpt-5.5). Companion opus pass:
> `2026-07-12-qa-failure-analysis.md`.

Source: `results/qa_pairs_pv4_super.txt` (2026-07-12 pv4 super sweep).

Implementation read before judging:

- `src/cloak/train/ladder_probes.py`: pv4 ladder/decision prompts, `lint_rung`,
  `locator_lint`, `lint_decision`, `_empty_gold`, `entail_score`, `_category_hit`,
  `validate_ladder`, `validate_decisions`.
- `src/cloak/train/reward.py`: pinned `Qwen3.5-0.8B` reader and the extractive
  `QA_PROMPT` / MC `MC_PROMPT`.
- `docs/issues/2026-07-11-ladder-decision-qa-question-design.md`: prior register.

Verification commands used for totals:

```bash
python3 - <<'PY'
from pathlib import Path
import re, collections
text=Path('results/qa_pairs_pv4_super.txt').read_text()
print('ladder markers', len(re.findall(r'^\[(?:kept|floor|ceiling)\s*\] rung', text, re.M)))
print('decision markers', len(re.findall(r'^\[(?:kept|floor|ceiling|unlinked)\s*\] Q:', text, re.M)))
print('gen rejects', len(re.findall(r'^\[(?:locator|lint|bad_rung)\s*\]', text, re.M)))
print('ladder statuses', collections.Counter(re.findall(r'^\[(kept|floor|ceiling)\s*\] rung', text, re.M)))
print('decision statuses', collections.Counter(re.findall(r'^\[(kept|floor|ceiling|unlinked)\s*\] Q:', text, re.M)))
print('gen gates', collections.Counter(re.findall(r'^\[(locator|lint|bad_rung)\s*\]', text, re.M)))
PY
```

Reconciled source totals:

- Ladder: 57 total = 21 kept + 36 rejected (13 floor, 23 ceiling).
- Decisions: 24 total = 1 kept + 23 rejected (11 floor, 10 unlinked, 2 ceiling).
- Generation-stage rejects: 19 total (13 locator, 5 lint, 1 bad_rung).
- Rejected entries audited below: 78 = 36 + 23 + 19.

## Verdict counts

Counts below are primary failure modes for the 78 rejected entries. Kept-entry
audit is separate.

| Primary mode | Count | Main fix surface |
|---|---:|---|
| bad QUESTION | 31 | teacher prompt / candidate target selection |
| CONTEXT LEAK | 23 | mostly working-as-intended rejection; for kept leaks, inversion/detection/matching |
| GATE artifact | 14 | gate code + detector/alias matching |
| READER failure | 9 | reader / validation prompt |
| SCORER mis-score | 1 | scorer |
| wrong GOLD | 0 | none certain from source |
| Total | 78 |  |

By source:

| Source | bad QUESTION | CONTEXT LEAK | GATE artifact | READER failure | SCORER mis-score | wrong GOLD | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ladder rejects | 16 | 12 | 0 | 7 | 1 | 0 | 36 |
| Decision rejects | 0 | 11 | 10 | 2 | 0 | 0 | 23 |
| Generation rejects | 15 | 0 | 4 | 0 | 0 | 0 | 19 |
| Total | 31 | 23 | 14 | 9 | 1 | 0 | 78 |

## Mode taxonomy

### bad QUESTION (31)

The question itself is the primary defect: ambiguous among several facts,
grounded in a fact not preserved by `OUT_HI`, asks an ontology/category not
stated in the reader context, names the target/finer surface, or uses another
hidden span as its locator. These are teacher-prompt failures.

Examples:

- `mts/0:L280` asks: "Which condition is documented as a health issue that was
  reviewed by Doctor Kumar?" Gold is `osteoporosis`, but the source/output list
  several reviewed conditions. The reader picks a neighbor.
- `aci/D2N004:L917` asks: "What health condition does James have according to
  the doctor's note?" Gold is `diabetes`, but `OUT_HI` names multiple health
  conditions.
- `aci/D2N007:L1340` asks: "What neck exam finding was documented as absent?"
  Gold is `thyroid enlargement`; this is a negated/absent finding omitted from
  `OUT_HI`, not a robust output-preservation probe.

Fix surface:

- Teacher prompt: reject low-salience DOC-only facts that `OUT_HI` did not
  preserve; for ladder, require a unique role anchor visible in the ceiling
  output.
- Teacher prompt: forbid "which condition was reviewed / mentioned / part of
  follow-up" when the output contains a list.
- Teacher prompt: forbid category questions whose gold category is only in the
  external lattice and not evidenced by the output text, unless using a reader
  that is explicitly allowed to do ontology inference.
- Teacher prompt / gate: for generation rejects, keep lint for named-target
  trivia; the source shows most lint rejects are real teacher failures, not false
  positives.

### CONTEXT LEAK (23)

The floor context really gives away the answer, usually through the rung-0
echo/inversion channel (`OUT_LO_FINAL` restores placeholders), a surviving
semantic label in `OUT_LO_P`, or MC options that make the placeholdered context
decidable. Most rejected floor entries are therefore working as intended:
validation correctly drops them. Kept entries with the same pattern are audited
later.

Examples:

- `aci/D2N001:L114`: floor `OUT_LO_FINAL` contains the exact target in "Reports
  therapy for depression is helping"; the floor reader answers `depression`.
- `aci/D2N005:L1059`: floor `OUT_LO_FINAL` still says "Atrial fibrillation";
  the question asks what condition the drug is for, so the answer is leaked.
- `aci/D2N006:L1226`: floor decision context says the condition is
  "musculoskeletal" and the plan continues exercise-like management, so the MC
  gold remains pickable after anonymization.

Fix surface:

- Rejected floor leaks are mostly working-as-intended: do not recover them as
  probes.
- For kept entries with floor leaks missed by the small reader, improve the
  floor reader or add deterministic floor-leak checks before accepting kept
  probes.
- For escaped placeholders in the kept audit, fix inversion/matching so
  `$\text{HEALTH\_CONDITION\_N}$` and `$\text{DRUG\_N}$` variants are restored
  or treated as placeholder echoes.

### GATE artifact (14)

The entry is rejected before validation because the gate cannot link it to a
detected span, because alias/subsurface matching is too strict, or because the
generation gate likely flags a question without an obvious forbidden locator.

Examples:

- `aci/D2N005:L1072`: decision `depends_on: ['Atrial fibrillation']` is
  unlinked even though the detected ladder surface is `afib`. This is an alias
  matching miss.
- `mts/1:L601`: decision `depends_on: ['C T results']` is unlinked for a CT
  imaging decision. This is a spacing/normalization and detection-coverage miss.
- `aci/D2N001:L170` (hypothesis): locator rejects "What broader category does
  the condition that the patient is receiving weekly therapy for belong to?"
  even though the visible question does not name another clinical span. This
  looks like an over-broad hidden-detail match.

Fix surface:

- Detection/matching: use lattice aliases and normalized abbreviation matching
  for `depends_on` (`afib` <-> `atrial fibrillation`, `C T` <-> `CT`).
- Gate code: record which hidden surface triggered `locator_lint`; without that
  field, false positives are hard to diagnose.
- Gate code: separate patient-name/location/other-PII locator matches from
  clinical-span locator matches. They have different downstream consequences.

### READER failure (9)

The answer is plainly available in the reader context, but the pinned
`Qwen3.5-0.8B` reader copies the wrong neighboring plan item, medication, or
option. These are not teacher-gold errors.

Examples:

- `aci/D2N003:L713`: `OUT_HI` states the kidney-stone plan including fluids and
  urine straining; the reader answers `Ultram`.
- `aci/D2N008:L1494`: `OUT_HI` states `Anemia` and a GI referral; the reader
  answers the referral action instead of the condition.
- `aci/D2N008:L1524` (decision): `OUT_HI` says to stay hydrated for the urologic
  condition; MC reader chooses the opposite fluid option.

Fix surface:

- Reader: use a stronger validation reader or at least a second-pass verifier
  for ceiling failures where the gold string appears in, or is directly
  paraphrased by, `OUT_HI`.
- Reader prompt: for ladder exact/category reads, discourage returning plan
  actions when the question asks for a condition/medication/category.
- Keep the small reader for training only if validation has a stronger offline
  build-time filter; otherwise it anti-selects good probes.

### SCORER mis-score (1 rejected, plus 1 kept-audit issue)

The binary containment scorer misses a reasonable answer because morphology or
clinical paraphrase is outside exact/acronym/token-containment.

Rejected example:

- `aci/D2N003:L731`: gold is `migraine`, `hi_answer` is `Migraines`, and the
  score is `0.0`. This should not be a ceiling reject.

Kept-audit example:

- `aci/D2N005:L1041`: rung-1 gold is `phalangeal fracture`; floor answer is
  "Fracture of the middle finger". That is a reasonable paraphrase of a
  phalangeal fracture, but the scorer gives `0.0`, keeping the probe.

Fix surface:

- Scorer: add light normalization for singular/plural medical nouns.
- Scorer: add alias/paraphrase acceptance for profile-backed anatomical
  variants where the profile can supply it; do not reintroduce broad token-F1
  that makes sibling categories pass.

### wrong GOLD (0)

No rejected entry is certainly a teacher-gold error against the DOC. There are
ambiguous source inconsistencies (for example one note mixes hypo/hyper-thyroid
wording), but the DOC and/or `OUT_HI` still support the gold used by the kept
or rejected entry. I did not count those as wrong GOLD.

## Entry-by-entry rejected audit

### Ladder rejects (36)

| Entry | Validation | Primary mode | Reason |
|---|---|---|---|
| `aci/D2N001:L114` depression r0 | floor | CONTEXT LEAK | `OUT_LO_FINAL` restores the exact condition in the therapy sentence. |
| `aci/D2N001:L120` hypertension r0 | ceiling | READER failure | `OUT_HI` has the condition and lisinopril/BP-monitoring plan; reader copies the dose-change action. |
| `aci/D2N001:L126` blood pressure medications r0 | floor | CONTEXT LEAK | Floor final still says "blood pressure medication". |
| `aci/D2N001:L132` nasal congestion r0 | ceiling | bad QUESTION | Target is a low-salience ROS detail not preserved in `OUT_HI`; question is grounded in DOC-only detail. |
| `aci/D2N001:L138` nasal congestion r1 | ceiling | bad QUESTION | Asks for an ontology category the clinician/output never assigns; reader says allergy. |
| `aci/D2N001:L144` echocardiogram r0 | floor | CONTEXT LEAK | Floor final restores the procedure and the cardiac-function context. |
| `mts/0:L262` hypertension r0 | floor | CONTEXT LEAK | Floor final restores the exact condition in the follow-up list. |
| `mts/0:L274` osteoarthritis r0 | floor | CONTEXT LEAK | Floor final restores the exact condition; the question also gives a world-knowledge clue. |
| `mts/0:L280` osteoporosis r0 | ceiling | bad QUESTION | Ambiguous among all conditions reviewed by the doctor. |
| `mts/0:L286` osteoporosis r1 | floor | bad QUESTION | "decreased bone density" gives away the fact/category without needing the note. |
| `mts/0:L292` hypothyroidism r0 | ceiling | bad QUESTION | Ambiguous follow-up-list locator; many conditions were part of follow-up. |
| `mts/0:L298` hypothyroidism r1 | ceiling | bad QUESTION | Same ambiguous locator plus ontology category absent from the output. |
| `mts/0:L304` allergic rhinitis r0 | floor | CONTEXT LEAK | Floor final restores the exact condition. |
| `aci/D2N002:L479` arthritis r0 | ceiling | bad QUESTION | Uses kidney-transplant medication-avoidance detail from DOC; that relation is not in `OUT_HI`. |
| `aci/D2N002:L485` arthritis r1 | ceiling | bad QUESTION | Awkward category question; asks for ontology not present in output and reader returns a plan rationale. |
| `mts/1:L571` migraine r0 | floor | CONTEXT LEAK | Floor final restores "migraine cocktail"; scorer accepts it for `migraine`. |
| `mts/1:L577` Morphine r0 | floor | CONTEXT LEAK | Floor final restores the medication. |
| `aci/D2N003:L713` kidney stones r0 | ceiling | READER failure | Ceiling context plainly contains the condition and fluids/urine-straining plan; reader answers the pain med. |
| `aci/D2N003:L725` imitrex r0 | floor | CONTEXT LEAK | Floor final restores the drug in migraine-medication context. |
| `aci/D2N003:L731` migraine r0 | ceiling | SCORER mis-score | `hi_answer: Migraines` should satisfy gold `migraine`; morphology makes score 0. |
| `aci/D2N004:L905` ibuprofen r0 | ceiling | READER failure | Ceiling context says the patient took Tylenol and ibuprofen; reader chooses a prescribed plan drug. |
| `aci/D2N004:L911` ibuprofen r1 | ceiling | bad QUESTION | Asks for a drug class not stated in `OUT_HI`; extractive reader returns another drug. |
| `aci/D2N004:L917` diabetes r0 | ceiling | bad QUESTION | "What health condition does James have" is ambiguous among multiple conditions. |
| `aci/D2N004:L923` meloxicam r0 | floor | CONTEXT LEAK | Floor final restores the prescribed medication. |
| `aci/D2N004:L929` lisinopril r0 | ceiling | bad QUESTION | "the medication mentioned in the transcript" is ambiguous; several medications are mentioned. |
| `aci/D2N005:L1059` afib r0 | floor | CONTEXT LEAK | Floor output still contains "Atrial fibrillation". |
| `aci/D2N006:L1207` arthritis r0 | floor | CONTEXT LEAK | Floor final restores the exact condition. |
| `aci/D2N007:L1328` chronic back pain r0 | ceiling | READER failure | Ceiling context states chronic back pain and PT; reader returns the PT action. |
| `aci/D2N007:L1340` thyroid enlargement r0 | ceiling | bad QUESTION | Negated absent finding is not preserved in `OUT_HI`; reader correctly abstains/blank. |
| `aci/D2N007:L1346` thyroid enlargement r1 | ceiling | bad QUESTION | Asks ontology of an absent finding; category is not in context. |
| `aci/D2N008:L1458` congestive heart failure r0 | ceiling | READER failure | Ceiling context contains the condition and the listed meds; reader copies the med list. |
| `aci/D2N008:L1476` seasonal allergies r1 | ceiling | bad QUESTION | "patient's condition" is under-specified in a note with several conditions; category is not stated. |
| `aci/D2N008:L1482` echocardiogram r0 | ceiling | bad QUESTION | Procedure was in DOC but omitted from `OUT_HI`; question targets a ceiling-absent fact. |
| `aci/D2N008:L1488` echocardiogram r1 | ceiling | bad QUESTION | Same ceiling omission plus category phrase not present in output. |
| `aci/D2N008:L1494` anemia r0 | ceiling | READER failure | Ceiling context states anemia and GI referral; reader returns the referral plan. |
| `aci/D2N008:L1500` anemia r1 | ceiling | READER failure | Ceiling context states `Anemia - Hematologic`; reader returns `Gastroenterology`. |

### Decision rejects (23)

| Entry | Validation | Primary mode | Reason |
|---|---|---|---|
| `aci/D2N001:L158` lipid-panel decision | unlinked | GATE artifact | `depends_on` names a lab not linked to a detected span; no training signal can be attached. |
| `mts/0:L324` medication-refill condition | floor | CONTEXT LEAK | Floor context preserves blood-pressure/refill context, so the answer remains pickable. |
| `aci/D2N002:L493` autoimmune-panel body system | ceiling | READER failure | A careful reader can choose musculoskeletal from arthritis/joint context; MC reader picks endocrine. |
| `aci/D2N002:L498` Synthroid specialist | floor | CONTEXT LEAK | Floor context preserves endocrine/Synthroid cue, making the specialist pickable. |
| `aci/D2N002:L503` joint-management provider | floor | CONTEXT LEAK | Floor plan explicitly preserves possible PT. |
| `mts/1:L596` non-opioid medication | floor | CONTEXT LEAK | Options/question make the cocktail pickable despite placeholdering. |
| `mts/1:L601` imaging study | unlinked | GATE artifact | `C T` spacing/detection mismatch leaves the decision unlinked. |
| `aci/D2N003:L739` imaging study | unlinked | GATE artifact | Procedure depends_on has no detected span id. |
| `aci/D2N003:L744` reflux refill | unlinked | GATE artifact | Medication depends_on has no detected span id. |
| `aci/D2N004:L937` lumbar-strain specialist | floor | CONTEXT LEAK | Floor context preserves musculoskeletal strain and PT referral. |
| `aci/D2N005:L1067` fracture body system | floor | CONTEXT LEAK | Floor context explicitly says fracture is musculoskeletal. |
| `aci/D2N005:L1072` cardiac-glycoside class | unlinked | GATE artifact | Alias mismatch: `Atrial fibrillation` does not link to detected `afib`. |
| `aci/D2N005:L1077` follow-up imaging | floor | CONTEXT LEAK | Floor plan preserves follow-up x-ray. |
| `aci/D2N006:L1221` gallbladder body system | unlinked | GATE artifact | Organ/finding is not linked to a detected lattice span. |
| `aci/D2N006:L1226` non-pharmacologic intervention | floor | CONTEXT LEAK | Floor context preserves pilates/exercise-like management. |
| `aci/D2N006:L1231` reflux body system | unlinked | GATE artifact | Reflux depends_on is not linked to a detected span. |
| `aci/D2N007:L1354` mood clinician | floor | CONTEXT LEAK | Floor context preserves psychiatric category and dose-increase context. |
| `aci/D2N007:L1359` lingering-discomfort body system | floor | CONTEXT LEAK | Floor context preserves musculoskeletal category. |
| `aci/D2N007:L1364` coronary-prevention medication | unlinked | GATE artifact | CABG-status depends_on is not linked to a detected span. |
| `aci/D2N008:L1514` endoscopic-referral body system | unlinked | GATE artifact | `low hemoglobin` does not link to the detected anemia/lattice span. |
| `aci/D2N008:L1519` heart-failure body system | floor | CONTEXT LEAK | Floor context preserves heart-failure/cardiac treatment context. |
| `aci/D2N008:L1524` urologic preventive measure | ceiling | READER failure | Ceiling context supports hydration; MC reader picks the opposite fluid option. |
| `aci/D2N008:L1529` GI specialist | unlinked | GATE artifact | Procedure-plan depends_on is not a detected sensitive span. |

### Generation-stage rejects (19)

| Entry | Gate | Primary mode | Reason |
|---|---|---|---|
| `aci/D2N001:L165` congestive heart failure r1 | locator | GATE artifact | Hypothesis: visible question does not name another clinical span; likely over-broad hidden-detail match (possibly patient name). |
| `aci/D2N001:L170` depression r1 | locator | GATE artifact | Hypothesis: no obvious other clinical span appears in the visible question. |
| `aci/D2N001:L175` hypertension r1 | locator | bad QUESTION | Uses other hidden facts (`lisinopril`, blood-pressure monitoring) to locate the target. |
| `aci/D2N001:L180` blood-pressure-medications r1 | locator | bad QUESTION | Names `hypertension`, another hidden clinical fact. |
| `aci/D2N001:L185` echocardiogram r1.0 | bad_rung | GATE artifact | Teacher emitted malformed/non-integer rung and duplicated the exact-value question. |
| `aci/D2N001:L189` lisinopril r1 | locator | bad QUESTION | Uses `hypertension` to locate the target drug. |
| `mts/0:L331` osteoarthritis r1 | lint | bad QUESTION | Names the exact target surface in the question. |
| `mts/0:L336` kidney stones r1 | lint | bad QUESTION | Contains the finer target token `stones`, giving away the target family. |
| `mts/1:L608` migraine r1 | locator | bad QUESTION | Uses `morphine`, another hidden span, to locate the target. |
| `aci/D2N003:L751` imitrex r1 | locator | bad QUESTION | Uses `migraines`, another hidden span, to locate the drug. |
| `aci/D2N003:L756` migraine r1 | locator | bad QUESTION | Uses `Imitrex`, another hidden span, to locate the condition. |
| `aci/D2N004:L944` congestive heart failure r1 | locator | bad QUESTION | Uses other hidden drugs (`lisinopril`, `lasix`) as the locator. |
| `aci/D2N004:L949` diabetes r1 | lint | bad QUESTION | Names the exact target surface. |
| `aci/D2N004:L954` meloxicam r1 | lint | bad QUESTION | Names the exact target surface. |
| `aci/D2N004:L959` lisinopril r1 | lint | bad QUESTION | Names the exact target surface. |
| `aci/D2N005:L1084` afib r1 | locator | bad QUESTION | Uses `digoxin`, another hidden span, to locate the target. |
| `aci/D2N008:L1536` congestive heart failure r1 | locator | bad QUESTION | Uses multiple hidden drugs and doses to locate the condition. |
| `aci/D2N008:L1541` kidney stones r1 | locator | GATE artifact | Hypothesis: visible question is role-grounded and does not name another obvious hidden clinical span; likely patient-name/hidden-detail overmatch. |
| `aci/D2N008:L1546` lisinopril r1 | locator | bad QUESTION | Uses heart failure, another hidden condition, to locate the medication. |

## Kept-entry audit

Kept total: 22 = 21 ladder + 1 decision. I classify 13 as acceptable for this
build and 9 as kept for the wrong or fragile reason.

Wrong/fragile kept entries:

| Entry | Why it is suspect | Likely fix surface |
|---|---|---|
| `aci/D2N001:L108` congestive heart failure r0 | Floor final contains the exact condition in assessment/plan, but floor reader answers another listed condition. Kept by reader noise on an ambiguous PMH question. | reader + teacher prompt |
| `aci/D2N001:L150` lisinopril r0 | Floor final contains the exact drug in plan, but floor reader answers only the dose. | reader / deterministic floor leak check |
| `mts/0:L268` hypertension r1 | Floor pre-inversion preserves "blood pressure management" and refill context; a stronger reader could infer the category. | context-leak check / teacher prompt |
| `mts/0:L316` kidney stones r0 | Question uses flank pain and hematuria, which are not in this DOC/OUT_HI; kept by world-knowledge association, not source grounding. | teacher prompt |
| `aci/D2N003:L719` kidney stones r1 | Floor pre-inversion gives back-pain, blood-in-urine, CT abdomen/pelvis, fluids and urine straining; the condition/category is inferable. | context-leak check / reader |
| `aci/D2N004:L899` congestive heart failure r0 | Floor final contains the exact condition; floor reader returns the medication plan instead. | reader / deterministic floor leak check |
| `aci/D2N005:L1041` distal phalanx fracture r1 | Floor answer "Fracture of the middle finger" is a reasonable paraphrase of `phalangeal fracture`; scorer rejects it. | scorer |
| `aci/D2N008:L1464` kidney stones r0 | Floor final still has escaped LaTeX placeholder `$\text{HEALTH\_CONDITION\_2}$` instead of inversion restoration; floor was artificially blinded. | inversion / placeholder normalization |
| `aci/D2N008:L1506` lisinopril r0 | Same escaped-placeholder issue for `$\text{DRUG\_1}$`; floor was artificially blinded. | inversion / placeholder normalization |

Acceptable kept entries:

| Entry | Why acceptable |
|---|---|
| `aci/D2N002:L467` hypothyroidism r0 | Ceiling preserves the fact; floor changes/obscures it. Source has hypo/hyper wording inconsistency, but ceiling supports the gold. |
| `aci/D2N002:L473` hypothyroidism r1 | Same as above; semantic category is not preserved by the floor. |
| `mts/0:L310` allergic rhinitis r1 | Floor only reaches a coarser allergy label; gold is the finer respiratory-allergy tier. |
| `mts/1:L583` Morphine r1 | Floor has a placeholder, not the drug/class. |
| `mts/1:L591` decision, opioid analgesic | Floor has a placeholder and cannot choose which opioid option; kept for the intended identity loss. |
| `aci/D2N005:L1035` distal phalanx fracture r0 | Floor has only a coarser fracture phrase, not the exact distal-phalanx value. |
| `aci/D2N005:L1047` digoxin r0 | Floor has `Drug 1`, not the exact medication. |
| `aci/D2N005:L1053` digoxin r1 | Floor has `Drug 1`, not the class. |
| `aci/D2N006:L1213` arthritis r1 | Floor preserves only a broad musculoskeletal label; gold is the finer bone-inflammation tier. |
| `aci/D2N007:L1316` depression r0 | Floor has generic health-condition text, not the exact condition. |
| `aci/D2N007:L1322` depression r1 | Floor has psychiatric/coarser context, not the finer mood-disorder tier. |
| `aci/D2N007:L1334` chronic back pain r1 | Floor has generic musculoskeletal/contextual symptoms, not the finer lumbar-pain tier. |
| `aci/D2N008:L1470` seasonal allergies r0 | Floor does not preserve the exact allergy attribution. |

## Register-novel findings

The 2026-07-11 register already names echo leaks, context inference, world
knowledge, reader limits, decision trivia/plan-readbacks, and rung redundancy.
The following were not already named as distinct failure modes:

1. Escaped-placeholder inversion drift: `aci/D2N008` floor outputs use
   `$\text{HEALTH\_CONDITION\_N}$` / `$\text{DRUG\_N}$` forms, and
   `OUT_LO_FINAL` does not restore them. Two rung-0 kept probes are therefore
   kept because the post-inversion floor is artificially blind.
2. Alias/normalization unlinked rejects: `afib` vs `Atrial fibrillation`,
   `C T` vs `CT`, and `low hemoglobin` vs anemia-like spans cause unlinked
   decision rejects even when a human can see the dependency.
3. Locator lint is not diagnosable: generation rejects do not record which
   hidden surface triggered `locator_lint`, so apparent false positives cannot
   be separated from true "uses another hidden span" defects without rerunning
   or instrumenting the gate.
4. Scorer morphology/paraphrase misses: plural `Migraines` vs `migraine` and
   "fracture of the middle finger" vs `phalangeal fracture` show misses that
   are narrower than the prior token-F1/sibling-category concern.

## Ranked fix list by recoverable probe count

"Recoverable" means entries that could plausibly become useful kept probes or
could prevent bad kept probes from entering the set. Floor CONTEXT LEAK rejects
are mostly not recoverable as probes; they are validation doing its job.

1. Teacher prompt and candidate-target filtering: 31 rejected bad QUESTION
   entries plus 4 wrong/fragile kept entries (`mts/0:L316`, `mts/0:L268`,
   and the ambiguous/context-inferable kept questions above). Highest yield.
   Focus on unique ceiling-visible role anchors, no list-wide locators, no
   DOC-only/ceiling-omitted facts, and no ontology category unless the output
   supports the category or the reader is allowed to infer it.
2. Detection/gate/matching normalization: 14 rejected GATE artifacts plus 2
   escaped-placeholder kept failures. Add alias-aware `depends_on` matching,
   CT/C T normalization, low-hemoglobin/anemia matching where profiles support
   it, escaped-placeholder inversion normalization, and record the locator
   trigger surface.
3. Reader upgrade or verifier: 9 rejected READER failures plus at least 4
   wrong kept entries where the floor reader missed leaked exact context. A
   stronger build-time reader/verifier would recover ceiling-available probes
   and stop accepting floor-leaked probes by reader noise.
4. Scorer sharpening: 1 rejected scorer miss plus 1 wrong kept semantic probe.
   Add plural normalization and profile-backed paraphrase aliases without
   broad token-F1.
5. Keep context-leak rejections as negative evidence: 23 rejected CONTEXT LEAK
   entries should remain rejected. The useful action is not to keep them, but to
   use their patterns to harden the teacher and add pre-reader leak checks.

## Conclusion

The build is cleaner than the prior register on parse/lint mechanics, but the
validated set is still not trustworthy as a reward signal without another pass.
The largest true failure is still question design: the teacher frequently asks
ambiguous, DOC-only, world-knowledge, or hidden-neighbor-located questions.
The second-order problem is validation selection bias: the small reader rejects
several ceiling-answerable probes and accidentally keeps several floor-leaked
ones. Finally, pv4 exposes a new implementation issue: escaped placeholder
syntax can bypass inversion and make rung-0 floor checks artificially easy to
pass.
