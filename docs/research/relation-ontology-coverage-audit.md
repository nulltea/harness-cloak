---
type: research
status: current
created: 2026-07-15
updated: 2026-07-15
tags: [qa-builder-v2, relations, detector, aci, ontology, coverage]
companion: docs/specs/qa-builder-v2.md
---

# Relation-ontology coverage audit — ACI

## Verdict

**Span → non-generalizable (PERSON, AGE, profession, nationality, org-medical-facility): nothing
worth adding — the audit confirms the existing relation anti-goals.** The extra zero-shot types are
either speaker-role noise, absent, or false positives, and none carries an explicit task-relevant
dependency to a generalizable clinical span.

**Span → span: two genuine gaps, both between already-detected types, both zero new detector work.**
1. **`medical-procedure → health-condition` (diagnostic yield)** — a diagnostic procedure and the
   finding it *establishes* ("the CT shows a stone"). Strongest candidate: both arguments are
   already-detected generalizable/lattice types, it recurs in ~8–10/15 notes, and it is directionally
   distinct from all six current relations. **Recommend adding.**
2. **`drug → dose`** — a prescribed drug and its strength ("oxycodone five milligrams"). Feasible
   (dose is already captured as `QUANTITY`) and recurs in ~8/15, but its privacy/utility relevance is
   weaker (dose is never anonymized). **Optional; secondary.**

Everything else fails the bar — details below.

## Method

- **Corpus / sample**: `aci`, 15 documents, deterministic `random.Random(0).sample` (doc IDs in
  `summary.json`). ACI = the *Ambient Clinical Intelligence* patient↔doctor dialogue-to-note corpus.
- **Detector**: `knowledgator/gliner-pii-large-v1.0` @ threshold 0.35, `profile="clinical"`, the
  production `QA_V2_CLINICAL_LABELS` map **plus** extra zero-shot labels `profession`, `nationality`,
  `ethnicity`, `religion`, `employer organization`. (`organization medical facility` is already in the
  production map.) gliner is zero-shot, so new types are added purely as label phrases.
- **Passes**: (A) deterministic co-occurrence tabulation of the new types against the generalizable
  clinical spans; (B) a qualitative Sonnet-5 eyeball of all 15 notes for *any* recurring explicit
  dependency the six current relations miss.
- Read-only: no substitution, no teacher call, one GPU process.
- Script: `scripts/spikes/relation_coverage_audit.py`; artifacts in
  `scratch/relation_coverage_audit/` (`spans.json`, `docs.txt`, `summary.json`).

## Part A — span → non-generalizable (the requested new types)

Detected counts across the 15 notes:

| new type | hits | what they actually are | relation candidate? |
|---|---|---|---|
| profession | 52 | ~48 clinician/provider roles (`doctor`×37, `nurse`×3, `cardiologist`, `podiatrist`, `vascular surgeon`, `pcp`, `family doctor`, `therapist`); 2 real patient jobs (`gardener`, `cook`); 1 misfire (`va`) | **No** |
| nationality | 0 | absent from ACI | **No** |
| ethnicity | 1 | `white` in "elderly white gentleman" (exam description) | **No** |
| religion | 1 | `religiously` in "self breast exams religiously" (adverb, false positive) | **No** |
| organization-medical-facility | 7 | generic facility words: `clinic`, `er`, `emergency room`, `infusion center/room`, `outpatient mri facility`, `walmart pharmacy` | **No** |

**Why none clears the bar:**

- **`profession` is 92% speaker-role noise.** The dominant surfaces (`doctor`, `nurse`, `cardiologist`,
  `pcp`, …) are exactly the dialogue `[doctor]` / `[nurse]` speaker roles the spec's anti-goals
  explicitly exclude ("A `[doctor]` speaker role is not a PERSON identity, and the compiler must not
  infer an unnamed clinician"). They are turn markers, not identities.
- **The two real patient professions carry no task-relevant dependency.** `gardener` is a diagnostic
  *activity* question ("are you a big gardener or … just started working in the yard") probing an
  injury cause, not a stable identity. `cook` refers to the patient's **wife** ("your wife being a
  great cook") — a third party, irrelevant to the single patient.
- **`nationality` / `ethnicity` / `religion`** are absent or false positives; even a true hit would be
  a generic attribute-holder relation, excluded by the anti-goals absent an independently justified
  privacy/task role, lattice, and counts.
- **`organization-medical-facility`** hits are generic facility words with **no generalization
  lattice**. Any dependency they carry (e.g. "referred to the infusion center") collapses toward the
  existing `referred_to` relation and could at best be a context/literal argument — it fails the
  detectability-plus-lattice requirement for a controlled type.

This is the anti-goals holding up under measurement: the missed connections here are precisely the
holder/speaker-role/attribute kind the ontology already ruled out.

## Part B — span → span (the eyeball pass)

The Sonnet eyeball found two dependencies the six current relations do **not** capture and that clear
the strict bar.

### 1. `medical-procedure → health-condition` — diagnostic yield (recommend)

A diagnostic procedure and the finding it establishes. Verbatim:

- "review the results of your **ct** and it does show a **stone** … in the proximal right ureter" (D2N050)
- "we did a **x-ray** of that right foot and i do notice **dorsal displacement** … a **lisfranc fracture**" (D2N054)
- "**barium swallow** … showed … **mild narrowing** … **esophagitis**" (D2N062)
- "**biopsy** … came back as grade two … **dcis**" (D2N031); "**endoscopy** … some **gastritis** … a slight **polyp**" (D2N020)

Clears the bar:
- **(a) explicit, maximally task-relevant** — the entire clinical point of imaging/labs is the finding
  they yield; not a holder fact.
- **(b) both arguments already detected** — procedure ∈ `medical-procedure`, every finding above ∈
  `health-condition`. **Zero new detector work.**
- **(c) two lattice-backed generalizable types** — the ideal case.
- **Distinct from the six.** It is *not* `monitored_by` reversed: `monitored_by` (condition→procedure)
  is ongoing surveillance of a *known* condition; this is a diagnostic procedure *establishing a new
  finding* — opposite direction, different clinical fact. Not `treated_with` (therapeutic).
- Recurs in **~8–10/15**. Requires negation handling (several notes have negative results: "x-ray … no
  fractures").

### 2. `drug → dose` (optional / secondary)

A prescribed drug and its strength. Verbatim: "oxycodone **five milligrams**" (D2N050); "omeprazole,
**40 milligrams**" and "allopurinol, **100 milligrams**" (D2N006); "carafate you take **one gram**" (D2N062).

Feasibility is strong: **dose is already captured** — `QA_V2_CLINICAL_LABELS` maps `dose → QUANTITY`,
and QUANTITY produced 25 in-sample hits dominated by real doses (`40 milligrams`, `one gram`, `two
puffs`, `twice a day`). So both arguments are detectable today with no new type — though QUANTITY
conflates dose with frequency and generic numbers, so the argument would need a dose filter.

Weaker on relevance: the drug is the controlled/generalizable argument, but the **dose is never
anonymized** (it is not identity), so this relation mostly tests dose recall across a generalized drug
("does generalizing *omeprazole → proton-pump inhibitor* still let the reader attach *40 mg*?"). That
is a marginal utility signal versus candidate 1. Recurs in ~8/15. If pursued, fold drug→frequency/route
(the full sig) into the same relation rather than adding a separate one.

### Rejected in the eyeball (brief)

- **condition → body-site/laterality / severity/grade** — the site/grade is almost always *fused
  inside the condition span* ("left shoulder pain", "grade two dcis"), so there is no separate entity
  to relate.
- **lab → value** — needs two new types (lab-name + value), lab-name isn't lattice-backed, and values
  are non-identity data far from the privacy target. Tempting (~6/15) but weaker than 1.
- **procedure → drug (interaction/hold)** ("eliquis" gating an epidural) — a real gap in
  `contraindicated_because_of` (which only allows →condition), but ~1/15.
- **drug → allergy**, **condition → temporal/status-change** — counterpart is the patient (holder,
  out of scope) or a non-entity status token; ~1–2/15.
- Detector data-quality noise to ignore: `dax`/`dragon` (ambient-scribe tools) detected as `drug`;
  `saps`/`sats` (SAT exam) as `health-condition`.

## Recommendation

1. **Add `medical-procedure → health-condition` (diagnostic yield)** to the relation inventory — it is
   the one clear coverage gap, needs no detector change, and is directionally distinct from the six.
   Before landing: (i) write its explicit contract + one worked example in the teacher prompt, (ii) add
   negation handling so negative results ("no fracture") don't become false positive findings, (iii)
   validate against `monitored_by` so the teacher doesn't relabel surveillance as diagnostic yield.
2. **Hold `drug → dose`** as optional/secondary — feasible but marginal for the privacy/utility
   objective; revisit only if diagnostic-yield lands and dose recall proves a measured utility signal.
3. **Do not add any span → non-generalizable relation.** The audit confirms the anti-goals empirically:
   the candidate anchors (profession, nationality, org-facility) carry no explicit task-relevant
   dependency in ACI.

## Definitions

- **generalizable / lattice-backed type** — a controlled span type (`drug`, `health-condition`,
  `medical-procedure`) that has a generalization lattice, so an argument can be answered at an allowed
  abstraction level rather than the verbatim surface.
- **non-generalizable / anchor type** — a detected type with no lattice (PERSON, AGE, profession,
  facility); usable only as a fixed anchor or a context/literal argument, never generalized.
- **contextual relation** — a directional, explicitly-stated task-relevant dependency between two
  detected spans, from which a QA item is compiled to test whether anonymization preserves utility.
- **linked decision argument vs context/literal argument** — a linked argument is a controlled frozen
  occurrence with a lattice-support property (gets routing links + representative-anchor check); a
  context/literal argument is an exact typed source string that is not generalized.
- **relation anti-goals** — the spec-level exclusions (`docs/specs/qa-builder-v2.md`): PERSON↔clinical
  holder relations, speaker-role/unnamed-clinician relations, generic attribute-holder relations, and
  `has_status`/`has_category`.
- **speaker role** — a `[doctor]`/`[nurse]`/`[patient]` dialogue turn marker; not a person identity.
- **ACI** — the Ambient Clinical Intelligence dialogue→note clinical corpus (`corpora/clinical/aci.jsonl`).

## Artifacts & sources

- Script: `scripts/spikes/relation_coverage_audit.py`
- Data: `scratch/relation_coverage_audit/{spans,summary}.json`, `docs.txt`
- Companion spec (anti-goals + relation inventory): `docs/specs/qa-builder-v2.md`
- Detector: `src/cloak/detect.py` (`QA_V2_CLINICAL_LABELS`); model
  `knowledgator/gliner-pii-large-v1.0` — <https://huggingface.co/knowledgator/gliner-pii-large-v1.0>
