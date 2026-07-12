---
type: dev-log
status: current
created: 2026-07-12
updated: 2026-07-12
tags: [qa-build, ladder-probes, decision-probes, fix-design, teacher-prompt, reader, pv4, gpt-5.6-sol]
companion: [docs/issues/2026-07-11-ladder-decision-qa-question-design.md,
            docs/dev/logs/2026-07-12-qa-fix-design-by-surface.md]
---

# QA probe-build failure report (GPT-5.6 Sol pass)

> Independent surface-classified fix design (gpt-5.6-sol, effort medium). Companion Claude pass:
> `2026-07-12-qa-fix-design-by-surface.md`. NOTE: this pass mis-identifies the teacher model as
> "Qwen3.6-35B-A3B" — the teacher is `nvidia/nemotron-3-super-120b-a12b`; Qwen3.5-0.8B is the
> reader. The substance (don't swap the teacher) is unaffected.

## Executive decision

The main defect is not teacher factual accuracy: both audits found essentially **0/78 wrong teacher golds**. The build loses or corrupts probes because the teacher is conditioned on the wrong evidence, the harness admits and links candidates too loosely, and the pinned 0.8B reader creates selection bias.

Current measured state:

| Artifact class | Total | Kept | Rejected |
|---|---:|---:|---:|
| Ladder candidates validated | 57 | 21 | 36 |
| Decision candidates validated | 24 | 1 | 23 |
| Generation-stage ladder rejects | 19 | — | 19 |
| All candidates | 100 | 22 | 78 |

Of the 22 keeps, the conservative audit finds about **13 trustworthy**, while the other audit finds about **18 trustworthy**. The disagreement is mostly whether context-inferable semantic questions should count as fragile or invalid. Both agree on at least three wrong-reason rung-0 keeps caused by floor-reader mistakes, and the Codex audit identifies two more rung-0 keeps invalidated by LaTeX placeholder residue.

The highest-yield route is:

1. Give the ladder teacher `OUT_HI`, polarity, and an explicit protected-span inventory.
2. Give the decision teacher only eligible detected targets, referenced by stable span IDs.
3. Improve reader prompting and use a stronger build-time adjudicator before considering a more expensive per-rollout reader.
4. Add deterministic floor-leak and ceiling-presence checks.
5. Repair detection/profile routing, then introduce class-stratified sampling under the ~13-probe budget.

Evidence: [primary audit](/home/timo/repos/agent-cloak/results/qa_pairs_pv4_failure_analysis.md), [independent audit](/home/timo/repos/agent-cloak/results/qa_pairs_pv4_failure_analysis_codex.md), [raw sweep](/home/timo/repos/agent-cloak/results/qa_pairs_pv4_super.txt), [issue register](/home/timo/repos/agent-cloak/docs/issues/2026-07-11-ladder-decision-qa-question-design.md).

---

# Surface 1 — teacher model

Teacher: `Qwen3.6-35B-A3B`, non-thinking, temperature 0.

## 1.1 Factual gold generation is not failing

Evidence:

- Both audits found **0 certain wrong golds among 78 rejects**.
- The one kept decision is genuinely useful.
- Ambiguous source wording exists, notably hypo/hyperthyroid language, but the generated gold remains supported by the DOC or ceiling output.

Root cause: none at the factual-knowledge level.

Fix: do not replace the teacher merely to improve medical knowledge. The observed failures are principally conditioning and instruction-following failures.

Expected recovered yield: approximately zero from a model-only swap.

Cost/risk: a model swap would invalidate caches and introduce an uncontrolled variable without addressing ceiling omission, eligible-target selection, or reader bias.

## 1.2 Instruction adherence is imperfect

Observed failures attributable partly to the model:

- **5 lint rejects** name the exact/finer fact despite explicit prohibition.
- **13 locator rejects** commonly use sibling sensitive facts as anchors.
- **1 `bad_rung` reject** emits JSON numeric `1.0` rather than an integer.
- It repeatedly writes list-wide ambiguous questions such as “What condition does the patient have?”
- It writes ontology questions whose answers are not stated or inferable under the extractive reader contract.

These are included in the audits’ **31 bad-question** failures, rather than being wrong golds.

Root cause:

- The prompt asks the model to invent a unique role from `doc_orig`, although it cannot see whether that role survives in `OUT_HI`.
- The model is not given a machine-readable inventory of forbidden sibling spans.
- It is asked to return free-form dependency phrases rather than target IDs.
- JSON mode constrains syntax, not semantic conformance; `1.0` is valid JSON.

### Fix

Keep the current teacher initially, but add:

1. Better conditioning described under Surface 2.
2. Constrained response schema where supported.
3. One bounded repair attempt for mechanically invalid output.
4. Machine-generated fields rather than teacher-generated fields wherever possible.

Pseudocode:

```python
schema = {
    "type": "object",
    "properties": {
        "probes": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "rung": {"type": "integer", "enum": [0, 1]},
                    "q": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                },
                "required": ["rung", "q", "evidence_quote"],
            },
        }
    },
    "required": ["probes"],
}

reply = teacher.generate(prompt, response_schema=schema)
errors = semantic_lint(reply, prompt_data)

if errors:
    repair = teacher.generate(
        REPAIR_PROMPT.format(
            original_json=reply,
            errors=errors,
            original_data=prompt_data,
        ),
        response_schema=schema,
    )
    reply = repair
```

Do not ask the teacher to emit `surface`, `gold`, aliases, rung phrases, polarity, or target IDs. The harness already knows them and should attach them deterministically.

Expected recovery:

- Direct mechanical recovery: the one integral-float rung.
- Combined with prompt changes: contributes to recovering a useful subset of the 13 locator rejects and preventing the 5 lint rejects.
- Realistic incremental effect beyond prompt redesign: **+1–3 trustworthy candidates**.

Risk: repair loops can make outputs appear valid without making them useful. Every repaired probe must still pass independent ceiling/floor validation.

---

# Surface 2 — teacher prompts and teacher-visible data

This is the highest-yield surface.

## 2.1 Ladder teacher does not see the ceiling output

Current ladder slots:

| Slot | Current data |
|---|---|
| `{output_kind}` | Corpus-derived label such as `clinical note` |
| `{doc}` | Entire original document, `doc_orig` |
| `{surface}` | Exact detected target surface |
| `{type}` | Runtime target type |
| `{sentence}` | One DOC sentence found by surface lookup |
| `{rung1}` | Finest profile generalization |

The ladder teacher does **not** receive `OUT_HI`, even though ceiling validation reads `OUT_HI`.

Measured consequence: about **6 of 23 ladder ceiling rejects** target facts omitted from `OUT_HI`:

- `nasal congestion` r0/r1 in D2N001.
- `thyroid enlargement` r0/r1 in D2N007.
- `echocardiogram` r0/r1 in D2N008.

Two further `arthritis` questions use a DOC-only relationship not stated in `OUT_HI`.

Root cause: generation and validation use different evidence sets.

### Fix: replace `LADDER_PROMPT`

Recommended replacement:

```text
You write two probe questions that test whether a {output_kind} preserved one
specific protected fact.

The questions will be answered from READER_CONTEXT only. The reader never sees
SOURCE_DOCUMENT. Use SOURCE_DOCUMENT only to understand the fact and its
polarity. Every locator and relationship used in a question must be explicitly
supported by READER_CONTEXT.

TARGET
- target_id: {target_id}
- exact value: {surface}
- type: {type}
- polarity: {polarity}
- finest truthful generalization: {rung1}
- source occurrence: {source_sentence}

OTHER PROTECTED SPANS
{protected_span_inventory}

SOURCE_DOCUMENT
{doc_orig}

READER_CONTEXT
{out_hi}

Write exactly two questions:

1. rung 0: the shortest correct answer must be the target's exact value.
2. rung 1: the shortest correct answer must be either the exact value or the
   finest truthful generalization. The question must ask for the property
   expressed by the generalization, not merely ask "what broader category".

Both questions MUST satisfy all rules:

- The answer and the relationship used to locate it are supported by
  READER_CONTEXT.
- The question uniquely identifies this target in READER_CONTEXT. If two or
  more facts could answer it, do not emit that question.
- Do not name or closely paraphrase the exact value, the generalization, or any
  alias of either.
- Do not use any item in OTHER PROTECTED SPANS as a locator, including a
  patient name, condition, drug, procedure, code, location, age, or demographic
  value.
- Do not identify the target by list position, order, section position,
  co-listing, or phrases such as "the condition mentioned/reviewed".
- Do not rely on general medical knowledge that is absent from READER_CONTEXT.
- If polarity is denied or screened, preserve that polarity explicitly. Never
  turn a denied or screened fact into an active diagnosis.
- Ask a wh-question ending in "?"; expect one short condition, medication,
  procedure, or category phrase, never a plan action, dose, list, sentence, or
  explanation.
- Quote the exact READER_CONTEXT text that proves the question is answerable.
  The evidence quote itself is metadata and is not shown to the reader.

If either rung cannot satisfy every rule, omit that rung rather than inventing
a locator.

Reply only as JSON:
{"probes": [
  {"rung": 0, "q": "...", "evidence_quote": "..."},
  {"rung": 1, "q": "...", "evidence_quote": "..."}
]}
```

### Exact slot data

- `{target_id}`: harness-assigned stable span ID. Used only for attribution.
- `{surface}`: detected original surface.
- `{type}`: profile-routing type after matcher resolution.
- `{polarity}`: `active`, `denied`, or `screened`; derived from the occurrence and dialogue response.
- `{rung1}`: profile’s first truthful generalization.
- `{source_sentence}`: preferably the full dialogue turn plus the immediate response for screened/denied facts, not merely `sentence_of(surface)`.
- `{protected_span_inventory}`: every detected sensitive span in the document, including placeholder-only, screened, abstained, and quasi spans. Each row should contain `span_id`, `surface`, `aliases`, `type`, and `polarity`. This is a forbidden-locator list.
- `{doc_orig}`: full original input, for factual interpretation and polarity.
- `{out_hi}`: actual ceiling output used by the ceiling reader. This becomes the authoritative question evidence.

Why include both `doc_orig` and `out_hi`:

- `doc_orig` supplies correct meaning and polarity.
- `out_hi` limits the question to what the remote task actually preserved.
- `evidence_quote` makes this contract machine-checkable.

Do not pass `OUT_LO_P` to the teacher in the first implementation. That would let generation overfit one realized floor output. Floor independence should remain an external validation criterion.

### Harness support

```python
if not quote_is_in_context(probe["evidence_quote"], out_hi):
    reject("bad_evidence_quote")

if target_not_supported(target, probe["evidence_quote"], out_hi):
    reject("ceiling_target_absent")

if not unique_answer_shape(probe, out_hi, target):
    reject("ambiguous_ceiling_locator")
```

Expected recovery:

- Avoids wasting all six ceiling-omitted ladder candidates.
- Repairs or avoids the two DOC-only mis-grounded arthritis questions.
- Could recover **5–9 of the 13 locator rejects**, depending on whether `OUT_HI` contains a unique non-sensitive role.
- Prevents several ambiguous ceiling failures rather than necessarily increasing raw count.
- Expected net: **+4–8 trustworthy ladder keeps**, after reader improvements.

Risk: many facts genuinely lack a non-sensitive unique locator. Omission is preferable to manufacturing a probe.

## 2.2 Role-only grounding collides with sibling sensitive spans

Measured: **13 locator generation rejects**. Most use another hidden condition, drug, procedure, or name:

- hypertension via lisinopril;
- blood-pressure medication via hypertension;
- migraine and Imitrex via each other;
- CHF via lisinopril/Lasix;
- afib via digoxin;
- some apparent false positives use the patient’s name or another non-clinical protected detail.

Root cause: the prompt says “use the fact’s clinical role,” but clinical roles are often represented by another protected fact. It does not provide an explicit forbidden inventory or ask for evidence from `OUT_HI`.

Fix: the ladder prompt above plus:

```python
other_spans = [
    {
        "id": span.id,
        "surface": span.surface,
        "aliases": profile_aliases(span),
        "type": span.type,
        "polarity": span.polarity,
    }
    for span in all_detected_spans
    if span.id != target.id
]
```

The locator gate should return trigger metadata:

```python
def locator_lint(question, target, forbidden_spans):
    q = normalize(question)
    hits = []

    for span in forbidden_spans:
        for form in [span.surface, *span.aliases]:
            if phrase_match(form, q):
                hits.append({
                    "span_id": span.id,
                    "surface": span.surface,
                    "matched_form": form,
                    "type": span.type,
                })

    return LocatorResult(ok=not hits, triggers=hits)
```

Expected recovery: **+3–7 candidates**, overlapping with the ceiling-aware prompt estimate.

Risk: merely loosening `locator_lint` would admit unanswerable questions. Improve prompt/data first; only fix demonstrable gate false positives.

## 2.3 Ambiguous and ontology-unsupported ladder questions

Measured:

- **5 ambiguous ladder ceiling rejects**.
- Several additional category questions ask for taxonomy not expressed in `OUT_HI`.
- Examples include a six-condition follow-up list, “What health condition does James have?”, and “What broader category does the patient’s condition belong to?”

Root cause:

- The prompt does not require a quoted unique ceiling anchor.
- Rung 1 asks the extractive reader to produce externally supplied lattice language even when the output never says it.
- Generic “broader category” wording gives the reader no target.

Fix:

- Require an exact `OUT_HI` evidence quote.
- Forbid list-wide locators.
- Give the teacher the target’s `OUT_HI` realization if the harness can align it.
- Require the rung-1 question to contain a target-specific, non-sensitive relationship.

Additional prompt diff:

```diff
- asking about the PROPERTY the generalization expresses
+ asking about the property expressed by the generalization, grounded in one
+ unique relationship explicitly present in READER_CONTEXT. Do not ask only
+ "what kind/category does the condition belong to?"
```

Admission rule:

```python
target_mentions = align_target_to_out_hi(
    target.surface,
    target.aliases,
    target.rung1,
    out_hi,
)

if not target_mentions:
    mark_probe_status(target, "ceiling_omitted")
    skip_teacher_call(target)
```

This does not require exact surface presence for rung 1: an aligned alias or supported finer/generalized phrase is acceptable. It does require some evidence that the target is represented.

Expected recovery: **+3–6 trustworthy probes**, overlapping with the reader fixes. More importantly, it removes low-value calls and false ceiling failures.

## 2.4 Decision teacher is not given the eligible protected-target set

Current decision slots:

| Slot | Current data |
|---|---|
| `{output_kind}` | Corpus-derived output label |
| `{k}` | Requested number of decisions |
| `{decision_kinds}` | Corpus-specific free-text task types |
| `{doc}` | Full `doc_orig` |
| `{out_hi}` | Ceiling output |

The teacher invents free-text `depends_on` phrases. It is not told which spans are detected, lattice-backed, or creditable.

Measured consequence: **10/24 decisions are unlinked**.

Approximately:

- **9** turn on out-of-lattice facts such as lipid panel, CT, Protonix, CABG/aspirin, low hemoglobin/endoscopy, gallbladder, or reflux.
- **1** is a clear linker false negative: detected `afib` versus `depends_on: "Atrial fibrillation"`.

Root cause: the prompt optimizes for salient decisions, while the harness requires a dependency on a detected lattice span. That requirement is absent from the teacher’s inputs.

### Fix: replace `DECISION_PROMPT`

```text
You design patient-specific multiple-choice decision probes for a
{output_kind}. A probe is useful only if losing the meaning of one listed
protected target makes the answer no longer determinable.

ELIGIBLE PROTECTED TARGETS
{eligible_targets}

SOURCE_DOCUMENT
{doc_orig}

READER_CONTEXT
{out_hi}

Write up to {k} decision probes. Use only targets from ELIGIBLE PROTECTED
TARGETS. Refer to dependencies by target_id; never invent a free-text
dependency.

Every probe MUST satisfy all rules:

1. The correct option is supported by READER_CONTEXT.
2. The answer depends on the meaning of one eligible target or on a documented
   relationship between two eligible targets.
3. The question must not name or closely paraphrase any dependency target,
   alias, or generalization.
4. The question must identify the situation through other non-protected,
   patient-specific evidence explicitly present in READER_CONTEXT.
5. The answer must not be a verbatim plan action, dose, interval, section
   label, assessment category column, or referral already stated in the
   question/context.
6. Do not ask generic world-knowledge questions such as which body system,
   specialist, or drug class corresponds to a named condition or drug.
7. Prefer documented relational decisions: why a treatment is selected or
   constrained; how one condition changes management of another; what
   follow-up is justified by a documented finding. Do not invent clinical
   recommendations beyond READER_CONTEXT.
8. Give 3-5 mutually exclusive options. Make distractors the same semantic
   kind and comparable in specificity. Do not make one option identifiable by
   wording or class polarity alone.
9. Supply one exact READER_CONTEXT evidence quote for each dependency and one
   quote supporting the correct option.
10. If no question meets every rule, return an empty list.

Reply only as JSON:
{"decisions": [{
  "q": "...",
  "options": ["...", "...", "..."],
  "gold": "...",
  "depends_on_ids": ["span-id"],
  "dependency_evidence": {
    "span-id": "exact quote from READER_CONTEXT"
  },
  "answer_evidence": "exact quote from READER_CONTEXT"
}]}
```

### Exact `eligible_targets` data

For each probe-eligible span:

```json
{
  "target_id": "health-condition:3",
  "surface": "arthritis",
  "aliases": ["..."],
  "type": "health-condition",
  "polarity": "active",
  "finest_generalization": "bone inflammation disease",
  "source_occurrence": "...",
  "ceiling_evidence": "..."
}
```

Include only spans that:

- resolved to a profile;
- are admitted for probing;
- have an aligned representation in `OUT_HI`;
- have a known polarity.

Do not include placeholder-only identifiers as decision targets, but include them separately as forbidden question locators.

Why pass exact surfaces and aliases even though questions may not name them: the teacher needs to understand the target and avoid accidentally leaking an alias.

### Better decision shape

Instead of:

```text
Which body system is associated with the condition described as stable with
fluid retention?
```

use a patient-specific relation only if the ceiling supports it:

```text
The patient has developed leg swelling after difficulty limiting salt. Which
management consideration is supported for the chronic problem affected by
this change?
```

Options must not simply reproduce the plan. If `OUT_HI` only states the plan and no rationale, omit the probe.

Expected recovery:

- Prevents roughly **9 out-of-lattice unlinked decisions** from consuming generation capacity.
- Directly recovers the afib decision once ID-based dependencies are used.
- The current data contain only one unquestionably good decision; after redesign, realistic yield remains approximately **1–3 trustworthy decisions per 10 documents**, not 9 recovered decisions.
- The benefit is mostly precision and budget efficiency, not raw decision count.

Risk: the ceiling output often lacks enough rationale for genuine reasoning questions. Returning zero is correct.

## 2.5 Decision floor leaks: plan, category, world knowledge, and options

Measured decision floor failures:

- **7** plan or structured-column readbacks.
- **3** world-knowledge answers from surviving anchors.
- **1** option-structure leak: three opioids versus one non-opioid.
- Total: **11 floor rejects**.

Root cause: prompt rules are semantically weak. “Do not ask for a value written verbatim” does not prevent paraphrased category-column reads or option-set giveaways.

Fixes:

- The replacement prompt explicitly forbids category columns and direct plan semantics.
- Add deterministic option lint:

```python
def lint_options(options, gold):
    signatures = [clinical_option_signature(x) for x in options]

    if unique_semantic_class(gold, signatures):
        return Reject("option_class_outlier")

    if unequal_specificity(gold, options):
        return Reject("option_specificity_outlier")

    if lexical_overlap_with_context(gold) >> max_other_overlap:
        return Reject("verbatim_option_leak")
```

- Treat these 11 as correct rejects; do not attempt to recover them unchanged.

Expected recovery: little direct recovery. It increases trustworthy yield by preventing teacher calls and reader slots from being spent on doomed candidates.

## 2.6 Polarity is hidden from the teacher

The old screening filter demoted interrogative/denied mentions. Demotion correctly closed the floor leak, as the recomputed D2N002 anchor demonstrates: Synthroid, Tylenol, transplant status, and related facts are now placeholdered at the floor.

Open defect: denied and screened facts become utility-silent because the teacher receives no explicit status.

Root cause: probe eligibility and polarity were conflated.

Fix:

```python
SpanCandidate(
    id=...,
    surface=...,
    type=...,
    role="lattice" | "placeholder" | "quasi",
    polarity="active" | "denied" | "screened",
    probe_eligible=...,
)
```

For dialogue screening, pass both turns:

```text
source occurrence:
doctor: Have you had any congestion?
patient: No.
polarity: denied
```

Prompt template for a negative:

```text
Which symptom did the patient deny when asked about upper-respiratory complaints?
```

Gold remains the exact symptom at rung 0; rung 1 must preserve the denied status.

Risk: current `sentence_of()` cannot reliably associate a question with the subsequent “No.” A dialogue-aware occurrence extractor is required before enabling negative probes.

Expected yield: likely **+1–3 clinically valuable probes per 10 documents**, but the larger gain is preventing RL from erasing pertinent negatives.

---

# Surface 3 — harness

## 3.1 Admission is coupled to probing and wastes the reader budget

Planned Issue 5 direction is correct: detected-sensitive admission, probe eligibility, and runtime probe sampling are separate decisions.

Root cause: the current flow largely treats “lattice matched” as “generate ladder probes,” even if the ceiling omits the fact or the target class is already overrepresented.

### Fix

```python
for span in all_detected_spans:
    span.sensitive = True                  # always hide at floor
    span.profile = resolve_profile(span)
    span.polarity = classify_polarity(span, doc)
    span.ceiling_alignment = align(span, out_hi)

    span.probe_eligible = (
        span.profile is not None
        and span.ceiling_alignment is not None
        and polarity_supported(span)
    )

candidate_probes = generate_for(span for span if span.probe_eligible)
validated = validate(candidate_probes, anchors)

selected = stratified_sample(
    validated,
    budget=13,
    strata=[
        "condition", "drug", "procedure",
        "active", "denied_or_screened",
        "exact", "semantic", "decision",
    ],
    minimums={
        "drug": 1,
        "procedure": 1,
        "denied_or_screened": 1,
        "decision": 1,
    },
)
```

The precise class minimums should be tuned from availability; do not force an invalid probe to fill a quota.

Expected result: raw kept count may fall, but trustworthy probes consumed per rollout should increase materially. Under a 13-probe wall, this is likely more valuable than generating additional condition questions.

## 3.2 Decision linking uses brittle substring matching

Current code concatenates `depends_on`, canonicalizes it, then tests:

```python
surface in dep_text
```

Measured:

- `afib` does not match `Atrial fibrillation`.
- `C T` does not match `CT`.
- `low hemoglobin` does not match anemia-like profile entries.
- About **10 unlinked decisions**, with one definite linker false negative and several detection/target-selection failures.

### Preferred fix

Use `depends_on_ids` from the teacher prompt and validate IDs against the supplied eligible inventory:

```python
def decision_span_ids(entry, eligible_by_id):
    ids = entry.get("depends_on_ids", [])
    if not ids:
        return Reject("missing_dependencies")
    if any(sid not in eligible_by_id for sid in ids):
        return Reject("unknown_dependency_id")
    return ids
```

### Compatibility fallback

```python
def link_dependency(dep, spans):
    dep_forms = normalized_forms(dep)

    for span in spans:
        span_forms = normalized_forms(
            span.surface,
            *profile_aliases(span),
            span.entry,
        )
        if dep_forms & span_forms:
            return span.id

    return None

def normalized_forms(*values):
    return {
        collapse_letter_spacing(
            singularize_medical_nouns(
                normalize_unicode(canon(v))
            )
        )
        for v in values
        if v
    }
```

Do not add unrestricted fuzzy matching across all spans. It can attach decisions to the wrong sibling and silently misassign reward.

Expected recovery: **one definite decision**, possibly one additional normalization case. The nine genuinely out-of-lattice decisions should be prevented at generation, not force-linked afterward.

## 3.3 Locator lint is opaque and may over-reject

The 13 locator rejects do not record the triggering surface. At least some questions appear clean when printed; possible triggers include patient names.

Fix:

```python
result = locator_lint(q, target, forbidden)
if not result.ok:
    reject_sink.append({
        ...,
        "gate": "locator",
        "locator_triggers": result.triggers,
    })
```

Separate trigger classes:

```python
trigger_class = (
    "clinical_sibling"
    if span.role == "lattice"
    else "protected_identifier"
)
```

Both remain invalid for reading from anonymized context, but separate reporting distinguishes prompt defects from normalization false positives.

Expected recovery: unknown until instrumented; likely **0–3**. Instrumentation is cheap and necessary before relaxing the gate.

## 3.4 Integral JSON numbers are rejected

One teacher output uses `"rung": 1.0`.

Fix:

```python
raw_rung = row.get("rung")

if isinstance(raw_rung, bool):
    reject("bad_rung")
elif isinstance(raw_rung, (int, float)) and float(raw_rung).is_integer():
    rung = int(raw_rung)
else:
    reject("bad_rung")

if rung not in (0, 1):
    reject("bad_rung")
```

Expected recovery: **one candidate**. Its question duplicated an exact-value shape, so it may still fail lint or validation; recovery is at most one.

## 3.5 Scorer hard-fails morphology and narrow paraphrases

Measured:

- `Migraines` versus `migraine` creates one false ceiling reject.
- “Fracture of the middle finger” versus `phalangeal fracture` likely creates one false floor miss and wrong keep.
- Binary containment correctly fixed the earlier shared-head-noun softness and must remain.

Fix:

```python
def semantic_forms(text):
    c = normalize_unicode(canon(text))
    forms = {c, medical_singularize(c)}
    forms |= profile_backed_aliases(c)
    return forms

def category_hit(answer, gold, aliases=()):
    answer_forms = semantic_forms(answer)
    gold_forms = set().union(
        semantic_forms(gold),
        *(semantic_forms(a) for a in aliases),
    )

    if answer_forms & gold_forms:
        return 1.0

    return float(any(
        content_tokens(g) <= content_tokens(a)
        for a in answer_forms
        for g in gold_forms
    ))
```

Use conservative suffix rules or a small inflector for plural normalization. Do not lemmatize into broad clinical equivalence, and do not restore token-F1.

Expected effect:

- Correctly reclassifies the migraine probe, although its floor may then reject it; trustworthy gain may be zero.
- Removes one wrong semantic keep.
- Primary value: correctness, not raw yield.

## 3.6 Rung-0 keeps can survive by reader noise

At least three agreed wrong-reason keeps:

- D2N001 CHF: floor contains CHF, reader selects hypertension.
- D2N001 lisinopril: floor contains lisinopril, reader returns dose.
- D2N004 CHF: floor contains CHF, reader returns plan.

The Codex audit additionally finds D2N008 kidney stones and lisinopril artificially kept due LaTeX placeholder residue.

Root cause: acceptance relies exclusively on a noisy reader even when the exact target or alias is visibly present in the floor context.

### Fix: deterministic pre-reader floor check for rung 0

```python
def exact_target_present(context, entry):
    forms = [
        entry["surface"],
        *entry.get("aliases", []),
    ]
    return any(boundary_phrase_match(form, context) for form in forms)

if rung == 0 and exact_target_present(out_lo_final, entry):
    verdict = "floor_exact_leak"
    lo_answer = "<deterministic exact/alias presence>"
    lo_score = 1.0
else:
    run_reader()
```

This is deliberately stricter than question-specific answerability: the all-placeholder floor should not preserve the exact protected value anywhere. If it does, the floor is not an exact-identity floor for that target.

Expected effect: removes **at least three and possibly five wrong keeps**. This lowers raw yield but substantially raises trustworthiness.

## 3.7 LaTeX placeholders bypass inversion

Observed forms:

```text
$\text{HEALTH\_CONDITION\_2}$
$\text{DRUG\_1}$
```

Current inversion handles bracketed `<TYPE_N>` and bare `TYPE_N`, single-pass, but not TeX escaping.

Measured consequence: at least **two suspect rung-0 keeps** in D2N008.

### Fix

Normalize only placeholder tokens from the actual `R`, preserving the current single-pass guarantee:

```python
def placeholder_forms(token):
    bare = token[1:-1]  # HEALTH_CONDITION_2
    escaped = bare.replace("_", r"\_")
    return {
        token,
        bare,
        rf"\text{{{escaped}}}",
        rf"$\text{{{escaped}}}$",
        rf"\(\text{{{escaped}}}\)",
    }

forms = {}
for token, entries in ph.items():
    surface = canonical_surface(entries)
    for form in placeholder_forms(token):
        forms[form] = surface

pattern = compile_longest_first(forms)
text, n = pattern.subn(lambda m: forms[m.group()], out_p)
```

Important: build one alternation and perform one substitution pass over the original output, as the current code does. Sequential substitutions can reprocess inserted sensitive surfaces.

Also extend residue detection so TeX placeholders count as `ph_residue` even before inversion.

Expected effect: invalidates or correctly floor-rejects **two wrong keeps**; prevents user-facing placeholder artifacts.

## 3.8 Detection threshold is knife-edge

Known example: a relevant span scores **0.339** against threshold **0.35**.

Root cause: a single hard threshold determines whether a fact receives a type/profile route, even though the difference is below any defensible calibration precision.

Fix: decouple hiding from probing with a three-way band:

```python
if score >= probe_threshold:
    sensitive = True
    probe_candidate = True
elif score >= hide_threshold:
    sensitive = True
    probe_candidate = False
    admission_reason = "low_confidence"
else:
    sensitive = False
```

Example starting policy:

```text
hide_threshold = 0.30
probe_threshold = 0.35
```

These values must be evaluated on labeled detection data; they are not a recommendation to normalize methods or alter privacy comparisons. This is detector admission calibration, not a privacy-comparison knob.

Expected recovery: dataset-dependent, likely small in this ten-document sample, but it closes high-risk floor leaks.

## 3.9 No symptom label

`DETECT_LABELS` includes condition, drug, and medical process, but not a first-class symptom/finding route. This contributes to missed congestion, low hemoglobin, pain, and similar clinically important targets or forces them into condition routing.

Fix options:

```python
DETECT_LABELS.update({
    "symptom": ("symptom", "lattice"),
    "clinical finding": ("clinical-finding", "lattice"),
})
```

Only make these lattice-bearing after adding corresponding profiles and polarity handling. Until then:

```python
"symptom": ("symptom", "placeholder")
```

is safer for floor privacy but produces no probe.

Expected yield: potentially meaningful once profiles exist, especially for denied symptoms; not an immediate cheap recovery.

Risk: adding labels changes GLiNER competition and may lower existing detections. Re-evaluate the full label set rather than appending labels blindly.

## 3.10 Medical-process versus health-condition routing mismatch

Example: transplant status can be detected as `medical process` while its profile entry lives under `health-condition`.

Root cause: runtime detector types and profile ontology types are treated as identical namespaces.

Fix: introduce explicit compatible routes:

```python
PROFILE_TYPE_ROUTES = {
    "medical-procedure": [
        "medical-procedure",
        "health-condition",       # only for status/history entries
    ],
    "health-condition": [
        "health-condition",
        "medical-procedure",      # only canonical status aliases
    ],
}
```

A better long-term model distinguishes:

```text
event: kidney transplant
status: history of kidney transplant / transplant recipient
```

The matcher should verify contextual compatibility rather than globally allowing cross-type matches.

Expected recovery: several previously abstained candidates across the corpus; exact ten-document yield is not established.

Risk: unrestricted cross-type matching creates wrong profiles. Require alias/profile evidence and context verification.

## 3.11 Stale profile embedding index

The stale `lattice_profiles.embindex.npz` forced exact-only or degraded retrieval, amplifying matcher abstention.

Fix:

1. Rebuild the index from the current profile source.
2. Record source hash, embedding model, dimension, and build timestamp.
3. Fail closed on metadata mismatch rather than silently using a stale index.

Pseudocode:

```python
expected = {
    "profiles_sha256": sha256(profile_json),
    "embedding_model": PROFILE_EMBED_MODEL,
    "dimension": expected_dim,
}

index = load_index()
if index.meta != expected:
    raise StaleProfileIndex(
        "rebuild lattice_profiles.embindex.npz before probe detection"
    )
```

Expected recovery: unknown without rerun, but necessary before interpreting detector/profile coverage. This can affect multiple spans per document.

## 3.12 Post-fix D2N002 confirms demotion is correct

The recomputed floor hides previously leaked Synthroid, Tylenol, kidney transplant, and other screened/abstained facts while allowing `invert()` to restore them in `OUT_LO_FINAL`.

Evidence: [D2N002 recompute](/home/timo/repos/agent-cloak/results/d2n002_floor_recompute.txt).

No reversal is warranted. The remaining work is polarity annotation and profile routing, not restoring deletion.

---

# Surface 4 — reader model and reader prompts

Reader: pinned `Qwen3.5-0.8B`, greedy/non-thinking.

## 4.1 Reader failures reject good ceiling probes

Measured: **9 reader failures**:

- Wrong neighbouring condition.
- Plan action instead of diagnosis.
- Medication list instead of condition.
- Dose instead of medication.
- Referral line instead of condition.
- Two MC mis-picks, including the opposite hydration option.
- Rung-1 category questions that the extractive contract does not support.

Root cause:

- The 0.8B model is weak on structured clinical notes and relation direction.
- `QA_PROMPT` asks for the shortest copied phrase but does not state the expected answer type.
- The same generic prompt serves conditions, drugs, procedures, and semantic categories.
- Rung-1 gold may not be copied from the note, contradicting the extractive instruction.
- The small reader’s mistakes affect both sides asymmetrically: it rejects good ceilings and lets floor leaks survive.

## 4.2 Replace the generic QA prompt with typed prompts

Recommended full template:

```text
Answer the question using only the note.

Expected answer type: {answer_type}
Target relation: {relation_hint}

Rules:
- Return only the entity or category requested by the question.
- If the question asks for a condition, return the condition name, not its
  treatment, dose, referral, status, or plan.
- If it asks for a medication, return the medication name or requested drug
  class, not its dose, indication, or action.
- If it asks for a procedure, return the procedure name, not its result.
- If it asks for a category, return the most specific category supported by
  the note; do not return a neighboring entity or plan action.
- Do not return a list unless the question explicitly requests a list.
- If the note does not uniquely support one answer of the expected type, reply
  exactly: NONE.

NOTE
{ctx}

QUESTION
{q}

ANSWER
```

### Exact slots

- `{ctx}`:
  - rung 0 validation/scoring: `OUT_HI_FINAL` or `OUT_LO_FINAL`;
  - rung 1: `OUT_HI_P`/`OUT_LO_P`, or `doc_p` under the pending branch;
  - never silently mix these contexts.
- `{q}`: generated probe question.
- `{answer_type}`: harness-derived enum, not teacher prose:
  - `condition name`
  - `medication name`
  - `procedure name`
  - `clinical category`
- `{relation_hint}`: short harness-derived instruction such as:
  - `identify the condition associated with the described management`
  - `identify the medication, not its dose`
  - `identify the condition category, not the referral`

The relation hint must not contain the gold or aliases.

Expected recovery:

- Likely fixes a useful subset of the nine ceiling failures, especially plan/dose/list errors.
- Also converts at least three wrong-reason keeps into correct floor rejects.
- Expected net: **+2–5 trustworthy keeps**, with higher precision even if raw count changes little.

## 4.3 Improve the MC prompt

Current MC prompt is directionally correct but too permissive.

Replacement:

```text
Choose the single option best supported by the note.

Use only patient-specific information stated in the note. Do not choose an
option merely because it is generally associated with a named disease, drug,
or procedure. Trace the question's described circumstances to the relevant
assessment or relationship, then compare every option.

If the note does not distinguish one option, reply exactly: NONE.
Otherwise reply with the exact option text and nothing else.

NOTE
{ctx}

QUESTION AND OPTIONS
{q}

ANSWER
```

`{q}` is the rendered question plus symmetrically shuffled options. `{ctx}` is `OUT_HI_P`/`OUT_LO_P`, or `doc_p` if that branch is chosen.

This should help the opposite-fluid and body-system mis-picks, but prompt changes alone will not give a 0.8B reader reliable multi-fact reasoning.

## 4.4 Stronger build-time adjudicator versus stronger rollout reader

Two separate roles should be explicit:

1. **Build validator:** decides whether a probe is trustworthy.
2. **Per-rollout scorer:** executes approximately 13 reads per document and is the cost wall.

Recommended staged design:

```python
result = deterministic_checks(probe, hi, lo)

if result.conclusive:
    return result

small = validate_with_qwen_0_8b(probe, hi, lo)

if small.is_clean_pass:
    # Optional sampling audit by stronger reader.
    return small

strong = adjudicate_with_stronger_reader(probe, hi, lo)
return consensus_rule(small, strong)
```

Acceptance policy:

- Reject if deterministic floor leak exists.
- Reject if target lacks ceiling alignment.
- For remaining candidates, require strong-reader ceiling success and strong-reader floor failure.
- Record whether the 0.8B reader also succeeds.
- Only schedule probes for rollout if the rollout reader can answer the ceiling reliably, unless the rollout reader is upgraded.

This prevents a stronger offline validator from admitting probes that the runtime scorer cannot use.

Alternatives considered:

- **Replace the 0.8B reader everywhere:** best semantic reliability, highest per-rollout cost; benchmark before choosing.
- **Use a stronger reader only at build time:** cheaper, but cannot recover runtime reward if the 0.8B reader still fails.
- **Typed 0.8B prompts plus deterministic checks:** lowest-cost first move and the recommended initial implementation.

Expected yield:

- Typed prompts and deterministic filters: **+2–5 trustworthy keeps net**.
- A stronger reader everywhere might recover **5–7 net**, but must pass the per-rollout performance gate.
- A stronger build-only reader mainly improves audit trust unless accepted probes are also 0.8B-readable.

---

# Ranked implementation order

Estimated gains overlap; they must not be summed mechanically.

## 1. Ceiling-aware target admission and ladder prompt

Implement:

- align targets to `OUT_HI`;
- pass `OUT_HI`, polarity, complete protected-span inventory, and evidence slots;
- require unique ceiling-visible evidence;
- skip ceiling-omitted facts before teacher generation.

Expected outcome: **+4–8 trustworthy ladder keeps**, while eliminating six known dead ceiling candidates and DOC-only grounding.

This is the largest likely yield recovery.

## 2. Typed reader prompts plus deterministic rung-0 floor-leak checks

Implement:

- typed condition/drug/procedure/category QA instructions;
- improved MC reasoning prompt;
- exact/alias presence check on `OUT_LO_FINAL`.

Expected outcome: recover **2–5** good ceiling probes and remove **3–5** wrong keeps. The primary result is a trustworthy selected set.

## 3. Eligible-target decision prompt with ID-based dependencies

Implement:

- supply only admitted lattice targets;
- return `depends_on_ids`;
- require patient-specific evidence and forbid plan/category readback;
- add option-set lint.

Expected outcome: one definite linker recovery, removal of roughly nine out-of-lattice generations, and a realistic decision yield of **1–3 trustworthy probes per ten documents**.

## 4. Detection/profile integrity repairs

Implement together:

- rebuild and hash-pin the embedding index;
- add verified cross-type routing for procedure/status facts;
- introduce hide/probe threshold bands;
- add symptom/finding detection initially as placeholder-only;
- attach active/denied/screened polarity.

Expected outcome: unknown direct count, but this is required before any new sweep can support coverage claims.

## 5. Small correctness fixes and class-stratified sampling

Implement:

- integral numeric rung coercion;
- TeX placeholder inversion;
- plural/profile-backed scorer normalization;
- locator trigger reporting;
- sample validated probes by target class and polarity under the ~13/doc budget.

Expected outcome: one possible candidate recovered, at least two wrong keeps corrected, and substantially better reward coverage per reader call.

---

# Interaction with the pending `doc_p` decision

Rung 0 remains on `out_final` in every branch.

## Branch A — rung 1 and decisions continue to read `out_p`

Data flow:

```text
teacher generation:
    doc_orig + out_hi + target metadata/polarity/inventory

anchor validation:
    rung 0: out_hi_final versus out_lo_final
    rung 1: out_hi_p versus out_lo_p
    decision: out_hi_p versus out_lo_p

rollout:
    rung 0: out_final
    rung 1: out_p
    decision: out_p
```

Required prompt wording:

- Call the teacher’s evidence `READER_CONTEXT = out_hi`.
- Tell it that the runtime context is a generated output, so every locator must survive summarization.
- Keep ceiling alignment against `OUT_HI`.

Advantages:

- Measures whether the remote model’s output preserves semantic utility.
- Matches the current pipeline’s end-to-end summarization task.

Risks:

- Ceiling omission remains fundamental; admission must filter it.
- Structured output columns can leak categories.
- The remote summarizer can introduce or remove relationships independently of substitution.

## Branch B — rung 1 and decisions move to `doc_p`

Data flow:

```text
teacher generation:
    doc_orig + target metadata/polarity/inventory
    optionally a ceiling reference for end-task relevance

anchor validation:
    rung 0: out_hi_final versus out_lo_final
    rung 1: doc_orig versus all-placeholder doc_lo
    decision: doc_orig versus all-placeholder doc_lo

rollout:
    rung 0: out_final
    rung 1: doc_p
    decision: doc_p
```

Prompt changes:

```diff
- The questions will be answered from READER_CONTEXT, which is the generated {output_kind}.
+ The questions will be answered directly from ANONYMIZED_DOCUMENT.

- Every locator must be supported by READER_CONTEXT.
+ Every locator must be supported by SOURCE_DOCUMENT and remain present after
+ replacing all OTHER PROTECTED SPANS with their typed placeholders.
```

Add a teacher-visible masked locator context:

```text
TARGET-MASKED DOCUMENT
{doc_with_target_original_but_all_other_sensitive_spans_placeholdered}
```

This is the appropriate grounding source for rung 1 and decisions under the `doc_p` branch. It directly tests whether the question’s locator survives anonymization without revealing sibling protected spans.

Advantages:

- Removes ceiling-output omission from semantic/decision probe validation.
- Makes the 13 locator failures easier to prevent because the teacher can see the exact grading context.
- Separates substitution utility from remote summarizer behavior.

Risks:

- It changes the measured object: semantic probes grade `doc_p`, not what the remote model actually carries into `out_final`.
- Questions can reward source-document detail the task output would never need.
- End-task relevance therefore still needs an `OUT_HI` admission or weighting signal, even if `doc_p` is the reader context.

## Fixes independent of the branch

Implement now without deciding the branch:

- polarity annotation;
- eligible-target IDs;
- profile/index repairs;
- TeX inversion;
- morphology normalization;
- deterministic rung-0 floor leak;
- typed reader prompts;
- locator trigger instrumentation;
- admission/probing separation;
- class-stratified runtime sampling.

The branch-dependent seam should be explicit:

```python
semantic_context = (
    anchors.out_p
    if semantic_read_surface == "out_p"
    else candidate.doc_p
)

decision_context = (
    anchors.out_p
    if decision_read_surface == "out_p"
    else candidate.doc_p
)
```

Do not let prompts or validation helpers infer this choice indirectly from variable names. Record it in artifact metadata because changing it changes the meaning of the reward and invalidates comparisons across builds.