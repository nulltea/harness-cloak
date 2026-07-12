---
type: dev-log
status: current
created: 2026-07-12
updated: 2026-07-12
tags: [qa-build, ladder-probes, decision-probes, fix-design, teacher-prompt, reader, pv4]
companion: [docs/issues/2026-07-11-ladder-decision-qa-question-design.md,
            docs/dev/logs/2026-07-12-qa-failure-analysis.md,
            docs/dev/logs/2026-07-12-qa-fix-design-by-surface-gpt56sol.md]
---

# QA-build fix design, classified by surface

Sources: `2026-07-12-qa-failure-analysis.md` (opus), `2026-07-12-qa-failure-analysis-codex.md`
(codex), the pv4 raw dump (`results/qa_pairs_pv4_super.txt`), the 2026-07-11 issue register
(Issues 1–5), and this session's live investigations (floor-leak demote fix `30e9564`, invert
bracket fix `34334c8`+`d799081`, D2N002 recompute). Parallel independent pass:
`2026-07-12-qa-fix-design-by-surface-gpt56sol.md`.

Where the two analysts' taxonomies disagree, entries were re-binned to the surface whose change
actually fixes them (codex's "bad QUESTION 31" splits across prompt-data, prompt-wording, and
span-selection surfaces; opus's "locator working-as-intended" is right about the *gate* but the
generation loss is still a prompt-surface problem).

Baseline: 22 kept / 100 (of which ~3–9 wrong-reason depending on strictness). Both analysts agree:
**teacher golds are never wrong (0/78)**; reader failures = 9; scorer = 1–2; the rest is question
design + gates.

---

## Surface 1 — Teacher model (`nvidia/nemotron-3-super-120b-a12b`)

**Issues attributable to the model itself: essentially none.**

| Issue | n | Evidence |
|---|---|---|
| Wrong golds | 0 | both analysts, independently |
| Malformed output | 1 | `"rung": 1.0` float (D2N001 echocardiogram) — a typing slip, gate-recoverable |
| Parse failures | 0 | pv3 fix stack (json_object + no max_tokens cap) holds |

**Fix: none at this surface. Keep the model.** The super-vs-ultra sweep already showed bigger ≠
better here (ultra leaked golds, produced no usable decision). Every observed "teacher failure"
traces to what the prompt asks for or what data it is (not) given — Surface 2. Do not spend an
escalation on the model until Surface 2 fixes are in and still failing.

---

## Surface 2 — Teacher prompts (wording + WHAT DATA IS PASSED)

This is the highest-loss surface: it owns the 13 locator gen-rejects, ~8 ceiling-omission/
mis-grounding rejects, ~5 ambiguity rejects, the decision tier's 7 plan-readbacks + 3
world-knowledge-anchor floors + 1 option leak, and ~9 out-of-lattice decision targets.
The unifying root cause: **the teacher is systematically under-informed.** It sees only
`doc_orig` (+ `out_hi` for decisions) and is asked to satisfy constraints it cannot check:
"answerable from the output" (it can't see what survives — ladder), "identify only via clinical
role" (it can't know which anchors are themselves sensitive), "determined by one detected fact"
(it can't see the detected-span inventory), "unanswerable after placeholdering" (it can't see
the floor). Each sub-issue below adds one missing piece of ground truth to the prompt.

### 2a. Ceiling-omission / mis-grounding (~8 ladder ceiling rejects)

Teacher grounds questions on doc facts or relations the note-writer drops (`nasal congestion`,
`echocardiogram` D2N008, the arthritis↔transplant-med-avoidance link).

**Branch on the pending doc_p decision:**
- **If rung-1/decision reads move to `doc_p`:** this issue disappears structurally — the ceiling
  anchor becomes `doc_orig`, which always contains the fact. No prompt change needed. (This is
  ~8 of 23 ceiling rejects recovered for free.)
- **If reads stay on `out_p`:** pass the ceiling output to the ladder teacher (the decision
  prompt already gets it) and gate grounding on it:

```
LADDER_PROMPT additions (out_p branch only):
  new slot:  {out_hi}   — inserted after the Document block:

  "A {output_kind} written from this document by the same model reads:
   {out_hi}

   Ground each question ONLY in facts and relationships that appear in BOTH the document and
   this {output_kind}. If the target fact does not appear in the {output_kind} at all, reply
   with {{"probes": []}} — do not invent a question for a fact the {output_kind} dropped."

  data change in ladder_probes_for_docs:
    todo.append({... "prompt": LADDER_PROMPT.format(..., out_hi=out_hi_of[d["id"]]) ...})
    # out_hi_of is already computed in both build paths; plumb it through like
    # decision_probes_for_docs already does.
```

### 2b. Ambiguity among co-listed siblings (~5 ladder ceiling rejects)

"Which condition was reviewed by Doctor Kumar?" when six conditions were reviewed; "what health
condition does James have" among several. The teacher doesn't know which sibling facts co-occur,
so it can't check uniqueness.

**Fix (data + wording): pass the sibling inventory and require unique selection.**

```
LADDER_PROMPT additions:
  new slot: {siblings} = the doc's OTHER detected lattice surfaces of the same runtime type
            (already available: the `other` list built for locator_lint — reuse it, filtered
            to same-type)

  new numbered requirement:
  "N. The document also mentions these other {type} facts: {siblings}. Your question must be
   answerable ONLY by the target fact — if any fact in this list also satisfies your question's
   description, rewrite the question until it does not. Never write questions of the form
   'which condition does the patient have / was reviewed / was discussed' when more than one
   qualifies."
```

### 2c. Locator bottleneck (13 gen-rejects, the semantic tier's biggest gen-stage loss)

The pv4 prompt permits grounding only via "what it is managed or treated with, its documented
status or course, or its clinical consequence" — but the *treatment* is usually itself a detected
sensitive span (lisinopril, digoxin, Imitrex), so the teacher walks straight into `locator_lint`.
The gate is correct; the prompt gives the teacher no legal anchor vocabulary and no list of what
is forbidden.

**Fix (data + wording): pass the forbidden-surface list explicitly and widen the legal anchor set
to non-sensitive context.**

```
LADDER_PROMPT changes:
  new slot: {hidden} = all detected surfaces for the doc (all_surfaces_of — already computed,
            currently used only by locator_lint AFTER generation; give it to the teacher BEFORE)

  replace requirement 2's anchor clause with:
  "2. identify which fact it asks about ONLY through NON-SENSITIVE context: the presenting
   complaint or symptom it explains, the body site or organ system involved, its documented
   time course (new / chronic / worsening / resolved), or the care setting. The following
   phrases are OTHER PROTECTED FACTS and will be hidden when the answer is graded — your
   question must not contain or paraphrase ANY of them: {hidden}. NEVER identify the fact by
   naming another medication, condition, procedure, or person; by its POSITION in a list; or
   by ENUMERATING its neighbours. [rest of requirement unchanged]"
```

Expected effect: most of the 13 locator rejects become legal candidates (they then still face
validation — pair with Surface 4 so the recovered candidates aren't lost to reader noise).

### 2d. Decision tier: plan-readbacks (7), world-knowledge anchors (3), option leak (1)

Answers sit verbatim in surviving plan/assessment lines ("physical therapy", the category
column), or an un-hidden anchor (Synthroid) gives them away, or the option set telegraphs
(3 opioids + 1 non-opioid, gold = the non-opioid). The pv4 no-naming rewrite fixed trivia but
the teacher still cannot check floor-answerability — because it never sees the floor.

**Fix (data + wording): show the teacher the floor output and make unanswerability-at-floor an
explicit, checkable requirement; constrain option construction.**

```
DECISION_PROMPT additions (the floor anchor out_lo_p is computed BEFORE decision generation in
both build paths — plumb it in like out_hi):

  new slot: {out_lo}  — inserted after the {out_hi} block:

  "The same {output_kind} with every protected fact replaced by a placeholder reads:
   {out_lo}

   N. Check each question against this anonymized version: if a careful reader could still pick
   the correct option from it — because the answer appears verbatim in a surviving plan or
   assessment line, because a surviving phrase (a drug name, a category label, an exercise
   type) implies it, or because the option set itself gives it away — the question is invalid.
   Rewrite or drop it.
   N+1. All options must be the same kind of thing and equally plausible for the anonymized
   version: never one option of a different class than the rest, and never an option that
   appears verbatim in the {output_kind}'s plan or assessment lines."
```

This converts 11 floor-rejects/doc-set worth of teacher effort into either valid probes or
early drops — and it is the cheapest form of "teacher self-validation" available, because the
floor text already exists at generation time.

### 2e. Out-of-lattice decision targets (~9 unlinked) + linking brittleness — solved together

The teacher writes decisions about salient non-span facts (labs, imaging) because it doesn't
know what the protected inventory is; separately, `_decision_span_ids` substring-matching drops
genuinely linkable decisions (`afib` vs "Atrial fibrillation", `C T` vs CT).

**Fix (data + contract): pass the detected span inventory WITH IDS and make the teacher cite the
id — eliminating both the targeting problem and the string-matching problem in one change.**

```
DECISION_PROMPT changes:
  new slot: {span_inventory} =
      "s1: hypothyroidism (condition)\n s2: arthritis (condition)\n s3: synthroid (drug)..."
      (built from spans_of[doc_id]; ids are the span ids validation already uses)

  requirement 1 gains: "...ONE specific clinical fact FROM THIS LIST:\n{span_inventory}\n"
  reply schema change: "depends_on_span_ids": ["s2"] replaces free-text depends_on
    (keep accepting depends_on as fallback for cache compatibility)

harness side (ladder_probes.py):
  def _decision_span_ids(entry):
      ids = entry.get("depends_on_span_ids")
      if ids: return [i for i in ids if i in known_span_ids]   # teacher-cited, exact
      # fallback: legacy substring path, now alias-aware:
      dep = canon(" ".join(entry.get("depends_on", [])))
      return [sid for sid, s in spans
              if canon(s.surface) in dep
              or any(canon(a) in dep for a in lookup_aliases(s.surface, s.type))]
```

Expected: the ~9 out-of-lattice decisions become in-inventory decisions (or honest early drops),
and the 1–2 alias false-negatives disappear.

### 2f. Polarity annotation (register Issue 5, open — scheduled with the redesign)

Pass `status: active | denied | screened` per span (the demoted spans' `_negated_or_screening`
result, kept as a field instead of a role change once probing resumes for them); require
status-faithful questions ("Which symptom did the patient deny on review of systems?"). Also
covers the negated exam finding (`thyroid enlargement`, "no thyroid enlargement") that burned
2 ceiling probes this sweep. Template belongs to the Issue-3/4 redesign session; the data
plumbing (status field on spans) can land any time.

---

## Surface 3 — Harness (build / gates / scoring / detection / inversion)

### 3a. Scorer morphology (1 rejected + 1 wrong-keep)

"Migraines" vs `migraine` scores a hard 0. Minimal, non-F1-reintroducing fix — singular/plural
normalization on content tokens inside the binary matcher:

```python
# ladder_probes.py
_PLURAL = re.compile(r"(?<=\w\w)s$")          # crude English plural; 'stones'->'stone', keeps 'is'
def _norm_tokens(text):                        # replaces _tokens in _category_hit ONLY
    return {_PLURAL.sub("", w) for w in _tokens(text)}

def _category_hit(answer, gold):
    if fact_score(answer, gold) == 1.0: return 1.0
    gt = _norm_tokens(gold)
    return 1.0 if gt and gt <= _norm_tokens(answer) else 0.0
```

Paraphrase gaps ("fracture of the middle finger" vs `phalangeal fracture`) are NOT fixable by
string rules without reintroducing sibling-blur; route them through profile aliases (the
acceptance set already folds aliases — add the paraphrase to the profile entry, a data fix).

### 3b. Float-rung coercion (1 gen-reject)

```python
rung = row.get("rung")
if isinstance(rung, float) and rung.is_integer(): rung = int(rung)
```

### 3c. Locator-lint diagnosability (codex finding: rejects not attributable)

```python
# locator_lint returns the triggering surface instead of bool; reject sink records it
def locator_lint(q, span_surface, other_surfaces):
    qt = _tokens(q or "")
    for surface in other_surfaces or []:
        ...
        if st and st <= qt: return surface     # falsy "" == pass, else the trigger
    return ""
# caller: trig = locator_lint(...); if trig: _rej(t, rung, q, "locator", gold, trigger=trig)
```

Two of the 13 locator rejects are unexplained by the visible question (codex hypothesis:
patient-name overmatch); this field settles it next sweep.

### 3d. Detection (threshold / labels / routing) — floor-integrity + coverage

- **Threshold 0.35 → 0.30**: ultram scored 0.339 (the only true recall miss in D2N002); the
  miner already runs 0.30 on the base model. Cheap sweep to verify junk rate stays acceptable.
- **Type-routing fallback before abstain**: `kidney transplant` detected as `medical process`
  but profiled under `health-condition` → abstain → (pre-fix) floor leak, (post-fix) demoted
  span that could have carried a lattice.

```python
m = matches.get(span_key(c["surface"], c["type"]))
if m is None and c["type"] in LATTICE_TYPE_SIBLINGS:      # {"medical-procedure": ["health-condition"], ...}
    for alt in LATTICE_TYPE_SIBLINGS[c["type"]]:
        m = matches_alt.get(span_key(c["surface"], alt))   # second match pass under sibling type
        if m: c = {**c, "type": alt}; break
```

- **Symptom label**: `joint pain` is undetectable by construction (no symptom-ish label in
  DETECT_LABELS). Adding `"symptom" -> ("symptom", "placeholder")` hides the symptom narrative
  at the floor — but note this cuts BOTH ways: symptoms are the main *legal anchor vocabulary*
  for 2c's grounding fix. Decision needed at redesign time: hide symptoms (tighter floor) or
  keep them visible (richer grounding). Do not do both.
- **Rebuild `lattice_profiles.embindex.npz`** (operational, blocks on the parallel session's
  profile work being committed): exact-only matching amplified abstains all sweep.

### 3e. Inversion: LaTeX-escaped placeholder form (codex kept-audit finding — NOT yet fixed)

D2N008's floor output carries `$\text{HEALTH\_CONDITION\_2}$` — the remote wrapped placeholders
in LaTeX. The single-pass fix (`d799081`) handles `<X>` and bare `X`, not this. Two rung-0
probes were kept against an artificially blind floor because of it.

```python
# extract.py — normalize known escape wrappers BEFORE the single-pass restoration:
_TEX = re.compile(r"\$\\text\{([^}]*)\}\$")
def _unwrap_escapes(text):
    return _TEX.sub(lambda m: m.group(1).replace(r"\_", "_"), text)
# _rule_prepass: text = _unwrap_escapes(text) as step 0; the existing alternation then matches.
# (markdown \<X\> deferred by d799081 — fold into the same normalizer.)
```

### 3f. Already fixed this session (for completeness, no action)

Decisions read `out_p` (validation + reward); MC reader prompt; symmetric shuffle; span-link
gate; `lint_decision` + pv4 no-naming; single rung-1 question; binary containment scoring;
screened/abstained span demotion (floor leak); invert bracket-stripped echoes (single-pass).

---

## Surface 4 — Reader (pinned Qwen3.5-0.8B + read prompts)

Owns: 9 ladder ceiling failures, 2 decision ceiling mis-picks, and 3–4 wrong-reason rung-0
keeps (floor mis-picks). Failure shape is consistent: **the extractive reader returns plan
actions, doses, or med lists when asked for a condition/drug name, and picks wrong neighbours
in lists.**

### 4a. Answer-type directive in QA_PROMPT (free, do first)

```
QA_PROMPT = ("Answer the question using ONLY the note below. Reply with the shortest exact "
             "answer copied from the note (a name, value, number, or phrase). "
             "If the question asks WHICH CONDITION, MEDICATION, or PROCEDURE, reply with that "
             "item's NAME alone — never a plan action, a dose, a list, or what was done about "
             "it. If the note does not contain the answer, reply exactly: NONE.\n\n"
             "Note:\n{ctx}\n\nQuestion: {q}\nAnswer:")
```

Directly targets 6–7 of the 9 ceiling failures (plan-action/dose/list returns). Costs nothing;
reader pin unchanged (prompt is part of the build, bump validation only).

### 4b. Asymmetric two-reader validation (the structural fix for anti-selection)

The insight both analysts converged on: ceiling errors *lose good probes*, floor errors *keep
bad ones* — and the two checks have different requirements. The ceiling check answers "can the
REWARD reader earn this probe?" (must use the pinned 0.8B, or kept probes are dead weight at
training time); the floor check answers "does the anonymized context leak?" (wants the most
competent reader available — a floor mis-pick mints a wrong-reason keep).

```python
# validate_ladder / validate_decisions: split the reader pair
hi_answer = reader_pinned(q)        # ceiling: the reward reader — earnability check
lo_answer = reader_strong(q)        # floor: strong reader (e.g. gemma E4B, already served) — leak check
# keep iff pinned-ceiling >= th AND strong-floor < th
```

- Build-time cost only (floor reads double in quality, not count); reward path untouched.
- Immediately flips the 3–4 wrong-reason echo-keeps to floor (correct), and stops the kept-count
  inflation both analysts flagged.
- The 9 pinned-reader ceiling failures are NOT rescued by this (deliberately — if the pinned
  reader can't answer at ceiling, the probe can't pay reward). Rescuing them needs 4a's prompt
  fix, and where that fails, the probe should stay dropped or the reward reader upgraded —
  which is the register's option (b), a cost decision (per-rollout), not a build decision.

### 4c. Reward-reader upgrade — hold

Escalating the pinned reader (0.8B → 1.7B/4B) is the fallback if 4a+4b leave semantic-tier
yield too thin. It re-pins the reward (re-gate) and raises every-rollout cost. Decide AFTER a
sweep with Surfaces 2–3 + 4a/4b in place; current evidence says most "reader" losses are
actually prompt-recoverable.

---

## Ranked implementation order

Grouped; within a group, order is by (yield ÷ effort). "Branch" = depends on the pending
doc_p-vs-out_p read-target decision.

| # | Fix | Surface | Est. effect | Cost |
|---|---|---|---|---|
| 1 | 4a answer-type directive | reader-prompt | rescues ~6 ceiling losses | trivial |
| 2 | 4b asymmetric two-reader validation | harness | kills wrong-reason keeps (~3–9); trustworthy kept set | small |
| 3 | 3a plural normalization + 3b float rung + 3c locator trigger | harness | +1–2, diagnosable gates | trivial |
| 4 | 3e LaTeX-escape inversion normalizer | harness | closes the last known echo bypass | small |
| 5 | 2e span-inventory + id-citation for decisions | teacher-data + harness | converts ~9 unlinked into real candidates; kills linking bugs | small |
| 6 | 2d floor-preview self-check + option constraints | teacher-data | converts ~11 floor drops into valid probes or early drops | small |
| 7 | 2c hidden-list + non-sensitive anchor vocabulary | teacher-data | recovers most of 13 locator rejects as candidates | small |
| 8 | 2b sibling-list uniqueness requirement | teacher-data | rescues ~5 ambiguity losses | small |
| 9 | 3d detection: threshold 0.30, type-routing fallback, embindex rebuild | harness | tighter floor, fewer abstains | medium |
| 10 | 2a out_hi to ladder teacher — **only on the out_p branch** | teacher-data | ~8 ceiling-omission losses (free on the doc_p branch) | small |
| 11 | 2f polarity annotation + symptom-label decision + decision reasoning-shapes | teacher-prompt redesign session | pertinent negatives; decision tier becomes reasoning tier | session |
| 12 | 4c reward-reader upgrade | reader | last resort, re-gates the reward | large |

Interaction summary with the pending decision: choosing **doc_p** makes #10 unnecessary and
shrinks #2a's loss class to zero; choosing **out_p** makes #10 mandatory and keeps the
note-writer's stochastic omissions as a permanent ~10% candidate tax. Everything else is
branch-independent.
