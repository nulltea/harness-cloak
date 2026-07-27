---
type: handoff
status: current
created: 2026-07-20
updated: 2026-07-27
tags: [qa-builder-v2, relations, ambiguity, reverse-framing, lattice-profiles, detector]
companion: docs/issues/qa-builder-dept.md
---

# QA-builder-v2 relations: open tasks, ambiguity work, set-valued proposal, detection gap

Context: after the 67-doc ACI QA-builder-v2 build (`results/qa_v2_aci_full/`), we audited the relation
drop channels. The dominant genuine loss is **answer ambiguity** (a `(relation, subject)` with ≥2
same-type answers is unanswerable as a unique-answer QA). This handoff records (1) the tasks approved
*before* the ambiguity work, (2) where the ambiguity/reverse-framing work landed + its uncommitted
code, (3) Codex's set-valued proposal for the genuinely multi-answer (multi-direction) case, and
(4) the lattice surface-match detection gap that blocks a chunk of the residual losses.

Report generator for any build: `python scripts/qa_build_report.py results/<dir>` (durable; writes
`<stem>.report.md` + `<stem>.gate_failure_probe.json` + `<stem>.lattice_quality.json`; zero-cost).

---

## 1. Previously-approved open tasks (pre-ambiguity)

All scoped, none yet implemented. Each is recall-safe / diagnostic; validate on a cached re-run.

- **Plural fold in `_source_literal_spans`** (`src/cloak/qa/builder.py` ~2663). The `(?!\w)`
  boundary blocks a singular context-literal (`x-ray`) from matching a plural source (`x-rays`), and
  `level`↔`levels`. Add an optional trailing plural on the final token, matched-extent-inclusive.
  Recovers ~3 net-new `tests_for` relations (`unknown_context_literal` Cause 2: D2N040/044/045 x-ray,
  D2N039 aldosterone level). Low risk; matcher is also used at evidence-range (~2950) and escalation
  (~3987), so re-run the escalation prefilter regression + one unit test.

- **Record leaking token + answer-level in `answer_leakage` audit evidence.** ~6/24 `answer_leakage`
  cases fire with no lexical overlap against the final answer (they leak a finer level or a generic
  word like "medical" from a "medical condition" coarse level). Evidence doesn't record which
  token/level triggered `_question_leaks_answer` (~1927) at the reject site (~4873). Diagnostic-only.

- **Stop gleaning re-authoring already-kept relations** (`_gleaning_targets` ambiguous branch,
  `qa_builder.py:4276`). It iterates `[*kept_relations, *rejections]`, re-handing KEPT relations to the
  paid gleaning teacher. Measured: 74/458 gleaning proposals re-author a decision-pair primary already
  kept; 52 come back kept and are all merged away as `primary_preferred` (this *is* the approved→kept
  gap), 22 come back rejected (wasted paid calls + inflated channels). Fix: iterate `rejections` only.
  Recall-safe (kept total unchanged), fewer paid calls. Add a unit test.

- **Gate `lattice_level_suspect` probe against false positives** (`_diagnose_coarser_readable`
  ~7345). Verified false-positive-prone: 8/10 flagged levels are used by a KEPT relation in the same
  doc (provably readable), and the real cause of those gate failures is answer ambiguity (178/312
  `three_point_gate_failed` have ≥2 competing same-type answers). Suppress the flag when the level is
  kept in-doc OR the subject has ≥2 competing same-type answers. Diagnostic-only (review flag, not a
  verdict). NOTE: the "bad lattice profiles" list I first produced from this probe was retracted —
  the probe does not isolate bad data.

Genuine lattice **data** defects (separate, from `scripts/qa_build_report.py`'s data-quality audit):
`essential hypertension family` / `subacromial tendon disorder family` (grouping-node artifacts),
`administration, physiological systems and anatomical regions, introduction` (injection/shot; opaque
SNOMED axis label, used 7×), `heart and great vessels, {bypass,replacement,repair}`, `lower joints,
{revision,replacement}`, `skin and breast, excision`, garbled surface keys (`echocardiogram
echocardiocardiogram`, `ultrasound elasto elastography`). Fix in `lattice_profiles.json`. (The
`essential hypertension family` case for `severe/uncontrolled hypertension` was already merged into
the canonical `hypertension` entry this session.)

---

## 2. Ambiguity / reverse-framing work — current state (UNCOMMITTED)

**Problem, quantified** (baseline `results/qa_v2_aci_full`, condition-subject relations
`prescribed_with`/`tests_for`/`procedure_for`): 145 genuinely-lost `three_point_gate_failed` ambiguity
instances / 61 ambiguous `(relation, subject)` groups / 36 docs. "Genuinely lost" = the
`(relation, subject_decision, object)` fact is kept nowhere in the doc.

**What was tried (all measured on 36 docs, free — cached v21 teacher + local reader):**

| approach | mechanism | result |
|---|---|---|
| Prompt-flip (teacher rewords) | repair-prompt instruction to flip orientation | fragile: net +2, **cannibalized** forward relations (teacher re-rolls whole doc); reverted |
| **Source 1** | doc-global deterministic reverse-framing (`answer_role=subject`), isolated pass | **+10 net, 0 regression** |
| **Source 1+2** | + judge-accepted opportunities (`recovered_by_escalation`) seed `{objects}` | **+12 net, 0 regression** |
| Source 2-full | flip against every detected object span in the block | 525 candidates, firehose, poor precision — **rejected** |
| Source 3 (reader-hint) | reader names the subject for a failed flip | **~0 net-new** — controlled hints already kept; new hints are uncontrolled (see §4). **Not built.** |

**The kept mechanism (Sources 1+2):** deterministic reverse-orientation QA. For an ambiguous
`(relation, subject)` group, flip every object (linked span or context literal) → `answer_role=subject`
(roles preserved, so the arg-type contract holds — no `invalid_argument_types`), with a per-relation
question template ("For what medical condition was the {object} prescribed/ordered/performed?").
Additive isolated doc-global pass; skips facts already kept forward. `_reverse_framed_proposals` +
`_REVERSE_FRAME_TEMPLATES` (`qa_builder.py` ~4553), `reverse_framing_only` compile mode, and the pass
in the orchestration (before `coverage_targets`, ~7224). Extrapolates to ~+20–25 kept / 0 loss on 67
docs, free (cached teacher). 238 tests pass.

**Uncommitted working-tree changes (this session):**
- `src/cloak/qa/builder.py`: reverse-framing (Sources 1+2), `reverse_framing_only` compile mode,
  doc-global pass, leakage-locator exemption for reverse questions (`answer_role==subject`), env-gated
  `_gate_debug` (`CLOAK_GATE_DEBUG_DIR`), `import inspect`.
- `src/cloak/llm.py`: env-gated reasoning-trace sidecar (`CLOAK_LLM_REASONING_DIR`); `OpenRouterRelationTeacher.include_reasoning` param (`CLOAK_TEACHER_REASONING=include`, scoped to the
  secondary/gleaning teacher so the primary cache still hits). Default OFF — prod unchanged.
- `scripts/qa_build_report.py` (new, durable report generator).
- `scripts/spikes/{judge_accepted,detected_unproposed}_flip_candidates.py` (throwaway analyses).
- Tests updated: `test_relation_qa_v2.py`, `test_qa_builder_v2.py` (238 pass).
- Pilot artifacts under `results/qa_v2_flip_pilot{1..6}/` are throwaway — safe to delete.

**Decision pending:** commit Sources 1+2 (bump `builder_pin`), run the full 67-doc free confirmation,
and pin down whether the runtime under-generation (34 candidates vs 180 theoretical) is only
skip-already-kept + grounding failures or also `span_label` resolution in the attempt records.

---

## 3. Codex's set-valued proposal — for genuinely multi-answer (multi-direction) relations

Reverse-framing (Sources 1+2) only fixes ambiguity where one *direction* is unique. For **genuinely
many-to-many** cases (both directions ambiguous — e.g. a test serving several conditions and a
condition served by several tests), Codex (session `019f7f8a-a337-7f00-a7f6-c330447f053d`, high effort,
read-only) proposed a **set-valued QA** mechanism. Verbatim-faithful summary:

**Mechanism.** One set-valued assertion per multi-answer query (additive schema alongside `semantic_qa`,
whose schema is currently closed):
- `subtype: contextual_relation`, `scoring_contract: {kind: semantic_qa_all, match: fact_score,
  response: json_string_array, aggregation: one_to_one_mean}`.
- `question`: an exhaustive wrapper over the already-sanitized singular question ("Return every distinct
  answer supported by the document for: <question>"). Must NOT state K or enumerate golds; all existing
  linked-surface substitution + answer/protected-term leakage checks rerun against every member.
- Two gold representations: **build-time** `answer_set.members` — each member keeps its current
  `accepted_values` + `answer_target` byte-for-byte (linked targets stay decision-scoped
  `(decision_id, required_property)`); **RL `out_final`** `gold_answers: list[list[str]]` — one inner
  equivalence class per distinct fact, copied from the constituent single probes.

**Scorer.** Parse the reader response as a one-line JSON array of spans; dedup by `canon`; compute the
**maximum-weight one-to-one assignment** between predicted items and gold members, edge = the existing
member scorer (`_linked_answer_score` for linked; `max(fact_score(...))` for literal). One prediction
satisfies at most one member — so `"loop diuretic and ACE inhibitor"` as one item cannot get 2/2, and
duplicated output cannot manufacture coverage.

**Three-point gate (K≥2):** pass iff `∀i m_i_original ≥ t AND ∀i m_i_representative ≥ t AND
∀i m_i_placeholder < t` (per-member conjunction, stronger than the scalar single-answer gate).

**Runtime score** = `max one-to-one matching weight / K` — dropping 1 of 3 facts caps at 2/3; full
credit needs all members; extra predictions don't reduce (recall channel).

**Minimal `reward.py` extension** (additive): `MULTI_QA_PROMPT` (one-line JSON array or `[]`),
`_read_all_batch` (same pinned `QA_MODEL`, deterministic, separate response contract + capacity),
`_one_to_one_mean(preds, gold_groups, edge_score)`, and a `scoring_contract.kind` dispatch inside
`fact_f1s` (single probes unchanged). `u_gold` needs no change — export the same K constituent fact
rows (`_score_facts` already averages per-fact log-probs). Malformed JSON must raise a scorer error,
NOT become utility zero.

**Detection / regression.** Group compiled (not `_answer_competing_surfaces`-diagnostic) candidates by
`relation / answer_role / answer_type / normalized question / non-answer-locator identity`; dedup
answers by linked `decision_id` or exact literal; a bucket goes multi only with ≥2 distinct compiled
answer identities. Insert between `compile_relations` and `validate_candidate_rows`. Union occurrence
links + decision requirements + reader evidence; keep constituent evidence under `evidence.members`.
`K==1` returns the original candidate byte-identical (existing tests pin it). `K≥2` tries the multi
gate first; if it fails, run the constituents through today's single gate (never destroy a usable
member). Scorer/reader changes still require a builder/reward pin bump.

**Ranked alternatives.** (1) exhaustive set QA + one-to-one all-of matching — **recommended**;
(2) fallback: separate questions with a source-grounded distinguisher — only when a genuine non-answer
locator exists; (3) **rejected**: one question with several accepted values scored max/"accept any" —
`_answer_score` is already `max(fact_score)`, so preserving one member gives full credit and the
reconstructor may drop the rest.

**Codex grounding corrections to note:** the three-point gate is `validate_context_assertions`
(`qa_builder.py:7667`), not `apply_repeated_reader_checks`; the QA-builder artifact scores context
assertions on `doc_p`, whereas `reward.fact_f1s` scores `out_final`. **Honest caveat (Codex's):**
whether MedGemma-4b reliably enumerates all members as JSON is unverified — needs a fresh gate run.

---

## 4. Detection gap: lattice surface-match blocks controllable conditions

Investigating why reverse-framing couldn't recover some ambiguity relations exposed a real upstream
bug. Conditions with lattice profiles are being left **uncontrolled** (`controlled=False`, no
decision) because the **detected surface variant does not match the profile key/alias** — the lattice
lookup (`lookup_entry`, `src/cloak/lattice/profiles.py`) is **exact** (canonical key + aliases +
plural-fold), not semantic. Confirmed:

| detected surface | `lookup_entry(health-condition)` |
|---|---|
| `prostate cancer` | ✓ → levels present |
| `prostate cancer issues` (D2N047) | ✗ no match → `controlled=False` |
| `back pain` (D2N009) | ✗ (profiles are `chronic back pain` / `acute lumbar`) |
| `enlarged prostate` (D2N047) | ✗ (no BPH profile) |

An uncontrolled span **cannot be a linked relation argument** (no decision → no lattice/anchor/answer
target), so e.g. `tests_for(prostate cancer → PSA)` — true in the source, PSA proposed, prostate cancer
detected — can be captured by **no** relation source (teacher, judge/miner, or any flip): the correct
subject is invisible to relations. This is the real blocker behind a slice of the residual ambiguity
losses (misattributed earlier to "detector gaps" / "reader-hint opportunities").

**Fixes (upstream of relations, pick one):**
1. **Alias coverage** — add the observed surface variants as profile aliases (`prostate cancer issues`
   → `prostate cancer`; `back pain` → `chronic back pain` or a bare `back pain` profile; add an
   `enlarged prostate`/BPH profile). Cheapest; per-surface.
2. **Wire the embedding index into decision-assignment** — the embindex (`profile_match.build_embindex`,
   built as build step 1) would semantically match `prostate cancer issues` → `prostate cancer`. If the
   env-build controlling path uses exact `lookup_levels` instead of the embindex, closing that gap fixes
   the whole class. **First step: confirm which path the env build uses** to choose fix 1 vs 2.

Impact: fixing this makes these conditions controlled → the normal pipeline captures their relations,
and Sources 1+2 extend to them for free. It also means the residual ambiguity is *not* a relation-
mechanism limit but a detector/lattice coverage limit.

---

## Suggested next steps (priority order)
1. Decide fix path for §4 (check env-build controlling path: exact-lookup vs embindex) — highest
   leverage, unblocks a class of relations.
2. Commit Sources 1+2 (bump `builder_pin`), full 67-doc free confirmation, update the report.
3. Implement the pre-ambiguity tasks (§1) — start with `_gleaning_targets` kept-exclusion (recall-safe,
   cuts paid gleaning cost) and the plural fold.
4. Build Codex's set-valued mechanism (§3) only if the genuinely many-to-many residual (after §1–2 and
   the §4 detector fix) justifies the reward-system change.
