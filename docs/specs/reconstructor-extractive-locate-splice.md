---
type: reference
status: current
created: 2026-07-09
updated: 2026-07-09
# 2026-07-09 second pass: doc-based claims re-verified against CURRENT code + a fresh
# fine-arms probe (scripts/spikes/reconstructor_issue_probes.py,
# scripts/spikes/reconstructor_residue_fresh_probe.py). See "Measured findings".
tags: [reconstructor, extraction, survived-recovery, locate-splice, do-no-harm, design-review]
companion: [docs/specs/extractor-frozen-rl-reward.md, docs/plans/2026-07-06-reconstructor-design3-plan.md, docs/plans/2026-07-05-survived-recovery-extractor.md, docs/specs/lattice-substitutor.md, research-wiki/training/2026-07-06-FT-reconstructor-v1-residue-edit.md]
# 2026-07-09: superseded in part — the extractor is now required to be a FROZEN,
# substitutor-independent RL reward component; the controlling design moved to
# docs/specs/extractor-frozen-rl-reward.md. This doc remains the failure-mode record for the
# generative editor and the locate/verify/splice mechanics it shares with the successor.
---

# Reconstructor re-evaluation: replace generative residue editing with extractive locate-then-splice

Re-evaluation of the residue-targeted generative edit reconstructor
(`docs/plans/2026-07-06-reconstructor-design3-plan.md`, implemented through commit `c2ad611`,
training **never run** — correctly blocked by its own pre-training audit gate). Verdict: the
generative seq2seq is the wrong tool for this task; the failure the audit surfaced is structural,
not a patchable gate bug. This spec documents the failure modes and proposes the replacement
design: **extractive locate-then-splice**, whose quality ceiling is measurable today with zero
training.

## Definitions

- **Cascade** — the deterministic inverter `cloak.extract.invert` / `_rule_prepass`: exact, fuzzy,
  and semantic fill matching. Recovers ~82% of survived spans (239/293 on the 151-doc measurement).
- **Residue** — the generalization entries in `R` whose fill the cascade could not locate in
  `out_p`: paraphrase-reworded fills and lossy-generalization mentions.
- **Survived span** — a generalization whose substituted content (the fill or a rewording of it)
  reached `out_p`. The reconstructor's denominator; 293/1059 on the measured corpus.
- **Splice** — replacing a located mention slice of `out_p` with the original surface from `R`.
- **Quote** — the survival judge's verbatim `out_p` snippet grounding a residue mention.
- **Do-no-harm** — hard constraint: only original surfaces from `R` may enter `out_final`; a false
  substitution is worse than a miss; cascade-resolved spans must never be altered.
- **Judge** — the locally served Qwen3.6-35B-A3B survival judge (llama-swap, `localhost:8060`).
  Local: sending it `out_p` + `R` has **zero remote-privacy cost**.

## Current state (evidence)

Implemented: helpers + `edit_guard` + data builder with admission gate + `reconstruct()` +
`classify_recovery` (commits `d0ca878`…`c2ad611`); pre-training audit spike `a31a55a`.

Data build (`results/build_reconstructor_data_fine.log`, `data/reconstructor_*.jsonl`):

| corpus | docs w/ residue | positive docs | admitted edits | no-op targets | logged degeneracies |
|---|--:|--:|--:|--:|--:|
| clinical | 56 | 22 | 32 | 34 | 31 |
| lexsum | 80 | 10 | 11 | 70 | 69 |

**43 admitted edits total.** The hand audit (`results/reconstructor_admit_audit*.txt`) found
false restorations among them — the admitted-target precision gate (< ~0.95 → do not train)
fired, and the training record (`FT-reconstructor v1`) is still `status: planned`.

Audit failure pattern — **quote-granularity superset splices**. The judge's quote is frequently a
superset of the actual mention, and splicing the original over the whole quote deletes flanking
content or rewrites non-target text:

- `'Type 2 Diabetes Mellitus'` → spliced to `'diabetes'` (deletes the modifier chain);
- `'Hypertension and Diabetes'` → `'diabetes'` (deletes a co-mention);
- `'refugee resettlement agencies'` → `'refugees'` (fill `a personal attribute`, type DEM — the
  quote is an ORG-like phrase; splice corrupts it);
- `'Aortic Valve Replacement and Ascending Aortic Aneurysm Repair'` → `'aortic valve replacement'`
  (halves a compound procedure);
- `'(Reflux/Nausea/Vomiting/Abdominal Pain):**'` → `'(reflux):**'` (destroys list content);
- one `type=UNKNOWN, fill=''` entry admitted (gate hole: empty fill should never reach admission).

Note where these live: DEM, MISC, UNKNOWN — types **outside** `_AMBIGUOUS_TYPES = {DATETIME,
QUANTITY, LOC, ORG}`, so they skip the mandatory NLI correspondence check entirely.

## Measured findings (2026-07-09 — current code + fresh fine-type arms)

The Jul-7 artifacts above are **coarse-typed** (restore maps: DEM 96, MISC 68, zero fine types;
built on the Jul-5 pilot arms) — they describe a pipeline that no longer runs. Every claim was
therefore re-tested: code-level probes against current `reconstruct.py`
(`scripts/spikes/reconstructor_issue_probes.py`, CPU-only) and a fresh 40-docs/corpus probe on
the fine arms (`scripts/spikes/reconstructor_residue_fresh_probe.py` →
`results/recon_fresh_probe.json`, `results/recon_fresh_admit_audit.txt`).

**Code-level probe results (current code):**

| claim | result |
|---|---|
| `edit_guard` accepts over-deletion around a correct splice | **CONFIRMED** (+3-word deletion passes) |
| empty-fill / `UNKNOWN` entries admitted by the gate | **CONFIRMED** |
| all-or-nothing: 1 unrelated word of drift discards a correct edit | **CONFIRMED** |
| flan-t5 1024-token truncation | **CONFIRMED** (26/56 clinical inputs, max 1384) |
| fine types bypass the correspondence gate | **REFUTED** — the real defect is the *reverse*: `_type_sane`'s fine word-lists hold family words only ("disease", "endocrine"), so genuine specific mentions ("Type 2 Diabetes Mellitus") are **rejected** — fine-type admission recall ≈ 0 — while MISC has no branch (always passes) and DEM's word list passes junk |

**Fresh-probe results (fine arms, 40 docs/corpus, judge-grounded):**

1. **The design-premise is dead.** Residue is now 102/195 clinical (52%) and 263/339 lexsum
   (78%) of generalizations — the cascade's fill-matching collapsed under fine lattice fills
   (long generic phrases like "a city in wisconsin" come back reworded as "Wisconsin city's").
   "Cascade gets ~82%, recover the last 18%" no longer describes the system.
2. **But the survived-and-locatable population is tiny:** only 40/365 residue entries have a
   judge-grounded SURVIVED/REWORDED quote. Most fine fills never reach `out_p` in locatable
   form — a utility question for the *substitutor*, not a reconstructor gap.
3. **Supervision scarcity is worse than the stale number:** admitted edits per 80 docs — current
   gate 8, correspondence-for-all 4, fixed gate (correspondence-for-all, word-list legs dropped) 7.
   A generative editor cannot be trained on 7 examples; that claim is now measured, not inferred.
4. **Quote-boundary noise is real but secondary:** of 40 grounded quotes, 16 overshoot 0 words,
   9 overshoot 1, 8 overshoot ≥2, 7 align to nothing. The correspondence gate filters most of
   the wild ones before splicing.
5. **Zero-training splice ceiling:** on the fixed-gate admitted set, 6/7 recovered (1 deletion)
   under both splice-at-quote and boundary-tight splice — after a proper correspondence gate the
   two modes barely differ; **the gate, not the splice mode, carries the safety.** Known caveat:
   preposition loss inside quotes ("jefferson parish in Louisiana" → "jefferson parish
   New Orleans").
6. **The dominant defects are upstream, reproduced deterministically:**
   - `bucket_quantity` garbles units and drops word-scales: `'$8.5 million'` →
     `'between 4.25 and 17 $illion'` (the `[kKmM]?` in the unit regex eats the leading "m";
     "million" never scales `v`). The remote model then emitted "17 billion dollars" — a ×1000
     error injected into `out_p`.
   - GeoNames single-representative mapping mislabels same-named entities: surface `Idaho` →
     fill `"a city in colorado"`; surface `California` → `"a city in new jersey"` — false fills
     that also poison any restoration decision.
   - Lattice-producer nonsense fills in the wild: `"an information"` for surface
     "nasal congestion" (health-condition) — the open issue register
     (`docs/issues/2026-07-09-lattice-producer-generation-quality-issue-register.md`) is visible
     end-to-end.

**Net effect on priorities:** fixing upstream fill quality and extending the cascade's fill
matcher for fine lattice fills dominate any reconstructor work — they address the 325 residue
entries that are upstream damage or matcher misses, versus ≤40 that any reconstructor could
even see.

## Failure modes

### Training

1. **Supervision scarcity is terminal, not incidental.** 43 admitted edits (fewer after the audit
   tightens the gate) is the entire positive supervision for a seq2seq that must learn full-document
   verbatim copying plus surgical edits. LoRA on flan-t5-base will memorize or collapse to copy.
   The addressable population itself is tiny — ~54 D-reworded mentions in 151 docs — so the data
   builder cannot be scaled out of this; more docs yields admitted edits at roughly 0.3/doc.
2. **Structural label noise from quote-boundary splicing.** The superset-splice failures above are
   not judge mistakes to prompt away: a quote is a grounding snippet, not a mention boundary.
   Building targets by `splice_at_quote` bakes destructive over-deletion into the gold labels. Any
   model trained on them learns to delete flanking content, and `edit_guard` would *accept* those
   edits at inference (a replace region may delete arbitrarily much as long as the insert is the
   tight surface and overlaps a fuzzy anchor) — do-no-harm bounds what enters, not what leaves.
3. **Admission-gate miscalibration, both directions (measured).** Mandatory correspondence
   (`fill ⊨ quote`) applies only to {DATETIME, QUANTITY, LOC, ORG}. Coarse legs leak: MISC has no
   `_type_sane` branch (always passes) and DEM's word list passes junk — where the Jul-7 audit
   found false restorations. Fine legs over-reject: the fine word-lists hold family words only,
   so genuine mentions ("Type 2 Diabetes Mellitus") fail `_type_sane` — fine admission recall ≈ 0.
   Fix: correspondence mandatory for every non-exact restore; drop the word-list legs as
   admission criteria (the fresh probe's "fixed" gate variant: 7 admits incl. the fine-type case
   the current gate's `corr_all` variant wrongly dropped).
4. **Stale upstream, retraining treadmill.** The build used coarse `DEM`/`MISC` types; the runtime
   substitutor is now fine-typed (`docs/specs/lattice-substitutor.md`) with lattice-cache fills
   ("computer and mathematical occupation") and underscore placeholders (`<HEALTH_CONDITION_1>`).
   Every upstream change — detector v7, lattice cache rebuilds, ranker policy, the currently-open
   lattice-producer quality issues — shifts the fill distribution and invalidates the training set.
   A learned full-text editor must be re-distilled and re-audited each time; at ~0.3 admitted
   edits/doc and real proxy cost per build, that treadmill is the dominant lifecycle cost.
5. **No-op dominance teaches silent uselessness.** lexsum is 70/80 no-op targets. The
   loss-minimizing behavior is "copy the input", which also always passes the guard — a model that
   does nothing looks safe and ships.
6. **Teacher circularity in evaluation.** `classify_recovery` anchors on the same judge quotes that
   built the training targets; quote-boundary bias appears identically in train and eval, so the
   eval over-credits agreement with the teacher rather than measuring recovery.

### Inference

7. **Full-document regeneration for 1–3 edits, with all-or-nothing acceptance.** flan-t5-base must
   reproduce ~500 words verbatim; any stray reword anywhere causes `edit_guard` to reject the
   *entire* candidate, discarding the good edits with the bad. Expected realized recovery ≈ cascade
   baseline, with GPU cost added. The architecture couples its safety mechanism to discarding its
   own work.
8. **Context-window truncation.** Clinical inputs median ~469 words (max 717) + restore map +
   instruction against `max_length=1024` tokens: the longest documents truncate, so tail residue is
   invisible at inference and truncated *targets* teach tail deletion at training.
9. **Guard anchor fragility.** Anchors come from `rapidfuzz.partial_ratio_alignment(fill, prepass)`
   at score ≥60. Generic fills ("something", "a disease") in long documents anchor arbitrarily or
   collide; the overlapping-anchor bail-out then rejects wholesale. High silent-fallback rate on
   exactly the documents with the most residue.
10. **Constrained decoding does not rescue this.** The plan's own upgrade path
    (`prefix_allowed_tokens_fn`) fixes hallucinated *tokens*, but not supervision scarcity (#1),
    label noise (#2), truncation (#8), or the full-copy cost (#7).

### Design level

11. **Wrong tool: this is a localization problem, not a generation problem.** Extraction is
    client-side and `R` holds every original surface — no content needs generating, ever. The only
    open question per residue entry is *where* (and *whether*) its mention sits in `out_p`. A
    generator introduces hallucination risk, which demands the guard, which then rejects the
    generator. The system fights itself.
12. **The data builder already contains the real reconstructor.** Judge-locate → admission gate →
    splice *is* the pipeline that produces the "gold" targets. Distilling it into a seq2seq is a
    speed optimization taken before the quality ceiling of the thing being distilled was ever
    measured — and the audit shows that ceiling is currently below the do-no-harm bar because of
    quote boundaries, which distillation inherits rather than fixes.
13. **Marginal value does not carry the complexity.** Ceiling: ~54 D-mentions of 1059
    generalizations (~5% of spans; ~18% of survived spans), of which the deterministic
    original-surface/acronym proposers (`docs/plans/2026-07-05-survived-recovery-extractor.md`)
    take a slice with no model. The trained-editor stack (builder + trainer + guard + 3 checkpoints
    + per-stratum eval + retraining treadmill) buys at most a few points of end-to-end recovery.
14. **Near-unfalsifiable success criteria at this n.** `wrong_insert == 0 ∧ harm_rate == 0 ∧
    recovered > cascade`, per stratum, on ~10–30 residue mentions per stratum, judged through the
    same teacher: single noisy quotes flip the outcome in either direction.

## Proposed design: extractive locate-then-splice

Keep the cascade exactly as is. For each residue entry `(surface s, fill f, type t)` run a
**locate → admit → splice** pipeline in which generation never occurs and every mention is accepted
or rejected independently.

**1. Locate (cheap-first ladder).**
- a. Deterministic proposers from the survived-recovery plan: exact-`s` lock (do-no-harm on leaked
  originals), acronym/initialism of `s`, lemma/morphology variants of `f` and `s`.
- b. The existing semantic candidate machinery (`cloak.extract` MiniLM windows) at a lower
  threshold, typed by `_type_sane`.
- c. **Local judge fallback** for what remains: the same Qwen prompt the data builder uses, one
  call per residue document (56/136 docs on the measured corpus). Local model — no remote-privacy
  cost; same infra the builder already pays for.

**2. Admit (one gate, all types).** `restorable` with three fixes: correspondence (`fill ⊨ quote`,
deterministic `_value_compatible` then NLI) becomes **mandatory for every non-exact restore**, not
only {DATETIME, QUANTITY, LOC, ORG}; empty/`UNKNOWN` fills are rejected outright; fine runtime
types are first-class (hyphenated types, underscore placeholders per the lattice-substitutor spec).

**3. Splice boundary-tight.** Never replace the whole quote. Token-align `f` (and `s`-variants)
against the quote and replace only the aligned sub-span, keeping flanking tokens
("Type 2 **Diabetes Mellitus**" → "Type 2 **diabetes** Mellitus" is still wrong — so when the
alignment does not cover the quote's full noun phrase head, shrink to the aligned head or abstain).
Alignment failure ⇒ abstain. This removes the superset-destruction class structurally instead of
asking a judge prompt or a learned editor to respect boundaries.

**4. Repair locally.** The existing article-agreement fixer (from the substitutor) on the spliced
sentence; nothing generative.

**5. Accept per mention.** Each splice carries its own gate verdict and stat
(`gen_locate_splice`, `gen_abstain`); one bad candidate no longer discards the rest.

**Optional learned component (deferred until measured need).** If the judge fallback is too slow or
misses systematically: a small **extractive pointer** — encoder QA head, query = `(t, f, s)`,
output = start/end span in `out_p` or null. Trained on the same distilled quotes but as
token-aligned span labels, which (a) need an order of magnitude less data than full-text targets,
(b) cannot hallucinate content by construction, (c) abstain natively via null. This is the correct
place for learning in this pipeline, and it is not needed to ship step 1–5.

### Why this beats the generative editor

- **Do-no-harm by construction, not by guard.** Only `R` originals can enter, only at located
  spans, only boundary-tight. `edit_guard` (the most intricate code in the module) becomes
  unnecessary; the safety budget moves to admission, where the audit shows the real errors live.
- **Zero training to reach the measurable ceiling.** The judge-locate path can be evaluated on
  held-out docs *today*; its recovery number upper-bounds any distilled model and directly answers
  whether learning is warranted at all.
- **No copy-fidelity, truncation, or all-or-nothing failure classes.** Nothing regenerates the
  document.
- **Robust to upstream drift.** Localization is a function of `(f, s, out_p)`, not of the
  substitutor's type schema or lattice fills; detector/lattice/ranker changes do not invalidate a
  training set because there isn't one (until the optional pointer, whose span labels are also
  cheaper to rebuild than audited full-text targets).
- **Complexity proportional to the prize.** ~5% of spans get a deterministic ladder + one local
  LLM call on residue docs, instead of a training pipeline + guard + eval treadmill.

### Tradeoffs (honest)

- **Distributed rewordings stay unrecovered.** A mention smeared across a clause or morphologically
  fused ("Minneapolis" → "in Minnesota" with preposition change) cannot be fixed by splicing. The
  generative editor could in principle handle these — but the measured D-cases are dominated by
  contiguous mentions, and the guard would have rejected the fancy rewrites anyway. Accept as a
  measured miss class; report it.
- **Judge latency at inference.** Qwen3.6-35B-A3B serially on residue docs is the heavy path.
  Mitigations: it runs only when deterministic + semantic proposers fail; batching per doc (one
  prompt covers all residue entries); the pointer distillation exists as the upgrade path once a
  measured win justifies it.
- **Splice grammar.** Boundary-tight splices inside reworded context can read awkwardly. The
  article fixer covers the common case; residual awkwardness is a utility cost to measure, not a
  correctness violation (no false facts are asserted).
- **Specificity-faithfulness calls remain policy.** When the remote model re-derives a *more
  specific* correct term ("Type 2 Diabetes Mellitus" inferred from Metformin context; fill was
  "a disease" for surface "diabetes"), restoring the doc_orig surface reduces specificity while
  restoring faithfulness. Position: faithfulness wins — `out_final` must not assert specifics
  `doc_orig` does not ground — but only under a boundary-tight splice; over a superset quote it is
  pure damage.

## Decision path (revised after the fresh probe)

1. **Fix upstream first — it dominates.** (a) `bucket_quantity` unit-garble + unscaled
   word-magnitudes (`'$8.5 million'` → `'between 4.25 and 17 $illion'`; deterministic, unit-tested
   fix); (b) GeoNames same-name entity-class errors (`Idaho` → "a city in colorado"); (c) the
   lattice-producer quality register. These corrupt `doc_p` (utility + a ×1000 numeric error
   observed in `out_p`) and account for a visible share of the residue explosion.
2. **Extend the cascade's fill matcher for fine lattice fills** (reworded generic phrases:
   "a city in wisconsin" → "Wisconsin city's"). 325/365 residue entries are matcher misses or
   upstream damage — a bigger recovery lever than any reconstructor.
3. **Reconstructor: locate-then-splice only, generative editor closed.** Measured: 7 admissible
   edits per 80 docs (no trainable population), 6/7 recovered by deterministic splice under the
   fixed gate. Ship the fixed admission gate (correspondence for all non-exact restores, empty
   fills rejected, word-list legs dropped) + splice with preposition-preserving boundaries; the
   judge-locate call is the fallback localizer. The extractive pointer stays deferred until the
   admissible population is 10× larger.
4. Re-measure end-to-end after steps 1–2; the survived/residue landscape will shift and the
   reconstructor's remaining scope should be re-derived from that measurement, not from this one.

## Non-goals

- No change to the cascade's resolved-span behavior (A/B/C paths).
- No remote calls anywhere in reconstruction; the judge is local.
- No recovery attempt for D-4 (model re-derived a different specific) beyond abstain — unchanged.
- No claim of end-to-end privacy/utility improvement without an attacker-measured result.
