---
type: research
status: current
created: 2026-07-09
updated: 2026-07-09
tags: [lattice-producer, coherence, grounding, langgraph, health-condition, medical-procedure, issue-register, creativity-collapse]
companion: [2026-07-08-lattice-producer-coherence-hardening.md]
---

# Issue register — lattice producer generation quality (health-condition, medical-procedure)

Root-cause investigation of `data/lattice_profiles/proposed/drug-health-procedure.proposed.json`
after the coherence-hardening plan's fixes (companion doc) were implemented and run. Damage
assessment (same-session): **613/646 (94.9%) health-condition entries and 163/163 (100%)
medical-procedure entries carry no level that is both semantically specific and backed by a
count that agrees with every other occurrence of that same label.** 0/1893 health-condition and
0/435 medical-procedure level-groundings are certifying — all `model-proposed`.

Issues below are split by where the fix lives: **deterministic-level** (queue/gate/graph code —
fixable without touching a prompt) vs **generative-level** (prompt text / context construction —
what the model is shown and told). A third section covers one item that isn't yet a confirmed
bug, just evidence gathered for a future decision. Ordered by severity within each section.

**Important correction to the companion plan's status:** Fix Areas 2, 3, 4, and 6 there are
marked complete and are implemented as specified — but issues #1, #2, #4 below show the code
paths that exercise them don't fire the way the plan assumed, for reasons outside what that plan
tested. The unit-test suite (255 passing) validates each mechanism in isolation; it does not
catch a queue-building path bypassing the mechanism entirely, or a control-flow edge that only
matters at real run scale (1000+ items, multi-hour, crash/resume cycles).

## Deterministic-level issues

### 1. `--category`-seeded queue items unconditionally skip `deterministic_lookup` — the primary cause

`queue.py:104-125` (`_queue_from_profile_categories`, used when a run's queue is seeded from
existing `lattice_profiles.json` surfaces rather than freshly-mined PII spans) hardcodes
`"force_model_proposal": True` on every item it builds, unconditionally (line 120). `route_selected`
(`graph.py:140-156`) checks that flag *before* anything else and routes straight to
`propose_with_llama_swap`, never calling `deterministic_lookup`.

**Evidence:** all 1798 items remaining in the `drug-health-procedure` run's queue carry the flag,
across all three runtime types (`drug`: 539, `health-condition`: 771, `medical-procedure`: 488 —
100% of each). All 4279 accepted rows produced by this run so far are `source_family:
model-proposed`; zero are `openfda`/`doid-is-a`/`icd10pcs-prefix`. The model compensates by writing
plausible-looking `count_evidence` text that *mentions* "openfda-ndc" / "ICD-10" while the
structured `source_family` underneath still reads `model-proposed` — i.e. it mimics the citation
style of real grounding without any lookup behind it.

**Why the plan's Fix Area 2 unit tests didn't catch it:** they test `reference_candidates_for` and
`deterministic_lookup` directly, and test that `route_after_deterministic` routes correctly *given*
a `deterministic_lookup` call happened. Nothing in the test suite builds a queue via
`_queue_from_profile_categories` and asserts `deterministic_lookup` is reachable from it.

**Fix direction:** `_queue_from_profile_categories` should not force model proposal at all —
category-seeded items are exactly the population most likely to already have a real surface for
`reference_candidates_for` to match against (drug/health-condition/medical-procedure canonical
values pulled straight from the profile). Removing line 120 lets `route_selected` fall through to
`deterministic_lookup` like every other item.

### 2. No check that a reused exact-vocabulary label's count agrees with the vocabulary's existing count for that label

`gates.py:196-206` is the only vocabulary-aware check in the gate. It fires the near-duplicate
check *only* when `not vocabulary.has_exact(level)` — so once a label is an exact string match to
something already accepted, **any** `proposed_count` is accepted for it, with no comparison to
what that same label was already assigned earlier in the run. `CanonicalVocabulary._seed_from_run`
(`vocabulary.py:59-78`) compounds this: it records a label's count only the first time it's seen
(`if key not in self._labels: self._labels[key] = ...`) and never updates it — there isn't even an
authoritative "current" count to check new occurrences against beyond the very first one.

**Evidence:** of labels reused ≥2 times in the accepted output, 151/166 (91%) in health-condition
and 13/13 (100%) in medical-procedure have more than one distinct `level_counts` value for the
identical string. Example: `"general medical condition"` appears with `450000000.0` in one entry
and `115000.0` in another; `"human medical condition"` ranges `4100.0` → `8500000000.0` (6 orders
of magnitude) across its 64 occurrences.

**Fix direction:** add a gate check — when `vocabulary.has_exact(level)`, compare the candidate's
`level_count` to the vocabulary's recorded value for that label (some tolerance band, since real
counts do have legitimate estimation variance) and reject/flag on large disagreement instead of
silently accepting. Requires `_seed_from_run` to expose the count it already has for a label, not
just membership.

### 3. Near-duplicate gate has a permanent one-time bypass; nothing reconciles two already-accepted synonyms

`gates.py:196-200`'s near-duplicate check runs only for a level not yet `has_exact`. The first
time each of two independent paraphrases of the same concept gets accepted (e.g. `"medical
condition"` and `"human medical condition"`), both become permanently exempt from the check for
the rest of the run — nothing subsequently notices they're 50%+ token-overlapping synonyms
occupying the same semantic slot.

**Evidence:** health-condition's broadest (top) level splits 98/64/55 across three unmerged
near-synonyms (`medical condition` / `human medical condition` / `general medical condition`),
which collectively account for 217/350 (62%) of that category's chains at the time reviewed.
Medical-procedure is worse: 100% of its accepted levels are `model-proposed`, and 39/163 (24%)
entries share the exact 3-label chain `(medical procedure, clinical service, human activity)` —
with another chunk of entries reversing the order of the first two labels (80 forward, 20
reversed for the `medical procedure`/`clinical service` pair alone), meaning the run can't even
agree internally which of its own two favorite buckets is the broader one.

**Fix direction:** this needs a continuous invariant, not a curated list (see issue #5) — a
lightweight periodic or on-accept pass that Jaccard-clusters the *already-accepted* vocabulary
against itself (not just new candidates against old) and collapses newly-detected clusters to one
representative before they can diverge further in count.

### 4. The one reconciliation pass (`normalize_coherence`) runs at most once, only at full batch/queue completion

`should_continue` (`graph.py:426-430`) routes to `normalize_coherence` only when
`processed >= max_items`; the only other entry point is `route_selected` (`graph.py:140-156`)
returning it when the queue is fully empty. For an unbounded, long-running, crash-prone job (this
one: 1798+ items, already crashed and manually resumed twice this session from `APITimeoutError`),
the reconciliation pass may never execute even once in practice — it depends on the *entire*
remaining queue finishing in one uninterrupted pass.

**Evidence:** `data/lattice_profiles/proposed/drug-health-procedure.proposed.json` as reviewed has
never had `normalize_coherence` applied to it (confirmed: `level_grounding` statuses are 100%
`model-proposed` with no `clusters_applied`/reordering signature, and the raw per-item count
contradictions in issues #2/#3 are still present verbatim). This turns issues #2 and #3's damage
into effectively permanent corruption for any run that doesn't finish end-to-end in one sitting —
which, at current per-item latency (single items observed taking 20+ minutes on the 35B model),
this run will not do for a very long time.

**Fix direction:** run `normalize_coherence` periodically (e.g. every N accepted items, or on a
wall-clock interval) instead of only at the true end — it's already idempotent-safe to call
repeatedly on the growing accepted set.

### 5. `CLUSTERS` (the only synonym-merge mechanism) is a static, hand-curated table with no path for a live run to extend it

`coherence.py:48-86`'s `CLUSTERS` dict was hand-built from the earlier drug-only manual cleanup. It
has no mechanism to learn a new synonym pair a live run invents — `"human medical condition"` and
`"general medical condition"` aren't in the `health-condition` cluster's variant set, so even a
`normalize_coherence` pass that *did* run (issue #4) would not merge them; they'd survive as
separate PAVA-ranked labels, still incoherent with `"medical condition"` and each other.

**Fix direction:** either (a) treat this as a live curation task — periodically diff the run's
accepted vocabulary against `CLUSTERS` and add missed variants by hand, matching the existing
workflow used for drug, or (b) replace/augment it with the same Jaccard-based auto-clustering
proposed for issue #3, applied at `normalize_coherence` time instead of (or in addition to) gate
time.

### 6. Vocabulary context ranking (`context_slice`) trusts raw, unvalidated model-proposed counts

`vocabulary.py:110-113` ranks the labels shown to the model by `-self._labels[label]` — raw
self-reported `proposed_count`, descending, no distinction between `certifying` and
`model-proposed` values. Because the model itself invents wildly inflated counts for the generic
sinks (up to `8500000000.0`), those labels dominate the front of `context_slice()`'s output for
every subsequent item, which is also the model's primary reuse cue (see generative issues #8, #10) — a
rich-get-richer loop on top of issue #3, entirely mechanical (the ranking function itself), even
though the values being ranked are generative in origin.

**Fix direction:** rank certifying-grounded labels ahead of model-proposed ones regardless of
count magnitude, or normalize/cap self-reported counts before they're allowed to influence
ranking.

### 7. No retry/backoff around the LLM call — a single timeout kills the whole process

`propose.py:229`'s `client.chat.completions.create(...)` has no timeout override and no retry.
Confirmed twice this session: the run process died with an uncaught `APITimeoutError` and had to
be manually restarted with `--resume` each time, losing nothing on disk (checkpointing works) but
requiring manual intervention every time an item runs long. Orthogonal to output quality, but
responsible for why this run has already needed two manual restarts in one evening.

**Fix direction:** wrap the call in a bounded retry (2-3 attempts, escalating timeout), and
separately, `--thinking-budget-tokens` (already a CLI flag, `scripts/run_lattice_producer.py:33`)
should probably have a non-`-1` (unbounded) default given how often single items are running
20+ minutes.

## Generative-level issues

### 8. [PRIMARY] Vocabulary context causes progressive creativity collapse over the run — not just correlated with genericness, causally driven by it

Bucketing `accepted.jsonl` into 5 equal chronological chunks (file append order = processing
order) shows the sink pattern is not static noise — it is a collapse that starts small and
compounds as the run progresses, tracking exactly when a sink label first becomes visible in
`context_slice()`.

**health-condition** (646 items, 5 chunks of ~129):

| chunk | fully-generic | new specific labels introduced | top sinks (count) |
|---|---|---|---|
| 1 (earliest) | 1% | 300 | `medical condition` (51) |
| 2 | 8% | 187 | `human medical condition` (37), `medical condition` (36), `general medical condition` (27) |
| 3 | **29%** | **20** | `human medical condition` (92), `general medical condition` (87) |
| 4 | 24% | 19 | `human medical condition` (98), `general medical condition` (86) |
| 5 (latest) | 18% | 25 | `human medical condition` (97), `general medical condition` (77) |

Chunk 1 has near-zero genericness and 300 fresh specific labels; `"human medical condition"` and
`"general medical condition"` don't exist yet. The collapse hits exactly at chunk 2→3, the moment
those two phrases first entered the vocabulary and became visible to later calls. From that point
the new-specific-label introduction rate crashes ~15× (187/chunk → ~20/chunk) and never recovers —
the model stops coining anything new and reuses whichever inflated sink is currently on top.

**medical-procedure** (163 items, 5 chunks of ~32) never has a healthy phase at all:

| chunk | fully-generic | new specific labels | top sinks |
|---|---|---|---|
| 1 | 66% | 8 | `medical procedure` (29), `clinical service` (26) |
| 3 | 75% | 4 | `human activity` overtakes both |
| 5 (latest) | 69% | **0** | `human activity` hits 32/35 items |

No golden window here because (issue #12) almost nothing grounds this category from item 1, so
the sink forms immediately instead of after a delay.

**Why this is the primary generative issue, not just one contributor among #9-#11 below:** issues
#3 (permanent near-dup bypass), #6 (count-based visibility ranking amplifying invented numbers),
and #9 (soft "reuse if it fits" instruction, no counts shown) are each individually plausible as
minor contributors — the chunk data is what shows they compose into an actual exploitation loop
with a visible trigger point and an irreversible one-way transition, not just steady-state noise.
The moment a sink label is inflated and visible, it wins every future call, permanently — visibility
itself triggers the collapse. Passing prior generations as context was meant to be a coherence
measure; empirically, on this run, it is the dominant force suppressing generation diversity,
outweighing whatever coherence benefit it provides (and per issues #2/#4, it does not even
reliably deliver that coherence benefit either).

**Fix direction:** any fix to #3/#6/#9 needs to be evaluated against this temporal signature
specifically — re-bucket a post-fix run the same way and confirm chunk 3-5 recovers something
close to chunk 1's new-specific-label rate, not just that aggregate genericness percentages drop.
A fix that reduces the *count* of generic labels but preserves the early-lock-in dynamic (e.g.
just adding more entries to `CLUSTERS`) would not address this issue.

### 9. The fixed prompt requires a "broadest useful generalization" tier for every single item

`propose.py:71-72` (the hardcoded instruction string) tells the model to order candidate levels
"from nearest truthful generalization to broadest useful generalization" for every item, with no
escape hatch for a surface that has no natural broad category beyond a truism. This is what
manufactures a catch-all tier on every single item that lacks a real anchor to terminate on
instead (which, per issue #1, was every item in this run). Not an error in the model's compliance —
it's doing exactly what it's told — but the instruction guarantees sink-label invention at scale
whenever the deterministic path isn't available.

**Fix direction:** needs product-level judgment, not just a code change — e.g. allow the model to
say "no useful broader tier beyond the nearest one" for genuinely narrow surfaces, rather than
forcing invention.

### 10. Vocabulary-reuse instruction is advisory only, and the context never shows counts

`canonical_vocabulary_instruction` (`propose.py:83-87`) says "if any label in
`canonical_vocabulary_slice` already fits, reuse it verbatim" — soft language, no consequence
framed for coining a near-synonym instead. Separately, `context_slice()` (`vocabulary.py:110-113`)
returns label **strings only** — the model is never shown what count is already on file for any
of them. Even a model that reuses a label string exactly right has no way to know what count
"agrees" with prior occurrences; it re-derives a number from scratch every time. This is the
generative-side mirror of deterministic issue #2 — the context doesn't supply the information the
gate doesn't check either.

**Fix direction:** change `canonical_vocabulary_slice` to carry `{label, count}` pairs, and change
the instruction to explicitly require reusing the attached count (not just the label string) when
reusing a label.

### 11. Vocabulary visibility is bottlenecked to 8 labels per call against a 350-770-entry vocabulary

`--max-context-rows` defaults to `8` (`scripts/run_lattice_producer.py:32`). Combined with issue
#6's count-based ranking, the vast majority of an already-large accepted vocabulary is invisible
to any single proposal call — for health-condition (350+ distinct accepted labels observed
mid-run) the model sees at most the top 8 by (untrusted) count, so it frequently cannot see the
"correct" existing label to reuse even when fully willing to.

**Fix direction:** raise `max_context_rows` for runtime types with large vocabularies, and/or make
the slice selection domain-aware (e.g. token-overlap with the current surface) rather than a flat
top-N by count.

### 12. No cross-item memory beyond the label-string vocabulary slice

Each `propose_with_llama_swap` call is independent — temperature 0.0, single-shot, no conversation
history. The only continuity mechanism across items is the vocabulary label list. This is a
structural property of the chosen architecture (stateless per-item calls), not a specific defect,
but it's the reason issues #2, #3, #8, and #10 are even possible: there is no shared session for
the model to stay self-consistent within, only the thin, lossy, and (per #8) irreversibly-biased
channel of a label-string list.

## Needs more evidence before classifying as a bug

### 13. ICD-10-PCS reference-source coverage for medical-procedure looks low even when the deterministic path is allowed to run

Sampled `reference_candidates_for` directly against 40 real queued surfaces per type, bypassing
issue #1 entirely (calling the function straight, not through the graph): health-condition (DOID)
hit 10/40 (25%) with genuine matches (`alcoholic cardiomyopathy` → `extrinsic cardiomyopathy`,
`kyphosis` → `spinal disease`, etc.); medical-procedure (ICD-10-PCS) hit only 2/40 (5%)
(`upper gi fluoroscopy` → an ICD-10-PCS prefix class; `examination of female reproductive system`
→ an ICD-10-PCS prefix class), missing on informal mined phrases like `upper endoscopy`,
`maneuvers`, `dietary indiscretion`, `back pain evaluation`.

This suggests that even after fixing issue #1, medical-procedure will keep falling through to free
generation for most of its surfaces — ICD-10-PCS's precise coded-procedure vocabulary is a
structurally poor match for informal clinically-mined phrases. Also notable: the *base*
`lattice_profiles.json` (before this run) already has 0 certifying groundings for both
health-condition and medical-procedure (vs. 143/539 for drug) — so there's no existing production
evidence this ever produced a meaningfully-grounded corpus for either type, at any point, not just
in this run.

**Not yet a confirmed bug** — need to decide whether 25%/5% coverage is acceptable as one input
among others (issue #1 fix will still help by seeding some real anchors into the shared
vocabulary even at that hit rate) or whether medical-procedure needs a second/different reference
source before the deterministic-first strategy is worth relying on for it at all.

## Suggested fix order

Issue #1 first — it's the switch that made every other symptom visible at 100% severity instead
of partial. Issue #8 is the primary generative-side issue — the temporal collapse it documents is
the actual mechanism turning #3/#6/#10's static weaknesses into runaway genericness, so any fix to
those three should be judged against #8's chunk-bucketed re-measurement, not just aggregate
genericness percentages. Re-run a small `--max-items` smoke batch after #1 alone and re-measure
the damage numbers above (including a fresh chunk breakdown per #8) before deciding how much of
#2-#12 is still worth fixing versus how much #1 alone resolves.

## Post-overhaul re-measurement (2026-07-09)

Overhaul implemented on branch `lattice-producer-overhaul` (plan
`docs/superpowers/plans/2026-07-09-lattice-producer-overhaul.md`). Re-measured with a fresh run,
`nvidia/nemotron-3-super-120b-a12b:free` via OpenRouter (a hosted stand-in, not the register's
Qwen3.6-35B baseline — so this validates pipeline mechanics + quality, not a matched-model #8
re-measurement).

### health-condition — 60-item batch (`data/lattice_runs/smoke-overhaul`): RESOLVED

- **Grounding (#1, #13-health):** 215/215 accepted levels `certifying` (130 `doid-is-a` + 85
  `deterministic-aset`), **0 model-proposed**. Baseline was 0/1893 certifying, 100% model-proposed.
  Removing the forced `force_model_proposal` (fix #1) routes health-condition surfaces to the DOID
  `is_a` path, which supplies real, source-backed levels; the model path is rarely reached.
- **Creativity collapse (#8):** eliminated. Per-chunk `fully_generic` stays ≤8.3% (was ramping
  1%→8%→**29%**→24%→18%) and `new_specific` holds steady ~20–31/chunk (was crashing 300→187→**20**).
  No one-way lock-in. NB: because levels now come from DOID's real ontology rather than free model
  generation, the collapse is moot — #8 was a symptom of #1 forcing the model to invent every level.
- **Count coherence (#2, #3, #6):** 0/24 reused labels disagree >4× (baseline 91%/100%); counts are
  real member-set sizes (monotone, e.g. `drug allergy` 52 → `allergic disease` 176 → `immune system
  disease` 896 → `disease of anatomical entity` 9162), not prevalence.
- **Chain length (#B):** no length-1 entries (baseline ~170 flagged); chains are 2–5 levels.

### medical-procedure — 12-record batch (`data/lattice_runs/smoke-mp`): #13 CONFIRMED, model-dependent

The dedicated `medical-procedure` batch ran (12 records, 26 accepted level-rows). It is the real
test of #13 (ICD-10-PCS surface coverage → model fallback) and of the model-path guardrails
(#9/#10/#11), which the health-condition run did not exercise.

- **Grounding (#13):** **22/26 (85%) `source_family: model-proposed`, only 4/26 deterministic**
  (`deterministic-aset` on the reused `medical procedure` / `medical and surgical procedure` /
  `obstetric procedure` anchors). The opposite of health-condition's 215/215 certifying — because
  ICD-10-PCS's coded-procedure vocabulary does not match informal clinically-mined phrases
  (`arthrography`, `alcohol consumption`, `upper endoscopy`), the deterministic path rarely fires
  and the model supplies the levels.
- **The "mimics grounding" pattern persists on the model path** (issue #1's signature, now
  isolated to model-proposed rows): `count_evidence` cites "CPT code range… 112 distinct codes",
  "SNOMED CT hierarchy… 85 concepts", "ICD-10-PCS… 120 codes" while `source_family` underneath
  reads `model-proposed` — invented counts wearing citation clothing.
- **Coherence held better than the pre-overhaul baseline:** the count-agreement gate +
  relevance-ranked slice kept reuse consistent (coherence report: 16 canonical levels, 0 level
  changes, 1 same-count collision) — so #2/#3/#6's *mechanisms* work on the model path; what they
  cannot do is make an invented count *certifying*.

**Verdict:** the deterministic-first strategy resolves health-condition (DOID) but **not**
medical-procedure — mp remains model-dependent, and no second procedure source was added. The
open decision from #13 stands: accept 85% model-proposed for mp, or add a procedure source
(SNOMED-CT procedure hierarchy / CPT) before relying on grounded counts for it.

### What resolved each issue

- **#1** (forced model proposal): fixed — the master switch, as predicted; drove the certifying %
  from 0 to 100 for health-condition.
- **#2/#3/#6** (count incoherence, sink ranking): fixed via corpus-membership-derived counts
  (coherence) + count-agreement gate + relevance-ranked vocabulary slice.
- **#4** (single normalize pass): periodic normalization (`--normalize-every`).
- **#7** (no retry): bounded retry + backoff, now also covering 429.
- **#8** (collapse): moot for health-condition (deterministic-first); the model-path guardrails
  (#9/#10/#11) are validated only once the medical-procedure run stresses the model path.
- **#9/#10/#11** (prompt genericness, advisory reuse, 8-row context): type-specific prompts,
  count-annotated + relevance-ranked vocabulary slice (default rows 8→20).
- **Counts vs k-floor (final-review seam):** the k-floor is now a chain-wide anonymization-time
  gate (keeps specific sub-floor rungs, requires the chain's broadest rung to reach the floor),
  per `anonymity.py`'s "legal iff aset ≥ k_floors" definition.

### Deferred / still open

- **#13** (ICD-10-PCS coverage): **measured, confirmed open** — the mp run fell through to the
  model on 85% of levels (above). Decision pending: accept model-proposed counts for
  medical-procedure, or add a SNOMED-CT-procedure / CPT reference source.
- **#5** (static `CLUSTERS` table): not extended; the relevance-ranked reuse + count-agreement gate
  reduce synonym proliferation at the source, and the mp coherence report (0 level changes)
  suggests it is now low-priority — re-evaluate only if a larger mp/model-path run regresses.
- Minor: the ≥2-level floor can discard a lone deterministic-`certifying` level (narrow; those
  chains are usually multi-tier).

## Status refresh (2026-07-09, late)

Fixed/open at a glance after the overhaul + hybrid-drug work:

| issue | status | resolved by |
|---|---|---|
| #1 forced model proposal | **fixed** | overhaul (drop `force_model_proposal`) |
| #2/#3/#6 count incoherence, sink ranking | **fixed** | membership-derived counts + count-agreement gate + relevance-ranked slice (`59b63fe`, `4e74eb6`) |
| #4 single normalize pass | **fixed** | periodic `--normalize-every` |
| #7 no retry/backoff | **fixed** | bounded retry incl. 429 (`9ff83b5`); empty-response tolerance (`3285369`) |
| #8 creativity collapse | **fixed for health-condition** (deterministic-first makes it moot); **unproven on the model path** (mp is model-dependent — re-bucket a larger mp run to confirm the guardrails hold) |
| #9/#10/#11 prompt genericness, advisory reuse, 8-row context | **fixed** | type-specific prompts, count-annotated slice, rows 8→20 |
| #12 no cross-item memory | structural (by design); mitigated by the count-annotated slice |
| #13 ICD-10-PCS coverage | **confirmed open** — mp 85% model-proposed; second source undecided |
| #5 static CLUSTERS | deferred — low priority post-overhaul |

New mechanisms since this register was written (not in the table above): hybrid drug augmentation
(`bc0d2f7`, `af52cbd`) and reference-source-miss routing to the model rather than the profile
cache (`639c9b5`) — these extend the drug path and the deterministic/model routing; they do not
change the health-condition (resolved) or medical-procedure (#13 open) verdicts.
