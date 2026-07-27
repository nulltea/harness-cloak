---
type: research
status: current
created: 2026-07-10
updated: 2026-07-27
tags: [lattice, lattice-profiles, health-condition, data-quality, qa-build, reward, deduplication,
       aliases, issue]
companion: [docs/specs/generalization-lattice-cache.md,
            docs/specs/RL/training-task-env.md]
---

# Issue: near-duplicate condition entries / aliases inflate QA-build spans and reward credit

> **Post-refactor note (2026-07-27):** module paths in this doc were updated to the
> regrouped `src/cloak` layout — see the path mapping in [the cleanup plan](../plans/2026-07-27-codebase-cleanup-refactor.md).
> Still named here but **deleted, not moved** in that cleanup: `scripts/build_probes.py` (superseded by `scripts/build_arms_artifact.py` + `scripts/build_qa_utility_artifact.py`).

## Summary

When the QA (ladder/decision) build detects clinical spans and looks up their lattices in
`data/lattice_profiles/lattice_profiles.json`, **multiple distinct surface strings for the same
underlying condition each resolve to a lattice and are treated as separate spans**. On the first
clinical validation doc this produced redundant paid teacher calls and inflated the per-rung
ceiling-reject rate. The root is two related facts about the `health-condition` profile:

1. **Alias variants both resolve to levels.** `hypertension` and `high blood pressure` are not
   canonical rows — they are *aliases*. Each resolves (correctly) to non-empty levels via the
   loader's alias index, so a document that mentions both surfaces yields **two lattice spans for
   one condition**, with the same or near-identical level ladders.
2. **Some near-synonym conditions are separate canonical rows with identical levels.** e.g.
   `diabetes mellitus` and `high blood sugar` are two canonical entries whose `levels` are
   byte-identical (`glucose metabolism disease → carbohydrate metabolic disorder → inherited
   metabolic disorder → disease of metabolism`). These are arguably one condition (disease vs its
   finding) split into two rows.

## Evidence (measured 2026-07-10)

- `health-condition` has **339 canonical entries**; **49 distinct level-lists are shared by >1
  canonical entry, covering 116 entries**. (Scan: group canonical rows by `tuple(levels)`.)
- **Most shared-level groups are NOT duplicates** — they are legitimately different diseases that
  share a generalization ancestor (`cellulitis`, `dermatitis`, `psoriasis` → `skin disease`;
  `epilepsy`, `migraine`, `brain edema` → `brain disease`). Correct behavior, not the issue.
- **The true near-duplicates are synonyms**, e.g.:
  - `diabetes mellitus` vs `high blood sugar` — separate canonical rows, identical levels.
  - `hypertension` / `high blood pressure` — both aliases resolving to the same
    cardiovascular ladder (`artery disease → vascular disease → cardiovascular system disease →
    disease of anatomical entity`).
  - `heart failure` / `congestive heart failure` — both resolve to the same
    (`heart disease → thoracic disease → disease of anatomical entity`) ladder.
- Internal consistency check is clean on one axis: **0 canonical entries list an alias that is
  itself another canonical key** — so this is not a canonical/alias collision, it is (a) multiple
  aliases of one condition surfacing in text, and (b) a minority of synonym rows.

## Impact

- **QA build (measured, 1-doc clinical validation, `results/ladder_build_detect_1doc.log`):** the
  detector emitted both `hypertension` and `high blood pressure`; both became lattice spans and
  each consumed a paid teacher call, and the redundant one drove ceiling-rejects (the note states
  one surface, so the reader lookup for the other fails). Of 11 lattice spans, several were
  redundant condition variants — wasted paid `nemotron-3-super` calls and a higher
  `reader_rung_reject_rate`.
- **RL reward (prospective):** the same fact appears as two ranker decision spans → per-span
  counterfactual credit is split across duplicates, and anonymity-set accounting double-counts one
  condition. The two-channel carrier reward would score the same fact twice.

## What is and isn't a defect

- **Not a defect:** different conditions sharing a category ancestor (the bulk of the 116). The
  lattice is *supposed* to converge siblings at a shared level.
- **Defect (profile-side):** genuine synonym rows split into separate canonicals with identical
  levels (`diabetes mellitus` / `high blood sugar`). These should be one canonical + aliases, per
  the schema's alias design (`docs/specs/generalization-lattice-cache.md`).
- **Not strictly a profile defect (build-side):** one condition legitimately having several
  aliases (`hypertension`/`high blood pressure`) is correct data; the redundancy is that the QA
  build / detector does not **collapse co-referent surfaces within a document** before spending
  teacher calls and creating spans.

## Candidate fixes (unresolved — for design)

1. **Build-side dedup (cheapest, local to the QA/reward path):** collapse detected spans of the
   same `runtime_type` that resolve to the **same canonical row / identical level-list**, keeping
   one representative surface per document. Removes the redundant paid calls and the split credit
   without touching the profile. Belongs in `_detect_docs` / span assembly.
2. **Profile-side merge (producer/merge):** fold true synonym canonical rows into one canonical +
   alias set in the lattice producer / `scripts/merge_lattice_profiles.py`, so
   `diabetes mellitus`/`high blood sugar`-type pairs are one row. Needs a synonym oracle (DOID
   exact-synonym edges are the natural source) — do not merge on identical-levels alone (that
   would wrongly merge sibling diseases).
3. **Alias coverage audit:** verify the alias sets are complete enough that co-referent surfaces
   resolve to one canonical (they do here), so build-side dedup by canonical id is reliable.

Fix 1 is the immediate unblocker for the QA build (and the RL reward's per-span credit); Fix 2 is
the durable data-quality correction. They are complementary.

## Sources

- Profile: `data/lattice_profiles/lattice_profiles.json` (`profiles.health-condition`, 339 rows;
  updated 2026-07-10, commit `9f1d0da`). Schema: `docs/specs/generalization-lattice-cache.md`.
- QA-build symptom: 1-doc validation run `results/ladder_build_detect_1doc.log` +
  `data/probes_ladder_validated.json` (teacher `nvidia/nemotron-3-super-120b-a12b`, reader
  `Qwen3.5-0.8B`); the ladder/decision build is `scripts/build_probes.py --detect`.
- Reward context: [training-task-env spec](../specs/RL/training-task-env.md) (per-span credit,
  two-channel carrier reward).
