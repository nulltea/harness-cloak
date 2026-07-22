---
type: research
status: current
created: 2026-07-22
updated: 2026-07-22
tags: [issue, lattice-profiles, anonymity-counts, count-provenance, rl-ranker, data-quality]
companion: [../specs/RL/interactive-ranker-v2.md,
            ../specs/qa-builder-v2.md]
---

# Issue: deferred lattice-profile count and hierarchy defects

The 2026-07-22 Ranker-v2 count audit found profile-quality defects that should be repaired, but
they no longer block RL delivery. Reward-facing counts are now sourced from each decision's
matched profile row; missing or inadmissible row-local evidence is represented explicitly as
provisional null data. No repair in this issue is approved for automatic application.

The complete human-review record is
[`results/ranker_v2/count_repair/proposed-edits.md`](../../results/ranker_v2/count_repair/proposed-edits.md).
Machine-readable diagnoses and execution evidence are in
[`profile-diagnoses.json`](../../results/ranker_v2/count_repair/profile-diagnoses.json),
[`repair-report.json`](../../results/ranker_v2/count_repair/repair-report.json), and
[`unresolved-queue.jsonl`](../../results/ranker_v2/count_repair/unresolved-queue.jsonl).

## Diagnosed defects

The review covered 52 profiles. The diagnosis classified them as:

- **31 merge-key mismatches:** the old frozen environment looked up a level string across a
  runtime type and used the maximum count, rather than using the count from the matched profile
  row. These are primarily environment count-sourcing defects; some underlying rows also merge
  distinct identities.
- **18 wrong authored-order defects:** the ladder order or level selection needs source-level
  repair rather than count sorting, clipping, or another reward-time workaround.
- **3 wrong count-evidence defects:** level values/order can be retained, but the evidence and
  resulting count need correction.

The proposed edits classify 29 profiles as count-only and 23 as order/fill-changing. The latter
invalidate affected decisions' existing QA support if eventually applied.

## Profiles requiring external evidence

Thirteen profiles cannot be repaired from the available local evidence and must remain
unresolved: `LOC:albania`, `LOC:armenia`, `LOC:central african republic`, `LOC:florida`,
`LOC:georgia`, `LOC:madagascar`, `LOC:namibia`, `LOC:portugal`, `LOC:vermont`,
`drug:acetaminophen`, `drug:ibuprofen`, `medical-procedure:blood tests`, and
`medical-procedure:hemoglobin a1c`.

Do not guess their missing levels, order, counts, or evidence. The exact failure and proposed
next evidence step for each profile is recorded in `proposed-edits.md`.

## Locally evidenced proposals awaiting confirmation

Three proposed edits have complete local DOID descendant evidence:

- `health-condition:adenoma`: replace the non-certifying cellular-proliferation count with the
  DOID-backed count while retaining the level sequence.
- `health-condition:bowel dysfunction`: replace the current ladder with the locally supported
  `intestinal disease` → `gastrointestinal system disease` → `disease of anatomical entity`
  chain and its DOID descendant counts.
- `health-condition:hypertension`: use the locally supported `heart disease` →
  `thoracic disease` → `disease of anatomical entity` chain and its DOID descendant counts.

These are **awaiting confirmation; do not auto-apply**. Their exact counts, grounding records,
and downstream classifications are in `proposed-edits.md` and the offline artifacts under
[`offline-run/`](../../results/ranker_v2/count_repair/offline-run/).

## Unapproved model queue

The unresolved producer queue contains **20 items**, each requiring one proposed request to
`Qwen3.6-35B-A3B`. No request was made and that model use is not approved. The queue includes
surface variants and repair evidence that collapse to fewer unique profile-level problems, so
its 20 items are distinct from the 13 unresolved profiles above. Preserve the queue as evidence;
do not call the model without explicit approval.

## Repair CLI defects

Two code defects were exposed by the offline-only run:

1. `scripts/run_lattice_producer.py` rejects an `--out` path outside
   `data/lattice_profiles/proposed/`, while the task's documented deterministic repair path is
   under `results/ranker_v2/count_repair/`. The guard conflicts with the supported plan path.
2. In the producer graph, `force_model_proposal` is evaluated before `offline_only`; therefore
   offline-only does not dominate forced model routing. Offline execution must fail closed or
   queue the item before any model branch can be selected.

Both deserve focused code fixes and regression tests. They were diagnosed only; neither was
changed as part of the frozen-environment migration.

## Resolution criteria

Resolve this issue only after every proposed profile edit is individually confirmed, all
external-evidence work is explicitly approved and recorded, the CLI defects have regression
tests, and the canonical profile plus embedding index are promoted through the normal validated
artifact workflow. Until then, RL consumes row-local grounded counts and tags missing evidence
as provisional rather than blocking delivery.
