---
type: reference
status: current
created: 2026-07-10
updated: 2026-07-10
tags: [extractor, frozen-extractor, verification, correspondence, calibration, do-no-harm, issue-register]
companion: [docs/specs/extractor-frozen-rl-reward.md, docs/plans/2026-07-09-frozen-extractor-implementation.md]
---

# Issue register — frozen zero-shot extractor (post-implementation)

The frozen extractor (`src/cloak/frozen_extractor.py`, spec
`docs/specs/extractor-frozen-rl-reward.md`) is implemented, reviewed, and merged (9-task SDD
branch + fix waves). Its machinery is correct and safe by construction on the smoke
(`results/extractor_vs_cascade_smoke.json`: 0 do-no-harm violations, 80 docs). **It is not yet
usable as the RL reward anchor**: a mini-calibration exposed that the verification gate cannot,
as built, separate real reworded mentions from generic-word echoes. Thresholds are therefore
still the spec's uncalibrated placeholders. This register lists what remains before the extractor
can recover any residue at an acceptable false-splice rate.

All evidence is measured, 2026-07-10, on the fine-arms calibration set
(`data/extractor_calibration_set.jsonl`, 66 cal docs / 28 audited gold mentions + 340 ABSENT
tripwires; `results/extractor_calibration_fit2.log`, `results/false_splice_dump.log`). This is a
diagnostic-grade forerunner of the spec's benchmark, not the benchmark.

## Issue 1 (Critical) — correspondence gate certifies on shared head-word, not restatement

**Symptom.** At the tightest swept setting (`ASSIGN_MARGIN=0.05`, `NLI_ENTAIL=0.80`): 12/28 gold
mentions recovered, but **47 false splices**. No swept threshold pair reaches false=0 — recovery
is flat at 12 across the whole grid while false count only grows as thresholds loosen, so the two
thresholds are **not the binding constraint**.

**Root cause.** The correspondence gate (`verify()`) asks NLI whether
`"The text mentions {fill}." ⊨ "The text mentions {chunk}."`. With generic fills and short
(1–6 word) candidate chunks, NLI returns *entailment* whenever the chunk shares the fill's head
word — it is scoring lexical overlap, not semantic restatement. Measured false splices of this
kind (from `false_splice_dump.log`):

- `fill "a state"` → chunk `"state"` (surface was `"U.S. District Court for the Central District
  of California"`) — matched the common word "state";
- `fill "a state"` → chunk `"State"` (surface `"Michigan"`);
- `fill "something"` → chunk `"which"` (surface `"American Sign Language"`);
- `fill "a symptom"` → chunk `"symptoms of"`; `fill "a disease"` → chunk `"disease"`;
- `fill "an organization"` → chunk `"court"` (surface `"Court"`).

~30 of the 47 are this mechanism. This is the exact "generic levels certify too easily" failure
pre-registered in the spec's Measured-limitation section and independently hit by the substitutor
profile-match spike (`docs/specs/substitutor-profile-match-retrieve-verify.md`).

**History.** The gate first shipped at *sentence* granularity
(`"The text mentions {fill}." ⊨ {full out_p sentence}`), which was **unsatisfiable** — 27/27 gold
pairs rejected at every threshold (recovery 0). Commit `92c8704` moved it to phrase granularity to
make true pairs admissible; that fixed the false negatives but opened this false-positive channel.
Neither granularity is right: the gate needs a *specificity* criterion, not a different premise.

**Minimum fix (design, not a tweak).** Require the chunk to entail the fill **and** carry content
beyond the fill's own head word: reject when the normalized chunk is a substring/lemma of the fill
(or vice-versa) with no added specific token (proper noun, digit-run, or out-of-fill content word).
This is the open design question the spec flagged; it needs its own small spike + re-fit, and it
may show the small NLI model is simply insufficient and an entailment *margin* against a
type-name hypothesis (chunk must entail the fill more than it entails the bare type name) is
required.

## Issue 2 (Important) — leaked-original surfaces reach the ladder instead of a tier-0 lock

**Symptom.** ~10 of the 47 false splices are the original surface **already standing verbatim** in
`out_p`, spliced over itself: `"lungs"`→`"Lungs"`, `"tylenol"`→`"Tylenol"`, `"march"`→`"March"`,
`"tenderness"`→`"Tenderness"`. The splice is idempotent (harmless — the smoke's do-no-harm check
passes), but the entry should never have been treated as recoverable residue: the surface leaked
through `doc_p` undetected, which is a *privacy* event, and restoring it is a no-op that pollutes
the recovery/false accounting.

**Root cause.** Tier 0 (`cloak.extract._rule_prepass`) resolves by locating the **fill**; it has no
exact-original-surface leak lock, so a leaked original with an absent fill falls through to the
residue and then to the ladder. The spec's Proposed-architecture stage 0 names an
"exact-surface leak lock" but `_rule_prepass` does not implement one.

**Minimum fix.** Add a leak-lock in tier 0 (or as a pre-ladder filter): when an entry's original
surface occurs verbatim (word-boundary) in `out_p`, mark it resolved/locked and exclude it from
residue — never route it to localization. Count it as a leak diagnostic, not a recovery. This also
tightens the `frozen_extractor` protected-span set (already excludes standing originals from splice
targets — Issue is that they still enter the *entry* list and get scored).

## Issue 3 (Minor) — calibration gold audit was over-aggressive

Building the calibration set, the coordinator hand-nulled 42/91 judge-proposed golds. At least one
was a genuine mention wrongly dropped: `chronic obstructive pulmonary disease` → `COPD`
(`aci/D2N015`) is a correct D-2 acronym recovery that the ladder makes and that was scored as a
false splice because its gold was nulled. A few `nulled`-bucket "false" splices in
`false_splice_dump.log` are similarly correct-but-unscored. Re-audit before the next fit; do not
treat the current 12/28 recovery or 47 false count as precise — both have audit noise of a few
entries. (The Issue-1 conclusion is robust to this: ~30 generic-word echoes are unambiguous
regardless.)

## What is NOT broken (so the register isn't misread)

- The ladder's control flow, determinism, do-no-harm-by-construction (protected spans + pre-splice
  assertion), margin rule, and RL opt-in wiring are all reviewed and correct; the smoke shows 0
  do-no-harm violations on 80 real docs.
- `models=None` path is byte-identical to the legacy cascade; merging changes no existing behavior.
- The absent-mention reality is unchanged and correct: ~48% of residue entries have no mention in
  `out_p` at all (the remote dropped the content under compressive tasks), and abstaining on those
  is right. Issues 1–2 concern only the addressable ~6% (reworded) + the leaked-original tail.

## Sequencing

1. Issue 2 (leak-lock) — smallest, removes ~10 false splices and a privacy-accounting bug; do first.
2. Issue 3 (re-audit) — cheap; needed before any number is trusted.
3. Issue 1 (specificity gate) — the real design work; spike + re-fit + held-out eval. Until it
   lands, the extractor's thresholds stay at spec placeholders and it is **not** wired into any RL
   run (the roundtrip path is opt-in and defaults to the legacy cascade — `extractor_models=None`).
4. Only after 1–3 clear false=0 on held-out: freeze thresholds (version bump) and, separately,
   build the full spec benchmark to re-certify at statistical weight.
