---
type: research
status: current
created: 2026-07-08
updated: 2026-07-27
tags: [rl, ranker, lattice, anonymity-counts, reward-design, feature-debt, issue-register]
companion: [../specs/RL/roundtrip-ranker-infiller.md,
            ../specs/offline-k-anonimity-risk-walk.md,
            ../specs/generalization-lattice-cache.md,
            2026-07-06-placeholder-gaming-reward-qa-necessity.md]
---

# Issue register — RL ranker environment, reward design, and lattice count pipeline

> **Post-refactor note (2026-07-27):** module paths in this doc were updated to the
> regrouped `src/cloak` layout — see the path mapping in [the cleanup plan](../plans/2026-07-27-codebase-cleanup-refactor.md).
> Still named here but **deleted, not moved** in that cleanup: `train/ranker.py` (the v1 feature row; the live one is assembled in `src/cloak/ranker/environment.py`).

Full-repo review 2026-07-08 (docs + working-tree code + data artifacts), prompted by the RL
null-result post-mortem. Issues ordered by fix priority. Each entry: what is wrong, evidence,
and the fix direction. Severity groups: **blocking** (invalidates or blocks training),
**degrading** (weakens training signal), **hygiene** (correctness debt, deferred).

## Blocking

### 1. Reward cannot express the training objective (placeholder-gaming / task-necessity)

The round-trip reward is recall-only over surface-recall probes; a placeholder echoes through
the remote model and `invert()` restores it, so placeholder often matches or beats a truthful
generalization. Floor-rejection — the only structural guard — is subtractive: it deleted 64%
of candidate probes (2,300/3,606) instead of rewarding generalization-over-placeholder. No one
has verified the corpora contain task-necessity structure (facts whose *category/relation*, not
surface, the task needs). Full analysis:
[placeholder-gaming issue](2026-07-06-placeholder-gaming-reward-qa-necessity.md).

Compounding mechanism (found by the corrected context-ablation follow-up,
`research-wiki/experiments/context-injection-surface-ablation.md`): gemma collapses to
bracketed templates on heavily anonymized `doc_p`, and the BC init *is* floor-walk — training
starts at the utility-collapse cliff where most facts yield zero reward and zero gradient (all
36 "never-recovered" facts recover at the ceiling).

**Fix direction:** redesign the remote task and/or QA-pair construction so that probes test a
downstream inference a generalization preserves and a placeholder breaks. Brainstorm scheduled;
candidate directions in the placeholder-gaming issue §"What a real fix requires".

### 2. Per-level counts never reach the legality mask

The runtime read path is `aset_count` (`src/cloak/lattice/anonymity.py:240`) → `lookup_count`
(`src/cloak/lattice/profiles.py:96-99`), which reads the single **row-level** `count` — the
per-level `level_counts` the producer writes into proposed artifacts (`merge.py:151-163`) and
that the coherence-cleaning spike orders and smooths are **not on the runtime read path at
all**. The uncommitted `_build_indexes` change (`lattice_profiles.py:81`) makes it worse: it
switches the level index from first-wins to **summing row-level counts across every surface
sharing a level** — neither the old semantics nor a membership count; it double-counts rows.

**Consequence:** the entire in-flight lattice-count fix (producer per-level counts, cleaned
drug-health-procedure artifact) changes zero legality decisions until fixed.

**Fix direction:** promote `level_counts` into the runtime profile schema
(spec: [offline-k-anonymity-risk-walk](../specs/offline-k-anonimity-risk-walk.md) §Required
Artifact Shape); make `lookup_count` per-level; define the multi-surface merge deliberately
(member-set union / max, never row-sum).

### 3. No certifying counts exist for domain types

The spec requires deterministic member-set inversion over a source universe; the deterministic
branch exists (`lattice_producer/counts.py:38-110`, `member_set` → `certifying`) but is dead
for the produced types: drug/health items are queued `force_model_proposal=True`
(`queue.py:130`), so all 2,679 levels in
`data/lattice_profiles/proposed/drug-health-procedure.proposed.json` carry the teacher's
`proposed_count` (`grounding: model-proposed`, `member_set_ref: None`). The cleaning spike
(`scripts/spikes/clean_drug_health_lattice_coherence.py`) makes those numbers *coherent*
(PAVA isotonic + real-world anchors), not *grounded* — its own docstring says so. The canonical
artifact is worse: 1,609 rows at the 1.0 fail-closed default, 808 at the forbidden 1000.0
legacy default.

**Fix direction (pick one, explicitly):** (a) wire real source universes through the existing
`member_set` branch — Mondo/DOID descendant sets for health-condition, ChEMBL/ATC for drug —
per the spec; or (b) re-scope the spec: model-proposed + coherence-cleaned counts are the
operating basis and the "certifying k-anonymity count" language is dropped. Either is honest;
treating fabricated counts as certifying is not.

### 1b. Round-trip reward is non-deterministic under concurrency (MEASURED 2026-07-09)

The spec's load-bearing invariant — `R_rt` deterministic given `doc_p` (§The one subtlety:
cache coherence, exact counterfactual credit, ExIt pool all rest on it) — **is violated by the
concurrent reward loop.** Measured with `scripts/spikes/reader_parallelism_smoke.py` (clinical,
cache-cold, temp-0): the same jobs at `workers=1` vs `workers=6` produce **different rewards on
6/18 (larger run) and 1/8 (smaller run) jobs**, parallel systematically higher by ~one
quantization step (~0.125). Stage-attributed: **`out_p` (gemma generation) itself differs on
5/8 jobs** — the root is llama.cpp batched-inference non-determinism (logits depend on the
concurrent batch composition; temp-0 greedy flips the argmax at token boundaries), inherited by
the reader.

**Scope:** `--rt-workers` defaults to 8, so *every run to date* (v3 smoke, support scan, the
2026-07-08 ablation) computed rewards under concurrency. Cached reward values depend on how many
jobs were in flight when first computed; the "exact" per-span counterfactual credit is
approximate; ExIt "keep the strict winner" can keep a concurrency-lucky rollout.

**Perf context:** the concurrency buys a real 3.1× wall speedup (119s→38s at 13.5 probes/job);
the reader does dominate wall as the v3 note said. So this is a genuine determinism-vs-throughput
tradeoff, not a free fix.

**Decision needed (empirical-honesty rule — do not engineer around silently):**
- (a) **Serialize generation** (`rt_workers=1` for the gen call, reader may stay parallel on a
  fixed `out_p`): restores determinism, ~3× slower.
- (b) **Accept as reward noise**, reframe the spec: `R_rt` is deterministic per
  (doc_p, concurrency) only; quantify the flip rate and fold it into the reward-quantization
  budget (it compounds the existing ~0.2-step quantization). Cache becomes concurrency-tagged.
- (c) **Investigate server flags** (llama-server determinism / single-slot gen pool) for a
  middle ground.
Recommendation pending user call; (a) for any run that needs the cache/counterfactual guarantees,
(b) only if the flip rate is small relative to the signal being measured.

## Degrading

### 4. Probe density is ~10% of target

Kept facts/doc: clinical 2.7 / lexsum 1.62 / wikibio 1.4
(`research-wiki/training/2026-07-06-RL-ranker-v3-roundtrip-pilot.md`) vs the strategy target of
10–20 (`docs/plans/2026-07-05-roundtrip-rl-strategy.md` §2). Reward quantization stays coarse
(~1/n_probes steps), groups tie, the DAPO filter discards work. Interacts with issue 1: a probe
redesign changes what counts as a probe — redesign first, then scale.

### 5. Policy feature debt (walk_risk, corpus one-hot, N_FEAT drift)

`action_features` (`train/ranker.py:23-39`, retired 2026-07-27; the v2 feature row is assembled in
`src/cloak/ranker/environment.py`) still carries walk_risk at index 1 and
the 5-wide corpus one-hot, both decided-removed in the RL spec (§Note — walk_risk, §Note —
corpus one-hot). Actual **N_FEAT = 30** (7 scalars + 18-wide type one-hot + 5 corpus): the
fine-DEM expansion widened `TYPES` to 18 and the spec's arithmetic (19) was never reconciled;
the module docstring still describes the 17-era layout. A 30-dim row dominated by mostly-zero
one-hots plus an orphaned privacy scalar is a plausible contributor to noisy gradients and poor
generalization. Cheap, already-decided edits; land them and update the spec table + docstring.

### 6. The context channel is unvalidated and its ablation is invalid as-run

After walk_risk + corpus removal the frozen ModernBERT CLS becomes the **sole** context
channel, and the one ablation that tested it
(`research-wiki/experiments/context-injection-surface-ablation.md`) is INVALID as-run: the
Level-1 marginal baseline held other spans at floor-walk — the utility-collapse operating
point — so the 64%-flat rate and the producer null are artifacts. Re-run with a
ceiling/realistic-mid baseline (same machinery, cheap) before betting the design on raw CLS;
contrastive embedder + mean-pool is the named upgrade path.

## Hygiene

### 7. Arms-artifact desync / re-gate discipline

`scripts/build_ranker_env.py:69-84` copies frozen `aset` values from the arms artifact
verbatim; no runtime path connects the env to the new lattice pipeline (the chain is offline:
`fine_lattice_profiles.json` → `build_arms_artifact.py` → arms JSON). Any profile rebuild —
including the issue-2 fix — shifts the legality mask and invalidates the cache/gate/policies
together (RL spec §The one subtlety). Needs an explicit rebuild + re-gate step; silent drift
would desync count semantics.

### 8. Cleaned-artifact schema drift

The cleaning spike writes `level_grounding` (singular) and pops `level_groundings`
(`clean_drug_health_lattice_coherence.py:356-357`), while `merge.py:validate_proposed_artifact`
(:184,193) requires the plural — the cleaned artifact fails the producer's own validator, and
the singular name collides with the per-candidate field in `counts.py:108`.

### 9. Direct identifiers are outside the learned loop

PERSON/CODE spans have no lattice, so `build_arms_artifact.action_table` drops them; they are
placeholdered by rule, never ranker decisions, never probed (documented in the
placeholder-gaming issue §Related finding). Acceptable for the pilot; bounds what any RL result
claims about the highest-risk identifiers.

### 10. Producer gaps

- **No NLI / grammar gate** in the proposal pipeline — `gates.py:67-122` is purely
  lexical/structural, contradicting the lattice-substitutor spec's NLI-entailment requirement.
- **Coverage:** no profile rows for `ethnicity`, `family-role`, `MISC` (lattice-eligible per
  `queue.py:LATTICE_RUNTIME_TYPES`).
- **Dead CLI flags:** `--model` / `--escalation-model` are ignored — the graph hardcodes
  `QWEN36_ESCALATION_MODEL` (`graph.py:168,250,259`).
- **Reader parallelism** across rollouts still not wired into the RL reward loop; the serial
  reader dominates wall on high-probe docs (v3 record §Cost).
- **Spec staleness:** RL spec pin table says remote max_tokens 512; code pins 1024
  (`roundtrip.py:17-26`). The v3 record is correct; update the spec.

## Attribution check (vs the 2026-07-08 post-mortem hypotheses)

- *Bad generalization lattice* — confirmed, but the in-flight fix is incomplete in two ways
  (issues 2, 3): counts don't reach the mask, and no path produces certifying counts.
- *Bad training tasks / credit* — confirmed as the top-priority item, with a sharper mechanism:
  the BC init sits at the remote model's utility-collapse point (issue 1).
- *Non-optimal policy design* — confirmed with a concrete number (N_FEAT 30 vs spec 19,
  walk_risk + corpus one-hot still live; issues 5, 6).
