---
type: training-experiment
status: planned
created: 2026-07-06
model: EncoderPolicy — ModernBERT-base (frozen encoder) + per-span action head, autoregressive over spans; BC'd from the floor-walk teacher
dataset: ranker_env_full — clinical 267 (aci+mts) + lexsum 161 + wikibio 160; 484 trainable docs / 7416 decision spans; validated round-trip probes (probes_validated.json, reader=Qwen3.5-0.8B, scorer v2)
result: pending — pre-flight complete (reader pinned, support scan PASS, pipeline smoke green); full pilot not yet run
tags: [ranker, stage1, roundtrip-reward, expert-iteration, rloo, count-floors, served-reader, pilot]
companion: ../../docs/specs/RL/roundtrip-ranker-infiller.md
supersedes-track: 2026-07-05-RL-ranker-v2-stage1-floor-env.md (surrogate reward; this is the round-trip track)
---

# Stage-1 ranker on the ROUND-TRIP reward — pilot (ExIt + RLOO, served reader)

First RL run on the **round-trip reward** (`R_rt` = realized fact recall on `out_final` over a
doc's validated train probes), superseding the surrogate-reward track (v1/v2). Spec:
[roundtrip-ranker-infiller.md](../../docs/specs/RL/roundtrip-ranker-infiller.md).

## Objective & hypothesis
Does learned per-span action selection (level generalization vs placeholder) beat the
floor-walk behavior-clone init on realized round-trip recall, at matched floors? Success =
greedy read-out moves off the BC init with `r_heldout` gain, per corpus (never averaged).
Pre-registered null (spec §pre-registered outcomes): ExIt *and* RLOO both flat at the
floor-walk = "selection adds little" — a legitimate finding.

## Pinned components (this run's re-gate surface; spec table is canonical)
- **Reward / remote task model:** gemma 4 (E4B), non-thinking, temp 0, max_tokens 1024, `:8060`.
- **QA reader:** **Qwen3.5-0.8B, served** on llama-swap (`UD-Q8_K_XL` GGUF, `-np 6`),
  **serial-per-context** (workers=1 → llama.cpp prompt-cache reuses the note-prefix KV),
  non-thinking, temp 0; **scorer `fact_score` v2** (canon + number-gate + containment +
  acronym). Selected this session over roberta-base-squad2 (extractive abstained ~40% on
  relational/section-structured notes, FM1) and over local fla (Qwen3.5's hybrid cache breaks
  cross-batch prefix-KV; served-serial is correct + ~6.6× faster than fanning across slots).
- **Probe teacher:** Qwen3.6-35B-A3B, non-thinking, prompt v3.
- **Extractor:** rule exact/fuzzy-90 + semantic-window (`invert(..., semantic=True)`).
- **π_rank:** ModernBERT-base encoder + per-span head; floors k_T = 100 all types.

## Training data
`ranker_env_full` (built this session): clinical 267 + lexsum 161 + wikibio 160 (wikibio +
qmsum added as new round-trip corpora; **qmsum dropped** — "summarize discussion" is a QA
desert, 86% ceiling-reject, see dev-log). Validated probes `probes_validated.json` (492 docs;
kept-facts/doc clinical 2.7 / lexsum 1.62 / wikibio 1.4; reader-F1 on kept ≈ 0.93–0.96).

## Training config (planned)
`scripts/train_ranker.py --reward roundtrip --policy encoder --exit-rounds 4 --exit-epochs 10
--epochs 5 --G 12 --cf-frac 0.25`, fixed floors (`--randomize-floors` is BC-only, hard-errors
in roundtrip). ExIt (ReST^EM) as workhorse, RLOO refiner (LOO baseline, no std division, DAPO
tie-filter, entropy 0.01, KL off), exact per-span counterfactual credit.

## Selection & operating point
Greedy read-out at the env floors; `r_train` AND `r_heldout` reported **per corpus, never
averaged** (spec). One policy = one (floor-config) point.

## Evaluation & success criteria
Round-trip realized recall on held-out probes/docs vs the floor-walk Pareto; whole-task-quality
regression gate (ROUGE-L/BERTScore/entity-F1 must not regress). No realized-privacy claim from
this record (Phase-5 attacker is separate).

## Pre-flight (done this session — gates for the pilot)
- **Support scan: PASS** (`results/roundtrip_support_scan.json`, 30 docs/corpus, 100 swaps):
  reward flips both directions above the quantization step (17 up-sig / 18 down-sig, max_abs
  1.0). Trainer gate now satisfied without `--force-ungated`.
- **Environment `aset` build bug (fixed — a bug, not a finding):** `build_arms_artifact`
  never wrote the per-action `aset`, so the k-floor mask rejected every level → 0/7416
  spans had ≥2 legal actions → total desert (the first smoke placeholdered everything, the
  first scan found 0 swaps). Fixed: compute `aset` in `action_table`; re-annotated the
  existing full arms/env post-hoc (no re-detect, no re-gate). After: 6075/7416 spans have a
  real decision. (Distinct from v2's null, whose env already had `aset`.)
- **Pipeline smoke: green** (2 docs, gate-legit): ExIt 2 winners (best 0.5 > BC 0.3125),
  RLOO gradient every epoch (no ties, entropy ~0.58), greedy `ph` 1.0→0.28, `r_heldout` 0.33.
  Confirms reward→policy plumbing learns on the aset-fixed env.

## Results
**pending** — full pilot not yet run.

## Cost / bottleneck (measured, pre-flight)
Reward round-trip ~5.5 rt/s (scan) and no catastrophic gemma↔reader swap-thrash (co-resident).
On high-probe docs the **serial reader dominates** (~1 rt/s at ~13 probes/doc) — the RL-loop
optimization is reader parallelism *across* rollouts (serial *within* a context); not yet wired
into `train_ranker`'s reward loop. Full-scale runs should measure whether the reader dominates
wall before scaling.

## Risks & caveats
- **Placeholder-gaming / QA-necessity** (open, pilot proceeds with current impl):
  [issue](../../docs/issues/2026-07-06-placeholder-gaming-reward-qa-necessity.md) — probes
  test surface recall, not task-necessity of the PII; floor-rejection is the only structural
  guard. Monitor greedy `ph` and the whole-task-quality regression gate.
- Direct identifiers (PERSON/CODE) are absent from the env (no lattice → dropped by
  `action_table`); the ranker decides quasi-identifiers only (names placeholdered by rule).
- Reader parallelism not yet in the RL loop (see Cost).

## Artifacts
`data/ranker_env_full.json`, `data/task_arms_full.json`, `data/probes_validated.json`
(gitignored) · `results/roundtrip_support_scan.json` (PASS) · smoke
`results/ranker_train_rt_enc_smoke.json` · code at branch `rl-reader-served-q8-aset-fix`
(commit `cc901c8`).

## Sources
Spec [roundtrip-ranker-infiller.md](../../docs/specs/RL/roundtrip-ranker-infiller.md);
corpus expansion [dev-log](../../docs/dev/logs/2026-07-06-qa-build-corpus-expansion.md);
reader/reward issue [placeholder-gaming](../../docs/issues/2026-07-06-placeholder-gaming-reward-qa-necessity.md);
predecessors (surrogate track) [v1](2026-07-04-RL-ranker-v1-stage1-bandit.md) ·
[v2](2026-07-05-RL-ranker-v2-stage1-floor-env.md).
