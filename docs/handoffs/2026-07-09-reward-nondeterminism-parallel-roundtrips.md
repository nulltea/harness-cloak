---
type: handoff
status: current
created: 2026-07-09
updated: 2026-07-09
tags: [rl, round-trip-reward, determinism, concurrency, exit, rloo, ranker, handoff]
companion: [docs/issues/2026-07-08-rl-env-and-lattice-count-issue-register.md,
            docs/specs/RL/roundtrip-ranker-infiller.md,
            docs/specs/RL/training-task-env.md,
            research-wiki/training/2026-07-06-RL-ranker-v3-roundtrip-pilot.md]
---

# Handoff — round-trip reward is non-deterministic under parallel round trips

## Focus for the next session

Design a **non-determinism-tolerant RL policy sampler / reward-evaluation path** for the
round-trip reward, given the measured finding below. The determinism-vs-throughput decision
(issue register §1b) is open and gates the RL pilot.

## The finding (measured 2026-07-09)

`R_rt` (round-trip reward) is **not deterministic given `doc_p`** when the reward loop runs
concurrent round trips. Full write-up + numbers + options: **issue register §1b**
(`docs/issues/2026-07-08-rl-env-and-lattice-count-issue-register.md`). Do not re-derive it —
read that section. One-paragraph recap:

- Smoke `scripts/spikes/reader_parallelism_smoke.py` (clinical, cache-cold, temp-0): same jobs
  at `workers=1` vs `workers=6` flip **~1/8–1/3 of per-doc rewards** by ~one quantization step
  (~0.125), parallel **systematically higher**.
- Root cause is the **generation stage**: stage-attribution showed `out_p` (gemma) differs on
  5/8 jobs — llama.cpp `-np` batched-inference reorders FP reductions, so concurrent batch
  composition changes logits and temp-0 argmax flips. The reader inherits it.
- `--rt-workers` defaults to 8, so **every run to date** (v3 smoke, support scan, the
  2026-07-08 ablation) computed rewards under this noise.
- Perf context: concurrency buys a real **3.1×** wall speedup (119s→38s at 13.5 probes/job) —
  the reader does dominate wall. So it is a genuine tradeoff, not a free fix.

## Why it specifically breaks ExIt (the intended workhorse)

ExIt selects by **max**, not average: `keep (doc, argmax(r)) if max(r) > baseline`, then SFT on
winners. Noise does not cancel under a max — it *chooses the supervised label*. Upward-biased
jitter + max-over-G selection = extreme-value bias: with a ~30% per-rollout flip and G=12, ~98%
of docs would emit a spurious "winner". Plus the cache (= ExIt candidate pool per the spec's
"one subtlety") stores concurrency-dependent values, so a rollout can win one round and lose the
next → no fixed point. RLOO tolerates the noise better (averaged gradient) but its "exact"
per-span counterfactual credit is no longer exact.

## Key architectural fact the next agent needs

There is **no separate "reader parallelism"** in the code. `roundtrip_batch(jobs, workers=N)`
(`src/cloak/train/roundtrip.py:43-63`) parallelizes **whole round trips** — each worker runs
`gemma.generate → invert → reader`. So `workers>1` parallelizes generation too, which is why the
non-determinism lands in `out_p`. The reader is serial *within* a job (prefix-KV reuse,
`reward.py:_read_batch`) and parallel *across* jobs only as a side effect of job-level batching.
The three reward call sites (ExIt `exit_round` :381, RLOO `train_roundtrip` :294, eval readout
:768 in `scripts/train_ranker.py`) all pass job lists to `roundtrip_batch`.

## Suggested design directions (for a non-determinism-tolerant sampler)

Not decisions — starting points for brainstorming:
1. **Two-phase reward:** generate serially (or from a deterministic cache) for a fixed,
   cache-coherent `out_p`; then parallelize only the reader over fixed `out_final`. Shrinks the
   blast radius to reader jitter; generation + counterfactuals stay exact. Note the reader
   (also llama.cpp temp-0) may still jitter under concurrency — measure it.
2. **Server-side determinism:** investigate llama-server single-slot / deterministic flags for
   the gen model; a dedicated `-np 1` generation pool co-resident with a parallel reader pool.
3. **Noise-tolerant selection for ExIt:** replicate-and-average `R_rt` before the max, or a
   margin threshold `max(r) > baseline + δ` with δ ≥ the measured jitter; RLOO-only (drop the
   max-selection) if averaging is cheaper than serializing.
4. **Reframe the reward** (empirical-honesty path): declare `R_rt` deterministic per
   (doc_p, gen-concurrency), tag the cache with concurrency, fold the flip rate into the
   quantization budget — only acceptable if the flip rate is small vs the signal, which the
   task-necessity probe redesign (below) is meant to enlarge.

Whatever is chosen, **re-run the support scan and any pilot under the chosen path** — prior PASS
verdicts were computed under the noisy loop.

## State of the broader work (context, not action items)

- **Reward/task redesign (approved, in progress):** ladder + decision probes so generalization
  beats placeholder on task-necessity. Spec `docs/specs/RL/training-task-env.md`; generator
  `src/cloak/train/ladder_probes.py` (validated on clinical: Qwen3.6 84% rung survival, pinned).
  Not yet wired into `scripts/build_probes.py`. This enlarges the reward signal, which also
  raises the noise-tolerance margin.
- **Issue register** `docs/issues/2026-07-08-rl-env-and-lattice-count-issue-register.md` — the
  full prioritized list. Landed since it was written: item 1 (per-level counts on the mask,
  committed by user in `9f1d0da`), item 3 (ranker features 30→24, `e326de1`), item 4 (ceiling
  ablation, `b2a5085`), item 5 measured (`5a15e8c`, this finding). **Item 2 (certifying counts:
  ontology member-sets vs re-scope-to-estimates) is still an open user decision.**
- **Ablation caveat:** the 2026-07-08 context ablation ran at `workers=6`, so its recall numbers
  carry this noise; the directional conclusion (surface-recall is flat at both baseline
  extremes) is robust to ~0.125 jitter, but re-run deterministically once §1b is decided if a
  precise number is needed.
- **Runtime lattice source is now `data/lattice_profiles/lattice_profiles.json`** (per-level
  `level_counts`); `annotate_lattice_counts.py` re-annotates arms `aset` from it without
  re-detecting.

## Reproduce the finding

```bash
# clinical env/arms re-annotated against lattice_profiles.json live in the session scratchpad;
# rebuild if gone: cp data/task_arms_full.json <arms>; \
#   PYTHONPATH=src .venv/bin/python scripts/annotate_lattice_counts.py --artifact <arms>; \
#   build_ranker_env.py --arms <arms> --out <env> --corpora clinical --n-docs 300 --skip-probes
PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/reader_parallelism_smoke.py \
  --env <clinical env> --arms <clinical arms> --probes data/probes_validated.json \
  --corpus clinical --n-docs 3 --g 6 --workers 6
# reports: correctness (serial vs parallel recall diff), STAGE (out_p vs out_final), PERF speedup
```
GPU: one AMD iGPU, one process at a time — check `rocm-smi --showpidgpus` and confirm before any
run (shared box). Reward needs `INFERDPT_LLM_CACHE` set.

## Suggested skills

- **superpowers:brainstorming** — before designing the non-determinism-tolerant sampler; this is
  a design fork (two-phase reward vs server flags vs noise-tolerant selection vs reframe), not a
  clear-spec implementation.
- **superpowers:writing-plans** — once the approach is chosen, to sequence the change + the
  mandatory support-scan re-run.
- **diagnose / diagnosing-bugs** — if isolating whether the *reader* (not just gen) is also
  non-deterministic under concurrency (direction 1's open question).
