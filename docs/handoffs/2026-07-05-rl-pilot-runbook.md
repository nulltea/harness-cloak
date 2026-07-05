---
type: handoff
status: current
created: 2026-07-05
updated: 2026-07-05
tags: [rl-pilot, runbook, round-trip-reward, probes, gates, extractor, handoff]
companion: [../specs/RL/roundtrip-ranker-infiller.md,
            ../plans/2026-07-05-roundtrip-ranker-pilot.md,
            ../plans/2026-07-05-roundtrip-rl-strategy.md]
---

# Handoff: run the RL ranker pilot; next — extractor/invert() improvements

Everything is implemented, reviewed (per-task + 8-angle verified + codex loop closed at 7/10
"almost" + final whole-branch READY-TO-MERGE), merged to `main`, and calibrated. What remains
is executing the tail of the runbook and, after the pilot, the extractor upgrade. Read first:
spec `docs/specs/RL/roundtrip-ranker-infiller.md` (pinned components table is the single
source of truth), pilot plan `docs/plans/2026-07-05-roundtrip-ranker-pilot.md` (Global
Constraints + runbook), progress ledger `.superpowers/sdd/progress.md`, review trail
`review-stage/AUTO_REVIEW.md`.

## Pins as of HEAD 6eb9f92 (change any ⇒ re-gate; enforced mechanically at 3 places)

- **Reward env**: `gemma 4 (E4B)`, non-thinking, temp 0, **max_tokens 1024** (512 truncated
  real clinical notes — measured), `RT_BASE_URL=http://localhost:8060/v1`; cache key includes
  base_url. All in `src/cloak/train/roundtrip.py`.
- **Probe teacher**: `Qwen3.6-35B-A3B` non-thinking, **prompt v3** (`PROMPT_VERSION=3`,
  full-gold + uniqueness + grader-aware + 8-type hints; measured +18% kept facts), 3
  questions/fact, **per-fact-max** realized recall. Teacher-tagged cache auto-retires
  other-teacher/pv entries.
- **Second remote (eval arm)**: LFM2.5-8B-A1B, thinking (it cannot not-think — measured;
  `results/thinking_mode_probe.json`).
- **Extractor**: rule exact/fuzzy-90, `cloak/extract.py` — audited, keep (see Next steps).
- All durable gen paths (InferDPT pipeline, latticecloak harnesses, dp_sweep, diagnostics)
  default to the gemma pin since 6eb9f92; historical Qwen-generated results are NOT
  comparable to new runs.

## Measured facts the next session should not re-derive

- Throughput: gemma **1,459 RT/h @ 6 workers** (measured at max_tokens 512 —
  `results/saturation_probe.log`; at 1024 expect ~½–⅔ of that; re-run
  `scripts/spikes/lfm_saturation_probe.py` if the estimate matters). llama-swap: gemma
  `-np 6` (parallel), **Qwen `-np 1`** (teacher calls serialize server-side), Qwen ttl 600s.
- Probe-yield history (12 clinical docs, fixed anchors/reader):
  1.17 → 1.33 (budget fix) → 2.17 (Qwen teacher) → 3.33 facts/doc (3-q multiplicity);
  v3 prompt A/B: 3.92 (`results/teacher_ab_p3-nothink.json`). Thinking teacher: abandoned,
  3–5× slower. Kept-probe quality is clamped by validation (~0.83–0.85 reader-F1) — only
  yield moves.
- **enron + aeslc are OUT of the pilot reward corpus** (measured finding: gemma replies are
  13–16-token pleasantries; ceiling rejection 100%). Replacement: **lexsum** (Multi-LexSum,
  161 docs materialized, long→short case-summary task; `results/lexsum_restatement.json`).
- Extractor audit (`results/extractor_miss_audit.json`, 75 level fills): exact 20% +
  fuzzy90 16% invert today; band60-90 44% but mostly spurious (cos 0.02–0.15 on generic
  fills); TRUE recoverable paraphrase ≈ 5–8%; absent 20%. Placeholder echo only 23.9% vs
  ~36% level — **the feared pro-placeholder reward bias is inverted; specificity survives.**

## State at handoff (in flight)

- `data/task_arms_pilot.json`: BUILT + count-annotated (80 clinical + 80 lexsum, detection
  done, 12 min). Frozen artifacts (`task_arms_tau0.02.json`, `ranker_env.json`) untouched.
- `data/ranker_env_pilot.json`: **build was RUNNING at handoff** (clinical phase done: 825
  spans, 63/80 probe-bearing, 510 v3 teacher calls cached; lexsum phase in progress).
  Check `results/build_env_pilot.log`; if it died, rerun:
  `PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_ranker_env.py --arms
  data/task_arms_pilot.json --out data/ranker_env_pilot.json --corpora clinical,lexsum
  --n-docs 80` (idempotent; teacher calls cached).
- Pilot target (user decision): **60 trainable docs, balanced 30 clinical / 30 lexsum**;
  80/80 candidates sized for ~42% yield. Top up `--n-docs` per corpus only if a corpus
  under-yields.

## Runbook — remaining steps, exact commands (all `-u`, logged to results/)

```bash
# 1. Probe build 5 on the pilot env (teacher mostly cached; anchors ≈ 320 RT):
INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python -u \
  scripts/build_probes.py --env data/ranker_env_pilot.json --arms data/task_arms_pilot.json \
  --corpora clinical,lexsum --n-docs 80 --workers 6 > results/build_probes_pilot5.log 2>&1
# GATE: results/probe_health.json — need >=30 docs/corpus with >=3 TRAIN facts (kept_facts
# per doc >= 4). Under-yield => raise --n-docs for that corpus and rerun (incremental).

# 2. Support scan (THE training gate; trainer verifies its meta itself):
INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python -u \
  scripts/spikes/roundtrip_support_scan.py --env data/ranker_env_pilot.json \
  --arms data/task_arms_pilot.json --probes data/probes_validated.json --n-docs 80 \
  --max-swaps 150 --workers 6 > results/support_scan_pilot.log 2>&1
# PASS requires quantization-exceeding moves BOTH directions. A desert = a finding, report it.

# 3. First smoke (movement canary):
INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python -u \
  scripts/train_ranker.py --reward roundtrip --env data/ranker_env_pilot.json \
  --arms data/task_arms_pilot.json --n-docs 80 --smoke > results/rt_smoke.log 2>&1

# 4. BEFORE the full run: write the training record (spec-then-results, v-schema):
#    research-wiki/training/2026-07-05-RL-ranker-v3-roundtrip-pilot.md
#    AND pass the perf gate (/auto-review-loop vs scripts/harness/perf_gate.md).
#    Known perf debt for that gate: QA reader is unbatched+uncached (~90% of wall
#    historically); RLOO refiner + cf paths batch per doc (exit_round shows the global
#    pattern). Fix or budget for it.

# 5. Pilot run — ExIt first, then refiner, FIXED floors (--randomize-floors hard-errors in
#    roundtrip mode; it is BC-only — do not "fix" this by removing the guard):
INFERDPT_LLM_CACHE=data/llm_cache PYTHONPATH=src:scripts .venv/bin/python -u \
  scripts/train_ranker.py --reward roundtrip --env data/ranker_env_pilot.json \
  --arms data/task_arms_pilot.json --n-docs 80 --exit-rounds 4 --exit-epochs 10 \
  --epochs 5 --G 12 --cf-frac 0.25 --policy encoder --rt-workers 6 \
  > results/rt_pilot.log 2>&1
# Read-outs: greedy_final has r_train AND r_heldout (out-of-sample); report per corpus,
# never averaged. Pre-registered null (ExIt+RLOO both flat at floor-walk) is a legitimate
# finding — see spec §pre-registered outcomes.
```

Trainer refuses to run without a PASS scan whose meta matches (model/base_url/probes path);
`--force-ungated` exists but prints a loud warning — don't.

## Next steps after the pilot: extractor / invert() improvements

The audit says the rule extractor is NOT the third-NULL channel (bias favors specificity),
so it was consciously kept for the pilot. The upgrade path, in order of value:
1. **Semantic-window matching** for the 5–8% recoverable band: MiniLM cosine over candidate
   windows located by fuzzy alignment in the 60–90 band, accept on high similarity +
   NLI/type sanity. `cloak/extract.py`'s own docstring pre-registers exactly this. Guard the
   extractor-gaming channel: monitor exact-vs-fuzzy(+semantic) recall gap per checkpoint.
2. **Learned reconstructor** (denoise/edit model over out_p + R) — the project-goal
   replacement; needs its own training-data story (the audit's band+absent rows are the
   seed corpus). Stage-2-adjacent build.
3. **Re-gate discipline**: ANY extractor change invalidates cached anchors, validated
   probes, scan verdicts, and trained policies together (pinning = cache coherence). The
   cheap moment is before a probe build, not after; budget one full probe+scan rebuild.

## Gotchas (operational, this environment)

- Subagents: ALWAYS `model: "opus"` (user rule, memory). Garbled zero-tool first replies
  happen — resume via SendMessage "proceed".
- codex reviews: MCP sandbox can't read local files; `codex exec` bwrap fails on this box —
  user approved `--sandbox danger-full-access` on a disposable snapshot worktree, with
  read-only conduct in the prompt. ALWAYS `< /dev/null` (stdin never closes otherwise —
  cost us 40 min once).
- `pkill -f` self-matches its own wrapper cmdline — filter `pgrep -af X | awk '$2 ~
  /python$/'` instead.
- `results/`, `corpora/`, `data/` artifacts are gitignored; findings live in the docs.
- The user's detector work happens in parallel sessions — never touch detector files;
  check `pgrep -af train_pii` before GPU-heavy local runs.
- One measurement at a time on the proxy (saturation numbers are only valid uncontended).

Suggested skills for the next session: superpowers:subagent-driven-development (any new
implementation), auto-review-loop (the perf gate, and inline-or-exec per the codex gotcha),
superpowers:verification-before-completion (before declaring the pilot launched/passed).
