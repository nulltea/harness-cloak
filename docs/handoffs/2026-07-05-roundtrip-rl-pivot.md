---
type: handoff
status: current
created: 2026-07-05
updated: 2026-07-05
tags: [round-trip-rl, reward, floors, ranker, pivot, known-issues]
companion: ../specs/RL/surrogate-ranker-infiller.md
---

# Handoff: pivot to round-trip RL (task-execution reward)

> **The round-trip support scan is the GATE for RL training** (user-mandated,
> 2026-07-05): in the round-trip environment the old constructed-arms gate is replaced by
> this much cheaper check (~30 min, mostly cacheable proxy calls) — no RL training run
> starts until it shows the realized reward can respond to the action space. It gates
> *training only*: spec/plan/lattice work proceeds freely around it. Rationale: both
> stage-1 NULLs came from training against a reward without support; this is the
> pre-flight that prevents a third.

**Decision being handed off (made 2026-07-05, user-approved): abandon local surrogate
utility rewards; train against the realized round-trip signal.** Three local surrogates
failed construct-validity in a row — u_qa (flat above bare findability; inversion
invariance), the mixed A+u_qa reward (privacy term's optimum was degenerate placeholder
collapse, KL leash forced to pin the policy), and u_gold (likelihood measures *steering*,
not retained information: cross-span "hypernym soup" drives the scorer below the
all-placeholder anchor on 13/20 docs). Every failure is documented with measurements; the
"remote" model is the free local proxy, so surrogate RL's only advantage was ~5–10× wall
time — no longer worth the validation tax.

Read first: spec third revision `docs/specs/RL/surrogate-ranker-infiller.md` (architecture
that SURVIVES the pivot: floors-only privacy, utility-only reward, floor grid = Pareto,
honesty boundaries); training records
`research-wiki/training/2026-07-04-RL-ranker-v1-stage1-bandit.md` and
`2026-07-05-RL-ranker-v2-stage1-floor-env.md` (the two NULLs + landscape addendum);
`review-stage/AUTO_REVIEW.md` (external review trail); u_gold post-mortem evidence in
`results/u_gold_landscape.json` + `.superpowers/sdd/task-2-report.md` (branch scratch).

## Repo state

- Branch `u-gold-reward` (current): u_gold scorer (`src/cloak/train/reward.py` — KEEP as
  diagnostic; reviewer-verified batching/anti-leak), landscape sanity spike + verified
  FAIL, spec third revision, records renamed to the v-schema. Plan
  `docs/plans/2026-07-05-u-gold-reward.md` Tasks 3–5 are CANCELLED by the pivot.
- `main`: floor migration merged (structural lattice risk, count floors, keep-original,
  floor-randomized trainer). Local main + branch are ahead of origin — nothing pushed.
- User's own detector work (FT-detector v4 `base-genfirst` run + records) is in flight in
  another session — do not touch `research-wiki/training/*detector*`,
  `scripts/build_pii_span_dataset.py`, `.gitignore`, or `data/models/`.
- GPU: one process at a time; the user's detector training may hold it — check
  `pgrep -af train_pii` before any GPU work; CPU prefix fallback works for smokes.

## Known issues that are NOT surrogate-RL-specific (they survive the pivot)

Environment / privacy mask:
1. **Lattice quality — absurd WordNet fills.** "female" → "an organism" (measured −12.8
   nats on its fact vs −3.4 under a placeholder). Passed the NLI gate via its word-sense
   ceiling (`docs/issues/rule-lattice-nli-gate-bypass.md`). These fills would mislead the
   REAL remote model too — direct realized-utility harm. No issue file yet for the general
   phenomenon; worth an audit spike (perplexity-screen all lattice fills vs originals).
2. **Famous-context priors.** Structural counts can't see world-knowledge pinning
   (measured: "LJM2" recovered from kept context). Escalation ladder pre-registered:
   population-weighted counts → per-surface overrides; eval attacker adjudicates.
3. **Floor calibration is thin.** Several count-vs-attacker cells n ≤ 3; QUANTITY floor
   rests on one item; DATETIME's 10–100 band non-monotone; the fuzzy hit@5 label inflates
   measured leakiness of lexically-close coarse fills (dates, ranges). Rule + caveats in
   spec §2-0b/§4.1.
4. **GeoNames reverse-count inflation** ≤2× (name/asciiname double-count; within-type
   consistent).
5. **Open-vocabulary MISC/OTHER have no count universe** — strict parse fails closed;
   default-deny 100; offline LM count-estimation is the planned build-time remedy.
6. **Per-span risk is blind to joint/relational leakage** until the E2 document head.
7. **Waiver-region coverage**: keep-vs-level trades are only trained in floor-grid points
   that include a waiver — the declared grid must contain ≥ 1 waiver-bearing config.

Detection (the privacy ceiling):
8. **Sibling-mention leak** (`docs/issues/detection-sibling-mention-leak.md`): undetected
   cleartext siblings of substituted spans are the dominant attacker channel; no
   downstream training closes it.
9. **Detection nondeterminism** across processes → everything runs off the frozen arms
   artifact (`data/task_arms_tau0.02.json`, annotate-in-place only, never re-detect).
10. **Detector type noise**: measured examples — "Morphine" typed as a direct identifier,
    "] james" tokenization garbage. Feeds bad spans into every stage.

Pipeline / environment mechanics:
11. **Echo vs absorption** (`docs/issues/remote-llm-echo-absorption.md`): absorption
    dominates for prose fills (82–95% leave no trace in out_p); extraction cannot win back
    absorbed content. Under round-trip RL this is no longer an unpriced channel — it IS
    part of the reward — but it shapes what utility can possibly be recovered.
12. **Static floor-walk teacher is jointly non-injective at high floors** (accepted:
    rollouts are dynamically masked; collision counts reported). Matters for any BC init.
13. **Probe supply is thin and clinical-heavy**: ~4.6 train probes/doc over 23 trainable
    docs (12 clinical / 8 enron / 3 aeslc) → realized fact recall is quantized in ~0.2
    steps. This was v1's noise floor and it applies UNCHANGED to round-trip reward.
    Expansion options queued in spec §4.2.
14. **Extractor pinning**: reward and eval call the deployed `invert()`; any extractor
    change re-gates and invalidates trained policies.
15. **E1 infiller is design-not-build-ready** (spec §3.4 build contract: grammar
    artifacts, slot vocabularies, decode constants, NLI-failure determinism).

RL-generic lessons (carry into round-trip design):
16. **Support before training** (learned twice): measure whether the reward can respond to
    the action space before any RL epochs — the round-trip version of
    `scripts/spikes/probe_flip_scan.py` is mandatory pre-flight.
17. **Leash arithmetic**: KL cost per span (−log p₀ of the target action, × kl_coef) must
    be compared against measured per-swap reward gains before choosing kl_coef; kl 0.01 is
    the current spec default, justified only for a non-degenerate reward.
18. **G ≥ 2 enforced** (NaN advantage guard); group advantage compares within one floor
    sample only; read-outs greedy at fixed floors, never averaged across floors.

Process gotchas (operational, this environment):
19. Codex MCP sandbox cannot read local files and silently falls back to the stale GitHub
    copy — always INLINE documents + evidence in reviewer prompts.
20. Subagent spawns occasionally return garbled zero-tool replies — resume via SendMessage
    with "proceed"; background children die when a subagent's turn ends — subagents must
    run long jobs foreground (chunked) and never end a turn to "wait".

## Next steps: round-trip RL, reward = task execution

The reward becomes the realized quantity itself: **fact recall on out_final over
train-split probes** (out_final = invert(RemoteLLM(task_prompt(doc_p)))). "Task execution"
focus: the remote model performs the actual task (clinical note → assessment/plan, email →
summary per `cloak/tasks.py` TASK_TEMPLATE); utility is what survives the full round trip
of task output — not doc_p-side answerability. Probes stay the measurement (whole-output
similarity is measured noise, spec §4.2); if task-level metrics beyond restated-fact
probes are wanted, that is a NEW measurement decision requiring its own validation.

Ordered:
1. **Spec fourth revision** (§5: reward = realized fact recall, u_gold demoted to
   diagnostic with its post-mortem cited; §6: gate replaced by the round-trip support
   scan; add: provider-coupling caveat + the second-remote-model eval arm as the guard,
   temp-0 + pinned proxy model + cache policy, wall-time budget ~2 h/run with pmap
   concurrency). Keep floors/grid/honesty sections untouched.
2. **Round-trip support scan — THE training gate** (new spike, pattern of
   `probe_flip_scan.py`): from the floor-walk baseline, ~100 single-action
   counterfactuals → generate out_p via proxy (workers=8, cached) → invert → per-probe
   realized fact recall deltas. Report: flippable probes, actual flips BOTH directions,
   per-swap delta magnitudes vs quantization. ~30 min — the cheap replacement for the old
   constructed-arms gate; **no RL run starts until it passes**. A support desert here is a
   real finding about the environment (walk near-optimal in realized terms), reported as
   such, not worked around.
3. If support exists: **RL-ranker v3 record** (spec-then-results, v-schema name e.g.
   `2026-07-0X-RL-ranker-v3-roundtrip-reward.md`) + trainer switch (rollout_reward calls
   the round trip; batch rollouts across docs through pmap; cache identical doc_p) + the
   first-smoke movement milestone, then the fixed-floor run, then randomized (§5.4
   protocols).
4. Independent of 1–3: **lattice-hygiene audit spike** (issue file + perplexity screen of
   all lattice fills; prune or NLI-tighten the absurd ones). Cheap, helps every branch —
   the "an organism" class of fills damages realized utility directly.
5. Branch handling: commit this handoff on `u-gold-reward`, merge to main (Tasks 1–2 are
   review-clean keepers; plan Tasks 3–5 cancelled), push only when the user says so.

Suggested skills for the next session: superpowers:subagent-driven-development (executing
the above as tasks), superpowers:writing-plans (the v4-spec + support-scan plan),
auto-review-loop (spec fourth revision, INLINE the doc per gotcha 19).
