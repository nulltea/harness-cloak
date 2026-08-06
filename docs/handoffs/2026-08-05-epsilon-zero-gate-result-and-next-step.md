---
type: handoff
status: current
created: 2026-08-05
updated: 2026-08-05
tags: [rl, ranker-v2, lexicographic, exact-ties, gate, codex-review, handoff]
companion: [docs/plans/2026-08-04-epsilon-zero-lexicographic-gate.md,
            research-wiki/experiments/2026-08-04-epsilon-zero-lexicographic-gate.md,
            docs/research/tie-ownership-root-cause-and-solution-space.md,
            docs/specs/RL/ties-by-design.md]
---

# Handoff — epsilon-zero lexicographic gate ran; Sol High recommends a constrained residual actor

## Where the work stands

Codex Sol High's plan [`docs/plans/2026-08-04-epsilon-zero-lexicographic-gate.md`](../plans/2026-08-04-epsilon-zero-lexicographic-gate.md) is implemented through Task 7 step 3. Tasks 1–5 are committed (`4cada47` pre-registration, `6b8940d` selector core, `f67ca48` candidate corpus, `c66a6d6` adjudication, `f81bf7c` comparators). Task 6's preflight is recorded in the experiment page.

**Uncommitted in the working tree:** the post-review fix wave to `scripts/spikes/epsilon_zero_lexicographic_gate.py` and `src/cloak/tests/test_epsilon_zero_lexicographic_gate.py`, plus the preflight section added to `research-wiki/experiments/2026-08-04-epsilon-zero-lexicographic-gate.md`. Nothing else in this session touched tracked files — `scripts/train_interactive_ranker.py` and the other dirty paths predate it (shared checkout; check `git diff --cached --name-only` before any commit).

**Canonical result:** `results/ranker_v2/architecture/epsilon-zero-lexicographic-gate.json` (gitignored), sha256 `1b2886765571e3b10fb447f9180e8f1592b080edf842e6cfc7eb2f43d867e833`, 63 per-document records. Two independent runs are byte-identical (`cmp` clean), the plan's `jq` contract check passes, and all four pre-registered input hashes are now enforced before evaluation. 42 gate tests pass; 207 across the four focused modules.

## Measured result

**Verdict `insufficient-primary-support`.** 59 primary documents, all support-complete (4 anchors each — the five walks deduplicate to four — 0 pin mismatches, 0 illegal, 0 incomplete, 0 parity failures, 9,295 validated candidates of 9,339 rows). Exact-optimal-set sizes: **48 singletons, 11 of size 2**. Exactly **one** primary document has positive gain: `aci/D2N019`, `G_d = 0.6259`, all 5 of its 5 decisions changed at bit-identical exact utility. Mean `G_d` 0.0106, median 0, bootstrap 95% one-sided lower bound 0.0. Campaign four: no positive gain.

Comparators after the softcap fix (report-only, `detached-s47` / `coupled-s47`): 19 / 18 cache hits of 126 greedy vectors; utility relation to the slate optimum splits 4 below, 13–14 equal, **1 above**; strictly-below fraction 21.1% / 22.2%; of the utility-feasible hits, 57.1% / 61.5% miss the max-count member; document-level, 3 documents show utility loss and 6 miss max count.

## Codex Sol High's last report (session `019f8fa3-5e69-7c72-bd5f-1f1ea12eb7b5`)

Full text: `codex-analysis-out.log` in this session's scratchpad (**ephemeral — /tmp**); the session history itself is the durable copy. Brief sent: `codex-analysis-brief.md` (same scratchpad). Substance below.

### Review findings

| # | severity | finding | state |
|---|---|---|---|
| 1 | P1 | Comparator replay omitted `utility_logit_softcap 25` from `training_config`, so the replayed forward function was not the archived policy's | **fixed** — options now reconstructed through the trainer's own `_apply_controller_options`, and the rebuilt `controller_transform` is asserted equal to the checkpoint's `semantic_contract` pin (`log1p-over-log1p-max-v1+softcap25+gain-evidence`) |
| 2 | P1 | "Below `U*`" conflated below / equal / above; expanded-cache comparator vectors can exceed the four-anchor optimum | **fixed** — three-way `utility_relation_to_exact_optimum`, separate below/above fractions, document-level counts alongside vector-level |
| 3 | P2 | Frozen hashes were recorded but never enforced | **fixed** — `--expect-sha256 NAME=HEX`; mismatch returns verdict `INVALID` before any evaluation (this also made the previously dead `INVALID` enum path live and tested) |
| 4 | minor | `invalid_reasons` never populated | **fixed** by 3 |
| 5 | minor | `unknown_document_excluded` counts rows not documents; count score reported as "privacy" | **fixed** — renamed `rows_for_non_retained_documents`; report keys now `profile_count_score` / `profile_count_key` / `exact_optimal_count_*` / `count_gap_to_lexicographic` / `chooses_count_max_inside_exact_set`. Plan-pinned dataclass field names in `src/cloak/ranker/lexicographic.py` were left alone |

All six declared deviations from the plan (extra CLI args, `count_state` keyword, per-document count-score memoisation, comparator scoring inside `run_gate`, fixture-specific test numbers, no worktree) were accepted.

### Interpretation, including two corrections to claims made earlier in the session

- The formal verdict is correct **for the frozen four-vector slate**; one positive document of 59 cannot establish a positive corpus mean.
- What the run measured is **incremental gain over BC-nearest tie-breaking**, not whether exact ties contain count variation. All 11 tied primary documents had count spread; BC-nearest already selected the max-count member in 10 of them.
- **Correction:** "cached pool size and tie-set size track each other exactly" is false. Log-count correlation is 0.56 on primary documents and **−0.24** on the campaign four.
- **Correction:** this gate does **not** establish that the additive controller fails to own ties — replay was wrong at the time, only 19/126 vectors hit cache, and the statistic pools profiles rather than documents.
- `aci/D2N019` is a valid existence proof within the pinned scorer and slate.
- The expanded-cache diagnostic justifies "a richer slate is worth one controlled attempt" but cannot estimate population prevalence, because its candidates were policy-adaptively sampled.

### Recommended next step — positive-λ constrained residual actor

`z = u_θ` at λ=0; `z = u_θ + r_φ(s,a)` at λ>0. Train `r_φ` primal-dual: maximise `E[P]` subject to `E[U_λ0 − U_λ>0] ≤ 0`. Count and measured-utility gradients reach the **same** residual logits; count still never enters the λ-blind tower; λ=0 stays an exact identity branch; per-document duals are training state, not inference calibration. Deletes from this arm: `alpha`, gain scalar, tie hinge, cycle projection, softcap, sensitivity regulariser, and all 0.044 tie labels. Codex frames this as the real gradient-path unification — one constrained positive-λ actor replacing three competing controller learners.

Experiment shape: split the 59 primary documents by frozen document hash, 47 train / 12 held out; two regimes only (λ=0, positive); four cycles, four rollouts per group, counterfactual budget two per group. Worst case `47×2×4×4 + 47×2×2×4 + 12×2×4 = 2,352` complete reward evaluations, 4–8 GPU hours after ~2–4 engineering days.

Pass criteria: count gradient reaches `r_φ` and exactly zero count gradient reaches `u_θ`; λ=0 logits bit-identical; every held-out greedy positive-λ vector has utility key ≥ its λ=0 key; positive one-sided document-bootstrap lower bound on held-out mean count gain; greedy separation retained across the final three cycles.

Failure readings, preregistered: utility violations → constrained optimisation cannot protect the primary objective, go to anchored utility estimation; constraints hold but zero count gain → representation or opportunity insufficient, then run the richer slate; train passes, held-out fails → document breadth is binding.

### Runners-up, in Codex's order

2. Anchored utility plus ranking — structurally sound, larger, still needs a deployment set-construction rule; use if constrained residual training violates utility.
3. **32-vector Sobol slate** — resolves oracle opportunity, not training. Construction: keep the four deduplicated anchors, add 28 vectors from a 32-point Sobol sequence, ordering each decision's legal menu by frozen count score → authored level → mode → action ID, mapping the Sobol coordinate to a legal-menu index while walking injectivity sequentially, continuing through later Sobol points on duplicates until 32 unique vectors or 256 attempts. Reads only document ID, environment, legal menus, authored order, frozen counts. Cost ≤ 28 new roundtrips per document = **1,652 remote generations + 1,652 batched reader jobs**, 3–8 hours.
4. `SCHEMA_NOTE` wiring — likely valuable, invalidates every cache row; do after the production task contract is decided.
5. Local tie floor / standalone evidence breadth / deployment-time probing — instrument work or inference cost; none repairs training now.

## Next steps

1. **Blocking on Timo:** which of Codex's options gets the next reward/GPU budget. Nothing below item 2 should start before that.
2. Finish the plan's paperwork, in Codex's required framing: Task 7 steps 4–5 (fill Results as "four-anchor slate yielded one primary incremental count gain; corpus effect unsupported" — **do not** write that exact ties are rare, and never "privacy improved"; hash; commit) and Task 8 (correct the `0.044` semantics in [`docs/specs/RL/ties-by-design.md`](../specs/RL/ties-by-design.md) regardless of verdict — distinguish measurement uncertainty, statistical equivalence, and user policy budget, and state that this gate uses none of them; append the decision-log entry as **slate-limited / inconclusive**, not adoption).
3. Do **not** run the attacker leg. No privacy claim is available from count scores.
4. If option 1 is chosen, write the training record first (`research-wiki/training/2026-08-XX-RL-ranker-v16-<slug>.md`, spec before run) and send the spec to Sol High before implementing — reward-side designs go to Codex first.

## Gotchas worth not rediscovering

- `codex exec resume` rejects a top-level `-s`; the working form is `codex exec -s read-only resume <SESSION_ID> -c model_reasoning_effort=high "<prompt>"`. Effort must be re-passed on every resume or it silently drops to medium.
- The Bash tool clamps `timeout` to 600 s, and a high-effort Codex analysis on this brief took ~950 s. Run it with `run_in_background`, not a long timeout. A run killed mid-flight leaves a prompt-only log and a duplicate prompt in the session history — this brief reached the session two or three times.
- A wrapper subagent that exits takes its Codex child with it. Prefer a direct background Bash call over an Agent wrapper for long Codex work.
- `utility_binding(...)["utility_weight_denominator"]` (policy mass) is **not** `artifact["documents"][doc]["utility_weight_denominator"]` (1.0 for most documents); `document_utility` and `_partitions` use the former. Comparing the wrong one suggests a nonexistent parity bug — 58 of 67 documents differ.
- `build_anchor_trajectories` produces **four** unique vectors per real document, not five.

## Suggested skills

- `superpowers:executing-plans` — for the Task 7–8 remainder; the plan is written and step-scoped.
- `superpowers:writing-plans` then `superpowers:brainstorming` — if the constrained residual actor is approved; it needs a spec before code.
- `codex:rescue` or a direct `codex exec -s read-only resume 019f8fa3-…` — reward-side design review before implementation, and the standing implement-then-cross-review split.
- `experiment-audit` — before any result claim from the next run.
- `superpowers:verification-before-completion` — evidence before assertions; this session's comparator numbers were wrong once already.
