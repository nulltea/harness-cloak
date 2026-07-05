---
type: plan
status: current
created: 2026-07-05
updated: 2026-07-05
tags: [reward, u-gold, gold-conditional, gate, sanity-check, ranker]
companion: ../specs/RL/surrogate-ranker-infiller.md
---

# u_gold Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> The normative contract for everything here is the spec's §5.1 (scorer + anti-leak +
> edge cases), §5.3 (leash 0.01, first-smoke milestone), §6 (gate report format + pass
> rule), and §4.2 (fact masks from probe machinery). Tasks below give interfaces and
> verification; exact semantics live in the spec — read the cited sections, not the whole
> spec.

**Goal:** Implement the utility-only u_gold reward (per-fact anti-leak gold-conditional
likelihood), validate it (landscape sanity check + gate), and switch the trainer — leaving
the stage-1 rerun unblocked.

**Global constraints** (repo rules, binding on every task): host `.venv` on GPU, one GPU
process at a time (NEVER kill an existing run; CPU fallback via
`HIP_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES=""` is acceptable for smokes), long runs
`python -u`, `INFERDPT_LLM_CACHE=data/llm_cache`, `__main__` assert self-checks (no
pytest), never re-detect / never rebuild the arms artifact, commit per task with trailer
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: u_gold scorer in `src/cloak/train/reward.py`

**Files:** Modify `src/cloak/train/reward.py` (add; do not remove u_qa — it is a
diagnostic now).

**Interfaces produced:**
- `GOLD_SCORER_MODEL: str` — pinned model id. Selection rule: try
  `Qwen/Qwen2.5-1.5B-Instruct` from HF cache/network; if unavailable (no network), fall
  back to `EleutherAI/pythia-410m` (already local) and say so in the report. Whichever
  loads is THE pin — record it in the module constant and the report.
- `gold_fact_spans(doc_gold: str, R_walk: list[dict]) -> list[dict]` — fact spans: unique
  substituted surfaces restated in gold, located via the same canonicalized exact→fuzzy≥85
  matcher `restated_probes` uses (reuse/refactor it — do NOT duplicate the matching
  logic). Each item: `{"surface", "gold_start", "gold_end"}` char offsets into gold.
- `u_gold(doc_p: str, R: list[dict], gold: str, facts: list[dict], anchors=None) ->
  tuple[float | None, dict]` — per-fact anti-leak scoring per spec §5.1: for each fact j,
  build `context_j = TASK_TEMPLATE-prompt(doc_p) + gold[:gold_start_j]` where in the gold
  prefix every OTHER fact surface is replaced via `generalize_text(_, R)` and earlier
  mentions of fact j's own surface are replaced by fact j's replacement in R (placeholder
  token or fill); score = mean token logprob of `gold[gold_start_j:gold_end_j]` under the
  scorer, teacher-forced (one batched forward across facts). Returns
  `(clipped normalized score | None if excluded, details)`. Edge-case rule (spec §5.1):
  empty facts or `|U_hi − U_lo| < 0.05` → `(None, {"excluded": reason})`.
- `u_gold_anchors(doc, facts, art_entry) -> {"U_hi", "U_lo"}` — U_hi on doc_orig (empty
  R), U_lo on the all-placeholder assembly (build it the way `reward_gate.py`'s
  all_placeholder arm does).

**Verification (`__main__` self-check):** toy doc (reuse reward.py's Sarah/Oslo example):
assert (a) ordering `score(keep) > score(coarse fill) > score(all-placeholder)` on the
"34" fact (raw scores, before normalization); (b) `u_gold` normalized ∈ [0,1] and
doc_orig ≈ 1, all-placeholder ≈ 0; (c) anti-leak: duplicating the fact earlier in gold
does NOT raise the placeholder variant's score (the self-mention replacement rule works);
(d) empty-fact and tiny-anchor exclusion return None. GPU or CPU both fine for the toy.

Commit: `feat: u_gold gold-conditional fact-likelihood reward (per-fact anti-leak scorer)`.

---

### Task 2: u_gold landscape sanity check (spike)

**Files:** Create `scripts/spikes/u_gold_landscape.py`.

Per spec §5.1 pre-registration, on the 23 trainable docs (env + artifact, floor-walk
baseline with the trainer's dynamic collision rule — copy the baseline construction from
`scripts/spikes/reward_landscape_probe.py`): (1) per doc: `u_gold(floor-walk)` vs
`u_gold(all-placeholder)` — report per-doc win rate; (2) single-swap directionality: for
every level→level swap pair where one fill has strictly larger `aset` (coarser), does
u_gold drop when coarsening? Report the fraction of specificity-consistent swaps; (3) the
excluded-doc count (edge-case rule). PASS (spec): floor-walk ≥ all-placeholder on a clear
majority of docs AND a clear majority of swaps specificity-consistent — print PASS/FAIL
with the numbers; write `results/u_gold_landscape.json`.

Commit: `feat: u_gold landscape sanity check (pre-registered, blocks training like the gate)`.

---

### Task 3: gate switch + gate run

**Files:** Modify `scripts/reward_gate.py`.

- U column becomes u_gold (u_qa retained as a diagnostic column `U_qa`); Spearman
  `U~realized` computed on u_gold. Add the spec §6 report fields: per-corpus fact-token
  coverage (fact spans per doc mean/min, empty-mask docs), anchor-health exclusions,
  usable-doc count and fraction, and the sanity-check verdict (read
  `results/u_gold_landscape.json`). Pass rule printed: Spearman positive on every corpus
  AND usable ≥ 80% per corpus AND sanity PASS — name the failing clause if any.
- RUN the gate (cached round trips; scorer forwards are the only new compute). Report the
  full per-corpus table.

Commit: `feat: u_gold gate — report format per spec §6 + run`.

---

### Task 4: trainer switch

**Files:** Modify `scripts/train_ranker.py` (+ `src/cloak/train/reward.py` import only).

- `rollout_reward` returns `u_gold` (drop A/stage1_reward from the reward path;
  `p6s`/ph_rate stay as logged diagnostics). Precompute per-doc facts + anchors at doc
  load; docs excluded by the edge rule are dropped from training with a printed count.
- Remove the alpha sweep (`--alphas` deleted); one run = one floor config (`--floors` is
  the operating-point arg). Output tag becomes the floor-config hash/name; logs keep a
  `u_qa`-style diagnostic if cheap, else drop.
- `--kl-coef` default 0.05 → **0.01** (spec §5.3).
- **First-smoke milestone (spec §2 Phase 1)**: run the CPU smoke; it MUST show movement
  (KL > 0.01 at some epoch or greedy ≠ BC on ≥ 1 span) — report which. A motionless smoke
  is BLOCKED status, not a soft warning.

Commit: `feat: trainer on u_gold utility-only reward (alpha retired, kl 0.01)`.

---

### Task 5: status + records

**Files:** Modify `docs/specs/RL/surrogate-ranker-infiller.md` (§6 status line only — set
to PASSED/FAILED with the measured numbers and date), append gate + sanity results to
`review-stage/AUTO_REVIEW.md` (loop-2 addendum), update `.superpowers/sdd/progress.md`.

Commit: `docs: u_gold gate + sanity results, spec status`.
