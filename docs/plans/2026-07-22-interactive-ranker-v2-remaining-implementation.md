---
type: plan
status: current
created: 2026-07-22
updated: 2026-07-22
tags: [rl, ranker, qa-builder, count-reward, structured-credit, implementation]
companion: [docs/specs/RL/interactive-ranker-v2.md,
            docs/specs/RL/interactive-ranker-v2-diagnostics.md,
            docs/specs/RL/interactive-ranker-v2-decision-log.md,
            docs/specs/qa-builder-v2.md,
            docs/plans/2026-07-22-core-rl-v2-qa-builder-alignment.md,
            docs/dev/logs/2026-07-22-rl-v2-implementation-inventory.md]
---

# Interactive Ranker v2 Remaining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. Use one implementation agent per task and the skill's
> review gates; do not dispatch parallel implementers against the same checkout.
>
> **Amended 2026-07-22 (Timo + Fable review):** (1) `codex/qa-builder-v2-clean` is now merged
> into `main` (`ca7f990`, main-wins on all evolved QA surfaces) — clean-branch reference code is
> retrievable via `git show codex/qa-builder-v2-clean:<path>`; (2) Task 2 no longer re-runs
> detection — the environment is rebuilt from the existing frozen arms artifact; (3) lattice
> profile repairs stop for Timo's explicit confirmation before promotion; (4) Tasks 14–15 are a
> deferred second phase requiring a separate green light; (5) implementation runs on GPT-5.6
> Sol High via `codex:rescue` with the coordinating Claude session as reviewer; (6) the
> counterfactual scheduler drops the "low-confidence links" input (v16 stores no confidence
> scalar).

**Goal:** Deliver one lambda-conditioned ranker that chooses KEEP, a truthful lattice level, or a
placeholder per stable policy decision using QA-builder-v2 whole-round-trip utility, exact
type-normalized count shaping, structured provisional credit, and bounded one-decision
counterfactual correction.

**Architecture:** Keep the current QA-builder/scorer implementation on `main` as the integration
base. Correct the frozen environment and utility-artifact contracts before writing optimization
code, then add focused modules for environment loading, count reward, conditional policy state,
utility measurement/cache, counterfactual scheduling, lambda selection, diagnostics, and hybrid
training. Reuse selected structured-credit and cache ideas from `codex/qa-builder-v2-clean` only
after repinning them to the current two-family assertion artifact; do not merge or cherry-pick that
branch wholesale.

**Tech Stack:** Python 3.12, PyTorch, Hugging Face Transformers, existing frozen ModernBERT
encoder, existing closed-box task/extractor/context-reader stack, JSON/JSONL content-addressed
artifacts, pytest.

## Global Constraints

- Work directly on `main` in `/home/timo/repos/agent-cloak` (Timo's direction): no new worktree;
  no other session touches this checkout. Check `git diff --cached --name-only` for staged
  strays before every commit.
- Implementation tasks run through the `codex:rescue` subagent on GPT-5.6 Sol High (high
  effort; a purely mechanical task may drop to medium). The coordinating Claude session (Fable)
  reviews every task diff against the task contract and review gates before the next task
  begins. LLM prompt strings (teacher/reader/attacker prompts, few-shot examples) are never
  delegated to codex — the coordinator authors and owns them.
- Never re-run span detection anywhere in this plan. The frozen detections in
  `results/qa_v2_aci_full/arms.json` are the single detection source; every environment
  rebuild derives from them.
- Use test-driven development: add the focused failing test, observe the intended failure, make
  the smallest production change, then run the focused suite before the task review gate.
- Do not install a new production dependency.
- Do not run an uncached external teacher, remote-task, extractor, reader, or attacker call without
  explicit user approval. Cache-only and local deterministic work proceeds without approval.
- QA and RL consume the same frozen occurrence/decision artifact; no training path redetects spans.
- The policy acts only on `ranker_selectable` decisions. Forced PERSON/CODE placeholders and other
  fixed rewrites remain in `doc_p` but never receive an action, log-probability, count score, or
  policy gradient.
- One action is selected per stable decision and applied to every mapped occurrence. Repeated
  occurrences do not multiply count reward. Preserve the formal-audit warning that repeated-context
  leakage remains unmodeled by this shaping signal.
- Context assertions score `doc_p`; delivered assertions score `out_final`. The utility vector is
  keyed only by stable assertion IDs and uses builder-authored weights with the fixed denominator.
- Missing context-family support does not renormalize utility and does not remove a policy decision.
- Linked occurrence sets are provisional routing evidence. Assertions with no policy-controllable
  dependency are residual/global credit; uncovered decisions receive complete-document fallback.
- Counterfactuals change exactly one policy decision across all mapped occurrences, rerun the full
  remote task, extraction, context scoring, and delivered scoring path, and substitute for the
  provisional loss at the same `1/G` weight.
- Count reward is experimental shaping, not measured privacy. Final comparisons use attacker
  outcomes at matched realized privacy and identical settings.
- The complete-count gate is fail-closed: 100% explicit admitted level counts, zero fallback
  gradient, zero nonmonotone profiles, and zero missing policy occurrence mappings.
- Run the smallest representative real-data vertical slice before full training. Passing unit tests
  or synthetic fixtures alone is not completion.

## Delivery Boundaries

The work is one dependency chain rather than independent subprojects:

1. **Contract and artifact alignment** — amend stale specifications, repair count provenance, and
   publish policy-scoped utility routing.
2. **Core RL components** — implement the v2 environment model, conditional sequential policy,
   count objective, structured utility, cache, and one-decision counterfactuals.
3. **Calibration and optimization** — build the utility-only trajectory pool, freeze the lambda
   menu and thresholds, then train with the hybrid loss.
4. **Validation and evidence** — pass the real-data vertical slice (Task 13, the end of this
   phase), then — after a separate green light — the full 67-document run and
   matched-realized-privacy evaluation (Tasks 14–15).

Do not begin a later boundary while a hard gate in an earlier boundary fails.

## Normative Coverage Matrix

| normative area | implementing tasks | decisive evidence |
|---|---:|---|
| frozen occurrence/decision environment | 1, 2, 4, 5 | policy/fixed partition and zero missing mappings |
| lambda-conditioned sequential policy | 5, 6, 12 | identity initialization, balanced exposure, responsive logits |
| type-normalized count shaping | 2, 3, 12 | complete-count PASS and exact legal-menu expectation |
| assertion-vector structured utility | 1, 4, 7, 8 | fixed-denominator linked/residual/fallback tests |
| utility-only BC and ExIt warm start | 9 | verified winners selected without count reward |
| contextual one-decision correction | 10 | full-round-trip pair and in-place `1/G` substitution |
| supported lambda selection | 11 | frozen pool, switch points, replay report, and menu manifest |
| diagnostic and threshold discipline | 11–13 | immutable spike and completed threshold manifest |
| hybrid RL optimization | 12, 13 | finite real-data update and component gradient report |
| realized-privacy evaluation | 14, 15 | pinned LLM attackers and matched-privacy adjudication |

## File Map

### Existing files to modify

- `docs/specs/RL/interactive-ranker-v2.md` — assertion-vector utility and policy-routing contract.
- `docs/specs/RL/interactive-ranker-v2-diagnostics.md` — live QA/count diagnostics and threshold
  manifest fields.
- `docs/specs/RL/training-task-env.md` — mark the old carrier reward contract stale for ranker v2.
- `docs/specs/qa-builder-v2.md` — policy decision, residual routing, and reader execution contract.
- `docs/specs/RL/interactive-ranker-v2-decision-log.md` — record the alignment decisions without
  reopening already approved forks.
- `src/cloak/train/qa_builder.py` — corrected utility artifact routing and batched runtime scoring.
- `src/cloak/train/ranker.py` — v2 action features and conditional policy head.
- `src/cloak/train/roundtrip.py` — complete utility-result execution and cache-facing interface.
- `src/cloak/train/utility_credit.py` — fixed-denominator linked/residual/fallback credit.
- `scripts/build_ranker_env.py` — emit the corrected count-complete frozen environment.
- `scripts/build_qa_utility_artifact.py` — emit policy-scoped utility artifacts directly.
- `scripts/train_ranker.py` — retain the predecessor CLI; expose only shared helpers needed by the
  new trainer and keep legacy behavior unchanged.

### New focused modules and scripts

- `src/cloak/train/ranker_environment.py` — immutable v2 environment loader and validation.
- `src/cloak/train/count_reward.py` — complete-count gate, type references, action scores, and exact
  expected count loss.
- `src/cloak/train/interactive_ranker.py` — trajectory sampling/replay and hybrid-loss assembly.
- `src/cloak/train/utility_cache.py` — append-only complete round-trip utility cache.
- `src/cloak/train/counterfactuals.py` — intervention construction, eligibility, pair loss, cache,
  and coverage/uncertainty scheduler.
- `src/cloak/train/lambda_menu.py` — offline calibration-pool frontier and supported-menu selector.
- `src/cloak/train/ranker_diagnostics.py` — spike measurements, threshold manifest, and run reports.
- `scripts/build_count_reward_state.py` — versioned count reward state and hard-gate report.
- `scripts/migrate_qa_utility_artifact.py` — deterministic v1-to-v2 policy-routing migration.
- `scripts/run_ranker_preflight.py` — frozen calibration, diagnostic spike, threshold, and menu
  workflow.
- `scripts/train_interactive_ranker.py` — thin production training CLI.
- `src/bench/ranker_policy.py` — benchmark adapter for a frozen conditional checkpoint and profile.
- `scripts/run_roundtrip_benchmark.py` — extend the existing benchmark CLI with explicit ranker
  artifact pins; retain one benchmark harness rather than create a second evaluator.

### New focused tests

- `src/cloak/tests/test_ranker_environment.py`
- `src/cloak/tests/test_count_reward.py`
- `src/cloak/tests/test_conditional_ranker.py`
- `src/cloak/tests/test_utility_cache.py`
- `src/cloak/tests/test_interactive_ranker.py`
- `src/cloak/tests/test_lambda_menu.py`
- `src/cloak/tests/test_ranker_diagnostics.py`
- `src/cloak/tests/test_bench_ranker_policy.py`
- `src/cloak/tests/test_bench_llm_privacy.py`

---

## Task 1: Pin the Live QA-to-RL Contract

**Files:**
- Modify: `docs/specs/RL/interactive-ranker-v2.md`
- Modify: `docs/specs/RL/interactive-ranker-v2-diagnostics.md`
- Modify: `docs/specs/RL/training-task-env.md`
- Modify: `docs/specs/qa-builder-v2.md`
- Modify: `docs/specs/RL/interactive-ranker-v2-decision-log.md`

**Produces:** One non-contradictory normative contract against which all later task reviews run.

- [ ] **Step 1: Replace the retired utility-vector vocabulary**

  In `interactive-ranker-v2.md`, define the complete round-trip vector as:

  ```text
  u_g = {assertion_id: score}
  U(g, Q) = sum_{q in Q} w_q u_g[q] / Z_d
  Z_d = utility_weight_denominator stored for document d
  ```

  State that `Z_d` is fixed even when a family is missing. Remove the old
  `{probe_id, decision_id, schema_id, global_id}` carrier wording and any implication that a subset
  is renormalized by its own weight sum.

- [ ] **Step 2: Define policy routing independently of measurement scope**

  Add these artifact fields to both the RL and QA specs:

  ```yaml
  document:
    policy_decision_ids: [decision_id]
    fixed_decision_ids: [decision_id]
    uncovered_policy_decision_ids: [decision_id]
    occurrence_to_decision: {occurrence_id: decision_id_or_null}
  assertion:
    occurrence_ids: [occurrence_id]
    policy_dependency_decision_ids: [decision_id]
    credit_routing: linked | residual
  ```

  Pin the rule: `credit_routing=linked` exactly when the assertion has at least one policy
  dependency; otherwise it is `residual`, including globally scoped assertions and assertions
  linked only to fixed rewrite decisions. A mixed fixed/policy hyperedge routes once to each unique
  policy dependency and never to a fixed decision.

- [ ] **Step 3: Correct utility partition and fallback equations**

  Replace `Q_global` with `Q_residual`, where residual means “not linked to a policy decision,” not
  merely `scope=global`. Preserve these equations:

  ```text
  U_link[g,j]     = U(g, {q : j in policy_dependency_decision_ids[q]})
  U_residual[g]   = U(g, {q : credit_routing[q] = residual})
  U_document[g]   = U(g, all accepted assertions for d)

  A_provisional[g,j] = A_link[g,j] + A_residual[g]  when j has linked assertions
  A_provisional[g,j] = A_document[g]                otherwise
  ```

  Explicitly prohibit adding `A_document` to a linked decision and prohibit including a linked
  assertion in `U_residual`.

- [ ] **Step 4: Correct reader batching claims**

  Amend both specs to distinguish one scorer submission from transport calls. Runtime scoring may
  flatten every context assertion from a rollout batch into one bounded work queue, but each
  assertion keeps its own pinned question, clause, excerpt, answer kind, and cache identity. Do not
  claim one wire/model generation per rollout unless a separately validated multi-question reader
  actually supplies that property. Preserve the existing reader semantics in this implementation.

- [ ] **Step 5: Mark predecessor reward text stale**

  Set the frontmatter of `training-task-env.md` to `status: stale`, add
  `superseded_by: docs/specs/RL/interactive-ranker-v2.md`, and explain that its carrier components
  remain historical inputs but are not the live ranker-v2 reward contract.

- [ ] **Step 6: Record the alignment decisions**

  Append one decision-log entry that records: assertion-ID vectors, fixed denominators,
  policy/fixed decision partition, residual routing, per-assertion reader semantics with batched
  scheduling, and selective porting from the clean branch. Include rejected alternatives:
  renormalizing present families, giving gradients to fixed decisions, treating fixed-only links as
  linked policy credit, and merging the divergent branch wholesale.

- [ ] **Step 7: Verify the normative docs**

  Run:

  ```bash
  rg -n "probe_id: score|decision_id: score|schema_id: score|Q_global|one batched reader request" \
    docs/specs/RL/interactive-ranker-v2.md \
    docs/specs/qa-builder-v2.md
  ```

  Expected: no stale normative occurrence. Historical quotations in the decision log are allowed
  only when labeled rejected or superseded.

- [ ] **Step 8: Commit the contract correction**

  ```bash
  git add docs/specs/RL/interactive-ranker-v2.md \
          docs/specs/RL/interactive-ranker-v2-diagnostics.md \
          docs/specs/RL/training-task-env.md \
          docs/specs/qa-builder-v2.md \
          docs/specs/RL/interactive-ranker-v2-decision-log.md
  git commit -m "docs: align ranker v2 with QA assertion utility"
  ```

## Task 2: Repair Count Provenance and Rebuild the Frozen Environment

**Files:**
- Modify: `data/lattice_profiles/lattice_profiles.json`
- Rebuild: `data/lattice_profiles/lattice_profiles.embindex.npz`
- Modify: `src/cloak/train/qa_builder.py`
- Modify: `scripts/build_arms_artifact.py`
- Modify: `scripts/build_ranker_env.py`
- Modify when defects originate upstream: `src/cloak/lattice_producer/coherence.py`
- Modify when defects originate upstream: `src/cloak/lattice_producer/merge.py`
- Test: `src/cloak/tests/test_build_arms_artifact_cli.py`
- Test: `src/cloak/tests/test_build_ranker_env_v2.py`

**Consumes:** Current ACI environment and lattice profiles referenced by
`results/qa_v2_aci_full/ranker-env.json`.

**Produces:** `results/ranker_v2/environment/ranker-env.json` with
`artifact_version=ranker-v2-environment-v2`, stable policy decisions, authored action order, and
explicit count provenance.

- [ ] **Step 1: Add failing environment-contract fixtures**

  Add tests that require every ranker-selectable decision to carry `profile_id` and every legal
  level action to carry these fields:

  ```python
  {
      "action_id": "sha256:f3c7c73aafefe377b54dd54c32b84b6d831461196c975c26f221163d7380f2eb",
      "mode": "level",
      "fill": "heart disease",
      "authored_level_index": 0,
      "count": 400.0,
      "count_grounding": {
          "status": "model-proposed",
          "source_family": "proposal-universe",
          "evidence_ref": "results/lattice-producer/run/item.json",
      },
  }
  ```

  Test that KEEP has no count, placeholder has no lattice count, all occurrences mapped to a
  policy decision are complete, forced decisions remain `ranker_selectable=false`, and authored
  level counts are non-decreasing by `authored_level_index`.

- [ ] **Step 2: Observe contract failures on the current environment**

  Run:

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_build_ranker_env_v2.py
  ```

  Expected: failures identify missing `profile_id`/grounding fields and nonmonotone level order.

- [ ] **Step 3: Generate a deterministic repair queue**

  Run the existing audit-to-producer path without model calls:

  ```bash
  mkdir -p results/ranker_v2/count_repair
  PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_lattice_repair_queue.py \
    --environment-audit results/qa_v2_aci_full/arms.environment-audit.json \
    --qa-audit results/qa_v2_aci_full_v16/aci_full.utility \
    --profiles data/lattice_profiles/lattice_profiles.json \
    --out results/ranker_v2/count_repair/repair-queue.jsonl \
    --triage-out results/ranker_v2/count_repair/identity-triage.jsonl \
    --report results/ranker_v2/count_repair/repair-report.json
  ```

  If the actual pinned profile path in the arms artifact differs, use that exact path and record it
  in `repair-report.json`; do not silently substitute another profile snapshot.

- [ ] **Step 4: Run offline repairs first**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -u scripts/run_lattice_producer.py \
    --run-dir results/ranker_v2/count_repair/offline-run \
    --profiles data/lattice_profiles/lattice_profiles.json \
    --queue results/ranker_v2/count_repair/repair-queue.jsonl \
    --out results/ranker_v2/count_repair/profiles-repaired.json \
    --offline-only \
    --workers 1
  ```

  Expected: deterministic source-backed/coherence repairs complete; unresolved model-proposed
  counts remain explicit queue failures. If unresolved entries require external calls, stop and
  request approval with the exact entry count, model, expected requests, and estimated cost.

- [ ] **Step 5: Repair causes, never reward-time symptoms**

  For each nonmonotone profile, determine whether the defect is wrong authored semantic order,
  wrong count evidence, or a merge-key mismatch. Correct that upstream source. Do not sort actions
  by count, clip a decreasing count upward, drop a level, or replace evidence with a default.

- [ ] **Step 5b: STOP — Timo confirms every profile edit before promotion**

  Present the proposed repairs as a reviewable report (per profile: current levels/counts, the
  diagnosed defect class, the exact proposed edit, and its evidence). Classify each edit by
  downstream cost: **count-only** (level values/order untouched — v16 utility migrates locally)
  versus **order/fill-changing** (authored ladder or fills change — invalidates the affected
  decisions' QA support and implies a partial paid QA rebuild, state the estimated call count).
  Apply NOTHING to the canonical profile artifact — directly or via subagent — until Timo has
  confirmed the edits. This checkpoint is a standing project rule, not a formality; a rejected
  edit goes back to diagnosis, never to a silent workaround.

- [ ] **Step 6: Validate and promote the repaired profile artifact (after confirmation)**

  Run the profile validator against `profiles-repaired.json`; require zero schema, count coverage,
  grounding, and monotonicity errors for every profile reached by a policy decision. Snapshot the
  prior canonical profile artifact in `results/`, then atomically promote the validated artifact and
  rebuild its embedding index:

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python - <<'PY'
  from cloak.lattice_profiles import load_profiles, validate_profile_artifact
  path = "results/ranker_v2/count_repair/profiles-repaired.json"
  errors = validate_profile_artifact(load_profiles(path))
  if errors:
      raise SystemExit("\n".join(errors))
  print("profile validation PASS")
  PY
  cp data/lattice_profiles/lattice_profiles.json \
     results/ranker_v2/count_repair/profiles-before.json
  cp results/ranker_v2/count_repair/profiles-repaired.json \
     data/lattice_profiles/lattice_profiles.json.next
  mv data/lattice_profiles/lattice_profiles.json.next \
     data/lattice_profiles/lattice_profiles.json
  PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_profile_embindex.py \
    data/lattice_profiles/lattice_profiles.json \
    --out data/lattice_profiles/lattice_profiles.embindex.npz
  ```

  If approved external producer calls were required, record their model, prompt pin, request count,
  and output hash in `repair-report.json` before promotion.

- [ ] **Step 7: Preserve provenance through the frozen-arms migration (no re-detection)**

  Do not re-run detection: the existing frozen arms artifact
  (`results/qa_v2_aci_full/arms.json`, the one pinned by v16) is the single detection source.
  Add a deterministic migration mode to the arms/environment path (e.g.
  `build_arms_artifact.py --from-arms <existing.json>` or an equivalent focused script) that
  keeps every document's frozen occurrences, decisions, offsets, and detector provenance
  byte-identical and re-derives ONLY the action menus from the repaired profile artifact.
  Decision/action serialization must preserve `profile_id`, `authored_level_index`, `count`,
  and the complete `count_grounding` record. Derive `action_id` from action semantics, not the
  normalized score. The environment hash must include every field that changes legality, action
  identity, count meaning, or utility support. The migration must refuse to mutate the canonical
  profile artifact while iterating documents (a detected alias that would mutate a profile is a
  hard error routed to the repair queue).

- [ ] **Step 8: Sanity-check the migration cost (light gate)**

  The migration is local and deterministic (profile matching against the embedding index is the
  only nontrivial step; no detector GPU pass, no model calls). Measure the one-document
  migration, confirm the 67-document estimate stays in single-digit minutes, and record the
  command and estimate in `results/ranker_v2/count_repair/repair-report.json`. The full
  `perf_gate.md` review workflow is not required for this step.

- [ ] **Step 9: Migrate the 67-document arms and environment artifacts**

  First run the smallest one-document check; then migrate all 67 documents only after it passes:

  ```bash
  mkdir -p results/ranker_v2/environment
  PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_arms_artifact.py \
    --from-arms results/qa_v2_aci_full/arms.json \
    --n-docs 1 \
    --out results/ranker_v2/environment/arms-smoke.json
  PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_ranker_env.py \
    --n-docs 1 \
    --corpora aci \
    --arms results/ranker_v2/environment/arms-smoke.json \
    --out results/ranker_v2/environment/ranker-env-smoke.json \
    --skip-probes
  ```

  After inspecting the smoke artifact, repeat over all 67 documents and output
  `results/ranker_v2/environment/arms.json` plus
  `results/ranker_v2/environment/ranker-env.json` (same commands without `--n-docs 1`).
  Exact flag names are an implementation detail; the frozen-detection reuse is not.

- [ ] **Step 10: Assert semantic compatibility with v16**

  Write a deterministic comparison in `test_build_ranker_env_v2.py` that reports whether document
  IDs, occurrence IDs, decision IDs, action IDs, action fills, modes, and authored order are
  unchanged from the v16-pinned environment. Classify the result:

  - **count-only compatible** — all listed semantic fields match; utility assertions may be locally
    rebound to the new environment hash;
  - **semantic change** — any listed field differs; the QA utility artifact must be rebuilt and
    regated rather than hash-rebound.

- [ ] **Step 11: Run the focused environment suite**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_build_ranker_env_v2.py \
    src/cloak/tests/test_build_arms_artifact_cli.py \
    src/cloak/tests/test_lattice_profiles.py \
    src/cloak/tests/test_lattice_producer_coherence.py
  ```

  Expected: PASS; artifact audit reports 67 documents, zero missing policy mappings, and zero
  nonmonotone policy profiles.

- [ ] **Step 12: Commit the environment correction**

  ```bash
  git add data/lattice_profiles/lattice_profiles.json \
          data/lattice_profiles/lattice_profiles.embindex.npz \
          src/cloak/train/qa_builder.py scripts/build_arms_artifact.py \
          scripts/build_ranker_env.py \
          src/cloak/lattice_producer/coherence.py \
          src/cloak/lattice_producer/merge.py \
          src/cloak/tests/test_build_arms_artifact_cli.py \
          src/cloak/tests/test_build_ranker_env_v2.py
  git commit -m "fix: preserve grounded counts in ranker environment"
  ```

  Add only upstream files actually changed; do not stage generated `results/` artifacts unless the
  repository's existing artifact policy requires them.

## Task 3: Build the Versioned Count Reward State

**Files:**
- Create: `src/cloak/train/count_reward.py`
- Create: `scripts/build_count_reward_state.py`
- Create: `src/cloak/tests/test_count_reward.py`

**Consumes:** `ranker-v2-environment-v2`.

**Produces:** `count-reward-state-v1`, a clause-level gate report, and the runtime interface
`CountReward.action_scores(decision_id, legal_action_ids)`.

- [ ] **Step 1: Define immutable count types and failing tests**

  Use these public types:

  ```python
  @dataclass(frozen=True)
  class CountActionScore:
      action_id: str
      decision_id: str
      runtime_type: str
      profile_id: str
      mode: str
      count: float | None
      score: float
      grounding_status: str | None
      source_family: str | None
      evidence_ref: str | None

  @dataclass(frozen=True)
  class TypeCountReference:
      runtime_type: str
      k_ref: float
      resolution: str
      profile_support: int
      low_reference_support: bool
      flat_count_signal: bool
  ```

  Tests must cover grounded-universe resolution, profile-balanced 95th percentile with at least 20
  profiles, max-profile fallback below 20 profiles, all-one flat signals, clipping, KEEP=0,
  placeholder=1, and rejection of missing/default/generic provenance.

- [ ] **Step 2: Observe the tests fail**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_count_reward.py
  ```

  Expected: import failure for `cloak.train.count_reward`.

- [ ] **Step 3: Implement the hard gate and type references**

  Implement:

  ```python
  validate_complete_counts(environment: Mapping) -> dict
  resolve_type_references(environment: Mapping) -> dict[str, TypeCountReference]
  build_count_reward_state(environment: Mapping) -> dict

  class CountReward:
      @classmethod
      from_artifact(cls, payload: Mapping) -> CountReward
      def action_scores(
          self, decision_id: str, legal_action_ids: Sequence[str]
      ) -> torch.Tensor
      def selected_document_score(
          self, action_vector: Mapping[str, str]
      ) -> float
  ```

  The gate reports every level's value, grounding status, source family, evidence reference, and
  clause result. It raises before state publication unless explicit coverage is 1.0, fallback
  gradient mass is 0.0, missing policy mappings are 0, and nonmonotone profiles are 0.
  The admitted status set for this experiment is exactly `certifying`, `model-proposed`, and
  `proposal-universe`. Model-proposed counts remain an experimental shaping signal and are reported
  separately; legacy defaults, inferred row fallbacks, and sentinel values are rejected.

- [ ] **Step 4: Implement exact expected count loss**

  Add:

  ```python
  def expected_count_loss(
      replay_steps: Sequence["ReplayedStep"],
      count_reward: CountReward,
      lambda_value: float,
      decision_count: int,
      rollout_count: int,
  ) -> torch.Tensor
  ```

  For every replayed state, calculate `sum(pi * p)` over the complete dynamic legal menu and return
  the negative sum weighted by `lambda_value / decision_count / rollout_count`. Do not use sampled
  `P_count`, RLOO, utility ties, or counterfactual eligibility in this function.

- [ ] **Step 5: Add the artifact CLI**

  `scripts/build_count_reward_state.py` accepts `--environment`, `--out`, and `--gate-report`, writes
  canonical sorted JSON, computes an artifact hash excluding the `artifact_hash` field itself, and
  exits nonzero on any hard-gate failure.

- [ ] **Step 6: Build the real count state**

  ```bash
  mkdir -p results/ranker_v2/reward
  PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_count_reward_state.py \
    --environment results/ranker_v2/environment/ranker-env.json \
    --out results/ranker_v2/reward/count-reward-state.json \
    --gate-report results/ranker_v2/reward/count-gate-report.json
  ```

  Expected: exit 0 and hard-gate PASS. Any failure blocks Tasks 4–15.

- [ ] **Step 7: Verify and commit**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_count_reward.py \
    src/cloak/tests/test_build_ranker_env_v2.py
  git add src/cloak/train/count_reward.py scripts/build_count_reward_state.py \
          src/cloak/tests/test_count_reward.py
  git commit -m "feat: add gated count reward state"
  ```

## Task 4: Publish a Policy-Scoped QA Utility Artifact

**Files:**
- Modify: `src/cloak/train/qa_builder.py`
- Modify: `scripts/build_qa_utility_artifact.py`
- Create: `scripts/migrate_qa_utility_artifact.py`
- Modify: `src/cloak/tests/test_qa_builder_v2.py`
- Modify: `src/cloak/tests/test_build_qa_utility_artifact_cli.py`

**Consumes:** v16 utility assertions and the corrected environment.

**Produces:** `utility-assertions-v2` at
`results/ranker_v2/qa/aci-full.utility` with policy dependency and residual routing.

- [ ] **Step 1: Add failing policy-routing tests**

  Cover four assertion shapes: policy-only link, fixed-only link, mixed policy/fixed link, and
  global assertion. Assert exact routing:

  ```python
  assert policy_only["policy_dependency_decision_ids"] == ["policy-a"]
  assert policy_only["credit_routing"] == "linked"
  assert fixed_only["policy_dependency_decision_ids"] == []
  assert fixed_only["credit_routing"] == "residual"
  assert mixed["policy_dependency_decision_ids"] == ["policy-a"]
  assert mixed["credit_routing"] == "linked"
  assert global_row["credit_routing"] == "residual"
  ```

  Also assert that literal JSON null remains `None`, never the string `"None"`.

- [ ] **Step 2: Observe failures on the current artifact compiler**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_qa_builder_v2.py \
    src/cloak/tests/test_build_qa_utility_artifact_cli.py
  ```

  Expected: failures for absent v2 routing fields and ambiguous `controlled_decision_ids`.

- [ ] **Step 3: Implement one canonical routing derivation**

  Add and reuse:

  ```python
  def policy_routing(
      assertion: Mapping,
      occurrence_to_decision: Mapping[str, str | None],
      policy_decision_ids: Collection[str],
  ) -> tuple[list[str], str]
  ```

  Deduplicate and sort dependency IDs. Reject unknown occurrence IDs. The builder and migration CLI
  must call this same function; neither may trust dependency IDs copied from an input artifact.

- [ ] **Step 4: Emit the v2 document contract**

  Replace ambiguous `controlled_decision_ids` with `policy_decision_ids` and
  `fixed_decision_ids`. Compute `uncovered_policy_decision_ids` from accepted linked assertions
  after routing. Keep occurrences, decisions, assertion weights, family budgets, missing-family
  budgets, and fixed denominator intact.

- [ ] **Step 5: Implement deterministic migration with compatibility proof**

  The migration CLI accepts `--input`, `--environment`, `--out`, and `--report`. It may rebind v16
  assertions only when Task 2 classified the environment `count-only compatible`. Its report must
  include old/new hashes, identical assertion IDs, identical scoring contracts, identical weights,
  policy/fixed decision counts, residual assertion counts, and before/after document utility parity
  on cached component vectors. A semantic environment change makes the CLI fail with
  `qa_rebuild_required`.

- [ ] **Step 6: Produce and inspect the 67-document artifact**

  ```bash
  mkdir -p results/ranker_v2/qa
  PYTHONPATH=src:scripts .venv/bin/python -u scripts/migrate_qa_utility_artifact.py \
    --input results/qa_v2_aci_full_v16/aci_full.utility \
    --environment results/ranker_v2/environment/ranker-env.json \
    --out results/ranker_v2/qa/aci-full.utility \
    --report results/ranker_v2/qa/migration-report.json
  ```

  If the CLI reports `qa_rebuild_required`, stop before external calls. Prepare the exact
  `build_qa_utility_artifact.py` rebuild command and request approval with cache-hit estimates and
  uncached teacher/reader call counts.

- [ ] **Step 7: Run the artifact gates**

  Add a real-artifact test that asserts 67 documents, 1,357 accepted assertions unless a rebuild
  intentionally changes admission, 701 policy decisions for a count-only migration, no fixed
  decision in `policy_decision_ids`, no unknown dependency, and exact fixed-denominator aggregation.

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_qa_builder_v2.py \
    src/cloak/tests/test_build_qa_utility_artifact_cli.py \
    src/cloak/tests/test_utility_credit.py
  ```

- [ ] **Step 8: Commit the artifact correction**

  ```bash
  git add src/cloak/train/qa_builder.py scripts/build_qa_utility_artifact.py \
          scripts/migrate_qa_utility_artifact.py \
          src/cloak/tests/test_qa_builder_v2.py \
          src/cloak/tests/test_build_qa_utility_artifact_cli.py
  git commit -m "fix: scope QA utility routing to policy decisions"
  ```

## Task 5: Add the Immutable V2 Environment and Trajectory Model

**Files:**
- Create: `src/cloak/train/ranker_environment.py`
- Create: `src/cloak/train/interactive_ranker.py`
- Create: `src/cloak/tests/test_ranker_environment.py`
- Create: `src/cloak/tests/test_interactive_ranker.py`

**Consumes:** Corrected environment and stable action IDs.

**Produces:** Typed document episodes, deterministic dynamic masks, sampled action vectors, and
gradient-bearing trajectory replay.

- [ ] **Step 1: Define failing loader and identity tests**

  Require these public types:

  ```python
  @dataclass(frozen=True)
  class RankerAction:
      action_id: str
      mode: str
      fill: str | None
      authored_level_index: int | None
      runtime_type: str

  @dataclass(frozen=True)
  class RankerDecision:
      decision_id: str
      profile_id: str
      runtime_type: str
      canonical_key: str
      occurrence_ids: tuple[str, ...]
      actions: tuple[RankerAction, ...]

  @dataclass(frozen=True)
  class RankerDocument:
      doc_id: str
      corpus: str
      text: str
      occurrences: tuple[Mapping, ...]
      policy_decisions: tuple[RankerDecision, ...]
      fixed_decisions: tuple[RankerDecision, ...]

  @dataclass(frozen=True)
  class SampledStep:
      decision_id: str
      legal_action_ids: tuple[str, ...]
      selected_action_id: str
      claimed_fills_before: tuple[str, ...]

  @dataclass(frozen=True)
  class SampledTrajectory:
      doc_id: str
      lambda_profile: str
      steps: tuple[SampledStep, ...]
      action_vector: Mapping[str, str]
  ```

  Tests must reject duplicate decision/action IDs, unordered decisions, missing mapped occurrences,
  fixed decisions in the policy set, action-vector omissions, and environment/hash mismatches.

- [ ] **Step 2: Implement `load_ranker_environment`**

  ```python
  load_ranker_environment(path: Path) -> dict[str, RankerDocument]
  ```

  Read only `ranker-v2-environment-v2`. Do not adapt legacy `spans`, `k_floors`, `tau`, `p6`, active
  floors, or corpus policy features. Sort policy decisions by first occurrence offset with
  `decision_id` as deterministic tie-breaker.

- [ ] **Step 3: Implement dynamic legal masks and assembly**

  ```python
  def legal_action_ids(
      decision: RankerDecision,
      claimed_fills: Mapping[str, str],
      reserved_fixed_fills: Collection[str],
  ) -> tuple[str, ...]

  def assemble_action_vector(
      document: RankerDocument,
      action_vector: Mapping[str, str],
  ) -> tuple[str, list[dict]]
  ```

  KEEP and placeholder stay legal. A level fill claimed by another policy decision or colliding
  with a fixed exact rewrite is masked before sampling. Every occurrence of one decision receives
  the same action. Assembly raises on a collision rather than repairing a sampled suffix.

- [ ] **Step 4: Separate no-grad sampling from gradient replay**

  Implement:

  ```python
  @torch.no_grad()
  sample_trajectory(policy, document, lambda_profile, *, greedy, generator) -> SampledTrajectory

  replay_trajectory(policy, document, trajectory, lambda_profile) -> ReplayedTrajectory
  ```

  Sampling stores stable IDs and masks without retaining an autograd graph across remote scoring.
  Replay recomputes states under the unchanged policy and returns ordered log-probabilities,
  entropies, and complete legal distributions. It raises if any replayed legal menu differs from
  the sampled menu.

- [ ] **Step 5: Verify deterministic repeated-occurrence behavior**

  Add a fixture with two occurrences sharing one decision. Assert one sampled log-probability, one
  count contribution, and two rewritten occurrences. Add a second decision whose level fill
  collides and assert the later menu masks it.

- [ ] **Step 6: Run tests and commit**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_ranker_environment.py \
    src/cloak/tests/test_interactive_ranker.py
  git add src/cloak/train/ranker_environment.py \
          src/cloak/train/interactive_ranker.py \
          src/cloak/tests/test_ranker_environment.py \
          src/cloak/tests/test_interactive_ranker.py
  git commit -m "feat: add ranker v2 episode model"
  ```

## Task 6: Implement the Conditional Sequential Policy

**Files:**
- Modify: `src/cloak/train/ranker.py`
- Create: `src/cloak/tests/test_conditional_ranker.py`

**Consumes:** `RankerDocument`, dynamic legal menus, count scores, and a supported lambda-menu
manifest.

**Produces:** `ConditionalRankerPolicy.log_probs` and `.sample` with exact lambda-zero identity
initialization and previous-decision state.

- [ ] **Step 1: Add failing feature and identity tests**

  Define the v2 action feature vector as mode one-hot, normalized authored level position, number of
  levels, type one-hot, and frozen normalized count score. Remove `p6`, `log10_aset`, active-floor,
  and corpus features from the v2 path. Tests must show:

  - identical unconditioned logits across every supported profile immediately after initialization;
  - changing a profile changes relative logits after nonzero conditioning weights are set;
  - adding one scalar equally to all logits is not the conditioning implementation;
  - previous selected actions can change a later decision's logits even when its legal menu is
    unchanged.

- [ ] **Step 2: Implement occurrence-context aggregation**

  Reuse the frozen ModernBERT encoder. Embed one bounded context window per occurrence, cache by
  `(environment_hash, encoder_pin, occurrence_id)`, and mean-pool occurrence embeddings per stable
  decision. Embed non-null action fill text once by `(encoder_pin, action_id)`; use learned mode
  vectors for KEEP and placeholder endpoints.

- [ ] **Step 3: Implement sequential state and lambda conditioning**

  Add:

  ```python
  class ConditionalRankerPolicy(nn.Module):
      begin_document(self, document: RankerDocument, profile: LambdaProfile) -> PolicyState
      def log_probs(
          self,
          state: "PolicyState",
          decision: RankerDecision,
          legal_action_ids: Sequence[str],
          profile: "LambdaProfile",
      ) -> torch.Tensor
      def advance(
          self, state: "PolicyState", decision: RankerDecision, action_id: str
      ) -> PolicyState
  ```

  Use a `GRUCell` over the selected decision/action representation. Condition the scoring head with
  both normalized ordered magnitude and learned profile identity through FiLM or explicit
  action-feature cross terms. Initialize profile embeddings and cross terms to zero, FiLM scale to
  one, and FiLM bias to zero.

- [ ] **Step 4: Preserve the predecessor policy**

  Keep `RankerPolicy` and `EncoderPolicy` callable for historical scripts and tests. Do not change
  their checkpoint format. Put every new feature name and dimension behind the
  `ConditionalRankerPolicy` interface.

- [ ] **Step 5: Run focused tests and commit**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_conditional_ranker.py \
    src/cloak/tests/test_train_roundtrip_mode.py
  git add src/cloak/train/ranker.py src/cloak/tests/test_conditional_ranker.py
  git commit -m "feat: add lambda-conditioned sequential ranker"
  ```

## Task 7: Implement Complete Utility Scoring and Cache

**Files:**
- Modify: `src/cloak/train/qa_builder.py`
- Modify: `src/cloak/train/roundtrip.py`
- Create: `src/cloak/train/utility_cache.py`
- Create: `src/cloak/tests/test_utility_cache.py`
- Modify: `src/cloak/tests/test_roundtrip.py`
- Modify: `src/cloak/tests/test_qa_builder_v2.py`

**Consumes:** A complete stable action vector and `utility-assertions-v2`.

**Produces:** A complete `UtilityResult` and append-only cache safe for base and counterfactual
rollouts.

- [ ] **Step 1: Define the result and cache contract in failing tests**

  Use:

  ```python
  @dataclass(frozen=True)
  class UtilityResult:
      doc_id: str
      action_vector: Mapping[str, str]
      doc_p: str
      out_p: str
      out_final: str
      component_scores: Mapping[str, float]
      utility: float
      result_hash: str
  ```

  Require exact assertion-ID coverage, finite scores in `[0,1]`, fixed-denominator utility parity,
  complete pins, truncation detection, duplicate-key conflict detection, and no partial cache entry
  after a failed stage.

- [ ] **Step 2: Port only the validated clean-branch cache concepts**

  Inspect with:

  ```bash
  git show codex/qa-builder-v2-clean:src/cloak/train/roundtrip.py
  git show codex/qa-builder-v2-clean:scripts/train_ranker.py
  ```

  Reimplement the append-only JSONL validation in `utility_cache.py`. Repin request identity to
  document ID, ordered decision/action vector, rendered `doc_p` hash, environment hash, utility
  artifact hash and binding, task prompt/model pin, extractor pin, reader pin, scorer version, and
  execution-contract version. Never accept the clean branch's obsolete builder/scorer pins.

- [ ] **Step 3: Add staged batch execution**

  Implement:

  ```python
  def score_roundtrip_batch(
      requests: Sequence["UtilityRequest"],
      *,
      cache: UtilityCache,
      remote_workers: int,
      reader_workers: int,
      reader_refresh: bool = False,
  ) -> list[UtilityResult]
  ```

  Execute cache misses by stage: render all inputs; generate all `out_p`; extract all `out_final`;
  compute deterministic delivered assertions; flatten context assertions into one bounded reader
  work queue; reconstruct one complete vector per rollout; validate; then atomically append complete
  results. Preserve per-assertion excerpts and answer contracts. Reader refresh changes the cache
  identity and is used only for pinned determinism/reverification steps.

- [ ] **Step 4: Benchmark the actual reader contract**

  Add counters for rollouts, context assertions, reader work items, transport calls, cache hits,
  latency by stage, and peak concurrency. Run one cached and one approved uncached ACI document.
  Record the result in the RL-ranker training record created in Task 11. If transport remains one
  generation per assertion, report that honestly; do not label it one model call per rollout.

- [ ] **Step 5: Verify cache reuse and invalidation**

  Tests must show that duplicate action vectors in one submitted batch dispatch once, a later call
  is a cache hit, a changed reader pin misses, a changed assertion weight misses, a changed
  environment hash misses, and failed generation/extraction/reader work writes no entry.

- [ ] **Step 6: Run tests and commit**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_utility_cache.py \
    src/cloak/tests/test_roundtrip.py \
    src/cloak/tests/test_qa_builder_v2.py
  git add src/cloak/train/qa_builder.py src/cloak/train/roundtrip.py \
          src/cloak/train/utility_cache.py \
          src/cloak/tests/test_utility_cache.py \
          src/cloak/tests/test_roundtrip.py \
          src/cloak/tests/test_qa_builder_v2.py
  git commit -m "feat: cache complete QA utility round trips"
  ```

## Task 8: Wire Fixed-Denominator Structured Utility Credit

**Files:**
- Modify: `src/cloak/train/utility_credit.py`
- Modify: `src/cloak/tests/test_utility_credit.py`
- Modify: `src/cloak/tests/test_interactive_ranker.py`

**Consumes:** Rollout component vectors and v2 policy-routing metadata.

**Produces:** `DocumentUtilityCredit` keyed by `(rollout_index, decision_id)` and a provisional
utility loss with one term per rollout-decision pair.

- [ ] **Step 1: Add failing routing and denominator tests**

  Include policy-only, fixed-only residual, mixed hyperedge, global residual, linked uncovered,
  missing-family, duplicate occurrence, multi-decision, and tied-component fixtures. Assert that a
  0.4 delivered-family document cannot score above 0.4 when its 0.6 context family is absent.

- [ ] **Step 2: Implement explicit score partitions**

  Expose:

  ```python
  @dataclass(frozen=True)
  class DocumentUtilityCredit:
      document_utility: tuple[float, ...]
      linked_utility: Mapping[str, tuple[float, ...]]
      residual_utility: tuple[float, ...]
      provisional_advantage: Mapping[tuple[int, str], float]
      route: Mapping[str, str]

  document_utility(component_scores: Mapping[str, float], artifact: Mapping, doc_id: str) -> float
  def provisional_credit(
      component_vectors: Sequence[Mapping[str, float]], artifact: Mapping, doc_id: str
  ) -> DocumentUtilityCredit
  ```

  Every partial score uses builder weights divided by the document's fixed denominator. Catch and
  report a missing assertion score rather than treating it as zero.

- [ ] **Step 3: Implement provisional utility loss**

  In `interactive_ranker.py`, add:

  ```python
  def provisional_utility_loss(
      replayed: Sequence["ReplayedTrajectory"],
      credit: DocumentUtilityCredit,
  ) -> torch.Tensor
  ```

  Return `(1/G) * sum(-A[g,j] * log_pi[g,j])`. Component-level ties create a zero utility term for
  that pair but do not remove the trajectory from count, entropy, KL, or diagnostics.

- [ ] **Step 4: Run tests and commit**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_utility_credit.py \
    src/cloak/tests/test_interactive_ranker.py
  git add src/cloak/train/utility_credit.py \
          src/cloak/tests/test_utility_credit.py \
          src/cloak/tests/test_interactive_ranker.py
  git commit -m "feat: route fixed-denominator utility credit"
  ```

## Task 9: Adapt Behavior Cloning and Utility-Only ExIt

**Files:**
- Modify: `src/cloak/train/interactive_ranker.py`
- Modify: `scripts/train_interactive_ranker.py`
- Modify: `src/cloak/tests/test_interactive_ranker.py`

**Consumes:** Conditional policy in identity mode and complete cached utility scoring.

**Produces:** A verified utility-only warm-start checkpoint and frozen per-document trajectory pool.

- [ ] **Step 1: Add deterministic BC teacher tests**

  Define a support-preserving teacher that chooses the most utility-preserving legal non-KEEP level
  from authored order and falls back to placeholder only when no such level is legal. Assert stable
  action IDs, dynamic injectivity, full decision coverage, and exact replay.

- [ ] **Step 2: Implement BC against stable IDs**

  Train with cross-entropy over dynamic legal menus. Record action-mode/type distributions and
  document `U`/`P_count` points. Do not import floor legality or positional surface keys from the
  predecessor trainer.

- [ ] **Step 3: Implement utility-only ExIt collection**

  For each document, score the BC reference and `G` sampled lambda-zero trajectories using pure
  `U_document`. Keep only a candidate that strictly beats the reference, then serially reverify the
  candidate and reference with `reader_refresh=true`. Store verified winners and all valid cached
  candidates; do not apply count or lambda to selection.

- [ ] **Step 4: Separate winner collection from profile cloning**

  Before the supported menu exists, store winner action vectors in
  `results/ranker_v2/calibration/exit-winners.json`. After Task 11 freezes the menu, replay each
  winner target once under every supported profile and clone it into the conditional head. This
  adds no remote calls and prevents random profile embeddings from damaging the utility warm start.

- [ ] **Step 5: Collect real winners cache-first**

  Run the collection CLI with `--cache-only` on the three smoke documents selected for Task 13.
  If any required trajectory is absent, print the exact remote-task and context-reader work-item
  counts and request approval before dispatch. Do not manufacture ExIt winners from fixture scores
  or skip serial reverification to satisfy the calibration-pool requirement.

- [ ] **Step 6: Verify utility-only behavior**

  Test that a lower-utility/higher-count candidate never wins ExIt, ties never replace BC, failed
  serial reverification drops the candidate, and identical cached candidates do not dispatch twice.

- [ ] **Step 7: Run tests and commit**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_interactive_ranker.py \
    src/cloak/tests/test_train_roundtrip_mode.py
  git add src/cloak/train/interactive_ranker.py \
          scripts/train_interactive_ranker.py \
          src/cloak/tests/test_interactive_ranker.py
  git commit -m "feat: add ranker v2 BC and utility-only ExIt"
  ```

## Task 10: Implement One-Decision Counterfactuals and the Scheduler

**Files:**
- Create: `src/cloak/train/counterfactuals.py`
- Modify: `src/cloak/train/interactive_ranker.py`
- Modify: `src/cloak/tests/test_counterfactual_credit.py`
- Modify: `src/cloak/tests/test_interactive_ranker.py`

**Consumes:** Sampled trajectories, current policy replay, complete utility cache, dependency
coverage, entropy, and fixed scheduler configuration.

**Produces:** Valid one-decision intervention requests, measured `delta_U`, bounded pair losses,
and auditable scheduler reports.

- [ ] **Step 1: Add failing intervention tests**

  Test adjacent finer/coarser selection, balanced direction, KEEP/placeholder endpoint reserve,
  duplicate-text skip, equal-action skip, illegal alternative skip, collision skip, all-occurrence
  rewrite, no suffix resampling, and complete vector rescoring.

- [ ] **Step 2: Implement alternative eligibility**

  Expose:

  ```python
  @dataclass(frozen=True)
  class CounterfactualRequest:
      doc_id: str
      rollout_index: int
      decision_id: str
      selected_action_id: str
      alternative_action_id: str
      direction: str
      priority_tier: int

  def eligible_alternatives(
      document: RankerDocument,
      trajectory: SampledTrajectory,
      decision_id: str,
  ) -> tuple[str, ...]
  ```

  Use authored level adjacency. Endpoints are eligible only through the configured minority reserve.
  An alternative that conflicts with a later selected fill is ineligible; never repair the suffix.

- [ ] **Step 3: Implement the coverage-and-uncertainty scheduler**

  The scheduler receives a fixed call budget. Reserve exactly 20% using seeded uniform sampling over
  every eligible decision-rollout pair. Allocate the remaining 80% lexicographically by: no linked
  assertion; only multi-decision (hyperedge) links; high entropy; unseen adjacent pair; oldest
  measured pair. (The spec's former "low-confidence links" input is removed — v16 stores no
  dependency-confidence scalar, and inventing one from provenance would be an unvalidated
  reward-allocation heuristic; Task 1 amends the spec accordingly.) Sample uniformly within ties. Priority changes selection probability only, never
  reward or loss magnitude. Require the frozen budget to be divisible by five, then set
  `uniform_budget = budget // 5`; reject rather than round a nonconforming budget.

- [ ] **Step 4: Implement complete interventions and pair loss**

  Construct the alternative action vector, assemble `doc_p`, run `score_roundtrip_batch`, and compute:

  ```text
  delta_U = U_selected - U_alternative
  q_pair = pi(selected) / (pi(selected) + pi(alternative))
  L_cf = -delta_U * (q_pair - 1/2)
  ```

  Derive `q_pair` from gradient-bearing replay over the full original menu. Do not calculate it from
  detached sampling probabilities.

- [ ] **Step 5: Implement in-place utility substitution**

  Add:

  ```python
  def hybrid_utility_loss(
      replayed,
      provisional_credit,
      counterfactual_losses: Mapping[tuple[int, str], torch.Tensor],
  ) -> torch.Tensor
  ```

  For each `(g,j)`, use exactly one term: pair loss when tested, otherwise provisional REINFORCE
  loss. Divide the final sum by `G`; do not average counterfactual terms separately and do not add
  them on top of provisional terms.

- [ ] **Step 6: Add scheduler/cache diagnostics**

  Report exact budget, uniform/priority allocations, endpoint fraction, direction balance, cache
  hits, never-measured eligible decisions, pair age, zero/sign/magnitude `delta_U`, and skip reasons.

- [ ] **Step 7: Run tests and commit**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_counterfactual_credit.py \
    src/cloak/tests/test_interactive_ranker.py \
    src/cloak/tests/test_utility_cache.py
  git add src/cloak/train/counterfactuals.py \
          src/cloak/train/interactive_ranker.py \
          src/cloak/tests/test_counterfactual_credit.py \
          src/cloak/tests/test_interactive_ranker.py
  git commit -m "feat: add ranker counterfactual scheduler"
  ```

## Task 11: Freeze the Calibration Pool, Diagnostics, and Lambda Menu

**Files:**
- Create: `src/cloak/train/lambda_menu.py`
- Create: `src/cloak/train/ranker_diagnostics.py`
- Create: `scripts/run_ranker_preflight.py`
- Create: `src/cloak/tests/test_lambda_menu.py`
- Create: `src/cloak/tests/test_ranker_diagnostics.py`
- Create: `research-wiki/training/2026-07-22-RL-ranker-v5-interactive-count-conditioned.md`

**Consumes:** BC, verified ExIt winners, rule anchors, support-scan trajectories, measured adjacent
counterfactuals, count state, and cached utility vectors from train/development data only.

**Produces:** Immutable calibration pool, raw switch points, diagnostic spike, threshold manifest,
accepted three-to-five-profile lambda menu, and pre-run training record.

- [ ] **Step 1: Write the training record before any preflight run**

  Populate every required section from the project training-experiment schema. Set
  `status: planned`, `result: pending`, pin the environment/utility/count artifacts, define the
  train/development split, state that final attacker data is excluded from calibration, and list the
  exact success/stop criteria from the diagnostics spec. If implementation begins after 2026-07-22,
  use the actual run date in the filename while retaining the next available `RL-ranker` version on
  `main`.

- [ ] **Step 2: Add failing frontier tests**

  Cover exact action-vector deduplication, exact `(U,P)` tie multiplicity, weak dominance removal,
  upper convex envelope construction, positive switch-point calculation, per-document total weight
  one, weighted log-lambda quantiles, nearest-observed snapping, replay-signature merging, and at
  most two deterministic replacement passes.

- [ ] **Step 3: Implement calibration-pool construction**

  Include BC, verified ExIt winner, KEEP walk, minimum-count non-KEEP walk, midpoint-level walk,
  all-placeholder walk, sampled support trajectories, and measured adjacent counterfactual vectors.
  Store ordered action vectors, `U`, `P_count`, component vector, count provenance, and every reward
  pin. A document contributes switch points only with at least three distinct `(U,P)` points but
  remains in replay validation otherwise.

- [ ] **Step 4: Predeclare unresolved threshold rules**

  Write `results/ranker_v2/preflight/threshold-rules.json` before the spike. For each unresolved
  field, encode measurement definition, candidate selection rule, allowed split, support rule,
  deterministic tie handling, and action (`block`, `ablation`, `reduce_scope`, or `report_only`). Do
  not fill a numeric value from observed full-training or attacker outcomes.

- [ ] **Step 5: Implement the diagnostic spike**

  Emit every required measurement from `interactive-ranker-v2-diagnostics.md`: unique trajectories
  and `(U,P)` points; frontier/switch spread; winner signatures; utility quantization and reader
  jitter; linked/residual/fallback counts; uncovered decisions; counterfactual zero/sign/magnitude;
  flat/clipped menus and adjacent `delta p`; collisions/lost opportunity; support by corpus/type/
  profile/provenance.

- [ ] **Step 6: Freeze the threshold manifest**

  Apply the predeclared rules exactly once. Hard-code only normative invariants:

  ```yaml
  explicit_count_coverage: 1.0
  fallback_count_gradient_mass: 0.0
  missing_occurrence_decision_mappings: 0
  nonmonotone_profiles: 0
  lambda_zero_identity: exact
  ```

  Every run-relevant empirical field must contain a frozen numeric value; use `report_only` only when
  the rules declared that measurement incapable of changing scope or verdict.

- [ ] **Step 7: Select and replay the lambda menu**

  Start with zero. For a four-profile menu, use weighted switch-point quantiles 0.25, 0.60, and 0.90
  in log space, snapped to observed switch points. Merge equivalent replay signatures. Apply the
  frozen adjacent winner-change threshold, nondecreasing selected `P_count`, lambda-zero exact
  utility identity, all-placeholder ceiling, and corpus/type support gates. Stop rather than padding
  if fewer than three supported profiles remain.

- [ ] **Step 8: Run the cache-only preflight**

  ```bash
  mkdir -p results/ranker_v2/preflight
  PYTHONPATH=src:scripts .venv/bin/python -u scripts/run_ranker_preflight.py \
    --environment results/ranker_v2/environment/ranker-env.json \
    --utility-artifact results/ranker_v2/qa/aci-full.utility \
    --count-state results/ranker_v2/reward/count-reward-state.json \
    --utility-cache results/ranker_v2/cache/utility-results.jsonl \
    --exit-winners results/ranker_v2/calibration/exit-winners.json \
    --threshold-rules results/ranker_v2/preflight/threshold-rules.json \
    --out-dir results/ranker_v2/preflight \
    --cache-only
  ```

  Expected: PASS only if all required trajectories already exist. If cache misses block the spike,
  report exact missing action vectors and request approval before rerunning without `--cache-only`.

- [ ] **Step 9: Run tests and commit**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_lambda_menu.py \
    src/cloak/tests/test_ranker_diagnostics.py
  git add src/cloak/train/lambda_menu.py src/cloak/train/ranker_diagnostics.py \
          scripts/run_ranker_preflight.py \
          src/cloak/tests/test_lambda_menu.py \
          src/cloak/tests/test_ranker_diagnostics.py \
          research-wiki/training/2026-07-22-RL-ranker-v5-interactive-count-conditioned.md
  git commit -m "feat: freeze ranker lambda preflight"
  ```

## Task 12: Implement the Lambda-Conditioned Hybrid Trainer

**Files:**
- Modify: `src/cloak/train/interactive_ranker.py`
- Modify: `scripts/train_interactive_ranker.py`
- Modify: `src/cloak/tests/test_interactive_ranker.py`
- Modify: `src/cloak/tests/test_conditional_ranker.py`
- Modify: `src/cloak/tests/test_ranker_diagnostics.py`

**Consumes:** Frozen environment, utility artifact/cache, count state, threshold manifest, supported
lambda menu, verified ExIt winners, and counterfactual scheduler.

**Produces:** One conditional checkpoint, one fixed lambda-zero control, per-epoch diagnostics, and
restartable training state.

- [ ] **Step 1: Add failing objective-composition tests**

  Verify numerically on a two-rollout/two-decision fixture:

  ```text
  H(d)  = (1/G) sum_g,j H(pi_gj)
  KL(d) = (1/G) sum_g,j KL(pi_gj || pi_ref_gj)

  L(d) = (1/G) sum_g,j ell[g,j]
       + L_count(d)
       - beta * H(d)
       + eta * KL(d)

  L = mean_d L(d)
  ```

  Assert counterfactual substitution preserves `1/G`, count gradients survive utility ties,
  count loss is not divided twice by decisions, entropy/KL coefficients do not change with lambda,
  and no sampled `P_count` enters RLOO.

- [ ] **Step 2: Implement balanced profile scheduling**

  Use a seeded Latin-cycle schedule so every document sees every supported profile once in each
  block of `|Lambda|` epochs. Record exposure by document, corpus, runtime type, and profile. One
  profile is fixed for all `G` trajectories of a document group.

- [ ] **Step 3: Initialize and clone the warm start**

  Instantiate the production conditional policy from the frozen menu, import the unconditioned
  BC/ExIt weights, assert exact profile identity before conditional updates, then clone every
  verified ExIt winner under every supported profile. Save the immutable post-clone checkpoint as
  the KL reference. Start with KL disabled; enable it only if the frozen collapse rule fires.

- [ ] **Step 4: Implement one optimizer step without graph retention**

  For each document group: sample `G` trajectories under `torch.no_grad`; score cache hits/misses;
  calculate structured credit; schedule and score counterfactuals; replay every sampled trajectory
  with gradients; assemble utility/count/entropy/KL terms; check all values finite; backpropagate;
  clip only if a value is frozen in the training record; step optimizer; then release the graph.

- [ ] **Step 5: Implement checkpoint and pin safety**

  Save model, optimizer, epoch, random-generator states, profile schedule state, environment hash,
  utility/count/menu/threshold hashes, policy architecture pin, cache paths, and code revision.
  Resume only when every pin matches exactly.

- [ ] **Step 6: Emit mandatory epoch reports**

  Report detached gradient norm and absolute weighted advantage mass for linked, residual,
  fallback, counterfactual, count, entropy, and KL terms. Stratify action modes, count score,
  utility, entropy, collision effects, and exposure by profile/corpus/type. Report scheduler budget
  and cache behavior separately from reward magnitude.

- [ ] **Step 7: Implement the thin CLI**

  Require explicit paths for every frozen artifact and output. Refuse to run when any preflight gate
  fails or hash differs. Include `--cache-only`, `--max-docs`, `--max-epochs`, `--rollouts`,
  `--remote-workers`, `--reader-workers`, `--seed`, and `--fixed-lambda-zero-control`; do not expose
  a flag that bypasses count, mapping, menu, or threshold gates.

- [ ] **Step 8: Run tests and commit**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_interactive_ranker.py \
    src/cloak/tests/test_conditional_ranker.py \
    src/cloak/tests/test_ranker_diagnostics.py \
    src/cloak/tests/test_counterfactual_credit.py \
    src/cloak/tests/test_count_reward.py
  git add src/cloak/train/interactive_ranker.py \
          scripts/train_interactive_ranker.py \
          src/cloak/tests/test_interactive_ranker.py \
          src/cloak/tests/test_conditional_ranker.py \
          src/cloak/tests/test_ranker_diagnostics.py
  git commit -m "feat: train interactive ranker v2"
  ```

## Task 13: Pass the Representative Real-Data Vertical Slice

**Files:**
- Modify: `research-wiki/training/2026-07-22-RL-ranker-v5-interactive-count-conditioned.md`
- Create at runtime: `results/ranker_v2/smoke/`

**Consumes:** The complete implementation and frozen preflight artifacts.

**Produces:** A real ACI end-to-end policy update with inspected outputs from every required reward
family. This is the first point at which the implementation may be described as operational on real
data.

- [ ] **Step 1: Freeze falsifiable smoke criteria before running**

  In the training record, name three ACI documents selected from train/development data: one with
  both context and delivered assertions, one with no accepted context assertion, and one with a
  repeated policy decision. Require: complete round trips; nonempty `out_p`/`out_final`; context and
  delivered component scores where present; residual and fallback routing; finite policy update;
  exact lambda-zero initialization; different post-update relative logits for at least two profiles
  on a non-flat menu; executed/cached counterfactual pair; and no hard-gate failure.

- [ ] **Step 2: Check GPU availability and exclusivity**

  ```bash
  .venv/bin/python -c 'import torch; print(torch.cuda.is_available())'
  pgrep -af 'python.*(train|eval|llama|ranker)' || true
  ```

  Expected: CUDA/ROCm availability is `True`; no conflicting GPU training process. Do not start a
  second GPU process.

- [ ] **Step 3: Run cache-only first**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -u scripts/train_interactive_ranker.py \
    --environment results/ranker_v2/environment/ranker-env.json \
    --utility-artifact results/ranker_v2/qa/aci-full.utility \
    --count-state results/ranker_v2/reward/count-reward-state.json \
    --lambda-menu results/ranker_v2/preflight/lambda-menu.json \
    --threshold-manifest results/ranker_v2/preflight/threshold-manifest.json \
    --utility-cache results/ranker_v2/cache/utility-results.jsonl \
    --out-dir results/ranker_v2/smoke \
    --max-docs 3 \
    --max-epochs 1 \
    --rollouts 2 \
    --seed 0 \
    --cache-only \
    --fixed-lambda-zero-control
  ```

  If misses remain, request approval with exact remote/reader counts; then rerun without
  `--cache-only`.

- [ ] **Step 4: Inspect actual artifacts**

  Record counts and at least one successful and one low-scoring example for `doc_p`, `out_p`,
  `out_final`, context components, delivered components, residual/fallback credit,
  counterfactual `delta_U`, action modes, and count scores. A zero-useful-output or generation-error
  run fails even if the process exits zero.

- [ ] **Step 5: Run the performance gate before larger work**

  Use the repository performance prompt and `auto-review-loop` against the proposed full command.
  Confirm batch sizes, remote and reader concurrency, cache-hit expectation, GPU saturation, peak
  memory, and wall-time estimate. If estimated wall time exceeds 10 minutes, confirm utilization on
  the smoke before launching the full run.

- [ ] **Step 6: Run the full local verification suite**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_build_ranker_env_v2.py \
    src/cloak/tests/test_build_qa_utility_artifact_cli.py \
    src/cloak/tests/test_qa_builder_v2.py \
    src/cloak/tests/test_ranker_environment.py \
    src/cloak/tests/test_count_reward.py \
    src/cloak/tests/test_conditional_ranker.py \
    src/cloak/tests/test_utility_credit.py \
    src/cloak/tests/test_utility_cache.py \
    src/cloak/tests/test_counterfactual_credit.py \
    src/cloak/tests/test_lambda_menu.py \
    src/cloak/tests/test_ranker_diagnostics.py \
    src/cloak/tests/test_interactive_ranker.py \
    src/cloak/tests/test_roundtrip.py \
    src/cloak/tests/test_train_roundtrip_mode.py
  ```

- [ ] **Step 7: Commit smoke evidence**

  Update the training record with measured smoke results, exact commands, cache/call counts, and
  artifact paths. Keep status `planned` unless the record's declared run is only the smoke; use
  `running` when the full training launch begins.

  ```bash
  git add research-wiki/training/2026-07-22-RL-ranker-v5-interactive-count-conditioned.md
  git commit -m "docs: record interactive ranker smoke"
  ```

---

**PHASE BOUNDARY — separate green light required.** Tasks 1–13 deliver runnable training
(the real-data vertical slice is the acceptance point). Tasks 14–15 — benchmark integration,
full 67-document training, and matched-realized-privacy evaluation — are deferred until Timo
reviews the Task 13 evidence and explicitly approves this second phase. Do not begin Task 14
on the strength of a passing Task 13 alone.

---

## Task 14: Integrate the Policy and LLM Attackers into the Benchmark

**Files:**
- Create: `src/bench/ranker_policy.py`
- Modify: `src/bench/runner.py`
- Modify: `src/bench/privacy.py`
- Modify: `src/bench/metrics.py`
- Modify: `src/bench/schema.py`
- Modify: `src/bench/registry.py`
- Modify: `scripts/run_roundtrip_benchmark.py`
- Create: `src/cloak/tests/test_bench_ranker_policy.py`
- Create: `src/cloak/tests/test_bench_llm_privacy.py`

**Consumes:** Frozen conditional-policy, environment, menu, and benchmark contracts.

**Produces:** One pinned benchmark path for conditional ranker traces and model-bearing realized
privacy attacks.

- [ ] **Step 1: Add the conditional-policy benchmark adapter**

  Implement `src/bench/ranker_policy.py` so the benchmark loads the frozen environment, count
  state, lambda menu, and checkpoint once; looks up the benchmark document by stable document ID;
  ranks only its policy decisions; applies fixed decisions from the same environment; and emits
  `doc_p`, `R`, ordered action vector, profile identity, and every artifact hash. Reject a benchmark
  document absent from the frozen environment rather than redetecting it or silently falling back
  to the predecessor substitutor.

- [ ] **Step 2: Implement the model-bearing privacy attackers**

  The current benchmark validates attacker-model flags but scores only deterministic exact-match
  attacks. Add cached model-bearing implementations for the three declared stages:

  ```text
  attack_attributes(doc_p, original policy-decision values, disclosed mechanism)
      -> per-value exact/generalized recovery and document attack success

  attack_reconstruction(doc_p, original policy-decision values, disclosed mechanism)
      -> top-k span recovery and document attack success

  attack_leak_through(out_final, original policy-decision values)
      -> per-value recovery and document attack success
  ```

  Pin prompt version, model, temperature, output schema, retry policy, and cache namespace in
  `BenchmarkConfig`. Require structured output tied to stable decision IDs. Score against frozen
  source values outside the attacker prompt-construction code. Keep deterministic attackers as
  diagnostics, never as the headline realized-privacy measure for this run.

- [ ] **Step 3: Add benchmark CLI pins and failing tests**

  Extend `scripts/run_roundtrip_benchmark.py` with `--ranker-checkpoint`, `--ranker-environment`,
  `--ranker-count-state`, `--ranker-lambda-menu`, `--lambda-profile`, and `--config`. The config file
  contains the already frozen task, extractor, attacker, detector, prompt, retry, and cache pins;
  explicit CLI values may only match, not override, conflicting config values. Tests must prove profile
  selection changes only the policy input, every trace carries the exact pins, missing documents
  fail closed, malformed attacker output marks the evaluation row invalid and never counts as an
  attacker miss/privacy success, and attacker cache identity changes with input/model/prompt/schema. Add an
  `aci_interactive_ranker` suite in `src/bench/registry.py` whose held-out document IDs are exactly
  those frozen in the training record; do not use `primary_utility`, whose corpora are not present in
  the current ACI-only environment.

- [ ] **Step 4: Run the benchmark implementation tests**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_bench_ranker_policy.py \
    src/cloak/tests/test_bench_llm_privacy.py \
    src/cloak/tests/test_bench_runner.py \
    src/cloak/tests/test_bench_metrics_privacy_report.py \
    src/cloak/tests/test_bench_schema.py
  ```

- [ ] **Step 5: Exercise the adapter without external calls**

  Run the benchmark with `--stub-remote` and mocked structured attacker responses for one held-out
  ACI document at lambda zero and one nonzero profile. Inspect both traces and assert they contain
  the same frozen occurrences, different profile identities, complete policy/fixed action records,
  and distinct config hashes only where the profile is part of the config.

- [ ] **Step 6: Commit the benchmark integration**

  ```bash
  git add src/bench/ranker_policy.py src/bench/runner.py src/bench/privacy.py \
          src/bench/metrics.py src/bench/schema.py src/bench/registry.py \
          scripts/run_roundtrip_benchmark.py \
          src/cloak/tests/test_bench_ranker_policy.py \
          src/cloak/tests/test_bench_llm_privacy.py
  git commit -m "feat: benchmark conditional ranker privacy"
  ```

## Task 15: Run Full Training and Matched-Privacy Evaluation

**Files:**
- Modify: `research-wiki/training/2026-07-22-RL-ranker-v5-interactive-count-conditioned.md`
- Create at runtime: `results/ranker_v2/train/`
- Create at runtime: `results/ranker_v2/eval/`

**Consumes:** Passed smoke, performance gate, and benchmark integration.

**Produces:** Conditional checkpoint, fixed lambda-zero control, per-profile held-out utility and
attacker outcomes, and an empirically honest verdict.

- [ ] **Step 1: Freeze the full command in the training record**

  Replace only run-size values established by the preflight/performance gate. Keep reward weights,
  lambda profiles, thresholds, reader/task/extractor pins, entropy/KL coefficients, scheduler
  budget, and acceptance tests frozen across the conditioned model and controls.

- [ ] **Step 2: Launch the full 67-document run unbuffered**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -u scripts/train_interactive_ranker.py \
    --environment results/ranker_v2/environment/ranker-env.json \
    --utility-artifact results/ranker_v2/qa/aci-full.utility \
    --count-state results/ranker_v2/reward/count-reward-state.json \
    --lambda-menu results/ranker_v2/preflight/lambda-menu.json \
    --threshold-manifest results/ranker_v2/preflight/threshold-manifest.json \
    --utility-cache results/ranker_v2/cache/utility-results.jsonl \
    --out-dir results/ranker_v2/train \
    --seed 0 \
    --fixed-lambda-zero-control
  ```

  Add only the pre-registered run-size/concurrency flags from the training record. Do not improvise
  a threshold or reward change after seeing results.

- [ ] **Step 3: Evaluate every supported profile independently**

  Measure whole-task utility on `out_final`, QA context/delivered components, action modes,
  `P_count`, type/provenance behavior, collision diagnostics, attacker success on `doc_p`, and
  leak-through on `out_final`. Run the fixed lambda-zero utility control under identical settings.

  Copy the pre-registered model and prompt pins from the training record into
  `results/ranker_v2/eval/evaluation-config.json`, run `--dry-run` first for each profile, and inspect
  every manifest. Then request explicit approval for model-bearing attacker calls with exact
  item/profile/stage request counts. The approved live workflow is:

  ```bash
  .venv/bin/python - <<'PY' > results/ranker_v2/eval/profile-names.txt
  import json
  menu = json.load(open("results/ranker_v2/preflight/lambda-menu.json"))
  for profile in menu["profiles"]:
      print(profile["name"])
  PY

  while IFS= read -r profile; do
    PYTHONPATH=src:scripts .venv/bin/python -u scripts/run_roundtrip_benchmark.py \
      --config results/ranker_v2/eval/evaluation-config.json \
      --suite aci_interactive_ranker \
      --substitutor ranker-v2 \
      --privacy-setting "profile=${profile}" \
      --ranker-checkpoint results/ranker_v2/train/conditioned.pt \
      --ranker-environment results/ranker_v2/environment/ranker-env.json \
      --ranker-count-state results/ranker_v2/reward/count-reward-state.json \
      --ranker-lambda-menu results/ranker_v2/preflight/lambda-menu.json \
      --lambda-profile "${profile}" \
      --output-dir "results/ranker_v2/eval/${profile}"
  done < results/ranker_v2/eval/profile-names.txt
  ```

  The config hash and profile name must appear in every trace. Never change the config after
  inspecting profile outcomes.

- [ ] **Step 4: Adjudicate only at matched realized privacy**

  Use the frozen document-bootstrap procedure and utility non-inferiority margin. Compare methods
  only at matched attacker-measured privacy. Do not normalize each model to equal count score and do
  not adjust lambda after attacker results. If count score rises while realized privacy worsens,
  report proxy failure and terminate count-based privacy claims for that region.

- [ ] **Step 5: Complete the training record**

  Set `status: done` only after successful generated artifacts have been inspected. Record all
  measured wins and regressions, unsupported corpora/types/profiles, failures, costs, artifact
  hashes, exact commands, and predecessor/successor links. Distinguish implemented, rejected,
  unvalidated, and failed behavior.

- [ ] **Step 6: Run final review and verification**

  Run the focused full suite from Task 13, the benchmark suite from Task 14, then:

  ```bash
  .venv/bin/python -m compileall -q src/cloak/train src/bench scripts/train_interactive_ranker.py \
    scripts/run_ranker_preflight.py scripts/run_roundtrip_benchmark.py
  git diff --check
  ```

  The coordinating Claude session performs the critical whole-change review (it reviewed every
  task and holds the full context). Request one additional independent codex
  `adversarial-review` pass if the coordinator's review leaves unresolved
  architecture/correctness risk.

- [ ] **Step 7: Commit final evidence**

  ```bash
  git add research-wiki/training/2026-07-22-RL-ranker-v5-interactive-count-conditioned.md
  git commit -m "docs: record ranker v2 realized-privacy results"
  ```

---

## Task Review Gates

Every SDD task review must answer these questions before the next task begins:

1. **Contract:** Does the diff implement only the task's declared public interfaces and preserve
   the global constraints?
2. **Identity:** Are document, occurrence, decision, action, assertion, and artifact identities
   stable and content-addressed where required?
3. **Credit:** Can any fixed decision, missing assertion, duplicate occurrence, counterfactual pair,
   or count fallback receive unintended gradient?
4. **Pins and cost:** Does every cache/scorer result bind all semantically relevant pins, and are
   uncached external calls explicit rather than hidden in tests or constructors?
5. **Evidence:** Did the exact focused test command pass, and did any real-data task inspect actual
   generated outputs rather than only process status?

## Final Definition of Done

The current phase (Tasks 1–13) is done when the statements through the vertical-slice item hold;
the last two statements belong to the deferred evaluation phase (Tasks 14–15). RL-v2 as a whole
is complete only when all statements below are true:

- The current 67-document ACI environment passes the complete-count and occurrence-mapping gates.
- The current QA artifact exposes only policy decisions to RL and preserves fixed-only evidence as
  residual utility.
- One conditional policy receives an approved finite lambda menu and satisfies exact identity at
  initialization.
- Utility gradients use linked/residual/fallback routing; count gradients use exact legal-menu
  expectations; tested counterfactual terms substitute at the same `1/G` weight.
- BC and ExIt are utility-only, and ExIt winners are reverified before cloning.
- The calibration pool, switch points, menu, diagnostic spike, threshold manifest, utility cache,
  scheduler report, checkpoints, and training record are hash-linked and reproducible.
- A representative real ACI slice produces usable `doc_p`, `out_p`, `out_final`, both utility
  families where present, residual/fallback credit, count gradients, and measured counterfactuals.
- The full conditioned run and fixed lambda-zero control are evaluated per supported profile, with
  attacker outcomes and utility compared only at matched realized privacy.
- The exact verification commands, counts, examples, failures, and caveats are recorded in the
  training record. Interfaces, mocks, and synthetic fixtures are never presented as full
  implementation evidence.
