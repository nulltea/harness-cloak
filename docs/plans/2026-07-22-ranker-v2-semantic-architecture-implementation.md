---
type: plan
status: current
created: 2026-07-22
updated: 2026-07-22
tags: [rl, ranker, semantic-privacy, candidate-attention, selected-action-memory,
       additive-controller, implementation]
companion: [docs/specs/RL/interactive-ranker-v2.md,
            docs/specs/RL/interactive-ranker-v2-decision-log.md,
            docs/specs/RL/interactive-ranker-v2-diagnostics.md,
            docs/specs/RL/ranker-v2-architecture.md,
            docs/specs/RL/ranker-v2-architecture-decision-log.md,
            docs/specs/qa-builder-v2.md,
            docs/dev/logs/2026-07-22-rl-v2-implementation-inventory.md,
            docs/plans/2026-07-22-core-rl-v2-qa-builder-alignment.md]
supersedes: docs/plans/2026-07-22-interactive-ranker-v2-remaining-implementation.md
---

# Ranker v2 Semantic Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking. Each task gets one implementation agent and the skill's review
> gates before the next task begins.

**Goal:** Replace the current count-bearing FiLM/GRU actor with the selected semantic ranker:
candidate-conditioned full-document utility, an independently pretrained semantic log-count head,
selected-action cross-attention memory, and one explicit additive lambda controller.

**Architecture:** Keep the current interactive RL-v2 environment, utility cache, structured credit,
counterfactual scheduler, lambda menu, and training loop as integration infrastructure. Add a new
`SemanticRankerPolicy` beside the current `ConditionalRankerPolicy`, which remains an explicit
diagnostic baseline. The selected policy shares one frozen, revision-pinned BioClinical ModernBERT
encoder for document and ordered source/candidate representations, but has separate trainable
utility and privacy projections and strict gradient boundaries.

**Tech Stack:** Python 3.12, PyTorch 2.10, Transformers 5.12, pinned
`thomas-sounack/BioClinical-ModernBERT-base` revision
`c3648aa87af95837c809e6f0c5f85d08160db437`, JSON manifests, tensor caches loaded with
`torch.load(..., weights_only=True)`, pytest, the existing QA-builder-v2 round-trip environment.

## Global Constraints

- Execute against the current `main` implementation at or after commit `28631e8`. Do not assume
  transcript context; read the five companion RL architecture/reward specs before Task 1.
- Preserve unrelated local changes, including `.superpowers/sdd/task-3-report.md` and uncommitted
  specification files. Never reset, clean, or overwrite user work.
- Follow the root `AGENTS.md` model table exactly. Implementation agents use GPT-5.6 Terra High by
  default; Terra Medium is allowed only for mechanical artifact/CLI edits. GPT-5.6 Sol High is
  review-only and must never implement. Never enable fast/priority service mode.
- Do not create commits unless the user explicitly requests them. Task boundaries and review reports
  replace the writing-plans skill's usual commit step.
- Use TDD: add the focused failing test, run it and observe the intended failure, implement the
  smallest production change, then run the focused suite and task review gate.
- Do not install or upgrade a production dependency.
- Do not make an uncached remote-task, reader, teacher, extractor, or attacker call without explicit
  user approval. Cache-only preflights and local GPU work may proceed.
- Before any GPU run, verify `.venv/bin/python -c 'import torch; print(torch.cuda.is_available())'`
  prints `True`. Only one GPU process may run at once.
- Before any run expected to exceed ten minutes, pass the repository performance gate with the
  standardized prompt in `scripts/harness/perf_gate.md`; record estimated wall time and measured
  device utilization in the run log.
- Use the pinned encoder revision with remote model code disabled. A different revision is a new
  experiment and must not reuse representation caches.
- The encoder remains frozen and in evaluation mode. No adapter, LoRA, partial unfreeze, or full
  fine-tuning is part of this plan.
- The actor never receives raw/normalized count, predicted privacy as a generic feature, authored
  level index, menu size, lambda identity, profile identity, count provenance, or QA routing IDs.
- True own-profile counts supervise the privacy head and exact local objective only. Model-proposed
  counts remain separately reported experimental targets. Missing/default counts never become
  implicit training labels.
- Preserve deterministic first-occurrence decision order. Alternative orders are diagnostics only.
- Preserve one decision and one gradient record per normalized surface. Repeated detector spans are
  occurrence-level evidence and are all rewritten by the selected decision, but do not duplicate
  count score or selected-action memory rows.
- Preserve the current QA assertion vector, fixed denominator, linked/residual/fallback credit,
  in-place counterfactual substitution, complete round-trip cache, and realized-privacy honesty
  rules.
- The current ACI artifacts may validate implementation mechanics. Because the pinned encoder saw
  ACI during pretraining, ACI results cannot select the encoder or support out-of-corpus
  generalization claims. Architecture promotion remains explicitly unvalidated until a non-ACI
  clinical slice exists.
- A passing synthetic suite is not completion. The final task runs the smallest representative
  real-data vertical slice and inspects generated artifacts; if required remote cache entries are
  absent, report the system as implemented but end-to-end unvalidated.

## Current WIP and Migration Boundary

The implementation session must begin by confirming these facts in the checkout:

| Current surface | State | Treatment in this plan |
|---|---|---|
| `ConditionalRankerPolicy` in `src/cloak/train/ranker.py` | count, authored position, menu size, profile embeddings, FiLM, GRU | retain unchanged as `legacy-film-gru` diagnostic baseline |
| `CountReward` and `expected_count_loss` in `src/cloak/train/count_reward.py` | type-normalized direct-count fallback | retain for fallback; do not feed it into the semantic actor |
| `interactive_ranker.py` trajectory, structured credit, counterfactual, cache, and training loop | implemented WIP | adapt through protocol-compatible semantic outputs |
| `scripts/train_interactive_ranker.py` | instantiates only current policy and count artifact | add explicit architecture selection and semantic artifacts |
| ACI ranker environment and QA artifact | real 67-document artifacts | reuse without rerunning ACI detection |

The selected implementation must not rename the old policy to look compliant. It introduces a new
checkpoint version and architecture pin, and requires explicit `semantic-v1` selection at every
CLI entry point.

## Normative Coverage Matrix

| Requirement | Implementing tasks | Required evidence |
|---|---:|---|
| own-profile count targets isolated from actor | 1, 6, 7 | schema test and zero count-gradient into utility |
| pinned shared frozen encoder and caches | 2 | cache identity, overlap deduplication, exact revision |
| ordered source/candidate relation | 2, 3 | reversal and candidate-swap tests |
| semantic log-count distribution | 3 | profile-held-out NLL, ordering, calibration report |
| candidate-conditioned complete-document context | 4 | local/long-range/target-marker intervention tests |
| selected-action cross-attention | 5 | empty-zero, permutation, selectivity, no-history/GRU arms |
| explicit additive lambda controller | 6 | exact lambda-zero identity and local monotonicity |
| finite controller-scale balance | 7 | opposing utility/count gradients on `alpha_raw` |
| BC, ExIt, structured hybrid RL integration | 7 | focused trainer tests and checkpoint round trip |
| architecture fitness and shortcut gates | 8 | frozen diagnostic manifest and arm report |
| representative real-data vertical slice | 9 | inspected cache/checkpoint/report artifacts |

## File Map

### Create

- `src/cloak/train/profile_count.py` — strict own-profile target artifact and exact-target runtime.
- `src/cloak/train/ranker_representation.py` — pinned frozen encoder, pair serialization, document
  token bank, and content-addressed cache.
- `src/cloak/train/ranker_privacy.py` — privacy projection/head, grouped dataset split, losses,
  metrics, and checkpoint loading.
- `src/cloak/train/semantic_ranker.py` — context readout, selected-action memory, utility tower,
  additive controller, and policy facade.
- `src/cloak/train/ranker_architecture_diagnostics.py` — shortcut, context, history, and order
  diagnostics with immutable reports.
- `scripts/build_profile_count_targets.py` — publish the selected architecture's count-target state.
- `scripts/build_ranker_representation_cache.py` — build frozen document/relation caches.
- `scripts/train_ranker_privacy_head.py` — pretrain and evaluate the semantic privacy head.
- `scripts/run_ranker_architecture_spike.py` — run matched representation/context/history arms.
- `src/cloak/tests/test_profile_count.py`
- `src/cloak/tests/test_ranker_representation.py`
- `src/cloak/tests/test_ranker_privacy.py`
- `src/cloak/tests/test_semantic_ranker.py`
- `src/cloak/tests/test_ranker_architecture_diagnostics.py`
- `research-wiki/training/2026-07-22-RL-ranker-v6-semantic-privacy-context.md` — write before the
  first real optimizer run.

### Modify

- `src/cloak/train/interactive_ranker.py` — replay semantic outputs and enforce gradient ownership.
- `scripts/train_interactive_ranker.py` — load semantic artifacts and emit semantic checkpoints.
- `src/cloak/tests/test_interactive_ranker.py` — semantic objective, warm-start, and checkpoint tests.
- `src/cloak/tests/test_conditional_ranker.py` — keep the old policy tests explicitly labelled as
  legacy baseline tests; do not weaken them.
- `src/cloak/train/ranker_diagnostics.py` and `src/cloak/tests/test_ranker_diagnostics.py` — include
  semantic-head, alpha, shortcut, and order metrics in run reports.

## Task 1: Publish Strict Own-Profile Count Targets

**Files:**
- Create: `src/cloak/train/profile_count.py`
- Create: `scripts/build_profile_count_targets.py`
- Create: `src/cloak/tests/test_profile_count.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ProfileActionTarget:
    decision_id: str
    action_id: str
    profile_id: str
    runtime_type: str
    mode: str
    log_count: float | None
    profile_score: float
    grounding_status: str | None
    source_family: str | None

class ProfileCountTargets:
    @classmethod
    def from_artifact(cls, payload: Mapping) -> "ProfileCountTargets": ...
    def action_scores(self, decision_id: str, action_ids: Sequence[str]) -> torch.Tensor: ...
    def target_rows(self, *, eligible_only: bool = True) -> tuple[ProfileActionTarget, ...]: ...
    def selected_document_score(self, action_vector: Mapping[str, str]) -> float: ...

def build_profile_count_targets(environment: Mapping, *, strict: bool) -> dict: ...
```

- [ ] **Step 1: Write failing target-construction tests**

  Add fixtures with two profiles, KEEP, two levels, and placeholder. Assert:

  ```python
  assert targets.action_scores("d1", ("keep", "fine", "coarse", "placeholder")).tolist() == [
      0.0,
      math.log(10) / math.log(100),
      1.0,
      1.0,
  ]
  assert targets.target_rows()[0].profile_id == "drug:aspirin"
  ```

  Also assert that strict construction rejects missing counts, missing grounding, duplicate profile
  IDs, nonmonotone authored ladders, multi-level all-one denominators, fallback/default provenance,
  and an action from a different decision. Assert that a one-level profile scores its sole level
  `1.0` and is tagged `singleton_profile_normalization`, even when its count is one.

- [ ] **Step 2: Observe the intended failure**

  Run:

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q src/cloak/tests/test_profile_count.py
  ```

  Expected: collection fails because `cloak.train.profile_count` does not exist.

- [ ] **Step 3: Implement the immutable artifact**

  Parse only `ranker_selectable` decisions from the raw `ranker-v2-environment-v2` artifact. For
  every level use the action's own `count` and `count_grounding`; never use `aset`,
  `coarseness_rank`, action order as a count, or `CountReward.type_references`. Compute natural-log
  targets against the maximum admitted level log count in that profile. Emit:

  ```json
  {
    "artifact_version": "ranker-v2-profile-count-targets-v1",
    "environment_hash": "sha256:...",
    "gate_mode": "strict",
    "decision_actions": {"decision_id": ["action_id"]},
    "action_targets": {"action_id": {"profile_score": 0.5}},
    "profile_tags": {"profile_id": ["singleton_profile_normalization"]},
    "gate_report": {},
    "artifact_hash": "sha256:..."
  }
  ```

  `strict=False` may publish a diagnostic artifact, but every incomplete decision is marked
  `privacy_head_eligible: false`; its level actions receive no semantic-head target and no validated
  exact count gradient. KEEP remains zero and placeholder one. The diagnostic mode may support
  plumbing smoke tests only.

- [ ] **Step 4: Implement and test the CLI**

  The CLI is:

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_profile_count_targets.py \
    --environment results/ranker_v2/environment/ranker-env.json \
    --out results/ranker_v2/reward/profile-count-targets.json \
    --gate-report results/ranker_v2/reward/profile-count-gate.json \
    --strict
  ```

  A failed strict gate writes the report, does not write the target artifact, and exits `2`.
  Canonical JSON hashing excludes `artifact_hash` from its own payload.

- [ ] **Step 5: Run focused verification**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_profile_count.py \
    src/cloak/tests/test_count_reward.py \
    src/cloak/tests/test_ranker_environment.py
  ```

  Expected: all pass; the existing type-normalized fallback tests remain unchanged.

**Review gate:** Confirm no true count, profile score, authored index, or menu size is exposed by a
policy-facing method. `ProfileCountTargets.action_scores` is consumed only by the exact objective,
calibration pool, and diagnostics.

## Task 2: Build the Frozen Shared Representation Store

**Files:**
- Create: `src/cloak/train/ranker_representation.py`
- Create: `scripts/build_ranker_representation_cache.py`
- Create: `src/cloak/tests/test_ranker_representation.py`

**Interfaces:**

```python
ENCODER_ID = "thomas-sounack/BioClinical-ModernBERT-base"
ENCODER_REVISION = "c3648aa87af95837c809e6f0c5f85d08160db437"

@dataclass(frozen=True)
class DocumentTokenBank:
    doc_id: str
    states: torch.Tensor          # [unique_source_tokens, 768], CPU float32
    offsets: torch.Tensor         # [unique_source_tokens, 2], int64
    chunk_membership: tuple[tuple[int, ...], ...]

@dataclass(frozen=True)
class RelationFeatures:
    decision_id: str
    action_id: str
    type_mean: torch.Tensor
    source_mean: torch.Tensor
    candidate_mean: torch.Tensor
    pair: torch.Tensor            # concat(type, source, candidate, candidate-source, source*candidate)
    candidate_only: torch.Tensor  # candidate encoded alone; diagnostics only
    independent_pair: torch.Tensor  # source/candidate separately encoded; diagnostics only

class RankerRepresentationStore:
    @classmethod
    def open(cls, manifest_path: Path) -> "RankerRepresentationStore": ...
    def document(self, doc_id: str) -> DocumentTokenBank: ...
    def relation(self, decision_id: str, action_id: str) -> RelationFeatures: ...
```

- [ ] **Step 1: Write failing serialization and cache tests**

  Use a deterministic stub tokenizer/encoder. Test exact relation rendering:

  ```text
  TYPE: health-condition
  SOURCE: kidney transplant
  CANDIDATE: solid organ transplant
  ```

  KEEP renders the canonical source. Placeholder renders `unspecified health condition`,
  `unspecified drug`, `unspecified medical procedure`, or `unspecified location`; it never embeds
  `<TYPE_N>`, action IDs, counts, or authored indices. Assert that reversing source and candidate
  changes the signed pair feature.

  For a document longer than one chunk, assert each source token offset appears once in the final
  bank even when it appears in two overlapping chunks. Assert that cache identity changes for model
  revision, tokenizer revision, chunk length, overlap, field-serialization version, or source hash.

- [ ] **Step 2: Observe the intended failure**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_ranker_representation.py
  ```

  Expected: missing module failure.

- [ ] **Step 3: Implement the frozen encoder adapter**

  Load tokenizer and model with:

  ```python
  tokenizer = AutoTokenizer.from_pretrained(ENCODER_ID, revision=ENCODER_REVISION)
  model = AutoModel.from_pretrained(
      ENCODER_ID,
      revision=ENCODER_REVISION,
      trust_remote_code=False,
  )
  model.eval()
  for parameter in model.parameters():
      parameter.requires_grad_(False)
  ```

  Reject a slow tokenizer because offset mappings are required. Use 512 total tokens and 64 source
  tokens of overlap. Exclude model special tokens from the persisted source bank. When an offset is
  encoded in several chunks, average its frozen states once and preserve the set of source chunk
  IDs; never present duplicate overlap rows to attention.

- [ ] **Step 4: Implement field-aware ordered relation pooling**

  Build ordinary-text field spans before tokenization and map offsets back to type/source/candidate
  masks. Reject a relation if any nonempty field has zero retained tokens. Persist all three masked
  means plus the exact ordered pair feature. KEEP, level, and placeholder use the same path.

  For diagnostic baselines, also encode candidate text alone and source/candidate independently
  through the same frozen checkpoint. Cache these separately versioned vectors; do not derive the
  candidate-only baseline from the joint candidate field, because that field has already attended
  to source and type tokens.

- [ ] **Step 5: Implement the content-addressed store and CLI**

  Write one weights-only tensor file per document and one per unique relation, plus a canonical JSON
  manifest. Validate file SHA-256 before loading. The manifest records environment hash, encoder and
  tokenizer revisions, hidden size `768`, chunk settings, serialization version, offset format,
  document hashes, relation keys, and tensor-file hashes.

  The CLI supports `--doc-id` for a tiny slice and `--cache-only-model` to fail before network access
  when the pinned model is not already available locally.

- [ ] **Step 6: Run focused verification**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_ranker_representation.py \
    src/cloak/tests/test_ranker_environment.py
  ```

  Expected: all pass without downloading or loading the real encoder.

**Review gate:** Inspect the serialized relation text and cache manifest. Search the module for
`count`, `aset`, `authored_level_index`, `profile_score`, and `lambda`; none may be a representation
input. Occurrence offsets are allowed only in the document token bank and later utility features.

## Task 3: Pretrain the Semantic Log-Count Head

**Files:**
- Create: `src/cloak/train/ranker_privacy.py`
- Create: `scripts/train_ranker_privacy_head.py`
- Create: `src/cloak/tests/test_ranker_privacy.py`
- Modify: `src/cloak/train/ranker_diagnostics.py`
- Modify: `src/cloak/tests/test_ranker_diagnostics.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class PrivacyPrediction:
    mu_log_count: torch.Tensor
    sigma_log_count: torch.Tensor

class SemanticPrivacyHead(nn.Module):
    def forward(self, pair_features: torch.Tensor) -> PrivacyPrediction: ...

def profile_normalize_predictions(
    prediction: PrivacyPrediction,
    modes: Sequence[str],
) -> torch.Tensor: ...

def privacy_training_loss(
    prediction: PrivacyPrediction,
    log_count_targets: torch.Tensor,
    profile_slices: Sequence[slice],
    profile_score_targets: torch.Tensor,
    *,
    rho: float,
    gamma: float,
) -> Mapping[str, torch.Tensor]: ...
```

- [ ] **Step 1: Write failing distribution and isolation tests**

  Assert `sigma >= 1e-4`, finite NLL, tied counts excluded from pairwise ranking, complete-menu
  normalization, exact KEEP/placeholder endpoints, and one-level handling. Backpropagate the privacy
  loss and assert gradients reach only the privacy projection/head, never a supplied utility
  projection.

  Add grouped split tests proving every `profile_id` belongs to exactly one of train/dev/test and
  every supported runtime type is reported per split.

- [ ] **Step 2: Observe the intended failure**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q src/cloak/tests/test_ranker_privacy.py
  ```

- [ ] **Step 3: Implement the head and losses**

  Use a separate trainable privacy projection and a two-output MLP:

  ```python
  projected = privacy_projection(pair_features)
  mu_raw, sigma_raw = privacy_head(projected).chunk(2, dim=-1)
  mu = F.softplus(mu_raw).squeeze(-1)
  sigma = 1e-4 + F.softplus(sigma_raw).squeeze(-1)
  ```

  The loss returns named scalar terms `nll`, `pairwise_rank`, `profile_huber`, and `total`.
  `pairwise_rank` compares all unequal target pairs within one profile using logistic ranking;
  `profile_huber` compares normalized predicted means against Task 1 targets. Exclude KEEP and
  placeholder from learned count metrics.

- [ ] **Step 4: Implement baselines and held-out metrics**

  Evaluate against:

  1. authored-position plus action-mode/runtime-type MLP;
  2. action-mode/runtime-type-only MLP;
  3. independently encoded candidate-only frozen vector with the same downstream head budget;
  4. stable train-profile mean, which cannot memorize held-out profile IDs.

  Report NLL, median absolute log error, median multiplicative error, 95% interval coverage,
  within-menu pairwise accuracy, Spearman rank correlation, profile-relative calibration error,
  and selected-action profile-relative regret by runtime type and grounding status.

- [ ] **Step 5: Implement checkpoint and CLI contracts**

  The checkpoint version is `ranker-v2-semantic-privacy-v1`. It binds environment hash,
  profile-target artifact hash, representation manifest hash, encoder revision, split manifest,
  projection/head dimensions, loss weights, seeds, and metric report hash. Loading fails before
  state-dict mutation on any mismatch.

  The CLI runs three explicit seeds, accepts `--max-steps` for a smoke, and writes checkpoints and
  reports under `results/ranker_v2/architecture/privacy/`. It never calls the remote task or reader.

- [ ] **Step 6: Run focused verification**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_ranker_privacy.py \
    src/cloak/tests/test_ranker_diagnostics.py
  ```

**Review gate:** The report must expose profile-held-out results and every baseline. A seen-profile
win is insufficient. The privacy checkpoint may proceed to integration only when the frozen
diagnostic manifest records the relative promotion verdict; ACI/doc generalization is not claimed.

## Task 4: Implement Candidate-Conditioned Full-Document Context

**Files:**
- Create: `src/cloak/train/semantic_ranker.py`
- Create: `src/cloak/tests/test_semantic_ranker.py`

**Interfaces introduced in this task:**

```python
@dataclass(frozen=True)
class DecisionTokenFeatures:
    role_ids: torch.Tensor
    relative_position_ids: torch.Tensor
    document_position_ids: torch.Tensor
    occurrence_token_indices: tuple[tuple[int, ...], ...]

class CandidateContextReadout(nn.Module):
    def forward(
        self,
        token_bank: DocumentTokenBank,
        decision_features: DecisionTokenFeatures,
        utility_relations: torch.Tensor,
    ) -> torch.Tensor: ...  # [actions, context_dim]
```

- [ ] **Step 1: Write failing complete-context tests**

  Use a synthetic token bank where the relevant evidence can be placed locally or in the last
  chunk. Assert:

  - changing candidate relation changes attention and utility context;
  - changing distant evidence changes only the candidate that queries it;
  - every mapped occurrence contributes to the target branch;
  - removing target markers changes the target summary;
  - duplicated overlap membership does not double a token's attention mass;
  - a document token bank is encoded once and reused across all candidates.

- [ ] **Step 2: Observe the intended failure**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_semantic_ranker.py -k context
  ```

- [ ] **Step 3: Implement frozen target/position features**

  For each source token assign one role: current-decision occurrence, another controlled
  occurrence, or ordinary token. Use signed nearest-current-occurrence distance buckets:

  ```text
  <=-128, -127..-64, -63..-32, -31..-16, -15..-8, -7..-4,
  -3..-2, -1, 0, +1, +2..+3, +4..+7, +8..+15,
  +16..+31, +32..+63, +64..+127, >=+128
  ```

  Use 16 equal-width original-document position bins. These trainable embeddings derive only from
  frozen source coordinates. QA IDs and dependency fields are prohibited.

- [ ] **Step 4: Implement the three context branches**

  Project `r_U(j,a)` into the candidate query. Compute:

  ```text
  c_target = attention(query, pooled occurrence states + occurrence positions)
  c_local  = attention(query, unique tokens from target-containing chunks)
  c_global = attention(query, all unique source tokens)
  c_context = projection([c_target; c_local; c_global])
  ```

  Use one multi-head attention layer per branch with the same configured head count. Do not add a
  document Transformer, retrieval policy, or encoder fine-tuning.

- [ ] **Step 5: Add explicit local baselines**

  Implement `context_mode` values `local-cls-mean`, `target-bidirectional`, and
  `full-candidate-attention`. They share frozen inputs and utility-head budget so Task 8 can run a
  matched ablation. Production configuration accepts only `full-candidate-attention` after its gate
  passes.

- [ ] **Step 6: Run focused verification**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_semantic_ranker.py -k 'context or token_features'
  ```

**Review gate:** Confirm no document truncation, no unconditional mean over repeated occurrences,
and no candidate-independent full-document summary in the selected arm.

## Task 5: Implement Selected-Action Cross-Attention Memory

**Files:**
- Modify: `src/cloak/train/semantic_ranker.py`
- Modify: `src/cloak/tests/test_semantic_ranker.py`

**Interfaces introduced in this task:**

```python
@dataclass(frozen=True)
class SelectedActionRecord:
    utility_relation: torch.Tensor
    action_mode_id: int
    runtime_type_id: int
    source_position_pool: torch.Tensor

class SelectedActionMemory(nn.Module):
    def forward(
        self,
        candidate_queries: torch.Tensor,
        records: tuple[SelectedActionRecord, ...],
    ) -> torch.Tensor: ...
```

- [ ] **Step 1: Write failing memory invariants**

  Assert:

  ```python
  assert torch.equal(memory(queries, ()), torch.zeros_like(expected))
  assert torch.allclose(memory(queries, records), memory(queries, tuple(reversed(records))))
  ```

  Add intervention cases where changing one semantically relevant prior selected action changes the
  current candidate history, while changing an unrelated record does not materially change it under
  a hand-set attention fixture. Assert one repeated-surface decision creates one record even when it
  has several occurrences.

- [ ] **Step 2: Observe the intended failure**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_semantic_ranker.py -k memory
  ```

- [ ] **Step 3: Implement the selected memory**

  Construct each row from utility relation, mode embedding, runtime-type embedding, and a
  deterministic mean of all original occurrence-position embeddings. Query with
  `[r_U(j,a); c_context(j,a)]`. Use one `nn.MultiheadAttention` cross-attention block and no memory
  self-attention. Do not add selection-step embeddings. The exact empty-memory result is zero,
  without a learned null token.

- [ ] **Step 4: Implement diagnostic baselines**

  Add `history_mode` values:

  - `none` — exact zero history;
  - `utility-gru` — a corrected GRU over the same `SelectedActionRecord` fields only;
  - `selected-cross-attention` — selected production arm.

  The corrected GRU cannot call legacy `_decision_action_inputs` and cannot receive count,
  authored index, menu size, lambda, profile identity, or QA routing fields.

- [ ] **Step 5: Pin first-occurrence semantics**

  Add a test showing the production walk consumes `RankerDocument.policy_decisions` exactly in
  frozen first-occurrence order. Add diagnostic helpers that replay reverse and seeded orders but
  never change the production sampler's default.

- [ ] **Step 6: Run focused verification**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_semantic_ranker.py -k 'memory or history or order'
  ```

**Review gate:** Inspect every memory field and gradient path. Count/privacy losses must not reach
memory parameters. Memory permutation invariance applies to rows available in the same prefix; it
does not claim that causal prefix membership is order-independent.

## Task 6: Compose the Semantic Policy and Additive Controller

**Files:**
- Modify: `src/cloak/train/semantic_ranker.py`
- Modify: `src/cloak/tests/test_semantic_ranker.py`
- Modify: `scripts/train_interactive_ranker.py`

**Interfaces introduced in this task:**

```python
@dataclass(frozen=True)
class SemanticPolicyState:
    document: RankerDocument
    profile: LambdaProfile
    selected_records: tuple[SelectedActionRecord, ...]

@dataclass(frozen=True)
class ActionDistribution:
    action_ids: tuple[str, ...]
    utility_logits: torch.Tensor
    mu_log_count: torch.Tensor
    sigma_log_count: torch.Tensor
    predicted_privacy: torch.Tensor
    combined_logits: torch.Tensor
    log_probs: torch.Tensor
    count_log_probs: torch.Tensor

class SemanticRankerPolicy(nn.Module):
    def begin_document(self, document: RankerDocument, profile: LambdaProfile) -> SemanticPolicyState: ...
    def distribution(self, state, decision, legal_action_ids, profile) -> ActionDistribution: ...
    def log_probs(self, state, decision, legal_action_ids, profile) -> torch.Tensor: ...
    def advance(self, state, decision, action_id) -> SemanticPolicyState: ...
```

- [ ] **Step 1: Write failing shortcut and controller tests**

  Assert that changing true target counts after representation/privacy inference cannot change
  utility logits, predicted privacy, or combined logits. Assert changing authored indices cannot
  change utility logits or raw privacy predictions. Adding a real candidate may change profile-menu
  normalization and is therefore not an invariance test. Assert changing lambda changes only the
  additive controller term.

  Assert exact identities:

  ```python
  assert torch.equal(dist_zero.combined_logits, dist_zero.utility_logits)
  assert torch.equal(dist_zero.log_probs, torch.log_softmax(dist_zero.utility_logits, 0))
  assert policy.alpha.item() >= 0.0
  ```

  Numerically verify expected predicted privacy is non-decreasing over the supported lambda menu for
  one fixed state/legal menu.

- [ ] **Step 2: Observe the intended failure**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_semantic_ranker.py -k 'policy or lambda or shortcut'
  ```

- [ ] **Step 3: Implement separate utility and privacy projections**

  The frozen `RelationFeatures.pair` feeds two different trainable projections. Load and freeze the
  validated privacy projection/head checkpoint. The utility tower consumes `r_U`, candidate context,
  an explicit projected context-times-relation interaction, mode/type embeddings, and selected
  history. It receives no privacy value or count target.

- [ ] **Step 4: Implement complete-menu privacy normalization**

  Predict every action in the complete decision menu before legal masking. Normalize predicted
  means across all lattice levels in that menu; then set KEEP to zero and placeholder to one. A
  dynamic collision mask may remove an action from the softmax but must not renormalize its siblings'
  predicted privacy scores.

- [ ] **Step 5: Implement the controller and count-gradient view**

  Initialize:

  ```python
  self.alpha_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
  alpha = F.softplus(self.alpha_raw)
  g = math.log1p(profile.value) / math.log1p(self.max_lambda)
  combined = utility_logits + alpha * g * predicted_privacy.detach()
  count_combined = utility_logits.detach() + alpha * g * predicted_privacy.detach()
  ```

  `log_probs` comes from `combined`; `count_log_probs` comes from `count_combined`. At lambda zero,
  bypass arithmetic and return the exact utility tensor before softmax so bitwise identity is
  testable.

- [ ] **Step 6: Add explicit CLI architecture selection**

  Add required `--policy-architecture semantic-v1|legacy-film-gru`. `semantic-v1` requires
  `--representation-manifest`, `--privacy-checkpoint`, and `--profile-count-targets`.
  `legacy-film-gru` requires the old `--count-state`. Never infer architecture from whichever files
  happen to exist.

- [ ] **Step 7: Run focused verification**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_semantic_ranker.py \
    src/cloak/tests/test_conditional_ranker.py
  ```

**Review gate:** Search the selected policy state dict and forward signature. There must be no
profile embedding, FiLM layer, count feature, menu-size feature, authored-position feature, or
legacy GRU. The old policy remains executable only under its explicit baseline name.

## Task 7: Integrate Replay, Hybrid Gradients, Warm Start, and Checkpoints

**Files:**
- Modify: `src/cloak/train/interactive_ranker.py`
- Modify: `src/cloak/tests/test_interactive_ranker.py`
- Modify: `scripts/train_interactive_ranker.py`

**Interface changes:**

```python
@dataclass(frozen=True)
class ReplayedStep:
    decision_id: str
    selected_action_id: str
    legal_action_ids: tuple[str, ...]
    log_prob: torch.Tensor
    log_probs: torch.Tensor
    count_log_probs: torch.Tensor
    utility_logits: torch.Tensor
    predicted_privacy: torch.Tensor
    entropy: torch.Tensor
```

- [ ] **Step 1: Write failing gradient-ownership tests**

  Construct one two-action state where utility prefers KEEP and exact count prefers placeholder.
  Assert:

  ```python
  assert grad_norm(count_loss, policy.utility_parameters()) == 0.0
  assert grad_norm(count_loss, policy.history_parameters()) == 0.0
  assert grad_norm(utility_loss, policy.privacy_parameters()) == 0.0
  assert grad_norm(count_loss, (policy.alpha_raw,)) > 0.0
  assert grad_norm(utility_loss, (policy.alpha_raw,)) > 0.0
  assert torch.sign(alpha_count_grad) != torch.sign(alpha_utility_grad)
  ```

  This opposing-gradient test is load-bearing: a privacy-only `alpha` objective is a failure even if
  aggregate loss decreases.

- [ ] **Step 2: Observe the intended failure**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_interactive_ranker.py -k 'semantic or alpha or gradient_isolation'
  ```

- [ ] **Step 3: Replay complete semantic distributions**

  Extend `TrajectoryPolicy` with `distribution` when available. `replay_trajectory` stores ordinary
  and count-detached distributions from the same state and legal menu. Keep the old protocol path
  for `legacy-film-gru`, but fail if `semantic-v1` returns only scalar selected log-probability.

- [ ] **Step 4: Implement exact profile-target loss**

  Add:

  ```python
  def expected_profile_count_loss(
      replay_steps: Sequence[ReplayedStep],
      targets: ProfileCountTargets,
      lambda_value: float,
      decision_count: int,
      rollout_count: int,
  ) -> torch.Tensor:
      expectations = []
      for step in replay_steps:
          exact = targets.action_scores(step.decision_id, step.legal_action_ids).to(
              step.count_log_probs
          )
          expectations.append(torch.sum(step.count_log_probs.exp() * exact))
      return -torch.stack(expectations).sum() * (
          lambda_value / decision_count / rollout_count
      )
  ```

  The legacy architecture continues to call `expected_count_loss`; semantic-v1 calls only the new
  function.

- [ ] **Step 5: Correct BC and ExIt warm start**

  BC and ExIt run at lambda zero and update utility projection, context readout, selected-action
  memory, and utility head. They do not update the frozen privacy head or `alpha_raw`, and they clone
  each verified trajectory once rather than once per profile identity. Preserve current remote-call,
  cache, verification, and passenger-action semantics.

- [ ] **Step 6: Version checkpoints and architecture pins**

  Publish `ranker-v2-semantic-policy-v1`. Bind environment, utility artifact, profile targets,
  representation manifest, privacy checkpoint, lambda menu, threshold manifest, encoder revision,
  context/history modes, feature schema, controller transform, optimizer, RNG, and schedule. Reject
  a legacy checkpoint before loading any tensor into a semantic policy and vice versa.

- [ ] **Step 7: Extend run diagnostics**

  Report utility/count/entropy/KL gradient norms separately for utility parameters, history
  parameters, privacy parameters, and `alpha_raw`; alpha value/gradient by lambda; predicted versus
  exact selected privacy; singleton/provenance strata; and lambda-zero exact-identity failures.

- [ ] **Step 8: Run focused verification**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_interactive_ranker.py \
    src/cloak/tests/test_semantic_ranker.py \
    src/cloak/tests/test_utility_credit.py \
    src/cloak/tests/test_counterfactual_credit.py
  ```

**Review gate:** Reconstruct the complete hybrid objective from named components and verify the
counterfactual term still substitutes in place at the same `1/G` weight. Inspect autograd results,
not only parameter-name filters.

## Task 8: Implement and Run the Architecture Fitness Spike

**Files:**
- Create: `src/cloak/train/ranker_architecture_diagnostics.py`
- Create: `scripts/run_ranker_architecture_spike.py`
- Create: `src/cloak/tests/test_ranker_architecture_diagnostics.py`
- Modify: `docs/specs/RL/interactive-ranker-v2-diagnostics.md`

**Produces:** One immutable diagnostic report comparing all approved arms under matched data,
parameter budgets, seeds, and cache state.

- [ ] **Step 1: Write failing diagnostic-contract tests**

  Require report sections:

  ```text
  relation_representation
  semantic_privacy_transfer
  context_readout
  action_history
  metadata_shortcut
  lambda_controller
  decision_order
  operational_cost
  contamination_boundary
  promotion_verdict
  ```

  Assert the report refuses to compare arms with different profile splits, utility cases, seeds,
  trainable-head budgets, encoder revisions, or cache manifests.

- [ ] **Step 2: Implement matched arms**

  Run:

  - relation: candidate-only, independent shared-model bi-encoder, selected ordered joint pair;
  - context: local CLS/mean, bidirectional target pooling, selected full-document attention;
  - history: none, corrected utility-only GRU, selected-action cross-attention;
  - shortcut: metadata-only actor, legacy FiLM/GRU actor, selected semantic actor;
  - order: first-occurrence, reverse, and three seeded alternative walks.

  Project every representation/context/history arm to the same output width and use the same
  downstream diagnostic head. Report module-specific trainable counts separately; do not pad an arm
  with unused parameters or add learned layers merely to equalize totals.

- [ ] **Step 3: Implement intervention measurements**

  Include context swaps, candidate swaps, source/candidate reversal, relevant/irrelevant prior-action
  interventions, memory-row permutation, target-marker ablation, full-document ablation, non-middle
  utility cases, and count-metadata mutation after privacy inference. Geometry is diagnostic only;
  promotion uses held-out outcome prediction, ordering, calibration, and intervention behavior.

- [ ] **Step 4: Implement relative promotion rules**

  Across three seeds, require the selected arm's paired bootstrap 95% confidence interval for
  improvement over both primary baselines to exclude zero on the load-bearing metric. It must not
  regress local-only utility, violate shortcut/invariance tests, or exceed the recorded operational
  budget. If selected-action memory does not beat no history on multi-decision utility, choose no
  history; do not promote the GRU automatically.

  Absolute empirical thresholds that the diagnostics spec intentionally leaves unregistered cause
  `promotion_verdict: NEEDS_THRESHOLD_REGISTRATION`, not an invented pass. The report contains the
  observed distributions needed for a subsequent frozen threshold-manifest amendment.

- [ ] **Step 5: Record the ACI contamination boundary**

  The script labels ACI context results `development_only_encoder_contaminated`. It may validate
  plumbing and reject a broken architecture, but cannot emit `PROMOTE` for encoder selection or
  out-of-corpus context generalization without a separately frozen non-ACI clinical manifest.

- [ ] **Step 6: Run focused tests**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_ranker_architecture_diagnostics.py \
    src/cloak/tests/test_ranker_diagnostics.py
  ```

- [ ] **Step 7: Run the smallest cached/local spike**

  First use two ACI documents, all relation rows, three seeds, and cached utility outputs. Before a
  GPU run, execute the GPU sanity command and confirm no other GPU process is active. Write outputs
  under `results/ranker_v2/architecture/spike/` and inspect at least one passing and one failing
  example from every measurement family.

**Review gate:** The implementation can proceed to a smoke optimizer step with a diagnostic-only
verdict, but no architecture-promotion or generalization claim is allowed until the manifest's
absolute thresholds and non-ACI boundary are satisfied.

## Task 9: Run the Representative Semantic-Ranker Vertical Slice

**Files:**
- Create: `research-wiki/training/2026-07-22-RL-ranker-v6-semantic-privacy-context.md`
- Modify: `scripts/train_interactive_ranker.py`
- Modify: `src/cloak/tests/test_interactive_ranker.py`

**Produces:** A real-data semantic checkpoint, component reports, and an honest validation status.

- [ ] **Step 1: Write the training record before running**

  Follow the repository training-experiment schema. Pin the two-document ACI smoke slice, exact
  environment/QA/profile-target/representation/privacy/menu/threshold hashes, model revision,
  context/history arms, optimizer, rollout count, counterfactual budget, cache-only policy, success
  criteria, artifact paths, and known ACI contamination. Set `status: planned` and `result: pending`.

- [ ] **Step 2: Add a cache-only semantic CLI integration test**

  Invoke `bc`, `exit-collect`, and `train` with `--policy-architecture semantic-v1` and stubbed
  semantic artifacts. Assert a missing remote result exits with the existing machine-readable
  cache-miss contract and launches no call. Assert all checkpoint pins survive save/load.

- [ ] **Step 3: Build the two-document frozen representation cache**

  Use the same two documents selected by the existing preflight artifact. Verify GPU availability,
  build unbuffered, record wall time/peak memory, and inspect manifest hashes, token counts,
  occurrence coverage, relation counts, and cache files.

- [ ] **Step 4: Run the privacy-head smoke**

  Train on the full eligible profile dataset with `--max-steps` set to the smallest value that
  produces finite train/dev metrics and a loadable checkpoint. This is a mechanics smoke, not the
  three-seed promotion run. Inspect predicted count/interval/order examples for KEEP, one fine level,
  one coarse level, and placeholder.

- [ ] **Step 5: Run cache-only BC, ExIt, and one hybrid step**

  Use two documents, at least two rollouts, one lambda-zero and one nonzero episode, one scheduled
  counterfactual when cached, and the semantic policy. If cache entries are absent, stop and report
  exact remote-task and context-reader work counts; request approval before any live call.

- [ ] **Step 6: Inspect required artifacts**

  Report:

  - document, decision, occurrence, action, and relation counts;
  - context and delivered assertion counts and accepted scores;
  - ordinary and counterfactual cache hits/misses;
  - utility/count/entropy/KL terms;
  - gradient norms by utility/history/privacy/alpha parameter group;
  - exact lambda-zero identity;
  - predicted versus exact selected privacy by action mode/type/provenance;
  - selected-action memory row counts and order-sensitivity diagnostics;
  - checkpoint and report hashes.

- [ ] **Step 7: Run final focused and broad verification**

  ```bash
  PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
    src/cloak/tests/test_profile_count.py \
    src/cloak/tests/test_ranker_representation.py \
    src/cloak/tests/test_ranker_privacy.py \
    src/cloak/tests/test_semantic_ranker.py \
    src/cloak/tests/test_ranker_architecture_diagnostics.py \
    src/cloak/tests/test_interactive_ranker.py \
    src/cloak/tests/test_conditional_ranker.py \
    src/cloak/tests/test_ranker_environment.py \
    src/cloak/tests/test_count_reward.py \
    src/cloak/tests/test_counterfactual_credit.py \
    src/cloak/tests/test_utility_credit.py \
    src/cloak/tests/test_utility_cache.py \
    src/cloak/tests/test_lambda_menu.py
  ```

  Then run `git diff --check` and the repository-configured formatter/linter commands already used
  by adjacent modules. Do not add a formatter if none is configured.

- [ ] **Step 8: Complete the training record honestly**

  Set `status: done` only if a real semantic optimizer step completed and produced nontrivial utility
  components, finite opposing alpha gradients, and a loadable checkpoint. Otherwise keep the record
  `planned` or `running` and state the exact unvalidated boundary. Record wins and regressions; do
  not call a synthetic-only or cache-miss run complete.

**Review gate:** Completion means the selected architecture runs through the real QA/RL pipeline on
the representative slice. It does not mean the semantic architecture is promoted, counts measure
privacy, ACI establishes generalization, or matched-realized-privacy evaluation is complete.

## Task Review Gates

After each task, the coordinator reviews:

1. exact compliance with the task interfaces and normative specs;
2. absence of actor metadata shortcuts and forbidden gradient paths;
3. preservation of legacy baseline behavior and existing RL-v2 reward contracts;
4. focused test evidence including the observed RED state;
5. artifact identity and fail-closed behavior;
6. no unrelated edits or silent scope expansion.

Critical whole-change review uses GPT-5.6 Sol High only after implementation is complete. If an
independent Claude review is requested, use `claude-opus-4-8` High through `delegate-to-claude` by
default; reserve `claude-fable-5` High for demonstrated unresolved architecture/research issues.
Neither reviewer implements fixes unless the user separately delegates implementation.

## Final Definition of Done

- The selected `semantic-v1` policy has no count/position/menu/lambda shortcut inputs.
- The frozen encoder revision, document chunks, offsets, relation serialization, and caches are
  content-addressed and validated.
- The privacy head predicts a calibrated log-count distribution and remains frozen during hybrid
  RL.
- Exact profile-relative count shaping reaches only `alpha_raw`; utility reaches utility/history
  parameters and may oppose count through `alpha_raw`.
- Lambda zero is bitwise identical to the utility policy before masking.
- Candidate-conditioned document attention observes target, local, repeated-occurrence, and global
  evidence without truncation or overlap duplication.
- Selected-action cross-attention is count-blind, row-permutation invariant, and evaluated against
  no-history and corrected-GRU baselines.
- First-occurrence order is the production walk; alternative-order sensitivity is reported.
- Existing structured utility, counterfactual substitution, cache, and lambda schedule tests remain
  green.
- A representative real-data optimizer step and artifact inspection either succeeds or is explicitly
  reported unvalidated with the exact missing external work.
- No claim exceeds the evidence: count remains shaping, ACI remains encoder-contaminated development
  data, and final method comparison still requires held-out attackers at matched realized privacy.
