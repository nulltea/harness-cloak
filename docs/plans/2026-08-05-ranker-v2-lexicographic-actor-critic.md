---
type: plan
status: current
created: 2026-08-05
updated: 2026-08-05
tags:
  - rl
  - ranker-v2
  - lexicographic-rl
  - primal-dual
  - actor-critic
  - low-rank-adapters
  - freeze-policy
companion:
  - docs/specs/RL/ranker-v2-architecture.md
  - docs/specs/RL/ranker-v2-architecture-decision-log.md
  - docs/specs/RL/interactive-ranker-v2.md
  - docs/specs/RL/interactive-ranker-v2-decision-log.md
  - docs/research/ranker-v2-trainable-multi-objective-rl-review.md
  - docs/html/interactive-ranker-v2.html
---

# Ranker v2 Lexicographic Actor-Critic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failed additive count controller with a trainable utility-first/count-second
policy that can adapt post-encoder semantics, preserve the selected utility policy exactly at
`lambda-0`, and serve a checkpoint-pinned finite menu of user utility/privacy preferences.

**Architecture:** Implement three tiers. Tier 1 is the permanently frozen BioClinical ModernBERT
substrate. Tier 2 is the candidate/context/history utility branch trained by BC, verified ExIt, and
utility-only structured RL, then frozen. Tier 3 is a private lexicographic semantic side path:
rank-4 low-rank deltas over an explicit allowlist of frozen Tier-2 maps plus a zero-initialized
normalized GELU-16 residual head. Utility and count policy gradients update Tier 3 only; separate
utility/count critics provide detached baselines; a learned finite-menu setting embedding conditions
the residual head; per-document-setting training-only dual variables enforce each setting's explicit
utility slack. The first run is a four-document cache-only strict-setting mechanism test, not a larger
training or deployment claim.

**Tech Stack:** Python 3.12, PyTorch, `dataclasses`, JSON/JSONL artifacts, `pytest`, the existing
ranker-v2 reward cache, and the existing `scripts/train_interactive_ranker.py` entry point.

## Global Constraints

- Preserve first-occurrence decision order, dynamic injectivity masking, and per-surface decisions.
- Preserve `semantic-v1` checkpoint loading for historical evaluation; do not silently reinterpret
  an additive checkpoint as a lexicographic checkpoint.
- The frozen utility-reference branch must remain bit-identical after lexicographic training.
- Count values, normalized count scores, authored level positions, numeric utility slack, and dual
  values must not enter Tier 1, Tier 2, Tier-3 semantic features, or frozen utility logits.
- The served policy requires one registered `lambda_setting_id`. `lambda-0` bypasses Tier 3 and is
  bit-identical to the frozen utility branch. Positive settings condition the residual head through
  a learned finite-menu embedding; arbitrary floats and unregistered settings fail closed.
- The pinned lambda menu contains three to five ordered settings. `lambda-0` is utility identity,
  `lambda-1` is strict count-second with zero slack, and later settings carry strictly increasing
  explicit document-level utility slacks. Lambda never multiplies count reward or an advantage.
- The Tier-3 actor receives count information only through sampled policy gradients.
- Tier-3 adapters are rank `4`, zero-product initialized, and restricted to the pinned
  post-encoder semantic target allowlist. No adapter may target the clinical encoder or utility head.
- The residual head is exactly `LayerNorm(4608, affine=False) -> Linear(4608, 16, bias=True) -> add
  Embedding(lambda_setting_id, 16) -> LayerNorm(16, affine=False) -> GELU -> Linear(16, 1,
  bias=False)`, with the final map and the `lambda-0` row initialized to exact zero.
- The utility critic consumes detached Tier-2 features; the count critic consumes detached Tier-3
  features. They share no trainable parameters.
- Positive settings use one direct nonnegative dual per `(document, lambda_setting_id)`. The first
  mechanism run activates only `lambda-1` with $\tau(\lambda_1)=0$; `lambda-0` remains the immutable
  control. PPO clip is $0.2$, entropy coefficient $0.01$, and previous-policy KL coefficient $0.01$.
- Preserve the fixed policy-role utility denominator, linked/residual/fallback credit, and in-place
  counterfactual substitution. Do not standardize utility or count advantages by batch standard
  deviation.
- The count target remains exact profile-relative return-to-go from the frozen count artifact.
  This is a temporary experimental shortcut; the final reward provider is a separately trained,
  frozen k-anonymity estimator behind the same interface. Changing providers requires a new reward
  pin and retraining, not a change to the lambda-conditioned actor interface.
- No deterministic selector, additive `alpha`, privacy-head controller, count-to-gain edge, tie
  hinge, cycle projection, gap scaling, utility softcap, fixed-reference KL, or profile-sensitivity
  regularizer may be active in the new forward or loss path.
- Use dollar-sign LaTeX notation in every Markdown file.
- Use the host `.venv`; run the smallest real-data slice before the four-document run.
- Do not claim held-out transfer, realized privacy, or production readiness from the mechanism run.

## File Map

- `src/cloak/ranker/semantic.py`: expose the frozen Tier-1 substrate, canonical selected-action
  records, Tier-2 utility features, and utility logits without changing legacy behavior.
- `src/cloak/ranker/lexicographic_actor.py`: low-rank adapter primitives, the Tier-3 semantic side
  stack, residual head, two critics, and the served lexicographic policy wrapper.
- `src/cloak/ranker/lexicographic_objective.py`: PPO surrogates, critic losses, document dual state,
  and actor/dual objective composition.
- `src/cloak/ranker/lexicographic_artifacts.py`: utility-reference and lexicographic checkpoint
  schemas, hashing, validation, save, and load.
- `src/cloak/ranker/interactive.py`: served-policy sampling/replay, structured advantages,
  counterfactual substitution, and document-group training orchestration.
- `scripts/train_interactive_ranker.py`: CLI wiring and run orchestration only.
- `src/cloak/tests/test_lexicographic_actor.py`: architecture and gradient ownership tests.
- `src/cloak/tests/test_lexicographic_objective.py`: numerical objective and dual tests.
- `src/cloak/tests/test_lexicographic_artifacts.py`: artifact fail-closed tests.
- `src/cloak/tests/test_interactive_ranker.py`: integrated trajectory, credit, and training tests.
- `src/cloak/tests/test_train_interactive_ranker.py`: CLI contract tests.
- `research-wiki/training/2026-08-05-RL-ranker-v8-lexicographic-primal-dual.md`: spec-before-run
  record and measured result.

---

### Task 1: Separate the Frozen Substrate from the Utility Stack

**Files:**
- Modify: `src/cloak/ranker/semantic.py`
- Modify: `src/cloak/tests/test_semantic_ranker.py`

**Interfaces:**
- Consumes: current `SemanticRankerPolicy.distribution`, document/relation caches,
  `SemanticPolicyState`, legal action IDs, canonical selected-action records, and dynamic masking.
- Produces:

```python
@dataclass(frozen=True)
class CanonicalSelectedActionRecord:
    decision_id: str
    action_id: str
    pair_features: torch.Tensor
    action_mode_id: int
    runtime_type_id: int
    document_position_ids: torch.Tensor
    occurrence_token_indices: tuple[tuple[int, ...], ...]

@dataclass(frozen=True)
class FrozenSemanticSubstrate:
    action_ids: tuple[str, ...]
    relation_bank: torch.Tensor
    document_token_bank: torch.Tensor
    target_masks: tuple[torch.Tensor, ...]
    local_masks: tuple[torch.Tensor, ...]
    occurrence_positions: tuple[torch.Tensor, ...]
    action_mode_ids: torch.Tensor
    runtime_type_ids: torch.Tensor
    selected_action_records: tuple[CanonicalSelectedActionRecord, ...]

@dataclass(frozen=True)
class UtilityActionStack:
    action_ids: tuple[str, ...]
    substrate: FrozenSemanticSubstrate
    utility_features: torch.Tensor
    utility_logits: torch.Tensor

def semantic_substrate(
    self,
    state: SemanticPolicyState,
    decision: RankerDecision,
    legal_action_ids: Sequence[str],
) -> FrozenSemanticSubstrate: ...

def utility_action_stack(
    self,
    substrate: FrozenSemanticSubstrate,
) -> UtilityActionStack: ...

def tier1_parameter_names(policy: SemanticRankerPolicy) -> tuple[str, ...]: ...
def tier2_parameter_names(policy: SemanticRankerPolicy) -> tuple[str, ...]: ...

def freeze_utility_branch(policy: SemanticRankerPolicy) -> FrozenBranchManifest: ...
```

`utility_features` is the current 4,608-dimensional `utility_inputs`, restricted to legal actions
in `action_ids` order. `utility_logits` is the current utility-head output before every historical
additive controller transform. The substrate contains detached frozen encoder outputs and canonical
raw policy history, not Tier-2 projected `utility_relation` or memory rows. Refactor
`SemanticPolicyState.selected_records` to store `CanonicalSelectedActionRecord`; the legacy utility
memory path must reconstruct its current `SelectedActionRecord` values from the canonical record
without numerical change. Dataclasses do not make tensors immutable;
constructors must detach substrate tensors and tests must reject tensors carrying a gradient
function. `freeze_utility_branch` freezes all Tier-1/Tier-2 parameters and returns their sorted
names, shapes, dtypes, and state hash for the architecture pin.

- [ ] **Step 1: Write an extraction parity test**

Add a test that constructs a semantic policy with deterministic weights, computes the current
legacy `distribution`, calls `semantic_substrate` then `utility_action_stack`, and asserts:

```python
assert utility.action_ids == distribution.action_ids
torch.testing.assert_close(utility.utility_logits, distribution.utility_logits, rtol=0, atol=0)
assert utility.utility_features.shape == (len(utility.action_ids), 4608)
assert utility.substrate.action_ids == utility.action_ids
```

Repeat after advancing one selected action so selected-action cross-attention is exercised. Assert
that a canonical selected-action record is identical before branch-specific projection and that
the reconstructed legacy memory row is bit-identical to the pre-refactor row.

- [ ] **Step 2: Run the parity test and confirm it fails**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
  src/cloak/tests/test_semantic_ranker.py -k 'semantic_substrate or utility_action_stack'
```

Expected: failure because the substrate/utility-stack interfaces do not exist.

- [ ] **Step 3: Extract the substrate and utility stack without duplicating semantics**

Split the existing path at the frozen-encoder/post-encoder boundary. `semantic_substrate` owns
cached token/relation states, masks, action metadata, and canonical history records.
`utility_action_stack` owns the existing Tier-2 relation projection, candidate-conditioned context
readout, selected-action memory projection/attention, interaction assembly, and utility head.
Refactor legacy `distribution` to call both methods, then continue through its unchanged historical
controller code. Preserve exact tensor ordering and operations so legacy logits remain bit-identical.

- [ ] **Step 4: Add freeze-contract tests**

Assert that `freeze_utility_branch` freezes every Tier-1/Tier-2 parameter used to produce
`utility_features` and `utility_logits`, leaves historical controller/privacy parameters unchanged,
returns stable sorted names and hash, and raises if the semantic schema is unknown. Assert there is
no trainable path from a scalar loss on `utility_features` or `utility_logits` into Tier 1/Tier 2
after freezing.

- [ ] **Step 5: Run semantic regression tests**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
  src/cloak/tests/test_semantic_ranker.py
```

Expected: all tests pass; existing `semantic-v1` distributions remain bit-identical in the new
parity fixture.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/cloak/ranker/semantic.py src/cloak/tests/test_semantic_ranker.py
git commit -m "refactor: expose frozen semantic substrate"
```

---

### Task 2: Add the Tier-3 Semantic Side Path, Residual Head, and Critics

**Files:**
- Create: `src/cloak/ranker/lexicographic_actor.py`
- Create: `src/cloak/tests/test_lexicographic_actor.py`

**Interfaces:**
- Consumes: `FrozenSemanticSubstrate`, `UtilityActionStack`, exact legal action IDs, and the frozen
  Tier-2 module registry.
- Produces:

```python
@dataclass(frozen=True)
class LambdaSetting:
    setting_id: str
    ordinal: int
    display_label: str
    utility_slack: float
    mode: Literal["utility-identity", "lexicographic"]

@dataclass(frozen=True)
class LambdaMenu:
    scope: Literal["mechanism", "deployment"]
    settings: tuple[LambdaSetting, ...]
    content_hash: str

    def require(self, setting_id: str) -> LambdaSetting: ...

@dataclass(frozen=True)
class AdapterTargetSpec:
    module_name: str
    in_features: int
    out_features: int
    rank: int = 4

class ZeroProductLowRankDelta(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4) -> None: ...
    def forward(self, inputs: torch.Tensor) -> torch.Tensor: ...

class LexicographicSemanticSideStack(nn.Module):
    def __init__(
        self,
        target_specs: Sequence[AdapterTargetSpec],
    ) -> None: ...

    def forward(
        self,
        frozen_utility_base: SemanticRankerPolicy,
        substrate: FrozenSemanticSubstrate,
    ) -> torch.Tensor: ...

@dataclass(frozen=True)
class LexicographicDistribution:
    lambda_setting_id: str
    action_ids: tuple[str, ...]
    utility_logits: torch.Tensor
    lexicographic_features: torch.Tensor
    residual_logits: torch.Tensor
    logits: torch.Tensor
    log_probs: torch.Tensor
    entropy: torch.Tensor

class LexicographicResidualHead(nn.Module):
    def __init__(
        self,
        lambda_menu: LambdaMenu,
        feature_dim: int = 4608,
        hidden_dim: int = 16,
    ) -> None: ...
    def forward(
        self,
        action_features: torch.Tensor,
        lambda_setting_id: str,
    ) -> torch.Tensor: ...

class MenuValueCritic(nn.Module):
    def __init__(
        self,
        lambda_menu: LambdaMenu,
        feature_dim: int = 4608,
        hidden_dim: int = 16,
    ) -> None: ...
    def forward(
        self,
        action_features: torch.Tensor,
        lambda_setting_id: str,
    ) -> torch.Tensor: ...

class LexicographicRankerPolicy(nn.Module):
    policy_architecture = "lexicographic-side-r4-gelu16-lambda-v1"

    def __init__(
        self,
        utility_base: SemanticRankerPolicy,
        adapter_targets: Sequence[AdapterTargetSpec],
        lambda_menu: LambdaMenu,
    ) -> None: ...

    def begin_document(self, document: RankerDocument) -> SemanticPolicyState: ...

    def distribution(
        self,
        state: SemanticPolicyState,
        decision: RankerDecision,
        legal_action_ids: Sequence[str],
        lambda_setting_id: str,
    ) -> LexicographicDistribution: ...

    def advance(
        self,
        state: SemanticPolicyState,
        decision: RankerDecision,
        action_id: str,
    ) -> SemanticPolicyState: ...

    def utility_reference_distribution(...) -> UtilityReferenceDistribution: ...
```

The public served wrapper must expose:

```python
def rank(
    document: RankerDocument,
    policy: LexicographicRankerPolicy,
    lambda_setting_id: str,
) -> Mapping[str, str]: ...
```

It validates the ID against the checkpoint-pinned menu before the first policy forward, holds it
constant for the complete sequential walk, and records the stable ID, display label, and slack in
the report. It does not accept an arbitrary numeric lambda, a per-decision override, or a separate
utility-slack argument.

The adapter target allowlist is fixed by the architecture specification:

```text
utility_projection
context_readout.token_projection
context_readout.query_projection
context_readout.target_attention.{q,k,v,out}
context_readout.local_attention.{q,k,v,out}
context_readout.global_attention.{q,k,v,out}
context_readout.context_projection
memory.record_projection
memory.query_projection
memory.cross_attention.{q,k,v,out}
```

Resolve every name once at construction, verify dimensions, and reject missing, duplicate, extra,
encoder, utility-head, or historical-controller targets. Treat each attention Q/K/V/output map as
a separate logical target even though PyTorch stores Q/K/V in packed `in_proj_weight`; use a
branch-local functional attention call with $W+\Delta W$, not an in-place packed-weight mutation.

Tier 3 recomputes the post-encoder semantic path with effective maps $W+\Delta W$ while Tier 2
continues to use frozen $W$. Do not mutate, monkey-patch, or temporarily swap parameters on the
utility policy; branch-local functional calls or explicit wrappers must make concurrent forward
behavior unambiguous. The side stack owns only adapter parameters; it must not register the frozen
utility base as a second child module or duplicate its tensors in the state dict. The policy passes
its single frozen base into the side-stack forward. Both branches consume the same canonical
selected-action records and derive their memory projections independently.

`MenuValueCritic` uses `Linear(4608, 16, bias=True) -> GELU`, mean-pools legal actions, and emits
one scalar through `Linear(16, 1, bias=True)`. Add a detached encoding of the finite-menu setting
before the scalar head. Instantiate `utility_critic` over detached `x_U` and `count_critic` over
detached `x_lex`; no parameter object may be shared.

- [ ] **Step 1: Write zero-initialization and identity tests**

Construct a four-entry deployment fixture and a two-entry mechanism fixture. Reject a mechanism menu
that is not exactly `{lambda-0, lambda-1}`, a deployment menu outside three-to-five entries,
duplicate IDs or ordinals, a non-identity ordinal zero, a missing strict positive zero-slack entry,
negative slacks, or non-monotone positive slacks. Canonical serialization must reproduce
`content_hash` exactly, and a mechanism menu must be marked non-deployable.

Assert every adapter output factor and the final residual weight are exactly zero. For fresh and
advanced policy states:

```python
state = policy.begin_document(document)
served = policy.distribution(state, decision, legal, "lambda-1")
reference = policy.utility_reference_distribution(state, decision, legal)
torch.testing.assert_close(served.utility_logits, reference.logits, rtol=0, atol=0)
torch.testing.assert_close(served.residual_logits, torch.zeros_like(served.residual_logits), rtol=0, atol=0)
torch.testing.assert_close(served.logits, reference.logits, rtol=0, atol=0)
```

Also assert the Tier-3 pre-head feature equals `x_U` at zero delta. This is a mechanism invariant,
not merely a distribution-level coincidence.

For `lambda-0`, assert exact identity before and after arbitrary Tier-3 optimizer steps:

```python
identity = policy.distribution(state, decision, legal, "lambda-0")
torch.testing.assert_close(identity.logits, reference.logits, rtol=0, atol=0)
torch.testing.assert_close(identity.log_probs, reference.log_probs, rtol=0, atol=0)
```

- [ ] **Step 2: Run the new tests and confirm they fail**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
  src/cloak/tests/test_lexicographic_actor.py
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement zero-product low-rank deltas**

Use rank `4`. Initialize one factor with the module's standard small random initialization and the
output factor to exact zero. Unit tests must show: initial delta output is exactly zero; both factors
have a gradient after a two-step sequence (the zero output factor moves first); effective-map output
equals the frozen map at initialization; and no base parameter is registered in the actor optimizer.

- [ ] **Step 4: Implement the Tier-3 semantic side stack**

Reuse the semantic operation graph exposed in Task 1. Apply deltas only at the pinned target maps.
Preserve target/local/global masks, repeated-occurrence aggregation, legal action ordering, and
selected-action memory semantics. Add intervention tests where changing relevant context or a prior
selected action changes `x_lex`, while an irrelevant prior action does not.

- [ ] **Step 5: Implement the exact residual architecture**

Implement:

```python
self.layers = nn.Sequential(
    nn.LayerNorm(4608, elementwise_affine=False),
    nn.Linear(4608, 16, bias=True),
    # Add the active learned 16-D lambda-setting embedding here.
    nn.LayerNorm(16, elementwise_affine=False),
    nn.GELU(),
    nn.Linear(16, 1, bias=False),
)
nn.init.zeros_(self.layers[-1].weight)
```

Implement the setting embedding outside `nn.Sequential` so its row lookup, row-zero invariant, and
checkpoint schema are explicit. Add it to the first linear output before `LayerNorm(16)`. Reject
feature dimensions other than 4,608. Do not add count, dual, authored-position, numeric slack, or
continuous lambda inputs. `lambda-0` must bypass the complete residual path rather than relying only
on a zero embedding.

- [ ] **Step 6: Implement the single served policy and internal reference replay**

For positive settings, the served path computes
`utility_logits.detach() + residual_logits(x_lex, lambda_setting_id)`. For `lambda-0`, return the
utility distribution directly and execute no Tier-3 module. The utility-reference method executes
Tier 2 only and exists for reference building and parity audits. Detach the Tier-1 substrate and
utility logits, but do not detach Tier-3 features or the active positive setting row before the
residual head. Preserve exact action order and masks.

- [ ] **Step 7: Add architecture and gradient tests**

Test all of the following:

1. utility-reference logits are unchanged after an optimizer step on Tier 3;
2. utility and count-shaped synthetic losses both reach the residual head and at least one adapter;
3. neither loss reaches Tier 1 or Tier 2;
4. utility-critic loss reaches only the utility critic from detached `x_U`;
5. count-critic loss reaches only the count critic from detached `x_lex`;
6. the target registry resolves exactly the pinned allowlist and emits the exact trainable count;
7. changing action order and applying the corresponding inverse permutation preserves each
   critic value;
8. count/action-position/numeric-slack/dual tensors are absent from every Tier-3 forward signature;
9. changing positive setting IDs can change residual logits, but can never change frozen utility
   logits or legal masks;
10. unsupported settings fail closed and `lambda-0` receives exactly zero Tier-3 gradient;
11. a constant additive shift to all action features does not produce an action-index shortcut.

- [ ] **Step 8: Run actor tests**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
  src/cloak/tests/test_lexicographic_actor.py \
  src/cloak/tests/test_semantic_ranker.py
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 2**

```bash
git add src/cloak/ranker/lexicographic_actor.py \
  src/cloak/tests/test_lexicographic_actor.py
git commit -m "feat: add lexicographic semantic side path"
```

---

### Task 3: Implement PPO Surrogates, Critics, and Document Duals

**Files:**
- Create: `src/cloak/ranker/lexicographic_objective.py`
- Create: `src/cloak/tests/test_lexicographic_objective.py`

**Interfaces:**
- Consumes: current/old selected-action log probabilities, detached utility/count advantages,
  entropy, old-policy distributions, critic predictions/targets, document IDs, and utility
  references.
- Produces:

```python
@dataclass(frozen=True)
class ActorObjectiveTerms:
    total: torch.Tensor
    utility: torch.Tensor
    count: torch.Tensor
    entropy: torch.Tensor
    previous_policy_kl: torch.Tensor

@dataclass(frozen=True)
class CriticObjectiveTerms:
    utility: torch.Tensor
    count: torch.Tensor

class DocumentDualState(nn.Module):
    def value(self, doc_id: str, lambda_setting_id: str) -> torch.Tensor: ...
    def project_nonnegative_(self) -> None: ...

def clipped_policy_loss(
    selected_log_probs: torch.Tensor,
    old_selected_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor: ...

def compose_actor_objective(
    utility_policy_loss: torch.Tensor,
    count_policy_loss: torch.Tensor,
    entropy: torch.Tensor,
    previous_policy_kl: torch.Tensor,
    dual_value: torch.Tensor,
    beta: float,
    eta: float,
) -> ActorObjectiveTerms: ...

def compose_critic_objective(
    utility_predictions: torch.Tensor,
    utility_targets: torch.Tensor,
    count_predictions: torch.Tensor,
    count_targets: torch.Tensor,
) -> CriticObjectiveTerms: ...

def dual_loss(
    dual_value: torch.Tensor,
    utility_reference: torch.Tensor,
    observed_utility: torch.Tensor,
    utility_slack: float,
) -> torch.Tensor: ...
```

`utility_slack` is resolved from the active `LambdaSetting` by the training orchestrator. It is not
accepted from a free scalar CLI/config field, and the objective record must retain the setting ID
and menu hash that supplied it.

- [ ] **Step 1: Write analytical PPO tests**

Use one positive and one negative advantage to verify the unclipped and clipped branches against
hand-computed values. Assert utility and count losses are computed separately and are never
concatenated or jointly standardized.

- [ ] **Step 2: Write dual direction and isolation tests**

Initialize one `(document, positive setting)` dual to zero. Verify a positive violation produces a negative derivative with
respect to the dual parameter, so gradient descent raises it; verify slack produces the opposite
direction after the dual is positive; verify `project_nonnegative_` clamps negative values to zero.
Assert `dual_loss.backward()` reaches no actor or critic parameter, one setting's update cannot
change another setting's dual, and requesting a dual for `lambda-0` fails closed.

- [ ] **Step 3: Run objective tests and confirm they fail**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
  src/cloak/tests/test_lexicographic_objective.py
```

Expected: import failure because the module does not exist.

- [ ] **Step 4: Implement the losses in native units**

Use `torch.nn.functional.smooth_l1_loss(..., beta=1.0)` for each critic. Validate finite scalar
inputs, nonnegative coefficients, equal tensor shapes, unique document IDs, and
`0 < clip_epsilon < 1`. Detach advantages, critic targets, utility references, observed utility in
the dual loss, and the dual value in the actor objective.

- [ ] **Step 5: Add the required total actor equation**

Implement exactly:

```python
total = (
    count_policy_loss
    + dual_value.detach() * utility_policy_loss
    - beta * entropy
    + eta * previous_policy_kl
)
```

The KL is between the current served policy and the frozen previous-policy snapshot for the same
legal menu. It is not KL to BC, ExIt, or the frozen utility-reference branch.

- [ ] **Step 6: Run objective and gradient tests**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
  src/cloak/tests/test_lexicographic_objective.py \
  src/cloak/tests/test_lexicographic_actor.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/cloak/ranker/lexicographic_objective.py \
  src/cloak/tests/test_lexicographic_objective.py
git commit -m "feat: add lexicographic primal dual losses"
```

---

### Task 4: Integrate Served Trajectories and Structured Credit

**Files:**
- Modify: `src/cloak/ranker/interactive.py`
- Modify: `src/cloak/tests/test_interactive_ranker.py`

**Interfaces:**
- Consumes: `LexicographicRankerPolicy`, existing reward scoring,
  `ProfileCountTargets`, linked/residual/fallback utility credit, and counterfactual scheduler/cache.
- Produces:

```python
@dataclass(frozen=True)
class LexicographicTrajectory:
    doc_id: str
    lambda_setting_id: str
    steps: tuple[SampledStep, ...]
    action_vector: Mapping[str, str]

@dataclass(frozen=True)
class LexicographicReplayedStep:
    lambda_setting_id: str
    decision_id: str
    selected_action_id: str
    legal_action_ids: tuple[str, ...]
    selected_log_prob: torch.Tensor
    old_selected_log_prob: torch.Tensor
    log_probs: torch.Tensor
    old_log_probs: torch.Tensor
    entropy: torch.Tensor
    detached_utility_features: torch.Tensor
    detached_lexicographic_features: torch.Tensor
    count_reward: torch.Tensor
    count_return_to_go: torch.Tensor

@dataclass(frozen=True)
class LexicographicDocumentUpdate:
    doc_id: str
    lambda_setting_id: str
    trajectories: tuple[LexicographicTrajectory, ...]
    actor_terms: ActorObjectiveTerms
    critic_terms: CriticObjectiveTerms
    dual_loss: torch.Tensor
    observed_utility: float
    utility_reference: float
    count_score: float
    dual_value_before: float
    dual_value_after: float
    gradient_norms: Mapping[str, Mapping[str, float]]

def sample_lexicographic_trajectory(
    policy: LexicographicRankerPolicy,
    document: RankerDocument,
    *,
    lambda_setting_id: str,
    greedy: bool,
    generator: torch.Generator | None,
) -> LexicographicTrajectory: ...

def replay_lexicographic_trajectory(
    policy: LexicographicRankerPolicy,
    old_policy: LexicographicRankerPolicy,
    document: RankerDocument,
    trajectory: LexicographicTrajectory,
    profile_targets: ProfileCountTargets,
) -> tuple[LexicographicReplayedStep, ...]: ...

def count_returns_to_go(
    step_rewards: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]: ...

def train_lexicographic_document_group(
    policy: LexicographicRankerPolicy,
    old_policy: LexicographicRankerPolicy,
    document: RankerDocument,
    *,
    lambda_setting_id: str,
    rollouts: int,
    utility_artifact: Mapping[str, object],
    utility_reference: DocumentUtilityReference,
    profile_targets: ProfileCountTargets,
    cache: UtilityCache,
    dual_state: DocumentDualState,
    actor_optimizer: torch.optim.Optimizer,
    utility_critic_optimizer: torch.optim.Optimizer,
    count_critic_optimizer: torch.optim.Optimizer,
    dual_optimizer: torch.optim.Optimizer,
    ppo_clip: float,
    beta: float,
    eta: float,
    lambda_menu: LambdaMenu,
    counterfactual_budget: int,
    endpoint_budget: int,
    seed: int,
    cache_only: bool,
) -> LexicographicDocumentUpdate: ...
```

Do not rename or reinterpret the legacy `SampledTrajectory.lambda_profile` field. The new path gets
new typed records so historical checkpoints and reports remain loadable.

- [ ] **Step 1: Write count return-to-go tests**

For three per-step rewards `(0.1, 0.2, 0.3)`, assert the returns are `(0.6, 0.5, 0.3)` within
floating tolerance. Add an injectivity fixture where an early action removes a later fill and
assert the sampled later reward follows the actual legal trajectory rather than an independent
per-menu expectation.

- [ ] **Step 2: Write utility-credit preservation tests**

Construct linked, residual, and uncovered decisions and assert the served lexicographic path produces
the same provisional utility advantages as the current hybrid path before PPO clipping. Add a
counterfactual-tested pair and assert its pair term substitutes in place with the same $1/G$
weighting; it must not be added as a second loss.

- [ ] **Step 3: Run the focused tests and confirm they fail**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
  src/cloak/tests/test_interactive_ranker.py \
  -k 'lexicographic or count_returns_to_go'
```

Expected: failures because the new records/functions do not exist.

- [ ] **Step 4: Implement served-policy sampling and replay**

Reuse the existing first-occurrence walk, injectivity mask, fill claiming, and `policy.advance`
logic. Validate one `lambda_setting_id` before the walk and hold it fixed through every decision.
Freeze a deep-copied old served policy once per document-group update. Record current and old
probabilities under the identical setting, prefix, and legal menu. Replay the frozen utility reference
synchronously for parity and $b_d$ diagnostics; it is never sampled for actor gradients. On every
advance, append one canonical selected-action record and let Tier 2/Tier 3 derive independent memory
projections from it.

Add a balanced-setting scheduler for deployment-scope menus: one positive setting per document
group, each document visits each positive setting exactly once per cycle, and document-specific
orders are deterministic Latin rotations under the run seed. `lambda-0` is evaluated once per cycle
for identity but never sent to the actor or dual optimizer. The two-entry mechanism menu therefore
trains only `lambda-1`.

- [ ] **Step 5: Assemble actor advantages**

Build utility advantages from existing linked/residual/fallback RLOO. Preserve exact
counterfactual substitution. Build count rewards as
$r^P_{g,j}=p_j(a_{g,j})/|D_d|$ and backward cumulative returns. Subtract detached critic values;
do not standardize either objective.

- [ ] **Step 6: Implement one document-group update**

Use four separate optimizers or parameter groups with disjoint ownership:

```text
utility critic -> utility critic parameters
count critic   -> count critic parameters
actor          -> Tier-3 adapters + residual-head parameters
                 + active positive lambda-setting embedding row
dual           -> current `(document, lambda_setting_id)` direct dual parameter
```

Run critic steps first, actor second, dual third, then project all duals nonnegative. Record raw
returns, advantages, PPO ratios, clipped fractions, gradient norms by family, utility violation,
dual value, greedy utility key, and greedy count score.

- [ ] **Step 7: Add a two-decision synthetic mechanism test**

The fixture must contain:

1. one exact utility tie with different count returns;
2. one count-preferred action with lower utility;
3. a zero-initialized Tier-3 branch.

After several deterministic updates, assert the first decision moves toward the count-preferred
action, the second document dual rises and prevents utility loss, and frozen utility-reference
logits remain bit-identical. Assert an adapter changes after the residual head becomes nonzero.

- [ ] **Step 8: Run integrated tests**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
  src/cloak/tests/test_interactive_ranker.py \
  src/cloak/tests/test_lexicographic_actor.py \
  src/cloak/tests/test_lexicographic_objective.py
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 4**

```bash
git add src/cloak/ranker/interactive.py src/cloak/tests/test_interactive_ranker.py
git commit -m "feat: train lexicographic trajectories"
```

---

### Task 5: Add Utility-Reference and Checkpoint Contracts

**Files:**
- Create: `src/cloak/ranker/lexicographic_artifacts.py`
- Create: `src/cloak/tests/test_lexicographic_artifacts.py`

**Interfaces:**
- Consumes: frozen utility-base checkpoint, frozen environment/representation/utility/count pins,
  training document IDs, pinned rollout seeds, cached action vectors, and measured utility rows.
- Produces:

```python
UTILITY_BASE_VERSION = "ranker-v2-utility-base-v1"
UTILITY_REFERENCE_VERSION = "ranker-v2-utility-reference-v1"
LAMBDA_MENU_VERSION = "ranker-v2-lexicographic-lambda-menu-v1"
LEXICOGRAPHIC_POLICY_VERSION = "ranker-v2-lexicographic-side-policy-v3"
LEXICOGRAPHIC_TRAINING_STATE_VERSION = "ranker-v2-lexicographic-side-training-state-v3"

@dataclass(frozen=True)
class DocumentUtilityReference:
    doc_id: str
    rollout_seeds: tuple[int, ...]
    action_vectors: tuple[Mapping[str, str], ...]
    utility_keys: tuple[str, ...]
    utilities: tuple[float, ...]
    mean_utility: float
    utility_spread: float

def build_utility_reference(
    policy: LexicographicRankerPolicy,
    documents: Sequence[RankerDocument],
    rollout_seeds: Sequence[int],
    utility_artifact: Mapping[str, object],
    utility_cache: UtilityCache,
    artifact_pins: Mapping[str, str],
    *,
    cache_only: bool,
) -> Mapping[str, object]: ...

def import_utility_base_from_semantic_checkpoint(
    first_checkpoint: Path,
    second_checkpoint: Path,
    target_policy: SemanticRankerPolicy,
    artifact_pins: Mapping[str, str],
) -> Mapping[str, object]: ...

def save_lexicographic_checkpoint(
    path: Path,
    policy: LexicographicRankerPolicy,
    lambda_menu: LambdaMenu,
    artifact_pins: Mapping[str, str],
    architecture_pin: str,
) -> None: ...

def load_lexicographic_checkpoint(
    path: Path,
    policy: LexicographicRankerPolicy,
    expected_artifact_pins: Mapping[str, str],
    expected_architecture_pin: str,
    expected_lambda_menu_hash: str,
) -> Mapping[str, object]: ...

def save_lexicographic_training_state(
    path: Path,
    *,
    policy: LexicographicRankerPolicy,
    dual_state: DocumentDualState,
    optimizers: Mapping[str, torch.optim.Optimizer],
    epoch: int,
    epoch_reports: Sequence[Mapping[str, object]],
    utility_reference_hash: str,
    training_config: Mapping[str, object],
) -> None: ...

def load_lexicographic_training_state(
    path: Path,
    *,
    policy: LexicographicRankerPolicy,
    dual_state: DocumentDualState,
    optimizers: Mapping[str, torch.optim.Optimizer],
    expected_utility_reference_hash: str,
    expected_training_config: Mapping[str, object],
) -> Mapping[str, object]: ...
```

- [ ] **Step 1: Write utility-reference determinism tests**

Build the same fixture twice with seeds `(0, 1, 2, 3, 4, 5, 6, 7)` plus the greedy vector. Assert
byte-identical canonical JSON, sorted document records, exact utility keys retained separately from
floats, and a stable content hash.

- [ ] **Step 2: Write fail-closed checkpoint tests**

Assert load rejects:

1. a `semantic-v1` additive checkpoint;
2. an altered utility-base hash;
3. an altered representation or reward pin;
4. a checkpoint whose Tier-1 or Tier-2 parameters are trainable;
5. an unknown residual-head schema;
6. a changed adapter rank, target allowlist, target dimension, or initialization schema;
7. a trainable encoder or utility-head parameter in the actor optimizer;
8. a training-state document set that differs from the dual-state document set;
9. a changed lambda-menu hash, setting order, setting mode, or utility slack;
10. a mechanism-scope menu loaded as a deployable policy;
11. a missing or trainable `lambda-0` embedding row;
12. the historical additive `ranker-v2-lambda-menu-v1` schema.

Add an import test proving that `import_utility_base_from_semantic_checkpoint` copies only the
allowlisted Tier-1, Tier-2 semantic/history, and utility-head tensors from a historical `semantic-v1`
checkpoint and copies no privacy, `alpha_raw`, gain, or controller state.

- [ ] **Step 3: Run artifact tests and confirm they fail**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
  src/cloak/tests/test_lexicographic_artifacts.py
```

Expected: import failure because the module does not exist.

- [ ] **Step 4: Implement the utility-reference schema**

Include hashes for the environment, utility artifact, count targets, representation manifest,
utility-base checkpoint, code revision, selected document IDs, rollout seeds, and cache path. A
cache-only build fails on any missing vector; it must not call the remote generator or reader.

- [ ] **Step 5: Implement separate deployment and training checkpoints**

The deployable policy checkpoint contains the frozen Tier-1/Tier-2 policy, frozen-parameter
manifest and state hash, Tier-3 adapter state, residual head, canonical selected-action schema, and
architecture pin. It also contains the canonical deployment-scope lambda menu, its content hash,
and every finite-menu setting-embedding row. The training state additionally contains both critics,
document-setting duals, four optimizer states, RNG state, completed epoch, utility-reference hash,
training config, active-setting schedule, and epoch reports. Do not place dual values, numeric
slacks, or critics in the served policy forward interface.

A mechanism checkpoint may contain a two-entry `scope: mechanism` menu, but it must carry an
explicit `deployable: false` marker and fail closed in the deployment loader. Changing menu labels,
order, size, modes, or slacks changes the menu hash and requires a new policy checkpoint.

`import_utility_base_from_semantic_checkpoint` is an explicit migration, not a permissive loader.
It requires two historical pre-RL reference checkpoints from the same seed/config, verifies their
allowlisted Tier-1/Tier-2 utility tensors are bit-identical, writes only those tensors into
`ranker-v2-utility-base-v1`, and records both source hashes. Any allowlisted mismatch blocks the
migration.

- [ ] **Step 6: Run artifact tests**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
  src/cloak/tests/test_lexicographic_artifacts.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 5**

```bash
git add src/cloak/ranker/lexicographic_artifacts.py \
  src/cloak/tests/test_lexicographic_artifacts.py
git commit -m "feat: pin lexicographic training artifacts"
```

---

### Task 6: Add Explicit Lexicographic CLI Workflows

**Files:**
- Modify: `scripts/train_interactive_ranker.py`
- Create: `src/cloak/tests/test_train_interactive_ranker.py`

**Interfaces:**
- Consumes: Tasks 1--5 and existing artifact loaders/reward executor.
- Produces three new subcommands:

```text
migrate-utility-base
build-utility-reference
train-lexicographic
```

- [ ] **Step 1: Write parser contract tests**

Assert `migrate-utility-base` requires two historical pre-RL reference checkpoints and an output
utility-base checkpoint. Assert `build-utility-reference` requires the frozen utility-base checkpoint, output manifest,
environment, representation manifest, utility artifact, count targets, utility cache, explicit
document IDs, and pinned seeds. Assert `train-lexicographic` requires the utility reference and
separate policy/training-state/report outputs plus `--lambda-menu`.

Assert the lexicographic command rejects an arbitrary numeric `--lambda` or standalone
`--utility-slack`, and rejects all superseded controls: `--privacy-checkpoint`, `--alpha-init`,
`--controller-gap-scaling`, `--alpha-utility-routing`,
`--utility-logit-softcap`, `--profile-sensitivity-reg`, `--controller-gain`, `--count-to-gain`,
`--tie-mode`, `--tie-projection-lr`, and `--gain-penalty`.

Assert menu loading rejects an unknown setting ID, duplicate ordinal, a nonzero identity slack,
missing strict zero-slack positive setting, non-monotone positive slacks, a deployment menu outside
three to five settings, and any attempt to publish a mechanism-scope checkpoint. Assert reports and
served inference require `--lambda-setting-id` and record both its stable ID and display label.
Assert the historical additive `ranker-v2-lambda-menu-v1` artifact fails closed rather than being
reinterpreted as a utility-slack menu.
Add an end-to-end served-wrapper test proving that changing the selected registered ID changes only
the Tier-3 condition, while all decisions in one document observe the same ID and `lambda-0` stays
bit-identical to the frozen utility branch.

- [ ] **Step 2: Run parser tests and confirm they fail**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
  src/cloak/tests/test_train_interactive_ranker.py
```

Expected: failure because the subcommands do not exist.

- [ ] **Step 3: Add `migrate-utility-base`**

Load the two source checkpoints through the existing strict historical loader, compare the complete
allowlisted utility tensor set bit-for-bit, reject every mismatch, create a fresh semantic policy,
copy only the allowlisted tensors, freeze them, and save `ranker-v2-utility-base-v1`. Print source
hashes, copied parameter names/count, and excluded controller parameter names/count.

- [ ] **Step 4: Add `build-utility-reference`**

Default seeds to `0,1,2,3,4,5,6,7`, always include the greedy vector, and require `--cache-only` for
the first mechanism artifact. Emit a canonical JSON manifest and print its hash plus document,
vector, cache-hit, and utility-key counts.

- [ ] **Step 5: Add `train-lexicographic`**

Expose these exact defaults:

```text
--max-epochs 8
--rollouts 8
--actor-learning-rate 1e-4
--adapter-rank 4
--residual-hidden-dim 16
--critic-learning-rate 3e-4
--dual-learning-rate 1e-3
--ppo-clip 0.2
--beta 0.01
--eta 0.01
--counterfactual-budget 5
--endpoint-budget 1
--seed 47
```

Require `--cache-only` for the bounded run. Load utility slack only from the pinned menu entry;
never expose it as an independent CLI override. The mechanism run accepts only its two-entry
non-deployable menu and trains strict `lambda-1`; a deployable workflow requires a three-to-five-entry
deployment menu and a complete setting schedule.

- [ ] **Step 6: Record the exact architecture/training pin**

The config must include the policy version, frozen Tier-1/Tier-2 parameter names and state hash,
adapter target allowlist, resolved dimensions, rank, zero-product initialization schema, exact
Tier-3 parameter count, residual schema, critic schemas, dual schema, old-policy KL direction, all
learning rates, clip/coefficient values, rollout and counterfactual budgets, utility-reference hash,
frozen utility-base hash, document IDs, and all reward/representation hashes.
Record the canonical lambda-menu payload and hash, scope, selected setting schedule, active setting
per document group, setting-embedding schema, and document-setting dual keys. The report must make
clear that lambda selects an explicit utility tolerance and does not multiply reward or logits.

- [ ] **Step 7: Run CLI and integration tests**

Run:

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
  src/cloak/tests/test_train_interactive_ranker.py \
  src/cloak/tests/test_lexicographic_artifacts.py \
  src/cloak/tests/test_interactive_ranker.py
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 6**

```bash
git add scripts/train_interactive_ranker.py \
  src/cloak/tests/test_train_interactive_ranker.py
git commit -m "feat: add lexicographic ranker workflows"
```

---

### Task 7: Verify the Complete Mechanism in Tests

**Files:**
- Modify only if a failure exposes a defect in Tasks 1--6.

**Interfaces:**
- Consumes: all new modules and existing semantic/interactive tests.
- Produces: a reviewed, test-clean implementation before any real-data execution.

- [ ] **Step 1: Run the focused lexicographic suite**

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
  src/cloak/tests/test_lexicographic_actor.py \
  src/cloak/tests/test_lexicographic_objective.py \
  src/cloak/tests/test_lexicographic_artifacts.py \
  src/cloak/tests/test_train_interactive_ranker.py
```

Expected: all tests pass.

- [ ] **Step 2: Run the adjacent ranker regression suite**

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q \
  src/cloak/tests/test_semantic_ranker.py \
  src/cloak/tests/test_interactive_ranker.py \
  src/cloak/tests/test_lexicographic_ranker.py \
  src/cloak/tests/test_ranker_architecture_diagnostics.py
```

Expected: all tests pass; the deterministic lexicographic selector remains an offline oracle only.

- [ ] **Step 3: Run the required gradient audit**

Add or invoke one test that prints and asserts the parameter-family gradient matrix:

```text
                   Tier 1  Tier 2  adapters  residual  active setting  U critic  P critic  dual
utility actor         0       0       >0        >0           >0           0         0       0
count actor           0       0       >0        >0           >0           0         0       0
utility critic        0       0        0         0            0          >0         0       0
count critic          0       0        0         0            0           0        >0       0
dual loss             0       0        0         0            0           0         0      >0
```

Require exact zero, not a tolerance, for disconnected parameter families. Because both the adapter
output factors and residual final map start at zero, assert the path-faithful sequence explicitly:
the residual final map receives gradient on update one, adapter output factors become reachable
after that map moves, and adapter input factors become reachable after their output factors move.
Do not misclassify this intentional staged reachability as disconnection.
For `lambda-0`, require exact zero in every Tier-3 and setting-embedding column and bit-identical
served logits before and after arbitrary positive-setting optimizer steps. For positive settings,
only the active setting row may receive a gradient.

- [ ] **Step 4: Run a CPU one-document synthetic training smoke**

Use the synthetic environment/cache fixture and run two epochs. Inspect the emitted policy
checkpoint, training state, epoch JSONL, dual values, gradient norms, and utility-reference identity
count. A passing unit suite without these artifacts is insufficient.

- [ ] **Step 5: Request code review**

Use `superpowers:requesting-code-review` with one critical whole-change reviewer. Require review of
the gradient graph, utility identity, PPO sign/clipping, dual sign/projection, cache-only behavior,
checkpoint separation, and legacy checkpoint compatibility. Apply fixes and rerun Steps 1--4.

- [ ] **Step 6: Commit verification fixes**

Stage only files changed for the lexicographic implementation and use a focused commit message.

---

### Task 8: Run the Bounded Real-Data Mechanism Gate

**Files:**
- Create before execution:
  `research-wiki/training/2026-08-05-RL-ranker-v8-lexicographic-primal-dual.md`
- Produce under the gitignored result root:
  `results/ranker_v2/architecture/lexicographic-primal-dual-s47/`
- Update after execution:
  `research-wiki/training/2026-08-05-RL-ranker-v8-lexicographic-primal-dual.md`

**Interfaces:**
- Consumes frozen inputs:
  - `results/ranker_v2/environment/ranker-env.json`
  - `results/ranker_v2/architecture/representation-full/manifest.json`
  - `results/ranker_v2/reward/profile-count-targets.json`
  - `results/ranker_v2/qa/aci-full.utility`
  - `results/ranker_v2/architecture/count_to_gain/kl-ref-detached-s47.pt`
  - `results/ranker_v2/architecture/count_to_gain/kl-ref-coupled-s47.pt`
  - an isolated copy of `results/ranker_v2/architecture/count_to_gain/cache-coupled-s47.jsonl`
- Produces a utility-base checkpoint, utility-reference manifest, Tier-3 policy checkpoint,
  two-entry mechanism lambda-menu manifest, training state, epoch reports, and the v8 training record.

- [ ] **Step 1: Write the v8 spec-before-run record**

Use the required training-record frontmatter and sections. State the narrow hypothesis:

> A zero-initialized private post-encoder semantic side path trained by count and dual-weighted
> utility PPO surrogates can produce stable greedy count movement on cache-rich documents without
> changing the frozen utility reference or lowering any document's exact utility key.

Pin a two-entry `scope: mechanism` menu containing `lambda-0` utility identity and strict
`lambda-1` with zero slack. Mark every produced policy checkpoint non-deployable. This run tests the
conditioned mechanism only; it does not choose or validate a user-facing three-to-five-setting menu.

List the four documents exactly: `aci/D2N005`, `aci/D2N027`, `aci/D2N031`, and `aci/D2N063`.
State that one seed is a mechanism test and cannot establish generalization or privacy.

- [ ] **Step 2: Pass the performance gate before GPU use**

Run the repository's standardized performance review against the intended command using
`scripts/harness/perf_gate.md`. Confirm one GPU process, batched policy forwards, bounded reader
workers, cache-only reward, and expected wall time. Do not launch if the run duplicates static
encoder/relation-bank work per rollout, rebuilds the adapter registry inside forwards, or leaves the
GPU idle because of Python per-pair loops. The intentional second post-encoder Tier-3 semantic pass
is not a performance defect; it should still batch all legal actions and lockstep rollout states.

- [ ] **Step 3: Migrate the frozen utility base**

Run `migrate-utility-base` over the detached and coupled seed-47 KL-reference checkpoints. Those
references were captured before their respective historical additive training arms; the migration must
prove their Tier-1/Tier-2 utility tensors are bit-identical before importing them. Do not import from
`count_to_gain/coupled-s47.pt` or `count_to_gain/detached-s47.pt`, because those checkpoints include
completed additive training. Save the migrated checkpoint under:

```text
results/ranker_v2/architecture/lexicographic-primal-dual-s47/utility-base.pt
```

Record both source hashes, the complete copied/excluded parameter lists, the resulting utility-base
hash, and its frozen-reference greedy vectors. A source mismatch blocks the run and requires a separate
utility-base reconstruction plan; do not choose one source opportunistically.

- [ ] **Step 4: Build the utility-reference manifest**

Run `build-utility-reference` on the four documents with seeds `0..7`, greedy vectors, and the
isolated cache. The command must be cache-only and produce zero reward calls. Save:

```text
results/ranker_v2/architecture/lexicographic-primal-dual-s47/utility-reference.json
```

Inspect every document record: no missing vector, no pin mismatch, finite mean utility, exact keys
present, and the utility-base hash equal to `utility-base.pt`.

- [ ] **Step 5: Run a one-document, one-epoch real-data smoke**

Use `aci/D2N005`, seed 47, 8 rollouts, the pinned mechanism menu, active setting `lambda-1`, and
cache-only mode. Save under a `smoke/` child directory. Evaluate `lambda-0` through the public served
branch before and after the update. Before continuing, inspect:

1. utility-reference identity failures equal zero;
2. Tier-3 initialization-parity failures equal zero;
3. utility and count actor gradients satisfy the staged residual-head, adapter-output, and
   adapter-input reachability contract when eligible;
4. actor gradients into Tier 1 and Tier 2 are exactly zero;
5. the correct document dual changes in the correct direction;
6. no cache miss or reward call occurs;
7. all policy, training-state, and JSONL artifacts load successfully.
8. the menu hash is identical across config, checkpoint, training state, and report;
9. the active dual key is exactly `(aci/D2N005, lambda-1)` and no `lambda-0` dual exists.

- [ ] **Step 6: Run seed 47 for four documents and eight epochs**

Use the exact defaults from Task 6. Write:

```text
policy.pt
training-state.pt
epochs.jsonl
summary.json
utility-cache.jsonl
```

under `results/ranker_v2/architecture/lexicographic-primal-dual-s47/`. Keep the cache isolated from
historical arms even in cache-only mode so accidental mutation cannot contaminate comparisons.

- [ ] **Step 7: Adjudicate the preregistered mechanism gates**

Pass requires all of the following:

1. zero utility-reference identity failures at every update and snapshot;
2. zero Tier-3 initialization-parity failures before training;
3. utility and count actor terms both satisfy the staged residual-head, adapter-output, and
   adapter-input reachability contract on eligible documents;
4. actor gradients are exactly zero in Tier 1 and Tier 2, and count gradients are exactly zero in
   the utility critic;
5. the adapter allowlist, rank, dimensions, and exact trainable count match the architecture pin;
6. for every document and each of the final three snapshots, the served greedy exact utility key
   is no lower than the frozen-reference key;
7. at least one opportunity-bearing document has strictly positive greedy count-score separation
   in all final three snapshots;
8. no passing document relies only on a mid-run positive peak that later becomes zero;
9. dual values rise on violations, fall or remain at zero under slack, and are not accidentally one
   shared scalar across documents or settings;
10. no superseded controller or selector appears in the architecture pin or gradient report;
11. `lambda-0` remains an exact identity after every positive-setting update;
12. only the active positive setting row receives setting-embedding gradients;
13. zero reward calls and zero unresolved cache misses.

Failure readings are fixed:

- utility-reference or initialization-parity failure: implementation defect; stop;
- disconnected count or utility actor gradient after staged warm-up: objective/representation-route
  defect; stop;
- utility-key regression: primal-dual mechanism failure at $\tau=0$; reject;
- no stable count movement with utility retained: mechanism insufficient; do not enlarge training;
- critic underperformance alone: keep critics report-only and rerun only if actor advantages did not
  depend on them;
- cache miss: artifact/support failure; do not convert the run into a live reward expansion.

- [ ] **Step 8: Escalate to two more seeds only after a pass**

If and only if seed 47 passes every gate, rerun unchanged with seeds 17 and 29. Three-seed success
authorizes a separately planned document-held-out experiment. It does not authorize full-corpus
training, deployment, or a privacy claim.

- [ ] **Step 9: Complete and commit the v8 record**

Fill the record with exact commands, wall time, hardware, artifact hashes, per-document final-three
utility/count/dual trajectories, gradient ownership, critic diagnostics, cache behavior, and the
honest verdict. Commit the record and implementation docs; do not commit gitignored model/results
artifacts.

---

## Explicitly Deferred Work

The following work is not part of this implementation plan and requires a new design or experiment
record after the mechanism gate:

1. document-held-out transfer or a 63/67/240-document training run;
2. attacker-based realized-privacy evaluation and matched-realized-privacy comparison;
3. selecting, preregistering, training, and validating a deployable three-to-five-setting lambda
   menu with nonzero utility slacks;
4. wiring `SCHEMA_NOTE`, which changes the reward pin and invalidates the existing utility cache;
5. deleting legacy additive code, checkpoints, diagnostics, or the offline lexicographic selector.

## Sources

- [Trainable multi-objective RL review](../research/ranker-v2-trainable-multi-objective-rl-review.md).
- [Ranker v2 architecture](../specs/RL/ranker-v2-architecture.md).
- [Interactive ranker v2](../specs/RL/interactive-ranker-v2.md).
- [Ranker v2 architecture decision log](../specs/RL/ranker-v2-architecture-decision-log.md).
- [Interactive ranker v2 decision log](../specs/RL/interactive-ranker-v2-decision-log.md).
- [Interactive ranker v2 architecture report](../html/interactive-ranker-v2.html).
- [Lexicographic Multi-Objective Reinforcement Learning](../../research-wiki/papers/skalse2022_lexicographic_morl.md)
  ([arXiv 2212.13769](https://arxiv.org/abs/2212.13769)).
- [Reward Constrained Policy Optimization](../../research-wiki/papers/tessler2018_reward_constrained_policy.md)
  ([arXiv 1805.11074](https://arxiv.org/abs/1805.11074)).
- [Constrained Policy Optimization](../../research-wiki/papers/achiam2017_constrained_policy_optimization.md)
  ([arXiv 1705.10528](https://arxiv.org/abs/1705.10528)).
- [A Distributional View on Multi-Objective Policy Optimization](../../research-wiki/papers/abdolmaleki2020_distributional_view_multiobjective.md)
  ([arXiv 2005.07513](https://arxiv.org/abs/2005.07513)).
