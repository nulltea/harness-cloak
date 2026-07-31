"""Stable-ID trajectory sampling and replay for the ranker-v2 environment."""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import torch
import torch.nn.functional as F

from cloak.ranker.environment import (
    RankerAction,
    RankerDecision,
    RankerDocument,
    _action_by_id,
    _case_adjust,
    _fill_key,
    _fixed_fill_claims,
    _occurrence_maps,
    assemble_action_vector,
    legal_action_ids,
)
from cloak.ranker.profile_count import ProfileCountTargets
from cloak.reward.utility_cache import UtilityCache, UtilityRequest, UtilityResult, stable_hash
from cloak.reward.utility_credit import (
    DocumentUtilityCredit,
    document_utility,
    provisional_credit,
)


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


@dataclass(frozen=True)
class ReplayedTrajectory:
    doc_id: str
    lambda_profile: str
    steps: tuple[ReplayedStep, ...]


@dataclass(frozen=True)
class BehaviorCloningResult:
    trajectories: tuple[SampledTrajectory, ...]
    epoch_losses: tuple[float, ...]
    action_mode_counts: Mapping[str, int]
    runtime_type_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "action_mode_counts", MappingProxyType(dict(self.action_mode_counts))
        )
        object.__setattr__(
            self,
            "runtime_type_counts",
            MappingProxyType(dict(self.runtime_type_counts)),
        )


@dataclass(frozen=True)
class TrajectoryPoint:
    trajectory: SampledTrajectory
    utility: float
    count_score: float
    component_scores: Mapping[str, float]
    result_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_scores",
            MappingProxyType(dict(self.component_scores)),
        )

    @property
    def action_vector(self) -> Mapping[str, str]:
        return self.trajectory.action_vector


@dataclass(frozen=True)
class ExitDocumentCollection:
    doc_id: str
    reference: TrajectoryPoint
    candidates: tuple[TrajectoryPoint, ...]
    winner: TrajectoryPoint | None
    reverified_reference_utility: float | None
    verification_status: str


@dataclass(frozen=True)
class ExitCollection:
    documents: tuple[ExitDocumentCollection, ...]


class CacheOnlyMissError(RuntimeError):
    """The next scoring phase needs uncached transport work."""

    def __init__(
        self,
        *,
        phase: str,
        remote_tasks: int,
        context_reader_work_items: int,
    ):
        self.phase = phase
        self.remote_tasks = remote_tasks
        self.context_reader_work_items = context_reader_work_items
        super().__init__(
            f"cache-only {phase} miss: remote_tasks={remote_tasks} "
            f"context_reader_work_items={context_reader_work_items}"
        )


@dataclass(frozen=True)
class HybridDocumentObjective:
    total: torch.Tensor
    utility: torch.Tensor
    count: torch.Tensor
    entropy: torch.Tensor
    kl: torch.Tensor
    beta: float
    eta: float


@dataclass(frozen=True)
class LatinCycleSchedule:
    profile_names: tuple[str, ...]
    profile_values: tuple[float, ...]
    offsets_by_document: Mapping[str, int]
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "offsets_by_document",
            MappingProxyType(dict(self.offsets_by_document)),
        )

    def profile_for(self, doc_id: str, epoch: int):
        if epoch < 0:
            raise ValueError("schedule epoch must be nonnegative")
        try:
            offset = self.offsets_by_document[doc_id]
        except KeyError as error:
            raise ValueError(f"schedule lacks document {doc_id}") from error
        from cloak.ranker.environment import LambdaProfile

        index = (offset + epoch) % len(self.profile_names)
        return LambdaProfile(self.profile_names[index], self.profile_values[index])


@dataclass(frozen=True)
class WarmStartResult:
    identity_verified: bool
    verified_winner_count: int
    clone_target_count: int
    clone_loss: float
    reference_state_dict: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_state_dict",
            MappingProxyType({
                key: value.detach().clone()
                for key, value in self.reference_state_dict.items()
            }),
        )


@dataclass(frozen=True)
class DocumentTrainingResult:
    doc_id: str
    corpus: str
    profile_name: str
    rollout_count: int
    loss: float
    utility: float
    count_score: float
    entropy: float
    collision_count: int
    action_modes: Mapping[str, int]
    runtime_type_exposure: Mapping[str, int]
    gradient_norms: Mapping[str, float]
    absolute_weighted_mass: Mapping[str, float]
    scheduler_diagnostics: Mapping[str, Any]
    cache_metrics: Mapping[str, Any]
    action_vector_hashes: tuple[str, ...] = ()
    runtime_type_metrics: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    parameter_group_gradient_norms: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )
    alpha_diagnostics: Mapping[str, float] = field(default_factory=dict)
    privacy_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    lambda_zero_identity_failures: int = 0

    def __post_init__(self) -> None:
        for name in (
            "action_modes", "runtime_type_exposure", "gradient_norms",
            "absolute_weighted_mass", "scheduler_diagnostics", "cache_metrics",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(
            self,
            "runtime_type_metrics",
            MappingProxyType({
                key: MappingProxyType(dict(value))
                for key, value in self.runtime_type_metrics.items()
            }),
        )
        object.__setattr__(
            self,
            "parameter_group_gradient_norms",
            MappingProxyType({
                key: MappingProxyType(dict(value))
                for key, value in self.parameter_group_gradient_norms.items()
            }),
        )
        object.__setattr__(
            self, "alpha_diagnostics", MappingProxyType(dict(self.alpha_diagnostics))
        )
        object.__setattr__(
            self, "privacy_diagnostics", MappingProxyType(dict(self.privacy_diagnostics))
        )


@dataclass(frozen=True)
class HybridTrainingResult:
    epoch_reports: tuple[Mapping[str, Any], ...]
    schedule: LatinCycleSchedule
    pair_history: Mapping
    kl_enabled: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "epoch_reports",
            tuple(MappingProxyType(dict(row)) for row in self.epoch_reports),
        )
        object.__setattr__(
            self, "pair_history", MappingProxyType(dict(self.pair_history)),
        )


def _trajectory_from_action_vector(
    document: RankerDocument,
    action_vector: Mapping[str, str],
    profile: Any,
) -> SampledTrajectory:
    occurrence_by_id, _ = _occurrence_maps(document)
    reserved = tuple(_fixed_fill_claims(document, occurrence_by_id))
    expected = {decision.decision_id for decision in document.policy_decisions}
    if set(action_vector) != expected:
        raise ValueError(f"stored action vector differs for {document.doc_id}")
    claimed: dict[str, str] = {}
    steps = []
    ordered_vector = {}
    for decision in document.policy_decisions:
        menu = legal_action_ids(decision, claimed, reserved)
        selected = str(action_vector[decision.decision_id])
        if selected not in menu:
            raise ValueError(
                f"stored action is dynamically illegal for {decision.decision_id}"
            )
        claimed_before = tuple(sorted(claimed))
        action = _action_by_id(decision, selected)
        if action.mode == "level":
            assert action.fill is not None
            claimed.setdefault(_fill_key(action.fill), decision.decision_id)
        steps.append(SampledStep(
            decision_id=decision.decision_id,
            legal_action_ids=menu,
            selected_action_id=selected,
            claimed_fills_before=claimed_before,
        ))
        ordered_vector[decision.decision_id] = selected
    return SampledTrajectory(
        doc_id=document.doc_id,
        lambda_profile=profile,
        steps=tuple(steps),
        action_vector=MappingProxyType(ordered_vector),
    )


def _import_unconditioned_state(
    policy: torch.nn.Module,
    source_state: Mapping[str, torch.Tensor],
) -> None:
    target = policy.state_dict()
    # The controller gain head is created AFTER behavior cloning (tie-ownership
    # escalation); its zero-initialized parameters are legitimately absent from
    # the BC checkpoint and keep their init values. Everything else stays a
    # strict one-to-one import.
    optional = {
        name for name in target
        if name.startswith("gain_head.") and name not in source_state
    }
    if set(source_state) != set(target) - optional:
        raise ValueError("BC checkpoint parameters differ from conditional policy")
    imported = {}
    for name, target_value in target.items():
        if name in optional:
            imported[name] = target_value.detach().clone()
            continue
        source_value = source_state[name]
        if source_value.shape != target_value.shape:
            raise ValueError(f"BC parameter shape differs: {name}")
        imported[name] = source_value.detach().clone()
    policy.load_state_dict(imported, strict=True)


def _policy_architecture_name(policy: Any) -> str:
    declared = getattr(policy, "policy_architecture", None)
    if declared is not None:
        if declared not in {"semantic-v1", "legacy-film-gru"}:
            raise ValueError(f"unsupported policy architecture: {declared}")
        return declared
    if type(policy).__name__ == "SemanticRankerPolicy":
        return "semantic-v1"
    return "legacy-film-gru"


@torch.no_grad()
def assert_exact_lambda_zero_identity(
    policy: TrajectoryPolicy,
    documents: Sequence[RankerDocument],
    lambda_zero: Any,
) -> None:
    """Require the semantic controller's exact lambda-zero utility bypass."""

    if float(lambda_zero.value) != 0.0:
        raise ValueError("semantic warm start requires lambda zero")
    for document in documents:
        trajectory = behavior_clone_trajectory(document, lambda_zero)
        replayed = replay_trajectory(policy, document, trajectory, lambda_zero)
        for step in replayed.steps:
            expected = torch.log_softmax(step.utility_logits, dim=0)
            if not torch.equal(step.log_probs, expected):
                raise ValueError(
                    f"lambda-zero identity failed before warm start for {document.doc_id}"
                )


def initialize_hybrid_warm_start(
    policy: torch.nn.Module,
    documents: Sequence[RankerDocument],
    profiles: Sequence[Any],
    *,
    bc_state_dict: Mapping[str, torch.Tensor],
    exit_winners: Mapping,
    optimizer: torch.optim.Optimizer,
) -> WarmStartResult:
    """Import BC weights and clone verified winners without changing privacy control."""

    documents = tuple(documents)
    profiles = tuple(profiles)
    documents_by_id = {document.doc_id: document for document in documents}
    if len(documents_by_id) != len(documents) or not documents:
        raise ValueError("warm start requires unique nonempty documents")
    if exit_winners.get("artifact_version") != "ranker-v2-exit-winners-v1":
        raise ValueError("unsupported ExIt winner artifact")
    _import_unconditioned_state(policy, bc_state_dict)
    zero_profiles = tuple(
        profile for profile in profiles if float(profile.value) == 0.0
    )
    if len(zero_profiles) != 1:
        raise ValueError("semantic warm start requires exactly one lambda-zero profile")
    clone_profiles = zero_profiles
    assert_exact_lambda_zero_identity(policy, documents, zero_profiles[0])
    winners = []
    for row in exit_winners.get("documents", ()):
        if row.get("verification_status") != "verified" or row.get("winner") is None:
            continue
        doc_id = str(row.get("doc_id"))
        if doc_id not in documents_by_id:
            raise ValueError(f"ExIt winner document is unavailable: {doc_id}")
        winner = row["winner"]
        if not isinstance(winner, Mapping) or not isinstance(
            winner.get("action_vector"), Mapping
        ):
            raise ValueError(f"ExIt winner vector is invalid: {doc_id}")
        winners.append((documents_by_id[doc_id], winner["action_vector"]))

    train = getattr(policy, "train", None)
    if callable(train):
        train()
    total_loss = 0.0
    target_count = 0
    for document, vector in winners:
        for profile in clone_profiles:
            trajectory = _trajectory_from_action_vector(document, vector, profile)
            optimizer.zero_grad(set_to_none=True)
            replayed = replay_trajectory(policy, document, trajectory, profile)
            if not replayed.steps:
                raise ValueError("verified ExIt winner has no policy decisions")
            loss = -torch.stack([step.log_prob for step in replayed.steps]).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("non-finite ExIt clone loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(replayed.steps)
            target_count += len(replayed.steps)
    reference = {
        key: value.detach().clone() for key, value in policy.state_dict().items()
    }
    return WarmStartResult(
        identity_verified=True,
        verified_winner_count=len(winners),
        clone_target_count=target_count,
        clone_loss=total_loss / target_count if target_count else 0.0,
        reference_state_dict=reference,
    )


def expected_profile_count_loss(
    replay_steps: Sequence[ReplayedStep],
    targets: ProfileCountTargets,
    lambda_value: float,
    decision_count: int,
    rollout_count: int,
) -> torch.Tensor:
    """Exact profile-relative shaping loss over the count-detached distribution."""

    steps = tuple(replay_steps)
    if not steps:
        raise ValueError("profile count loss requires replay steps")
    if not isinstance(targets, ProfileCountTargets):
        raise TypeError("profile count loss requires ProfileCountTargets")
    if (
        isinstance(lambda_value, bool)
        or not isinstance(lambda_value, int | float)
        or not math.isfinite(float(lambda_value))
        or float(lambda_value) < 0.0
        or not isinstance(decision_count, int)
        or isinstance(decision_count, bool)
        or decision_count <= 0
        or not isinstance(rollout_count, int)
        or isinstance(rollout_count, bool)
        or rollout_count <= 0
    ):
        raise ValueError("profile count loss normalization is invalid")
    expectations = []
    for step in steps:
        if (
            step.count_log_probs.ndim != 1
            or len(step.count_log_probs) != len(step.legal_action_ids)
            or not bool(torch.isfinite(step.count_log_probs).all())
        ):
            raise ValueError("profile count loss received an invalid distribution")
        exact = targets.action_scores(
            step.decision_id, step.legal_action_ids,
        ).to(step.count_log_probs)
        expectations.append(torch.sum(step.count_log_probs.exp() * exact))
    return -torch.stack(expectations).sum() * (
        float(lambda_value) / decision_count / rollout_count
    )


def compose_hybrid_document_objective(
    replayed: Sequence[ReplayedTrajectory],
    *,
    utility_loss: torch.Tensor,
    profile_targets: ProfileCountTargets,
    lambda_value: float,
    beta: float,
    eta: float,
    reference_log_probs: Sequence[Sequence[torch.Tensor]] | None,
    kl_direction: str = "forward",
) -> HybridDocumentObjective:
    """Compose the rollout-normalized hybrid objective for one document group.

    kl_direction "forward" penalizes KL(pi || ref); "reverse" penalizes
    KL(ref || pi), whose gradient (~ pi - ref) stays alive when pi saturates —
    the failure mode of the lambda-3 coin-flip root cause (decision log
    2026-07-30).
    """

    replayed = tuple(replayed)
    if not replayed:
        raise ValueError("hybrid objective requires at least one rollout")
    if utility_loss.ndim != 0 or not bool(torch.isfinite(utility_loss)):
        raise ValueError("hybrid utility loss must be a finite scalar")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in (lambda_value, beta, eta)
    ):
        raise ValueError("lambda, beta, and eta must be finite and nonnegative")
    decision_count = len(replayed[0].steps)
    if decision_count == 0 or any(
        len(trajectory.steps) != decision_count for trajectory in replayed
    ):
        raise ValueError("hybrid objective requires equal nonempty decision walks")
    if len({trajectory.doc_id for trajectory in replayed}) != 1:
        raise ValueError("hybrid objective rollouts must share one document")
    replay_steps = tuple(
        step for trajectory in replayed for step in trajectory.steps
    )
    count = expected_profile_count_loss(
        replay_steps,
        profile_targets,
        lambda_value=float(lambda_value),
        decision_count=decision_count,
        rollout_count=len(replayed),
    )
    entropy = torch.stack([step.entropy for step in replay_steps]).sum() / len(replayed)
    if reference_log_probs is None:
        if eta != 0.0:
            raise ValueError("positive KL coefficient requires reference distributions")
        kl = utility_loss.new_zeros(())
    else:
        reference = tuple(tuple(row) for row in reference_log_probs)
        if len(reference) != len(replayed) or any(
            len(row) != decision_count for row in reference
        ):
            raise ValueError("reference distributions differ from replayed trajectories")
        kl_terms = []
        for trajectory, reference_row in zip(replayed, reference, strict=True):
            for step, reference_step in zip(
                trajectory.steps, reference_row, strict=True,
            ):
                reference_step = reference_step.to(
                    device=step.log_probs.device, dtype=step.log_probs.dtype,
                )
                if reference_step.shape != step.log_probs.shape or not bool(
                    torch.isfinite(reference_step).all()
                ):
                    raise ValueError("reference distribution differs from replay menu")
                if kl_direction == "forward":
                    kl_terms.append(torch.sum(
                        step.log_probs.exp() * (step.log_probs - reference_step)
                    ))
                elif kl_direction == "reverse":
                    kl_terms.append(torch.sum(
                        reference_step.exp() * (reference_step - step.log_probs)
                    ))
                else:
                    raise ValueError(f"unsupported KL direction: {kl_direction!r}")
        kl = torch.stack(kl_terms).sum() / len(replayed)
    total = utility_loss + count - float(beta) * entropy + float(eta) * kl
    if not all(bool(torch.isfinite(value)) for value in (count, entropy, kl, total)):
        raise ValueError("hybrid objective contains a non-finite term")
    return HybridDocumentObjective(
        total=total,
        utility=utility_loss,
        count=count,
        entropy=entropy,
        kl=kl,
        beta=float(beta),
        eta=float(eta),
    )


TIE_EXACT_ATOL = 1e-9
TIE_EXIT_BOUND = 0.044


def record_tie_evidence(
    ledger: dict,
    evidence_rows: Sequence[Mapping[str, Any]],
    current_round: int,
) -> None:
    """Append value-bearing probe evidence to the tie-ownership ledger."""
    for row in evidence_rows:
        pair = tuple(sorted((
            str(row["selected_action_id"]), str(row["alternative_action_id"]),
        )))
        key = (str(row["doc_id"]), str(row["decision_id"]), pair[0], pair[1])
        ledger.setdefault(key, []).append({
            "delta_u": float(row["delta_u"]),
            "context_hash": str(row["context_hash"]),
            "round": int(current_round),
        })


def compute_tie_labels(
    ledger: Mapping,
    documents: Mapping[str, RankerDocument],
    profile_targets: ProfileCountTargets,
    *,
    min_contexts: int = 3,
    before_round: int | None = None,
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]:
    """Qualified tie pairs per (doc, decision) from the VERIFIABLE core only.

    A pair qualifies iff it has >= min_contexts records with DISTINCT
    surrounding-context hashes whose |delta_u| is exactly zero (within
    TIE_EXACT_ATOL), and ZERO records with |delta_u| > TIE_EXIT_BOUND
    (monotone-until-contradicted exit on the measured bound). Records in the
    noise band neither qualify nor disqualify. Absence of evidence means
    UNKNOWN, never tied. Each qualified pair is oriented (a_plus, a_minus) by
    the frozen profile count score; score-equal pairs are dropped (no
    preference to enforce). before_round implements the one-cycle label lag.
    """
    labels: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for (doc_id, decision_id, action_a, action_b), records in ledger.items():
        usable = [
            record for record in records
            if before_round is None or record["round"] < before_round
        ]
        if not usable:
            continue
        if any(abs(r["delta_u"]) > TIE_EXIT_BOUND for r in usable):
            continue
        exact_contexts = {
            r["context_hash"] for r in usable
            if abs(r["delta_u"]) <= TIE_EXACT_ATOL
        }
        if len(exact_contexts) < min_contexts:
            continue
        score_a = float(profile_targets.action_scores(decision_id, (action_a,))[0])
        score_b = float(profile_targets.action_scores(decision_id, (action_b,))[0])
        if score_a == score_b:
            continue
        pair = (action_a, action_b) if score_a > score_b else (action_b, action_a)
        labels.setdefault((doc_id, decision_id), []).append(pair)
    return {key: tuple(sorted(pairs)) for key, pairs in labels.items()}


def tie_margin_loss(
    policy: Any,
    document: RankerDocument,
    trajectory: SampledTrajectory,
    labels: Mapping[tuple[str, str], Sequence[tuple[str, str]]],
    *,
    max_profile: Any,
    margin: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Controller-only hinge enforcing measured tie ownership at max lambda.

    For each qualified tied pair (a_plus, a_minus) of a labeled decision, the
    max-lambda sampling margin z(a_plus) - z(a_minus) must exceed `margin`.
    Utility logits and privacy scores are detached; alpha_raw is detached —
    the gradient reaches ONLY the gain residual (the critical gradient split).
    Macro-averaged per decision. Returns (hinge, residual_penalty,
    satisfied_pairs, total_pairs).
    """
    if getattr(policy, "controller_gain_mode", None) != "evidence":
        raise ValueError("tie margin loss requires the evidence controller gain")
    g_max = math.log1p(float(max_profile.value)) / math.log1p(
        float(policy.max_lambda)
    )
    if g_max <= 0.0:
        raise ValueError("tie margin loss requires a nonzero max-lambda profile")
    state = policy.begin_document(document, max_profile)
    per_decision_terms: list[torch.Tensor] = []
    residual_penalties: list[torch.Tensor] = []
    satisfied = total = 0
    gap_scaled = getattr(policy, "controller_gap_scaling", None) == "utility-gap"
    for decision in document.policy_decisions:
        menu = tuple(action.action_id for action in decision.actions)
        pairs = labels.get((document.doc_id, decision.decision_id), ())
        if pairs:
            row = policy.distribution(state, decision, menu, max_profile)
            utility = row.utility_logits.detach()
            privacy = row.predicted_privacy.detach()
            residual = policy.gain_head(
                policy_tie_pooled_inputs(policy, state, decision)
            ).squeeze(-1)
            alpha_tie = F.softplus(policy.alpha_raw.detach() + residual)
            scale = (
                (utility.max() - utility.min()) if gap_scaled
                else utility.new_ones(())
            )
            index = {action_id: position for position, action_id in enumerate(menu)}
            terms = []
            for a_plus, a_minus in pairs:
                if a_plus not in index or a_minus not in index:
                    continue
                z_gap = (
                    utility[index[a_plus]] - utility[index[a_minus]]
                ) + alpha_tie * g_max * scale * (
                    privacy[index[a_plus]] - privacy[index[a_minus]]
                )
                total += 1
                if float(z_gap) >= margin:
                    satisfied += 1
                terms.append(torch.relu(
                    torch.as_tensor(float(margin), dtype=z_gap.dtype) - z_gap
                ))
            if terms:
                per_decision_terms.append(torch.stack(terms).mean())
                residual_penalties.append(residual ** 2)
        selected = trajectory.action_vector[decision.decision_id]
        state = policy.advance(state, decision, selected)
    prototype = policy.alpha_raw
    if not per_decision_terms:
        zero = prototype.new_zeros(())
        return zero, zero.clone(), 0, 0
    hinge = torch.stack(per_decision_terms).mean()
    penalty = torch.stack(residual_penalties).mean()
    return hinge, penalty, satisfied, total


def policy_tie_pooled_inputs(
    policy: Any, state: Any, decision: RankerDecision,
) -> torch.Tensor:
    """The gain head's pooled, detached feature input for one decision."""
    actions, pair_features, token_bank, features = policy._decision_inputs(
        state, decision
    )
    utility_relations = policy.utility_projection(pair_features)
    contexts = policy.context_readout(token_bank, features, utility_relations)
    histories = policy.memory(
        torch.cat([utility_relations, contexts], dim=-1),
        state.selected_records,
    )
    mode_ids, runtime_type_ids = policy._category_ids(actions, decision)
    interaction = policy.interaction_projection(
        utility_relations * policy.context_to_relation(contexts)
    )
    return torch.cat(
        [
            utility_relations, contexts, interaction,
            policy.action_mode_embedding(mode_ids),
            policy.runtime_type_embedding(runtime_type_ids),
            histories,
        ],
        dim=-1,
    ).detach().mean(dim=0)


def profile_sensitivity_loss(
    replayed: Sequence["ReplayedTrajectory"],
    policy: Any,
    profiles: Sequence[Any],
    target_kl_per_unit: float,
) -> torch.Tensor:
    """Penalize deviation of adjacent-profile sensitivity from a measured target.

    Scaled-diversity regularizer in the D3PO sense (tie-ownership fork,
    decision log 2026-07-30): profile separation becomes an explicit objective.
    Because the utility tower is lambda-blind, the conditional distribution at
    ANY profile is analytically reconstructible from a replayed step's utility
    logits and privacy scores (z = u + alpha*g(lambda)*gap*p_hat; lambda-zero
    is the exact identity), so no extra forward passes are needed. For each
    replayed decision and each adjacent profile pair the loss is
    (KL(pi_k || pi_k+1) - target * (g_k+1 - g_k))^2; the gradient reaches
    alpha and the utility shape.
    """
    if not 0.0 <= float(target_kl_per_unit) < math.inf:
        raise ValueError("profile sensitivity target must be finite and nonnegative")
    alpha = policy.alpha
    max_lambda = float(policy.max_lambda)
    ramp = [
        math.log1p(float(profile.value)) / math.log1p(max_lambda)
        for profile in profiles
    ]
    gap_scaled = getattr(policy, "controller_gap_scaling", None) == "utility-gap"
    terms = []
    for trajectory in replayed:
        for step in trajectory.steps:
            utility = step.utility_logits
            privacy = step.predicted_privacy.detach()
            gap = (
                (utility.max() - utility.min()).detach()
                if gap_scaled else utility.new_ones(())
            )
            log_distributions = [
                torch.log_softmax(
                    utility + alpha * g * gap * privacy if g > 0.0 else utility,
                    dim=0,
                )
                for g in ramp
            ]
            for k in range(len(ramp) - 1):
                delta_g = ramp[k + 1] - ramp[k]
                if delta_g <= 0.0:
                    continue
                lower, upper = log_distributions[k], log_distributions[k + 1]
                kl = torch.sum(lower.exp() * (lower - upper))
                terms.append((kl - float(target_kl_per_unit) * delta_g) ** 2)
    if not terms:
        raise ValueError("profile sensitivity loss requires replayed decisions")
    return torch.stack(terms).mean()


def measure_profile_sensitivity_target(
    policy: TrajectoryPolicy,
    documents: Sequence[RankerDocument],
    profiles: Sequence[Any],
) -> float:
    """Measured adjacent-profile KL per unit g under the calibrated warm start.

    Sets the sensitivity regularizer's target from a measurement (never
    invented): the greedy-walk decisions of every document, scored with the
    same analytic reconstruction the loss uses.
    """
    total_kl = 0.0
    total_dg = 0.0
    with torch.no_grad():
        for document in documents:
            greedy = sample_trajectory(
                policy, document, profiles[0], greedy=True, generator=None,
            )
            replayed = replay_trajectory(policy, document, greedy, profiles[0])
            alpha = policy.alpha
            max_lambda = float(policy.max_lambda)
            ramp = [
                math.log1p(float(profile.value)) / math.log1p(max_lambda)
                for profile in profiles
            ]
            gap_scaled = (
                getattr(policy, "controller_gap_scaling", None) == "utility-gap"
            )
            for step in replayed.steps:
                utility = step.utility_logits
                privacy = step.predicted_privacy
                gap = (
                    (utility.max() - utility.min())
                    if gap_scaled else utility.new_ones(())
                )
                logs = [
                    torch.log_softmax(
                        utility + alpha * g * gap * privacy if g > 0.0 else utility,
                        dim=0,
                    )
                    for g in ramp
                ]
                for k in range(len(ramp) - 1):
                    delta_g = ramp[k + 1] - ramp[k]
                    if delta_g <= 0.0:
                        continue
                    total_kl += float(torch.sum(
                        logs[k].exp() * (logs[k] - logs[k + 1])
                    ))
                    total_dg += delta_g
    if total_dg <= 0.0:
        raise ValueError("sensitivity target measurement found no profile gaps")
    return total_kl / total_dg


def synchronous_profile_snapshot(
    policy: TrajectoryPolicy,
    documents: Sequence[RankerDocument],
    profiles: Sequence[Any],
    profile_targets: ProfileCountTargets,
    *,
    samples: int = 64,
    seed: int = 0,
    tie_labels: Mapping | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate every profile from ONE policy state (confound-free readout).

    Latin-cycle group means observe each profile several optimizer steps apart,
    mixing lambda-conditioning with policy evolution (root-cause adjudication,
    decision log 2026-07-30). This snapshot samples action vectors per profile
    from the same checkpoint — count scores need no remote scoring — plus the
    greedy walk's per-decision expected privacy score and utility-logit range.
    Uses its own generator so training RNG streams are untouched.
    """
    generator = torch.Generator().manual_seed(seed)
    snapshot: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for document in documents:
            decision_ids = [
                decision.decision_id for decision in document.policy_decisions
            ]
            per_profile: dict[str, Any] = {}
            for profile in profiles:
                sampled_p = []
                for _ in range(samples):
                    trajectory = sample_trajectory(
                        policy, document, profile,
                        greedy=False, generator=generator,
                    )
                    scores = [
                        float(profile_targets.action_scores(
                            decision_id,
                            (trajectory.action_vector[decision_id],),
                        )[0])
                        for decision_id in decision_ids
                    ]
                    sampled_p.append(sum(scores) / len(scores))
                greedy = sample_trajectory(
                    policy, document, profile, greedy=True, generator=None,
                )
                replayed = replay_trajectory(policy, document, greedy, profile)
                decisions = []
                greedy_scores = []
                gain_enabled = getattr(policy, "controller_gain_mode", None)
                if gain_enabled:
                    from cloak.ranker.semantic import decision_controller_alpha
                    gain_state = policy.begin_document(document, profile)
                for decision, step in zip(
                    document.policy_decisions, replayed.steps, strict=True,
                ):
                    action_scores = torch.tensor([
                        float(profile_targets.action_scores(
                            decision.decision_id, (action_id,),
                        )[0])
                        for action_id in step.legal_action_ids
                    ], dtype=step.log_probs.dtype)
                    probabilities = step.log_probs.exp().cpu()
                    greedy_scores.append(float(profile_targets.action_scores(
                        decision.decision_id, (step.selected_action_id,),
                    )[0]))
                    entry = {
                        "decision_id": decision.decision_id,
                        "greedy_action_score": greedy_scores[-1],
                        "expected_p": float(
                            torch.sum(probabilities * action_scores)
                        ),
                        "utility_logit_range": float(
                            step.utility_logits.max() - step.utility_logits.min()
                        ),
                    }
                    if gain_enabled:
                        entry["controller_alpha"] = decision_controller_alpha(
                            policy, gain_state, decision, step.legal_action_ids,
                        )
                    decisions.append(entry)
                row = {
                    "sampled_P_mean": sum(sampled_p) / len(sampled_p),
                    "sampled_P_count": samples,
                    "greedy_P": sum(greedy_scores) / len(greedy_scores),
                    "decisions": decisions,
                }
                if tie_labels is not None and float(profile.value) > 0.0:
                    # Hard lexicographic ORACLE from the same evidence: the
                    # z-greedy walk, but within each decision's qualified tie
                    # set the max-count-score member is selected. A free
                    # ceiling + evidence-sufficiency diagnostic.
                    oracle_scores = []
                    oracle_state = policy.begin_document(document, profile)
                    occurrence_by_id, _ = _occurrence_maps(document)
                    reserved = tuple(
                        _fixed_fill_claims(document, occurrence_by_id)
                    )
                    claimed: dict[str, str] = {}
                    for decision in document.policy_decisions:
                        menu = legal_action_ids(decision, claimed, reserved)
                        log_probs = _checked_log_probs(
                            policy, oracle_state, decision, menu, profile,
                        )
                        pick = menu[int(torch.argmax(log_probs).item())]
                        pairs = tie_labels.get(
                            (document.doc_id, decision.decision_id), (),
                        )
                        tied_with_pick = {pick}
                        for a_plus, a_minus in pairs:
                            if pick in (a_plus, a_minus):
                                tied_with_pick.update((a_plus, a_minus))
                        tied_in_menu = [a for a in tied_with_pick if a in menu]
                        pick = max(
                            tied_in_menu,
                            key=lambda a: float(profile_targets.action_scores(
                                decision.decision_id, (a,),
                            )[0]),
                        )
                        oracle_scores.append(float(
                            profile_targets.action_scores(
                                decision.decision_id, (pick,),
                            )[0]
                        ))
                        action = _action_by_id(decision, pick)
                        if action.mode == "level":
                            assert action.fill is not None
                            claimed.setdefault(
                                _fill_key(action.fill), decision.decision_id,
                            )
                        oracle_state = policy.advance(
                            oracle_state, decision, pick,
                        )
                    row["oracle_greedy_P"] = (
                        sum(oracle_scores) / len(oracle_scores)
                    )
                per_profile[str(profile.name)] = row
            snapshot[document.doc_id] = per_profile
    return snapshot


def build_latin_cycle_schedule(
    documents: Sequence[RankerDocument],
    profiles: Sequence[Any],
    *,
    seed: int,
) -> LatinCycleSchedule:
    """Freeze seeded per-document offsets for a balanced profile cycle."""

    documents = tuple(documents)
    profiles = tuple(profiles)
    if not documents or len({document.doc_id for document in documents}) != len(documents):
        raise ValueError("Latin cycle requires unique nonempty documents")
    names = tuple(str(profile.name) for profile in profiles)
    values = tuple(float(profile.value) for profile in profiles)
    if not names or len(set(names)) != len(names):
        raise ValueError("Latin cycle requires unique nonempty profiles")
    if any(not name for name in names) or any(
        not math.isfinite(value) or value < 0.0 for value in values
    ):
        raise ValueError("Latin cycle profiles are invalid")
    permutation = list(range(len(profiles)))
    random.Random(seed).shuffle(permutation)
    offsets = {
        doc_id: permutation[index % len(permutation)]
        for index, doc_id in enumerate(sorted(document.doc_id for document in documents))
    }
    return LatinCycleSchedule(
        profile_names=names,
        profile_values=values,
        offsets_by_document=offsets,
        seed=int(seed),
    )


def profile_exposure_report(
    documents: Sequence[RankerDocument],
    schedule: LatinCycleSchedule,
    epochs: Collection[int],
) -> dict[str, Any]:
    """Count scheduled document-group and decision exposure without reward weighting."""

    by_document: dict[str, Counter[str]] = {}
    by_corpus: dict[str, Counter[str]] = {}
    by_type: dict[str, Counter[str]] = {}
    by_profile: Counter[str] = Counter()
    for document in documents:
        document_counts: Counter[str] = Counter()
        corpus_counts = by_corpus.setdefault(document.corpus, Counter())
        runtime_types = Counter(
            decision.runtime_type for decision in document.policy_decisions
        )
        for epoch in epochs:
            profile = schedule.profile_for(document.doc_id, int(epoch))
            document_counts[profile.name] += 1
            corpus_counts[profile.name] += 1
            by_profile[profile.name] += 1
            for runtime_type, count in runtime_types.items():
                by_type.setdefault(runtime_type, Counter())[profile.name] += count
        by_document[document.doc_id] = document_counts
    return {
        "by_document": {
            key: dict(sorted(value.items())) for key, value in sorted(by_document.items())
        },
        "by_corpus": {
            key: dict(sorted(value.items())) for key, value in sorted(by_corpus.items())
        },
        "by_type": {
            key: dict(sorted(value.items())) for key, value in sorted(by_type.items())
        },
        "by_profile": dict(sorted(by_profile.items())),
    }


def provisional_utility_loss(
    replayed: Sequence["ReplayedTrajectory"],
    credit: DocumentUtilityCredit,
) -> torch.Tensor:
    """Return the rollout-normalized provisional REINFORCE utility term."""
    rollout_count = len(replayed)
    if rollout_count == 0:
        raise ValueError("provisional utility loss requires at least one rollout")
    if (
        len(credit.document_utility) != rollout_count
        or len(credit.residual_utility) != rollout_count
        or any(len(values) != rollout_count for values in credit.linked_utility.values())
    ):
        raise ValueError("utility credit rollout count differs from replayed trajectories")

    replayed_pairs: set[tuple[int, str]] = set()
    terms: list[torch.Tensor] = []
    for rollout_index, trajectory in enumerate(replayed):
        seen_decisions: set[str] = set()
        for step in trajectory.steps:
            if step.decision_id in seen_decisions:
                raise ValueError(
                    f"replayed trajectory repeats decision {step.decision_id!r}"
                )
            seen_decisions.add(step.decision_id)
            pair = (rollout_index, step.decision_id)
            replayed_pairs.add(pair)
            if pair not in credit.provisional_advantage:
                continue
            terms.append(-float(credit.provisional_advantage[pair]) * step.log_prob)

    # Credit prices every artifact policy decision; load-time demotion (scope,
    # zero-coverage) removes some from the replayed walk, so extra credit pairs
    # are expected and unused. Missing pairs stay a hard error.
    if not replayed_pairs <= set(credit.provisional_advantage):
        missing = sorted(replayed_pairs - set(credit.provisional_advantage))
        raise ValueError(f"credit lacks replayed trajectory pairs: {missing[:4]}")
    if not terms:
        raise ValueError("provisional utility loss requires at least one decision pair")
    return torch.stack(terms).sum() / rollout_count


def hybrid_utility_loss(
    replayed: Sequence["ReplayedTrajectory"],
    provisional_credit: DocumentUtilityCredit,
    counterfactual_losses: Mapping[tuple[int, str], torch.Tensor],
) -> torch.Tensor:
    """Substitute measured pair losses for provisional terms, once per pair."""

    rollout_count = len(replayed)
    if rollout_count == 0:
        raise ValueError("hybrid utility loss requires at least one rollout")
    if (
        len(provisional_credit.document_utility) != rollout_count
        or len(provisional_credit.residual_utility) != rollout_count
        or any(
            len(values) != rollout_count
            for values in provisional_credit.linked_utility.values()
        )
    ):
        raise ValueError("utility credit rollout count differs from replayed trajectories")

    replayed_pairs: set[tuple[int, str]] = set()
    step_by_pair: dict[tuple[int, str], ReplayedStep] = {}
    for rollout_index, trajectory in enumerate(replayed):
        for step in trajectory.steps:
            pair = (rollout_index, step.decision_id)
            if pair in replayed_pairs:
                raise ValueError(f"replayed trajectory repeats decision {step.decision_id!r}")
            replayed_pairs.add(pair)
            step_by_pair[pair] = step
    # See provisional_utility_loss: demoted decisions leave unused credit pairs.
    if not replayed_pairs <= set(provisional_credit.provisional_advantage):
        missing = sorted(replayed_pairs - set(provisional_credit.provisional_advantage))
        raise ValueError(f"credit lacks replayed trajectory pairs: {missing[:4]}")
    unknown = sorted(set(counterfactual_losses) - replayed_pairs)
    if unknown:
        raise ValueError(f"unknown counterfactual loss pairs: {unknown}")
    for pair, loss in counterfactual_losses.items():
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError(f"counterfactual loss for {pair} must be a scalar tensor")
        if not bool(torch.isfinite(loss)):
            raise ValueError(f"counterfactual loss for {pair} is non-finite")

    terms = []
    for pair in sorted(replayed_pairs):
        if pair in counterfactual_losses:
            terms.append(counterfactual_losses[pair])
        else:
            terms.append(
                -float(provisional_credit.provisional_advantage[pair])
                * step_by_pair[pair].log_prob
            )
    return torch.stack(terms).sum() / rollout_count


class TrajectoryPolicy(Protocol):
    def begin_document(self, document: RankerDocument, profile: Any) -> Any: ...

    def log_probs(
        self,
        state: Any,
        decision: RankerDecision,
        legal_action_ids: Sequence[str],
        profile: Any,
    ) -> torch.Tensor: ...

    def distribution(
        self,
        state: Any,
        decision: RankerDecision,
        legal_action_ids: Sequence[str],
        profile: Any,
    ) -> Any: ...

    def advance(
        self, state: Any, decision: RankerDecision, action_id: str
    ) -> Any: ...


def _checked_log_probs(
    policy: TrajectoryPolicy,
    state: Any,
    decision: RankerDecision,
    menu: tuple[str, ...],
    profile: Any,
) -> torch.Tensor:
    if not menu:
        raise ValueError(f"empty legal action menu for {decision.decision_id}")
    log_probs = policy.log_probs(state, decision, menu, profile)
    if not isinstance(log_probs, torch.Tensor):
        raise TypeError(f"policy log_probs must return a tensor for {decision.decision_id}")
    if log_probs.ndim != 1 or len(log_probs) != len(menu):
        raise ValueError(f"incomplete policy distribution for {decision.decision_id}")
    if not bool(torch.isfinite(log_probs).all()):
        raise ValueError(f"non-finite policy distribution for {decision.decision_id}")
    return log_probs


def _checked_semantic_distribution(
    policy: TrajectoryPolicy,
    state: Any,
    decision: RankerDecision,
    menu: tuple[str, ...],
    profile: Any,
) -> Any:
    distribution = getattr(policy, "distribution", None)
    if not callable(distribution):
        raise ValueError(
            "semantic-v1 replay requires a complete distribution, not scalar log-probs"
        )
    row = distribution(state, decision, menu, profile)
    if tuple(getattr(row, "action_ids", ())) != menu:
        raise ValueError(
            f"semantic complete distribution menu differs for {decision.decision_id}"
        )
    for name in (
        "log_probs", "count_log_probs", "utility_logits", "predicted_privacy",
    ):
        tensor = getattr(row, name, None)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 1
            or len(tensor) != len(menu)
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(
                f"semantic complete distribution has invalid {name} for "
                f"{decision.decision_id}"
            )
    return row


@torch.no_grad()
def sample_trajectory(
    policy: TrajectoryPolicy,
    document: RankerDocument,
    lambda_profile,
    *,
    greedy: bool,
    generator: torch.Generator | None,
) -> SampledTrajectory:
    """Sample stable action IDs without retaining an autograd graph."""

    occurrence_by_id, _ = _occurrence_maps(document)
    reserved = tuple(_fixed_fill_claims(document, occurrence_by_id))
    claimed: dict[str, str] = {}
    state = policy.begin_document(document, lambda_profile)
    steps: list[SampledStep] = []
    action_vector: dict[str, str] = {}
    for decision in document.policy_decisions:
        menu = legal_action_ids(decision, claimed, reserved)
        claimed_before = tuple(sorted(claimed))
        log_probs = _checked_log_probs(policy, state, decision, menu, lambda_profile)
        if greedy:
            selected_index = int(torch.argmax(log_probs).item())
        else:
            # Draw on CPU so the caller's CPU generator works for any policy device.
            selected_index = int(
                torch.multinomial(
                    log_probs.exp().cpu(), 1, generator=generator
                ).item()
            )
        selected_action_id = menu[selected_index]
        action = _action_by_id(decision, selected_action_id)
        if action.mode == "level":
            assert action.fill is not None
            claimed.setdefault(_fill_key(action.fill), decision.decision_id)
        steps.append(SampledStep(
            decision_id=decision.decision_id,
            legal_action_ids=menu,
            selected_action_id=selected_action_id,
            claimed_fills_before=claimed_before,
        ))
        action_vector[decision.decision_id] = selected_action_id
        state = policy.advance(state, decision, selected_action_id)
    return SampledTrajectory(
        doc_id=document.doc_id,
        lambda_profile=lambda_profile,
        steps=tuple(steps),
        action_vector=MappingProxyType(action_vector),
    )


def dominant_trajectory_probability(
    policy: TrajectoryPolicy,
    document: RankerDocument,
    lambda_profile,
) -> float:
    """Exact probability of the greedy legal walk under the current policy.

    The greedy walk is the (near-)dominant complete action vector; its
    probability p_hat drives support-scaled rollouts: a group of R rollouts is
    fully degenerate with probability ~p_hat^R (small-document credit-support
    fork, decision log 2026-07-29).
    """
    occurrence_by_id, _ = _occurrence_maps(document)
    reserved = tuple(_fixed_fill_claims(document, occurrence_by_id))
    claimed: dict[str, str] = {}
    state = policy.begin_document(document, lambda_profile)
    log_probability = 0.0
    with torch.no_grad():
        for decision in document.policy_decisions:
            menu = legal_action_ids(decision, claimed, reserved)
            log_probs = _checked_log_probs(
                policy, state, decision, menu, lambda_profile,
            )
            index = int(torch.argmax(log_probs).item())
            log_probability += float(log_probs[index])
            action = _action_by_id(decision, menu[index])
            if action.mode == "level":
                assert action.fill is not None
                claimed.setdefault(_fill_key(action.fill), decision.decision_id)
            state = policy.advance(state, decision, menu[index])
    return math.exp(log_probability)


def support_scaled_rollouts(
    p_hat: float,
    base: int,
    *,
    cap: int = 32,
    degenerate_target: float = 0.05,
) -> int:
    """Smallest rollout count holding P[all rollouts identical] under target."""
    if not 0.0 < p_hat < 1.0:
        return cap if p_hat >= 1.0 else base
    needed = math.ceil(math.log(degenerate_target) / math.log(p_hat))
    return max(base, min(cap, needed))


def replay_trajectory(
    policy: TrajectoryPolicy,
    document: RankerDocument,
    trajectory: SampledTrajectory,
    lambda_profile,
) -> ReplayedTrajectory:
    """Replay a sampled stable-ID trajectory with a fresh autograd graph."""

    occurrence_by_id, _ = _occurrence_maps(document)
    if trajectory.doc_id != document.doc_id:
        raise ValueError("trajectory document id differs from replay document")
    if len(trajectory.steps) != len(document.policy_decisions):
        raise ValueError("trajectory step count differs from policy decision count")
    expected_vector_ids = {decision.decision_id for decision in document.policy_decisions}
    if set(trajectory.action_vector) != expected_vector_ids:
        raise ValueError("trajectory action vector is incomplete")

    reserved = tuple(_fixed_fill_claims(document, occurrence_by_id))
    claimed: dict[str, str] = {}
    state = policy.begin_document(document, lambda_profile)
    replayed: list[ReplayedStep] = []
    for decision, sampled in zip(document.policy_decisions, trajectory.steps):
        if sampled.decision_id != decision.decision_id:
            raise ValueError("trajectory decision order differs from environment")
        menu = legal_action_ids(decision, claimed, reserved)
        if menu != sampled.legal_action_ids:
            raise ValueError(
                f"replayed legal menu differs for {decision.decision_id}"
            )
        claimed_before = tuple(sorted(claimed))
        if claimed_before != sampled.claimed_fills_before:
            raise ValueError(f"replayed claimed fills differ for {decision.decision_id}")
        if sampled.selected_action_id not in menu:
            raise ValueError(f"sampled action is not legal for {decision.decision_id}")
        if trajectory.action_vector[decision.decision_id] != sampled.selected_action_id:
            raise ValueError(f"trajectory action vector differs for {decision.decision_id}")

        if _policy_architecture_name(policy) == "semantic-v1":
            distribution = _checked_semantic_distribution(
                policy, state, decision, menu, lambda_profile,
            )
            log_probs = distribution.log_probs
            count_log_probs = distribution.count_log_probs
            utility_logits = distribution.utility_logits
            predicted_privacy = distribution.predicted_privacy
        else:
            log_probs = _checked_log_probs(
                policy, state, decision, menu, lambda_profile,
            )
            count_log_probs = log_probs
            utility_logits = log_probs
            predicted_privacy = torch.zeros_like(log_probs)
        selected_index = menu.index(sampled.selected_action_id)
        selected_log_prob = log_probs[selected_index]
        entropy = -(log_probs.exp() * log_probs).sum()
        replayed.append(ReplayedStep(
            decision_id=decision.decision_id,
            selected_action_id=sampled.selected_action_id,
            legal_action_ids=menu,
            log_prob=selected_log_prob,
            log_probs=log_probs,
            count_log_probs=count_log_probs,
            utility_logits=utility_logits,
            predicted_privacy=predicted_privacy,
            entropy=entropy,
        ))

        action = _action_by_id(decision, sampled.selected_action_id)
        if action.mode == "level":
            assert action.fill is not None
            claimed.setdefault(_fill_key(action.fill), decision.decision_id)
        state = policy.advance(state, decision, sampled.selected_action_id)
    return ReplayedTrajectory(
        doc_id=document.doc_id,
        lambda_profile=lambda_profile,
        steps=tuple(replayed),
    )


def behavior_clone_trajectory(
    document: RankerDocument,
    lambda_profile: Any,
) -> SampledTrajectory:
    """Build the lambda-independent, support-preserving BC teacher walk."""

    occurrence_by_id, _ = _occurrence_maps(document)
    reserved = tuple(_fixed_fill_claims(document, occurrence_by_id))
    claimed: dict[str, str] = {}
    steps: list[SampledStep] = []
    action_vector: dict[str, str] = {}
    for decision in document.policy_decisions:
        menu = legal_action_ids(decision, claimed, reserved)
        claimed_before = tuple(sorted(claimed))
        legal = set(menu)
        level_actions = [
            action for action in decision.actions
            if action.mode == "level" and action.action_id in legal
        ]
        if level_actions:
            if any(action.authored_level_index is None for action in level_actions):
                raise ValueError(
                    f"level action lacks authored order for {decision.decision_id}"
                )
            authored_indices = [
                int(action.authored_level_index) for action in level_actions
            ]
            if len(authored_indices) != len(set(authored_indices)):
                raise ValueError(
                    f"duplicate authored level index for {decision.decision_id}"
                )
            selected = min(
                level_actions,
                key=lambda action: int(action.authored_level_index),
            )
        else:
            placeholders = [
                action for action in decision.actions
                if action.mode == "placeholder" and action.action_id in legal
            ]
            if len(placeholders) != 1:
                raise ValueError(
                    f"BC teacher requires one placeholder for {decision.decision_id}"
                )
            selected = placeholders[0]

        if selected.mode == "level":
            assert selected.fill is not None
            claimed.setdefault(_fill_key(selected.fill), decision.decision_id)
        steps.append(SampledStep(
            decision_id=decision.decision_id,
            legal_action_ids=menu,
            selected_action_id=selected.action_id,
            claimed_fills_before=claimed_before,
        ))
        action_vector[decision.decision_id] = selected.action_id

    return SampledTrajectory(
        doc_id=document.doc_id,
        lambda_profile=lambda_profile,
        steps=tuple(steps),
        action_vector=MappingProxyType(action_vector),
    )


def behavior_clone(
    policy: TrajectoryPolicy,
    documents: Sequence[RankerDocument],
    *,
    lambda_zero: Any,
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> BehaviorCloningResult:
    """Fit teacher stable IDs by cross-entropy over each dynamic legal menu."""

    documents = tuple(documents)
    if not documents:
        raise ValueError("behavior cloning requires at least one document")
    if epochs <= 0:
        raise ValueError("behavior cloning epochs must be positive")
    if (
        _policy_architecture_name(policy) == "semantic-v1"
        and float(lambda_zero.value) != 0.0
    ):
        raise ValueError("semantic behavior cloning requires lambda zero")
    trajectories = tuple(
        behavior_clone_trajectory(document, lambda_zero)
        for document in documents
    )
    modes: Counter[str] = Counter()
    runtime_types: Counter[str] = Counter()
    documents_by_id = {document.doc_id: document for document in documents}
    if len(documents_by_id) != len(documents):
        raise ValueError("behavior cloning document IDs must be unique")
    for document, trajectory in zip(documents, trajectories, strict=True):
        decisions = {
            decision.decision_id: decision for decision in document.policy_decisions
        }
        for step in trajectory.steps:
            decision = decisions[step.decision_id]
            action = _action_by_id(decision, step.selected_action_id)
            modes[action.mode] += 1
            runtime_types[decision.runtime_type] += 1

    train = getattr(policy, "train", None)
    if callable(train):
        train()
    epoch_losses: list[float] = []
    for _ in range(epochs):
        total_loss = 0.0
        decision_count = 0
        for trajectory in trajectories:
            optimizer.zero_grad(set_to_none=True)
            replayed = replay_trajectory(
                policy,
                documents_by_id[trajectory.doc_id],
                trajectory,
                lambda_zero,
            )
            if not replayed.steps:
                raise ValueError(
                    f"behavior cloning document has no policy decisions: {trajectory.doc_id}"
                )
            loss = -torch.stack([step.log_prob for step in replayed.steps]).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("non-finite behavior cloning loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(replayed.steps)
            decision_count += len(replayed.steps)
        epoch_losses.append(total_loss / decision_count)
    return BehaviorCloningResult(
        trajectories=trajectories,
        epoch_losses=tuple(epoch_losses),
        action_mode_counts=dict(sorted(modes.items())),
        runtime_type_counts=dict(sorted(runtime_types.items())),
    )


def trajectory_point(
    trajectory: SampledTrajectory,
    result: UtilityResult,
    *,
    count_reward: ProfileCountTargets,
    utility_artifact: Mapping,
) -> TrajectoryPoint:
    """Bind a complete cached component vector to its pure U and diagnostic P_count."""

    if result.doc_id != trajectory.doc_id:
        raise ValueError("utility result document differs from trajectory")
    if dict(result.action_vector) != dict(trajectory.action_vector):
        raise ValueError("utility result action vector differs from trajectory")
    utility = document_utility(
        result.component_scores, utility_artifact, trajectory.doc_id,
    )
    if not trajectory.action_vector:
        raise ValueError("trajectory point requires at least one policy decision")
    count_score = sum(
        float(count_reward.action_scores(decision_id, (action_id,))[0])
        for decision_id, action_id in trajectory.action_vector.items()
    ) / len(trajectory.action_vector)
    return TrajectoryPoint(
        trajectory=trajectory,
        utility=utility,
        count_score=count_score,
        component_scores=result.component_scores,
        result_hash=result.result_hash,
    )


def _vector_key(
    document: RankerDocument, action_vector: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (decision.decision_id, str(action_vector[decision.decision_id]))
        for decision in document.policy_decisions
    )


def _context_work_items(request: UtilityRequest) -> int:
    return sum(
        row.get("doc_id") == request.document.doc_id
        and row.get("status", "accepted") == "accepted"
        and row.get("family") == "context"
        for row in request.utility_artifact.get("assertions", {}).values()
    )


def _require_cached(
    requests: Sequence[UtilityRequest],
    *,
    cache: UtilityCache,
    reader_refresh: bool,
    phase: str,
) -> None:
    """Preflight the same validated identities used by the staged scorer."""

    # Local import avoids the roundtrip -> trajectory module dependency at import time.
    from cloak.reward.roundtrip import _cache_identity, _validate_request_readers

    unique: dict[str, tuple[UtilityRequest, Mapping]] = {}
    for request in requests:
        _validate_request_readers(request)
        doc_p, _ = assemble_action_vector(request.document, request.action_vector)
        identity = _cache_identity(
            request, doc_p, reader_refresh=reader_refresh,
        )
        unique.setdefault(cache.request_identity(identity), (request, identity))

    missing: list[UtilityRequest] = []
    for request, identity in unique.values():
        if cache.lookup(identity) is None:
            missing.append(request)
    if missing:
        raise CacheOnlyMissError(
            phase=phase,
            remote_tasks=len(missing),
            context_reader_work_items=sum(_context_work_items(row) for row in missing),
        )


def _default_score_batch(requests, **kwargs):
    from cloak.reward.roundtrip import score_roundtrip_batch

    return score_roundtrip_batch(requests, **kwargs)


def score_trajectories(
    documents: Sequence[RankerDocument],
    trajectories: Sequence[SampledTrajectory],
    *,
    utility_artifact: Mapping,
    environment_hash: str,
    count_reward: ProfileCountTargets,
    cache: UtilityCache,
    remote_workers: int,
    reader_workers: int,
    score_batch: Callable[..., Sequence[UtilityResult]] | None = None,
    cache_only: bool = False,
) -> tuple[TrajectoryPoint, ...]:
    """Score complete trajectories and bind their fixed-denominator U/P points."""

    documents = tuple(documents)
    trajectories = tuple(trajectories)
    documents_by_id = {document.doc_id: document for document in documents}
    if len(documents_by_id) != len(documents):
        raise ValueError("trajectory scoring document IDs must be unique")
    requests = []
    for trajectory in trajectories:
        try:
            document = documents_by_id[trajectory.doc_id]
        except KeyError as error:
            raise ValueError(
                f"trajectory document not supplied: {trajectory.doc_id}"
            ) from error
        requests.append(UtilityRequest(
            document=document,
            action_vector=trajectory.action_vector,
            utility_artifact=utility_artifact,
            environment_hash=environment_hash,
        ))
    if cache_only:
        _require_cached(
            requests, cache=cache, reader_refresh=False, phase="initial",
        )
    scorer = score_batch or _default_score_batch
    results = tuple(scorer(
        requests,
        cache=cache,
        remote_workers=remote_workers,
        reader_workers=reader_workers,
        reader_refresh=False,
    ))
    if len(results) != len(trajectories):
        raise ValueError("utility scorer returned the wrong trajectory result count")
    return tuple(
        trajectory_point(
            trajectory,
            result,
            count_reward=count_reward,
            utility_artifact=utility_artifact,
        )
        for trajectory, result in zip(trajectories, results, strict=True)
    )


def _leave_one_out(values: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in values)
    if len(values) < 2:
        raise ValueError("hybrid utility diagnostics require at least two rollouts")
    total = sum(values)
    return tuple(
        value - (total - value) / (len(values) - 1) for value in values
    )


def _utility_family_terms_and_mass(
    replayed: Sequence[ReplayedTrajectory],
    credit: DocumentUtilityCredit,
    counterfactual_losses: Mapping[tuple[int, str], torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    rollout_count = len(replayed)
    prototype = replayed[0].steps[0].log_prob
    terms: dict[str, list[torch.Tensor]] = {
        "linked": [], "residual": [], "fallback": [], "counterfactual": [],
    }
    residual = _leave_one_out(credit.residual_utility)
    document = _leave_one_out(credit.document_utility)
    linked = {
        decision_id: _leave_one_out(values)
        for decision_id, values in credit.linked_utility.items()
    }
    for rollout_index, trajectory in enumerate(replayed):
        for step in trajectory.steps:
            pair = (rollout_index, step.decision_id)
            if pair in counterfactual_losses:
                terms["counterfactual"].append(counterfactual_losses[pair])
                continue
            route = credit.route[step.decision_id]
            if route == "linked":
                terms["linked"].append(
                    -linked[step.decision_id][rollout_index] * step.log_prob
                )
                terms["residual"].append(
                    -residual[rollout_index] * step.log_prob
                )
            elif route == "document":
                terms["fallback"].append(
                    -document[rollout_index] * step.log_prob
                )
            else:
                raise ValueError(f"unsupported utility credit route: {route}")
    losses = {
        name: (
            torch.stack(rows).sum() / rollout_count
            if rows else prototype.new_zeros(())
        )
        for name, rows in terms.items()
    }
    masses = {
        name: sum(abs(float(row.detach())) for row in rows) / rollout_count
        for name, rows in terms.items()
    }
    return losses, masses


def _reference_distributions(
    reference_policy: TrajectoryPolicy | None,
    document: RankerDocument,
    trajectories: Sequence[SampledTrajectory],
    profile: Any,
) -> tuple[tuple[torch.Tensor, ...], ...] | None:
    if reference_policy is None:
        return None
    with torch.no_grad():
        return tuple(
            tuple(
                step.log_probs.detach().clone()
                for step in replay_trajectory(
                    reference_policy, document, trajectory, profile,
                ).steps
            )
            for trajectory in trajectories
        )


def _gradient_norm(
    term: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> float:
    if not term.requires_grad:
        return 0.0
    parameters = tuple(
        parameter for parameter in parameters if parameter.requires_grad
    )
    if not parameters:
        return 0.0
    gradients = torch.autograd.grad(
        term,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared = sum(
        float(torch.sum(gradient.detach() ** 2))
        for gradient in gradients if gradient is not None
    )
    return math.sqrt(squared)


def _unique_parameters(
    values: Sequence[torch.nn.Parameter],
) -> tuple[torch.nn.Parameter, ...]:
    seen: set[int] = set()
    result = []
    for parameter in values:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return tuple(result)


def semantic_parameter_groups(
    policy: torch.nn.Module,
) -> Mapping[str, tuple[torch.nn.Parameter, ...]]:
    """Return disjoint semantic ownership groups used only for diagnostics."""

    if _policy_architecture_name(policy) != "semantic-v1":
        return MappingProxyType({})

    def declared_or_modules(method_name: str, module_names: Sequence[str]):
        method = getattr(policy, method_name, None)
        if callable(method):
            return _unique_parameters(tuple(method()))
        parameters = []
        for name in module_names:
            module = getattr(policy, name, None)
            if isinstance(module, torch.nn.Module):
                parameters.extend(module.parameters())
        return _unique_parameters(tuple(parameters))

    utility = declared_or_modules("utility_parameters", (
        "utility_projection", "context_readout", "context_to_relation",
        "interaction_projection", "action_mode_embedding",
        "runtime_type_embedding", "utility_head",
    ))
    history = declared_or_modules("history_parameters", ("memory",))
    privacy = declared_or_modules("privacy_parameters", ("privacy_head",))
    alpha = (getattr(policy, "alpha_raw"),)
    if not isinstance(alpha[0], torch.nn.Parameter):
        raise ValueError("semantic-v1 policy lacks alpha_raw")
    groups = {
        "utility": utility,
        "history": history,
        "privacy": privacy,
        "alpha": alpha,
    }
    owners: dict[int, str] = {}
    for name, parameters in groups.items():
        for parameter in parameters:
            prior = owners.setdefault(id(parameter), name)
            if prior != name:
                raise ValueError(
                    f"semantic parameter belongs to both {prior} and {name}"
                )
    return MappingProxyType(groups)


def semantic_gradient_diagnostics(
    terms: Mapping[str, torch.Tensor],
    policy: torch.nn.Module,
) -> Mapping[str, Mapping[str, float]]:
    groups = semantic_parameter_groups(policy)
    return MappingProxyType({
        name: MappingProxyType({
            group: _gradient_norm(term, parameters)
            for group, parameters in groups.items()
        })
        for name, term in terms.items()
    })


def train_hybrid_document_group(
    policy: torch.nn.Module,
    reference_policy: TrajectoryPolicy | None,
    document: RankerDocument,
    profile: Any,
    *,
    rollouts: int,
    utility_artifact: Mapping,
    environment_hash: str,
    profile_targets: ProfileCountTargets,
    cache: UtilityCache,
    optimizer: torch.optim.Optimizer,
    beta: float,
    eta: float,
    counterfactual_budget: int,
    endpoint_budget: int,
    pair_history: Mapping,
    seed: int,
    current_round: int,
    remote_workers: int,
    reader_workers: int,
    generator: torch.Generator,
    score_batch: Callable[..., Sequence[UtilityResult]] | None = None,
    scheduler: Callable | None = None,
    counterfactual_executor: Callable | None = None,
    cache_only: bool = False,
    max_grad_norm: float | None = None,
    counterfactual_coverage: str = "fixed",
    kl_direction: str = "forward",
    profile_sensitivity_coefficient: float = 0.0,
    profile_sensitivity_target: float = 0.0,
    sensitivity_profiles: Sequence[Any] | None = None,
    tie_evidence: dict | None = None,
    tie_labels: Mapping | None = None,
    tie_coefficient: float = 0.0,
    tie_margin: float = 0.1,
    gain_penalty_coefficient: float = 1e-3,
    tie_max_profile: Any = None,
) -> DocumentTrainingResult:
    """Run one complete sampled/scored/replayed optimizer step for a document group."""

    if rollouts < 2:
        raise ValueError("hybrid training requires at least two rollouts")
    architecture = _policy_architecture_name(policy)
    if max_grad_norm is not None and (
        not math.isfinite(float(max_grad_norm)) or max_grad_norm <= 0.0
    ):
        raise ValueError("frozen max_grad_norm must be finite and positive")
    from cloak.ranker.counterfactuals import (
        execute_counterfactuals,
        pair_history_key,
        schedule_counterfactuals,
    )

    scheduler = scheduler or schedule_counterfactuals
    counterfactual_executor = counterfactual_executor or execute_counterfactuals
    trajectories = tuple(
        sample_trajectory(
            policy, document, profile, greedy=False, generator=generator,
        )
        for _ in range(rollouts)
    )
    hits_before = cache.hits
    points = score_trajectories(
        (document,),
        trajectories,
        utility_artifact=utility_artifact,
        environment_hash=environment_hash,
        count_reward=profile_targets,
        cache=cache,
        remote_workers=remote_workers,
        reader_workers=reader_workers,
        score_batch=score_batch,
        cache_only=cache_only,
    )
    trajectory_cache_metrics = dict(cache.last_batch_metrics)
    credit = provisional_credit(
        tuple(point.component_scores for point in points),
        utility_artifact,
        document.doc_id,
    )
    replayed = tuple(
        replay_trajectory(policy, document, trajectory, profile)
        for trajectory in trajectories
    )
    effective_budget = counterfactual_budget
    effective_endpoint = endpoint_budget
    scheduler_options: dict[str, Any] = {}
    executor_options: dict[str, Any] = {}
    if counterfactual_coverage == "degeneracy":
        # Small-document credit-support fork (decision log 2026-07-29): dedup
        # identical-vector pairs, broadcast measured losses back to them, and
        # widen the unique-intervention budget when the group is degenerate so
        # every decision of the dominant vector can be probed.
        unique_vectors = len({
            tuple(sorted(trajectory.action_vector.items()))
            for trajectory in trajectories
        })
        if unique_vectors <= 3:
            coverage_budget = min(
                15, 5 * math.ceil(len(document.policy_decisions) / 5),
            )
            if coverage_budget > counterfactual_budget:
                effective_budget = coverage_budget
                effective_endpoint = (
                    coverage_budget * endpoint_budget // counterfactual_budget
                )
        scheduler_options["dedup_identical_vectors"] = True
        executor_options["broadcast"] = True
    requests, scheduler_diagnostics = scheduler(
        {document.doc_id: document},
        trajectories,
        replayed,
        utility_artifact=utility_artifact,
        credit_routes={document.doc_id: credit.route},
        pair_history=pair_history,
        budget=effective_budget,
        endpoint_budget=effective_endpoint,
        seed=seed,
        current_round=current_round,
        **scheduler_options,
    )
    if cache_only and requests:
        utility_requests = []
        for request in requests:
            alternative = dict(trajectories[request.rollout_index].action_vector)
            alternative[request.decision_id] = request.alternative_action_id
            utility_requests.append(UtilityRequest(
                document=document,
                action_vector=alternative,
                utility_artifact=utility_artifact,
                environment_hash=environment_hash,
            ))
        _require_cached(
            utility_requests,
            cache=cache,
            reader_refresh=False,
            phase="counterfactual",
        )
    counterfactual_losses, scheduler_diagnostics = counterfactual_executor(
        requests,
        {document.doc_id: document},
        trajectories,
        replayed,
        utility_artifact=utility_artifact,
        environment_hash=environment_hash,
        cache=cache,
        remote_workers=remote_workers,
        reader_workers=reader_workers,
        scheduler_diagnostics=scheduler_diagnostics,
        score_batch=score_batch,
        **executor_options,
    )
    counterfactual_cache_metrics = dict(cache.last_batch_metrics)
    if tie_evidence is not None:
        record_tie_evidence(
            tie_evidence,
            scheduler_diagnostics.get("evidence_rows", ()),
            current_round,
        )
    if requests:
        if not isinstance(pair_history, dict):
            raise ValueError("counterfactual pair history must be mutable during training")
        for request in requests:
            pair_history[pair_history_key(
                request.doc_id,
                request.rollout_index,
                request.decision_id,
                request.selected_action_id,
                request.alternative_action_id,
            )] = current_round
    utility_loss = hybrid_utility_loss(replayed, credit, counterfactual_losses)
    reference_log_probs = _reference_distributions(
        reference_policy, document, trajectories, profile,
    )
    objective = compose_hybrid_document_objective(
        replayed,
        utility_loss=utility_loss,
        profile_targets=profile_targets,
        lambda_value=float(profile.value),
        beta=beta,
        eta=eta,
        reference_log_probs=reference_log_probs,
        kl_direction=kl_direction,
    )
    total_objective = objective.total
    tie_term = None
    tie_penalty_term = None
    if tie_coefficient > 0.0 and tie_labels is not None:
        # Families must be identical across every group of an epoch, so the
        # tie terms are ALWAYS emitted when the machinery is on — exact zeros
        # for documents without qualified pairs.
        if tie_max_profile is None:
            raise ValueError("tie margin loss requires the max-lambda profile")
        hinge, residual_penalty, tie_satisfied, tie_total = tie_margin_loss(
            policy, document, trajectories[0], tie_labels,
            max_profile=tie_max_profile, margin=tie_margin,
        )
        scheduler_diagnostics["tie_pairs_total"] = tie_total
        scheduler_diagnostics["tie_pairs_satisfied"] = tie_satisfied
        tie_term = float(tie_coefficient) * hinge
        tie_penalty_term = float(gain_penalty_coefficient) * residual_penalty
        total_objective = total_objective + tie_term + tie_penalty_term
    sensitivity_term = None
    if profile_sensitivity_coefficient > 0.0:
        if not sensitivity_profiles:
            raise ValueError(
                "profile sensitivity regularizer requires the full profile menu"
            )
        sensitivity_term = float(profile_sensitivity_coefficient) * (
            profile_sensitivity_loss(
                replayed, policy, sensitivity_profiles,
                profile_sensitivity_target,
            )
        )
        total_objective = total_objective + sensitivity_term
    family_terms, absolute_mass = _utility_family_terms_and_mass(
        replayed, credit, counterfactual_losses,
    )
    family_terms.update({
        "count": objective.count,
        "entropy": -float(beta) * objective.entropy,
        "KL": float(eta) * objective.kl,
    })
    if sensitivity_term is not None:
        family_terms["profile_sensitivity"] = sensitivity_term
    if tie_term is not None:
        family_terms["tie_margin"] = tie_term
        family_terms["gain_penalty"] = tie_penalty_term
    reconstructed = torch.stack(tuple(family_terms.values())).sum()
    # fp32 tolerance: the family split rounds -(l+r)·logp as two products and
    # sums in a different order than the objective; at ~200 terms the drift
    # legitimately exceeds 1e-7 (seed-29 production run, 2026-07-29).
    if not torch.allclose(reconstructed, total_objective, rtol=1e-5, atol=1e-6):
        raise ValueError(
            "hybrid family terms do not reconstruct the objective: "
            f"{float(reconstructed)!r} vs {float(total_objective)!r}"
        )
    parameters = tuple(
        parameter for parameter in policy.parameters() if parameter.requires_grad
    )
    gradient_norms = {
        name: _gradient_norm(term, parameters)
        for name, term in family_terms.items()
    }
    parameter_group_gradient_norms = (
        semantic_gradient_diagnostics(family_terms, policy)
        if architecture == "semantic-v1"
        else {}
    )
    alpha_diagnostics: dict[str, float] = {}
    if architecture == "semantic-v1":
        alpha_raw = getattr(policy, "alpha_raw")
        alpha_gradient = torch.autograd.grad(
            total_objective, alpha_raw, retain_graph=True, allow_unused=True,
        )[0]
        alpha_value = getattr(policy, "alpha", None)
        if callable(alpha_value):
            alpha_value = alpha_value()
        if not isinstance(alpha_value, torch.Tensor):
            alpha_value = torch.nn.functional.softplus(alpha_raw)
        alpha_diagnostics = {
            "value": float(alpha_value.detach()),
            "gradient": (
                float(alpha_gradient.detach()) if alpha_gradient is not None else 0.0
            ),
        }
    absolute_mass.update({
        name: abs(float(family_terms[name].detach()))
        for name in family_terms
        if name not in absolute_mass
    })
    optimizer.zero_grad(set_to_none=True)
    total_objective.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(parameters, float(max_grad_norm))
    if any(
        parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    ):
        raise ValueError("hybrid optimizer produced a non-finite gradient")
    optimizer.step()

    decisions = {
        decision.decision_id: decision for decision in document.policy_decisions
    }
    modes: Counter[str] = Counter()
    collisions = 0
    for trajectory in trajectories:
        for step in trajectory.steps:
            action = _action_by_id(
                decisions[step.decision_id], step.selected_action_id,
            )
            modes[action.mode] += 1
            collisions += len(step.legal_action_ids) < len(
                decisions[step.decision_id].actions
            )
    type_exposure = Counter(
        decision.runtime_type for decision in document.policy_decisions
    )
    type_rows: dict[str, dict[str, Any]] = {}
    for runtime_type, exposure in sorted(type_exposure.items()):
        selected_scores = []
        entropies = []
        type_modes: Counter[str] = Counter()
        type_collisions = 0
        for trajectory, replay in zip(trajectories, replayed, strict=True):
            replay_by_decision = {
                step.decision_id: step for step in replay.steps
            }
            for step in trajectory.steps:
                decision = decisions[step.decision_id]
                if decision.runtime_type != runtime_type:
                    continue
                action = _action_by_id(decision, step.selected_action_id)
                type_modes[action.mode] += 1
                type_collisions += len(step.legal_action_ids) < len(decision.actions)
                selected_scores.append(float(profile_targets.action_scores(
                    decision.decision_id, (step.selected_action_id,),
                )[0]))
                entropies.append(float(
                    replay_by_decision[step.decision_id].entropy.detach()
                ))
        type_rows[runtime_type] = {
            "exposure": exposure,
            "utility": sum(credit.document_utility) / rollouts,
            "count_score": _mean(selected_scores),
            "entropy": _mean(entropies),
            "collisions": type_collisions,
            "action_modes": dict(sorted(type_modes.items())),
        }
    stage_cache_metrics = {
        "trajectory": trajectory_cache_metrics,
        "counterfactual": counterfactual_cache_metrics,
    }
    numeric_cache_keys = {
        key
        for row in stage_cache_metrics.values()
        for key, value in row.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    cache_metrics = {
        key: sum(float(row.get(key, 0.0)) for row in stage_cache_metrics.values())
        for key in sorted(numeric_cache_keys)
    }
    cache_metrics.setdefault("cache_hits", cache.hits - hits_before)
    cache_metrics["requested_rollouts"] = rollouts
    cache_metrics["stages"] = stage_cache_metrics
    privacy_diagnostics: dict[str, Any] = {}
    lambda_zero_identity_failures = 0
    if architecture == "semantic-v1":
        target_rows = {
            row.action_id: row for row in profile_targets.target_rows()
        }
        level_counts = Counter(
            row.decision_id for row in target_rows.values() if row.mode == "level"
        )
        selected_count = 0
        predicted_sum = 0.0
        exact_sum = 0.0
        absolute_error_sum = 0.0
        strata: dict[str, dict[str, float]] = {}
        for replay in replayed:
            for step in replay.steps:
                selected_index = step.legal_action_ids.index(step.selected_action_id)
                predicted = float(step.predicted_privacy[selected_index].detach())
                exact = float(profile_targets.action_scores(
                    step.decision_id, (step.selected_action_id,),
                )[0])
                target = target_rows[step.selected_action_id]
                normalization = (
                    "singleton_profile"
                    if level_counts[step.decision_id] == 1
                    else "multi_level_profile"
                )
                provenance = target.grounding_status or f"{target.mode}_endpoint"
                source_family = target.source_family or "none"
                stratum = (
                    f"{normalization}|grounding_status={provenance}|"
                    f"source_family={source_family}"
                )
                values = strata.setdefault(stratum, {
                    "count": 0, "predicted_sum": 0.0, "exact_sum": 0.0,
                    "absolute_error_sum": 0.0,
                })
                error = abs(predicted - exact)
                values["count"] += 1
                values["predicted_sum"] += predicted
                values["exact_sum"] += exact
                values["absolute_error_sum"] += error
                selected_count += 1
                predicted_sum += predicted
                exact_sum += exact
                absolute_error_sum += error
                if float(profile.value) == 0.0 and not torch.equal(
                    step.log_probs,
                    torch.log_softmax(step.utility_logits, dim=0),
                ):
                    lambda_zero_identity_failures += 1
        privacy_diagnostics = {
            "selected_count": selected_count,
            "predicted_sum": predicted_sum,
            "exact_sum": exact_sum,
            "absolute_error_sum": absolute_error_sum,
            "strata": strata,
        }
    return DocumentTrainingResult(
        doc_id=document.doc_id,
        corpus=document.corpus,
        profile_name=str(profile.name),
        rollout_count=rollouts,
        loss=float(total_objective.detach()),
        utility=sum(credit.document_utility) / rollouts,
        count_score=sum(point.count_score for point in points) / rollouts,
        entropy=float(objective.entropy.detach()),
        collision_count=collisions,
        action_modes=dict(sorted(modes.items())),
        runtime_type_exposure=dict(sorted(type_exposure.items())),
        gradient_norms=gradient_norms,
        absolute_weighted_mass=absolute_mass,
        scheduler_diagnostics=scheduler_diagnostics,
        cache_metrics=cache_metrics,
        action_vector_hashes=tuple(
            stable_hash(list(_vector_key(document, trajectory.action_vector)))
            for trajectory in trajectories
        ),
        runtime_type_metrics=type_rows,
        parameter_group_gradient_norms=parameter_group_gradient_norms,
        alpha_diagnostics=alpha_diagnostics,
        privacy_diagnostics=privacy_diagnostics,
        lambda_zero_identity_failures=lambda_zero_identity_failures,
    )


def _mean(rows: Sequence[float]) -> float:
    return sum(rows) / len(rows) if rows else 0.0


def build_epoch_report(
    epoch: int,
    groups: Sequence[DocumentTrainingResult],
) -> dict[str, Any]:
    """Aggregate detached training diagnostics without mixing allocation and reward."""

    groups = tuple(groups)
    if epoch < 0 or not groups:
        raise ValueError("epoch report requires a nonnegative epoch and document groups")
    families = tuple(groups[0].gradient_norms)
    if any(
        tuple(group.gradient_norms) != families
        or tuple(group.absolute_weighted_mass) != families
        for group in groups
    ):
        raise ValueError("epoch groups have inconsistent diagnostic families")

    def strata(key):
        grouped: dict[str, list[DocumentTrainingResult]] = {}
        for group in groups:
            grouped.setdefault(str(key(group)), []).append(group)
        return {
            name: {
                "groups": len(rows),
                "utility": _mean([row.utility for row in rows]),
                "count_score": _mean([row.count_score for row in rows]),
                "entropy": _mean([row.entropy for row in rows]),
                "collisions": sum(row.collision_count for row in rows),
                "action_modes": dict(sorted(sum(
                    (Counter(row.action_modes) for row in rows), Counter()
                ).items())),
            }
            for name, rows in sorted(grouped.items())
        }

    runtime_types: dict[str, int] = Counter()
    runtime_rows: dict[str, list[Mapping[str, Any]]] = {}
    for group in groups:
        runtime_types.update(group.runtime_type_exposure)
        for runtime_type, row in group.runtime_type_metrics.items():
            runtime_rows.setdefault(runtime_type, []).append(row)
    scheduler_rows = [dict(group.scheduler_diagnostics) for group in groups]
    cache_rows = [dict(group.cache_metrics) for group in groups]
    numeric_scheduler = {
        key: sum(float(row.get(key, 0)) for row in scheduler_rows)
        for key in ("budget", "uniform_allocation", "priority_allocation", "cache_hits")
    }
    numeric_cache = {
        key: sum(float(row.get(key, 0)) for row in cache_rows)
        for key in sorted({key for row in cache_rows for key in row})
        if all(isinstance(row.get(key, 0), int | float) for row in cache_rows)
    }
    numeric_scheduler["delta_u"] = [row.get("delta_u", {}) for row in scheduler_rows]
    numeric_scheduler["groups"] = scheduler_rows
    report = {
        "report_version": "ranker-v2-epoch-report-v1",
        "epoch": epoch,
        "document_groups": len(groups),
        "loss": _mean([group.loss for group in groups]),
        "term_families": {
            family: {
                "detached_gradient_norm": _mean([
                    group.gradient_norms[family] for group in groups
                ]),
                "absolute_weighted_mass": sum(
                    group.absolute_weighted_mass[family] for group in groups
                ),
            }
            for family in families
        },
        "profiles": strata(lambda row: row.profile_name),
        "corpora": strata(lambda row: row.corpus),
        "runtime_types": {
            name: {
                "exposure": count,
                "utility": _mean([
                    float(row["utility"]) for row in runtime_rows.get(name, ())
                ]),
                "count_score": _mean([
                    float(row["count_score"]) for row in runtime_rows.get(name, ())
                ]),
                "entropy": _mean([
                    float(row["entropy"]) for row in runtime_rows.get(name, ())
                ]),
                "collisions": sum(
                    int(row["collisions"]) for row in runtime_rows.get(name, ())
                ),
                "action_modes": dict(sorted(sum(
                    (
                        Counter(row["action_modes"])
                        for row in runtime_rows.get(name, ())
                    ),
                    Counter(),
                ).items())),
            }
            for name, count in sorted(runtime_types.items())
        },
        "conditional_samples": {
            group.doc_id: {
                group.profile_name: list(group.action_vector_hashes),
            }
            for group in groups
        },
        "scheduler": numeric_scheduler,
        "cache": numeric_cache,
    }
    semantic_rows = [
        group for group in groups if group.parameter_group_gradient_norms
    ]
    if semantic_rows:
        semantic_families = tuple(
            semantic_rows[0].parameter_group_gradient_norms
        )
        parameter_groups = tuple(
            next(iter(
                semantic_rows[0].parameter_group_gradient_norms.values()
            ))
        )
        if any(
            tuple(row.parameter_group_gradient_norms) != semantic_families
            or any(
                tuple(values) != parameter_groups
                for values in row.parameter_group_gradient_norms.values()
            )
            for row in semantic_rows
        ):
            raise ValueError("epoch groups have inconsistent semantic diagnostics")

        selected_count = sum(
            int(row.privacy_diagnostics.get("selected_count", 0))
            for row in semantic_rows
        )
        selected = {
            "count": selected_count,
            "predicted_mean": (
                sum(float(row.privacy_diagnostics.get("predicted_sum", 0.0))
                    for row in semantic_rows) / selected_count
                if selected_count else None
            ),
            "exact_mean": (
                sum(float(row.privacy_diagnostics.get("exact_sum", 0.0))
                    for row in semantic_rows) / selected_count
                if selected_count else None
            ),
            "mean_absolute_error": (
                sum(float(row.privacy_diagnostics.get("absolute_error_sum", 0.0))
                    for row in semantic_rows) / selected_count
                if selected_count else None
            ),
        }
        strata_totals: dict[str, dict[str, float]] = {}
        for row in semantic_rows:
            for name, values in row.privacy_diagnostics.get("strata", {}).items():
                total = strata_totals.setdefault(name, {
                    "count": 0, "predicted_sum": 0.0, "exact_sum": 0.0,
                    "absolute_error_sum": 0.0,
                })
                for key in total:
                    total[key] += values.get(key, 0)
        privacy_strata = {
            name: {
                "count": int(values["count"]),
                "predicted_mean": values["predicted_sum"] / values["count"],
                "exact_mean": values["exact_sum"] / values["count"],
                "mean_absolute_error": (
                    values["absolute_error_sum"] / values["count"]
                ),
            }
            for name, values in sorted(strata_totals.items())
            if values["count"]
        }
        alpha_by_lambda = {}
        for profile_name in sorted({row.profile_name for row in semantic_rows}):
            rows = [row for row in semantic_rows if row.profile_name == profile_name]
            alpha_by_lambda[profile_name] = {
                key: _mean([
                    float(row.alpha_diagnostics[key]) for row in rows
                ])
                for key in ("value", "gradient")
            }
        report["semantic_diagnostics"] = {
            "parameter_group_gradient_norms": {
                family: {
                    group: _mean([
                        row.parameter_group_gradient_norms[family][group]
                        for row in semantic_rows
                    ])
                    for group in parameter_groups
                }
                for family in semantic_families
            },
            "alpha_by_lambda": alpha_by_lambda,
            "selected_privacy": selected,
            "privacy_strata": privacy_strata,
            "lambda_zero_identity_failures": sum(
                row.lambda_zero_identity_failures for row in semantic_rows
            ),
        }
    return report


def _canonical_vector_hash(values: Sequence[str]) -> str:
    counts = Counter(values)
    if not counts:
        raise ValueError("conditional responsiveness lacks sampled vectors")
    maximum = max(counts.values())
    return min(value for value, count in counts.items() if count == maximum)


def _block_responsiveness(
    reports: Sequence[Mapping],
    profile_names: Sequence[str],
) -> dict[str, int | float]:
    samples: dict[str, dict[str, str]] = {}
    for report in reports:
        for doc_id, profile_rows in report.get("conditional_samples", {}).items():
            for profile_name, hashes in profile_rows.items():
                if profile_name in samples.setdefault(doc_id, {}):
                    raise ValueError("profile block repeats a document/profile assignment")
                samples[doc_id][profile_name] = _canonical_vector_hash(hashes)
    expected = set(profile_names)
    incomplete = sorted(
        doc_id for doc_id, rows in samples.items() if set(rows) != expected
    )
    if incomplete:
        raise ValueError(f"profile block has incomplete document exposure: {incomplete}")
    eligible = 0
    changed = 0
    for rows in samples.values():
        for left, right in zip(profile_names, profile_names[1:]):
            eligible += 1
            changed += rows[left] != rows[right]
    return {
        "eligible_adjacent_pairs": eligible,
        "changed_adjacent_pairs": changed,
        "adjacent_winner_change": changed / eligible if eligible else 0.0,
    }


def train_hybrid_policy(
    *,
    policy: Any,
    reference_policy: Any,
    documents: Sequence[RankerDocument],
    profiles: Sequence[Any],
    schedule: LatinCycleSchedule,
    optimizer: Any,
    utility_artifact: Mapping,
    environment_hash: str,
    profile_targets: ProfileCountTargets,
    cache: Any,
    threshold_manifest: Mapping,
    max_epochs: int,
    rollouts: int,
    beta: float,
    eta: float,
    counterfactual_budget: int,
    endpoint_budget: int,
    pair_history: Mapping,
    seed: int,
    remote_workers: int,
    reader_workers: int,
    generator: torch.Generator,
    start_epoch: int = 0,
    existing_reports: Sequence[Mapping] = (),
    kl_enabled: bool = False,
    cache_only: bool = False,
    max_grad_norm: float | None = None,
    score_batch: Callable[..., Sequence[UtilityResult]] | None = None,
    group_trainer: Callable = train_hybrid_document_group,
    epoch_callback: Callable[[int, Sequence[Mapping], Mapping, bool], None] | None = None,
    rollout_scaling: str = "fixed",
    counterfactual_coverage: str = "fixed",
    kl_schedule: str = "collapse-trigger",
    kl_direction: str = "forward",
    synchronous_profile_eval: bool = False,
    profile_sensitivity_coefficient: float = 0.0,
    profile_sensitivity_target: float = 0.0,
    tie_evidence: dict | None = None,
    tie_mode: str = "none",
    tie_coefficient: float = 1.0,
    tie_margin: float = 0.1,
    tie_min_contexts: int = 3,
    tie_projection_lr: float = 1e-2,
    gain_penalty_coefficient: float = 1e-3,
) -> HybridTrainingResult:
    """Train document groups in a seeded balanced profile cycle."""

    documents = tuple(documents)
    profiles = tuple(profiles)
    if max_epochs <= start_epoch or start_epoch < 0:
        raise ValueError("hybrid epoch range must contain at least one epoch")
    if not documents or not profiles:
        raise ValueError("hybrid training requires documents and profiles")
    if schedule.profile_names != tuple(profile.name for profile in profiles) or (
        set(schedule.offsets_by_document) != {document.doc_id for document in documents}
    ):
        raise ValueError("hybrid schedule differs from documents or profiles")
    reports = [dict(row) for row in existing_reports]
    history = pair_history if isinstance(pair_history, dict) else dict(pair_history)
    if kl_schedule == "always-on":
        # KL-anchor fix (decision log 2026-07-30): the collapse trigger is
        # aggregate-level and fires too late to recover a saturated policy;
        # anchor to the calibrated reference from epoch zero instead.
        kl_enabled = True
    elif kl_schedule != "collapse-trigger":
        raise ValueError(f"unsupported KL schedule: {kl_schedule!r}")
    documents_by_id = {document.doc_id: document for document in documents}
    tie_labels: dict = {}
    if tie_mode not in ("none", "online", "cycle"):
        raise ValueError(f"unsupported tie mode: {tie_mode!r}")
    for epoch in range(start_epoch, max_epochs):
        if tie_mode != "none" and tie_evidence is not None and (
            epoch % len(profiles) == 0
        ):
            # One-cycle label lag: qualification uses only evidence measured
            # in strictly earlier rounds (round-3 adjudication 2026-07-31).
            tie_labels = compute_tie_labels(
                tie_evidence, documents_by_id, profile_targets,
                min_contexts=tie_min_contexts, before_round=epoch,
            )
            if tie_mode == "cycle" and tie_labels and epoch > start_epoch:
                # Arm B: full-batch controller-only projection at the cycle
                # boundary; the utility tower and global alpha are untouched.
                projection_optimizer = torch.optim.Adam(
                    policy.gain_head.parameters(), lr=float(tie_projection_lr),
                )
                for _ in range(25):
                    hinge_terms = []
                    penalty_terms = []
                    unsatisfied = 0
                    for document in documents:
                        greedy = sample_trajectory(
                            policy, document, profiles[-1],
                            greedy=True, generator=None,
                        )
                        hinge, penalty, satisfied, total_pairs = tie_margin_loss(
                            policy, document, greedy, tie_labels,
                            max_profile=profiles[-1], margin=tie_margin,
                        )
                        if total_pairs:
                            hinge_terms.append(hinge)
                            penalty_terms.append(penalty)
                            unsatisfied += total_pairs - satisfied
                    if not hinge_terms or unsatisfied == 0:
                        break
                    projection_loss = (
                        float(tie_coefficient) * torch.stack(hinge_terms).mean()
                        + float(gain_penalty_coefficient)
                        * torch.stack(penalty_terms).mean()
                    )
                    projection_optimizer.zero_grad(set_to_none=True)
                    projection_loss.backward()
                    projection_optimizer.step()
        epoch_kl_enabled = kl_enabled
        groups = []
        for document_index, document in enumerate(documents):
            profile = schedule.profile_for(document.doc_id, epoch)
            group_rollouts = rollouts
            if rollout_scaling == "support":
                group_rollouts = support_scaled_rollouts(
                    dominant_trajectory_probability(policy, document, profile),
                    rollouts,
                )
            group_options: dict[str, Any] = {}
            if counterfactual_coverage != "fixed":
                group_options["counterfactual_coverage"] = counterfactual_coverage
            if kl_direction != "forward":
                group_options["kl_direction"] = kl_direction
            if profile_sensitivity_coefficient > 0.0:
                group_options["profile_sensitivity_coefficient"] = (
                    profile_sensitivity_coefficient
                )
                group_options["profile_sensitivity_target"] = (
                    profile_sensitivity_target
                )
                group_options["sensitivity_profiles"] = profiles
            if tie_mode != "none" and tie_evidence is not None:
                group_options["tie_evidence"] = tie_evidence
                group_options["tie_max_profile"] = profiles[-1]
                if tie_mode == "online":
                    group_options["tie_labels"] = tie_labels
                    group_options["tie_coefficient"] = tie_coefficient
                    group_options["tie_margin"] = tie_margin
                    group_options["gain_penalty_coefficient"] = (
                        gain_penalty_coefficient
                    )
            groups.append(group_trainer(
                policy,
                reference_policy,
                document,
                profile,
                rollouts=group_rollouts,
                **group_options,
                utility_artifact=utility_artifact,
                environment_hash=environment_hash,
                profile_targets=profile_targets,
                cache=cache,
                optimizer=optimizer,
                beta=beta,
                eta=eta if epoch_kl_enabled else 0.0,
                counterfactual_budget=counterfactual_budget,
                endpoint_budget=endpoint_budget,
                pair_history=history,
                seed=seed + epoch * len(documents) + document_index,
                current_round=epoch,
                remote_workers=remote_workers,
                reader_workers=reader_workers,
                generator=generator,
                score_batch=score_batch,
                cache_only=cache_only,
                max_grad_norm=max_grad_norm,
            ))
        report = build_epoch_report(epoch, groups)
        report["exposure"] = profile_exposure_report(documents, schedule, (epoch,))
        if len(profiles) > 1 and (epoch + 1) % len(profiles) == 0:
            block = (*reports, report)[-len(profiles):]
            responsiveness = _block_responsiveness(
                block, tuple(profile.name for profile in profiles),
            )
            report["conditional_responsiveness"] = responsiveness
            if not kl_enabled and collapse_rule_fires(threshold_manifest, report):
                kl_enabled = True
        else:
            report["conditional_responsiveness"] = {
                "eligible_adjacent_pairs": 0,
                "changed_adjacent_pairs": 0,
                "adjacent_winner_change": None,
            }
        report["kl_enabled_for_epoch"] = bool(eta > 0.0 and epoch_kl_enabled)
        if synchronous_profile_eval:
            report["synchronous_profiles"] = synchronous_profile_snapshot(
                policy, documents, profiles, profile_targets,
                samples=64, seed=seed + 90000 + epoch,
                tie_labels=tie_labels if tie_mode != "none" else None,
            )
        reports.append(report)
        if epoch_callback is not None:
            epoch_callback(epoch, tuple(reports), history, kl_enabled)
    return HybridTrainingResult(
        epoch_reports=tuple(reports),
        schedule=schedule,
        pair_history=history,
        kl_enabled=kl_enabled,
    )


_SEMANTIC_DIRECT_ARTIFACT_PIN_KEYS = frozenset({
    "environment_hash",
    "utility_artifact_hash",
    "profile_target_artifact_hash",
    "representation_manifest_hash",
    "lambda_menu_hash",
    "threshold_manifest_hash",
})
_SEMANTIC_LEARNED_ARTIFACT_PIN_KEYS = (
    _SEMANTIC_DIRECT_ARTIFACT_PIN_KEYS | {"privacy_checkpoint_hash"}
)


def _validate_hybrid_pins(pins: Mapping[str, str]) -> dict[str, str]:
    supplied = set(pins)
    expected = (
        _SEMANTIC_LEARNED_ARTIFACT_PIN_KEYS
        if "privacy_checkpoint_hash" in supplied
        else _SEMANTIC_DIRECT_ARTIFACT_PIN_KEYS
    )
    if supplied != expected or any(
        not isinstance(value, str) or not value for value in pins.values()
    ):
        raise ValueError("hybrid checkpoint artifact pins are incomplete")
    return dict(sorted(pins.items()))


def _semantic_policy_contract(policy: torch.nn.Module) -> dict[str, Any]:
    store = getattr(policy, "representation_store", None)
    manifest = getattr(store, "manifest", {})
    encoder_revision = getattr(policy, "encoder_revision", None)
    if encoder_revision is None and isinstance(manifest, Mapping):
        encoder = manifest.get("encoder", {})
        if isinstance(encoder, Mapping):
            encoder_revision = encoder.get("revision")
    context_mode = getattr(policy, "context_mode", None)
    if context_mode is None:
        context_mode = getattr(getattr(policy, "context_readout", None), "context_mode", None)
    history_mode = getattr(policy, "history_mode", None)
    if history_mode is None:
        history_mode = getattr(getattr(policy, "memory", None), "history_mode", None)
    feature_schema = getattr(policy, "feature_schema", (
        "utility_relation", "candidate_context", "context_relation_interaction",
        "action_mode", "runtime_type", "selected_history",
    ))
    controller_transform = getattr(
        policy, "controller_transform", "log1p-over-log1p-max-v1",
    )
    contract = {
        "encoder_revision": encoder_revision,
        "context_mode": context_mode,
        "history_mode": history_mode,
        "feature_schema": list(feature_schema),
        "controller_transform": controller_transform,
    }
    privacy_signal = getattr(policy, "privacy_signal", None)
    if privacy_signal is not None:
        if not isinstance(privacy_signal, Mapping):
            raise ValueError("semantic policy privacy signal provenance is invalid")
        contract["privacy_signal"] = dict(privacy_signal)
    if (
        not all(isinstance(contract[key], str) and contract[key] for key in (
            "encoder_revision", "context_mode", "history_mode",
            "controller_transform",
        ))
        or not contract["feature_schema"]
        or any(
            not isinstance(value, str) or not value
            for value in contract["feature_schema"]
        )
    ):
        raise ValueError("semantic policy checkpoint contract is incomplete")
    return contract


def policy_architecture_pin(policy: torch.nn.Module) -> str:
    """Hash the conditional policy's structural contract, excluding learned values."""

    architecture = _policy_architecture_name(policy)
    profiles = getattr(policy, "supported_profiles", ())
    supported_values = getattr(policy, "supported_lambda_values", ())
    payload = {
        "policy_architecture": architecture,
        "class": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "profiles": [
            {"name": profile.name, "value": float(profile.value)}
            for profile in profiles
        ],
        "max_menu_value": getattr(policy, "max_menu_value", None),
        "encoder_pin": getattr(policy, "encoder_pin", None),
        "encoder_dim": getattr(policy, "encoder_dim", None),
        "hidden_dim": getattr(policy, "hidden_dim", None),
        "context_window": getattr(policy, "context_window", None),
        "profile_embedding_dim": getattr(policy, "profile_embedding_dim", None),
        "action_feature_names": list(getattr(policy, "action_feature_names", ())),
        "state_layout": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in policy.state_dict().items()
        },
    }
    if architecture == "semantic-v1":
        payload["supported_lambda_values"] = [
            float(value) for value in supported_values
        ]
        payload["semantic_contract"] = _semantic_policy_contract(policy)
    return stable_hash(payload)


def _schedule_payload(schedule: LatinCycleSchedule) -> dict[str, Any]:
    return {
        "profile_names": schedule.profile_names,
        "profile_values": schedule.profile_values,
        "offsets_by_document": dict(schedule.offsets_by_document),
        "seed": schedule.seed,
    }


def _schedule_from_payload(payload: Mapping) -> LatinCycleSchedule:
    if set(payload) != {
        "profile_names", "profile_values", "offsets_by_document", "seed",
    }:
        raise ValueError("hybrid checkpoint schedule state is invalid")
    return LatinCycleSchedule(
        profile_names=tuple(payload["profile_names"]),
        profile_values=tuple(float(value) for value in payload["profile_values"]),
        offsets_by_document={
            str(key): int(value)
            for key, value in payload["offsets_by_document"].items()
        },
        seed=int(payload["seed"]),
    )


def save_hybrid_checkpoint(
    path: str | Path,
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    generator: torch.Generator,
    schedule: LatinCycleSchedule,
    artifact_pins: Mapping[str, str],
    architecture_pin: str,
    cache_paths: Mapping[str, str],
    code_revision: str,
    training_config: Mapping[str, Any],
    pair_history: Mapping,
    kl_enabled: bool,
    epoch_reports: Sequence[Mapping],
    tie_evidence: Mapping | None = None,
) -> None:
    """Atomically save all mutable trainer and reproducibility state."""

    if epoch < 0:
        raise ValueError("hybrid checkpoint epoch must be nonnegative")
    architecture = _policy_architecture_name(policy)
    pins = _validate_hybrid_pins(artifact_pins)
    if not architecture_pin or not code_revision or not cache_paths or not training_config or any(
        not isinstance(key, str) or not isinstance(value, str) or not key or not value
        for key, value in cache_paths.items()
    ):
        raise ValueError("hybrid checkpoint architecture/cache/revision pins are incomplete")
    payload = {
        "checkpoint_version": (
            "ranker-v2-semantic-policy-v1"
            if architecture == "semantic-v1"
            else "ranker-v2-hybrid-checkpoint-v1"
        ),
        "policy_architecture": architecture,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "rng_states": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "generator": generator.get_state(),
        },
        "schedule": _schedule_payload(schedule),
        "artifact_pins": pins,
        "architecture_pin": architecture_pin,
        "cache_paths": dict(sorted(cache_paths.items())),
        "code_revision": code_revision,
        "training_config": dict(sorted(training_config.items())),
        "pair_history": dict(pair_history),
        "tie_evidence": dict(tie_evidence) if tie_evidence is not None else {},
        "kl_enabled": bool(kl_enabled),
        "epoch_reports": tuple(dict(row) for row in epoch_reports),
    }
    if architecture == "semantic-v1":
        payload["semantic_contract"] = _semantic_policy_contract(policy)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_hybrid_checkpoint(
    path: str | Path,
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    expected_artifact_pins: Mapping[str, str],
    expected_architecture_pin: str,
    expected_cache_paths: Mapping[str, str],
    expected_code_revision: str,
    expected_training_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every frozen pin before restoring any mutable trainer state."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    architecture = _policy_architecture_name(policy)
    expected_version = (
        "ranker-v2-semantic-policy-v1"
        if architecture == "semantic-v1"
        else "ranker-v2-hybrid-checkpoint-v1"
    )
    if (
        checkpoint.get("policy_architecture", "legacy-film-gru") != architecture
        or checkpoint.get("checkpoint_version") != expected_version
    ):
        raise ValueError("hybrid checkpoint policy architecture differs")
    if architecture == "semantic-v1" and checkpoint.get(
        "semantic_contract"
    ) != _semantic_policy_contract(policy):
        raise ValueError("semantic checkpoint architecture contract differs")
    if checkpoint.get("artifact_pins") != _validate_hybrid_pins(expected_artifact_pins):
        raise ValueError("hybrid checkpoint artifact pins differ")
    if checkpoint.get("architecture_pin") != expected_architecture_pin:
        raise ValueError("hybrid checkpoint architecture pin differs")
    if checkpoint.get("cache_paths") != dict(sorted(expected_cache_paths.items())):
        raise ValueError("hybrid checkpoint cache paths differ")
    if checkpoint.get("code_revision") != expected_code_revision:
        raise ValueError("hybrid checkpoint code revision differs")
    if checkpoint.get("training_config") != dict(sorted(expected_training_config.items())):
        raise ValueError("hybrid checkpoint training configuration differs")
    schedule = _schedule_from_payload(checkpoint.get("schedule", {}))
    rng = checkpoint.get("rng_states")
    if not isinstance(rng, Mapping) or set(rng) != {"python", "torch", "generator"}:
        raise ValueError("hybrid checkpoint RNG state is incomplete")
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    generator.set_state(rng["generator"])
    return {
        "epoch": int(checkpoint["epoch"]),
        "schedule": schedule,
        "pair_history": dict(checkpoint.get("pair_history", {})),
        "tie_evidence": dict(checkpoint.get("tie_evidence", {})),
        "kl_enabled": bool(checkpoint.get("kl_enabled", False)),
        "epoch_reports": tuple(checkpoint.get("epoch_reports", ())),
        "training_config": dict(checkpoint["training_config"]),
    }


def collapse_rule_fires(
    threshold_manifest: Mapping,
    epoch_report: Mapping,
) -> bool:
    """Apply the frozen adjacent-profile responsiveness boundary exactly once."""

    try:
        threshold = float(
            threshold_manifest["feasibility_gates"]["min_adjacent_winner_change"]
        )
        observed = float(
            epoch_report["conditional_responsiveness"]["adjacent_winner_change"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("collapse responsiveness rule is incomplete") from error
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (
        threshold, observed,
    )):
        raise ValueError("collapse responsiveness rule is invalid")
    return observed < threshold


def collect_exit_winners(
    policy: TrajectoryPolicy,
    documents: Sequence[RankerDocument],
    *,
    lambda_zero: Any,
    rollouts_per_document: int,
    utility_artifact: Mapping,
    environment_hash: str,
    count_reward: ProfileCountTargets,
    cache: UtilityCache,
    remote_workers: int,
    reader_workers: int,
    generator: torch.Generator | None,
    score_batch: Callable[..., Sequence[UtilityResult]] | None = None,
    cache_only: bool = False,
) -> ExitCollection:
    """Collect and serially reverify strict pure-utility improvements over BC."""

    documents = tuple(documents)
    if not documents:
        raise ValueError("ExIt collection requires at least one document")
    if len({document.doc_id for document in documents}) != len(documents):
        raise ValueError("ExIt document IDs must be unique")
    if rollouts_per_document <= 0:
        raise ValueError("ExIt rollouts_per_document must be positive")
    scorer = score_batch or _default_score_batch

    pools: list[tuple[RankerDocument, SampledTrajectory, tuple[SampledTrajectory, ...]]] = []
    initial_requests: list[UtilityRequest] = []
    request_trajectories: list[SampledTrajectory] = []
    for document in documents:
        reference = behavior_clone_trajectory(document, lambda_zero)
        sampled = tuple(
            sample_trajectory(
                policy, document, lambda_zero, greedy=False, generator=generator,
            )
            for _ in range(rollouts_per_document)
        )
        reference_key = _vector_key(document, reference.action_vector)
        unique_candidates: dict[
            tuple[tuple[str, str], ...], SampledTrajectory
        ] = {}
        for trajectory in sampled:
            key = _vector_key(document, trajectory.action_vector)
            if key != reference_key:
                unique_candidates.setdefault(key, trajectory)
        candidates = tuple(unique_candidates.values())
        pools.append((document, reference, candidates))
        for trajectory in (reference, *candidates):
            initial_requests.append(UtilityRequest(
                document=document,
                action_vector=trajectory.action_vector,
                utility_artifact=utility_artifact,
                environment_hash=environment_hash,
            ))
            request_trajectories.append(trajectory)

    if cache_only:
        _require_cached(
            initial_requests,
            cache=cache,
            reader_refresh=False,
            phase="initial",
        )
    initial_results = tuple(scorer(
        initial_requests,
        cache=cache,
        remote_workers=remote_workers,
        reader_workers=reader_workers,
        reader_refresh=False,
    ))
    if len(initial_results) != len(initial_requests):
        raise ValueError("utility scorer returned the wrong initial result count")
    points_by_vector: dict[tuple[str, tuple[tuple[str, str], ...]], TrajectoryPoint] = {}
    for request, trajectory, result in zip(
        initial_requests, request_trajectories, initial_results, strict=True,
    ):
        point = trajectory_point(
            trajectory,
            result,
            count_reward=count_reward,
            utility_artifact=utility_artifact,
        )
        key = (request.document.doc_id, _vector_key(request.document, point.action_vector))
        previous = points_by_vector.setdefault(key, point)
        if previous != point:
            raise ValueError("conflicting utility results for one action vector")

    staged: list[
        tuple[RankerDocument, TrajectoryPoint, tuple[TrajectoryPoint, ...], TrajectoryPoint | None]
    ] = []
    refresh_requests: list[UtilityRequest] = []
    for document, reference_trajectory, candidate_trajectories in pools:
        reference = points_by_vector[
            (document.doc_id, _vector_key(document, reference_trajectory.action_vector))
        ]
        candidates = tuple(
            points_by_vector[(
                document.doc_id, _vector_key(document, trajectory.action_vector)
            )]
            for trajectory in candidate_trajectories
        )
        better = [point for point in candidates if point.utility > reference.utility]
        proposed = max(better, key=lambda point: point.utility, default=None)
        staged.append((document, reference, candidates, proposed))
        if proposed is not None:
            for point in (proposed, reference):
                refresh_requests.append(UtilityRequest(
                    document=document,
                    action_vector=point.action_vector,
                    utility_artifact=utility_artifact,
                    environment_hash=environment_hash,
                ))

    if cache_only and refresh_requests:
        _require_cached(
            refresh_requests,
            cache=cache,
            reader_refresh=True,
            phase="reverification",
        )

    collected: list[ExitDocumentCollection] = []
    for document, reference, candidates, proposed in staged:
        if proposed is None:
            collected.append(ExitDocumentCollection(
                doc_id=document.doc_id,
                reference=reference,
                candidates=candidates,
                winner=None,
                reverified_reference_utility=None,
                verification_status="not_strictly_better",
            ))
            continue
        try:
            candidate_result = tuple(scorer(
                (UtilityRequest(
                    document=document,
                    action_vector=proposed.action_vector,
                    utility_artifact=utility_artifact,
                    environment_hash=environment_hash,
                ),),
                cache=cache,
                remote_workers=remote_workers,
                reader_workers=reader_workers,
                reader_refresh=True,
            ))
            if len(candidate_result) != 1:
                raise ValueError("utility scorer returned wrong candidate refresh count")
            refreshed_candidate = trajectory_point(
                proposed.trajectory,
                candidate_result[0],
                count_reward=count_reward,
                utility_artifact=utility_artifact,
            )
            reference_result = tuple(scorer(
                (UtilityRequest(
                    document=document,
                    action_vector=reference.action_vector,
                    utility_artifact=utility_artifact,
                    environment_hash=environment_hash,
                ),),
                cache=cache,
                remote_workers=remote_workers,
                reader_workers=reader_workers,
                reader_refresh=True,
            ))
            if len(reference_result) != 1:
                raise ValueError("utility scorer returned wrong reference refresh count")
            refreshed_reference = trajectory_point(
                reference.trajectory,
                reference_result[0],
                count_reward=count_reward,
                utility_artifact=utility_artifact,
            )
        except RuntimeError:
            collected.append(ExitDocumentCollection(
                doc_id=document.doc_id,
                reference=reference,
                candidates=candidates,
                winner=None,
                reverified_reference_utility=None,
                verification_status="refresh_failed",
            ))
            continue

        verified = refreshed_candidate.utility > refreshed_reference.utility
        collected.append(ExitDocumentCollection(
            doc_id=document.doc_id,
            reference=reference,
            candidates=candidates,
            winner=refreshed_candidate if verified else None,
            reverified_reference_utility=refreshed_reference.utility,
            verification_status=(
                "verified" if verified else "refresh_not_strictly_better"
            ),
        ))
    return ExitCollection(documents=tuple(collected))


def _point_payload(point: TrajectoryPoint) -> dict[str, Any]:
    return {
        "action_vector": dict(point.action_vector),
        "utility": point.utility,
        "count_score": point.count_score,
        "component_scores": dict(point.component_scores),
        "result_hash": point.result_hash,
    }


def write_exit_winners(
    path: str | Path,
    collection: ExitCollection,
    *,
    pins: Mapping[str, str],
) -> dict[str, Any]:
    """Atomically publish a content-free, hash-bound ExIt calibration pool."""

    semantic_pins = frozenset({
        "environment_hash", "profile_target_artifact_hash",
        "representation_manifest_hash", "utility_artifact_hash",
        "policy_checkpoint_hash",
    })
    allowed_pin_sets = (semantic_pins, semantic_pins | {"privacy_checkpoint_hash"})
    if set(pins) not in allowed_pin_sets or any(
        not isinstance(value, str) or not value for value in pins.values()
    ):
        raise ValueError(
            "ExIt pins must match one architecture pin set exactly: "
            f"{[sorted(values) for values in allowed_pin_sets]}"
        )
    documents = []
    for record in collection.documents:
        documents.append({
            "doc_id": record.doc_id,
            "reference": _point_payload(record.reference),
            "candidates": [_point_payload(point) for point in record.candidates],
            "winner": (
                _point_payload(record.winner) if record.winner is not None else None
            ),
            "reverified_reference_utility": record.reverified_reference_utility,
            "verification_status": record.verification_status,
        })
    payload: dict[str, Any] = {
        "artifact_version": "ranker-v2-exit-winners-v1",
        "pins": dict(sorted(pins.items())),
        "documents": documents,
        "summary": {
            "document_count": len(documents),
            "candidate_count": sum(len(row.candidates) for row in collection.documents),
            "winner_count": sum(row.winner is not None for row in collection.documents),
        },
    }
    payload["artifact_hash"] = stable_hash(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return payload
