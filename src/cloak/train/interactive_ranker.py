"""Stable-ID trajectory sampling and replay for the ranker-v2 environment."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import torch

from cloak.runtime_types import placeholder_token, placeholder_type_token
from cloak.train.ranker_environment import (
    RankerAction,
    RankerDecision,
    RankerDocument,
)
from cloak.train.count_reward import CountReward
from cloak.train.utility_cache import UtilityCache, UtilityRequest, UtilityResult, stable_hash
from cloak.train.utility_credit import DocumentUtilityCredit, document_utility


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
    legal_action_ids: tuple[str, ...]
    selected_action_id: str
    log_prob: torch.Tensor
    entropy: torch.Tensor
    log_probs: torch.Tensor


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

    if replayed_pairs != set(credit.provisional_advantage):
        raise ValueError("credit pairs differ from replayed trajectory pairs")
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
    if replayed_pairs != set(provisional_credit.provisional_advantage):
        raise ValueError("credit pairs differ from replayed trajectory pairs")
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

    def advance(
        self, state: Any, decision: RankerDecision, action_id: str
    ) -> Any: ...


def _fill_key(fill: str) -> str:
    return fill.lower()


def _action_by_id(decision: RankerDecision, action_id: str) -> RankerAction:
    matches = [action for action in decision.actions if action.action_id == action_id]
    if len(matches) != 1:
        raise ValueError(
            f"unknown or duplicate action {action_id!r} for {decision.decision_id}"
        )
    return matches[0]


def _occurrence_maps(
    document: RankerDocument,
) -> tuple[dict[str, Mapping], dict[str, tuple[int, str]]]:
    by_id: dict[str, Mapping] = {}
    for occurrence in document.occurrences:
        occurrence_id = occurrence.get("occurrence_id")
        if not isinstance(occurrence_id, str) or occurrence_id in by_id:
            raise ValueError(f"duplicate or invalid occurrence id in {document.doc_id}")
        start = occurrence.get("start")
        end = occurrence.get("end")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end <= start
            or end > len(document.text)
            or document.text[start:end] != occurrence.get("surface")
        ):
            raise ValueError(f"source occurrence mismatch for {occurrence_id}")
        by_id[occurrence_id] = occurrence

    first_offsets: dict[str, tuple[int, str]] = {}
    all_decisions = (*document.policy_decisions, *document.fixed_decisions)
    decision_ids: set[str] = set()
    action_ids: set[str] = set()
    mapped_ids: set[str] = set()
    for decision in all_decisions:
        if decision.decision_id in decision_ids:
            raise ValueError(f"duplicate decision id: {decision.decision_id}")
        decision_ids.add(decision.decision_id)
        if not decision.occurrence_ids:
            raise ValueError(f"missing mapped occurrence for {decision.decision_id}")
        starts = []
        for occurrence_id in decision.occurrence_ids:
            occurrence = by_id.get(occurrence_id)
            if occurrence is None:
                raise ValueError(f"missing mapped occurrence for {decision.decision_id}")
            if occurrence.get("decision_id") != decision.decision_id:
                raise ValueError(f"occurrence mapping mismatch for {decision.decision_id}")
            if occurrence_id in mapped_ids:
                raise ValueError(f"occurrence mapped more than once: {occurrence_id}")
            mapped_ids.add(occurrence_id)
            starts.append(int(occurrence["start"]))
        first_offsets[decision.decision_id] = (
            min(starts), decision.decision_id,
        )
        if not decision.actions:
            raise ValueError(f"decision {decision.decision_id} has no actions")
        for action in decision.actions:
            if action.action_id in action_ids:
                raise ValueError(f"duplicate action id: {action.action_id}")
            action_ids.add(action.action_id)
            if action.mode not in {"level", "keep", "placeholder"}:
                raise ValueError(f"unsupported action mode: {action.mode}")
            if action.mode == "level" and not action.fill:
                raise ValueError(f"level action {action.action_id} has no fill")

    policy_ids = {decision.decision_id for decision in document.policy_decisions}
    fixed_ids = {decision.decision_id for decision in document.fixed_decisions}
    if policy_ids & fixed_ids:
        raise ValueError("fixed decision present in policy set")
    for fixed in document.fixed_decisions:
        if len(fixed.actions) != 1:
            raise ValueError(f"fixed decision {fixed.decision_id} must have one action")

    for occurrence_id, occurrence in by_id.items():
        decision_id = occurrence.get("decision_id")
        if occurrence.get("controlled") is True and occurrence_id not in mapped_ids:
            raise ValueError(f"missing mapped occurrence for {decision_id}")
        if decision_id is not None and decision_id not in decision_ids:
            raise ValueError(f"unknown occurrence decision: {decision_id}")

    expected_policy_order = tuple(
        sorted(document.policy_decisions, key=lambda row: first_offsets[row.decision_id])
    )
    if document.policy_decisions != expected_policy_order:
        raise ValueError(f"unordered policy decisions in {document.doc_id}")
    return by_id, first_offsets


def legal_action_ids(
    decision: RankerDecision,
    claimed_fills: Mapping[str, str],
    reserved_fixed_fills: Collection[str],
) -> tuple[str, ...]:
    """Return the stable-ID menu after dynamic injectivity masking."""

    normalized_claims = {
        _fill_key(str(fill)): owner for fill, owner in claimed_fills.items()
    }
    reserved = {_fill_key(str(fill)) for fill in reserved_fixed_fills}
    legal: list[str] = []
    for action in decision.actions:
        if action.mode in {"keep", "placeholder"}:
            legal.append(action.action_id)
            continue
        if action.mode != "level" or not action.fill:
            raise ValueError(f"invalid action semantics for {action.action_id}")
        fill_key = _fill_key(action.fill)
        owner = normalized_claims.get(fill_key)
        if fill_key not in reserved and (owner is None or owner == decision.decision_id):
            legal.append(action.action_id)
    return tuple(legal)


def _fixed_fill_claims(
    document: RankerDocument, occurrence_by_id: Mapping[str, Mapping]
) -> dict[str, str]:
    claims: dict[str, str] = {}
    for decision in document.fixed_decisions:
        action = decision.actions[0]
        if action.mode == "placeholder":
            continue
        if action.mode == "level":
            fills = (action.fill,)
        else:
            fills = tuple(
                str(occurrence_by_id[occurrence_id]["surface"])
                for occurrence_id in decision.occurrence_ids
            )
        for fill in fills:
            if not fill:
                raise ValueError(f"fixed decision {decision.decision_id} has no exact fill")
            key = _fill_key(fill)
            owner = claims.setdefault(key, decision.decision_id)
            if owner != decision.decision_id:
                raise ValueError(f"fixed fill collision: {fill!r}")
    return claims


def _case_adjust(fill: str, text: str, start: int) -> str:
    previous = text[:start].rstrip()
    sentence_start = not previous or previous[-1] in ".!?\n"
    return (fill[0].upper() if sentence_start else fill[0].lower()) + fill[1:]


def assemble_action_vector(
    document: RankerDocument,
    action_vector: Mapping[str, str],
) -> tuple[str, list[dict]]:
    """Render one complete policy vector and all automatic fixed decisions."""

    occurrence_by_id, first_offsets = _occurrence_maps(document)
    expected_ids = {decision.decision_id for decision in document.policy_decisions}
    supplied_ids = set(action_vector)
    missing = sorted(expected_ids - supplied_ids)
    if missing:
        raise ValueError(f"action-vector omissions: {missing}")
    extra = sorted(supplied_ids - expected_ids)
    if extra:
        raise ValueError(f"unknown policy decision ids: {extra}")

    selected: dict[str, RankerAction] = {}
    for decision in document.policy_decisions:
        selected[decision.decision_id] = _action_by_id(
            decision, str(action_vector[decision.decision_id])
        )
    for decision in document.fixed_decisions:
        selected[decision.decision_id] = decision.actions[0]

    fixed_claims = _fixed_fill_claims(document, occurrence_by_id)
    claims = dict(fixed_claims)
    for decision in document.policy_decisions:
        action = selected[decision.decision_id]
        if action.mode != "level":
            continue
        assert action.fill is not None
        key = _fill_key(action.fill)
        owner = claims.setdefault(key, decision.decision_id)
        if owner != decision.decision_id:
            raise ValueError(
                f"action-vector fill collision for {decision.decision_id}: {action.fill!r}"
            )

    placeholder_by_decision: dict[str, str] = {}
    counters: dict[str, int] = {}
    all_decisions = sorted(
        (*document.policy_decisions, *document.fixed_decisions),
        key=lambda row: first_offsets[row.decision_id],
    )
    for decision in all_decisions:
        if selected[decision.decision_id].mode != "placeholder":
            continue
        token_type = placeholder_type_token(decision.runtime_type)
        counters[token_type] = counters.get(token_type, 0) + 1
        placeholder_by_decision[decision.decision_id] = placeholder_token(
            decision.runtime_type, counters[token_type]
        )

    replacements: list[dict] = []
    for occurrence in document.occurrences:
        decision_id = occurrence.get("decision_id")
        if decision_id is None:
            continue
        action = selected.get(str(decision_id))
        if action is None:
            raise ValueError(f"unresolved controlled occurrence decision {decision_id}")
        start = int(occurrence["start"])
        end = int(occurrence["end"])
        original = document.text[start:end]
        if action.mode == "keep":
            replacement = original
        elif action.mode == "placeholder":
            replacement = placeholder_by_decision[str(decision_id)]
        else:
            assert action.fill is not None
            replacement = _case_adjust(action.fill, document.text, start)
        replacements.append({
            "occurrence_id": occurrence["occurrence_id"],
            "decision_id": str(decision_id),
            "start": start,
            "end": end,
            "surface": original,
            "replacement": replacement,
            "action_id": action.action_id,
            "mode": action.mode,
        })

    ordered = sorted(replacements, key=lambda row: (row["start"], row["end"]))
    for left, right in zip(ordered, ordered[1:]):
        if right["start"] < left["end"]:
            raise ValueError("overlapping occurrences cannot be assembled")
    rendered = document.text
    for row in reversed(ordered):
        rendered = rendered[:row["start"]] + row["replacement"] + rendered[row["end"]:]
    return rendered, ordered


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
            selected_index = int(
                torch.multinomial(
                    log_probs.exp(), 1, generator=generator
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

        log_probs = _checked_log_probs(policy, state, decision, menu, lambda_profile)
        selected_index = menu.index(sampled.selected_action_id)
        selected_log_prob = log_probs[selected_index]
        entropy = -(log_probs.exp() * log_probs).sum()
        replayed.append(ReplayedStep(
            decision_id=decision.decision_id,
            legal_action_ids=menu,
            selected_action_id=sampled.selected_action_id,
            log_prob=selected_log_prob,
            entropy=entropy,
            log_probs=log_probs,
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
    count_reward: CountReward,
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
    from cloak.train.roundtrip import _cache_identity, _validate_request_readers

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
    from cloak.train.roundtrip import score_roundtrip_batch

    return score_roundtrip_batch(requests, **kwargs)


def score_trajectories(
    documents: Sequence[RankerDocument],
    trajectories: Sequence[SampledTrajectory],
    *,
    utility_artifact: Mapping,
    environment_hash: str,
    count_reward: CountReward,
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


def collect_exit_winners(
    policy: TrajectoryPolicy,
    documents: Sequence[RankerDocument],
    *,
    lambda_zero: Any,
    rollouts_per_document: int,
    utility_artifact: Mapping,
    environment_hash: str,
    count_reward: CountReward,
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

    required_pins = {
        "environment_hash", "count_state_hash", "utility_artifact_hash",
        "policy_checkpoint_hash",
    }
    if set(pins) != required_pins or any(
        not isinstance(value, str) or not value for value in pins.values()
    ):
        raise ValueError(f"ExIt pins must be exactly {sorted(required_pins)}")
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
