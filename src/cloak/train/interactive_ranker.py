"""Stable-ID trajectory sampling and replay for the ranker-v2 environment."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import torch

from cloak.runtime_types import placeholder_token, placeholder_type_token
from cloak.train.ranker_environment import (
    RankerAction,
    RankerDecision,
    RankerDocument,
)
from cloak.train.utility_credit import DocumentUtilityCredit


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
