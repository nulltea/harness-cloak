"""One-decision utility counterfactuals and their fixed-budget scheduler."""
from __future__ import annotations

import copy
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from cloak.ranker.interactive import (
    ReplayedStep,
    ReplayedTrajectory,
    SampledStep,
    SampledTrajectory,
)
from cloak.ranker.environment import (
    RankerAction,
    RankerDecision,
    RankerDocument,
    assemble_action_vector,
)
from cloak.reward.utility_cache import (
    UtilityCache,
    UtilityRequest,
    UtilityResult,
    stable_hash,
)
from cloak.reward.utility_credit import decision_delta_utility, document_utility


@dataclass(frozen=True)
class CounterfactualRequest:
    doc_id: str
    rollout_index: int
    decision_id: str
    selected_action_id: str
    alternative_action_id: str
    direction: str
    priority_tier: int


@dataclass(frozen=True)
class _EligiblePair:
    document: RankerDocument
    trajectory: SampledTrajectory
    rollout_index: int
    decision: RankerDecision
    replayed_step: ReplayedStep
    adjacent: tuple[str, ...]
    endpoints: tuple[str, ...]
    priority: tuple[Any, ...]
    profile_id: str


def _decision(document: RankerDocument, decision_id: str) -> RankerDecision:
    matches = [
        decision for decision in document.policy_decisions
        if decision.decision_id == decision_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate policy decision: {decision_id}")
    return matches[0]


def _sampled_step(
    trajectory: SampledTrajectory, decision_id: str,
) -> SampledStep:
    matches = [step for step in trajectory.steps if step.decision_id == decision_id]
    if len(matches) != 1:
        raise ValueError(f"trajectory lacks unique decision step: {decision_id}")
    return matches[0]


def _actions(decision: RankerDecision) -> dict[str, RankerAction]:
    result = {action.action_id: action for action in decision.actions}
    if len(result) != len(decision.actions):
        raise ValueError(f"duplicate action ID for {decision.decision_id}")
    return result


def _structural_alternatives(
    decision: RankerDecision, selected_action_id: str,
) -> tuple[str, ...]:
    actions = _actions(decision)
    try:
        selected = actions[selected_action_id]
    except KeyError as error:
        raise ValueError(
            f"selected action is unknown for {decision.decision_id}: {selected_action_id}"
        ) from error
    levels = [action for action in decision.actions if action.mode == "level"]
    if any(action.authored_level_index is None for action in levels):
        raise ValueError(f"level action lacks authored index for {decision.decision_id}")
    levels.sort(key=lambda action: int(action.authored_level_index))
    indices = [int(action.authored_level_index) for action in levels]
    if len(indices) != len(set(indices)):
        raise ValueError(f"duplicate authored index for {decision.decision_id}")
    keep = [action for action in decision.actions if action.mode == "keep"]
    placeholder = [
        action for action in decision.actions if action.mode == "placeholder"
    ]
    if len(keep) != 1 or len(placeholder) != 1:
        raise ValueError(
            f"counterfactual decision requires KEEP and placeholder: {decision.decision_id}"
        )

    if selected.mode == "level":
        position = levels.index(selected)
        adjacent = []
        if position > 0:
            adjacent.append(levels[position - 1].action_id)
        if position + 1 < len(levels):
            adjacent.append(levels[position + 1].action_id)
        return (*adjacent, keep[0].action_id, placeholder[0].action_id)
    if not levels:
        return ()
    if selected.mode == "keep":
        return (levels[0].action_id,)
    if selected.mode == "placeholder":
        return (levels[-1].action_id,)
    raise ValueError(f"unsupported selected action mode: {selected.mode}")


def _eligible_with_reasons(
    document: RankerDocument,
    trajectory: SampledTrajectory,
    decision_id: str,
) -> tuple[tuple[str, ...], Counter[str]]:
    if trajectory.doc_id != document.doc_id:
        raise ValueError("trajectory document differs from counterfactual document")
    decision = _decision(document, decision_id)
    step = _sampled_step(trajectory, decision_id)
    if trajectory.action_vector.get(decision_id) != step.selected_action_id:
        raise ValueError(f"trajectory selected action mismatch for {decision_id}")
    selected_action_id = step.selected_action_id
    selected_vector = dict(trajectory.action_vector)
    selected_doc, _ = assemble_action_vector(document, selected_vector)
    eligible: list[str] = []
    reasons: Counter[str] = Counter()
    for alternative_action_id in _structural_alternatives(
        decision, selected_action_id,
    ):
        if alternative_action_id == selected_action_id:
            reasons["equal_action"] += 1
            continue
        if alternative_action_id not in step.legal_action_ids:
            reasons["illegal_alternative"] += 1
            continue
        alternative_vector = dict(selected_vector)
        alternative_vector[decision_id] = alternative_action_id
        try:
            alternative_doc, _ = assemble_action_vector(
                document, alternative_vector,
            )
        except ValueError as error:
            if "collision" in str(error):
                reasons["collision"] += 1
                continue
            raise
        if alternative_doc == selected_doc:
            reasons["duplicate_text"] += 1
            continue
        eligible.append(alternative_action_id)
    if not eligible:
        reasons["no_eligible_alternative"] += 1
    return tuple(eligible), reasons


def eligible_alternatives(
    document: RankerDocument,
    trajectory: SampledTrajectory,
    decision_id: str,
) -> tuple[str, ...]:
    """Return legal adjacent and endpoint alternatives without suffix repair."""

    return _eligible_with_reasons(document, trajectory, decision_id)[0]


def pair_history_key(
    doc_id: str,
    rollout_index: int,
    decision_id: str,
    selected_action_id: str,
    alternative_action_id: str,
) -> tuple[str, int, str, str, str]:
    left, right = sorted((selected_action_id, alternative_action_id))
    return (doc_id, int(rollout_index), decision_id, left, right)


def _direction(
    decision: RankerDecision,
    selected_action_id: str,
    alternative_action_id: str,
) -> str:
    actions = _actions(decision)
    selected = actions[selected_action_id]
    alternative = actions[alternative_action_id]
    if selected.mode == "keep" or alternative.mode == "keep":
        return "keep"
    if selected.mode == "placeholder" or alternative.mode == "placeholder":
        return "placeholder"
    assert selected.authored_level_index is not None
    assert alternative.authored_level_index is not None
    return (
        "finer"
        if alternative.authored_level_index < selected.authored_level_index
        else "coarser"
    )


def _dependency_rows(
    utility_artifact: Mapping, document: RankerDocument, decision_id: str,
) -> tuple[Mapping, ...]:
    try:
        assertion_ids = utility_artifact["documents"][document.doc_id]["assertion_ids"]
        assertions = utility_artifact["assertions"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"utility artifact lacks counterfactual document {document.doc_id}"
        ) from error
    return tuple(
        assertions[assertion_id]
        for assertion_id in assertion_ids
        if assertions[assertion_id].get("credit_routing") == "linked"
        and decision_id
        in assertions[assertion_id].get("policy_dependency_decision_ids", ())
    )


def _history_for_pair(
    pair: _EligiblePair,
    alternative_action_id: str,
    pair_history: Mapping[tuple[str, int, str, str, str], int],
) -> int | None:
    return pair_history.get(pair_history_key(
        pair.document.doc_id,
        pair.rollout_index,
        pair.decision.decision_id,
        pair.trajectory.action_vector[pair.decision.decision_id],
        alternative_action_id,
    ))


def _age_summary(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def schedule_counterfactuals(
    documents: Mapping[str, RankerDocument],
    trajectories: Sequence[SampledTrajectory],
    replayed: Sequence[ReplayedTrajectory],
    *,
    utility_artifact: Mapping,
    credit_routes: Mapping[str, Mapping[str, str]],
    pair_history: Mapping[tuple[str, int, str, str, str], int],
    budget: int,
    endpoint_budget: int,
    seed: int,
    current_round: int,
    dedup_identical_vectors: bool = False,
) -> tuple[tuple[CounterfactualRequest, ...], dict[str, Any]]:
    """Allocate the exact 20/80 call budget without exposing reward values.

    With dedup_identical_vectors, rollouts sharing a complete action vector
    contribute pairs only once per decision — a measured intervention is exact
    for every duplicate (small-document credit-support fork, decision log
    2026-07-29), so spending budget on duplicates is pure waste; the executor's
    broadcast substitutes the measured loss into the duplicate pairs.
    """

    if (
        not isinstance(budget, int)
        or isinstance(budget, bool)
        or budget <= 0
        or budget % 5
    ):
        raise ValueError("counterfactual budget must be positive and divisible by five")
    if not isinstance(endpoint_budget, int) or isinstance(endpoint_budget, bool):
        raise ValueError("endpoint_budget must be an integer")
    if endpoint_budget < 0 or endpoint_budget * 2 >= budget:
        raise ValueError("endpoint_budget must be a nonnegative minority of budget")
    if current_round < 0:
        raise ValueError("current_round must be nonnegative")
    trajectories = tuple(trajectories)
    replayed = tuple(replayed)
    if len(trajectories) != len(replayed):
        raise ValueError("sampled and replayed rollout counts differ")

    skip_reasons: Counter[str] = Counter()
    pairs: list[_EligiblePair] = []
    never_measured_ids: set[tuple[str, int, str]] = set()
    seen_vector_decisions: set[tuple[tuple[tuple[str, str], ...], str]] = set()
    deduplicated_pairs = 0
    for rollout_index, (trajectory, replay) in enumerate(
        zip(trajectories, replayed, strict=True)
    ):
        try:
            document = documents[trajectory.doc_id]
        except KeyError as error:
            raise ValueError(
                f"counterfactual document not supplied: {trajectory.doc_id}"
            ) from error
        if replay.doc_id != trajectory.doc_id:
            raise ValueError("sampled and replayed document IDs differ")
        replay_steps = {step.decision_id: step for step in replay.steps}
        if len(replay_steps) != len(replay.steps):
            raise ValueError("replayed trajectory repeats a decision")
        routes = credit_routes.get(document.doc_id)
        if routes is None:
            raise ValueError(f"missing credit routes for {document.doc_id}")
        vector_items = tuple(sorted(trajectory.action_vector.items()))
        for decision in document.policy_decisions:
            try:
                replayed_step = replay_steps[decision.decision_id]
                route = routes[decision.decision_id]
            except KeyError as error:
                raise ValueError(
                    f"missing replay/credit route for {decision.decision_id}"
                ) from error
            if dedup_identical_vectors:
                vector_decision = (vector_items, decision.decision_id)
                if vector_decision in seen_vector_decisions:
                    deduplicated_pairs += 1
                    skip_reasons["duplicate_vector_pair"] += 1
                    continue
                seen_vector_decisions.add(vector_decision)
            eligible, reasons = _eligible_with_reasons(
                document, trajectory, decision.decision_id,
            )
            skip_reasons.update(reasons)
            actions = _actions(decision)
            selected_action_id = trajectory.action_vector[decision.decision_id]
            adjacent = tuple(
                action_id for action_id in eligible
                if actions[action_id].mode == "level"
                and actions[selected_action_id].mode == "level"
            )
            endpoints = tuple(
                action_id for action_id in eligible if action_id not in adjacent
            )
            if not adjacent and not (endpoints and endpoint_budget > 0):
                skip_reasons["no_adjacent_alternative"] += 1
                continue
            linked_rows = _dependency_rows(
                utility_artifact, document, decision.decision_id,
            )
            no_linked = route == "document"
            if route not in {"document", "linked"}:
                raise ValueError(f"invalid credit route for {decision.decision_id}")
            if no_linked != (not linked_rows):
                raise ValueError(
                    f"credit route and utility dependencies differ for {decision.decision_id}"
                )
            only_hyperedges = bool(linked_rows) and all(
                len(set(row.get("policy_dependency_decision_ids", ()))) > 1
                for row in linked_rows
            )
            histories = [
                pair_history.get(pair_history_key(
                    document.doc_id,
                    rollout_index,
                    decision.decision_id,
                    selected_action_id,
                    alternative,
                ))
                for alternative in (*adjacent, *endpoints)
            ]
            ages = [current_round - value for value in histories if value is not None]
            if any(age < 0 for age in ages):
                raise ValueError("pair history lies after current_round")
            priority = (
                -int(no_linked),
                -int(only_hyperedges),
                -float(replayed_step.entropy.detach()),
            )
            pair = _EligiblePair(
                document=document,
                trajectory=trajectory,
                rollout_index=rollout_index,
                decision=decision,
                replayed_step=replayed_step,
                adjacent=adjacent,
                endpoints=endpoints,
                priority=priority,
                profile_id=str(trajectory.lambda_profile),
            )
            pairs.append(pair)
            if all(
                _history_for_pair(pair, action_id, pair_history) is None
                for action_id in (*adjacent, *endpoints)
            ):
                never_measured_ids.add((
                    document.doc_id, rollout_index, decision.decision_id,
                ))

    # Budget is a cap, not an exact demand (production defect 2026-07-29: a
    # lambda-zero group whose rollouts converge to all-KEEP has zero
    # adjacent-capable pairs, and small groups can have fewer eligible pairs
    # than budget). Scarce groups schedule what capacity allows; the slot loop
    # below degrades per slot instead of raising.
    rng = random.Random(seed)
    uniform_budget = budget // 5
    endpoint_slots = set(rng.sample(range(budget), endpoint_budget))
    direction_counts: Counter[str] = Counter()
    profile_direction_counts: Counter[tuple[str, str]] = Counter()
    selected_rows: list[tuple[_EligiblePair, str, str, tuple[Any, ...] | None]] = []
    selected_ages: list[int] = []
    remaining = list(pairs)

    def candidates_for(pair: _EligiblePair, *, endpoint: bool) -> tuple[str, ...]:
        return pair.endpoints if endpoint else pair.adjacent

    def preserves_remaining_slots(pair: _EligiblePair, slot: int) -> bool:
        after = [candidate for candidate in remaining if candidate is not pair]
        future_slots = range(slot + 1, budget)
        adjacent_needed = sum(index not in endpoint_slots for index in future_slots)
        endpoints_needed = sum(index in endpoint_slots for index in future_slots)
        return (
            len(after) >= adjacent_needed + endpoints_needed
            and sum(bool(candidate.adjacent) for candidate in after) >= adjacent_needed
            and sum(bool(candidate.endpoints) for candidate in after) >= endpoints_needed
        )

    def balanced_options(
        pair: _EligiblePair,
        candidates: tuple[str, ...],
        *,
        endpoint: bool,
    ) -> tuple[tuple[str, ...], tuple[int, int]]:
        by_direction: dict[str, list[str]] = {}
        selected_action_id = pair.trajectory.action_vector[pair.decision.decision_id]
        for action_id in candidates:
            direction = _direction(pair.decision, selected_action_id, action_id)
            by_direction.setdefault(direction, []).append(action_id)
        minimum = min(
            profile_direction_counts[(pair.profile_id, direction)]
            for direction in by_direction
        )
        directions = sorted(
            direction for direction in by_direction
            if profile_direction_counts[(pair.profile_id, direction)] == minimum
        )
        balanced = tuple(
            action_id
            for direction in directions
            for action_id in sorted(by_direction[direction])
        )
        history = {
            action_id: _history_for_pair(pair, action_id, pair_history)
            for action_id in balanced
        }
        if not endpoint:
            unseen = [action_id for action_id in balanced if history[action_id] is None]
        else:
            unseen = []
        if unseen:
            return tuple(unseen), (-1, 1)
        measured = [
            (current_round - int(value), action_id)
            for action_id, value in history.items() if value is not None
        ]
        oldest_age = max((age for age, _ in measured), default=-1)
        oldest = tuple(
            action_id for age, action_id in measured if age == oldest_age
        )
        if oldest:
            return oldest, (0, oldest_age)
        return balanced, (0, -1)

    retyped_slots = 0
    for slot in range(budget):
        endpoint = slot in endpoint_slots
        pool = [
            pair for pair in remaining
            if candidates_for(pair, endpoint=endpoint)
            and preserves_remaining_slots(pair, slot)
        ]
        if not pool:
            # Degrade instead of raising: first drop the lookahead (it presumes
            # every future slot is fillable, which scarce groups violate), then
            # retype the slot to the other probe kind, then stop scheduling.
            # The strict path above is untouched, so full-capacity groups
            # behave exactly as before.
            pool = [
                pair for pair in remaining
                if candidates_for(pair, endpoint=endpoint)
            ]
            if not pool:
                endpoint = not endpoint
                pool = [
                    pair for pair in remaining
                    if candidates_for(pair, endpoint=endpoint)
                ]
                if pool:
                    retyped_slots += 1
            if not pool:
                break
        if slot < uniform_budget:
            pair = rng.choice(pool)
            options, _history_priority = balanced_options(
                pair, candidates_for(pair, endpoint=endpoint), endpoint=endpoint,
            )
            effective_priority = None
        else:
            ranked: list[tuple[tuple[Any, ...], _EligiblePair, tuple[str, ...]]] = []
            for candidate_pair in pool:
                options, history_priority = balanced_options(
                    candidate_pair,
                    candidates_for(candidate_pair, endpoint=endpoint),
                    endpoint=endpoint,
                )
                unseen_rank, age = history_priority
                effective = (*candidate_pair.priority, unseen_rank, -age)
                ranked.append((effective, candidate_pair, options))
            best_priority = min(priority for priority, _, _ in ranked)
            tied = [row for row in ranked if row[0] == best_priority]
            effective_priority, pair, options = rng.choice(tied)
        alternative_action_id = rng.choice(options)
        selected_action_id = pair.trajectory.action_vector[pair.decision.decision_id]
        direction = _direction(pair.decision, selected_action_id, alternative_action_id)
        selected_rows.append((pair, alternative_action_id, direction, effective_priority))
        remaining.remove(pair)
        direction_counts[direction] += 1
        profile_direction_counts[(pair.profile_id, direction)] += 1
        measured_round = _history_for_pair(pair, alternative_action_id, pair_history)
        if measured_round is not None:
            age = current_round - measured_round
            if age < 0:
                raise ValueError("pair history lies after current_round")
            selected_ages.append(age)

    tier_by_priority = {
        priority: tier
        for tier, priority in enumerate(sorted({
            priority for _, _, _, priority in selected_rows if priority is not None
        }))
    }
    requests: list[CounterfactualRequest] = []
    for slot, (pair, alternative_action_id, direction, effective_priority) in enumerate(
        selected_rows
    ):
        priority_tier = -1 if slot < uniform_budget else tier_by_priority[effective_priority]
        selected_action_id = pair.trajectory.action_vector[pair.decision.decision_id]
        request = CounterfactualRequest(
            doc_id=pair.document.doc_id,
            rollout_index=pair.rollout_index,
            decision_id=pair.decision.decision_id,
            selected_action_id=selected_action_id,
            alternative_action_id=alternative_action_id,
            direction=direction,
            priority_tier=priority_tier,
        )
        requests.append(request)

    skip_reasons["budget_unselected"] += len(pairs) - len(selected_rows)
    directions_by_profile: dict[str, dict[str, int]] = {}
    for (profile_id, direction), count in sorted(profile_direction_counts.items()):
        directions_by_profile.setdefault(profile_id, {})[direction] = count
    diagnostics: dict[str, Any] = {
        "budget": budget,
        "scheduled": len(selected_rows),
        "slot_shortfall": budget - len(selected_rows),
        "retyped_slots": retyped_slots,
        "deduplicated_pairs": deduplicated_pairs,
        "uniform_allocation": uniform_budget,
        "priority_allocation": budget - uniform_budget,
        "endpoint_fraction": endpoint_budget / budget,
        "direction_balance": dict(sorted(direction_counts.items())),
        "direction_balance_by_profile": directions_by_profile,
        "cache_hits": 0,
        "never_measured_eligible_decisions": len(never_measured_ids),
        "never_measured_eligible_decision_ids": [
            f"{doc_id}:{rollout_index}:{decision_id}"
            for doc_id, rollout_index, decision_id in sorted(never_measured_ids)
        ],
        "pair_age": _age_summary(selected_ages),
        "delta_u": {"count": 0},
        "skip_reasons": dict(sorted(skip_reasons.items())),
    }
    return tuple(requests), diagnostics


def _default_score_batch(requests, **kwargs):
    from cloak.reward.roundtrip import score_roundtrip_batch

    return score_roundtrip_batch(requests, **kwargs)


def execute_counterfactuals(
    requests: Sequence[CounterfactualRequest],
    documents: Mapping[str, RankerDocument],
    trajectories: Sequence[SampledTrajectory],
    replayed: Sequence[ReplayedTrajectory],
    *,
    utility_artifact: Mapping,
    environment_hash: str,
    cache: UtilityCache,
    remote_workers: int,
    reader_workers: int,
    scheduler_diagnostics: Mapping[str, Any],
    score_batch: Callable[..., Sequence[UtilityResult]] | None = None,
    broadcast: bool = False,
) -> tuple[dict[tuple[int, str], torch.Tensor], dict[str, Any]]:
    """Measure complete one-decision vectors and build graph-bearing pair losses.

    With broadcast, a measured delta-U also substitutes a pair loss into every
    other rollout whose complete action vector is identical to the probed one —
    exact by construction (identical selected vector, counterfactual vector, and
    replay state), at zero extra scorer calls (small-document credit-support
    fork, decision log 2026-07-29).
    """

    requests = tuple(requests)
    trajectories = tuple(trajectories)
    replayed = tuple(replayed)
    if len(trajectories) != len(replayed):
        raise ValueError("sampled and replayed rollout counts differ")
    request_pairs = [(request.rollout_index, request.decision_id) for request in requests]
    if len(request_pairs) != len(set(request_pairs)):
        raise ValueError("counterfactual requests repeat a rollout-decision pair")

    utility_requests: list[UtilityRequest] = []
    validated: list[
        tuple[CounterfactualRequest, SampledTrajectory, ReplayedStep, dict[str, str]]
    ] = []
    for request in requests:
        if request.rollout_index < 0 or request.rollout_index >= len(trajectories):
            raise ValueError("counterfactual rollout_index is out of range")
        trajectory = trajectories[request.rollout_index]
        replay = replayed[request.rollout_index]
        try:
            document = documents[request.doc_id]
        except KeyError as error:
            raise ValueError(f"counterfactual document not supplied: {request.doc_id}") from error
        if trajectory.doc_id != request.doc_id or replay.doc_id != request.doc_id:
            raise ValueError("counterfactual request document differs from rollout")
        if trajectory.action_vector.get(request.decision_id) != request.selected_action_id:
            raise ValueError("counterfactual request selected action differs from rollout")
        if request.alternative_action_id not in eligible_alternatives(
            document, trajectory, request.decision_id,
        ):
            raise ValueError("counterfactual request alternative is ineligible")
        decision = _decision(document, request.decision_id)
        if request.direction != _direction(
            decision,
            request.selected_action_id,
            request.alternative_action_id,
        ):
            raise ValueError("counterfactual request direction disagrees with actions")
        replay_steps = [
            step for step in replay.steps if step.decision_id == request.decision_id
        ]
        if len(replay_steps) != 1:
            raise ValueError("counterfactual replay lacks a unique decision step")
        replayed_step = replay_steps[0]
        sampled_step = _sampled_step(trajectory, request.decision_id)
        if (
            replayed_step.legal_action_ids != sampled_step.legal_action_ids
            or replayed_step.log_probs.ndim != 1
            or len(replayed_step.log_probs) != len(sampled_step.legal_action_ids)
        ):
            raise ValueError("counterfactual replay must preserve the full original legal menu")
        if (
            replayed_step.selected_action_id != request.selected_action_id
            or request.alternative_action_id not in replayed_step.legal_action_ids
        ):
            raise ValueError("counterfactual request differs from full replay menu")
        alternative_vector = dict(trajectory.action_vector)
        alternative_vector[request.decision_id] = request.alternative_action_id
        difference = [
            decision_id for decision_id in trajectory.action_vector
            if trajectory.action_vector[decision_id] != alternative_vector[decision_id]
        ]
        if difference != [request.decision_id]:
            raise ValueError("counterfactual intervention must change exactly one decision")
        validated.append((request, trajectory, replayed_step, alternative_vector))
        for vector in (trajectory.action_vector, alternative_vector):
            utility_requests.append(UtilityRequest(
                document=document,
                action_vector=vector,
                utility_artifact=utility_artifact,
                environment_hash=environment_hash,
            ))

    scorer = score_batch or _default_score_batch
    hits_before = cache.hits
    results = tuple(scorer(
        utility_requests,
        cache=cache,
        remote_workers=remote_workers,
        reader_workers=reader_workers,
        reader_refresh=False,
    ))
    if len(results) != len(utility_requests):
        raise ValueError("utility scorer returned wrong counterfactual result count")

    losses: dict[tuple[int, str], torch.Tensor] = {}
    deltas: list[float] = []
    evidence_rows: list[dict[str, Any]] = []
    broadcast_pairs = 0
    for index, (request, trajectory, replayed_step, alternative_vector) in enumerate(validated):
        selected_result = results[2 * index]
        alternative_result = results[2 * index + 1]
        if (
            selected_result.doc_id != request.doc_id
            or alternative_result.doc_id != request.doc_id
        ):
            raise ValueError("counterfactual utility result document mismatch")
        if dict(selected_result.action_vector) != dict(trajectory.action_vector):
            raise ValueError("selected utility result action vector mismatch")
        if dict(alternative_result.action_vector) != alternative_vector:
            raise ValueError("alternative utility result action vector mismatch")
        selected_utility = document_utility(
            selected_result.component_scores, utility_artifact, request.doc_id,
        )
        alternative_utility = document_utility(
            alternative_result.component_scores, utility_artifact, request.doc_id,
        )
        delta_u = selected_utility - alternative_utility
        selected_index = replayed_step.legal_action_ids.index(request.selected_action_id)
        alternative_index = replayed_step.legal_action_ids.index(
            request.alternative_action_id
        )
        pair_log_probs = torch.stack((
            replayed_step.log_probs[selected_index],
            replayed_step.log_probs[alternative_index],
        ))
        q_pair = torch.softmax(pair_log_probs, dim=0)[0]
        pair_loss = -float(delta_u) * (q_pair - 0.5)
        if not bool(torch.isfinite(pair_loss)):
            raise ValueError("non-finite counterfactual pair loss")
        losses[(request.rollout_index, request.decision_id)] = pair_loss
        deltas.append(float(delta_u))
        surrounding = {
            key: value for key, value in trajectory.action_vector.items()
            if key != request.decision_id
        }
        # Tie qualification compares against a resolution threshold, so it needs
        # the decision's OWN assertions in the decision's own units, not the whole
        # document's (measurement revision, 2026-08-03). The pair loss above keeps
        # the document-level delta deliberately: it is the objective's unit.
        attributed_document, attributed_linked = decision_delta_utility(
            selected_result.component_scores,
            alternative_result.component_scores,
            utility_artifact,
            request.doc_id,
            request.decision_id,
        )
        evidence_rows.append({
            "doc_id": request.doc_id,
            "decision_id": request.decision_id,
            "selected_action_id": request.selected_action_id,
            "alternative_action_id": request.alternative_action_id,
            "delta_u": float(delta_u),
            "delta_u_attributed": float(attributed_document),
            "delta_u_linked": (
                None if attributed_linked is None else float(attributed_linked)
            ),
            "context_hash": stable_hash(sorted(surrounding.items())),
        })
        if broadcast:
            selected_items = tuple(sorted(trajectory.action_vector.items()))
            for other_index, other_replay in enumerate(replayed):
                if other_index == request.rollout_index:
                    continue
                other_items = tuple(sorted(
                    trajectories[other_index].action_vector.items()
                ))
                if other_items != selected_items:
                    continue
                other_pair = (other_index, request.decision_id)
                if other_pair in losses:
                    continue
                other_steps = [
                    step for step in other_replay.steps
                    if step.decision_id == request.decision_id
                ]
                if (
                    len(other_steps) != 1
                    or other_steps[0].legal_action_ids
                    != replayed_step.legal_action_ids
                ):
                    continue
                other_step = other_steps[0]
                other_q = torch.softmax(torch.stack((
                    other_step.log_probs[selected_index],
                    other_step.log_probs[alternative_index],
                )), dim=0)[0]
                other_loss = -float(delta_u) * (other_q - 0.5)
                if not bool(torch.isfinite(other_loss)):
                    raise ValueError("non-finite broadcast counterfactual pair loss")
                losses[other_pair] = other_loss
                broadcast_pairs += 1

    diagnostics = copy.deepcopy(dict(scheduler_diagnostics))
    # Value-bearing utility evidence for the tie-ownership ledger (round-3
    # adjudication 2026-07-31): per-probe delta-U keyed by the surrounding
    # action-vector context. Broadcast duplicates share the context and are
    # deliberately not repeated (qualification counts DISTINCT contexts).
    diagnostics["evidence_rows"] = evidence_rows
    diagnostics["broadcast_pairs"] = broadcast_pairs
    diagnostics["cache_hits"] = int(
        cache.last_batch_metrics.get("cache_hits", cache.hits - hits_before)
    )
    absolute = [abs(value) for value in deltas]
    diagnostics["delta_u"] = {
        "count": len(deltas),
        "zero": sum(value == 0.0 for value in deltas),
        "positive": sum(value > 0.0 for value in deltas),
        "negative": sum(value < 0.0 for value in deltas),
        "mean_abs": sum(absolute) / len(absolute) if absolute else 0.0,
        "max_abs": max(absolute, default=0.0),
    }
    return losses, diagnostics
