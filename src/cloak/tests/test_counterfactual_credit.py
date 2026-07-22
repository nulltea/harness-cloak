import random
import sys
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from types import MappingProxyType

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import train_ranker as tr  # noqa: E402
from test_train_roundtrip_mode import _doc, fake_roundtrip  # noqa: E402


def test_counterfactual_credits_the_flipped_span(monkeypatch):
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip)
    torch.manual_seed(0)
    doc = _doc()
    policy = tr.RankerPolicy()
    # a rollout that KEPT the level fill (reward 1.0); counterfactual placeholder -> 0.0
    choice, logps, _, doc_p, _, _ = tr.sample_rollout(doc, doc["spans"], doc["feats"], policy,
                                                      greedy=True)
    if choice["metformin"]["mode"] != "level":   # force the level action for determinism
        choice = {"metformin": doc["spans"][0]["actions"][0]}
        lp = policy.log_probs(doc["feats"][0], doc["spans"][0]["legal"])
        logps = [lp[doc["spans"][0]["legal"].index(0)]]
    term, n_cf = tr.counterfactual_terms(doc, policy, choice, logps, base_r=1.0,
                                         frac=1.0, rng=random.Random(0), rt_workers=1)
    assert n_cf == 1
    # adv_span = base_r - r_cf = 1.0 - 0.0 = 1.0; term = -(adv * logp) > 0
    assert term.item() > 0
    term.backward()   # gradient flows to the policy
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())


def test_counterfactual_skips_placeholder_spans(monkeypatch):
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip)
    doc = _doc()
    policy = tr.RankerPolicy()
    ph_action = doc["spans"][0]["actions"][1]
    choice = {"metformin": ph_action}
    lp = policy.log_probs(doc["feats"][0], doc["spans"][0]["legal"])
    logps = [lp[doc["spans"][0]["legal"].index(1)]]
    term, n_cf = tr.counterfactual_terms(doc, policy, choice, logps, base_r=0.0,
                                         frac=1.0, rng=random.Random(0), rt_workers=1)
    assert n_cf == 0 and term == 0.0   # placeholder IS the counterfactual; nothing to flip


from cloak.train.interactive_ranker import (  # noqa: E402
    ReplayedStep,
    ReplayedTrajectory,
    SampledStep,
    SampledTrajectory,
    assemble_action_vector,
)
from cloak.train.ranker_environment import (  # noqa: E402
    RankerAction,
    RankerDecision,
    RankerDocument,
)
from cloak.train.utility_cache import UtilityCache, make_result, stable_hash  # noqa: E402


def _action(decision_id, name, mode, fill=None, index=None):
    return RankerAction(
        action_id=f"{decision_id}-{name}",
        mode=mode,
        fill=fill,
        authored_level_index=index,
        runtime_type="TYPE",
    )


def _decision(decision_id, occurrence_ids, *, collision_fill=None):
    return RankerDecision(
        decision_id=decision_id,
        profile_id=f"profile-{decision_id}",
        runtime_type="TYPE",
        canonical_key=decision_id,
        occurrence_ids=occurrence_ids,
        actions=(
            _action(decision_id, "fine", "level", f"fine {decision_id}", 0),
            _action(decision_id, "middle", "level", f"middle {decision_id}", 1),
            _action(
                decision_id,
                "coarse",
                "level",
                collision_fill or f"coarse {decision_id}",
                2,
            ),
            _action(decision_id, "keep", "keep"),
            _action(decision_id, "placeholder", "placeholder"),
        ),
    )


def _document(*, collision=False):
    text = "Alpha met Beta and alpha near Gamma."
    occurrences = (
        MappingProxyType({
            "occurrence_id": "o-d1-1", "decision_id": "d1",
            "start": 0, "end": 5, "surface": "Alpha", "controlled": True,
        }),
        MappingProxyType({
            "occurrence_id": "o-d2", "decision_id": "d2",
            "start": 10, "end": 14, "surface": "Beta", "controlled": True,
        }),
        MappingProxyType({
            "occurrence_id": "o-d1-2", "decision_id": "d1",
            "start": 19, "end": 24, "surface": "alpha", "controlled": True,
        }),
        MappingProxyType({
            "occurrence_id": "o-d3", "decision_id": "d3",
            "start": 30, "end": 35, "surface": "Gamma", "controlled": True,
        }),
    )
    d2_selected_fill = "middle d2"
    return RankerDocument(
        doc_id="fixture/counterfactual",
        corpus="aci",
        text=text,
        occurrences=occurrences,
        policy_decisions=(
            _decision(
                "d1", ("o-d1-1", "o-d1-2"),
                collision_fill=d2_selected_fill if collision else None,
            ),
            _decision("d2", ("o-d2",)),
            _decision("d3", ("o-d3",)),
        ),
        fixed_decisions=(),
    )


def _trajectory(document=None, *, legal_override=None):
    document = document or _document()
    action_vector = {
        decision.decision_id: f"{decision.decision_id}-middle"
        for decision in document.policy_decisions
    }
    claimed = ()
    steps = []
    for decision in document.policy_decisions:
        menu = tuple(action.action_id for action in decision.actions)
        if legal_override and decision.decision_id in legal_override:
            menu = tuple(legal_override[decision.decision_id])
        steps.append(SampledStep(
            decision_id=decision.decision_id,
            legal_action_ids=menu,
            selected_action_id=action_vector[decision.decision_id],
            claimed_fills_before=claimed,
        ))
        claimed = (*claimed, f"middle {decision.decision_id}")
    return SampledTrajectory(
        doc_id=document.doc_id,
        lambda_profile="lambda-zero",
        steps=tuple(steps),
        action_vector=MappingProxyType(action_vector),
    )


def _replayed(document=None, *, entropies=None):
    document = document or _document()
    entropies = entropies or {"d1": 0.1, "d2": 0.8, "d3": 0.4}
    steps = []
    logits_by_decision = {}
    for decision in document.policy_decisions:
        menu = tuple(action.action_id for action in decision.actions)
        logits = torch.tensor([0.0, 0.4, -0.2, -0.5, -0.8], requires_grad=True)
        log_probs = torch.log_softmax(logits, dim=0)
        logits_by_decision[decision.decision_id] = logits
        steps.append(ReplayedStep(
            decision_id=decision.decision_id,
            selected_action_id=f"{decision.decision_id}-middle",
            legal_action_ids=menu,
            log_prob=log_probs[1],
            log_probs=log_probs,
            count_log_probs=log_probs,
            utility_logits=log_probs,
            predicted_privacy=torch.zeros_like(log_probs),
            entropy=torch.tensor(float(entropies[decision.decision_id])),
        ))
    return ReplayedTrajectory(
        doc_id=document.doc_id,
        lambda_profile="lambda-zero",
        steps=tuple(steps),
    ), logits_by_decision


def _utility_artifact(document=None):
    document = document or _document()
    artifact = {
        "artifact_version": "utility-assertions-v2",
        "environment_hash": "env-hash",
        "reader_pin": {
            "model": "fixture-reader", "endpoint": "fixture-endpoint",
            "prompt_version": "fixture-v1", "response_schema": {"type": "string"},
            "revision": "fixture-revision",
        },
        "documents": {
            document.doc_id: {
                "assertion_ids": ["a-residual", "a-hyperedge"],
                "policy_decision_ids": ["d1", "d2", "d3"],
                "utility_weight_denominator": 1.0,
            },
        },
        "assertions": {
            "a-residual": {
                "assertion_id": "a-residual", "doc_id": document.doc_id,
                "family": "delivered", "status": "accepted", "weight": 0.5,
                "credit_routing": "residual", "policy_dependency_decision_ids": [],
            },
            "a-hyperedge": {
                "assertion_id": "a-hyperedge", "doc_id": document.doc_id,
                "family": "context", "status": "accepted", "weight": 0.5,
                "credit_routing": "linked",
                "policy_dependency_decision_ids": ["d2", "d3"],
            },
        },
    }
    artifact["artifact_hash"] = stable_hash(artifact)
    return artifact


def test_counterfactual_request_contract_is_exact_and_frozen():
    from cloak.train.counterfactuals import CounterfactualRequest

    assert [field.name for field in fields(CounterfactualRequest)] == [
        "doc_id", "rollout_index", "decision_id", "selected_action_id",
        "alternative_action_id", "direction", "priority_tier",
    ]
    request = CounterfactualRequest("d", 0, "j", "a", "b", "finer", 1)
    with pytest.raises(FrozenInstanceError):
        request.direction = "coarser"


def test_eligible_alternatives_are_adjacent_levels_plus_endpoints():
    from cloak.train.counterfactuals import eligible_alternatives

    assert eligible_alternatives(_document(), _trajectory(), "d1") == (
        "d1-fine", "d1-coarse", "d1-keep", "d1-placeholder",
    )


def test_eligibility_skips_illegal_duplicate_text_and_equal_actions():
    from cloak.train.counterfactuals import eligible_alternatives

    document = _document()
    d1 = document.policy_decisions[0]
    duplicate = replace(d1.actions[2], fill=d1.actions[1].fill)
    document = replace(
        document,
        policy_decisions=(replace(d1, actions=(*d1.actions[:2], duplicate, *d1.actions[3:])),
                          *document.policy_decisions[1:]),
    )
    legal = tuple(
        action.action_id for action in document.policy_decisions[0].actions
        if action.action_id != "d1-fine"
    )
    trajectory = _trajectory(document, legal_override={"d1": legal})

    assert eligible_alternatives(document, trajectory, "d1") == (
        "d1-keep", "d1-placeholder",
    )


def test_eligibility_rejects_alternative_colliding_with_later_selected_fill():
    from cloak.train.counterfactuals import eligible_alternatives

    document = _document(collision=True)
    assert eligible_alternatives(document, _trajectory(document), "d1") == (
        "d1-fine", "d1-keep", "d1-placeholder",
    )


def test_scheduler_enforces_budget_split_priority_balance_and_endpoint_reserve():
    from cloak.train.counterfactuals import schedule_counterfactuals

    document = _document()
    trajectories = (_trajectory(document), _trajectory(document))
    replayed = (_replayed(document)[0], _replayed(document)[0])
    requests, diagnostics = schedule_counterfactuals(
        {document.doc_id: document},
        trajectories,
        replayed,
        utility_artifact=_utility_artifact(document),
        credit_routes={document.doc_id: {
            "d1": "document", "d2": "linked", "d3": "linked",
        }},
        pair_history={},
        budget=5,
        endpoint_budget=1,
        seed=17,
        current_round=10,
    )

    assert len(requests) == 5
    assert sum(request.priority_tier == -1 for request in requests) == 1
    assert {(request.rollout_index, request.decision_id) for request in requests} >= {
        (0, "d1"), (1, "d1"),
    }
    assert diagnostics["budget"] == 5
    assert diagnostics["uniform_allocation"] == 1
    assert diagnostics["priority_allocation"] == 4
    assert diagnostics["endpoint_fraction"] == pytest.approx(0.2)
    assert sum(diagnostics["direction_balance"].values()) == 5
    assert abs(
        diagnostics["direction_balance"].get("finer", 0)
        - diagnostics["direction_balance"].get("coarser", 0)
    ) <= 1
    assert diagnostics["direction_balance_by_profile"] == {
        "lambda-zero": diagnostics["direction_balance"],
    }
    assert diagnostics["never_measured_eligible_decisions"] == 6
    assert diagnostics["pair_age"]["count"] == 0
    assert diagnostics["cache_hits"] == 0
    assert diagnostics["delta_u"]["count"] == 0
    assert diagnostics["skip_reasons"]["budget_unselected"] == 1


def test_scheduler_balances_keep_and_placeholder_inside_endpoint_reserve():
    from cloak.train.counterfactuals import schedule_counterfactuals

    document = _document()
    trajectories = tuple(_trajectory(document) for _ in range(4))
    replayed = tuple(_replayed(document)[0] for _ in range(4))
    requests, diagnostics = schedule_counterfactuals(
        {document.doc_id: document},
        trajectories,
        replayed,
        utility_artifact=_utility_artifact(document),
        credit_routes={document.doc_id: {
            "d1": "document", "d2": "linked", "d3": "linked",
        }},
        pair_history={},
        budget=10,
        endpoint_budget=2,
        seed=9,
        current_round=0,
    )

    endpoint_requests = [
        request for request in requests if request.direction in {"keep", "placeholder"}
    ]
    assert len(endpoint_requests) == 2
    assert {request.direction for request in endpoint_requests} == {"keep", "placeholder"}
    assert diagnostics["endpoint_fraction"] == pytest.approx(0.2)


def test_selected_endpoint_pair_is_eligible_only_inside_endpoint_reserve():
    from cloak.train.counterfactuals import schedule_counterfactuals

    document = _document()
    base = _trajectory(document)
    action_vector = dict(base.action_vector)
    action_vector["d1"] = "d1-keep"
    trajectory = replace(
        base,
        steps=(replace(base.steps[0], selected_action_id="d1-keep"), *base.steps[1:]),
        action_vector=MappingProxyType(action_vector),
    )
    base_replay, _ = _replayed(document)
    d1_log_probs = base_replay.steps[0].log_probs
    replayed = replace(
        base_replay,
        steps=(
            replace(
                base_replay.steps[0],
                selected_action_id="d1-keep",
                log_prob=d1_log_probs[3],
            ),
            *base_replay.steps[1:],
        ),
    )

    requests, _ = schedule_counterfactuals(
        {document.doc_id: document},
        (trajectory, trajectory),
        (replayed, replayed),
        utility_artifact=_utility_artifact(document),
        credit_routes={document.doc_id: {
            "d1": "document", "d2": "linked", "d3": "linked",
        }},
        pair_history={},
        budget=5,
        endpoint_budget=1,
        seed=5,
        current_round=0,
    )

    d1_requests = [request for request in requests if request.decision_id == "d1"]
    assert len(d1_requests) == 1
    assert d1_requests[0].selected_action_id == "d1-keep"
    assert d1_requests[0].alternative_action_id == "d1-fine"
    assert d1_requests[0].direction == "keep"


@pytest.mark.parametrize("budget", [0, 3, 6, 5.0])
def test_scheduler_rejects_nonpositive_or_nondivisible_budget(budget):
    from cloak.train.counterfactuals import schedule_counterfactuals

    document = _document()
    with pytest.raises(ValueError, match="positive and divisible by five"):
        schedule_counterfactuals(
            {document.doc_id: document},
            (_trajectory(document),),
            (_replayed(document)[0],),
            utility_artifact=_utility_artifact(document),
            credit_routes={document.doc_id: {
                "d1": "document", "d2": "linked", "d3": "linked",
            }},
            pair_history={},
            budget=budget,
            endpoint_budget=0,
            seed=0,
            current_round=0,
        )


def test_scheduler_rejects_fractional_endpoint_budget():
    from cloak.train.counterfactuals import schedule_counterfactuals

    document = _document()
    with pytest.raises(ValueError, match="endpoint_budget must be an integer"):
        schedule_counterfactuals(
            {document.doc_id: document},
            (_trajectory(document), _trajectory(document)),
            (_replayed(document)[0], _replayed(document)[0]),
            utility_artifact=_utility_artifact(document),
            credit_routes={document.doc_id: {
                "d1": "document", "d2": "linked", "d3": "linked",
            }},
            pair_history={},
            budget=5,
            endpoint_budget=0.5,
            seed=0,
            current_round=0,
        )


def test_scheduler_is_seeded_and_reports_measured_pair_age():
    from cloak.train.counterfactuals import pair_history_key, schedule_counterfactuals

    document = _document()
    trajectories = (_trajectory(document), _trajectory(document))
    replayed = (_replayed(document)[0], _replayed(document)[0])
    history = {}
    for rollout_index, trajectory in enumerate(trajectories):
        for decision in document.policy_decisions:
            for alternative in (f"{decision.decision_id}-fine", f"{decision.decision_id}-coarse",
                                f"{decision.decision_id}-keep",
                                f"{decision.decision_id}-placeholder"):
                history[pair_history_key(
                    document.doc_id, rollout_index, decision.decision_id,
                    trajectory.action_vector[decision.decision_id], alternative,
                )] = 2
    kwargs = dict(
        utility_artifact=_utility_artifact(document),
        credit_routes={document.doc_id: {
            "d1": "document", "d2": "linked", "d3": "linked",
        }},
        pair_history=history,
        budget=5,
        endpoint_budget=1,
        seed=23,
        current_round=10,
    )
    first, diagnostics = schedule_counterfactuals(
        {document.doc_id: document}, trajectories, replayed, **kwargs,
    )
    second, _ = schedule_counterfactuals(
        {document.doc_id: document}, trajectories, replayed, **kwargs,
    )

    assert first == second
    assert diagnostics["never_measured_eligible_decisions"] == 0
    assert diagnostics["pair_age"] == {
        "count": 5, "min": 8, "max": 8, "mean": 8.0,
    }


def test_scheduler_balances_direction_before_history_preference():
    from cloak.train.counterfactuals import pair_history_key, schedule_counterfactuals

    document = _document()
    trajectories = (_trajectory(document), _trajectory(document))
    replayed = (_replayed(document)[0], _replayed(document)[0])
    history = {}
    for rollout_index, trajectory in enumerate(trajectories):
        for decision in document.policy_decisions:
            history[pair_history_key(
                document.doc_id,
                rollout_index,
                decision.decision_id,
                trajectory.action_vector[decision.decision_id],
                f"{decision.decision_id}-coarse",
            )] = 2
    requests, _ = schedule_counterfactuals(
        {document.doc_id: document},
        trajectories,
        replayed,
        utility_artifact=_utility_artifact(document),
        credit_routes={document.doc_id: {
            "d1": "document", "d2": "linked", "d3": "linked",
        }},
        pair_history=history,
        budget=5,
        endpoint_budget=1,
        seed=17,
        current_round=10,
    )

    adjacent = [
        request for request in requests
        if request.direction not in {"keep", "placeholder"}
    ]
    counts = {
        direction: sum(request.direction == direction for request in adjacent)
        for direction in ("finer", "coarser")
    }
    assert abs(counts["finer"] - counts["coarser"]) <= 1
    assert counts["finer"] > 0
    assert counts["coarser"] > 0


def test_scheduler_fixes_endpoint_stratum_before_unseen_pair_priority():
    from cloak.train.counterfactuals import pair_history_key, schedule_counterfactuals

    document = _document()
    trajectories = (_trajectory(document), _trajectory(document))
    replayed = tuple(
        _replayed(
            document,
            entropies={"d1": 0.5, "d2": 0.5, "d3": 0.5},
        )[0]
        for _ in trajectories
    )
    artifact = _utility_artifact(document)
    artifact["assertions"]["a-hyperedge"][
        "policy_dependency_decision_ids"
    ] = ["d1", "d2", "d3"]
    history = {}
    for rollout_index, trajectory in enumerate(trajectories):
        for decision in document.policy_decisions:
            for suffix in ("fine", "coarse", "keep", "placeholder"):
                if (
                    rollout_index == 0
                    and decision.decision_id == "d1"
                    and suffix == "fine"
                ):
                    continue
                history[pair_history_key(
                    document.doc_id,
                    rollout_index,
                    decision.decision_id,
                    trajectory.action_vector[decision.decision_id],
                    f"{decision.decision_id}-{suffix}",
                )] = 2

    requests, _ = schedule_counterfactuals(
        {document.doc_id: document},
        trajectories,
        replayed,
        utility_artifact=artifact,
        credit_routes={document.doc_id: {
            "d1": "linked", "d2": "linked", "d3": "linked",
        }},
        pair_history=history,
        budget=5,
        endpoint_budget=1,
        seed=9,
        current_round=10,
    )

    prioritized = next(
        request for request in requests
        if request.rollout_index == 0 and request.decision_id == "d1"
    )
    assert prioritized.alternative_action_id == "d1-fine"
    assert prioritized.direction == "finer"


def test_scheduler_priority_tiers_are_lexicographic_by_route_class():
    from cloak.train.counterfactuals import schedule_counterfactuals

    document = _document()
    trajectories = (_trajectory(document), _trajectory(document))
    replayed = tuple(
        _replayed(
            document,
            entropies={"d1": 0.5, "d2": 0.5, "d3": 0.5},
        )[0]
        for _ in trajectories
    )
    artifact = _utility_artifact(document)
    artifact["documents"][document.doc_id]["assertion_ids"].append("a-single")
    artifact["assertions"]["a-single"] = {
        "assertion_id": "a-single",
        "doc_id": document.doc_id,
        "family": "context",
        "status": "accepted",
        "weight": 0.1,
        "credit_routing": "linked",
        "policy_dependency_decision_ids": ["d3"],
    }

    requests, _ = schedule_counterfactuals(
        {document.doc_id: document},
        trajectories,
        replayed,
        utility_artifact=artifact,
        credit_routes={document.doc_id: {
            "d1": "document", "d2": "linked", "d3": "linked",
        }},
        pair_history={},
        budget=5,
        endpoint_budget=0,
        seed=0,
        current_round=0,
    )

    tiers = {
        decision_id: min(
            request.priority_tier for request in requests
            if request.priority_tier >= 0 and request.decision_id == decision_id
        )
        for decision_id in ("d1", "d2", "d3")
    }
    assert tiers["d1"] < tiers["d2"] < tiers["d3"]


def test_scheduler_entropy_precedes_unseen_and_pair_age():
    from cloak.train.counterfactuals import pair_history_key, schedule_counterfactuals

    document = _document()
    trajectories = (_trajectory(document), _trajectory(document))
    replayed = tuple(
        _replayed(
            document,
            entropies={"d1": 0.9, "d2": 0.5, "d3": 0.1},
        )[0]
        for _ in trajectories
    )
    artifact = _utility_artifact(document)
    artifact["documents"][document.doc_id]["assertion_ids"] = ["a-residual"]
    history = {}
    for rollout_index, trajectory in enumerate(trajectories):
        for decision in document.policy_decisions:
            for suffix in ("fine", "coarse"):
                history[pair_history_key(
                    document.doc_id,
                    rollout_index,
                    decision.decision_id,
                    trajectory.action_vector[decision.decision_id],
                    f"{decision.decision_id}-{suffix}",
                )] = 0 if decision.decision_id == "d3" else 9

    requests, _ = schedule_counterfactuals(
        {document.doc_id: document},
        trajectories,
        replayed,
        utility_artifact=artifact,
        credit_routes={document.doc_id: {
            "d1": "document", "d2": "document", "d3": "document",
        }},
        pair_history=history,
        budget=5,
        endpoint_budget=0,
        seed=0,
        current_round=10,
    )

    tiers = {
        decision_id: min(
            request.priority_tier for request in requests
            if request.priority_tier >= 0 and request.decision_id == decision_id
        )
        for decision_id in ("d1", "d2", "d3")
    }
    assert tiers["d1"] < tiers["d2"] < tiers["d3"]


def test_scheduler_uses_oldest_measured_pair_after_earlier_ties():
    from cloak.train.counterfactuals import pair_history_key, schedule_counterfactuals

    document = _document()
    trajectories = (_trajectory(document), _trajectory(document))
    replayed = tuple(
        _replayed(
            document,
            entropies={"d1": 0.5, "d2": 0.5, "d3": 0.5},
        )[0]
        for _ in trajectories
    )
    artifact = _utility_artifact(document)
    artifact["documents"][document.doc_id]["assertion_ids"] = ["a-residual"]
    measured_round = {"d1": 0, "d2": 5, "d3": 9}
    history = {}
    for rollout_index, trajectory in enumerate(trajectories):
        for decision in document.policy_decisions:
            for suffix in ("fine", "coarse"):
                history[pair_history_key(
                    document.doc_id,
                    rollout_index,
                    decision.decision_id,
                    trajectory.action_vector[decision.decision_id],
                    f"{decision.decision_id}-{suffix}",
                )] = measured_round[decision.decision_id]

    requests, _ = schedule_counterfactuals(
        {document.doc_id: document},
        trajectories,
        replayed,
        utility_artifact=artifact,
        credit_routes={document.doc_id: {
            "d1": "document", "d2": "document", "d3": "document",
        }},
        pair_history=history,
        budget=5,
        endpoint_budget=0,
        seed=0,
        current_round=10,
    )

    tiers = {
        decision_id: min(
            request.priority_tier for request in requests
            if request.priority_tier >= 0 and request.decision_id == decision_id
        )
        for decision_id in ("d1", "d2", "d3")
    }
    assert tiers["d1"] < tiers["d2"] < tiers["d3"]


def test_executor_changes_one_decision_everywhere_rescores_complete_vectors_and_uses_replay_graph(
    tmp_path,
):
    from cloak.train.counterfactuals import (
        CounterfactualRequest,
        execute_counterfactuals,
    )

    document = _document()
    trajectory = _trajectory(document)
    replayed, logits = _replayed(document)
    artifact = _utility_artifact(document)
    calls = []

    def scorer(requests, *, cache, remote_workers, reader_workers, reader_refresh=False):
        del remote_workers, reader_workers
        assert reader_refresh is False
        cache.last_batch_metrics = {"cache_hits": 2}
        output = []
        for request in requests:
            rendered, replacements = assemble_action_vector(
                request.document, request.action_vector,
            )
            calls.append((dict(request.action_vector), replacements))
            utility = 0.8 if request.action_vector["d1"] == "d1-middle" else 0.3
            output.append(make_result(
                doc_id=request.document.doc_id,
                action_vector=request.action_vector,
                doc_p=rendered,
                out_p="fixture-remote",
                out_final="fixture-final",
                component_scores={"a-residual": utility, "a-hyperedge": utility},
                utility=utility,
            ))
        return output

    losses, diagnostics = execute_counterfactuals(
        (CounterfactualRequest(
            document.doc_id, 0, "d1", "d1-middle", "d1-fine", "finer", 0,
        ),),
        {document.doc_id: document},
        (trajectory,),
        (replayed,),
        utility_artifact=artifact,
        environment_hash="env-hash",
        cache=UtilityCache(tmp_path / "cache.jsonl"),
        remote_workers=1,
        reader_workers=1,
        score_batch=scorer,
        scheduler_diagnostics={
            "budget": 1, "uniform_allocation": 0, "priority_allocation": 1,
            "endpoint_fraction": 0.0, "direction_balance": {"finer": 1},
            "cache_hits": 0, "never_measured_eligible_decisions": 1,
            "pair_age": {"count": 0, "min": None, "max": None, "mean": None},
            "delta_u": {"count": 0}, "skip_reasons": {},
        },
    )

    assert len(calls) == 2
    selected_vector, alternative_vector = calls[0][0], calls[1][0]
    assert selected_vector == dict(trajectory.action_vector)
    assert [key for key in selected_vector if selected_vector[key] != alternative_vector[key]] == [
        "d1"
    ]
    alternative_d1 = [row for row in calls[1][1] if row["decision_id"] == "d1"]
    assert len(alternative_d1) == 2
    assert {row["action_id"] for row in alternative_d1} == {"d1-fine"}
    assert calls[1][0]["d2"] == "d2-middle"
    assert losses.keys() == {(0, "d1")}
    selected_step = replayed.steps[0]
    q_pair = torch.softmax(torch.stack([
        selected_step.log_probs[1], selected_step.log_probs[0],
    ]), dim=0)[0]
    assert losses[(0, "d1")].item() == pytest.approx(
        float(-0.5 * (q_pair.detach() - 0.5))
    )
    losses[(0, "d1")].backward()
    assert logits["d1"].grad is not None
    assert logits["d1"].grad.abs().sum().item() > 0
    assert diagnostics["delta_u"] == {
        "count": 1, "zero": 0, "positive": 1, "negative": 0,
        "mean_abs": 0.5, "max_abs": 0.5,
    }
    assert diagnostics["cache_hits"] == 2


def test_executor_rejects_result_bound_to_another_document(tmp_path):
    from cloak.train.counterfactuals import (
        CounterfactualRequest,
        execute_counterfactuals,
    )

    document = _document()
    trajectory = _trajectory(document)
    replayed, _ = _replayed(document)

    def scorer(requests, **kwargs):
        return [
            make_result(
                doc_id="wrong/document",
                action_vector=request.action_vector,
                doc_p="fixture",
                out_p="fixture",
                out_final="fixture",
                component_scores={"a-residual": 0.5, "a-hyperedge": 0.5},
                utility=0.5,
            )
            for request in requests
        ]

    with pytest.raises(ValueError, match="result document mismatch"):
        execute_counterfactuals(
            (CounterfactualRequest(
                document.doc_id, 0, "d1", "d1-middle", "d1-fine", "finer", 0,
            ),),
            {document.doc_id: document},
            (trajectory,),
            (replayed,),
            utility_artifact=_utility_artifact(document),
            environment_hash="env-hash",
            cache=UtilityCache(tmp_path / "cache.jsonl"),
            remote_workers=1,
            reader_workers=1,
            score_batch=scorer,
            scheduler_diagnostics={
                "budget": 1, "uniform_allocation": 0, "priority_allocation": 1,
                "endpoint_fraction": 0.0, "direction_balance": {"finer": 1},
                "cache_hits": 0, "never_measured_eligible_decisions": 1,
                "pair_age": {"count": 0, "min": None, "max": None, "mean": None},
                "delta_u": {"count": 0}, "skip_reasons": {},
            },
        )


def test_executor_rejects_equal_selected_and_alternative_without_scoring(tmp_path):
    from cloak.train.counterfactuals import (
        CounterfactualRequest,
        execute_counterfactuals,
    )

    document = _document()
    scorer_calls = []
    with pytest.raises(ValueError, match="alternative is ineligible"):
        execute_counterfactuals(
            (CounterfactualRequest(
                document.doc_id, 0, "d1", "d1-middle", "d1-middle", "finer", 0,
            ),),
            {document.doc_id: document},
            (_trajectory(document),),
            (_replayed(document)[0],),
            utility_artifact=_utility_artifact(document),
            environment_hash="env-hash",
            cache=UtilityCache(tmp_path / "cache.jsonl"),
            remote_workers=1,
            reader_workers=1,
            score_batch=lambda *args, **kwargs: scorer_calls.append(args),
            scheduler_diagnostics={
                "budget": 1, "uniform_allocation": 0, "priority_allocation": 1,
                "endpoint_fraction": 0.0, "direction_balance": {"finer": 1},
                "cache_hits": 0, "never_measured_eligible_decisions": 1,
                "pair_age": {"count": 0, "min": None, "max": None, "mean": None},
                "delta_u": {"count": 0}, "skip_reasons": {},
            },
        )
    assert scorer_calls == []


def test_executor_rejects_direction_label_that_disagrees_with_action_semantics(tmp_path):
    from cloak.train.counterfactuals import (
        CounterfactualRequest,
        execute_counterfactuals,
    )

    document = _document()
    with pytest.raises(ValueError, match="direction disagrees"):
        execute_counterfactuals(
            (CounterfactualRequest(
                document.doc_id, 0, "d1", "d1-middle", "d1-fine", "coarser", 0,
            ),),
            {document.doc_id: document},
            (_trajectory(document),),
            (_replayed(document)[0],),
            utility_artifact=_utility_artifact(document),
            environment_hash="env-hash",
            cache=UtilityCache(tmp_path / "cache.jsonl"),
            remote_workers=1,
            reader_workers=1,
            score_batch=lambda *args, **kwargs: (),
            scheduler_diagnostics={
                "budget": 1, "uniform_allocation": 0, "priority_allocation": 1,
                "endpoint_fraction": 0.0, "direction_balance": {"finer": 1},
                "cache_hits": 0, "never_measured_eligible_decisions": 1,
                "pair_age": {"count": 0, "min": None, "max": None, "mean": None},
                "delta_u": {"count": 0}, "skip_reasons": {},
            },
        )


def test_executor_rejects_replay_that_is_not_the_full_original_menu(tmp_path):
    from cloak.train.counterfactuals import (
        CounterfactualRequest,
        execute_counterfactuals,
    )

    document = _document()
    replayed, _ = _replayed(document)
    d1 = replayed.steps[0]
    replayed = replace(
        replayed,
        steps=(replace(
            d1,
            legal_action_ids=d1.legal_action_ids[:3],
            log_probs=d1.log_probs[:3],
        ), *replayed.steps[1:]),
    )
    with pytest.raises(ValueError, match="full original legal menu"):
        execute_counterfactuals(
            (CounterfactualRequest(
                document.doc_id, 0, "d1", "d1-middle", "d1-fine", "finer", 0,
            ),),
            {document.doc_id: document},
            (_trajectory(document),),
            (replayed,),
            utility_artifact=_utility_artifact(document),
            environment_hash="env-hash",
            cache=UtilityCache(tmp_path / "cache.jsonl"),
            remote_workers=1,
            reader_workers=1,
            score_batch=lambda *args, **kwargs: (),
            scheduler_diagnostics={
                "budget": 1, "uniform_allocation": 0, "priority_allocation": 1,
                "endpoint_fraction": 0.0, "direction_balance": {"finer": 1},
                "cache_hits": 0, "never_measured_eligible_decisions": 1,
                "pair_age": {"count": 0, "min": None, "max": None, "mean": None},
                "delta_u": {"count": 0}, "skip_reasons": {},
            },
        )


def test_request_priority_tier_never_changes_pair_loss(tmp_path):
    from cloak.train.counterfactuals import (
        CounterfactualRequest,
        execute_counterfactuals,
    )

    document = _document()
    trajectory = _trajectory(document)
    replayed, _ = _replayed(document)

    def scorer(requests, **kwargs):
        return [
            make_result(
                doc_id=request.document.doc_id,
                action_vector=request.action_vector,
                doc_p="fixture",
                out_p="fixture",
                out_final="fixture",
                component_scores={
                    "a-residual": 0.7 if request.action_vector["d1"] == "d1-middle" else 0.2,
                    "a-hyperedge": 0.7 if request.action_vector["d1"] == "d1-middle" else 0.2,
                },
                utility=0.7 if request.action_vector["d1"] == "d1-middle" else 0.2,
            )
            for request in requests
        ]

    losses = []
    for priority_tier in (-1, 9):
        result, _ = execute_counterfactuals(
            (CounterfactualRequest(
                document.doc_id, 0, "d1", "d1-middle", "d1-fine", "finer",
                priority_tier,
            ),),
            {document.doc_id: document},
            (trajectory,),
            (replayed,),
            utility_artifact=_utility_artifact(document),
            environment_hash="env-hash",
            cache=UtilityCache(tmp_path / f"cache-{priority_tier}.jsonl"),
            remote_workers=1,
            reader_workers=1,
            score_batch=scorer,
            scheduler_diagnostics={
                "budget": 1, "uniform_allocation": 0, "priority_allocation": 1,
                "endpoint_fraction": 0.0, "direction_balance": {"finer": 1},
                "cache_hits": 0, "never_measured_eligible_decisions": 1,
                "pair_age": {"count": 0, "min": None, "max": None, "mean": None},
                "delta_u": {"count": 0}, "skip_reasons": {},
            },
        )
        losses.append(result[(0, "d1")])
    assert torch.equal(losses[0], losses[1])
