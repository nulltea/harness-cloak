from dataclasses import FrozenInstanceError, fields, replace
from types import MappingProxyType

import pytest
import torch

from cloak.train.ranker_environment import (
    RankerAction,
    RankerDecision,
    RankerDocument,
)


def _action(owner: str, suffix: str, mode: str, fill=None, index=None):
    return RankerAction(
        action_id=f"{owner}-{suffix}",
        mode=mode,
        fill=fill,
        authored_level_index=index,
        runtime_type="TYPE" if owner != "fixed" else "FIXED",
    )


def _decision(
    decision_id: str,
    occurrence_ids: tuple[str, ...],
    *,
    fixed: bool = False,
):
    if fixed:
        actions = (_action("fixed", "placeholder", "placeholder"),)
    else:
        actions = (
            _action(decision_id, "level", "level", "shared fill", 0),
            _action(decision_id, "keep", "keep", decision_id),
            _action(decision_id, "placeholder", "placeholder"),
        )
    return RankerDecision(
        decision_id=decision_id,
        profile_id=None if fixed else f"profile-{decision_id}",
        runtime_type="FIXED" if fixed else "TYPE",
        canonical_key=decision_id,
        occurrence_ids=occurrence_ids,
        actions=actions,
    )


def _document(*, unordered=False):
    alpha = _decision("alpha", ("o-alpha-1", "o-alpha-2"))
    beta = _decision("beta", ("o-beta",))
    policy = (beta, alpha) if unordered else (alpha, beta)
    occurrences = (
        MappingProxyType({
            "occurrence_id": "o-alpha-1", "decision_id": "alpha",
            "start": 0, "end": 5, "surface": "Alpha", "controlled": True,
        }),
        MappingProxyType({
            "occurrence_id": "o-beta", "decision_id": "beta",
            "start": 10, "end": 14, "surface": "Beta", "controlled": True,
        }),
        MappingProxyType({
            "occurrence_id": "o-alpha-2", "decision_id": "alpha",
            "start": 19, "end": 24, "surface": "alpha", "controlled": True,
        }),
        MappingProxyType({
            "occurrence_id": "o-fixed", "decision_id": "fixed",
            "start": 25, "end": 27, "surface": "ID", "controlled": True,
        }),
    )
    return RankerDocument(
        doc_id="fixture/doc",
        corpus="fixture",
        text="Alpha met Beta and alpha ID.",
        occurrences=occurrences,
        policy_decisions=policy,
        fixed_decisions=(_decision("fixed", ("o-fixed",), fixed=True),),
    )


class StubPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = torch.nn.Parameter(torch.tensor([3.0, 0.0, 2.0]))
        self.menus = []
        self.profiles = []
        self.grad_modes = []

    def begin_document(self, document, profile):
        self.profiles.append(profile)
        return ()

    def log_probs(self, state, decision, legal_action_ids, profile):
        self.grad_modes.append(torch.is_grad_enabled())
        self.menus.append((decision.decision_id, tuple(legal_action_ids), state))
        self.profiles.append(profile)
        mode_index = {
            action.action_id: {"level": 0, "keep": 1, "placeholder": 2}[action.mode]
            for action in decision.actions
        }
        logits = torch.stack([self.scores[mode_index[action_id]] for action_id in legal_action_ids])
        return torch.log_softmax(logits, dim=0)

    def advance(self, state, decision, action_id):
        return (*state, action_id)


def test_public_trajectory_dataclasses_are_exact_frozen_and_opaque():
    from cloak.train.interactive_ranker import SampledStep, SampledTrajectory

    assert [field.name for field in fields(SampledStep)] == [
        "decision_id", "legal_action_ids", "selected_action_id",
        "claimed_fills_before",
    ]
    assert [field.name for field in fields(SampledTrajectory)] == [
        "doc_id", "lambda_profile", "steps", "action_vector",
    ]
    profile = object()
    trajectory = SampledTrajectory(
        doc_id="d", lambda_profile=profile, steps=(),
        action_vector=MappingProxyType({}),
    )
    assert trajectory.lambda_profile is profile
    with pytest.raises(FrozenInstanceError):
        trajectory.doc_id = "changed"
    with pytest.raises(TypeError):
        trajectory.action_vector["d"] = "a"


def test_legal_action_ids_masks_only_level_collisions():
    from cloak.train.interactive_ranker import legal_action_ids

    decision = _decision("alpha", ("o-alpha-1", "o-alpha-2"))
    assert legal_action_ids(decision, {}, ()) == tuple(
        action.action_id for action in decision.actions
    )
    assert legal_action_ids(decision, {"shared fill": "other"}, ()) == (
        "alpha-keep", "alpha-placeholder",
    )
    assert legal_action_ids(decision, {"shared fill": "alpha"}, ()) == tuple(
        action.action_id for action in decision.actions
    )
    assert legal_action_ids(decision, {}, {"SHARED FILL"}) == (
        "alpha-keep", "alpha-placeholder",
    )


def test_sampling_masks_later_collision_and_repeated_decision_counts_once():
    from cloak.train.interactive_ranker import sample_trajectory

    policy = StubPolicy()
    profile = object()
    trajectory = sample_trajectory(
        policy, _document(), profile, greedy=True, generator=None,
    )

    assert trajectory.lambda_profile is profile
    assert len(trajectory.steps) == 2
    assert trajectory.steps[0].selected_action_id == "alpha-level"
    assert trajectory.steps[0].claimed_fills_before == ()
    assert trajectory.steps[1].claimed_fills_before == ("shared fill",)
    assert trajectory.steps[1].legal_action_ids == ("beta-keep", "beta-placeholder")
    assert trajectory.steps[1].selected_action_id == "beta-placeholder"
    assert sum(step.selected_action_id.endswith("-level") for step in trajectory.steps) == 1
    assert policy.grad_modes == [False, False]
    assert all(seen is profile for seen in policy.profiles)


def test_assembly_applies_one_choice_to_repeated_occurrences_and_auto_fixed():
    from cloak.train.interactive_ranker import assemble_action_vector

    rendered, replacements = assemble_action_vector(
        _document(),
        {"alpha": "alpha-level", "beta": "beta-placeholder"},
    )
    alpha_rows = [row for row in replacements if row["decision_id"] == "alpha"]
    assert len(alpha_rows) == 2
    assert {row["action_id"] for row in alpha_rows} == {"alpha-level"}
    assert len([row for row in replacements if row["decision_id"] == "fixed"]) == 1
    assert "Shared fill" in rendered
    assert "shared fill" in rendered
    assert rendered.count("<TYPE_1>") == 1
    assert rendered.count("<FIXED_1>") == 1


def test_assembly_rejects_omissions_unknown_actions_and_collisions():
    from cloak.train.interactive_ranker import assemble_action_vector

    with pytest.raises(ValueError, match="action-vector omissions"):
        assemble_action_vector(_document(), {"alpha": "alpha-level"})
    with pytest.raises(ValueError, match="unknown policy decision"):
        assemble_action_vector(
            _document(),
            {"alpha": "alpha-level", "beta": "beta-placeholder", "extra": "x"},
        )
    with pytest.raises(ValueError, match="collision"):
        assemble_action_vector(
            _document(), {"alpha": "alpha-level", "beta": "beta-level"}
        )


def test_sampling_reserves_fixed_exact_rewrites():
    from cloak.train.interactive_ranker import sample_trajectory

    document = _document()
    fixed_level = replace(
        document.fixed_decisions[0],
        actions=(_action("fixed", "level", "level", "shared fill", 0),),
    )
    document = replace(document, fixed_decisions=(fixed_level,))
    trajectory = sample_trajectory(
        StubPolicy(), document, object(), greedy=True, generator=None,
    )
    assert trajectory.steps[0].legal_action_ids == (
        "alpha-keep", "alpha-placeholder",
    )


def test_replay_returns_gradient_tensors_and_complete_ordered_distributions():
    from cloak.train.interactive_ranker import replay_trajectory, sample_trajectory

    document = _document()
    policy = StubPolicy()
    trajectory = sample_trajectory(
        policy, document, "profile", greedy=True, generator=None,
    )
    policy.menus.clear()
    replayed = replay_trajectory(policy, document, trajectory, "profile")

    assert [step.decision_id for step in replayed.steps] == ["alpha", "beta"]
    assert policy.grad_modes[-2:] == [True, True]
    assert tuple(step.legal_action_ids for step in replayed.steps) == tuple(
        step.legal_action_ids for step in trajectory.steps
    )
    assert all(step.log_prob.requires_grad for step in replayed.steps)
    assert all(step.entropy.requires_grad for step in replayed.steps)
    assert [tuple(step.log_probs.shape) for step in replayed.steps] == [
        (3,), (2,),
    ]
    assert torch.allclose(
        torch.stack([step.log_probs.exp().sum() for step in replayed.steps]),
        torch.ones(2),
    )
    (-sum(step.log_prob for step in replayed.steps)).backward()
    assert policy.scores.grad is not None


def test_replay_raises_when_sampled_menu_or_claimed_state_differs():
    from cloak.train.interactive_ranker import replay_trajectory, sample_trajectory

    document = _document()
    trajectory = sample_trajectory(
        StubPolicy(), document, "profile", greedy=True, generator=None,
    )
    bad_menu = replace(
        trajectory,
        steps=(
            trajectory.steps[0],
            replace(trajectory.steps[1], legal_action_ids=("beta-placeholder",)),
        ),
    )
    with pytest.raises(ValueError, match="replayed legal menu differs"):
        replay_trajectory(StubPolicy(), document, bad_menu, "profile")

    bad_claims = replace(
        trajectory,
        steps=(
            trajectory.steps[0],
            replace(trajectory.steps[1], claimed_fills_before=()),
        ),
    )
    with pytest.raises(ValueError, match="claimed fills differ"):
        replay_trajectory(StubPolicy(), document, bad_claims, "profile")


@pytest.mark.parametrize("operation", ["sample", "assemble", "replay"])
def test_operations_reject_unordered_policy_decisions(operation):
    import cloak.train.interactive_ranker as interactive

    document = _document(unordered=True)
    with pytest.raises(ValueError, match="unordered policy decisions"):
        if operation == "sample":
            interactive.sample_trajectory(
                StubPolicy(), document, "profile", greedy=True, generator=None,
            )
        elif operation == "assemble":
            interactive.assemble_action_vector(
                document, {"alpha": "alpha-level", "beta": "beta-placeholder"}
            )
        else:
            ordered = _document()
            trajectory = interactive.sample_trajectory(
                StubPolicy(), ordered, "profile", greedy=True, generator=None,
            )
            interactive.replay_trajectory(
                StubPolicy(), document, trajectory, "profile"
            )


def _replayed(log_probabilities):
    from cloak.train.interactive_ranker import ReplayedStep, ReplayedTrajectory

    trajectories = []
    for rollout_index, values in enumerate(log_probabilities):
        steps = tuple(
            ReplayedStep(
                decision_id=decision_id,
                legal_action_ids=(f"{decision_id}-selected",),
                selected_action_id=f"{decision_id}-selected",
                log_prob=log_prob,
                entropy=torch.zeros((), requires_grad=True),
                log_probs=log_prob.reshape(1),
            )
            for decision_id, log_prob in zip(("p1", "p2"), values, strict=True)
        )
        trajectories.append(ReplayedTrajectory(
            doc_id="d1", lambda_profile=f"profile-{rollout_index}", steps=steps,
        ))
    return tuple(trajectories)


def test_provisional_utility_loss_has_one_term_per_pair_and_divides_only_by_rollouts():
    from cloak.train.interactive_ranker import provisional_utility_loss
    from cloak.train.utility_credit import DocumentUtilityCredit

    logs = tuple(
        torch.tensor(value, requires_grad=True)
        for value in (-0.1, -0.2, -0.3, -0.4)
    )
    replayed = _replayed((logs[:2], logs[2:]))
    credit = DocumentUtilityCredit(
        document_utility=(0.0, 0.0), linked_utility={}, residual_utility=(0.0, 0.0),
        provisional_advantage={
            (0, "p1"): 1.0, (0, "p2"): 2.0,
            (1, "p1"): -1.0, (1, "p2"): -2.0,
        },
        route={"p1": "linked", "p2": "document"},
    )

    loss = provisional_utility_loss(replayed, credit)

    assert loss.item() == pytest.approx(-0.3)
    loss.backward()
    assert [value.grad.item() for value in logs] == pytest.approx([
        -0.5, -1.0, 0.5, 1.0,
    ])


def test_tied_pair_contributes_a_zero_term_without_detaching_its_log_probability():
    from cloak.train.interactive_ranker import provisional_utility_loss
    from cloak.train.utility_credit import DocumentUtilityCredit

    logs = tuple(
        torch.tensor(value, requires_grad=True)
        for value in (-0.1, -0.2, -0.3, -0.4)
    )
    credit = DocumentUtilityCredit(
        document_utility=(0.0, 0.0), linked_utility={}, residual_utility=(0.0, 0.0),
        provisional_advantage={
            (0, "p1"): 1.0, (0, "p2"): 0.0,
            (1, "p1"): -1.0, (1, "p2"): 0.0,
        },
        route={"p1": "linked", "p2": "linked"},
    )

    loss = provisional_utility_loss(_replayed((logs[:2], logs[2:])), credit)
    loss.backward()

    assert logs[1].grad is not None and logs[1].grad.item() == 0.0
    assert logs[3].grad is not None and logs[3].grad.item() == 0.0


def test_provisional_utility_loss_rejects_missing_or_extra_credit_pairs():
    from cloak.train.interactive_ranker import provisional_utility_loss
    from cloak.train.utility_credit import DocumentUtilityCredit

    logs = tuple(torch.tensor(-0.1, requires_grad=True) for _ in range(4))
    credit = DocumentUtilityCredit(
        document_utility=(0.0, 0.0), linked_utility={}, residual_utility=(0.0, 0.0),
        provisional_advantage={(0, "p1"): 0.0}, route={"p1": "linked"},
    )

    with pytest.raises(ValueError, match="credit pairs differ from replayed trajectory pairs"):
        provisional_utility_loss(_replayed((logs[:2], logs[2:])), credit)


def test_hybrid_utility_loss_substitutes_pair_terms_in_place_and_divides_once():
    from cloak.train.interactive_ranker import hybrid_utility_loss
    from cloak.train.utility_credit import DocumentUtilityCredit

    logs = tuple(
        torch.tensor(value, requires_grad=True)
        for value in (-0.1, -0.2, -0.3, -0.4)
    )
    pair_loss = torch.tensor(0.7, requires_grad=True)
    credit = DocumentUtilityCredit(
        document_utility=(0.0, 0.0), linked_utility={}, residual_utility=(0.0, 0.0),
        provisional_advantage={
            (0, "p1"): 1.0, (0, "p2"): 2.0,
            (1, "p1"): -1.0, (1, "p2"): -2.0,
        },
        route={"p1": "linked", "p2": "document"},
    )

    loss = hybrid_utility_loss(
        _replayed((logs[:2], logs[2:])),
        credit,
        {(0, "p2"): pair_loss},
    )

    assert loss.item() == pytest.approx(-0.15)
    loss.backward()
    assert logs[0].grad.item() == pytest.approx(-0.5)
    assert logs[1].grad is None
    assert logs[2].grad.item() == pytest.approx(0.5)
    assert logs[3].grad.item() == pytest.approx(1.0)
    assert pair_loss.grad.item() == pytest.approx(0.5)


def test_hybrid_utility_loss_rejects_unknown_pair_or_nonscalar_loss():
    from cloak.train.interactive_ranker import hybrid_utility_loss
    from cloak.train.utility_credit import DocumentUtilityCredit

    logs = tuple(torch.tensor(-0.1, requires_grad=True) for _ in range(4))
    replayed = _replayed((logs[:2], logs[2:]))
    credit = DocumentUtilityCredit(
        document_utility=(0.0, 0.0), linked_utility={}, residual_utility=(0.0, 0.0),
        provisional_advantage={
            (0, "p1"): 0.0, (0, "p2"): 0.0,
            (1, "p1"): 0.0, (1, "p2"): 0.0,
        },
        route={"p1": "linked", "p2": "document"},
    )

    with pytest.raises(ValueError, match="unknown counterfactual loss pairs"):
        hybrid_utility_loss(replayed, credit, {(2, "p1"): torch.tensor(0.0)})
    with pytest.raises(ValueError, match="scalar tensor"):
        hybrid_utility_loss(replayed, credit, {(0, "p1"): torch.zeros(2)})


def _count_reward(*, alpha_level=0.2, beta_level=0.9):
    from cloak.train.count_reward import CountActionScore, CountReward

    rows = {}
    decision_actions = {}
    for decision in _document().policy_decisions:
        decision_actions[decision.decision_id] = tuple(
            action.action_id for action in decision.actions
        )
        for action in decision.actions:
            score = {
                "level": alpha_level if decision.decision_id == "alpha" else beta_level,
                "keep": 0.0,
                "placeholder": 1.0,
            }[action.mode]
            rows[action.action_id] = CountActionScore(
                action_id=action.action_id,
                decision_id=decision.decision_id,
                runtime_type=decision.runtime_type,
                profile_id=decision.profile_id,
                mode=action.mode,
                count=None,
                score=score,
                grounding_status=None,
                source_family=None,
                evidence_ref=None,
            )
    rows["other-placeholder"] = CountActionScore(
        action_id="other-placeholder",
        decision_id="other-document-decision",
        runtime_type="TYPE",
        profile_id="profile-other",
        mode="placeholder",
        count=None,
        score=1.0,
        grounding_status=None,
        source_family=None,
        evidence_ref=None,
    )
    decision_actions["other-document-decision"] = ("other-placeholder",)
    return CountReward(rows, decision_actions)


def _utility_artifact():
    from cloak.train.utility_cache import stable_hash

    artifact = {
        "artifact_version": "utility-assertions-v2",
        "environment_hash": "env-hash",
        "reader_pin": {
            "model": "fixture-reader",
            "endpoint": "fixture-endpoint",
            "prompt_version": "fixture-prompt-v1",
            "response_schema": {"type": "string"},
            "revision": "fixture-revision",
        },
        "documents": {
            "fixture/doc": {
                "assertion_ids": ["a-delivered"],
                "policy_decision_ids": ["alpha", "beta"],
                "utility_weight_denominator": 1.0,
            },
        },
        "assertions": {
            "a-delivered": {
                "assertion_id": "a-delivered",
                "doc_id": "fixture/doc",
                "family": "delivered",
                "status": "accepted",
                "weight": 1.0,
                "credit_routing": "residual",
                "policy_dependency_decision_ids": [],
            },
        },
    }
    artifact["artifact_hash"] = stable_hash(artifact)
    return artifact


def _utility_result(action_vector, utility):
    from cloak.train.utility_cache import make_result

    return make_result(
        doc_id="fixture/doc",
        action_vector=action_vector,
        doc_p="rendered",
        out_p="remote",
        out_final="final",
        component_scores={"a-delivered": utility},
        utility=utility,
    )


class SequencePolicy(StubPolicy):
    """Select configured full vectors while preserving the trajectory interface."""

    def __init__(self, vectors):
        super().__init__()
        self.vectors = iter(vectors)
        self.current = None

    def begin_document(self, document, profile):
        self.current = next(self.vectors)
        return ()

    def log_probs(self, state, decision, legal_action_ids, profile):
        selected = self.current[decision.decision_id]
        if selected not in legal_action_ids:
            raise AssertionError(f"configured action is dynamically illegal: {selected}")
        logits = torch.full((len(legal_action_ids),), -100.0)
        logits[legal_action_ids.index(selected)] = 0.0
        return torch.log_softmax(logits, dim=0)


def test_bc_teacher_uses_authored_legal_levels_then_placeholder_and_replays_exactly():
    from cloak.train.interactive_ranker import (
        behavior_clone_trajectory,
        replay_trajectory,
    )

    document = _document()
    profile = object()
    teacher = behavior_clone_trajectory(document, profile)

    assert teacher.lambda_profile is profile
    assert [step.decision_id for step in teacher.steps] == ["alpha", "beta"]
    assert [step.selected_action_id for step in teacher.steps] == [
        "alpha-level", "beta-placeholder",
    ]
    assert teacher.steps[1].legal_action_ids == ("beta-keep", "beta-placeholder")
    assert teacher.steps[1].claimed_fills_before == ("shared fill",)
    assert dict(teacher.action_vector) == {
        "alpha": "alpha-level", "beta": "beta-placeholder",
    }

    replayed = replay_trajectory(StubPolicy(), document, teacher, profile)
    assert tuple(step.legal_action_ids for step in replayed.steps) == tuple(
        step.legal_action_ids for step in teacher.steps
    )
    assert [step.selected_action_id for step in replayed.steps] == [
        "alpha-level", "beta-placeholder",
    ]


def test_bc_teacher_uses_authored_index_not_action_tuple_position():
    from cloak.train.interactive_ranker import behavior_clone_trajectory

    document = _document()
    alpha = document.policy_decisions[0]
    broad = _action("alpha", "broad", "level", "broad fill", 1)
    alpha = replace(alpha, actions=(broad, *alpha.actions))
    document = replace(
        document,
        policy_decisions=(alpha, document.policy_decisions[1]),
    )

    teacher = behavior_clone_trajectory(document, "lambda-zero")

    assert teacher.steps[0].legal_action_ids[0] == "alpha-broad"
    assert teacher.steps[0].selected_action_id == "alpha-level"


def test_behavior_clone_trains_cross_entropy_and_records_stable_id_distributions():
    from cloak.train.interactive_ranker import behavior_clone

    policy = StubPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.2)
    initial = policy.scores.detach().clone()

    report = behavior_clone(
        policy,
        (_document(),),
        lambda_zero="lambda-zero",
        optimizer=optimizer,
        epochs=2,
    )

    assert len(report.epoch_losses) == 2
    assert all(value > 0.0 for value in report.epoch_losses)
    assert report.action_mode_counts == {"level": 1, "placeholder": 1}
    assert report.runtime_type_counts == {"TYPE": 2}
    assert tuple(report.trajectories[0].action_vector) == ("alpha", "beta")
    assert not torch.equal(policy.scores.detach(), initial)


def test_trajectory_point_recomputes_fixed_denominator_utility_and_count_score():
    from cloak.train.interactive_ranker import (
        behavior_clone_trajectory,
        trajectory_point,
    )

    trajectory = behavior_clone_trajectory(_document(), "lambda-zero")
    result = _utility_result(trajectory.action_vector, 0.7)
    point = trajectory_point(
        trajectory,
        result,
        count_reward=_count_reward(),
        utility_artifact=_utility_artifact(),
    )

    assert point.utility == pytest.approx(0.7)
    assert point.count_score == pytest.approx(0.6)
    assert point.component_scores == {"a-delivered": 0.7}
    assert point.result_hash == result.result_hash


def _exit_scorer(initial_utilities, refreshed_utilities=None, *, fail_refresh=False):
    calls = []
    refreshed_utilities = refreshed_utilities or initial_utilities

    def score(requests, *, cache, remote_workers, reader_workers, reader_refresh=False):
        del cache, remote_workers, reader_workers
        vectors = [tuple(request.action_vector.items()) for request in requests]
        calls.append((reader_refresh, vectors))
        if reader_refresh and fail_refresh:
            raise RuntimeError("fixture refresh failure")
        table = refreshed_utilities if reader_refresh else initial_utilities
        return [
            _utility_result(request.action_vector, table[tuple(request.action_vector.items())])
            for request in requests
        ]

    return score, calls


def test_exit_selects_strict_pure_utility_winner_and_reverifies_serially(tmp_path):
    from cloak.train.interactive_ranker import collect_exit_winners
    from cloak.train.utility_cache import UtilityCache

    reference = (("alpha", "alpha-level"), ("beta", "beta-placeholder"))
    winner = (("alpha", "alpha-keep"), ("beta", "beta-level"))
    scorer, calls = _exit_scorer(
        {reference: 0.5, winner: 0.8},
        {reference: 0.55, winner: 0.75},
    )
    policy = SequencePolicy((dict(winner),))

    collection = collect_exit_winners(
        policy,
        (_document(),),
        lambda_zero="lambda-zero",
        rollouts_per_document=1,
        utility_artifact=_utility_artifact(),
        environment_hash="env-hash",
        count_reward=_count_reward(),
        cache=UtilityCache(tmp_path / "cache.jsonl"),
        remote_workers=1,
        reader_workers=1,
        score_batch=scorer,
        generator=torch.Generator().manual_seed(0),
    )

    record = collection.documents[0]
    assert len(record.candidates) == 1
    assert record.winner is not None
    assert tuple(record.winner.action_vector.items()) == winner
    assert record.winner.utility == pytest.approx(0.75)
    assert [refresh for refresh, _ in calls] == [False, True, True]
    assert calls[1][1] == [winner]
    assert calls[2][1] == [reference]


def test_exit_lower_utility_higher_count_candidate_never_wins(tmp_path):
    from cloak.train.interactive_ranker import collect_exit_winners
    from cloak.train.utility_cache import UtilityCache

    reference = (("alpha", "alpha-level"), ("beta", "beta-placeholder"))
    higher_count = (("alpha", "alpha-placeholder"), ("beta", "beta-level"))
    scorer, calls = _exit_scorer({reference: 0.5, higher_count: 0.4})
    collection = collect_exit_winners(
        SequencePolicy((dict(higher_count),)),
        (_document(),),
        lambda_zero="lambda-zero",
        rollouts_per_document=1,
        utility_artifact=_utility_artifact(),
        environment_hash="env-hash",
        count_reward=_count_reward(alpha_level=0.2, beta_level=0.9),
        cache=UtilityCache(tmp_path / "cache.jsonl"),
        remote_workers=1,
        reader_workers=1,
        score_batch=scorer,
        generator=torch.Generator().manual_seed(0),
    )

    record = collection.documents[0]
    assert record.candidates[0].utility < record.reference.utility
    assert record.candidates[0].count_score > record.reference.count_score
    assert record.winner is None
    assert record.verification_status == "not_strictly_better"
    assert calls == [(False, [reference, higher_count])]


@pytest.mark.parametrize(
    ("candidate_utility", "fail_refresh"),
    [(0.5, False), (0.8, True)],
)
def test_exit_tie_never_replaces_bc_and_failed_refresh_drops_candidate(
    tmp_path, candidate_utility, fail_refresh,
):
    from cloak.train.interactive_ranker import collect_exit_winners
    from cloak.train.utility_cache import UtilityCache

    reference = (("alpha", "alpha-level"), ("beta", "beta-placeholder"))
    candidate = (("alpha", "alpha-keep"), ("beta", "beta-level"))
    scorer, _ = _exit_scorer(
        {reference: 0.5, candidate: candidate_utility},
        fail_refresh=fail_refresh,
    )
    collection = collect_exit_winners(
        SequencePolicy((dict(candidate),)),
        (_document(),),
        lambda_zero="lambda-zero",
        rollouts_per_document=1,
        utility_artifact=_utility_artifact(),
        environment_hash="env-hash",
        count_reward=_count_reward(),
        cache=UtilityCache(tmp_path / "cache.jsonl"),
        remote_workers=1,
        reader_workers=1,
        score_batch=scorer,
        generator=torch.Generator().manual_seed(0),
    )

    assert collection.documents[0].winner is None
    expected = "not_strictly_better" if not fail_refresh else "refresh_failed"
    assert collection.documents[0].verification_status == expected


def test_exit_refresh_contract_errors_fail_closed_instead_of_becoming_dropped_candidates(
    tmp_path,
):
    from cloak.train.interactive_ranker import collect_exit_winners
    from cloak.train.utility_cache import UtilityCache

    reference = (("alpha", "alpha-level"), ("beta", "beta-placeholder"))
    candidate = (("alpha", "alpha-keep"), ("beta", "beta-level"))

    def scorer(requests, *, cache, remote_workers, reader_workers, reader_refresh=False):
        del cache, remote_workers, reader_workers
        if reader_refresh:
            raise ValueError("corrupt cache identity")
        return [
            _utility_result(
                request.action_vector,
                {reference: 0.5, candidate: 0.8}[tuple(request.action_vector.items())],
            )
            for request in requests
        ]

    with pytest.raises(ValueError, match="corrupt cache identity"):
        collect_exit_winners(
            SequencePolicy((dict(candidate),)),
            (_document(),),
            lambda_zero="lambda-zero",
            rollouts_per_document=1,
            utility_artifact=_utility_artifact(),
            environment_hash="env-hash",
            count_reward=_count_reward(),
            cache=UtilityCache(tmp_path / "cache.jsonl"),
            remote_workers=1,
            reader_workers=1,
            score_batch=scorer,
            generator=torch.Generator().manual_seed(0),
        )


def test_exit_deduplicates_identical_candidates_before_scoring(tmp_path):
    from cloak.train.interactive_ranker import collect_exit_winners
    from cloak.train.utility_cache import UtilityCache

    reference = (("alpha", "alpha-level"), ("beta", "beta-placeholder"))
    candidate = (("alpha", "alpha-keep"), ("beta", "beta-level"))
    scorer, calls = _exit_scorer({reference: 0.5, candidate: 0.4})
    collection = collect_exit_winners(
        SequencePolicy((dict(candidate), dict(candidate))),
        (_document(),),
        lambda_zero="lambda-zero",
        rollouts_per_document=2,
        utility_artifact=_utility_artifact(),
        environment_hash="env-hash",
        count_reward=_count_reward(),
        cache=UtilityCache(tmp_path / "cache.jsonl"),
        remote_workers=1,
        reader_workers=1,
        score_batch=scorer,
        generator=torch.Generator().manual_seed(0),
    )

    assert len(collection.documents[0].candidates) == 1
    assert calls == [(False, [reference, candidate])]


def test_exit_cache_only_reports_exact_unique_initial_work_and_dispatches_nothing(
    tmp_path, monkeypatch,
):
    import cloak.train.roundtrip as roundtrip
    from cloak.train.interactive_ranker import CacheOnlyMissError, collect_exit_winners
    from cloak.train.utility_cache import UtilityCache, stable_hash

    artifact = _utility_artifact()
    artifact["documents"]["fixture/doc"]["assertion_ids"].append("a-context")
    artifact["assertions"]["a-context"] = {
        "assertion_id": "a-context",
        "doc_id": "fixture/doc",
        "family": "context",
        "status": "accepted",
        "weight": 1.0,
        "credit_routing": "residual",
        "policy_dependency_decision_ids": [],
    }
    artifact["artifact_hash"] = stable_hash({
        key: value for key, value in artifact.items() if key != "artifact_hash"
    })
    monkeypatch.setattr(roundtrip, "_validate_request_readers", lambda request: None)
    candidate = {"alpha": "alpha-keep", "beta": "beta-level"}
    document = replace(_document(), corpus="aci")
    scorer_calls = []

    def forbidden_score(*args, **kwargs):
        scorer_calls.append((args, kwargs))
        raise AssertionError("cache-only miss dispatched scoring")

    with pytest.raises(CacheOnlyMissError) as captured:
        collect_exit_winners(
            SequencePolicy((candidate, candidate)),
            (document,),
            lambda_zero="lambda-zero",
            rollouts_per_document=2,
            utility_artifact=artifact,
            environment_hash="env-hash",
            count_reward=_count_reward(),
            cache=UtilityCache(tmp_path / "cache.jsonl"),
            remote_workers=1,
            reader_workers=1,
            score_batch=forbidden_score,
            generator=torch.Generator().manual_seed(0),
            cache_only=True,
        )

    assert captured.value.phase == "initial"
    assert captured.value.remote_tasks == 2
    assert captured.value.context_reader_work_items == 2
    assert scorer_calls == []


def test_exit_cache_only_preflights_refresh_pairs_after_cached_selection(monkeypatch):
    import cloak.train.roundtrip as roundtrip
    from cloak.train.interactive_ranker import CacheOnlyMissError, collect_exit_winners
    from cloak.train.utility_cache import UtilityCache

    monkeypatch.setattr(roundtrip, "_validate_request_readers", lambda request: None)
    reference = (("alpha", "alpha-level"), ("beta", "beta-placeholder"))
    candidate = (("alpha", "alpha-keep"), ("beta", "beta-level"))
    scorer, calls = _exit_scorer({reference: 0.5, candidate: 0.8})

    class InitialOnlyCache:
        request_identity = staticmethod(UtilityCache.request_identity)

        def lookup(self, identity):
            return object() if not identity["reader_refresh"] else None

    with pytest.raises(CacheOnlyMissError) as captured:
        collect_exit_winners(
            SequencePolicy((dict(candidate),)),
            (replace(_document(), corpus="aci"),),
            lambda_zero="lambda-zero",
            rollouts_per_document=1,
            utility_artifact=_utility_artifact(),
            environment_hash="env-hash",
            count_reward=_count_reward(),
            cache=InitialOnlyCache(),
            remote_workers=1,
            reader_workers=1,
            score_batch=scorer,
            generator=torch.Generator().manual_seed(0),
            cache_only=True,
        )

    assert captured.value.phase == "reverification"
    assert captured.value.remote_tasks == 2
    assert captured.value.context_reader_work_items == 0
    assert calls == [(False, [reference, candidate])]


def test_exit_artifact_contains_only_ids_scores_and_pins(tmp_path):
    from cloak.train.interactive_ranker import (
        collect_exit_winners,
        write_exit_winners,
    )
    from cloak.train.utility_cache import UtilityCache

    reference = (("alpha", "alpha-level"), ("beta", "beta-placeholder"))
    candidate = (("alpha", "alpha-keep"), ("beta", "beta-level"))
    scorer, _ = _exit_scorer(
        {reference: 0.5, candidate: 0.8},
        {reference: 0.55, candidate: 0.75},
    )
    collection = collect_exit_winners(
        SequencePolicy((dict(candidate),)),
        (_document(),),
        lambda_zero="lambda-zero",
        rollouts_per_document=1,
        utility_artifact=_utility_artifact(),
        environment_hash="env-hash",
        count_reward=_count_reward(),
        cache=UtilityCache(tmp_path / "cache.jsonl"),
        remote_workers=1,
        reader_workers=1,
        score_batch=scorer,
        generator=torch.Generator().manual_seed(0),
    )
    out = tmp_path / "exit-winners.json"
    payload = write_exit_winners(
        out,
        collection,
        pins={
            "environment_hash": "env-hash",
            "count_state_hash": "count-hash",
            "utility_artifact_hash": "utility-hash",
            "policy_checkpoint_hash": "policy-hash",
        },
    )

    assert out.exists()
    assert payload["artifact_version"] == "ranker-v2-exit-winners-v1"
    assert payload["summary"] == {
        "document_count": 1, "candidate_count": 1, "winner_count": 1,
    }
    encoded = out.read_text()
    assert "Alpha met Beta" not in encoded
    assert "remote" not in encoded
    assert "final" not in encoded
    assert payload["artifact_hash"].startswith("sha256:")


@pytest.mark.parametrize(
    ("command", "extra"),
    [
        ("bc", ["--out-checkpoint", "bc.pt"]),
        (
            "exit-collect",
            ["--checkpoint", "bc.pt", "--out", "exit.json", "--rollouts", "2"],
        ),
    ],
)
def test_interactive_cli_has_only_task_subcommands_and_requires_all_artifact_paths(
    command, extra,
):
    import train_interactive_ranker

    parser = train_interactive_ranker.build_parser()
    common = [
        "--environment", "environment.json",
        "--count-state", "count.json",
        "--utility-artifact", "utility.json",
        "--utility-cache", "cache.jsonl",
    ]
    args = parser.parse_args([command, *common, *extra])
    assert args.command == command
    assert vars(args)["environment"] == "environment.json"
    assert vars(args)["count_state"] == "count.json"
    assert vars(args)["utility_artifact"] == "utility.json"
    assert vars(args)["utility_cache"] == "cache.jsonl"
    selected = parser.parse_args([
        command, *common, *extra, "--doc-id", "doc-a", "--doc-id", "doc-b",
    ])
    assert selected.doc_ids == ["doc-a", "doc-b"]

    with pytest.raises(SystemExit):
        parser.parse_args([command, *common[:-2], *extra])

    assert set(parser._subparsers._group_actions[0].choices) == {
        "bc", "exit-collect", "train",
    }


def test_interactive_cli_cache_only_miss_is_machine_readable_and_nonzero(
    monkeypatch, capsys,
):
    import train_interactive_ranker
    from cloak.train.interactive_ranker import CacheOnlyMissError

    def blocked(args):
        raise CacheOnlyMissError(
            phase="initial", remote_tasks=3, context_reader_work_items=7,
        )

    monkeypatch.setattr(train_interactive_ranker, "_run_exit_collect", blocked)
    with pytest.raises(SystemExit) as captured:
        train_interactive_ranker.main([
            "exit-collect",
            "--environment", "environment.json",
            "--count-state", "count.json",
            "--utility-artifact", "utility.json",
            "--utility-cache", "cache.jsonl",
            "--checkpoint", "bc.pt",
            "--out", "exit.json",
            "--rollouts", "2",
            "--cache-only",
        ])

    assert captured.value.code == 2
    assert capsys.readouterr().err.strip() == (
        "CACHE_ONLY_MISS phase=initial remote_tasks=3 "
        "context_reader_work_items=7"
    )


def test_train_cli_requires_every_frozen_artifact_output_and_runtime_control():
    import train_interactive_ranker

    parser = train_interactive_ranker.build_parser()
    args = parser.parse_args([
        "train",
        "--environment", "environment.json",
        "--count-state", "count.json",
        "--utility-artifact", "utility.json",
        "--utility-cache", "cache.jsonl",
        "--threshold-manifest", "threshold.json",
        "--lambda-menu", "menu.json",
        "--exit-winners", "exit.json",
        "--bc-checkpoint", "bc.pt",
        "--out-checkpoint", "conditional.pt",
        "--kl-reference-checkpoint", "reference.pt",
        "--epoch-reports", "epochs.jsonl",
        "--fixed-lambda-zero-control", "control.pt",
        "--max-docs", "3",
        "--max-epochs", "4",
        "--rollouts", "2",
        "--remote-workers", "2",
        "--reader-workers", "3",
        "--seed", "17",
        "--cache-only",
    ])

    assert args.command == "train"
    assert args.fixed_lambda_zero_control == "control.pt"
    assert args.max_docs == 3 and args.max_epochs == 4 and args.rollouts == 2
    assert args.cache_only is True
    train_actions = parser._subparsers._group_actions[0].choices["train"]._actions
    option_names = {option for action in train_actions for option in action.option_strings}
    assert not any(
        token in option
        for option in option_names
        for token in ("bypass", "skip-gate", "ignore-gate")
    )
    with pytest.raises(SystemExit):
        parser.parse_args([
            "train",
            "--environment", "environment.json",
            "--count-state", "count.json",
            "--utility-artifact", "utility.json",
            "--utility-cache", "cache.jsonl",
        ])


def test_train_cli_dispatches_train_and_preserves_cache_only_stop(monkeypatch, capsys):
    import train_interactive_ranker
    from cloak.train.interactive_ranker import CacheOnlyMissError

    def blocked(args):
        assert args.command == "train"
        raise CacheOnlyMissError(
            phase="counterfactual", remote_tasks=2,
            context_reader_work_items=5,
        )

    monkeypatch.setattr(train_interactive_ranker, "_run_train", blocked, raising=False)
    with pytest.raises(SystemExit) as captured:
        train_interactive_ranker.main([
            "train",
            "--environment", "environment.json",
            "--count-state", "count.json",
            "--utility-artifact", "utility.json",
            "--utility-cache", "cache.jsonl",
            "--threshold-manifest", "threshold.json",
            "--lambda-menu", "menu.json",
            "--exit-winners", "exit.json",
            "--bc-checkpoint", "bc.pt",
            "--out-checkpoint", "conditional.pt",
            "--kl-reference-checkpoint", "reference.pt",
            "--epoch-reports", "epochs.jsonl",
            "--fixed-lambda-zero-control", "control.pt",
            "--max-epochs", "1",
            "--rollouts", "2",
            "--cache-only",
        ])

    assert captured.value.code == 2
    assert capsys.readouterr().err.strip() == (
        "CACHE_ONLY_MISS phase=counterfactual remote_tasks=2 "
        "context_reader_work_items=5"
    )


def _objective_fixture():
    from cloak.train.count_reward import CountActionScore, CountReward
    from cloak.train.interactive_ranker import (
        ReplayedStep,
        ReplayedTrajectory,
    )
    from cloak.train.utility_credit import DocumentUtilityCredit

    probability = torch.tensor([0.75, 0.25], dtype=torch.float64)
    log_probs = probability.log().requires_grad_()
    trajectories = []
    for rollout_index in range(2):
        steps = tuple(
            ReplayedStep(
                decision_id=decision_id,
                legal_action_ids=(f"{decision_id}-selected", f"{decision_id}-other"),
                selected_action_id=f"{decision_id}-selected",
                log_prob=log_probs[0],
                entropy=-(log_probs.exp() * log_probs).sum(),
                log_probs=log_probs,
            )
            for decision_id in ("first", "second")
        )
        trajectories.append(ReplayedTrajectory(
            doc_id="fixture",
            lambda_profile="lambda-two",
            steps=steps,
        ))
    action_rows = {}
    decision_actions = {}
    for decision_id in ("first", "second"):
        ids = (f"{decision_id}-selected", f"{decision_id}-other")
        decision_actions[decision_id] = ids
        for action_id, score in zip(ids, (0.0, 1.0), strict=True):
            action_rows[action_id] = CountActionScore(
                action_id=action_id,
                decision_id=decision_id,
                runtime_type="TYPE",
                profile_id="profile",
                mode="level",
                count=1.0,
                score=score,
                grounding_status="certifying",
                source_family="fixture",
                evidence_ref="fixture",
            )
    credit = DocumentUtilityCredit(
        document_utility=(0.5, 0.5),
        linked_utility={"first": (0.5, 0.5)},
        residual_utility=(0.0, 0.0),
        provisional_advantage={
            (0, "first"): 1.0,
            (0, "second"): 2.0,
            (1, "first"): 0.0,
            (1, "second"): -1.0,
        },
        route={"first": "linked", "second": "document"},
    )
    return tuple(trajectories), CountReward(action_rows, decision_actions), credit


def test_hybrid_document_objective_matches_two_rollout_two_decision_equation():
    from cloak.train.interactive_ranker import (
        compose_hybrid_document_objective,
        hybrid_utility_loss,
    )

    replayed, count_reward, credit = _objective_fixture()
    counterfactual = {(0, "second"): replayed[0].steps[1].log_prob * 0.0 + 0.3}
    utility = hybrid_utility_loss(replayed, credit, counterfactual)
    reference = tuple(
        tuple(torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64)) for _ in row.steps)
        for row in replayed
    )

    objective = compose_hybrid_document_objective(
        replayed,
        utility_loss=utility,
        count_reward=count_reward,
        lambda_value=2.0,
        beta=0.1,
        eta=0.2,
        reference_log_probs=reference,
    )

    per_step_entropy = -(0.75 * torch.log(torch.tensor(0.75))
                         + 0.25 * torch.log(torch.tensor(0.25)))
    per_step_kl = (0.75 * torch.log(torch.tensor(1.5))
                   + 0.25 * torch.log(torch.tensor(0.5)))
    assert float(objective.utility.detach()) == pytest.approx(0.15)
    assert float(objective.count.detach()) == pytest.approx(-0.5)
    assert float(objective.entropy.detach()) == pytest.approx(2 * per_step_entropy)
    assert float(objective.kl.detach()) == pytest.approx(2 * per_step_kl)
    assert float(objective.total.detach()) == pytest.approx(
        0.15 - 0.5 - 0.1 * 2 * per_step_entropy + 0.2 * 2 * per_step_kl
    )
    assert objective.beta == 0.1
    assert objective.eta == 0.2


def test_count_gradient_survives_tied_utility_and_is_not_divided_twice():
    from cloak.train.interactive_ranker import (
        compose_hybrid_document_objective,
        hybrid_utility_loss,
    )
    from cloak.train.utility_credit import DocumentUtilityCredit

    replayed, count_reward, _ = _objective_fixture()
    tied = DocumentUtilityCredit(
        document_utility=(0.5, 0.5),
        linked_utility={"first": (0.5, 0.5)},
        residual_utility=(0.0, 0.0),
        provisional_advantage={
            (rollout, decision): 0.0
            for rollout in range(2)
            for decision in ("first", "second")
        },
        route={"first": "linked", "second": "document"},
    )
    utility = hybrid_utility_loss(replayed, tied, {})
    objective = compose_hybrid_document_objective(
        replayed,
        utility_loss=utility,
        count_reward=count_reward,
        lambda_value=2.0,
        beta=0.0,
        eta=0.0,
        reference_log_probs=None,
    )

    assert float(utility.detach()) == 0.0
    assert float(objective.count.detach()) == pytest.approx(-0.5)
    objective.total.backward()
    assert replayed[0].steps[0].log_probs.grad is not None
    assert torch.count_nonzero(replayed[0].steps[0].log_probs.grad) > 0


def test_absolute_weighted_mass_does_not_cancel_opposite_rollout_terms():
    from cloak.train.interactive_ranker import _utility_family_terms_and_mass
    from cloak.train.utility_credit import DocumentUtilityCredit

    replayed, _, _ = _objective_fixture()
    credit = DocumentUtilityCredit(
        document_utility=(1.0, 0.0),
        linked_utility={},
        residual_utility=(0.0, 0.0),
        provisional_advantage={
            (rollout, decision): (1.0 if rollout == 0 else -1.0)
            for rollout in range(2)
            for decision in ("first", "second")
        },
        route={"first": "document", "second": "document"},
    )

    losses, masses = _utility_family_terms_and_mass(replayed, credit, {})

    assert float(losses["fallback"].detach()) == pytest.approx(0.0)
    assert masses["fallback"] > 0.0


def test_seeded_latin_cycle_balances_every_document_and_records_exposure():
    from cloak.train.interactive_ranker import (
        build_latin_cycle_schedule,
        profile_exposure_report,
    )
    from cloak.train.ranker import LambdaProfile

    documents = tuple(
        replace(_document(), doc_id=f"fixture/{index}", corpus=("a" if index < 2 else "b"))
        for index in range(4)
    )
    profiles = tuple(
        LambdaProfile(name, value)
        for name, value in (("zero", 0.0), ("middle", 1.0), ("high", 3.0))
    )
    schedule = build_latin_cycle_schedule(documents, profiles, seed=19)
    assignments = {
        document.doc_id: tuple(
            schedule.profile_for(document.doc_id, epoch).name
            for epoch in range(len(profiles))
        )
        for document in documents
    }

    assert all(set(values) == {profile.name for profile in profiles}
               for values in assignments.values())
    assert schedule == build_latin_cycle_schedule(documents, profiles, seed=19)
    assert schedule != build_latin_cycle_schedule(documents, profiles, seed=20)
    report = profile_exposure_report(documents, schedule, range(len(profiles)))
    assert all(
        count == 1
        for profiles_by_document in report["by_document"].values()
        for count in profiles_by_document.values()
    )
    assert sum(report["by_profile"].values()) == len(documents) * len(profiles)
    assert set(report["by_corpus"]) == {"a", "b"}
    assert set(report["by_type"]) == {"TYPE"}


def test_one_hybrid_optimizer_step_samples_without_graph_and_reports_all_families(
    tmp_path,
):
    import copy

    from cloak.train.interactive_ranker import train_hybrid_document_group
    from cloak.train.ranker import LambdaProfile
    from cloak.train.utility_cache import UtilityCache

    policy = StubPolicy()
    reference = copy.deepcopy(policy)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    initial = policy.scores.detach().clone()
    score_calls = []

    def scorer(requests, **kwargs):
        score_calls.append((len(requests), kwargs["reader_refresh"]))
        kwargs["cache"].last_batch_metrics = {
            "cache_hits": 1, "transport_calls": 2, "reader_work_items": 3,
        }
        return [_utility_result(request.action_vector, 0.5) for request in requests]

    def scheduler(documents, trajectories, replayed, **kwargs):
        assert set(documents) == {"fixture/doc"}
        assert len(trajectories) == len(replayed) == 2
        assert kwargs["budget"] == 5
        assert all(row.lambda_profile.name == "lambda-two" for row in trajectories)
        return (), {
            "budget": 5,
            "uniform_allocation": 1,
            "priority_allocation": 4,
            "cache_hits": 0,
            "delta_u": {"count": 0},
            "skip_reasons": {},
        }

    def executor(requests, documents, trajectories, replayed, **kwargs):
        assert requests == ()
        kwargs["cache"].last_batch_metrics = {
            "cache_hits": 2, "transport_calls": 3, "reader_work_items": 4,
        }
        return {}, dict(kwargs["scheduler_diagnostics"])

    result = train_hybrid_document_group(
        policy,
        reference,
        _document(),
        LambdaProfile("lambda-two", 2.0),
        rollouts=2,
        utility_artifact=_utility_artifact(),
        environment_hash="env-hash",
        count_reward=_count_reward(),
        cache=UtilityCache(tmp_path / "cache.jsonl"),
        optimizer=optimizer,
        beta=0.01,
        eta=0.0,
        counterfactual_budget=5,
        endpoint_budget=1,
        pair_history={},
        seed=7,
        current_round=0,
        remote_workers=1,
        reader_workers=1,
        generator=torch.Generator().manual_seed(3),
        score_batch=scorer,
        scheduler=scheduler,
        counterfactual_executor=executor,
    )

    assert not torch.equal(policy.scores.detach(), initial)
    assert policy.grad_modes[:4] == [False, False, False, False]
    assert all(policy.grad_modes[index] for index in range(4, len(policy.grad_modes)))
    assert score_calls == [(2, False)]
    assert set(result.gradient_norms) == {
        "linked", "residual", "fallback", "counterfactual",
        "count", "entropy", "KL",
    }
    assert set(result.absolute_weighted_mass) == set(result.gradient_norms)
    assert result.absolute_weighted_mass["count"] > 0.0
    assert result.rollout_count == 2
    assert result.profile_name == "lambda-two"
    assert result.scheduler_diagnostics["budget"] == 5
    assert result.cache_metrics["requested_rollouts"] == 2
    assert result.cache_metrics["cache_hits"] == 3
    assert result.cache_metrics["transport_calls"] == 5
    assert result.cache_metrics["reader_work_items"] == 7
    assert set(result.cache_metrics["stages"]) == {"trajectory", "counterfactual"}
    assert result.action_modes
    assert result.runtime_type_exposure == {"TYPE": 2}
    assert result.runtime_type_metrics["TYPE"]["exposure"] == 2
    assert result.runtime_type_metrics["TYPE"]["action_modes"]
    assert result.runtime_type_metrics["TYPE"]["count_score"] >= 0.0


def test_epoch_report_keeps_scheduler_budget_separate_from_reward_magnitude(tmp_path):
    from cloak.train.interactive_ranker import DocumentTrainingResult, build_epoch_report

    group = DocumentTrainingResult(
        doc_id="fixture/doc",
        corpus="fixture",
        profile_name="lambda-two",
        rollout_count=2,
        loss=1.25,
        utility=0.5,
        count_score=0.7,
        entropy=0.4,
        collision_count=1,
        action_modes={"level": 2, "placeholder": 2},
        runtime_type_exposure={"TYPE": 2},
        gradient_norms={name: 0.1 for name in (
            "linked", "residual", "fallback", "counterfactual",
            "count", "entropy", "KL",
        )},
        absolute_weighted_mass={name: 0.2 for name in (
            "linked", "residual", "fallback", "counterfactual",
            "count", "entropy", "KL",
        )},
        scheduler_diagnostics={"budget": 5, "delta_u": {"mean_abs": 0.8}},
        cache_metrics={"cache_hits": 3, "remote_tasks": 1},
        runtime_type_metrics={
            "TYPE": {
                "exposure": 2,
                "utility": 0.5,
                "count_score": 0.7,
                "entropy": 0.4,
                "collisions": 1,
                "action_modes": {"level": 2, "placeholder": 2},
            },
        },
    )

    report = build_epoch_report(4, (group,))

    assert report["epoch"] == 4
    assert report["term_families"]["counterfactual"] == {
        "detached_gradient_norm": pytest.approx(0.1),
        "absolute_weighted_mass": pytest.approx(0.2),
    }
    assert report["scheduler"]["budget"] == 5
    assert "delta_u" in report["scheduler"]
    assert report["cache"] == {"cache_hits": 3, "remote_tasks": 1}
    assert report["profiles"]["lambda-two"]["utility"] == pytest.approx(0.5)
    assert report["corpora"]["fixture"]["count_score"] == pytest.approx(0.7)
    assert report["runtime_types"]["TYPE"] == {
        "exposure": 2,
        "utility": pytest.approx(0.5),
        "count_score": pytest.approx(0.7),
        "entropy": pytest.approx(0.4),
        "collisions": 1,
        "action_modes": {"level": 2, "placeholder": 2},
    }


def test_failed_counterfactual_execution_does_not_advance_pair_history(tmp_path):
    import copy

    from cloak.train.counterfactuals import CounterfactualRequest
    from cloak.train.interactive_ranker import train_hybrid_document_group
    from cloak.train.ranker import LambdaProfile
    from cloak.train.utility_cache import UtilityCache

    policy = StubPolicy()
    history = {}

    def scorer(requests, **kwargs):
        return [_utility_result(request.action_vector, 0.5) for request in requests]

    def scheduler(documents, trajectories, replayed, **kwargs):
        del documents, replayed, kwargs
        step = trajectories[0].steps[0]
        alternative = next(
            action_id for action_id in step.legal_action_ids
            if action_id != step.selected_action_id
        )
        return (CounterfactualRequest(
            doc_id=trajectories[0].doc_id,
            rollout_index=0,
            decision_id=step.decision_id,
            selected_action_id=step.selected_action_id,
            alternative_action_id=alternative,
            direction="fixture",
            priority_tier=0,
        ),), {"budget": 5}

    def failed_executor(*args, **kwargs):
        raise RuntimeError("fixture scoring failure")

    with pytest.raises(RuntimeError, match="fixture scoring failure"):
        train_hybrid_document_group(
            policy,
            copy.deepcopy(policy),
            _document(),
            LambdaProfile("lambda-two", 2.0),
            rollouts=2,
            utility_artifact=_utility_artifact(),
            environment_hash="env-hash",
            count_reward=_count_reward(),
            cache=UtilityCache(tmp_path / "cache.jsonl"),
            optimizer=torch.optim.SGD(policy.parameters(), lr=0.1),
            beta=0.0,
            eta=0.0,
            counterfactual_budget=5,
            endpoint_budget=1,
            pair_history=history,
            seed=0,
            current_round=4,
            remote_workers=1,
            reader_workers=1,
            generator=torch.Generator().manual_seed(2),
            score_batch=scorer,
            scheduler=scheduler,
            counterfactual_executor=failed_executor,
        )

    assert history == {}


def test_hybrid_checkpoint_round_trip_restores_optimizer_rng_schedule_and_pins(
    tmp_path,
):
    import random

    from cloak.train.interactive_ranker import (
        build_latin_cycle_schedule,
        load_hybrid_checkpoint,
        save_hybrid_checkpoint,
    )
    from cloak.train.ranker import LambdaProfile

    policy = StubPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    profiles = (LambdaProfile("zero", 0.0), LambdaProfile("high", 2.0))
    schedule = build_latin_cycle_schedule((_document(),), profiles, seed=11)
    generator = torch.Generator().manual_seed(23)
    pins = {
        "environment_hash": "env",
        "utility_artifact_hash": "utility",
        "count_state_hash": "count",
        "lambda_menu_hash": "menu",
        "threshold_manifest_hash": "threshold",
        "exit_winners_hash": "exit",
        "bc_checkpoint_hash": "bc",
    }
    architecture_pin = "sha256:architecture"
    path = tmp_path / "checkpoint.pt"
    save_hybrid_checkpoint(
        path,
        policy=policy,
        optimizer=optimizer,
        epoch=3,
        generator=generator,
        schedule=schedule,
        artifact_pins=pins,
        architecture_pin=architecture_pin,
        cache_paths={"utility": "cache.jsonl"},
        code_revision="revision",
        training_config={"learning_rate": 0.01, "beta": 0.1, "eta": 0.2},
        pair_history={("doc", 0, "decision", "a", "b"): 2},
        kl_enabled=True,
        epoch_reports=({"epoch": 3},),
    )
    saved_scores = policy.scores.detach().clone()
    saved_generator_state = generator.get_state().clone()
    with torch.no_grad():
        policy.scores.add_(10.0)
    generator.manual_seed(99)
    random.seed(99)

    resumed = load_hybrid_checkpoint(
        path,
        policy=policy,
        optimizer=optimizer,
        generator=generator,
        expected_artifact_pins=pins,
        expected_architecture_pin=architecture_pin,
        expected_cache_paths={"utility": "cache.jsonl"},
        expected_code_revision="revision",
        expected_training_config={"learning_rate": 0.01, "beta": 0.1, "eta": 0.2},
    )

    assert torch.equal(policy.scores.detach(), saved_scores)
    assert torch.equal(generator.get_state(), saved_generator_state)
    assert resumed["epoch"] == 3
    assert resumed["schedule"] == schedule
    assert resumed["kl_enabled"] is True
    assert resumed["pair_history"] == {("doc", 0, "decision", "a", "b"): 2}
    assert resumed["epoch_reports"] == ({"epoch": 3},)
    assert resumed["training_config"] == {
        "learning_rate": 0.01, "beta": 0.1, "eta": 0.2,
    }


def test_hybrid_checkpoint_refuses_any_pin_mismatch_before_loading_state(tmp_path):
    from cloak.train.interactive_ranker import (
        build_latin_cycle_schedule,
        load_hybrid_checkpoint,
        save_hybrid_checkpoint,
    )
    from cloak.train.ranker import LambdaProfile

    policy = StubPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    generator = torch.Generator().manual_seed(1)
    schedule = build_latin_cycle_schedule(
        (_document(),), (LambdaProfile("zero", 0.0),), seed=0,
    )
    pins = {
        "environment_hash": "env", "utility_artifact_hash": "utility",
        "count_state_hash": "count", "lambda_menu_hash": "menu",
        "threshold_manifest_hash": "threshold", "exit_winners_hash": "exit",
        "bc_checkpoint_hash": "bc",
    }
    path = tmp_path / "checkpoint.pt"
    save_hybrid_checkpoint(
        path,
        policy=policy,
        optimizer=optimizer,
        epoch=0,
        generator=generator,
        schedule=schedule,
        artifact_pins=pins,
        architecture_pin="architecture",
        cache_paths={"utility": "cache"},
        code_revision="revision",
        training_config={"learning_rate": 0.1, "beta": 0.0, "eta": 0.0},
        pair_history={},
        kl_enabled=False,
        epoch_reports=(),
    )
    before = policy.scores.detach().clone()
    mismatched = dict(pins, lambda_menu_hash="different")

    with pytest.raises(ValueError, match="artifact pins"):
        load_hybrid_checkpoint(
            path,
            policy=policy,
            optimizer=optimizer,
            generator=generator,
            expected_artifact_pins=mismatched,
            expected_architecture_pin="architecture",
            expected_cache_paths={"utility": "cache"},
            expected_code_revision="revision",
            expected_training_config={"learning_rate": 0.1, "beta": 0.0, "eta": 0.0},
        )
    assert torch.equal(policy.scores.detach(), before)


def test_kl_enables_only_when_frozen_collapse_threshold_fires():
    from cloak.train.interactive_ranker import collapse_rule_fires

    manifest = {"feasibility_gates": {"min_adjacent_winner_change": 0.2}}
    assert collapse_rule_fires(
        manifest, {"conditional_responsiveness": {"adjacent_winner_change": 0.1}},
    ) is True
    assert collapse_rule_fires(
        manifest, {"conditional_responsiveness": {"adjacent_winner_change": 0.3}},
    ) is False
    with pytest.raises(ValueError, match="responsiveness"):
        collapse_rule_fires(manifest, {})


def test_hybrid_training_loop_uses_latin_profiles_and_enables_kl_after_block():
    from cloak.train.interactive_ranker import (
        DocumentTrainingResult,
        build_latin_cycle_schedule,
        train_hybrid_policy,
    )
    from cloak.train.ranker import LambdaProfile

    documents = tuple(
        replace(_document(), doc_id=f"fixture/{index}") for index in range(3)
    )
    profiles = (
        LambdaProfile("zero", 0.0),
        LambdaProfile("middle", 1.0),
        LambdaProfile("high", 2.0),
    )
    schedule = build_latin_cycle_schedule(documents, profiles, seed=13)
    calls = []
    checkpoints = []

    def group_trainer(policy, reference_policy, document, profile, **kwargs):
        del policy, reference_policy
        calls.append((kwargs["current_round"], document.doc_id, profile.name, kwargs["eta"]))
        families = {
            name: 0.0 for name in (
                "linked", "residual", "fallback", "counterfactual",
                "count", "entropy", "KL",
            )
        }
        return DocumentTrainingResult(
            doc_id=document.doc_id,
            corpus=document.corpus,
            profile_name=profile.name,
            rollout_count=2,
            loss=0.0,
            utility=0.5,
            count_score=profile.value / 2.0,
            entropy=0.4,
            collision_count=0,
            action_modes={"level": 2},
            runtime_type_exposure={"TYPE": 2},
            gradient_norms=families,
            absolute_weighted_mass=families,
            scheduler_diagnostics={
                "budget": 5, "uniform_allocation": 1,
                "priority_allocation": 4, "delta_u": {"count": 0},
            },
            cache_metrics={"cache_hits": 0},
            action_vector_hashes=("sha256:same", "sha256:same"),
        )

    result = train_hybrid_policy(
        policy=object(),
        reference_policy=object(),
        documents=documents,
        profiles=profiles,
        schedule=schedule,
        optimizer=object(),
        utility_artifact={},
        environment_hash="env",
        count_reward=object(),
        cache=object(),
        threshold_manifest={
            "feasibility_gates": {"min_adjacent_winner_change": 0.2},
        },
        max_epochs=4,
        rollouts=2,
        beta=0.01,
        eta=0.3,
        counterfactual_budget=5,
        endpoint_budget=1,
        pair_history={},
        seed=7,
        remote_workers=1,
        reader_workers=1,
        generator=torch.Generator().manual_seed(4),
        group_trainer=group_trainer,
        epoch_callback=lambda epoch, reports, history, enabled: checkpoints.append(
            (epoch, len(reports), dict(history), enabled)
        ),
    )

    first_block = calls[: len(documents) * len(profiles)]
    for document in documents:
        assert {
            profile for _, doc_id, profile, _ in first_block
            if doc_id == document.doc_id
        } == {profile.name for profile in profiles}
    assert all(eta == 0.0 for *_, eta in first_block)
    assert all(eta == 0.3 for *_, eta in calls[len(first_block):])
    assert result.kl_enabled is True
    assert len(result.epoch_reports) == 4
    assert result.epoch_reports[2]["conditional_responsiveness"] == {
        "eligible_adjacent_pairs": 6,
        "changed_adjacent_pairs": 0,
        "adjacent_winner_change": 0.0,
    }
    assert result.epoch_reports[0]["exposure"]["by_profile"]
    assert [row[:2] for row in checkpoints] == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert checkpoints[2][3] is True
