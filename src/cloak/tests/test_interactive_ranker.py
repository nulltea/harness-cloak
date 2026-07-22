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

    assert set(parser._subparsers._group_actions[0].choices) == {"bc", "exit-collect"}


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
