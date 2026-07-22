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
