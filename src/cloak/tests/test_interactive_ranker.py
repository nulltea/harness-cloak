import math
from dataclasses import FrozenInstanceError, fields, replace
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from cloak.ranker.environment import (
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


class SemanticGradientPolicy(torch.nn.Module):
    policy_architecture = "semantic-v1"
    encoder_revision = "stub-encoder-revision"
    context_mode = "full-candidate-attention"
    history_mode = "selected-cross-attention"
    feature_schema = (
        "utility_relation",
        "candidate_context",
        "context_relation_interaction",
        "action_mode",
        "runtime_type",
        "selected_history",
    )
    controller_transform = "log1p-over-log1p-max-v1"

    def __init__(self):
        super().__init__()
        self.utility_scores = torch.nn.Parameter(torch.tensor([0.0, 2.0, 0.0]))
        self.history_scale = torch.nn.Parameter(torch.tensor(0.25))
        self.privacy_scale = torch.nn.Parameter(torch.tensor(0.5))
        self.alpha_raw = torch.nn.Parameter(
            torch.tensor(math.log(math.expm1(1.0)))
        )
        self.max_lambda = 2.0
        self.distribution_calls = 0

    @property
    def alpha(self):
        return torch.nn.functional.softplus(self.alpha_raw)

    def utility_parameters(self):
        return (self.utility_scores,)

    def history_parameters(self):
        return (self.history_scale,)

    def privacy_parameters(self):
        return (self.privacy_scale,)

    def begin_document(self, document, profile):
        return ()

    def distribution(self, state, decision, legal_action_ids, profile):
        self.distribution_calls += 1
        mode_indices = {
            action.action_id: {
                "level": 0,
                "keep": 1,
                "placeholder": 2,
            }[action.mode]
            for action in decision.actions
        }
        indices = torch.tensor([mode_indices[row] for row in legal_action_ids])
        history_direction = torch.tensor([0.0, 1.0, -1.0])[indices]
        utility = (
            self.utility_scores[indices]
            + len(state) * self.history_scale * history_direction
        )
        predicted = torch.tensor([0.5, 0.0, 1.0])[indices]
        mu = predicted + self.privacy_scale * torch.tensor([0.2, -0.1, 0.1])[indices]
        sigma = torch.full_like(mu, 0.25)
        if float(profile.value) == 0.0:
            combined = utility
            count_combined = utility.detach()
        else:
            magnitude = math.log1p(float(profile.value)) / math.log1p(self.max_lambda)
            controller = self.alpha * magnitude * predicted.detach()
            combined = utility + controller
            count_combined = utility.detach() + controller
        return SimpleNamespace(
            action_ids=tuple(legal_action_ids),
            utility_logits=utility,
            mu_log_count=mu,
            sigma_log_count=sigma,
            predicted_privacy=predicted,
            combined_logits=combined,
            log_probs=torch.log_softmax(combined, dim=0),
            count_log_probs=torch.log_softmax(count_combined, dim=0),
        )

    def log_probs(self, state, decision, legal_action_ids, profile):
        return self.distribution(
            state, decision, legal_action_ids, profile
        ).log_probs

    def advance(self, state, decision, action_id):
        return (*state, action_id)


def _profile_targets():
    from cloak.ranker.profile_count import ProfileActionTarget, ProfileCountTargets

    rows = {}
    decision_actions = {}
    for decision in _document().policy_decisions:
        decision_actions[decision.decision_id] = tuple(
            action.action_id for action in decision.actions
        )
        for action in decision.actions:
            score = {"level": 0.5, "keep": 0.0, "placeholder": 1.0}[action.mode]
            rows[action.action_id] = ProfileActionTarget(
                decision_id=decision.decision_id,
                action_id=action.action_id,
                profile_id=str(decision.profile_id),
                runtime_type=decision.runtime_type,
                mode=action.mode,
                log_count=0.5 if action.mode == "level" else None,
                profile_score=score,
                grounding_status="grounded" if action.mode == "level" else None,
                source_family="fixture" if action.mode == "level" else None,
            )
    return ProfileCountTargets(
        rows,
        decision_actions,
        {decision_id: True for decision_id in decision_actions},
    )


def test_public_trajectory_dataclasses_are_exact_frozen_and_opaque():
    from cloak.ranker.interactive import SampledStep, SampledTrajectory

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
    from cloak.ranker.environment import legal_action_ids

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
    from cloak.ranker.interactive import sample_trajectory

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
    from cloak.ranker.environment import assemble_action_vector

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
    from cloak.ranker.environment import assemble_action_vector

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
    from cloak.ranker.interactive import sample_trajectory

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
    from cloak.ranker.interactive import replay_trajectory, sample_trajectory

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


def test_semantic_replay_captures_both_distributions_from_one_forward_per_step():
    from cloak.ranker.interactive import (
        ReplayedStep,
        behavior_clone_trajectory,
        replay_trajectory,
    )
    from cloak.ranker.environment import LambdaProfile

    assert [field.name for field in fields(ReplayedStep)] == [
        "decision_id",
        "selected_action_id",
        "legal_action_ids",
        "log_prob",
        "log_probs",
        "count_log_probs",
        "utility_logits",
        "predicted_privacy",
        "entropy",
    ]
    policy = SemanticGradientPolicy()
    profile = LambdaProfile("high", 2.0)
    trajectory = behavior_clone_trajectory(_document(), profile)

    replayed = replay_trajectory(policy, _document(), trajectory, profile)

    assert policy.distribution_calls == len(trajectory.steps)
    assert all(step.count_log_probs.requires_grad for step in replayed.steps)
    assert all(step.utility_logits.requires_grad for step in replayed.steps)
    assert all(step.predicted_privacy.shape == step.log_probs.shape for step in replayed.steps)
    assert all(
        torch.equal(step.log_prob, step.log_probs[
            step.legal_action_ids.index(step.selected_action_id)
        ])
        for step in replayed.steps
    )


def test_semantic_replay_fails_closed_when_policy_only_returns_scalar_log_probs():
    from cloak.ranker.interactive import (
        behavior_clone_trajectory,
        replay_trajectory,
    )
    from cloak.ranker.environment import LambdaProfile

    class IncompleteSemanticPolicy(StubPolicy):
        policy_architecture = "semantic-v1"

    profile = LambdaProfile("high", 2.0)
    trajectory = behavior_clone_trajectory(_document(), profile)
    with pytest.raises(ValueError, match="complete distribution"):
        replay_trajectory(
            IncompleteSemanticPolicy(), _document(), trajectory, profile
        )


def _two_action_semantic_step(policy, profile):
    from cloak.ranker.interactive import ReplayedStep

    decision = _document().policy_decisions[0]
    legal = ("alpha-keep", "alpha-placeholder")
    row = policy.distribution((), decision, legal, profile)
    selected_index = legal.index("alpha-keep")
    return row, ReplayedStep(
        decision_id=decision.decision_id,
        selected_action_id="alpha-keep",
        legal_action_ids=legal,
        log_prob=row.log_probs[selected_index],
        log_probs=row.log_probs,
        count_log_probs=row.count_log_probs,
        utility_logits=row.utility_logits,
        predicted_privacy=row.predicted_privacy,
        entropy=-(row.log_probs.exp() * row.log_probs).sum(),
    )


def test_semantic_expected_profile_count_loss_uses_detached_count_distribution():
    from cloak.ranker.interactive import expected_profile_count_loss
    from cloak.ranker.environment import LambdaProfile

    policy = SemanticGradientPolicy()
    profile = LambdaProfile("high", 2.0)
    row, step = _two_action_semantic_step(policy, profile)

    loss = expected_profile_count_loss(
        (step,),
        _profile_targets(),
        lambda_value=profile.value,
        decision_count=1,
        rollout_count=1,
    )

    exact = torch.tensor([0.0, 1.0], dtype=row.count_log_probs.dtype)
    assert torch.equal(loss, -2.0 * torch.sum(row.count_log_probs.exp() * exact))


def test_semantic_hybrid_objective_uses_profile_targets_not_legacy_count_reward():
    from cloak.ranker.interactive import (
        ReplayedTrajectory,
        compose_hybrid_document_objective,
        expected_profile_count_loss,
    )
    from cloak.ranker.environment import LambdaProfile

    policy = SemanticGradientPolicy()
    profile = LambdaProfile("high", 2.0)
    _, step = _two_action_semantic_step(policy, profile)
    replayed = (ReplayedTrajectory(
        doc_id=_document().doc_id,
        lambda_profile=profile,
        steps=(step,),
    ),)
    utility = -step.log_prob

    objective = compose_hybrid_document_objective(
        replayed,
        utility_loss=utility,
        profile_targets=_profile_targets(),
        lambda_value=profile.value,
        beta=0.0,
        eta=0.0,
        reference_log_probs=None,
    )

    assert torch.equal(
        objective.count,
        expected_profile_count_loss(
            (step,), _profile_targets(), profile.value, 1, 1,
        ),
    )


def test_semantic_utility_and_exact_count_alpha_gradients_oppose_with_isolation():
    from cloak.ranker.interactive import (
        _gradient_norm,
        expected_profile_count_loss,
    )
    from cloak.ranker.environment import LambdaProfile

    policy = SemanticGradientPolicy()
    profile = LambdaProfile("high", 2.0)
    row, step = _two_action_semantic_step(policy, profile)
    utility_loss = -row.log_probs[0]
    count_loss = expected_profile_count_loss(
        (step,),
        _profile_targets(),
        lambda_value=profile.value,
        decision_count=1,
        rollout_count=1,
    )

    assert _gradient_norm(count_loss, policy.utility_parameters()) == 0.0
    assert _gradient_norm(count_loss, policy.history_parameters()) == 0.0
    assert _gradient_norm(utility_loss, policy.privacy_parameters()) == 0.0
    assert _gradient_norm(count_loss, (policy.alpha_raw,)) > 0.0
    assert _gradient_norm(utility_loss, (policy.alpha_raw,)) > 0.0
    alpha_count_grad = torch.autograd.grad(
        count_loss, policy.alpha_raw, retain_graph=True
    )[0]
    alpha_utility_grad = torch.autograd.grad(
        utility_loss, policy.alpha_raw, retain_graph=True
    )[0]
    assert torch.sign(alpha_count_grad) != torch.sign(alpha_utility_grad)
    assert alpha_count_grad < 0.0
    assert alpha_utility_grad > 0.0


def test_replay_raises_when_sampled_menu_or_claimed_state_differs():
    from cloak.ranker.interactive import replay_trajectory, sample_trajectory

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
    import cloak.ranker.interactive as interactive

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
    from cloak.ranker.interactive import ReplayedStep, ReplayedTrajectory

    trajectories = []
    for rollout_index, values in enumerate(log_probabilities):
        steps = tuple(
            ReplayedStep(
                decision_id=decision_id,
                selected_action_id=f"{decision_id}-selected",
                legal_action_ids=(f"{decision_id}-selected",),
                log_prob=log_prob,
                log_probs=log_prob.reshape(1),
                count_log_probs=log_prob.reshape(1),
                utility_logits=log_prob.reshape(1),
                predicted_privacy=torch.zeros(1),
                entropy=torch.zeros((), requires_grad=True),
            )
            for decision_id, log_prob in zip(("p1", "p2"), values, strict=True)
        )
        trajectories.append(ReplayedTrajectory(
            doc_id="d1", lambda_profile=f"profile-{rollout_index}", steps=steps,
        ))
    return tuple(trajectories)


def test_provisional_utility_loss_has_one_term_per_pair_and_divides_only_by_rollouts():
    from cloak.ranker.interactive import provisional_utility_loss
    from cloak.reward.utility_credit import DocumentUtilityCredit

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
    from cloak.ranker.interactive import provisional_utility_loss
    from cloak.reward.utility_credit import DocumentUtilityCredit

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


def test_provisional_utility_loss_rejects_missing_but_tolerates_extra_credit_pairs():
    from cloak.ranker.interactive import provisional_utility_loss
    from cloak.reward.utility_credit import DocumentUtilityCredit

    logs = tuple(torch.tensor(-0.1, requires_grad=True) for _ in range(4))
    missing = DocumentUtilityCredit(
        document_utility=(0.0, 0.0), linked_utility={}, residual_utility=(0.0, 0.0),
        provisional_advantage={(0, "p1"): 0.0}, route={"p1": "linked"},
    )
    with pytest.raises(ValueError, match="credit lacks replayed trajectory pairs"):
        provisional_utility_loss(_replayed((logs[:2], logs[2:])), missing)

    # Extra pairs (load-time-demoted decisions the walk never visits) are unused.
    extra = DocumentUtilityCredit(
        document_utility=(0.0, 0.0), linked_utility={}, residual_utility=(0.0, 0.0),
        provisional_advantage={
            (0, "p1"): 1.0, (0, "p2"): 2.0, (0, "demoted"): 9.0,
            (1, "p1"): -1.0, (1, "p2"): -2.0, (1, "demoted"): -9.0,
        },
        route={"p1": "linked", "p2": "document", "demoted": "document"},
    )
    loss = provisional_utility_loss(_replayed((logs[:2], logs[2:])), extra)
    expected = -(1.0 * -0.1 + 2.0 * -0.1 + -1.0 * -0.1 + -2.0 * -0.1) / 2
    assert torch.isclose(loss, torch.tensor(expected))


def test_hybrid_utility_loss_substitutes_pair_terms_in_place_and_divides_once():
    from cloak.ranker.interactive import hybrid_utility_loss
    from cloak.reward.utility_credit import DocumentUtilityCredit

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
    from cloak.ranker.interactive import hybrid_utility_loss
    from cloak.reward.utility_credit import DocumentUtilityCredit

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
    from cloak.ranker.profile_count import ProfileActionTarget, ProfileCountTargets

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
            rows[action.action_id] = ProfileActionTarget(
                decision_id=decision.decision_id,
                action_id=action.action_id,
                profile_id=str(decision.profile_id),
                runtime_type=decision.runtime_type,
                mode=action.mode,
                log_count=None,
                profile_score=score,
                grounding_status=None,
                source_family=None,
            )
    rows["other-placeholder"] = ProfileActionTarget(
        decision_id="other-document-decision",
        action_id="other-placeholder",
        profile_id="profile-other",
        runtime_type="TYPE",
        mode="placeholder",
        log_count=None,
        profile_score=1.0,
        grounding_status=None,
        source_family=None,
    )
    decision_actions["other-document-decision"] = ("other-placeholder",)
    return ProfileCountTargets(
        rows,
        decision_actions,
        {decision_id: True for decision_id in decision_actions},
    )


def _utility_artifact():
    from cloak.reward.utility_cache import stable_hash

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
                "assertion_ids": ["a-delivered", "a-linked"],
                "policy_decision_ids": ["alpha", "beta"],
                # Total weight 2.0, of which the policy mass is a-linked's 1.0.
                "utility_weight_denominator": 2.0,
            },
        },
        "assertions": {
            # Monitoring mass: no policy dependency, so it is reported per component
            # but excluded from the credited utility.
            "a-delivered": {
                "assertion_id": "a-delivered",
                "doc_id": "fixture/doc",
                "family": "delivered",
                "status": "accepted",
                "weight": 1.0,
                "credit_routing": "residual",
                "policy_dependency_decision_ids": [],
            },
            "a-linked": {
                "assertion_id": "a-linked",
                "doc_id": "fixture/doc",
                "family": "delivered",
                "status": "accepted",
                "weight": 1.0,
                "credit_routing": "linked",
                "policy_dependency_decision_ids": ["alpha"],
            },
        },
    }
    artifact["artifact_hash"] = stable_hash(artifact)
    return artifact


def _utility_result(action_vector, utility, monitoring=None):
    """Both components score `utility` by default, so the credited (policy-only)
    utility, the all-assertion cache parity value, and `utility` all coincide."""
    from cloak.reward.utility_cache import make_result

    monitoring = utility if monitoring is None else monitoring
    return make_result(
        doc_id="fixture/doc",
        action_vector=action_vector,
        doc_p="rendered",
        out_p="remote",
        out_final="final",
        component_scores={"a-delivered": monitoring, "a-linked": utility},
        utility=(monitoring + utility) / 2.0,
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
    from cloak.ranker.interactive import (
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
    from cloak.ranker.interactive import behavior_clone_trajectory

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
    from cloak.ranker.interactive import behavior_clone

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


def test_semantic_behavior_clone_updates_only_utility_and_history_at_lambda_zero():
    from cloak.ranker.interactive import behavior_clone
    from cloak.ranker.environment import LambdaProfile

    policy = SemanticGradientPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.2)
    utility_before = policy.utility_scores.detach().clone()
    history_before = policy.history_scale.detach().clone()
    privacy_before = policy.privacy_scale.detach().clone()
    alpha_before = policy.alpha_raw.detach().clone()

    behavior_clone(
        policy,
        (_document(),),
        lambda_zero=LambdaProfile("zero", 0.0),
        optimizer=optimizer,
        epochs=1,
    )

    assert not torch.equal(policy.utility_scores.detach(), utility_before)
    assert not torch.equal(policy.history_scale.detach(), history_before)
    assert torch.equal(policy.privacy_scale.detach(), privacy_before)
    assert torch.equal(policy.alpha_raw.detach(), alpha_before)


def test_semantic_warm_start_clones_each_verified_winner_once_at_lambda_zero():
    from cloak.ranker.interactive import initialize_hybrid_warm_start
    from cloak.ranker.environment import LambdaProfile

    policy = SemanticGradientPolicy()
    bc_state = {
        name: value.detach().clone()
        for name, value in policy.state_dict().items()
    }
    privacy_before = policy.privacy_scale.detach().clone()
    alpha_before = policy.alpha_raw.detach().clone()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    winner = {
        "artifact_version": "ranker-v2-exit-winners-v1",
        "documents": [{
            "doc_id": _document().doc_id,
            "verification_status": "verified",
            "winner": {
                "action_vector": {
                    "alpha": "alpha-keep",
                    "beta": "beta-level",
                },
            },
        }],
    }
    profiles = (
        LambdaProfile("zero", 0.0),
        LambdaProfile("middle", 1.0),
        LambdaProfile("high", 2.0),
    )

    result = initialize_hybrid_warm_start(
        policy,
        (_document(),),
        profiles,
        bc_state_dict=bc_state,
        exit_winners=winner,
        optimizer=optimizer,
    )

    assert result.identity_verified is True
    assert result.verified_winner_count == 1
    assert result.clone_target_count == len(_document().policy_decisions)
    assert torch.equal(policy.privacy_scale.detach(), privacy_before)
    assert torch.equal(policy.alpha_raw.detach(), alpha_before)


def test_trajectory_point_recomputes_fixed_denominator_utility_and_count_score():
    from cloak.ranker.interactive import (
        behavior_clone_trajectory,
        trajectory_point,
    )

    trajectory = behavior_clone_trajectory(_document(), "lambda-zero")
    # Monitoring 0.1, policy 0.7 -> the reported all-assertion utility is 0.4, the
    # credited policy-only utility is 1.0*0.7/1.0 = 0.7.
    result = _utility_result(trajectory.action_vector, 0.7, monitoring=0.1)
    point = trajectory_point(
        trajectory,
        result,
        count_reward=_count_reward(),
        utility_artifact=_utility_artifact(),
    )

    assert result.utility == pytest.approx(0.4)
    assert point.utility == pytest.approx(0.7)
    assert point.count_score == pytest.approx(0.6)
    assert point.component_scores == {"a-delivered": 0.1, "a-linked": 0.7}
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
    from cloak.ranker.interactive import collect_exit_winners
    from cloak.reward.utility_cache import UtilityCache

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
    from cloak.ranker.interactive import collect_exit_winners
    from cloak.reward.utility_cache import UtilityCache

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
    from cloak.ranker.interactive import collect_exit_winners
    from cloak.reward.utility_cache import UtilityCache

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
    from cloak.ranker.interactive import collect_exit_winners
    from cloak.reward.utility_cache import UtilityCache

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
    from cloak.ranker.interactive import collect_exit_winners
    from cloak.reward.utility_cache import UtilityCache

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
    import cloak.reward.roundtrip as roundtrip
    from cloak.ranker.interactive import CacheOnlyMissError, collect_exit_winners
    from cloak.reward.utility_cache import UtilityCache, stable_hash

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
    import cloak.reward.roundtrip as roundtrip
    from cloak.ranker.interactive import CacheOnlyMissError, collect_exit_winners
    from cloak.reward.utility_cache import UtilityCache

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
    from cloak.ranker.interactive import (
        collect_exit_winners,
        write_exit_winners,
    )
    from cloak.reward.utility_cache import UtilityCache

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
            "profile_target_artifact_hash": "targets-hash",
            "representation_manifest_hash": "representations-hash",
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
        "--policy-architecture", "semantic-v1",
        "--environment", "environment.json",
        "--representation-manifest", "representations.json",
        "--profile-count-targets", "targets.json",
        "--utility-artifact", "utility.json",
        "--utility-cache", "cache.jsonl",
    ]
    args = parser.parse_args([command, *common, *extra])
    assert args.command == command
    assert vars(args)["environment"] == "environment.json"
    assert vars(args)["profile_count_targets"] == "targets.json"
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
    from cloak.ranker.interactive import CacheOnlyMissError

    def blocked(args):
        raise CacheOnlyMissError(
            phase="initial", remote_tasks=3, context_reader_work_items=7,
        )

    monkeypatch.setattr(train_interactive_ranker, "_run_exit_collect", blocked)
    with pytest.raises(SystemExit) as captured:
        train_interactive_ranker.main([
            "exit-collect",
            "--policy-architecture", "semantic-v1",
            "--environment", "environment.json",
            "--representation-manifest", "representations.json",
            "--profile-count-targets", "targets.json",
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


@pytest.mark.parametrize(
    ("command", "runner_name", "extra"),
    [
        ("bc", "_run_bc", ["--out-checkpoint", "bc.pt"]),
        (
            "exit-collect", "_run_exit_collect",
            ["--checkpoint", "bc.pt", "--out", "exit.json", "--rollouts", "2"],
        ),
        (
            "train", "_run_train",
            [
                "--threshold-manifest", "threshold.json",
                "--lambda-menu", "menu.json",
                "--exit-winners", "exit.json",
                "--bc-checkpoint", "bc.pt",
                "--out-checkpoint", "policy.pt",
                "--kl-reference-checkpoint", "reference.pt",
                "--epoch-reports", "epochs.jsonl",
                "--fixed-lambda-zero-control", "control.pt",
                "--max-epochs", "1", "--rollouts", "2",
            ],
        ),
    ],
)
def test_semantic_cli_cache_only_stop_covers_every_training_stage(
    monkeypatch, capsys, command, runner_name, extra,
):
    import train_interactive_ranker
    from cloak.ranker.interactive import CacheOnlyMissError

    calls = []

    def blocked(args):
        calls.append(args)
        assert args.policy_architecture == "semantic-v1"
        assert args.cache_only is True
        assert args.representation_manifest == "representations.json"
        assert args.privacy_checkpoint == "privacy.pt"
        assert args.profile_count_targets == "profile-targets.json"
        assert args.allow_development_privacy_checkpoint is True
        raise CacheOnlyMissError(
            phase="initial", remote_tasks=4,
            context_reader_work_items=6,
        )

    monkeypatch.setattr(train_interactive_ranker, runner_name, blocked)
    with pytest.raises(SystemExit) as captured:
        train_interactive_ranker.main([
            command,
            "--policy-architecture", "semantic-v1",
            "--environment", "environment.json",
            "--representation-manifest", "representations.json",
            "--privacy-checkpoint", "privacy.pt",
            "--profile-count-targets", "profile-targets.json",
            "--allow-development-privacy-checkpoint",
            "--utility-artifact", "utility.json",
            "--utility-cache", "cache.jsonl",
            "--cache-only",
            *extra,
        ])

    assert captured.value.code == 2
    assert len(calls) == 1
    assert capsys.readouterr().err.strip() == (
        "CACHE_ONLY_MISS phase=initial remote_tasks=4 "
        "context_reader_work_items=6"
    )


def test_semantic_bc_config_round_trips_every_architecture_pin():
    import train_interactive_ranker

    policy = SimpleNamespace(
        representation_store=SimpleNamespace(manifest={
            "encoder": {"id": "encoder-model-pin"},
        }),
        encoder_revision="encoder-revision",
        context_mode="full-candidate-attention",
        history_mode="selected-cross-attention",
        feature_schema=("utility_relation", "candidate_context"),
        controller_transform="log1p-over-log1p-max-v1",
    )

    config = train_interactive_ranker._semantic_bc_policy_config(policy)
    assert config == {
        "policy_architecture": "semantic-v1",
        "encoder_pin": "encoder-model-pin",
        "encoder_revision": "encoder-revision",
        "context_mode": "full-candidate-attention",
        "history_mode": "selected-cross-attention",
        "feature_schema": ["utility_relation", "candidate_context"],
        "controller_transform": "log1p-over-log1p-max-v1",
    }
    train_interactive_ranker._validate_semantic_bc_policy_config(config, policy)
    with pytest.raises(ValueError, match="semantic policy contract"):
        train_interactive_ranker._validate_semantic_bc_policy_config(
            dict(config, history_mode="none"), policy,
        )


def test_train_cli_requires_every_frozen_artifact_output_and_runtime_control():
    import train_interactive_ranker

    parser = train_interactive_ranker.build_parser()
    args = parser.parse_args([
        "train",
        "--policy-architecture", "semantic-v1",
        "--environment", "environment.json",
        "--representation-manifest", "representations.json",
        "--profile-count-targets", "targets.json",
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
            "--policy-architecture", "semantic-v1",
            "--environment", "environment.json",
            "--utility-artifact", "utility.json",
            "--utility-cache", "cache.jsonl",
        ])


def test_semantic_cli_factory_uses_only_frozen_local_artifact_contracts(monkeypatch):
    import train_interactive_ranker
    from cloak.ranker.environment import LambdaProfile

    contract = SimpleNamespace(pair_dim=20)
    store = SimpleNamespace(manifest={"encoder": {"hidden_size": 4}})
    captured = {}
    sentinel = object()

    monkeypatch.setattr(
        train_interactive_ranker.RankerRepresentationStore,
        "open",
        lambda path: store,
    )
    monkeypatch.setattr(
        train_interactive_ranker,
        "_privacy_checkpoint_contract",
        lambda path: contract,
    )
    monkeypatch.setattr(
        train_interactive_ranker.SemanticRankerPolicy,
        "from_privacy_checkpoint",
        lambda **kwargs: captured.update(kwargs) or sentinel,
    )
    args = SimpleNamespace(
        representation_manifest="representations.json",
        privacy_checkpoint="privacy.pt",
    )
    profiles = (
        LambdaProfile("zero", 0.0),
        LambdaProfile("high", 2.0),
    )

    result = train_interactive_ranker._semantic_training_policy(
        args, (_document(),), profiles,
    )

    assert result is sentinel
    assert captured["representation_store"] is store
    assert captured["privacy_checkpoint_contract"] is contract
    assert captured["supported_profiles"] == profiles
    assert captured["max_lambda"] == 2.0
    assert captured["token_dim"] == captured["relation_dim"] == 4
    assert captured["pair_dim"] == 20
    assert captured["num_heads"] == 1
    assert captured["runtime_types"] == ("TYPE",)


def test_train_cli_dispatches_train_and_preserves_cache_only_stop(monkeypatch, capsys):
    import train_interactive_ranker
    from cloak.ranker.interactive import CacheOnlyMissError

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
            "--policy-architecture", "semantic-v1",
            "--environment", "environment.json",
            "--representation-manifest", "representations.json",
            "--profile-count-targets", "targets.json",
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
    from cloak.ranker.interactive import (
        ReplayedStep,
        ReplayedTrajectory,
    )
    from cloak.ranker.profile_count import ProfileActionTarget, ProfileCountTargets
    from cloak.reward.utility_credit import DocumentUtilityCredit

    probability = torch.tensor([0.75, 0.25], dtype=torch.float64)
    log_probs = probability.log().requires_grad_()
    trajectories = []
    for rollout_index in range(2):
        steps = tuple(
            ReplayedStep(
                decision_id=decision_id,
                selected_action_id=f"{decision_id}-selected",
                legal_action_ids=(f"{decision_id}-selected", f"{decision_id}-other"),
                log_prob=log_probs[0],
                log_probs=log_probs,
                count_log_probs=log_probs,
                utility_logits=log_probs,
                predicted_privacy=torch.zeros_like(log_probs),
                entropy=-(log_probs.exp() * log_probs).sum(),
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
            action_rows[action_id] = ProfileActionTarget(
                action_id=action_id,
                decision_id=decision_id,
                runtime_type="TYPE",
                profile_id="profile",
                mode="level",
                log_count=1.0,
                profile_score=score,
                grounding_status="certifying",
                source_family="fixture",
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
    targets = ProfileCountTargets(
        action_rows,
        decision_actions,
        {decision_id: True for decision_id in decision_actions},
    )
    return tuple(trajectories), targets, credit


def test_hybrid_document_objective_matches_two_rollout_two_decision_equation():
    from cloak.ranker.interactive import (
        compose_hybrid_document_objective,
        hybrid_utility_loss,
    )

    replayed, profile_targets, credit = _objective_fixture()
    counterfactual = {(0, "second"): replayed[0].steps[1].log_prob * 0.0 + 0.3}
    utility = hybrid_utility_loss(replayed, credit, counterfactual)
    reference = tuple(
        tuple(torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64)) for _ in row.steps)
        for row in replayed
    )

    objective = compose_hybrid_document_objective(
        replayed,
        utility_loss=utility,
        profile_targets=profile_targets,
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


def test_reverse_kl_direction_matches_manual_formula_and_keeps_saturated_gradient():
    from cloak.ranker.interactive import (
        compose_hybrid_document_objective,
        hybrid_utility_loss,
    )

    replayed, profile_targets, credit = _objective_fixture()
    utility = hybrid_utility_loss(replayed, credit, {})
    reference = tuple(
        tuple(torch.log(torch.tensor([0.5, 0.5], dtype=torch.float64)) for _ in row.steps)
        for row in replayed
    )
    objective = compose_hybrid_document_objective(
        replayed,
        utility_loss=utility,
        profile_targets=profile_targets,
        lambda_value=2.0,
        beta=0.0,
        eta=0.2,
        reference_log_probs=reference,
        kl_direction="reverse",
    )
    # KL(ref || pi) with pi = (0.75, 0.25), ref = (0.5, 0.5), per step
    per_step = (0.5 * torch.log(torch.tensor(0.5 / 0.75))
                + 0.5 * torch.log(torch.tensor(0.5 / 0.25)))
    assert float(objective.kl.detach()) == pytest.approx(2 * float(per_step))

    with pytest.raises(ValueError, match="unsupported KL direction"):
        compose_hybrid_document_objective(
            replayed,
            utility_loss=utility,
            profile_targets=profile_targets,
            lambda_value=2.0,
            beta=0.0,
            eta=0.2,
            reference_log_probs=reference,
            kl_direction="sideways",
        )

    # gradient survives saturation: pi -> one-hot, forward KL grad ~ 0, reverse stays
    sharp = torch.tensor([12.0, -12.0], requires_grad=True)
    log_pi = torch.log_softmax(sharp, dim=0)
    ref = torch.log(torch.tensor([0.5, 0.5]))
    forward = torch.sum(log_pi.exp() * (log_pi - ref))
    reverse = torch.sum(ref.exp() * (ref - log_pi))
    grad_f = torch.autograd.grad(forward, sharp, retain_graph=True)[0]
    grad_r = torch.autograd.grad(reverse, sharp)[0]
    assert float(grad_r.abs().max()) > 100 * float(grad_f.abs().max())


def test_count_gradient_survives_tied_utility_and_is_not_divided_twice():
    from cloak.ranker.interactive import (
        compose_hybrid_document_objective,
        hybrid_utility_loss,
    )
    from cloak.reward.utility_credit import DocumentUtilityCredit

    replayed, profile_targets, _ = _objective_fixture()
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
        profile_targets=profile_targets,
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
    from cloak.ranker.interactive import _utility_family_terms_and_mass
    from cloak.reward.utility_credit import DocumentUtilityCredit

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
    from cloak.ranker.interactive import (
        build_latin_cycle_schedule,
        profile_exposure_report,
    )
    from cloak.ranker.environment import LambdaProfile

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

    from cloak.ranker.interactive import train_hybrid_document_group
    from cloak.ranker.environment import LambdaProfile
    from cloak.reward.utility_cache import UtilityCache

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
        profile_targets=_count_reward(),
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


def test_semantic_hybrid_step_routes_profile_loss_and_emits_owned_gradients(tmp_path):
    from cloak.ranker.interactive import train_hybrid_document_group
    from cloak.ranker.environment import LambdaProfile
    from cloak.reward.utility_cache import UtilityCache

    policy = SemanticGradientPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.05)

    def scorer(requests, **kwargs):
        kwargs["cache"].last_batch_metrics = {}
        return [_utility_result(request.action_vector, 0.5) for request in requests]

    def scheduler(*args, **kwargs):
        del args, kwargs
        return (), {
            "budget": 0, "uniform_allocation": 0, "priority_allocation": 0,
            "cache_hits": 0, "delta_u": {}, "skip_reasons": {},
        }

    def executor(requests, *args, **kwargs):
        assert requests == ()
        kwargs["cache"].last_batch_metrics = {}
        return {}, dict(kwargs["scheduler_diagnostics"])

    result = train_hybrid_document_group(
        policy,
        None,
        _document(),
        LambdaProfile("high", 2.0),
        rollouts=2,
        utility_artifact=_utility_artifact(),
        environment_hash="env-hash",
        profile_targets=_profile_targets(),
        cache=UtilityCache(tmp_path / "cache.jsonl"),
        optimizer=optimizer,
        beta=0.0,
        eta=0.0,
        counterfactual_budget=0,
        endpoint_budget=0,
        pair_history={},
        seed=2,
        current_round=0,
        remote_workers=1,
        reader_workers=1,
        generator=torch.Generator().manual_seed(5),
        score_batch=scorer,
        scheduler=scheduler,
        counterfactual_executor=executor,
    )

    assert result.parameter_group_gradient_norms["count"]["utility"] == 0.0
    assert result.parameter_group_gradient_norms["count"]["history"] == 0.0
    assert result.parameter_group_gradient_norms["count"]["alpha"] > 0.0
    assert result.parameter_group_gradient_norms["fallback"]["privacy"] == 0.0
    assert result.alpha_diagnostics["value"] >= 0.0
    assert result.privacy_diagnostics["selected_count"] == 4
    assert result.privacy_diagnostics["strata"]


def test_epoch_report_keeps_scheduler_budget_separate_from_reward_magnitude(tmp_path):
    from cloak.ranker.interactive import DocumentTrainingResult, build_epoch_report

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


def test_epoch_report_extends_semantic_gradient_alpha_privacy_diagnostics():
    from cloak.ranker.interactive import DocumentTrainingResult, build_epoch_report

    families = (
        "linked", "residual", "fallback", "counterfactual",
        "count", "entropy", "KL",
    )
    groups = {name: 0.0 for name in ("utility", "history", "privacy", "alpha")}
    parameter_norms = {name: dict(groups) for name in families}
    parameter_norms["count"]["alpha"] = 0.75
    parameter_norms["linked"]["utility"] = 0.5
    row = DocumentTrainingResult(
        doc_id="fixture/doc",
        corpus="fixture",
        profile_name="lambda-two",
        rollout_count=1,
        loss=0.1,
        utility=0.2,
        count_score=0.3,
        entropy=0.4,
        collision_count=0,
        action_modes={"keep": 1},
        runtime_type_exposure={"TYPE": 1},
        gradient_norms={name: 0.1 for name in families},
        absolute_weighted_mass={name: 0.2 for name in families},
        scheduler_diagnostics={"budget": 0},
        cache_metrics={},
        parameter_group_gradient_norms=parameter_norms,
        alpha_diagnostics={"value": 1.0, "gradient": -0.25},
        privacy_diagnostics={
            "selected_count": 2,
            "predicted_sum": 1.1,
            "exact_sum": 1.0,
            "absolute_error_sum": 0.3,
            "strata": {
                "singleton_profile|grounded": {
                    "count": 1,
                    "predicted_sum": 0.6,
                    "exact_sum": 0.5,
                    "absolute_error_sum": 0.1,
                },
            },
        },
        lambda_zero_identity_failures=1,
    )

    report = build_epoch_report(0, (row,))
    diagnostics = report["semantic_diagnostics"]

    assert diagnostics["parameter_group_gradient_norms"]["count"] == {
        "utility": 0.0, "history": 0.0, "privacy": 0.0, "alpha": 0.75,
    }
    assert diagnostics["alpha_by_lambda"]["lambda-two"] == {
        "value": 1.0, "gradient": -0.25,
    }
    assert diagnostics["selected_privacy"] == {
        "count": 2,
        "predicted_mean": pytest.approx(0.55),
        "exact_mean": pytest.approx(0.5),
        "mean_absolute_error": pytest.approx(0.15),
    }
    assert diagnostics["privacy_strata"]["singleton_profile|grounded"]["count"] == 1
    assert diagnostics["lambda_zero_identity_failures"] == 1


def test_failed_counterfactual_execution_does_not_advance_pair_history(tmp_path):
    import copy

    from cloak.ranker.counterfactuals import CounterfactualRequest
    from cloak.ranker.interactive import train_hybrid_document_group
    from cloak.ranker.environment import LambdaProfile
    from cloak.reward.utility_cache import UtilityCache

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
            profile_targets=_count_reward(),
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

    from cloak.ranker.interactive import (
        build_latin_cycle_schedule,
        load_hybrid_checkpoint,
        save_hybrid_checkpoint,
    )
    from cloak.ranker.environment import LambdaProfile

    policy = StubPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    profiles = (LambdaProfile("zero", 0.0), LambdaProfile("high", 2.0))
    schedule = build_latin_cycle_schedule((_document(),), profiles, seed=11)
    generator = torch.Generator().manual_seed(23)
    pins = _semantic_checkpoint_pins()
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
    from cloak.ranker.interactive import (
        build_latin_cycle_schedule,
        load_hybrid_checkpoint,
        save_hybrid_checkpoint,
    )
    from cloak.ranker.environment import LambdaProfile

    policy = StubPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    generator = torch.Generator().manual_seed(1)
    schedule = build_latin_cycle_schedule(
        (_document(),), (LambdaProfile("zero", 0.0),), seed=0,
    )
    pins = _semantic_checkpoint_pins()
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


def _semantic_checkpoint_pins():
    return {
        "environment_hash": "env",
        "utility_artifact_hash": "utility",
        "profile_target_artifact_hash": "profile-targets",
        "representation_manifest_hash": "representations",
        "privacy_checkpoint_hash": "privacy",
        "lambda_menu_hash": "menu",
        "threshold_manifest_hash": "threshold",
    }


def test_semantic_checkpoint_publishes_frozen_contract_and_round_trips(tmp_path):
    from cloak.ranker.interactive import (
        build_latin_cycle_schedule,
        load_hybrid_checkpoint,
        policy_architecture_pin,
        save_hybrid_checkpoint,
    )
    from cloak.ranker.environment import LambdaProfile

    policy = SemanticGradientPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(4)
    schedule = build_latin_cycle_schedule(
        (_document(),), (LambdaProfile("zero", 0.0),), seed=7,
    )
    architecture_pin = policy_architecture_pin(policy)
    path = tmp_path / "semantic.pt"
    save_hybrid_checkpoint(
        path,
        policy=policy,
        optimizer=optimizer,
        epoch=2,
        generator=generator,
        schedule=schedule,
        artifact_pins=_semantic_checkpoint_pins(),
        architecture_pin=architecture_pin,
        cache_paths={"utility": "cache"},
        code_revision="revision",
        training_config={"learning_rate": 0.01, "beta": 0.0, "eta": 0.0},
        pair_history={},
        kl_enabled=False,
        epoch_reports=(),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert payload["checkpoint_version"] == "ranker-v2-semantic-policy-v1"
    assert payload["policy_architecture"] == "semantic-v1"
    assert payload["semantic_contract"] == {
        "encoder_revision": "stub-encoder-revision",
        "context_mode": "full-candidate-attention",
        "history_mode": "selected-cross-attention",
        "feature_schema": list(policy.feature_schema),
        "controller_transform": "log1p-over-log1p-max-v1",
    }
    saved = policy.alpha_raw.detach().clone()
    with torch.no_grad():
        policy.alpha_raw.add_(5.0)
    load_hybrid_checkpoint(
        path,
        policy=policy,
        optimizer=optimizer,
        generator=generator,
        expected_artifact_pins=_semantic_checkpoint_pins(),
        expected_architecture_pin=architecture_pin,
        expected_cache_paths={"utility": "cache"},
        expected_code_revision="revision",
        expected_training_config={"learning_rate": 0.01, "beta": 0.0, "eta": 0.0},
    )
    assert torch.equal(policy.alpha_raw.detach(), saved)


def test_semantic_and_legacy_checkpoints_reject_each_other_before_tensor_load(tmp_path):
    from cloak.ranker.interactive import (
        build_latin_cycle_schedule,
        load_hybrid_checkpoint,
        policy_architecture_pin,
        save_hybrid_checkpoint,
    )
    from cloak.ranker.environment import LambdaProfile

    semantic = SemanticGradientPolicy()
    semantic_optimizer = torch.optim.SGD(semantic.parameters(), lr=0.1)
    generator = torch.Generator().manual_seed(1)
    schedule = build_latin_cycle_schedule(
        (_document(),), (LambdaProfile("zero", 0.0),), seed=0,
    )
    path = tmp_path / "semantic.pt"
    save_hybrid_checkpoint(
        path,
        policy=semantic,
        optimizer=semantic_optimizer,
        epoch=0,
        generator=generator,
        schedule=schedule,
        artifact_pins=_semantic_checkpoint_pins(),
        architecture_pin=policy_architecture_pin(semantic),
        cache_paths={"utility": "cache"},
        code_revision="revision",
        training_config={"learning_rate": 0.1},
        pair_history={},
        kl_enabled=False,
        epoch_reports=(),
    )
    legacy = StubPolicy()
    legacy_before = legacy.scores.detach().clone()

    with pytest.raises(ValueError, match="architecture"):
        load_hybrid_checkpoint(
            path,
            policy=legacy,
            optimizer=torch.optim.SGD(legacy.parameters(), lr=0.1),
            generator=torch.Generator(),
            expected_artifact_pins=_semantic_checkpoint_pins(),
            expected_architecture_pin=policy_architecture_pin(legacy),
            expected_cache_paths={"utility": "cache"},
            expected_code_revision="revision",
            expected_training_config={"learning_rate": 0.1},
        )
    assert torch.equal(legacy.scores.detach(), legacy_before)

    legacy_path = tmp_path / "legacy.pt"
    legacy_optimizer = torch.optim.SGD(legacy.parameters(), lr=0.1)
    legacy_pins = _semantic_checkpoint_pins()
    save_hybrid_checkpoint(
        legacy_path,
        policy=legacy,
        optimizer=legacy_optimizer,
        epoch=0,
        generator=torch.Generator().manual_seed(1),
        schedule=schedule,
        artifact_pins=legacy_pins,
        architecture_pin=policy_architecture_pin(legacy),
        cache_paths={"utility": "cache"},
        code_revision="revision",
        training_config={"learning_rate": 0.1},
        pair_history={},
        kl_enabled=False,
        epoch_reports=(),
    )
    semantic_before = semantic.alpha_raw.detach().clone()
    with pytest.raises(ValueError, match="architecture"):
        load_hybrid_checkpoint(
            legacy_path,
            policy=semantic,
            optimizer=semantic_optimizer,
            generator=torch.Generator(),
            expected_artifact_pins=_semantic_checkpoint_pins(),
            expected_architecture_pin=policy_architecture_pin(semantic),
            expected_cache_paths={"utility": "cache"},
            expected_code_revision="revision",
            expected_training_config={"learning_rate": 0.1},
        )
    assert torch.equal(semantic.alpha_raw.detach(), semantic_before)


def test_kl_enables_only_when_frozen_collapse_threshold_fires():
    from cloak.ranker.interactive import collapse_rule_fires

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
    from cloak.ranker.interactive import (
        DocumentTrainingResult,
        build_latin_cycle_schedule,
        train_hybrid_policy,
    )
    from cloak.ranker.environment import LambdaProfile

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
        profile_targets=object(),
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

    # always-on schedule: KL from epoch zero, direction forwarded to groups
    calls.clear()
    always = train_hybrid_policy(
        policy=object(),
        reference_policy=object(),
        documents=documents,
        profiles=profiles,
        schedule=schedule,
        optimizer=object(),
        utility_artifact={},
        environment_hash="env",
        profile_targets=object(),
        cache=object(),
        threshold_manifest={
            "feasibility_gates": {"min_adjacent_winner_change": 0.2},
        },
        max_epochs=1,
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
        kl_schedule="always-on",
    )
    assert all(eta == 0.3 for *_, eta in calls)
    assert always.kl_enabled is True
    assert always.epoch_reports[0]["kl_enabled_for_epoch"] is True


def test_scope_demotion_moves_uncovered_and_out_of_scope_menus_to_fixed_keep():
    import train_interactive_ranker

    document = _document()
    provider = SimpleNamespace(
        has_targets=lambda decision_id, action_ids: decision_id == "alpha"
    )

    updated, demoted = (
        train_interactive_ranker._demote_out_of_scope_decisions(
            (document,), provider, scope_types=frozenset({"TYPE"})
        )
    )

    assert demoted == 1
    result = updated[0]
    assert tuple(d.decision_id for d in result.policy_decisions) == ("alpha",)
    assert result.fixed_decisions[-1].decision_id == "beta"
    assert [a.mode for a in result.fixed_decisions[-1].actions] == ["keep"]
    covered, unchanged = (
        train_interactive_ranker._demote_out_of_scope_decisions(
            (document,), SimpleNamespace(has_targets=lambda d, a: True),
            scope_types=frozenset({"TYPE"}),
        )
    )
    assert unchanged == 0
    assert covered[0] is document

    # Out-of-scope runtime type is demoted even with full coverage; provider=None
    # (learned path) skips the coverage check but still enforces scope.
    scoped, scope_demoted = (
        train_interactive_ranker._demote_out_of_scope_decisions(
            (document,), None, scope_types=frozenset({"OTHER"})
        )
    )
    assert scope_demoted == 2
    assert not scoped[0].policy_decisions
    assert [d.decision_id for d in scoped[0].fixed_decisions[-2:]] == ["alpha", "beta"]


def test_scope_demotion_rejects_menu_without_unique_keep():
    import train_interactive_ranker

    document = _document()
    beta = document.policy_decisions[1]
    no_keep = replace(
        beta,
        actions=tuple(a for a in beta.actions if a.mode != "keep"),
    )
    broken = replace(
        document, policy_decisions=(document.policy_decisions[0], no_keep)
    )

    with pytest.raises(ValueError, match="unique keep action"):
        train_interactive_ranker._demote_out_of_scope_decisions(
            (broken,), SimpleNamespace(has_targets=lambda d, a: False),
            scope_types=frozenset({"TYPE"}),
        )


def test_zero_signal_documents_are_dropped_from_training():
    import train_interactive_ranker

    documents = (_document(), replace(_document(), doc_id="fixture/nolink"))
    artifact = {"assertions": {
        "a1": {"doc_id": "fixture/doc", "status": "accepted",
               "answer_target": {"kind": "linked_decision", "decision_id": "alpha"}},
        "a2": {"doc_id": "fixture/nolink", "status": "accepted",
               "answer_target": {"kind": "literal", "expected_values": ["x"]}},
        "a3": {"doc_id": "fixture/nolink", "status": "rejected",
               "answer_target": {"kind": "linked_decision", "decision_id": "beta"}},
    }}
    retained, dropped = train_interactive_ranker._drop_zero_signal_documents(
        documents, artifact
    )
    assert [d.doc_id for d in retained] == ["fixture/doc"]
    assert dropped == ("fixture/nolink",)

    with pytest.raises(ValueError, match="zero policy-linked"):
        train_interactive_ranker._drop_zero_signal_documents(
            (documents[1],), artifact
        )


def test_assembly_alias_group_fills_restore_instead_of_retaining():
    from cloak.ranker.environment import assemble_action_vector
    from cloak.reward.extract import invert

    gamma = _decision("gamma", ("o-g1", "o-g2"))
    document = RankerDocument(
        doc_id="fixture/alias",
        corpus="fixture",
        text="Foo met Bar.",
        occurrences=(
            MappingProxyType({
                "occurrence_id": "o-g1", "decision_id": "gamma",
                "start": 0, "end": 3, "surface": "Foo", "controlled": True,
            }),
            MappingProxyType({
                "occurrence_id": "o-g2", "decision_id": "gamma",
                "start": 8, "end": 11, "surface": "Bar", "controlled": True,
            }),
        ),
        policy_decisions=(gamma,),
        fixed_decisions=(),
    )
    doc_p, replacements = assemble_action_vector(document, {"gamma": "gamma-level"})
    assert all("restore_policy" not in row for row in replacements)

    # Every echo restores to the group's first (canonical) alias surface;
    # case-adjusted duplicates may log gen_absent but no fill text survives.
    out_final, stats = invert(doc_p, replacements)
    assert stats["gen_retained"] == 0
    assert "fill" not in out_final.lower()
    assert "Foo" in out_final


def test_support_scaled_rollouts_targets_degeneracy_probability():
    from cloak.ranker.interactive import support_scaled_rollouts

    # p_hat^R <= 0.05: p=0.9 needs 29, p=0.97 hits the cap, diverse docs stay at base
    assert support_scaled_rollouts(0.90, 8) == 29
    assert support_scaled_rollouts(0.97, 8) == 32
    assert support_scaled_rollouts(0.10, 8) == 8
    assert support_scaled_rollouts(1.0, 8) == 32
    assert support_scaled_rollouts(0.0, 8) == 8
    for p_hat in (0.5, 0.8, 0.9, 0.95):
        rollouts = support_scaled_rollouts(p_hat, 8)
        assert p_hat ** rollouts <= 0.05 or rollouts == 32


def test_dominant_trajectory_probability_matches_greedy_walk():
    import math

    from cloak.ranker.interactive import (
        dominant_trajectory_probability,
        sample_trajectory,
        replay_trajectory,
    )
    from test_semantic_ranker import _direct_count_policy

    policy, document, _, profiles, _ = _direct_count_policy()
    p_hat = dominant_trajectory_probability(policy, document, profiles[0])
    assert 0.0 < p_hat <= 1.0
    greedy = sample_trajectory(
        policy, document, profiles[0], greedy=True, generator=None,
    )
    replayed = replay_trajectory(policy, document, greedy, profiles[0])
    expected = math.exp(sum(float(step.log_prob) for step in replayed.steps))
    assert p_hat == pytest.approx(expected, rel=1e-5)


def test_synchronous_profile_snapshot_reads_all_profiles_from_one_policy():
    from cloak.ranker.interactive import synchronous_profile_snapshot
    from test_semantic_ranker import _direct_count_policy

    policy, document, decision, profiles, scores = _direct_count_policy()
    score_by_action = {
        action.action_id: value
        for action, value in zip(decision.actions, scores, strict=True)
    }

    class StubTargets:
        def action_scores(self, decision_id, action_ids):
            del decision_id
            return tuple(score_by_action[a] for a in action_ids)

    snapshot = synchronous_profile_snapshot(
        policy, (document,), profiles, StubTargets(), samples=8, seed=3,
    )
    per_profile = snapshot[document.doc_id]
    assert set(per_profile) == {profile.name for profile in profiles}
    for name, row in per_profile.items():
        assert 0.0 <= row["sampled_P_mean"] <= 1.0
        assert row["sampled_P_count"] == 8
        assert 0.0 <= row["greedy_P"] <= 1.0
        assert len(row["decisions"]) == len(document.policy_decisions)
        for entry in row["decisions"]:
            assert 0.0 <= entry["expected_p"] <= 1.0
            assert entry["utility_logit_range"] >= 0.0
    # deterministic given the seed, and the same policy state serves all profiles
    again = synchronous_profile_snapshot(
        policy, (document,), profiles, StubTargets(), samples=8, seed=3,
    )
    assert again == snapshot


def test_profile_sensitivity_loss_matches_manual_kl_and_reaches_alpha():
    import math

    from cloak.ranker.interactive import profile_sensitivity_loss
    from cloak.ranker.environment import LambdaProfile
    from test_semantic_ranker import _direct_count_policy

    policy, document, decision, profiles, _ = _direct_count_policy()
    from cloak.ranker.interactive import replay_trajectory, sample_trajectory

    greedy = sample_trajectory(
        policy, document, profiles[0], greedy=True, generator=None,
    )
    replayed = replay_trajectory(policy, document, greedy, profiles[0])

    loss = profile_sensitivity_loss(
        (replayed,), policy, profiles, target_kl_per_unit=0.0,
    )
    assert loss.ndim == 0 and float(loss) >= 0.0

    # manual reconstruction for the single decision, target 0:
    step = replayed.steps[0]
    alpha = policy.alpha
    ramp = [
        math.log1p(float(p.value)) / math.log1p(float(policy.max_lambda))
        for p in profiles
    ]
    logs = [
        torch.log_softmax(
            step.utility_logits + alpha * g * step.predicted_privacy.detach()
            if g > 0.0 else step.utility_logits,
            dim=0,
        )
        for g in ramp
    ]
    manual = []
    for k in range(len(ramp) - 1):
        dg = ramp[k + 1] - ramp[k]
        if dg <= 0.0:
            continue
        kl = torch.sum(logs[k].exp() * (logs[k] - logs[k + 1]))
        manual.append(kl ** 2)
    expected = torch.stack(manual).mean()
    assert float(loss) == pytest.approx(float(expected), rel=1e-5)

    # gradient reaches alpha
    policy.zero_grad(set_to_none=True)
    loss.backward()
    assert policy.alpha_raw.grad is not None
    assert float(policy.alpha_raw.grad.abs()) > 0.0


def test_measure_profile_sensitivity_target_is_finite_and_positive():
    from cloak.ranker.interactive import measure_profile_sensitivity_target
    from test_semantic_ranker import _direct_count_policy

    policy, document, decision, profiles, _ = _direct_count_policy()
    target = measure_profile_sensitivity_target(policy, (document,), profiles)
    assert math.isfinite(target) and target >= 0.0


def test_tie_evidence_ledger_and_verifiable_core_labels():
    from cloak.ranker.interactive import (
        compute_tie_labels,
        record_tie_evidence,
    )

    class StubTargets:
        def action_scores(self, decision_id, action_ids):
            scores = {"keep": 0.0, "L0": 0.4, "L1": 0.7}
            return tuple(scores[a] for a in action_ids)

    ledger = {}
    rows = [
        {"doc_id": "d", "decision_id": "j", "selected_action_id": "keep",
         "alternative_action_id": "L0", "delta_u": 0.0, "context_hash": f"c{i}"}
        for i in range(3)
    ]
    record_tie_evidence(ledger, rows, current_round=0)
    labels = compute_tie_labels(ledger, {}, StubTargets(), min_contexts=3, before_round=1)
    assert labels == {("d", "j"): (("L0", "keep"),)}  # oriented by count score

    # one-cycle lag: evidence at round >= before_round is invisible
    assert compute_tie_labels(ledger, {}, StubTargets(), min_contexts=3, before_round=0) == {}

    # contradiction exit: a single above-bound record disqualifies permanently
    record_tie_evidence(ledger, [{
        "doc_id": "d", "decision_id": "j", "selected_action_id": "L0",
        "alternative_action_id": "keep", "delta_u": 0.2, "context_hash": "c9",
    }], current_round=1)
    assert compute_tie_labels(ledger, {}, StubTargets(), min_contexts=3, before_round=2) == {}

    # noise-band records neither qualify nor disqualify
    ledger2 = {}
    record_tie_evidence(ledger2, rows[:2], current_round=0)
    record_tie_evidence(ledger2, [{
        "doc_id": "d", "decision_id": "j", "selected_action_id": "keep",
        "alternative_action_id": "L0", "delta_u": 0.02, "context_hash": "cx",
    }], current_round=0)
    assert compute_tie_labels(ledger2, {}, StubTargets(), min_contexts=3, before_round=1) == {}

    # distinct contexts required: duplicate context hashes do not count twice
    ledger3 = {}
    record_tie_evidence(ledger3, [dict(rows[0]), dict(rows[0]), dict(rows[0])], current_round=0)
    assert compute_tie_labels(ledger3, {}, StubTargets(), min_contexts=3, before_round=1) == {}


def test_tie_margin_loss_routes_gradient_to_gain_head_only():
    from cloak.ranker.interactive import sample_trajectory, tie_margin_loss
    from cloak.ranker.semantic import enable_controller_gain
    from test_semantic_ranker import _direct_count_policy

    policy, document, decision, profiles, scores = _direct_count_policy()
    enable_controller_gain(policy, "evidence", hidden_dim=8)
    policy.float()

    # label the two highest-score actions as a qualified tie (orientation by score)
    ordered = sorted(
        zip(decision.actions, scores), key=lambda pair: pair[1], reverse=True,
    )
    a_plus, a_minus = ordered[0][0].action_id, ordered[-1][0].action_id
    labels = {(document.doc_id, decision.decision_id): ((a_plus, a_minus),)}

    greedy = sample_trajectory(policy, document, profiles[-1], greedy=True, generator=None)
    hinge, penalty, satisfied, total = tie_margin_loss(
        policy, document, greedy, labels, max_profile=profiles[-1], margin=100.0,
    )
    assert total == 1 and hinge.ndim == 0 and penalty.ndim == 0
    assert satisfied == 0  # margin forced unattainable -> hinge active

    policy.zero_grad(set_to_none=True)
    (hinge + penalty).backward()
    gain_grads = [p_.grad for p_ in policy.gain_head.parameters() if p_.grad is not None]
    assert gain_grads and any(float(g.abs().sum()) > 0 for g in gain_grads)
    assert policy.alpha_raw.grad is None or float(policy.alpha_raw.grad.abs()) == 0.0
    tower_grads = [p_.grad for p_ in policy.utility_head.parameters() if p_.grad is not None]
    assert not tower_grads or all(float(g.abs().sum()) == 0.0 for g in tower_grads)
    policy.controller_gain_mode = None


def test_evidence_gain_count_path_trains_alpha_raw_not_residual():
    from cloak.ranker.semantic import enable_controller_gain
    from test_semantic_ranker import _direct_count_policy

    policy, document, decision, profiles, _ = _direct_count_policy()
    enable_controller_gain(policy, "evidence", hidden_dim=8)
    policy.float()
    menu = tuple(a.action_id for a in decision.actions)
    state = policy.begin_document(document, profiles[-1])
    row = policy.distribution(state, decision, menu, profiles[-1])
    policy.zero_grad(set_to_none=True)
    row.count_log_probs.sum().backward()
    assert policy.alpha_raw.grad is not None and float(policy.alpha_raw.grad.abs()) > 0
    gain_grads = [p_.grad for p_ in policy.gain_head.parameters() if p_.grad is not None]
    assert not gain_grads or all(float(g.abs().sum()) == 0.0 for g in gain_grads)
    # sampling path DOES train the residual
    policy.zero_grad(set_to_none=True)
    state2 = policy.begin_document(document, profiles[-1])
    row2 = policy.distribution(state2, decision, menu, profiles[-1])
    row2.log_probs.sum().backward()
    gain_grads = [p_.grad for p_ in policy.gain_head.parameters() if p_.grad is not None]
    assert gain_grads and any(float(g.abs().sum()) > 0 for g in gain_grads)
    policy.controller_gain_mode = None
