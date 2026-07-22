import math
from dataclasses import FrozenInstanceError, fields
from types import MappingProxyType

import pytest
import torch

import cloak.train.ranker as ranker
from cloak.train.count_reward import (
    CountActionScore,
    CountReward,
    expected_count_loss,
)
from cloak.train.interactive_ranker import replay_trajectory, sample_trajectory
from cloak.train.ranker_environment import (
    RankerAction,
    RankerDecision,
    RankerDocument,
)


PROFILES = (
    ("utility", 0.0),
    ("balanced", 1.0),
    ("privacy", 3.0),
)


def _symbols():
    missing = [
        name for name in ("LambdaProfile", "PolicyState", "ConditionalRankerPolicy")
        if not hasattr(ranker, name)
    ]
    if missing:
        pytest.fail(f"conditional ranker API is not implemented: {missing}")
    return ranker.LambdaProfile, ranker.PolicyState, ranker.ConditionalRankerPolicy


class StubEncoder:
    embedding_dim = 4

    def __init__(self):
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(tuple(texts))
        rows = []
        for text in texts:
            code_sum = sum(ord(char) for char in text)
            rows.append([
                len(text) / 100.0,
                (code_sum % 97) / 97.0,
                len(text.split()) / 10.0,
                ((code_sum // 97) % 89) / 89.0,
            ])
        return torch.tensor(rows, dtype=torch.float32)


def _action(owner, suffix, mode, fill, authored_level_index=None):
    return RankerAction(
        action_id=f"{owner}-{suffix}",
        mode=mode,
        fill=fill,
        authored_level_index=authored_level_index,
        runtime_type="LOC",
    )


def _decision(owner, occurrence_ids, fills):
    actions = tuple(
        _action(owner, f"level-{index}", "level", fill, index)
        for index, fill in enumerate(fills)
    ) + (
        _action(owner, "keep", "keep", owner),
        _action(owner, "placeholder", "placeholder", None),
    )
    return RankerDecision(
        decision_id=owner,
        profile_id=f"profile-{owner}",
        runtime_type="LOC",
        canonical_key=owner,
        occurrence_ids=occurrence_ids,
        actions=actions,
    )


def _document():
    text = "Alpha met alpha near Beta."
    alpha = _decision(
        "alpha", ("occ-alpha-1", "occ-alpha-2"),
        ("specific place", "a region"),
    )
    beta = _decision(
        "beta", ("occ-beta",), ("a landmark", "an area"),
    )
    occurrences = (
        MappingProxyType({
            "occurrence_id": "occ-alpha-1", "decision_id": "alpha",
            "start": 0, "end": 5, "surface": "Alpha", "controlled": True,
        }),
        MappingProxyType({
            "occurrence_id": "occ-alpha-2", "decision_id": "alpha",
            "start": 10, "end": 15, "surface": "alpha", "controlled": True,
        }),
        MappingProxyType({
            "occurrence_id": "occ-beta", "decision_id": "beta",
            "start": 21, "end": 25, "surface": "Beta", "controlled": True,
        }),
    )
    return RankerDocument(
        doc_id="fixture/doc",
        corpus="fixture",
        text=text,
        occurrences=occurrences,
        policy_decisions=(alpha, beta),
        fixed_decisions=(),
    )


def _count_reward(document):
    scores = {
        "alpha-level-0": 0.25,
        "alpha-level-1": 0.75,
        "alpha-keep": 0.0,
        "alpha-placeholder": 1.0,
        "beta-level-0": 0.30,
        "beta-level-1": 0.80,
        "beta-keep": 0.0,
        "beta-placeholder": 1.0,
    }
    action_rows = {}
    decision_actions = {}
    for decision in document.policy_decisions:
        decision_actions[decision.decision_id] = tuple(
            action.action_id for action in decision.actions
        )
        for action in decision.actions:
            action_rows[action.action_id] = CountActionScore(
                action_id=action.action_id,
                decision_id=decision.decision_id,
                runtime_type=decision.runtime_type,
                profile_id=decision.profile_id,
                mode=action.mode,
                count=None,
                score=scores[action.action_id],
                grounding_status=None,
                source_family=None,
                evidence_ref=None,
            )
    return CountReward(action_rows, decision_actions)


def _profiles():
    LambdaProfile, _, _ = _symbols()
    return tuple(LambdaProfile(name, value) for name, value in PROFILES)


def _policy(*, encoder=None):
    _, _, ConditionalRankerPolicy = _symbols()
    document = _document()
    encoder = encoder or StubEncoder()
    torch.manual_seed(7)
    policy = ConditionalRankerPolicy(
        count_reward=_count_reward(document),
        supported_profiles=_profiles(),
        max_menu_value=3.0,
        environment_hash="sha256:fixture-environment",
        encoder_pin="stub-encoder-v1",
        encoder=encoder,
        hidden_dim=12,
    )
    return policy, document, encoder


def test_lambda_profile_is_small_frozen_value_and_magnitude_is_log_normalized():
    LambdaProfile, PolicyState, _ = _symbols()

    assert [field.name for field in fields(LambdaProfile)] == ["name", "value"]
    profile = LambdaProfile("balanced", 1.0)
    with pytest.raises(FrozenInstanceError):
        profile.value = 2.0

    policy, document, _ = _policy()
    state = policy.begin_document(document, profile)
    assert isinstance(state, PolicyState)
    assert state.lambda_magnitude == pytest.approx(
        math.log1p(1.0) / math.log1p(3.0)
    )


def test_v2_action_features_have_only_declared_layout_and_frozen_count_scores():
    policy, document, _ = _policy()
    decision = document.policy_decisions[0]
    action_ids = tuple(action.action_id for action in decision.actions)

    features = policy.action_features(decision, action_ids)

    expected_names = (
        "mode:level", "mode:keep", "mode:placeholder",
        "authored_level_position", "number_of_levels",
        *(f"type:{runtime_type}" for runtime_type in policy.runtime_types),
        "count_score",
    )
    assert policy.action_feature_names == expected_names
    assert features.shape == (4, len(expected_names))
    assert torch.equal(features[:, :3], torch.tensor([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]))
    assert torch.equal(features[:, 3], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.equal(features[:, 4], torch.full((4,), 2.0))
    loc_index = policy.action_feature_names.index("type:LOC")
    assert torch.equal(features[:, loc_index], torch.ones(4))
    assert torch.allclose(features[:, -1], torch.tensor([0.25, 0.75, 0.0, 1.0]))
    assert not features.requires_grad
    assert all(
        forbidden not in name
        for name in policy.action_feature_names
        for forbidden in ("p6", "aset", "floor", "corpus")
    )


def test_occurrence_and_fill_embeddings_are_batched_cached_and_mean_pooled():
    encoder = StubEncoder()
    policy, document, _ = _policy(encoder=encoder)

    policy.begin_document(document, _profiles()[0])

    assert sorted(len(call) for call in encoder.calls) == [3, 4]
    assert set(policy.context_embedding_cache) == {
        ("sha256:fixture-environment", "stub-encoder-v1", "occ-alpha-1"),
        ("sha256:fixture-environment", "stub-encoder-v1", "occ-alpha-2"),
        ("sha256:fixture-environment", "stub-encoder-v1", "occ-beta"),
    }
    assert set(policy.action_embedding_cache) == {
        ("stub-encoder-v1", "alpha-level-0"),
        ("stub-encoder-v1", "alpha-level-1"),
        ("stub-encoder-v1", "beta-level-0"),
        ("stub-encoder-v1", "beta-level-1"),
    }
    expected = torch.stack([
        policy.context_embedding_cache[
            ("sha256:fixture-environment", "stub-encoder-v1", occurrence_id)
        ]
        for occurrence_id in ("occ-alpha-1", "occ-alpha-2")
    ]).mean(dim=0)
    assert torch.equal(policy.decision_context(document, document.policy_decisions[0]), expected)

    call_count = len(encoder.calls)
    policy.begin_document(document, _profiles()[2])
    assert len(encoder.calls) == call_count


def test_all_profiles_have_exactly_identical_policy_at_identity_initialization():
    policy, document, _ = _policy()
    decision = document.policy_decisions[0]
    menu = tuple(action.action_id for action in decision.actions)

    distributions = []
    for profile in _profiles():
        state = policy.begin_document(document, profile)
        distributions.append(policy.log_probs(state, decision, menu, profile))

    assert all(torch.equal(distributions[0], row) for row in distributions[1:])
    assert torch.count_nonzero(policy.profile_embeddings.weight) == 0
    assert torch.count_nonzero(policy.film.weight) == 0
    assert torch.count_nonzero(policy.film.bias) == 0
    gamma, beta = policy.film_parameters(_profiles()[2])
    assert torch.equal(gamma, torch.ones_like(gamma))
    assert torch.equal(beta, torch.zeros_like(beta))


def test_nonzero_magnitude_and_profile_identity_condition_relative_logits():
    policy, document, _ = _policy()
    decision = document.policy_decisions[0]
    menu = tuple(action.action_id for action in decision.actions)
    count_row = (
        policy.action_feature_offset
        + policy.action_feature_names.index("count_score")
    )
    magnitude_column = policy.profile_embedding_dim

    with torch.no_grad():
        policy.film.weight[count_row, magnitude_column] = 1.0
    utility, _, privacy = _profiles()
    lp_utility = policy.log_probs(
        policy.begin_document(document, utility), decision, menu, utility,
    )
    lp_privacy = policy.log_probs(
        policy.begin_document(document, privacy), decision, menu, privacy,
    )
    delta = lp_privacy - lp_utility
    assert not torch.allclose(lp_privacy, lp_utility)
    assert not torch.allclose(delta, delta[0].expand_as(delta))

    with torch.no_grad():
        policy.film.weight.zero_()
        policy.profile_embeddings.weight.zero_()
        privacy_index = policy.profile_index[privacy.name]
        policy.profile_embeddings.weight[privacy_index, 0] = 1.0
        policy.film.weight[count_row, 0] = 1.0
    lp_identity = policy.log_probs(
        policy.begin_document(document, privacy), decision, menu, privacy,
    )
    lp_zero_identity = policy.log_probs(
        policy.begin_document(document, utility), decision, menu, utility,
    )
    assert not torch.allclose(lp_identity, lp_zero_identity)


def test_selected_action_gru_state_changes_later_relative_logits():
    policy, document, _ = _policy()
    profile = _profiles()[1]
    first, second = document.policy_decisions
    second_menu = tuple(action.action_id for action in second.actions)
    initial = policy.begin_document(document, profile)

    after_level = policy.advance(initial, first, "alpha-level-0")
    after_placeholder = policy.advance(initial, first, "alpha-placeholder")
    level_lp = policy.log_probs(after_level, second, second_menu, profile)
    placeholder_lp = policy.log_probs(
        after_placeholder, second, second_menu, profile,
    )

    assert not torch.allclose(after_level.hidden, after_placeholder.hidden)
    assert not torch.allclose(level_lp, placeholder_lp)
    delta = level_lp - placeholder_lp
    assert not torch.allclose(delta, delta[0].expand_as(delta))


def test_unsupported_or_mid_document_profile_changes_are_rejected():
    LambdaProfile, _, _ = _symbols()
    policy, document, _ = _policy()
    supported = _profiles()[1]
    unsupported = LambdaProfile("not-in-menu", 1.0)
    with pytest.raises(ValueError, match="unsupported lambda profile"):
        policy.begin_document(document, unsupported)

    state = policy.begin_document(document, supported)
    decision = document.policy_decisions[0]
    menu = tuple(action.action_id for action in decision.actions)
    with pytest.raises(ValueError, match="profile changed within document"):
        policy.log_probs(state, decision, menu, _profiles()[2])


def test_policy_sample_returns_a_stable_legal_action_id_and_log_probability():
    policy, document, _ = _policy()
    profile = _profiles()[1]
    decision = document.policy_decisions[0]
    menu = tuple(action.action_id for action in decision.actions)
    state = policy.begin_document(document, profile)

    action_id, log_probability = policy.sample(
        state, decision, menu, profile, greedy=True,
    )

    assert action_id in menu
    assert log_probability.ndim == 0
    assert torch.isfinite(log_probability)


def test_conditional_policy_integrates_sampling_replay_and_exact_count_loss():
    policy, document, _ = _policy()
    profile = _profiles()[2]
    generator = torch.Generator().manual_seed(19)

    trajectory = sample_trajectory(
        policy, document, profile, greedy=False, generator=generator,
    )
    replayed = replay_trajectory(policy, document, trajectory, profile)
    loss = expected_count_loss(
        replayed.steps,
        policy.count_reward,
        lambda_value=profile.value,
        decision_count=len(document.policy_decisions),
        rollout_count=1,
    )

    assert len(trajectory.steps) == len(document.policy_decisions) == 2
    assert len(replayed.steps) == 2
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    gradients = [
        parameter.grad for parameter in policy.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_legacy_ranker_policy_contract_and_feature_dimension_are_unchanged():
    span = {
        "type": "LOC",
        "actions": [
            {"mode": "level", "fill": "region", "p6": 0.4, "aset": 10.0},
            {"mode": "placeholder", "fill": None, "p6": 0.0},
        ],
    }
    features = ranker.action_features(span, floor=1.0)
    legacy = ranker.RankerPolicy(hidden=8)

    assert features.shape == (2, ranker.N_FEAT)
    assert list(legacy.state_dict()) == [
        "net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias",
        "net.4.weight", "net.4.bias",
    ]
    assert legacy.log_probs(features, [0, 1]).shape == (2,)


def test_warm_start_imports_single_profile_weights_then_clones_exit_winner_everywhere():
    from cloak.train.interactive_ranker import initialize_hybrid_warm_start

    LambdaProfile, _, ConditionalRankerPolicy = _symbols()
    document = _document()
    source = ConditionalRankerPolicy(
        count_reward=_count_reward(document),
        supported_profiles=(LambdaProfile("utility", 0.0),),
        max_menu_value=0.0,
        environment_hash="sha256:fixture-environment",
        encoder_pin="stub-encoder-v1",
        encoder=StubEncoder(),
        hidden_dim=12,
    )
    with torch.no_grad():
        source.head[-1].bias.fill_(0.25)
    target, _, _ = _policy()
    winner = {
        "artifact_version": "ranker-v2-exit-winners-v1",
        "documents": [{
            "doc_id": document.doc_id,
            "verification_status": "verified",
            "winner": {
                "action_vector": {
                    "alpha": "alpha-level-1",
                    "beta": "beta-level-1",
                },
            },
        }],
    }
    optimizer = torch.optim.SGD(target.parameters(), lr=0.05)

    result = initialize_hybrid_warm_start(
        target,
        (document,),
        _profiles(),
        bc_state_dict=source.state_dict(),
        exit_winners=winner,
        optimizer=optimizer,
    )

    assert result.identity_verified is True
    assert result.verified_winner_count == 1
    assert result.clone_target_count == len(_profiles()) * 2
    assert result.clone_loss > 0.0
    assert torch.equal(target.head[-1].bias, source.head[-1].bias)
    assert set(result.reference_state_dict) == set(target.state_dict())
    saved = result.reference_state_dict["head.2.bias"].clone()
    with torch.no_grad():
        target.head[-1].bias.add_(1.0)
    assert torch.equal(result.reference_state_dict["head.2.bias"], saved)


def test_warm_start_rejects_nonidentical_profiles_before_exit_updates():
    from cloak.train.interactive_ranker import assert_exact_profile_identity

    policy, document, _ = _policy()
    count_row = (
        policy.action_feature_offset
        + policy.action_feature_names.index("count_score")
    )
    with torch.no_grad():
        policy.film.weight[count_row, policy.profile_embedding_dim] = 1.0

    with pytest.raises(ValueError, match="profile identity"):
        assert_exact_profile_identity(policy, (document,), _profiles())


def test_policy_architecture_pin_binds_layout_not_learned_weights():
    from cloak.train.interactive_ranker import policy_architecture_pin

    first, _, _ = _policy()
    second, _, _ = _policy()
    assert policy_architecture_pin(first) == policy_architecture_pin(second)
    with torch.no_grad():
        second.head[-1].bias.add_(1.0)
    assert policy_architecture_pin(first) == policy_architecture_pin(second)

    _, _, ConditionalRankerPolicy = _symbols()
    document = _document()
    different = ConditionalRankerPolicy(
        count_reward=_count_reward(document),
        supported_profiles=_profiles(),
        max_menu_value=3.0,
        environment_hash="sha256:fixture-environment",
        encoder_pin="stub-encoder-v1",
        encoder=StubEncoder(),
        hidden_dim=10,
    )
    assert policy_architecture_pin(first) != policy_architecture_pin(different)
