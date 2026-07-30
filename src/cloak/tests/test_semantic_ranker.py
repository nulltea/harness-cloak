from __future__ import annotations

import inspect
import math
from dataclasses import fields, replace
from pathlib import Path
from types import MappingProxyType

import pytest
import torch

import cloak.ranker.semantic as semantic_ranker_module
from cloak.ranker.environment import LambdaProfile
from cloak.ranker.environment import (
    RankerAction,
    RankerDecision,
    RankerDocument,
)
from cloak.ranker.privacy import (
    DirectCountPrivacyProvider,
    PrivacyCheckpointContract,
    PrivacyPrediction,
    SemanticPrivacyHead,
    save_privacy_checkpoint,
)
from cloak.ranker.representation import DocumentTokenBank, RelationFeatures
from cloak.ranker.semantic import (
    CONTEXT_MODES,
    CURRENT_OCCURRENCE_ROLE,
    ORDINARY_ROLE,
    OTHER_CONTROLLED_ROLE,
    CandidateContextReadout,
    DecisionTokenFeatures,
    build_decision_token_features,
    signed_distance_bucket,
    stack_utility_relations,
)


def _fixture() -> tuple[DocumentTokenBank, RankerDocument, RankerDecision]:
    states = torch.tensor([
        [10.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    offsets = torch.tensor([
        [0, 5], [10, 15], [20, 25], [30, 35],
        [40, 45], [50, 55], [60, 65], [70, 75],
    ], dtype=torch.int64)
    bank = DocumentTokenBank(
        doc_id="fixture/doc",
        states=states,
        offsets=offsets,
        chunk_membership=((0,), (0,), (0, 1), (1,), (1,), (2,), (3,), (3,)),
    )
    occurrences = (
        MappingProxyType({
            "occurrence_id": "current-first",
            "decision_id": "current-decision",
            "start": 10,
            "end": 15,
            "surface": "aaaaa",
            "controlled": True,
        }),
        MappingProxyType({
            "occurrence_id": "other",
            "decision_id": "other-decision",
            "start": 30,
            "end": 35,
            "surface": "bbbbb",
            "controlled": True,
        }),
        MappingProxyType({
            "occurrence_id": "current-second",
            "decision_id": "current-decision",
            "start": 40,
            "end": 45,
            "surface": "ccccc",
            "controlled": True,
        }),
    )
    decision = RankerDecision(
        decision_id="current-decision",
        profile_id="TYPE:source",
        runtime_type="TYPE",
        canonical_key="source",
        occurrence_ids=("current-first", "current-second"),
        actions=(),
    )
    source_text = list("x" * 80)
    source_text[10:15] = "aaaaa"
    source_text[30:35] = "bbbbb"
    source_text[40:45] = "ccccc"
    document = RankerDocument(
        doc_id="fixture/doc",
        corpus="fixture",
        text="".join(source_text),
        occurrences=occurrences,
        policy_decisions=(decision,),
        fixed_decisions=(),
    )
    return bank, document, decision


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (-1000, 0), (-128, 0), (-127, 1), (-64, 1), (-63, 2), (-32, 2),
        (-31, 3), (-16, 3), (-15, 4), (-8, 4), (-7, 5), (-4, 5),
        (-3, 6), (-2, 6), (-1, 7), (0, 8), (1, 9), (2, 10), (3, 10),
        (4, 11), (7, 11), (8, 12), (15, 12), (16, 13), (31, 13),
        (32, 14), (63, 14), (64, 15), (127, 15), (128, 16), (1000, 16),
    ],
)
def test_token_features_use_exact_signed_distance_buckets(distance, expected):
    assert signed_distance_bucket(distance) == expected


def test_token_features_mark_all_occurrences_roles_distances_and_document_bins():
    bank, document, decision = _fixture()

    features = build_decision_token_features(bank, document, decision)

    assert isinstance(features, DecisionTokenFeatures)
    assert features.role_ids.dtype == torch.int64
    assert features.relative_position_ids.dtype == torch.int64
    assert features.document_position_ids.dtype == torch.int64
    assert features.role_ids.tolist() == [
        ORDINARY_ROLE,
        CURRENT_OCCURRENCE_ROLE,
        ORDINARY_ROLE,
        OTHER_CONTROLLED_ROLE,
        CURRENT_OCCURRENCE_ROLE,
        ORDINARY_ROLE,
        ORDINARY_ROLE,
        ORDINARY_ROLE,
    ]
    assert features.relative_position_ids.tolist() == [7, 8, 9, 7, 8, 9, 10, 10]
    assert features.document_position_ids.tolist() == [0, 2, 4, 6, 8, 10, 12, 14]
    assert features.occurrence_token_indices == ((1,), (4,))


def _identity_attention(module: torch.nn.MultiheadAttention) -> None:
    dimension = module.embed_dim
    with torch.no_grad():
        module.in_proj_weight.zero_()
        module.in_proj_bias.zero_()
        for block in range(3):
            start = block * dimension
            module.in_proj_weight[start:start + dimension].copy_(torch.eye(dimension))
        module.out_proj.weight.copy_(torch.eye(dimension))
        module.out_proj.bias.zero_()


def _readout(
    context_mode: str = "full-candidate-attention",
) -> CandidateContextReadout:
    module = CandidateContextReadout(
        token_dim=4,
        relation_dim=4,
        context_dim=4,
        num_heads=1,
        context_mode=context_mode,
        dropout=0.0,
    )
    with torch.no_grad():
        module.token_projection.weight.copy_(torch.eye(4))
        module.token_projection.bias.zero_()
        module.query_projection.weight.copy_(torch.eye(4))
        module.query_projection.bias.zero_()
        module.role_embedding.weight.zero_()
        module.role_embedding.weight[CURRENT_OCCURRENCE_ROLE, 2] = 5.0
        module.relative_position_embedding.weight.zero_()
        module.document_position_embedding.weight.zero_()
        module.occurrence_position_embedding.weight.zero_()
        module.context_projection.weight.zero_()
        module.context_projection.bias.zero_()
        for branch in range(3):
            start = branch * 4
            module.context_projection.weight[:, start:start + 4].copy_(torch.eye(4))
    _identity_attention(module.target_attention)
    _identity_attention(module.local_attention)
    _identity_attention(module.global_attention)
    module.eval()
    return module


def _queries() -> torch.Tensor:
    return torch.tensor([[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]])


def test_context_stacks_task2_relation_pairs_for_one_decision():
    relations = tuple(
        RelationFeatures(
            decision_id="current-decision",
            action_id=f"action-{index}",
            type_mean=torch.zeros(1),
            source_mean=torch.zeros(1),
            candidate_mean=torch.zeros(1),
            pair=torch.tensor([float(index), 1.0, 2.0, 3.0]),
            candidate_only=torch.zeros(1),
            independent_pair=torch.zeros(1),
        )
        for index in range(2)
    )

    stacked = stack_utility_relations(relations)

    assert torch.equal(stacked, torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 1.0, 2.0, 3.0],
    ]))


def test_context_module_contains_no_prohibited_supervision_inputs():
    source = Path(semantic_ranker_module.__file__).read_text().casefold()
    assert "assertion_id" not in source
    assert "dependency" not in source


def test_context_candidate_relation_changes_attention_and_utility_context():
    bank, document, decision = _fixture()
    features = build_decision_token_features(bank, document, decision)
    module = _readout()

    contexts, weights = module.forward_with_attention(bank, features, _queries())

    assert contexts.shape == (2, 4)
    assert not torch.equal(contexts[0], contexts[1])
    assert not torch.equal(weights["global"][0], weights["global"][1])
    assert not torch.equal(weights["target"][0], weights["target"][1])


def test_context_distant_evidence_changes_only_candidate_that_queries_it():
    bank, document, decision = _fixture()
    features = build_decision_token_features(bank, document, decision)
    module = _readout()
    baseline = module(bank, features, _queries())
    changed_states = bank.states.clone()
    changed_states[6, 1] = 20.0
    changed = module(replace(bank, states=changed_states), features, _queries())

    assert torch.allclose(changed[0], baseline[0], atol=1e-6, rtol=0)
    assert not torch.allclose(changed[1], baseline[1], atol=1e-3, rtol=0)


def test_context_target_branch_uses_every_occurrence_and_candidate_conditioning():
    bank, document, decision = _fixture()
    features = build_decision_token_features(bank, document, decision)
    module = _readout()

    _, weights = module.forward_with_attention(bank, features, _queries())

    assert weights["target"].shape == (2, 2)
    assert torch.all(weights["target"] > 0)
    assert weights["target"][0, 0] > weights["target"][0, 1]
    assert weights["target"][1, 1] > weights["target"][1, 0]


def test_context_removing_target_markers_changes_target_summary():
    bank, document, decision = _fixture()
    features = build_decision_token_features(bank, document, decision)
    module = _readout()
    marked, _ = module.forward_with_attention(bank, features, _queries())
    unmarked_features = replace(
        features, role_ids=torch.full_like(features.role_ids, ORDINARY_ROLE)
    )
    unmarked, _ = module.forward_with_attention(
        bank, unmarked_features, _queries()
    )

    assert not torch.equal(marked, unmarked)


def test_context_overlap_membership_never_duplicates_attention_rows():
    bank, document, decision = _fixture()
    features = build_decision_token_features(bank, document, decision)
    module = _readout()
    duplicated, weights = module.forward_with_attention(bank, features, _queries())
    memberships = list(bank.chunk_membership)
    memberships[2] = (0,)
    single_membership = replace(bank, chunk_membership=tuple(memberships))
    single = module(single_membership, features, _queries())

    assert weights["global"].shape[-1] == bank.states.shape[0]
    assert weights["local"].shape[-1] == 5
    assert torch.equal(duplicated, single)


def test_context_projects_document_bank_once_for_all_candidates():
    bank, document, decision = _fixture()
    features = build_decision_token_features(bank, document, decision)
    module = _readout()
    calls = []
    handle = module.token_projection.register_forward_hook(
        lambda _module, inputs, _output: calls.append(inputs[0].shape)
    )
    try:
        module(bank, features, _queries())
    finally:
        handle.remove()

    assert calls == [bank.states.shape]


@pytest.mark.parametrize("context_mode", CONTEXT_MODES)
def test_context_modes_share_output_width_and_frozen_inputs(context_mode):
    bank, document, decision = _fixture()
    features = build_decision_token_features(bank, document, decision)
    module = _readout(context_mode)

    output = module(bank, features, _queries())

    assert output.shape == (2, 4)
    assert module.context_projection.in_features == 12
    assert module.context_projection.out_features == 4


def test_context_modes_are_explicit_ablation_surfaces_and_production_is_full_only():
    bank, document, decision = _fixture()
    features = build_decision_token_features(bank, document, decision)
    changed_states = bank.states.clone()
    changed_states[6, 1] = 20.0
    changed_bank = replace(bank, states=changed_states)

    for mode in ("local-cls-mean", "target-bidirectional"):
        module = _readout(mode)
        assert torch.equal(
            module(bank, features, _queries()),
            module(changed_bank, features, _queries()),
        )
        with pytest.raises(ValueError, match="production"):
            CandidateContextReadout(
                token_dim=4,
                relation_dim=4,
                context_dim=4,
                num_heads=1,
                context_mode=mode,
                production=True,
            )

    production = CandidateContextReadout(
        token_dim=4,
        relation_dim=4,
        context_dim=4,
        num_heads=1,
        context_mode="full-candidate-attention",
        production=True,
    )
    assert production.context_mode == "full-candidate-attention"


def _memory(history_mode: str = "selected-cross-attention"):
    memory = semantic_ranker_module.SelectedActionMemory(
        relation_dim=4,
        context_dim=4,
        history_dim=4,
        num_heads=1,
        action_mode_count=3,
        runtime_type_count=4,
        history_mode=history_mode,
        dropout=0.0,
    )
    with torch.no_grad():
        memory.record_projection.weight.zero_()
        memory.record_projection.bias.zero_()
        memory.record_projection.weight[:, :4].copy_(torch.eye(4))
        memory.query_projection.weight.zero_()
        memory.query_projection.bias.zero_()
        memory.query_projection.weight[:, :4].copy_(torch.eye(4))
        memory.action_mode_embedding.weight.zero_()
        memory.runtime_type_embedding.weight.zero_()
        memory.source_position_embedding.weight.zero_()
    _identity_attention(memory.cross_attention)
    memory.eval()
    return memory


def _record(utility_relation: torch.Tensor):
    return semantic_ranker_module.SelectedActionRecord(
        utility_relation=utility_relation,
        action_mode_id=1,
        runtime_type_id=2,
        source_position_pool=torch.zeros(4),
    )


def test_memory_empty_prefix_and_none_history_are_exact_zero():
    queries = torch.randn(3, 8)
    selected = _memory()
    no_history = _memory("none")
    expected = torch.zeros(3, 4)

    assert torch.equal(selected(queries, ()), expected)
    assert torch.equal(no_history(queries, (_record(torch.ones(4)),)), expected)


def test_memory_rows_are_permutation_invariant_within_one_causal_prefix():
    memory = _memory()
    queries = torch.tensor([
        [20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    records = (
        _record(torch.tensor([1.0, 0.0, 0.0, 0.0])),
        _record(torch.tensor([0.0, 1.0, 0.0, 0.0])),
        _record(torch.tensor([0.0, 0.0, 1.0, 0.0])),
    )

    forward = memory(queries, records)
    reversed_rows = memory(queries, tuple(reversed(records)))

    assert torch.allclose(forward, reversed_rows, atol=1e-6, rtol=1e-6)


def test_memory_intervention_is_candidate_conditioned_and_selective():
    memory = _memory()
    query = torch.tensor([
        [20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])
    relevant = _record(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    unrelated = _record(torch.tensor([0.0, 1.0, 0.0, 0.0]))
    baseline = memory(query, (relevant, unrelated))
    relevant_changed = memory(
        query,
        (_record(torch.tensor([2.0, 0.0, 0.0, 0.0])), unrelated),
    )
    unrelated_changed = memory(
        query,
        (relevant, _record(torch.tensor([0.0, 2.0, 0.0, 0.0]))),
    )

    assert not torch.allclose(relevant_changed, baseline, atol=0.5, rtol=0)
    assert torch.allclose(unrelated_changed, baseline, atol=1e-3, rtol=0)


def test_memory_repeated_surface_decision_builds_one_record_with_all_positions():
    bank, document, decision = _fixture()
    features = build_decision_token_features(bank, document, decision)
    memory = _memory()
    with torch.no_grad():
        rows = torch.arange(16, dtype=torch.float32).unsqueeze(1).expand(-1, 4)
        memory.source_position_embedding.weight.copy_(rows)

    record = memory.build_record(
        utility_relation=torch.ones(4),
        action_mode_id=1,
        runtime_type_id=2,
        decision_features=features,
    )

    assert isinstance(record, semantic_ranker_module.SelectedActionRecord)
    assert record.source_position_pool.shape == (4,)
    assert torch.equal(record.source_position_pool, torch.full((4,), 5.0))


@pytest.mark.parametrize(
    "history_mode",
    semantic_ranker_module.HISTORY_MODES,
)
def test_history_modes_share_output_width_and_record_fields(history_mode):
    memory = _memory(history_mode)
    output = memory(
        torch.randn(2, 8),
        (_record(torch.ones(4)), _record(torch.arange(4, dtype=torch.float32))),
    )

    assert output.shape == (2, 4)
    assert memory.record_projection.in_features == 16
    assert memory.record_projection.out_features == 4


def test_history_production_accepts_selected_cross_attention_only():
    for history_mode in ("none", "utility-gru"):
        with pytest.raises(ValueError, match="production"):
            semantic_ranker_module.SelectedActionMemory(
                relation_dim=4,
                context_dim=4,
                history_dim=4,
                num_heads=1,
                action_mode_count=3,
                runtime_type_count=4,
                history_mode=history_mode,
                production=True,
            )


def test_memory_has_only_allowed_fields_one_cross_attention_and_no_step_state():
    assert [field.name for field in fields(
        semantic_ranker_module.SelectedActionRecord
    )] == [
        "utility_relation",
        "action_mode_id",
        "runtime_type_id",
        "source_position_pool",
    ]
    memory = _memory()
    attentions = [
        module
        for module in memory.modules()
        if isinstance(module, torch.nn.MultiheadAttention)
    ]
    assert attentions == [memory.cross_attention]
    assert all("step" not in name for name, _ in memory.named_parameters())
    source = "\n".join((
        inspect.getsource(semantic_ranker_module.SelectedActionRecord),
        inspect.getsource(semantic_ranker_module.SelectedActionMemory),
    )).casefold()
    for prohibited in (
        "_decision_action_inputs",
        "predicted_privacy",
        "authored_index",
        "authored_level_index",
        "menu_size",
        "profile_id",
        "profile_identity",
        "assertion_id",
        "qa_assertion",
        "dependency_id",
        "qa_dependency",
        "lambda",
    ):
        assert prohibited not in source


def test_count_privacy_only_loss_has_no_gradient_path_to_memory_parameters():
    memory = _memory()
    shared_relation = torch.ones(1, 4, requires_grad=True)
    record = _record(shared_relation.squeeze(0))
    memory(torch.cat([shared_relation, torch.zeros(1, 4)], dim=-1), (record,))
    privacy_projection = torch.nn.Linear(4, 1)

    privacy_projection(shared_relation).sum().backward()

    assert all(parameter.grad is None for parameter in memory.parameters())
    assert shared_relation.grad is not None


def _ordered_document() -> RankerDocument:
    decisions = tuple(
        RankerDecision(
            decision_id=decision_id,
            profile_id=f"TYPE:{decision_id}",
            runtime_type="TYPE",
            canonical_key=decision_id,
            occurrence_ids=(f"occurrence-{decision_id}",),
            actions=(),
        )
        for decision_id in ("first", "second", "third", "fourth")
    )
    return RankerDocument(
        doc_id="ordered/doc",
        corpus="fixture",
        text="ordered",
        occurrences=(),
        policy_decisions=decisions,
        fixed_decisions=(),
    )


def test_order_helpers_preserve_frozen_production_first_occurrence_walk():
    document = _ordered_document()
    frozen = document.policy_decisions

    assert semantic_ranker_module.production_decision_order(document) is frozen
    assert semantic_ranker_module.reverse_diagnostic_decision_order(
        document
    ) == tuple(reversed(frozen))
    first_seeded = semantic_ranker_module.seeded_diagnostic_decision_order(
        document, seed=17
    )
    second_seeded = semantic_ranker_module.seeded_diagnostic_decision_order(
        document, seed=17
    )
    assert first_seeded == second_seeded
    assert set(first_seeded) == set(frozen)
    assert document.policy_decisions is frozen
    assert semantic_ranker_module.production_decision_order(document) is frozen


class _StubRepresentationStore:
    def __init__(
        self,
        bank: DocumentTokenBank,
        relations: tuple[RelationFeatures, ...],
    ):
        self.bank = bank
        self.relations = {
            (relation.decision_id, relation.action_id): relation
            for relation in relations
        }
        self.target_counts = {
            relation.action_id: index + 1
            for index, relation in enumerate(relations)
        }

    def document(self, doc_id: str) -> DocumentTokenBank:
        if doc_id != self.bank.doc_id:
            raise KeyError(doc_id)
        return self.bank

    def relation(self, decision_id: str, action_id: str) -> RelationFeatures:
        return self.relations[(decision_id, action_id)]


class _StubPrivacyHead(torch.nn.Module):
    pair_dim = 4

    def __init__(self):
        super().__init__()
        self.privacy_projection = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            self.privacy_projection.weight.copy_(
                torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            )

    def forward(self, pair_features, *, count_basis=None):
        assert count_basis is None
        raw = self.privacy_projection(pair_features).squeeze(-1)
        return PrivacyPrediction(
            mu_log_count=torch.nn.functional.softplus(raw),
            sigma_log_count=torch.full_like(raw, 0.3),
        )


def _semantic_fixture():
    bank, document, original = _fixture()
    document = replace(
        document,
        occurrences=tuple(
            MappingProxyType({
                **occurrence,
                "controlled": False,
                "decision_id": None,
            })
            if occurrence["occurrence_id"] == "other"
            else occurrence
            for occurrence in document.occurrences
        ),
    )
    actions = (
        RankerAction(
            action_id="current-level-low",
            mode="level",
            fill="broad place",
            authored_level_index=0,
            runtime_type="LOC",
        ),
        RankerAction(
            action_id="current-level-high",
            mode="level",
            fill="specific place",
            authored_level_index=1,
            runtime_type="LOC",
        ),
        RankerAction(
            action_id="current-keep",
            mode="keep",
            fill="source",
            authored_level_index=None,
            runtime_type="LOC",
        ),
        RankerAction(
            action_id="current-placeholder",
            mode="placeholder",
            fill=None,
            authored_level_index=None,
            runtime_type="LOC",
        ),
    )
    decision = replace(
        original,
        profile_id="LOC:source",
        runtime_type="LOC",
        actions=actions,
    )
    document = replace(document, policy_decisions=(decision,))
    pair_rows = (
        torch.tensor([0.25, 1.0, 0.0, 0.0]),
        torch.tensor([1.50, 0.0, 1.0, 0.0]),
        torch.tensor([4.00, 0.0, 0.0, 1.0]),
        torch.tensor([-2.0, 1.0, 1.0, 0.0]),
    )
    relations = tuple(
        RelationFeatures(
            decision_id=decision.decision_id,
            action_id=action.action_id,
            type_mean=torch.zeros(1),
            source_mean=torch.zeros(1),
            candidate_mean=torch.zeros(1),
            pair=pair,
            candidate_only=torch.zeros(1),
            independent_pair=torch.zeros(1),
        )
        for action, pair in zip(actions, pair_rows, strict=True)
    )
    return _StubRepresentationStore(bank, relations), document, decision


def _semantic_policy(*, privacy_head=None):
    store, document, decision = _semantic_fixture()
    profiles = (
        LambdaProfile("utility", 0.0),
        LambdaProfile("balanced", 1.0),
        LambdaProfile("privacy", 3.0),
    )
    torch.manual_seed(31)
    policy = semantic_ranker_module.SemanticRankerPolicy(
        representation_store=store,
        privacy_head=privacy_head or _StubPrivacyHead(),
        supported_profiles=profiles,
        max_lambda=3.0,
        token_dim=4,
        pair_dim=4,
        relation_dim=4,
        context_dim=4,
        history_dim=4,
        utility_hidden_dim=8,
        num_heads=1,
        runtime_types=("LOC",),
        dropout=0.0,
    )
    policy.eval()
    return policy, store, document, decision, profiles


def _direct_count_policy():
    store, document, decision = _semantic_fixture()
    scores = (0.2, 0.7, 0.0, 1.0)
    artifact = {
        "artifact_hash": "sha256:direct-targets",
        "environment_hash": "sha256:environment",
        "action_targets": {
            action.action_id: {
                "action_id": action.action_id,
                "decision_id": decision.decision_id,
                "mode": action.mode,
                "profile_score": score,
                "log_count": 1.0 if action.mode == "level" else None,
                "grounding_status": (
                    "certifying" if action.mode == "level" else None
                ),
                "profile_id": decision.profile_id,
            }
            for action, score in zip(decision.actions, scores, strict=True)
        },
    }
    profiles = (
        LambdaProfile("utility", 0.0),
        LambdaProfile("privacy", 3.0),
    )
    torch.manual_seed(31)
    policy = semantic_ranker_module.SemanticRankerPolicy.from_direct_count_targets(
        representation_store=store,
        profile_count_targets=artifact,
        supported_profiles=profiles,
        max_lambda=3.0,
        token_dim=4,
        pair_dim=4,
        relation_dim=4,
        context_dim=4,
        history_dim=4,
        utility_hidden_dim=8,
        num_heads=1,
        runtime_types=("LOC",),
        dropout=0.0,
    )
    policy.eval()
    return policy, document, decision, profiles, scores


def _full_menu(decision: RankerDecision) -> tuple[str, ...]:
    return tuple(action.action_id for action in decision.actions)


def test_policy_interfaces_and_zero_lambda_are_exact_identities():
    policy, _, document, decision, profiles = _semantic_policy()
    state = policy.begin_document(document, profiles[0])
    distribution = policy.distribution(
        state, decision, _full_menu(decision), profiles[0]
    )

    assert [field.name for field in fields(
        semantic_ranker_module.SemanticPolicyState
    )] == ["document", "profile", "selected_records"]
    assert [field.name for field in fields(
        semantic_ranker_module.ActionDistribution
    )] == [
        "action_ids",
        "utility_logits",
        "mu_log_count",
        "sigma_log_count",
        "predicted_privacy",
        "combined_logits",
        "log_probs",
        "count_log_probs",
    ]
    assert distribution.combined_logits is distribution.utility_logits
    assert torch.equal(distribution.combined_logits, distribution.utility_logits)
    assert torch.equal(
        distribution.log_probs,
        torch.log_softmax(distribution.utility_logits, dim=0),
    )
    assert torch.equal(
        policy.log_probs(state, decision, _full_menu(decision), profiles[0]),
        distribution.log_probs,
    )
    assert policy.alpha.item() >= 0.0
    assert policy.alpha.item() == pytest.approx(1.0)


def test_policy_count_metadata_mutation_after_inference_is_a_noop_shortcut():
    policy, store, document, decision, profiles = _semantic_policy()
    state = policy.begin_document(document, profiles[1])
    before = policy.distribution(
        state, decision, _full_menu(decision), profiles[1]
    )

    store.target_counts = {
        action_id: 10_000 - value
        for action_id, value in store.target_counts.items()
    }
    after = policy.distribution(
        state, decision, _full_menu(decision), profiles[1]
    )

    assert torch.equal(after.utility_logits, before.utility_logits)
    assert torch.equal(after.predicted_privacy, before.predicted_privacy)
    assert torch.equal(after.combined_logits, before.combined_logits)


def test_policy_authored_index_mutation_cannot_change_semantic_predictions():
    policy, _, document, decision, profiles = _semantic_policy()
    baseline = policy.distribution(
        policy.begin_document(document, profiles[1]),
        decision,
        _full_menu(decision),
        profiles[1],
    )
    changed_actions = tuple(
        replace(
            action,
            authored_level_index=(
                100 - int(action.authored_level_index)
                if action.authored_level_index is not None
                else None
            ),
        )
        for action in decision.actions
    )
    changed_decision = replace(decision, actions=changed_actions)
    changed_document = replace(
        document, policy_decisions=(changed_decision,)
    )
    changed = policy.distribution(
        policy.begin_document(changed_document, profiles[1]),
        changed_decision,
        _full_menu(changed_decision),
        profiles[1],
    )

    assert torch.equal(changed.utility_logits, baseline.utility_logits)
    assert torch.equal(changed.mu_log_count, baseline.mu_log_count)
    assert torch.equal(changed.sigma_log_count, baseline.sigma_log_count)


def test_lambda_changes_only_detached_additive_controller_term():
    policy, _, document, decision, profiles = _semantic_policy()
    rows = [
        policy.distribution(
            policy.begin_document(document, profile),
            decision,
            _full_menu(decision),
            profile,
        )
        for profile in profiles
    ]

    assert all(torch.equal(rows[0].utility_logits, row.utility_logits) for row in rows)
    assert all(
        torch.equal(rows[0].predicted_privacy, row.predicted_privacy)
        for row in rows
    )
    for profile, row in zip(profiles[1:], rows[1:], strict=True):
        magnitude = math.log1p(profile.value) / math.log1p(policy.max_lambda)
        expected = (
            row.utility_logits
            + policy.alpha * magnitude * row.predicted_privacy.detach()
        )
        assert torch.equal(row.combined_logits, expected)


def test_direct_count_policy_composes_exact_artifact_scores():
    from cloak.ranker.interactive import _semantic_policy_contract

    policy, document, decision, profiles, scores = _direct_count_policy()
    policy.encoder_revision = "stub-revision"

    row = policy.distribution(
        policy.begin_document(document, profiles[1]),
        decision,
        _full_menu(decision),
        profiles[1],
    )

    expected_scores = torch.tensor(scores, dtype=row.utility_logits.dtype)
    expected_logits = row.utility_logits + policy.alpha * expected_scores
    assert isinstance(policy.privacy_provider, DirectCountPrivacyProvider)
    assert policy.privacy_head is None
    assert torch.equal(row.predicted_privacy, expected_scores)
    assert torch.equal(row.combined_logits, expected_logits)
    assert policy.privacy_signal == {
        "kind": "direct-count",
        "targets_artifact_hash": "sha256:direct-targets",
        "environment_hash": "sha256:environment",
    }
    assert _semantic_policy_contract(policy)["privacy_signal"] == (
        policy.privacy_signal
    )


def test_policy_complete_menu_privacy_normalization_precedes_legal_masking():
    policy, _, document, decision, profiles = _semantic_policy()
    state = policy.begin_document(document, profiles[1])
    full = policy.distribution(state, decision, _full_menu(decision), profiles[1])
    masked_ids = (
        "current-level-low",
        "current-keep",
        "current-placeholder",
    )
    masked = policy.distribution(state, decision, masked_ids, profiles[1])

    low_index = full.action_ids.index("current-level-low")
    assert full.predicted_privacy[low_index] < 1.0
    assert torch.equal(masked.predicted_privacy[0], full.predicted_privacy[low_index])
    assert masked.predicted_privacy[1].item() == 0.0
    assert masked.predicted_privacy[2].item() == 1.0


def test_policy_expected_predicted_privacy_is_locally_lambda_monotone():
    policy, _, document, decision, profiles = _semantic_policy()
    expected_privacy = []
    for profile in profiles:
        row = policy.distribution(
            policy.begin_document(document, profile),
            decision,
            _full_menu(decision),
            profile,
        )
        expected_privacy.append(
            float(
                (row.log_probs.exp() * row.predicted_privacy).sum().detach()
            )
        )

    assert expected_privacy == sorted(expected_privacy)


def test_policy_count_gradient_view_updates_only_global_controller():
    policy, _, document, decision, profiles = _semantic_policy()
    row = policy.distribution(
        policy.begin_document(document, profiles[1]),
        decision,
        _full_menu(decision),
        profiles[1],
    )

    (-row.count_log_probs[-1]).backward()

    assert policy.alpha_raw.grad is not None
    assert torch.isfinite(policy.alpha_raw.grad)
    assert all(
        parameter.grad is None
        for name, parameter in policy.named_parameters()
        if name != "alpha_raw"
    )


def test_policy_advance_appends_one_selected_record_and_protocol_replays():
    from cloak.ranker.interactive import replay_trajectory, sample_trajectory

    policy, _, document, decision, profiles = _semantic_policy()
    initial = policy.begin_document(document, profiles[1])
    advanced = policy.advance(initial, decision, "current-level-low")

    assert initial.selected_records == ()
    assert len(advanced.selected_records) == 1
    assert advanced.document is initial.document
    generator = torch.Generator().manual_seed(5)
    trajectory = sample_trajectory(
        policy, document, profiles[1], greedy=False, generator=generator
    )
    replayed = replay_trajectory(policy, document, trajectory, profiles[1])
    assert len(trajectory.steps) == len(replayed.steps) == 1
    assert replayed.steps[0].log_probs.shape == (4,)


def test_policy_state_dict_has_no_legacy_or_shortcut_parameters():
    policy, _, _, _, _ = _semantic_policy()
    names = tuple(policy.state_dict())
    prohibited = (
        "profile_embedding",
        "film",
        "count_feature",
        "menu_size",
        "authored",
        "gru",
    )

    assert not any(token in name.casefold() for name in names for token in prohibited)
    assert policy.utility_projection.in_features == 4
    assert policy.utility_head[0].in_features == 24
    assert (
        policy.utility_projection
        is not policy.privacy_head.privacy_projection
    )
    assert policy.memory.history_mode == "selected-cross-attention"
    assert not any(isinstance(module, torch.nn.GRU) for module in policy.modules())
    for method in (policy.distribution, policy.log_probs, policy.advance):
        argument_names = tuple(inspect.signature(method).parameters)
        assert not any(
            token in argument.casefold()
            for argument in argument_names
            for token in (
                "count",
                "authored",
                "menu_size",
                "profile_id",
                "assertion",
                "dependency",
            )
        )
    assert all(
        not parameter.requires_grad
        for parameter in policy.privacy_head.parameters()
    )
    assert policy.privacy_head.training is False


def _privacy_contract() -> PrivacyCheckpointContract:
    return PrivacyCheckpointContract(
        environment_hash="sha256:environment",
        profile_target_artifact_hash="sha256:targets",
        representation_manifest_hash="sha256:representations",
        encoder_revision="stub-revision",
        split_manifest_hash="sha256:splits",
        pair_dim=4,
        projection_dim=1,
        hidden_dim=0,
        count_basis_size=0,
        count_basis_categories=(),
        rho=0.0,
        gamma=1.0,
        seeds=(3, 5, 7),
        training_seed=5,
        metric_report_hash="sha256:metrics",
        diagnostic_manifest_hash="sha256:diagnostics",
        counterexample_set_hash="sha256:counterexamples",
        run_protocol="promotion",
        seed_count=3,
        promotion_verdict="PROMOTE",
        target_mean=1.0,
        target_std=0.5,
        sigma_fixed=0.7,
        feature_schema="type-source-candidate-hadamard-v1",
        optimizer="AdamW",
        learning_rate=3e-4,
        weight_decay=0.01,
        gradient_clip=1.0,
    )


def test_policy_factory_loads_and_freezes_validated_privacy_checkpoint(tmp_path):
    store, document, decision = _semantic_fixture()
    contract = _privacy_contract()
    source = SemanticPrivacyHead(
        pair_dim=4, projection_dim=1, hidden_dim=0
    )
    with torch.no_grad():
        source.privacy_projection.weight.fill_(0.125)
        source.target_mean.fill_(contract.target_mean)
        source.target_std.fill_(contract.target_std)
        source.sigma_fixed.fill_(contract.sigma_fixed)
    checkpoint = tmp_path / "privacy.pt"
    save_privacy_checkpoint(checkpoint, source, contract)

    policy = semantic_ranker_module.SemanticRankerPolicy.from_privacy_checkpoint(
        representation_store=store,
        privacy_checkpoint=checkpoint,
        privacy_checkpoint_contract=contract,
        supported_profiles=(
            LambdaProfile("utility", 0.0),
            LambdaProfile("privacy", 3.0),
        ),
        max_lambda=3.0,
        token_dim=4,
        relation_dim=4,
        context_dim=4,
        history_dim=4,
        utility_hidden_dim=8,
        num_heads=1,
        runtime_types=("LOC",),
        dropout=0.0,
    )

    assert torch.equal(
        policy.privacy_head.privacy_projection.weight,
        source.privacy_projection.weight,
    )
    assert all(
        not parameter.requires_grad
        for parameter in policy.privacy_head.parameters()
    )
    row = policy.distribution(
        policy.begin_document(document, LambdaProfile("utility", 0.0)),
        decision,
        _full_menu(decision),
        LambdaProfile("utility", 0.0),
    )
    assert torch.isfinite(row.mu_log_count).all()


def test_policy_factory_rejects_iteration_checkpoint_without_explicit_override(
    tmp_path,
):
    store, _document, _decision = _semantic_fixture()
    contract = replace(
        _privacy_contract(),
        seeds=(3,),
        training_seed=3,
        seed_count=1,
        run_protocol="iteration",
        promotion_verdict="NEEDS_MULTI_SEED_EVIDENCE",
        counterexample_set_hash=None,
    )
    source = SemanticPrivacyHead(pair_dim=4, projection_dim=1, hidden_dim=0)
    with torch.no_grad():
        source.target_mean.fill_(contract.target_mean)
        source.target_std.fill_(contract.target_std)
        source.sigma_fixed.fill_(contract.sigma_fixed)
    checkpoint = tmp_path / "iteration.pt"
    save_privacy_checkpoint(checkpoint, source, contract)
    kwargs = dict(
        representation_store=store,
        privacy_checkpoint=checkpoint,
        privacy_checkpoint_contract=contract,
        supported_profiles=(
            LambdaProfile("utility", 0.0),
            LambdaProfile("privacy", 3.0),
        ),
        max_lambda=3.0,
        token_dim=4,
        relation_dim=4,
        context_dim=4,
        history_dim=4,
        utility_hidden_dim=8,
        num_heads=1,
        runtime_types=("LOC",),
        dropout=0.0,
    )

    with pytest.raises(ValueError, match="promotion"):
        semantic_ranker_module.SemanticRankerPolicy.from_privacy_checkpoint(
            **kwargs
        )
    policy = semantic_ranker_module.SemanticRankerPolicy.from_privacy_checkpoint(
        **kwargs, allow_development_privacy_checkpoint=True
    )
    assert policy.privacy_head.training is False


def test_policy_cli_requires_explicit_architecture_specific_artifacts():
    from scripts import train_interactive_ranker

    parser = train_interactive_ranker.build_parser()
    shared = [
        "bc",
        "--environment", "environment.json",
        "--utility-artifact", "utility.json",
        "--utility-cache", "utility.jsonl",
        "--out-checkpoint", "policy.pt",
    ]
    semantic = parser.parse_args([
        *shared,
        "--policy-architecture", "semantic-v1",
        "--representation-manifest", "representations.json",
        "--profile-count-targets", "targets.json",
    ])
    learned = parser.parse_args([
        *shared,
        "--policy-architecture", "semantic-v1",
        "--representation-manifest", "representations.json",
        "--privacy-checkpoint", "privacy.pt",
        "--profile-count-targets", "targets.json",
    ])

    assert semantic.policy_architecture == "semantic-v1"
    assert semantic.representation_manifest == "representations.json"
    assert semantic.privacy_checkpoint is None
    assert semantic.profile_count_targets == "targets.json"
    assert learned.privacy_checkpoint == "privacy.pt"
    with pytest.raises(SystemExit):
        parser.parse_args([*shared, "--profile-count-targets", "targets.json"])
    with pytest.raises(SystemExit):
        parser.parse_args([
            *shared,
            "--policy-architecture", "semantic-v1",
            "--representation-manifest", "representations.json",
        ])
    with pytest.raises(SystemExit):
        parser.parse_args([
            *shared,
            "--policy-architecture", "semantic-v1",
            "--profile-count-targets", "targets.json",
        ])


def test_policy_caches_static_decision_inputs_and_invalidates_on_apply():
    policy, store, document, decision, profiles = _semantic_policy()
    document_calls = []
    original_document = store.document

    def counting_document(doc_id):
        document_calls.append(doc_id)
        return original_document(doc_id)

    store.document = counting_document
    state = policy.begin_document(document, profiles[0])
    menu = tuple(action.action_id for action in decision.actions)

    first = policy.distribution(state, decision, menu, profiles[0])
    calls_after_first = len(document_calls)
    second = policy.distribution(state, decision, menu, profiles[0])

    assert calls_after_first >= 1
    assert len(document_calls) == calls_after_first
    assert torch.equal(first.log_probs, second.log_probs)

    policy.float()
    assert not policy._token_bank_cache
    assert not policy._pair_feature_cache
    assert not policy._decision_feature_cache
    third = policy.distribution(state, decision, menu, profiles[0])
    assert len(document_calls) > calls_after_first
    assert torch.equal(first.log_probs, third.log_probs)


def test_alpha_utility_routing_scales_alpha_gradient_only():
    policy, store, document, decision, profiles = _semantic_policy()
    menu = tuple(action.action_id for action in decision.actions)
    nonzero = profiles[1]

    def alpha_grad_and_logits(routing):
        policy.alpha_utility_routing = routing
        policy.zero_grad(set_to_none=True)
        state = policy.begin_document(document, nonzero)
        row = policy.distribution(state, decision, menu, nonzero)
        row.log_probs.sum().backward()
        grad = float(policy.alpha_raw.grad)
        return grad, row.combined_logits.detach().clone(), row.count_log_probs.detach().clone()

    base_grad, base_logits, base_count = alpha_grad_and_logits(None)
    routed_grad, routed_logits, routed_count = alpha_grad_and_logits("per-decision")

    decision_count = len(document.policy_decisions)
    assert torch.equal(base_logits, routed_logits)          # forward identical
    assert torch.equal(base_count, routed_count)            # count channel untouched
    assert routed_grad == pytest.approx(base_grad / decision_count, rel=1e-5)
    policy.alpha_utility_routing = None


def test_controller_gap_scaling_multiplies_by_detached_logit_range():
    policy, store, document, decision, profiles = _semantic_policy()
    menu = tuple(action.action_id for action in decision.actions)
    nonzero = profiles[2]  # max lambda -> g = 1

    state = policy.begin_document(document, nonzero)
    base = policy.distribution(state, decision, menu, nonzero)
    policy.controller_gap_scaling = "utility-gap"
    scaled = policy.distribution(state, decision, menu, nonzero)
    policy.controller_gap_scaling = None

    gap = (base.utility_logits.max() - base.utility_logits.min()).item()
    base_shift = base.combined_logits - base.utility_logits
    scaled_shift = scaled.combined_logits - scaled.utility_logits
    assert torch.allclose(scaled_shift, base_shift * gap, atol=1e-6)
    # lambda-zero untouched
    zero_state = policy.begin_document(document, profiles[0])
    policy.controller_gap_scaling = "utility-gap"
    row = policy.distribution(zero_state, decision, menu, profiles[0])
    policy.controller_gap_scaling = None
    assert torch.equal(row.combined_logits, row.utility_logits)


def test_switch_threshold_calibration_flips_argmax_at_threshold():
    policy, document, decision, profiles, _ = _direct_count_policy()
    menu = tuple(action.action_id for action in decision.actions)
    max_profile = profiles[-1]  # g = 1, so alpha is the controller pressure

    raw_median, norm_median = semantic_ranker_module.switch_threshold_calibration(
        policy, (document,), profiles,
    )
    assert math.isfinite(raw_median) and raw_median > 0.0
    assert math.isfinite(norm_median) and norm_median > 0.0

    def combined_vs_utility_argmax(alpha, *, gap_scaling=None):
        policy.controller_gap_scaling = gap_scaling
        semantic_ranker_module.calibrate_alpha(policy, alpha)
        policy.controller_gap_scaling = None
        assert float(policy.alpha) == pytest.approx(alpha, rel=1e-5)
        policy.controller_gap_scaling = gap_scaling
        state = policy.begin_document(document, max_profile)
        with torch.no_grad():
            row = policy.distribution(state, decision, menu, max_profile)
        policy.controller_gap_scaling = None
        return int(torch.argmax(row.combined_logits)), int(torch.argmax(row.utility_logits))

    below, utility_star = combined_vs_utility_argmax(raw_median * 0.99)
    above, _ = combined_vs_utility_argmax(raw_median * 1.01)
    assert below == utility_star
    assert above != utility_star

    # The gap-normalized threshold plays the same switching role once the
    # controller is scaled by the per-menu utility-logit range.
    below_scaled, _ = combined_vs_utility_argmax(
        norm_median * 0.99, gap_scaling="utility-gap",
    )
    above_scaled, _ = combined_vs_utility_argmax(
        norm_median * 1.01, gap_scaling="utility-gap",
    )
    assert below_scaled == utility_star
    assert above_scaled != utility_star


def test_calibrate_alpha_rejects_non_positive_or_non_finite_targets():
    policy, _, _, _, _ = _semantic_policy()
    for target in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            semantic_ranker_module.calibrate_alpha(policy, target)


def test_utility_logit_softcap_bounds_and_preserves_order_and_lambda_zero():
    policy, document, decision, profiles, _ = _direct_count_policy()
    menu = tuple(a.action_id for a in decision.actions)
    state = policy.begin_document(document, profiles[0])
    base = policy.distribution(state, decision, menu, profiles[0])

    policy.utility_logit_softcap = 0.05  # tiny cap to force visible squashing
    policy.float()  # invalidate feature caches so the forward re-runs
    capped_state = policy.begin_document(document, profiles[0])
    capped = policy.distribution(capped_state, decision, menu, profiles[0])
    policy.utility_logit_softcap = None

    assert float(capped.utility_logits.abs().max()) <= 0.05 + 1e-6
    assert torch.equal(
        torch.argsort(base.utility_logits), torch.argsort(capped.utility_logits),
    )
    # lambda-zero stays the exact identity under capping
    assert torch.equal(capped.combined_logits, capped.utility_logits)


def test_learned_controller_gain_zero_init_identity_bound_and_gradient():
    from cloak.ranker.semantic import decision_controller_alpha, enable_controller_gain

    policy, document, decision, profiles, _ = _direct_count_policy()
    menu = tuple(a.action_id for a in decision.actions)
    nonzero = profiles[-1]

    state = policy.begin_document(document, nonzero)
    base = policy.distribution(state, decision, menu, nonzero)

    enable_controller_gain(policy, "learned", hidden_dim=8, bound=1.5)
    policy.float()  # drop feature caches
    gain_state = policy.begin_document(document, nonzero)
    gained = policy.distribution(gain_state, decision, menu, nonzero)

    # zero-init residual -> exact identity with the global-alpha controller
    assert torch.allclose(base.combined_logits, gained.combined_logits, atol=1e-6)
    assert decision_controller_alpha(
        policy, gain_state, decision, menu,
    ) == pytest.approx(float(policy.alpha), rel=1e-5)

    # bound: even a huge residual cannot push alpha past softplus(raw + bound)
    with torch.no_grad():
        policy.gain_head[-1].bias.fill_(1000.0)
    policy.float()
    bounded_state = policy.begin_document(document, nonzero)
    ceiling = float(torch.nn.functional.softplus(policy.alpha_raw + 1.5))
    assert decision_controller_alpha(
        policy, bounded_state, decision, menu,
    ) == pytest.approx(ceiling, rel=1e-4)
    with torch.no_grad():
        policy.gain_head[-1].bias.zero_()

    # gradient reaches the gain head through the count channel
    policy.float()
    policy.zero_grad(set_to_none=True)
    grad_state = policy.begin_document(document, nonzero)
    row = policy.distribution(grad_state, decision, menu, nonzero)
    row.count_log_probs.sum().backward()
    grads = [p.grad for p in policy.gain_head.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)

    # lambda-zero stays the exact identity
    zero_state = policy.begin_document(document, profiles[0])
    zero = policy.distribution(zero_state, decision, menu, profiles[0])
    assert torch.equal(zero.combined_logits, zero.utility_logits)
    del policy.gain_head
    policy.controller_gain_mode = None


def test_random_controller_gain_is_deterministic_and_parameter_free():
    from cloak.ranker.semantic import (
        _random_gain_offset,
        decision_controller_alpha,
        enable_controller_gain,
    )

    policy, document, decision, profiles, _ = _direct_count_policy()
    menu = tuple(a.action_id for a in decision.actions)
    parameter_count = sum(1 for _ in policy.parameters())
    enable_controller_gain(policy, "random", bound=1.5)
    assert sum(1 for _ in policy.parameters()) == parameter_count

    state = policy.begin_document(document, profiles[-1])
    first = decision_controller_alpha(policy, state, decision, menu)
    second = decision_controller_alpha(policy, state, decision, menu)
    assert first == second
    offset = _random_gain_offset(decision.decision_id, 1.5)
    assert abs(offset) <= 1.5
    assert _random_gain_offset("some-other-decision", 1.5) != offset
    policy.controller_gain_mode = None
