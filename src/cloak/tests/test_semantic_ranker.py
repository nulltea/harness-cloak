from __future__ import annotations

import inspect
from dataclasses import fields, replace
from pathlib import Path
from types import MappingProxyType

import pytest
import torch

from cloak.train.ranker_environment import RankerDecision, RankerDocument
from cloak.train.ranker_representation import DocumentTokenBank, RelationFeatures
import cloak.train.semantic_ranker as semantic_ranker_module
from cloak.train.semantic_ranker import (
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
    document = RankerDocument(
        doc_id="fixture/doc",
        corpus="fixture",
        text="x" * 80,
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
