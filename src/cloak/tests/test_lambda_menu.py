from __future__ import annotations

from dataclasses import replace

import pytest


def _point(
    doc_id: str,
    name: str,
    utility: float,
    count_score: float,
    *,
    corpus: str = "fixture",
    modes: tuple[str, ...] = ("level",),
    runtime_types: tuple[str, ...] = ("TYPE",),
):
    from cloak.train.lambda_menu import CalibrationPoint

    return CalibrationPoint(
        doc_id=doc_id,
        corpus=corpus,
        sources=("support",),
        ordered_action_vector=(("decision", f"action-{name}"),),
        utility=utility,
        count_score=count_score,
        component_scores={"assertion": utility},
        count_provenance={
            "action": {
                "mode": modes[0],
                "grounding_status": "certifying",
                "source_family": "fixture",
            },
        },
        reward_pins={
            "environment_hash": "env",
            "utility_artifact_hash": "utility",
            "profile_target_artifact_hash": "count",
            "execution_contract_version": "execution",
        },
        action_modes=modes,
        runtime_types=runtime_types,
        result_hash=f"result-{doc_id}-{name}",
    )


def test_exact_action_vectors_deduplicate_and_merge_sources():
    from cloak.train.lambda_menu import deduplicate_action_vectors

    original = _point("doc", "same", 0.8, 0.2)
    duplicate = replace(original, sources=("counterfactual",))

    result = deduplicate_action_vectors((original, duplicate))

    assert len(result) == 1
    assert result[0].sources == ("counterfactual", "support")


def test_conflicting_duplicate_action_vector_is_rejected():
    from cloak.train.lambda_menu import deduplicate_action_vectors

    original = _point("doc", "same", 0.8, 0.2)
    conflict = replace(original, utility=0.7)

    with pytest.raises(ValueError, match="conflicting duplicate action vector"):
        deduplicate_action_vectors((original, conflict))


def test_exact_up_ties_retain_multiplicity_and_canonical_vector():
    from cloak.train.lambda_menu import merge_exact_point_ties

    right = _point("doc", "z", 0.8, 0.2)
    left = _point("doc", "a", 0.8, 0.2)

    merged = merge_exact_point_ties((right, left))

    assert len(merged) == 1
    assert merged[0].tie_multiplicity == 2
    assert merged[0].canonical_action_vector == left.ordered_action_vector


def test_weak_dominance_removes_points_with_one_strict_coordinate():
    from cloak.train.lambda_menu import merge_exact_point_ties, remove_weakly_dominated

    points = merge_exact_point_ties((
        _point("doc", "winner", 0.9, 0.6),
        _point("doc", "lower-u", 0.8, 0.6),
        _point("doc", "lower-p", 0.9, 0.5),
        _point("doc", "tradeoff", 1.0, 0.2),
    ))

    kept = remove_weakly_dominated(points)

    assert {(point.utility, point.count_score) for point in kept} == {
        (0.9, 0.6), (1.0, 0.2),
    }


def test_upper_convex_envelope_removes_point_below_chord():
    from cloak.train.lambda_menu import (
        merge_exact_point_ties,
        remove_weakly_dominated,
        upper_convex_envelope,
    )

    points = merge_exact_point_ties((
        _point("doc", "left", 1.0, 0.0),
        _point("doc", "below", 0.70, 0.5),
        _point("doc", "right", 0.5, 1.0),
    ))

    envelope = upper_convex_envelope(remove_weakly_dominated(points))

    assert [(point.count_score, point.utility) for point in envelope] == [
        (0.0, 1.0), (1.0, 0.5),
    ]


def test_positive_switch_points_have_per_document_total_weight_one():
    from cloak.train.lambda_menu import document_frontier

    frontier = document_frontier((
        _point("doc", "left", 1.0, 0.0),
        _point("doc", "middle", 0.8, 0.5),
        _point("doc", "right", 0.4, 1.0),
    ))

    assert [point.value for point in frontier.switch_points] == pytest.approx([0.4, 0.8])
    assert sum(point.weight for point in frontier.switch_points) == pytest.approx(1.0)
    assert all(point.doc_id == "doc" for point in frontier.switch_points)


def test_document_with_fewer_than_three_distinct_points_stays_replay_only():
    from cloak.train.lambda_menu import document_frontier

    frontier = document_frontier((
        _point("doc", "left", 1.0, 0.0),
        _point("doc", "right", 0.5, 1.0),
    ))

    assert len(frontier.envelope) == 2
    assert frontier.switch_points == ()
    assert frontier.switch_eligible is False


def test_weighted_log_quantiles_and_nearest_observed_snapping_are_deterministic():
    from cloak.train.lambda_menu import SwitchPoint, snap_to_observed, weighted_log_quantiles

    switches = (
        SwitchPoint("a", 0.1, 0.5, (), ()),
        SwitchPoint("b", 1.0, 0.25, (), ()),
        SwitchPoint("c", 100.0, 0.25, (), ()),
    )

    assert weighted_log_quantiles(switches, (0.25, 0.60, 0.90)) == (0.1, 1.0, 100.0)
    assert snap_to_observed(10 ** 0.5, (1.0, 10.0)) == 1.0


def test_equivalent_replay_signatures_merge_lambda_values():
    from cloak.train.lambda_menu import merge_equivalent_lambdas

    pool = (
        _point("doc-a", "utility", 1.0, 0.0),
        _point("doc-a", "middle", 0.9, 0.6),
        _point("doc-a", "count", 0.2, 1.0),
        _point("doc-b", "utility", 1.0, 0.0),
        _point("doc-b", "middle", 0.9, 0.6),
        _point("doc-b", "count", 0.2, 1.0),
    )

    merged, signatures = merge_equivalent_lambdas((0.0, 0.2, 0.3, 2.0), pool)

    assert merged == (0.0, 0.2, 2.0)
    assert signatures[0.2] == signatures[0.3]
    assert signatures[0.0] != signatures[2.0]


def _menu_pool():
    points = []
    for index, switch in enumerate((0.1, 1.0, 10.0, 100.0)):
        doc_id = f"doc-{index}"
        points.extend((
            _point(doc_id, "utility", 1.0, 0.0),
            _point(doc_id, "middle", 1.0 - 0.001 * switch, 0.001),
            _point(
                doc_id,
                "count",
                1.0 - 0.002 * switch,
                0.002,
                modes=("placeholder",),
            ),
        ))
    return tuple(points)


def test_menu_acceptance_starts_at_zero_and_preserves_scalarization_invariants():
    from cloak.train.lambda_menu import select_lambda_menu

    artifact = select_lambda_menu(
        _menu_pool(),
        menu_size=4,
        min_adjacent_winner_change=0.2,
        max_placeholder_fraction=1.0,
        min_supported_documents_by_corpus=1,
        min_supported_decisions_by_type=1,
    )

    assert artifact["verdict"] == "PASS"
    assert artifact["values"][0] == 0.0
    assert 3 <= len(artifact["values"]) <= 5
    assert artifact["replay_report"]["lambda_zero_identity"] is True
    selected = artifact["replay_report"]["selected_count_score"]
    assert selected == sorted(selected)
    assert artifact["replacement_passes"] <= 2


def test_menu_replacement_is_bounded_to_two_deterministic_passes():
    from cloak.train.lambda_menu import select_lambda_menu

    artifact = select_lambda_menu(
        _menu_pool(),
        menu_size=4,
        min_adjacent_winner_change=0.9,
        max_placeholder_fraction=1.0,
        min_supported_documents_by_corpus=1,
        min_supported_decisions_by_type=1,
    )

    assert artifact["verdict"] == "FAIL"
    assert artifact["replacement_passes"] == 2
    assert artifact["failure_reasons"]


def test_menu_stops_instead_of_padding_fewer_than_three_signatures():
    from cloak.train.lambda_menu import select_lambda_menu

    pool = (
        _point("doc", "utility", 1.0, 0.0),
        _point("doc", "count", 0.0, 1.0),
    )

    artifact = select_lambda_menu(
        pool,
        menu_size=4,
        min_adjacent_winner_change=0.0,
        max_placeholder_fraction=1.0,
        min_supported_documents_by_corpus=1,
        min_supported_decisions_by_type=1,
    )

    assert artifact["verdict"] == "FAIL"
    assert artifact["values"] == [0.0]
    assert "fewer_than_three_supported_profiles" in artifact["failure_reasons"]


def test_anchor_trajectories_cover_every_required_source_after_vector_deduplication():
    from test_ranker_diagnostics import _count_reward, _documents

    from cloak.train.lambda_menu import build_anchor_trajectories

    anchors = build_anchor_trajectories(_documents()["doc"], _count_reward())

    assert set(source for row in anchors for source in row.sources) == {
        "behavior_cloning",
        "keep_walk",
        "minimum_count_non_keep_walk",
        "midpoint_level_walk",
        "all_placeholder_walk",
    }
    assert len({row.ordered_action_vector for row in anchors}) == len(anchors)


def test_calibration_pool_stores_complete_pins_components_and_count_provenance():
    from test_ranker_diagnostics import (
        _count_reward,
        _documents,
        _utility_artifact,
    )

    from cloak.train.lambda_menu import (
        build_anchor_trajectories,
        calibration_point_from_result,
        freeze_calibration_pool,
    )
    from cloak.train.utility_cache import make_result

    document = _documents()["doc"]
    candidate = next(
        row for row in build_anchor_trajectories(document, _count_reward())
        if row.ordered_action_vector == (("decision", "fine"),)
    )
    result = make_result(
        doc_id="doc",
        action_vector=dict(candidate.ordered_action_vector),
        doc_p="rendered",
        out_p="remote",
        out_final="final",
        component_scores={"linked": 0.8, "residual": 0.6},
        utility=0.7,
    )
    point = calibration_point_from_result(
        candidate,
        result,
        count_reward=_count_reward(),
        count_state={
            "action_targets": {
                "fine": {
                    "mode": "level",
                    "profile_id": "profile",
                    "grounding_status": "certifying",
                    "source_family": "fixture",
                },
            },
        },
        utility_artifact=_utility_artifact(),
        reward_pins={
            "environment_hash": "env",
            "utility_artifact_hash": "utility",
            "profile_target_artifact_hash": "count",
            "execution_contract_version": "execution",
        },
    )
    artifact = freeze_calibration_pool(
        (point,),
        split_by_doc={"doc": "development"},
        reward_pins=point.reward_pins,
    )

    assert point.utility == pytest.approx(0.7)
    assert point.count_score == pytest.approx(0.1)
    assert point.component_scores == {"linked": 0.8, "residual": 0.6}
    assert point.count_provenance["fine"]["source_family"] == "fixture"
    assert artifact["artifact_version"] == "ranker-v2-calibration-pool-v1"
    assert artifact["artifact_hash"].startswith("sha256:")
    assert artifact["documents"]["doc"]["split"] == "development"
