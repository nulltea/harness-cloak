from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
import json
from types import SimpleNamespace

import pytest

from cloak.train.profile_count import ProfileActionTarget, ProfileCountTargets
from cloak.train.ranker_environment import RankerAction, RankerDecision, RankerDocument


def _documents():
    decision = RankerDecision(
        decision_id="decision",
        profile_id="profile",
        runtime_type="TYPE",
        canonical_key="key",
        occurrence_ids=("occurrence",),
        actions=(
            RankerAction(
                action_id="fine",
                mode="level",
                fill="generalized",
                authored_level_index=0,
                runtime_type="TYPE",
            ),
            RankerAction(
                action_id="coarse",
                mode="level",
                fill="coarse generalized",
                authored_level_index=1,
                runtime_type="TYPE",
            ),
            RankerAction(
                action_id="keep",
                mode="keep",
                fill="Entity",
                authored_level_index=None,
                runtime_type="TYPE",
            ),
            RankerAction(
                action_id="placeholder",
                mode="placeholder",
                fill=None,
                authored_level_index=None,
                runtime_type="TYPE",
            ),
        ),
    )
    document = RankerDocument(
        doc_id="doc",
        corpus="aci",
        text="Entity",
        occurrences=(MappingProxyType({
            "occurrence_id": "occurrence",
            "decision_id": "decision",
            "start": 0,
            "end": 6,
            "surface": "Entity",
            "controlled": True,
        }),),
        policy_decisions=(decision,),
        fixed_decisions=(),
    )
    return {document.doc_id: document}


def _count_reward():
    rows = {
        "fine": ProfileActionTarget(
            "decision", "fine", "profile", "TYPE", "level", 1.0, 0.1,
            "certifying", "fixture",
        ),
        "coarse": ProfileActionTarget(
            "decision", "coarse", "profile", "TYPE", "level", 10.0, 0.6,
            "certifying", "fixture",
        ),
        "keep": ProfileActionTarget(
            "decision", "keep", "profile", "TYPE", "keep", None, 0.0, None, None,
        ),
        "placeholder": ProfileActionTarget(
            "decision", "placeholder", "profile", "TYPE", "placeholder", None,
            1.0, None, None,
        ),
    }
    return ProfileCountTargets(
        rows, {"decision": tuple(rows)}, {"decision": True},
    )


def _point(name, utility, count_score, mode):
    from cloak.train.lambda_menu import CalibrationPoint

    return CalibrationPoint(
        doc_id="doc",
        corpus="fixture",
        sources=(name,),
        ordered_action_vector=(("decision", name),),
        utility=utility,
        count_score=count_score,
        component_scores={"linked": utility, "residual": utility},
        count_provenance={
            name: {
                "mode": mode,
                "grounding_status": "certifying" if mode == "level" else None,
                "source_family": "fixture" if mode == "level" else None,
            },
        },
        reward_pins={
            "environment_hash": "env",
            "utility_artifact_hash": "utility",
            "profile_target_artifact_hash": "count",
            "execution_contract_version": "execution",
        },
        action_modes=(mode,),
        runtime_types=("TYPE",),
        result_hash=f"result-{name}",
    )


def _pool():
    return (
        _point("keep", 1.0, 0.0, "keep"),
        _point("fine", 0.8, 0.1, "level"),
        _point("coarse", 0.5, 0.6, "level"),
        _point("placeholder", 0.1, 1.0, "placeholder"),
    )


def _utility_artifact():
    from cloak.train.utility_cache import stable_hash

    artifact = {
        "artifact_version": "utility-assertions-v2",
        "environment_hash": "env",
        "reader_pin": {
            "model": "fixture-reader",
            "endpoint": "fixture-endpoint",
            "prompt_version": "fixture-prompt",
            "response_schema": {"type": "object"},
            "revision": "fixture-revision",
        },
        "documents": {
            "doc": {
                "assertion_ids": ["linked", "residual"],
                "policy_decision_ids": ["decision"],
                "uncovered_policy_decision_ids": [],
                "utility_weight_denominator": 1.0,
            },
        },
        "assertions": {
            "linked": {
                "assertion_id": "linked",
                "doc_id": "doc",
                "family": "context",
                "status": "accepted",
                "weight": 0.5,
                "credit_routing": "linked",
                "policy_dependency_decision_ids": ["decision"],
            },
            "residual": {
                "assertion_id": "residual",
                "doc_id": "doc",
                "family": "delivered",
                "status": "accepted",
                "weight": 0.5,
                "credit_routing": "residual",
                "policy_dependency_decision_ids": [],
            },
        },
    }
    artifact["artifact_hash"] = stable_hash(artifact)
    return artifact


def _count_state():
    return {
        "artifact_version": "ranker-v2-profile-count-targets-v1",
        "artifact_hash": "count",
        "environment_hash": "env",
        "gate_mode": "diagnostic",
        "profile_tags": {},
        "action_targets": {
            "fine": {
                "action_id": "fine", "decision_id": "decision", "mode": "level",
                "runtime_type": "TYPE", "profile_score": 0.1, "log_count": 1.0,
                "grounding_status": "certifying", "source_family": "fixture",
                "profile_id": "profile",
            },
            "coarse": {
                "action_id": "coarse", "decision_id": "decision", "mode": "level",
                "runtime_type": "TYPE", "profile_score": 0.6, "log_count": 10.0,
                "grounding_status": "certifying", "source_family": "fixture",
                "profile_id": "profile",
            },
        },
        "gate_report": {
            "verdict": "PASS",
            "strict_verdict": "PASS",
            "summary": {"level_actions": 2, "admitted_level_actions": 2, "gap_count": 0},
            "missing_policy_mappings": [],
            "nonmonotone_profiles": [],
        },
    }


def _menu_artifact():
    return {
        "values": [0.0, 1.0],
        "replay_report": {
            "winner_signatures": ["signature-zero", "signature-one"],
            "winner_change": [1.0],
            "selected_count_score": [0.0, 0.6],
            "lambda_zero_identity": True,
            "placeholder_fraction": [0.0, 0.0],
        },
    }


def test_default_threshold_rules_predeclare_every_required_field():
    from cloak.train.ranker_diagnostics import (
        default_threshold_rules,
        required_empirical_threshold_fields,
        validate_threshold_rules,
    )

    rules = validate_threshold_rules(default_threshold_rules())

    assert set(rules["fields"]) == set(required_empirical_threshold_fields())
    for row in rules["fields"].values():
        assert set(row) == {
            "measurement_definition", "candidate_rule", "allowed_split",
            "support_rule", "tie_handling", "action",
        }
        assert row["allowed_split"] in {"train", "development", "train+development"}
        assert row["action"] in {"block", "ablation", "reduce_scope", "report_only"}
        assert "attacker" not in str(row).casefold()


def test_diagnostic_spike_emits_every_measurement_family_and_hash():
    from cloak.train.ranker_diagnostics import build_diagnostic_spike

    spike = build_diagnostic_spike(
        _pool(),
        documents=_documents(),
        utility_artifact=_utility_artifact(),
        count_reward=_count_reward(),
        count_state=_count_state(),
        menu_artifact=_menu_artifact(),
        split_by_doc={"doc": "development"},
        reader_jitter=(0.0, 0.02, -0.01),
        counterfactual_records=(
            {"doc_id": "doc", "decision_id": "decision", "delta_u": 0.0},
            {"doc_id": "doc", "decision_id": "decision", "delta_u": 0.2},
            {"doc_id": "doc", "decision_id": "decision", "delta_u": -0.1},
        ),
    )

    assert spike["artifact_version"] == "ranker-v2-diagnostic-spike-v1"
    assert spike["artifact_hash"].startswith("sha256:")
    assert set(spike["measurements"]) == {
        "trajectory_support",
        "frontier_switch_support",
        "winner_signatures",
        "utility_resolution",
        "credit_coverage",
        "counterfactual_support",
        "count_signal",
        "injectivity",
        "support",
    }
    assert spike["measurements"]["trajectory_support"]["unique_trajectories"] == 4
    assert spike["measurements"]["counterfactual_support"]["zero_count"] == 1
    assert spike["measurements"]["credit_coverage"] == {
        "linked_assertions": 1,
        "residual_assertions": 1,
        "fallback_decisions": 0,
        "uncovered_decisions": 0,
        "by_document": {
            "doc": {
                "linked_assertions": 1,
                "residual_assertions": 1,
                "fallback_decisions": 0,
                "uncovered_decisions": 0,
            },
        },
        "by_corpus": {
            "aci": {
                "linked_assertions": 1,
                "residual_assertions": 1,
                "fallback_decisions": 0,
                "uncovered_decisions": 0,
            },
        },
    }
    assert spike["measurements"]["count_signal"]["adjacent_delta_p"]["count"] == 1
    assert spike["measurements"]["utility_resolution"][
        "quantization_step_by_document"
    ] == {"doc": pytest.approx(0.2)}
    assert spike["measurements"]["utility_resolution"]["reader_jitter_by_split"][
        "development"
    ]["abs_max"] == pytest.approx(0.02)
    assert spike["measurements"]["counterfactual_support"]["by_type"]["TYPE"] == {
        "count": 3,
        "zero_count": 1,
        "positive_count": 1,
        "negative_count": 1,
    }
    assert spike["measurements"]["injectivity"]["by_type"]["TYPE"][
        "eligible_decisions"
    ] == 4
    assert spike["measurements"]["support"]["profile_support"] == {
        "0.0": {
            "documents_by_corpus": {"fixture": 1},
            "decisions_by_type": {"TYPE": 1},
        },
        "1.0": {
            "documents_by_corpus": {"fixture": 1},
            "decisions_by_type": {"TYPE": 1},
        },
    }
    assert spike["measurements"]["support"]["decision_count"] == 1
    assert spike["measurements"]["support"]["documents_by_corpus_by_split"] == {
        "development": {"fixture": 1},
    }


def test_threshold_manifest_freezes_hard_invariants_and_numeric_empirical_values():
    from cloak.train.ranker_diagnostics import (
        default_threshold_rules,
        freeze_threshold_manifest,
    )

    spike = {
        "artifact_version": "ranker-v2-diagnostic-spike-v1",
        "artifact_hash": "sha256:spike",
        "dataset_hash": "sha256:dataset",
        "measurements": {
            "trajectory_support": {
                "distinct_points_by_document": {"doc": 4},
            },
            "frontier_switch_support": {
                "winner_change_floor": 0.1,
                "positive_switch_documents_by_corpus": {"fixture": 1},
            },
            "count_signal": {"flat_menu_fraction": 0.0},
            "injectivity": {"collision_rate": 0.0, "lost_count_opportunity_max": 0.0},
            "counterfactual_support": {"nonzero_rate": 2 / 3},
            "utility_resolution": {
                "reader_jitter_abs_max": 0.02,
                "reader_jitter_by_split": {
                    "development": {"abs_max": 0.02},
                },
            },
            "support": {
                "documents_by_corpus": {"fixture": 1},
                "decisions_by_type": {"TYPE": 1},
                "documents_by_corpus_by_split": {
                    "development": {"fixture": 1},
                },
            },
        },
    }
    manifest = freeze_threshold_manifest(
        default_threshold_rules(),
        spike,
        pins={
            "reward_version": "reward",
            "environment_hash": "env",
            "span_decision_artifact_hash": "span",
            "utility_component_artifact_hash": "utility",
            "count_artifact_hash": "count",
        },
    )

    assert manifest["hard_gates"] == {
        "explicit_count_coverage": 1.0,
        "fallback_count_gradient_mass": 0.0,
        "missing_occurrence_decision_mappings": 0,
        "nonmonotone_profiles": 0,
        "lambda_zero_identity": "exact",
    }
    for section in ("feasibility_gates", "scheduler"):
        assert all(isinstance(value, int | float) for value in manifest[section].values())
    assert isinstance(manifest["acceptance"]["confidence_level"], float)
    assert isinstance(manifest["acceptance"]["utility_noninferiority_margin"], float)
    assert isinstance(manifest["acceptance"]["minimum_supported_documents"], int)
    assert manifest["artifact_hash"].startswith("sha256:")


def test_report_only_cannot_be_used_for_a_run_relevant_threshold():
    from cloak.train.ranker_diagnostics import default_threshold_rules, validate_threshold_rules

    rules = default_threshold_rules()
    rules["fields"]["feasibility_gates.min_adjacent_winner_change"][
        "action"
    ] = "report_only"

    with pytest.raises(ValueError, match="run-relevant threshold cannot be report_only"):
        validate_threshold_rules(rules)


def test_threshold_manifest_stops_when_reader_jitter_is_unmeasured():
    from cloak.train.ranker_diagnostics import (
        default_threshold_rules,
        freeze_threshold_manifest,
    )

    spike = {
        "artifact_version": "ranker-v2-diagnostic-spike-v1",
        "artifact_hash": "sha256:spike",
        "dataset_hash": "sha256:dataset",
        "measurements": {
            "frontier_switch_support": {
                "winner_change_floor": 0.1,
                "positive_switch_documents_by_corpus": {"fixture": 1},
            },
            "support": {
                "documents_by_corpus": {"fixture": 2},
                "decisions_by_type": {"TYPE": 2},
                "documents_by_corpus_by_split": {
                    "development": {"fixture": 1},
                },
            },
            "count_signal": {"flat_menu_fraction": 0.0},
            "injectivity": {"collision_rate": 0.0, "lost_count_opportunity_max": 0.0},
            "counterfactual_support": {"nonzero_rate": 0.5},
            "utility_resolution": {
                "reader_jitter_by_split": {"development": {"abs_max": None}},
            },
        },
    }

    with pytest.raises(ValueError, match="finite numeric value"):
        freeze_threshold_manifest(
            default_threshold_rules(),
            spike,
            pins={
                "reward_version": "reward",
                "environment_hash": "env",
                "span_decision_artifact_hash": "span",
                "utility_component_artifact_hash": "utility",
                "count_artifact_hash": "count",
            },
        )


def test_reader_jitter_uses_paired_cached_refresh_vectors_only():
    from cloak.train.ranker_diagnostics import reader_jitter_from_cache
    from cloak.train.utility_cache import make_result

    artifact = _utility_artifact()
    base = make_result(
        doc_id="doc",
        action_vector={"decision": "fine"},
        doc_p="rendered",
        out_p="remote",
        out_final="final",
        component_scores={"linked": 0.8, "residual": 0.6},
        utility=0.7,
    )
    refreshed = make_result(
        doc_id="doc",
        action_vector={"decision": "fine"},
        doc_p="rendered",
        out_p="remote-refresh",
        out_final="final-refresh",
        component_scores={"linked": 0.4, "residual": 0.6},
        utility=0.5,
    )
    identity = {
        "doc_id": "doc",
        "ordered_action_vector": [["decision", "fine"]],
        "environment_hash": "env",
        "utility_artifact_hash": artifact["artifact_hash"],
    }
    cache = SimpleNamespace(entries={
        "base": ({**identity, "reader_refresh": False}, base),
        "refresh": ({**identity, "reader_refresh": True}, refreshed),
    })

    jitter = reader_jitter_from_cache(
        cache,
        utility_artifact=artifact,
        split_by_doc={"doc": "development"},
    )

    assert jitter == {"development": (pytest.approx(-0.2),)}


def test_cache_only_missing_report_names_exact_vectors_and_work_counts(tmp_path: Path):
    from cloak.train.lambda_menu import CalibrationTrajectory
    from cloak.train.ranker_diagnostics import cache_only_missing_report
    from cloak.train.utility_cache import UtilityCache

    candidate = CalibrationTrajectory(
        doc_id="doc",
        corpus="fixture",
        sources=("bc",),
        ordered_action_vector=(("decision", "fine"),),
        action_modes=("level",),
        runtime_types=("TYPE",),
    )
    report = cache_only_missing_report(
        (candidate,),
        documents=_documents(),
        utility_artifact=_utility_artifact(),
        environment_hash="env",
        cache=UtilityCache(tmp_path / "empty.jsonl"),
    )

    assert report["missing_action_vector_count"] == 1
    assert report["remote_tasks"] == 1
    assert report["context_reader_work_items"] == 1
    assert report["missing"][0]["ordered_action_vector"] == [["decision", "fine"]]
    assert report["dispatched"] is False


def test_preflight_cli_cleanly_stops_and_writes_exact_cache_misses(
    tmp_path: Path, capsys,
):
    from run_ranker_preflight import main

    from cloak.train.ranker_diagnostics import default_threshold_rules

    root = Path(__file__).resolve().parents[3]
    rules = tmp_path / "threshold-rules.json"
    rules.write_text(json.dumps(default_threshold_rules()), encoding="utf-8")
    out_dir = tmp_path / "preflight"

    status = main([
        "--environment", str(root / "results/ranker_v2/environment/ranker-env.json"),
        "--utility-artifact", str(root / "results/ranker_v2/qa/aci-full.utility"),
        "--profile-count-targets",
        str(root / "results/ranker_v2/reward/profile-count-targets.json"),
        "--utility-cache", str(tmp_path / "empty-cache.jsonl"),
        "--exit-winners", str(tmp_path / "missing-exit-winners.json"),
        "--threshold-rules", str(rules),
        "--out-dir", str(out_dir),
        "--cache-only",
    ])

    line = capsys.readouterr().out.strip()
    report = json.loads((out_dir / "cache-misses.json").read_text())
    assert status == 2
    assert line.startswith("PREFLIGHT CACHE_ONLY_STOP ")
    assert f"missing_action_vectors={report['missing_action_vector_count']}" in line
    assert f"remote_tasks={report['remote_tasks']}" in line
    assert (
        f"context_reader_work_items={report['context_reader_work_items']}" in line
    )
    assert line.endswith("dispatched=false")
    assert report["missing"]


def _training_artifact_fixtures():
    from cloak.train.utility_cache import stable_hash

    environment = {
        "artifact_version": "ranker-v2-environment-v2",
        "frozen_environment": {"environment_hash": "env", "documents": {}},
    }
    count_state = {
        "artifact_version": "ranker-v2-profile-count-targets-v1",
        "environment_hash": "env",
        "gate_report": {
            "verdict": "PASS",
            "missing_policy_mappings": [],
            "nonmonotone_profiles": [],
        },
    }
    count_state["artifact_hash"] = stable_hash(count_state)
    utility = {
        "artifact_version": "utility-assertions-v2",
        "environment_hash": "env",
    }
    utility["artifact_hash"] = stable_hash(utility)
    menu = {
        "artifact_version": "ranker-v2-lambda-menu-v1",
        "verdict": "PASS",
        "values": [0.0, 1.0, 3.0],
        "profile_names": ["zero", "middle", "high"],
    }
    menu["artifact_hash"] = stable_hash(menu)
    threshold = {
        "artifact_version": "ranker-v2-threshold-manifest-v1",
        "environment_hash": "env",
        "utility_component_artifact_hash": utility["artifact_hash"],
        "count_artifact_hash": count_state["artifact_hash"],
        "hard_gates": {
            "explicit_count_coverage": 1.0,
            "fallback_count_gradient_mass": 0.0,
            "missing_occurrence_decision_mappings": 0,
            "nonmonotone_profiles": 0,
            "lambda_zero_identity": "exact",
        },
        "feasibility_gates": {
            "min_adjacent_winner_change": 0.2,
        },
        "scheduler": {
            "call_budget": 5,
            "endpoint_fraction": 0.2,
        },
    }
    threshold["artifact_hash"] = stable_hash(threshold)
    exit_winners = {
        "artifact_version": "ranker-v2-exit-winners-v1",
        "pins": {
            "environment_hash": "env",
            "profile_target_artifact_hash": count_state["artifact_hash"],
            "representation_manifest_hash": "representations",
            "utility_artifact_hash": utility["artifact_hash"],
            "policy_checkpoint_hash": "bc-file",
        },
        "documents": [],
    }
    exit_winners["artifact_hash"] = stable_hash(exit_winners)
    bc_checkpoint = {
        "checkpoint_version": "ranker-v2-bc-v1",
        "pins": {
            "environment_hash": "env",
            "profile_target_artifact_hash": count_state["artifact_hash"],
            "representation_manifest_hash": "representations",
            "utility_artifact_hash": utility["artifact_hash"],
        },
        "policy_config": {"policy_architecture": "semantic-v1"},
        "policy_state_dict": {},
    }
    return environment, count_state, utility, menu, threshold, exit_winners, bc_checkpoint


def test_train_artifact_validation_accepts_only_hash_bound_passing_gates():
    from train_interactive_ranker import _validate_train_artifacts

    fixtures = _training_artifact_fixtures()
    validated = _validate_train_artifacts(
        *fixtures,
        bc_checkpoint_hash="bc-file",
        representation_manifest_hash="representations",
    )

    assert [profile.name for profile in validated["profiles"]] == [
        "zero", "middle", "high",
    ]
    assert validated["counterfactual_budget"] == 5
    assert validated["endpoint_budget"] == 1
    assert validated["artifact_pins"]["lambda_menu_hash"] == fixtures[3][
        "artifact_hash"
    ]

    broken = list(fixtures)
    broken[1] = dict(fixtures[1], artifact_version="count-reward-state-v1")
    with pytest.raises(ValueError, match="profile count target artifact"):
        _validate_train_artifacts(
            *broken,
            bc_checkpoint_hash="bc-file",
            representation_manifest_hash="representations",
        )

    with pytest.raises(ValueError, match="artifact hashes are incomplete"):
        _validate_train_artifacts(*fixtures, bc_checkpoint_hash="bc-file")

    broken = list(fixtures)
    broken[4] = dict(fixtures[4], utility_component_artifact_hash="different")
    with pytest.raises(ValueError, match="utility artifact hash"):
        _validate_train_artifacts(
            *broken,
            bc_checkpoint_hash="bc-file",
            representation_manifest_hash="representations",
        )

    broken = list(fixtures)
    broken[2] = dict(fixtures[2], unexpected_stale_field=True)
    with pytest.raises(ValueError, match="utility artifact hash"):
        _validate_train_artifacts(
            *broken,
            bc_checkpoint_hash="bc-file",
            representation_manifest_hash="representations",
        )


def _privacy_metrics(nll, pairwise, calibration, regret=0.05):
    profiles = {
        f"profile-{index}": {
            "within_menu_pairwise_accuracy": pairwise,
            "profile_relative_calibration_error": calibration,
            "selected_action_regret": regret,
        }
        for index in range(6)
    }
    return {
        "overall": {
            "nll": nll,
            "within_menu_pairwise_accuracy": pairwise,
            "profile_relative_calibration_error": calibration,
            "selected_action_regret": regret,
            "median_absolute_log_error": 0.7,
            "interval_95_coverage": 0.9,
        },
        "by_runtime_type": {
            "drug": {}, "health-condition": {},
        },
        "by_grounding_status": {
            "certifying": {}, "model-proposed": {},
        },
        "by_source_family": {"fixture": {}},
        "by_profile": profiles,
    }


def _privacy_seed_report():
    baselines = {
        "authored_position_mode_type": _privacy_metrics(0.1, 1.0, 0.08, 0.0),
        "mode_type_only": _privacy_metrics(2.1, 0.55, 0.35, 0.2),
        "candidate_only": _privacy_metrics(1.8, 0.65, 0.25, 0.08),
        "train_profile_mean": _privacy_metrics(2.3, 0.5, 0.4, 0.25),
    }
    return {
        "seed": 11,
        "profile_held_out": True,
        "splits": {
            "dev": {
                "semantic": _privacy_metrics(17.6, 0.8, 0.1, 0.005),
                "baselines": baselines,
            },
            "test": {
                "semantic": _privacy_metrics(17.6, 0.75, 0.12, 0.005),
                "baselines": baselines,
            },
        },
    }


def test_privacy_diagnostic_manifest_requires_held_out_metrics_and_all_baselines():
    from cloak.train.ranker_diagnostics import build_privacy_diagnostic_manifest

    manifest = build_privacy_diagnostic_manifest(
        [_privacy_seed_report()],
        split_manifest_hash="sha256:split",
        metric_report_hash="sha256:metrics",
    )

    assert manifest["artifact_version"] == "ranker-v2-semantic-privacy-diagnostic-v2"
    assert manifest["profile_held_out"] is True
    assert manifest["policy_fitness_gate"]["controller_metrics_verdict"] == (
        "NEEDS_MULTI_SEED_EVIDENCE"
    )
    assert manifest["policy_fitness_gate"]["lexical_counterexamples"] == {
        "status": "N/A",
        "reason": "counterexample_set_absent",
        "artifact_hash": None,
    }
    assert manifest["relative_promotion"]["verdict"] == (
        "NEEDS_MULTI_SEED_AND_COUNTEREXAMPLE_SET"
    )
    assert manifest["distributional_audit_gate"]["verdict"] == "REPORT_ONLY"
    assert set(manifest["report_only_metrics"]) == {
        "nll", "interval_95_coverage", "sigma_fixed",
        "median_absolute_log_error",
    }
    authored = manifest["relative_promotion"]["seed_verdicts"][0][
        "comparisons"
    ]["authored_position_mode_type"]
    assert authored["role"] == "oracle_ceiling"
    assert "within_menu_pairwise_accuracy" not in authored["blocking_metrics"]
    assert manifest["aci_document_generalization_claimed"] is False
    assert set(manifest["required_baselines"]) == {
        "authored_position_mode_type",
        "mode_type_only",
        "candidate_only",
        "train_profile_mean",
    }
    assert manifest["artifact_hash"].startswith("sha256:")

    incomplete = _privacy_seed_report()
    del incomplete["splits"]["test"]["baselines"]["candidate_only"]
    with pytest.raises(ValueError, match="baseline"):
        build_privacy_diagnostic_manifest(
            [incomplete],
            split_manifest_hash="sha256:split",
            metric_report_hash="sha256:metrics",
        )

    unsupported = _privacy_seed_report()
    unsupported["splits"]["test"]["semantic"]["overall"][
        "within_menu_pairwise_accuracy"
    ] = None
    manifest = build_privacy_diagnostic_manifest(
        [unsupported],
        split_manifest_hash="sha256:split",
        metric_report_hash="sha256:metrics",
    )
    assert manifest["policy_fitness_gate"]["controller_metrics_verdict"] == "FAIL"


def test_privacy_gate_ignores_bad_nll_but_requires_candidate_controller_fitness():
    from cloak.train.ranker_diagnostics import build_privacy_diagnostic_manifest

    report = _privacy_seed_report()
    report["splits"]["test"]["semantic"]["overall"]["nll"] = 1e9
    manifest = build_privacy_diagnostic_manifest(
        [report],
        split_manifest_hash="sha256:split",
        metric_report_hash="sha256:metrics",
    )
    assert manifest["policy_fitness_gate"]["controller_metrics_verdict"] == (
        "NEEDS_MULTI_SEED_EVIDENCE"
    )

    report["splits"]["test"]["semantic"]["overall"][
        "profile_relative_calibration_error"
    ] = 0.3
    report["splits"]["test"]["semantic"]["overall"][
        "within_menu_pairwise_accuracy"
    ] = 0.6
    failed = build_privacy_diagnostic_manifest(
        [report],
        split_manifest_hash="sha256:split",
        metric_report_hash="sha256:metrics",
    )
    assert failed["policy_fitness_gate"]["controller_metrics_verdict"] == (
        "NEEDS_MULTI_SEED_EVIDENCE"
    )


def test_privacy_gate_requires_paired_bootstrap_improvement_across_three_seeds():
    from copy import deepcopy

    from cloak.train.ranker_diagnostics import build_privacy_diagnostic_manifest

    reports = []
    for seed in (11, 22, 33):
        report = deepcopy(_privacy_seed_report())
        report["seed"] = seed
        reports.append(report)

    manifest = build_privacy_diagnostic_manifest(
        reports,
        split_manifest_hash="sha256:split",
        metric_report_hash="sha256:metrics",
        counterexample_report={
            "artifact_hash": "sha256:counterexamples",
            "verdict": "PASS",
        },
    )

    assert manifest["policy_fitness_gate"]["controller_metrics_verdict"] == "PASS"
    assert manifest["relative_promotion"]["verdict"] == "PROMOTE"
    paired = manifest["policy_fitness_gate"]["candidate_only_paired_bootstrap"]
    assert paired["profile_count"] == 6
    assert paired["profile_relative_calibration_improvement"]["ci_95"][0] > 0
    assert paired["within_menu_ordering_improvement"]["ci_95"][0] > 0
    assert manifest["preregistered_margins"] == {
        "candidate_calibration": 0.01,
        "candidate_ordering": 0.01,
        "candidate_regret": 0.01,
        "authored_calibration": 0.05,
        "authored_regret": 0.01,
    }


def test_point_improvement_without_bootstrap_support_does_not_pass():
    from copy import deepcopy

    from cloak.train.ranker_diagnostics import build_privacy_diagnostic_manifest

    reports = []
    for seed in (11, 22, 33):
        report = deepcopy(_privacy_seed_report())
        report["seed"] = seed
        semantic_profiles = report["splits"]["test"]["semantic"]["by_profile"]
        candidate_profiles = report["splits"]["test"]["baselines"][
            "candidate_only"
        ]["by_profile"]
        for index, profile_id in enumerate(semantic_profiles):
            candidate = candidate_profiles[profile_id]
            semantic_profiles[profile_id][
                "profile_relative_calibration_error"
            ] = candidate["profile_relative_calibration_error"] + (
                -0.02 if index < 3 else 0.02
            )
            semantic_profiles[profile_id][
                "within_menu_pairwise_accuracy"
            ] = candidate["within_menu_pairwise_accuracy"] + (
                0.02 if index < 3 else -0.02
            )
        reports.append(report)

    manifest = build_privacy_diagnostic_manifest(
        reports,
        split_manifest_hash="sha256:split",
        metric_report_hash="sha256:metrics",
        counterexample_report={
            "artifact_hash": "sha256:counterexamples",
            "verdict": "PASS",
        },
    )

    assert manifest["policy_fitness_gate"]["controller_metrics_verdict"] == "FAIL"
    assert manifest["relative_promotion"]["verdict"] == "FAIL"
