import json
from dataclasses import replace

import pytest


SEEDS = (11, 23, 37)


def _contract():
    from cloak.ranker.architecture_diagnostics import MatchedArmContract

    return MatchedArmContract(
        profile_split_hash="sha256:profiles",
        utility_cases_hash="sha256:utility-cases",
        seeds=SEEDS,
        trainable_head_budget=128,
        projection_width=16,
        diagnostic_head_hash="sha256:shared-head",
        encoder_revision="encoder-revision",
        cache_manifest_hash="sha256:cache",
    )


def _metrics(family, arm):
    from cloak.ranker.architecture_diagnostics import (
        LOAD_BEARING_METRICS,
        SELECTED_ARMS,
    )

    selected = SELECTED_ARMS[family]
    value = 0.9 if arm == selected else 0.5
    if family == "action_history" and arm == "utility-gru":
        value = 0.6
    return {
        seed: {
            LOAD_BEARING_METRICS[family]: value + index * 0.01,
            "local_utility": 0.8 + index * 0.01,
            "privacy_ordering": 0.75 + index * 0.01,
            "privacy_calibration_error": 0.12 - index * 0.01,
        }
        for index, seed in enumerate(SEEDS)
    }


def _interventions(family, arm):
    from cloak.ranker.architecture_diagnostics import (
        INTERVENTIONS_BY_FAMILY,
        SELECTED_ARMS,
    )

    if arm != SELECTED_ARMS[family]:
        return {}
    return {
        name: {"passed": True, "observed_effect": 0.25}
        for name in INTERVENTIONS_BY_FAMILY[family]
    }


def _arm(family, arm, *, contract=None):
    from cloak.ranker.architecture_diagnostics import ArmMeasurement

    return ArmMeasurement(
        family=family,
        arm=arm,
        matched_contract=contract or _contract(),
        seed_metrics=_metrics(family, arm),
        trainable_parameters={
            "projection": 32 + len(arm),
            "diagnostic_head": 128,
        },
        interventions=_interventions(family, arm),
        operational_cost={
            "incremental_latency_ms": 2.0,
            "peak_memory_bytes": 1024.0,
            "cache_bytes": 2048.0,
            "cache_build_seconds": 0.5,
        },
        diagnostic_distributions={
            "held_out_outcomes": [0.4, 0.8],
            "by_runtime_type": {"TYPE": [0.4, 0.8]},
            "by_grounding_status": {"grounded": [0.4, 0.8]},
        },
    )


def _arms():
    from cloak.ranker.architecture_diagnostics import APPROVED_ARMS

    return tuple(
        _arm(family, arm)
        for family, family_arms in APPROVED_ARMS.items()
        for arm in family_arms
    )


def _thresholds():
    return {
        "max_incremental_latency_ms": 5.0,
        "max_peak_memory_bytes": 4096.0,
        "max_cache_bytes": 8192.0,
        "max_cache_build_seconds": 2.0,
        "max_order_utility_range": 0.5,
    }


def test_report_has_exact_required_sections_all_arms_and_content_hash():
    from cloak.ranker.architecture_diagnostics import (
        APPROVED_ARMS,
        REPORT_SECTIONS,
        build_architecture_spike_report,
    )

    report = build_architecture_spike_report(
        _arms(),
        registered_thresholds=_thresholds(),
        aci_context=False,
        non_aci_manifest_hash="sha256:non-aci",
    )

    assert set(REPORT_SECTIONS).issubset(report)
    for family in (
        "relation_representation", "context_readout", "action_history",
        "metadata_shortcut", "decision_order",
    ):
        assert set(report[family]["arms"]) == set(APPROVED_ARMS[family])
    assert report["artifact_version"] == "ranker-v2-architecture-spike-v1"
    assert report["artifact_hash"].startswith("sha256:")
    assert report["operational_cost"]["maxima"]["cache_build_seconds"] == 0.5
    assert report["semantic_privacy_transfer"]["diagnostic_distributions"]
    encoded = json.dumps(report, sort_keys=True, allow_nan=False)
    assert "padding" not in encoded.casefold()


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("profile_split_hash", "sha256:different", "profile split"),
        ("utility_cases_hash", "sha256:different", "utility cases"),
        ("seeds", (3, 5, 7), "seeds"),
        ("trainable_head_budget", 256, "trainable-head budget"),
        ("encoder_revision", "different", "encoder revision"),
        ("cache_manifest_hash", "sha256:different", "cache manifest"),
    ],
)
def test_report_refuses_unmatched_arm_contracts(field, replacement, message):
    from cloak.ranker.architecture_diagnostics import (
        build_architecture_spike_report,
    )

    arms = list(_arms())
    changed_contract = replace(_contract(), **{field: replacement})
    changed_metrics = arms[1].seed_metrics
    if field == "seeds":
        changed_metrics = {
            new_seed: dict(arms[1].seed_metrics[old_seed])
            for old_seed, new_seed in zip(SEEDS, replacement, strict=True)
        }
    changed_parameters = arms[1].trainable_parameters
    if field == "trainable_head_budget":
        changed_parameters = {
            **dict(arms[1].trainable_parameters),
            "diagnostic_head": replacement,
        }
    arms[1] = replace(
        arms[1],
        matched_contract=changed_contract,
        seed_metrics=changed_metrics,
        trainable_parameters=changed_parameters,
    )
    with pytest.raises(ValueError, match=message):
        build_architecture_spike_report(
            arms,
            registered_thresholds=_thresholds(),
            aci_context=False,
            non_aci_manifest_hash="sha256:non-aci",
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("projection_width", 32, "projection width"),
        ("diagnostic_head_hash", "sha256:different", "diagnostic head"),
    ],
)
def test_report_refuses_nonshared_projection_or_diagnostic_head(
    field, replacement, message,
):
    from cloak.ranker.architecture_diagnostics import (
        build_architecture_spike_report,
    )

    arms = list(_arms())
    arms[-1] = replace(
        arms[-1], matched_contract=replace(_contract(), **{field: replacement}),
    )
    with pytest.raises(ValueError, match=message):
        build_architecture_spike_report(
            arms,
            registered_thresholds=_thresholds(),
            aci_context=False,
            non_aci_manifest_hash="sha256:non-aci",
        )


def test_arm_measurement_refuses_a_misreported_shared_head_budget():
    row = _arm("context_readout", "local-cls-mean")

    with pytest.raises(ValueError, match="diagnostic-head trainable count"):
        replace(
            row,
            trainable_parameters={
                **dict(row.trainable_parameters),
                "diagnostic_head": 64,
            },
        )


def test_matched_contract_loader_does_not_coerce_missing_hashes_to_strings():
    from cloak.ranker.architecture_diagnostics import MatchedArmContract

    payload = _contract().to_dict()
    payload["profile_split_hash"] = None

    with pytest.raises(ValueError, match="profile_split_hash"):
        MatchedArmContract.from_mapping(payload)


def test_required_intervention_measurement_cannot_be_omitted():
    from cloak.ranker.architecture_diagnostics import (
        build_architecture_spike_report,
    )

    arms = list(_arms())
    selected_relation = next(
        index for index, row in enumerate(arms)
        if row.family == "relation_representation" and row.arm == "joint-pair"
    )
    interventions = dict(arms[selected_relation].interventions)
    del interventions["candidate_swap"]
    arms[selected_relation] = replace(
        arms[selected_relation], interventions=interventions,
    )

    with pytest.raises(ValueError, match="candidate_swap"):
        build_architecture_spike_report(
            arms,
            registered_thresholds=_thresholds(),
            aci_context=False,
            non_aci_manifest_hash="sha256:non-aci",
        )


def test_relative_promotion_uses_paired_three_seed_cis_and_all_hard_rules():
    from cloak.ranker.architecture_diagnostics import (
        build_architecture_spike_report,
    )

    report = build_architecture_spike_report(
        _arms(),
        registered_thresholds=_thresholds(),
        aci_context=False,
        non_aci_manifest_hash="sha256:non-aci",
    )

    assert report["promotion_verdict"]["verdict"] == "PROMOTE"
    comparisons = report["promotion_verdict"]["paired_bootstrap"]
    assert comparisons
    assert all(row["confidence_interval_95"][0] > 0.0 for row in comparisons)
    assert all(row["seeds"] == list(SEEDS) for row in comparisons)
    assert report["action_history"]["selected_arm"] == "selected-cross-attention"
    assert report["promotion_verdict"]["local_utility_regression"] is False
    assert report["promotion_verdict"]["shortcut_invariance_passed"] is True
    assert report["promotion_verdict"]["operational_budget_passed"] is True


def test_local_regression_shortcut_failure_and_cost_overrun_each_reject():
    from cloak.ranker.architecture_diagnostics import (
        build_architecture_spike_report,
    )

    local_rows = list(_arms())
    context_index = next(
        index for index, row in enumerate(local_rows)
        if row.family == "context_readout"
        and row.arm == "full-candidate-attention"
    )
    context = local_rows[context_index]
    local_rows[context_index] = replace(
        context,
        seed_metrics={
            seed: {**dict(values), "local_utility": 0.1}
            for seed, values in context.seed_metrics.items()
        },
    )

    shortcut_rows = list(_arms())
    shortcut_index = next(
        index for index, row in enumerate(shortcut_rows)
        if row.family == "metadata_shortcut" and row.arm == "semantic"
    )
    shortcut = shortcut_rows[shortcut_index]
    interventions = {
        name: dict(values) for name, values in shortcut.interventions.items()
    }
    interventions["count_metadata_mutation"]["passed"] = False
    shortcut_rows[shortcut_index] = replace(
        shortcut, interventions=interventions,
    )

    cost_rows = list(_arms())
    cost_rows[0] = replace(
        cost_rows[0],
        operational_cost={
            **dict(cost_rows[0].operational_cost),
            "incremental_latency_ms": 10.0,
        },
    )

    for rows in (local_rows, shortcut_rows, cost_rows):
        report = build_architecture_spike_report(
            rows,
            registered_thresholds=_thresholds(),
            aci_context=False,
            non_aci_manifest_hash="sha256:non-aci",
        )
        assert report["promotion_verdict"]["verdict"] == "REJECT"


def test_history_chooses_none_when_memory_does_not_beat_it_and_never_promotes_gru():
    from cloak.ranker.architecture_diagnostics import (
        LOAD_BEARING_METRICS,
        build_architecture_spike_report,
    )

    arms = list(_arms())
    for index, row in enumerate(arms):
        if row.family != "action_history":
            continue
        values = {
            "none": 0.8,
            "utility-gru": 0.95,
            "selected-cross-attention": 0.79,
        }
        metrics = {
            seed: {
                **dict(row.seed_metrics[seed]),
                LOAD_BEARING_METRICS["action_history"]: values[row.arm],
            }
            for seed in SEEDS
        }
        arms[index] = replace(row, seed_metrics=metrics)

    report = build_architecture_spike_report(
        arms,
        registered_thresholds=_thresholds(),
        aci_context=False,
        non_aci_manifest_hash="sha256:non-aci",
    )

    assert report["action_history"]["selected_arm"] == "none"
    assert report["action_history"]["gru_auto_promotion_allowed"] is False
    assert report["promotion_verdict"]["verdict"] == "PROMOTE"


def test_unregistered_absolute_thresholds_never_become_an_invented_pass():
    from cloak.ranker.architecture_diagnostics import (
        build_architecture_spike_report,
    )

    report = build_architecture_spike_report(
        _arms(),
        registered_thresholds={},
        aci_context=False,
        non_aci_manifest_hash="sha256:non-aci",
    )

    assert report["promotion_verdict"]["verdict"] == (
        "NEEDS_THRESHOLD_REGISTRATION"
    )
    assert set(report["promotion_verdict"]["missing_thresholds"]) == set(
        _thresholds()
    )


def test_aci_contamination_is_a_hard_promotion_boundary():
    from cloak.ranker.architecture_diagnostics import (
        build_architecture_spike_report,
    )

    report = build_architecture_spike_report(
        _arms(),
        registered_thresholds=_thresholds(),
        aci_context=True,
        non_aci_manifest_hash=None,
    )

    assert report["contamination_boundary"] == {
        "label": "development_only_encoder_contaminated",
        "non_aci_manifest_hash": None,
        "encoder_selection_promotion_allowed": False,
        "out_of_corpus_generalization_allowed": False,
    }
    assert report["promotion_verdict"]["verdict"] == (
        "DEVELOPMENT_ONLY_ENCODER_CONTAMINATED"
    )
    assert report["promotion_verdict"]["verdict"] != "PROMOTE"


def test_encoder_promotion_requires_an_explicit_non_aci_manifest_even_when_not_aci():
    from cloak.ranker.architecture_diagnostics import (
        build_architecture_spike_report,
    )

    report = build_architecture_spike_report(
        _arms(),
        registered_thresholds=_thresholds(),
        aci_context=False,
        non_aci_manifest_hash=None,
    )

    assert report["contamination_boundary"]["label"] == "non_aci_manifest_missing"
    assert report["promotion_verdict"]["verdict"] == "NEEDS_NON_ACI_VALIDATION"
    assert report["promotion_verdict"]["verdict"] != "PROMOTE"


def test_runner_enumerates_every_approved_arm_under_one_contract():
    from cloak.ranker.architecture_diagnostics import (
        APPROVED_ARMS,
        run_architecture_spike,
    )

    calls = []

    def evaluator(family, arm, contract):
        calls.append((family, arm, contract))
        return _arm(family, arm, contract=contract)

    report = run_architecture_spike(
        _contract(),
        evaluator,
        registered_thresholds=_thresholds(),
        aci_context=False,
        non_aci_manifest_hash="sha256:non-aci",
    )

    assert [(family, arm) for family, arm, _ in calls] == [
        (family, arm)
        for family, arms in APPROVED_ARMS.items()
        for arm in arms
    ]
    assert all(contract == _contract() for _, _, contract in calls)
    assert report["promotion_verdict"]["verdict"] == "PROMOTE"


def test_runner_rejects_an_evaluator_that_switches_contracts():
    from cloak.ranker.architecture_diagnostics import run_architecture_spike

    def evaluator(family, arm, contract):
        if family == "relation_representation" and arm == "candidate-only":
            contract = replace(contract, cache_manifest_hash="sha256:different")
        return _arm(family, arm, contract=contract)

    with pytest.raises(ValueError, match="runner contract"):
        run_architecture_spike(
            _contract(),
            evaluator,
            registered_thresholds=_thresholds(),
            aci_context=False,
            non_aci_manifest_hash="sha256:non-aci",
        )


def test_cli_builds_content_addressed_report_from_synthetic_measurements(tmp_path):
    import run_ranker_architecture_spike

    payload = {
        "matched_contract": _contract().to_dict(),
        "measurements": [row.to_dict() for row in _arms()],
        "registered_thresholds": _thresholds(),
        "aci_context": True,
        "non_aci_manifest_hash": None,
    }
    source = tmp_path / "measurements.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    destination = tmp_path / "report.json"

    report = run_ranker_architecture_spike.main([
        "--measurements", str(source),
        "--out", str(destination),
    ])

    assert destination.exists()
    assert json.loads(destination.read_text()) == report
    assert report["promotion_verdict"]["verdict"] == (
        "DEVELOPMENT_ONLY_ENCODER_CONTAMINATED"
    )
