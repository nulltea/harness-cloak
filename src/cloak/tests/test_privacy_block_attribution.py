from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from cloak.train.ranker_privacy import (
    PrivacyExample,
    build_privacy_attribution_model,
    select_privacy_attribution_features,
)


sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "scripts" / "spikes")
)
from privacy_block_attribution import (  # noqa: E402
    build_profile_folds,
    compute_paired_profile_deltas,
)


def _example(
    profile_id: str,
    runtime_type: str,
    *,
    source_family: str = "grounded",
) -> PrivacyExample:
    blocks = torch.tensor([
        10.0, 11.0,  # type
        20.0, 21.0,  # source
        30.0, 31.0,  # joint candidate
        40.0, 41.0,  # candidate - source (prohibited)
        50.0, 51.0,  # Hadamard
    ])
    return PrivacyExample(
        decision_id=f"decision-{profile_id}",
        action_id=f"action-{profile_id}",
        profile_id=profile_id,
        runtime_type=runtime_type,
        grounding_status="grounded",
        source_family=source_family,
        authored_position=0,
        pair_features=blocks,
        candidate_only=torch.tensor([60.0, 61.0]),
        log_count_target=1.0,
        profile_score_target=1.0,
        independent_pair=torch.tensor([
            70.0, 71.0,  # independent source
            80.0, 81.0,  # independent candidate
            85.0, 86.0,  # candidate - source (prohibited)
            90.0, 91.0,  # product
        ]),
    )


@pytest.mark.parametrize(
    ("arm", "expected", "expected_dim", "expected_parameters"),
    [
        ("candidate_projected", [60.0, 61.0], 2, 9),
        ("candidate_native", [60.0, 61.0], 2, 3),
        ("joint_candidate", [30.0, 31.0], 2, 3),
        ("joint_candidate_source", [30.0, 31.0, 20.0, 21.0], 4, 5),
        ("joint_candidate_source_hadamard", [30.0, 31.0, 20.0, 21.0, 50.0, 51.0], 6, 7),
        (
            "joint_full",
            [10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 50.0, 51.0],
            8,
            9,
        ),
        (
            "bi_encoder",
            [70.0, 71.0, 80.0, 81.0, 90.0, 91.0],
            6,
            7,
        ),
    ],
)
def test_attribution_feature_blocks_are_exact_and_have_native_widths(
    arm: str,
    expected: list[float],
    expected_dim: int,
    expected_parameters: int,
):
    features = select_privacy_attribution_features((_example("p", "drug"),), arm)
    model = build_privacy_attribution_model(
        arm=arm,
        pair_dim=10,
        candidate_dim=2,
    )

    assert features.shape == (1, expected_dim)
    assert torch.equal(features[0], torch.tensor(expected))
    assert model.pair_dim == expected_dim
    assert model.projection_dim == 1
    assert model.hidden_dim == 0
    assert model.count_basis_size == 0
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        expected_parameters
    )
    if arm == "candidate_projected":
        assert hasattr(model, "fixed_projection")
        assert model.privacy_projection.in_features == 8
    else:
        assert not hasattr(model, "fixed_projection")
        assert model.privacy_projection.in_features == expected_dim


def test_profile_folds_are_grouped_runtime_stratified_and_exhaust_train_dev():
    rows = tuple(
        _example(f"{runtime_type}-{index}", runtime_type)
        for runtime_type in ("drug", "health-condition")
        for index in range(10)
    )
    manifest = {
        "artifact_hash": "sha256:split",
        "profiles": {
            "train": [
                f"{runtime_type}-{index}"
                for runtime_type in ("drug", "health-condition")
                for index in range(8)
            ],
            "dev": [
                f"{runtime_type}-{index}"
                for runtime_type in ("drug", "health-condition")
                for index in range(8, 10)
            ],
            "test": ["drug-test", "health-condition-test"],
        },
    }

    folds = build_profile_folds(rows, manifest, fold_count=5, seed=20260723)

    assert len(folds) == 5
    observed_dev = []
    for fold in folds:
        train = set(fold["train_profiles"])
        dev = set(fold["dev_profiles"])
        assert train.isdisjoint(dev)
        assert len(train | dev) == 20
        assert fold["runtime_type_counts"]["dev"] == {
            "drug": 2,
            "health-condition": 2,
        }
        observed_dev.extend(dev)
    assert sorted(observed_dev) == sorted(row.profile_id for row in rows)


def test_profile_fold_construction_rejects_spent_test_profile_leakage():
    rows = tuple(
        [_example(f"drug-{index}", "drug") for index in range(5)]
        + [_example("drug-test", "drug")]
    )
    manifest = {
        "artifact_hash": "sha256:split",
        "profiles": {
            "train": [f"drug-{index}" for index in range(4)],
            "dev": ["drug-4"],
            "test": ["drug-test"],
        },
    }

    with pytest.raises(ValueError, match="spent test profile"):
        build_profile_folds(rows, manifest, fold_count=5, seed=7)


def _run(
    arm: str,
    profile_id: str,
    seed: int,
    ordering: float,
    calibration: float,
    *,
    runtime_type: str,
    source_family: str,
) -> dict:
    return {
        "arm": arm,
        "fold": 0,
        "seed": seed,
        "profiles": {
            profile_id: {
                "runtime_type": runtime_type,
                "source_family": source_family,
                "metrics": {
                    "within_menu_pairwise_accuracy": ordering,
                    "profile_relative_calibration_error": calibration,
                    "selected_action_regret": calibration,
                    "median_absolute_log_error": calibration,
                },
            }
        },
    }


def test_paired_deltas_are_profile_paired_seed_collapsed_and_stratified():
    runs = []
    for seed, jitter in ((11, 0.0), (29, 0.01), (47, -0.01)):
        runs.extend([
            _run(
                "candidate_projected", "p1", seed, 0.5 + jitter, 0.4 - jitter,
                runtime_type="drug", source_family="grounded",
            ),
            _run(
                "candidate_native", "p1", seed, 0.7 + jitter, 0.3 - jitter,
                runtime_type="drug", source_family="grounded",
            ),
            _run(
                "candidate_projected", "p2", seed, 0.8 + jitter, 0.2 - jitter,
                runtime_type="health-condition", source_family="curated",
            ),
            _run(
                "candidate_native", "p2", seed, 0.6 + jitter, 0.25 - jitter,
                runtime_type="health-condition", source_family="curated",
            ),
        ])

    report = compute_paired_profile_deltas(
        runs,
        comparisons=(("candidate_projected", "candidate_native"),),
        tie_tolerance=1e-6,
    )["candidate_projected-candidate_native"]

    assert report["positive_delta_means"] == "right_arm_improves"
    assert report["metrics"]["within_menu_pairwise_accuracy"] == {
        "median": 0.0,
        "wins": 1,
        "ties": 0,
        "losses": 1,
    }
    assert report["metrics"]["profile_relative_calibration_error"] == {
        "median": pytest.approx(0.025),
        "wins": 1,
        "ties": 0,
        "losses": 1,
    }
    assert report["by_runtime_type"]["drug"][
        "within_menu_pairwise_accuracy"
    ]["median"] == pytest.approx(0.2)
    assert report["by_source_family"]["curated"][
        "profile_relative_calibration_error"
    ]["median"] == pytest.approx(-0.05)
    assert report["profile_deltas"]["p1"][
        "within_menu_pairwise_accuracy"
    ] == pytest.approx(0.2)
    assert report["profile_deltas"]["p2"][
        "profile_relative_calibration_error"
    ] == pytest.approx(-0.05)
