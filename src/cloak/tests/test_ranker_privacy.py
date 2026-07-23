from __future__ import annotations

import math
import hashlib
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from cloak.train.ranker_privacy import (
    CHECKPOINT_VERSION,
    REQUIRED_BASELINES,
    PrivacyCheckpointContract,
    PrivacyExample,
    PrivacyPrediction,
    SemanticPrivacyHead,
    TrainProfileMeanBaseline,
    build_grouped_profile_split,
    build_neural_privacy_models,
    evaluate_privacy_predictions,
    load_privacy_checkpoint,
    privacy_training_loss,
    profile_normalize_predictions,
    save_privacy_checkpoint,
)
from cloak.train.ranker_environment import (
    RankerAction,
    RankerDecision,
    RankerDocument,
)
from cloak.train.ranker_representation import (
    DocumentTokenBank,
    RelationFeatures,
    build_representation_store,
)


def test_semantic_privacy_head_predicts_valid_log_count_distribution():
    torch.manual_seed(7)
    model = SemanticPrivacyHead(pair_dim=6, projection_dim=4, hidden_dim=5)

    prediction = model(torch.arange(24, dtype=torch.float32).reshape(4, 6) / 10)

    assert prediction.mu_log_count.shape == (4,)
    assert prediction.sigma_log_count.shape == (4,)
    assert torch.all(prediction.mu_log_count >= 0)
    assert torch.all(prediction.sigma_log_count >= 1e-4)
    assert torch.isfinite(prediction.mu_log_count).all()
    assert torch.isfinite(prediction.sigma_log_count).all()


def test_optional_count_basis_is_a_privacy_only_unknown_aware_input():
    torch.manual_seed(9)
    model = SemanticPrivacyHead(
        pair_dim=6,
        projection_dim=4,
        hidden_dim=5,
        count_basis_size=3,
    )
    pair_features = torch.randn(2, 6)

    absent = model(pair_features)
    explicit_unknown = model(pair_features, count_basis=torch.zeros(2, dtype=torch.long))
    known = model(pair_features, count_basis=torch.tensor([1, 2]))

    assert torch.equal(absent.mu_log_count, explicit_unknown.mu_log_count)
    assert not torch.equal(absent.mu_log_count, known.mu_log_count)
    assert model.privacy_projection.in_features == 9


def test_profile_normalization_uses_complete_level_menu_and_exact_endpoints():
    prediction = PrivacyPrediction(
        mu_log_count=torch.tensor([50.0, 2.0, 4.0, 80.0]),
        sigma_log_count=torch.ones(4),
    )

    scores = profile_normalize_predictions(
        prediction, ("keep", "level", "level", "placeholder")
    )

    assert torch.equal(scores, torch.tensor([0.0, 0.5, 1.0, 1.0]))


def test_profile_normalization_assigns_single_level_one():
    prediction = PrivacyPrediction(
        mu_log_count=torch.tensor([10.0, 0.0, 20.0]),
        sigma_log_count=torch.ones(3),
    )

    scores = profile_normalize_predictions(
        prediction, ("keep", "level", "placeholder")
    )

    assert torch.equal(scores, torch.tensor([0.0, 1.0, 1.0]))


@pytest.mark.parametrize(
    "modes",
    [
        ("level", "placeholder"),
        ("keep", "level"),
        ("keep", "keep", "level", "placeholder"),
        ("keep", "level", "placeholder", "unknown"),
    ],
)
def test_profile_normalization_rejects_incomplete_or_invalid_menus(modes):
    prediction = PrivacyPrediction(
        mu_log_count=torch.ones(len(modes)),
        sigma_log_count=torch.ones(len(modes)),
    )

    with pytest.raises(ValueError, match="menu"):
        profile_normalize_predictions(prediction, modes)


def test_privacy_loss_is_finite_and_returns_exact_named_scalar_terms():
    prediction = PrivacyPrediction(
        mu_log_count=torch.tensor([0.2, 1.8, 2.4, 0.8, 2.2]),
        sigma_log_count=torch.tensor([0.5, 0.7, 0.8, 0.6, 0.9]),
    )
    targets = torch.tensor([0.0, math.log(10), math.log(10), math.log(2), math.log(8)])
    profile_scores = torch.tensor([0.0, 1.0, 1.0, 1 / 3, 1.0])

    losses = privacy_training_loss(
        prediction,
        targets,
        (slice(0, 3), slice(3, 5)),
        profile_scores,
        rho=0.4,
        gamma=0.7,
    )

    assert set(losses) == {"nll", "pairwise_rank", "profile_huber", "total"}
    assert all(value.shape == () and torch.isfinite(value) for value in losses.values())
    assert torch.allclose(
        losses["total"],
        losses["nll"] + 0.4 * losses["pairwise_rank"] + 0.7 * losses["profile_huber"],
    )


def test_pairwise_loss_excludes_tied_targets():
    prediction = PrivacyPrediction(
        mu_log_count=torch.tensor([0.1, 9.0]),
        sigma_log_count=torch.ones(2),
    )

    losses = privacy_training_loss(
        prediction,
        torch.tensor([2.0, 2.0]),
        (slice(0, 2),),
        torch.tensor([1.0, 1.0]),
        rho=1.0,
        gamma=0.0,
    )

    assert losses["pairwise_rank"].item() == 0.0


def test_privacy_loss_reaches_only_privacy_projection_and_head():
    torch.manual_seed(11)
    model = SemanticPrivacyHead(pair_dim=6, projection_dim=4, hidden_dim=5)
    utility_projection = nn.Linear(6, 4)
    prediction = model(torch.randn(4, 6))

    losses = privacy_training_loss(
        prediction,
        torch.tensor([0.0, 1.0, 0.5, 2.0]),
        (slice(0, 2), slice(2, 4)),
        torch.tensor([0.0, 1.0, 0.25, 1.0]),
        rho=0.5,
        gamma=0.5,
    )
    losses["total"].backward()

    assert all(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in model.privacy_projection.parameters()
    )
    assert all(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in model.privacy_head.parameters()
    )
    assert all(parameter.grad is None for parameter in utility_projection.parameters())


def _split_rows():
    return [
        {"profile_id": f"{runtime_type}:profile-{index}", "runtime_type": runtime_type}
        for runtime_type in ("drug", "health-condition")
        for index in range(6)
    ]


def test_grouped_split_is_disjoint_deterministic_and_type_stratified():
    manifest = build_grouped_profile_split(
        _split_rows(), seed=19, source_hash="sha256:targets"
    )
    repeated = build_grouped_profile_split(
        list(reversed(_split_rows())), seed=19, source_hash="sha256:targets"
    )

    split_sets = {name: set(values) for name, values in manifest["profiles"].items()}
    assert set(split_sets) == {"train", "dev", "test"}
    assert set.union(*split_sets.values()) == {
        row["profile_id"] for row in _split_rows()
    }
    assert not (split_sets["train"] & split_sets["dev"])
    assert not (split_sets["train"] & split_sets["test"])
    assert not (split_sets["dev"] & split_sets["test"])
    assert all(
        set(manifest["runtime_type_counts"][split])
        == {"drug", "health-condition"}
        for split in ("train", "dev", "test")
    )
    assert manifest == repeated
    assert manifest["artifact_hash"].startswith("sha256:")


def test_grouped_split_fails_when_type_coverage_is_impossible():
    rows = [
        {"profile_id": f"drug:{index}", "runtime_type": "drug"}
        for index in range(2)
    ]

    with pytest.raises(ValueError, match="three profiles"):
        build_grouped_profile_split(rows, seed=3, source_hash="sha256:targets")


def _privacy_examples() -> tuple[PrivacyExample, ...]:
    rows = []
    specifications = (
        ("drug:p1", "drug", "certifying", "openfda", (0.0, 2.0)),
        ("drug:p2", "drug", "model-proposed", "model-proposed", (1.0, 3.0)),
        ("health:p1", "health-condition", "certifying", "doid", (0.5, 2.5)),
    )
    for profile_index, (profile_id, runtime_type, grounding, source_family, targets) in enumerate(
        specifications
    ):
        maximum = max(targets)
        for position, target in enumerate(targets):
            rows.append(PrivacyExample(
                decision_id=f"decision-{profile_index}",
                action_id=f"action-{profile_index}-{position}",
                profile_id=profile_id,
                runtime_type=runtime_type,
                grounding_status=grounding,
                source_family=source_family,
                authored_position=position,
                pair_features=torch.tensor([
                    target, float(position), float(profile_index), 1.0,
                ]),
                candidate_only=torch.tensor([target, float(position)]),
                log_count_target=target,
                profile_score_target=target / maximum if maximum else 1.0,
            ))
    return tuple(rows)


def test_neural_baselines_have_required_names_and_matching_head_budget():
    models = build_neural_privacy_models(
        pair_dim=4,
        candidate_dim=2,
        runtime_types=("drug", "health-condition"),
        projection_dim=3,
        hidden_dim=5,
    )

    assert set(models) == {"semantic", *REQUIRED_BASELINES[:-1]}
    assert REQUIRED_BASELINES == (
        "authored_position_mode_type",
        "mode_type_only",
        "candidate_only",
        "train_profile_mean",
    )
    assert all(model.projection_dim == 3 for model in models.values())
    assert all(model.hidden_dim == 5 for model in models.values())


def test_train_profile_mean_is_stable_by_type_without_profile_memorization():
    training_rows = _privacy_examples()[:4]
    baseline = TrainProfileMeanBaseline.fit(training_rows)
    held_out = tuple(
        replace(row, profile_id="unseen-profile") for row in _privacy_examples()[4:]
    )

    first = baseline.predict(held_out)
    second = baseline.predict(tuple(reversed(held_out)))

    assert not hasattr(baseline, "profile_means")
    assert set(baseline.runtime_type_means) == {"drug"}
    assert torch.equal(first.mu_log_count, torch.flip(second.mu_log_count, dims=(0,)))
    assert torch.all(first.sigma_log_count >= 1e-4)


def test_held_out_metrics_report_every_metric_and_required_stratum():
    examples = _privacy_examples()
    targets = torch.tensor([row.log_count_target for row in examples])
    prediction = PrivacyPrediction(targets, torch.full_like(targets, 0.5))

    report = evaluate_privacy_predictions(examples, prediction)

    metric_names = {
        "nll",
        "median_absolute_log_error",
        "median_multiplicative_error",
        "interval_95_coverage",
        "within_menu_pairwise_accuracy",
        "spearman",
        "profile_relative_calibration_error",
        "selected_action_regret",
    }
    assert set(report["overall"]) == metric_names
    assert set(report["by_runtime_type"]) == {"drug", "health-condition"}
    assert set(report["by_grounding_status"]) == {"certifying", "model-proposed"}
    assert set(report["by_source_family"]) == {"doid", "model-proposed", "openfda"}
    assert report["overall"]["median_absolute_log_error"] == 0.0
    assert report["overall"]["median_multiplicative_error"] == 1.0
    assert report["overall"]["interval_95_coverage"] == 1.0
    assert report["overall"]["within_menu_pairwise_accuracy"] == 1.0
    assert report["overall"]["spearman"] == 1.0
    assert report["overall"]["profile_relative_calibration_error"] == 0.0
    assert report["overall"]["selected_action_regret"] == 0.0


def _checkpoint_contract() -> PrivacyCheckpointContract:
    return PrivacyCheckpointContract(
        environment_hash="sha256:environment",
        profile_target_artifact_hash="sha256:targets",
        representation_manifest_hash="sha256:representations",
        encoder_revision="encoder-revision",
        split_manifest_hash="sha256:split",
        pair_dim=4,
        projection_dim=3,
        hidden_dim=5,
        count_basis_size=0,
        count_basis_categories=(),
        rho=0.5,
        gamma=0.25,
        seeds=(11, 22, 33),
        training_seed=11,
        metric_report_hash="sha256:metrics",
    )


def test_checkpoint_round_trip_and_mismatch_fails_before_model_mutation(tmp_path: Path):
    torch.manual_seed(13)
    source = SemanticPrivacyHead(pair_dim=4, projection_dim=3, hidden_dim=5)
    checkpoint_path = tmp_path / "privacy.pt"
    save_privacy_checkpoint(checkpoint_path, source, _checkpoint_contract())
    saved_state = {
        name: value.detach().clone() for name, value in source.state_dict().items()
    }

    target = SemanticPrivacyHead(pair_dim=4, projection_dim=3, hidden_dim=5)
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.fill_(42.0)
    before_failure = {
        name: value.detach().clone() for name, value in target.state_dict().items()
    }
    mismatched = replace(_checkpoint_contract(), environment_hash="sha256:different")

    with pytest.raises(ValueError, match="contract mismatch"):
        load_privacy_checkpoint(checkpoint_path, target, mismatched)

    assert all(
        torch.equal(target.state_dict()[name], value)
        for name, value in before_failure.items()
    )
    loaded = load_privacy_checkpoint(checkpoint_path, target, _checkpoint_contract())
    assert loaded["checkpoint_version"] == CHECKPOINT_VERSION
    assert all(
        torch.equal(target.state_dict()[name], value)
        for name, value in saved_state.items()
    )


def test_checkpoint_rejects_invalid_state_shapes_before_mutation(tmp_path: Path):
    source = SemanticPrivacyHead(pair_dim=4, projection_dim=3, hidden_dim=5)
    checkpoint_path = tmp_path / "privacy.pt"
    save_privacy_checkpoint(checkpoint_path, source, _checkpoint_contract())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    first_name = next(iter(checkpoint["model_state_dict"]))
    checkpoint["model_state_dict"][first_name] = torch.zeros(1)
    torch.save(checkpoint, checkpoint_path)

    target = SemanticPrivacyHead(pair_dim=4, projection_dim=3, hidden_dim=5)
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.fill_(42.0)
    before = {
        name: value.detach().clone() for name, value in target.state_dict().items()
    }

    with pytest.raises(ValueError, match="state shape mismatch"):
        load_privacy_checkpoint(checkpoint_path, target, _checkpoint_contract())

    assert all(torch.equal(target.state_dict()[name], value) for name, value in before.items())


def test_checkpoint_refuses_internally_inconsistent_contracts(tmp_path: Path):
    model = SemanticPrivacyHead(pair_dim=4, projection_dim=3, hidden_dim=5)

    with pytest.raises(ValueError, match="model contract mismatch"):
        save_privacy_checkpoint(
            tmp_path / "wrong-model.pt",
            model,
            replace(_checkpoint_contract(), pair_dim=5),
        )

    with pytest.raises(ValueError, match="count basis"):
        PrivacyCheckpointContract(
            **{
                **_checkpoint_contract().__dict__,
                "count_basis_size": 2,
                "count_basis_categories": ("<unknown>",),
            }
        )


class SyntheticRepresentationEncoder:
    encoder_id = "stub-encoder"
    encoder_revision = "stub-revision"
    tokenizer_id = "stub-tokenizer"
    tokenizer_revision = "stub-tokenizer-revision"
    hidden_size = 2
    chunk_length = 8
    source_token_overlap = 2
    field_serialization_version = "stub-fields-v1"

    def encode_document(self, doc_id: str, _text: str) -> DocumentTokenBank:
        return DocumentTokenBank(
            doc_id,
            torch.ones((1, 2), dtype=torch.float32),
            torch.tensor([[0, 1]], dtype=torch.int64),
            ((0,),),
        )

    def encode_relation(
        self,
        decision_id: str,
        action_id: str,
        _runtime_type: str,
        _source: str,
        _candidate: str,
    ) -> RelationFeatures:
        value = (sum(action_id.encode("utf-8")) % 17) / 10
        source = torch.tensor([value, value + 0.1], dtype=torch.float32)
        candidate = torch.tensor([value + 0.2, value + 0.3], dtype=torch.float32)
        type_mean = torch.tensor([0.5, 1.0], dtype=torch.float32)
        return RelationFeatures(
            decision_id=decision_id,
            action_id=action_id,
            type_mean=type_mean,
            source_mean=source,
            candidate_mean=candidate,
            pair=torch.cat([
                type_mean, source, candidate, candidate - source, source * candidate,
            ]),
            candidate_only=candidate,
            independent_pair=torch.cat([
                source, candidate, candidate - source, source * candidate,
            ]),
        )


def _synthetic_documents_and_targets():
    documents = {}
    action_targets = {}
    decision_actions = {}
    eligibility = {}
    for runtime_type in ("drug", "health-condition"):
        for profile_index in range(3):
            decision_id = f"{runtime_type}-decision-{profile_index}"
            profile_id = f"{runtime_type}:profile-{profile_index}"
            actions = (
                RankerAction(
                    f"{decision_id}-level-0", "level", "narrow", 0, runtime_type,
                ),
                RankerAction(
                    f"{decision_id}-level-1", "level", "broad", 1, runtime_type,
                ),
                RankerAction(
                    f"{decision_id}-keep", "keep", "source", None, runtime_type,
                ),
                RankerAction(
                    f"{decision_id}-placeholder", "placeholder", None, None, runtime_type,
                ),
            )
            decision = RankerDecision(
                decision_id=decision_id,
                profile_id=profile_id,
                runtime_type=runtime_type,
                canonical_key="source",
                occurrence_ids=(f"occurrence-{decision_id}",),
                actions=actions,
            )
            doc_id = f"fixture/{decision_id}"
            documents[doc_id] = RankerDocument(
                doc_id=doc_id,
                corpus="fixture",
                text="source",
                occurrences=(),
                policy_decisions=(decision,),
                fixed_decisions=(),
            )
            decision_actions[decision_id] = [action.action_id for action in actions]
            eligibility[decision_id] = True
            for action in actions:
                if action.mode == "level":
                    log_count = float(action.authored_level_index * 2)
                    profile_score = float(action.authored_level_index)
                    grounding = "certifying" if profile_index % 2 == 0 else "model-proposed"
                    source_family = "fixture" if grounding == "certifying" else "model-proposed"
                else:
                    log_count = None
                    profile_score = 0.0 if action.mode == "keep" else 1.0
                    grounding = None
                    source_family = None
                action_targets[action.action_id] = {
                    "decision_id": decision_id,
                    "action_id": action.action_id,
                    "profile_id": profile_id,
                    "runtime_type": runtime_type,
                    "mode": action.mode,
                    "log_count": log_count,
                    "profile_score": profile_score,
                    "grounding_status": grounding,
                    "source_family": source_family,
                }
    artifact = {
        "artifact_version": "ranker-v2-profile-count-targets-v1",
        "environment_hash": "sha256:environment",
        "gate_mode": "diagnostic",
        "decision_actions": decision_actions,
        "decision_eligibility": eligibility,
        "action_targets": action_targets,
        "profile_tags": {},
        "gate_report": {"strict_verdict": "FAIL"},
    }
    encoded = json.dumps(
        artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    artifact["artifact_hash"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return documents, artifact


def test_cli_runs_three_seed_stub_smoke_and_writes_bound_artifacts(
    tmp_path: Path, monkeypatch, capsys
):
    documents, targets = _synthetic_documents_and_targets()
    target_path = tmp_path / "profile-targets.json"
    target_path.write_text(json.dumps(targets))
    representation_manifest = build_representation_store(
        documents,
        environment_hash="sha256:environment",
        out_dir=tmp_path / "representations",
        encoder=SyntheticRepresentationEncoder(),
    )
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(json.dumps({
        "frozen_environment": {"environment_hash": "sha256:environment"},
    }))
    out_dir = tmp_path / "privacy"
    cli = importlib.import_module("train_ranker_privacy_head")
    monkeypatch.setattr(cli, "load_ranker_environment", lambda _path: documents)
    monkeypatch.setattr(sys, "argv", [
        "train_ranker_privacy_head.py",
        "--environment", str(environment_path),
        "--profile-targets", str(target_path),
        "--representation-manifest", str(representation_manifest),
        "--out-dir", str(out_dir),
        "--seeds", "11", "22", "33",
        "--split-seed", "5",
        "--projection-dim", "3",
        "--hidden-dim", "4",
        "--max-steps", "1",
        "--use-count-basis",
    ])

    cli.main()

    assert "seeds=11,22,33" in capsys.readouterr().out
    split_manifest = json.loads((out_dir / "split-manifest.json").read_text())
    metrics = json.loads((out_dir / "metrics.json").read_text())
    diagnostics = json.loads((out_dir / "diagnostic-manifest.json").read_text())
    representation_identity = json.loads(
        representation_manifest.read_text()
    )["manifest_hash"]
    assert split_manifest["artifact_version"] == "ranker-v2-profile-split-v1"
    assert [row["seed"] for row in metrics["seed_reports"]] == [11, 22, 33]
    assert set(diagnostics["required_baselines"]) == set(REQUIRED_BASELINES)
    for seed in (11, 22, 33):
        checkpoint = torch.load(
            out_dir / f"seed-{seed}" / "checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )
        contract = checkpoint["contract"]
        assert checkpoint["checkpoint_version"] == CHECKPOINT_VERSION
        assert contract["environment_hash"] == "sha256:environment"
        assert contract["profile_target_artifact_hash"] == targets["artifact_hash"]
        assert contract["representation_manifest_hash"] == representation_identity
        assert contract["split_manifest_hash"] == split_manifest["artifact_hash"]
        assert contract["seeds"] == [11, 22, 33] or contract["seeds"] == (11, 22, 33)
        assert contract["metric_report_hash"] == metrics["artifact_hash"]
        assert contract["count_basis_categories"][0] == "<unknown>"


def test_cli_environment_hash_reader_rejects_mismatch(tmp_path: Path):
    cli = importlib.import_module("train_ranker_privacy_head")
    path = tmp_path / "environment.json"
    path.write_text(json.dumps({
        "frozen_environment": {"environment_hash": "sha256:environment"},
    }))

    assert cli._environment_hash(path) == "sha256:environment"

    path.write_text(json.dumps({"frozen_environment": {}}))
    with pytest.raises(ValueError, match="environment hash"):
        cli._environment_hash(path)


def test_metric_subset_survives_diverged_log_error():
    """One diverged |log error| (> 709) must not overflow the multiplicative metric."""
    import torch

    from cloak.train.ranker_privacy import PrivacyExample, PrivacyPrediction, _metric_subset

    def example(target):
        return PrivacyExample(
            decision_id="d", action_id="a", profile_id="p", runtime_type="drug",
            grounding_status="certifying", source_family="doid", authored_position=0,
            pair_features=torch.zeros(4), candidate_only=torch.zeros(4),
            log_count_target=target, profile_score_target=0.5,
        )

    examples = [example(1.0), example(2.0), example(3.0)]
    prediction = PrivacyPrediction(
        mu_log_count=torch.tensor([1.1, 2.1, 900.0]),
        sigma_log_count=torch.tensor([0.5, 0.5, 0.5]),
    )
    metrics = _metric_subset(examples, prediction, [0.1, 0.2, 0.3], (0, 1, 2))
    assert metrics["median_multiplicative_error"] < 2.0


def test_normalization_survives_all_zero_predictions():
    """Saturated all-zero predicted means must yield zero scores, not a crash."""
    import torch

    from cloak.train.ranker_privacy import (
        PrivacyPrediction,
        _normalize_level_means,
        profile_normalize_predictions,
    )

    zero_means = torch.zeros(3)
    normalized = _normalize_level_means(zero_means, [slice(0, 3)])
    assert torch.equal(normalized, torch.zeros(3))

    prediction = PrivacyPrediction(
        mu_log_count=torch.zeros(4), sigma_log_count=torch.full((4,), 0.5)
    )
    scores = profile_normalize_predictions(prediction, ["keep", "level", "level", "placeholder"])
    assert scores[0].item() == 0.0 and scores[3].item() == 1.0
    assert scores[1].item() == 0.0 and scores[2].item() == 0.0
