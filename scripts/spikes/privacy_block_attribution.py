#!/usr/bin/env python3
"""Run the profile-grouped nested privacy feature-block attribution spike."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from cloak.ranker.environment import load_ranker_environment
from cloak.ranker.privacy import (
    PREDICTION_TIE_TOLERANCE,
    PRIVACY_ATTRIBUTION_ARMS,
    PrivacyExample,
    load_privacy_examples,
    train_privacy_attribution_fold,
)
from cloak.ranker.representation import RankerRepresentationStore


ARTIFACT_VERSION = "ranker-v2-privacy-block-attribution-v2"
FOLD_VERSION = "ranker-v2-privacy-block-attribution-folds-v1"
FOLD_COUNT = 5
FOLD_SEED = 20260723
SEEDS = (11, 29, 47)
COMPARISONS = (
    ("candidate_projected", "candidate_native"),
    ("candidate_native", "joint_candidate"),
    ("joint_candidate", "joint_candidate_source"),
    ("joint_candidate_source", "joint_candidate_source_hadamard"),
    ("joint_candidate_source_hadamard", "joint_full"),
    ("candidate_native", "bi_encoder"),
)
METRICS = (
    "within_menu_pairwise_accuracy",
    "profile_relative_calibration_error",
    "selected_action_regret",
    "median_absolute_log_error",
)
LOWER_IS_BETTER = frozenset(METRICS[1:])
DEFAULT_OUT_DIR = Path("results/ranker_v2/architecture/block-attribution")


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"artifact must be a mapping: {path}")
    return value


def _validate_content_hash(artifact: Mapping[str, Any], name: str) -> str:
    artifact_hash = artifact.get(name)
    if not isinstance(artifact_hash, str) or not artifact_hash.startswith("sha256:"):
        raise ValueError(f"artifact lacks {name}")
    expected = _stable_hash({
        key: value for key, value in artifact.items() if key != name
    })
    if artifact_hash != expected:
        raise ValueError(f"artifact {name} mismatch")
    return artifact_hash


def build_profile_folds(
    examples: Sequence[PrivacyExample],
    split_manifest: Mapping[str, Any],
    *,
    fold_count: int = FOLD_COUNT,
    seed: int = FOLD_SEED,
) -> tuple[dict[str, Any], ...]:
    """Create deterministic runtime-stratified folds from train+dev profiles only."""
    if fold_count < 2 or not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("privacy attribution fold protocol is invalid")
    profiles = split_manifest.get("profiles")
    if not isinstance(profiles, Mapping) or set(profiles) != {
        "train", "dev", "test"
    }:
        raise ValueError("privacy attribution requires the existing three-way split")
    partitions = {
        name: frozenset(str(value) for value in profiles[name])
        for name in ("train", "dev", "test")
    }
    if (
        any(not values for values in partitions.values())
        or any(
            partitions[left] & partitions[right]
            for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
        )
    ):
        raise ValueError("privacy attribution source split is invalid")
    observed = {row.profile_id for row in examples}
    leaked = observed & partitions["test"]
    if leaked:
        raise ValueError("spent test profile leaked into attribution folds")
    admitted = partitions["train"] | partitions["dev"]
    if observed != admitted:
        raise ValueError(
            "attribution rows must exactly cover source train+dev profiles"
        )

    profile_types: dict[str, str] = {}
    for row in examples:
        previous = profile_types.setdefault(row.profile_id, row.runtime_type)
        if previous != row.runtime_type:
            raise ValueError("privacy attribution profile crosses runtime types")
    by_type: dict[str, list[str]] = defaultdict(list)
    for profile_id, runtime_type in profile_types.items():
        by_type[runtime_type].append(profile_id)
    fold_dev: list[list[str]] = [[] for _ in range(fold_count)]
    for runtime_type, profile_ids in sorted(by_type.items()):
        if len(profile_ids) < fold_count:
            raise ValueError(
                f"runtime type {runtime_type} has fewer profiles than folds"
            )
        ordered = sorted(
            profile_ids,
            key=lambda profile_id: hashlib.sha256(
                f"{seed}\0{runtime_type}\0{profile_id}".encode("utf-8")
            ).hexdigest(),
        )
        for index, profile_id in enumerate(ordered):
            fold_dev[index % fold_count].append(profile_id)

    folds = []
    for fold_index, dev_profiles in enumerate(fold_dev):
        dev = frozenset(dev_profiles)
        train = admitted - dev
        runtime_type_counts = {
            split: {
                runtime_type: sum(
                    profile_types[profile_id] == runtime_type
                    for profile_id in profile_ids
                )
                for runtime_type in sorted(by_type)
            }
            for split, profile_ids in (("train", train), ("dev", dev))
        }
        folds.append({
            "fold": fold_index,
            "train_profiles": sorted(train),
            "dev_profiles": sorted(dev),
            "runtime_type_counts": runtime_type_counts,
        })
    return tuple(folds)


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize empty attribution deltas")
    return float(statistics.median(values))


def _summarize_deltas(
    values: Sequence[float],
    *,
    tie_tolerance: float,
) -> dict[str, float | int]:
    median = _median(values)
    return {
        "median": 0.0 if abs(median) <= tie_tolerance else median,
        "wins": sum(value > tie_tolerance for value in values),
        "ties": sum(abs(value) <= tie_tolerance for value in values),
        "losses": sum(value < -tie_tolerance for value in values),
    }


def _profile_source_families(profile: Mapping[str, Any]) -> tuple[str, ...]:
    source_families = profile.get("source_families")
    if source_families is None:
        source_family = profile.get("source_family")
        source_families = [source_family]
    if (
        not isinstance(source_families, Sequence)
        or isinstance(source_families, (str, bytes))
        or not source_families
        or any(not isinstance(value, str) or not value for value in source_families)
    ):
        raise ValueError("attribution profile source families are invalid")
    return tuple(sorted(set(source_families)))


def compute_paired_profile_deltas(
    runs: Sequence[Mapping[str, Any]],
    *,
    comparisons: Sequence[tuple[str, str]] = COMPARISONS,
    tie_tolerance: float = PREDICTION_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Pair arms by profile and seed, then collapse seed deltas by median."""
    if tie_tolerance < 0 or not math.isfinite(tie_tolerance):
        raise ValueError("attribution tie tolerance must be finite and nonnegative")
    indexed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    metadata: dict[str, tuple[str, tuple[str, ...]]] = {}
    for run in runs:
        arm = run.get("arm")
        seed = run.get("seed")
        profiles = run.get("profiles")
        if (
            arm not in PRIVACY_ATTRIBUTION_ARMS
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or not isinstance(profiles, Mapping)
        ):
            raise ValueError("attribution run is malformed")
        for profile_id, profile in profiles.items():
            if not isinstance(profile_id, str) or not isinstance(profile, Mapping):
                raise ValueError("attribution profile result is malformed")
            runtime_type = profile.get("runtime_type")
            metrics = profile.get("metrics")
            if not isinstance(runtime_type, str) or not isinstance(metrics, Mapping):
                raise ValueError("attribution profile metadata is malformed")
            profile_metadata = (
                runtime_type,
                _profile_source_families(profile),
            )
            previous = metadata.setdefault(profile_id, profile_metadata)
            if previous != profile_metadata:
                raise ValueError("attribution profile strata changed across runs")
            key = (arm, profile_id, seed)
            if key in indexed:
                raise ValueError("duplicate attribution arm/profile/seed result")
            indexed[key] = metrics

    output = {}
    for left_arm, right_arm in comparisons:
        comparison_id = f"{left_arm}-{right_arm}"
        profiles_by_arm = {
            arm: {
                profile_id
                for candidate_arm, profile_id, _ in indexed
                if candidate_arm == arm
            }
            for arm in (left_arm, right_arm)
        }
        if (
            not profiles_by_arm[left_arm]
            or profiles_by_arm[left_arm] != profiles_by_arm[right_arm]
        ):
            raise ValueError(f"unpaired profiles for comparison {comparison_id}")
        profile_deltas: dict[str, dict[str, float]] = {}
        for profile_id in sorted(profiles_by_arm[left_arm]):
            seeds_by_arm = {
                arm: {
                    seed
                    for candidate_arm, candidate_profile, seed in indexed
                    if candidate_arm == arm and candidate_profile == profile_id
                }
                for arm in (left_arm, right_arm)
            }
            if seeds_by_arm[left_arm] != seeds_by_arm[right_arm]:
                raise ValueError(
                    f"unpaired seeds for {comparison_id}:{profile_id}"
                )
            collapsed = {}
            for metric in METRICS:
                seed_deltas = []
                for seed in sorted(seeds_by_arm[left_arm]):
                    left = indexed[(left_arm, profile_id, seed)].get(metric)
                    right = indexed[(right_arm, profile_id, seed)].get(metric)
                    if left is None or right is None:
                        continue
                    left_value = float(left)
                    right_value = float(right)
                    if not all(math.isfinite(value) for value in (
                        left_value, right_value
                    )):
                        raise ValueError("attribution metric must be finite")
                    delta = right_value - left_value
                    if metric in LOWER_IS_BETTER:
                        delta = -delta
                    seed_deltas.append(delta)
                if seed_deltas:
                    collapsed[metric] = _median(seed_deltas)
            profile_deltas[profile_id] = collapsed

        def summaries(profile_ids: Sequence[str]) -> dict[str, Any]:
            return {
                metric: _summarize_deltas(
                    [
                        profile_deltas[profile_id][metric]
                        for profile_id in profile_ids
                        if metric in profile_deltas[profile_id]
                    ],
                    tie_tolerance=tie_tolerance,
                )
                for metric in METRICS
                if any(
                    metric in profile_deltas[profile_id]
                    for profile_id in profile_ids
                )
            }

        all_profiles = sorted(profile_deltas)
        runtime_types = sorted({metadata[value][0] for value in all_profiles})
        source_families = sorted({
            source_family
            for value in all_profiles
            for source_family in metadata[value][1]
        })
        output[comparison_id] = {
            "left_arm": left_arm,
            "right_arm": right_arm,
            "positive_delta_means": "right_arm_improves",
            "seed_collapse": "median",
            "profile_deltas": profile_deltas,
            "metrics": summaries(all_profiles),
            "by_runtime_type": {
                runtime_type: summaries([
                    profile_id
                    for profile_id in all_profiles
                    if metadata[profile_id][0] == runtime_type
                ])
                for runtime_type in runtime_types
            },
            "by_source_family": {
                source_family: summaries([
                    profile_id
                    for profile_id in all_profiles
                    if source_family in metadata[profile_id][1]
                ])
                for source_family in source_families
            },
        }
    return output


def _summary_markdown(report: Mapping[str, Any], report_name: str) -> str:
    lines = [
        "# Privacy block attribution",
        "",
        f"Artifact: `{report_name}`",
        "",
        "Positive paired deltas mean the right-hand arm improves.",
        "",
        "| Arm | Width | Trainable | Ordering | Calibration | Regret | Median log error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    runs = report["runs"]
    for arm in PRIVACY_ATTRIBUTION_ARMS:
        arm_runs = [run for run in runs if run["arm"] == arm]
        lines.append(
            "| {arm} | {width} | {parameters} | {ordering:.6g} | "
            "{calibration:.6g} | {regret:.6g} | {error:.6g} |".format(
                arm=arm,
                width=arm_runs[0]["feature_dim"],
                parameters=arm_runs[0]["trainable_parameters"],
                ordering=_median([
                    run["metrics"]["within_menu_pairwise_accuracy"]
                    for run in arm_runs
                    if run["metrics"]["within_menu_pairwise_accuracy"] is not None
                ]),
                calibration=_median([
                    run["metrics"]["profile_relative_calibration_error"]
                    for run in arm_runs
                ]),
                regret=_median([
                    run["metrics"]["selected_action_regret"] for run in arm_runs
                ]),
                error=_median([
                    run["metrics"]["median_absolute_log_error"] for run in arm_runs
                ]),
            )
        )
    lines.extend([
        "",
        "| Comparison | Metric | Median favorable delta | W/T/L |",
        "|---|---|---:|---:|",
    ])
    for comparison, comparison_report in report["paired_profile_deltas"].items():
        for metric in METRICS:
            value = comparison_report["metrics"].get(metric)
            if value is None:
                continue
            lines.append(
                f"| {comparison} | {metric} | {value['median']:.6g} | "
                f"{value['wins']}/{value['ties']}/{value['losses']} |"
            )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--profile-targets", required=True)
    parser.add_argument("--representation-manifest", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    environment_path = Path(args.environment)
    target_path = Path(args.profile_targets)
    representation_path = Path(args.representation_manifest)
    split_path = Path(args.split_manifest)
    out_dir = Path(args.out_dir)
    environment_artifact = _read_mapping(environment_path)
    targets = _read_mapping(target_path)
    split_manifest = _read_mapping(split_path)
    target_hash = _validate_content_hash(targets, "artifact_hash")
    split_hash = _validate_content_hash(split_manifest, "artifact_hash")
    if split_manifest.get("source_hash") != target_hash:
        raise ValueError("split manifest is not bound to the profile targets")
    profiles = split_manifest.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("split manifest lacks profile partitions")
    admitted_profiles = tuple(profiles.get("train", ())) + tuple(
        profiles.get("dev", ())
    )
    documents = load_ranker_environment(environment_path)
    store = RankerRepresentationStore.open(representation_path)
    environment_hash = targets.get("environment_hash")
    frozen_environment = environment_artifact.get("frozen_environment")
    authored_environment_hash = (
        frozen_environment.get("environment_hash")
        if isinstance(frozen_environment, Mapping) else None
    )
    if (
        not isinstance(environment_hash, str)
        or authored_environment_hash != environment_hash
        or store.manifest.get("environment_hash") != environment_hash
    ):
        raise ValueError("attribution input environment hashes differ")
    examples = load_privacy_examples(
        documents,
        targets,
        store,
        admitted_profile_ids=admitted_profiles,
    )
    folds = build_profile_folds(examples, split_manifest)
    fold_manifest = {
        "artifact_version": FOLD_VERSION,
        "source_split_manifest_hash": split_hash,
        "fold_count": FOLD_COUNT,
        "seed": FOLD_SEED,
        "folds": list(folds),
    }
    fold_manifest["artifact_hash"] = _stable_hash(fold_manifest)
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    runs = []
    for fold in folds:
        for seed in SEEDS:
            for arm in PRIVACY_ATTRIBUTION_ARMS:
                _, run = train_privacy_attribution_fold(
                    examples,
                    train_profiles=fold["train_profiles"],
                    validation_profiles=fold["dev_profiles"],
                    arm=arm,
                    seed=seed,
                    device=device,
                )
                run["fold"] = fold["fold"]
                runs.append(run)

    representation_hash = store.manifest.get("manifest_hash")
    encoder_revision = store.manifest.get("encoder", {}).get("revision")
    if not all(
        isinstance(value, str) and value
        for value in (environment_hash, representation_hash, encoder_revision)
    ):
        raise ValueError("input artifacts lack attribution identity pins")
    report = {
        "artifact_version": ARTIFACT_VERSION,
        "pins": {
            "environment_hash": environment_hash,
            "profile_target_artifact_hash": target_hash,
            "representation_manifest_hash": representation_hash,
            "split_manifest_hash": split_hash,
            "encoder_revision": encoder_revision,
        },
        "protocol": {
            "arms": list(PRIVACY_ATTRIBUTION_ARMS),
            "comparisons": [f"{left}-{right}" for left, right in COMPARISONS],
            "folds": FOLD_COUNT,
            "fold_source_partitions": ["train", "dev"],
            "test_partition_status": "excluded-spent",
            "seeds": list(SEEDS),
            "mean_head": "linear",
            "rho": 0.0,
            "gamma": 1.0,
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "max_updates": 500,
            "evaluation_interval": 10,
            "patience": 10,
            "selection": [
                "profile_relative_calibration_error",
                "within_menu_pairwise_accuracy",
                "mean_absolute_log_error_guard",
            ],
            "prediction_tie_tolerance": PREDICTION_TIE_TOLERANCE,
        },
        "fold_manifest": fold_manifest,
        "runs": runs,
        "paired_profile_deltas": compute_paired_profile_deltas(runs),
    }
    report["artifact_hash"] = _stable_hash(report)
    digest = report["artifact_hash"].removeprefix("sha256:")
    report_name = f"block-attribution-{digest}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / report_name).write_text(json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n")
    (out_dir / "summary.md").write_text(
        _summary_markdown(report, report_name)
    )
    print(json.dumps({
        "status": "complete",
        "artifact": str(out_dir / report_name),
        "artifact_hash": report["artifact_hash"],
        "runs": len(runs),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
