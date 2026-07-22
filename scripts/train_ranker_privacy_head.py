#!/usr/bin/env python3
"""Pretrain the Ranker-v2 semantic log-count head from frozen representations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from cloak.train.ranker_diagnostics import build_privacy_diagnostic_manifest
from cloak.train.ranker_environment import load_ranker_environment
from cloak.train.ranker_privacy import (
    METRIC_REPORT_VERSION,
    PrivacyCheckpointContract,
    build_grouped_profile_split,
    load_privacy_examples,
    save_privacy_checkpoint,
    train_privacy_seed,
)
from cloak.train.ranker_representation import RankerRepresentationStore


DEFAULT_OUT_DIR = Path("results/ranker_v2/architecture/privacy")


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ) + "\n")


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"artifact must be a mapping: {path}")
    return value


def _environment_hash(path: Path) -> str:
    artifact = _read_mapping(path)
    frozen = artifact.get("frozen_environment")
    environment_hash = (
        frozen.get("environment_hash") if isinstance(frozen, dict) else None
    )
    if not isinstance(environment_hash, str) or not environment_hash:
        raise ValueError("environment hash is missing")
    return environment_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--profile-targets", required=True)
    parser.add_argument("--representation-manifest", required=True)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seeds", nargs=3, type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=1729)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--rho", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--use-count-basis", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = tuple(args.seeds)
    if len(set(seeds)) != 3:
        raise ValueError("privacy pretraining requires three distinct explicit seeds")
    if args.projection_dim <= 0 or args.hidden_dim <= 0:
        raise ValueError("privacy model dimensions must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    environment_path = Path(args.environment)
    target_path = Path(args.profile_targets)
    representation_path = Path(args.representation_manifest)
    out_dir = Path(args.out_dir)
    targets = _read_mapping(target_path)
    store = RankerRepresentationStore.open(representation_path)
    environment_hash = targets.get("environment_hash")
    if not isinstance(environment_hash, str) or not environment_hash:
        raise ValueError("profile targets lack environment hash")
    if store.manifest.get("environment_hash") != environment_hash:
        raise ValueError("representation environment hash mismatch")
    if _environment_hash(environment_path) != environment_hash:
        raise ValueError("profile-target environment hash mismatch")
    target_hash = targets.get("artifact_hash")
    if not isinstance(target_hash, str) or not target_hash:
        raise ValueError("profile targets lack artifact hash")
    encoder_revision = store.manifest.get("encoder", {}).get("revision")
    if not isinstance(encoder_revision, str) or not encoder_revision:
        raise ValueError("representation manifest lacks encoder revision")

    documents = load_ranker_environment(environment_path)
    examples = load_privacy_examples(documents, targets, store)
    split_manifest = build_grouped_profile_split(
        examples,
        seed=args.split_seed,
        source_hash=target_hash,
    )
    _write_json(out_dir / "split-manifest.json", split_manifest)

    models = {}
    seed_reports = []
    for seed in seeds:
        model, report = train_privacy_seed(
            examples,
            split_manifest,
            seed=seed,
            projection_dim=args.projection_dim,
            hidden_dim=args.hidden_dim,
            rho=args.rho,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            use_count_basis=args.use_count_basis,
        )
        models[seed] = model
        seed_reports.append(report)
        _write_json(out_dir / f"seed-{seed}" / "metrics.json", report)

    metric_report = {
        "artifact_version": METRIC_REPORT_VERSION,
        "profile_held_out": True,
        "split_manifest_hash": split_manifest["artifact_hash"],
        "seeds": list(seeds),
        "seed_reports": seed_reports,
    }
    metric_report["artifact_hash"] = _stable_hash(metric_report)
    _write_json(out_dir / "metrics.json", metric_report)

    representation_hash = _file_hash(representation_path)
    for seed in seeds:
        model = models[seed]
        contract = PrivacyCheckpointContract(
            environment_hash=environment_hash,
            profile_target_artifact_hash=target_hash,
            representation_manifest_hash=representation_hash,
            encoder_revision=encoder_revision,
            split_manifest_hash=split_manifest["artifact_hash"],
            pair_dim=model.pair_dim,
            projection_dim=model.projection_dim,
            hidden_dim=model.hidden_dim,
            count_basis_size=model.count_basis_size,
            count_basis_categories=tuple(
                seed_reports[seeds.index(seed)]["count_basis_categories"]
            ),
            rho=float(args.rho),
            gamma=float(args.gamma),
            seeds=seeds,
            training_seed=seed,
            metric_report_hash=metric_report["artifact_hash"],
        )
        save_privacy_checkpoint(
            out_dir / f"seed-{seed}" / "checkpoint.pt", model, contract
        )

    diagnostic_manifest = build_privacy_diagnostic_manifest(
        seed_reports,
        split_manifest_hash=split_manifest["artifact_hash"],
        metric_report_hash=metric_report["artifact_hash"],
    )
    _write_json(out_dir / "diagnostic-manifest.json", diagnostic_manifest)
    print(
        "semantic privacy pretraining published: "
        f"seeds={','.join(str(seed) for seed in seeds)} "
        f"steps={args.max_steps} -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
