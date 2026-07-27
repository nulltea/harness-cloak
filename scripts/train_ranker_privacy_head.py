#!/usr/bin/env python3
"""Pretrain the Ranker-v2 semantic log-count head from frozen representations."""
from __future__ import annotations

import argparse
import hashlib
import json

import torch
from pathlib import Path
from typing import Any

from cloak.ranker.diagnostics import build_privacy_diagnostic_manifest
from cloak.ranker.environment import load_ranker_environment
from cloak.ranker.privacy import (
    METRIC_REPORT_VERSION,
    PrivacyCheckpointContract,
    build_grouped_profile_split,
    load_privacy_examples,
    save_privacy_checkpoint,
    train_privacy_seed,
)
from cloak.ranker.representation import RankerRepresentationStore


DEFAULT_OUT_DIR = Path("results/ranker_v2/architecture/privacy")


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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
    parser.add_argument(
        "--counterexample-set",
        help="optional evaluated lexical/semantic counterexample artifact",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=1729)
    parser.add_argument("--escalated-mean-head", action="store_true")
    parser.add_argument("--rank-difference-ablation", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--profile-batch-size", type=int, default=32)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--minimum-improvement", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--use-count-basis", action="store_true")
    parser.add_argument("--device", default="auto",
                        help="training device: auto|cpu|cuda (eval stays on CPU)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = tuple(args.seeds)
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("privacy pretraining requires distinct explicit seeds")
    if len(seeds) == 2:
        raise ValueError("use one iteration seed or at least three promotion seeds")
    if not 0 < args.max_steps <= 500:
        raise ValueError("--max-steps must be in [1, 500]")
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device in {"cpu", "cuda"}:
        device = args.device
    else:
        raise ValueError("--device must be auto, cpu, or cuda")

    environment_path = Path(args.environment)
    target_path = Path(args.profile_targets)
    representation_path = Path(args.representation_manifest)
    out_dir = Path(args.out_dir)
    targets = _read_mapping(target_path)
    counterexample_report = None
    if args.counterexample_set:
        counterexample_artifact = _read_mapping(Path(args.counterexample_set))
        counterexample_hash = counterexample_artifact.get("artifact_hash")
        expected_counterexample_hash = _stable_hash({
            key: value for key, value in counterexample_artifact.items()
            if key != "artifact_hash"
        })
        counterexample_verdict = counterexample_artifact.get(
            "privacy_head_gate_verdict"
        )
        if counterexample_hash != expected_counterexample_hash:
            raise ValueError("counterexample-set artifact hash mismatch")
        if counterexample_verdict not in {"PASS", "FAIL"}:
            raise ValueError("counterexample-set gate verdict is invalid")
        counterexample_report = {
            "artifact_hash": counterexample_hash,
            "verdict": counterexample_verdict,
        }
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
    projection_dim = 32 if args.escalated_mean_head else 1
    rho = 0.05 if args.rank_difference_ablation else 0.0
    for seed in seeds:
        model, report = train_privacy_seed(
            examples,
            split_manifest,
            seed=seed,
            projection_dim=projection_dim,
            hidden_dim=0,
            rho=rho,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            use_count_basis=args.use_count_basis,
            device=device,
            weight_decay=args.weight_decay,
            gradient_clip=args.gradient_clip,
            profile_batch_size=args.profile_batch_size,
            evaluation_interval=args.evaluation_interval,
            patience=args.patience,
            minimum_improvement=args.minimum_improvement,
        )
        models[seed] = model
        seed_reports.append(report)
        _write_json(out_dir / f"seed-{seed}" / "metrics.json", report)

    run_protocol = "iteration" if len(seeds) == 1 else "promotion"
    metric_report = {
        "artifact_version": METRIC_REPORT_VERSION,
        "profile_held_out": True,
        "run_protocol": run_protocol,
        "seed_count": len(seeds),
        "split_manifest_hash": split_manifest["artifact_hash"],
        "seeds": list(seeds),
        "seed_reports": seed_reports,
    }
    metric_report["artifact_hash"] = _stable_hash(metric_report)
    _write_json(out_dir / "metrics.json", metric_report)

    diagnostic_manifest = build_privacy_diagnostic_manifest(
        seed_reports,
        split_manifest_hash=split_manifest["artifact_hash"],
        metric_report_hash=metric_report["artifact_hash"],
        counterexample_report=counterexample_report,
    )
    _write_json(out_dir / "diagnostic-manifest.json", diagnostic_manifest)

    representation_hash = store.manifest.get("manifest_hash")
    if not isinstance(representation_hash, str) or not representation_hash:
        raise ValueError("representation manifest lacks manifest hash")
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
            rho=rho,
            gamma=float(args.gamma),
            seeds=seeds,
            training_seed=seed,
            metric_report_hash=metric_report["artifact_hash"],
            diagnostic_manifest_hash=diagnostic_manifest["artifact_hash"],
            counterexample_set_hash=(
                counterexample_report["artifact_hash"]
                if counterexample_report is not None else None
            ),
            run_protocol=run_protocol,
            seed_count=len(seeds),
            promotion_verdict=diagnostic_manifest[
                "relative_promotion"
            ]["verdict"],
            target_mean=float(model.target_mean),
            target_std=float(model.target_std),
            sigma_fixed=float(model.sigma_fixed),
            feature_schema="type-source-candidate-hadamard-v1",
            optimizer="AdamW",
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            gradient_clip=float(args.gradient_clip),
        )
        save_privacy_checkpoint(
            out_dir / f"seed-{seed}" / "checkpoint.pt", model, contract
        )

    print(
        "semantic privacy pretraining published: "
        f"seeds={','.join(str(seed) for seed in seeds)} "
        f"steps={args.max_steps} -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
