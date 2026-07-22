#!/usr/bin/env python3
"""Thin entry point for ranker-v2 behavior cloning and ExIt collection."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

from cloak.train.count_reward import CountReward
from cloak.train.interactive_ranker import (
    CacheOnlyMissError,
    behavior_clone,
    collect_exit_winners,
    score_trajectories,
    write_exit_winners,
)
from cloak.train.ranker import ConditionalRankerPolicy, LambdaProfile
from cloak.train.ranker_environment import load_ranker_environment
from cloak.train.utility_cache import UtilityCache


LAMBDA_ZERO = LambdaProfile("lambda-zero", 0.0)


def _add_artifact_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--environment", required=True)
    parser.add_argument("--count-state", required=True)
    parser.add_argument("--utility-artifact", required=True)
    parser.add_argument("--utility-cache", required=True)


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remote-workers", type=int, default=1)
    parser.add_argument("--reader-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument(
        "--doc-id",
        dest="doc_ids",
        action="append",
        help="restrict execution to an explicit document ID; repeat for a smoke set",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)

    bc = subcommands.add_parser("bc")
    _add_artifact_paths(bc)
    _add_runtime_options(bc)
    bc.add_argument("--out-checkpoint", required=True)
    bc.add_argument("--epochs", type=int, default=1)
    bc.add_argument("--learning-rate", type=float, default=1e-4)
    bc.add_argument("--encoder-pin", default="answerdotai/ModernBERT-base")

    collect = subcommands.add_parser("exit-collect")
    _add_artifact_paths(collect)
    _add_runtime_options(collect)
    collect.add_argument("--checkpoint", required=True)
    collect.add_argument("--out", required=True)
    collect.add_argument("--rollouts", type=int, required=True)
    return parser


def _read_json(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"artifact must be a mapping: {path}")
    return value


def _file_hash(path: str | Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_inputs(args) -> tuple[dict, tuple, dict, CountReward, dict, UtilityCache, str]:
    environment_artifact = _read_json(args.environment)
    environment_hash = environment_artifact.get("frozen_environment", {}).get(
        "environment_hash"
    )
    if not isinstance(environment_hash, str) or not environment_hash:
        raise ValueError("ranker environment lacks environment_hash")
    documents_by_id = load_ranker_environment(Path(args.environment))
    selected_ids = getattr(args, "doc_ids", None)
    if selected_ids:
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("duplicate --doc-id")
        unknown = sorted(set(selected_ids) - set(documents_by_id))
        if unknown:
            raise ValueError(f"unknown --doc-id values: {unknown}")
        documents = tuple(documents_by_id[doc_id] for doc_id in selected_ids)
    else:
        documents = tuple(documents_by_id.values())
    count_state = _read_json(args.count_state)
    if count_state.get("environment_hash") != environment_hash:
        raise ValueError("count state environment hash mismatch")
    count_reward = CountReward.from_artifact(count_state)
    utility_artifact = _read_json(args.utility_artifact)
    if utility_artifact.get("environment_hash") != environment_hash:
        raise ValueError("utility artifact environment hash mismatch")
    cache = UtilityCache(args.utility_cache)
    return (
        environment_artifact,
        documents,
        count_state,
        count_reward,
        utility_artifact,
        cache,
        environment_hash,
    )


def _policy(
    count_reward: CountReward,
    environment_hash: str,
    *,
    encoder_pin: str,
) -> ConditionalRankerPolicy:
    return ConditionalRankerPolicy(
        count_reward,
        (LAMBDA_ZERO,),
        max_menu_value=0.0,
        environment_hash=environment_hash,
        encoder_pin=encoder_pin,
    )


def _point_payload(point) -> dict[str, Any]:
    return {
        "doc_id": point.trajectory.doc_id,
        "action_vector": dict(point.action_vector),
        "utility": point.utility,
        "count_score": point.count_score,
        "component_scores": dict(point.component_scores),
        "result_hash": point.result_hash,
    }


def _run_bc(args) -> None:
    (
        _, documents, count_state, count_reward, utility_artifact, cache,
        environment_hash,
    ) = _load_inputs(args)
    torch.manual_seed(args.seed)
    policy = _policy(
        count_reward, environment_hash, encoder_pin=args.encoder_pin,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    report = behavior_clone(
        policy,
        documents,
        lambda_zero=LAMBDA_ZERO,
        optimizer=optimizer,
        epochs=args.epochs,
    )
    points = score_trajectories(
        documents,
        report.trajectories,
        utility_artifact=utility_artifact,
        environment_hash=environment_hash,
        count_reward=count_reward,
        cache=cache,
        remote_workers=args.remote_workers,
        reader_workers=args.reader_workers,
        cache_only=args.cache_only,
    )
    checkpoint = {
        "checkpoint_version": "ranker-v2-bc-v1",
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "policy_config": {"encoder_pin": args.encoder_pin},
        "pins": {
            "environment_hash": environment_hash,
            "count_state_hash": count_state.get("artifact_hash"),
            "utility_artifact_hash": utility_artifact.get("artifact_hash"),
        },
        "bc": {
            "epoch_losses": report.epoch_losses,
            "action_mode_counts": dict(report.action_mode_counts),
            "runtime_type_counts": dict(report.runtime_type_counts),
            "document_points": [_point_payload(point) for point in points],
        },
    }
    destination = Path(args.out_checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    print(
        f"BC PASS documents={len(documents)} decisions="
        f"{sum(len(row.steps) for row in report.trajectories)} "
        f"checkpoint={destination}",
        flush=True,
    )


def _run_exit_collect(args) -> None:
    (
        _, documents, count_state, count_reward, utility_artifact, cache,
        environment_hash,
    ) = _load_inputs(args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_version") != "ranker-v2-bc-v1":
        raise ValueError("unsupported behavior-cloning checkpoint")
    expected_pins = {
        "environment_hash": environment_hash,
        "count_state_hash": count_state.get("artifact_hash"),
        "utility_artifact_hash": utility_artifact.get("artifact_hash"),
    }
    if checkpoint.get("pins") != expected_pins:
        raise ValueError("behavior-cloning checkpoint artifact pins differ")
    encoder_pin = checkpoint.get("policy_config", {}).get("encoder_pin")
    if not isinstance(encoder_pin, str) or not encoder_pin:
        raise ValueError("behavior-cloning checkpoint lacks encoder pin")
    policy = _policy(count_reward, environment_hash, encoder_pin=encoder_pin)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    generator = torch.Generator().manual_seed(args.seed)
    collection = collect_exit_winners(
        policy,
        documents,
        lambda_zero=LAMBDA_ZERO,
        rollouts_per_document=args.rollouts,
        utility_artifact=utility_artifact,
        environment_hash=environment_hash,
        count_reward=count_reward,
        cache=cache,
        remote_workers=args.remote_workers,
        reader_workers=args.reader_workers,
        generator=generator,
        cache_only=args.cache_only,
    )
    payload = write_exit_winners(
        args.out,
        collection,
        pins={
            **expected_pins,
            "policy_checkpoint_hash": _file_hash(args.checkpoint),
        },
    )
    print(
        f"ExIt PASS documents={payload['summary']['document_count']} "
        f"candidates={payload['summary']['candidate_count']} "
        f"winners={payload['summary']['winner_count']} out={args.out}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "bc":
            _run_bc(args)
        else:
            _run_exit_collect(args)
    except CacheOnlyMissError as error:
        print(
            f"CACHE_ONLY_MISS phase={error.phase} "
            f"remote_tasks={error.remote_tasks} "
            f"context_reader_work_items={error.context_reader_work_items}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
