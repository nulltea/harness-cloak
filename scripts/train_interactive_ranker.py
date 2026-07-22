#!/usr/bin/env python3
"""Thin entry point for ranker-v2 behavior cloning and ExIt collection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from cloak.train.count_reward import CountReward
from cloak.train.interactive_ranker import (
    CacheOnlyMissError,
    behavior_clone,
    build_latin_cycle_schedule,
    collect_exit_winners,
    initialize_hybrid_warm_start,
    load_hybrid_checkpoint,
    policy_architecture_pin,
    save_hybrid_checkpoint,
    score_trajectories,
    train_hybrid_policy,
    write_exit_winners,
)
from cloak.train.ranker import ConditionalRankerPolicy, LambdaProfile
from cloak.train.ranker_environment import load_ranker_environment
from cloak.train.utility_cache import UtilityCache, stable_hash


LAMBDA_ZERO = LambdaProfile("lambda-zero", 0.0)

_HARD_TRAINING_GATES = {
    "explicit_count_coverage": 1.0,
    "fallback_count_gradient_mass": 0.0,
    "missing_occurrence_decision_mappings": 0,
    "nonmonotone_profiles": 0,
    "lambda_zero_identity": "exact",
}


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

    train = subcommands.add_parser("train")
    _add_artifact_paths(train)
    _add_runtime_options(train)
    train.add_argument("--threshold-manifest", required=True)
    train.add_argument("--lambda-menu", required=True)
    train.add_argument("--exit-winners", required=True)
    train.add_argument("--bc-checkpoint", required=True)
    train.add_argument("--out-checkpoint", required=True)
    train.add_argument("--kl-reference-checkpoint", required=True)
    train.add_argument("--epoch-reports", required=True)
    train.add_argument("--fixed-lambda-zero-control", required=True)
    train.add_argument("--resume")
    train.add_argument("--max-docs", type=int)
    train.add_argument("--max-epochs", type=int, required=True)
    train.add_argument("--rollouts", type=int, required=True)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--beta", type=float, default=0.01)
    train.add_argument("--eta", type=float, default=0.01)
    train.add_argument("--encoder-pin", default="answerdotai/ModernBERT-base")
    return parser


def _read_json(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"artifact must be a mapping: {path}")
    return value


def _file_hash(path: str | Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_artifact_hash(artifact: dict, label: str) -> str:
    value = artifact.get("artifact_hash")
    if not isinstance(value, str) or value != stable_hash({
        key: row for key, row in artifact.items() if key != "artifact_hash"
    }):
        raise ValueError(f"{label} hash mismatch")
    return value


def _validate_train_artifacts(
    environment: dict,
    count_state: dict,
    utility_artifact: dict,
    lambda_menu: dict,
    threshold_manifest: dict,
    exit_winners: dict,
    bc_checkpoint: dict,
    *,
    bc_checkpoint_hash: str,
) -> dict[str, Any]:
    """Validate the complete frozen trainer binding without mutating runtime state."""

    environment_hash = environment.get("frozen_environment", {}).get(
        "environment_hash"
    )
    if not isinstance(environment_hash, str) or not environment_hash:
        raise ValueError("training environment lacks environment_hash")
    count_hash = count_state.get("artifact_hash")
    utility_hash = utility_artifact.get("artifact_hash")
    if count_state.get("environment_hash") != environment_hash:
        raise ValueError("count state environment hash mismatch")
    if utility_artifact.get("environment_hash") != environment_hash:
        raise ValueError("utility artifact environment hash mismatch")
    count_gate = count_state.get("gate_report", {})
    if count_gate.get("verdict") != "PASS":
        raise ValueError("count gate did not pass")
    if count_gate.get("missing_policy_mappings"):
        raise ValueError("count mapping gate did not pass")
    if count_gate.get("nonmonotone_profiles"):
        raise ValueError("count monotonicity gate did not pass")

    if threshold_manifest.get("environment_hash") != environment_hash:
        raise ValueError("threshold environment hash mismatch")
    if threshold_manifest.get("utility_component_artifact_hash") != utility_hash:
        raise ValueError("threshold utility artifact hash mismatch")
    if threshold_manifest.get("count_artifact_hash") != count_hash:
        raise ValueError("threshold count artifact hash mismatch")
    if threshold_manifest.get("hard_gates") != _HARD_TRAINING_GATES:
        raise ValueError("threshold hard gates differ from normative values")
    _require_artifact_hash(count_state, "count state")
    _require_artifact_hash(utility_artifact, "utility artifact")

    if lambda_menu.get("artifact_version") != "ranker-v2-lambda-menu-v1":
        raise ValueError("unsupported lambda menu")
    if lambda_menu.get("verdict") != "PASS":
        raise ValueError("lambda menu gate did not pass")
    values = lambda_menu.get("values")
    names = lambda_menu.get("profile_names")
    if (
        not isinstance(values, list)
        or not isinstance(names, list)
        or len(values) != len(names)
        or not 3 <= len(values) <= 5
        or values[0] != 0.0
        or any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in values
        )
        or [float(value) for value in values] != sorted(set(float(value) for value in values))
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("lambda menu schema or ordering is invalid")

    if threshold_manifest.get("artifact_version") != "ranker-v2-threshold-manifest-v1":
        raise ValueError("unsupported threshold manifest")
    scheduler = threshold_manifest.get("scheduler", {})
    budget = scheduler.get("call_budget")
    endpoint_fraction = scheduler.get("endpoint_fraction")
    if (
        not isinstance(budget, int)
        or isinstance(budget, bool)
        or budget <= 0
        or budget % 5
        or isinstance(endpoint_fraction, bool)
        or not isinstance(endpoint_fraction, int | float)
        or not 0.0 <= float(endpoint_fraction) < 0.5
    ):
        raise ValueError("threshold scheduler configuration is invalid")
    endpoint_budget_float = budget * float(endpoint_fraction)
    if not endpoint_budget_float.is_integer():
        raise ValueError("threshold endpoint budget is not integral")

    if exit_winners.get("artifact_version") != "ranker-v2-exit-winners-v1":
        raise ValueError("unsupported ExIt winner artifact")
    expected_common = {
        "environment_hash": environment_hash,
        "count_state_hash": count_hash,
        "utility_artifact_hash": utility_hash,
    }
    exit_pins = exit_winners.get("pins", {})
    if any(exit_pins.get(key) != value for key, value in expected_common.items()):
        raise ValueError("ExIt winner artifact pins differ")
    if exit_pins.get("policy_checkpoint_hash") != bc_checkpoint_hash:
        raise ValueError("ExIt winner BC checkpoint hash differs")
    if bc_checkpoint.get("checkpoint_version") != "ranker-v2-bc-v1":
        raise ValueError("unsupported behavior-cloning checkpoint")
    if bc_checkpoint.get("pins") != expected_common:
        raise ValueError("behavior-cloning checkpoint artifact pins differ")

    menu_hash = _require_artifact_hash(lambda_menu, "lambda menu")
    threshold_hash = _require_artifact_hash(threshold_manifest, "threshold manifest")
    exit_hash = _require_artifact_hash(exit_winners, "ExIt winner")
    profiles = tuple(
        LambdaProfile(name, float(value))
        for name, value in zip(names, values, strict=True)
    )
    return {
        "profiles": profiles,
        "counterfactual_budget": budget,
        "endpoint_budget": int(endpoint_budget_float),
        "artifact_pins": {
            "environment_hash": environment_hash,
            "utility_artifact_hash": str(utility_hash),
            "count_state_hash": str(count_hash),
            "lambda_menu_hash": menu_hash,
            "threshold_manifest_hash": threshold_hash,
            "exit_winners_hash": exit_hash,
            "bc_checkpoint_hash": bc_checkpoint_hash,
        },
    }


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
    profiles: tuple[LambdaProfile, ...] = (LAMBDA_ZERO,),
    encoder=None,
) -> ConditionalRankerPolicy:
    return ConditionalRankerPolicy(
        count_reward,
        profiles,
        max_menu_value=max(profile.value for profile in profiles),
        environment_hash=environment_hash,
        encoder_pin=encoder_pin,
        encoder=encoder,
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


def _code_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if not revision:
        raise ValueError("git code revision is unavailable")
    return revision


def _save_reference_checkpoint(
    path: str | Path,
    *,
    state_dict: dict[str, torch.Tensor] | Any,
    artifact_pins: dict[str, str],
    architecture_pin: str,
    code_revision: str,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save({
        "checkpoint_version": "ranker-v2-kl-reference-v1",
        "policy_state_dict": dict(state_dict),
        "artifact_pins": dict(sorted(artifact_pins.items())),
        "architecture_pin": architecture_pin,
        "code_revision": code_revision,
    }, temporary)
    os.replace(temporary, destination)


def _load_reference_checkpoint(
    path: str | Path,
    *,
    artifact_pins: dict[str, str],
    architecture_pin: str,
    code_revision: str,
) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("checkpoint_version") != "ranker-v2-kl-reference-v1":
        raise ValueError("unsupported KL-reference checkpoint")
    if payload.get("artifact_pins") != dict(sorted(artifact_pins.items())):
        raise ValueError("KL-reference artifact pins differ")
    if payload.get("architecture_pin") != architecture_pin:
        raise ValueError("KL-reference architecture pin differs")
    if payload.get("code_revision") != code_revision:
        raise ValueError("KL-reference code revision differs")
    state = payload.get("policy_state_dict")
    if not isinstance(state, dict):
        raise ValueError("KL-reference policy state is invalid")
    return state


def _write_epoch_reports(
    path: str | Path,
    conditional_reports,
    control_reports,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    lines = []
    for run_name, reports in (
        ("conditional", conditional_reports),
        ("fixed-lambda-zero-control", control_reports),
    ):
        for report in reports:
            lines.append(json.dumps(
                {"run": run_name, **dict(report)},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ))
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _training_config(args, documents, *, fixed_control: bool) -> dict[str, Any]:
    return {
        "learning_rate": float(args.learning_rate),
        "beta": float(args.beta),
        "eta": float(args.eta),
        "max_grad_norm": None,
        "max_epochs": int(args.max_epochs),
        "rollouts": int(args.rollouts),
        "remote_workers": int(args.remote_workers),
        "reader_workers": int(args.reader_workers),
        "seed": int(args.seed),
        "fixed_lambda_zero_control": bool(fixed_control),
        "document_ids_hash": stable_hash(sorted(document.doc_id for document in documents)),
    }


def _run_train(args) -> None:
    (
        environment,
        loaded_documents,
        count_state,
        count_reward,
        utility_artifact,
        cache,
        environment_hash,
    ) = _load_inputs(args)
    if (
        args.max_epochs <= 0
        or args.rollouts < 2
        or args.remote_workers <= 0
        or args.reader_workers <= 0
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0.0
        or not math.isfinite(args.beta)
        or args.beta < 0.0
        or not math.isfinite(args.eta)
        or args.eta < 0.0
        or args.max_docs is not None and args.max_docs <= 0
    ):
        raise ValueError("invalid hybrid training runtime configuration")
    documents = tuple(sorted(loaded_documents, key=lambda row: row.doc_id))
    if args.max_docs is not None:
        documents = documents[:args.max_docs]
    if not documents:
        raise ValueError("hybrid training selected no documents")
    lambda_menu = _read_json(args.lambda_menu)
    threshold_manifest = _read_json(args.threshold_manifest)
    exit_winners = _read_json(args.exit_winners)
    bc_checkpoint = torch.load(
        args.bc_checkpoint, map_location="cpu", weights_only=True,
    )
    validated = _validate_train_artifacts(
        environment,
        count_state,
        utility_artifact,
        lambda_menu,
        threshold_manifest,
        exit_winners,
        bc_checkpoint,
        bc_checkpoint_hash=_file_hash(args.bc_checkpoint),
    )
    profiles = validated["profiles"]
    checkpoint_encoder = bc_checkpoint.get("policy_config", {}).get("encoder_pin")
    if checkpoint_encoder != args.encoder_pin:
        raise ValueError("trainer encoder pin differs from BC checkpoint")
    code_revision = _code_revision()
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    policy = _policy(
        count_reward,
        environment_hash,
        encoder_pin=args.encoder_pin,
        profiles=profiles,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    schedule = build_latin_cycle_schedule(documents, profiles, seed=args.seed)
    architecture_pin = policy_architecture_pin(policy)
    artifact_pins = validated["artifact_pins"]
    cache_paths = {
        "utility": str(Path(args.utility_cache).resolve()),
        "kl_reference": str(Path(args.kl_reference_checkpoint).resolve()),
    }
    training_config = _training_config(args, documents, fixed_control=False)
    pair_history = {}
    existing_reports = ()
    start_epoch = 0
    kl_enabled = False

    if args.resume:
        reference_state = _load_reference_checkpoint(
            args.kl_reference_checkpoint,
            artifact_pins=artifact_pins,
            architecture_pin=architecture_pin,
            code_revision=code_revision,
        )
        resumed = load_hybrid_checkpoint(
            args.resume,
            policy=policy,
            optimizer=optimizer,
            generator=generator,
            expected_artifact_pins=artifact_pins,
            expected_architecture_pin=architecture_pin,
            expected_cache_paths=cache_paths,
            expected_code_revision=code_revision,
            expected_training_config=training_config,
        )
        if resumed["schedule"] != schedule:
            raise ValueError("resume schedule differs from frozen Latin cycle")
        pair_history = resumed["pair_history"]
        existing_reports = resumed["epoch_reports"]
        start_epoch = resumed["epoch"] + 1
        kl_enabled = resumed["kl_enabled"]
    else:
        warm = initialize_hybrid_warm_start(
            policy,
            documents,
            profiles,
            bc_state_dict=bc_checkpoint["policy_state_dict"],
            exit_winners=exit_winners,
            optimizer=optimizer,
        )
        reference_state = dict(warm.reference_state_dict)
        _save_reference_checkpoint(
            args.kl_reference_checkpoint,
            state_dict=reference_state,
            artifact_pins=artifact_pins,
            architecture_pin=architecture_pin,
            code_revision=code_revision,
        )
    reference_policy = _policy(
        count_reward,
        environment_hash,
        encoder_pin=args.encoder_pin,
        profiles=profiles,
        encoder=policy.encoder,
    )
    reference_policy.load_state_dict(reference_state, strict=True)
    reference_policy.eval()
    for parameter in reference_policy.parameters():
        parameter.requires_grad_(False)

    def save_epoch(epoch, reports, history, enabled):
        save_hybrid_checkpoint(
            args.out_checkpoint,
            policy=policy,
            optimizer=optimizer,
            epoch=epoch,
            generator=generator,
            schedule=schedule,
            artifact_pins=artifact_pins,
            architecture_pin=architecture_pin,
            cache_paths=cache_paths,
            code_revision=code_revision,
            training_config=training_config,
            pair_history=history,
            kl_enabled=enabled,
            epoch_reports=reports,
        )

    conditional = train_hybrid_policy(
        policy=policy,
        reference_policy=reference_policy,
        documents=documents,
        profiles=profiles,
        schedule=schedule,
        optimizer=optimizer,
        utility_artifact=utility_artifact,
        environment_hash=environment_hash,
        count_reward=count_reward,
        cache=cache,
        threshold_manifest=threshold_manifest,
        max_epochs=args.max_epochs,
        rollouts=args.rollouts,
        beta=args.beta,
        eta=args.eta,
        counterfactual_budget=validated["counterfactual_budget"],
        endpoint_budget=validated["endpoint_budget"],
        pair_history=pair_history,
        seed=args.seed,
        remote_workers=args.remote_workers,
        reader_workers=args.reader_workers,
        generator=generator,
        start_epoch=start_epoch,
        existing_reports=existing_reports,
        kl_enabled=kl_enabled,
        cache_only=args.cache_only,
        epoch_callback=save_epoch,
    )

    zero_profiles = (profiles[0],)
    control = _policy(
        count_reward,
        environment_hash,
        encoder_pin=args.encoder_pin,
        profiles=zero_profiles,
        encoder=policy.encoder,
    )
    control_optimizer = torch.optim.Adam(
        control.parameters(), lr=args.learning_rate,
    )
    control_warm = initialize_hybrid_warm_start(
        control,
        documents,
        zero_profiles,
        bc_state_dict=bc_checkpoint["policy_state_dict"],
        exit_winners=exit_winners,
        optimizer=control_optimizer,
    )
    control_reference = _policy(
        count_reward,
        environment_hash,
        encoder_pin=args.encoder_pin,
        profiles=zero_profiles,
        encoder=policy.encoder,
    )
    control_reference.load_state_dict(control_warm.reference_state_dict, strict=True)
    control_reference.eval()
    for parameter in control_reference.parameters():
        parameter.requires_grad_(False)
    control_schedule = build_latin_cycle_schedule(
        documents, zero_profiles, seed=args.seed,
    )
    control_architecture_pin = policy_architecture_pin(control)
    control_config = _training_config(args, documents, fixed_control=True)
    control_generator = torch.Generator().manual_seed(args.seed)

    def save_control_epoch(epoch, reports, history, enabled):
        save_hybrid_checkpoint(
            args.fixed_lambda_zero_control,
            policy=control,
            optimizer=control_optimizer,
            epoch=epoch,
            generator=control_generator,
            schedule=control_schedule,
            artifact_pins=artifact_pins,
            architecture_pin=control_architecture_pin,
            cache_paths={"utility": str(Path(args.utility_cache).resolve())},
            code_revision=code_revision,
            training_config=control_config,
            pair_history=history,
            kl_enabled=enabled,
            epoch_reports=reports,
        )

    fixed_control = train_hybrid_policy(
        policy=control,
        reference_policy=control_reference,
        documents=documents,
        profiles=zero_profiles,
        schedule=control_schedule,
        optimizer=control_optimizer,
        utility_artifact=utility_artifact,
        environment_hash=environment_hash,
        count_reward=count_reward,
        cache=cache,
        threshold_manifest=threshold_manifest,
        max_epochs=args.max_epochs,
        rollouts=args.rollouts,
        beta=args.beta,
        eta=0.0,
        counterfactual_budget=validated["counterfactual_budget"],
        endpoint_budget=validated["endpoint_budget"],
        pair_history={},
        seed=args.seed,
        remote_workers=args.remote_workers,
        reader_workers=args.reader_workers,
        generator=control_generator,
        cache_only=args.cache_only,
        epoch_callback=save_control_epoch,
    )
    _write_epoch_reports(
        args.epoch_reports,
        conditional.epoch_reports,
        fixed_control.epoch_reports,
    )
    print(
        f"TRAIN PASS documents={len(documents)} profiles={len(profiles)} "
        f"epochs={args.max_epochs} conditional={args.out_checkpoint} "
        f"control={args.fixed_lambda_zero_control}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "bc":
            _run_bc(args)
        elif args.command == "exit-collect":
            _run_exit_collect(args)
        else:
            _run_train(args)
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
