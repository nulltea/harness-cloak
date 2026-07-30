#!/usr/bin/env python3
"""Thin entry point for ranker-v2 behavior cloning and ExIt collection."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from cloak.ranker.profile_count import ProfileCountTargets
from cloak.ranker.interactive import (
    _semantic_policy_contract,
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
from cloak.ranker.environment import LambdaProfile
from cloak.ranker.privacy import (
    CHECKPOINT_VERSION,
    DirectCountPrivacyProvider,
    PrivacyCheckpointContract,
)
from cloak.ranker.environment import load_ranker_environment
from cloak.ranker.representation import RankerRepresentationStore
from cloak.ranker.semantic import (
    SemanticRankerPolicy,
    calibrate_alpha,
    switch_threshold_calibration,
)
from cloak.reward.utility_cache import UtilityCache, stable_hash


LAMBDA_ZERO = LambdaProfile("lambda-zero", 0.0)
POLICY_ARCHITECTURES = ("semantic-v1",)

_HARD_TRAINING_GATES = {
    "explicit_count_coverage": 1.0,
    "fallback_count_gradient_mass": 0.0,
    "missing_occurrence_decision_mappings": 0,
    "nonmonotone_profiles": 0,
    "lambda_zero_identity": "exact",
}


class _PolicyArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        missing = [
            option
            for option, value in (
                ("--representation-manifest", parsed.representation_manifest),
                ("--profile-count-targets", parsed.profile_count_targets),
            )
            if not value
        ]
        if missing:
            self.error(
                f"{parsed.policy_architecture} requires {', '.join(missing)}"
            )
        return parsed


def _add_artifact_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--policy-architecture",
        required=True,
        choices=POLICY_ARCHITECTURES,
    )
    parser.add_argument("--environment", required=True)
    parser.add_argument("--representation-manifest")
    parser.add_argument("--privacy-checkpoint")
    parser.add_argument("--profile-count-targets")
    parser.add_argument(
        "--allow-development-privacy-checkpoint",
        action="store_true",
        help="allow a non-promoted privacy checkpoint for development-only runs",
    )
    parser.add_argument("--utility-artifact", required=True)
    parser.add_argument("--utility-cache", required=True)


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remote-workers", type=int, default=1)
    parser.add_argument("--reader-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--doc-id",
        dest="doc_ids",
        action="append",
        help="restrict execution to an explicit document ID; repeat for a smoke set",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _PolicyArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)

    bc = subcommands.add_parser("bc")
    _add_artifact_paths(bc)
    _add_runtime_options(bc)
    bc.add_argument("--out-checkpoint", required=True)
    bc.add_argument("--epochs", type=int, default=1)
    bc.add_argument("--learning-rate", type=float, default=1e-4)

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
    train.add_argument(
        "--alpha-utility-routing", choices=("none", "per-decision"), default="none",
    )
    train.add_argument(
        "--controller-gap-scaling", choices=("none", "utility-gap"), default="none",
    )
    train.add_argument(
        "--alpha-init", choices=("checkpoint", "switch-calibrated"),
        default="checkpoint",
    )
    train.add_argument(
        "--rollout-scaling", choices=("fixed", "support"), default="fixed",
    )
    train.add_argument(
        "--counterfactual-coverage", choices=("fixed", "degeneracy"),
        default="fixed",
    )
    train.add_argument(
        "--kl-schedule", choices=("collapse-trigger", "always-on"),
        default="collapse-trigger",
    )
    train.add_argument(
        "--kl-direction", choices=("forward", "reverse"), default="forward",
    )
    train.add_argument("--synchronous-profile-eval", action="store_true")
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
    representation_manifest_hash: str | None = None,
    privacy_checkpoint_hash: str | None = None,
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
        raise ValueError("count supervision environment hash mismatch")
    if utility_artifact.get("environment_hash") != environment_hash:
        raise ValueError("utility artifact environment hash mismatch")
    if count_state.get("artifact_version") != "ranker-v2-profile-count-targets-v1":
        raise ValueError("unsupported profile count target artifact")
    if not representation_manifest_hash:
        raise ValueError("semantic training artifact hashes are incomplete")

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
        "utility_artifact_hash": utility_hash,
        "profile_target_artifact_hash": count_hash,
        "representation_manifest_hash": representation_manifest_hash,
        **(
            {"privacy_checkpoint_hash": privacy_checkpoint_hash}
            if privacy_checkpoint_hash is not None
            else {}
        ),
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
            "profile_target_artifact_hash": str(count_hash),
            "representation_manifest_hash": str(representation_manifest_hash),
            **(
                {"privacy_checkpoint_hash": str(privacy_checkpoint_hash)}
                if privacy_checkpoint_hash is not None
                else {}
            ),
            "lambda_menu_hash": menu_hash,
            "threshold_manifest_hash": threshold_hash,
        },
    }


# Initial-RL scope (user decision 2026-07-23): the policy controls only the three
# clinical menu types; PERSON/CODE stay fixed placeholders; everything else (LOC)
# is demoted to fixed keep. Widen when the scope grows.
RANKER_SCOPE_TYPES = frozenset({"drug", "health-condition", "medical-procedure"})


def _demote_out_of_scope_decisions(
    documents, provider, scope_types: frozenset[str] = RANKER_SCOPE_TYPES,
):
    """Demote out-of-scope or count-uncovered menus to fixed keep decisions.

    Two demotion reasons: runtime type outside the initial-RL scope, and menus
    without direct-count coverage (privacy-head-ineligible; see
    docs/issues/2026-07-23-privacy-ineligible-decisions.md). `provider=None`
    skips the coverage check (learned-checkpoint path).
    """
    demoted = 0
    updated = []
    for document in documents:
        retained = []
        demoted_fixed = []
        for decision in document.policy_decisions:
            in_scope = decision.runtime_type in scope_types
            covered = provider is None or provider.has_targets(
                decision.decision_id,
                [action.action_id for action in decision.actions],
            )
            if in_scope and covered:
                retained.append(decision)
                continue
            keep_actions = tuple(
                action for action in decision.actions if action.mode == "keep"
            )
            if len(keep_actions) != 1:
                raise ValueError(
                    "demoted decision lacks a unique keep action: "
                    f"{decision.decision_id}"
                )
            demoted_fixed.append(
                dataclasses.replace(decision, actions=keep_actions)
            )
            demoted += 1
        if demoted_fixed:
            updated.append(dataclasses.replace(
                document,
                policy_decisions=tuple(retained),
                fixed_decisions=(*document.fixed_decisions, *demoted_fixed),
            ))
        else:
            updated.append(document)
    return tuple(updated), demoted


def _drop_zero_signal_documents(documents, utility_artifact):
    """Drop documents whose utility is constant for every action vector.

    A document with zero policy-linked assertions (no linked_decision /
    linked_decision_set answer targets) has identically-zero utility advantage:
    rollouts there are pure noise (pre-RL audit R1,
    docs/issues/2026-07-27-pre-rl-reward-audit.md).
    """
    linked: set[str] = set()
    for assertion in utility_artifact.get("assertions", {}).values():
        if assertion.get("status", "accepted") != "accepted":
            continue
        target = assertion.get("answer_target") or {}
        if str(target.get("kind", "")).startswith("linked"):
            linked.add(str(assertion.get("doc_id")))
    retained = tuple(d for d in documents if d.doc_id in linked)
    dropped = tuple(d.doc_id for d in documents if d.doc_id not in linked)
    if not retained:
        raise ValueError("every selected document has zero policy-linked assertions")
    return retained, dropped


def _load_inputs(
    args,
) -> tuple[dict, tuple, dict, ProfileCountTargets, dict, UtilityCache, str]:
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
    count_state = _read_json(args.profile_count_targets)
    if count_state.get("environment_hash") != environment_hash:
        raise ValueError("profile count targets environment hash mismatch")
    count_reward = ProfileCountTargets.from_artifact(count_state)
    provider = (
        None if getattr(args, "privacy_checkpoint", None)
        else DirectCountPrivacyProvider(count_state)
    )
    documents, demoted = _demote_out_of_scope_decisions(documents, provider)
    if demoted:
        print(
            f"ranker scope: demoted {demoted} out-of-scope or count-uncovered "
            f"policy decisions to fixed keep (scope: {sorted(RANKER_SCOPE_TYPES)})"
        )
    utility_artifact = _read_json(args.utility_artifact)
    if utility_artifact.get("environment_hash") != environment_hash:
        raise ValueError("utility artifact environment hash mismatch")
    documents, dropped = _drop_zero_signal_documents(documents, utility_artifact)
    if dropped:
        print(
            f"ranker signal: dropped {len(dropped)} documents with zero "
            f"policy-linked assertions (constant utility; "
            "docs/issues/2026-07-27-pre-rl-reward-audit.md R1): "
            + ", ".join(sorted(dropped))
        )
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


def _privacy_checkpoint_contract(path: str | Path) -> PrivacyCheckpointContract:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported semantic privacy checkpoint")
    raw = checkpoint.get("contract")
    if not isinstance(raw, dict):
        raise ValueError("semantic privacy checkpoint lacks its frozen contract")
    values = dict(raw)
    values["count_basis_categories"] = tuple(values.get("count_basis_categories", ()))
    values["seeds"] = tuple(values.get("seeds", ()))
    return PrivacyCheckpointContract(**values)


def _semantic_training_policy(
    args,
    documents: tuple,
    profiles: tuple[LambdaProfile, ...],
) -> SemanticRankerPolicy:
    """Build the semantic trainer from frozen local artifacts without an encoder."""

    store = RankerRepresentationStore.open(Path(args.representation_manifest))
    encoder = store.manifest["encoder"]
    hidden_size = int(encoder["hidden_size"])
    runtime_types = tuple(sorted({
        decision.runtime_type
        for document in documents
        for decision in document.policy_decisions
    }))
    if not runtime_types:
        raise ValueError("semantic training selected no runtime types")
    head_count = 8 if hidden_size % 8 == 0 else 1
    supported_profiles = profiles
    if max(float(profile.value) for profile in profiles) == 0.0:
        supported_profiles = (
            *profiles,
            LambdaProfile("bc-controller-unused", 1.0),
        )
    policy_kwargs = dict(
        representation_store=store,
        supported_profiles=supported_profiles,
        max_lambda=max(float(profile.value) for profile in supported_profiles),
        token_dim=hidden_size,
        pair_dim=5 * hidden_size,
        relation_dim=hidden_size,
        context_dim=hidden_size,
        history_dim=hidden_size,
        utility_hidden_dim=hidden_size,
        num_heads=head_count,
        runtime_types=runtime_types,
        dropout=0.0,
    )
    if args.privacy_checkpoint:
        contract = _privacy_checkpoint_contract(args.privacy_checkpoint)
        if contract.pair_dim != 5 * hidden_size:
            raise ValueError(
                "semantic privacy pair dimension differs from representations"
            )
        policy = SemanticRankerPolicy.from_privacy_checkpoint(
            privacy_checkpoint=Path(args.privacy_checkpoint),
            privacy_checkpoint_contract=contract,
            allow_development_privacy_checkpoint=getattr(
                args, "allow_development_privacy_checkpoint", False
            ),
            **policy_kwargs,
        )
    else:
        policy = SemanticRankerPolicy.from_direct_count_targets(
            profile_count_targets=_read_json(args.profile_count_targets),
            **policy_kwargs,
        )
    device = _resolve_device(args)
    return policy if device == "cpu" else policy.to(device)


def _resolve_device(args) -> str:
    requested = getattr(args, "device", "cpu")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _policy_input_pins(
    args,
    *,
    environment_hash: str,
    count_artifact: dict,
    utility_artifact: dict,
) -> dict[str, str]:
    manifest = _read_json(args.representation_manifest)
    manifest_hash = manifest.get("manifest_hash")
    privacy_checkpoint_hash = None
    if args.privacy_checkpoint:
        contract = _privacy_checkpoint_contract(args.privacy_checkpoint)
        expected = (
            environment_hash,
            count_artifact.get("artifact_hash"),
            manifest_hash,
        )
        actual = (
            contract.environment_hash,
            contract.profile_target_artifact_hash,
            contract.representation_manifest_hash,
        )
        if expected != actual:
            raise ValueError("semantic privacy checkpoint artifact pins differ")
        privacy_checkpoint_hash = _file_hash(args.privacy_checkpoint)
    else:
        provider = DirectCountPrivacyProvider(count_artifact)
        if provider.privacy_signal != {
            "kind": "direct-count",
            "targets_artifact_hash": count_artifact.get("artifact_hash"),
            "environment_hash": environment_hash,
        }:
            raise ValueError("direct-count privacy signal artifact pins differ")
    pins = {
        "environment_hash": environment_hash,
        "profile_target_artifact_hash": count_artifact.get("artifact_hash"),
        "representation_manifest_hash": manifest_hash,
        "utility_artifact_hash": utility_artifact.get("artifact_hash"),
        **(
            {"privacy_checkpoint_hash": privacy_checkpoint_hash}
            if privacy_checkpoint_hash is not None
            else {}
        ),
    }
    if any(not isinstance(value, str) or not value for value in pins.values()):
        raise ValueError("semantic policy input pins are incomplete")
    return pins


def _point_payload(point) -> dict[str, Any]:
    return {
        "doc_id": point.trajectory.doc_id,
        "action_vector": dict(point.action_vector),
        "utility": point.utility,
        "count_score": point.count_score,
        "component_scores": dict(point.component_scores),
        "result_hash": point.result_hash,
    }


def _semantic_bc_policy_config(
    policy: torch.nn.Module,
) -> dict[str, Any]:
    """Freeze every semantic architecture field needed by later CLI stages."""

    store = getattr(policy, "representation_store", None)
    manifest = getattr(store, "manifest", {})
    encoder = manifest.get("encoder", {}) if isinstance(manifest, Mapping) else {}
    encoder_pin = encoder.get("id") if isinstance(encoder, Mapping) else None
    if not isinstance(encoder_pin, str) or not encoder_pin:
        raise ValueError("semantic BC encoder pin is incomplete")
    return {
        "policy_architecture": "semantic-v1",
        "encoder_pin": encoder_pin,
        **_semantic_policy_contract(policy),
    }


def _validate_semantic_bc_policy_config(
    config: dict[str, Any],
    policy: torch.nn.Module,
) -> None:
    """Reject semantic BC architecture drift before loading policy tensors."""

    expected = {
        "policy_architecture": "semantic-v1",
        **_semantic_policy_contract(policy),
    }
    if (
        not isinstance(config, dict)
        or not isinstance(config.get("encoder_pin"), str)
        or not config["encoder_pin"]
        or any(config.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("semantic policy contract differs from BC checkpoint")


def _run_bc(args) -> None:
    (
        _, documents, count_state, count_reward, utility_artifact, cache,
        environment_hash,
    ) = _load_inputs(args)
    torch.manual_seed(args.seed)
    policy = _semantic_training_policy(args, documents, (LAMBDA_ZERO,))
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
    input_pins = _policy_input_pins(
        args,
        environment_hash=environment_hash,
        count_artifact=count_state,
        utility_artifact=utility_artifact,
    )
    checkpoint = {
        "checkpoint_version": "ranker-v2-bc-v1",
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "policy_config": _semantic_bc_policy_config(policy),
        "pins": input_pins,
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
    expected_pins = _policy_input_pins(
        args,
        environment_hash=environment_hash,
        count_artifact=count_state,
        utility_artifact=utility_artifact,
    )
    if checkpoint.get("pins") != expected_pins:
        raise ValueError("behavior-cloning checkpoint artifact pins differ")
    config = checkpoint.get("policy_config", {})
    if config.get("policy_architecture") != args.policy_architecture:
        raise ValueError("behavior-cloning checkpoint policy architecture differs")
    policy = _semantic_training_policy(args, documents, (LAMBDA_ZERO,))
    _validate_semantic_bc_policy_config(config, policy)
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


def _apply_controller_options(policy, args) -> None:
    """Apply the adopted controller-strength options to a trainer policy.

    Gap scaling changes the forward pass, so it also retags controller_transform:
    the semantic contract (and thus every architecture pin) diverges exactly when
    the controller semantics diverge (controller-strength fork, decision log
    2026-07-28).
    """
    if getattr(args, "alpha_utility_routing", "none") == "per-decision":
        policy.alpha_utility_routing = "per-decision"
    if getattr(args, "controller_gap_scaling", "none") == "utility-gap":
        policy.controller_gap_scaling = "utility-gap"
        policy.controller_transform = "log1p-over-log1p-max-utility-gap-v1"


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
        "alpha_utility_routing": getattr(args, "alpha_utility_routing", "none"),
        "controller_gap_scaling": getattr(args, "controller_gap_scaling", "none"),
        "alpha_init": getattr(args, "alpha_init", "checkpoint"),
        "rollout_scaling": getattr(args, "rollout_scaling", "fixed"),
        "counterfactual_coverage": getattr(
            args, "counterfactual_coverage", "fixed",
        ),
        "kl_schedule": getattr(args, "kl_schedule", "collapse-trigger"),
        "kl_direction": getattr(args, "kl_direction", "forward"),
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
    privacy_checkpoint_hash = None
    representation_manifest = _read_json(args.representation_manifest)
    representation_manifest_hash = representation_manifest.get("manifest_hash")
    if not isinstance(representation_manifest_hash, str) or not representation_manifest_hash:
        raise ValueError("representation manifest lacks manifest_hash")
    if args.privacy_checkpoint:
        privacy_checkpoint_hash = _file_hash(args.privacy_checkpoint)
        privacy_contract = _privacy_checkpoint_contract(args.privacy_checkpoint)
        privacy_contract.validate_for_policy()
        if (
            privacy_contract.environment_hash != environment_hash
            or privacy_contract.profile_target_artifact_hash
            != count_state.get("artifact_hash")
            or privacy_contract.representation_manifest_hash
            != representation_manifest_hash
        ):
            raise ValueError("semantic privacy checkpoint artifact pins differ")
    else:
        provider = DirectCountPrivacyProvider(count_state)
        if provider.privacy_signal["environment_hash"] != environment_hash:
            raise ValueError(
                "direct-count privacy signal environment hash differs"
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
        representation_manifest_hash=representation_manifest_hash,
        privacy_checkpoint_hash=privacy_checkpoint_hash,
    )
    profiles = validated["profiles"]
    if bc_checkpoint.get("policy_config", {}).get(
        "policy_architecture"
    ) != "semantic-v1":
        raise ValueError("trainer semantic architecture differs from BC checkpoint")
    code_revision = _code_revision()
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    policy = _semantic_training_policy(args, documents, profiles)
    _validate_semantic_bc_policy_config(
        bc_checkpoint.get("policy_config", {}), policy,
    )
    _apply_controller_options(policy, args)
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
        if getattr(args, "alpha_init", "checkpoint") == "switch-calibrated":
            raw_median, norm_median = switch_threshold_calibration(
                policy, documents, profiles,
            )
            target = (
                norm_median
                if getattr(args, "controller_gap_scaling", "none") == "utility-gap"
                else raw_median
            )
            calibrate_alpha(policy, target)
            print(
                f"alpha switch calibration: raw median {raw_median:.4f} | "
                f"gap-normalized {norm_median:.4f} | applied {target:.4f}",
                flush=True,
            )
            # KL anchors to the calibrated-init policy; anchoring to the
            # uncalibrated warm start would pull the controller back to dead.
            reference_state = {
                key: value.detach().clone()
                for key, value in policy.state_dict().items()
            }
        else:
            reference_state = dict(warm.reference_state_dict)
        _save_reference_checkpoint(
            args.kl_reference_checkpoint,
            state_dict=reference_state,
            artifact_pins=artifact_pins,
            architecture_pin=architecture_pin,
            code_revision=code_revision,
        )
    reference_policy = _semantic_training_policy(args, documents, profiles)
    _apply_controller_options(reference_policy, args)
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
        profile_targets=count_reward,
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
        rollout_scaling=getattr(args, "rollout_scaling", "fixed"),
        counterfactual_coverage=getattr(
            args, "counterfactual_coverage", "fixed",
        ),
        kl_schedule=getattr(args, "kl_schedule", "collapse-trigger"),
        kl_direction=getattr(args, "kl_direction", "forward"),
        synchronous_profile_eval=getattr(
            args, "synchronous_profile_eval", False,
        ),
    )

    zero_profiles = (profiles[0],)
    control = _semantic_training_policy(args, documents, profiles)
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
    control_reference = _semantic_training_policy(args, documents, profiles)
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
        profile_targets=count_reward,
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
        rollout_scaling=getattr(args, "rollout_scaling", "fixed"),
        counterfactual_coverage=getattr(
            args, "counterfactual_coverage", "fixed",
        ),
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
