#!/usr/bin/env python3
"""Freeze ranker-v2 calibration artifacts or stop on exact cache misses."""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cloak.train.count_reward import CountReward
from cloak.train.lambda_menu import (
    CalibrationTrajectory,
    build_anchor_trajectories,
    calibration_point_from_result,
    freeze_calibration_pool,
    select_lambda_menu,
)
from cloak.train.ranker_diagnostics import (
    build_diagnostic_spike,
    cache_only_missing_report,
    freeze_threshold_manifest,
    reader_jitter_from_cache,
    validate_threshold_rules,
)
from cloak.train.ranker_environment import RankerDocument, load_ranker_environment
from cloak.train.roundtrip import (
    UTILITY_EXECUTION_CONTRACT_VERSION,
    _cache_identity,
    score_roundtrip_batch,
)
from cloak.train.utility_cache import UtilityCache, UtilityRequest, UtilityResult, stable_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--utility-artifact", required=True)
    parser.add_argument("--count-state", required=True)
    parser.add_argument("--utility-cache", required=True)
    parser.add_argument("--exit-winners", required=True)
    parser.add_argument("--threshold-rules", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--remote-workers", type=int, default=1)
    parser.add_argument("--reader-workers", type=int, default=1)
    return parser


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"artifact must be a mapping: {path}")
    return value


def _write_json(path: str | Path, payload: Mapping) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _environment_hash(environment: Mapping) -> str:
    value = environment.get("frozen_environment", {}).get("environment_hash")
    if not isinstance(value, str) or not value:
        raise ValueError("ranker environment lacks environment_hash")
    return value


def _split_by_doc(doc_ids: Sequence[str]) -> dict[str, str]:
    import hashlib

    return {
        doc_id: (
            "development"
            if int(hashlib.sha256(
                ("ranker-v2-calibration-split-v1\0" + doc_id).encode()
            ).hexdigest(), 16) % 5 == 0
            else "train"
        )
        for doc_id in sorted(doc_ids)
    }


def _trajectory_from_vector(
    document: RankerDocument,
    action_vector: Mapping[str, str],
    source: str,
) -> CalibrationTrajectory:
    if set(action_vector) != {
        decision.decision_id for decision in document.policy_decisions
    }:
        raise ValueError(f"calibration vector decision set differs for {document.doc_id}")
    ordered = []
    modes = []
    runtime_types = []
    for decision in document.policy_decisions:
        action_id = str(action_vector[decision.decision_id])
        matches = [action for action in decision.actions if action.action_id == action_id]
        if len(matches) != 1:
            raise ValueError(f"calibration vector has unknown action: {action_id}")
        ordered.append((decision.decision_id, action_id))
        modes.append(matches[0].mode)
        runtime_types.append(decision.runtime_type)
    return CalibrationTrajectory(
        doc_id=document.doc_id,
        corpus=document.corpus,
        sources=(source,),
        ordered_action_vector=tuple(ordered),
        action_modes=tuple(modes),
        runtime_types=tuple(runtime_types),
    )


def _exit_trajectories(
    path: str | Path,
    documents: Mapping[str, RankerDocument],
    *,
    expected_pins: Mapping[str, str],
) -> tuple[tuple[CalibrationTrajectory, ...], str]:
    source = Path(path)
    if not source.exists():
        return (), "absent"
    artifact = _read_json(source)
    if artifact.get("artifact_version") != "ranker-v2-exit-winners-v1":
        raise ValueError("unsupported ExIt winner artifact")
    stored_hash = artifact.get("artifact_hash")
    expected_hash = stable_hash({
        key: value for key, value in artifact.items() if key != "artifact_hash"
    })
    if stored_hash != expected_hash:
        raise ValueError("ExIt winner artifact hash mismatch")
    pins = artifact.get("pins")
    for name in ("environment_hash", "count_state_hash", "utility_artifact_hash"):
        if not isinstance(pins, Mapping) or pins.get(name) != expected_pins[name]:
            raise ValueError(f"ExIt winner {name} differs")
    trajectories = []
    for row in artifact.get("documents", ()):
        doc_id = str(row.get("doc_id"))
        if doc_id not in documents:
            raise ValueError(f"ExIt winner document is unavailable: {doc_id}")
        entries = [("exit_reference", row.get("reference"))]
        entries.extend(("exit_sample", item) for item in row.get("candidates", ()))
        if row.get("winner") is not None:
            entries.append(("verified_exit_winner", row["winner"]))
        for label, point in entries:
            if not isinstance(point, Mapping) or not isinstance(
                point.get("action_vector"), Mapping
            ):
                raise ValueError(f"ExIt calibration point is invalid for {doc_id}")
            trajectories.append(_trajectory_from_vector(
                documents[doc_id], point["action_vector"], label,
            ))
    return tuple(trajectories), "loaded"


def _cache_trajectories(
    cache: UtilityCache,
    documents: Mapping[str, RankerDocument],
    *,
    environment_hash: str,
    utility_artifact_hash: str,
) -> tuple[CalibrationTrajectory, ...]:
    trajectories = []
    for identity, result in cache.entries.values():
        if (
            identity.get("environment_hash") != environment_hash
            or identity.get("utility_artifact_hash") != utility_artifact_hash
            or identity.get("reader_refresh") is not False
            or result.doc_id not in documents
        ):
            continue
        trajectories.append(_trajectory_from_vector(
            documents[result.doc_id], result.action_vector, "cached_complete_trajectory",
        ))
    return tuple(trajectories)


def _deduplicate_trajectories(
    trajectories: Sequence[CalibrationTrajectory],
) -> tuple[CalibrationTrajectory, ...]:
    unique: dict[tuple[str, tuple[tuple[str, str], ...]], CalibrationTrajectory] = {}
    for trajectory in trajectories:
        key = (trajectory.doc_id, trajectory.ordered_action_vector)
        previous = unique.get(key)
        if previous is None:
            unique[key] = trajectory
        else:
            unique[key] = CalibrationTrajectory(
                doc_id=trajectory.doc_id,
                corpus=trajectory.corpus,
                sources=tuple(sorted(set(previous.sources) | set(trajectory.sources))),
                ordered_action_vector=trajectory.ordered_action_vector,
                action_modes=trajectory.action_modes,
                runtime_types=trajectory.runtime_types,
            )
    return tuple(unique[key] for key in sorted(unique))


def _cached_result(
    trajectory: CalibrationTrajectory,
    *,
    document: RankerDocument,
    utility_artifact: Mapping,
    environment_hash: str,
    cache: UtilityCache,
) -> UtilityResult:
    request = UtilityRequest(
        document=document,
        action_vector=trajectory.action_vector,
        utility_artifact=utility_artifact,
        environment_hash=environment_hash,
    )
    from cloak.train.interactive_ranker import assemble_action_vector

    doc_p, _ = assemble_action_vector(document, trajectory.action_vector)
    identity = _cache_identity(request, doc_p, reader_refresh=False)
    result = cache.lookup(identity)
    if result is None:
        raise ValueError("preflight cache changed after admission")
    return result


def _candidate_menu(points) -> dict[str, Any]:
    return select_lambda_menu(
        points,
        menu_size=4,
        min_adjacent_winner_change=0.0,
        max_placeholder_fraction=1.0,
        min_supported_documents_by_corpus=1,
        min_supported_decisions_by_type=1,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    environment = _read_json(args.environment)
    environment_hash = _environment_hash(environment)
    documents = load_ranker_environment(Path(args.environment))
    utility_artifact = _read_json(args.utility_artifact)
    count_state = _read_json(args.count_state)
    if utility_artifact.get("environment_hash") != environment_hash:
        raise ValueError("utility artifact environment hash differs")
    if count_state.get("environment_hash") != environment_hash:
        raise ValueError("count state environment hash differs")
    count_reward = CountReward.from_artifact(count_state)
    rules = validate_threshold_rules(_read_json(args.threshold_rules))
    cache = UtilityCache(args.utility_cache)
    expected_exit_pins = {
        "environment_hash": environment_hash,
        "count_state_hash": str(count_state.get("artifact_hash")),
        "utility_artifact_hash": str(utility_artifact.get("artifact_hash")),
    }
    exit_trajectories, exit_status = _exit_trajectories(
        args.exit_winners, documents, expected_pins=expected_exit_pins,
    )
    anchors = tuple(
        trajectory
        for document in documents.values()
        for trajectory in build_anchor_trajectories(document, count_reward)
    )
    cached = _cache_trajectories(
        cache,
        documents,
        environment_hash=environment_hash,
        utility_artifact_hash=str(utility_artifact.get("artifact_hash")),
    )
    trajectories = _deduplicate_trajectories((*anchors, *exit_trajectories, *cached))
    missing = cache_only_missing_report(
        trajectories,
        documents=documents,
        utility_artifact=utility_artifact,
        environment_hash=environment_hash,
        cache=cache,
    )
    missing["exit_winners_status"] = exit_status
    missing["threshold_rules_hash"] = stable_hash(rules)
    out_dir = Path(args.out_dir)
    if missing["missing_action_vector_count"]:
        _write_json(out_dir / "cache-misses.json", missing)
        if args.cache_only:
            print(
                "PREFLIGHT CACHE_ONLY_STOP "
                f"missing_action_vectors={missing['missing_action_vector_count']} "
                f"remote_tasks={missing['remote_tasks']} "
                f"context_reader_work_items={missing['context_reader_work_items']} "
                f"details={out_dir / 'cache-misses.json'} dispatched=false",
                flush=True,
            )
            return 2
        requests = [
            UtilityRequest(
                document=documents[trajectory.doc_id],
                action_vector=trajectory.action_vector,
                utility_artifact=utility_artifact,
                environment_hash=environment_hash,
            )
            for trajectory in trajectories
        ]
        score_roundtrip_batch(
            requests,
            cache=cache,
            remote_workers=args.remote_workers,
            reader_workers=args.reader_workers,
            reader_refresh=False,
        )

    reward_pins = {
        "environment_hash": environment_hash,
        "utility_artifact_hash": str(utility_artifact.get("artifact_hash")),
        "count_state_hash": str(count_state.get("artifact_hash")),
        "execution_contract_version": UTILITY_EXECUTION_CONTRACT_VERSION,
    }
    points = tuple(
        calibration_point_from_result(
            trajectory,
            _cached_result(
                trajectory,
                document=documents[trajectory.doc_id],
                utility_artifact=utility_artifact,
                environment_hash=environment_hash,
                cache=cache,
            ),
            count_reward=count_reward,
            count_state=count_state,
            utility_artifact=utility_artifact,
            reward_pins=reward_pins,
        )
        for trajectory in trajectories
    )
    split_by_doc = _split_by_doc(tuple(documents))
    pool_artifact = freeze_calibration_pool(
        points, split_by_doc=split_by_doc, reward_pins=reward_pins,
    )
    candidate_menu = _candidate_menu(points)
    reader_jitter = reader_jitter_from_cache(
        cache,
        utility_artifact=utility_artifact,
        split_by_doc=split_by_doc,
    )
    spike = build_diagnostic_spike(
        points,
        documents=documents,
        utility_artifact=utility_artifact,
        count_reward=count_reward,
        count_state=count_state,
        menu_artifact=candidate_menu,
        split_by_doc=split_by_doc,
        reader_jitter=reader_jitter,
        counterfactual_records=(),
    )
    manifest = freeze_threshold_manifest(
        rules,
        spike,
        pins={
            "reward_version": UTILITY_EXECUTION_CONTRACT_VERSION,
            "environment_hash": environment_hash,
            "span_decision_artifact_hash": str(
                utility_artifact.get("environment_audit_hash") or environment_hash
            ),
            "utility_component_artifact_hash": str(utility_artifact.get("artifact_hash")),
            "count_artifact_hash": str(count_state.get("artifact_hash")),
        },
    )
    final_menu = select_lambda_menu(
        points,
        menu_size=4,
        min_adjacent_winner_change=float(
            manifest["feasibility_gates"]["min_adjacent_winner_change"]
        ),
        max_placeholder_fraction=float(
            manifest["feasibility_gates"]["max_all_placeholder_fraction"]
        ),
        min_supported_documents_by_corpus=int(
            manifest["feasibility_gates"]["min_supported_documents_by_corpus"]
        ),
        min_supported_decisions_by_type=int(
            manifest["feasibility_gates"]["min_supported_decisions_by_type"]
        ),
    )
    gate_report = {
        "report_version": "ranker-v2-preflight-gate-v1",
        "environment_hash": environment_hash,
        "clauses": {
            "count_state": count_state.get("gate_report", {}).get("verdict"),
            "missing_occurrence_decision_mappings": len(
                count_state.get("gate_report", {}).get("missing_policy_mappings", ())
            ),
            "nonmonotone_profiles": len(
                count_state.get("gate_report", {}).get("nonmonotone_profiles", ())
            ),
            "lambda_menu": final_menu["verdict"],
        },
        "verdict": "PASS" if final_menu["verdict"] == "PASS" else "FAIL",
    }
    outputs = {
        "calibration-pool.json": pool_artifact,
        "diagnostic-spike.json": spike,
        "threshold-manifest.json": manifest,
        "switch-points.json": {
            "artifact_version": "ranker-v2-switch-points-v1",
            "calibration_pool_hash": pool_artifact["artifact_hash"],
            "switch_points": final_menu["switch_points"],
        },
        "replay-report.json": final_menu["replay_report"],
        "lambda-menu.json": final_menu,
        "gate-report.json": gate_report,
    }
    for name, payload in outputs.items():
        _write_json(out_dir / name, payload)
    print(
        f"PREFLIGHT {gate_report['verdict']} documents={len(documents)} "
        f"trajectories={len(points)} profiles={len(final_menu['values'])}",
        flush=True,
    )
    return 0 if gate_report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
