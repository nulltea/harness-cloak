from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from cloak.train.count_reward import CountGateError
from cloak.train.profile_count import (
    ProfileActionTarget,
    ProfileCountTargets,
    build_profile_count_targets,
)


def _grounding(*, evidence: str = "fixture reviewed estimate") -> dict:
    return {
        "status": "model-proposed",
        "source_family": "fixture-ontology",
        "selector": "fixture vocabulary slice",
        "count_evidence": evidence,
        "member_set_ref": None,
    }


def _decision(
    decision_id: str,
    profile_id: str | None,
    counts: tuple[float | None, ...],
    *,
    runtime_type: str,
    canonical_key: str,
) -> dict:
    names = ("fine", "coarse") if len(counts) == 2 else ("only",)
    actions = [
        {
            "action_id": f"{decision_id}-{name}",
            "mode": "level",
            "fill": f"{canonical_key} {name}",
            "legal": True,
            "authored_level_index": index,
            "count": count,
            "count_grounding": _grounding() if count is not None else None,
        }
        for index, (name, count) in enumerate(zip(names, counts, strict=True))
    ]
    actions.extend([
        {
            "action_id": f"{decision_id}-keep",
            "mode": "keep",
            "fill": canonical_key,
            "legal": True,
        },
        {
            "action_id": f"{decision_id}-placeholder",
            "mode": "placeholder",
            "fill": None,
            "legal": True,
        },
    ])
    return {
        "decision_id": decision_id,
        "profile_id": profile_id,
        "runtime_type": runtime_type,
        "canonical_key": canonical_key,
        "occurrence_ids": [f"occurrence-{decision_id}"],
        "ranker_selectable": True,
        "actions": actions,
    }


def _environment() -> dict:
    decisions = [
        _decision(
            "d1", "drug:aspirin", (10.0, 100.0),
            runtime_type="drug", canonical_key="aspirin",
        ),
        _decision(
            "d2", "health-condition:migraine", (4.0, 16.0),
            runtime_type="health-condition", canonical_key="migraine",
        ),
    ]
    occurrences = [
        {
            "occurrence_id": f"occurrence-{decision['decision_id']}",
            "decision_id": decision["decision_id"],
        }
        for decision in decisions
    ]
    return {
        "artifact_version": "ranker-v2-environment-v2",
        "frozen_environment": {
            "artifact_version": "occurrence-decisions-v2",
            "environment_hash": "sha256:fixture-environment",
            "documents": {
                "fixture/doc": {
                    "corpus": "fixture",
                    "occurrences": occurrences,
                    "decisions": decisions,
                    "environment_document_hash": "sha256:fixture-document",
                },
            },
        },
        "corpora": {
            "fixture": {
                "fixture/doc": {
                    "policy_decision_ids": ["d1", "d2"],
                },
            },
        },
    }


def _build_runtime(environment: dict | None = None) -> tuple[dict, ProfileCountTargets]:
    artifact = build_profile_count_targets(environment or _environment(), strict=True)
    return artifact, ProfileCountTargets.from_artifact(artifact)


def _decision_by_id(environment: dict, decision_id: str) -> dict:
    decisions = environment["frozen_environment"]["documents"]["fixture/doc"]["decisions"]
    return next(row for row in decisions if row["decision_id"] == decision_id)


def test_public_types_are_exact_frozen_and_profile_relative():
    artifact, targets = _build_runtime()

    assert [field.name for field in fields(ProfileActionTarget)] == [
        "decision_id", "action_id", "profile_id", "runtime_type", "mode",
        "log_count", "profile_score", "grounding_status", "source_family",
    ]
    with pytest.raises(FrozenInstanceError):
        targets.target_rows()[0].profile_score = 0.0
    assert targets.action_scores(
        "d1", ("d1-keep", "d1-fine", "d1-coarse", "d1-placeholder")
    ).tolist() == [0.0, math.log(10) / math.log(100), 1.0, 1.0]
    assert targets.target_rows()[0].profile_id == "drug:aspirin"
    assert artifact["profile_tags"] == {}
    assert artifact["artifact_hash"].startswith("sha256:")


def test_scores_use_only_each_decisions_own_profile_counts():
    environment = _environment()
    first = _decision_by_id(environment, "d1")
    first["actions"][0]["aset"] = 1e12
    first["actions"][0]["coarseness_rank"] = 1e12
    second = _decision_by_id(environment, "d2")
    second["actions"][1]["count"] = 64.0

    _, targets = _build_runtime(environment)

    assert targets.action_scores("d1", ("d1-fine", "d1-coarse")).tolist() == [0.5, 1.0]
    assert targets.action_scores("d2", ("d2-fine", "d2-coarse")).tolist() == [
        math.log(4.0) / math.log(64.0), 1.0,
    ]


@pytest.mark.parametrize(
    ("mutation", "clause"),
    [
        ("missing_count", "explicit_coverage"),
        ("missing_grounding", "explicit_coverage"),
        ("nonmonotone", "nonmonotone_authored_ladders"),
        ("all_one", "positive_profile_denominators"),
        ("fallback", "explicit_coverage"),
        ("duplicate_profile", "duplicate_profile_ids"),
    ],
)
def test_strict_gate_rejects_invalid_profile_targets(mutation: str, clause: str):
    environment = _environment()
    first = _decision_by_id(environment, "d1")
    if mutation == "missing_count":
        first["actions"][0]["count"] = None
    elif mutation == "missing_grounding":
        first["actions"][0]["count_grounding"] = None
    elif mutation == "nonmonotone":
        first["actions"][0]["count"] = 100.0
        first["actions"][1]["count"] = 10.0
    elif mutation == "all_one":
        first["actions"][0]["count"] = 1.0
        first["actions"][1]["count"] = 1.0
    elif mutation == "fallback":
        first["actions"][0]["count_grounding"] = _grounding(
            evidence="default fallback count"
        )
    elif mutation == "duplicate_profile":
        second = _decision_by_id(environment, "d2")
        second["profile_id"] = "drug:aspirin"

    with pytest.raises(CountGateError) as raised:
        build_profile_count_targets(environment, strict=True)

    clauses = {row["clause"]: row["result"] for row in raised.value.report["clauses"]}
    assert clauses[clause] == "FAIL"


def test_singleton_count_one_scores_one_and_is_tagged():
    environment = _environment()
    singleton = _decision(
        "d3", "LOC:berlin", (1.0,), runtime_type="LOC", canonical_key="berlin"
    )
    document = environment["frozen_environment"]["documents"]["fixture/doc"]
    document["decisions"] = [singleton]
    document["occurrences"] = [{
        "occurrence_id": "occurrence-d3", "decision_id": "d3",
    }]
    environment["corpora"]["fixture"]["fixture/doc"]["policy_decision_ids"] = ["d3"]

    artifact, targets = _build_runtime(environment)

    assert targets.action_scores("d3", ("d3-only",)).tolist() == [1.0]
    assert artifact["profile_tags"] == {
        "LOC:berlin": ["singleton_profile_normalization"],
    }


def test_runtime_rejects_action_owned_by_a_different_decision():
    _, targets = _build_runtime()

    with pytest.raises(KeyError, match="d2-fine"):
        targets.action_scores("d1", ("d2-fine",))


def test_diagnostic_artifact_marks_incomplete_decision_and_withholds_level_targets():
    environment = _environment()
    first = _decision_by_id(environment, "d1")
    first["actions"][0]["count"] = None
    first["actions"][0]["count_grounding"] = None

    artifact = build_profile_count_targets(environment, strict=False)
    targets = ProfileCountTargets.from_artifact(artifact)

    assert artifact["gate_mode"] == "diagnostic"
    assert artifact["decision_eligibility"] == {"d1": False, "d2": True}
    d1_rows = [
        row for row in artifact["action_targets"].values()
        if row["decision_id"] == "d1"
    ]
    assert {row["mode"] for row in d1_rows} == {"keep", "placeholder"}
    assert {row["profile_score"] for row in d1_rows} == {0.0, 1.0}
    assert all(row.decision_id == "d2" for row in targets.target_rows())
    with pytest.raises(ValueError, match="not privacy-head eligible"):
        targets.action_scores("d1", ("d1-keep", "d1-placeholder"))


def test_selected_document_score_requires_complete_eligible_action_vector():
    _, targets = _build_runtime()

    assert targets.selected_document_score({
        "d1": "d1-fine", "d2": "d2-placeholder",
    }) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="missing decisions"):
        targets.selected_document_score({"d1": "d1-fine"})


def test_artifact_hash_is_canonical_and_verified():
    artifact, _ = _build_runtime()
    unhashed = {key: value for key, value in artifact.items() if key != "artifact_hash"}
    encoded = json.dumps(
        unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    assert artifact["artifact_hash"] == "sha256:" + hashlib.sha256(encoded).hexdigest()

    artifact["gate_mode"] = "diagnostic"
    with pytest.raises(ValueError, match="hash mismatch"):
        ProfileCountTargets.from_artifact(artifact)


def test_cli_strict_failure_and_diagnostic_publish(tmp_path, monkeypatch):
    module = importlib.import_module("build_profile_count_targets")
    environment = _environment()
    first = _decision_by_id(environment, "d1")
    first["actions"][0]["count"] = None
    first["actions"][0]["count_grounding"] = None
    environment_path = tmp_path / "environment.json"
    target_path = tmp_path / "profile-count-targets.json"
    strict_report_path = tmp_path / "strict-gate.json"
    diagnostic_report_path = tmp_path / "diagnostic-gate.json"
    environment_path.write_text(json.dumps(environment))

    monkeypatch.setattr(sys, "argv", [
        "build_profile_count_targets.py",
        "--environment", str(environment_path),
        "--out", str(target_path),
        "--gate-report", str(strict_report_path),
        "--strict",
    ])
    with pytest.raises(SystemExit) as raised:
        module.main()
    assert raised.value.code == 2
    assert not target_path.exists()
    strict_report = json.loads(strict_report_path.read_text())
    assert strict_report["verdict"] == "FAIL"
    assert strict_report["summary"]["privacy_head_ineligible_decisions"] == 1

    monkeypatch.setattr(sys, "argv", [
        "build_profile_count_targets.py",
        "--environment", str(environment_path),
        "--out", str(target_path),
        "--gate-report", str(diagnostic_report_path),
        "--diagnostic",
    ])
    module.main()

    payload = json.loads(target_path.read_text())
    assert payload["decision_eligibility"]["d1"] is False
    assert json.loads(diagnostic_report_path.read_text())["mode"] == "diagnostic"
    assert target_path.read_text() == json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ) + "\n"
