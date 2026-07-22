from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cloak.train.count_reward import (
    CountActionScore,
    CountGateError,
    CountReward,
    TypeCountReference,
    _build_count_reward_state,
    build_count_reward_state,
    expected_count_loss,
    resolve_type_references,
    validate_complete_counts,
)


ROOT = Path(__file__).resolve().parents[3]


def _grounding(
    status: str = "certifying",
    *,
    source_family: str = "fixture-ontology",
    universe_size: float | None = None,
    universe_ref: str | None = None,
) -> dict:
    if status == "certifying":
        row = {
            "status": status,
            "source_family": source_family,
            "member_set_ref": "fixture:members",
            "selector": "fixture.descendants",
        }
    elif status == "model-proposed":
        row = {
            "status": status,
            "source_family": source_family,
            "count_evidence": "fixture reviewed estimate",
            "selector": "fixture vocabulary slice",
            "member_set_ref": None,
        }
    else:
        row = {
            "status": status,
            "source_family": source_family,
            "member_set_ref": "generated-universe:fixture",
            "selector": "fixture generated group",
        }
    if universe_size is not None:
        row["universe_size"] = universe_size
    if universe_ref is not None:
        row["universe_ref"] = universe_ref
    return row


def _environment(
    profiles: list[tuple[str, list[float | None]]],
    *,
    runtime_type: str = "health-condition",
    groundings: dict[tuple[int, int], dict | None] | None = None,
) -> dict:
    decisions = []
    occurrences = []
    policy_ids = []
    groundings = groundings or {}
    for profile_index, (profile_id, counts) in enumerate(profiles):
        decision_id = f"decision-{profile_index}"
        occurrence_id = f"occurrence-{profile_index}"
        policy_ids.append(decision_id)
        occurrences.append({
            "occurrence_id": occurrence_id,
            "decision_id": decision_id,
            "start": profile_index * 10,
            "end": profile_index * 10 + 5,
        })
        actions = []
        for action_index, count in enumerate(counts):
            grounding = groundings.get(
                (profile_index, action_index), _grounding("model-proposed")
            )
            actions.append({
                "action_id": f"level-{profile_index}-{action_index}",
                "mode": "level",
                "fill": f"level {profile_index} {action_index}",
                "legal": True,
                "authored_level_index": action_index,
                "count": count,
                "count_grounding": grounding,
            })
        actions.extend([
            {
                "action_id": f"keep-{profile_index}",
                "mode": "keep",
                "fill": profile_id,
                "legal": True,
            },
            {
                "action_id": f"placeholder-{profile_index}",
                "mode": "placeholder",
                "fill": None,
                "legal": True,
            },
        ])
        decisions.append({
            "decision_id": decision_id,
            "profile_id": f"{runtime_type}:{profile_id}" if profile_id else None,
            "runtime_type": runtime_type,
            "canonical_key": profile_id,
            "occurrence_ids": [occurrence_id],
            "ranker_selectable": True,
            "actions": actions,
        })
    document = {
        "corpus": "aci",
        "occurrences": occurrences,
        "decisions": decisions,
        "environment_document_hash": "sha256:fixture-document",
    }
    return {
        "artifact_version": "ranker-v2-environment-v2",
        "frozen_environment": {
            "artifact_version": "occurrence-decisions-v2",
            "environment_hash": "sha256:fixture-environment",
            "documents": {"aci/D1": document},
        },
        "corpora": {"aci": {"aci/D1": {"policy_decision_ids": policy_ids}}},
    }


def test_public_count_types_are_exact_and_frozen():
    assert [field.name for field in fields(CountActionScore)] == [
        "action_id", "decision_id", "runtime_type", "profile_id", "mode", "count",
        "score", "grounding_status", "source_family", "evidence_ref",
    ]
    assert [field.name for field in fields(TypeCountReference)] == [
        "runtime_type", "k_ref", "resolution", "profile_support",
        "low_reference_support", "flat_count_signal",
    ]
    row = TypeCountReference("drug", 10.0, "max-profile-fallback", 1, True, False)
    with pytest.raises(FrozenInstanceError):
        row.k_ref = 20.0


def test_grounded_universe_resolution_precedes_profile_fallback():
    groundings = {
        (index, 0): _grounding(
            universe_size=5000.0, universe_ref="fixture:coherent-universe"
        )
        for index in range(2)
    }
    environment = _environment([("a", [10.0]), ("b", [30.0])], groundings=groundings)

    reference = resolve_type_references(environment)["health-condition"]

    assert reference == TypeCountReference(
        runtime_type="health-condition",
        k_ref=5000.0,
        resolution="grounded-universe",
        profile_support=2,
        low_reference_support=False,
        flat_count_signal=False,
    )


def test_profile_balanced_p95_uses_one_maximum_per_profile_at_twenty_profiles():
    environment = _environment([
        (f"profile-{index}", [float(index), float(index) / 2])
        for index in range(1, 21)
    ])

    reference = resolve_type_references(environment)["health-condition"]

    assert reference.resolution == "profile-balanced-p95"
    assert reference.profile_support == 20
    assert reference.k_ref == pytest.approx(19.05)
    assert reference.low_reference_support is False


def test_max_profile_fallback_below_twenty_profiles_is_tagged():
    environment = _environment([("a", [5.0]), ("b", [20.0])])

    reference = resolve_type_references(environment)["health-condition"]

    assert reference == TypeCountReference(
        runtime_type="health-condition",
        k_ref=20.0,
        resolution="max-profile-fallback",
        profile_support=2,
        low_reference_support=True,
        flat_count_signal=False,
    )


def test_all_one_type_has_flat_count_signal():
    environment = _environment([(f"profile-{index}", [1.0]) for index in range(20)])

    reference = resolve_type_references(environment)["health-condition"]

    assert reference.k_ref == 1.0
    assert reference.resolution == "flat-count-signal"
    assert reference.flat_count_signal is True


def test_gate_reports_every_level_and_all_strict_clauses():
    environment = _environment([("a", [2.0, 4.0])])

    report = validate_complete_counts(environment)

    assert report["verdict"] == "PASS"
    assert report["summary"] == {
        "level_actions": 2,
        "admitted_level_actions": 2,
        "gap_count": 0,
        "model_proposed_level_actions": 2,
    }
    assert {clause["clause"]: clause["result"] for clause in report["clauses"]} == {
        "explicit_coverage": "PASS",
        "fallback_gradient_mass": "PASS",
        "missing_policy_mappings": "PASS",
        "nonmonotone_profiles": "PASS",
    }
    level = report["levels"][0]
    assert level["count"] == 2.0
    assert level["grounding_status"] == "model-proposed"
    assert level["source_family"] == "fixture-ontology"
    assert level["evidence_ref"] == "fixture reviewed estimate"
    assert level["clause_results"] == {
        "explicit_count": "PASS",
        "accepted_status": "PASS",
        "status_evidence": "PASS",
        "nondefault_provenance": "PASS",
    }


@pytest.mark.parametrize(
    "grounding",
    [
        None,
        _grounding("legacy-default", source_family="legacy-default"),
        _grounding("model-proposed", source_family="GENERIC-sentinel"),
        {
            "status": "model-proposed",
            "source_family": "model-proposed",
            "selector": "fixture",
            "count_evidence": "default fallback count",
        },
    ],
)
def test_strict_gate_rejects_missing_default_and_generic_provenance(grounding):
    environment = _environment(
        [("a", [10.0])], groundings={(0, 0): grounding},
    )

    with pytest.raises(CountGateError) as raised:
        build_count_reward_state(environment)

    assert raised.value.report["summary"]["gap_count"] == 1
    assert raised.value.report["levels"][0]["admitted"] is False


def test_provisional_gap_tags_whole_decision_and_preserves_placeholder_endpoint():
    environment = _environment([("a", [10.0, None])])

    state = _build_count_reward_state(environment, provisional=True)
    reward = CountReward.from_artifact(state)

    assert state["gate_mode"] == "provisional"
    assert state["gate_report"]["verdict"] == "PASS"
    assert state["gate_report"]["strict_verdict"] == "FAIL"
    assert state["gate_report"]["provisional_tag_count_by_type"] == {
        "health-condition": 1,
    }
    assert state["provisional_decision_tags"] == [{
        "decision_id": "decision-0",
        "runtime_type": "health-condition",
        "profile_id": "health-condition:a",
        "gap_action_ids": ["level-0-1"],
    }]
    scores = reward.action_scores(
        "decision-0", ["level-0-0", "level-0-1", "keep-0", "placeholder-0"]
    )
    assert torch.equal(scores, torch.tensor([0.0, 0.0, 0.0, 1.0]))


@pytest.mark.parametrize("provisional", [False, True])
def test_nonmonotone_non_null_own_profile_ladder_raises_in_both_modes(provisional):
    environment = _environment([("a", [20.0, 10.0])])

    with pytest.raises(CountGateError, match="nonmonotone_profiles"):
        _build_count_reward_state(environment, provisional=provisional)


@pytest.mark.parametrize("provisional", [False, True])
def test_missing_occurrence_mapping_raises_in_both_modes(provisional):
    environment = _environment([("a", [10.0])])
    environment["frozen_environment"]["documents"]["aci/D1"]["occurrences"] = []

    with pytest.raises(CountGateError, match="missing_policy_mappings"):
        _build_count_reward_state(environment, provisional=provisional)


def test_runtime_scores_clip_and_keep_placeholder_endpoints_are_exact():
    groundings = {
        (0, index): _grounding(
            universe_size=100.0, universe_ref="fixture:universe"
        )
        for index in range(2)
    }
    environment = _environment([("a", [10.0, 1000.0])], groundings=groundings)
    reward = CountReward.from_artifact(build_count_reward_state(environment))

    scores = reward.action_scores(
        "decision-0", ["level-0-0", "level-0-1", "keep-0", "placeholder-0"]
    )

    assert torch.allclose(scores, torch.tensor([0.5, 1.0, 0.0, 1.0]))
    assert reward.selected_document_score({"decision-0": "level-0-0"}) == pytest.approx(0.5)


def test_expected_count_loss_is_exact_over_every_complete_legal_menu():
    environment = _environment([("a", [10.0])])
    reward = CountReward.from_artifact(build_count_reward_state(environment))
    first_logits = torch.tensor([0.0, math.log(3.0)], requires_grad=True)
    second_logits = torch.tensor([math.log(3.0), 0.0], requires_grad=True)
    replay_steps = [
        SimpleNamespace(
            decision_id="decision-0",
            legal_action_ids=("keep-0", "placeholder-0"),
            log_probs=torch.log_softmax(first_logits, dim=0),
        ),
        SimpleNamespace(
            decision_id="decision-0",
            legal_action_ids=("keep-0", "placeholder-0"),
            log_probs=torch.log_softmax(second_logits, dim=0),
        ),
    ]

    loss = expected_count_loss(
        replay_steps, reward, lambda_value=2.0, decision_count=2, rollout_count=2,
    )

    assert loss.item() == pytest.approx(-0.5)
    loss.backward()
    assert first_logits.grad is not None
    assert second_logits.grad is not None


def test_artifact_cli_writes_strict_failure_and_provisional_pass(tmp_path, monkeypatch):
    module = importlib.import_module("build_count_reward_state")
    environment = _environment([("a", [10.0, None])])
    environment_path = tmp_path / "environment.json"
    strict_out = tmp_path / "strict-state.json"
    strict_report = tmp_path / "strict-report.json"
    provisional_out = tmp_path / "count-state.json"
    provisional_report = tmp_path / "provisional-report.json"
    issue_path = tmp_path / "count-issues.md"
    environment_path.write_text(json.dumps(environment))
    issue_path.write_text("# Existing count issue\n")
    monkeypatch.setattr(module, "COUNT_DEFECT_ISSUE", issue_path)

    monkeypatch.setattr(sys, "argv", [
        "build_count_reward_state.py",
        "--environment", str(environment_path),
        "--out", str(strict_out),
        "--gate-report", str(strict_report),
    ])
    with pytest.raises(SystemExit) as strict_exit:
        module.main()
    assert strict_exit.value.code != 0
    assert not strict_out.exists()
    assert json.loads(strict_report.read_text())["summary"]["gap_count"] == 1

    monkeypatch.setattr(sys, "argv", [
        "build_count_reward_state.py",
        "--environment", str(environment_path),
        "--out", str(provisional_out),
        "--gate-report", str(provisional_report),
        "--provisional",
    ])
    module.main()

    payload = json.loads(provisional_out.read_text())
    hash_payload = {key: value for key, value in payload.items() if key != "artifact_hash"}
    encoded = json.dumps(
        hash_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    assert payload["artifact_hash"] == "sha256:" + hashlib.sha256(encoded).hexdigest()
    assert json.loads(provisional_report.read_text())["verdict"] == "PASS"
    assert len(payload["provisional_decision_tags"]) == 1
    assert provisional_out.read_text() == json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ) + "\n"
    assert "decision-0" in issue_path.read_text()
