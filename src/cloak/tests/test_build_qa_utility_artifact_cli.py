import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import build_qa_utility_artifact as builder_cli  # noqa: E402


WORKTREE = Path(__file__).resolve().parents[3]
HOST_REPO = WORKTREE.parents[1]


_COST_BUDGETS = {
    "base": {
        "remote_round_trips_per_rollout": 1,
        "context_reader_batches_per_rollout": 1,
    },
    "counterfactual": {
        "remote_round_trips_per_selected_pair": 1,
        "context_reader_batches_per_selected_pair": 1,
    },
}


def test_build_qa_utility_artifact_cli(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    out_path = tmp_path / "utility.json"
    manifest_path.write_text(json.dumps({
        "family_budgets": {"context": 0.6, "delivered": 0.4},
        "min_context_assertions": 0,
        "task_pin": "aci-v1",
        "reader_pin": "must-not-override-live-reader",
        "cost_budgets": {
            "base": {
                "remote_round_trips_per_rollout": 1,
                "context_reader_batches_per_rollout": 1,
            },
            "counterfactual": {
                "remote_round_trips_per_selected_pair": 1,
                "context_reader_batches_per_selected_pair": 1,
            },
        },
    }))

    result = subprocess.run(
        [sys.executable, str(WORKTREE / "scripts/build_qa_utility_artifact.py"),
         "--env", "data/ranker_env.json",
         "--arms", "data/task_arms_tau0.02.json",
         "--corpus", "clinical", "--doc-id", "aci/D2N002",
         "--threshold-manifest", str(manifest_path), "--out", str(out_path)],
        cwd=HOST_REPO,
        env={
            **os.environ,
            "PYTHONPATH": f"{WORKTREE / 'src'}:{WORKTREE / 'scripts'}",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    artifact = json.loads(out_path.read_text())
    assert artifact["artifact_version"] == "utility-assertions-v1"
    assert artifact["gate_manifest_hash"] == builder_cli._hash(
        artifact["threshold_manifest"]
    )
    assert artifact["task_pin"] == builder_cli.AciTaskAdapter.task_pin
    assert artifact["builder_pin"]["version"]
    assert artifact["teacher_pin"]["enabled"] is False
    assert artifact["reader_pin"]["pin_version"] == "qa-context-reader-v1"
    assert artifact["scorer_pin"]["reader"] == artifact["reader_pin"]
    assert artifact["threshold_manifest_pin"] == {
        "schema": "qa-threshold-manifest-v1",
        "sha256": artifact["gate_manifest_hash"],
    }
    assert artifact["cost_budgets"] == artifact["threshold_manifest"]["cost_budgets"]
    assert artifact["threshold_manifest"]["cost_budgets"] == {
        "base": {
            "remote_round_trips_per_rollout": 1,
            "context_reader_batches_per_rollout": 1,
        },
        "counterfactual": {
            "remote_round_trips_per_selected_pair": 1,
            "context_reader_batches_per_selected_pair": 1,
        },
    }
    assert list(artifact["documents"]) == ["aci/D2N002"]
    assert artifact["documents"]["aci/D2N002"]["missing_family_budgets"] == ["context"]
    assert any(
        row["scoring_contract"]["value"] == "hypothyroidism"
        for row in artifact["assertions"].values()
    )
    assert any(
        len(row["occurrence_ids"]) > 1
        for row in artifact["assertions"].values()
        if row["scope"] == "linked"
    )
    assert "OpenRouter" not in result.stdout

    report_line = next(
        line for line in result.stdout.splitlines() if line.startswith("qa preflight: ")
    )
    report = json.loads(report_line.removeprefix("qa preflight: "))
    assert report["documents"]["aci/D2N002"]["measurement_state"] == "partial"
    assert report["documents"]["aci/D2N002"]["accepted_assertion_count"] == 13
    assert report["documents"]["aci/D2N002"]["missing_family_budgets"] == ["context"]
    assert report["call_budget"]["base"]["remote_round_trips_per_rollout"] == 1
    assert report["call_budget"]["counterfactual"]["remote_round_trips_per_selected_pair"] == 1
    assert report["executed_remote_calls"] == 0


def test_build_cli_floor_override_changes_frozen_legality_and_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(HOST_REPO)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "family_budgets": {"context": 0.6, "delivered": 0.4},
        "cost_budgets": _COST_BUDGETS,
    }))
    common = [
        "--env", "data/ranker_env.json",
        "--arms", "data/task_arms_tau0.02.json",
        "--corpus", "clinical",
        "--doc-id", "aci/D2N002",
        "--threshold-manifest", str(manifest_path),
        "--out", str(tmp_path / "utility.json"),
    ]
    default_args = builder_cli.parse_args(common)
    override_args = builder_cli.parse_args([
        *common,
        "--floors", "DATETIME=1e30,DEM=1e30,MISC=1e30",
    ])

    _default_artifact, default_environment = builder_cli.build_from_files(
        default_args, return_frozen_environment=True
    )
    _override_artifact, override_environment = builder_cli.build_from_files(
        override_args, return_frozen_environment=True
    )

    assert override_environment["environment_hash"] != default_environment["environment_hash"]
    assert override_environment["effective_floors"]["DEM"] == 1e30
    assert all(
        action["mode"] == "placeholder" or not action["legal"]
        for decision in override_environment["documents"]["aci/D2N002"]["decisions"]
        for action in decision["actions"]
    )


def test_context_build_cli_preflights_against_full_frozen_environment(
    tmp_path, monkeypatch, capsys,
):
    doc_id = "aci/context-fixture"
    source = "Hypothyroidism is treated with Synthroid."
    spans = [
        {
            "surface": "Hypothyroidism",
            "type": "health-condition",
            "start": source.index("Hypothyroidism"),
            "end": source.index("Hypothyroidism") + len("Hypothyroidism"),
                "actions": [
                    {"fill": "an endocrine condition", "mode": "level", "aset": 100.0},
                    {"fill": "a condition", "mode": "level", "aset": 100.0},
                    {"fill": "Hypothyroidism", "mode": "level", "keep": True,
                     "aset": 100.0},
                {"fill": None, "mode": "placeholder"},
            ],
        },
        {
            "surface": "Synthroid",
            "type": "drug",
            "start": source.index("Synthroid"),
            "end": source.index("Synthroid") + len("Synthroid"),
                "actions": [
                    {"fill": "a thyroid medication", "mode": "level", "aset": 100.0},
                    {"fill": "a medication", "mode": "level", "aset": 100.0},
                    {"fill": "Synthroid", "mode": "level", "keep": True, "aset": 100.0},
                {"fill": None, "mode": "placeholder"},
            ],
        },
    ]
    occurrences = [
        {
            "start": span["start"], "end": span["end"], "surface": span["surface"],
            "type": span["type"], "action": "generalize",
            "replacement": span["actions"][0]["fill"],
            "lattice": [action["fill"] for action in span["actions"][:2]],
        }
        for span in spans
    ]
    env_path = tmp_path / "env.json"
    arms_path = tmp_path / "arms.json"
    manifest_path = tmp_path / "manifest.json"
    out_path = tmp_path / "utility.json"
    env_path.write_text(json.dumps({
        "corpora": {"clinical": {doc_id: {"spans": spans}}},
    }))
    arms_path.write_text(json.dumps({
        "clinical": {doc_id: {"tau_walk": [source, occurrences]}},
    }))
    manifest_path.write_text(json.dumps({
        "family_budgets": {"context": 0.6, "delivered": 0.4},
        "min_context_assertions": 1,
        "reader_threshold": 1.0,
        "reader_stability_repetitions": 1,
        "reader_option_permutations": 1,
        "reader_stability_threshold": 1.0,
        "cost_budgets": _COST_BUDGETS,
    }))
    monkeypatch.setattr(builder_cli, "_source_rows", lambda corpus, doc_ids: {
        doc_id: {"id": doc_id, "text": source, "gold_ref": source},
    })

    class Teacher:
        def propose(self, prompt):
            occurrence_ids = re.findall(
                r'"occurrence_id": "(sha256:[^"]+)"', prompt
            )
            return [{
                "relation": "treated_with",
                "argument_occurrence_ids": occurrence_ids,
                "support_properties": {
                    occurrence_ids[0]: "an endocrine condition",
                    occurrence_ids[1]: "a thyroid medication",
                },
                "answer_occurrence_id": occurrence_ids[1],
                "answer_property": "a thyroid medication",
                "question": "What treatment category is used for the endocrine disorder?",
                "evidence_quote": source,
            }]

    def reader(questions, context):
        if "<" in context:
            return [""] * len(questions)
        return ["a thyroid medication"] * len(questions)

    builder_cli.main([
        "--env", str(env_path),
        "--arms", str(arms_path),
        "--corpus", "clinical",
        "--doc-id", doc_id,
        "--threshold-manifest", str(manifest_path),
        "--out", str(out_path),
        "--relation-teacher",
    ], relation_teacher=Teacher(), reader=reader)

    artifact = json.loads(out_path.read_text())
    context_rows = [
        row for row in artifact["assertions"].values() if row["family"] == "context"
    ]
    assert len(context_rows) == 1
    assert len(context_rows[0]["expected_action_support"]["joint_anchor_action_vector"]) == 2
    report_line = next(
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("qa preflight: ")
    )
    report = json.loads(report_line.removeprefix("qa preflight: "))
    assert report["documents"][doc_id]["measurement_state"] == "measured"
    assert report["documents"][doc_id]["context_assertion_count"] == 1


@pytest.mark.parametrize("family_budgets", [
    {"context": 0.6},
    {"context": 0.6, "delivered": 0.4, "unknown": 0.1},
    {"context": 0.0, "delivered": 1.0},
    {"context": float("nan"), "delivered": 1.0},
    {"context": float("inf"), "delivered": 1.0},
])
def test_build_cli_rejects_invalid_family_budgets_before_source_loading(
    tmp_path, monkeypatch, family_budgets,
):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "family_budgets": family_budgets,
        "cost_budgets": _COST_BUDGETS,
    }))
    monkeypatch.setattr(
        builder_cli,
        "_source_rows",
        lambda *args: pytest.fail("source loading must not run"),
    )
    args = builder_cli.parse_args([
        "--env", "data/ranker_env.json",
        "--arms", "data/task_arms_tau0.02.json",
        "--corpus", "clinical",
        "--doc-id", "aci/D2N002",
        "--threshold-manifest", str(manifest_path),
        "--out", str(tmp_path / "out.json"),
    ])

    with pytest.raises(SystemExit, match="family budgets"):
        builder_cli.build_from_files(args)
