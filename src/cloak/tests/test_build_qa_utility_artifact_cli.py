import json
import os
import subprocess
from pathlib import Path

import train_ranker as tr


def test_build_qa_utility_artifact_cli(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    out_path = tmp_path / "utility.json"
    manifest_path.write_text(json.dumps({
        "family_budgets": {"context": 0.6, "delivered": 0.4},
        "min_context_assertions": 0,
        "task_pin": "aci-v1",
        "reader_pin": "reader-v1",
    }))

    result = subprocess.run(
        [".venv/bin/python", "scripts/build_qa_utility_artifact.py",
         "--env", "data/ranker_env.json",
         "--arms", "data/task_arms_tau0.02.json",
         "--corpus", "clinical", "--doc-id", "aci/D2N002",
         "--threshold-manifest", str(manifest_path), "--out", str(out_path)],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    artifact = json.loads(out_path.read_text())
    assert artifact["artifact_version"] == "utility-assertions-v1"
    assert artifact["gate_manifest_hash"].startswith("sha256:")
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

    report = tr.qa_utility_preflight_report(
        artifact,
        {"environment_hash": artifact["environment_hash"]},
    )
    assert report["documents"]["aci/D2N002"]["measurement_state"] == "partial"
    assert report["call_budget"]["base"]["remote_round_trips_per_rollout"] == 1
    assert report["call_budget"]["counterfactual"]["remote_round_trips_per_selected_pair"] == 1
    assert report["executed_remote_calls"] == 0
