import json
import os
import subprocess
from pathlib import Path

import build_qa_utility_artifact as qa_cli
from cloak.train.qa_builder import (
    freeze_ranker_environment,
    frozen_occurrences_from_arms,
)


def _d2n002_fixture():
    source = (
        "The patient is diagnosed with hypothyroidism. "
        "Hypothyroidism is treated with Synthroid. "
        "The clinician orders thyroid labs."
    )
    reference = """HISTORY OF PRESENT ILLNESS
62-year-old male with hypothyroidism.
ASSESSMENT
Hypothyroidism — Endocrine — Stable
PLAN
Hypothyroidism — Synthroid — thyroid labs
"""
    span_specs = [
        ("hypothyroidism", "health-condition", "endocrine condition"),
        ("Synthroid", "drug", "thyroid medication"),
        ("thyroid labs", "medical-procedure", "thyroid test"),
    ]
    spans = []
    arm_rows = []
    for index, (surface, runtime_type, generalization) in enumerate(span_specs, start=1):
        start = source.index(surface)
        end = start + len(surface)
        spans.append({
            "surface": surface,
            "type": runtime_type,
            "start": start,
            "end": end,
            "actions": [
                {"fill": generalization, "mode": "level"},
                {"fill": None, "mode": "placeholder"},
            ],
        })
        arm_rows.append({
            "surface": surface,
            "type": runtime_type,
            "start": start,
            "end": end,
            "score": 0.9,
            "action": "placeholder",
            "replacement": f"<{runtime_type.upper().replace('-', '_')}_{index}>",
            "lattice": [generalization],
        })
    environment = {
        "corpora": {"clinical": {"aci/D2N002": {"spans": spans}}},
    }
    arms = {
        "_meta": {"detector": {
            "config": "qa-v2-clinical",
            "model": "deterministic-test-detector",
            "threshold": 0.35,
        }},
        "clinical": {"aci/D2N002": {"tau_walk": [source, arm_rows]}},
    }
    return source, reference, environment, arms


class _InjectedRelationTeacher:
    def __init__(self, proposals):
        self.proposals = proposals
        self.calls = 0

    def propose(self, prompt):
        assert "aci/D2N002" in prompt
        self.calls += 1
        return self.proposals


def _acceptance_reader(questions, context):
    if "<" in context and ">" in context:
        return ["unknown" for _ in questions]
    answers = []
    for question in questions:
        if "condition category" in question:
            answers.append("endocrine condition")
        elif "procedure or test category" in question:
            answers.append("thyroid test")
        else:
            answers.append("thyroid medication")
    return answers


def test_action_renderer_uses_synthesized_keep_as_source_identity():
    source, _, environment, arms = _d2n002_fixture()
    occurrence_rows = frozen_occurrences_from_arms({
        "clinical": arms["clinical"],
    })
    frozen = freeze_ranker_environment(
        environment, occurrences_by_document=occurrence_rows
    )
    action_vector = {
        decision["decision_id"]: next(
            action["action_id"] for action in decision["actions"]
            if action["mode"] == "keep"
        )
        for decision in frozen["documents"]["aci/D2N002"]["decisions"]
    }
    render = qa_cli._action_renderer(
        environment,
        frozen,
        {"clinical": arms["clinical"]},
        "clinical",
        {"aci/D2N002": source},
    )

    assert render("aci/D2N002", action_vector) == source


def test_cli_writes_normative_and_derived_views_from_same_artifact(tmp_path, monkeypatch):
    artifact = {
        "artifact_version": "utility-assertions-v1",
        "artifact_hash": "artifact-hash",
        "family_budgets": {"context": 0.6, "delivered": 0.4},
        "documents": {},
        "assertions": {},
        "rejections": {"summary_by_reason": {}, "records": []},
    }
    monkeypatch.setattr(qa_cli, "build_from_files", lambda *args, **kwargs: artifact)
    output = tmp_path / "utility.json"

    qa_cli.main([
        "--env", "unused-env.json",
        "--arms", "unused-arms.json",
        "--doc-id", "aci/D2N002",
        "--threshold-manifest", "unused-manifest.json",
        "--out", str(output),
    ])

    assertions_path = tmp_path / "utility.assertions.json"
    qa_pairs_path = tmp_path / "utility.qa-pairs.json"
    assert json.loads(output.read_text()) == artifact
    assert json.loads(assertions_path.read_text())["source_artifact_hash"] == "artifact-hash"
    assert json.loads(qa_pairs_path.read_text())["source_artifact_hash"] == "artifact-hash"


def test_d2n002_acceptance_exports_substantive_artifact_without_external_calls(
    tmp_path, monkeypatch,
):
    source, reference, environment, arms = _d2n002_fixture()
    environment_path = tmp_path / "ranker-env.json"
    arms_path = tmp_path / "arms.json"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "d2n002.json"
    environment_path.write_text(json.dumps(environment))
    arms_path.write_text(json.dumps(arms))
    manifest_path.write_text(json.dumps({
        "family_budgets": {"context": 0.6, "delivered": 0.4},
        "structural_cap": 0.1,
        "min_context_assertions": 10,
        "reader_threshold": 1.0,
    }))
    monkeypatch.setattr(
        qa_cli,
        "_source_rows",
        lambda corpus, doc_ids: {
            "aci/D2N002": {"id": "aci/D2N002", "text": source, "gold_ref": reference}
        },
    )
    occurrence_rows = frozen_occurrences_from_arms({
        "clinical": arms["clinical"],
    })
    frozen = freeze_ranker_environment(
        environment, occurrences_by_document=occurrence_rows
    )
    frozen_document = frozen["documents"]["aci/D2N002"]
    occurrences = {
        row["runtime_type"]: row["occurrence_id"]
        for row in frozen_document["occurrences"]
    }
    decisions = {
        row["runtime_type"]: row["decision_id"]
        for row in frozen_document["decisions"]
    }
    valid_proposal = {
        "relation": "treated_with",
        "argument_occurrence_ids": [
            occurrences["health-condition"], occurrences["drug"],
        ],
        "support_properties": {
            occurrences["health-condition"]: "endocrine condition",
            occurrences["drug"]: "thyroid medication",
        },
        "answer_occurrence_id": occurrences["drug"],
        "answer_property": "thyroid medication",
        "question": "What treatment category is used for the diagnosed condition?",
        "evidence_quote": "Hypothyroidism is treated with Synthroid.",
    }
    teacher = _InjectedRelationTeacher([
        valid_proposal,
        {**valid_proposal, "question": "Malformed question"},
    ])

    qa_cli.main([
        "--env", str(environment_path),
        "--arms", str(arms_path),
        "--corpus", "clinical",
        "--doc-id", "aci/D2N002",
        "--threshold-manifest", str(manifest_path),
        "--out", str(output),
    ], relation_teacher=teacher, reader=_acceptance_reader)

    artifact = json.loads(output.read_text())
    subtypes = [row["subtype"] for row in artifact["assertions"].values()]
    assert teacher.calls == 1
    assert all(subtypes.count(subtype) > 0 for subtype in (
        "structure", "field", "exact_relation", "semantic_property",
        "contextual_relation",
    ))
    assert set(artifact["family_budgets"]) == {"context", "delivered"}
    document = artifact["documents"]["aci/D2N002"]
    assert set(document["present_family_budgets"]) == {"context", "delivered"}
    assert document["missing_family_budgets"] == []
    assert all(
        any(action["mode"] == "keep" and action["keep"] for action in decision["actions"])
        for decision in document["decisions"]
    )
    assert set(decisions.values()).issubset(document["controlled_decision_ids"])
    rejection = artifact["rejections"]["records"][0]
    assert {
        "rejection_id", "attempt_hash", "doc_id", "status", "reason",
        "detail_reason", "evidence",
    }.issubset(rejection)
    assert rejection["detail_reason"] == "invalid_question"

    assertions_view = json.loads((tmp_path / "d2n002.assertions.json").read_text())
    qa_pairs_view = json.loads((tmp_path / "d2n002.qa-pairs.json").read_text())
    grouped = assertions_view["documents"]["aci/D2N002"]["assertion_groups"]
    assert grouped["structure"] and grouped["field_content"]
    assert grouped["exact_relation"] and grouped["contextual"]
    decision_rows = qa_pairs_view["documents"]["aci/D2N002"]["decisions"].values()
    assert any(row["qa_pairs"] for row in decision_rows)
    assert any(row["rejections"] for row in decision_rows)

def test_build_qa_utility_artifact_cli_rejects_legacy_aci_detector_artifacts(tmp_path):
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

    assert result.returncode != 0
    assert "legacy or unsupported ACI decision types" in result.stderr
    assert "DEM" in result.stderr
    assert "MISC" in result.stderr
    assert not out_path.exists()
