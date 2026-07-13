import json
import os
import subprocess
from pathlib import Path

import build_qa_utility_artifact as qa_cli
import pytest
from cloak.train.qa_builder import (
    freeze_ranker_environment,
    frozen_occurrences_from_arms,
    validate_context_assertions,
)


TEST_READER_PIN = {
    "model": "deterministic-test-reader",
    "endpoint": "in-process",
    "prompt_version": "qa-reader-test-v1",
    "response_schema": {"type": "answers-array", "version": 1},
    "revision": "test-revision-1",
}


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
                {"fill": generalization, "mode": "level", "aset": 100},
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
    repeated_condition_start = source.index("Hypothyroidism")
    arm_rows.append({
        "surface": "Hypothyroidism",
        "type": "health-condition",
        "start": repeated_condition_start,
        "end": repeated_condition_start + len("Hypothyroidism"),
        "score": 0.9,
        "action": "placeholder",
        "replacement": "<HEALTH_CONDITION_4>",
        "lattice": ["endocrine condition"],
    })
    arm_rows.sort(key=lambda row: row["start"])
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
    pin = {
        "provider": "injected",
        "model": "deterministic-relation-teacher",
        "base_url": "in-process",
        "prompt_version": "qa-relation-teacher-test-v1",
        "response_schema": {"type": "relations-array", "version": 1},
        "revision": "test-revision-1",
    }

    def __init__(self, proposals):
        self.proposals = proposals
        self.calls = 0
        self.prompts = []

    def propose(self, prompt):
        assert "[S1:" in prompt
        assert "OCCURRENCE INVENTORY" not in prompt
        self.calls += 1
        self.prompts.append(prompt)
        return self.proposals


def _acceptance_reader(questions, context):
    support = {
        "endocrine condition": ("hypothyroidism", "endocrine condition"),
        "thyroid medication": ("synthroid", "thyroid medication"),
        "thyroid test": ("thyroid labs", "thyroid test"),
    }
    normalized_context = context.casefold()

    def supports_treated_with_relation():
        return any(
            f"{condition}{cue}{treatment}" in normalized_context
            for condition in support["endocrine condition"]
            for cue in (" is treated with ", " was treated with ")
            for treatment in support["thyroid medication"]
        )

    answers = []
    for question in questions:
        normalized_question = question.casefold()
        if (
            "treatment category" in normalized_question
            and "diagnosed condition" in normalized_question
        ):
            expected = "thyroid medication"
            supported = supports_treated_with_relation()
        elif "condition category" in normalized_question:
            expected = "endocrine condition"
            supported = any(
                value in normalized_context for value in support[expected]
            )
        elif "procedure or test category" in normalized_question:
            expected = "thyroid test"
            supported = any(
                value in normalized_context for value in support[expected]
            )
        elif "treatment category" in normalized_question:
            expected = "thyroid medication"
            supported = any(
                value in normalized_context for value in support[expected]
            )
        else:
            answers.append("unknown")
            continue
        answers.append(expected if supported else "unknown")
    return answers


_acceptance_reader.pin = dict(TEST_READER_PIN)


def test_acceptance_reader_rejects_broken_representative_rendering():
    assertion = {
        "assertion_id": "condition-category",
        "family": "context",
        "question": "What specific condition category is documented?",
        "accepted_values": ["endocrine condition"],
    }

    accepted, evidence = validate_context_assertions(
        [assertion],
        original_context="The patient is diagnosed with hypothyroidism.",
        representative_context="The patient has an unrelated condition.",
        placeholder_context="The patient is diagnosed with <HEALTH_CONDITION_1>.",
        reader=_acceptance_reader,
    )

    assert accepted == []
    assert evidence["condition-category"]["scores"] == {
        "original": 1.0,
        "representative": 0.0,
        "placeholder": 0.0,
    }


@pytest.mark.parametrize(
    "representative_context",
    [
        "An unrelated condition is treated with thyroid medication.",
        "Endocrine condition was discussed. Thyroid medication remained listed.",
    ],
)
def test_acceptance_reader_rejects_broken_contextual_relation(
    representative_context,
):
    assertion = {
        "assertion_id": "treated-with",
        "family": "context",
        "subtype": "contextual_relation",
        "question": "What treatment category is used for the diagnosed condition?",
        "accepted_values": ["thyroid medication"],
    }

    accepted, evidence = validate_context_assertions(
        [assertion],
        original_context="Hypothyroidism is treated with Synthroid.",
        representative_context=representative_context,
        placeholder_context=(
            "<HEALTH_CONDITION_1> is treated with <DRUG_2>."
        ),
        reader=_acceptance_reader,
    )

    assert accepted == []
    assert evidence["treated-with"]["scores"] == {
        "original": 1.0,
        "representative": 0.0,
        "placeholder": 0.0,
    }


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


def test_action_renderer_preserves_repeated_mixed_case_keep_occurrences():
    source = "Synthroid was continued; SYNTHROID remained listed; synthroid was refilled."
    surfaces = ["Synthroid", "SYNTHROID", "synthroid"]
    spans = []
    arm_rows = []
    search_start = 0
    actions = [
        {"fill": "thyroid medication", "mode": "level", "aset": 100},
        {"fill": None, "mode": "placeholder"},
    ]
    for index, surface in enumerate(surfaces, start=1):
        start = source.index(surface, search_start)
        end = start + len(surface)
        search_start = end
        spans.append({
            "surface": surface,
            "type": "drug",
            "start": start,
            "end": end,
            "actions": actions,
        })
        arm_rows.append({
            "surface": surface,
            "type": "drug",
            "start": start,
            "end": end,
            "action": "placeholder",
            "replacement": f"<DRUG_{index}>",
            "lattice": ["thyroid medication"],
        })
    environment = {
        "corpora": {"clinical": {"d1": {"spans": spans}}},
    }
    arms = {"clinical": {"d1": {"tau_walk": [source, arm_rows]}}}
    frozen = freeze_ranker_environment(
        environment,
        occurrences_by_document=frozen_occurrences_from_arms(arms),
    )
    decision = frozen["documents"]["d1"]["decisions"][0]
    keep = next(action for action in decision["actions"] if action["mode"] == "keep")
    render = qa_cli._action_renderer(
        environment, frozen, arms, "clinical", {"d1": source}
    )

    assert render("d1", {decision["decision_id"]: keep["action_id"]}) == source


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
        "reader_pin": TEST_READER_PIN,
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
        "evidence_start": source.index("Hypothyroidism is treated with Synthroid."),
    }
    teacher = _InjectedRelationTeacher([
        valid_proposal,
        {**valid_proposal, "question": "Malformed question"},
        {**valid_proposal, "relation": "invented_relation"},
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
    assert artifact["teacher_pin"] == teacher.pin
    assert artifact["reader_pin"] == TEST_READER_PIN
    assert artifact["builder_pin"] == "qa-builder-v2-assertion-compiler-v4"
    assert all(subtypes.count(subtype) > 0 for subtype in (
        "structure", "field", "content", "exact_relation", "semantic_property",
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
    rejections = artifact["rejections"]["records"]
    assert len(rejections) >= 2
    required_rejection_fields = {
        "rejection_id", "attempt_hash", "doc_id", "status", "reason",
        "detail_reason", "evidence",
    }
    assert all(required_rejection_fields.issubset(rejection) for rejection in rejections)
    assert {rejection["detail_reason"] for rejection in rejections}.issuperset({
        "invalid_question", "invalid_relation",
    })

    assertions_view = json.loads((tmp_path / "d2n002.assertions.json").read_text())
    qa_pairs_view = json.loads((tmp_path / "d2n002.qa-pairs.json").read_text())
    grouped = assertions_view["documents"]["aci/D2N002"]["assertion_groups"]
    assert grouped["structure"] and grouped["field_content"]
    assert grouped["exact_relation"] and grouped["contextual"]
    decision_rows = qa_pairs_view["documents"]["aci/D2N002"]["decisions"].values()
    assert any(row["qa_pairs"] for row in decision_rows)
    assert any(row["rejections"] for row in decision_rows)

    relation = next(
        row for row in artifact["assertions"].values()
        if row["subtype"] == "contextual_relation"
    )
    assert relation["evidence"]["proposal_hash"] == qa_cli._hash(valid_proposal)
    assert relation["evidence"]["prompt_hash"] == qa_cli._hash(teacher.prompts[0])
    assert teacher.prompts[0] not in output.read_text()
    render = qa_cli._action_renderer(
        environment, frozen, {"clinical": arms["clinical"]}, "clinical",
        {"aci/D2N002": source},
    )
    representative = render(
        "aci/D2N002",
        relation["expected_action_support"]["joint_anchor_action_vector"],
    )
    occurrences_by_id = {
        row["occurrence_id"]: row for row in frozen_document["occurrences"]
    }
    assert all(
        occurrences_by_id[occurrence_id]["surface"].casefold()
        not in representative.casefold()
        for occurrence_id in relation["occurrence_ids"]
    )


def test_default_reader_exposes_complete_structured_pin():
    pin = qa_cli.read_context_batch.pin

    assert set(pin).issuperset({
        "model", "endpoint", "prompt_version", "response_schema", "revision",
    })
    assert all(pin[key] for key in (
        "model", "endpoint", "prompt_version", "response_schema", "revision",
    ))


@pytest.mark.parametrize("reader_pin", [None, "reader-v1", {}, {"model": "reader"}])
def test_build_from_files_rejects_unstructured_reader_pin(
    tmp_path, monkeypatch, reader_pin,
):
    source, reference, environment, arms = _d2n002_fixture()
    environment_path = tmp_path / "ranker-env.json"
    arms_path = tmp_path / "arms.json"
    manifest_path = tmp_path / "manifest.json"
    environment_path.write_text(json.dumps(environment))
    arms_path.write_text(json.dumps(arms))
    manifest_path.write_text(json.dumps({
        "family_budgets": {"context": 0.6, "delivered": 0.4},
        "structural_cap": 0.1,
        "min_contextual_relation_assertions": 0,
        "reader_pin": reader_pin,
    }))
    monkeypatch.setattr(
        qa_cli,
        "_source_rows",
        lambda corpus, doc_ids: {
            "aci/D2N002": {"id": "aci/D2N002", "text": source, "gold_ref": reference}
        },
    )
    args = qa_cli.parse_args([
        "--env", str(environment_path),
        "--arms", str(arms_path),
        "--doc-id", "aci/D2N002",
        "--threshold-manifest", str(manifest_path),
        "--out", str(tmp_path / "artifact.json"),
    ])

    with pytest.raises(ValueError, match="reader_pin"):
        qa_cli.build_from_files(args, reader=_acceptance_reader)


def test_build_from_files_rejects_unpinned_injected_reader(tmp_path, monkeypatch):
    source, reference, environment, arms = _d2n002_fixture()
    environment_path = tmp_path / "ranker-env.json"
    arms_path = tmp_path / "arms.json"
    manifest_path = tmp_path / "manifest.json"
    environment_path.write_text(json.dumps(environment))
    arms_path.write_text(json.dumps(arms))
    manifest_path.write_text(json.dumps({
        "family_budgets": {"context": 0.6, "delivered": 0.4},
        "structural_cap": 0.1,
        "min_contextual_relation_assertions": 0,
        "reader_pin": TEST_READER_PIN,
    }))
    monkeypatch.setattr(
        qa_cli,
        "_source_rows",
        lambda corpus, doc_ids: {
            "aci/D2N002": {"id": "aci/D2N002", "text": source, "gold_ref": reference}
        },
    )
    args = qa_cli.parse_args([
        "--env", str(environment_path),
        "--arms", str(arms_path),
        "--doc-id", "aci/D2N002",
        "--threshold-manifest", str(manifest_path),
        "--out", str(tmp_path / "artifact.json"),
    ])

    with pytest.raises(ValueError, match="injected reader.*pin"):
        qa_cli.build_from_files(args, reader=lambda questions, context: [])


def test_build_from_files_rejects_relation_teacher_without_explicit_pin(
    tmp_path, monkeypatch,
):
    source, reference, environment, arms = _d2n002_fixture()
    environment_path = tmp_path / "ranker-env.json"
    arms_path = tmp_path / "arms.json"
    manifest_path = tmp_path / "manifest.json"
    environment_path.write_text(json.dumps(environment))
    arms_path.write_text(json.dumps(arms))
    manifest_path.write_text(json.dumps({
        "family_budgets": {"context": 0.6, "delivered": 0.4},
        "structural_cap": 0.1,
        "min_context_assertions": 0,
        "reader_pin": TEST_READER_PIN,
    }))
    monkeypatch.setattr(
        qa_cli,
        "_source_rows",
        lambda corpus, doc_ids: {
            "aci/D2N002": {"id": "aci/D2N002", "text": source, "gold_ref": reference}
        },
    )
    args = qa_cli.parse_args([
        "--env", str(environment_path),
        "--arms", str(arms_path),
        "--doc-id", "aci/D2N002",
        "--threshold-manifest", str(manifest_path),
        "--out", str(tmp_path / "artifact.json"),
    ])

    class TeacherWithoutPin:
        def propose(self, prompt):
            return []

    with pytest.raises(ValueError, match="relation teacher.*pin"):
        qa_cli.build_from_files(
            args, relation_teacher=TeacherWithoutPin(), reader=_acceptance_reader
        )


def test_build_qa_utility_artifact_cli_rejects_legacy_aci_detector_artifacts(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    out_path = tmp_path / "utility.json"
    manifest_path.write_text(json.dumps({
        "family_budgets": {"context": 0.6, "delivered": 0.4},
        "min_context_assertions": 0,
        "task_pin": "aci-v1",
        "reader_pin": qa_cli.read_context_batch.pin,
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
