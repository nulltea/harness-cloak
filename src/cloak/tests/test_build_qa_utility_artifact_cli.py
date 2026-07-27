import json
import os
import subprocess
import sys
from pathlib import Path

import build_qa_utility_artifact as qa_cli
import pytest
from cloak.qa.builder import validate_context_assertions
from cloak.qa.freeze import freeze_ranker_environment, frozen_occurrences_from_arms


TEST_READER_PIN = {
    "model": "deterministic-test-reader",
    "endpoint": "in-process",
    "prompt_version": "qa-reader-test-v1",
    "response_schema": {"type": "answers-array", "version": 1},
    "revision": "test-revision-1",
}


def _write_migration_fixture(tmp_path: Path, *, semantic_change: bool = False):
    policy_action = {
        "action_id": "action-policy", "mode": "level", "fill": "general",
    }
    fixed_action = {
        "action_id": "action-fixed", "mode": "placeholder", "fill": None,
    }
    old_decisions = [
        {"decision_id": "policy-a", "ranker_selectable": True,
         "actions": [policy_action]},
        {"decision_id": "fixed-a", "ranker_selectable": False,
         "actions": [fixed_action]},
    ]
    new_decisions = json.loads(json.dumps(old_decisions))
    new_decisions[0]["actions"][0].update({
        "count": 10, "count_grounding": {"source": "fixture"},
        "authored_level_index": 0,
    })
    if semantic_change:
        new_decisions[0]["actions"][0]["fill"] = "changed-general"
    occurrences = [
        {"occurrence_id": "o-policy", "decision_id": "policy-a", "controlled": True},
        {"occurrence_id": "o-fixed", "decision_id": "fixed-a", "controlled": True},
        {"occurrence_id": "o-null", "decision_id": None, "controlled": False},
    ]
    assertion_rows = [
        ("a-policy", ["o-policy"], "context", "contextual_relation"),
        ("a-fixed", ["o-fixed"], "delivered", "content"),
        ("a-mixed", ["o-policy", "o-fixed", "o-policy"], "delivered", "content"),
        ("a-global", [], "delivered", "structure"),
    ]
    assertions = {
        assertion_id: {
            "assertion_id": assertion_id,
            "doc_id": "d1",
            "status": "accepted",
            "scope": "linked" if occurrence_ids else "global",
            "occurrence_ids": occurrence_ids,
            "family": family,
            "subtype": subtype,
            "group_id": assertion_id,
            "weight": 0.25,
            "scoring_contract": {"kind": "fixture"},
            "evidence": {"validation": {"scores": {
                "original": 1.0, "representative": 0.75, "placeholder": 0.0,
            }}},
        }
        for assertion_id, occurrence_ids, family, subtype in assertion_rows
    }
    old_artifact = {
        "artifact_version": "utility-assertions-v1",
        "artifact_hash": "sha256:old-artifact",
        "environment_hash": "sha256:old-environment",
        "family_budgets": {"context": 0.5, "delivered": 0.5},
        "documents": {"d1": {
            "environment_document_hash": "sha256:old-document",
            "measurement_state": "measured",
            "utility_weight_denominator": 1.0,
            "present_family_budgets": ["context", "delivered"],
            "missing_family_budgets": [],
            "assertion_ids": list(assertions),
            "controlled_decision_ids": ["policy-a", "fixed-a"],
            "occurrence_to_decision": {
                "o-policy": "policy-a", "o-fixed": "fixed-a", "o-null": None,
            },
            "decision_keys": [],
            "occurrences": occurrences,
            "decisions": old_decisions,
            "uncovered_decision_ids": [],
        }},
        "assertions": assertions,
        "rejections": {"summary_by_reason": {}, "records": []},
    }
    new_environment = {
        "artifact_version": "ranker-v2-environment-v2",
        "frozen_environment": {
            "artifact_version": "occurrence-decisions-v2",
            "environment_hash": "sha256:new-environment",
            "documents": {"d1": {
                "environment_document_hash": "sha256:new-document",
                "occurrences": occurrences,
                "decisions": new_decisions,
            }},
        },
    }
    input_path = tmp_path / "input.utility"
    environment_path = tmp_path / "ranker-env.json"
    output_path = tmp_path / "output.utility"
    report_path = tmp_path / "migration-report.json"
    input_path.write_text(json.dumps(old_artifact))
    environment_path.write_text(json.dumps(new_environment))
    return input_path, environment_path, output_path, report_path


def _run_migration(input_path, environment_path, output_path, report_path):
    repo = Path(__file__).resolve().parents[3]
    return subprocess.run(
        [
            sys.executable, str(repo / "scripts/migrate_qa_utility_artifact.py"),
            "--input", str(input_path),
            "--environment", str(environment_path),
            "--out", str(output_path),
            "--report", str(report_path),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def test_migration_cli_rebinds_count_only_environment_with_canonical_routing(tmp_path):
    paths = _write_migration_fixture(tmp_path)
    result = _run_migration(*paths)

    assert result.returncode == 0, result.stderr
    artifact = json.loads(paths[2].read_text())
    report = json.loads(paths[3].read_text())
    assert artifact["artifact_version"] == "utility-assertions-v2"
    assert artifact["documents"]["d1"]["occurrence_to_decision"]["o-null"] is None
    rows = artifact["assertions"]
    assert rows["a-policy"]["policy_dependency_decision_ids"] == ["policy-a"]
    assert rows["a-fixed"]["credit_routing"] == "residual"
    assert rows["a-mixed"]["policy_dependency_decision_ids"] == ["policy-a"]
    assert rows["a-global"]["credit_routing"] == "residual"
    assert report["status"] == "count-only compatible"
    assert report["document_utility_parity"]["identical"] is True
    assert report["document_utility_parity"]["cached_vector_names"] == [
        "original", "placeholder", "representative",
    ]
    assert report["document_utility_parity"]["cached_score_count"] == 12


def test_migration_cli_rejects_semantic_environment_change(tmp_path):
    paths = _write_migration_fixture(tmp_path, semantic_change=True)
    result = _run_migration(*paths)

    assert result.returncode != 0
    assert not paths[2].exists()
    report = json.loads(paths[3].read_text())
    assert report["status"] == "qa_rebuild_required"
    assert report["compatibility"]["verdict"] == "semantic change"


def test_real_v16_migration_uses_task2_pinned_compatibility_reference(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    result = _run_migration(
        repo / "results/qa_v2_aci_full_v16/aci_full.utility",
        repo / "results/ranker_v2/environment/ranker-env.json",
        tmp_path / "aci-full.utility",
        tmp_path / "migration-report.json",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "migration-report.json").read_text())
    assert report["compatibility"]["verdict"] == "count-only compatible"
    assert report["compatibility"]["reference"] == (
        "results/qa_v2_aci_full/ranker-env.json"
    )
    assert report["input_inventory_audit"]["difference_count"] == 26
    assert report["input_inventory_audit"]["unknown_assertion_occurrence_links"] == 0
    assert report["input_inventory_audit"]["stale_joint_anchor_assertions"] == 8
    assert report["document_utility_parity"]["cached_component_count"] == 634
    assert report["document_utility_parity"]["cached_score_count"] == 1_902


def test_real_policy_scoped_utility_artifact_passes_routing_and_denominator_gates():
    repo = Path(__file__).resolve().parents[3]
    artifact_path = repo / "results/ranker_v2/qa/aci-full.utility"
    report_path = repo / "results/ranker_v2/qa/migration-report.json"
    assert artifact_path.exists(), "run migrate_qa_utility_artifact.py first"
    artifact = json.loads(artifact_path.read_text())
    report = json.loads(report_path.read_text())

    assert artifact["artifact_version"] == "utility-assertions-v2"
    assert len(artifact["documents"]) == 67
    assert len(artifact["assertions"]) == 1_357
    assert sum(
        len(document["policy_decision_ids"])
        for document in artifact["documents"].values()
    ) == 705
    assert sum(
        len(document["fixed_decision_ids"])
        for document in artifact["documents"].values()
    ) == 208

    for doc_id, document in artifact["documents"].items():
        policy_ids = set(document["policy_decision_ids"])
        fixed_ids = set(document["fixed_decision_ids"])
        assert policy_ids.isdisjoint(fixed_ids)
        assert "controlled_decision_ids" not in document
        assert all(
            value is None or isinstance(value, str)
            for value in document["occurrence_to_decision"].values()
        )
        covered = set()
        weight_sum = 0.0
        for assertion_id in document["assertion_ids"]:
            assertion = artifact["assertions"][assertion_id]
            assert assertion["doc_id"] == doc_id
            assert set(assertion["occurrence_ids"]).issubset(
                document["occurrence_to_decision"]
            )
            dependencies = set(assertion["policy_dependency_decision_ids"])
            assert dependencies.issubset(policy_ids)
            assert dependencies.isdisjoint(fixed_ids)
            assert assertion["credit_routing"] == (
                "linked" if dependencies else "residual"
            )
            covered.update(dependencies)
            weight_sum += float(assertion["weight"])
        assert document["uncovered_policy_decision_ids"] == [
            decision_id for decision_id in document["policy_decision_ids"]
            if decision_id not in covered
        ]
        expected_weight = sum(
            float(artifact["family_budgets"][family])
            for family in document["present_family_budgets"]
        )
        denominator = float(document["utility_weight_denominator"])
        assert weight_sum / denominator == pytest.approx(
            expected_weight / denominator, abs=1e-12
        )

    assert report["document_utility_parity"]["identical"] is True
    assert report["counts"]["policy_decisions"] == 705
    assert report["counts"]["fixed_decisions"] == 208


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


def _acceptance_reader(questions, context, clauses=None):
    support = {
        "endocrine condition": ("hypothyroidism", "endocrine condition"),
        "thyroid medication": ("synthroid", "thyroid medication"),
        "thyroid test": ("thyroid labs", "thyroid test"),
    }
    normalized_context = context.casefold()

    def supports_procedure_for_relation():
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
            supported = supports_procedure_for_relation()
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
    render = qa_cli._action_renderer(frozen, {"aci/D2N002": source})

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
    render = qa_cli._action_renderer(frozen, {"d1": source})

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
        "relation": "procedure_for",
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
    assert artifact["artifact_version"] == "utility-assertions-v2"
    assert artifact["builder_pin"] == "qa-builder-v2-assertion-compiler-v16"
    assert all(subtypes.count(subtype) > 0 for subtype in (
        "structure", "field", "content", "exact_relation",
        "contextual_relation",
    ))
    # semantic_property probes are disabled (SEMANTIC_PROPERTY_PROBES_DISABLED)
    assert "semantic_property" not in subtypes
    assert set(artifact["family_budgets"]) == {"context", "delivered"}
    document = artifact["documents"]["aci/D2N002"]
    assert set(document["present_family_budgets"]) == {"context", "delivered"}
    assert document["missing_family_budgets"] == []
    assert all(
        any(action["mode"] == "keep" and action["keep"] for action in decision["actions"])
        for decision in document["decisions"]
    )
    assert set(decisions.values()).issubset(document["policy_decision_ids"])
    assert document["fixed_decision_ids"] == []
    assert "controlled_decision_ids" not in document
    assert all(
        "policy_dependency_decision_ids" in row and "credit_routing" in row
        for row in artifact["assertions"].values()
    )
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
    render = qa_cli._action_renderer(frozen, {"aci/D2N002": source})
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
        qa_cli.build_from_files(args, reader=lambda questions, context, clauses=None: [])


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
        [sys.executable, "scripts/build_qa_utility_artifact.py",
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
    assert result.stderr
    assert not out_path.exists()
