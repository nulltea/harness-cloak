import json

import pytest

import cloak.train.roundtrip as roundtrip
import cloak.train.qa_builder as qa_builder
from cloak.train.qa_builder import (
    AciTaskAdapter,
    BatchedContextReader,
    artifact_views,
    assign_static_weights,
    build_utility_artifact,
    build_joint_representative_anchor,
    compile_relational_assertions,
    freeze_ranker_environment,
    package_utility_artifact,
    relation_teacher_prompt,
    score_utility,
    validate_context_assertions,
)


def test_artifact_views_project_complete_inspectable_records_without_mutation():
    actions = [
        {
            "action_id": "action-general",
            "mode": "level",
            "legal": True,
            "fill": "thyroid medication",
            "entails": ["thyroid medication"],
        },
        {
            "action_id": "action-keep",
            "mode": "keep",
            "keep": True,
            "legal": True,
            "fill": "synthroid",
            "entails": ["synthroid"],
        },
    ]
    validation = {
        "verdict": "accepted",
        "scores": {"original": 1.0, "representative": 1.0, "placeholder": 0.0},
    }
    delivered_rows = {
        "component-structure": {
            "assertion_id": "component-structure",
            "doc_id": "d1",
            "family": "delivered",
            "subtype": "structure",
            "group_id": "group-structure",
            "occurrence_ids": [],
            "evidence": {"authority": "human_reference"},
            "scoring_contract": {"kind": "required_sections", "sections": ["PLAN"]},
        },
        "component-field": {
            "assertion_id": "component-field",
            "doc_id": "d1",
            "family": "delivered",
            "subtype": "field",
            "group_id": "group-field",
            "occurrence_ids": [],
            "evidence": {"authority": "human_reference"},
            "scoring_contract": {"kind": "field_value", "value": "stable"},
        },
        "component-content": {
            "assertion_id": "component-content",
            "doc_id": "d1",
            "family": "delivered",
            "subtype": "content",
            "group_id": "group-content",
            "occurrence_ids": ["occurrence-drug"],
            "evidence": {"authority": "human_reference"},
            "scoring_contract": {"kind": "contains", "value": "Synthroid"},
        },
        "component-exact": {
            "assertion_id": "component-exact",
            "doc_id": "d1",
            "family": "delivered",
            "subtype": "exact_relation",
            "group_id": "group-exact",
            "occurrence_ids": ["occurrence-drug"],
            "evidence": {"authority": "human_reference"},
            "scoring_contract": {"kind": "exact_relation", "treatment": "Synthroid"},
        },
    }
    semantic = {
        "assertion_id": "component-semantic",
        "doc_id": "d1",
        "family": "context",
        "subtype": "semantic_property",
        "group_id": "group-semantic",
        "occurrence_ids": ["occurrence-drug"],
        "question": "What category is the treatment?",
        "accepted_values": ["thyroid medication"],
        "decision_requirements": {"decision-drug": "thyroid medication"},
        "expected_action_support": {
            "joint_anchor_action_vector": {"decision-drug": "action-general"},
            "joint_anchor_hash": "anchor-semantic",
            "property_level": {"decision-drug": "thyroid medication"},
        },
        "evidence": {
            "supporting_action_ids": ["action-general"],
            "validation": validation,
        },
    }
    contextual = {
        **semantic,
        "assertion_id": "component-contextual",
        "subtype": "contextual_relation",
        "group_id": "group-contextual",
        "relation": "treated_with",
        "question": "What treatment category is used?",
    }
    rejection = {
        "rejection_id": "rejection-1",
        "attempt_hash": "attempt-1",
        "doc_id": "d1",
        "status": "rejected",
        "reason": "invalid",
        "detail_reason": "invalid_question",
        "evidence": {"decision_id": "decision-drug"},
    }
    artifact = {
        "artifact_version": "utility-assertions-v1",
        "artifact_hash": "artifact-hash",
        "family_budgets": {"context": 0.6, "delivered": 0.4},
        "documents": {
            "d1": {
                "measurement_state": "measured",
                "occurrence_to_decision": {"occurrence-drug": "decision-drug"},
                "decisions": [{
                    "decision_id": "decision-drug",
                    "runtime_type": "drug",
                    "canonical_key": "synthroid",
                    "occurrence_ids": ["occurrence-drug"],
                    "actions": actions,
                }],
            },
        },
        "assertions": {**delivered_rows, semantic["assertion_id"]: semantic,
                       contextual["assertion_id"]: contextual},
        "rejections": {
            "summary_by_reason": {"invalid": 1},
            "records": [rejection],
        },
    }
    before = json.dumps(artifact, sort_keys=True)

    assertions_view, qa_pairs_view = artifact_views(artifact)

    grouped = assertions_view["documents"]["d1"]["assertion_groups"]
    assert [row["assertion_id"] for row in grouped["structure"]] == [
        "component-structure"
    ]
    assert [row["assertion_id"] for row in grouped["field_content"]] == [
        "component-content", "component-field"
    ]
    assert grouped["exact_relation"][0]["scoring_contract"]["kind"] == (
        "exact_relation"
    )
    assert {row["subtype"] for row in grouped["contextual"]} == {
        "semantic_property", "contextual_relation",
    }
    assert all(row["evidence"] for rows in grouped.values() for row in rows)

    decision = qa_pairs_view["documents"]["d1"]["decisions"]["decision-drug"]
    assert decision["occurrence_ids"] == ["occurrence-drug"]
    assert decision["legal_actions"] == actions
    assert [row["assertion_id"] for row in decision["qa_pairs"]] == [
        "component-contextual", "component-semantic",
    ]
    assert decision["qa_pairs"][1]["accepted_values"] == ["thyroid medication"]
    assert decision["qa_pairs"][1]["evidence"]["supporting_action_ids"] == [
        "action-general"
    ]
    assert decision["qa_pairs"][1]["evidence"]["validation"] == validation
    assert decision["rejections"] == [rejection]
    assert qa_pairs_view["documents"]["d1"]["unassigned_rejections"] == []
    assert assertions_view["rejections"]["records"] == [rejection]
    assert json.dumps(artifact, sort_keys=True) == before


def test_static_weights_keep_family_budgets_and_fixed_denominator():
    assertions = [
        {"assertion_id": "c1", "family": "context", "group_id": "condition:hypothyroid"},
        {"assertion_id": "c2", "family": "context", "group_id": "condition:hypothyroid"},
        {"assertion_id": "c3", "family": "context", "group_id": "condition:arthritis"},
        {"assertion_id": "d1", "family": "delivered", "group_id": "condition:hypothyroid"},
    ]

    weighted, document_state = assign_static_weights(
        assertions, {"context": 0.6, "delivered": 0.4}
    )

    assert {row["assertion_id"]: row["weight"] for row in weighted} == pytest.approx({
        "c1": 0.15,
        "c2": 0.15,
        "c3": 0.30,
        "d1": 0.40,
    })
    assert document_state == {
        "utility_weight_denominator": 1.0,
        "present_family_budgets": ["context", "delivered"],
        "missing_family_budgets": [],
    }


def test_missing_family_does_not_renormalize_surviving_weights():
    weighted, document_state = assign_static_weights(
        [{"assertion_id": "d1", "family": "delivered", "group_id": "schema:age"}],
        {"context": 0.6, "delivered": 0.4},
    )

    assert weighted[0]["weight"] == pytest.approx(0.4)
    assert document_state["utility_weight_denominator"] == pytest.approx(1.0)
    assert document_state["missing_family_budgets"] == ["context"]


def test_joint_anchor_uses_coarsest_entailing_actions_and_keep_elsewhere():
    decisions = [
        {
            "decision_id": "condition",
            "actions": [
                {"action_id": "keep-c", "mode": "keep", "legal": True, "entails": ["exact"]},
                {"action_id": "thyroid", "mode": "level", "legal": True,
                 "entails": ["thyroid", "endocrine"]},
                {"action_id": "endocrine", "mode": "level", "legal": True,
                 "entails": ["endocrine"]},
                {"action_id": "placeholder-c", "mode": "placeholder", "legal": True,
                 "entails": []},
            ],
        },
        {
            "decision_id": "drug",
            "actions": [
                {"action_id": "keep-d", "mode": "keep", "legal": True, "entails": ["exact"]},
                {"action_id": "thyroid-med", "mode": "level", "legal": True,
                 "entails": ["thyroid-treatment"]},
                {"action_id": "medication", "mode": "level", "legal": True,
                 "entails": ["medication"]},
            ],
        },
        {
            "decision_id": "age",
            "actions": [
                {"action_id": "keep-a", "mode": "keep", "legal": True, "entails": ["exact"]},
                {"action_id": "adult", "mode": "level", "legal": True, "entails": ["adult"]},
            ],
        },
    ]
    assertion = {
        "assertion_id": "a1",
        "occurrence_ids": ["o-condition", "o-drug"],
        "decision_requirements": {
            "condition": "endocrine",
            "drug": "thyroid-treatment",
        },
    }

    anchor = build_joint_representative_anchor(assertion, decisions)

    assert anchor["action_vector"] == {
        "condition": "endocrine",
        "drug": "thyroid-med",
        "age": "keep-a",
    }
    assert anchor["action_vector_hash"].startswith("sha256:")


def test_joint_anchor_rejects_missing_entailing_action():
    assertion = {
        "assertion_id": "a1",
        "decision_requirements": {"condition": "endocrine"},
    }
    decisions = [{
        "decision_id": "condition",
        "actions": [{"action_id": "placeholder", "mode": "placeholder", "legal": True,
                     "entails": []}],
    }]

    with pytest.raises(ValueError, match="no legal generalization"):
        build_joint_representative_anchor(assertion, decisions)


def test_context_validation_requires_original_and_generalization_but_not_placeholder():
    assertions = [{
        "assertion_id": "a1",
        "family": "context",
        "question": "What category of condition is documented?",
        "accepted_values": ["endocrine condition"],
    }]
    calls = []

    def reader(questions, context):
        calls.append((list(questions), context))
        answers = {
            "original": ["endocrine condition"],
            "generalized": ["endocrine condition"],
            "placeholder": ["NONE"],
        }
        return answers[context]

    accepted, evidence = validate_context_assertions(
        assertions,
        original_context="original",
        representative_context="generalized",
        placeholder_context="placeholder",
        reader=reader,
    )

    assert [row["assertion_id"] for row in accepted] == ["a1"]
    assert evidence["a1"]["verdict"] == "accepted"
    assert len(calls) == 3


def test_context_validation_rejects_unstable_reader_after_deterministic_option_permutations():
    assertions = [{
        "assertion_id": "a1",
        "family": "context",
        "question": "Which category is documented?",
        "options": ["endocrine", "musculoskeletal", "respiratory"],
        "accepted_values": ["endocrine"],
    }]
    calls = []

    def reader(questions, context):
        calls.append((list(questions), context))
        trial = (len(calls) - 1) // 3
        if context == "placeholder":
            return ["NONE"]
        if context == "generalized" and trial == 1:
            return ["NONE"]
        return ["endocrine"]

    accepted, evidence = validate_context_assertions(
        assertions,
        original_context="original",
        representative_context="generalized",
        placeholder_context="placeholder",
        reader=reader,
        stability_repetitions=1,
        option_permutations=2,
        stability_threshold=1.0,
    )

    assert accepted == []
    assert evidence["a1"]["verdict"] == "unstable"
    assert evidence["a1"]["stability"]["passing_fraction"] == pytest.approx(0.5)
    assert evidence["a1"]["stability"]["option_permutations"] == 2
    assert calls[0][0] != calls[3][0]


def test_builder_records_unstable_context_reader_without_accepting_assertion():
    frozen = {
        "environment_hash": "env-v1",
        "documents": {"d1": {
            "occurrences": [{"occurrence_id": "o1", "decision_id": "dec1"}],
            "decisions": [{
                "decision_id": "dec1",
                "actions": [
                    {"action_id": "keep", "mode": "keep", "legal": True},
                    {"action_id": "general", "mode": "level", "legal": True,
                     "entails": ["endocrine"]},
                    {"action_id": "placeholder", "mode": "placeholder", "legal": True},
                ],
            }],
        }},
    }

    class Adapter:
        def deterministic_candidates(self, doc_id, document, environment_document):
            return [{
                "family": "context", "scope": "linked", "subtype": "semantic_property",
                "occurrence_ids": ["o1"], "group_id": "condition:category",
                "question": "What category?", "accepted_values": ["endocrine"],
                "decision_requirements": {"dec1": "endocrine"},
            }]

    calls = []

    def reader(questions, context):
        calls.append(context)
        if context == "placeholder":
            return ["NONE"]
        if context == "generalized" and len(calls) > 4:
            return ["NONE"]
        return ["endocrine"]

    artifact = build_utility_artifact(
        frozen,
        Adapter(),
        {"d1": "original"},
        threshold_manifest={
            "family_budgets": {"context": 0.6, "delivered": 0.4},
            "reader_stability_repetitions": 2,
            "reader_option_permutations": 1,
            "reader_stability_threshold": 1.0,
        },
        pins={"gate_manifest_hash": "gate-v1"},
        reader=reader,
        render_action_vector=lambda doc_id, vector: (
            "placeholder" if vector["dec1"] == "placeholder" else "generalized"
        ),
    )

    assert artifact["assertions"] == {}
    assert artifact["rejections"]["summary_by_reason"] == {"unstable": 1}
    rejection = artifact["rejections"]["records"][0]
    assert rejection["reason"] == "unstable"
    assert rejection["doc_id"] == "d1"
    assert rejection["attempt_hash"].startswith("sha256:")
    assert rejection["evidence"]["validation"]["verdict"] == "unstable"
    assert rejection["evidence"]["joint_anchor_hash"].startswith("sha256:")


def test_builder_preserves_deterministic_not_generated_rejection_record():
    frozen = {
        "environment_hash": "env-v1",
        "documents": {"d1": {"occurrences": [], "decisions": []}},
    }
    deterministic_rejection = {
        "status": "rejected",
        "reason": "not_generated",
        "detail_reason": "no_safe_contextual_locator",
        "rejection_id": "sha256:deterministic",
        "evidence": {"source": "deterministic_template", "decision_id": "dec1"},
    }

    class Adapter:
        def deterministic_candidates(self, doc_id, document, environment_document):
            return [deterministic_rejection]

    artifact = build_utility_artifact(
        frozen,
        Adapter(),
        {"d1": "source"},
        threshold_manifest={
            "family_budgets": {"context": 0.6, "delivered": 0.4},
        },
        pins={"gate_manifest_hash": "gate-v1"},
        reader=lambda questions, context: [],
        render_action_vector=lambda doc_id, vector: "unused",
    )

    assert artifact["assertions"] == {}
    assert artifact["rejections"]["summary_by_reason"] == {"not_generated": 1}
    record = artifact["rejections"]["records"][0]
    assert record["rejection_id"] == "sha256:deterministic"
    assert record["doc_id"] == "d1"
    assert record["attempt_hash"].startswith("sha256:")


def test_builder_preserves_every_teacher_rejection_with_stable_summary():
    environment = _relation_environment()
    frozen = {
        "environment_hash": "env-v1",
        "documents": {"d1": environment},
    }
    source = "Hypothyroidism is treated with Synthroid."
    leaking = {
        "relation": "treated_with",
        "argument_occurrence_ids": ["o-condition", "o-drug"],
        "support_properties": {
            "o-condition": "an endocrine condition",
            "o-drug": "a thyroid medication",
        },
        "answer_occurrence_id": "o-drug",
        "answer_property": "a thyroid medication",
        "question": "Which treatment is used for hypothyroidism?",
        "evidence_quote": source,
    }

    class Adapter:
        def deterministic_candidates(self, doc_id, document, environment_document):
            return []

        def compile_relations(self, doc_id, document, environment_document, proposals):
            return compile_relational_assertions(
                doc_id, document, environment_document, proposals
            )

    class Teacher:
        def __init__(self):
            self.calls = 0

        def propose(self, prompt):
            self.calls += 1
            return [leaking, leaking]

    teacher = Teacher()
    artifact = build_utility_artifact(
        frozen,
        Adapter(),
        {"d1": source},
        threshold_manifest={
            "family_budgets": {"context": 0.6, "delivered": 0.4},
            "min_context_assertions": 1,
        },
        pins={"gate_manifest_hash": "gate-v1"},
        reader=lambda questions, context: [],
        render_action_vector=lambda doc_id, vector: "unused",
        relation_teacher=teacher,
    )

    records = artifact["rejections"]["records"]
    assert teacher.calls == 1
    assert artifact["rejections"]["summary_by_reason"] == {"leakage": 2}
    assert len(records) == 2
    assert len({row["rejection_id"] for row in records}) == 2
    assert len({row["attempt_hash"] for row in records}) == 2
    assert {row["doc_id"] for row in records} == {"d1"}
    assert {row["detail_reason"] for row in records} == {"protected_locator"}
    assert all(row["reason"] == "leakage" for row in records)
    assert all(row["proposal_hash"].startswith("sha256:") for row in records)


def test_runtime_scores_context_assertions_in_one_reader_batch():
    artifact = {
        "documents": {"d1": {"utility_weight_denominator": 1.0}},
        "assertions": {
            "c1": {"assertion_id": "c1", "doc_id": "d1", "family": "context",
                   "question": "Q1", "accepted_values": ["endocrine"], "weight": 0.3},
            "c2": {"assertion_id": "c2", "doc_id": "d1", "family": "context",
                   "question": "Q2", "accepted_values": ["arthritis"], "weight": 0.3},
            "d1": {"assertion_id": "d1", "doc_id": "d1", "family": "delivered",
                   "scoring_contract": {"kind": "contains", "value": "kidney transplant"},
                   "weight": 0.4},
        },
    }
    calls = []

    def reader(questions, context):
        calls.append((list(questions), context))
        return ["endocrine", "arthritis"]

    result = score_utility(
        artifact,
        "d1",
        doc_p="generalized document",
        out_final="History includes kidney transplant.",
        reader=reader,
    )

    assert len(calls) == 1
    assert result["component_scores"] == {"c1": 1.0, "c2": 1.0, "d1": 1.0}
    assert result["utility"] == pytest.approx(1.0)


def test_batched_context_reader_uses_one_model_request_for_all_questions():
    class Client:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt):
            self.prompts.append(prompt)
            return '{"answers":["endocrine condition","thyroid medication"]}'

    client = Client()
    reader = BatchedContextReader(client=client)

    answers = reader(["What condition category?", "What treatment category?"], "note")

    assert answers == ["endocrine condition", "thyroid medication"]
    assert len(client.prompts) == 1
    assert "What condition category?" in client.prompts[0]
    assert "What treatment category?" in client.prompts[0]


def test_roundtrip_utility_artifact_scores_doc_p_and_out_final(monkeypatch):
    class Remote:
        def generate(self, prompt):
            return "REMOTE OUT_P"

    calls = []

    def fake_score(artifact, doc_id, *, doc_p, out_final, reader):
        calls.append((artifact, doc_id, doc_p, out_final, reader))
        return {"component_scores": {"c1": 0.75}, "utility": 0.75}

    monkeypatch.setattr(roundtrip, "_remote", lambda: Remote())
    monkeypatch.setattr(roundtrip, "invert", lambda out_p, R: ("DELIVERED OUT_FINAL", None))
    monkeypatch.setattr(roundtrip, "score_utility", fake_score, raising=False)

    artifact = {"documents": {"d1": {}}, "assertions": {}}
    result = roundtrip.roundtrip_batch([{
        "corpus": "clinical",
        "doc_id": "d1",
        "doc_p": "ROLLOUT DOC_P",
        "R": [],
        "probes": [],
        "utility_artifact": artifact,
    }], workers=1)[0]

    assert result["recall"] == pytest.approx(0.75)
    assert result["component_scores"] == {"c1": 0.75}
    assert calls[0][1:4] == ("d1", "ROLLOUT DOC_P", "DELIVERED OUT_FINAL")


def test_compile_artifact_assigns_stable_ids_weights_and_uncovered_decisions():
    environment = {
        "environment_hash": "env-hash",
        "documents": {
            "d1": {
                "occurrences": [
                    {"occurrence_id": "o1", "decision_id": "dec1"},
                    {"occurrence_id": "o2", "decision_id": "dec2"},
                ],
                "decisions": [
                    {"decision_id": "dec1", "controlled": True},
                    {"decision_id": "dec2", "controlled": True},
                ],
            }
        },
    }
    candidates = {
        "d1": [
            {"family": "context", "scope": "linked", "subtype": "semantic_property",
             "occurrence_ids": ["o1"], "group_id": "hypothyroid:category",
             "question": "What category?", "accepted_values": ["endocrine"]},
            {"family": "delivered", "scope": "global", "subtype": "content",
             "occurrence_ids": [], "group_id": "document:age",
             "scoring_contract": {"kind": "contains", "value": "62-year-old"}},
        ]
    }

    first = package_utility_artifact(
        environment,
        candidates,
        family_budgets={"context": 0.6, "delivered": 0.4},
        pins={"task_pin": "task-v1", "reader_pin": "reader-v1"},
    )
    second = package_utility_artifact(
        environment,
        candidates,
        family_budgets={"context": 0.6, "delivered": 0.4},
        pins={"task_pin": "task-v1", "reader_pin": "reader-v1"},
    )

    assert first == second
    assert first["artifact_hash"].startswith("sha256:")
    assert len(first["assertions"]) == 2
    assert first["documents"]["d1"]["uncovered_decision_ids"] == ["dec2"]
    assert first["documents"]["d1"]["occurrence_to_decision"] == {
        "o1": "dec1", "o2": "dec2"
    }
    assert first["documents"]["d1"]["utility_weight_denominator"] == pytest.approx(1.0)


def test_compile_artifact_rejects_invalid_scope_links():
    environment = {
        "environment_hash": "env-hash",
        "documents": {"d1": {"occurrences": [], "decisions": []}},
    }

    with pytest.raises(ValueError, match="global assertion must not link occurrences"):
        package_utility_artifact(
            environment,
            {"d1": [{"family": "delivered", "scope": "global",
                      "occurrence_ids": ["missing"], "group_id": "g"}]},
            family_budgets={"context": 0.6, "delivered": 0.4},
            pins={},
        )


def test_freeze_ranker_environment_maps_repeated_occurrences_to_one_decision():
    ranker_env = {
        "corpora": {
            "clinical": {
                "d1": {
                    "spans": [
                        {"surface": "Synthroid", "type": "drug", "start": 10, "end": 19,
                         "actions": [
                             {"fill": "a thyroid medication", "mode": "level"},
                             {"fill": "Synthroid", "mode": "level", "keep": True},
                             {"fill": None, "mode": "placeholder"},
                         ]},
                        {"surface": "Synthroid", "type": "drug", "start": 40, "end": 49,
                         "actions": [
                             {"fill": "a thyroid medication", "mode": "level"},
                             {"fill": "Synthroid", "mode": "level", "keep": True},
                             {"fill": None, "mode": "placeholder"},
                         ]},
                    ]
                }
            }
        }
    }

    frozen = freeze_ranker_environment(ranker_env)
    document = frozen["documents"]["d1"]

    assert frozen["environment_hash"].startswith("sha256:")
    assert len(document["occurrences"]) == 2
    assert len(document["decisions"]) == 1
    assert {row["decision_id"] for row in document["occurrences"]} == {
        document["decisions"][0]["decision_id"]
    }
    assert [row["mode"] for row in document["decisions"][0]["actions"]] == [
        "level", "keep", "placeholder"
    ]


def test_freeze_ranker_environment_synthesizes_missing_keep_into_frozen_menu():
    ranker_env = {
        "corpora": {"clinical": {"d1": {"spans": [{
            "surface": "Synthroid",
            "type": "drug",
            "start": 10,
            "end": 19,
            "actions": [
                {"fill": "a thyroid medication", "mode": "level"},
                {"fill": None, "mode": "placeholder"},
            ],
        }]}}},
    }

    first = freeze_ranker_environment(ranker_env)
    second = freeze_ranker_environment(ranker_env)
    decision = first["documents"]["d1"]["decisions"][0]
    keep_actions = [row for row in decision["actions"] if row["mode"] == "keep"]

    assert keep_actions == [{
        "fill": "Synthroid",
        "mode": "keep",
        "keep": True,
        "legal": True,
        "entails": ["synthroid"],
        "action_id": keep_actions[0]["action_id"],
    }]
    assert keep_actions[0]["action_id"].startswith("sha256:")
    assert first == second


def test_freeze_ranker_environment_uses_all_frozen_arm_occurrences():
    ranker_env = {
        "corpora": {"clinical": {"d1": {"spans": [{
            "surface": "Synthroid", "type": "drug", "start": 40, "end": 49,
            "actions": [
                {"fill": "a thyroid medication", "mode": "level"},
                {"fill": "Synthroid", "mode": "level", "keep": True},
                {"fill": None, "mode": "placeholder"},
            ],
        }]}}},
    }
    arm_occurrences = {"d1": [
        {"surface": "Synthroid", "type": "drug", "start": 10, "end": 19,
         "lattice": ["a thyroid medication"], "score": 0.8},
        {"surface": "Synthroid", "type": "drug", "start": 40, "end": 49,
         "lattice": ["a thyroid medication"], "score": 0.9},
    ]}

    frozen = freeze_ranker_environment(
        ranker_env, occurrences_by_document=arm_occurrences
    )
    document = frozen["documents"]["d1"]

    assert [(row["start"], row["end"]) for row in document["occurrences"]] == [
        (10, 19), (40, 49)
    ]
    assert len(document["decisions"]) == 1
    assert document["decisions"][0]["occurrence_ids"] == [
        row["occurrence_id"] for row in document["occurrences"]
    ]


def test_high_level_builder_calls_teacher_once_then_compiles_and_validates():
    frozen = {
        "environment_hash": "env-v1",
        "documents": {"d1": {
            "occurrences": [{"occurrence_id": "o1", "decision_id": "dec1"}],
            "decisions": [{
                "decision_id": "dec1", "controlled": True,
                "actions": [
                    {"action_id": "keep", "mode": "keep", "legal": True,
                     "entails": ["exact"]},
                    {"action_id": "endocrine", "mode": "level", "legal": True,
                     "entails": ["endocrine"]},
                    {"action_id": "placeholder", "mode": "placeholder", "legal": True,
                     "entails": []},
                ],
            }],
        }},
    }

    class Adapter:
        def deterministic_candidates(self, doc_id, document, environment_document):
            return [{
                "family": "delivered", "scope": "global", "subtype": "content",
                "occurrence_ids": [], "group_id": "document:condition",
                "scoring_contract": {"kind": "contains", "value": "hypothyroidism"},
            }]

        def compile_relations(self, doc_id, document, environment_document, proposals):
            assert proposals == [{"raw": "proposal"}]
            return ([{
                "family": "context", "scope": "linked", "subtype": "contextual_relation",
                "occurrence_ids": ["o1"], "group_id": "condition:category",
                "question": "What category?", "accepted_values": ["endocrine"],
                "decision_requirements": {"dec1": "endocrine"},
            }], [])

    class Teacher:
        def __init__(self):
            self.prompts = []

        def propose(self, prompt):
            self.prompts.append(prompt)
            return [{"raw": "proposal"}]

    teacher = Teacher()

    def render(doc_id, action_vector):
        return "placeholder" if action_vector["dec1"] == "placeholder" else "generalized"

    def reader(questions, context):
        return ["NONE" if context == "placeholder" else "endocrine"] * len(questions)

    artifact = build_utility_artifact(
        frozen,
        Adapter(),
        {"d1": "original"},
        threshold_manifest={
            "family_budgets": {"context": 0.6, "delivered": 0.4},
            "min_context_assertions": 1,
            "reader_threshold": 1.0,
        },
        pins={"gate_manifest_hash": "gate-v1"},
        reader=reader,
        render_action_vector=render,
        relation_teacher=teacher,
    )

    assert len(teacher.prompts) == 1
    assert "d1" in teacher.prompts[0]
    assert artifact["documents"]["d1"]["measurement_state"] == "measured"
    context = next(row for row in artifact["assertions"].values()
                   if row["family"] == "context")
    assert context["subtype"] == "contextual_relation"
    assert context["expected_action_support"]["joint_anchor_hash"].startswith("sha256:")
    assert context["expected_action_support"]["property_level"] == {"dec1": "endocrine"}
    assert "property_levels" not in context["expected_action_support"]
    assert context["evidence"]["validation"]["scores"] == {
        "original": 1.0,
        "representative": 1.0,
        "placeholder": 0.0,
    }


def test_openrouter_relation_teacher_uses_pinned_nemotron(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, model, **kwargs):
            captured.update({"model": model, **kwargs})

        def generate(self, prompt):
            captured["prompt"] = prompt
            return '{"relations": [{"relation": "treated_with"}]}'

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setenv("CLOAK_LLM_CACHE", "/tmp/qa-builder-test-cache")
    monkeypatch.setattr("cloak.llm.LLMClient", FakeClient)

    teacher = qa_builder.OpenRouterRelationTeacher()
    relations = teacher.propose("document prompt")

    assert captured["model"] == "nvidia/nemotron-3-super-120b-a12b:free"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["api_key"] == "secret"
    assert captured["response_format"] == {"type": "json_object"}
    assert "max_tokens" not in captured
    assert relations == [{"relation": "treated_with"}]


def test_openrouter_relation_teacher_requires_content_addressed_cache(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.delenv("CLOAK_LLM_CACHE", raising=False)

    with pytest.raises(ValueError, match="CLOAK_LLM_CACHE"):
        qa_builder.OpenRouterRelationTeacher()


def test_aci_adapter_rejects_legacy_coarse_decision_types():
    environment = {
        "documents": {
            "aci/D2N002": {
                "decisions": [
                    {"decision_id": "bad-dem", "runtime_type": "DEM"},
                    {"decision_id": "bad-misc", "runtime_type": "MISC"},
                ]
            }
        }
    }

    with pytest.raises(ValueError, match="legacy or unsupported ACI decision types.*DEM.*MISC"):
        qa_builder.AciTaskAdapter({}).validate_environment(environment)


def _relation_environment():
    return {
        "occurrences": [
            {"occurrence_id": "o-condition", "decision_id": "d-condition",
             "surface": "hypothyroidism", "runtime_type": "health-condition"},
            {"occurrence_id": "o-drug", "decision_id": "d-drug",
             "surface": "Synthroid", "runtime_type": "drug"},
        ],
        "decisions": [
            {"decision_id": "d-condition", "actions": [
                {"action_id": "condition-general", "mode": "level",
                 "fill": "an endocrine condition", "entails": ["an endocrine condition"]},
                {"action_id": "condition-keep", "mode": "keep", "fill": "hypothyroidism",
                 "entails": ["hypothyroidism"]},
                {"action_id": "condition-placeholder", "mode": "placeholder", "entails": []},
            ]},
            {"decision_id": "d-drug", "actions": [
                {"action_id": "drug-general", "mode": "level",
                 "fill": "a thyroid medication", "entails": ["a thyroid medication"]},
                {"action_id": "drug-keep", "mode": "keep", "fill": "Synthroid",
                 "entails": ["synthroid"]},
                {"action_id": "drug-placeholder", "mode": "placeholder", "entails": []},
            ]},
        ],
    }


def _semantic_property_environment():
    return {
        "occurrences": [
            {
                "occurrence_id": "o-uncontrolled-condition",
                "decision_id": None,
                "surface": "hypothyroidism",
                "runtime_type": "health-condition",
                "controlled": False,
            },
            {
                "occurrence_id": "o-drug-1",
                "decision_id": "d-drug",
                "surface": "Synthroid",
                "runtime_type": "drug",
            },
            {
                "occurrence_id": "o-drug-2",
                "decision_id": "d-drug",
                "surface": "Synthroid",
                "runtime_type": "drug",
            },
        ],
        "decisions": [{
            "decision_id": "d-drug",
            "runtime_type": "drug",
            "controlled": True,
            "actions": [
                {
                    "action_id": "thyroid-specific",
                    "mode": "level",
                    "legal": True,
                    "fill": "teacher-bait-specific",
                    "entails": ["thyroid medication", "medication"],
                },
                {
                    "action_id": "thyroid-equivalent",
                    "mode": "level",
                    "legal": True,
                    "fill": "teacher-bait-equivalent",
                    "entails": ["thyroid medication", "medication"],
                },
                {
                    "action_id": "medication",
                    "mode": "level",
                    "legal": True,
                    "fill": "teacher-bait-coarse",
                    "entails": ["medication"],
                },
                {
                    "action_id": "opioid",
                    "mode": "level",
                    "legal": True,
                    "entails": ["opioid analgesic"],
                },
                {
                    "action_id": "runtime-type-only",
                    "mode": "level",
                    "legal": True,
                    "entails": ["drug"],
                },
                {
                    "action_id": "illegal-property",
                    "mode": "level",
                    "legal": False,
                    "entails": ["forbidden property"],
                },
                {
                    "action_id": "keep",
                    "mode": "keep",
                    "legal": True,
                    "entails": ["synthroid"],
                },
                {
                    "action_id": "placeholder",
                    "mode": "placeholder",
                    "legal": True,
                    "entails": [],
                },
            ],
        }],
    }


def test_semantic_property_candidates_use_legal_entails_and_link_repetitions():
    document = (
        "The patient with hypothyroidism takes Synthroid daily. "
        "Synthroid remains on the medication list."
    )

    candidates = AciTaskAdapter({}).semantic_property_candidates(
        "aci/D2N002", document, _semantic_property_environment()
    )

    accepted = [row for row in candidates if row.get("status") != "rejected"]
    assert {row["accepted_values"][0] for row in accepted} == {
        "thyroid medication", "opioid analgesic",
    }
    assert len(accepted) == 2
    assert all(row["occurrence_ids"] == ["o-drug-1", "o-drug-2"] for row in accepted)
    assert all(row["decision_requirements"] == {
        "d-drug": row["accepted_values"][0]
    } for row in accepted)
    assert all("Synthroid" not in row["question"] for row in accepted)
    assert all("hypothyroidism" not in row["question"].casefold() for row in accepted)
    assert all("[target item]" in row["question"] for row in accepted)
    assert all("[protected item]" in row["question"] for row in accepted)
    assert all(
        row["accepted_values"][0].casefold() not in row["question"].casefold()
        for row in accepted
    )
    assert "forbidden property" not in {
        row["accepted_values"][0] for row in accepted
    }
    type_only = [
        row for row in candidates if row.get("detail_reason") == "placeholder_type_only"
    ]
    assert len(type_only) == 2
    assert all(row["reason"] == "not_generated" for row in type_only)
    assert all("taking" in row["question"].casefold() for row in accepted)

    opioid = next(
        row for row in accepted if row["accepted_values"] == ["opioid analgesic"]
    )
    anchor = build_joint_representative_anchor(
        opioid, _semantic_property_environment()["decisions"]
    )
    assert anchor["action_vector"] == {"d-drug": "opioid"}


def test_semantic_property_records_not_generated_without_safe_context_locator():
    environment = _semantic_property_environment()
    environment["decisions"][0]["actions"] = [
        row for row in environment["decisions"][0]["actions"]
        if row["action_id"] in {"opioid", "keep", "placeholder"}
    ]

    first = AciTaskAdapter({}).semantic_property_candidates(
        "aci/D2N002", "Synthroid.", environment
    )
    second = AciTaskAdapter({}).semantic_property_candidates(
        "aci/D2N002", "Synthroid.", environment
    )

    assert first == second
    assert len(first) == 1
    rejection = first[0]
    assert rejection["status"] == "rejected"
    assert rejection["reason"] == "not_generated"
    assert rejection["detail_reason"] == "no_safe_contextual_locator"
    assert rejection["rejection_id"].startswith("sha256:")
    assert rejection["attempt_hash"].startswith("sha256:")
    assert rejection["doc_id"] == "aci/D2N002"
    assert rejection["evidence"]["decision_id"] == "d-drug"
    assert "Synthroid" not in json.dumps(rejection)
    assert "medication" not in json.dumps(rejection)


def test_semantic_property_records_not_generated_without_task_role_cue():
    environment = _semantic_property_environment()
    environment["decisions"][0]["actions"] = [
        row for row in environment["decisions"][0]["actions"]
        if row["action_id"] in {"opioid", "keep", "placeholder"}
    ]

    records = AciTaskAdapter({}).semantic_property_candidates(
        "aci/D2N002", "The patient mentioned Synthroid yesterday.", environment
    )

    assert len(records) == 1
    assert records[0]["status"] == "rejected"
    assert records[0]["reason"] == "not_generated"
    assert records[0]["detail_reason"] == "no_task_role_cue"


@pytest.mark.parametrize(("runtime_type", "surface", "property_level", "document"), [
    ("drug", "Aspirin", "a drug,", "The patient takes Aspirin daily."),
    (
        "health-condition", "asthma", "(an illness)",
        "The patient has a history of asthma.",
    ),
    (
        "medical-procedure", "CBC", "a test!",
        "The clinician ordered CBC.",
    ),
    ("LOC", "Berlin", "the place.", "The clinic is located in Berlin."),
])
def test_semantic_property_rejects_article_bearing_placeholder_meaning(
    runtime_type, surface, property_level, document,
):
    environment = {
        "occurrences": [{
            "occurrence_id": "o-target",
            "decision_id": "d-target",
            "surface": surface,
            "runtime_type": runtime_type,
        }],
        "decisions": [{
            "decision_id": "d-target",
            "runtime_type": runtime_type,
            "actions": [
                {"action_id": "generic", "mode": "level", "entails": [property_level]},
                {"action_id": "keep", "mode": "keep", "entails": [surface]},
                {"action_id": "placeholder", "mode": "placeholder", "entails": []},
            ],
        }],
    }

    records = AciTaskAdapter({}).semantic_property_candidates(
        "aci/article-generic", document, environment
    )

    assert len(records) == 1
    assert records[0]["status"] == "rejected"
    assert records[0]["reason"] == "not_generated"
    assert records[0]["detail_reason"] == "placeholder_type_only"


def test_semantic_property_requires_role_cue_attached_to_target_occurrence():
    environment = _semantic_property_environment()
    environment["decisions"][0]["actions"] = [
        row for row in environment["decisions"][0]["actions"]
        if row["action_id"] in {"opioid", "keep", "placeholder"}
    ]

    records = AciTaskAdapter({}).semantic_property_candidates(
        "aci/D2N002",
        "The patient takes aspirin and mentions Synthroid.",
        environment,
    )

    assert len(records) == 1
    assert records[0]["status"] == "rejected"
    assert records[0]["reason"] == "not_generated"
    assert records[0]["detail_reason"] == "no_task_role_cue"


def test_semantic_property_records_unsupported_runtime_type_attempt():
    environment = {
        "occurrences": [{
            "occurrence_id": "o-custom",
            "decision_id": "d-custom",
            "surface": "SecretValue",
            "runtime_type": "custom-sensitive-type",
        }],
        "decisions": [{
            "decision_id": "d-custom",
            "runtime_type": "custom-sensitive-type",
            "actions": [
                {"action_id": "specific", "mode": "level", "entails": ["special category"]},
                {"action_id": "keep", "mode": "keep", "entails": ["SecretValue"]},
                {"action_id": "placeholder", "mode": "placeholder", "entails": []},
            ],
        }],
    }

    records = AciTaskAdapter({}).semantic_property_candidates(
        "aci/custom", "SecretValue appears in the record.", environment
    )

    assert len(records) == 1
    assert records[0]["status"] == "rejected"
    assert records[0]["reason"] == "not_generated"
    assert records[0]["detail_reason"] == "unsupported_runtime_type"
    assert records[0]["doc_id"] == "aci/custom"
    assert records[0]["attempt_hash"].startswith("sha256:")


def test_relation_prompt_exposes_only_closed_ids_properties_and_source():
    prompt = relation_teacher_prompt(
        "aci/D2N002",
        "Hypothyroidism is treated with Synthroid.",
        _relation_environment(),
    )

    assert "aci/D2N002" in prompt
    assert "o-condition" in prompt
    assert "an endocrine condition" in prompt
    assert "treated_with" in prompt
    assert "Hypothyroidism is treated with Synthroid." in prompt
    assert "accepted_values" not in prompt


def test_relation_teacher_properties_come_only_from_legal_entails():
    environment = _relation_environment()
    environment["decisions"][1]["actions"][0]["fill"] = "teacher bait"
    prompt = relation_teacher_prompt(
        "aci/D2N002",
        "Hypothyroidism is treated with Synthroid.",
        environment,
    )

    assert "a thyroid medication" in prompt
    assert "teacher bait" not in prompt

    proposal = {
        "relation": "treated_with",
        "argument_occurrence_ids": ["o-condition", "o-drug"],
        "support_properties": {
            "o-condition": "an endocrine condition",
            "o-drug": "teacher bait",
        },
        "answer_occurrence_id": "o-drug",
        "answer_property": "teacher bait",
        "question": "What treatment category is used for the endocrine condition?",
        "evidence_quote": "Hypothyroidism is treated with Synthroid.",
    }

    accepted, rejected = compile_relational_assertions(
        "aci/D2N002",
        "Hypothyroidism is treated with Synthroid.",
        environment,
        [proposal],
    )

    assert accepted == []
    assert rejected[0]["reason"] == "invalid"
    assert rejected[0]["detail_reason"] == "invalid_property"


def test_contextual_relation_rejections_have_stable_ids_and_safe_evidence():
    source = "Hypothyroidism is treated with Synthroid."
    proposal = {
        "relation": "treated_with",
        "argument_occurrence_ids": ["o-condition", "o-drug"],
        "support_properties": {
            "o-condition": "an endocrine condition",
            "o-drug": "a thyroid medication",
        },
        "answer_occurrence_id": "o-drug",
        "answer_property": "a thyroid medication",
        "question": "Which treatment is used for hypothyroidism?",
        "evidence_quote": source,
    }

    _, first = compile_relational_assertions(
        "aci/D2N002", source, _relation_environment(), [proposal]
    )
    _, second = compile_relational_assertions(
        "aci/D2N002", source, _relation_environment(), [proposal]
    )

    assert first == second
    assert first[0]["reason"] == "leakage"
    assert first[0]["detail_reason"] == "protected_locator"
    assert first[0]["doc_id"] == "aci/D2N002"
    assert first[0]["attempt_hash"].startswith("sha256:")
    assert first[0]["rejection_id"].startswith("sha256:")
    assert first[0]["proposal_hash"].startswith("sha256:")
    assert first[0]["evidence"]["source"] == "relation_teacher"
    assert first[0]["evidence"]["evidence_span"] == [0, len(source)]
    assert source not in json.dumps(first[0])
    assert "hypothyroidism" not in json.dumps(first[0]).casefold()


def test_contextual_relation_rejection_hashes_unknown_teacher_argument_ids():
    source = "Hypothyroidism is treated with Synthroid."
    proposal = {
        "relation": "treated_with",
        "argument_occurrence_ids": ["o-condition", "Hypothyroidism"],
        "evidence_quote": source,
    }

    accepted, rejected = compile_relational_assertions(
        "aci/D2N002", source, _relation_environment(), [proposal]
    )

    assert accepted == []
    assert rejected[0]["reason"] == "invalid"
    assert rejected[0]["detail_reason"] == "invalid_arguments"
    assert rejected[0]["evidence"]["argument_occurrence_ids"][0] == "o-condition"
    assert rejected[0]["evidence"]["argument_occurrence_ids"][1].startswith("sha256:")
    assert "hypothyroidism" not in json.dumps(rejected[0]).casefold()


@pytest.mark.parametrize("source", [
    "Hypothyroidism is treated with metformin. Synthroid is listed separately.",
    "Hypothyroidism is treated with metformin\nSynthroid is listed separately",
])
def test_relational_compiler_rejects_cross_sentence_false_link(source):
    proposal = {
        "subtype": "contextual_relation",
        "relation": "treated_with",
        "argument_occurrence_ids": ["o-condition", "o-drug"],
        "support_properties": {
            "o-condition": "an endocrine condition",
            "o-drug": "a thyroid medication",
        },
        "answer_occurrence_id": "o-drug",
        "answer_property": "a thyroid medication",
        "question": "What treatment category is used for the endocrine condition?",
        "evidence_quote": source,
    }

    accepted, rejected = compile_relational_assertions(
        "aci/D2N002", source, _relation_environment(), [proposal]
    )

    assert accepted == []
    assert rejected[0]["reason"] == "invalid"
    assert rejected[0]["detail_reason"] == "invalid_evidence"


def test_relational_compiler_rejects_extra_entity_inside_connector():
    source = "Hypothyroidism and diabetes is treated with Synthroid."
    proposal = {
        "subtype": "contextual_relation",
        "relation": "treated_with",
        "argument_occurrence_ids": ["o-condition", "o-drug"],
        "support_properties": {
            "o-condition": "an endocrine condition",
            "o-drug": "a thyroid medication",
        },
        "answer_occurrence_id": "o-drug",
        "answer_property": "a thyroid medication",
        "question": "What treatment category is used for the endocrine condition?",
        "evidence_quote": source,
    }

    accepted, rejected = compile_relational_assertions(
        "aci/D2N002", source, _relation_environment(), [proposal]
    )

    assert accepted == []
    assert rejected[0]["reason"] == "invalid"
    assert rejected[0]["detail_reason"] == "invalid_evidence"


def test_relational_compiler_rejects_answer_modifier_and_alias_leakage():
    source = "Hypothyroidism is treated with Synthroid."
    environment = _relation_environment()
    environment["occurrences"][1]["aliases"] = ["levothyroxine sodium"]

    def proposal(question):
        return {
            "subtype": "contextual_relation",
            "relation": "treated_with",
            "argument_occurrence_ids": ["o-condition", "o-drug"],
            "support_properties": {
                "o-condition": "an endocrine condition",
                "o-drug": "a thyroid medication",
            },
            "answer_occurrence_id": "o-drug",
            "answer_property": "a thyroid medication",
            "question": question,
            "evidence_quote": source,
        }

    _, answer_rejection = compile_relational_assertions(
        "aci/D2N002", source, environment,
        [proposal("Which thyroid treatment is used for the endocrine condition?")],
    )
    _, alias_rejection = compile_relational_assertions(
        "aci/D2N002", source, environment,
        [proposal("Which levothyroxine treatment is used for the endocrine condition?")],
    )

    assert answer_rejection[0]["detail_reason"] == "answer_leakage"
    assert alias_rejection[0]["detail_reason"] == "protected_locator"


@pytest.mark.parametrize("alias", ["Li", "UK", "AF"])
def test_relational_compiler_rejects_short_protected_alias_leakage(alias):
    source = "Hypothyroidism is treated with Synthroid."
    environment = _relation_environment()
    environment["occurrences"][1]["aliases"] = [alias]
    proposal = {
        "subtype": "contextual_relation",
        "relation": "treated_with",
        "argument_occurrence_ids": ["o-condition", "o-drug"],
        "support_properties": {
            "o-condition": "an endocrine condition",
            "o-drug": "a thyroid medication",
        },
        "answer_occurrence_id": "o-drug",
        "answer_property": "a thyroid medication",
        "question": f"Which {alias} treatment is used for the endocrine condition?",
        "evidence_quote": source,
    }

    accepted, rejected = compile_relational_assertions(
        "aci/D2N002", source, environment, [proposal]
    )

    assert accepted == []
    assert rejected[0]["reason"] == "leakage"
    assert rejected[0]["detail_reason"] == "protected_locator"


def test_relational_compiler_rejects_non_contextual_proposed_subtype():
    source = "Hypothyroidism is treated with Synthroid."
    proposal = {
        "subtype": "semantic_property",
        "relation": "treated_with",
        "argument_occurrence_ids": ["o-condition", "o-drug"],
        "support_properties": {
            "o-condition": "an endocrine condition",
            "o-drug": "a thyroid medication",
        },
        "answer_occurrence_id": "o-drug",
        "answer_property": "a thyroid medication",
        "question": "What treatment category is used for the endocrine condition?",
        "evidence_quote": source,
    }

    accepted, rejected = compile_relational_assertions(
        "aci/D2N002", source, _relation_environment(), [proposal]
    )

    assert accepted == []
    assert rejected[0]["reason"] == "invalid"
    assert rejected[0]["detail_reason"] == "invalid_subtype"


def test_relational_compiler_derives_gold_and_links_from_frozen_inventory():
    source = "Hypothyroidism is treated with Synthroid."
    proposals = [{
        "relation": "treated_with",
        "argument_occurrence_ids": ["o-condition", "o-drug"],
        "support_properties": {
            "o-condition": "an endocrine condition",
            "o-drug": "a thyroid medication",
        },
        "answer_occurrence_id": "o-drug",
        "answer_property": "a thyroid medication",
        "question": "What treatment category is used for the endocrine condition?",
        "evidence_quote": source,
    }]

    accepted, rejected = compile_relational_assertions(
        "aci/D2N002", source, _relation_environment(), proposals
    )

    assert rejected == []
    assert accepted == [{
        "family": "context",
        "scope": "linked",
        "subtype": "contextual_relation",
        "relation": "treated_with",
        "occurrence_ids": ["o-condition", "o-drug"],
        "group_id": "relation:treated_with:o-condition:o-drug",
        "question": "What treatment category is used for the endocrine condition?",
        "accepted_values": ["a thyroid medication"],
        "decision_requirements": {
            "d-condition": "an endocrine condition",
            "d-drug": "a thyroid medication",
        },
        "evidence": {"source_quotes": [source]},
    }]


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"evidence_quote": "Synthroid is mentioned elsewhere."}, "invalid_evidence"),
        ({"answer_property": "a medicine"}, "invalid_property"),
        ({"question": "Is a thyroid medication used?"}, "answer_leakage"),
        ({"relation": "invented_relation"}, "invalid_relation"),
    ],
)
def test_relational_compiler_rejects_unfrozen_or_leaking_proposals(change, reason):
    source = "Hypothyroidism is treated with Synthroid."
    proposal = {
        "relation": "treated_with",
        "argument_occurrence_ids": ["o-condition", "o-drug"],
        "support_properties": {
            "o-condition": "an endocrine condition",
            "o-drug": "a thyroid medication",
        },
        "answer_occurrence_id": "o-drug",
        "answer_property": "a thyroid medication",
        "question": "What treatment category is used for the endocrine condition?",
        "evidence_quote": source,
    }
    proposal.update(change)

    accepted, rejected = compile_relational_assertions(
        "aci/D2N002", source, _relation_environment(), [proposal]
    )

    assert accepted == []
    assert rejected[0]["reason"] == (
        "leakage" if reason == "answer_leakage" else "invalid"
    )
    assert rejected[0]["detail_reason"] == reason


@pytest.mark.parametrize(
    ("source", "proposal_change", "reason"),
    [
        (
            "Hypothyroidism and Synthroid are listed in the chart.",
            {"evidence_quote": "Hypothyroidism and Synthroid are listed in the chart."},
            "invalid_evidence",
        ),
        (
            "Hypothyroidism is treated with Synthroid.",
            {"argument_polarities": {"o-condition": "negated", "o-drug": "active"}},
            "invalid_polarity",
        ),
        (
            "Hypothyroidism is treated with Synthroid. "
            "Hypothyroidism is not treated with Synthroid.",
            {"evidence_quote": "Hypothyroidism is treated with Synthroid."},
            "source_contradiction",
        ),
    ],
)
def test_relational_compiler_requires_direct_noncontradictory_authoritative_evidence(
    source, proposal_change, reason
):
    proposal = {
        "relation": "treated_with",
        "argument_occurrence_ids": ["o-condition", "o-drug"],
        "support_properties": {
            "o-condition": "an endocrine condition",
            "o-drug": "a thyroid medication",
        },
        "answer_occurrence_id": "o-drug",
        "answer_property": "a thyroid medication",
        "question": "What treatment category is used for the endocrine condition?",
        "evidence_quote": source,
    }
    proposal.update(proposal_change)

    accepted, rejected = compile_relational_assertions(
        "aci/D2N002", source, _relation_environment(), [proposal]
    )

    assert accepted == []
    assert len(rejected) == 1
    assert rejected[0]["proposal_index"] == 0
    assert rejected[0]["reason"] == "invalid"
    assert rejected[0]["detail_reason"] == reason
    assert rejected[0]["rejection_id"].startswith("sha256:")
    assert rejected[0]["proposal_hash"].startswith("sha256:")
    assert rejected[0]["evidence"]["source"] == "relation_teacher"


def test_aci_adapter_builds_delivered_facts_only_from_authoritative_reference():
    environment = _relation_environment()
    adapter = AciTaskAdapter({
        "aci/D2N002": (
            "A 62-year-old male has hypothyroidism and takes Synthroid."
        )
    })

    candidates = adapter.deterministic_candidates(
        "aci/D2N002",
        "A 62-year-old male has hypothyroidism and takes Synthroid for treatment.",
        environment,
    )

    delivered = [row for row in candidates if row.get("family") == "delivered"]
    assert {row["scoring_contract"]["value"] for row in delivered} == {
        "62-year-old", "male", "hypothyroidism", "Synthroid"
    }
    condition = next(row for row in delivered
                     if row["scoring_contract"]["value"] == "hypothyroidism")
    assert condition["scope"] == "linked"
    assert condition["occurrence_ids"] == ["o-condition"]


def _aci_delivered_environment():
    occurrences = [
        ("o-condition", "d-condition", "hypothyroidism", "health-condition"),
        ("o-treatment", "d-treatment", "Synthroid", "drug"),
        ("o-test", "d-test", "thyroid labs", "medical-procedure"),
    ]
    return {
        "occurrences": [
            {
                "occurrence_id": occurrence_id,
                "decision_id": decision_id,
                "surface": surface,
                "runtime_type": runtime_type,
            }
            for occurrence_id, decision_id, surface, runtime_type in occurrences
        ],
        "decisions": [
            {"decision_id": decision_id, "controlled": True}
            for _, decision_id, _, _ in occurrences
        ],
    }


def test_aci_delivered_candidates_compile_reference_backed_groups_with_capped_structure():
    reference = """HISTORY OF PRESENT ILLNESS
62-year-old male with hypothyroidism.
ASSESSMENT
Hypothyroidism — Endocrine — Stable
PLAN
Hypothyroidism — Synthroid — thyroid labs
"""
    adapter = AciTaskAdapter({"aci/D2N002": reference})
    candidates = adapter.delivered_candidates(
        "aci/D2N002",
        "A ceiling output with no authoritative facts.",
        reference,
        _aci_delivered_environment(),
    )

    contracts = {row["scoring_contract"]["kind"] for row in candidates}
    assert contracts == {"contains", "field_value", "required_sections", "exact_relation"}
    assert {row["subtype"] for row in candidates} == {
        "content", "field", "structure", "exact_relation",
    }
    structure = next(row for row in candidates if row["subtype"] == "structure")
    assert structure["group_id"] == "structure:required_sections"
    assert structure["scoring_contract"] == {
        "kind": "required_sections",
        "sections": ["HISTORY OF PRESENT ILLNESS", "ASSESSMENT", "PLAN"],
        "parseability": {
            "assessment": {"kind": "rows", "count": 1},
            "plan": {"kind": "rows", "count": 1},
        },
    }
    assert "fields" not in structure["scoring_contract"]

    artifact = package_utility_artifact(
        {"environment_hash": "env", "documents": {"aci/D2N002": _aci_delivered_environment()}},
        {"aci/D2N002": candidates},
        family_budgets={"context": 0.6, "delivered": 0.4},
        structural_cap=0.1,
        pins={},
    )
    structural_weight = sum(
        row["weight"] for row in artifact["assertions"].values()
        if row["subtype"] == "structure"
    )
    assert structural_weight == pytest.approx(0.04)
    assert sum(row["weight"] for row in artifact["assertions"].values()) == pytest.approx(0.4)


@pytest.mark.parametrize("structural_cap", [None, "0.1", True, -0.1, 1.1])
def test_delivered_structure_requires_a_frozen_numeric_cap(structural_cap):
    reference = """HISTORY OF PRESENT ILLNESS
62-year-old male with hypothyroidism.
ASSESSMENT
Hypothyroidism — Endocrine — Stable
PLAN
Hypothyroidism — Synthroid — thyroid labs
"""
    candidates = AciTaskAdapter({"aci/D2N002": reference}).delivered_candidates(
        "aci/D2N002", "unused", reference, _aci_delivered_environment()
    )

    with pytest.raises(ValueError, match="structural_cap"):
        package_utility_artifact(
            {"environment_hash": "env", "documents": {"aci/D2N002": _aci_delivered_environment()}},
            {"aci/D2N002": candidates},
            family_budgets={"context": 0.6, "delivered": 0.4},
            structural_cap=structural_cap,
            pins={},
        )


def test_aci_parser_accepts_schema_valid_inline_headings():
    reference = """HISTORY OF PRESENT ILLNESS: 62-year-old male with hypothyroidism.
ASSESSMENT: Hypothyroidism — Endocrine — Stable
PLAN: Hypothyroidism — Synthroid — thyroid labs
"""

    candidates = AciTaskAdapter({"aci/D2N002": reference}).delivered_candidates(
        "aci/D2N002", "unused", reference, _aci_delivered_environment()
    )

    structure = next(row for row in candidates if row["subtype"] == "structure")
    assert structure["scoring_contract"]["sections"] == [
        "HISTORY OF PRESENT ILLNESS", "ASSESSMENT", "PLAN",
    ]
    assert structure["scoring_contract"]["parseability"] == {
        "assessment": {"kind": "rows", "count": 1},
        "plan": {"kind": "rows", "count": 1},
    }
    assert any(row["subtype"] == "exact_relation" for row in candidates)


@pytest.mark.parametrize(
    "reference",
    [
        """HISTORY OF PRESENT ILLNESS
No acute concerns.
ASSESSMENT
none
PLAN
none
""",
        """HISTORY OF PRESENT ILLNESS: No acute concerns.
ASSESSMENT: none
PLAN: none
""",
    ],
)
def test_aci_structure_accepts_heading_only_and_inline_none_sections(reference):
    candidates = AciTaskAdapter({"aci/D2N002": reference}).delivered_candidates(
        "aci/D2N002", "unused", reference, _aci_delivered_environment()
    )
    structure = next(row for row in candidates if row["subtype"] == "structure")
    artifact = {
        "documents": {"d1": {"utility_weight_denominator": 1.0}},
        "assertions": {
            "structure": {
                "assertion_id": "structure", "doc_id": "d1", "family": "delivered",
                "weight": 1.0, "scoring_contract": structure["scoring_contract"],
            },
        },
    }

    valid = score_utility(
        artifact, "d1", doc_p="unused", out_final=reference,
        reader=lambda questions, context: pytest.fail("reader must not be called"),
    )
    malformed = score_utility(
        artifact,
        "d1",
        doc_p="unused",
        out_final=reference.replace("ASSESSMENT: none", "ASSESSMENT: no concerns").replace(
            "ASSESSMENT\nnone", "ASSESSMENT\nno concerns"
        ),
        reader=lambda questions, context: pytest.fail("reader must not be called"),
    )

    assert valid["component_scores"] == {"structure": 1.0}
    assert malformed["component_scores"] == {"structure": 0.0}


def test_aci_compiler_abstains_from_duplicate_assessment_condition_contracts():
    reference = """HISTORY OF PRESENT ILLNESS
Follow-up note.
ASSESSMENT
Hypothyroidism — Endocrine — Stable
Hypothyroidism — Autoimmune — Active
Arthritis — Musculoskeletal — Stable
PLAN
Hypothyroidism — Synthroid — thyroid labs
Arthritis — physical therapy — mobility assessment
"""

    candidates = AciTaskAdapter({"aci/D2N002": reference}).delivered_candidates(
        "aci/D2N002", "unused", reference, _aci_delivered_environment()
    )
    assessment_fields = [
        row["scoring_contract"] for row in candidates
        if row["subtype"] == "field"
        and row["scoring_contract"].get("section") == "ASSESSMENT"
    ]
    relations = [
        row["scoring_contract"] for row in candidates
        if row["subtype"] == "exact_relation"
    ]

    assert {(row["row"], row["field"]) for row in assessment_fields} == {
        ("Arthritis", "category"),
        ("Arthritis", "status"),
    }
    assert [row["condition"] for row in relations] == ["Arthritis"]


def test_score_utility_evaluates_every_deterministic_delivered_contract():
    artifact = {
        "documents": {"d1": {"utility_weight_denominator": 1.0}},
        "assertions": {
            "content": {
                "assertion_id": "content", "doc_id": "d1", "family": "delivered",
                "weight": 0.25,
                "scoring_contract": {"kind": "contains", "value": "hypothyroidism"},
            },
            "field": {
                "assertion_id": "field", "doc_id": "d1", "family": "delivered",
                "weight": 0.25,
                "scoring_contract": {
                    "kind": "field_value", "section": "ASSESSMENT",
                    "row": "hypothyroidism", "field": "status", "value": "stable",
                },
            },
            "structure": {
                "assertion_id": "structure", "doc_id": "d1", "family": "delivered",
                "weight": 0.25,
                "scoring_contract": {
                    "kind": "required_sections",
                    "sections": ["HISTORY OF PRESENT ILLNESS", "ASSESSMENT", "PLAN"],
                },
            },
            "relation": {
                "assertion_id": "relation", "doc_id": "d1", "family": "delivered",
                "weight": 0.25,
                "scoring_contract": {
                    "kind": "exact_relation", "section": "PLAN",
                    "condition": "hypothyroidism", "treatment": "Synthroid",
                    "test": "thyroid labs",
                },
            },
        },
    }
    delivered = """HISTORY OF PRESENT ILLNESS
62-year-old male with hypothyroidism.
ASSESSMENT
Hypothyroidism — Endocrine — Stable
PLAN
Hypothyroidism — Synthroid — thyroid labs
"""

    result = score_utility(
        artifact,
        "d1",
        doc_p="unused",
        out_final=delivered,
        reader=lambda questions, context: pytest.fail("reader must not be called"),
    )

    assert result == {
        "component_scores": {
            "content": 1.0, "field": 1.0, "relation": 1.0, "structure": 1.0,
        },
        "utility": 1.0,
    }


    malformed = score_utility(
        artifact,
        "d1",
        doc_p="unused",
        out_final="""HISTORY OF PRESENT ILLNESS
62-year-old male with hypothyroidism.
ASSESSMENT
Hypothyroidism stable
PLAN
Hypothyroidism — Synthroid — thyroid labs
""",
        reader=lambda questions, context: pytest.fail("reader must not be called"),
    )

    assert malformed["component_scores"]["structure"] == 0.0


def test_package_rejects_dangling_occurrence_decision_links():
    environment = {
        "environment_hash": "env-hash",
        "documents": {
            "d1": {
                "occurrences": [{"occurrence_id": "o1", "decision_id": "missing"}],
                "decisions": [{"decision_id": "dec1", "controlled": True}],
            }
        },
    }

    with pytest.raises(ValueError, match="unknown decision links"):
        package_utility_artifact(
            environment,
            {"d1": [{
                "family": "delivered", "scope": "linked", "subtype": "content",
                "occurrence_ids": ["o1"], "group_id": "condition:1",
                "scoring_contract": {"kind": "contains", "value": "condition"},
            }]},
            family_budgets={"context": 0.6, "delivered": 0.4},
            pins={},
        )


def test_package_rejects_duplicate_assertion_ids():
    environment = {
        "environment_hash": "env-hash",
        "documents": {"d1": {"occurrences": [], "decisions": []}},
    }
    candidate = {
        "assertion_id": "ast:duplicate", "family": "delivered", "scope": "global",
        "subtype": "content", "occurrence_ids": [], "group_id": "document:1",
        "scoring_contract": {"kind": "contains", "value": "condition"},
    }

    with pytest.raises(ValueError, match="duplicate assertion id"):
        package_utility_artifact(
            environment,
            {"d1": [candidate, {**candidate, "group_id": "document:2"}]},
            family_budgets={"context": 0.6, "delivered": 0.4},
            pins={},
        )


def test_relational_compiler_rejects_illegal_argument_runtime_types():
    source = "Hypothyroidism is treated with arthritis."
    proposal = {
        "relation": "treated_with",
        "argument_occurrence_ids": ["o-condition", "o-drug"],
        "support_properties": {
            "o-condition": "an endocrine condition",
            "o-drug": "a thyroid medication",
        },
        "answer_occurrence_id": "o-drug",
        "answer_property": "a thyroid medication",
        "question": "What treatment category is used for the endocrine condition?",
        "evidence_quote": source,
    }
    environment = _relation_environment()
    environment["occurrences"][1]["runtime_type"] = "health-condition"
    environment["occurrences"][1]["surface"] = "arthritis"

    accepted, rejected = compile_relational_assertions(
        "aci/D2N002", source, environment, [proposal]
    )

    assert accepted == []
    assert len(rejected) == 1
    assert rejected[0]["proposal_index"] == 0
    assert rejected[0]["reason"] == "invalid"
    assert rejected[0]["detail_reason"] == "invalid_argument_types"
    assert rejected[0]["rejection_id"].startswith("sha256:")
    assert rejected[0]["proposal_hash"].startswith("sha256:")


def test_teacher_abstention_records_missing_context_without_retry():
    frozen = {
        "environment_hash": "env-v1",
        "documents": {"d1": {
            "occurrences": [],
            "decisions": [],
        }},
    }

    class Adapter:
        def deterministic_candidates(self, doc_id, document, environment_document):
            return [{
                "family": "delivered", "scope": "global", "subtype": "content",
                "occurrence_ids": [], "group_id": "document:condition",
                "scoring_contract": {"kind": "contains", "value": "hypothyroidism"},
            }]

        def compile_relations(self, doc_id, document, environment_document, proposals):
            raise AssertionError("an abstaining teacher must not be compiled")

    class AbstainingTeacher:
        def __init__(self):
            self.calls = 0

        def propose(self, prompt):
            self.calls += 1
            return []

    teacher = AbstainingTeacher()
    artifact = build_utility_artifact(
        frozen,
        Adapter(),
        {"d1": "hypothyroidism"},
        threshold_manifest={
            "family_budgets": {"context": 0.6, "delivered": 0.4},
            "min_context_assertions": 1,
        },
        pins={"gate_manifest_hash": "gate-v1"},
        reader=lambda questions, context: [],
        render_action_vector=lambda doc_id, action_vector: "unused",
        relation_teacher=teacher,
    )

    assert teacher.calls == 1
    assert artifact["documents"]["d1"]["measurement_state"] == "partial"
    assert artifact["rejections"]["summary_by_reason"] == {"not_generated": 1}


def test_relational_compiler_rejects_protected_locator_leakage():
    source = "Hypothyroidism is treated with Synthroid."
    proposal = {
        "relation": "treated_with",
        "argument_occurrence_ids": ["o-condition", "o-drug"],
        "support_properties": {
            "o-condition": "an endocrine condition",
            "o-drug": "a thyroid medication",
        },
        "answer_occurrence_id": "o-drug",
        "answer_property": "a thyroid medication",
        "question": "Which treatment is used for hypothyroidism?",
        "evidence_quote": source,
    }

    accepted, rejected = compile_relational_assertions(
        "aci/D2N002", source, _relation_environment(), [proposal]
    )

    assert accepted == []
    assert len(rejected) == 1
    assert rejected[0]["proposal_index"] == 0
    assert rejected[0]["reason"] == "leakage"
    assert rejected[0]["detail_reason"] == "protected_locator"
    assert rejected[0]["rejection_id"].startswith("sha256:")
    assert rejected[0]["proposal_hash"].startswith("sha256:")


def test_delivered_only_scoring_is_deterministic_without_reader_call():
    artifact = {
        "documents": {"d1": {"utility_weight_denominator": 1.0}},
        "assertions": {
            "d1": {
                "assertion_id": "d1", "doc_id": "d1", "family": "delivered",
                "weight": 1.0,
                "scoring_contract": {"kind": "contains", "value": "kidney transplant"},
            }
        },
    }

    result = score_utility(
        artifact,
        "d1",
        doc_p="unused",
        out_final="History includes kidney transplant.",
        reader=lambda questions, context: pytest.fail("reader must not be called"),
    )

    assert result == {"component_scores": {"d1": 1.0}, "utility": 1.0}


def test_runtime_component_scores_follow_stable_assertion_id_order():
    artifact = {
        "documents": {"d1": {"utility_weight_denominator": 1.0}},
        "assertions": {
            "z": {
                "assertion_id": "z", "doc_id": "d1", "family": "delivered",
                "weight": 0.5,
                "scoring_contract": {"kind": "contains", "value": "kidney transplant"},
            },
            "a": {
                "assertion_id": "a", "doc_id": "d1", "family": "delivered",
                "weight": 0.5,
                "scoring_contract": {"kind": "contains", "value": "female"},
            },
        },
    }

    result = score_utility(
        artifact,
        "d1",
        doc_p="unused",
        out_final="A female has a kidney transplant.",
        reader=lambda questions, context: pytest.fail("reader must not be called"),
    )

    assert list(result["component_scores"]) == ["a", "z"]


def test_freeze_ranker_environment_uses_action_semantics_not_menu_position_for_ids():
    actions = [
        {"fill": "a thyroid medication", "mode": "level", "aset": 10.0},
        {"fill": "Synthroid", "mode": "level", "keep": True, "aset": 1.0},
        {"fill": None, "mode": "placeholder"},
    ]

    def environment(action_rows):
        return {"corpora": {"clinical": {"d1": {"spans": [{
            "surface": "Synthroid", "type": "drug", "start": 10, "end": 19,
            "actions": action_rows,
        }]}}}}

    first = freeze_ranker_environment(environment(actions))
    second = freeze_ranker_environment(environment(list(reversed(actions))))
    first_ids = {
        row["fill"]: row["action_id"]
        for row in first["documents"]["d1"]["decisions"][0]["actions"]
    }
    second_ids = {
        row["fill"]: row["action_id"]
        for row in second["documents"]["d1"]["decisions"][0]["actions"]
    }

    assert first_ids == second_ids


def test_freeze_ranker_environment_preserves_uncontrolled_frozen_occurrence():
    ranker_env = {"corpora": {"clinical": {"d1": {"spans": [{
        "surface": "Synthroid", "type": "drug", "start": 10, "end": 19,
        "actions": [
            {"fill": "a thyroid medication", "mode": "level"},
            {"fill": "Synthroid", "mode": "level", "keep": True},
            {"fill": None, "mode": "placeholder"},
        ],
    }]}}}}

    frozen = freeze_ranker_environment(
        ranker_env,
        occurrences_by_document={"d1": [{
            "surface": "Unrelated", "type": "drug", "start": 30, "end": 39,
        }]},
    )

    occurrence = frozen["documents"]["d1"]["occurrences"][0]
    assert occurrence["surface"] == "Unrelated"
    assert occurrence["controlled"] is False
    assert occurrence["decision_id"] is None


def test_frozen_occurrences_from_arms_carries_detector_provenance():
    arms = {
        "clinical": {
            "aci/D2N002": {
                "tau_walk": ["doc", [{
                    "surface": "Synthroid",
                    "type": "drug",
                    "start": 10,
                    "end": 19,
                    "score": 0.91,
                    "lattice": ["thyroid medication"],
                }]]
            }
        }
    }
    detector_pin = {
        "config": "qa-v2-clinical",
        "model": "knowledgator/gliner-pii-large-v1.0",
        "threshold": 0.35,
    }

    occurrences = qa_builder.frozen_occurrences_from_arms(
        arms, detector_provenance=detector_pin
    )

    assert occurrences["aci/D2N002"][0]["detector_provenance"] == {
        **detector_pin,
        "score": 0.91,
    }
