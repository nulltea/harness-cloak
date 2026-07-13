import copy

import pytest

import cloak.train.roundtrip as roundtrip
import cloak.train.qa_builder as qa_builder
from cloak.train.utility_credit import provisional_advantages
from cloak.train.qa_builder import (
    AciTaskAdapter,
    BatchedContextReader,
    OpenRouterRelationTeacher,
    assign_static_weights,
    build_utility_artifact,
    build_joint_representative_anchor,
    compile_relational_assertions,
    freeze_ranker_environment,
    normalize_threshold_manifest,
    package_utility_artifact,
    relation_teacher_prompt,
    score_utility,
    validate_context_assertions,
)


COST_BUDGETS = {
    "base": {
        "remote_round_trips_per_rollout": 1,
        "context_reader_batches_per_rollout": 1,
    },
    "counterfactual": {
        "remote_round_trips_per_selected_pair": 1,
        "context_reader_batches_per_selected_pair": 1,
    },
}


@pytest.mark.parametrize("field", [
    "reader_stability_repetitions",
    "reader_option_permutations",
])
@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_threshold_manifest_requires_integer_reader_counts(field, value):
    manifest = {
        "family_budgets": {"context": 0.6, "delivered": 0.4},
        "cost_budgets": COST_BUDGETS,
        "reader_stability_repetitions": 1,
        "reader_option_permutations": 1,
    }
    manifest[field] = value

    with pytest.raises(ValueError, match="reader.*integer"):
        normalize_threshold_manifest(manifest)


@pytest.mark.parametrize("value", [True, 1.0, "1", -1])
def test_threshold_manifest_requires_non_boolean_nonnegative_min_context_assertions(value):
    with pytest.raises(ValueError, match="min_context_assertions"):
        normalize_threshold_manifest({
            "family_budgets": {"context": 0.6, "delivered": 0.4},
            "cost_budgets": COST_BUDGETS,
            "min_context_assertions": value,
        })


def test_threshold_manifest_freezes_wall_time_budgets():
    manifest = normalize_threshold_manifest({
        "family_budgets": {"context": 0.6, "delivered": 0.4},
        "cost_budgets": COST_BUDGETS,
        "wall_time_budgets": {
            "artifact_build_seconds_per_document": 3.0,
            "base_seconds_per_rollout": 2.0,
            "counterfactual_seconds_per_selected_pair": 2.0,
        },
    })

    assert manifest["wall_time_budgets"]["artifact_build_seconds_per_document"] == 3.0


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


@pytest.mark.parametrize("family_budgets", [
    {"context": 1.0},
    {"context": 0.5, "delivered": 0.5, "unknown": 0.1},
    {"context": 0.0, "delivered": 1.0},
    {"context": float("nan"), "delivered": 1.0},
    {"context": float("inf"), "delivered": 1.0},
])
def test_builder_requires_exact_positive_finite_family_budgets(family_budgets):
    with pytest.raises(ValueError, match="family budgets"):
        package_utility_artifact(
            {"environment_hash": "env-v1", "documents": {}},
            {},
            family_budgets=family_budgets,
            pins={},
        )


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


def test_aci_d2n002_joint_anchor_keeps_unrelated_decisions_above_floor():
    frozen = freeze_ranker_environment({
        "k_floors": {"DEM": 1e30, "OTHER": 1e30},
        "corpora": {"clinical": {"aci/D2N002": {"spans": [
            {
                "surface": "arthritis", "type": "DEM", "start": 0, "end": 9,
                "actions": [
                    {"fill": "a disease", "mode": "level", "aset": 1e31},
                    {"fill": "arthritis", "mode": "level", "keep": True, "aset": 1.0},
                    {"fill": None, "mode": "placeholder"},
                ],
            },
            {
                "surface": "tylenol", "type": "OTHER", "start": 10, "end": 17,
                "actions": [
                    {"fill": "a medicine", "mode": "level", "aset": 60.0},
                    {"fill": "tylenol", "mode": "level", "keep": True, "aset": 1.0},
                    {"fill": None, "mode": "placeholder"},
                ],
            },
        ]}}},
    })
    decisions = frozen["documents"]["aci/D2N002"]["decisions"]
    linked = next(row for row in decisions if row["canonical_key"] == "arthritis")
    unrelated = next(row for row in decisions if row["canonical_key"] == "tylenol")

    anchor = build_joint_representative_anchor({
        "decision_requirements": {linked["decision_id"]: "a disease"},
    }, decisions)

    assert next(action for action in linked["actions"] if action["action_id"] == anchor[
        "action_vector"][linked["decision_id"]
    ])["mode"] == "level"
    assert next(action for action in unrelated["actions"] if action["action_id"] == anchor[
        "action_vector"][unrelated["decision_id"]
    ])["mode"] == "keep"


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


def test_freeze_preserves_lattice_entailment_closure_for_representative_anchors():
    frozen = freeze_ranker_environment({
        "corpora": {"clinical": {"d1": {"spans": [{
            "surface": "lisinopril",
            "type": "drug",
            "start": 0,
            "end": 10,
            "actions": [
                {"fill": "an ACE inhibitor", "mode": "level", "aset": 100.0},
                {"fill": "an antihypertensive", "mode": "level", "aset": 100.0},
                {"fill": "a medicine", "mode": "level", "aset": 100.0},
                {"fill": "lisinopril", "mode": "level", "keep": True, "aset": 100.0},
                {"fill": None, "mode": "placeholder"},
            ],
        }]}}},
    })
    decision = frozen["documents"]["d1"]["decisions"][0]
    actions = {row["fill"]: row for row in decision["actions"]}

    assert actions["an ACE inhibitor"]["entails"] == [
        "an ace inhibitor", "an antihypertensive", "a medicine",
    ]
    assert actions["an antihypertensive"]["entails"] == [
        "an antihypertensive", "a medicine",
    ]
    assert actions["a medicine"]["entails"] == ["a medicine"]
    assert "an ace inhibitor" not in actions["a medicine"]["entails"]

    decision_id = decision["decision_id"]
    broad = build_joint_representative_anchor(
        {"decision_requirements": {decision_id: "a medicine"}}, [decision]
    )
    narrow = build_joint_representative_anchor(
        {"decision_requirements": {decision_id: "an ACE inhibitor"}}, [decision]
    )

    assert broad["action_vector"] == {decision_id: actions["a medicine"]["action_id"]}
    assert narrow["action_vector"] == {
        decision_id: actions["an ACE inhibitor"]["action_id"]
    }


def test_freeze_binds_action_legality_and_identity_to_effective_floors():
    environment = {
        "k_floors": {"drug": 100.0, "OTHER": 50.0},
        "corpora": {"clinical": {"d1": {"spans": [{
            "surface": "lisinopril",
            "type": "drug",
            "start": 0,
            "end": 10,
            "actions": [
                {"fill": "an ACE inhibitor", "mode": "level", "aset": 20.0},
                {"fill": "a medicine", "mode": "level", "aset": 100.0},
                {"fill": "lisinopril", "mode": "level", "keep": True, "aset": 1.0},
                {"fill": None, "mode": "placeholder"},
            ],
        }]}}},
    }

    frozen = freeze_ranker_environment(environment)
    lower_floor = freeze_ranker_environment(environment, floors={"drug": 10.0})
    actions = {
        row["fill"]: row["legal"]
        for row in frozen["documents"]["d1"]["decisions"][0]["actions"]
    }

    assert frozen["effective_floors"] == {"OTHER": 50.0, "drug": 100.0}
    assert actions == {
        "an ACE inhibitor": False,
        "a medicine": True,
        "lisinopril": True,
        None: True,
    }
    assert lower_floor["effective_floors"] == {"OTHER": 50.0, "drug": 10.0}
    assert lower_floor["environment_hash"] != frozen["environment_hash"]


def test_freeze_binds_source_and_authoritative_reference_identity():
    environment = {"corpora": {"clinical": {"d1": {"spans": []}}}}

    frozen = freeze_ranker_environment(
        environment,
        source_documents={"d1": "source document"},
        authoritative_references={"d1": "gold reference"},
    )
    changed_source = freeze_ranker_environment(
        environment,
        source_documents={"d1": "changed source"},
        authoritative_references={"d1": "gold reference"},
    )
    changed_reference = freeze_ranker_environment(
        environment,
        source_documents={"d1": "source document"},
        authoritative_references={"d1": "changed reference"},
    )

    assert frozen["documents"]["d1"]["source_hash"] == qa_builder._stable_hash(
        "source document"
    )
    assert frozen["documents"]["d1"]["authoritative_reference_hash"] == (
        qa_builder._stable_hash("gold reference")
    )
    assert changed_source["environment_hash"] != frozen["environment_hash"]
    assert changed_reference["environment_hash"] != frozen["environment_hash"]


def test_builder_rejects_under_floor_representative_action():
    frozen = freeze_ranker_environment({
        "k_floors": {"drug": 100.0, "OTHER": 100.0},
        "corpora": {"clinical": {"d1": {"spans": [{
            "surface": "Synthroid",
            "type": "drug",
            "start": 0,
            "end": 9,
            "actions": [
                {"fill": "a thyroid medication", "mode": "level", "aset": 10.0},
                {"fill": "Synthroid", "mode": "level", "keep": True, "aset": 1.0},
                {"fill": None, "mode": "placeholder"},
            ],
        }]}}},
    })
    decision = frozen["documents"]["d1"]["decisions"][0]
    occurrence = frozen["documents"]["d1"]["occurrences"][0]

    class Adapter:
        task_pin = {"adapter": "test", "version": "v1"}

        def deterministic_candidates(self, doc_id, document, environment_document):
            return [{
                "family": "context",
                "scope": "linked",
                "subtype": "semantic_property",
                "occurrence_ids": [occurrence["occurrence_id"]],
                "group_id": "drug:category",
                "question": "What treatment category?",
                "accepted_values": ["a thyroid medication"],
                "decision_requirements": {
                    decision["decision_id"]: "a thyroid medication",
                },
            }]

    artifact = build_utility_artifact(
        frozen,
        Adapter(),
        {"d1": "Synthroid"},
        threshold_manifest={
            "family_budgets": {"context": 0.6, "delivered": 0.4},
            "cost_budgets": COST_BUDGETS,
        },
        pins={},
        reader=lambda questions, context: ["a thyroid medication"],
        render_action_vector=lambda doc_id, vector: "generalized",
    )

    assert artifact["assertions"] == {}
    assert artifact["rejections"]["summary_by_reason"] == {"unsupported": 1}


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


def test_context_validation_refreshes_repeated_reader_trials():
    assertions = [{
        "assertion_id": "a1",
        "family": "context",
        "question": "Which category is documented?",
        "accepted_values": ["endocrine"],
    }]
    refreshes = []

    def reader(questions, context, *, refresh=False):
        refreshes.append(refresh)
        return ["NONE" if context == "placeholder" else "endocrine"]

    accepted, _evidence = validate_context_assertions(
        assertions,
        original_context="original",
        representative_context="generalized",
        placeholder_context="placeholder",
        reader=reader,
        stability_repetitions=3,
    )

    assert [row["assertion_id"] for row in accepted] == ["a1"]
    assert refreshes == [False, False, False, True, True, True, True, True, True]


def test_context_validation_repeated_trials_require_refresh_capability():
    assertions = [{
        "assertion_id": "a1",
        "family": "context",
        "question": "Which category is documented?",
        "accepted_values": ["endocrine"],
    }]

    def reader(questions, context):
        return ["NONE" if context == "placeholder" else "endocrine"]

    with pytest.raises(TypeError, match="refresh"):
        validate_context_assertions(
            assertions,
            original_context="original",
            representative_context="generalized",
            placeholder_context="placeholder",
            reader=reader,
            stability_repetitions=2,
        )


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

    def reader(questions, context, *, refresh=False):
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
            "cost_budgets": COST_BUDGETS,
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
    assert set(result) == {"component_scores"}


def test_runtime_renders_option_questions_like_validation_permutation_zero():
    assertion = {
        "assertion_id": "c1",
        "doc_id": "d1",
        "family": "context",
        "question": "Which category is documented?",
        "options": ["respiratory", "endocrine", "musculoskeletal"],
        "accepted_values": ["endocrine"],
        "weight": 1.0,
    }
    artifact = {
        "documents": {"d1": {"utility_weight_denominator": 1.0}},
        "assertions": {"c1": assertion},
    }
    calls = []

    result = score_utility(
        artifact,
        "d1",
        doc_p="generalized document",
        out_final="unused",
        reader=lambda questions, context: calls.append(list(questions)) or ["endocrine"],
    )

    assert calls == [[qa_builder._permuted_reader_question(assertion, 0)]]
    assert result["component_scores"] == {"c1": 1.0}


def test_packaged_runtime_option_order_matches_prepackaging_validation():
    candidate = {
        "family": "context",
        "scope": "global",
        "subtype": "semantic_property",
        "occurrence_ids": [],
        "group_id": "condition:category",
        "question": "Which category is documented?",
        "options": ["respiratory", "endocrine", "musculoskeletal"],
        "accepted_values": ["endocrine"],
    }
    validation_questions = []

    def validation_reader(questions, context):
        validation_questions.extend(questions)
        return ["NONE" if context == "placeholder" else "endocrine"]

    accepted, _evidence = validate_context_assertions(
        [candidate],
        original_context="original",
        representative_context="representative",
        placeholder_context="placeholder",
        reader=validation_reader,
    )
    artifact = package_utility_artifact(
        {
            "environment_hash": "env-v1",
            "documents": {"d1": {"occurrences": [], "decisions": []}},
        },
        {"d1": accepted},
        family_budgets={"context": 0.6, "delivered": 0.4},
        pins={},
    )
    runtime_questions = []

    score_utility(
        artifact,
        "d1",
        doc_p="representative",
        out_final="unused",
        reader=lambda questions, context: runtime_questions.extend(questions) or ["endocrine"],
    )

    assert runtime_questions == [validation_questions[0]]


def test_batched_context_reader_uses_one_model_request_for_all_questions():
    class Client:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt, *, refresh=False):
            self.prompts.append(prompt)
            return '{"answers":["endocrine condition","thyroid medication"]}'

    client = Client()
    reader = BatchedContextReader(client=client)

    answers = reader(["What condition category?", "What treatment category?"], "note")

    assert answers == ["endocrine condition", "thyroid medication"]
    assert len(client.prompts) == 1
    assert "What condition category?" in client.prompts[0]
    assert "What treatment category?" in client.prompts[0]


def test_context_reader_pin_covers_live_prompt_schema_endpoint_and_decoding():
    pin = qa_builder.context_reader_pin()

    assert pin["pin_version"] == "qa-context-reader-v1"
    assert pin["model"] == qa_builder.QA_MODEL
    assert pin["base_url"] == qa_builder.QA_BASE_URL
    assert pin["prompt"]["version"]
    assert pin["prompt"]["sha256"].startswith("sha256:")
    assert pin["response_schema"]["version"]
    assert pin["response_schema"]["schema"] == {
        "type": "object",
        "required": ["answers"],
        "properties": {"answers": {"type": "array", "items": {"type": "string"}}},
        "additionalProperties": False,
    }
    assert pin["decoding"] == {
        "temperature": 0.0,
        "max_tokens": 512,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }


def test_batched_context_reader_refresh_reaches_generate():
    calls = []

    class Client:
        def generate(self, prompt, *, refresh=False):
            calls.append(refresh)
            return '{"answers":["endocrine"]}'

    reader = BatchedContextReader(client=Client())

    assert reader(["What category?"], "note", refresh=True) == ["endocrine"]
    assert calls == [True]


def test_roundtrip_utility_artifact_scores_doc_p_and_out_final(monkeypatch):
    class Remote:
        def generate(self, prompt):
            return "REMOTE OUT_P"

    calls = []

    def fake_score(artifact, doc_id, *, doc_p, out_final, reader, reader_refresh=False):
        calls.append((artifact, doc_id, doc_p, out_final, reader, reader_refresh))
        return {"component_scores": {"c1": 0.75}}

    monkeypatch.setattr(roundtrip, "_remote", lambda: Remote())
    monkeypatch.setattr(roundtrip, "invert", lambda out_p, R: ("DELIVERED OUT_FINAL", None))
    monkeypatch.setattr(roundtrip, "score_utility", fake_score, raising=False)

    artifact = {
        "documents": {"d1": {
            "assertion_ids": ["c1"],
            "utility_weight_denominator": 1.0,
        }},
        "assertions": {"c1": {
            "assertion_id": "c1",
            "doc_id": "d1",
            "status": "accepted",
            "weight": 1.0,
        }},
    }
    result = roundtrip.roundtrip_batch([{
        "corpus": "clinical",
        "doc_id": "d1",
        "doc_p": "ROLLOUT DOC_P",
        "R": [],
        "probes": [],
        "utility_artifact": artifact,
    }], workers=1, reader_refresh=True)[0]

    assert result["recall"] == pytest.approx(0.75)
    assert result["component_scores"] == {"c1": 0.75}
    assert calls[0][1:4] == ("d1", "ROLLOUT DOC_P", "DELIVERED OUT_FINAL")
    assert calls[0][-1] is True


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
    assert first["documents"]["d1"]["weight_groups"] == {
        "context": {
            "hypothyroid:category": {
                "assertion_ids": [next(assertion_id for assertion_id, row in first["assertions"].items()
                                  if row["family"] == "context")],
                "weight": pytest.approx(0.6),
            }
        },
        "delivered": {
            "document:age": {
                "assertion_ids": [next(assertion_id for assertion_id, row in first["assertions"].items()
                                  if row["family"] == "delivered")],
                "weight": pytest.approx(0.4),
            }
        },
    }


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
                "family": "context", "scope": "linked", "subtype": "semantic_property",
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
            "cost_budgets": COST_BUDGETS,
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
    assert context["expected_action_support"]["joint_anchor_hash"].startswith("sha256:")
    assert context["expected_action_support"]["property_level"] == {"dec1": "endocrine"}
    assert "property_levels" not in context["expected_action_support"]
    assert artifact["teacher_pin"]["production"] is False
    assert context["evidence"]["validation"]["scores"] == {
        "original": 1.0,
        "representative": 1.0,
        "placeholder": 0.0,
    }
    assert artifact["threshold_manifest"] == {
        "family_budgets": {"context": 0.6, "delivered": 0.4},
        "cost_budgets": COST_BUDGETS,
        "min_context_assertions": 1,
        "reader_threshold": 1.0,
        "reader_stability_repetitions": 1,
        "reader_option_permutations": 1,
        "reader_stability_threshold": 1.0,
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
    assert relations == [{"relation": "treated_with"}]


def test_openrouter_relation_teacher_requires_content_addressed_cache(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.delenv("CLOAK_LLM_CACHE", raising=False)

    with pytest.raises(ValueError, match="CLOAK_LLM_CACHE"):
        qa_builder.OpenRouterRelationTeacher()


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


def test_relation_prompt_exposes_only_closed_ids_properties_and_source():
    prompt = relation_teacher_prompt(
        "aci/D2N002",
        "Hypothyroidism is treated with Synthroid.",
        _relation_environment(),
        authoritative_reference=(
            "Assessment: hypothyroidism. Plan: continue Synthroid and monitor thyroid labs."
        ),
    )

    assert "aci/D2N002" in prompt
    assert "o-condition" in prompt
    assert "an endocrine condition" in prompt
    assert "treated_with" in prompt
    assert "AUTHORITATIVE REFERENCE EVIDENCE" in prompt
    assert "continue Synthroid and monitor thyroid labs" in prompt
    assert "Hypothyroidism is treated with Synthroid." in prompt
    assert "accepted_values" not in prompt


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
    assert rejected[0]["reason"] == reason


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
    assert rejected == [{"proposal_index": 0, "reason": reason}]


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

    delivered = [row for row in candidates if row["family"] == "delivered"]
    assert {row["scoring_contract"]["value"] for row in delivered} == {
        "62-year-old", "male", "hypothyroidism", "Synthroid"
    }
    condition = next(row for row in delivered
                     if row["scoring_contract"]["value"] == "hypothyroidism")
    assert condition["scope"] == "linked"
    assert condition["occurrence_ids"] == ["o-condition"]


def test_builder_prefers_linked_aci_age_fact_over_global_duplicate():
    environment = copy.deepcopy(_relation_environment())
    environment["occurrences"].append({
        "occurrence_id": "o-age",
        "decision_id": "dec-age",
        "surface": "62-year-old",
        "runtime_type": "age",
        "controlled": True,
    })
    environment["decisions"].append({
        "decision_id": "dec-age",
        "controlled": True,
        "runtime_type": "age",
        "canonical_key": "62-year-old",
    })
    adapter = AciTaskAdapter({
        "aci/D2N002": "A 62-year-old male has hypothyroidism and takes Synthroid."
    })
    candidates = adapter.deterministic_candidates(
        "aci/D2N002",
        "A 62-year-old male has hypothyroidism and takes Synthroid.",
        environment,
    )
    artifact = package_utility_artifact(
        {"environment_hash": "env-v1", "documents": {"aci/D2N002": environment}},
        {"aci/D2N002": candidates},
        family_budgets={"context": 0.6, "delivered": 0.4},
        pins={},
    )
    age_rows = [
        row for row in artifact["assertions"].values()
            if row.get("scoring_contract", {}).get("value") == "62-year-old"
    ]

    assert len(age_rows) == 1
    assert age_rows[0]["scope"] == "linked"
    assert age_rows[0]["occurrence_ids"] == ["o-age"]


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


def test_package_excludes_uncontrolled_occurrences_from_credit_mapping():
    environment = {
        "environment_hash": "env-hash",
        "documents": {"d1": {
            "occurrences": [
                {"occurrence_id": "o1", "decision_id": "dec1", "controlled": True},
                {"occurrence_id": "o-uncontrolled", "decision_id": None,
                 "controlled": False},
            ],
            "decisions": [
                {"decision_id": "dec1", "controlled": True,
                 "runtime_type": "drug", "canonical_key": "synthroid"},
                {"decision_id": "uncontrolled", "controlled": False,
                 "runtime_type": "drug", "canonical_key": "unrelated"},
            ],
        }},
    }
    global_candidate = {
        "family": "delivered", "scope": "global", "subtype": "content",
        "occurrence_ids": [], "group_id": "document:unlinked-evidence",
        "scoring_contract": {"kind": "contains", "value": "unrelated"},
    }

    artifact = package_utility_artifact(
        environment,
        {"d1": [global_candidate]},
        family_budgets={"context": 0.6, "delivered": 0.4},
        pins={},
    )

    assert artifact["documents"]["d1"]["occurrence_to_decision"] == {"o1": "dec1"}
    assert artifact["documents"]["d1"]["decision_keys"] == [{
        "decision_id": "dec1", "runtime_type": "drug", "canonical_key": "synthroid",
    }]
    assertion_id = artifact["documents"]["d1"]["assertion_ids"][0]
    assert provisional_advantages(
        [{assertion_id: 1.0}, {assertion_id: 0.0}],
        artifact,
        artifact["documents"]["d1"]["occurrence_to_decision"],
        doc_id="d1",
    ) == pytest.approx({(0, "dec1"): 0.4, (1, "dec1"): -0.4})
    with pytest.raises(ValueError, match="uncontrolled occurrence"):
        package_utility_artifact(
            environment,
            {"d1": [{**global_candidate, "scope": "linked",
                      "occurrence_ids": ["o-uncontrolled"]}]},
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
    assert rejected == [{"proposal_index": 0, "reason": "invalid_argument_types"}]


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
            "cost_budgets": COST_BUDGETS,
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


def test_builder_emits_authoritative_transitive_pins_and_manifest_identity():
    frozen = {"environment_hash": "env-v1", "documents": {}}

    class Adapter:
        task_pin = {"adapter": "test-task", "version": "v1"}

    artifact = build_utility_artifact(
        frozen,
        Adapter(),
        {},
        threshold_manifest={
            "family_budgets": {"context": 0.6, "delivered": 0.4},
            "cost_budgets": COST_BUDGETS,
        },
        pins={"task_pin": "forged", "reader_pin": "forged"},
        reader=lambda questions, context: [],
        render_action_vector=lambda doc_id, vector: "unused",
    )

    assert artifact["task_pin"] == Adapter.task_pin
    assert artifact["builder_pin"]["version"]
    assert artifact["teacher_pin"]["enabled"] is False
    assert artifact["reader_pin"]["production"] is False
    assert artifact["reader_pin"] != qa_builder.context_reader_pin()
    assert artifact["scorer_pin"]["reader"] == qa_builder.context_reader_pin()
    assert artifact["gate_manifest_hash"] == qa_builder._stable_hash(
        artifact["threshold_manifest"]
    )
    assert artifact["threshold_manifest_pin"] == {
        "schema": "qa-threshold-manifest-v1",
        "sha256": artifact["gate_manifest_hash"],
    }


def test_builder_stamps_exact_production_reader_and_teacher_dependencies():
    artifact = build_utility_artifact(
        {"environment_hash": "env-v1", "documents": {}},
        AciTaskAdapter({}),
        {},
        threshold_manifest={
            "family_budgets": {"context": 0.6, "delivered": 0.4},
            "cost_budgets": COST_BUDGETS,
        },
        pins={},
        reader=qa_builder.read_context_batch,
        render_action_vector=lambda doc_id, vector: "unused",
        relation_teacher=object.__new__(OpenRouterRelationTeacher),
    )

    assert artifact["reader_pin"] == qa_builder.context_reader_pin()
    assert artifact["teacher_pin"] == qa_builder.relation_teacher_pin(True)


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
    assert rejected == [{"proposal_index": 0, "reason": "protected_locator"}]


def _valid_relation_proposal():
    source = "Hypothyroidism is treated with Synthroid."
    return source, {
        "relation": "treated_with",
        "argument_occurrence_ids": ["o-condition", "o-drug"],
        "support_properties": {
            "o-condition": "an endocrine condition",
            "o-drug": "a thyroid medication",
        },
        "answer_occurrence_id": "o-drug",
        "answer_property": "a thyroid medication",
        "question": "What treatment category is used for the endocrine disorder?",
        "evidence_quote": source,
    }


def test_relational_compiler_rejects_partial_answer_clue():
    source, proposal = _valid_relation_proposal()
    proposal["question"] = "Which thyroid category is used for the endocrine disorder?"

    accepted, rejected = compile_relational_assertions(
        "aci/D2N002", source, _relation_environment(), [proposal]
    )

    assert accepted == []
    assert rejected == [{"proposal_index": 0, "reason": "answer_leakage"}]


def test_relational_compiler_rejects_frozen_answer_alias():
    source, proposal = _valid_relation_proposal()
    environment = copy.deepcopy(_relation_environment())
    environment["decisions"][1]["actions"][0]["aliases"] = ["levothyroxine class"]
    proposal["question"] = "Which levothyroxine class is used for the endocrine disorder?"

    accepted, rejected = compile_relational_assertions(
        "aci/D2N002", source, environment, [proposal]
    )

    assert accepted == []
    assert rejected == [{"proposal_index": 0, "reason": "answer_leakage"}]


def test_relational_compiler_rejects_unlinked_protected_surface_or_alias():
    source, proposal = _valid_relation_proposal()
    environment = copy.deepcopy(_relation_environment())
    environment["occurrences"].append({
        "occurrence_id": "o-unlinked",
        "decision_id": None,
        "surface": "Boston",
        "aliases": ["Beantown"],
        "runtime_type": "location",
        "controlled": False,
    })
    proposal["question"] = "Which treatment category is used near Beantown?"

    accepted, rejected = compile_relational_assertions(
        "aci/D2N002", source, environment, [proposal]
    )

    assert accepted == []
    assert rejected == [{"proposal_index": 0, "reason": "protected_locator"}]


def test_relational_compiler_lints_protected_leakage_in_options():
    source, proposal = _valid_relation_proposal()
    proposal["options"] = ["supportive care", "Synthroid", "physical therapy"]

    accepted, rejected = compile_relational_assertions(
        "aci/D2N002", source, _relation_environment(), [proposal]
    )

    assert accepted == []
    assert rejected == [{"proposal_index": 0, "reason": "protected_locator"}]


def test_relational_compiler_ignores_one_character_protected_tokens():
    source, proposal = _valid_relation_proposal()
    environment = copy.deepcopy(_relation_environment())
    environment["occurrences"].append({
        "occurrence_id": "o-short",
        "decision_id": None,
        "surface": "A",
        "aliases": ["I"],
        "runtime_type": "misc",
        "controlled": False,
    })

    accepted, rejected = compile_relational_assertions(
        "aci/D2N002", source, environment, [proposal]
    )

    assert rejected == []
    assert len(accepted) == 1


def test_relational_compiler_rejects_exact_short_protected_phrase():
    source, proposal = _valid_relation_proposal()
    environment = copy.deepcopy(_relation_environment())
    environment["occurrences"].append({
        "occurrence_id": "o-hiv",
        "decision_id": None,
        "surface": "HIV",
        "aliases": [],
        "runtime_type": "health-condition",
        "controlled": False,
    })
    proposal["question"] = "Which treatment category is used for HIV?"

    accepted, rejected = compile_relational_assertions(
        "aci/D2N002", source, environment, [proposal]
    )

    assert accepted == []
    assert rejected == [{"proposal_index": 0, "reason": "protected_locator"}]


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

    assert result == {"component_scores": {"d1": 1.0}}


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
            "aliases": ["Other name"],
        }]},
    )

    occurrence = frozen["documents"]["d1"]["occurrences"][0]
    assert occurrence["surface"] == "Unrelated"
    assert occurrence["controlled"] is False
    assert occurrence["decision_id"] is None
    assert occurrence["aliases"] == ["Other name"]


def _semantic_fixture(source="Arthritis is treated with Tylenol."):
    environment = freeze_ranker_environment({
        "corpora": {"clinical": {"d1": {"spans": [
            {"surface": "Arthritis", "type": "health-condition", "start": 0, "end": 9,
             "aliases": ["joint disease"], "actions": [
                 {"fill": "a disease", "mode": "level", "aset": 100.0},
                 {"fill": "Arthritis", "mode": "level", "keep": True, "aset": 1.0},
                 {"fill": None, "mode": "placeholder"},
             ]},
            {"surface": "Tylenol", "type": "drug", "start": 25, "end": 32,
             "actions": [
                 {"fill": "a medicine", "mode": "level", "aset": 100.0},
                 {"fill": "Tylenol", "mode": "level", "keep": True, "aset": 1.0},
                 {"fill": None, "mode": "placeholder"},
             ]},
        ]}}},
    }, source_documents={"d1": source}, authoritative_references={"d1": ""})
    return source, environment


def test_aci_deterministic_semantic_candidate_masks_all_protected_locators():
    source, environment = _semantic_fixture()
    candidates = AciTaskAdapter({"d1": ""}).deterministic_candidates(
        "d1", source, environment["documents"]["d1"]
    )
    context = next(row for row in candidates if row["family"] == "context")

    assert context["subtype"] == "semantic_property"
    assert "[BLANK]" in context["question"] and "[SENSITIVE]" in context["question"]
    assert "arthritis" not in context["question"].lower()
    assert "tylenol" not in context["question"].lower()
    assert context["accepted_values"] == ["a disease"]


def test_masked_local_context_blanks_selected_target_and_masks_repeated_target_values():
    document = "Dr. Kumar and Doctor Kumar improve with Tylenol."
    target = {"surface": "Dr. Kumar", "start": 0, "end": 9}

    context = qa_builder._masked_local_context(
        document, target, ["Doctor Kumar", "Tylenol"]
    )

    assert context.count("[BLANK]") == 1
    assert context.count("[SENSITIVE]") == 2
    assert "kumar" not in context.lower()
    assert "tylenol" not in context.lower()


def test_deterministic_semantic_candidate_accepts_through_fake_reader():
    source, environment = _semantic_fixture()
    decisions = environment["documents"]["d1"]["decisions"]

    def render(_doc_id, vector):
        selected = [next(action for action in decision["actions"]
                         if action["action_id"] == vector[decision["decision_id"]])
                    for decision in decisions]
        return "all placeholders" if all(action["mode"] == "placeholder" for action in selected) else source

    artifact = build_utility_artifact(
        environment, AciTaskAdapter({"d1": ""}), {"d1": source},
        threshold_manifest={"family_budgets": {"context": 0.6, "delivered": 0.4},
                            "cost_budgets": COST_BUDGETS, "min_context_assertions": 1},
        pins={},
        reader=lambda questions, context, **_kwargs: [
            "" if context == "all placeholders" else "a disease" for _ in questions
        ],
        render_action_vector=render,
    )

    assert any(row["subtype"] == "semantic_property" for row in artifact["assertions"].values())


def test_aci_delivered_facts_fall_back_to_explicit_source_with_authority_metadata():
    source = "A 62-year-old male with arthritis is seen."
    environment = freeze_ranker_environment({
        "corpora": {"clinical": {"d1": {"spans": [{
            "surface": "arthritis", "type": "health-condition", "start": 25, "end": 34,
            "actions": [{"fill": "a disease", "mode": "level", "aset": 100.0},
                        {"fill": "arthritis", "mode": "level", "keep": True, "aset": 1.0},
                        {"fill": None, "mode": "placeholder"}],
        }]}}},
    })
    rows = AciTaskAdapter({"d1": "brief reference omits facts"}).deterministic_candidates(
        "d1", source, environment["documents"]["d1"]
    )
    delivered = [row for row in rows if row["family"] == "delivered"]

    assert {row["scoring_contract"]["value"] for row in delivered} >= {"arthritis", "62-year-old", "male"}
    assert all(row["evidence"]["authority"] == "doc_orig" for row in delivered)


def test_context_reader_failure_marks_only_that_document_build_failed():
    source, environment = _semantic_fixture()
    artifact = build_utility_artifact(
        environment, AciTaskAdapter({"d1": ""}), {"d1": source},
        threshold_manifest={"family_budgets": {"context": 0.6, "delivered": 0.4},
                            "cost_budgets": COST_BUDGETS, "min_context_assertions": 1},
        pins={}, reader=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("reader down")),
        render_action_vector=lambda *_args: "placeholder context",
    )

    state = artifact["documents"]["d1"]
    assert state["measurement_state"] == "build_failed"
    assert state["build_failure"]["stage"] == "context_reader"
