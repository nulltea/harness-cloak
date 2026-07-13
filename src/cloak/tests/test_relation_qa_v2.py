import pytest
import cloak.train.qa_builder as qa_builder

from cloak.train.qa_builder import (
    OpenRouterRelationTeacher,
    compile_relational_assertions,
    relation_context_candidates,
    relation_evidence_windows,
    relation_teacher_response_format,
    relation_teacher_prompt,
)


def _environment(source):
    return {
        "occurrences": [
            {"occurrence_id": "condition", "decision_id": "d-condition", "surface": "Hypothyroidism", "start": 0, "end": 14, "runtime_type": "health-condition"},
            {"occurrence_id": "drug", "decision_id": "d-drug", "surface": "Synthroid", "start": source.index("Synthroid"), "end": source.index("Synthroid") + 9, "runtime_type": "drug"},
        ],
        "decisions": [
            {"decision_id": "d-condition", "actions": [{"mode": "level", "legal": True, "entails": ["endocrine condition"]}]},
            {"decision_id": "d-drug", "actions": [{"mode": "level", "legal": True, "entails": ["thyroid medication"]}]},
        ],
    }


def _proposal(source):
    return {
        "relation": "prescribed_with",
        "arguments": [
            {"role": "subject", "kind": "linked", "occurrence_id": "condition", "support_property": "endocrine condition"},
            {"role": "object", "kind": "linked", "occurrence_id": "drug", "support_property": "thyroid medication"},
        ],
        "question": "Which medication class is used for the endocrine disorder?",
        "accepted_answers": ["hormone replacement therapy"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
        "evidence_quote": source,
    }


def test_teacher_authored_relation_qa_is_preserved_and_drug_is_prescribed_not_treated():
    source = "Hypothyroidism is treated with Synthroid."
    accepted, rejected = compile_relational_assertions("d2", source, _environment(source), [_proposal(source)])

    assert rejected == []
    assert accepted[0]["relation"] == "prescribed_with"
    assert accepted[0]["question"] == _proposal(source)["question"]
    assert accepted[0]["accepted_values"] == ["hormone replacement therapy"]
    assert accepted[0]["decision_requirements"] == {"d-condition": "endocrine condition", "d-drug": "thyroid medication"}


def test_treated_with_rejects_drug_and_context_argument_stays_unlinked():
    source = "Hypothyroidism is treated with Synthroid. Arthritis was referred to physical therapy."
    environment = _environment(source)
    arthritis_start = source.index("Arthritis")
    environment["occurrences"].append({"occurrence_id": "arthritis", "decision_id": "d-arthritis", "surface": "Arthritis", "start": arthritis_start, "end": arthritis_start + 9, "runtime_type": "health-condition"})
    environment["decisions"].append({"decision_id": "d-arthritis", "actions": [{"mode": "level", "legal": True, "entails": ["joint condition"]}]})
    bad = {**_proposal(source), "relation": "treated_with"}
    physical = next(
        candidate for candidate in relation_context_candidates(source)
        if candidate["literal"] == "physical therapy"
    )
    context = {
        "relation": "referred_to",
        "arguments": [
            {"role": "subject", "kind": "linked", "occurrence_id": "arthritis", "support_property": "joint condition"},
            {"role": "object", "kind": "context", "context_candidate_id": physical["context_candidate_id"]},
        ],
        "question": "Which referral modality was selected?",
        "accepted_answers": ["physical therapy service"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
        "evidence_quote": "Arthritis was referred to physical therapy.",
        "evidence_start": source.index("Arthritis"),
    }
    accepted, rejected = compile_relational_assertions("d2", source, environment, [bad, context])

    assert rejected[0]["detail_reason"] == "invalid_argument_types"
    assert accepted[0]["occurrence_ids"] == ["arthritis"]
    assert accepted[0]["decision_requirements"] == {"d-arthritis": "joint condition"}


def test_prompt_requests_exhaustive_accounting_and_teacher_semantic_qa():
    source = "Hypothyroidism is treated with Synthroid."
    prompt = relation_teacher_prompt("d2", source, _environment(source))

    assert "candidate_accounting" in prompt
    assert "accepted answers" in prompt
    assert "prescribed_with" in prompt
    assert "treated_with: condition or diagnosis -> medical procedure" in prompt
    assert "EVIDENCE CARDS" in prompt


def test_context_candidate_inventory_is_finite_typed_but_not_globally_prompted():
    source = (
        "Hypothyroidism is treated with Synthroid. "
        "Order some thyroid labs. Arthritis was referred to physical therapy."
    )
    candidates = relation_context_candidates(source)
    by_literal = {candidate["literal"]: candidate for candidate in candidates}
    prompt = relation_teacher_prompt("d2", source, _environment(source))

    assert by_literal["thyroid labs"]["runtime_type"] == "test"
    assert by_literal["physical therapy"]["runtime_type"] == "procedure"
    assert all(candidate["context_candidate_id"] not in prompt for candidate in candidates)
    assert "CONTEXT CANDIDATE INVENTORY" not in prompt


def test_teacher_response_schema_binds_roles_and_candidate_ledger_to_inventory():
    source = (
        "Hypothyroidism is treated with Synthroid. "
        "Order some thyroid labs. Arthritis was referred to physical therapy."
    )
    environment = _environment(source)
    response_format = relation_teacher_response_format(environment, source)
    schema = response_format["json_schema"]["schema"]
    arguments = schema["properties"]["relations"]["items"]["properties"]["arguments"]
    ledger = schema["properties"]["candidate_accounting"]
    expected_labels = [row["span_label"] for row in qa_builder.relation_teacher_span_inventory(environment)]

    linked, context = arguments["items"]["anyOf"]
    assert linked["properties"]["role"]["enum"] == ["subject", "object"]
    assert linked["properties"]["kind"]["const"] == "linked"
    assert context["properties"]["kind"]["const"] == "context"
    assert linked["properties"]["literal"]["const"] is None
    assert context["properties"]["span_label"]["const"] is None
    assert "evidence_window_id" not in schema["properties"]["relations"]["items"]["properties"]
    assert ledger["minItems"] == ledger["maxItems"] == len(expected_labels)
    assert ledger["items"] is False
    assert [row["properties"]["candidate_label"]["const"] for row in ledger["prefixItems"]] == expected_labels


def test_context_argument_must_reference_inventory_candidate():
    source = "Arthritis was referred to physical therapy."
    arthritis_start = source.index("Arthritis")
    environment = {
        "occurrences": [{"occurrence_id": "arthritis", "decision_id": "d-arthritis", "surface": "Arthritis", "start": arthritis_start, "end": arthritis_start + 9, "runtime_type": "health-condition"}],
        "decisions": [{"decision_id": "d-arthritis", "actions": [{"mode": "level", "legal": True, "entails": ["joint condition"]}]}],
    }
    proposal = {
        "relation": "referred_to",
        "arguments": [
            {"role": "subject", "kind": "linked", "occurrence_id": "arthritis", "support_property": "joint condition"},
            {"role": "object", "kind": "context", "context_candidate_id": "not-inventory"},
        ],
        "question": "Which referral modality was selected?",
        "accepted_answers": ["rehabilitation service"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
        "evidence_quote": source,
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert accepted == []
    assert rejected[0]["detail_reason"] == "unknown_context_candidate"


def test_teacher_evidence_window_binds_arguments_to_one_exact_source_span():
    source = "Hypothyroidism is treated with Synthroid."
    environment = _environment(source)
    windows = relation_evidence_windows(source, environment)
    assert len(windows) == 1
    assert {
        (row["relation"], row["subject_candidate_id"], row["object_candidate_id"])
        for row in windows[0]["eligible_pairs"]
    } >= {("prescribed_with", "condition", "drug")}
    proposal = {
        **_proposal(source),
        "evidence_window_id": windows[0]["evidence_window_id"],
    }
    proposal.pop("evidence_quote")

    accepted, rejected = compile_relational_assertions(
        "d2", source, environment, [proposal]
    )

    assert rejected == []
    assert accepted[0]["evidence"]["source_span"] == {
        "start": 0, "end": len(source),
        "quote_hash": qa_builder._stable_hash(source),
    }


def test_evidence_window_allows_adjacent_condition_then_ordered_context_with_cue():
    source = "Hypothyroidism is stable. Order thyroid labs. Synthroid is continued."
    environment = _environment(source)
    environment["occurrences"] = [environment["occurrences"][0]]
    environment["decisions"] = [environment["decisions"][0]]
    labs = next(
        row for row in relation_context_candidates(source)
        if row["literal"] == "thyroid labs"
    )
    window = next(
        row for row in relation_evidence_windows(source, environment)
        if {"condition", labs["context_candidate_id"]}.issubset(row["candidate_ids"])
    )
    proposal = {
        "relation": "monitored_by",
        "arguments": [
            {"role": "subject", "kind": "linked", "occurrence_id": "condition", "support_property": "endocrine condition"},
            {"role": "object", "kind": "context", "context_candidate_id": labs["context_candidate_id"]},
        ],
        "question": "Which testing modality was ordered for the endocrine condition?",
        "accepted_answers": ["laboratory evaluation"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
        "evidence_window_id": window["evidence_window_id"],
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == []
    assert accepted[0]["relation"] == "monitored_by"


def test_openrouter_teacher_requires_complete_candidate_accounting(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt):
            return '{"relations": [], "candidate_accounting": []}'

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("CLOAK_LLM_CACHE", "/tmp/test-cache")
    monkeypatch.setattr("cloak.llm.LLMClient", FakeClient)

    proposals = OpenRouterRelationTeacher().propose("prompt")

    assert proposals == []
    assert proposals.candidate_accounting == []


def test_openrouter_teacher_reports_an_empty_provider_completion_explicitly(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.last_completion_state = {"outcome": "no_choices"}

        def generate(self, prompt):
            return ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("CLOAK_LLM_CACHE", "/tmp/test-cache")
    monkeypatch.setattr("cloak.llm.LLMClient", FakeClient)

    with pytest.raises(ValueError, match="teacher_no_choices"):
        OpenRouterRelationTeacher().propose("prompt")


def test_openrouter_teacher_accepts_a_document_bound_response_schema(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt, **kwargs):
            captured.update(kwargs)
            return '{"relations": [], "candidate_accounting": []}'

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("CLOAK_LLM_CACHE", "/tmp/test-cache")
    monkeypatch.setattr("cloak.llm.LLMClient", FakeClient)

    bound_format = {"type": "json_schema", "json_schema": {"name": "bound"}}
    OpenRouterRelationTeacher().propose("prompt", response_format=bound_format)

    assert captured["response_format"] == bound_format


def test_candidate_accounting_must_cover_exactly_the_prompted_inventory():
    source = "Hypothyroidism is treated with Synthroid. Order thyroid labs."
    environment = _environment(source)
    candidate_ids = [row["span_label"] for row in qa_builder.relation_teacher_span_inventory(environment)]
    accounting = [
        {"candidate_label": candidate_id, "state": "exhausted_no_relation", "reason": "No explicit relation."}
        for candidate_id in candidate_ids
    ]

    accepted = qa_builder._validated_candidate_accounting(accounting, environment, source)

    assert {row["candidate_label"] for row in accepted} == set(candidate_ids)
    with pytest.raises(ValueError, match="exactly one record"):
        qa_builder._validated_candidate_accounting(accounting[:-1], environment, source)


def test_v4_prompt_and_schema_use_source_labels_not_internal_inventory():
    source = "Hypothyroidism is treated with Synthroid."
    environment = _environment(source)

    prompt = relation_teacher_prompt("d2", source, environment)
    schema = relation_teacher_response_format(environment, source)["json_schema"]["schema"]
    argument_variants = schema["properties"]["relations"]["items"]["properties"]["arguments"]["items"]["anyOf"]

    assert "[S1: Hypothyroidism | condition | levels: endocrine condition]" in prompt
    assert "[S2: Synthroid | drug | levels: thyroid medication]" in prompt
    assert "OCCURRENCE INVENTORY" not in prompt
    assert "SOURCE EVIDENCE WINDOWS" not in prompt
    assert '"occurrence_id"' not in prompt
    assert "evidence_window_id" not in schema["properties"]["relations"]["items"]["properties"]
    assert argument_variants[0]["properties"]["span_label"]["enum"] == ["S1", "S2"]
    assert argument_variants[1]["properties"]["literal"]["type"] == "string"
    assert schema["properties"]["candidate_accounting"]["prefixItems"][0]["properties"]["candidate_label"]["const"] == "S1"


def test_v4_compiler_derives_anchor_and_resolves_exact_context_literal():
    source = "Hypothyroidism is monitored by thyroid labs."
    environment = {
        "occurrences": [{
            "occurrence_id": "condition", "decision_id": "d-condition",
            "surface": "Hypothyroidism", "start": 0, "end": 14,
            "runtime_type": "health-condition",
        }],
        "decisions": [{
            "decision_id": "d-condition",
            "actions": [{"mode": "level", "legal": True, "entails": ["endocrine condition"]}],
        }],
    }
    proposal = {
        "relation": "monitored_by",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S1", "support_property": "endocrine condition", "literal": None},
            {"role": "object", "kind": "context", "span_label": None, "support_property": None, "literal": "thyroid labs"},
        ],
        "question": "Which testing modality follows the endocrine condition?",
        "accepted_answers": ["thyroid labs"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == []
    assert accepted[0]["relation"] == "monitored_by"
    assert accepted[0]["evidence"]["source_span"]["start"] == 0
    assert accepted[0]["evidence"]["source_span"]["end"] == len(source)


def test_v4_compiler_uses_a_bounded_plan_section_anchor_for_treatment():
    source = (
        "Arthritis.\n"
        "• Medical Reasoning: Symptoms have worsened.\n"
        "• Additional Testing: Order a panel.\n"
        "• Medical Treatment: Initiate Ultram 50 mg every 6 hours as needed."
    )
    ultram_start = source.index("Ultram")
    environment = {
        "occurrences": [
            {"occurrence_id": "arthritis", "decision_id": "d-arthritis", "surface": "Arthritis", "start": 0, "end": 9, "runtime_type": "health-condition"},
            {"occurrence_id": "ultram", "decision_id": "d-ultram", "surface": "Ultram", "start": ultram_start, "end": ultram_start + 6, "runtime_type": "drug"},
        ],
        "decisions": [
            {"decision_id": "d-arthritis", "actions": [{"mode": "level", "legal": True, "entails": ["joint condition"]}]},
            {"decision_id": "d-ultram", "actions": [{"mode": "level", "legal": True, "entails": ["opioid analgesic"]}]},
        ],
    }
    proposal = {
        "relation": "prescribed_with",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S1", "support_property": "joint condition"},
            {"role": "object", "kind": "linked", "span_label": "S2", "support_property": "opioid analgesic"},
        ],
        "question": "What medication class was started for the joint disorder?",
        "accepted_answers": ["opioid analgesic"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == []
    assert accepted[0]["evidence"]["source_span"]["quote_hash"] == qa_builder._stable_hash(source)
