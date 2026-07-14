import json

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
    arguments = schema["properties"]["context_relations"]["items"]["properties"]["arguments"]
    ledger = schema["properties"]["candidate_accounting"]
    expected_labels = [row["span_label"] for row in qa_builder.relation_teacher_span_inventory(environment)]

    linked, context = arguments["anyOf"][0]["prefixItems"]
    assert linked["properties"]["role"]["const"] == "subject"
    assert context["properties"]["role"]["const"] == "object"
    assert linked["properties"]["kind"]["const"] == "linked"
    assert context["properties"]["kind"]["const"] == "context"
    assert linked["properties"]["literal"]["const"] is None
    assert context["properties"]["span_label"]["const"] is None
    assert "evidence_window_id" not in schema["properties"]["context_relations"]["items"]["properties"]
    assert ledger["minItems"] == ledger["maxItems"] == len(expected_labels)
    assert ledger["items"] is False
    assert [row["properties"]["candidate_label"]["const"] for row in ledger["prefixItems"]] == expected_labels


def test_teacher_pin_reflects_sectioned_v8_contract_and_uncapped_token_budgets():
    # Token caps repeatedly produced empty/truncated teacher replies: the r16
    # smoke's reasoning trace was cut mid-source-scan before three further
    # explicit relations. The contract carries no completion or reasoning
    # cap; only the reasoning-trace exclusion remains, and the sectioned
    # prompt/schema (v8/r20) must repin caches.
    assert "max_tokens" not in qa_builder.RELATION_TEACHER_GENERATION_CONFIG
    assert qa_builder.RELATION_TEACHER_GENERATION_CONFIG["reasoning"] == {"exclude": True}
    assert qa_builder.RELATION_TEACHER_PROMPT_VERSION == "qa-relation-teacher-v18"
    assert qa_builder.RELATION_TEACHER_RESPONSE_SCHEMA["version"] == 9
    assert qa_builder.RELATION_TEACHER_REVISION == "qa-relation-teacher-r30"


def test_prompt_worked_examples_are_source_independent_and_level_based():
    # Worked examples must not leak the document's own entities: verbatim
    # excerpts turn the examples into an answer key the teacher can copy,
    # confounding any per-document result. Examples use unrelated conditions.
    source = "Hypothyroidism is treated with Synthroid. Arthritis exacerbation, prescribe Ultram."
    environment = _environment(source)
    arth = source.index("Arthritis")
    environment["occurrences"].append({"occurrence_id": "arthritis", "decision_id": "d-arth", "surface": "Arthritis", "start": arth, "end": arth + 9, "runtime_type": "health-condition"})
    environment["decisions"].append({"decision_id": "d-arth", "actions": [{"mode": "level", "legal": True, "entails": ["joint disease"]}]})
    prompt = relation_teacher_prompt("d2", source, environment)
    instructions = prompt.split("DETECTED SPANS")[0]

    assert 'Safe question: "Which medication category was prescribed for the neurological disorder?"' in prompt
    assert 'Accepted answer: "triptan"' in prompt
    assert "never the source words" in prompt
    # The example scenario is independent of any document's spans.
    for example_entity in ("migraine", "sumatriptan", "cataract", "diabetes", "asthma", "beta-blockers"):
        assert example_entity in instructions
    # No document span/level leaks into the worked examples.
    for leaked in ("arthritis", "ultram", "hypothyroidism", "synthroid", "opioid analgesic", "joint disease"):
        assert leaked not in instructions.lower()


def test_prompt_anchors_labels_to_the_relation_sentence_and_deduplicates_facts():
    # The v5 live smoke paired literals with the FIRST label of a value
    # (S1/S2) instead of the label at the relation sentence, and spent 10 of
    # 12 slots on per-label copies of the same facts, crowding out
    # monitored_by. Answers used controlled surfaces ("ultram") instead of
    # levels.
    source = "Hypothyroidism is treated with Synthroid."
    prompt = relation_teacher_prompt("d2", source, _environment(source))

    assert "sentence that states the relation" in prompt
    assert "Emit each distinct fact once" in prompt
    assert "duplicate_mention" in prompt
    assert "listed levels, never its source text" in prompt
    assert "reason for every row" in prompt


def test_sectioned_wire_schema_separates_span_and_context_relations():
    # A/B spike (2026-07-14): the sectioned single call produced 4-6 literal
    # proposals in 3/3 draws vs 0-3 under the mixed single list; the wire
    # schema now requires span_relations (linked+linked only) and
    # context_relations (exactly one linked + one literal).
    source = "Hypothyroidism is treated with Synthroid."
    environment = _environment(source)
    schema = relation_teacher_response_format(environment, source)["json_schema"]["schema"]

    assert schema["required"] == ["span_relations", "context_relations", "candidate_accounting"]
    span_shapes = schema["properties"]["span_relations"]["items"]["properties"]["arguments"]["anyOf"]
    context_shapes = schema["properties"]["context_relations"]["items"]["properties"]["arguments"]["anyOf"]
    span_kinds = {
        tuple(branch["properties"]["kind"]["const"] for branch in shape["prefixItems"])
        for shape in span_shapes
    }
    context_kinds = {
        tuple(branch["properties"]["kind"]["const"] for branch in shape["prefixItems"])
        for shape in context_shapes
    }
    assert span_kinds == {("linked", "linked")}
    assert context_kinds == {("linked", "context"), ("context", "linked")}


def test_openrouter_teacher_parses_sectioned_relation_lists(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, prompt, **kwargs):
            return json.dumps({
                "span_relations": [{"relation": "prescribed_with"}],
                "context_relations": [{"relation": "monitored_by"}],
                "candidate_accounting": [],
            })

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("CLOAK_LLM_CACHE", "/tmp/test-cache")
    monkeypatch.setattr("cloak.llm.LLMClient", FakeClient)

    proposals = OpenRouterRelationTeacher().propose("prompt")

    assert [row["relation"] for row in proposals] == ["prescribed_with", "monitored_by"]


def test_prompt_defines_the_response_record_fields_and_linked_argument_rule():
    # The observed Nemotron reasoning trace planned a text format ("format?
    # Not fully specified") and then fell into the all-context wire branch;
    # the prompt must name the record contents the decoder will demand.
    source = "Hypothyroidism is treated with Synthroid."
    prompt = relation_teacher_prompt("d2", source, _environment(source))

    assert "span_label" in prompt
    assert "support_property" in prompt
    assert "exactly one linked S-label argument" in prompt
    assert "verbatim" in prompt
    assert "Never quote a displayed span as a context literal" in prompt
    assert "Example span_relations record" in prompt
    assert "Example context_relations record" in prompt


def test_response_schema_forbids_zero_linked_argument_pairs_and_fixes_roles():
    source = "Hypothyroidism is treated with Synthroid."
    environment = _environment(source)
    schema = relation_teacher_response_format(environment, source)["json_schema"]["schema"]

    shapes = [
        shape
        for section in ("span_relations", "context_relations")
        for shape in schema["properties"][section]["items"]["properties"]["arguments"]["anyOf"]
    ]
    kinds = [
        tuple(branch["properties"]["kind"]["const"] for branch in shape["prefixItems"])
        for shape in shapes
    ]
    # The observed Nemotron failure emitted context+context, which the compiler
    # always rejects (missing_linked_argument); it must be unrepresentable.
    assert ("context", "context") not in kinds
    assert set(kinds) == {("linked", "linked"), ("linked", "context"), ("context", "linked")}
    for shape in shapes:
        assert shape["items"] is False
        assert shape["minItems"] == shape["maxItems"] == 2
        roles = [branch["properties"]["role"]["const"] for branch in shape["prefixItems"]]
        assert roles == ["subject", "object"]
        for branch in shape["prefixItems"]:
            if branch["properties"]["kind"]["const"] == "linked":
                assert branch["properties"]["span_label"]["enum"] == ["S1", "S2"]
                assert branch["properties"]["literal"]["const"] is None
            else:
                assert branch["properties"]["span_label"]["const"] is None
                assert branch["properties"]["support_property"]["const"] is None
                assert branch["properties"]["literal"] == {"type": "string", "minLength": 1}


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


def test_prescribed_with_accepts_explicit_continue_on_prescription():
    # The relation inventory promises "prescribed, continued, taken, or used";
    # D2N002 grounds hypothyroidism -> Synthroid as "continue you on the
    # synthroid", which the cue contract must accept.
    source = "For your Hypothyroidism, I will continue you on the Synthroid."
    environment = _environment(source)
    environment["occurrences"][0]["start"] = source.index("Hypothyroidism")
    environment["occurrences"][0]["end"] = source.index("Hypothyroidism") + 14
    proposal = {
        "relation": "prescribed_with",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S1", "support_property": "endocrine condition", "literal": None},
            {"role": "object", "kind": "linked", "span_label": "S2", "support_property": "thyroid medication", "literal": None},
        ],
        "question": "Which medication class is continued for the endocrine disorder?",
        "accepted_answers": ["hormone replacement therapy"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == []
    assert accepted[0]["relation"] == "prescribed_with"


def test_candidate_accounting_reasons_are_sanitized_of_protected_terms():
    # The observed r16 ledger repeated protected surfaces in its reasons
    # ("no explicit relation found for kidney transplant"); the compiler must
    # sanitize reasons before they can reach the artifact.
    source = "Hypothyroidism is treated with Synthroid."
    environment = _environment(source)
    environment["occurrences"][0]["aliases"] = ["underactive thyroid"]
    accounting = [
        {"candidate_label": "S1", "state": "exhausted_no_relation",
         "reason": "no explicit relation found for hypothyroidism"},
        {"candidate_label": "S2", "state": "emitted",
         "reason": "Synthroid used for the underactive thyroid problem"},
    ]

    validated = qa_builder._validated_candidate_accounting(accounting, environment, source)

    reasons = {row["candidate_label"]: row["reason"] for row in validated}
    assert "hypothyroidism" not in reasons["S1"].casefold()
    assert "synthroid" not in reasons["S2"].casefold()
    assert "underactive thyroid" not in reasons["S2"].casefold()
    assert reasons["S1"] == "no explicit relation found for [protected]"
    assert all(row["reason"].strip() for row in validated)


def test_context_literal_is_typed_by_the_relation_slot_when_lexical_rules_cannot():
    # The v5 live smoke grounded hypothyroidism -> "synthroid" ("continue you
    # on the synthroid") but the closed lexical rules cover only
    # test/procedure/provider/status/category, so the drug literal died as
    # untyped_context_literal. The spec's literal contract adds "the relation
    # object's permitted slot class"; grounding, cue, leakage, and reader
    # gates remain the real filters.
    source = "for your hypothyroidism , i will continue you on the synthroid ."
    environment = {
        "occurrences": [{
            "occurrence_id": "condition", "decision_id": "d-condition",
            "surface": "hypothyroidism", "start": source.index("hypothyroidism"),
            "end": source.index("hypothyroidism") + 14, "runtime_type": "health-condition",
        }],
        "decisions": [{
            "decision_id": "d-condition",
            "actions": [{"mode": "level", "legal": True, "entails": ["endocrine condition"]}],
        }],
    }
    proposal = {
        "relation": "prescribed_with",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S1", "support_property": "endocrine condition", "literal": None},
            {"role": "object", "kind": "context", "span_label": None, "support_property": None, "literal": "synthroid"},
        ],
        "question": "Which medication is continued for the endocrine disorder?",
        "accepted_answers": ["synthroid"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == []
    assert accepted[0]["relation"] == "prescribed_with"
    assert accepted[0]["occurrence_ids"] == ["condition"]


def test_context_literal_resolving_onto_a_controlled_span_is_rejected_as_leakage():
    # Spec: a literal that also resolves to a protected controlled span is
    # rejected; the teacher must reference that span by its S-label instead.
    source = "for your arthritis , i will prescribe some ultram ."
    ultram_start = source.index("ultram")
    environment = {
        "occurrences": [
            {"occurrence_id": "arthritis", "decision_id": "d-arthritis", "surface": "arthritis", "start": source.index("arthritis"), "end": source.index("arthritis") + 9, "runtime_type": "health-condition"},
            {"occurrence_id": "ultram", "decision_id": "d-ultram", "surface": "ultram", "start": ultram_start, "end": ultram_start + 6, "runtime_type": "drug"},
        ],
        "decisions": [
            {"decision_id": "d-arthritis", "actions": [{"mode": "level", "legal": True, "entails": ["joint condition"]}]},
            {"decision_id": "d-ultram", "actions": [{"mode": "level", "legal": True, "entails": ["opioid analgesic"]}]},
        ],
    }
    proposal = {
        "relation": "prescribed_with",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S1", "support_property": "joint condition", "literal": None},
            {"role": "object", "kind": "context", "span_label": None, "support_property": None, "literal": "ultram"},
        ],
        "question": "Which medication class is started for the joint condition?",
        "accepted_answers": ["opioid analgesic"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert accepted == []
    assert rejected[0]["detail_reason"] == "protected_context_literal"
    assert rejected[0]["reason"] == "leakage"


def test_context_literal_matching_an_uncontrolled_detected_span_is_rejected():
    # The synthroid case: the drug is detected (so it is substituted to a
    # placeholder at render) but has no lattice decision, so the teacher can
    # only reference it as a context literal. That literal cannot survive the
    # generalized/placeholder docs, so it is rejected up front rather than
    # dying later at the three-point gate. (protected_context_literal does not
    # catch it because the span is uncontrolled — no decision to resolve onto.)
    source = "for your hypothyroidism , continue the synthroid ."
    syn = source.index("synthroid")
    environment = {
        "occurrences": [
            {"occurrence_id": "hypo", "decision_id": "d-hypo", "surface": "hypothyroidism",
             "start": source.index("hypothyroidism"), "end": source.index("hypothyroidism") + 14,
             "runtime_type": "health-condition"},
            # detected drug, uncontrolled (no lattice decision) but still
            # substituted at render -> not caught by protected_context_literal
            {"occurrence_id": "syn", "surface": "synthroid", "start": syn, "end": syn + 9,
             "runtime_type": "drug", "controlled": False},
        ],
        "decisions": [
            {"decision_id": "d-hypo", "actions": [{"mode": "level", "legal": True,
                                                   "entails": ["thyroid gland disease"]}]},
        ],
    }
    proposal = {
        "relation": "prescribed_with",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S1",
             "support_property": "thyroid gland disease", "literal": None},
            {"role": "object", "kind": "context", "span_label": None,
             "support_property": None, "literal": "synthroid"},
        ],
        "question": "Which medication was continued for the thyroid gland disease?",
        "accepted_answers": ["synthroid"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert accepted == []
    assert rejected[0]["detail_reason"] == "literal_will_be_substituted"


def test_leakage_repair_recolors_subject_side_and_preserves_answer_floor():
    # "thyroid gland disease" (subject level, in the question) collides on
    # "thyroid" with the answer "thyroid hormonal medication" -> answer_leakage.
    # Repair rewrites the question's subject reference to the coarser legal level
    # "endocrine system disease" and leaves the answer floor intact.
    decisions = {
        "d-hypo": {"actions": [{"mode": "level", "legal": True,
                                "entails": ["thyroid gland disease", "endocrine system disease"]}]},
        "d-syn": {"actions": [{"mode": "level", "legal": True,
                               "entails": ["thyroid hormonal medication", "hormonal therapy agent"]}]},
    }
    occurrences = {
        "hypo": {"occurrence_id": "hypo", "decision_id": "d-hypo", "runtime_type": "health-condition"},
        "syn": {"occurrence_id": "syn", "decision_id": "d-syn", "runtime_type": "drug"},
    }
    arguments = [
        {"role": "subject", "kind": "linked", "occurrence_id": "hypo", "support_property": "thyroid gland disease"},
        {"role": "object", "kind": "linked", "occurrence_id": "syn", "support_property": "thyroid hormonal medication"},
    ]
    question = "Which medication was prescribed for the thyroid gland disease?"

    result = qa_builder._repair_leaked_relation(
        question, ["thyroid hormonal medication"], arguments, "object",
        ["health-condition", "drug"], decisions, occurrences,
    )

    assert result is not None
    new_question, values, args, repair = result
    assert repair["kind"] == "subject_level_recolor" and repair["floor_lowered"] is False
    assert repair["to_level"] == "endocrine system disease"
    assert "endocrine system disease" in new_question and "thyroid gland disease" not in new_question
    assert values == ["thyroid hormonal medication"]  # answer floor unchanged
    assert not qa_builder._question_leaks_answer(new_question, values[0], ["health-condition", "drug"])


def test_leakage_repair_answer_side_fallback_lowers_floor():
    # Subject has no alternative legal level, so the only way to clear the
    # collision is to coarsen the answer -> flagged floor_lowered.
    decisions = {
        "d-hypo": {"actions": [{"mode": "level", "legal": True, "entails": ["thyroid gland disease"]}]},
        "d-syn": {"actions": [{"mode": "level", "legal": True,
                               "entails": ["thyroid hormonal medication", "hormonal therapy agent"]}]},
    }
    occurrences = {
        "hypo": {"occurrence_id": "hypo", "decision_id": "d-hypo", "runtime_type": "health-condition"},
        "syn": {"occurrence_id": "syn", "decision_id": "d-syn", "runtime_type": "drug"},
    }
    arguments = [
        {"role": "subject", "kind": "linked", "occurrence_id": "hypo", "support_property": "thyroid gland disease"},
        {"role": "object", "kind": "linked", "occurrence_id": "syn", "support_property": "thyroid hormonal medication"},
    ]
    question = "Which medication was prescribed for the thyroid gland disease?"

    result = qa_builder._repair_leaked_relation(
        question, ["thyroid hormonal medication"], arguments, "object",
        ["health-condition", "drug"], decisions, occurrences,
    )

    assert result is not None
    new_question, values, args, repair = result
    assert repair["kind"] == "answer_level_recolor" and repair["floor_lowered"] is True
    assert values == ["hormonal therapy agent"]
    assert new_question == question  # question untouched on answer-side repair


def test_leakage_repair_returns_none_when_no_legal_recoloring_clears_it():
    # Every level on both sides carries "thyroid" -> unrepairable, caller rejects.
    decisions = {
        "d-hypo": {"actions": [{"mode": "level", "legal": True, "entails": ["thyroid gland disease"]}]},
        "d-syn": {"actions": [{"mode": "level", "legal": True, "entails": ["thyroid hormone agent"]}]},
    }
    occurrences = {
        "hypo": {"occurrence_id": "hypo", "decision_id": "d-hypo", "runtime_type": "health-condition"},
        "syn": {"occurrence_id": "syn", "decision_id": "d-syn", "runtime_type": "drug"},
    }
    arguments = [
        {"role": "subject", "kind": "linked", "occurrence_id": "hypo", "support_property": "thyroid gland disease"},
        {"role": "object", "kind": "linked", "occurrence_id": "syn", "support_property": "thyroid hormone agent"},
    ]
    question = "Which medication was prescribed for the thyroid gland disease?"

    assert qa_builder._repair_leaked_relation(
        question, ["thyroid hormone agent"], arguments, "object",
        ["health-condition", "drug"], decisions, occurrences,
    ) is None


def test_contraindicated_because_of_accepts_explicit_cannot_take_wording():
    # D2N002 grounds the contraindication as "you ca n't take some of those
    # anti-inflammatory medications because of your kidney transplant"; the
    # cue lives before the subject argument, so the cue window must cover the
    # clause holding the arguments, not only the text between them.
    source = (
        "again , you ca n't take some of those anti-inflammatory medications "
        "because of your kidney transplant , so it will be a struggle ."
    )
    transplant_start = source.index("kidney transplant")
    environment = {
        "occurrences": [{
            "occurrence_id": "transplant", "decision_id": "d-transplant",
            "surface": "kidney transplant", "start": transplant_start,
            "end": transplant_start + 17, "runtime_type": "health-condition",
        }],
        "decisions": [{
            "decision_id": "d-transplant",
            "actions": [{"mode": "level", "legal": True, "entails": ["solid organ transplant"]}],
        }],
    }
    proposal = {
        "relation": "contraindicated_because_of",
        "arguments": [
            {"role": "subject", "kind": "context", "span_label": None, "support_property": None, "literal": "some of those anti-inflammatory medications"},
            {"role": "object", "kind": "linked", "span_label": "S1", "support_property": "solid organ transplant", "literal": None},
        ],
        "question": "Which medication group cannot be taken because of the transplant history?",
        "accepted_answers": ["anti-inflammatory medications"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == []
    assert accepted[0]["relation"] == "contraindicated_because_of"


def test_multi_sentence_turn_anchors_search_cues_in_the_argument_clauses():
    # In D2N002 the contraindication lives inside a long doctor turn: the
    # anchor quote spans several sentences and the "ca n't take" cue precedes
    # the subject argument, so the cue window must be the clause holding the
    # arguments, not the text between them.
    source = (
        "[doctor] you are doing well . again , you ca n't take some of those "
        "anti-inflammatory medications because of your kidney transplant , so "
        "it will be a struggle . let us move on .\n[patient] okay ."
    )
    transplant_start = source.index("kidney transplant")
    environment = {
        "occurrences": [{
            "occurrence_id": "transplant", "decision_id": "d-transplant",
            "surface": "kidney transplant", "start": transplant_start,
            "end": transplant_start + 17, "runtime_type": "health-condition",
        }],
        "decisions": [{
            "decision_id": "d-transplant",
            "actions": [{"mode": "level", "legal": True, "entails": ["solid organ transplant"]}],
        }],
    }
    proposal = {
        "relation": "contraindicated_because_of",
        "arguments": [
            {"role": "subject", "kind": "context", "span_label": None, "support_property": None, "literal": "anti-inflammatory medications"},
            {"role": "object", "kind": "linked", "span_label": "S1", "support_property": "solid organ transplant", "literal": None},
        ],
        "question": "Which medication group cannot be taken because of the solid organ transplant?",
        "accepted_answers": ["anti-inflammatory medications"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == []
    assert accepted[0]["relation"] == "contraindicated_because_of"


def test_causes_or_explains_accepts_explicit_exacerbation_attribution():
    # D2N002 explains the knee pain as "an acute exacerbation of your
    # arthritis"; explicit clinical attribution wordings (exacerbation of,
    # due to, secondary to) belong in the closed cue set alongside
    # causes/explains.
    source = (
        "[doctor] so for your knee pain , i think that this is an acute "
        "exacerbation of your arthritis , okay ? so i wan na go ahead and "
        "prescribe some ultram .\n[patient] okay ."
    )
    arthritis_start = source.index("arthritis")
    environment = {
        "occurrences": [{
            "occurrence_id": "arthritis", "decision_id": "d-arthritis",
            "surface": "arthritis", "start": arthritis_start,
            "end": arthritis_start + 9, "runtime_type": "health-condition",
        }],
        "decisions": [{
            "decision_id": "d-arthritis",
            "actions": [{"mode": "level", "legal": True, "entails": ["bone inflammation disease"]}],
        }],
    }
    proposal = {
        "relation": "causes_or_explains",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S1", "support_property": "bone inflammation disease", "literal": None},
            {"role": "object", "kind": "context", "span_label": None, "support_property": None, "literal": "knee pain"},
        ],
        "question": "What symptom is attributed to the bone inflammation disease?",
        "accepted_answers": ["knee pain"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == []
    assert accepted[0]["relation"] == "causes_or_explains"


def test_spoken_ellipsis_does_not_split_a_clause_in_the_direct_support_gate():
    # ACI transcripts contain hesitation ellipses ("i wan na go ahead and if
    # ... and prescribe some ultram"); dots in a run are not sentence
    # boundaries and must not make same-sentence arguments non-adjacent.
    source = (
        "[doctor] this is an acute exacerbation of your arthritis , okay ? "
        "so i wan na go ahead and if ... and prescribe some ultram 50 mg .\n"
        "[patient] okay ."
    )
    arthritis_start = source.index("arthritis")
    ultram_start = source.index("ultram")
    environment = {
        "occurrences": [
            {"occurrence_id": "arthritis", "decision_id": "d-arthritis", "surface": "arthritis", "start": arthritis_start, "end": arthritis_start + 9, "runtime_type": "health-condition"},
            {"occurrence_id": "ultram", "decision_id": "d-ultram", "surface": "ultram", "start": ultram_start, "end": ultram_start + 6, "runtime_type": "drug"},
        ],
        "decisions": [
            {"decision_id": "d-arthritis", "actions": [{"mode": "level", "legal": True, "entails": ["bone inflammation disease"]}]},
            {"decision_id": "d-ultram", "actions": [{"mode": "level", "legal": True, "entails": ["opioid analgesic"]}]},
        ],
    }
    proposal = {
        "relation": "prescribed_with",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S1", "support_property": "bone inflammation disease", "literal": None},
            {"role": "object", "kind": "linked", "span_label": "S2", "support_property": "opioid analgesic", "literal": None},
        ],
        "question": "Which medication category was prescribed for the bone inflammation disease?",
        "accepted_answers": ["opioid analgesic"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == []
    assert accepted[0]["relation"] == "prescribed_with"


def test_linked_surface_in_qa_is_substituted_with_the_selected_level():
    # Three consecutive live smokes wrote the linked span's surface into the
    # question/answers ("for the arthritis?", answer "ultram") despite
    # escalating prompt guidance. The compiler substitutes the teacher's own
    # selected support_property for that argument's protected surface and
    # re-runs the leakage gates; unrelated protected surfaces still reject.
    source = "Hypothyroidism is treated with Synthroid."
    proposal = {
        "relation": "prescribed_with",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S1", "support_property": "endocrine condition", "literal": None},
            {"role": "object", "kind": "linked", "span_label": "S2", "support_property": "thyroid medication", "literal": None},
        ],
        "question": "Which medication was prescribed for the Hypothyroidism?",
        "accepted_answers": ["Synthroid"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, _environment(source), [proposal])

    assert rejected == []
    assert accepted[0]["question"] == "Which medication was prescribed for the endocrine condition?"
    assert accepted[0]["accepted_values"] == ["thyroid medication"]
    assert accepted[0]["evidence"]["sanitized_qa"] is True


_ASSESSMENT_BLOCK = (
    "[doctor] so i just wan na go over my assessment and my plan .\n"
    "[patient] mm-hmm .\n"
    "[doctor] so for your knee pain , i think this is an acute exacerbation of your "
    "arthritis . so i wan na prescribe some ultram .\n"
    "[patient] okay .\n"
    "[doctor] okay ? i also wan na go ahead and just order an autoimmune panel .\n"
    "[patient] sure .\n"
    "[doctor] for your second problem , your hypothyroidism , i wan na order a thyroid panel .\n"
)


def _assessment_environment():
    src = _ASSESSMENT_BLOCK
    arthritis = src.index("arthritis")
    panel = src.index("autoimmune panel")
    thyroid = src.index("hypothyroidism")
    return src, {
        "occurrences": [
            {"occurrence_id": "arthritis", "decision_id": "d-arthritis", "surface": "arthritis", "start": arthritis, "end": arthritis + 9, "runtime_type": "health-condition"},
            {"occurrence_id": "panel", "decision_id": "d-panel", "surface": "autoimmune panel", "start": panel, "end": panel + 16, "runtime_type": "medical-procedure"},
            {"occurrence_id": "thyroid", "decision_id": "d-thyroid", "surface": "hypothyroidism", "start": thyroid, "end": thyroid + 14, "runtime_type": "health-condition"},
        ],
        "decisions": [
            {"decision_id": "d-arthritis", "actions": [{"mode": "level", "legal": True, "entails": ["joint condition"]}]},
            {"decision_id": "d-panel", "actions": [{"mode": "level", "legal": True, "entails": ["immunology panel"]}]},
            {"decision_id": "d-thyroid", "actions": [{"mode": "level", "legal": True, "entails": ["endocrine condition"]}]},
        ],
    }


def _span_pair(relation, subj, subj_level, obj, obj_level, question, answer):
    return {
        "relation": relation,
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": subj, "support_property": subj_level, "literal": None},
            {"role": "object", "kind": "linked", "span_label": obj, "support_property": obj_level, "literal": None},
        ],
        "question": question,
        "accepted_answers": [answer],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }


def test_multiturn_span_pair_grounds_within_one_problem_block():
    # D2N002's second true span pair: the autoimmune panel is ordered to
    # evaluate the arthritis, but a patient turn sits between them. A
    # within-problem-block anchor must ground it. S1=arthritis, S3=panel.
    source, environment = _assessment_environment()
    # source order: S1=arthritis, S2=autoimmune panel, S3=hypothyroidism.
    proposal = _span_pair(
        "monitored_by", "S1", "joint condition", "S2", "immunology panel",
        "What testing was ordered to evaluate the joint condition?", "immunology panel",
    )

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == []
    assert accepted[0]["relation"] == "monitored_by"
    assert accepted[0]["occurrence_ids"] == ["arthritis", "panel"]


def test_treated_with_indication_grounds_inside_a_speaker_turn_anchor():
    # D2N002: "you've had the kidney transplant a few years ago for some
    # polycystic kidneys" sits in one doctor turn with preamble, so the anchor
    # re-splits into >2 sub-clauses. The reversed indication connector must be
    # consulted there, not only in a clean 2-clause fixture.
    source = (
        "[doctor] okay . all right . now , i know that you've had the kidney transplant "
        "a few years ago for some polycystic kidneys .\n[patient] mm-hmm .\n"
    )
    kt = source.index("kidney transplant")
    environment = {
        "occurrences": [{
            "occurrence_id": "transplant", "decision_id": "d-transplant",
            "surface": "kidney transplant", "start": kt, "end": kt + 17,
            "runtime_type": "health-condition",
        }],
        "decisions": [{
            "decision_id": "d-transplant",
            "actions": [{"mode": "level", "legal": True, "entails": ["solid organ transplant"]}],
        }],
    }
    proposal = {
        "relation": "treated_with",
        "arguments": [
            {"role": "subject", "kind": "context", "span_label": None, "support_property": None, "literal": "polycystic kidneys"},
            {"role": "object", "kind": "linked", "span_label": "S1", "support_property": "solid organ transplant", "literal": None},
        ],
        "question": "Which procedure category addressed the polycystic kidneys?",
        "accepted_answers": ["solid organ transplant"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == []
    assert accepted[0]["relation"] == "treated_with"
    assert accepted[0]["occurrence_ids"] == ["transplant"]


def test_multiturn_anchor_rejects_link_across_a_problem_switch():
    # arthritis (first problem) must not link to the thyroid panel ordered
    # under "for your second problem".
    source, environment = _assessment_environment()
    thyroid_panel = source.index("thyroid panel")
    environment["occurrences"].append(
        {"occurrence_id": "tpanel", "decision_id": "d-tpanel", "surface": "thyroid panel",
         "start": thyroid_panel, "end": thyroid_panel + 13, "runtime_type": "medical-procedure"})
    environment["decisions"].append(
        {"decision_id": "d-tpanel", "actions": [{"mode": "level", "legal": True, "entails": ["thyroid testing"]}]})
    proposal = _span_pair(
        "monitored_by", "S1", "joint condition", "S4", "thyroid testing",
        "What panel was ordered for the joint condition?", "thyroid testing",
    )

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert accepted == []
    assert rejected[0]["detail_reason"] == "invalid_evidence"


def test_problem_block_anchor_rejects_a_hedged_conditional_relation():
    # The conditional PT referral ("if symptoms continue ... possibly referral")
    # must not become an asserted fact even though both ends share the block.
    source = (
        "[doctor] so for your knee pain , this is your arthritis . "
        "if your symptoms continue , we'll possibly refer you to physical therapy .\n"
        "[patient] okay .\n"
        "[doctor] for your second problem , your hypothyroidism .\n"
    )
    arthritis = source.index("arthritis")
    environment = {
        "occurrences": [{
            "occurrence_id": "arthritis", "decision_id": "d-arthritis", "surface": "arthritis",
            "start": arthritis, "end": arthritis + 9, "runtime_type": "health-condition",
        }],
        "decisions": [{
            "decision_id": "d-arthritis",
            "actions": [{"mode": "level", "legal": True, "entails": ["joint condition"]}],
        }],
    }
    proposal = {
        "relation": "referred_to",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S1", "support_property": "joint condition", "literal": None},
            {"role": "object", "kind": "context", "span_label": None, "support_property": None, "literal": "physical therapy"},
        ],
        "question": "Which service was named for the joint condition?",
        "accepted_answers": ["physical therapy"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert accepted == []
    assert rejected[0]["detail_reason"] == "hedged_relation"


def test_prompt_permits_within_problem_block_multiturn_links():
    source, environment = _assessment_environment()
    prompt = relation_teacher_prompt("d2", source, environment)

    assert "SAME problem discussion" in prompt
    assert "different problem discussion" in prompt
    assert "conditional or hypothetical" in prompt


def _transplant_chain():
    return [
        {"node": "keep", "answer_aliases": ["kidney transplant", "renal transplant"],
         "entailed_properties": ["solid organ transplant", "medical condition"]},
        {"node": "solid organ transplant", "answer_aliases": ["solid organ transplant"],
         "entailed_properties": ["solid organ transplant", "medical condition"]},
        {"node": "medical condition", "answer_aliases": ["medical condition"],
         "entailed_properties": ["medical condition"]},
        {"node": "placeholder", "answer_aliases": [], "entailed_properties": []},
    ]


def test_linked_answer_score_rewards_keep_and_supported_generalization_only():
    chain = _transplant_chain()
    req = "solid organ transplant"
    # KEEP (source + alias) and the exact supported level all get full credit.
    assert qa_builder._linked_answer_score("kidney transplant", chain, req) == 1.0
    assert qa_builder._linked_answer_score("renal transplant", chain, req) == 1.0
    assert qa_builder._linked_answer_score("a solid organ transplant", chain, req) == 1.0
    # coarser-than-required and placeholder/NONE get nothing.
    assert qa_builder._linked_answer_score("medical condition", chain, req) == 0.0
    assert qa_builder._linked_answer_score("NONE", chain, req) == 0.0
    assert qa_builder._linked_answer_score("", chain, req) == 0.0


def test_linked_answer_resolution_is_decision_scoped():
    # A phrase that is not in this decision's chain does not resolve, even if it
    # would be valid for some other decision.
    chain = _transplant_chain()
    assert qa_builder._resolve_semantic_node(chain, "hypothyroidism") is None
    assert qa_builder._resolve_semantic_node(chain, "kidney transplant")["node"] == "keep"


def test_accepted_answers_and_question_are_underscore_normalized():
    # The teacher inconsistently snake-cases answers ("solid_organ_transplant");
    # under lexical fact_score that tokenizes as one token vs three, so
    # normalize underscores to spaces before storing/scoring.
    source = "Hypothyroidism is treated with Synthroid."
    proposal = {
        "relation": "prescribed_with",
        "arguments": [
            {"role": "subject", "kind": "linked", "occurrence_id": "condition", "support_property": "endocrine condition"},
            {"role": "object", "kind": "linked", "occurrence_id": "drug", "support_property": "thyroid medication"},
        ],
        "question": "Which medication_class treats the endocrine disorder?",
        "accepted_answers": ["hormone_replacement_therapy"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
        "evidence_quote": source,
    }

    accepted, rejected = compile_relational_assertions("d2", source, _environment(source), [proposal])

    assert rejected == []
    assert accepted[0]["accepted_values"] == ["hormone replacement therapy"]
    assert accepted[0]["question"] == "Which medication class treats the endocrine disorder?"


def test_prompt_prefers_the_most_specific_generalization_level():
    # The teacher was picking the vacuous root level (e.g. "medical condition")
    # as the answer, which discriminates nothing; the prompt must ask for the
    # most specific applicable level.
    source = "Hypothyroidism is treated with Synthroid."
    prompt = relation_teacher_prompt("d2", source, _environment(source))
    assert "most specific" in prompt
    assert "not the broadest" in prompt


def test_treated_with_accepts_procedure_form_condition_via_indication_connector():
    # D2N002 (and its reference verbatim): "you've had the kidney transplant a
    # few years ago for some polycystic kidneys". The transplant is detector-
    # typed health-condition, so treated_with's procedure slot needs the
    # closed procedure-form lexicon; the "<procedure> for <condition>"
    # indication form needs a reversed connector.
    source = "you've had the kidney transplant a few years ago for some polycystic kidneys ."
    transplant_start = source.index("kidney transplant")
    environment = {
        "occurrences": [{
            "occurrence_id": "transplant", "decision_id": "d-transplant",
            "surface": "kidney transplant", "start": transplant_start,
            "end": transplant_start + 17, "runtime_type": "health-condition",
        }],
        "decisions": [{
            "decision_id": "d-transplant",
            "actions": [{"mode": "level", "legal": True, "entails": ["solid organ transplant"]}],
        }],
    }
    proposal = {
        "relation": "treated_with",
        "arguments": [
            {"role": "subject", "kind": "context", "span_label": None, "support_property": None, "literal": "polycystic kidneys"},
            {"role": "object", "kind": "linked", "span_label": "S1", "support_property": "solid organ transplant", "literal": None},
        ],
        "question": "Which procedure category addressed the polycystic kidneys?",
        "accepted_answers": ["solid organ transplant"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == []
    assert accepted[0]["relation"] == "treated_with"
    assert accepted[0]["occurrence_ids"] == ["transplant"]


def test_prompt_displays_dual_class_for_procedure_form_conditions():
    source = "you've had the kidney transplant a few years ago for some polycystic kidneys ."
    transplant_start = source.index("kidney transplant")
    environment = {
        "occurrences": [{
            "occurrence_id": "transplant", "decision_id": "d-transplant",
            "surface": "kidney transplant", "start": transplant_start,
            "end": transplant_start + 17, "runtime_type": "health-condition",
        }],
        "decisions": [{
            "decision_id": "d-transplant",
            "actions": [{"mode": "level", "legal": True, "entails": ["solid organ transplant"]}],
        }],
    }

    prompt = relation_teacher_prompt("d2", source, environment)

    assert "[S1: kidney transplant | condition/procedure | levels: solid organ transplant]" in prompt


def test_plain_condition_still_cannot_fill_the_procedure_slot():
    # The dual class applies only to the closed procedure-form lexicon; an
    # ordinary condition surface must keep failing treated_with's object slot.
    source = "the knee pain was managed for some arthritis ."
    arthritis_start = source.index("arthritis")
    environment = {
        "occurrences": [{
            "occurrence_id": "arthritis", "decision_id": "d-arthritis",
            "surface": "arthritis", "start": arthritis_start,
            "end": arthritis_start + 9, "runtime_type": "health-condition",
        }],
        "decisions": [{
            "decision_id": "d-arthritis",
            "actions": [{"mode": "level", "legal": True, "entails": ["joint condition"]}],
        }],
    }
    proposal = {
        "relation": "treated_with",
        "arguments": [
            {"role": "subject", "kind": "context", "span_label": None, "support_property": None, "literal": "knee pain"},
            {"role": "object", "kind": "linked", "span_label": "S1", "support_property": "joint condition", "literal": None},
        ],
        "question": "Which procedure category addressed the knee pain?",
        "accepted_answers": ["joint condition"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert accepted == []
    assert rejected[0]["detail_reason"] == "invalid_argument_types"


def test_question_may_use_declared_generalization_level_tokens():
    # "kidney transplant" generalizes to the declared legal level "solid organ
    # transplant"; the spec directs questions to use that level, so its tokens
    # cannot count as protected residue. Full-term containment and non-level
    # tokens ("kidney") still leak.
    allowed = {"kidney transplant": frozenset({"solid", "organ", "transplant"})}
    assert not qa_builder._question_leaks_protected_term(
        "Which medication group cannot be taken because of the solid organ transplant?",
        ["kidney transplant"], allowed,
    )
    assert qa_builder._question_leaks_protected_term(
        "Which medication group cannot be taken because of the kidney issue?",
        ["kidney transplant"], allowed,
    )
    assert qa_builder._question_leaks_protected_term(
        "Is the kidney transplant relevant?", ["kidney transplant"], allowed,
    )


def test_ledger_supports_duplicate_mention_and_wire_schema_requires_reasons():
    # The v5 live smoke fabricated one relation per duplicate S-label (10 of 12
    # slots) because `emitted` was the only way to cover a repeated mention,
    # and returned empty reasons the wire schema permitted, invalidating the
    # whole ledger. Repeated mentions need their own state and reasons must be
    # wire-required.
    source = "Hypothyroidism is treated with Synthroid."
    environment = _environment(source)
    accounting = [
        {"candidate_label": "S1", "state": "emitted", "reason": "used in prescribed_with"},
        {"candidate_label": "S2", "state": "duplicate_mention", "reason": "same value as the S2 fact"},
    ]

    validated = qa_builder._validated_candidate_accounting(accounting, environment, source)
    assert [row["state"] for row in validated] == ["emitted", "duplicate_mention"]

    ledger_item = qa_builder.RELATION_TEACHER_RESPONSE_FORMAT[
        "json_schema"]["schema"]["properties"]["candidate_accounting"]["items"]
    assert "duplicate_mention" in ledger_item["properties"]["state"]["enum"]
    assert ledger_item["properties"]["reason"]["minLength"] == 1

    bound = relation_teacher_response_format(environment, source)
    bound_rows = bound["json_schema"]["schema"]["properties"]["candidate_accounting"]["prefixItems"]
    for row in bound_rows:
        assert "duplicate_mention" in row["properties"]["state"]["enum"]
        assert row["properties"]["reason"]["minLength"] == 1


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
    argument_shapes = schema["properties"]["span_relations"]["items"]["properties"]["arguments"]["anyOf"]

    assert "[S1: Hypothyroidism | condition | levels: endocrine condition]" in prompt
    assert "[S2: Synthroid | drug | levels: thyroid medication]" in prompt
    assert "OCCURRENCE INVENTORY" not in prompt
    assert "SOURCE EVIDENCE WINDOWS" not in prompt
    assert '"occurrence_id"' not in prompt
    assert "evidence_window_id" not in schema["properties"]["span_relations"]["items"]["properties"]
    assert argument_shapes[0]["prefixItems"][0]["properties"]["span_label"]["enum"] == ["S1", "S2"]
    context_shapes = schema["properties"]["context_relations"]["items"]["properties"]["arguments"]["anyOf"]
    assert context_shapes[0]["prefixItems"][1]["properties"]["literal"]["type"] == "string"
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


def _person_anchor_environment(source):
    hy = source.index("hypothyroidism")
    return {
        "occurrences": [
            {"occurrence_id": "p1", "surface": "andrew", "start": source.index("andrew"),
             "end": source.index("andrew") + 6, "runtime_type": "PERSON",
             "anchor": True, "anchor_token": "<PERSON_1>", "controlled": False},
            {"occurrence_id": "c1", "decision_id": "d-c", "surface": "hypothyroidism",
             "start": hy, "end": hy + 14, "runtime_type": "health-condition", "controlled": True},
        ],
        "decisions": [
            {"decision_id": "d-c", "runtime_type": "health-condition",
             "canonical_key": "hypothyroidism", "controlled": True,
             "actions": [{"mode": "level", "legal": True,
                          "entails": ["thyroid gland disease", "endocrine system disease"]}]},
        ],
    }


def test_person_anchored_relation_compiles_with_placeholder_anchor_subject():
    source = "andrew has hypothyroidism ."
    proposal = {
        "relation": "has_condition",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "P1",
             "support_property": None, "literal": None},
            {"role": "object", "kind": "linked", "span_label": "S1",
             "support_property": "thyroid gland disease", "literal": None},
        ],
        "question": "Which condition does <PERSON_1> have?",
        "accepted_answers": ["thyroid gland disease"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }
    accepted, rejected = compile_relational_assertions(
        "d", source, _person_anchor_environment(source), [proposal])

    assert accepted, rejected
    row = accepted[0]
    assert row["relation"] == "has_condition"
    assert row["answer_target"] == {
        "kind": "linked_decision", "decision_id": "d-c",
        "required_property": "thyroid gland disease"}
    # anchor enters occurrence_ids (grounding) but carries no decision requirement
    assert "p1" in row["occurrence_ids"] and "c1" in row["occurrence_ids"]
    assert row["decision_requirements"] == {"d-c": "thyroid gland disease"}


def test_person_anchor_cannot_be_the_answer():
    source = "andrew has hypothyroidism ."
    proposal = {
        "relation": "has_condition",
        "answer_role": "subject",  # person as the answer -> illegal
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "P1",
             "support_property": None, "literal": None},
            {"role": "object", "kind": "linked", "span_label": "S1",
             "support_property": "thyroid gland disease", "literal": None},
        ],
        "question": "Who has the thyroid gland disease?",
        "accepted_answers": ["andrew"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }
    accepted, rejected = compile_relational_assertions(
        "d", source, _person_anchor_environment(source), [proposal])

    assert accepted == []
    assert any(r["detail_reason"] == "answer_is_anchor" for r in rejected)
