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


def _cross_clause_anchor_inputs(source):
    condition = "Muscle strain"
    medication = "Meloxicam"
    condition_start = source.index(condition)
    medication_start = source.index(medication)
    occurrences = {
        "condition": {
            "occurrence_id": "condition", "decision_id": "d-condition",
            "surface": condition, "start": condition_start,
            "end": condition_start + len(condition), "runtime_type": "health-condition",
        },
        "medication": {
            "occurrence_id": "medication", "decision_id": "d-medication",
            "surface": medication, "start": medication_start,
            "end": medication_start + len(medication), "runtime_type": "drug",
        },
    }
    arguments = [
        {"role": "subject", "kind": "linked", "occurrence_id": "condition",
         "surface": condition, "runtime_type": "health-condition",
         "support_property": "muscle injury"},
        {"role": "object", "kind": "linked", "occurrence_id": "medication",
         "surface": medication, "runtime_type": "drug",
         "support_property": "anti-inflammatory medication"},
    ]
    return arguments, occurrences


def _cross_clause_anchor(source):
    arguments, occurrences = _cross_clause_anchor_inputs(source)
    return (*qa_builder._derived_relation_anchor(
        source, arguments, occurrences, {}, "prescribed_with", qa_builder.ACI_RELATION_CONTRACT,
    ), arguments, occurrences)


def _cross_clause_context_anchor(source, literal="follow-up x-ray"):
    condition = "Distal phalanx fracture"
    condition_start = source.index(condition)
    occurrences = {
        "condition": {
            "occurrence_id": "condition", "decision_id": "d-condition",
            "surface": condition, "start": condition_start,
            "end": condition_start + len(condition), "runtime_type": "health-condition",
        },
    }
    arguments = [
        {"role": "subject", "kind": "linked", "occurrence_id": "condition",
         "surface": condition, "runtime_type": "health-condition",
         "support_property": "finger or toe bone fracture"},
        {"role": "object", "kind": "context", "literal": literal,
         "runtime_type": "", "start": None, "end": None},
    ]
    return (*qa_builder._derived_relation_anchor(
        source, arguments, occurrences, {}, "tests_for", qa_builder.ACI_RELATION_CONTRACT,
    ), arguments, occurrences)


def test_cross_clause_span_anchor_stitches_argument_clauses_and_reader_turns():
    source = (
        "Muscle strain was assessed.\n"
        "The patient denies fever.\n"
        "Meloxicam was prescribed for pain."
    )
    quote, span, clause_ranges, error, kind, arguments, _ = _cross_clause_anchor(source)

    assert error is None and kind == "stitched_clauses"
    assert "denies fever" not in quote
    assert quote == "Muscle strain was assessed.\nMeloxicam was prescribed for pain."
    assert span == (0, len(source))
    clauses = qa_builder._source_clause_spans(source)
    assert clause_ranges == [clauses[0], clauses[2]]
    assert qa_builder._source_turns_for_ranges(source, clause_ranges) == [0, 2]
    assert qa_builder._relation_quote_has_lexical_cue_support(
        "prescribed_with", quote, arguments, qa_builder.ACI_RELATION_CONTRACT,
        allow_adjacent_clauses=True,
    )


def test_cross_clause_span_anchor_rejects_problem_switch_marker():
    source = (
        "Muscle strain was assessed. "
        "For your next problem, the patient denies fever. "
        "Meloxicam was prescribed for pain."
    )
    _, span, clause_ranges, error, kind, _, _ = _cross_clause_anchor(source)

    assert span is None and clause_ranges is None
    assert error == "invalid_evidence" and kind is None


def test_cross_clause_span_anchor_bridges_the_exam_to_plan_transition():
    # The condition is stated in the exam and its drug in that problem's plan -- the
    # "assessment and plan" / "for your first problem" transition of the SAME problem must
    # NOT block the bridge (only a switch to a different problem does).
    source = (
        "Muscle strain was assessed.\n"
        "So let's go over my assessment and my plan.\n"
        "For your first problem, Meloxicam was prescribed for pain."
    )
    quote, span, clause_ranges, error, kind, _, _ = _cross_clause_anchor(source)

    assert error is None and kind == "stitched_clauses"
    assert span is not None and clause_ranges is not None
    assert "assessment and my plan" not in quote  # the transition clause is elided, not sent


def test_cross_clause_span_anchor_rejects_pairs_beyond_global_distance():
    source = "Muscle strain was assessed. " + " ".join(
        f"Unrelated clause {index}." for index in range(qa_builder._CROSS_CLAUSE_ANCHOR_MAX_DISTANCE)
    ) + " Meloxicam was prescribed for pain."
    _, span, clause_ranges, error, kind, _, _ = _cross_clause_anchor(source)

    assert span is None and clause_ranges is None
    assert error == "invalid_evidence" and kind is None


def test_cross_clause_stitch_without_relation_cue_fails_support_check():
    source = (
        "Muscle strain was assessed.\n"
        "The patient denies fever.\n"
        "Meloxicam appears in the medication list."
    )
    quote, _, _, error, kind, arguments, _ = _cross_clause_anchor(source)

    assert error is None and kind == "stitched_clauses"
    assert not qa_builder._relation_quote_has_lexical_cue_support(
        "prescribed_with", quote, arguments, qa_builder.ACI_RELATION_CONTRACT,
        allow_adjacent_clauses=True,
    )
    assert not qa_builder._relation_quote_has_direct_support(
        "prescribed_with", quote, arguments, qa_builder.ACI_RELATION_CONTRACT,
        allow_adjacent_clauses=True, require_lexical_cue=True,
    )


def test_cross_clause_context_anchor_stitches_linked_and_literal_clauses():
    source = (
        "Distal phalanx fracture was assessed.\n"
        "The patient denies fever.\n"
        "Follow-up x-ray was ordered."
    )
    quote, span, clause_ranges, error, kind, arguments, _ = _cross_clause_context_anchor(source)

    assert error is None and kind == "stitched_clauses"
    assert "denies fever" not in quote
    assert quote == "Distal phalanx fracture was assessed.\nFollow-up x-ray was ordered."
    assert span == (0, len(source))
    assert clause_ranges == [
        qa_builder._source_clause_spans(source)[0],
        qa_builder._source_clause_spans(source)[2],
    ]
    assert arguments[1]["literal"] == "Follow-up x-ray"
    assert qa_builder._source_turns_for_ranges(source, clause_ranges) == [0, 2]


def test_cross_clause_context_anchor_rejects_literal_beyond_global_distance():
    source = "Distal phalanx fracture was assessed. " + " ".join(
        f"Unrelated clause {index}." for index in range(qa_builder._CROSS_CLAUSE_ANCHOR_MAX_DISTANCE)
    ) + " Follow-up x-ray was ordered."
    _, span, clause_ranges, error, kind, _, _ = _cross_clause_context_anchor(source)

    assert span is None and clause_ranges is None
    assert error == "unknown_context_literal" and kind is None


def test_cross_clause_context_anchor_rejects_problem_switch_before_literal():
    source = (
        "Distal phalanx fracture was assessed. "
        "For your next problem, the patient denies fever. "
        "Follow-up x-ray was ordered."
    )
    _, span, clause_ranges, error, kind, _, _ = _cross_clause_context_anchor(source)

    assert span is None and clause_ranges is None
    assert error == "unknown_context_literal" and kind is None


def test_cross_clause_context_anchor_picks_nearest_qualifying_literal_occurrence():
    source = (
        "Distal phalanx fracture was assessed. "
        "The patient denies fever. "
        "Follow-up x-ray was ordered first. "
        "The patient was given instructions. "
        "Follow-up x-ray was ordered later."
    )
    _, _, clause_ranges, error, kind, arguments, _ = _cross_clause_context_anchor(source)

    first_xray = source.index("Follow-up x-ray")
    assert error is None and kind == "stitched_clauses"
    assert arguments[1]["start"] == first_xray
    assert clause_ranges == [
        qa_builder._source_clause_spans(source)[0],
        qa_builder._source_clause_spans(source)[2],
    ]


def test_cross_clause_context_anchor_reanchors_to_nearest_same_decision_sibling():
    source = (
        "Distal phalanx fracture was identified. "
        "The patient denies fever. "
        "The diagnosis remains distal phalanx fracture. "
        "Follow-up x-ray was scheduled."
    )
    first = source.index("Distal phalanx fracture")
    second = source.index("distal phalanx fracture", first + 1)
    occurrences = {
        "fracture-early": {
            "occurrence_id": "fracture-early", "decision_id": "d-fracture",
            "surface": "Distal phalanx fracture", "start": first,
            "end": first + len("Distal phalanx fracture"), "runtime_type": "health-condition",
        },
        "fracture-plan": {
            "occurrence_id": "fracture-plan", "decision_id": "d-fracture",
            "surface": "distal phalanx fracture", "start": second,
            "end": second + len("distal phalanx fracture"), "runtime_type": "health-condition",
        },
    }
    arguments = [
        {"role": "subject", "kind": "linked", "occurrence_id": "fracture-early",
         "surface": "Distal phalanx fracture", "runtime_type": "health-condition",
         "support_property": "finger bone fracture"},
        {"role": "object", "kind": "context", "literal": "follow-up x-ray",
         "runtime_type": "", "start": None, "end": None},
    ]

    first_anchor = qa_builder._derived_relation_anchor(
        source, arguments, occurrences, {}, "tests_for", qa_builder.ACI_RELATION_CONTRACT,
    )
    second_anchor = qa_builder._derived_relation_anchor(
        source, arguments, occurrences, {}, "tests_for", qa_builder.ACI_RELATION_CONTRACT,
    )

    assert first_anchor[3] is None and first_anchor[4] == "stitched_clauses"
    assert second_anchor[3] is None and second_anchor[4] == "clause"
    assert arguments[0]["occurrence_id"] == "fracture-plan"
    assert "diagnosis remains distal phalanx fracture" in second_anchor[0]
    assert "Follow-up x-ray was scheduled" in second_anchor[0]


def test_bounded_stitched_relation_reaches_reader_gate_without_lexical_cue():
    source = (
        "Distal phalanx fracture was identified. "
        "The patient denies fever. "
        "Follow-up x-ray was discussed."
    )
    fracture_start = source.index("Distal phalanx fracture")
    environment = {
        "occurrences": [{
            "occurrence_id": "fracture", "decision_id": "d-fracture",
            "surface": "Distal phalanx fracture", "start": fracture_start,
            "end": fracture_start + len("Distal phalanx fracture"),
            "runtime_type": "health-condition",
        }],
        "decisions": [{
            "decision_id": "d-fracture",
            "actions": [{"mode": "level", "legal": True,
                         "entails": ["finger bone fracture"]}],
        }],
    }
    proposal = {
        "relation": "tests_for",
        "answer_role": "subject",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S1",
             "support_property": "finger bone fracture", "literal": None},
            {"role": "object", "kind": "context", "span_label": None,
             "support_property": None, "literal": "follow-up x-ray"},
        ],
        "question": "For what medical condition was follow-up x-ray discussed?",
        "accepted_answers": ["finger bone fracture"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == [], rejected
    assert accepted[0]["evidence"]["source_clause_ranges"] == [
        list(qa_builder._source_clause_spans(source)[0]),
        list(qa_builder._source_clause_spans(source)[2]),
    ]


def test_teacher_authored_relation_qa_is_preserved_and_drug_is_prescribed_not_treated():
    source = "Hypothyroidism is treated with Synthroid."
    accepted, rejected = compile_relational_assertions("d2", source, _environment(source), [_proposal(source)])

    assert rejected == []
    assert accepted[0]["relation"] == "prescribed_with"
    assert accepted[0]["question"] == _proposal(source)["question"]
    assert accepted[0]["accepted_values"] == ["hormone replacement therapy"]
    assert accepted[0]["decision_requirements"] == {"d-condition": "endocrine condition", "d-drug": "thyroid medication"}


def test_procedure_for_rejects_drug_and_context_argument_stays_unlinked():
    source = "Hypothyroidism is treated with Synthroid. Arthritis was referred to physical therapy."
    environment = _environment(source)
    arthritis_start = source.index("Arthritis")
    environment["occurrences"].append({"occurrence_id": "arthritis", "decision_id": "d-arthritis", "surface": "Arthritis", "start": arthritis_start, "end": arthritis_start + 9, "runtime_type": "health-condition"})
    environment["decisions"].append({"decision_id": "d-arthritis", "actions": [{"mode": "level", "legal": True, "entails": ["joint condition"]}]})
    bad = {**_proposal(source), "relation": "procedure_for"}
    physical = next(
        candidate for candidate in relation_context_candidates(source)
        if candidate["literal"] == "physical therapy"
    )
    context = {
        "relation": "procedure_for",
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
    assert "procedure_for: condition or diagnosis -> medical procedure" in prompt
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
    assert qa_builder.RELATION_TEACHER_PROMPT_VERSION == "qa-relation-teacher-v21"
    assert qa_builder.RELATION_TEACHER_RESPONSE_SCHEMA["version"] == 9
    assert qa_builder.RELATION_TEACHER_REVISION == "qa-relation-teacher-r32"


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

    assert 'Safe question: "Which medication was prescribed for the neurological disorder?"' in prompt
    assert 'Accepted answer: "triptan"' in prompt
    assert 'Safe question: "For what medical condition was hemoglobin A1c ordered?"' in prompt
    assert 'Accepted answer: "metabolic disorder"' in prompt
    assert "answer_role to the linked argument's role" in prompt
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
    # tests_for. Answers used controlled surfaces ("ultram") instead of
    # levels.
    source = "Hypothyroidism is treated with Synthroid."
    prompt = relation_teacher_prompt("d2", source, _environment(source))

    assert "sentence that states the relation" in prompt
    assert "Emit each distinct fact" in prompt
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
                "context_relations": [{"relation": "tests_for"}],
                "candidate_accounting": [],
            })

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("CLOAK_LLM_CACHE", "/tmp/test-cache")
    monkeypatch.setattr("cloak.llm.LLMClient", FakeClient)

    proposals = OpenRouterRelationTeacher().propose("prompt")

    assert [row["relation"] for row in proposals] == ["prescribed_with", "tests_for"]


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
    assert "answer_role subject" in prompt


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
        "relation": "procedure_for",
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
        "relation": "tests_for",
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
    assert accepted[0]["relation"] == "tests_for"


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


def test_openrouter_teacher_retries_empty_then_reports_it_explicitly(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.last_completion_state = {"outcome": "no_choices"}

        def generate(self, prompt, **kwargs):
            calls.append(kwargs.get("refresh", False))
            return ""  # persistently empty -> retries exhaust, then raise

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("CLOAK_LLM_CACHE", "/tmp/test-cache")
    monkeypatch.setattr("cloak.llm.LLMClient", FakeClient)

    with pytest.raises(ValueError, match="teacher_no_choices"):
        OpenRouterRelationTeacher().propose("prompt")
    # empty is retried (refresh=True) before giving up, like a throttle
    assert len(calls) == qa_builder._TEACHER_EMPTY_RETRIES + 1
    assert calls == [False] + [True] * qa_builder._TEACHER_EMPTY_RETRIES


def test_openrouter_teacher_recovers_when_a_retry_returns_content(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.last_completion_state = {"outcome": "no_choices"}
            self.n = 0

        def generate(self, prompt, **kwargs):
            self.n += 1
            return "" if self.n == 1 else json.dumps(
                {"span_relations": [], "context_relations": [], "candidate_accounting": []})

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("CLOAK_LLM_CACHE", "/tmp/test-cache")
    monkeypatch.setattr("cloak.llm.LLMClient", FakeClient)
    # first reply empty, retry returns valid content -> no error
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


def test_context_literal_on_a_controlled_span_is_promoted_to_a_linked_argument():
    # Fix B: a literal that also resolves to a DETECTED (controlled) span with a legal
    # generalization level is promoted to a linked argument on that decision -- a detected
    # entity is substituted in doc_p, so its generalized level is a hideable answer --
    # instead of being rejected as leakage. (An uncontrolled span stays rejected: see the
    # next test.)
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

    assert len(accepted) == 1
    assert accepted[0]["relation"] == "prescribed_with"
    assert not any(r.get("detail_reason") == "protected_context_literal" for r in rejected)


def test_context_literal_on_uncontrolled_detected_span_is_allowed():
    # An UNCONTROLLED detected span (controlled False / no decision, e.g. "physical therapy"
    # with no lattice entry) is detector evidence, not a ranker-controlled span: it carries no
    # action and is NEVER rewritten in any scored render, so the literal survives verbatim and
    # is a valid answer. The literal_will_be_substituted guard must NOT fire on it (it fires
    # only for controlled surfaces the ranker actually substitutes); the three-point gate then
    # decides. Uses a procedure with no decision to mirror the physical-therapy case.
    source = "for your arthritis , we may refer you to physical therapy ."
    pt = source.index("physical therapy")
    environment = {
        "occurrences": [
            {"occurrence_id": "arth", "decision_id": "d-arth", "surface": "arthritis",
             "start": source.index("arthritis"), "end": source.index("arthritis") + 9,
             "runtime_type": "health-condition"},
            # detected procedure, uncontrolled (no lattice decision) -> not rewritten at render
            {"occurrence_id": "pt", "surface": "physical therapy", "start": pt, "end": pt + 16,
             "runtime_type": "medical-procedure", "controlled": False},
        ],
        "decisions": [
            {"decision_id": "d-arth", "actions": [{"mode": "level", "legal": True,
                                                   "entails": ["joint inflammation disease"]}]},
        ],
    }
    proposal = {
        "relation": "procedure_for",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S1",
             "support_property": "joint inflammation disease", "literal": None},
            {"role": "object", "kind": "context", "span_label": None,
             "support_property": None, "literal": "physical therapy"},
        ],
        "question": "What procedure may be referred for the joint inflammation disease?",
        "accepted_answers": ["physical therapy"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    # the uncontrolled literal must NOT be rejected as will-be-substituted
    assert not any(r["detail_reason"] == "literal_will_be_substituted" for r in rejected)


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
    # strict repair: recolor the locator to the coarser level AND strip the answer's own token
    # ("medication") from the question -- no generic-word whitelist. Answer floor unchanged.
    assert repair["kind"] == "strict_answer_token_strip" and repair["floor_lowered"] is False
    assert repair["locator_to"] == "endocrine system disease"
    assert new_question == "Which was prescribed for the endocrine system disease?"
    assert "thyroid gland disease" not in new_question and "medication" not in new_question
    assert values == ["thyroid hormonal medication"]  # answer floor unchanged
    assert not qa_builder._question_leaks_answer(new_question, values[0], "")  # strict, no exempt


def test_leakage_repair_never_coarsens_the_answer_floor_rejects_instead():
    # The locator "thyroid gland disease" leaks "thyroid" and has NO coarser legal level, so the
    # only tokens strippable are outside the locator ("medication") -- which cannot clear the
    # in-locator "thyroid" leak. The repair must NOT coarsen the answer floor (that path is
    # retired); it returns None and the caller rejects honestly.
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

    assert result is None  # no floor-lowering fallback; reject honestly


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
    # sanitize swaps the protected surface for the subject level; then the strict leak repair
    # strips the answer's own token "medication" from the question (answer = "thyroid medication").
    assert accepted[0]["question"] == "Which was prescribed for the endocrine condition?"
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
        "tests_for", "S1", "joint condition", "S2", "immunology panel",
        "What testing was ordered to evaluate the joint condition?", "immunology panel",
    )

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == []
    assert accepted[0]["relation"] == "tests_for"
    assert accepted[0]["occurrence_ids"] == ["arthritis", "panel"]


def test_procedure_for_indication_grounds_inside_a_speaker_turn_anchor():
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
        "relation": "procedure_for",
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
    assert accepted[0]["relation"] == "procedure_for"
    assert accepted[0]["occurrence_ids"] == ["transplant"]


def _kidney_environment(source):
    # D2N002: both spans controlled. The subject's level "cystic kidney disease"
    # shares the token "kidney" with the object's raw surface "kidney transplant".
    kt = source.index("kidney transplant")
    pk = source.index("polycystic kidneys")
    return {
        "occurrences": [
            {"occurrence_id": "transplant", "decision_id": "d-transplant",
             "surface": "kidney transplant", "start": kt, "end": kt + 17,
             "runtime_type": "health-condition"},
            {"occurrence_id": "poly", "decision_id": "d-poly",
             "surface": "polycystic kidneys", "start": pk, "end": pk + 18,
             "runtime_type": "health-condition"},
        ],
        "decisions": [
            {"decision_id": "d-transplant", "actions": [{"mode": "level", "legal": True,
             "entails": ["solid organ transplant", "medical condition"]}]},
            {"decision_id": "d-poly", "actions": [{"mode": "level", "legal": True,
             "entails": ["cystic kidney disease", "kidney disease"]}]},
        ],
    }


_KIDNEY_SOURCE = (
    "[doctor] okay . all right . now , i know that you've had the kidney transplant "
    "a few years ago for some polycystic kidneys .\n[patient] mm-hmm .\n"
)


def test_protected_locator_exempts_a_sibling_arguments_level_token():
    # The question's "kidney" comes from the subject's published level "cystic
    # kidney disease", not from the object surface "kidney transplant". Pooling the
    # relation's argument levels must let this privacy-safe relation compile.
    environment = _kidney_environment(_KIDNEY_SOURCE)
    proposal = {
        "relation": "procedure_for",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S2",
             "support_property": "cystic kidney disease", "literal": None},
            {"role": "object", "kind": "linked", "span_label": "S1",
             "support_property": "solid organ transplant", "literal": None},
        ],
        "question": "Which procedure was used to treat the cystic kidney disease?",
        "accepted_answers": ["solid organ transplant"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }
    accepted, rejected = compile_relational_assertions(
        "d2", _KIDNEY_SOURCE, environment, [proposal])

    assert rejected == [], rejected
    assert accepted[0]["relation"] == "procedure_for"


def test_protected_locator_still_blocks_a_discriminative_surface_token():
    # "polycystic" is in no legal level, so it never enters the argument pool and
    # must still be rejected — the exemption only forgives published-level tokens.
    environment = _kidney_environment(_KIDNEY_SOURCE)
    proposal = {
        "relation": "procedure_for",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S2",
             "support_property": "cystic kidney disease", "literal": None},
            {"role": "object", "kind": "linked", "span_label": "S1",
             "support_property": "solid organ transplant", "literal": None},
        ],
        "question": "Which procedure was used for the polycystic condition?",
        "accepted_answers": ["solid organ transplant"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }
    accepted, rejected = compile_relational_assertions(
        "d2", _KIDNEY_SOURCE, environment, [proposal])

    assert accepted == []
    assert any(r["detail_reason"] == "protected_locator" for r in rejected)


def _ct_stone_environment(source):
    ct = source.index("ct scan"); ks = source.index("kidney stone")
    return {
        "occurrences": [
            {"occurrence_id": "ct", "decision_id": "d-ct", "surface": "ct scan",
             "start": ct, "end": ct + 7, "runtime_type": "medical-procedure"},
            {"occurrence_id": "stone", "decision_id": "d-stone", "surface": "kidney stone",
             "start": ks, "end": ks + 12, "runtime_type": "health-condition"},
        ],
        "decisions": [
            {"decision_id": "d-ct", "actions": [{"mode": "level", "legal": True,
             "entails": ["imaging study"]}]},
            {"decision_id": "d-stone", "actions": [{"mode": "level", "legal": True,
             "entails": ["urinary tract stone"]}]},
        ],
    }


def test_tests_for_discovery_compiles():
    # condition <- diagnostic test that discovered it (reversed "shows" form).
    # S1 = ct scan (test, earlier), S2 = kidney stone (condition, later).
    source = "[doctor] we did a ct scan and it shows a kidney stone .\n[patient] okay .\n"
    environment = _ct_stone_environment(source)
    proposal = {
        "relation": "tests_for",
        "answer_role": "subject",  # the discovered finding is the answer
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S2",
             "support_property": "urinary tract stone", "literal": None},   # condition (finding)
            {"role": "object", "kind": "linked", "span_label": "S1",
             "support_property": "imaging study", "literal": None},          # diagnostic test
        ],
        "question": "What did the imaging study reveal?",
        "accepted_answers": ["urinary tract stone"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }
    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == [], rejected
    assert accepted[0]["relation"] == "tests_for"
    assert accepted[0]["answer_target"]["required_property"] == "urinary tract stone"


def test_context_literal_relation_compiles_with_linked_subject_answer():
    source = "[doctor] for diabetes, hemoglobin A1c was ordered .\n"
    diabetes_start = source.index("diabetes")
    environment = {
        "occurrences": [{
            "occurrence_id": "diabetes", "decision_id": "d-diabetes",
            "surface": "diabetes", "start": diabetes_start,
            "end": diabetes_start + len("diabetes"),
            "runtime_type": "health-condition",
        }],
        "decisions": [{
            "decision_id": "d-diabetes",
            "actions": [{"mode": "level", "legal": True,
                         "entails": ["metabolic disorder"]}],
        }],
    }
    proposal = {
        "relation": "tests_for",
        # answer_role deliberately OMITTED (the real teacher frequently does): the compiler must
        # still force the linked condition as the answer, since the uncontrolled literal can never be.
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S1",
             "support_property": "metabolic disorder", "literal": None},
            {"role": "object", "kind": "context", "span_label": None,
             "support_property": None, "literal": "hemoglobin A1c"},
        ],
        "question": "For what medical condition was hemoglobin A1c ordered?",
        "accepted_answers": ["metabolic disorder"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == [], rejected
    assert accepted[0]["answer_target"] == {
        "kind": "linked_decision",
        "decision_id": "d-diabetes",
        "required_property": "metabolic disorder",
    }
    assert accepted[0]["accepted_values"] == ["metabolic disorder"]


def test_tests_for_grounds_via_semantic_support_when_cue_absent():
    # "scheduled a ct scan for your kidney stone" carries no order/monitor/show cue, but
    # the quote semantically entails the test<->condition relation, so the NLI support
    # fallback grounds it (fixed cue lists drift; NLI covers valid out-of-list phrasing).
    source = "[doctor] we scheduled a ct scan for your kidney stone .\n[patient] okay .\n"
    environment = _ct_stone_environment(source)
    proposal = {
        "relation": "tests_for",
        "answer_role": "subject",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S2",
             "support_property": "urinary tract stone", "literal": None},
            {"role": "object", "kind": "linked", "span_label": "S1",
             "support_property": "imaging study", "literal": None},
        ],
        "question": "What did the imaging study reveal?",
        "accepted_answers": ["urinary tract stone"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }
    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert len(accepted) == 1
    assert accepted[0]["relation"] == "tests_for"


def test_grounds_via_same_value_sibling_when_teacher_names_history_mention():
    # arthritis appears twice (same decision): a history-list mention far from the
    # drug, and the prescribe-sentence mention. The teacher names the history one;
    # the compiler must remap to the sibling that grounds with ultram.
    source = ("[doctor] past medical history includes arthritis , noted years ago .\n"
              "[patient] okay .\n"
              "[doctor] your heart sounds normal and your blood pressure is fine .\n"
              "[patient] good .\n"
              "[doctor] now for your arthritis , i will prescribe ultram .\n")
    a1 = source.index("arthritis")
    a2 = source.index("arthritis", a1 + 1)
    ul = source.index("ultram")
    environment = {
        "occurrences": [
            {"occurrence_id": "arth-hist", "decision_id": "d-arth", "surface": "arthritis",
             "start": a1, "end": a1 + 9, "runtime_type": "health-condition"},
            {"occurrence_id": "arth-plan", "decision_id": "d-arth", "surface": "arthritis",
             "start": a2, "end": a2 + 9, "runtime_type": "health-condition"},
            {"occurrence_id": "ultram", "decision_id": "d-ultram", "surface": "ultram",
             "start": ul, "end": ul + 6, "runtime_type": "drug"},
        ],
        "decisions": [
            {"decision_id": "d-arth", "actions": [{"mode": "level", "legal": True,
             "entails": ["bone inflammation disease"]}]},
            {"decision_id": "d-ultram", "actions": [{"mode": "level", "legal": True,
             "entails": ["opioid analgesic"]}]},
        ],
    }
    # Inventory is one label per decision: S1 = arthritis (represented by the earliest
    # occurrence arth-hist), S2 = ultram. The teacher can only name S1, which resolves to
    # the history mention; the compiler must remap to the plan-sentence sibling to ground.
    proposal = {
        "relation": "prescribed_with",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S1",
             "support_property": "bone inflammation disease", "literal": None},
            {"role": "object", "kind": "linked", "span_label": "S2",
             "support_property": "opioid analgesic", "literal": None},
        ],
        "question": "Which medication category was prescribed for the bone inflammation disease?",
        "accepted_answers": ["opioid analgesic"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }
    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == [], rejected
    assert accepted[0]["relation"] == "prescribed_with"
    # remapped to the plan-sentence sibling, not the history mention
    assert "arth-plan" in accepted[0]["occurrence_ids"]
    assert "arth-hist" not in accepted[0]["occurrence_ids"]


def _conditional_referral_environment(source):
    ar = source.index("arthritis"); pt = source.index("physical therapy")
    return {
        "occurrences": [
            {"occurrence_id": "arth", "decision_id": "d-arth", "surface": "arthritis",
             "start": ar, "end": ar + 9, "runtime_type": "health-condition"},
            {"occurrence_id": "pt", "decision_id": "d-pt", "surface": "physical therapy",
             "start": pt, "end": pt + 16, "runtime_type": "medical-procedure"},
        ],
        "decisions": [
            {"decision_id": "d-arth", "actions": [{"mode": "level", "legal": True,
             "entails": ["musculoskeletal system disease"]}]},
            {"decision_id": "d-pt", "actions": [{"mode": "level", "legal": True,
             "entails": ["manual therapy"]}]},
        ],
    }


_CONDITIONAL_SOURCE = ("[doctor] for your arthritis , if symptoms continue we'll "
                       "possibly refer you to physical therapy .\n[patient] okay .\n")


def _conditional_referral_proposal(question):
    return {
        "relation": "procedure_for",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S1",
             "support_property": "musculoskeletal system disease", "literal": None},
            {"role": "object", "kind": "linked", "span_label": "S2",
             "support_property": "manual therapy", "literal": None},
        ],
        "question": question,
        "accepted_answers": ["manual therapy"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }


def test_conditional_relation_kept_with_conditional_question():
    env = _conditional_referral_environment(_CONDITIONAL_SOURCE)
    proposal = _conditional_referral_proposal(
        "What procedure may the patient be referred to for the musculoskeletal system disease?")
    accepted, rejected = compile_relational_assertions("d2", _CONDITIONAL_SOURCE, env, [proposal])

    assert rejected == [], rejected
    assert accepted[0]["relation"] == "procedure_for"


def test_conditional_relation_rejected_with_definite_question():
    env = _conditional_referral_environment(_CONDITIONAL_SOURCE)
    proposal = _conditional_referral_proposal(
        "What procedure was performed for the musculoskeletal system disease?")
    accepted, rejected = compile_relational_assertions("d2", _CONDITIONAL_SOURCE, env, [proposal])

    assert accepted == []
    assert any(r["detail_reason"] == "hedged_relation" for r in rejected)


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
        "tests_for", "S1", "joint condition", "S4", "thyroid testing",
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
        "relation": "procedure_for",
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
    assert "conditional or planned statement" in prompt
    assert "phrase its question conditionally" in prompt


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


def test_procedure_for_accepts_procedure_form_condition_via_indication_connector():
    # D2N002 (and its reference verbatim): "you've had the kidney transplant a
    # few years ago for some polycystic kidneys". The transplant is detector-
    # typed health-condition, so procedure_for's procedure slot needs the
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
        "relation": "procedure_for",
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
    assert accepted[0]["relation"] == "procedure_for"
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
    # ordinary condition surface must keep failing procedure_for's object slot.
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
        "relation": "procedure_for",
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
        "relation": "tests_for",
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
    assert accepted[0]["relation"] == "tests_for"
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




def test_lattice_level_suspect_probe_flags_coarser_readable(monkeypatch):
    """The coarser-level diagnostic: a relation whose fine locator level is unreadable but a
    coarser legal level in the same chain reads -> report the offending surface + levels."""
    import cloak.train.qa_builder as qb

    decisions = [{
        "decision_id": "d1", "canonical_key": "arthritis", "runtime_type": "health-condition",
        "actions": [
            {"action_id": "fine", "mode": "level", "legal": True, "coarseness_rank": 0.0,
             "fill": "joint inflammation disease",
             "entails": ["joint inflammation disease", "musculoskeletal system disease"]},
            {"action_id": "coarse", "mode": "level", "legal": True, "coarseness_rank": 1.0,
             "fill": "musculoskeletal system disease",
             "entails": ["musculoskeletal system disease"]},
        ],
    }]
    candidate = {
        "question": "Which medication was prescribed for the joint inflammation disease?",
        "decision_requirements": {"d1": "joint inflammation disease"},
        "evidence": {"reader_turns": [0]},
    }
    fill_by_action = {a["action_id"]: a["fill"] for a in decisions[0]["actions"]}

    def render(doc_id, vector):
        return f"note: patient with {fill_by_action[vector['d1']]} takes a drug"

    # reader only extracts once the COARSE level is what got rendered
    def reader(questions, context):
        return ["opioid analgesic" if "musculoskeletal system disease" in context else ""]

    monkeypatch.setattr(qb, "_turn_excerpt", lambda ctx, turns, window=0: ctx)
    monkeypatch.setattr(qb, "_context_answer_score", lambda row, ans, chain: 1.0 if ans else 0.0)

    result = qb._diagnose_coarser_readable(
        candidate, decisions, [],
        doc_id="aci/TEST", render_action_vector=render, reader=reader,
        chain_by_decision={}, reader_threshold=1.0,
    )
    assert result is not None
    assert result["surface"] == "arthritis"
    assert result["unreadable_level"] == "joint inflammation disease"
    assert result["readable_coarser_level"] == "musculoskeletal system disease"


def test_lattice_level_suspect_probe_silent_when_no_level_reads(monkeypatch):
    """No coarser level reads -> genuine reader limit, not a data issue -> no suspect."""
    import cloak.train.qa_builder as qb

    decisions = [{
        "decision_id": "d1", "canonical_key": "arthritis", "runtime_type": "health-condition",
        "actions": [
            {"action_id": "fine", "mode": "level", "legal": True, "coarseness_rank": 0.0,
             "fill": "joint inflammation disease",
             "entails": ["joint inflammation disease", "musculoskeletal system disease"]},
            {"action_id": "coarse", "mode": "level", "legal": True, "coarseness_rank": 1.0,
             "fill": "musculoskeletal system disease", "entails": ["musculoskeletal system disease"]},
        ],
    }]
    candidate = {
        "question": "Which medication was prescribed for the joint inflammation disease?",
        "decision_requirements": {"d1": "joint inflammation disease"},
        "evidence": {"reader_turns": [0]},
    }
    monkeypatch.setattr(qb, "_turn_excerpt", lambda ctx, turns, window=0: ctx)
    monkeypatch.setattr(qb, "_context_answer_score", lambda row, ans, chain: 0.0)
    result = qb._diagnose_coarser_readable(
        candidate, decisions, [], doc_id="aci/TEST",
        render_action_vector=lambda d, v: "x", reader=lambda q, c: [""],
        chain_by_decision={}, reader_threshold=1.0,
    )
    assert result is None


def test_relation_support_opportunities_counts_distinct_compiler_shaped_facts():
    source = "Hypothyroidism is treated with Synthroid."

    opportunities = qa_builder.relation_support_opportunities(source, _environment(source))

    assert [(row["relation"], row["scope"]) for row in opportunities] == [
        ("prescribed_with", "span_span"),
    ]


def test_relation_escalation_trigger_uses_manifest_policy_per_scope():
    policy = {
        "version": "structural-opportunity-v1",
        "min_opportunities": {"span_span": 2, "span_literal": 2},
        "coverage_fraction": {"span_span": 0.8, "span_literal": 0.5},
        "scope_caps": {"span_span": 6, "span_literal": 6},
    }

    targets = qa_builder.relation_escalation_targets(
        {"span_span": 4, "span_literal": 1}, policy,
    )

    assert targets == {"span_span": 4, "span_literal": 0}
    assert qa_builder.needs_relation_escalation(
        {"span_span": 3, "span_literal": 0}, targets,
    )
    assert not qa_builder.needs_relation_escalation(
        {"span_span": 4, "span_literal": 0}, targets,
    )


def test_merge_kept_relations_prefers_primary_and_keeps_single_teacher_facts():
    occurrences = {
        "condition": {"decision_id": "d-condition"},
        "drug": {"decision_id": "d-drug"},
    }
    primary = {
        "relation": "prescribed_with",
        "question": "primary question?",
        "evidence": {"arguments": [
            {"role": "subject", "kind": "linked", "occurrence_id": "condition"},
            {"role": "object", "kind": "linked", "occurrence_id": "drug"},
        ]},
    }
    secondary_duplicate = {**primary, "question": "secondary duplicate?"}
    secondary_only = {
        "relation": "tests_for",
        "question": "secondary only?",
        "evidence": {"arguments": [
            {"role": "subject", "kind": "linked", "occurrence_id": "condition"},
            {"role": "object", "kind": "context", "literal": "thyroid labs"},
        ]},
    }

    merged, disposition = qa_builder.merge_kept_relation_rows(
        [primary], [secondary_duplicate, secondary_only], occurrences,
    )

    assert [row["question"] for row in merged] == ["primary question?", "secondary only?"]
    assert disposition == {"primary_only": 0, "primary_preferred": 1, "secondary_only": 1}


def test_compile_relations_deduplicates_same_fact_across_sibling_occurrences():
    source = "Hypothyroidism is treated with Synthroid. Hypothyroidism is treated with Synthroid."
    first_condition = source.index("Hypothyroidism")
    first_drug = source.index("Synthroid")
    second_condition = source.index("Hypothyroidism", first_condition + 1)
    second_drug = source.index("Synthroid", first_drug + 1)
    environment = {
        "occurrences": [
            {"occurrence_id": "c1", "decision_id": "condition", "surface": "Hypothyroidism", "start": first_condition, "end": first_condition + 14, "runtime_type": "health-condition"},
            {"occurrence_id": "d1", "decision_id": "drug", "surface": "Synthroid", "start": first_drug, "end": first_drug + 9, "runtime_type": "drug"},
            {"occurrence_id": "c2", "decision_id": "condition", "surface": "Hypothyroidism", "start": second_condition, "end": second_condition + 14, "runtime_type": "health-condition"},
            {"occurrence_id": "d2", "decision_id": "drug", "surface": "Synthroid", "start": second_drug, "end": second_drug + 9, "runtime_type": "drug"},
        ],
        "decisions": [
            {"decision_id": "condition", "actions": [{"mode": "level", "legal": True, "entails": ["endocrine condition"]}]},
            {"decision_id": "drug", "actions": [{"mode": "level", "legal": True, "entails": ["thyroid medication"]}]},
        ],
    }
    first = {
        "relation": "prescribed_with",
        "arguments": [
            {"role": "subject", "kind": "linked", "occurrence_id": "c1", "support_property": "endocrine condition"},
            {"role": "object", "kind": "linked", "occurrence_id": "d1", "support_property": "thyroid medication"},
        ],
        "question": "Which drug was prescribed for the endocrine condition?",
        "accepted_answers": ["thyroid medication"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
        "evidence_quote": "Hypothyroidism is treated with Synthroid.",
        "evidence_start": first_condition,
    }
    second = {
        **first,
        "arguments": [
            {"role": "subject", "kind": "linked", "occurrence_id": "c2", "support_property": "endocrine condition"},
            {"role": "object", "kind": "linked", "occurrence_id": "d2", "support_property": "thyroid medication"},
        ],
        "evidence_start": second_condition,
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [first, second])

    assert len(accepted) == 1
    assert [row["detail_reason"] for row in rejected] == ["duplicate_fact_group"]


def test_source_literal_spans_equates_case_and_whitespace_only():
    # FM2: the teacher re-cases/re-spaces a literal; the resolver maps to the exact source
    # span so grounding stays exact, but never matches a different token sequence.
    document = "please order an mri today, and recheck   hemoglobin a1c next month."
    assert [document[s:e] for s, e in qa_builder._source_literal_spans(document, "MRI")] == ["mri"]
    assert [document[s:e] for s, e in
            qa_builder._source_literal_spans(document, "hemoglobin A1c")] == ["hemoglobin a1c"]
    # collapsed whitespace variant still resolves to the real (multi-space) source span
    assert [document[s:e] for s, e in
            qa_builder._source_literal_spans(document, "hemoglobin   a1c")] == ["hemoglobin a1c"]
    # a genuinely absent / different token never matches
    assert qa_builder._source_literal_spans(document, "ultrasound") == []
    assert qa_builder._source_literal_spans(document, "mri scan") == []
    # non-word-edge literals must still resolve (regression that \b would have broken)
    punct = "the (MRI) and C++ were noted; a1c-panel drawn."
    assert [punct[s:e] for s, e in qa_builder._source_literal_spans(punct, "(MRI)")] == ["(MRI)"]
    assert [punct[s:e] for s, e in qa_builder._source_literal_spans(punct, "C++")] == ["C++"]
    # a hyphen-prefixed fragment must NOT match a longer hyphenated token
    assert qa_builder._source_literal_spans(punct, "a1c-") == []


def test_protected_term_generic_category_word_is_not_a_leak():
    leaks = qa_builder._question_leaks_protected_term
    # "medication" is a detected drug SURFACE (protected term) AND a drug placeholder word ->
    # a question using the generic slot word is not an identity leak.
    allowed = {"medication": frozenset({"medication", "drug", "treatment"})}
    assert leaks("Which medication was prescribed for the mood disorder?",
                 ["medication"], allowed) is False
    # a single-token surface riding its own published COARSER level is safe: "diabetes" is
    # authorized because the level "diabetes mellitus" contributes {diabetes, mellitus}.
    assert leaks("Which medication was prescribed for the diabetes mellitus?",
                 ["diabetes"], {"diabetes": frozenset({"diabetes", "mellitus", "condition"})}) is False
    # defect-2: a raw brand whose only would-be authorization is a level EQUAL to the surface is
    # rejected -- the caller drops that level from `allowed`, so "Synthroid" stays discriminative.
    assert leaks("Which Synthroid dose was used?", ["Synthroid"],
                 {"Synthroid": frozenset({"medication"})}) is True   # surface-echoing level excluded upstream
    # a raw brand/name token stays discriminative and still leaks
    assert leaks("Which medication like prozac was prescribed?", ["prozac"], allowed) is True
    # a short alias tokenizes to nothing (sub-3-char) and is never exempted
    assert leaks("Which AF treatment is used?", ["AF"], {"AF": frozenset()}) is True
    # a multi-token raw surface is never exempted, even if each token is authorized
    assert leaks("Was this seen by john smith?", ["john smith"],
                 {"john smith": frozenset({"john", "smith"})}) is True


def test_answer_competing_surfaces_flags_multi_procedure_scope():
    # Ambiguity monitor: reflux's problem block holds two controlled procedures, so the answered
    # one ("dietary modifications") has a co-valid competitor ("ultrasound") a free-form reader
    # could name instead. Scope is the subject decision's problem block. Deterministic, diagnostic.
    doc = ("for your first problem , your reflux , get an ultrasound and continue dietary "
           "modifications . for your second problem , your arthritis , order an autoimmune panel .")
    def span(sub): return (doc.index(sub), doc.index(sub) + len(sub))
    rs, re_ = span("reflux"); us, ue = span("ultrasound")
    ds, de = span("dietary modifications"); ps, pe = span("autoimmune panel")
    occ = {
        "reflux": {"occurrence_id": "reflux", "decision_id": "d-reflux", "runtime_type": "health-condition",
                   "controlled": True, "surface": "reflux", "start": rs, "end": re_},
        "us": {"occurrence_id": "us", "decision_id": "d-us", "runtime_type": "medical-procedure",
               "controlled": True, "surface": "ultrasound", "start": us, "end": ue},
        "diet": {"occurrence_id": "diet", "decision_id": "d-diet", "runtime_type": "medical-procedure",
                 "controlled": True, "surface": "dietary modifications", "start": ds, "end": de},
        "panel": {"occurrence_id": "panel", "decision_id": "d-panel", "runtime_type": "medical-procedure",
                  "controlled": True, "surface": "autoimmune panel", "start": ps, "end": pe},
    }
    subj = {"kind": "linked", "occurrence_id": "reflux"}
    answer_arg = {"kind": "linked", "occurrence_id": "diet"}
    target = {"kind": "linked_decision", "decision_id": "d-diet"}
    f = qa_builder._answer_competing_surfaces
    # ultrasound is in reflux's block; the arthritis-block panel is NOT counted (different problem)
    assert f(doc, subj, answer_arg, target, occ) == ["ultrasound"]
    # sole procedure in the subject block -> no competitor
    assert f(doc, subj, answer_arg, target, {k: occ[k] for k in ("reflux", "diet", "panel")}) == []


def test_turn_excerpt_stitches_discontiguous_turns_and_elides_the_gap():
    ctx = "\n".join(f"t{i}" for i in range(10))
    # adjacent turns merge into one contiguous region (no elision) -- co-located relation unchanged
    assert qa_builder._turn_excerpt(ctx, [2, 3], window=0) == "t2\nt3"
    # far-apart turns are stitched surgically: only the two regions, gap elided (not t3..t6)
    out = qa_builder._turn_excerpt(ctx, [2, 7], window=0)
    assert out == f"t2\n{qa_builder._TURN_EXCERPT_ELISION}\nt7"
    assert "t4" not in out and "t5" not in out  # the noisy middle is never handed to the reader
    # windows that bridge the gap re-merge into one region (no elision)
    assert qa_builder._turn_excerpt(ctx, [2, 4], window=1) == "t1\nt2\nt3\nt4\nt5"
    # empty / out-of-range falls back to the full context (diverged-render safety)
    assert qa_builder._turn_excerpt(ctx, [], window=0) == ctx
    assert qa_builder._turn_excerpt(ctx, [99], window=0) == ctx


def test_gleaning_targets_classifies_by_taxonomy():
    occ = {n: {"occurrence_id": n, "decision_id": f"d-{n}", "runtime_type": t,
               "controlled": True, "surface": n, "start": s, "end": s + 5}
           for n, t, s in [("reflux", "health-condition", 0), ("diet", "medical-procedure", 20),
                           ("us", "medical-procedure", 50), ("hf", "health-condition", 70),
                           ("ace", "drug", 90)]}

    def rel(subj, obj):
        return [{"role": "subject", "kind": "linked", "occurrence_id": subj, "support_property": "x"},
                {"role": "object", "kind": "linked", "occurrence_id": obj, "support_property": "y"}]

    ambiguous = {"relation": "procedure_for", "detail_reason": "protected_answer",
                 "evidence": {"arguments": rel("reflux", "diet"), "answer_competing": ["ultrasound"]}}
    fixable = {"relation": "prescribed_with", "detail_reason": "invalid_evidence",
               "evidence": {"arguments": rel("hf", "ace")}}
    legit = {"relation": "tests_for", "detail_reason": "no_task_role_cue",
             "evidence": {"arguments": rel("hf", "us")}}
    both = {"relation": "tests_for", "detail_reason": "invalid_evidence",  # fixable AND ambiguous
            "evidence": {"arguments": rel("reflux", "us"), "answer_competing": ["dietary modifications"]}}
    opp_key = qa_builder._relation_fact_key("prescribed_with", rel("reflux", "ace"), occ)
    opportunities = [{"relation": "prescribed_with", "fact_key": opp_key,
                      "arguments": rel("reflux", "ace"), "evidence_span": [0, 100]}]

    targets = qa_builder._gleaning_targets(
        "", [], [ambiguous, fixable, legit, both], opportunities, occ)
    by = {(t["relation"], t["kind"]) for t in targets}

    assert ("procedure_for", "ambiguous") in by            # answer_competing -> ambiguous
    assert ("prescribed_with", "fixable") in by            # invalid_evidence -> fixable
    assert ("prescribed_with", "missed") in by             # opportunity never proposed -> missed
    assert ("tests_for", "ambiguous") in by                # dedup priority: ambiguous beats fixable
    assert ("tests_for", "fixable") not in by
    # no_task_role_cue is 100% legitimate -> never a target
    assert not any(t["relation"] == "tests_for" and t["kind"] == "fixable" for t in targets)
    assert all(t.get("reason") != "no_task_role_cue" for t in targets)
    amb = next(t for t in targets if t["kind"] == "ambiguous" and t["relation"] == "procedure_for")
    assert amb["competing"] == ["ultrasound"] and "uniquely" in amb["hint"].lower()
    fx = next(t for t in targets if t["kind"] == "fixable")
    assert fx["reason"] == "invalid_evidence" and "anchor" in fx["hint"].lower()


def test_gleaning_targets_never_drops_a_fixable_reject_without_compiled_args():
    # Real rejections often carry no evidence.arguments (e.g. invalid_evidence whose args never
    # grounded). The conservative rule forbids a false-negative: they must still be targets.
    ambiguous = {"relation": "procedure_for", "detail_reason": "protected_answer",
                 "rejection_id": "sha256:aaa",
                 "evidence": {"answer_competing": ["right upper quadrant ultrasound"]}}
    fixable = {"relation": "prescribed_with", "detail_reason": "invalid_evidence",
               "rejection_id": "sha256:bbb", "evidence": {}}
    legit = {"relation": "tests_for", "detail_reason": "source_contradiction",
             "rejection_id": "sha256:ccc", "evidence": {}}

    targets = qa_builder._gleaning_targets("", [], [ambiguous, fixable, legit], [], {})
    kinds = {t["kind"] for t in targets}

    assert "ambiguous" in kinds and "fixable" in kinds
    assert all(t["fact_key"][0] == "reject" for t in targets)  # keyed by rejection identity
    assert not any(t.get("reason") == "source_contradiction" for t in targets)
    assert len(targets) == 2  # legitimate reject excluded, both fixable/ambiguous kept


def test_gleaning_targets_flags_mispaired_context_literal_for_repair():
    # A literal->linked relation the grounding/reader could not confirm for its paired condition
    # (three_point_gate_failed / unknown_context_literal) becomes a fixable repair target carrying the
    # literal, so the teacher can re-pair it -- never silently re-paired here.
    def rel(reason, literal):
        return {"relation": "tests_for", "detail_reason": reason, "rejection_id": "sha256:" + reason,
                "evidence": {"arguments": [
                    {"role": "subject", "kind": "linked", "occurrence_id": "chf",
                     "support_property": "heart disease"},
                    {"role": "object", "kind": "context", "literal": literal},
                ]}}
    # a span->span three_point_gate_failed must NOT be treated as a mispaired literal
    span_span = {"relation": "tests_for", "detail_reason": "three_point_gate_failed",
                 "rejection_id": "sha256:ss", "evidence": {"arguments": [
                     {"role": "subject", "kind": "linked", "occurrence_id": "a", "support_property": "x"},
                     {"role": "object", "kind": "linked", "occurrence_id": "b", "support_property": "y"}]}}
    occ = {"chf": {"occurrence_id": "chf", "decision_id": "d-chf"}}
    targets = qa_builder._gleaning_targets(
        "", [],
        [rel("three_point_gate_failed", "hemoglobin A1c"),
         rel("unknown_context_literal", "ct scan"), span_span],
        [], occ)
    mis = [t for t in targets if t.get("reason") == "mispaired_context_literal"]
    assert len(mis) == 2  # both the reader-failed and the ungroundable literal->linked cases only
    for t in mis:
        assert t["kind"] == "fixable"
        assert any(a.get("kind") == "context" and a.get("literal") for a in t.get("arguments") or [])
    # the span->span three_point_gate_failed produced no mispaired target
    assert len(targets) == 2


def test_rejection_safe_arguments_strips_controlled_surface_keeps_uncontrolled_literal():
    args = [
        {"role": "subject", "kind": "linked", "occurrence_id": "o1",
         "surface": "hypothyroidism", "support_property": "an endocrine condition"},
        {"role": "object", "kind": "context", "literal": "dietary modifications",
         "runtime_type": "medical-procedure", "start": 5, "end": 9},
    ]
    safe = qa_builder._rejection_safe_arguments(args)
    blob = json.dumps(safe).casefold()
    # a linked argument's controlled surface is stripped (protected leak)...
    assert "hypothyroidism" not in blob
    assert safe[0]["occurrence_id"] == "o1" and safe[0]["support_property"] == "an endocrine condition"
    # ...but a context argument's UNCONTROLLED literal is retained (needed to re-pair a mispaired
    # literal in the gleaning repair pass; it is kept verbatim in doc_p, so not a protected leak).
    assert safe[1]["literal"] == "dietary modifications"
    assert safe[1]["runtime_type"] == "medical-procedure"


def test_relation_repair_prompt_restricts_to_targets_and_lists_hints():
    # reflux + dietary(target) + heart-failure/ace(non-target) in the same doc; the repair prompt
    # must show only the target's spans/cards and a REPAIR TARGETS line with the fix hint.
    # non-target entities chosen to NOT appear in the worked-examples text
    source = ("for your first problem , your reflux , continue dietary modifications . "
              "for your second problem , your insomnia , continue the zolpidem .")
    def sp(x): return source.index(x)
    environment = {
        "occurrences": [
            {"occurrence_id": "reflux", "decision_id": "d-reflux", "surface": "reflux",
             "start": sp("reflux"), "end": sp("reflux") + 6, "runtime_type": "health-condition"},
            {"occurrence_id": "diet", "decision_id": "d-diet", "surface": "dietary modifications",
             "start": sp("dietary modifications"), "end": sp("dietary modifications") + 21,
             "runtime_type": "medical-procedure"},
            {"occurrence_id": "ins", "decision_id": "d-ins", "surface": "insomnia",
             "start": sp("insomnia"), "end": sp("insomnia") + 8, "runtime_type": "health-condition"},
            {"occurrence_id": "zol", "decision_id": "d-zol", "surface": "zolpidem",
             "start": sp("zolpidem"), "end": sp("zolpidem") + 8, "runtime_type": "drug"},
        ],
        "decisions": [
            {"decision_id": "d-reflux", "actions": [{"mode": "level", "legal": True, "entails": ["gastrointestinal condition"]}]},
            {"decision_id": "d-diet", "actions": [{"mode": "level", "legal": True, "entails": ["dietary intervention"]}]},
            {"decision_id": "d-ins", "actions": [{"mode": "level", "legal": True, "entails": ["sleep disorder"]}]},
            {"decision_id": "d-zol", "actions": [{"mode": "level", "legal": True, "entails": ["sedative"]}]},
        ],
    }
    inv = {str(r["decision_id"]): r["span_label"] for r in qa_builder.relation_teacher_span_inventory(environment)}
    target = {"kind": "ambiguous", "relation": "procedure_for", "hint": "make it uniquely answerable",
              "arguments": [
                  {"role": "subject", "kind": "linked", "occurrence_id": "reflux"},
                  {"role": "object", "kind": "linked", "occurrence_id": "diet"}]}
    prompt = qa_builder.relation_repair_prompt("d", source, environment, [target])

    assert "REPAIR TARGETS" in prompt and "procedure_for" in prompt
    assert "make it uniquely answerable" in prompt and "AMBIGUOUS" in prompt
    # the target's spans are shown; the non-target insomnia/zolpidem block is excluded
    assert inv["d-reflux"] in prompt and inv["d-diet"] in prompt
    assert "insomnia" not in prompt.split("SOURCE DOCUMENT")[0]  # only in the source, not spans/cards
    assert "zolpidem" not in prompt.split("SOURCE DOCUMENT")[0]
    # safety-critical privacy instruction survives
    assert "the QUESTION must never contain the accepted answer" in prompt
