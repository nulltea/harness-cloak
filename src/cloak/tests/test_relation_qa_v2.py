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


def test_cross_clause_span_anchor_keeps_pairs_beyond_soft_global_distance():
    source = "Muscle strain was assessed. " + " ".join(
        f"Unrelated clause {index}." for index in range(qa_builder._CROSS_CLAUSE_ANCHOR_MAX_DISTANCE)
    ) + " Meloxicam was prescribed for pain."
    quote, span, clause_ranges, error, kind, _, _ = _cross_clause_anchor(source)

    assert error is None and kind == "stitched_clauses"
    assert span == (0, len(source))
    assert "Unrelated clause 0" not in quote
    assert clause_ranges == [
        qa_builder._source_clause_spans(source)[0],
        qa_builder._source_clause_spans(source)[-1],
    ]


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


def test_cross_clause_context_anchor_keeps_literal_beyond_soft_global_distance():
    source = "Distal phalanx fracture was assessed. " + " ".join(
        f"Unrelated clause {index}." for index in range(qa_builder._CROSS_CLAUSE_ANCHOR_MAX_DISTANCE)
    ) + " Follow-up x-ray was ordered."
    quote, span, clause_ranges, error, kind, _, _ = _cross_clause_context_anchor(source)

    assert error is None and kind == "stitched_clauses"
    assert span == (0, len(source))
    assert "Unrelated clause 0" not in quote
    assert clause_ranges == [
        qa_builder._source_clause_spans(source)[0],
        qa_builder._source_clause_spans(source)[-1],
    ]


def test_plan_section_anchor_narrows_reader_turns_to_argument_clauses():
    source = (
        "Arthritis.\n"
        "• Medical Reasoning: Symptoms have worsened.\n"
        "• Additional Testing: The patient denies fever.\n"
        "• Medical Treatment: Meloxicam was prescribed for arthritis.\n"
    )
    arthritis = source.index("Arthritis")
    meloxicam = source.index("Meloxicam")
    arguments = [
        {"role": "subject", "kind": "linked", "occurrence_id": "condition",
         "surface": "Arthritis", "runtime_type": "health-condition",
         "support_property": "joint condition"},
        {"role": "object", "kind": "linked", "occurrence_id": "medication",
         "surface": "Meloxicam", "runtime_type": "drug",
         "support_property": "anti-inflammatory medication"},
    ]
    occurrences = {
        "condition": {"occurrence_id": "condition", "decision_id": "d-condition",
                      "surface": "Arthritis", "start": arthritis, "end": arthritis + 9,
                      "runtime_type": "health-condition"},
        "medication": {"occurrence_id": "medication", "decision_id": "d-medication",
                       "surface": "Meloxicam", "start": meloxicam, "end": meloxicam + 9,
                       "runtime_type": "drug"},
    }

    quote, span, clause_ranges, error, kind = qa_builder._derived_relation_anchor(
        source, arguments, occurrences, {}, "prescribed_with", qa_builder.ACI_RELATION_CONTRACT,
    )

    assert error is None and kind == "plan_section"
    assert span == (0, len(source))
    assert "patient denies fever" in quote  # full envelope remains source evidence
    assert clause_ranges == [
        qa_builder._source_clause_spans(source)[0],
        qa_builder._source_clause_spans(source)[3],
    ]
    assert qa_builder._source_turns_for_ranges(source, clause_ranges) == [0, 3]


def test_problem_block_anchor_narrows_reader_turns_to_argument_turns():
    source = (
        "[doctor] assessment and plan for your first problem, arthritis is active.\n"
        "[patient] okay.\n"
        "[doctor] meloxicam was prescribed for arthritis.\n"
        "[doctor] for your second problem, hypothyroidism.\n"
    )
    arthritis = source.index("arthritis")
    meloxicam = source.index("meloxicam")
    arguments = [
        {"role": "subject", "kind": "linked", "occurrence_id": "arthritis",
         "surface": "arthritis", "runtime_type": "health-condition",
         "support_property": "joint condition"},
        {"role": "object", "kind": "linked", "occurrence_id": "meloxicam",
         "surface": "meloxicam", "runtime_type": "drug",
         "support_property": "anti-inflammatory medication"},
    ]
    occurrences = {
        "arthritis": {"occurrence_id": "arthritis", "decision_id": "d-arthritis",
                      "surface": "arthritis", "start": arthritis, "end": arthritis + 9,
                      "runtime_type": "health-condition"},
        "meloxicam": {"occurrence_id": "meloxicam", "decision_id": "d-meloxicam",
                       "surface": "meloxicam", "start": meloxicam, "end": meloxicam + 9,
                       "runtime_type": "drug"},
    }

    quote, span, clause_ranges, error, kind = qa_builder._derived_relation_anchor(
        source, arguments, occurrences, {}, "prescribed_with", qa_builder.ACI_RELATION_CONTRACT,
    )

    assert error is None and kind == "problem_block"
    assert span == qa_builder._shared_problem_block(source, [
        (arthritis, arthritis + 9), (meloxicam, meloxicam + 9),
    ])
    assert "[patient] okay" in quote  # full envelope remains source evidence
    assert clause_ranges == [
        qa_builder._source_clause_spans(source)[0],
        qa_builder._source_clause_spans(source)[2],
    ]
    assert qa_builder._source_turns_for_ranges(source, clause_ranges) == [0, 2]


def test_compile_records_beyond_soft_cross_clause_cap_without_rejecting():
    source = "Muscle strain was assessed. " + " ".join(
        f"Unrelated clause {index}." for index in range(qa_builder._CROSS_CLAUSE_ANCHOR_MAX_DISTANCE)
    ) + " Meloxicam was prescribed for Muscle strain."
    condition_start = source.index("Muscle strain")
    medication_start = source.index("Meloxicam")
    environment = {
        "occurrences": [
            {"occurrence_id": "condition", "decision_id": "d-condition",
             "surface": "Muscle strain", "start": condition_start,
             "end": condition_start + len("Muscle strain"), "runtime_type": "health-condition"},
            {"occurrence_id": "medication", "decision_id": "d-medication",
             "surface": "Meloxicam", "start": medication_start,
             "end": medication_start + len("Meloxicam"), "runtime_type": "drug"},
        ],
        "decisions": [
            {"decision_id": "d-condition", "actions": [{"mode": "level", "legal": True,
                "entails": ["muscle injury"]}]},
            {"decision_id": "d-medication", "actions": [{"mode": "level", "legal": True,
                "entails": ["anti-inflammatory medication"]}]},
        ],
    }
    proposal = {
        "relation": "prescribed_with",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S1",
             "support_property": "muscle injury"},
            {"role": "object", "kind": "linked", "span_label": "S2",
             "support_property": "anti-inflammatory medication"},
        ],
        "question": "What medication category was prescribed for the muscle injury?",
        "accepted_answers": ["anti-inflammatory medication"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == []
    assert accepted[0]["evidence"]["anchor_diagnostics"] == [{
        "kind": "soft_cross_clause_cap_exceeded",
        "clause_distance": qa_builder._CROSS_CLAUSE_ANCHOR_MAX_DISTANCE + 1,
        "soft_cap": qa_builder._CROSS_CLAUSE_ANCHOR_MAX_DISTANCE,
    }]


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


def test_linked_relation_gold_is_derived_from_selected_property():
    source = "Hypothyroidism is treated with Synthroid."
    accepted, rejected = compile_relational_assertions("d2", source, _environment(source), [_proposal(source)])

    assert rejected == []
    assert accepted[0]["relation"] == "prescribed_with"
    assert accepted[0]["question"] == _proposal(source)["question"]
    assert accepted[0]["accepted_values"] == ["thyroid medication"]
    assert accepted[0]["decision_requirements"] == {"d-condition": "endocrine condition", "d-drug": "thyroid medication"}


def test_span_span_relation_flips_orientation_answer_is_subject():
    # Ambiguity flip (phase 1): two linked spans, answer_role=subject -> the DRUG is the question
    # locator and the CONDITION is the answer, so "which condition is <drug> for?" is unique even
    # when the condition has several drugs. Same directional fact; only QA orientation changes.
    source = "Hypothyroidism is treated with Synthroid."
    proposal = {
        "relation": "prescribed_with",
        "answer_role": "subject",
        "arguments": [
            {"role": "subject", "kind": "linked", "occurrence_id": "condition",
             "support_property": "endocrine condition"},
            {"role": "object", "kind": "linked", "occurrence_id": "drug",
             "support_property": "thyroid medication"},
        ],
        "question": "What medical condition was the thyroid medication prescribed for?",
        "accepted_answers": ["endocrine condition"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
        "evidence_quote": source,
    }
    accepted, rejected = compile_relational_assertions("d2", source, _environment(source), [proposal])

    assert rejected == [], rejected
    assert accepted[0]["accepted_values"] == ["endocrine condition"]          # subject is the answer
    assert accepted[0]["answer_target"]["decision_id"] == "d-condition"        # target = subject decision
    assert accepted[0]["answer_target"]["required_property"] == "endocrine condition"


def test_source_literal_spans_plural_fold():
    S = qa_builder._source_literal_spans
    doc = "reviewed the x-rays and ordered morning aldosterone levels today"
    assert [doc[a:b] for a, b in S(doc, "x-ray")] == ["x-rays"]                 # singular -> plural source
    assert [doc[a:b] for a, b in S(doc, "aldosterone level")] == ["aldosterone levels"]
    doc2 = "ordered an x-ray today"
    assert [doc2[a:b] for a, b in S(doc2, "x-rays")] == ["x-ray"]  # plural literal -> singular source
    assert S("went into labor", "lab") == []                                    # boundary: no over-match


def test_reverse_framed_proposals_only_ambiguous_and_reverse_unique():
    occurrences = {
        "oc": {"occurrence_id": "oc", "decision_id": "d-cond"},
        "oa": {"occurrence_id": "oa", "decision_id": "d-a"},
        "ob": {"occurrence_id": "ob", "decision_id": "d-b"},
        "oc2": {"occurrence_id": "oc2", "decision_id": "d-cond2"},
        "os": {"occurrence_id": "os", "decision_id": "d-shared"},
    }
    span_labels = {"S1": "oc", "S2": "oa", "S3": "ob", "S4": "oc2", "S5": "os"}

    def rel(subj, obj, sp_o):
        return {"relation": "prescribed_with", "arguments": [
            {"role": "subject", "kind": "linked", "span_label": subj, "support_property": "heart disease"},
            {"role": "object", "kind": "linked", "span_label": obj, "support_property": sp_o}]}

    proposals = [
        rel("S1", "S2", "loop diuretic"),      # heart disease -> three drugs = ambiguous group;
        rel("S1", "S3", "ace inhibitor"),      # flip EVERY object in the group (reader filters),
        rel("S1", "S5", "shared drug"),        # including the shared one
        rel("S4", "S5", "shared drug"),        # S4 has a single object -> NOT ambiguous -> no flip
    ]
    variants = qa_builder._reverse_framed_proposals(proposals, span_labels, occurrences)
    qs = {v["question"] for v in variants}
    assert qs == {
        "For what medical condition was the loop diuretic prescribed?",
        "For what medical condition was the ace inhibitor prescribed?",
        "For what medical condition was the shared drug prescribed?",
    }  # all 3 objects of the ambiguous subject S1; nothing for single-object S4
    for v in variants:
        assert v["answer_role"] == "subject"
        assert v["accepted_answers"] == ["heart disease"]
        assert v["_reverse_framed"] is True


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
        "question": "For what medical condition were thyroid labs ordered?",
        "accepted_answers": ["endocrine condition"],
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
    evidence = accepted[0]["evidence"]
    assert evidence["reader_clauses"] == qa_builder._source_reader_clause_refs(source, [
        (source.index("anti-inflammatory medications"),
         source.index("anti-inflammatory medications") + len("anti-inflammatory medications")),
        (transplant_start, transplant_start + len("kidney transplant")),
    ])
    excerpt = qa_builder._reader_excerpt(source, evidence)
    assert "you are doing well" not in excerpt
    assert "let us move on" not in excerpt


def test_turn_excerpt_narrows_a_long_speaker_turn_to_pinned_clauses():
    source = (
        "[doctor] unrelated intake detail . arthritis is active . unrelated review detail . "
        "meloxicam was prescribed for arthritis . unrelated closing detail .\n[patient] okay ."
    )
    arthritis = source.index("arthritis")
    meloxicam = source.index("meloxicam")

    clauses = qa_builder._source_reader_clause_refs(
        source,
        [(arthritis, arthritis + len("arthritis")), (meloxicam, meloxicam + len("meloxicam"))],
    )
    excerpt = qa_builder._turn_excerpt(source, [0], window=0, core_clauses=clauses)

    assert "arthritis is active" in excerpt
    assert "meloxicam was prescribed" in excerpt
    assert "unrelated intake detail" not in excerpt
    assert "unrelated review detail" not in excerpt
    assert "unrelated closing detail" not in excerpt


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
        "question": "What medical condition explains knee pain?",
        "accepted_answers": ["bone inflammation disease"],
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
    # Sanitize swaps the protected surface for the subject level. The relation-contract-derived
    # generic answer word "medication" remains legal for the canonical linked answer.
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
        "tests_for", "S1", "joint condition", "S2", "immunology panel",
        "What testing was ordered to evaluate the joint condition?", "immunology panel",
    )

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == []
    assert accepted[0]["relation"] == "tests_for"
    assert accepted[0]["occurrence_ids"] == ["arthritis", "panel"]
    assert accepted[0]["evidence"]["reader_turns"] == [2, 4]
    assert accepted[0]["evidence"]["source_clause_ranges"] == [
        list(qa_builder._source_clause_spans(source)[2]),
        list(qa_builder._source_clause_spans(source)[4]),
    ]


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


def test_linked_answer_ignores_unrelated_split_surface_in_protected_answer_check():
    source = (
        "[doctor] type 2 diabetes requires hemoglobin a1c monitoring .\n"
        "[doctor] lastly , what about your diabetes ?\n"
    )
    type_two_start = source.index("type 2 diabetes")
    diabetes_start = source.index("diabetes", type_two_start + len("type 2 diabetes"))
    environment = {
        "occurrences": [
            {"occurrence_id": "type-two", "decision_id": "d-type-two",
             "surface": "type 2 diabetes", "start": type_two_start,
             "end": type_two_start + len("type 2 diabetes"),
             "runtime_type": "health-condition"},
            {"occurrence_id": "diabetes-mention", "decision_id": "d-diabetes-mention",
             "surface": "diabetes", "start": diabetes_start,
             "end": diabetes_start + len("diabetes"),
             "runtime_type": "health-condition"},
        ],
        "decisions": [
            {"decision_id": "d-type-two", "actions": [{"mode": "level", "legal": True,
             "entails": ["diabetes mellitus", "medical condition"]}]},
            {"decision_id": "d-diabetes-mention", "actions": [{"mode": "level", "legal": True,
             "entails": ["glucose metabolism disease", "disease of metabolism"]}]},
        ],
    }
    proposal = {
        "relation": "tests_for",
        "answer_role": "subject",
        "arguments": [
            {"role": "subject", "kind": "linked", "span_label": "S1",
             "support_property": "diabetes mellitus", "literal": None},
            {"role": "object", "kind": "context", "span_label": None,
             "support_property": None, "literal": "hemoglobin a1c"},
        ],
        "question": "For what medical condition was hemoglobin a1c monitored?",
        "accepted_answers": ["diabetes mellitus"],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }

    accepted, rejected = compile_relational_assertions("d2", source, environment, [proposal])

    assert rejected == [], rejected
    assert accepted[0]["accepted_values"] == ["diabetes mellitus"]
    assert accepted[0]["answer_target"] == {
        "kind": "linked_decision",
        "decision_id": "d-type-two",
        "required_property": "diabetes mellitus",
    }


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


def test_conditional_relation_definite_question_flagged_not_blocked():
    # DEMOTED 2026-07-19: a hedged source + definite question is a non-blocking diagnostic, not a
    # reject (regex hedge/question detectors too blunt; see docs/issues/qa-builder-dept.md). The
    # relation reaches the reader gate; compile records the modality diagnostic on its evidence.
    env = _conditional_referral_environment(_CONDITIONAL_SOURCE)
    proposal = _conditional_referral_proposal(
        "What procedure was performed for the musculoskeletal system disease?")
    accepted, rejected = compile_relational_assertions("d2", _CONDITIONAL_SOURCE, env, [proposal])

    assert not any(r.get("detail_reason") == "hedged_relation" for r in rejected), rejected
    assert accepted, "hedged relation must no longer be blocked from the reader"
    assert accepted[0]["evidence"]["modality_diagnostics"] == ["hedged_source_definite_question"]


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


def test_problem_block_hedged_relation_not_blocked_by_hedge_gate():
    # The conditional PT referral ("if symptoms continue ... possibly referral") is no longer
    # rejected as `hedged_relation` (demoted 2026-07-19). With the hedge out of the way this
    # particular relation falls to a DIFFERENT downstream gate — the context-literal object forces
    # the condition as the scored answer, which the question names → answer_leakage. The point:
    # the hedge gate itself no longer blocks.
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

    reasons = {r.get("detail_reason") for r in rejected}
    assert "hedged_relation" not in reasons, rejected     # the hedge gate no longer blocks
    assert "answer_leakage" in reasons                    # caught by a real downstream gate instead


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


def test_linked_teacher_answers_do_not_override_canonical_gold():
    # Linked answers are scored against their selected lattice property, not
    # teacher-authored lexical answer text.
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
    assert accepted[0]["accepted_values"] == ["thyroid medication"]
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
    assert accepted[0]["evidence"]["reader_turns"] == [0, 3]
    assert accepted[0]["evidence"]["source_clause_ranges"] == [
        list(qa_builder._source_clause_spans(source)[0]),
        list(qa_builder._source_clause_spans(source)[3]),
    ]




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
    def reader(questions, context, clauses=None):
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
        render_action_vector=lambda d, v: "x", reader=lambda q, c, cl=None: [""],
        chain_by_decision={}, reader_threshold=1.0,
    )
    assert result is None


def test_relation_support_opportunities_counts_distinct_compiler_shaped_facts():
    source = "Hypothyroidism is treated with Synthroid."

    opportunities = qa_builder.relation_support_opportunities(source, _environment(source))

    # The opportunity miner keeps its lexical-cue precision filter (unlike the compiler,
    # see docs/issues/qa-builder-dept.md): only the cued prescribed_with pair survives.
    assert [(row["relation"], row["scope"]) for row in opportunities] == [
        ("prescribed_with", "span_span"),
    ]


def test_relation_support_opportunities_augment_adds_gazetteer_unreachable_literal():
    # The gazetteer emits only test/procedure/status/category literals, so a DRUG literal object is
    # structurally unreachable. The prefilter augment recovers it; cue-matched ("treated with") so no
    # judge is needed. Also asserts the no-drop invariant (augmented set superset of baseline).
    source = "Hypothyroidism was treated with levothyroxine daily."
    env = {
        "occurrences": [
            {"occurrence_id": "condition", "decision_id": "d-condition", "surface": "Hypothyroidism",
             "start": 0, "end": 14, "runtime_type": "health-condition"},
        ],
        "decisions": [
            {"decision_id": "d-condition",
             "actions": [{"mode": "level", "legal": True, "entails": ["endocrine condition"]}]},
        ],
    }
    baseline = qa_builder.relation_support_opportunities(source, env)
    assert not any(row["relation"] == "prescribed_with" for row in baseline)

    start = source.index("levothyroxine")
    drug_literal = {
        "context_candidate_id": "context:levo", "kind": "context_literal", "runtime_type": "drug",
        "literal": "levothyroxine", "start": start, "end": start + len("levothyroxine"),
        "provenance": "llm_prefilter",
    }
    augmented = qa_builder.relation_support_opportunities(
        source, env, extra_context_candidates=[drug_literal])

    base_keys = {qa_builder._stable_hash(row["fact_key"]) for row in baseline}
    aug_keys = {qa_builder._stable_hash(row["fact_key"]) for row in augmented}
    assert base_keys <= aug_keys  # no-drop invariant
    assert any(row["relation"] == "prescribed_with" and row["scope"] == "span_literal"
               for row in augmented)


def test_lattice_level_suspect_suppressed_for_ambiguity_and_kept_sibling():
    probe = {"surface": "tsh", "runtime_type": "lab", "unreadable_level": "thyroid test",
             "readable_coarser_level": "lab test", "chain": []}
    document = {"decisions": [{"decision_id": "d1", "canonical_key": "tsh"},
                              {"decision_id": "d2", "canonical_key": "cbc"}]}

    def artifact(extra_evidence, kept_decision):
        return {
            "documents": {"aci/D": document},
            "assertions": {"a1": {"doc_id": "aci/D", "decision_requirements": {kept_decision: "x"}}},
            "rejections": {"records": [{
                "doc_id": "aci/D", "detail_reason": "three_point_gate_failed",
                "evidence": {"validation": {"scores": {}}, "lattice_probe": probe, **extra_evidence}}]},
        }

    def suspects(art):
        flags = qa_builder.compute_review_flags(art)
        return sum(1 for lst in flags.values() for f in lst if f.get("code") == "lattice_level_suspect")

    # unambiguous AND tsh never kept in-doc (only cbc kept) -> genuine suspect, fires
    assert suspects(artifact({}, "d2")) == 1
    # ambiguity-driven failure -> suppressed
    assert suspects(artifact({"answer_competing": ["a", "b"]}, "d2")) == 0
    # tsh's decision is used by a kept relation (provably readable) -> suppressed
    assert suspects(artifact({}, "d1")) == 0


def test_display_locators_strips_lead_verbs_and_drops_superset_duplicates():
    out = qa_builder._display_locators(
        ["order a thyroid panel", "referral to physical therapy", "physical therapy", "creatinine"])
    assert "thyroid panel" in out                     # leading verb + article stripped
    assert "physical therapy" in out
    assert "referral to physical therapy" not in out  # near-duplicate superset dropped
    assert "creatinine" in out


def test_literal_reverse_assertions_builds_compound_condition_answer():
    occurrences = {
        "cond": {"occurrence_id": "cond", "decision_id": "d-c", "surface": "CHF", "start": 0, "end": 3,
                 "runtime_type": "health-condition", "controlled": True},
    }
    decisions_by_id = {"d-c": {"decision_id": "d-c", "actions": [
        {"mode": "level", "legal": True, "entails": ["heart disease"]}]}}
    document = "CHF; ordered bmp and a lipid panel."

    def opp(literal):
        start = document.index(literal)
        return {"relation": "tests_for", "scope": "span_literal", "recovered_by_escalation": True,
                "arguments": [
                    {"role": "subject", "kind": "linked", "occurrence_id": "cond"},
                    {"role": "object", "kind": "context", "literal": literal,
                     "start": start, "end": start + len(literal)}]}

    rows = qa_builder._literal_reverse_assertions(
        document, [opp("bmp"), opp("a lipid panel")], occurrences, decisions_by_id)

    assert len(rows) == 1
    row = rows[0]
    assert row["relation"] == "tests_for"
    assert row["answer_target"] == {"kind": "linked_decision", "decision_id": "d-c",
                                    "required_property": "heart disease"}
    assert row["accepted_values"] == ["heart disease"]
    assert "bmp" in row["question"] and "lipid panel" in row["question"]  # compound locator
    assert "CHF" not in row["question"]                                   # answer never in question
    assert row["occurrence_ids"] == ["cond"]                              # only the condition is linked
    # a non-judge-accepted opportunity is ignored
    assert qa_builder._literal_reverse_assertions(
        document, [{**opp("bmp"), "recovered_by_escalation": False}], occurrences, decisions_by_id) == []


def test_first_noncolliding_level_escalates_past_foreign_surface():
    levels = ["congestive heart failure", "heart disease", "medical condition"]
    # finest level is byte-identical to ANOTHER decision's protected surface -> escalate
    assert qa_builder._first_noncolliding_level(
        levels, ["congestive heart failure"]) == "heart disease"
    # whole-word containment of a foreign surface also collides
    assert qa_builder._first_noncolliding_level(
        ["kidney transplant history", "organ history"], ["kidney transplant"]) == "organ history"
    # every level colliding -> fall back to finest (compile's leak check stays the backstop)
    assert qa_builder._first_noncolliding_level(
        ["heart disease"], ["heart disease"]) == "heart disease"
    # no foreign collision -> finest wins as before
    assert qa_builder._first_noncolliding_level(levels, ["hypertension"]) == levels[0]


def test_stage_proposal_locator_escalates_on_foreign_surface_collision():
    occurrences, decisions, opportunity = _stage_fixture()
    # a DIFFERENT decision's raw surface equals the condition's finest level
    occurrences = dict(occurrences, other={
        "occurrence_id": "other", "decision_id": "d-o", "surface": "congestive heart failure",
        "start": 60, "end": 84, "runtime_type": "health-condition", "controlled": True})
    [plan] = qa_builder._deterministic_relation_plans(
        [opportunity("tests_for", "cond", "test1")], occurrences, decisions)

    forward = qa_builder._deterministic_stage_proposal(
        plan, "forward", "cardiac imaging study", occurrences, decisions,
        {"d-c": "S1", "d-t": "S4"})

    # locator escalated past the colliding finest level to "heart disease"
    assert forward["question"] == (
        "Which test or investigation was ordered to evaluate the heart disease?")


def test_gleaning_targets_drop_literal_collision_protected_locator():
    occurrences = {
        "cond": {"occurrence_id": "cond", "decision_id": "d-c", "start": 0, "end": 3},
    }
    arguments = [
        {"role": "subject", "kind": "linked", "occurrence_id": "cond"},
        {"role": "object", "kind": "context", "literal": "kidney function"},
    ]
    def rejection(extra_evidence):
        return {"relation": "tests_for", "detail_reason": "protected_locator",
                "evidence": {"arguments": arguments, **extra_evidence}}

    # literal-collision leak: dead weight, never a repair target
    assert qa_builder._gleaning_targets(
        "CHF; checked kidney function.", [],
        [rejection({"leak_source": "context_literal"})], [], occurrences) == []
    # an ordinary protected_locator (recolorable) stays fixable
    targets = qa_builder._gleaning_targets(
        "CHF; checked kidney function.", [], [rejection({})], [], occurrences)
    assert [t["kind"] for t in targets] == ["fixable"]
    assert targets[0]["reason"] == "protected_locator"


def test_finer_level_readability_certifies_reward_band():
    decisions = [{
        "decision_id": "d-c",
        "canonical_key": "chf",
        "actions": [
            {"action_id": "keep-c", "mode": "keep", "legal": True},
            {"action_id": "act-fine", "mode": "level", "legal": True, "coarseness_rank": 10,
             "fill": "congestive heart failure",
             "entails": ["congestive heart failure", "heart disease"]},
            {"action_id": "act-coarse", "mode": "level", "legal": True, "coarseness_rank": 20,
             "fill": "heart disease", "entails": ["heart disease"]},
            {"action_id": "ph-c", "mode": "placeholder", "legal": True, "entails": []},
        ],
    }]
    candidate = {
        "question": "For what medical condition was the loop diuretic prescribed?",
        "decision_requirements": {"d-c": "heart disease"},
        "answer_target": {"kind": "linked_decision", "decision_id": "d-c",
                          "required_property": "heart disease"},
        "evidence": {"reader_turns": [0]},
    }
    chains_ok = {"d-c": [
        {"node": "congestive heart failure", "answer_aliases": ["congestive heart failure"],
         "entailed_properties": ["congestive heart failure", "heart disease"]},
        {"node": "heart disease", "answer_aliases": ["heart disease"],
         "entailed_properties": ["heart disease"]},
    ]}

    def render(doc_id, vector):
        return ("on congestive heart failure meds" if vector.get("d-c") == "act-fine"
                else "on heart disease meds")

    def reader(questions, context, clauses=None):
        return ["congestive heart failure" if "congestive heart failure" in context
                else "heart disease"] * len(questions)

    scores = qa_builder._finer_level_readability(
        candidate, decisions, [], doc_id="d", render_action_vector=render,
        reader=reader, chain_by_decision=chains_ok)
    # the finer render is read and its node ENTAILS the supported level -> band certified
    assert scores == {"congestive heart failure": {"score": 1.0, "render": "ok"}}

    # broken finer alias: the reader echo cannot resolve -> unreadable finer level recorded as 0
    chains_broken = {"d-c": [
        {"node": "congestive heart failure", "answer_aliases": [],
         "entailed_properties": ["congestive heart failure", "heart disease"]},
        {"node": "heart disease", "answer_aliases": ["heart disease"],
         "entailed_properties": ["heart disease"]},
    ]}
    scores = qa_builder._finer_level_readability(
        candidate, decisions, [], doc_id="d", render_action_vector=render,
        reader=reader, chain_by_decision=chains_broken)
    assert scores == {"congestive heart failure": {"score": 0.0, "render": "ok"}}

    # supported level already finest -> empty band -> no check
    finest_pinned = {**candidate,
                     "decision_requirements": {"d-c": "congestive heart failure"},
                     "answer_target": {"kind": "linked_decision", "decision_id": "d-c",
                                       "required_property": "congestive heart failure"}}
    assert qa_builder._finer_level_readability(
        finest_pinned, decisions, [], doc_id="d", render_action_vector=render,
        reader=reader, chain_by_decision=chains_ok) is None
    # set rows are skipped (members pinned finest)
    set_row = {**candidate, "answer_target": {"kind": "linked_decision_set", "members": []}}
    assert qa_builder._finer_level_readability(
        set_row, decisions, [], doc_id="d", render_action_vector=render,
        reader=reader, chain_by_decision=chains_ok) is None


def test_write_finer_level_failures_emits_lattice_worklist(tmp_path):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "scripts"))
    import build_qa_utility_artifact as bqa
    artifact = {
        "reader_threshold": 1.0,
        "documents": {"aci/DX": {
            "decisions": [{"decision_id": "d-c", "canonical_key": "chf",
                           "runtime_type": "health-condition"}],
            "occurrences": [
                {"occurrence_id": "cond", "decision_id": "d-c", "surface": "chf"},
                {"occurrence_id": "drug1", "decision_id": "d-1", "surface": "lasix"}],
        }},
        "assertions": {
            "a1": {  # one unreadable finer level -> emitted
                "subtype": "contextual_relation", "relation": "prescribed_with",
                "doc_id": "aci/DX",
                "question": "For what medical condition was the loop diuretic prescribed?",
                "answer_target": {"kind": "linked_decision", "decision_id": "d-c",
                                  "required_property": "heart disease"},
                "evidence": {"reader_turns": [0], "arguments": [
                    {"role": "subject", "kind": "linked", "occurrence_id": "cond"},
                    {"role": "object", "kind": "linked", "occurrence_id": "drug1"}],
                    "validation": {"finer_levels": {
                        "congestive heart failure": 0.0, "cardiac failure": 1.0}}},
            },
            "a2": {  # full band readable -> NOT emitted
                "subtype": "contextual_relation", "relation": "tests_for", "doc_id": "aci/DX",
                "question": "q2?", "answer_target": {"kind": "linked_decision",
                                                     "decision_id": "d-c",
                                                     "required_property": "heart disease"},
                "evidence": {"reader_turns": [0], "arguments": [],
                             "validation": {"finer_levels": {"cardiac failure": 1.0}}},
            },
        },
    }
    sources = {"aci/DX": "chf on lasix daily."}

    # a HARD-mode rejection (never an assertion) is also emitted, from its rejection evidence
    artifact["rejections"] = {"summary_by_reason": {"unsupported": 1}, "records": [{
        "doc_id": "aci/DX", "relation": "tests_for", "reason": "unsupported",
        "detail_reason": "finer_level_unreadable",
        "evidence": {"reader_turns": [0],
                     "arguments": [{"role": "subject", "kind": "linked",
                                    "occurrence_id": "cond"}],
                     "question": "Which test was ordered for the heart disease?",
                     "answer_target": {"kind": "linked_decision", "decision_id": "d-c",
                                       "required_property": "heart disease"},
                     "finer_levels": {"congestive heart failure": 0.0}},
    }]}

    path, count = bqa.write_finer_level_failures(artifact, sources, tmp_path / "probe.json")

    assert count == 2 and path.name == "probe.finer-level-failures.jsonl"
    row, rejected_row = [json.loads(line) for line in path.read_text().splitlines()]
    assert row["profile"] == {"key": "chf", "runtime_type": "health-condition"}
    assert row["levels_rejected"] == ["congestive heart failure"]
    # accepted = the verified ladder: gate-passed supported level + band-confirmed finer levels
    assert row["levels_accepted"] == ["cardiac failure", "heart disease"]
    assert row["rejection_causes"] == {"congestive heart failure": "read_failed"}
    assert row["relation"] == {"type": "prescribed_with", "provenance": "teacher_primary",
                               "answer_role": "subject", "subject": "chf", "object": "lasix",
                               "supported_level": "heart disease"}
    assert row["doc_context"] == "chf on lasix daily."
    assert rejected_row["relation"]["type"] == "tests_for"
    assert rejected_row["question"] == "Which test was ordered for the heart disease?"
    assert rejected_row["levels_rejected"] == ["congestive heart failure"]
    assert rejected_row["levels_accepted"] == ["heart disease"]  # never empty: supported is in


def test_reader_outcome_route_signatures():
    def rejection(original, representative, placeholder, teacher_id=None):
        evidence = {"validation": {"scores": {
            "original": original, "representative": representative, "placeholder": placeholder}}}
        if teacher_id:
            evidence["teacher_id"] = teacher_id
        return {"evidence": evidence}

    route = qa_builder._reader_outcome_route
    # orig/rep verdict disagreement (either direction) or placeholder-only readable -> lattice
    assert route(rejection(1.0, 0.0, 0.0)) == "lattice_suspect"
    assert route(rejection(0.0, 1.0, 0.0)) == "lattice_suspect"
    assert route(rejection(0.0, 0.0, 1.0)) == "lattice_suspect"
    # read nowhere: no_relation ONLY for deterministic authorship; teacher rejections keep repair
    assert route(rejection(0.0, 0.0, 0.0, teacher_id="deterministic")) == "no_relation"
    assert route(rejection(0.0, 0.0, 0.0)) is None
    assert route(rejection(0.0, 0.0, 0.0, teacher_id="gpt_oss")) is None
    # readable everywhere INCLUDING the placeholder floor (1,1,1): the floor cannot
    # discriminate the fact and re-authoring cannot change that -- excluded from repair/glean
    assert route(rejection(1.0, 1.0, 1.0)) == "floor_answerable"
    assert route(rejection(1.0, 1.0, 1.0, teacher_id="gpt_oss")) == "floor_answerable"
    # no reader scores (compile-time rejection) -> untouched
    assert route({"evidence": {}}) is None
    # hard finer-level-check rejection: lattice-owned, never repair-targeted -- routed on the
    # detail_reason alone (its supported-level scores are a passing 1,1,0 pattern)
    assert route({"detail_reason": "finer_level_unreadable",
                  **rejection(1.0, 1.0, 0.0)}) == "lattice_suspect"


def test_gleaning_targets_reader_routing_excludes_without_resurrecting():
    occurrences = {
        "cond": {"occurrence_id": "cond", "decision_id": "d-c", "start": 0, "end": 3},
        "test1": {"occurrence_id": "test1", "decision_id": "d-t", "start": 10, "end": 13},
    }
    document = "CHF; got bmp today."
    arguments = [{"role": "subject", "kind": "linked", "occurrence_id": "cond"},
                 {"role": "object", "kind": "linked", "occurrence_id": "test1"}]
    fact_key = qa_builder._relation_fact_key("tests_for", arguments, occurrences)
    opportunity = {"relation": "tests_for", "scope": "span_span", "fact_key": fact_key,
                   "arguments": arguments}

    def stage_rejection(scores):
        return {"relation": "tests_for", "detail_reason": "three_point_gate_failed",
                "evidence": {"arguments": arguments, "teacher_id": "deterministic",
                             "answer_competing": ["other test"],
                             "validation": {"scores": scores}}}

    # reader-verified no-relation: neither an ambiguous target NOR resurrected as missed
    nowhere = stage_rejection({"original": 0.0, "representative": 0.0, "placeholder": 0.0})
    assert qa_builder._gleaning_targets(document, [], [nowhere], [opportunity], occurrences) == []
    # lattice signature: excluded too (any authorship)
    disagree = stage_rejection({"original": 1.0, "representative": 0.0, "placeholder": 0.0})
    disagree["evidence"].pop("teacher_id")
    assert qa_builder._gleaning_targets(document, [], [disagree], [opportunity], occurrences) == []
    # control: the same competing-answer rejection WITHOUT an exclusion signature is a target
    ambiguous = stage_rejection({"original": 0.0, "representative": 0.0, "placeholder": 0.0})
    ambiguous["evidence"].pop("teacher_id")
    targets = qa_builder._gleaning_targets(document, [], [ambiguous], [opportunity], occurrences)
    assert [t["kind"] for t in targets] == ["ambiguous"]


def test_gleaning_targets_skip_facts_kept_by_any_row():
    occurrences = {
        "cond": {"occurrence_id": "cond", "decision_id": "d-c", "start": 0, "end": 3},
        "drug1": {"occurrence_id": "drug1", "decision_id": "d-1", "start": 10, "end": 15},
    }
    document = "CHF on lasix; ordered bmp and cxr."
    subject = {"role": "subject", "kind": "linked", "occurrence_id": "cond"}
    drug = {"role": "object", "kind": "linked", "occurrence_id": "drug1"}
    # fact kept via the deterministic REVERSE flip...
    kept_reverse = {"relation": "prescribed_with",
                    "evidence": {"arguments": [subject, drug]}}
    # ...while the teacher's FORWARD attempt was rejected as ambiguous
    forward_reject = {"relation": "prescribed_with", "detail_reason": "three_point_gate_failed",
                      "evidence": {"arguments": [subject, drug],
                                   "answer_competing": ["other drug"]}}
    # compound literal row (3 args -> no 2-arg fact key) covering two literal pairs
    kept_compound = {"relation": "tests_for", "evidence": {"arguments": [
        subject,
        {"role": "object", "kind": "context", "literal": "bmp", "start": 22, "end": 25},
        {"role": "object", "kind": "context", "literal": "cxr", "start": 30, "end": 33},
    ]}}

    def literal_opportunity(literal):
        return {"relation": "tests_for", "scope": "span_literal",
                "fact_key": ("tests_for", ("linked_decision", "d-c"), ("context_literal", literal)),
                "arguments": [subject,
                              {"role": "object", "kind": "context", "literal": literal}]}

    targets = qa_builder._gleaning_targets(
        document,
        [kept_reverse, kept_compound],
        [forward_reject],
        [literal_opportunity("bmp"), literal_opportunity("cxr"), literal_opportunity("troponin")],
        occurrences,
    )

    kinds = {(t["kind"], t["relation"]) for t in targets}
    # the kept-forward-rejected fact and the compound-covered literal pairs are NOT re-targeted;
    # only the genuinely uncovered opportunity remains
    assert kinds == {("missed", "tests_for")}
    [target] = targets
    assert target["fact_key"] == ("tests_for", ("linked_decision", "d-c"),
                                  ("context_literal", "troponin"))
    # without the kept rows, the ambiguous forward rejection IS a target again
    targets_unkept = qa_builder._gleaning_targets(
        document, [], [forward_reject],
        [literal_opportunity("troponin")], occurrences)
    assert {t["kind"] for t in targets_unkept} == {"ambiguous", "missed"}
    # a compound REJECTION also counts its pairs as proposed: they never resurrect as "missed"
    compound_reject = {"relation": "tests_for", "detail_reason": "three_point_gate_failed",
                       "evidence": {"arguments": kept_compound["evidence"]["arguments"]}}
    targets_rejected = qa_builder._gleaning_targets(
        document, [], [compound_reject],
        [literal_opportunity("bmp"), literal_opportunity("cxr")], occurrences)
    assert [t["kind"] for t in targets_rejected] == []


def _stage_fixture():
    occurrences = {
        "cond": {"occurrence_id": "cond", "decision_id": "d-c", "surface": "CHF",
                 "start": 0, "end": 3, "runtime_type": "health-condition", "controlled": True},
        "drug1": {"occurrence_id": "drug1", "decision_id": "d-1", "surface": "lasix",
                  "start": 10, "end": 15, "runtime_type": "drug", "controlled": True},
        "drug2": {"occurrence_id": "drug2", "decision_id": "d-2", "surface": "carvedilol",
                  "start": 20, "end": 30, "runtime_type": "drug", "controlled": True},
        "test1": {"occurrence_id": "test1", "decision_id": "d-t", "surface": "echocardiogram",
                  "start": 40, "end": 54, "runtime_type": "test", "controlled": True},
    }
    decisions = {
        "d-c": {"decision_id": "d-c", "actions": [
            {"mode": "level", "legal": True, "entails": ["congestive heart failure"]},
            {"mode": "level", "legal": True, "entails": ["heart disease"]},
            {"mode": "level", "legal": True, "entails": ["medical condition"]}]},
        "d-1": {"decision_id": "d-1", "actions": [
            {"mode": "level", "legal": True, "entails": ["loop diuretic"]},
            {"mode": "level", "legal": True, "entails": ["medication"]}]},
        "d-2": {"decision_id": "d-2", "actions": [
            {"mode": "level", "legal": True, "entails": ["beta blocker"]}]},
        "d-t": {"decision_id": "d-t", "actions": [
            {"mode": "level", "legal": True, "entails": ["cardiac imaging study"]}]},
    }

    def opportunity(relation, subject_id, object_id):
        return {"relation": relation, "scope": "span_span", "fact_key": (relation, subject_id, object_id),
                "arguments": [
                    {"role": "subject", "kind": "linked", "occurrence_id": subject_id},
                    {"role": "object", "kind": "linked", "occurrence_id": object_id}]}

    return occurrences, decisions, opportunity


def test_deterministic_relation_plans_singleton_forward_multi_reverse():
    occurrences, decisions, opportunity = _stage_fixture()
    opportunities = [
        opportunity("prescribed_with", "cond", "drug1"),   # 2 drug objects -> ambiguous
        opportunity("prescribed_with", "cond", "drug2"),
        opportunity("tests_for", "cond", "test1"),         # singleton -> forward + reverse fallback
        {"relation": "tests_for", "scope": "span_literal", "arguments": []},  # not a span plan
    ]

    plans = qa_builder._deterministic_relation_plans(opportunities, occurrences, decisions)

    by_relation = {}
    for plan in plans:
        by_relation.setdefault(plan["relation"], []).append(plan)
    prescribed = by_relation["prescribed_with"]
    assert [plan["directions"] for plan in prescribed if not plan.get("compound")] == [
        ["reverse"], ["reverse"]]
    # ambiguous group also gets ONE compound plan, ordered AFTER its per-object flips
    assert prescribed[-1].get("compound") is True
    assert len(prescribed[-1]["objects"]) == 2
    assert prescribed[-1]["fact_keys"] == [plan["fact_key"] for plan in prescribed[:-1]]
    # singleton groups never get a compound plan
    assert [plan["directions"] for plan in by_relation["tests_for"]] == [["forward", "reverse"]]
    tests_plan = by_relation["tests_for"][0]
    assert tests_plan["fact_key"] == qa_builder._relation_fact_key(
        "tests_for", [tests_plan["subject"], tests_plan["object"]], occurrences)
    # an uncontrolled argument disqualifies the pair entirely
    uncontrolled = dict(occurrences, test1={**occurrences["test1"], "controlled": False})
    assert qa_builder._deterministic_relation_plans(
        [opportunity("tests_for", "cond", "test1")], uncontrolled, decisions) == []


def test_deterministic_answer_levels_coarsest_first_with_skip():
    occurrences, decisions, _ = _stage_fixture()
    # coarsest -> finest, with the bare type-word level ("medical condition") skipped
    assert qa_builder._deterministic_answer_levels(decisions["d-c"]) == [
        "heart disease", "congestive heart failure"]
    # every level degenerate -> one trial at the finest level, never an empty search
    only_type_word = {"decision_id": "d-x", "actions": [
        {"mode": "level", "legal": True, "entails": ["medication"]}]}
    assert qa_builder._deterministic_answer_levels(only_type_word) == ["medication"]


def test_deterministic_stage_proposal_directions():
    occurrences, decisions, opportunity = _stage_fixture()
    [plan] = qa_builder._deterministic_relation_plans(
        [opportunity("tests_for", "cond", "test1")], occurrences, decisions)
    span_labels = {"d-c": "S1", "d-t": "S4"}

    forward = qa_builder._deterministic_stage_proposal(
        plan, "forward", "cardiac imaging study", occurrences, decisions, span_labels)
    assert forward["answer_role"] == "object"
    assert forward["accepted_answers"] == ["cardiac imaging study"]
    # locator = subject at its FINEST level; answer argument carries the trial level
    assert forward["question"] == (
        "Which test or investigation was ordered to evaluate the congestive heart failure?")
    assert [argument["span_label"] for argument in forward["arguments"]] == ["S1", "S4"]
    assert forward["arguments"][1]["support_property"] == "cardiac imaging study"

    reverse = qa_builder._deterministic_stage_proposal(
        plan, "reverse", "heart disease", occurrences, decisions, span_labels)
    assert reverse["answer_role"] == "subject"
    assert reverse["accepted_answers"] == ["heart disease"]
    assert reverse["question"] == "For what medical condition was the cardiac imaging study ordered?"
    assert reverse["arguments"][0]["support_property"] == "heart disease"


def test_compound_span_reverse_row_locators_and_requirements():
    occurrences, decisions, opportunity = _stage_fixture()
    plans = qa_builder._deterministic_relation_plans(
        [opportunity("prescribed_with", "cond", "drug1"),
         opportunity("prescribed_with", "cond", "drug2")], occurrences, decisions)
    compound = plans[-1]
    assert compound.get("compound") is True

    row = qa_builder._compound_span_reverse_row(
        "CHF x on lasix and carvedilol; echocardiogram done later today ok",
        "prescribed_with", compound["subject"], compound["objects"],
        occurrences, decisions, "heart disease")

    # every object at its FINEST level forms the locator; the subject answers at the trial level
    assert "loop diuretic" in row["question"] and "beta blocker" in row["question"]
    assert row["accepted_values"] == ["heart disease"]
    assert row["answer_target"] == {"kind": "linked_decision", "decision_id": "d-c",
                                    "required_property": "heart disease"}
    assert row["decision_requirements"] == {
        "d-c": "heart disease", "d-1": "loop diuretic", "d-2": "beta blocker"}
    assert row["occurrence_ids"] == ["cond", "drug1", "drug2"]
    assert (row["evidence"] or {})["run_id"] == "deterministic_stage"


def test_set_forward_row_members_and_requirements():
    occurrences, decisions, opportunity = _stage_fixture()
    plans = qa_builder._deterministic_relation_plans(
        [opportunity("prescribed_with", "cond", "drug1"),
         opportunity("prescribed_with", "cond", "drug2")], occurrences, decisions)
    compound = plans[-1]

    row = qa_builder._set_forward_row(
        "CHF x on lasix and carvedilol; echocardiogram done later today ok",
        "prescribed_with", compound["subject"], compound["objects"], occurrences, decisions)

    # subject locator at its finest level in the question; members = objects at THEIR finest
    assert "congestive heart failure" in row["question"]
    assert row["answer_target"] == {"kind": "linked_decision_set", "members": [
        {"decision_id": "d-1", "required_property": "loop diuretic"},
        {"decision_id": "d-2", "required_property": "beta blocker"}]}
    assert row["decision_requirements"] == {
        "d-c": "congestive heart failure", "d-1": "loop diuretic", "d-2": "beta blocker"}
    assert row["scoring_contract"] == {"kind": "semantic_qa", "match": "set_recall"}
    assert (row["evidence"] or {})["run_id"] == "deterministic_stage"


def test_context_answer_score_set_recall_one_to_one():
    row = {"answer_target": {"kind": "linked_decision_set", "members": [
        {"decision_id": "d-1", "required_property": "loop diuretic"},
        {"decision_id": "d-2", "required_property": "beta blocker"}]}}
    chains = {
        "d-1": [{"node": "loop diuretic", "answer_aliases": ["loop diuretic"],
                 "entailed_properties": ["loop diuretic", "medication"]}],
        "d-2": [{"node": "beta blocker", "answer_aliases": ["beta blocker"],
                 "entailed_properties": ["beta blocker", "medication"]}],
    }
    score = qa_builder._context_answer_score
    # full recall; the extra uncontrolled literal prediction is ignored (recall, not precision)
    assert score(row, '["loop diuretic", "beta blocker", "bmp"]', chains) == 1.0
    # one member missing -> partial recall (< threshold 1.0 -> gate-fails on that render)
    assert score(row, '["loop diuretic"]', chains) == 0.5
    # one prediction cannot satisfy two members (one-to-one matching)
    assert score(row, '["loop diuretic", "loop diuretic"]', chains) == 0.5
    # malformed / non-array reply -> zero
    assert score(row, "loop diuretic and beta blocker", chains) == 0.0
    assert score(row, "[]", chains) == 0.0


def test_validate_context_assertions_routes_set_rows_to_set_reader():
    row = {
        "family": "context", "question": "List EVERY distinct medication.",
        "answer_target": {"kind": "linked_decision_set", "members": [
            {"decision_id": "d-1", "required_property": "loop diuretic"}]},
        "evidence": {"reader_turns": [0]},
    }
    chains = {"d-1": [{"node": "loop diuretic", "answer_aliases": ["loop diuretic"],
                       "entailed_properties": ["loop diuretic"]}]}
    calls = {"span": 0, "set": 0}

    def span_reader(questions, context, clauses=None):
        calls["span"] += 1
        return ["never used for set rows"] * len(questions)

    def set_reader(questions, context, clauses=None):
        calls["set"] += 1
        # answerable on original+representative, hidden on placeholder
        return ['["loop diuretic"]' if "lasix" in context or "loop diuretic" in context
                else "[]"] * len(questions)

    accepted, evidence = qa_builder.validate_context_assertions(
        [row],
        original_context="on lasix daily",
        representative_context="on loop diuretic daily",
        placeholder_context="on [MEDICATION] daily",
        reader=span_reader,
        chain_by_decision=chains,
        set_reader=set_reader,
    )
    assert len(accepted) == 1
    assert calls == {"span": 0, "set": 3}
    [row_evidence] = evidence.values()
    assert row_evidence["scores"] == {"original": 1.0, "representative": 1.0, "placeholder": 0.0}


def test_literal_reverse_groups_widened_seed_carries_fact_key():
    occurrences = {
        "cond": {"occurrence_id": "cond", "decision_id": "d-c", "surface": "CHF", "start": 0,
                 "end": 3, "runtime_type": "health-condition", "controlled": True},
    }
    cue_pass = {"relation": "tests_for", "scope": "span_literal", "recovered_by_escalation": False,
                "fact_key": ("tests_for", ("linked_decision", "d-c"), ("context_literal", "bmp")),
                "arguments": [
                    {"role": "subject", "kind": "linked", "occurrence_id": "cond"},
                    {"role": "object", "kind": "context", "literal": "bmp", "start": 13, "end": 16}]}

    assert qa_builder._literal_reverse_groups([cue_pass], occurrences) == {}
    groups = qa_builder._literal_reverse_groups(
        [cue_pass], occurrences, judge_recovered_only=False)
    assert set(groups) == {("tests_for", "d-c")}
    [entry] = groups[("tests_for", "d-c")]["literals"].values()
    assert entry["literal"] == "bmp"
    assert entry["fact_key"] == cue_pass["fact_key"]


def test_llm_prefilter_context_candidates_locates_and_types_phrases():
    source = "Hypothyroidism was treated with levothyroxine; ordered a thyroid panel."
    env = {
        "occurrences": [
            {"occurrence_id": "c", "decision_id": "d", "surface": "Hypothyroidism",
             "start": 0, "end": 14, "runtime_type": "health-condition", "controlled": True},
        ],
        "decisions": [],
    }

    def propose(prompt):  # stub set-call: drug for prescribed_with, test for tests_for (fenced), else []
        if "medication or drug" in prompt:
            return '["levothyroxine"]'
        if "diagnostic test" in prompt:
            return '```json\n["thyroid panel"]\n```'
        return "[]"

    candidates = qa_builder.llm_prefilter_context_candidates(source, env, propose)

    typed = {(row["literal"], row["runtime_type"]) for row in candidates}
    assert ("levothyroxine", "drug") in typed
    assert ("thyroid panel", "test") in typed
    for row in candidates:  # every literal is a verbatim source span
        assert source[row["start"]:row["end"]].lower() == row["literal"].lower()
        assert row["provenance"] == "llm_prefilter"


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


def test_source_literal_spans_equates_unicode_dash_variants():
    # LLM teachers emit unicode dashes (gpt-oss: "x‑ray" U+2011) for source ASCII "x-ray";
    # all dash variants are equated so the literal still grounds.
    document = "we did an x-ray and a follow-up ct scan."
    assert [document[s:e] for s, e in
            qa_builder._source_literal_spans(document, "x‑ray")] == ["x-ray"]      # U+2011
    assert [document[s:e] for s, e in
            qa_builder._source_literal_spans(document, "follow–up")] == ["follow-up"]  # U+2013
    # symmetric: ASCII literal still grounds an ASCII source dash
    assert [document[s:e] for s, e in
            qa_builder._source_literal_spans(document, "x-ray")] == ["x-ray"]
    # a dash must NOT be equated with whitespace or nothing (no over-match)
    assert qa_builder._source_literal_spans("state of art care", "state-of-art") == []


def test_evidence_quote_grounding_equates_unicode_dashes():
    # teacher emits a unicode dash in the evidence quote for an ASCII-hyphen source; grounding
    # must still resolve, to the correct-length ORIGINAL-document span (dash fold is 1-char->1-char).
    doc = "[doctor] show me the right knee x-ray. no fracture."   # ASCII hyphen in source
    quote = "show me the right knee x‑ray"                        # U+2011 in the teacher quote
    span, err = qa_builder._resolve_relation_evidence_span(doc, quote, None)
    assert err is None and span is not None
    assert doc[span[0]:span[1]] == "show me the right knee x-ray"  # exact source substring
    assert span[1] - span[0] == len(quote)                        # length preserved
    # no-regression: case stays exact, and a genuinely-absent quote still fails
    assert qa_builder._exact_substring_starts(doc, "SHOW me") == []
    assert qa_builder._resolve_relation_evidence_span(doc, "mri scan", None) == (None, "invalid_evidence")


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


def test_gleaning_targets_excludes_kept_relations_from_ambiguous():
    # A kept relation already answered uniquely at the reader gate; its multi-answer coverage is
    # handled deterministically by reverse-framing, so it must NEVER be handed back as an ambiguous
    # gleaning target (which would re-author an already-kept fact).
    occ = {n: {"occurrence_id": n, "decision_id": f"d-{n}", "runtime_type": t,
               "controlled": True, "surface": n, "start": s, "end": s + 5}
           for n, t, s in [("reflux", "health-condition", 0), ("diet", "medical-procedure", 20)]}
    kept = {"relation": "procedure_for", "evidence": {
        "answer_competing": ["ultrasound"],
        "arguments": [
            {"role": "subject", "kind": "linked", "occurrence_id": "reflux", "support_property": "x"},
            {"role": "object", "kind": "linked", "occurrence_id": "diet", "support_property": "y"}]}}
    assert qa_builder._gleaning_targets("", [kept], [], [], occ) == []


def test_answer_leak_tokens_reports_overlap_minus_exempt():
    # the discriminative answer tokens present in the question are the leak; exemptions remove them
    assert qa_builder._answer_leak_tokens("was the aspirin prescribed?", "aspirin", "") == {"aspirin"}
    assert qa_builder._answer_leak_tokens(
        "was the aspirin prescribed?", "aspirin", "", extra_exempt_tokens=["aspirin"]) == set()
    assert qa_builder._answer_leak_tokens("what condition was treated?", "aspirin", "") == set()


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


def test_gleaning_targets_routes_hedge_flagged_reader_failure_to_repair():
    # The hedge guard is now a non-blocking diagnostic, but a hedge-flagged relation the READER
    # then could not confirm is routed back to repair to re-phrase conditionally (restores the
    # pre-demotion repair path, gated on actually failing the reader).
    hedged = {"relation": "procedure_for", "detail_reason": "three_point_gate_failed",
              "rejection_id": "sha256:hhh",
              "evidence": {"modality_diagnostics": ["hedged_source_definite_question"]}}
    # a plain reader failure WITHOUT the hedge flag (and not a mispaired literal) is NOT routed
    plain = {"relation": "prescribed_with", "detail_reason": "three_point_gate_failed",
             "rejection_id": "sha256:ppp", "evidence": {}}

    targets = qa_builder._gleaning_targets("", [], [hedged, plain], [], {})

    hedge_targets = [t for t in targets if t.get("reason") == "hedged_relation"]
    assert len(hedge_targets) == 1
    assert hedge_targets[0]["kind"] == "fixable"
    assert "conditional" in hedge_targets[0]["hint"].lower()
    assert all(t["relation"] != "prescribed_with" for t in targets)  # plain gate-fail not routed


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
    # targets+evidence are combined: the target's own source clause is inlined, no separate cards
    assert "EVIDENCE CARDS" not in prompt
    # grouped by region: the target's clause is shown once on a SOURCE line above its target
    assert 'SOURCE: "for your first problem , your reflux , continue dietary modifications' in prompt
    # the full-source copy is dropped as redundant; the per-target clause is the only source shown
    assert "SOURCE DOCUMENT" not in prompt
    # the target's spans are shown; the non-target insomnia/zolpidem block never appears
    assert inv["d-reflux"] in prompt and inv["d-diet"] in prompt
    assert "insomnia" not in prompt and "zolpidem" not in prompt
    # safety-critical privacy instruction survives
    assert "the QUESTION must never contain the accepted answer" in prompt


def test_relation_repair_prompt_writes_each_fix_hint_once():
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
    hint = qa_builder._GLEANING_MISSED_HINT
    def missed(subject_id, object_id, relation):
        return {"kind": "missed", "relation": relation, "hint": hint,
                "arguments": [
                    {"role": "subject", "kind": "linked", "occurrence_id": subject_id},
                    {"role": "object", "kind": "linked", "occurrence_id": object_id}]}
    fixable = {"kind": "fixable", "relation": "prescribed_with", "reason": "answer_leakage",
               "hint": qa_builder._GLEANING_FIX_HINTS["answer_leakage"],
               "arguments": [
                   {"role": "subject", "kind": "linked", "occurrence_id": "ins"},
                   {"role": "object", "kind": "linked", "occurrence_id": "zol"}]}

    prompt = qa_builder.relation_repair_prompt(
        "d", source, environment,
        [missed("reflux", "diet", "procedure_for"), missed("ins", "zol", "prescribed_with"),
         fixable])

    # the shared MISSED hint appears exactly ONCE (in the FIX GUIDE), not per target
    assert prompt.count(hint) == 1
    assert prompt.count(qa_builder._GLEANING_FIX_HINTS["answer_leakage"]) == 1
    assert "FIX GUIDE" in prompt
    # target lines reference the guide by tag instead of inlining the hint
    assert "MISSED" in prompt and "fix: answer_leakage" in prompt


# --- Relation-support escalation: no-regression invariant + accept-only cascade ---
from cloak.train.relation_support_gate import RelationSupportCascade, build_medgemma_judge


def _cueless_env():
    # condition + drug co-located with no relation cue verb -> cue gate misses the pair
    source = "Hypothyroidism and Synthroid."
    return source, _environment(source)


def test_relation_support_escalator_none_matches_cue_only():
    source, env = _cueless_env()
    base = qa_builder.relation_support_opportunities(source, env)
    assert qa_builder.relation_support_opportunities(source, env, escalator=None) == base


def test_causes_or_explains_cue_ok_is_judge_gated():
    # a cue-OK causal pair ("arthritis causes fever"): cue-only mode keeps it, but with an escalator
    # the judge filters it -- a rejecting judge drops the cue-ok pair, an accepting judge keeps it.
    source = "The arthritis causes fever in this patient."
    def sp(x): return source.index(x)
    env = {
        "occurrences": [
            {"occurrence_id": "art", "decision_id": "d-art", "surface": "arthritis",
             "start": sp("arthritis"), "end": sp("arthritis") + 9, "runtime_type": "health-condition"},
            {"occurrence_id": "fev", "decision_id": "d-fev", "surface": "fever",
             "start": sp("fever"), "end": sp("fever") + 5, "runtime_type": "health-condition"},
        ],
        "decisions": [
            {"decision_id": "d-art", "actions": [{"mode": "level", "legal": True, "entails": ["joint disease"]}]},
            {"decision_id": "d-fev", "actions": [{"mode": "level", "legal": True, "entails": ["febrile illness"]}]},
        ],
    }

    def causal(opps):
        return [o for o in opps if o["relation"] == "causes_or_explains"]

    # cue-only (no escalator): the cue-ok causal pair is kept
    assert causal(qa_builder.relation_support_opportunities(source, env))
    # escalator present: judge is the authority -> reject drops it, accept keeps it
    assert not causal(qa_builder.relation_support_opportunities(source, env, escalator=lambda **k: False))
    assert causal(qa_builder.relation_support_opportunities(source, env, escalator=lambda **k: True))


def test_causes_or_explains_proximity_cap_rejects_wide_anchor():
    # local (single clause) causal pair is eligible; a far-apart pair (>1 clause, wide anchor) is
    # dropped BEFORE cue/escalation -- a causal claim between whole-note-apart findings is
    # co-occurrence, not causation. Accept-all escalator so only the cap can remove it.
    def env_for(source, a_sp, b_sp):
        return {
            "occurrences": [
                {"occurrence_id": "a", "decision_id": "d-a", "surface": "arthritis",
                 "start": a_sp, "end": a_sp + 9, "runtime_type": "health-condition"},
                {"occurrence_id": "b", "decision_id": "d-b", "surface": "fever",
                 "start": b_sp, "end": b_sp + 5, "runtime_type": "health-condition"},
            ],
            "decisions": [
                {"decision_id": "d-a", "actions": [{"mode": "level", "legal": True, "entails": ["joint disease"]}]},
                {"decision_id": "d-b", "actions": [{"mode": "level", "legal": True, "entails": ["febrile illness"]}]},
            ],
        }

    def causal(opps):
        return [o for o in opps if o["relation"] == "causes_or_explains"]

    local = "The arthritis explains the fever here."
    env_local = env_for(local, local.index("arthritis"), local.index("fever"))
    assert causal(qa_builder.relation_support_opportunities(
        local, env_local, escalator=lambda **k: True))

    far = ("The arthritis was documented today. The patient rested well overnight. "
           "A separate note. Much later on the fever developed.")
    env_far = env_for(far, far.index("arthritis"), far.index("fever"))
    assert not causal(qa_builder.relation_support_opportunities(
        far, env_far, escalator=lambda **k: True))


def test_relation_support_escalation_is_additive_superset():
    source, env = _cueless_env()
    base = qa_builder.relation_support_opportunities(source, env)
    base_keys = {tuple(o["fact_key"]) for o in base}

    accept_all = qa_builder.relation_support_opportunities(source, env, escalator=lambda **k: True)
    reject_all = qa_builder.relation_support_opportunities(source, env, escalator=lambda **k: False)
    acc_keys = {tuple(o["fact_key"]) for o in accept_all}

    # NO-REGRESSION: every cue-only opportunity survives escalation.
    assert base_keys <= acc_keys
    # A rejecting escalator is byte-identical to the cue-only set (escalator only recovers).
    assert {tuple(o["fact_key"]) for o in reject_all} == base_keys
    # The co-located cue-less pair is genuinely recovered (strict addition), and flagged.
    assert len(acc_keys) > len(base_keys)
    recovered = [o for o in accept_all if o["recovered_by_escalation"]]
    assert recovered
    assert all(not o["recovered_by_escalation"]
               for o in accept_all if tuple(o["fact_key"]) in base_keys)


def _pair(subject="afib", object_="digoxin"):
    return [{"role": "subject", "kind": "linked", "surface": subject},
            {"role": "object", "kind": "linked", "surface": object_}]


def test_cascade_mednli_accepts_and_short_circuits_judge():
    judge_calls = []

    def judge(**kwargs):
        judge_calls.append(kwargs)
        return False  # would reject; must not be consulted

    casc = RelationSupportCascade(judge, mednli_entail=lambda p, h: 0.99, mednli_threshold=0.98)
    assert casc(relation="prescribed_with", quote="q", arguments=_pair()) is True
    assert judge_calls == []  # accept-only tier short-circuited the LLM


def test_cascade_defers_to_judge_when_mednli_below_threshold():
    casc = RelationSupportCascade(judge=lambda **k: True,
                                  mednli_entail=lambda p, h: 0.10, mednli_threshold=0.98)
    assert casc(relation="prescribed_with", quote="q", arguments=_pair()) is True
    casc_rej = RelationSupportCascade(judge=lambda **k: False,
                                      mednli_entail=lambda p, h: 0.10, mednli_threshold=0.98)
    assert casc_rej(relation="prescribed_with", quote="q", arguments=_pair()) is False


def test_cascade_defers_without_regressing_on_no_pair_or_unknown_relation():
    casc = RelationSupportCascade(judge=lambda **k: True)
    # missing object -> no judgeable pair -> defer (no recovery, no regression)
    assert casc(relation="prescribed_with", quote="q",
                arguments=[{"role": "subject", "surface": "x"}]) is False
    # relation with no claim template -> defer
    assert casc(relation="not_a_relation", quote="q", arguments=_pair()) is False


class _StubClient:
    def __init__(self, reply):
        self.reply = reply

    def generate(self, prompt, system=None):
        return self.reply


def test_medgemma_judge_parses_and_is_accept_biased_on_error():
    assert build_medgemma_judge(_StubClient('{"asserted": true, "why": "x"}'))(premise="p", claim="c") is True
    assert build_medgemma_judge(_StubClient('{"asserted": false, "why": "x"}'))(premise="p", claim="c") is False
    # recall-first default: unparseable reply accepts rather than dropping a possibly-true relation
    assert build_medgemma_judge(_StubClient("garbage"))(premise="p", claim="c") is True
    assert build_medgemma_judge(_StubClient("garbage"), accept_on_error=False)(premise="p", claim="c") is False
    # causes_or_explains flips to REJECT-on-error (combinatorial; can't afford accept-on-glitch),
    # and contraindicated_because_of too (adversative: co-occurrence usually asserts the OPPOSITE
    # relation), while other relations keep the recall-first accept-on-error default.
    j = build_medgemma_judge(_StubClient("garbage"))
    assert j(premise="p", claim="c", relation="causes_or_explains") is False
    assert j(premise="p", claim="c", relation="contraindicated_because_of") is False
    assert j(premise="p", claim="c", relation="prescribed_with") is True


def test_contraindication_judge_rule_requires_quoted_avoidance_cue():
    from cloak.train.relation_support_gate import _judge_system
    contra = _judge_system("contraindicated_because_of")
    # extractive-quote grounding (same defense as the causal rule): true requires quoting the
    # exact avoidance phrase in a "cue" field, else false
    assert 'quote, in a "cue" field' in contra
    assert "cannot quote" in contra
    # the inversion trap is a worked example: a drug titrated FOR a condition is not contraindicated
    assert "increase the lamotrigine" in contra
    assert '"asserted": false, "cue": ""' in contra


def test_judge_system_is_per_relation_isolated():
    from cloak.train.relation_support_gate import _judge_system
    causal = _judge_system("causes_or_explains")
    contra = _judge_system("contraindicated_because_of")
    prescribed = _judge_system("prescribed_with")

    # each relation's own rule appears ONLY in its own prompt (no cross-relation contamination)
    assert "CAUSAL claim" in causal and "CONTRAINDICATION" not in causal
    assert "CONTRAINDICATION" in contra and "CAUSAL claim" not in contra
    # prescribed_with needs no extra rule beyond the general grounding ones
    assert "CAUSAL claim" not in prescribed and "CONTRAINDICATION" not in prescribed
    # only the target relation's worked examples are present (off-corpus entities)
    assert "due to your pneumonia" in causal  # causal example
    assert "due to your pneumonia" not in prescribed
    assert "start you on sumatriptan" in prescribed  # prescribed_with example
    assert "start you on sumatriptan" not in causal
    # shared grounding + answer format are always present
    for prompt in (causal, contra, prescribed):
        assert "careful clinical NLP annotator" in prompt
        assert '"asserted": true|false' in prompt


def test_informative_context_judge_parses_and_is_accept_biased_on_error():
    from cloak.train.relation_support_gate import build_informative_context_judge

    assert build_informative_context_judge(
        _StubClient('{"informative": true, "why": "x"}'))(locator="l", category="drug") is True
    assert build_informative_context_judge(
        _StubClient('{"informative": false, "why": "x"}'))(locator="l", category="drug") is False
    assert build_informative_context_judge(_StubClient("garbage"))(locator="l", category="drug") is True
    assert build_informative_context_judge(
        _StubClient("garbage"), accept_on_error=False)(locator="l", category="drug") is False


class _BatchEscalator:
    """Escalator exposing judge_batch -> the miner must use the batch path."""
    def __init__(self, verdict):
        self.verdict = verdict
        self.batches = []

    def judge_batch(self, calls):
        self.batches.append(list(calls))
        return [self.verdict] * len(calls)


def test_miner_uses_batch_escalator_and_preserves_invariant():
    source, env = _cueless_env()
    base_keys = {tuple(o["fact_key"]) for o in qa_builder.relation_support_opportunities(source, env)}

    reject = _BatchEscalator(False)
    reject_out = qa_builder.relation_support_opportunities(source, env, escalator=reject)
    # batch path was taken (one judge_batch call), and rejecting batch == cue-only
    assert reject.batches and {tuple(o["fact_key"]) for o in reject_out} == base_keys

    accept = _BatchEscalator(True)
    accept_out = qa_builder.relation_support_opportunities(source, env, escalator=accept)
    acc_keys = {tuple(o["fact_key"]) for o in accept_out}
    assert base_keys <= acc_keys and len(acc_keys) > len(base_keys)
    # every call payload carries the fields a real judge needs
    assert accept.batches
    for call in accept.batches[0]:
        assert {"relation", "quote", "arguments", "anchor_kind", "document"} <= set(call)


def test_cascade_judge_batch_is_medgemma_only_and_concurrent_safe():
    seen = []

    def judge(*, premise, claim, relation=None):
        seen.append(claim)
        return "prescribed" in claim  # accept only the prescribed_with claim

    # mednli present but MUST be ignored by judge_batch (MedGemma-only path)
    casc = RelationSupportCascade(judge, mednli_entail=lambda p, h: 1.0, mednli_threshold=0.0)
    calls = [
        {"relation": "prescribed_with", "quote": "q", "arguments": _pair("afib", "digoxin")},
        {"relation": "tests_for", "quote": "q", "arguments": _pair("afib", "echocardiogram")},
        {"relation": "prescribed_with", "quote": "q", "arguments": [{"role": "subject", "surface": "x"}]},
    ]
    out = casc.judge_batch(calls)
    assert out == [True, False, False]  # 3rd has no object -> False; mednli's 1.0 never short-circuits
    assert len(seen) == 2  # judge consulted for the two well-formed pairs, not the no-object one


def test_claim_directionality_places_drug_test_first():
    from cloak.train.relation_support_gate import _claim
    # miner passes subject=condition, object=drug/test/procedure for these relations
    assert _claim("prescribed_with", "afib", "digoxin") == "digoxin is a medication prescribed to treat afib"
    assert _claim("tests_for", "chronic back pain", "x-ray") == "x-ray is a test or scan done to investigate chronic back pain"
    assert _claim("procedure_for", "back pain", "physical therapy") == "physical therapy is a procedure or treatment given for back pain"
    # subject-first relations are unchanged
    assert _claim("contraindicated_because_of", "lasix", "edema") == "the patient must NOT be given lasix because they have edema"
    assert _claim("causes_or_explains", "edema", "mitral regurgitation") == "edema causes or explains mitral regurgitation"


def test_linked_answer_score_contiguity_and_verbosity_cap():
    from cloak.train.qa_builder import _linked_answer_score, _resolve_semantic_node
    chain = [
        {"answer_aliases": ["kidney stones"], "entailed_properties": ["urolithiasis", "kidney disease"]},
        {"answer_aliases": ["urolithiasis"], "entailed_properties": ["urolithiasis"]},
    ]
    # singular answer against plural alias resolves + credits (inflection folds both sides)
    assert _resolve_semantic_node(chain, "kidney stone") is not None
    assert _linked_answer_score("kidney stone", chain, "urolithiasis") == 1.0
    # determiners/stopwords are ignored: "the kidney stones" is still an exact span
    assert _linked_answer_score("the kidney stones", chain, "urolithiasis") == 1.0
    assert _linked_answer_score("urolithiasis", chain, "urolithiasis") == 1.0
    # legitimate verbosity within the cap resolves: hedge prefix, qualifier tail, compound span
    # (the 2026-07-21 scorer A/B measured these as 32/37 of strict-equality's false negatives)
    assert _linked_answer_score("possible kidney stone", chain, "urolithiasis") == 1.0
    assert _linked_answer_score("recurrent kidney stones on the left", chain, "urolithiasis") == 1.0
    # NON-CONTIGUOUS alias tokens never resolve (scattered-token containment was unearned)
    assert _linked_answer_score("kidney pain from stones", chain, "urolithiasis") == 0.0
    # whole-sentence echo over the verbosity cap never resolves
    assert _linked_answer_score(
        "the doctor will order an ultrasound scan to check whether kidney stones explain it",
        chain, "urolithiasis") == 0.0
    chain2 = [{"answer_aliases": ["medication"], "entailed_properties": ["drug"]}]
    assert _linked_answer_score("medications", chain2, "drug") == 1.0
    assert _linked_answer_score("some medications", chain2, "drug") == 1.0
    # '-ss' words are not truncated (no false fold), unrelated terms don't resolve
    chain3 = [{"answer_aliases": ["abscess"], "entailed_properties": ["lesion"]}]
    assert _linked_answer_score("abscess", chain3, "lesion") == 1.0
    assert _linked_answer_score("headache", chain3, "lesion") == 0.0


def test_all_occurrence_judge_premise_elides_middle_not_contiguous():
    """All-occurrence premise = the occurrence-clauses joined (2\\n13\\n14), NOT the span 2..14."""
    from cloak.train.qa_builder import _all_occurrence_judge_premise, _source_clause_spans
    doc = (
        "[doctor] the patient has gout in the past history .\n"
        "[patient] okay yes .\n"
        "[doctor] talking about the gout again now .\n"
        "[doctor] and we started allopurinol today ."
    )
    clauses = _source_clause_spans(doc)
    assert len(clauses) == 4  # one clause per speaker turn
    gout1 = doc.index("gout")
    gout2 = doc.index("gout", gout1 + 1)
    allo = doc.index("allopurinol")
    occurrences = {
        "o1": {"occurrence_id": "o1", "decision_id": "d_gout", "start": gout1, "end": gout1 + 4},
        "o2": {"occurrence_id": "o2", "decision_id": "d_gout", "start": gout2, "end": gout2 + 4},
        "o3": {"occurrence_id": "o3", "decision_id": "d_allo", "start": allo, "end": allo + 11},
    }
    arguments = [
        {"role": "subject", "kind": "linked", "occurrence_id": "o1"},
        {"role": "object", "kind": "linked", "occurrence_id": "o3"},
    ]
    premise = _all_occurrence_judge_premise(doc, arguments, occurrences, clauses)
    # includes BOTH gout mentions (turns 0 and 2) and allopurinol (turn 3)
    assert "past history" in premise and "gout again now" in premise and "allopurinol" in premise
    # elides the middle turn that mentions neither entity (turn 1)
    assert "okay yes" not in premise
    # it is the joined clauses, not the contiguous 2..14 envelope
    assert premise == "\n".join(
        doc[l:r].strip() for l, r in (clauses[0], clauses[2], clauses[3])
    )


def test_escalation_prefilter_coordination_and_negation():
    from cloak.train.qa_builder import _escalation_prefilter_reason
    # coordination: three conditions in a flat list -> siblings, not a relation
    doc = "past medical history significant for gout , depression and hypertension ."
    g, d, h = doc.index("gout"), doc.index("depression"), doc.index("hypertension")
    ctx = {
        "c1": {"start": g, "end": g + 4}, "c2": {"start": d, "end": d + 10},
        "c3": {"start": h, "end": h + 12},
    }
    coord_args = [  # two CONDITIONS in a flat list -> same-class siblings
        {"role": "subject", "kind": "context", "literal": "gout",
         "runtime_type": "health-condition", "start": g, "end": g + 4},
        {"role": "object", "kind": "context", "literal": "hypertension",
         "runtime_type": "health-condition", "start": h, "end": h + 12},
    ]
    assert _escalation_prefilter_reason(doc, coord_args, {}, ctx) == "coordination_sibling"

    # cross-type "and" (condition + drug) is NOT coordination -- can be a real prescribed_with
    cross_args = [
        {"role": "subject", "kind": "context", "literal": "gout",
         "runtime_type": "health-condition", "start": g, "end": g + 4},
        {"role": "object", "kind": "context", "literal": "hypertension",
         "runtime_type": "drug", "start": h, "end": h + 12},
    ]
    assert _escalation_prefilter_reason(doc, cross_args, {}, ctx) is None

    # "with" is NOT a coordinator for a general pair: a qualifier/complication can be a real relation
    doc2 = "chronic heart failure with diastolic dysfunction was noted ."
    hf, dd = doc2.index("heart failure"), doc2.index("diastolic dysfunction")
    with_args = [
        {"role": "subject", "kind": "context", "literal": "heart failure", "start": hf, "end": hf + 13},
        {"role": "object", "kind": "context", "literal": "diastolic dysfunction",
         "start": dd, "end": dd + 21},
    ]
    assert _escalation_prefilter_reason(doc2, with_args, {}, {}) is None
    # ...but for causes_or_explains the "with" comorbidity connector is co-occurrence, not causation
    assert _escalation_prefilter_reason(
        doc2, with_args, {}, {}, "causes_or_explains") == "cooccurrence_connector"

    # causes_or_explains coordination WITH an intervening quantifier ("and some") -- the gap the
    # plain and/or coordination regex missed, letting the co-occurrence pair reach the judge
    doc5 = "there is some edema and some erythema of the right knee ."
    ed, er = doc5.index("edema"), doc5.index("erythema")
    cooc_args = [
        {"role": "subject", "kind": "context", "literal": "edema", "start": ed, "end": ed + 5},
        {"role": "object", "kind": "context", "literal": "erythema", "start": er, "end": er + 8},
    ]
    assert _escalation_prefilter_reason(
        doc5, cooc_args, {}, {}, "causes_or_explains") == "cooccurrence_connector"
    # a genuine causal verb in the gap survives even under causes_or_explains (not pure co-occurrence)
    doc6 = "the leg swelling is caused by her heart failure ."
    sw, hf2 = doc6.index("leg swelling"), doc6.index("heart failure")
    causal_args = [
        {"role": "subject", "kind": "context", "literal": "heart failure", "start": hf2, "end": hf2 + 13},
        {"role": "object", "kind": "context", "literal": "leg swelling", "start": sw, "end": sw + 12},
    ]
    assert _escalation_prefilter_reason(
        doc6, causal_args, {}, {}, "causes_or_explains") is None

    # negation: an argument explicitly asserted absent
    doc3 = "the workup shows no evidence of coronary artery disease today ."
    cad = doc3.index("coronary artery disease")
    neg_args = [
        {"role": "subject", "kind": "context", "literal": "coronary artery disease",
         "start": cad, "end": cad + 23},
        {"role": "object", "kind": "context", "literal": "workup",
         "start": doc3.index("workup"), "end": doc3.index("workup") + 6},
    ]
    assert _escalation_prefilter_reason(doc3, neg_args, {}, {}) == "negation_scope"

    # genuine relation with a predicate between the arguments -> not junk
    doc4 = "the gout was treated with allopurinol daily ."
    gg, aa = doc4.index("gout"), doc4.index("allopurinol")
    real_args = [
        {"role": "subject", "kind": "context", "literal": "gout", "start": gg, "end": gg + 4},
        {"role": "object", "kind": "context", "literal": "allopurinol", "start": aa, "end": aa + 11},
    ]
    assert _escalation_prefilter_reason(doc4, real_args, {}, {}) is None


def test_relation_repair_prompt_shares_clause_pool_across_targets():
    """A clause referenced by multiple targets is inlined ONCE in SOURCE CLAUSES and cited by label;
    the target header shows label + surface (e.g. 'S1 (gout)')."""
    source = ("[doctor] you have gout so we started allopurinol and colchicine .\n"
              "[patient] okay , thanks .")
    def sp(x): return source.index(x)
    environment = {
        "occurrences": [
            {"occurrence_id": "g", "decision_id": "d-g", "surface": "gout",
             "start": sp("gout"), "end": sp("gout") + 4, "runtime_type": "health-condition"},
            {"occurrence_id": "a", "decision_id": "d-a", "surface": "allopurinol",
             "start": sp("allopurinol"), "end": sp("allopurinol") + 11, "runtime_type": "drug"},
            {"occurrence_id": "c", "decision_id": "d-c", "surface": "colchicine",
             "start": sp("colchicine"), "end": sp("colchicine") + 10, "runtime_type": "drug"},
        ],
        "decisions": [
            {"decision_id": "d-g", "actions": [{"mode": "level", "legal": True, "entails": ["arthritis"]}]},
            {"decision_id": "d-a", "actions": [{"mode": "level", "legal": True, "entails": ["urate lowering drug"]}]},
            {"decision_id": "d-c", "actions": [{"mode": "level", "legal": True, "entails": ["anti-inflammatory"]}]},
        ],
    }
    t1 = {"kind": "missed", "relation": "prescribed_with", "hint": "emit if supported",
          "arguments": [{"role": "subject", "kind": "linked", "occurrence_id": "g"},
                        {"role": "object", "kind": "linked", "occurrence_id": "a"}]}
    t2 = {"kind": "missed", "relation": "prescribed_with", "hint": "emit if supported",
          "arguments": [{"role": "subject", "kind": "linked", "occurrence_id": "g"},
                        {"role": "object", "kind": "linked", "occurrence_id": "c"}]}
    prompt = qa_builder.relation_repair_prompt("d", source, environment, [t1, t2])
    clause_text = "you have gout so we started allopurinol and colchicine"
    # both targets share the clause -> one region -> the clause is inlined exactly ONCE, not per target
    assert prompt.count(clause_text) == 1
    assert prompt.count("SOURCE:") == 1          # a single region groups both targets
    assert "[1]" in prompt and "[2]" in prompt   # both targets listed under it
    # header carries label + surface for matching
    inv = {str(r["decision_id"]): r["span_label"] for r in qa_builder.relation_teacher_span_inventory(environment)}
    assert f"{inv['d-g']} (gout)" in prompt


def _mini_repair_env():
    source = "for the reflux continue omeprazole ."
    def sp(x): return source.index(x)
    env = {
        "occurrences": [
            {"occurrence_id": "r", "decision_id": "d-r", "surface": "reflux",
             "start": sp("reflux"), "end": sp("reflux") + 6, "runtime_type": "health-condition"},
            {"occurrence_id": "o", "decision_id": "d-o", "surface": "omeprazole",
             "start": sp("omeprazole"), "end": sp("omeprazole") + 10, "runtime_type": "drug"},
        ],
        "decisions": [
            {"decision_id": "d-r", "actions": [{"mode": "level", "legal": True, "entails": ["gastrointestinal condition"]}]},
            {"decision_id": "d-o", "actions": [{"mode": "level", "legal": True, "entails": ["ppi"]}]},
        ],
    }
    return source, env


def _span_label_enum(fmt):
    return (fmt["json_schema"]["schema"]["properties"]["span_relations"]["items"]
            ["properties"]["arguments"]["anyOf"][0]["prefixItems"][0]["properties"]["span_label"]["enum"])


def test_repair_response_schema_scoped_to_shown_labels():
    source, env = _mini_repair_env()
    inv = {str(r["decision_id"]): r["span_label"] for r in qa_builder.relation_teacher_span_inventory(env)}
    full = qa_builder.relation_teacher_response_format(env, source)
    scoped = qa_builder.relation_teacher_response_format(env, source, allowed_labels={inv["d-r"]})
    # full schema: both labels emittable + accounting ledger covers both
    assert set(_span_label_enum(full)) == {inv["d-r"], inv["d-o"]}
    assert full["json_schema"]["schema"]["properties"]["candidate_accounting"]["minItems"] == 2
    # scoped schema: only the shown label; ledger covers exactly it
    assert _span_label_enum(scoped) == [inv["d-r"]]
    ledger = scoped["json_schema"]["schema"]["properties"]["candidate_accounting"]
    assert ledger["minItems"] == 1 and ledger["maxItems"] == 1


def test_relation_repair_prompt_exposes_shown_labels():
    source, env = _mini_repair_env()
    target = {"kind": "missed", "relation": "prescribed_with", "hint": "emit if supported",
              "arguments": [{"role": "subject", "kind": "linked", "occurrence_id": "r"},
                            {"role": "object", "kind": "linked", "occurrence_id": "o"}]}
    out: set = set()
    prompt = qa_builder.relation_repair_prompt("d", source, env, [target], shown_labels_out=out)
    inv_labels = {r["span_label"] for r in qa_builder.relation_teacher_span_inventory(env)}
    assert out and out <= inv_labels          # populated, subset of inventory
    # every exposed label appears in the rendered DETECTED SPANS
    for lab in out:
        assert f"[{lab}:" in prompt
