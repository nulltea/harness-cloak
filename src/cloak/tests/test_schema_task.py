import importlib

import pytest


def _schema_task():
    try:
        return importlib.import_module("cloak.train.schema_task")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing schema task module: {exc}")


def _parse_sections(text):
    return _schema_task().parse_sections(text)


def _schema_field_score(out_final_text, out_hi_text, acceptance_sets=None):
    return _schema_task().schema_field_score(out_final_text, out_hi_text, acceptance_sets)


def test_schema_prompt_constants_are_additive_templates():
    tasks = importlib.import_module("cloak.tasks")
    assert hasattr(tasks, "SCHEMA_NOTE")
    assert hasattr(tasks, "SCHEMA_CASE")
    assert hasattr(tasks, "SCHEMA_TEMPLATE")
    assert hasattr(tasks, "SCHEMA_CORPORA")

    SCHEMA_NOTE = tasks.SCHEMA_NOTE
    SCHEMA_CASE = tasks.SCHEMA_CASE
    SCHEMA_TEMPLATE = tasks.SCHEMA_TEMPLATE
    SCHEMA_CORPORA = tasks.SCHEMA_CORPORA

    assert "CHIEF COMPLAINT: one line." in SCHEMA_NOTE
    assert 'formatted "problem — category — status"' in SCHEMA_NOTE
    assert 'write "none"' in SCHEMA_NOTE
    assert "PARTIES: one line." in SCHEMA_CASE
    assert 'formatted "claim — category — status"' in SCHEMA_CASE
    assert SCHEMA_TEMPLATE == {
        "aci": SCHEMA_NOTE,
        "mts": SCHEMA_NOTE,
        "clinical": SCHEMA_NOTE,
        "lexsum": SCHEMA_CASE,
    }
    assert SCHEMA_CORPORA == frozenset({"aci", "mts", "clinical", "lexsum"})


def test_parse_sections_extracts_well_formed_clinical_rows():
    parsed = _parse_sections(
        """
        CHIEF COMPLAINT: shortness of breath
        HISTORY OF PRESENT ILLNESS:
        The patient reports worsening edema.
        ASSESSMENT:
        congestive heart failure — cardiovascular condition — worsening
        hypertension - cardiovascular condition - stable
        PLAN:
        congestive heart failure – start Lasix – cardiology follow-up
        hypertension - continue lisinopril - primary care follow-up
        """
    )

    assert parsed["chief_complaint"] == "shortness of breath"
    assert parsed["history_of_present_illness"] == "The patient reports worsening edema."
    assert parsed["assessment"] == [
        {
            "problem": "congestive heart failure",
            "category": "cardiovascular condition",
            "status": "worsening",
        },
        {
            "problem": "hypertension",
            "category": "cardiovascular condition",
            "status": "stable",
        },
    ]
    assert parsed["plan"] == [
        {
            "problem": "congestive heart failure",
            "action": "start Lasix",
            "follow_up": "cardiology follow-up",
        },
        {
            "problem": "hypertension",
            "action": "continue lisinopril",
            "follow_up": "primary care follow-up",
        },
    ]


def test_parse_sections_keeps_embedded_dash_inside_field_value():
    parsed = _parse_sections(
        """
        ASSESSMENT:
        post–COVID syndrome — respiratory — stable
        """
    )

    assert parsed["assessment"] == [
        {
            "problem": "post–COVID syndrome",
            "category": "respiratory",
            "status": "stable",
        }
    ]


def test_parse_sections_missing_and_none_sections_are_empty():
    parsed = _parse_sections(
        """
        CHIEF COMPLAINT:
        none
        HISTORY OF PRESENT ILLNESS: none
        ASSESSMENT: none
        """
    )

    assert parsed["chief_complaint"] == ""
    assert parsed["history_of_present_illness"] == ""
    assert parsed["assessment"] == []
    assert parsed["plan"] == []


def test_parse_sections_tolerates_malformed_rows_without_raising():
    parsed = _parse_sections(
        """
        assessment
        diabetes -- endocrine -- uncontrolled -- extra
        malformed row without separators
        asthma - respiratory
        PLAN:
        diabetes - adjust insulin
        asthma - refill inhaler - pulmonology
        """
    )

    assert parsed["assessment"] == [
        {"problem": "diabetes", "category": "endocrine", "status": "uncontrolled"}
    ]
    assert parsed["plan"] == [
        {"problem": "asthma", "action": "refill inhaler", "follow_up": "pulmonology"}
    ]


def test_schema_field_score_aligns_rows_and_penalizes_missing_gold_problems():
    ceiling = """
    ASSESSMENT:
    congestive heart failure — cardiovascular condition — worsening
    hypertension — cardiovascular condition — stable
    PLAN:
    congestive heart failure — start Lasix — cardiology
    hypertension — continue lisinopril — primary care
    """
    out_final = """
    PLAN:
    hypertension — continue lisinopril — primary care
    congestive heart failure — start Lasix — cardiology
    diabetes — adjust insulin — endocrinology
    ASSESSMENT:
    hypertension — cardiovascular condition — stable
    congestive heart failure — cardiovascular condition — worsening
    diabetes — endocrine condition — uncontrolled
    """

    assert _schema_field_score(out_final, ceiling) == pytest.approx(1.0)

    missing = """
    ASSESSMENT:
    congestive heart failure — cardiovascular condition — worsening
    PLAN:
    congestive heart failure — start Lasix — cardiology
    """

    assert _schema_field_score(missing, ceiling) == pytest.approx(0.5)


def test_schema_field_score_counts_duplicate_problem_rows_positionally():
    ceiling = """
    ASSESSMENT:
    asthma — respiratory condition — stable
    asthma — respiratory condition — stable
    PLAN:
    asthma — continue inhaler — primary care
    asthma — continue inhaler — primary care
    """
    out_final = """
    ASSESSMENT:
    asthma — respiratory condition — stable
    PLAN:
    asthma — continue inhaler — primary care
    """

    assert _schema_field_score(out_final, ceiling) == pytest.approx(0.5)


def test_schema_field_score_uses_category_acceptance_sets():
    ceiling = """
    ASSESSMENT:
    hypothyroidism — endocrine condition — stable
    PLAN:
    hypothyroidism — continue levothyroxine — primary care
    """
    out_final = """
    ASSESSMENT:
    hypothyroidism — hypothyroidism — stable
    PLAN:
    hypothyroidism — continue levothyroxine — primary care
    """

    without_acceptance = _schema_field_score(out_final, ceiling)
    with_acceptance = _schema_field_score(
        out_final,
        ceiling,
        acceptance_sets={"hypothyroidism": ["hypothyroidism", "endocrine condition"]},
    )

    assert without_acceptance < 1.0
    assert with_acceptance == pytest.approx(1.0)


def test_schema_field_score_scores_lexsum_case_rows():
    ceiling = """
    CLAIMS:
    breach of contract — contract — proven
    negligence — tort — dismissed
    OUTCOME:
    breach of contract — damages — plaintiff wins
    negligence — no remedy — defendant wins
    """
    out_final = """
    OUTCOME:
    negligence — no remedy — defendant wins
    breach of contract — damages — plaintiff wins
    CLAIMS:
    negligence — tort — dismissed
    breach of contract — contract — proven
    """

    assert _schema_field_score(out_final, ceiling) == pytest.approx(1.0)

    accepted_category = """
    CLAIMS:
    breach of contract — contract claim — proven
    negligence — tort — dismissed
    OUTCOME:
    breach of contract — damages — plaintiff wins
    negligence — no remedy — defendant wins
    """

    assert _schema_field_score(
        accepted_category,
        ceiling,
        acceptance_sets={"breach of contract": ["contract claim"]},
    ) == pytest.approx(1.0)

    missing = """
    CLAIMS:
    breach of contract — contract — proven
    OUTCOME:
    breach of contract — damages — plaintiff wins
    """

    assert _schema_field_score(missing, ceiling) == pytest.approx(0.5)


def test_schema_field_score_returns_none_when_ceiling_has_no_rows():
    assert _schema_field_score("ASSESSMENT: asthma - respiratory - stable", "PLAN: none") is None
