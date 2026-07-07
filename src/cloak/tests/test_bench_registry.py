from bench.registry import (
    SUITES,
    available_suites,
    gold_spans_from_text,
    load_items,
)


def test_primary_utility_suite_round_robins_core_task_domains():
    items = load_items("primary_utility", limit=6, seed=7)

    domains = {item.domain for item in items}

    assert {"clinical", "legal", "biography"} <= domains
    assert all(item.reference_outputs for item in items)
    assert all(item.doc_orig for item in items)


def test_detector_and_privacy_datasets_are_not_primary_utility():
    assert SUITES["primary_utility"] == ["clinical", "lexsum", "wikibio"]
    assert "pii-bench" in SUITES["detector_coverage"]
    assert "synthetic-financial-pii" in SUITES["detector_coverage"]
    assert "rat-bench" in SUITES["privacy_stress"]


def test_email_controls_are_separate_from_primary():
    primary = {item.corpus for item in load_items("primary_utility", limit=12, seed=0)}
    controls = {item.corpus for item in load_items("email_controls", limit=4, seed=0)}

    assert primary <= {"clinical", "lexsum", "wikibio"}
    assert controls <= {"aeslc", "enron"}
    assert not primary & controls


def test_gold_spans_from_text_marks_reference_restated_strings():
    doc = "Martha Collins is 50 years old and lives in Oslo."
    refs = ["Martha Collins is a 50-year-old patient."]

    spans = gold_spans_from_text(doc, refs)
    surfaces = {span.surface for span in spans}

    assert "Martha Collins" in surfaces
    assert "50 years old" in surfaces
    assert any(span.task_relevance == "gold_restated" for span in spans)


def test_available_suites_reports_missing_optional_corpora():
    suites = available_suites()

    assert suites["primary_utility"]["available"]
    assert "clinical" in suites["primary_utility"]["available"]
    assert "detector_coverage" in suites
    assert "missing" in suites["detector_coverage"]
