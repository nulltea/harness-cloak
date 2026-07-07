from dataclasses import dataclass

from bench.baselines import (
    make_all_placeholder_record,
    make_coarsest_text_record,
    make_no_privacy_record,
    make_oracle_extractor_record,
)


@dataclass
class SpanLike:
    start: int
    end: int
    text: str
    type: str
    score: float = 1.0
    chain: int = 0


def test_no_privacy_record_keeps_text_and_records_keep_actions():
    text = "Martha is 50 years old."

    doc_p, R = make_no_privacy_record(text, [SpanLike(0, 6, "Martha", "PERSON")])

    assert doc_p == text
    assert R[0]["action"] == "keep"
    assert R[0]["replacement"] == "Martha"


def test_all_placeholder_record_replaces_each_span():
    text = "Martha is in Oslo."
    spans = [SpanLike(0, 6, "Martha", "PERSON"), SpanLike(13, 17, "Oslo", "LOC")]

    doc_p, R = make_all_placeholder_record(text, spans)

    assert "Martha" not in doc_p
    assert "Oslo" not in doc_p
    assert "<PERSON_1>" in doc_p
    assert "<LOC_1>" in doc_p
    assert [r["action"] for r in R] == ["placeholder", "placeholder"]


def test_coarsest_text_record_uses_text_when_available_else_placeholder():
    text = "Martha is a cardiologist."
    spans = [SpanLike(12, 24, "cardiologist", "profession")]

    doc_p, R = make_coarsest_text_record(text, spans)

    assert "cardiologist" not in doc_p
    assert R[0]["replacement"]
    assert R[0]["action"] in {"generalize", "placeholder"}


def test_oracle_extractor_only_replaces_echoed_replacements():
    out_p = "<PERSON_1> is a healthcare worker."
    R = [
        {"surface": "Martha", "replacement": "<PERSON_1>", "action": "placeholder", "type": "PERSON"},
        {"surface": "cardiologist", "replacement": "healthcare worker", "action": "generalize", "type": "profession"},
        {"surface": "Oslo", "replacement": "a city", "action": "generalize", "type": "LOC"},
    ]

    out_final = make_oracle_extractor_record(out_p, R)

    assert out_final == "Martha is a cardiologist."
    assert "Oslo" not in out_final
