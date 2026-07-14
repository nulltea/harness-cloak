"""substitute() batched matcher pre-pass + R match provenance."""
import cloak.substitute as sub
from cloak.detect import Span
from cloak.profile_match import MatchResult, span_key
from cloak.substitute import prepare_spans_for_substitution


def _spans(text, *triples):
    # adapted to real cloak.detect.Span: score + source are required fields
    out = []
    for surface, typ in triples:
        i = text.index(surface)
        out.append(Span(start=i, end=i + len(surface), text=surface, type=typ,
                        score=0.9, source="gliner"))
    return out


def test_prepass_feeds_lattice_and_records_match(monkeypatch):
    text = "He is diabetic and takes aspirin, then insulin."
    spans = _spans(text, ("diabetic", "health-condition"), ("aspirin", "drug"),
                   ("insulin", "drug"))
    monkeypatch.setattr(sub, "coref_chains", lambda t, s: s)
    monkeypatch.setattr(sub, "walk_risk", lambda *a, **k: 0.0)
    submitted = []

    def fake_batch(items, **kw):
        submitted.extend(items)
        return {
            span_key("diabetic", "health-condition"):
                MatchResult(["endocrine condition"], "semantic", False, 0.84,
                            "diabetes", nli=0.91),
            span_key("aspirin", "drug"):
                MatchResult(["analgesic drug"], "exact", True, 1.0, "aspirin"),
            span_key("insulin", "drug"): None,   # abstained -> teacher-cache/placeholder path
        }
    monkeypatch.setattr(sub, "match_spans_batch", fake_batch)
    doc_p, R = sub.substitute(text, spans, tau=2.0)   # tau>1: accept first level

    assert {(s, t) for s, t, _ in submitted} == {("diabetic", "health-condition"),
                                                 ("aspirin", "drug"), ("insulin", "drug")}
    by_surface = {r["surface"]: r for r in R}
    assert by_surface["diabetic"]["match"] == {"kind": "semantic", "entry": "diabetes",
                                               "similarity": 0.84, "nli": 0.91}
    assert by_surface["diabetic"]["replacement"] == "endocrine condition"
    assert by_surface["aspirin"]["match"] == {"kind": "exact", "entry": "aspirin"}
    assert "match" not in by_surface["insulin"]   # abstain: placeholder, no provenance
    assert by_surface["insulin"]["replacement"].startswith("<DRUG_")


def test_rule_and_direct_types_not_submitted(monkeypatch):
    text = "Sarah paid 120,000 dollars in 2019."
    spans = _spans(text, ("Sarah", "PERSON"), ("120,000 dollars", "QUANTITY"),
                   ("2019", "DATETIME"))
    monkeypatch.setattr(sub, "coref_chains", lambda t, s: s)
    monkeypatch.setattr(sub, "walk_risk", lambda *a, **k: 0.0)
    called = []
    monkeypatch.setattr(sub, "match_spans_batch",
                        lambda items, **kw: called.extend(items) or {})
    sub.substitute(text, spans, tau=2.0)
    assert called == []   # no profile-backed spans -> pre-pass not consulted


def test_repeat_surface_copies_match(monkeypatch):
    text = "diabetic today; still diabetic tomorrow."
    spans = _spans(text, ("diabetic", "health-condition"))
    j = text.rindex("diabetic")
    spans.append(Span(start=j, end=j + len("diabetic"),
                      text="diabetic", type="health-condition",
                      score=0.9, source="gliner"))
    monkeypatch.setattr(sub, "coref_chains", lambda t, s: s)
    monkeypatch.setattr(sub, "walk_risk", lambda *a, **k: 0.0)
    m = MatchResult(["endocrine condition"], "semantic", False, 0.84, "diabetes", nli=0.9)
    monkeypatch.setattr(sub, "match_spans_batch",
                        lambda items, **kw: {span_key("diabetic", "health-condition"): m})
    _, R = sub.substitute(text, spans, tau=2.0)
    assert all(r["match"]["entry"] == "diabetes" for r in R)   # repeat reuses match too


def test_explicit_native_name_label_bypasses_wordnet_role_retyping():
    span = Span(0, 6, "andrew", "PERSON", 0.95, "gliner", raw_label="name")
    prepared, rejected = prepare_spans_for_substitution(
        "andrew arrived", [span], reject_demographic_other=True
    )
    assert [(row.text, row.type) for row in prepared] == [("andrew", "PERSON")]
    assert rejected == []


def test_qa_v2_runtime_contract_rejects_residual_demographic_fallback():
    span = Span(
        0,
        7,
        "patient",
        "PERSON",
        0.85,
        "presidio",
        raw_label="PERSON",
        recognizer="SpacyRecognizer",
    )
    prepared, rejected = prepare_spans_for_substitution(
        "patient arrived", [span], reject_demographic_other=True
    )
    assert prepared == []
    assert rejected[0]["status"] == "post_detection_rejected"
    assert rejected[0]["reason"] == "qa_v2_forbidden_demographic_other"
    assert rejected[0]["proposed_runtime_type"] == "demographic-other"


def test_substitution_record_preserves_winning_detector_provenance(monkeypatch):
    provenance = {
        "source": "gliner",
        "raw_label": "name",
        "score": 0.95,
        "candidates": [{"status": "accepted", "surface": "Andrew"}],
    }
    span = Span(
        0,
        6,
        "Andrew",
        "PERSON",
        0.95,
        "gliner",
        raw_label="name",
        detector_provenance=provenance,
    )
    monkeypatch.setattr(sub, "coref_chains", lambda _text, rows: rows)

    _doc_p, records = sub.substitute("Andrew arrived", [span])

    assert records[0]["detector_provenance"] == provenance


def test_substitution_record_synthesizes_missing_detector_provenance(monkeypatch):
    span = Span(
        0,
        6,
        "Andrew",
        "PERSON",
        0.95,
        "gliner",
        raw_label="name",
    )
    monkeypatch.setattr(sub, "coref_chains", lambda _text, rows: rows)

    _doc_p, records = sub.substitute("Andrew arrived", [span])

    assert records[0]["detector_provenance"] == {
        "source": "gliner",
        "raw_label": "name",
        "recognizer": None,
        "score": 0.95,
    }
