from types import SimpleNamespace

from cloak.detection.detect import (
    Detector,
    PROFILES,
    QA_V2_CLINICAL_LABELS,
    Span,
    _chunks,
    _clinical_presidio_view,
    _dedupe_with_diagnostics,
    _native_clinical_code_rejection,
    _stop_words,
)


def test_clinical_presidio_view_compacts_split_contractions_without_moving_offsets():
    text = "i wan na leave and i'm gon na call Andrew"
    view, events = _clinical_presidio_view(text)

    assert len(view) == len(text)
    assert view == "i wanna  leave and i'm gonna  call Andrew"
    assert [
        (event.start, event.end, event.surface, event.analysis_surface)
        for event in events
    ] == [
        (2, 8, "wan na", "wanna "),
        (23, 29, "gon na", "gonna "),
    ]
    assert all(text[event.start:event.end] == event.surface for event in events)


def test_native_clinical_code_contract_requires_identifier_shape():
    assert _native_clinical_code_rejection("heart rate", "medical code") == (
        "clinical_code_without_identifier_shape"
    )
    assert _native_clinical_code_rejection("two out of six", "medical code") == (
        "clinical_code_without_identifier_shape"
    )
    assert _native_clinical_code_rejection("E11.9", "medical code") is None
    assert _native_clinical_code_rejection("AB123456", "healthcare number") is None
    assert _native_clinical_code_rejection("943 476 5919", "healthcare number") is None
    assert _native_clinical_code_rejection("heart rate", "condition") is None


def test_qa_v2_native_labels_map_explicit_names_to_person():
    assert QA_V2_CLINICAL_LABELS["name"] == "PERSON"
    assert QA_V2_CLINICAL_LABELS["first name"] == "PERSON"
    assert QA_V2_CLINICAL_LABELS["last name"] == "PERSON"


def test_dedupe_diagnostics_record_same_type_and_cross_type_losers():
    spans = [
        Span(0, 6, "Andrew", "PERSON", 0.91, "gliner", raw_label="name"),
        Span(
            0,
            6,
            "Andrew",
            "PERSON",
            0.85,
            "presidio",
            raw_label="PERSON",
            recognizer="SpacyRecognizer",
        ),
        Span(0, 12, "Andrew Smith", "MISC", 0.40, "gliner", raw_label="condition"),
    ]

    winners, diagnostics = _dedupe_with_diagnostics(spans)

    assert [(span.text, span.type, span.source) for span in winners] == [
        ("Andrew", "PERSON", "gliner")
    ]
    statuses = {(row["source"], row["runtime_type"]): row for row in diagnostics}
    assert statuses[("gliner", "PERSON")]["status"] == "accepted"
    assert statuses[("presidio", "PERSON")]["status"] == "overlap_loser"
    assert statuses[("presidio", "PERSON")]["winner"]["source"] == "gliner"
    assert statuses[("gliner", "MISC")]["status"] == "overlap_loser"


class FakeGLiNER:
    config = SimpleNamespace(max_len=None)

    def __init__(self, entities):
        self.entities = entities
        self.calls = []

    def batch_predict_entities(self, texts, labels, threshold, batch_size):
        self.calls.append(list(texts))
        return [list(self.entities) for _text in texts]


class FakePresidio:
    def __init__(self, entities=()):
        self.entities = list(entities)

    def analyze(self, *, text, language):
        return list(self.entities)


def fake_qa_v2_detector(*, gliner_entities, presidio_entities=()):
    detector = Detector.__new__(Detector)
    detector.threshold = 0.35
    detector.batch_size = 16
    detector.fine_dem = False
    detector.label2type = dict(QA_V2_CLINICAL_LABELS)
    detector.labels = list(detector.label2type)
    detector.gliner = FakeGLiNER(gliner_entities)
    detector.max_words = None
    detector.presidio = FakePresidio(presidio_entities)
    detector.profile = PROFILES["clinical"]
    detector.stop_words = _stop_words(detector.profile)
    return detector


def test_detect_with_diagnostics_attributes_contract_rejections_and_normalization():
    detector = fake_qa_v2_detector(
        gliner_entities=[
            {"start": 0, "end": 6, "text": "andrew", "label": "name", "score": 0.95},
            {
                "start": 27,
                "end": 37,
                "text": "heart rate",
                "label": "medical code",
                "score": 0.56,
            },
        ],
        presidio_entities=[],
    )
    result = detector.detect_with_diagnostics("andrew says i wan na check heart rate")

    assert [(span.text, span.type, span.raw_label) for span in result.spans] == [
        ("andrew", "PERSON", "name")
    ]
    assert any(
        row["reason"] == "clinical_code_without_identifier_shape"
        for row in result.candidates
    )
    assert [event.surface for event in result.normalizations] == ["wan na"]
    assert detector.detect("andrew says i wan na check heart rate") == result.spans


def test_detect_many_with_diagnostics_flattens_chunks_and_preserves_document_order():
    texts = [
        "first document says wan na leave",
        "second document says gon na stay",
    ]
    detector = fake_qa_v2_detector(gliner_entities=[])

    results = detector.detect_many_with_diagnostics(texts)

    expected_chunks = [chunk for source in texts for _offset, chunk in _chunks(source)]
    assert detector.gliner.calls == [expected_chunks]
    assert [[event.surface for event in result.normalizations] for result in results] == [
        ["wan na"],
        ["gon na"],
    ]
