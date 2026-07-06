import cloak.extract as ex


def test_detector_pointer_constructs_audit_detector_checkpoint(monkeypatch):
    import cloak.detect as detect

    seen = {}

    class FakeDetector:
        def __init__(self, gliner_model, **kwargs):
            seen["gliner_model"] = gliner_model
            seen["kwargs"] = kwargs

        def detect(self, text):
            return []

    monkeypatch.setattr(detect, "Detector", FakeDetector)
    out, stats = ex.invert_detector_pointer(
        "No usable typed candidate here.",
        [{"action": "generalize", "surface": "Boston", "type": "LOC",
          "replacement": "a city in Massachusetts"}],
    )

    assert out == "No usable typed candidate here."
    assert seen["gliner_model"] == "data/models/pii_gliner_multidomain/checkpoint-2479"
    assert stats["gen_abstain"] == 1


def test_semantic_window_inverts_recoverable_paraphrase(monkeypatch):
    def fake_scores(fill, snippets):
        return [0.9 if s == "that Massachusetts city" else 0.2 for s in snippets]

    monkeypatch.setattr(ex, "_semantic_scores", fake_scores)
    out, stats = ex.invert(
        "She now lives in that Massachusetts city.",
        [{"action": "generalize", "surface": "Boston", "type": "LOC",
          "replacement": "a city in Massachusetts"}],
    )

    assert out == "She now lives in Boston."
    assert stats["gen_semantic"] == 1
    assert stats["gen_absent"] == 0


def test_semantic_window_abstains_on_close_runner_up(monkeypatch):
    monkeypatch.setattr(ex, "_semantic_scores", lambda fill, snippets: [0.9] * len(snippets))

    out, stats = ex.invert(
        "She mentioned that Massachusetts city and another Massachusetts city.",
        [{"action": "generalize", "surface": "Boston", "type": "LOC",
          "replacement": "a city in Massachusetts"}],
    )

    assert out == "She mentioned that Massachusetts city and another Massachusetts city."
    assert stats["gen_semantic"] == 0
    assert stats["gen_absent"] == 1


def test_semantic_window_abstains_on_type_sanity_failure(monkeypatch):
    monkeypatch.setattr(ex, "_semantic_scores", lambda fill, snippets: [0.95] * len(snippets))

    out, stats = ex.invert(
        "The early filing was completed.",
        [{"action": "generalize", "surface": "January 2019", "type": "DATETIME",
          "replacement": "early 2019"}],
    )

    assert out == "The early filing was completed."
    assert stats["gen_semantic"] == 0
    assert stats["gen_absent"] == 1


def test_detector_pointer_assigns_typed_residue(monkeypatch):
    def fake_scores(query, snippets):
        return [0.9 if s == "Massachusetts city" else 0.1 for s in snippets]

    monkeypatch.setattr(ex, "_pointer_scores", fake_scores)
    out, stats = ex.invert_detector_pointer(
        "She now lives in that Massachusetts city.",
        [{"action": "generalize", "surface": "Boston", "type": "LOC",
          "replacement": "a city in Massachusetts"}],
        detector=lambda text: [{"start": text.index("Massachusetts"),
                                "end": text.index("city") + len("city"),
                                "text": "Massachusetts city",
                                "type": "LOC",
                                "score": 0.99}],
    )

    assert out == "She now lives in that Boston."
    assert stats["gen_pointer"] == 1
    assert stats["gen_abstain"] == 0
    assert stats["gen_absent"] == 0


def test_detector_pointer_abstains_on_close_competing_span(monkeypatch):
    monkeypatch.setattr(ex, "_pointer_scores", lambda query, snippets: [0.9] * len(snippets))
    out, stats = ex.invert_detector_pointer(
        "She saw one Massachusetts city and another Massachusetts city.",
        [{"action": "generalize", "surface": "Boston", "type": "LOC",
          "replacement": "a city in Massachusetts"}],
        detector=lambda text: [
            {"start": text.index("one"), "end": text.index("and") - 1,
             "text": "one Massachusetts city", "type": "LOC", "score": 0.99},
            {"start": text.index("another"), "end": text.rindex("city") + len("city"),
             "text": "another Massachusetts city", "type": "LOC", "score": 0.99},
        ],
    )

    assert out == "She saw one Massachusetts city and another Massachusetts city."
    assert stats["gen_pointer"] == 0
    assert stats["gen_abstain"] == 1
    assert stats["gen_absent"] == 1


def test_detector_pointer_preserves_rule_prepass(monkeypatch):
    def fail_detector(text):
        raise AssertionError("detector should not run when exact/fuzzy pre-pass resolves all entries")

    monkeypatch.setattr(ex, "_pointer_scores", lambda query, snippets: [])
    out, stats = ex.invert_detector_pointer(
        "She takes a drug daily.",
        [{"action": "generalize", "surface": "lasix", "type": "MISC",
          "replacement": "a drug"}],
        detector=fail_detector,
    )

    assert out == "She takes lasix daily."
    assert stats["gen_exact"] == 1
    assert stats["gen_pointer"] == 0
    assert stats["gen_abstain"] == 0
