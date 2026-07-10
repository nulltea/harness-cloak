import cloak.frozen_extractor as fx
import cloak.train.roundtrip as rt


class _StubClient:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.reply


def test_roundtrip_default_uses_legacy_invert_without_extractor_version(monkeypatch):
    calls = []
    stub = _StubClient("remote output")

    def fake_invert(out_p, R):
        calls.append((out_p, R))
        return "legacy final", {"unused": True}

    monkeypatch.setattr(rt, "_remote", lambda: stub)
    monkeypatch.setattr(rt, "invert", fake_invert)
    monkeypatch.setattr(rt, "fact_f1s", lambda out_final, probes, refresh=False: [1.0])

    R = [{"surface": "Alice", "replacement": "<PERSON_1>"}]
    result = rt.roundtrip_batch(
        [{"corpus": "enron", "doc_p": "<PERSON_1> sent mail.", "R": R,
          "probes": [{"surface": "Alice", "question": "Who sent mail?"}]}],
        workers=1,
    )[0]

    assert calls == [("remote output", R)]
    assert result["out_final"] == "legacy final"
    assert result["f1s"] == [1.0]
    assert result["recall"] == 1.0
    assert "extractor_version" not in result


def test_roundtrip_optin_uses_frozen_extractor_and_stamps_version(monkeypatch):
    calls = []
    stub = _StubClient("remote output")
    models = {"encoder": object(), "nli": object(), "mlm": object()}

    def fake_extract(doc_p, R, out_p, *, models):
        calls.append((doc_p, R, out_p, models))
        return "frozen final", {"unused": True}

    monkeypatch.setattr(rt, "_remote", lambda: stub)
    monkeypatch.setattr(rt, "fact_f1s", lambda out_final, probes, refresh=False: [1.0])
    monkeypatch.setattr(fx, "extract", fake_extract)
    monkeypatch.setattr(fx, "extractor_version", lambda: "fx-test-version")

    R = [{"surface": "Alice", "replacement": "a person"}]
    result = rt.roundtrip_batch(
        [{"corpus": "enron", "doc_p": "a person sent mail.", "R": R,
          "probes": [{"surface": "Alice", "question": "Who sent mail?"}]}],
        workers=1,
        extractor_models=models,
    )[0]

    assert calls == [("a person sent mail.", R, "remote output", models)]
    assert result["out_final"] == "frozen final"
    assert result["extractor_version"] == "fx-test-version"
