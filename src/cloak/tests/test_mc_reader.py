"""MC reader template: decisions must not go through the extractive QA prompt."""

import cloak.train.reward as rw


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def generate(self, prompt, refresh=False):
        self.prompts.append(prompt)
        return self.reply


def test_read_mc_batch_uses_mc_template_not_extractive_qa(monkeypatch):
    fake = FakeClient("route to endocrinology")
    monkeypatch.setattr(rw, "_qa_client", lambda: fake)

    q = rw.decision_prompt("Which route?", ["primary care", "route to endocrinology"])
    out = rw._read_mc_batch([q], "NOTE TEXT")

    assert out == ["route to endocrinology"]
    p = fake.prompts[0]
    assert "NOTE TEXT" in p and "Which route?" in p and "- primary care" in p
    # the extractive framing that made non-verbatim options unanswerable must be gone
    assert "copied from the note" not in p
    assert "NONE" not in p


def test_read_batch_still_uses_extractive_template(monkeypatch):
    fake = FakeClient("hypothyroidism")
    monkeypatch.setattr(rw, "_qa_client", lambda: fake)

    out = rw._read_batch(["What diagnosis?"], "NOTE TEXT")

    assert out == ["hypothyroidism"]
    assert "copied from the note" in fake.prompts[0]


def test_read_mc_batch_empty_questions_and_none_reply(monkeypatch):
    assert rw._read_mc_batch([], "NOTE") == []
    fake = FakeClient("NONE")
    monkeypatch.setattr(rw, "_qa_client", lambda: fake)
    assert rw._read_mc_batch([rw.decision_prompt("Q?", ["a", "b"])], "NOTE") == [""]
