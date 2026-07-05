"""Round-trip reward wiring — offline (LLM + reader monkeypatched)."""
import cloak.train.roundtrip as rt


class _StubClient:
    def __init__(self, replies):
        self.replies = replies
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.replies[len(self.prompts) - 1]


def test_roundtrip_batch_inverts_and_scores(monkeypatch):
    # remote echoes the fill; invert() must map it back; probes scored on out_final
    stub = _StubClient(["Patient is a fifty-something female with chest pain."])
    monkeypatch.setattr(rt, "_remote", lambda: stub)
    monkeypatch.setattr(rt, "fact_f1s", lambda out, probes: [1.0 if "50" in out else 0.0])
    jobs = [{"corpus": "clinical",
             "doc_p": "a fifty-something female reports chest pain",
             "R": [{"surface": "50-year-old", "type": "DEM", "action": "generalize",
                    "replacement": "fifty-something"}],
             "probes": [{"surface": "50-year-old", "question": "How old is the patient?"}]}]
    res = rt.roundtrip_batch(jobs, workers=1)
    assert len(res) == 1
    assert "50-year-old" in res[0]["out_final"]          # inversion fired
    assert res[0]["recall"] == 1.0 and res[0]["f1s"] == [1.0]
    assert "fifty-something female reports" in stub.prompts[0]   # doc_p reached the template


def test_roundtrip_batch_no_probes_gives_none(monkeypatch):
    stub = _StubClient(["anything"])
    monkeypatch.setattr(rt, "_remote", lambda: stub)
    res = rt.roundtrip_batch([{"corpus": "enron", "doc_p": "x", "R": [], "probes": []}],
                             workers=1)
    assert res[0]["recall"] is None and res[0]["f1s"] == []


def test_fact_f1s_matches_fact_recall(monkeypatch):
    import cloak.train.reward as rw
    monkeypatch.setattr(rw, "_qa_answer", lambda q, c: "42 mg")
    probes = [{"surface": "42 mg", "question": "What dose?"},
              {"surface": "Oslo", "question": "Where?"}]
    f1s = rw.fact_f1s("text", probes)
    assert f1s[0] == 1.0 and f1s[1] == 0.0
    assert rw.fact_recall("text", probes) == sum(f1s) / 2
