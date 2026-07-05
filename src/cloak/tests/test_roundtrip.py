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


def test_fact_recall_is_per_fact_max_mean_over_facts(monkeypatch):
    import cloak.train.reward as rw
    # three questions -> two are the SAME fact after canon ("42 mg"/"42 milligrams"),
    # the third a distinct fact. fact score = max over its questions, mean over facts.
    monkeypatch.setattr(rw, "fact_f1s", lambda out, ps: [0.4, 0.9, 0.2])
    probes = [{"surface": "42 mg", "question": "q1"},
              {"surface": "42 milligrams", "question": "q2"},   # same fact as q1
              {"surface": "Oslo", "question": "q3"}]            # distinct fact
    # fact "42 mg": max(0.4, 0.9) = 0.9 ; fact "oslo": 0.2 ; mean = 0.55 (not the 0.5 mean)
    assert rw.fact_recall("text", probes) == (0.9 + 0.2) / 2


def test_fact_recall_none_without_probes():
    import cloak.train.reward as rw
    assert rw.fact_recall("text", []) is None
