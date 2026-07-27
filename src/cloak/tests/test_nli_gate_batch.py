"""nli_gate_batch: one pipeline call for many (entity, context, candidates) jobs."""
import cloak.lattice.core as cl


class FakeNLI:
    """Mimics the transformers pipeline: returns entailment score per pair."""

    def __init__(self, scores):
        self.scores = list(scores)  # consumed in call order
        self.calls = 0
        self.pair_counts = []

    def __call__(self, pairs, top_k=None, truncation=True):
        self.calls += 1
        self.pair_counts.append(len(pairs))
        out = []
        for _ in pairs:
            s = self.scores.pop(0)
            out.append([{"label": "entailment", "score": s},
                        {"label": "neutral", "score": 1 - s}])
        return out


def test_batch_matches_single_and_returns_scores(monkeypatch):
    jobs = [
        ("Oslo", "She lives in Oslo.", ["a city in Norway", "a bank"]),
        ("diabetes", "He has diabetes.", ["a chronic condition"]),
    ]
    monkeypatch.setattr(cl, "_nli", FakeNLI([0.9, 0.2, 0.8]))
    got = cl.nli_gate_batch(jobs, thresh=0.6)
    assert got[0] == [("a city in Norway", 0.9)]  # 0.2 below thresh
    assert got[1] == [("a chronic condition", 0.8)]

    # single-job wrapper: same filtering, plain list, one underlying call
    monkeypatch.setattr(cl, "_nli", FakeNLI([0.9, 0.2]))
    assert cl.nli_gate("Oslo", "She lives in Oslo.",
                       ["a city in Norway", "a bank"]) == ["a city in Norway"]


def test_batch_is_one_pipeline_call(monkeypatch):
    fake = FakeNLI([0.9, 0.9, 0.9])
    monkeypatch.setattr(cl, "_nli", fake)
    cl.nli_gate_batch([("a", "x a y.", ["p"]), ("b", "x b y.", ["q", "r"])])
    assert fake.calls == 1 and fake.pair_counts == [3]


def test_batch_preserves_per_job_fail_closed(monkeypatch):
    fake = FakeNLI([0.9])
    monkeypatch.setattr(cl, "_nli", fake)
    jobs = [
        ("missing", "entity not in this context.", ["c1"]),   # no sentence hit -> []
        ("self", "the self sentence.", ["self reference"]),   # self-ref filtered -> []
        ("ok", "an ok sentence.", ["fine phrase"]),
    ]
    got = cl.nli_gate_batch(jobs)
    assert got[0] == [] and got[1] == [] and got[2] == [("fine phrase", 0.9)]
    assert fake.pair_counts == [1]  # only the viable pair hit the pipeline


def test_empty_jobs_no_pipeline_call(monkeypatch):
    fake = FakeNLI([])
    monkeypatch.setattr(cl, "_nli", fake)
    assert cl.nli_gate_batch([]) == []
    assert cl.nli_gate_batch([("e", "no hit here.", [])]) == [[]]
    assert fake.calls == 0
