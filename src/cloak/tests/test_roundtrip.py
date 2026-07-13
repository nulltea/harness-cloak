"""Round-trip reward wiring — offline (LLM + reader monkeypatched)."""
import pytest

import cloak.extract as extract
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
    monkeypatch.setattr(rt, "fact_f1s",
                        lambda out, probes, refresh=False: [1.0 if "50" in out else 0.0])
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


def test_remote_gen_client_uses_single_flight(monkeypatch):
    seen = {}

    class FakeLLMClient:
        def __init__(self, model, **kwargs):
            seen["model"] = model
            seen["kwargs"] = kwargs

        def generate(self, prompt, **kwargs):
            return "unused"

    import cloak.llm as llm
    monkeypatch.setenv("CLOAK_LLM_CACHE", "test-cache")
    monkeypatch.setattr(llm, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(rt, "_client", None)

    try:
        rt._remote()
    finally:
        monkeypatch.setattr(rt, "_client", None)

    assert seen["model"] == rt.RT_MODEL
    assert seen["kwargs"]["single_flight"] is True


def test_reader_refresh_reaches_reader_but_never_gen(monkeypatch):
    import cloak.train.reward as rw

    gen_refreshes = []
    reader_refreshes = []

    class FakeGen:
        def generate(self, prompt, **kwargs):
            gen_refreshes.append(kwargs.get("refresh", False))
            return "Alice received the numbers."

    class FakeReader:
        def generate(self, prompt, refresh=None, **kwargs):
            reader_refreshes.append(refresh)
            return "Alice"

    monkeypatch.setattr(rt, "_remote", lambda: FakeGen())
    monkeypatch.setattr(rw, "_qa_client", lambda: FakeReader())
    monkeypatch.setattr(rt, "fact_f1s", rw.fact_f1s)

    jobs = [{"corpus": "enron",
             "doc_p": "Alice received the numbers.",
             "R": [],
             "probes": [{"surface": "Alice", "question": "Who received the numbers?"}]}]

    rt.roundtrip_batch(jobs, workers=1)
    rt.roundtrip_batch(jobs, workers=1, reader_refresh=True)

    assert reader_refreshes == [False, True]
    assert gen_refreshes == [False, False]


def test_roundtrip_batch_no_probes_gives_none(monkeypatch):
    stub = _StubClient(["anything"])
    monkeypatch.setattr(rt, "_remote", lambda: stub)
    res = rt.roundtrip_batch([{"corpus": "enron", "doc_p": "x", "R": [], "probes": []}],
                             workers=1)
    assert res[0]["recall"] is None and res[0]["f1s"] == []


def test_utility_artifact_roundtrip_aggregates_builder_components(monkeypatch):
    monkeypatch.setattr(rt, "_remote", lambda: _StubClient(["remote output"]))
    monkeypatch.setattr(rt, "invert", lambda out_p, replacements: ("final output", None))
    monkeypatch.setattr(
        rt,
        "score_utility",
        lambda *args, **kwargs: {"component_scores": {"context": 0.5, "delivered": 1.0}},
    )
    artifact = {
        "documents": {"d1": {
            "assertion_ids": ["context", "delivered"],
            "utility_weight_denominator": 1.0,
        }},
        "assertions": {
            "context": {
                "assertion_id": "context", "doc_id": "d1", "status": "accepted",
                "weight": 0.6,
            },
            "delivered": {
                "assertion_id": "delivered", "doc_id": "d1", "status": "accepted",
                "weight": 0.4,
            },
        },
    }

    result = rt.roundtrip_batch([{
        "corpus": "clinical",
        "doc_id": "d1",
        "doc_p": "generalized",
        "R": [],
        "probes": [],
        "utility_artifact": artifact,
    }], workers=1)[0]

    assert result["component_scores"] == {"context": 0.5, "delivered": 1.0}
    assert result["recall"] == pytest.approx(0.7)


def test_roundtrip_reward_pin_tracks_actual_task_prompt_and_invert(monkeypatch):
    baseline = rt.roundtrip_reward_pin({"scorer": "utility-v1"}, corpus="clinical")

    assert baseline["task_prompt"]["sha256"].startswith("sha256:")
    assert baseline["extractor"]["version"]
    assert baseline["extractor"]["sha256"].startswith("sha256:")
    assert set(baseline["extractor"]["modules"]) == {
        "cloak.extract",
        "cloak.runtime_types",
    }
    assert set(baseline["extractor"]["packages"]) == {
        "huggingface-hub",
        "numpy",
        "rapidfuzz",
        "sentence-transformers",
        "tokenizers",
        "torch",
        "transformers",
    }
    assert baseline["extractor"]["semantic_model"] == {
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    }

    with monkeypatch.context() as context:
        context.setitem(rt.TASK_TEMPLATE, "clinical", "changed prompt\n{doc}")
        changed_prompt = rt.roundtrip_reward_pin(
            {"scorer": "utility-v1"}, corpus="clinical"
        )
    with monkeypatch.context() as context:
        context.setattr(
            extract,
            "INVERT_EXTRACTOR_VERSION",
            "invert-rule-cascade-test-change",
            raising=False,
        )
        changed_invert = rt.roundtrip_reward_pin(
            {"scorer": "utility-v1"}, corpus="clinical"
        )
    with monkeypatch.context() as context:
        context.setattr(
            extract,
            "_module_source_hash",
            lambda module: f"sha256:changed:{module.__name__}",
            raising=False,
        )
        changed_helper = rt.roundtrip_reward_pin(
            {"scorer": "utility-v1"}, corpus="clinical"
        )
    with monkeypatch.context() as context:
        context.setattr(
            extract,
            "_distribution_version",
            lambda name: f"changed-{name}",
            raising=False,
        )
        changed_package = rt.roundtrip_reward_pin(
            {"scorer": "utility-v1"}, corpus="clinical"
        )

    assert changed_prompt != baseline
    assert changed_invert != baseline
    assert changed_helper != baseline
    assert changed_package != baseline


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
