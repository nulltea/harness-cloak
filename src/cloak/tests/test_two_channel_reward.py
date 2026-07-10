"""Two-channel carrier reward wiring -- offline reader/generator fakes."""

import pytest

import cloak.train.roundtrip as rt


class _StubClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.replies[len(self.prompts) - 1]


def _job(**overrides):
    job = {
        "corpus": "clinical",
        "doc_p": "Patient has a generalized endocrine condition.",
        "R": [],
        "probes": [{"surface": "hypothyroidism", "question": "What diagnosis?"}],
        "ladder": [
            {
                "surface": "hypothyroidism",
                "rung": 0,
                "q": "What exact diagnosis is documented?",
                "rungs": ["hypothyroidism", "an endocrine condition"],
            },
            {
                "surface": "hypothyroidism",
                "rung": 1,
                "q": "What kind of diagnosis is documented?",
                "rungs": ["hypothyroidism", "an endocrine condition"],
            },
        ],
    }
    job.update(overrides)
    return job


def test_ladder_exact_reads_out_final_and_semantic_reads_out_p(monkeypatch):
    stub = _StubClient(["OUT_P: patient has an endocrine condition."])
    calls = []

    def fake_read(questions, context, refresh=False):
        calls.extend((q, context, refresh) for q in questions)
        out = []
        for q in questions:
            if "exact" in q:
                out.append("hypothyroidism" if context.startswith("OUT_FINAL") else "")
            elif "kind" in q:
                out.append("an endocrine condition" if context.startswith("OUT_P") else "")
            else:
                out.append("")
        return out

    monkeypatch.setattr(rt, "_remote", lambda: stub)
    monkeypatch.setattr(rt, "invert", lambda out_p, R: ("OUT_FINAL: hypothyroidism.", None))
    monkeypatch.setattr(rt, "_read_batch", fake_read)

    res = rt.roundtrip_batch([_job()], workers=1)[0]

    assert res["recall"] == pytest.approx(1.0)
    assert res["f1s"] == [1.0]
    assert ("What exact diagnosis is documented?", "OUT_FINAL: hypothyroidism.", False) in calls
    assert ("What kind of diagnosis is documented?",
            "OUT_P: patient has an endocrine condition.", False) in calls


def test_placeholder_fill_gets_echo_only_when_semantic_tier_not_entailed(monkeypatch):
    stub = _StubClient(["OUT_P: patient has <CONDITION_1>.",
                        "OUT_P: patient has <CONDITION_1>."])

    def fake_read(questions, context, refresh=False):
        return ["hypothyroidism" if "exact" in q else "" for q in questions]

    monkeypatch.setattr(rt, "_remote", lambda: stub)
    monkeypatch.setattr(rt, "invert", lambda out_p, R: ("OUT_FINAL: hypothyroidism.", None))
    monkeypatch.setattr(rt, "_read_batch", fake_read)

    res = rt.roundtrip_batch([_job()], workers=1)[0]

    assert res["span_parts"] == [0.5]
    assert res["recall"] == pytest.approx(0.5)


def test_carrier_combines_available_components_unweighted(monkeypatch):
    stub = _StubClient(["OUT_P: patient has <CONDITION_1>.",
                        "OUT_P: patient has <CONDITION_1>."])
    decision_prompts = []
    shuffle_seed_keys = []

    def fake_read(questions, context, refresh=False):
        out = []
        for q in questions:
            if q.startswith("What exact"):
                out.append("hypothyroidism")
            elif "Options:" in q:
                decision_prompts.append(q)
                out.append("route to endocrinology" if len(decision_prompts) == 1 else "primary care")
            else:
                out.append("")
        return out

    monkeypatch.setattr(rt, "_remote", lambda: stub)
    monkeypatch.setattr(rt, "invert", lambda out_p, R: ("OUT_FINAL: hypothyroidism.", None))
    monkeypatch.setattr(rt, "_read_batch", fake_read)
    monkeypatch.setattr(
        rt,
        "mc_shuffle",
        lambda options, seed_key: shuffle_seed_keys.append(seed_key) or list(reversed(options)),
    )
    monkeypatch.setattr(rt, "schema_field_score", lambda out_final, out_hi: 1.0)
    decisions = [
        {"q": "Which route?", "options": ["primary care", "route to endocrinology"],
         "gold": "route to endocrinology", "span_ids": ["s0"]},
        {"q": "What follow-up?", "options": ["primary care", "cardiology"],
         "gold": "cardiology", "span_ids": ["s0"]},
    ]

    res = rt.roundtrip_batch([
        _job(decisions=decisions, schema=True, out_hi="CEILING")
    ], workers=1)[0]

    assert res["decision_score"] == pytest.approx(0.5)
    assert res["schema_score"] == pytest.approx(1.0)
    assert res["recall"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    assert len(shuffle_seed_keys) == 2
    assert shuffle_seed_keys[0] != shuffle_seed_keys[1]
    assert all("Options:" in p for p in decision_prompts)


def test_schema_component_requires_flag_and_out_hi(monkeypatch):
    stub = _StubClient(["OUT_P: patient has <CONDITION_1>.",
                        "OUT_P: patient has <CONDITION_1>."])
    schema_calls = []

    monkeypatch.setattr(rt, "_remote", lambda: stub)
    monkeypatch.setattr(rt, "invert", lambda out_p, R: ("OUT_FINAL: hypothyroidism.", None))
    monkeypatch.setattr(
        rt,
        "_read_batch",
        lambda questions, context, refresh=False: [
            "hypothyroidism" if "exact" in q else "" for q in questions
        ],
    )
    monkeypatch.setattr(
        rt,
        "schema_field_score",
        lambda out_final, out_hi: schema_calls.append((out_final, out_hi)) or 1.0,
    )

    res = rt.roundtrip_batch([
        _job(schema=True),
        _job(schema=False, out_hi="CEILING"),
    ], workers=1)

    assert [r.get("schema_score") for r in res] == [None, None]
    assert schema_calls == []
    assert "EXACTLY these sections" in stub.prompts[0]
    assert "standard note sections" in stub.prompts[1]


def test_reader_refresh_reaches_echo_semantic_and_decision_channels(monkeypatch):
    stub = _StubClient(["OUT_P: patient has an endocrine condition."])
    refreshes = []

    def fake_read(questions, context, refresh=False):
        refreshes.extend(refresh for _ in questions)
        return ["hypothyroidism" if "exact" in q else "an endocrine condition"
                for q in questions]

    monkeypatch.setattr(rt, "_remote", lambda: stub)
    monkeypatch.setattr(rt, "invert", lambda out_p, R: ("OUT_FINAL: hypothyroidism.", None))
    monkeypatch.setattr(rt, "_read_batch", fake_read)

    rt.roundtrip_batch([
        _job(decisions=[{"q": "Which route?", "options": ["endocrinology", "primary care"],
                         "gold": "endocrinology"}])
    ], workers=1, reader_refresh=True)

    assert refreshes == [True, True, True]


def test_legacy_job_result_shape_and_score_are_unchanged(monkeypatch):
    stub = _StubClient(["OUT_P"])

    monkeypatch.setattr(rt, "_remote", lambda: stub)
    monkeypatch.setattr(rt, "invert", lambda out_p, R: ("OUT_FINAL", None))
    monkeypatch.setattr(rt, "fact_f1s", lambda out, probes, refresh=False: [0.25, 0.75])
    job = {
        "corpus": "enron",
        "doc_p": "x",
        "R": [],
        "probes": [
            {"surface": "Alice", "question": "q1"},
            {"surface": "Alice", "question": "q2"},
        ],
    }

    res = rt.roundtrip_batch([job], workers=1)[0]

    assert res == {"out_p": "OUT_P", "out_final": "OUT_FINAL",
                   "f1s": [0.25, 0.75], "recall": 0.75}
