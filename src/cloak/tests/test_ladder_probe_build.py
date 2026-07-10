import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import build_probes  # noqa: E402
from build_probes import _with_validated_rung0, ladder_health_row  # noqa: E402
from cloak.train import ladder_probes as lp  # noqa: E402
from cloak.train.ladder_probes import (  # noqa: E402
    locator_lint,
    mc_shuffle,
    validate_decisions,
    validate_ladder,
)


def test_span_levels_drops_keep_and_sorts_by_aset():
    span_levels = getattr(lp, "span_levels", None)
    assert callable(span_levels)

    span = {
        "surface": "heart failure",
        "actions": [
            {"mode": "level", "fill": "a physical condition", "aset": 1313},
            {"mode": "level", "fill": "a cardiovascular disease", "aset": 28},
            {"mode": "level", "fill": "heart failure", "aset": 1.0},
            {"mode": "placeholder", "fill": "<HEALTH_CONDITION_1>"},
        ],
    }

    assert span_levels(span) == ["a cardiovascular disease", "a physical condition"]


def test_validate_ladder_keeps_only_ceiling_pass_floor_fail_rungs():
    entries = [
        {
            "id": "keep",
            "surface": "hypothyroidism",
            "rungs": ["hypothyroidism", "an endocrine condition"],
            "rung": 1,
            "q": "What body-system category is treated with Synthroid?",
        },
        {
            "id": "ceiling-fail",
            "surface": "metformin",
            "rungs": ["metformin", "a diabetes medication"],
            "rung": 1,
            "q": "What medication class is being continued?",
        },
        {
            "id": "floor-pass",
            "surface": "asthma",
            "rungs": ["asthma", "a respiratory condition"],
            "rung": 1,
            "q": "What condition type prompted the inhaler plan?",
        },
    ]

    hi = {
        entries[0]["q"]: "hypothyroidism",
        entries[1]["q"]: "unknown",
        entries[2]["q"]: "asthma",
    }
    lo = {
        entries[0]["q"]: "",
        entries[1]["q"]: "",
        entries[2]["q"]: "a respiratory condition",
    }

    kept, rows = validate_ladder(entries, hi.get, hi.get, lo.get, lo.get, th=0.5)

    assert [e["id"] for e in kept] == ["keep"]
    assert {r["id"]: r["verdict"] for r in rows} == {
        "keep": "kept",
        "ceiling-fail": "ceiling",
        "floor-pass": "floor",
    }


def test_validate_ladder_threshold_boundaries_are_inclusive_for_hi_strict_for_lo():
    entries = [
        {
            "id": "hi-boundary",
            "surface": "alpha beta gamma",
            "rungs": ["alpha beta gamma"],
            "rung": 0,
            "q": "What is the first boundary fact?",
        },
        {
            "id": "lo-boundary",
            "surface": "delta epsilon zeta",
            "rungs": ["delta epsilon zeta"],
            "rung": 0,
            "q": "What is the second boundary fact?",
        },
    ]

    hi = {
        entries[0]["q"]: "alpha",
        entries[1]["q"]: "delta epsilon zeta",
    }
    lo = {
        entries[0]["q"]: "",
        entries[1]["q"]: "delta",
    }

    kept, rows = validate_ladder(entries, hi.get, hi.get, lo.get, lo.get, th=0.5)

    assert [e["id"] for e in kept] == ["hi-boundary"]
    assert {r["id"]: r["verdict"] for r in rows} == {
        "hi-boundary": "kept",
        "lo-boundary": "floor",
    }
    assert {r["id"]: r["hi_score"] for r in rows}["hi-boundary"] == 0.5
    assert {r["id"]: r["lo_score"] for r in rows}["lo-boundary"] == 0.5


def test_validate_ladder_semantic_rungs_use_pre_inversion_floor():
    entry = {
        "id": "semantic",
        "surface": "heart failure",
        "rungs": ["heart failure", "a cardiovascular disease"],
        "rung": 1,
        "q": "What condition category prompted the cardiology follow-up?",
    }
    q = entry["q"]

    kept, rows = validate_ladder(
        [entry],
        lambda _q: "heart failure",
        lambda _q: "a cardiovascular disease",
        lambda _q: "heart failure",
        lambda _q: "",
        th=0.5,
    )

    assert [e["id"] for e in kept] == ["semantic"]
    assert rows[0]["verdict"] == "kept"
    assert rows[0]["q"] == q
    assert rows[0]["lo_answer"] == ""


def test_locator_lint_drops_cross_span_question():
    assert locator_lint(
        "What condition is managed with daily medication?",
        "hypothyroidism",
        ["Synthroid", "Dr. Lee"],
    )
    assert not locator_lint(
        "What condition did Dr. Lee treat?",
        "hypothyroidism",
        ["Synthroid", "Dr. Lee"],
    )


def test_mc_shuffle_is_deterministic_per_seed_and_varies_across_seeds():
    options = ["continue medication", "stop medication", "refer urgently", "observe"]

    assert mc_shuffle(options, "doc-1:q-1:hi") == mc_shuffle(options, "doc-1:q-1:hi")
    assert mc_shuffle(options, "doc-1:q-1:hi") != mc_shuffle(options, "doc-1:q-1:lo")
    assert options == ["continue medication", "stop medication", "refer urgently", "observe"]


def test_validate_decisions_tags_spans_from_depends_on_canon_substring():
    entries = [
        {
            "id": "d1",
            "q": "What is the appropriate medication decision?",
            "options": ["continue Synthroid", "stop Synthroid", "refer to neurology"],
            "gold": "continue Synthroid",
            "depends_on": ["history of hypothyroidism treated with Synthroid"],
            "detected_spans": [
                {"id": "s-condition", "surface": "hypothyroidism"},
                {"id": "s-drug", "surface": "Synthroid"},
            ],
        },
        {
            "id": "d2",
            "q": "What follow-up interval is supported?",
            "options": ["1 week", "6 months", "no follow-up"],
            "gold": "6 months",
            "depends_on": ["stable control is documented"],
            "detected_spans": [{"id": "s-condition", "surface": "hypothyroidism"}],
        },
        {
            "id": "d3",
            "q": "Which follow-up route is supported?",
            "options": ["routine primary care", "emergency department", "cardiology referral"],
            "gold": "routine primary care",
            "depends_on": ["stable control is documented"],
            "detected_spans": [{"id": "s-condition", "surface": "hypothyroidism"}],
        },
    ]

    hi = {
        entries[0]["q"]: "continue Synthroid",
        entries[1]["q"]: "no follow-up",
        entries[2]["q"]: "routine primary care",
    }
    kept, rows = validate_decisions(
        entries,
        lambda q, _opts: hi[q],
        lambda _q, _opts: None,
    )

    assert [e["id"] for e in kept] == ["d1", "d3"]
    assert kept[0]["span_ids"] == ["s-condition", "s-drug"]
    assert kept[1]["span_ids"] == []
    assert {r["id"]: r["verdict"] for r in rows} == {
        "d1": "kept",
        "d2": "ceiling",
        "d3": "kept",
    }


def test_validated_rung0_cache_entries_prevent_reteaching(monkeypatch, tmp_path):
    docs = [{"id": "doc-1", "text": "Hypothyroidism is treated with Synthroid."}]
    spans = [
        {
            "surface": "hypothyroidism",
            "type": "condition",
            "actions": [{"mode": "level", "fill": "an endocrine condition"}],
        }
    ]
    spans_of = {"doc-1": spans}
    cache_path = tmp_path / "ladder_probes.json"

    class FakeTeacher:
        calls = 0

        def generate(self, _prompt):
            FakeTeacher.calls += 1
            return "not json"

    monkeypatch.setattr(lp, "_teacher", lambda _model, _base_url: FakeTeacher())

    first = lp.ladder_probes_for_docs(
        docs,
        spans_of,
        "clinical",
        workers=1,
        model="fake-teacher",
        cache_path=cache_path,
    )
    assert first == {"doc-1": []}
    assert FakeTeacher.calls == 1

    entries = _with_validated_rung0(
        first["doc-1"],
        spans,
        {
            "hypothyroidism": {
                "surface": "hypothyroidism",
                "question": "What condition is treated with Synthroid?",
            }
        },
        teacher="fake-teacher",
        pv=lp.LADDER_PV,
    )
    cache_path.write_text(json.dumps({"doc-1": entries}, indent=1))

    second = lp.ladder_probes_for_docs(
        docs,
        spans_of,
        "clinical",
        workers=1,
        model="fake-teacher",
        cache_path=cache_path,
    )

    assert FakeTeacher.calls == 1
    assert second["doc-1"][0]["source"] == "probes_validated"
    assert second["doc-1"][0]["teacher"] == "fake-teacher"
    assert second["doc-1"][0]["pv"] == lp.LADDER_PV


def test_ladder_health_row_reports_reader_rejects_tiers_and_decisions():
    row = ladder_health_row(
        docs=4,
        spans=5,
        rung_candidates=10,
        rung_kept=7,
        decisions_kept=6,
    )

    assert row["reader_rung_reject_rate"] == 0.3
    assert row["tiers_per_span_kept"] == 1.4
    assert row["decisions_kept_per_doc"] == 1.5


def test_validated_artifact_meta_contains_reward_pins():
    helper = getattr(build_probes, "validated_artifact", None)
    assert callable(helper)

    artifact = helper(
        {"doc-1": []},
        {"doc-1": []},
        {
            "th": 0.5,
            "corpora": ["clinical"],
            "env_path": "data/ranker_env.json",
            "built_at": "2026-07-10T12:00:00",
        },
    )

    assert artifact["ladder"] == {"doc-1": []}
    assert artifact["decisions"] == {"doc-1": []}
    assert set(artifact["meta"]) >= {
        "teacher",
        "reader",
        "rt_model",
        "th",
        "ladder_pv",
        "decision_pv",
        "corpora",
        "determinism",
        "env_path",
        "built_at",
    }
    assert artifact["meta"]["determinism"] == "workers1"
