import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from build_probes import ladder_health_row  # noqa: E402
from cloak.train.ladder_probes import (  # noqa: E402
    locator_lint,
    mc_shuffle,
    validate_decisions,
    validate_ladder,
)


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

    kept, rows = validate_ladder(entries, hi.get, lo.get, th=0.5)

    assert [e["id"] for e in kept] == ["keep"]
    assert {r["id"]: r["verdict"] for r in rows} == {
        "keep": "kept",
        "ceiling-fail": "ceiling",
        "floor-pass": "floor",
    }


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
    ]

    kept, rows = validate_decisions(
        entries,
        lambda _q, _opts: "continue Synthroid",
        lambda _q, _opts: None,
    )

    assert [e["id"] for e in kept] == ["d1"]
    assert kept[0]["span_ids"] == ["s-condition", "s-drug"]
    assert {r["id"]: r["verdict"] for r in rows} == {"d1": "kept", "d2": "ceiling"}


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
