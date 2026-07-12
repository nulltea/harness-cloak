import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import build_probes  # noqa: E402
from build_probes import (  # noqa: E402
    _negated_or_screening,
    _with_validated_rung0,
    ladder_health_row,
)
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


def test_entail_score_is_binary_no_shared_head_noun_credit():
    from cloak.train.ladder_probes import entail_score

    rungs = ["hypertension", "artery disease", "vascular disease"]
    # sibling category sharing only the head noun 'disease' must score 0, not F1 0.5
    assert entail_score("vascular disease", rungs, 1) == 0.0
    # finer-than-gold answer containing the gold's content tokens is a full hit
    assert entail_score("coronary artery disease", rungs, 1) == 1.0
    # article difference must not break containment
    assert entail_score("cardiovascular disease",
                        ["heart failure", "a cardiovascular disease"], 1) == 1.0
    # acronym path survives (fact_score exact hit)
    assert entail_score("CHF", ["congestive heart failure"], 0) == 1.0


def test_validate_ladder_partial_token_answers_reject_not_partial_score():
    entries = [
        {
            "id": "partial-hi",
            "surface": "alpha beta gamma",
            "rungs": ["alpha beta gamma"],
            "rung": 0,
            "q": "What is the first boundary fact?",
        },
        {
            "id": "partial-lo",
            "surface": "delta epsilon zeta",
            "rungs": ["delta epsilon zeta"],
            "rung": 0,
            "q": "What is the second boundary fact?",
        },
    ]
    hi = {
        entries[0]["q"]: "alpha",                    # one shared token -> 0.0 -> ceiling
        entries[1]["q"]: "delta epsilon zeta noted",  # containment -> 1.0
    }
    lo = {
        entries[0]["q"]: "",
        entries[1]["q"]: "delta",                    # one shared token -> 0.0 -> NOT floor
    }

    kept, rows = validate_ladder(entries, hi.get, hi.get, lo.get, lo.get, th=0.5)

    assert [e["id"] for e in kept] == ["partial-lo"]
    assert {r["id"]: r["verdict"] for r in rows} == {
        "partial-hi": "ceiling",
        "partial-lo": "kept",
    }
    assert all(r["hi_score"] in (0.0, 1.0) and r["lo_score"] in (0.0, 1.0) for r in rows)


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


def test_validate_ladder_semantic_rungs_use_pre_inversion_ceiling():
    # Guards the CEILING side of F2: rung >= 1 must score against out_p, not out_final.
    # reader_hi_final returns a non-entailing answer; a regression routing the ceiling check
    # to out_final would score 0 and reject the rung ("ceiling"), so keeping it proves out_p
    # was read at the ceiling anchor.
    entry = {
        "id": "semantic",
        "surface": "heart failure",
        "rungs": ["heart failure", "a cardiovascular disease"],
        "rung": 1,
        "q": "What condition category prompted the cardiology follow-up?",
    }

    kept, rows = validate_ladder(
        [entry],
        lambda _q: "",                       # hi_final: would fail if wrongly used for rung 1
        lambda _q: "a cardiovascular disease",  # hi_p: the correct semantic-channel anchor
        lambda _q: "heart failure",
        lambda _q: "",
        th=0.5,
    )

    assert [e["id"] for e in kept] == ["semantic"]
    assert rows[0]["verdict"] == "kept"
    assert rows[0]["hi_answer"] == "a cardiovascular disease"


def test_validate_ladder_accepts_canonical_alias_at_rung0():
    # rung 0 gold is the detected surface; the note wrote a synonym alias (HTN). The alias
    # acceptance set keeps the rung; without it the surface-exact score would ceiling-reject.
    entry = {
        "id": "alias",
        "surface": "hypertension",
        "rungs": ["hypertension", "artery disease"],
        "aliases": ["htn", "high blood pressure"],
        "rung": 0,
        "q": "What condition is listed as an active problem?",
    }
    kept, rows = validate_ladder(
        [entry],
        lambda _q: "HTN",   # ceiling out_final wrote the abbreviation
        lambda _q: "HTN",
        lambda _q: "",      # floor: hidden
        lambda _q: "",
        th=0.5,
    )
    assert [e["id"] for e in kept] == ["alias"]
    assert rows[0]["hi_score"] == 1.0

    # without the alias set, the same answer misses -> ceiling reject
    kept2, _ = validate_ladder(
        [{**entry, "aliases": []}],
        lambda _q: "HTN", lambda _q: "HTN", lambda _q: "", lambda _q: "", th=0.5,
    )
    assert kept2 == []


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

    assert [e["id"] for e in kept] == ["d1"]
    assert kept[0]["span_ids"] == ["s-condition", "s-drug"]
    assert {r["id"]: r["verdict"] for r in rows} == {
        "d1": "kept",
        "d2": "unlinked",
        "d3": "unlinked",
    }
    assert rows[1]["hi_pick"] is None and rows[1]["lo_pick"] is None


def test_validate_decisions_hi_and_lo_read_the_same_option_order():
    # hi/lo shuffles differed by seed suffix, so keep/floor verdicts carried option-order
    # noise on the positional-bias-prone small reader (measured: the one kept decision of
    # the 2026-07-12 super sweep survived only via a floor mis-pick under a different order)
    seen = {}
    entry = {
        "id": "d1",
        "q": "Which route is supported?",
        "options": ["primary care", "endocrinology", "cardiology", "neurology"],
        "gold": "endocrinology",
        "depends_on": ["hypothyroidism"],
        "detected_spans": [{"id": "s0", "surface": "hypothyroidism"}],
    }

    def hi(q, options):
        seen["hi"] = list(options)
        return "endocrinology"

    def lo(q, options):
        seen["lo"] = list(options)
        return "primary care"

    kept, rows = validate_decisions([entry], hi, lo)

    assert seen["hi"] == seen["lo"]
    assert rows[0]["verdict"] == "kept"


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


def test_ladder_probes_scopes_cache_to_current_spans_and_rungs(monkeypatch, tmp_path):
    # Regression: the return/reuse must be scoped to THIS run's detected spans + current lattice,
    # not to (teacher, pv) alone. A prior run left two contaminating entries in the cache:
    #  - 'dragon' (a surface not detected this run, env-style rungs) -> must NOT be returned;
    #  - 'hypertension' with STALE rungs (lattice changed since) -> must be re-generated, and the
    #    stale-rung entry must NOT be returned.
    docs = [{"id": "d1", "text": "Patient with hypertension."}]
    span = {"surface": "hypertension", "type": "health-condition"}
    spans_of = {"d1": [span]}
    monkeypatch.setattr(lp, "span_levels",
                        lambda s: ["artery disease"] if s["surface"] == "hypertension" else [])

    cache_path = tmp_path / "ladder_probes.json"
    cache_path.write_text(json.dumps({"d1": [
        {"surface": "dragon", "rung": 0, "q": "q?", "a": "dragon",
         "rungs": ["dragon", "a mythical monster"], "teacher": "fake", "pv": lp.LADDER_PV},
        {"surface": "hypertension", "rung": 1, "q": "old?", "a": "a cardiovascular disease",
         "rungs": ["hypertension", "a cardiovascular disease"], "teacher": "fake",
         "pv": lp.LADDER_PV},
    ]}))

    class FakeTeacher:
        def generate(self, _prompt):
            return ('[{"rung": 0, "q": "What condition is present?", "a": "hypertension"},'
                    ' {"rung": 1, "q": "What cardiovascular category?", "a": "artery disease"}]')

    monkeypatch.setattr(lp, "_teacher", lambda _m, _b: FakeTeacher())

    out = lp.ladder_probes_for_docs(docs, spans_of, "clinical", workers=1,
                                    model="fake", cache_path=cache_path)
    returned = out["d1"]
    assert all(e["surface"] != "dragon" for e in returned)                 # stale surface gone
    hyp = [e for e in returned if e["surface"] == "hypertension"]
    assert hyp and all(e["rungs"] == ["hypertension", "artery disease"] for e in hyp)  # fresh rungs
    assert all(e["rungs"] != ["hypertension", "a cardiovascular disease"] for e in hyp)  # not stale


def test_negated_or_screening_drops_denials_not_documented_conditions():
    # screening question + explicit denial -> drop (fact only ruled out, not documented)
    assert _negated_or_screening("Have you had any fever or chills, cough, congestion?")
    assert _negated_or_screening("Patient denies chest pain.")
    assert _negated_or_screening("Exam was negative for lymphadenopathy.")
    # a documented condition with 'no changes' MUST be kept (patient HAS it) — the over-drop trap
    assert not _negated_or_screening(
        "No changes or concerns were reported regarding hypertension.")
    assert not _negated_or_screening(
        "Doctor Kumar followed up regarding your hypertension, osteoarthritis, and kidney stones.")


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


def test_lint_decision_rejects_questions_naming_a_lattice_fact():
    from cloak.train.ladder_probes import lint_decision

    surfaces = ["mammogram", "congestive heart failure"]
    # world-knowledge trivia shape: names the fact, asks a generic property of it
    assert not lint_decision("Which body system does a mammogram primarily evaluate?",
                             surfaces)
    assert not lint_decision(
        "Which specialist should manage the congestive heart failure noted in the plan?",
        surfaces)
    # circumstance-grounded question that names no fact passes
    assert lint_decision(
        "Which specialist should follow up the condition managed with daily medication?",
        surfaces)
    assert lint_decision("Which route is supported?", [])


def test_decision_probes_lint_drops_fact_naming_questions(monkeypatch, tmp_path):
    docs = [{"id": "d1", "text": "Patient has hypothyroidism, treated with Synthroid."}]
    reply = json.dumps({"decisions": [
        {"q": "Which body system does hypothyroidism affect?",
         "options": ["Endocrine", "Cardiac", "Renal"], "gold": "Endocrine",
         "depends_on": ["hypothyroidism"]},
        {"q": "Which specialist should follow up the condition managed with daily medication?",
         "options": ["Endocrinologist", "Cardiologist", "Nephrologist"],
         "gold": "Endocrinologist", "depends_on": ["hypothyroidism"]},
    ]})

    class FakeTeacher:
        def generate(self, _prompt):
            return reply

    monkeypatch.setattr(lp, "_teacher", lambda _m, _b: FakeTeacher())

    out = lp.decision_probes_for_docs(
        docs, {"d1": "CEILING NOTE"}, "clinical", workers=1, model="fake",
        cache_path=tmp_path / "decision_probes.json",
        lattice_surfaces_of={"d1": ["hypothyroidism", "Synthroid"]},
    )

    kept_qs = [e["q"] for e in out["d1"]]
    assert kept_qs == [
        "Which specialist should follow up the condition managed with daily medication?"
    ]


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


def test_ladder_generation_keeps_only_rung0_and_rung1(monkeypatch, tmp_path):
    # decided resolution for nested lattices: ONE semantic question pinned at rung 1;
    # teacher replies at coarser rungs are rejected as bad_rung, and the prompt no longer
    # shows the coarser rungs at all (no question can select among nested levels anyway)
    docs = [{"id": "d1", "text": "Patient has hypertension."}]
    span = {"surface": "hypertension", "type": "health-condition"}
    monkeypatch.setattr(
        lp, "span_levels",
        lambda s: ["artery disease", "vascular disease", "a physical condition"])

    prompts = []

    class FakeTeacher:
        def generate(self, prompt):
            prompts.append(prompt)
            return json.dumps({"probes": [
                {"rung": 0, "q": "What condition needs daily monitoring?"},
                {"rung": 1, "q": "What category of condition is being managed?"},
                {"rung": 2, "q": "What broader kind of condition is present?"},
            ]})

    monkeypatch.setattr(lp, "_teacher", lambda _m, _b: FakeTeacher())

    rejects = []
    out = lp.ladder_probes_for_docs(
        docs, {"d1": [span]}, "clinical", workers=1, model="fake",
        cache_path=tmp_path / "ladder_probes.json", reject_sink=rejects)

    assert sorted(e["rung"] for e in out["d1"]) == [0, 1]
    assert [r["gate"] for r in rejects] == ["bad_rung"]
    assert rejects[0]["rung"] == 2
    # full ladder still recorded on the entries (acceptance sets need it) ...
    assert all(e["rungs"] == ["hypertension", "artery disease", "vascular disease",
                              "a physical condition"] for e in out["d1"])
    # ... but the teacher only ever sees rung 0 and rung 1
    assert "artery disease" in prompts[0]
    assert "vascular disease" not in prompts[0]
    assert "a physical condition" not in prompts[0]
