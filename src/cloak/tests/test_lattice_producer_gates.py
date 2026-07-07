from cloak.lattice_producer.counts import compile_level_counts
from cloak.lattice_producer.gates import gate_candidates


def test_count_compiler_counts_generated_universe_as_proposal_only(tmp_path):
    generated = tmp_path / "generated_universe.jsonl"
    generated.write_text(
        "\n".join(
            [
                '{"runtime_type":"profession","canonical_value":"cardiologist","proposed_levels":["medical specialist","healthcare worker"]}',
                '{"runtime_type":"profession","canonical_value":"surgeon","proposed_levels":["medical specialist","healthcare worker"]}',
                '{"runtime_type":"profession","canonical_value":"teacher","proposed_levels":["education worker","healthcare worker"]}',
            ]
        )
        + "\n"
    )
    item = {"runtime_type": "profession", "entry_origin": "generated-universe"}
    candidates = [{"level": "healthcare worker"}, {"level": "medical specialist"}]

    compiled = compile_level_counts(item, candidates, generated_universe_path=generated)

    assert [c["level"] for c in compiled] == ["medical specialist", "healthcare worker"]
    assert compiled[0]["level_count"] == 2.0
    assert compiled[0]["level_grounding"]["status"] == "proposal-universe"
    assert compiled[1]["level_count"] == 3.0


def test_count_compiler_fails_closed_without_source_grounding(tmp_path):
    compiled = compile_level_counts(
        {"runtime_type": "ORG", "surface": "Acme"},
        [{"level": "a commercial establishment"}],
        generated_universe_path=tmp_path / "missing.jsonl",
    )

    assert compiled[0]["level_count"] == 1.0
    assert compiled[0]["level_grounding"]["status"] == "fail-closed"


def test_gate_rejects_leaks_and_routes_below_floor_to_diagnostics():
    item = {"item_id": "p1", "runtime_type": "profession", "surface": "cardiologist"}
    candidates = [
        {"level": "cardiologist specialist", "level_count": 1000.0, "level_grounding": {"status": "certifying"}},
        {"level": "a profession", "level_count": 1000.0, "level_grounding": {"status": "certifying"}},
        {"level": "medical specialist", "level_count": 2.0, "level_grounding": {"status": "certifying"}},
        {"level": "healthcare worker", "level_count": 120.0, "level_grounding": {"status": "certifying"}},
    ]

    result = gate_candidates(item, candidates)

    assert [r["level"] for r in result.accepted] == ["healthcare worker"]
    assert {r["reason"] for r in result.rejected} == {"self_leak", "type_name_phrase"}
    assert result.diagnostics[0]["reason"] == "below_floor"


def test_gate_allows_generated_universe_counts_but_marks_them_non_certifying():
    item = {"item_id": "p2", "runtime_type": "profession", "surface": "cardiologist"}
    candidates = [
        {
            "level": "medical specialist",
            "level_count": 2.0,
            "level_grounding": {"status": "proposal-universe"},
        }
    ]

    result = gate_candidates(item, candidates)

    assert result.accepted[0]["level"] == "medical specialist"
    assert result.accepted[0]["level_grounding"]["status"] == "proposal-universe"
