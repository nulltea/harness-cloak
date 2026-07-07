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


def test_count_compiler_keeps_model_counts_non_certifying(tmp_path):
    compiled = compile_level_counts(
        {"runtime_type": "profession", "surface": "privacy engineer"},
        [
            {
                "level": "privacy and security software professional",
                "source_family": "model-proposed",
                "proposed_count": 180,
                "selector": "model-domain-cluster:privacy-security-software",
                "count_evidence": "Includes privacy engineering, security engineering, GRC engineering, and software privacy roles.",
                "rationale": "Preserves the privacy/security/software context.",
            }
        ],
        generated_universe_path=tmp_path / "missing.jsonl",
    )

    assert compiled[0]["level_count"] == 180.0
    assert compiled[0]["level_grounding"] == {
        "status": "model-proposed",
        "source_family": "model-proposed",
        "selector": "model-domain-cluster:privacy-security-software",
        "member_set_ref": None,
        "count_evidence": "Includes privacy engineering, security engineering, GRC engineering, and software privacy roles.",
    }


def test_count_compiler_fails_closed_for_model_level_without_count_evidence(tmp_path):
    compiled = compile_level_counts(
        {"runtime_type": "profession", "surface": "privacy engineer"},
        [{"level": "professional worker", "source_family": "model-proposed", "proposed_count": 1000}],
        generated_universe_path=tmp_path / "missing.jsonl",
    )

    assert compiled[0]["level_count"] == 1.0
    assert compiled[0]["level_grounding"]["status"] == "fail-closed"
    assert compiled[0]["level_grounding"]["source_family"] == "model-proposed"


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


def test_gate_fails_closed_for_model_proposals_missing_aliases_and_evidence():
    item = {"item_id": "p3", "runtime_type": "profession", "surface": "privacy engineer"}
    candidates = [
        {
            "level": "architecture and engineering occupation",
            "source_family": "model-proposed",
            "level_count": 1000.0,
            "level_grounding": {"status": "model-proposed", "source_family": "model-proposed"},
            "rationale": "",
        }
    ]

    result = gate_candidates(item, candidates)

    assert result.accepted == []
    assert result.diagnostics[0]["reason"] == "missing_aliases"


def test_gate_fails_closed_for_flat_generic_model_chain():
    item = {
        "item_id": "p4",
        "runtime_type": "profession",
        "surface": "beer cicerone",
        "aliases": ["beer sommelier"],
    }
    candidates = [
        {
            "level": "worker",
            "source_family": "model-proposed",
            "level_count": 1000.0,
            "level_grounding": {
                "status": "model-proposed",
                "source_family": "model-proposed",
                "selector": "model-domain-cluster:generic-worker",
                "count_evidence": "Generic broad worker category.",
            },
            "rationale": "A beer cicerone is a worker.",
        },
        {
            "level": "production worker",
            "source_family": "model-proposed",
            "level_count": 1000.0,
            "level_grounding": {
                "status": "model-proposed",
                "source_family": "model-proposed",
                "selector": "model-domain-cluster:generic-production",
                "count_evidence": "Generic broad production category.",
            },
            "rationale": "A beer cicerone works near beverage production.",
        },
    ]

    result = gate_candidates(item, candidates)

    assert result.accepted == []
    assert {row["reason"] for row in result.diagnostics} == {"flat_model_counts", "weak_semantic_relevance"}


def test_gate_accepts_model_chain_with_aliases_counts_and_domain_evidence():
    item = {
        "item_id": "p5",
        "runtime_type": "profession",
        "surface": "privacy engineer",
        "aliases": ["data protection engineer"],
    }
    candidates = [
        {
            "level": "privacy and security software professional",
            "aliases": ["data protection engineer"],
            "source_family": "model-proposed",
            "level_count": 180.0,
            "level_grounding": {
                "status": "model-proposed",
                "source_family": "model-proposed",
                "selector": "model-domain-cluster:privacy-security-software",
                "count_evidence": "Includes privacy engineering, security engineering, GRC engineering, and software privacy roles.",
            },
            "rationale": "Preserves the privacy/security/software context without naming the exact profession.",
        },
        {
            "level": "software security and compliance professional",
            "aliases": ["data protection engineer"],
            "source_family": "model-proposed",
            "level_count": 420.0,
            "level_grounding": {
                "status": "model-proposed",
                "source_family": "model-proposed",
                "selector": "model-domain-cluster:software-security-compliance",
                "count_evidence": "Broader software security, privacy compliance, and governance roles.",
            },
            "rationale": "Still true for a privacy engineer and broader than the first level.",
        },
    ]

    result = gate_candidates(item, candidates)

    assert [row["level"] for row in result.accepted] == [
        "privacy and security software professional",
        "software security and compliance professional",
    ]
    assert result.diagnostics == []
