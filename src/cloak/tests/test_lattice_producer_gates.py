import json

from cloak.lattice_producer.counts import compile_level_counts
from cloak.lattice_producer.gates import gate_candidates


def test_compile_drops_frozenset_member_set_so_record_is_json_serializable(tmp_path):
    # a deterministic reference candidate carries a frozenset member_set; the compiled record must
    # keep the count + member_set_ref but drop the raw frozenset (not JSON-serializable).
    candidate = {
        "level": "extrinsic cardiomyopathy",
        "source_family": "doid-is-a",
        "selector": "doid.is_a_descendants(DOID:0050700)",
        "member_set": frozenset({"DOID:1", "DOID:2", "DOID:3"}),
        "member_set_ref": "doid:is_a_descendants:DOID:0050700",
    }
    compiled = compile_level_counts(
        {"runtime_type": "health-condition", "surface": "alcoholic cardiomyopathy"},
        [candidate],
        generated_universe_path=tmp_path / "missing.jsonl",
    )
    assert "member_set" not in compiled[0]
    assert compiled[0]["level_count"] == 3.0
    assert compiled[0]["level_grounding"]["status"] == "certifying"
    json.dumps(compiled[0])  # must not raise


def _proposed(path, rt, entries):
    path.write_text(json.dumps({
        "artifact_role": "proposal", "proposal_scope": "producer-processed-only",
        "profiles": {rt: entries},
    }))


def _model_cand(level, count):
    return {
        "level": level, "source_family": "model-proposed", "level_count": count,
        "level_grounding": {"status": "model-proposed", "count_evidence": "e", "selector": "s"},
        "count_evidence": "e", "selector": "s", "rationale": "r",
    }


def test_gate_flags_count_disagreement_on_reused_exact_label(tmp_path):
    out = tmp_path / "proposed.json"
    _proposed(out, "drug", {"x": {"levels": ["analgesic"], "level_counts": {"analgesic": 20}}})
    item = {"runtime_type": "drug", "surface": "ibuprofen", "aliases": ["advil"]}
    # exact reuse of "analgesic" but count 5000 vs recorded 20 -> disagreement
    res = gate_candidates(item, [_model_cand("analgesic", 5000)], proposed_out=str(out))
    assert any(d.get("reason") == "count_disagreement" for d in res.diagnostics)


def test_gate_flags_item_with_single_level(tmp_path):
    out = tmp_path / "proposed.json"
    out.write_text('{"profiles": {"drug": {}}}')
    item = {"runtime_type": "drug", "surface": "ibuprofen", "aliases": ["advil"]}
    # a lone candidate that clears every per-candidate gate (non-anchor level so it dodges the
    # count-agreement check, count above the k-floor) -- it survives to `accepted` alone, so the
    # >=2 floor must divert it. ("analgesic"/30 from the original brief instead trips
    # count_disagreement against the real anchor recorded=150, never reaching accepted.)
    cand = _model_cand("nonsteroidal anti-inflammatory drug", 5000)
    res = gate_candidates(item, [cand], proposed_out=str(out))
    # one accepted level is below the >=2 floor -> the surviving level is diverted to diagnostics
    assert any(d.get("reason") == "too_few_levels" for d in res.diagnostics)
    assert not res.accepted


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


def test_gate_rejects_leaks_and_keeps_subfloor_rungs_when_chain_reaches_floor():
    # eligible=False exempts this fixture from the item-level >=2 chain floor: this test isolates
    # the per-candidate self_leak/type_name_phrase rules and the k-floor semantics. The k-floor is
    # an anonymization-time legality test (anonymity.py), NOT a per-rung drop: a specific sub-floor
    # rung ("medical specialist", k=2) is KEPT as a granular lattice option, because the chain's
    # broadest rung ("healthcare worker", k=120) reaches the floor so the release-time walk has a
    # legal target.
    item = {"item_id": "p1", "runtime_type": "profession", "surface": "cardiologist", "eligible": False}
    candidates = [
        {"level": "cardiologist specialist", "level_count": 1000.0, "level_grounding": {"status": "certifying"}},
        {"level": "a profession", "level_count": 1000.0, "level_grounding": {"status": "certifying"}},
        {"level": "medical specialist", "level_count": 2.0, "level_grounding": {"status": "certifying"}},
        {"level": "healthcare worker", "level_count": 120.0, "level_grounding": {"status": "certifying"}},
    ]

    result = gate_candidates(item, candidates)

    assert [r["level"] for r in result.accepted] == ["medical specialist", "healthcare worker"]
    assert {r["reason"] for r in result.rejected} == {"self_leak", "type_name_phrase"}
    assert not result.diagnostics


def test_gate_diverts_whole_chain_when_broadest_rung_below_floor():
    # every truthful rung is below the k-floor of 100, so no rung can serve as a legal
    # anonymization target -> the whole chain is diverted (chain_below_floor) for a broader tier,
    # rather than persisting an entry the release-time walk could never anonymize safely.
    item = {"item_id": "p1b", "runtime_type": "profession", "surface": "cardiologist", "eligible": False}
    candidates = [
        {"level": "interventional cardiology specialist", "level_count": 8.0, "level_grounding": {"status": "certifying"}},
        {"level": "cardiac care specialist", "level_count": 40.0, "level_grounding": {"status": "certifying"}},
    ]

    result = gate_candidates(item, candidates)

    assert not result.accepted
    assert {r["reason"] for r in result.diagnostics} == {"chain_below_floor"}


def test_gate_exempts_proposal_universe_chain_from_chain_floor():
    # proposal-universe rungs carry provisional counts, so a chain made only of them is NOT diverted
    # by the chain-floor check even when every count is below the floor.
    item = {"item_id": "p1c", "runtime_type": "profession", "surface": "cardiologist", "eligible": False}
    candidates = [
        {"level": "medical specialist", "level_count": 2.0, "level_grounding": {"status": "proposal-universe"}},
        {"level": "healthcare worker", "level_count": 3.0, "level_grounding": {"status": "proposal-universe"}},
    ]

    result = gate_candidates(item, candidates)

    assert [r["level"] for r in result.accepted] == ["medical specialist", "healthcare worker"]
    assert not result.diagnostics


def test_gate_allows_generated_universe_counts_but_marks_them_non_certifying():
    # eligible=False: isolates the proposal-universe per-candidate path; exempt from the >=2 floor.
    item = {"item_id": "p2", "runtime_type": "profession", "surface": "cardiologist", "eligible": False}
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


def _drug_candidate(level, *, reused=None, level_count=5000.0):
    grounding = {
        "status": "model-proposed",
        "source_family": "model-proposed",
        "selector": f"model-domain-cluster:{level}",
        "count_evidence": "Estimated from clinical formulary and pharmacology references.",
    }
    candidate = {
        "level": level,
        "aliases": ["some brand name"],
        "source_family": "model-proposed",
        "level_count": level_count,
        "level_grounding": grounding,
        "rationale": "Truthful generalization for this entry.",
    }
    if reused is not None:
        candidate["reused_canonical_label"] = reused
    return candidate


def _candidate_with_aliases(level, aliases, runtime_type="drug", level_count=5000.0):
    return {
        "level": level,
        "aliases": aliases,
        "source_family": "model-proposed",
        "level_count": level_count,
        "level_grounding": {
            "status": "model-proposed",
            "source_family": "model-proposed",
            "selector": f"model-domain-cluster:{level}",
            "count_evidence": "Estimated from clinical formulary and pharmacology references.",
        },
        "rationale": "Truthful generalization for this entry.",
    }


def test_gate_routes_zero_domain_overlap_chain_to_diagnostics():
    # real example from the reviewed run: "baseball" filed under health-condition with levels
    # ["sport", "game", "human activity"] -- none of which share a token with any seeded
    # health-condition vocabulary label -- passed every other gate and was accepted.
    item = {"item_id": "hc1", "runtime_type": "health-condition", "surface": "baseball", "aliases": []}
    candidates = [
        _candidate_with_aliases("sport", ["athletic activity"], "health-condition"),
        _candidate_with_aliases("game", ["recreational game"], "health-condition"),
        _candidate_with_aliases("human activity", ["general activity"], "health-condition"),
    ]

    result = gate_candidates(item, candidates)

    assert result.accepted == []
    assert all(row["reason"] == "no_domain_overlap" for row in result.diagnostics)


def test_gate_allows_narrow_proposal_when_chain_has_any_domain_overlap():
    # a narrow, legitimate level not itself in the seeded vocabulary must still be accepted as
    # long as SOME level in the same chain touches the domain -- this gate only targets chains
    # with zero domain relevance anywhere, not every level individually.
    item = {"item_id": "hc2", "runtime_type": "health-condition", "surface": "chlamydia", "aliases": []}
    candidates = [
        _candidate_with_aliases(
            "bacterial sexually transmitted infection", ["chlamydial infection"], "health-condition", level_count=200.0
        ),
        _candidate_with_aliases("infectious disease", ["contagious illness"], "health-condition", level_count=1400.0),
    ]

    result = gate_candidates(item, candidates)

    assert [row["level"] for row in result.accepted] == [
        "bacterial sexually transmitted infection",
        "infectious disease",
    ]


def test_gate_routes_generic_filler_aliases_to_diagnostics():
    # real example from the reviewed run: "mcnuggates" (a McDonald's menu item) got aliased as
    # if it were a real drug with templated, contentless filler instead of the model admitting
    # it didn't recognize the surface. Every alias's tokens being a subset of generic filler
    # words is the tell -- a real alias set always includes at least one specific, identifying
    # term (a brand name, an active ingredient, a synonym).
    item = {"item_id": "d4", "runtime_type": "drug", "surface": "mcnuggates", "aliases": []}
    candidates = [
        _candidate_with_aliases(
            "medicinal compound",
            ["medicinal compound", "pharmaceutical preparation", "clinical agent"],
        )
    ]

    result = gate_candidates(item, candidates)

    assert result.accepted == []
    assert result.diagnostics[0]["reason"] == "generic_filler_aliases"


def test_gate_allows_single_generic_sounding_alias() -> None:
    # a lone alias can't be checked against "ALL aliases are filler" meaningfully -- this must
    # not become a blanket ban on any alias containing a common word.
    # eligible=False: isolates the single-generic-alias per-candidate rule; exempt from the >=2 floor.
    item = {"item_id": "d5", "runtime_type": "drug", "surface": "gabapentin", "aliases": [], "eligible": False}
    candidates = [_candidate_with_aliases("nonsteroidal anti-inflammatory drug", ["pharmaceutical agent"])]

    result = gate_candidates(item, candidates)

    assert [row["level"] for row in result.accepted] == ["nonsteroidal anti-inflammatory drug"]


def test_gate_routes_low_model_confidence_surface_to_diagnostics():
    item = {"item_id": "d6", "runtime_type": "drug", "surface": "gabapentin", "aliases": []}
    candidate = _candidate_with_aliases("gaba analog", ["neurontin"])
    candidate["surface_confidence"] = "ambiguous"

    result = gate_candidates(item, [candidate])

    assert result.accepted == []
    assert result.diagnostics[0]["reason"] == "low_confidence_surface"


def test_gate_backstops_short_surface_even_when_model_claims_high_confidence():
    # real example from the reviewed run: "pt" resolved to "pentazocine" (real clinical PT
    # almost always means physical therapy or prothrombin time), reported confidently by the
    # model. The backstop doesn't trust the model's own surface_confidence field for short
    # surfaces; it forces the disambiguation outcome anyway. ("pt" is deliberately chosen so it
    # isn't also a substring of the resolved level, unlike e.g. "cad"/"cadmium" -- this test is
    # about the length backstop specifically, not the separate pre-existing self-leak check.)
    item = {"item_id": "d7", "runtime_type": "drug", "surface": "pt", "aliases": []}
    candidate = _candidate_with_aliases("pentazocine analog", ["central nervous system modulator"])
    candidate["surface_confidence"] = "high"

    result = gate_candidates(item, [candidate])

    assert result.accepted == []
    assert result.diagnostics[0]["reason"] == "low_confidence_surface"


def test_gate_allows_long_surface_with_high_confidence():
    # eligible=False: isolates the long-surface/high-confidence per-candidate path; exempt from floor.
    item = {"item_id": "d9", "runtime_type": "drug", "surface": "gabapentin", "aliases": [], "eligible": False}
    candidate = _candidate_with_aliases("nonsteroidal anti-inflammatory drug", ["neurontin"])
    candidate["surface_confidence"] = "high"

    result = gate_candidates(item, [candidate])

    assert [row["level"] for row in result.accepted] == ["nonsteroidal anti-inflammatory drug"]


def test_gate_routes_unreused_near_duplicate_of_canonical_label_to_diagnostics():
    # "pharmaceutical product" is a near-duplicate paraphrase of the seeded anchor
    # "pharmaceutical compound" -- proposing it as new phrasing without reusing the existing
    # canonical label is exactly the paraphrase-proliferation failure mode this gate exists for.
    item = {"item_id": "d1", "runtime_type": "drug", "surface": "ibuprofen", "aliases": []}
    candidates = [_drug_candidate("pharmaceutical product")]

    result = gate_candidates(item, candidates)

    assert result.accepted == []
    assert result.diagnostics[0]["reason"] == "unreused_near_duplicate_label"
    assert "pharmaceutical compound" in result.diagnostics[0]["near_duplicates"]


def test_gate_accepts_exact_canonical_label_without_reused_flag():
    # eligible=False: isolates the exact-canonical-label per-candidate rule; exempt from the >=2 floor.
    item = {"item_id": "d2", "runtime_type": "drug", "surface": "ibuprofen", "aliases": [], "eligible": False}
    # count matched to the recorded anchor magnitude so the new count-agreement gate doesn't fire;
    # this test is about reuse of the exact label, not count disagreement.
    candidates = [_drug_candidate("pharmaceutical compound", level_count=2_800_000.0)]

    result = gate_candidates(item, candidates)

    assert [row["level"] for row in result.accepted] == ["pharmaceutical compound"]


def test_gate_catches_duplicate_of_a_label_this_run_already_accepted(tmp_path):
    # the real fix for paraphrase proliferation outside the static anchor set: an earlier item
    # in THIS run already had "renal excretion agent" accepted (not a hand-curated anchor, not
    # in any reference file) -- a later item proposing the near-duplicate "renal elimination
    # agent" without reusing it must be caught, exactly as if it duplicated an anchor. ("agent"
    # alone isn't enough to trigger this against the static anchors -- see the sibling test --
    # so this genuinely isolates the dynamic, run-grown vocabulary's contribution.)
    proposed = tmp_path / "run.proposed.json"
    proposed.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_role": "proposal",
                "proposal_scope": "producer-processed-only",
                "profiles": {
                    "drug": {
                        "some diuretic": {
                            "levels": ["renal excretion agent"],
                            "level_counts": {"renal excretion agent": 42.0},
                        }
                    }
                },
            }
        )
    )
    item = {"item_id": "d10", "runtime_type": "drug", "surface": "metoprolol", "aliases": []}
    candidates = [_drug_candidate("renal elimination agent")]

    result = gate_candidates(item, candidates, proposed_out=str(proposed))

    assert result.accepted == []
    assert result.diagnostics[0]["reason"] == "unreused_near_duplicate_label"
    assert "renal excretion agent" in result.diagnostics[0]["near_duplicates"]


def test_gate_without_proposed_out_falls_back_to_static_anchors_only():
    # same candidate label as above, but gate_candidates called the old way (no proposed_out)
    # and with no prior run history -- "renal elimination agent" only shares the generic token
    # "agent" with any static anchor (below the near-duplicate threshold), so it must be
    # accepted cleanly. Proves the dynamic behavior is additive, not silently always-on.
    # eligible=False: isolates the static-anchor fallback per-candidate path; exempt from the >=2 floor.
    item = {"item_id": "d11", "runtime_type": "drug", "surface": "metoprolol", "aliases": [], "eligible": False}
    candidates = [_drug_candidate("renal elimination agent")]

    result = gate_candidates(item, candidates)

    assert [row["level"] for row in result.accepted] == ["renal elimination agent"]


def test_gate_accepts_new_phrasing_explicitly_marked_as_not_reused_but_unrelated():
    # a genuinely novel, unrelated label (no vocabulary overlap at all) must not be blocked --
    # this gate only targets near-duplicates of an existing canonical label, not all new labels.
    # eligible=False: isolates the novel-unrelated-label per-candidate rule; exempt from the >=2 floor.
    item = {"item_id": "d3", "runtime_type": "drug", "surface": "ibuprofen", "aliases": [], "eligible": False}
    candidates = [_drug_candidate("nonsteroidal anti-inflammatory drug", reused=False)]

    result = gate_candidates(item, candidates)

    assert [row["level"] for row in result.accepted] == ["nonsteroidal anti-inflammatory drug"]


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
