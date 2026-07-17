import json

from cloak.train.qa_audit import (
    build_legacy_environment_audit,
    build_qa_audit,
    write_audit_sidecars,
)


def _arms():
    return {
        "_meta": {"detector": {"model": "test-detector"}},
        "aci": {
            "aci/D1": {
                "tau_walk": ["doc", [
                    {
                        "surface": "diabetes mellitus", "type": "health-condition",
                        "start": 0, "end": 17, "action": "generalize",
                        "replacement": "endocrine condition",
                        "lattice": ["endocrine condition", "<HEALTH_CONDITION_1>"],
                        "match": {"entry": "diabetes mellitus"},
                        "profile_match": {"outcome": "exact", "reason": "exact_entry"},
                    },
                    {
                        "surface": "diabetes", "type": "health-condition",
                        "start": 25, "end": 33, "action": "generalize",
                        "replacement": "glucose metabolism disease",
                        "lattice": ["glucose metabolism disease", "<HEALTH_CONDITION_1>"],
                        "match": {"entry": "diabetes"},
                        "profile_match": {"outcome": "semantic", "reason": "semantic_certified"},
                    },
                    {
                        "surface": "physical therapy", "type": "medical-procedure",
                        "start": 40, "end": 56, "action": "keep", "uncontrolled": True,
                        "replacement": "physical therapy", "lattice": ["<MEDICAL_PROCEDURE_1>"],
                        "profile_match": {"outcome": "abstained", "reason": "retrieval_empty"},
                    },
                ]],
                "detector_diagnostics": {
                    "accepted": [
                        {"text": "diabetes mellitus", "type": "health-condition", "start": 0, "end": 17},
                        {"text": "diabetes", "type": "health-condition", "start": 25, "end": 33},
                        {"text": "physical therapy", "type": "medical-procedure", "start": 40, "end": 56},
                    ],
                    "post_detection_rejections": [
                        {"surface": "patient", "proposed_runtime_type": "demographic-other",
                         "reason": "qa_v2_forbidden_demographic_other"},
                    ],
                },
            },
        },
    }


def test_environment_audit_reports_unprofiled_conflicts_and_degenerate_menu():
    audit = build_legacy_environment_audit(_arms())
    codes = [event["code"] for event in audit["events"]]

    assert "unprofiled_profile_backed_span" in codes
    assert "semantic_profile_match" in codes
    assert "coarse_or_degenerate_action_menu" in codes
    assert "controlled_subspan_profile_conflict" in codes
    assert "post_detection_rejection" in codes
    assert audit["audit_hash"].startswith("sha256:")
    assert build_legacy_environment_audit(_arms())["audit_hash"] == audit["audit_hash"]


def test_environment_audit_does_not_misclassify_admission_rejects_or_direct_rows():
    arms = _arms()
    document = arms["aci"]["aci/D1"]
    document["tau_walk"][1].append({
        "surface": "female", "type": "gender", "start": 60, "end": 66,
        "action": "placeholder", "replacement": "<GENDER_1>", "lattice": [],
    })
    document["detector_diagnostics"]["accepted"].append({
        "text": "acute exacerbation", "type": "health-condition", "start": 70, "end": 88,
    })
    document["detector_diagnostics"]["post_detection_rejections"].append({
        "surface": "acute exacerbation", "proposed_runtime_type": "health-condition",
        "start": 70, "end": 88, "reason": "qa_v2_low_confidence_health_condition",
    })

    audit = build_legacy_environment_audit(arms)
    unprofiled_surfaces = {
        event["entity_refs"].get("surface") for event in audit["events"]
        if event["code"] == "unprofiled_profile_backed_span"
    }
    coarse_surfaces = {
        event["entity_refs"].get("surface") for event in audit["events"]
        if event["code"] == "coarse_or_degenerate_action_menu"
    }
    assert "acute exacerbation" not in unprofiled_surfaces
    assert "female" not in coarse_surfaces


def test_environment_audit_scores_unprofiled_span_for_repair_routing():
    document = "Physical therapy was recommended."
    start = document.index("Physical therapy")
    arms = {
        "aci": {
            "aci/D1": {
                "tau_walk": [document, [{
                    "surface": "Physical therapy", "type": "medical-procedure",
                    "start": start, "end": start + len("Physical therapy"),
                    "action": "keep", "uncontrolled": True,
                    "lattice": ["<MEDICAL_PROCEDURE_1>"],
                    "profile_match": {"outcome": "abstained", "reason": "no_certified_match"},
                }]],
                "detector_diagnostics": {"accepted": [{
                    "text": "Physical therapy", "type": "medical-procedure",
                    "start": start, "end": start + len("Physical therapy"),
                }], "post_detection_rejections": []},
            },
        },
    }

    audit = build_legacy_environment_audit(
        arms,
        source_documents={"aci/D1": document},
        self_type_nli_batch_fn=lambda jobs: [[(candidates[0], 0.8)] for _surface, _context, candidates in jobs],
        semantic_embed_fn=lambda texts: [[1.0] for _ in texts],
        pair_nli_batch_fn=lambda jobs: [],
    )

    diagnostic = next(event for event in audit["events"]
                      if event["code"] == "unprofiled_self_type_diagnostic")
    assert diagnostic["evidence"]["prep_filtered"] is False
    assert diagnostic["evidence"]["source_excerpt"] == document


def test_environment_audit_joins_post_rejection_by_raw_detector_label():
    arms = _arms()
    document = arms["aci"]["aci/D1"]
    document["detector_diagnostics"]["accepted"].append({
        "text": "allergies", "type": "health-condition", "start": 70, "end": 79,
    })
    document["detector_diagnostics"]["post_detection_rejections"].append({
        "surface": "allergies", "raw_label": "condition", "start": 70, "end": 79,
        "reason": "qa_v2_low_confidence_health_condition",
    })

    audit = build_legacy_environment_audit(arms)
    assert "allergies" not in {
        event["entity_refs"].get("surface") for event in audit["events"]
        if event["code"] == "detector_to_walk_drop"
    }
    rejection = next(event for event in audit["events"] if event["code"] == "post_detection_rejection"
                     and event["entity_refs"].get("surface") == "allergies")
    assert rejection["entity_refs"]["runtime_type"] == "health-condition"


def test_environment_audit_distinguishes_unattributed_walk_drops_from_match_abstentions():
    arms = _arms()
    arms["aci"]["aci/D1"]["detector_diagnostics"]["accepted"].append({
        "text": "unfrozen condition", "type": "health-condition", "start": 70, "end": 88,
    })

    audit = build_legacy_environment_audit(arms)
    drop = next(event for event in audit["events"] if event["code"] == "detector_to_walk_drop")
    assert drop["entity_refs"]["surface"] == "unfrozen condition"
    assert "unfrozen condition" not in {
        event["entity_refs"].get("surface") for event in audit["events"]
        if event["code"] == "unprofiled_profile_backed_span"
    }


def test_legacy_environment_audit_logs_cross_profile_ladders_and_self_type_scores(tmp_path):
    document = "The patient has type 2 diabetes. The diabetes is monitored."
    specific = "type 2 diabetes"
    general = "diabetes"
    specific_start = document.index(specific)
    general_start = document.rindex(general)
    arms = {
        "aci": {
            "aci/D1": {
                "tau_walk": [document, [
                    {
                        "surface": specific, "type": "health-condition",
                        "start": specific_start, "end": specific_start + len(specific),
                        "action": "placeholder", "replacement": "<HEALTH_CONDITION_1>",
                        "lattice": ["diabetes mellitus", "medical condition"],
                        "match": {"entry": "type 2 diabetes"},
                        "profile_match": {"outcome": "exact", "reason": "exact_entry"},
                    },
                    {
                        "surface": general, "type": "health-condition",
                        "start": general_start, "end": general_start + len(general),
                        "action": "placeholder", "replacement": "<HEALTH_CONDITION_2>",
                        "lattice": ["glucose metabolism disease", "medical condition"],
                        "match": {"entry": "diabetes"},
                        "profile_match": {"outcome": "exact", "reason": "exact_entry"},
                    },
                ]],
                "detector_diagnostics": {"accepted": [], "post_detection_rejections": []},
            },
        },
    }
    profiles = {
        "schema_version": 1, "created": "test", "sources": {}, "profiles": {
            "health-condition": {
                "type 2 diabetes": {
                    "aliases": [], "count": 10.0,
                    "levels": ["diabetes mellitus", "medical condition"],
                    "level_counts": {"diabetes mellitus": 1000.0, "medical condition": 100000.0},
                },
                "diabetes": {
                    "aliases": [], "count": 20.0,
                    "levels": ["glucose metabolism disease", "medical condition"],
                    "level_counts": {"glucose metabolism disease": 500.0, "medical condition": 100000.0},
                },
            },
        },
    }
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(json.dumps(profiles))

    def fake_embed(texts):
        assert texts == [specific, general]
        return [[1.0, 0.0], [0.9, 0.1]]

    def pair_nli(jobs):
        return [[(candidates[0], 0.9)] if surface == specific else []
                for surface, _context, candidates in jobs]

    def self_type_nli(jobs):
        return [[(candidates[0], 0.2)] for _surface, _context, candidates in jobs]

    audit = build_legacy_environment_audit(
        arms,
        source_documents={"aci/D1": document},
        profiles_path=profile_path,
        semantic_embed_fn=fake_embed,
        pair_nli_batch_fn=pair_nli,
        self_type_nli_batch_fn=self_type_nli,
    )
    ladder = next(observation for observation in audit["observations"]
                  if observation["code"] == "controlled_ladder_metrics"
                  and observation["entity_refs"]["surface"] == specific)
    assert ladder["evidence"]["first_level_log10_jump"] == 2.0
    assert ladder["evidence"]["count_monotone"] is True
    assert "controlled_ladder_issue" not in {
        event["code"] for event in audit["events"]
    }
    candidate = next(event for event in audit["events"]
                     if event["code"] == "cross_profile_coreference_candidate")
    assert candidate["evidence"]["candidate_kind"] == "specific_to_general_candidate"
    assert candidate["entity_refs"]["left_surface"] == specific
    self_type = [event for event in audit["events"]
                 if event["code"] == "controlled_self_type_diagnostic"]
    assert len(self_type) == 2
    assert {event["evidence"]["self_type_score"] for event in self_type} == {0.2}


def test_ladder_metrics_become_review_events_only_for_structural_failures(tmp_path):
    profiles = {
        "schema_version": 1, "created": "test", "sources": {}, "profiles": {
            "health-condition": {
                "bad ladder": {
                    "aliases": [], "count": 2.0,
                    "levels": ["narrow condition", "broad condition"],
                    "level_counts": {"narrow condition": 100.0, "broad condition": 10.0},
                },
            },
        },
    }
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(json.dumps(profiles))
    arms = {
        "aci": {
            "aci/D1": {
                "tau_walk": ["Bad ladder.", [{
                    "surface": "Bad ladder", "type": "health-condition", "start": 0, "end": 10,
                    "lattice": ["narrow condition", "broad condition"],
                    "match": {"entry": "bad ladder"},
                    "profile_match": {"outcome": "exact", "reason": "exact_entry"},
                }]],
                "detector_diagnostics": {"accepted": [], "post_detection_rejections": []},
            },
        },
    }

    audit = build_legacy_environment_audit(arms, profiles_path=profile_path)
    assert len(audit["observations"]) == 1
    issue = next(event for event in audit["events"] if event["code"] == "controlled_ladder_issue")
    assert issue["evidence"]["issues"] == ["nonmonotone_level_counts"]


def test_qa_audit_normalizes_rejections_and_soft_cross_clause_risk(tmp_path):
    environment_audit = build_legacy_environment_audit(_arms())
    artifact = {
        "review_flags": {"aci/D1": [{
            "code": "lattice_level_suspect", "stage": "gate", "fix_class": "data_lattice",
            "severity": "warn", "detail": {"surface": "diabetes"},
        }]},
        "rejections": {"records": [{
            "doc_id": "aci/D1", "reason": "three_point_gate_failed", "relation": "tests_for",
            "question": "For what condition was x-ray ordered?", "evidence": {"validation": {"scores": {}}},
        }]},
        "assertions": {"a1": {
            "doc_id": "aci/D1", "relation": "tests_for", "question": "For what condition?",
            "evidence": {"anchor_diagnostics": [{
                "kind": "soft_cross_clause_cap_exceeded", "distance": 12,
            }]},
        }},
        "relation_candidate_accounting": {"aci/D1": [{"state": "ledger_inconsistent"}]},
        "relation_gleaning": {"aci/D1": {"triggered": True, "returned_count": 0}},
        "relation_coverage": {"aci/D1": {
            "opportunity_count": 2,
            "unresolved_targets": [
                {"kind": "missed", "relation": "tests_for", "fact_key_hash": "sha256:missed"},
                {"kind": "fixable", "relation": "procedure_for", "reason": "invalid_evidence",
                 "fact_key_hash": "sha256:repair"},
            ],
        }},
    }
    audit = build_qa_audit(artifact, environment_audit=environment_audit)
    codes = {event["code"] for event in audit["events"]}
    assert {"lattice_level_suspect", "relation_rejected:three_point_gate_failed",
            "soft_cross_clause_cap_exceeded", "teacher_candidate:ledger_inconsistent",
            "gleaning_returned_no_relations", "teacher_missed_structural_opportunity",
            "repairable_relation_still_rejected"} <= codes
    assert audit["metadata"]["environment_audit_hash"] == environment_audit["audit_hash"]

    output = tmp_path / "audit"
    json_path, jsonl_path, markdown_path = write_audit_sidecars(audit, output)
    dotted_output = tmp_path / "artifact.arms.environment-audit"
    dotted_json_path, dotted_jsonl_path, dotted_markdown_path = write_audit_sidecars(
        audit, dotted_output,
    )
    assert json.loads(json_path.read_text())["audit_hash"] == audit["audit_hash"]
    jsonl_records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert len(jsonl_records) == len(audit["events"]) + len(audit["observations"])
    assert {record["record_kind"] for record in jsonl_records} == {
        "event", *( ["observation"] if audit["observations"] else [] )
    }
    assert "soft_cross_clause_cap_exceeded" in markdown_path.read_text()
    assert dotted_json_path.name == "artifact.arms.environment-audit.json"
    assert dotted_jsonl_path.name == "artifact.arms.environment-audit.jsonl"
    assert dotted_markdown_path.name == "artifact.arms.environment-audit.md"
