import json

import render_qa_build_report as report


def test_render_report_shows_text_inventory_and_relation_qa(tmp_path, monkeypatch):
    artifact_path = tmp_path / "qa.json"
    arms_path = tmp_path / "arms.json"
    out_p_path = tmp_path / "out_p.txt"
    artifact_path.write_text(json.dumps({
        "documents": {"aci/D2N002": {
            "occurrences": [
                {"surface": "Andrew", "runtime_type": "PERSON"},
                {"surface": "Andrew", "runtime_type": "PERSON"},
                {"surface": "Synthroid", "runtime_type": "drug"},
            ],
        }},
        "assertions": {
            "schema": {
                "doc_id": "aci/D2N002", "subtype": "structure",
                "scoring_contract": {"kind": "required_sections", "sections": ["PLAN"]},
            },
            "relation": {
                "doc_id": "aci/D2N002", "subtype": "contextual_relation",
                "relation": "prescribed_with", "question": "Which medication class is used?",
                "accepted_values": ["hormone replacement therapy"],
                "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
                "evidence": {"arguments": [{"kind": "linked", "surface": "Synthroid"}]},
            },
        },
        "relation_candidate_accounting": {"aci/D2N002": [
            {"candidate_id": "o1", "state": "emitted", "reason": "Explicit evidence."},
        ]},
    }))
    arms_path.write_text(json.dumps({
        "clinical": {"aci/D2N002": {"tau_walk": ["anonymized note", []]}},
    }))
    out_p_path.write_text("remote model output")
    monkeypatch.setattr(report, "load_task_docs", lambda corpus: [{
        "id": "aci/D2N002", "text": "original note", "gold_ref": "reference output",
    }])

    rendered = report.render_report(
        artifact_path, arms_path, corpus="clinical", doc_id="aci/D2N002",
        out_p_path=out_p_path,
    )

    assert "## doc_orig\n\n```text\noriginal note\n```" in rendered
    assert "## out_hi (reference output)\n\n```text\nreference output\n```" in rendered
    assert "## doc_p (tau_walk)\n\n```text\nanonymized note\n```" in rendered
    assert "## out_p (remote model output)\n\n```text\nremote model output\n```" in rendered
    assert "### PERSON (2)" in rendered
    assert "### Structural schemas (deterministic)" in rendered
    assert "prescribed_with" in rendered
    assert "- Status: kept" in rendered
    assert "- Reason: accepted" in rendered
    assert "hormone replacement therapy" in rendered
    assert "### Candidate accounting" in rendered
    assert "Explicit evidence." in rendered


def test_render_report_shows_rejected_teacher_attempt_and_allows_missing_out_p(tmp_path, monkeypatch):
    artifact_path = tmp_path / "qa.json"
    arms_path = tmp_path / "arms.json"
    artifact_path.write_text(json.dumps({
        "documents": {"d1": {"occurrences": []}},
        "assertions": {},
        "relation_generation": {"d1": [{
            "proposal_index": 0,
            "relation": "prescribed_with",
            "arguments": [{"role": "subject", "kind": "linked", "occurrence_id": "o1"}],
            "question": "Which medication category was selected?",
            "accepted_answers": ["opioid analgesic"],
            "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
            "evidence_window_id": "window-1",
            "status": "rejected",
            "reason": "protected_answer",
        }]},
        "relation_candidate_accounting": {"d1": []},
    }))
    arms_path.write_text(json.dumps({"clinical": {"d1": {"tau_walk": ["anonymized", []]}}}))
    monkeypatch.setattr(report, "load_task_docs", lambda corpus: [{
        "id": "d1", "text": "original", "gold_ref": "reference",
    }])

    rendered = report.render_report(artifact_path, arms_path, corpus="clinical", doc_id="d1")

    assert "## out_p (remote model output)\n\n```text\nNot supplied.\n```" in rendered
    assert "- Status: rejected" in rendered
    assert "- Reason: protected_answer" in rendered
    assert "opioid analgesic" in rendered
