import json
from pathlib import Path

from cloak.lattice_producer.graph import (
    QWEN36_ESCALATION_MODEL,
    build_graph,
    compile_level_counts_node,
    deterministic_lookup,
    gate_candidates_node,
    persist_proposed_artifact_node,
    propose_with_llama_swap_node,
    requeue_rejected_item,
    route_selected,
    route_after_gate,
    run_producer,
)
from cloak.lattice_producer.state import make_initial_state, thread_id_for_run


def _profiles(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created": "2026-07-07",
                "sources": {},
                "profiles": {},
            }
        )
    )


def _fake_openai_response(content: str):
    class Message:
        pass

    class Choice:
        pass

    class Response:
        pass

    message = Message()
    message.content = content
    choice = Choice()
    choice.message = message
    response = Response()
    response.choices = [choice]
    return response


def _fake_client_returning(content: str):
    class FakeCompletions:
        def create(self, **kwargs):
            return _fake_openai_response(content)

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    return FakeClient


def test_dynamic_vocabulary_makes_item_two_see_item_ones_accepted_label(monkeypatch, tmp_path: Path) -> None:
    """End-to-end proof (no GPU, no network -- the OpenAI client is mocked) that the dynamic
    half of the canonical vocabulary actually works through the real graph nodes, not just in
    isolated unit calls: two items run in sequence through
    propose_with_llama_swap_node -> compile_level_counts_node -> gate_candidates_node ->
    persist_proposed_artifact_node. Item 1 proposes "renal excretion agent" (not a static
    anchor, not in any reference file) and gets it accepted and persisted. Item 2 then proposes
    the near-duplicate paraphrase "renal elimination agent", unreused. If the dynamic vocabulary
    genuinely works, item 2 must be caught -- and it can *only* be caught by seeing item 1's own
    run history, since neither label exists in any static anchor.
    """
    import cloak.lattice_producer.propose as propose_module

    profiles = tmp_path / "profiles.json"
    _profiles(profiles)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    proposed_out = tmp_path / "proposed.json"
    monkeypatch.setenv("INFERDPT_LLM_CACHE", str(tmp_path / "cache"))

    state = make_initial_state(
        run_id="dynamic-vocab-e2e",
        run_dir=run_dir,
        profiles_path=profiles,
        proposed_out=proposed_out,
    )

    # --- item 1: "aleve" -> "renal excretion agent", accepted and persisted ---
    item1_payload = json.dumps(
        {
            "aliases": ["some brand"],
            "surface_confidence": "high",
            "candidates": [
                {
                    "level": "renal excretion agent",
                    "proposed_count": 4200,
                    "count_evidence": "Estimated from clinical formulary references.",
                    "selector": "model-domain-cluster:renal",
                    "rationale": "Truthful generalization for this entry.",
                    "reused_canonical_label": False,
                }
            ],
        }
    )
    monkeypatch.setattr(propose_module, "OpenAI", _fake_client_returning(item1_payload))

    state["current_item"] = {"item_id": "drug:aleve", "runtime_type": "drug", "surface": "aleve", "aliases": []}
    state.update(propose_with_llama_swap_node(state))
    state.update(compile_level_counts_node(state))
    state.update(gate_candidates_node(state))

    assert [row["level"] for row in state["accepted_rows"]] == ["renal excretion agent"]
    persist_proposed_artifact_node(state)

    # confirm it's genuinely durable on disk before item 2 ever runs
    written = json.loads(proposed_out.read_text())
    assert written["profiles"]["drug"]["aleve"]["levels"] == ["renal excretion agent"]

    # --- item 2: "metoprolol" -> unreused near-duplicate paraphrase ---
    item2_payload = json.dumps(
        {
            "aliases": ["another brand"],
            "surface_confidence": "high",
            "candidates": [
                {
                    "level": "renal elimination agent",
                    "proposed_count": 5500,
                    "count_evidence": "Estimated from clinical formulary references.",
                    "selector": "model-domain-cluster:renal",
                    "rationale": "Truthful generalization for this entry.",
                    "reused_canonical_label": False,
                }
            ],
        }
    )
    monkeypatch.setattr(propose_module, "OpenAI", _fake_client_returning(item2_payload))

    state["current_item"] = {"item_id": "drug:metoprolol", "runtime_type": "drug", "surface": "metoprolol", "aliases": []}
    state.update(propose_with_llama_swap_node(state))
    state.update(compile_level_counts_node(state))
    state.update(gate_candidates_node(state))

    assert state["accepted_rows"] == []
    assert state["diagnostic_rows"][0]["reason"] == "unreused_near_duplicate_label"
    assert "renal excretion agent" in state["diagnostic_rows"][0]["near_duplicates"]


def test_deterministic_lookup_prefers_reference_source_over_profile_cache(monkeypatch, tmp_path: Path) -> None:
    import cloak.lattice_producer.graph as graph_module

    reference_hit = [
        {
            "level": "aminoketone",
            "source_family": "openfda-pharm-class",
            "selector": "openfda_ndc.pharm_class == 'Aminoketone [EPC]'",
            "member_set": frozenset({"bupropion hydrochloride"}),
            "member_set_ref": "openfda-ndc:pharm_class:Aminoketone [EPC]",
        }
    ]
    monkeypatch.setattr(graph_module, "reference_candidates_for", lambda item: reference_hit)
    monkeypatch.setattr(graph_module, "lookup_levels", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("profile cache should not be consulted when a reference source hits")))

    profiles = tmp_path / "profiles.json"
    _profiles(profiles)
    state = {"current_item": {"surface": "bupropion", "runtime_type": "drug"}, "profiles_path": profiles}

    result = deterministic_lookup(state)

    assert result == {"current_candidates": reference_hit}


def test_deterministic_lookup_falls_back_to_profile_cache_when_no_reference_hit(monkeypatch, tmp_path: Path) -> None:
    import cloak.lattice_producer.graph as graph_module

    monkeypatch.setattr(graph_module, "reference_candidates_for", lambda item: None)

    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created": "2026-07-07",
                "sources": {},
                "profiles": {"profession": {"teacher": {"aliases": [], "levels": ["education worker"], "source_ids": [], "count": 1}}},
            }
        )
    )
    state = {"current_item": {"surface": "teacher", "runtime_type": "profession"}, "profiles_path": profiles}

    result = deterministic_lookup(state)

    assert result == {"current_candidates": [{"level": "education worker", "source_family": "deterministic"}]}


def test_graph_builds_langgraph_app_with_persistent_thread_id(tmp_path: Path) -> None:
    graph = build_graph()
    state = make_initial_state(
        run_id="offline-smoke",
        run_dir=tmp_path / "run",
        profiles_path=tmp_path / "profiles.json",
        proposed_out=tmp_path / "proposed.json",
        offline_only=True,
    )

    assert callable(getattr(graph, "compile"))
    assert thread_id_for_run(state["run_id"]).startswith("lattice-producer:")
    assert len(thread_id_for_run(state["run_id"])) < 255


def test_force_model_proposal_routes_selected_item_to_model_node(tmp_path: Path) -> None:
    state = make_initial_state(
        run_id="forced-model",
        run_dir=tmp_path / "run",
        profiles_path=tmp_path / "profiles.json",
        proposed_out=tmp_path / "proposed.json",
    )
    state["current_item"] = {
        "item_id": "drug:aspirin",
        "runtime_type": "drug",
        "surface": "aspirin",
        "force_model_proposal": True,
    }

    assert route_selected(state) == "propose_with_llama_swap"


def test_default_state_uses_qwen36_as_only_proposal_model(tmp_path: Path) -> None:
    state = make_initial_state(
        run_id="retry-smoke",
        run_dir=tmp_path / "run",
        profiles_path=tmp_path / "profiles.json",
        proposed_out=tmp_path / "proposed.json",
    )

    assert state["model"] == QWEN36_ESCALATION_MODEL
    assert state["escalation_model"] == QWEN36_ESCALATION_MODEL


def test_rejected_item_requeues_once_with_rejection_feedback(tmp_path: Path) -> None:
    state = make_initial_state(
        run_id="retry-smoke",
        run_dir=tmp_path / "run",
        profiles_path=tmp_path / "profiles.json",
        proposed_out=tmp_path / "proposed.json",
    )
    state["current_item"] = {
        "item_id": "profession:privacy engineer",
        "runtime_type": "profession",
        "surface": "privacy engineer",
    }
    state["accepted_rows"] = []
    state["rejected_rows"] = []
    state["diagnostic_rows"] = [
        {
            "item_id": "profession:privacy engineer",
            "level": "architecture and engineering occupation",
            "reason": "missing_aliases",
        }
    ]

    assert route_after_gate(state) == "requeue_rejected_item"
    retry = requeue_rejected_item(state)

    assert retry["current_item"]["retry_attempt"] == 1
    assert retry["current_item"]["rejection_feedback"] == [
        {
            "reason": "missing_aliases",
            "level": "architecture and engineering occupation",
            "count": None,
            "grounding_status": None,
        }
    ]
    assert retry["rejected_rows"] == []
    assert retry["diagnostic_rows"] == []


def test_rejected_retry_records_after_one_escalation_attempt(tmp_path: Path) -> None:
    state = make_initial_state(
        run_id="retry-smoke",
        run_dir=tmp_path / "run",
        profiles_path=tmp_path / "profiles.json",
        proposed_out=tmp_path / "proposed.json",
    )
    state["current_item"] = {
        "item_id": "profession:privacy engineer",
        "runtime_type": "profession",
        "surface": "privacy engineer",
        "retry_attempt": 1,
    }
    state["accepted_rows"] = []
    state["rejected_rows"] = []
    state["diagnostic_rows"] = [{"reason": "weak_semantic_relevance"}]

    assert route_after_gate(state) == "record_item_result"


def test_retry_proposal_node_uses_qwen36_escalation_model(monkeypatch, tmp_path: Path) -> None:
    seen = {}

    def fake_propose(item, **kwargs):
        seen["item"] = item
        seen.update(kwargs)
        return {"candidates": [{"level": "software privacy professional"}]}

    monkeypatch.setattr("cloak.lattice_producer.graph.propose_with_llama_swap", fake_propose)
    state = make_initial_state(
        run_id="retry-smoke",
        run_dir=tmp_path / "run",
        profiles_path=tmp_path / "profiles.json",
        proposed_out=tmp_path / "proposed.json",
        model="gemma 4 (E4B)",
    )
    Path(state["run_dir"]).mkdir()
    (Path(state["run_dir"]) / "proposals.jsonl").touch()
    state["current_item"] = {
        "item_id": "profession:privacy engineer",
        "runtime_type": "profession",
        "surface": "privacy engineer",
        "retry_attempt": 1,
    }

    result = propose_with_llama_swap_node(state)

    assert seen["model"] == QWEN36_ESCALATION_MODEL
    assert seen["escalation_model"] == QWEN36_ESCALATION_MODEL
    assert result["current_candidates"][0]["level"] == "software privacy professional"


def test_first_pass_proposal_node_uses_qwen36_even_if_state_model_is_gemma(monkeypatch, tmp_path: Path) -> None:
    seen = {}

    def fake_propose(item, **kwargs):
        seen["item"] = item
        seen.update(kwargs)
        return {"candidates": [{"level": "software privacy professional"}]}

    monkeypatch.setattr("cloak.lattice_producer.graph.propose_with_llama_swap", fake_propose)
    state = make_initial_state(
        run_id="retry-smoke",
        run_dir=tmp_path / "run",
        profiles_path=tmp_path / "profiles.json",
        proposed_out=tmp_path / "proposed.json",
        model="gemma 4 (E4B)",
    )
    Path(state["run_dir"]).mkdir()
    (Path(state["run_dir"]) / "proposals.jsonl").touch()
    state["current_item"] = {
        "item_id": "profession:privacy engineer",
        "runtime_type": "profession",
        "surface": "privacy engineer",
    }

    propose_with_llama_swap_node(state)

    assert seen["model"] == QWEN36_ESCALATION_MODEL


def test_offline_graph_persists_accepted_item_before_review_interrupt(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    proposed = tmp_path / "proposed.json"
    run_dir = tmp_path / "run"
    queue = tmp_path / "queue.jsonl"
    _profiles(profiles)
    queue.write_text(
        json.dumps(
            {
                "item_id": "profession:cardiologist",
                "task_kind": "generated-universe",
                "runtime_type": "profession",
                "detector_label_family": "profession",
                "surface": "cardiologist",
                "canonical_value": "cardiologist",
                "aliases": [],
                "entry_origin": "generated-universe",
                "proposed_levels": ["medical specialist", "healthcare worker"],
            }
        )
        + "\n"
    )

    result = run_producer(
        run_dir=run_dir,
        profiles_path=profiles,
        proposed_out=proposed,
        queue_path=queue,
        offline_only=True,
        max_items=1,
        review_decision="approve-proposed-only",
    )

    artifact = json.loads(proposed.read_text())
    assert result["accepted"] == 2
    assert artifact["artifact_role"] == "proposal"
    assert artifact["profiles"]["profession"]["cardiologist"]["levels"]
    assert not (run_dir / "run_state.json").exists()


def test_experiment_log_contains_one_line_per_processed_entry(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    proposed = tmp_path / "proposed.json"
    run_dir = tmp_path / "run"
    queue = tmp_path / "queue.jsonl"
    _profiles(profiles)
    rows = [
        {
            "item_id": "profession:cardiologist",
            "task_kind": "generated-universe",
            "runtime_type": "profession",
            "detector_label_family": "profession",
            "surface": "cardiologist",
            "canonical_value": "cardiologist",
            "entry_origin": "generated-universe",
            "proposed_levels": ["medical specialist"],
        },
        {
            "item_id": "profession:teacher",
            "task_kind": "generated-universe",
            "runtime_type": "profession",
            "detector_label_family": "profession",
            "surface": "teacher",
            "canonical_value": "teacher",
            "entry_origin": "generated-universe",
            "proposed_levels": ["education worker"],
        },
    ]
    queue.write_text("".join(json.dumps(row) + "\n" for row in rows))

    run_producer(
        run_dir=run_dir,
        profiles_path=profiles,
        proposed_out=proposed,
        queue_path=queue,
        offline_only=True,
        max_items=2,
        review_decision="approve-proposed-only",
    )

    log = (run_dir / "EXPERIMENT_LOG.md").read_text()
    assert "- item_id: profession:cardiologist" in log
    assert "- item_id: profession:teacher" in log
    assert log.count("- item_id:") == 2


def test_run_writes_readable_review_report(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    proposed = tmp_path / "proposed.json"
    run_dir = tmp_path / "run"
    queue = tmp_path / "queue.jsonl"
    _profiles(profiles)
    queue.write_text(
        json.dumps(
            {
                "item_id": "profession:cardiologist",
                "task_kind": "generated-universe",
                "runtime_type": "profession",
                "detector_label_family": "profession",
                "surface": "cardiologist",
                "canonical_value": "cardiologist",
                "entry_origin": "generated-universe",
                "proposed_levels": ["medical specialist"],
            }
        )
        + "\n"
    )

    run_producer(
        run_dir=run_dir,
        profiles_path=profiles,
        proposed_out=proposed,
        queue_path=queue,
        offline_only=True,
        max_items=1,
        review_decision="approve-proposed-only",
    )

    report = (run_dir / "REVIEW_REPORT.md").read_text()
    assert "profession:cardiologist" in report
    assert "medical specialist" in report
    assert str(proposed) in report


def test_review_report_includes_aliases_evidence_and_warning_reasons(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    proposed = tmp_path / "proposed.json"
    run_dir = tmp_path / "run"
    queue = tmp_path / "queue.jsonl"
    _profiles(profiles)
    queue.write_text(
        json.dumps(
            {
                "item_id": "profession:privacy engineer",
                "runtime_type": "profession",
                "detector_label_family": "profession",
                "surface": "privacy engineer",
                "canonical_value": "privacy engineer",
                "marked_context_sentence": "The client works as a [SPAN]privacy engineer[/SPAN].",
            }
        )
        + "\n"
    )
    run_dir.mkdir()
    for name in ("proposals.jsonl", "accepted.jsonl", "rejected.jsonl", "diagnostics.jsonl"):
        (run_dir / name).touch()
    (run_dir / "accepted.jsonl").write_text(
        json.dumps(
            {
                "item_id": "profession:privacy engineer",
                "runtime_type": "profession",
                "level": "privacy and security software professional",
                "aliases": ["data protection engineer"],
                "level_count": 180.0,
                "level_grounding": {
                    "status": "model-proposed",
                    "source_family": "model-proposed",
                    "count_evidence": "Includes privacy engineering and security engineering roles.",
                },
                "rationale": "Preserves privacy/security/software context.",
            }
        )
        + "\n"
    )
    (run_dir / "diagnostics.jsonl").write_text(
        json.dumps(
            {
                "item_id": "profession:beer cicerone",
                "runtime_type": "profession",
                "surface": "beer cicerone",
                "level": "worker",
                "reason": "weak_semantic_relevance",
            }
        )
        + "\n"
    )

    run_producer(
        run_dir=run_dir,
        profiles_path=profiles,
        proposed_out=proposed,
        queue_path=queue,
        offline_only=True,
        max_items=0,
        review_decision="approve-proposed-only",
    )

    report = (run_dir / "REVIEW_REPORT.md").read_text()
    assert "aliases: data protection engineer" in report
    assert "count_evidence: Includes privacy engineering and security engineering roles." in report
    assert "rationale: Preserves privacy/security/software context." in report
    assert "weak_semantic_relevance" in report
