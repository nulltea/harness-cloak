import json
from pathlib import Path

from cloak.lattice.producer.graph import (
    QWEN36_ESCALATION_MODEL,
    augment_with_model_node,
    build_graph,
    compile_level_counts_node,
    deterministic_lookup,
    gate_candidates_node,
    merge_anchor_and_model,
    persist_proposed_artifact_node,
    propose_with_llama_swap_node,
    requeue_rejected_item,
    route_after_deterministic,
    route_selected,
    route_after_gate,
    run_producer,
    should_continue,
)
from cloak.lattice.producer.state import make_initial_state, thread_id_for_run


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
    import cloak.lattice.producer.propose as propose_module

    profiles = tmp_path / "profiles.json"
    _profiles(profiles)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    proposed_out = tmp_path / "proposed.json"
    monkeypatch.setenv("CLOAK_LLM_CACHE", str(tmp_path / "cache"))

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
                },
                {
                    "level": "renal system therapeutic",
                    "proposed_count": 15000,
                    "count_evidence": "Estimated from clinical formulary references.",
                    "selector": "model-domain-cluster:renal",
                    "rationale": "Broader truthful generalization for this entry.",
                    "reused_canonical_label": False,
                },
            ],
        }
    )
    monkeypatch.setattr(propose_module, "OpenAI", _fake_client_returning(item1_payload))

    state["current_item"] = {"item_id": "drug:aleve", "runtime_type": "drug", "surface": "aleve", "aliases": []}
    state.update(propose_with_llama_swap_node(state))
    state.update(compile_level_counts_node(state))
    state.update(gate_candidates_node(state))

    assert [row["level"] for row in state["accepted_rows"]] == [
        "renal excretion agent",
        "renal system therapeutic",
    ]
    persist_proposed_artifact_node(state)

    # confirm it's genuinely durable on disk before item 2 ever runs
    written = json.loads(proposed_out.read_text())
    assert written["profiles"]["drug"]["aleve"]["levels"] == [
        "renal excretion agent",
        "renal system therapeutic",
    ]

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


def test_should_continue_triggers_periodic_normalization():
    state = {"processed": 50, "max_items": None, "normalize_every": 50}
    assert should_continue(state) == "normalize_coherence"
    state = {"processed": 49, "max_items": None, "normalize_every": 50}
    assert should_continue(state) == "select_next_item"


def test_periodic_normalize_resumes_loop_but_exhausted_validates():
    from cloak.lattice.producer.graph import _route_after_normalize

    # periodic trigger: item cleared by record_item_result, queue not exhausted -> resume
    resume = {"processed": 50, "max_items": None, "current_item": None, "queue_exhausted": False}
    assert _route_after_normalize(resume) == "select_next_item"
    # genuine queue exhaustion -> validate
    exhausted = {"processed": 50, "max_items": None, "current_item": None, "queue_exhausted": True}
    assert _route_after_normalize(exhausted) == "validate_proposed_artifact"
    # max_items hit -> validate
    maxed = {"processed": 5, "max_items": 5, "current_item": None, "queue_exhausted": False}
    assert _route_after_normalize(maxed) == "validate_proposed_artifact"


def test_normalize_coherence_node_applies_entity_merge(tmp_path, monkeypatch):
    proposed = tmp_path / "proposed.json"
    proposed.write_text(json.dumps({
        "schema_version": 1, "created": "2026-07-10", "sources": {}, "profiles": {
            "health-condition": {
                "blorbitis": {"aliases": [], "levels": ["organ disease"],
                              "source_ids": ["t:1"], "count": 10.0}}},
        "artifact_role": "proposal", "proposal_scope": "producer-processed-only"}))
    calls = {}
    def fake_merge(artifact, **kwargs):
        calls["profiles"] = artifact["profiles"]
        return {"types": {"health-condition": {"merged": [], "review": [], "gate_scored": 0}},
                "duplicate_surface_claims": {}}
    monkeypatch.setattr("cloak.lattice.producer.graph.apply_entity_merge", fake_merge)
    state = {"proposed_out": str(proposed), "profiles_path": str(tmp_path / "canon.json"),
             "run_id": "test-run", "run_dir": str(tmp_path)}
    from cloak.lattice.producer.graph import normalize_coherence_node
    normalize_coherence_node(state)
    assert "health-condition" in calls["profiles"]
    assert (tmp_path / "entity_merge_report.json").exists()


def test_deterministic_lookup_prefers_reference_source_over_profile_cache(monkeypatch, tmp_path: Path) -> None:
    import cloak.lattice.producer.graph as graph_module

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


def test_merge_anchor_and_model_keeps_certifying_anchor_and_drops_anchor_paraphrase() -> None:
    anchor = {
        "level": "benzodiazepine",
        "source_family": "openfda-pharm-class",
        "selector": "openfda_ndc.pharm_class == 'Benzodiazepine [EPC]'",
        "member_set": frozenset({"alprazolam", "lorazepam"}),
        "member_set_ref": "openfda-ndc:pharm_class:Benzodiazepine [EPC]",
    }
    model_candidates = [
        {"level": "benzodiazepine derivative", "source_family": "model-proposed"},
        {"level": "central nervous system depressant", "source_family": "model-proposed"},
        {"level": "medication", "source_family": "model-proposed"},
    ]

    merged = merge_anchor_and_model([anchor], model_candidates)

    assert [candidate["level"] for candidate in merged] == [
        "benzodiazepine",
        "central nervous system depressant",
        "medication",
    ]
    assert merged[0] is anchor
    assert merged[0]["member_set"] == frozenset({"alprazolam", "lorazepam"})


def test_insufficient_deterministic_anchor_routes_to_model_augmentation(tmp_path: Path) -> None:
    state = make_initial_state(
        run_id="hybrid-anchor-routing",
        run_dir=tmp_path / "run",
        profiles_path=tmp_path / "profiles.json",
        proposed_out=tmp_path / "proposed.json",
    )
    state["current_item"] = {"item_id": "drug:diazepam", "runtime_type": "drug", "surface": "diazepam"}
    state["current_candidates"] = [
        {
            "level": "benzodiazepine",
            "source_family": "openfda-pharm-class",
            "member_set": frozenset({"alprazolam", "lorazepam"}),
        }
    ]

    assert route_after_deterministic(state) == "augment_with_model"


def test_sufficient_deterministic_chain_routes_directly_to_count_compilation(tmp_path: Path) -> None:
    state = make_initial_state(
        run_id="deterministic-chain-routing",
        run_dir=tmp_path / "run",
        profiles_path=tmp_path / "profiles.json",
        proposed_out=tmp_path / "proposed.json",
    )
    state["current_item"] = {"item_id": "health-condition:asthma", "runtime_type": "health-condition", "surface": "asthma"}
    state["current_candidates"] = [
        {"level": "asthma", "source_family": "doid-is-a", "member_set": frozenset({"asthma"})},
        {
            "level": "respiratory system disease",
            "source_family": "doid-is-a",
            "member_set": frozenset(str(idx) for idx in range(150)),
        },
    ]

    assert route_after_deterministic(state) == "compile_level_counts"


def test_hybrid_anchor_augmentation_keeps_certifying_nearest_rung(monkeypatch, tmp_path: Path) -> None:
    import cloak.lattice.producer.graph as graph_module

    profiles = tmp_path / "profiles.json"
    _profiles(profiles)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "generated_universe.jsonl").touch()
    (run_dir / "proposals.jsonl").touch()
    proposed_out = tmp_path / "proposed.json"

    anchor = {
        "level": "benzodiazepine",
        "source_family": "openfda-pharm-class",
        "selector": "openfda_ndc.pharm_class == 'Benzodiazepine [EPC]'",
        "member_set": frozenset({"alprazolam", "lorazepam"}),
        "member_set_ref": "openfda-ndc:pharm_class:Benzodiazepine [EPC]",
    }
    monkeypatch.setattr(graph_module, "reference_candidates_for", lambda item: [anchor])

    def fake_propose(item, **kwargs):
        return {
            "surface_confidence": "high",
            "aliases": ["diazepam"],
            "candidates": [
                {
                    "level": "benzodiazepine derivative",
                    "aliases": ["diazepam"],
                    "proposed_count": 150,
                    "count_evidence": "Broad drug class with more members than the source EPC anchor.",
                    "selector": "model-domain-cluster:benzodiazepine",
                    "rationale": "Paraphrases the deterministic anchor and should be dropped.",
                    "source_family": "model-proposed",
                },
                {
                    "level": "central nervous system depressant",
                    "aliases": ["diazepam"],
                    "proposed_count": 5000,
                    "count_evidence": "Includes sedatives, hypnotics, anxiolytics, and related medications.",
                    "selector": "model-domain-cluster:cns-depressant",
                    "rationale": "A broader truthful pharmacologic tier above benzodiazepines.",
                    "source_family": "model-proposed",
                    "reused_canonical_label": True,
                },
                {
                    "level": "medication",
                    "aliases": ["diazepam"],
                    "proposed_count": 20000,
                    "count_evidence": "Covers marketed therapeutic drug products.",
                    "selector": "model-domain-cluster:medication",
                    "rationale": "The broadest truthful drug tier.",
                    "source_family": "model-proposed",
                    "reused_canonical_label": True,
                },
            ],
        }

    monkeypatch.setattr(graph_module, "propose_with_llama_swap", fake_propose)
    state = make_initial_state(
        run_id="hybrid-anchor-e2e",
        run_dir=run_dir,
        profiles_path=profiles,
        proposed_out=proposed_out,
    )
    state["current_item"] = {
        "item_id": "drug:diazepam",
        "runtime_type": "drug",
        "surface": "diazepam",
        "canonical_value": "diazepam",
        "aliases": ["diazepam"],
    }

    state.update(deterministic_lookup(state))
    assert route_after_deterministic(state) == "augment_with_model"
    state.update(augment_with_model_node(state))
    state.update(compile_level_counts_node(state))
    state.update(gate_candidates_node(state))

    assert [row["level"] for row in state["accepted_rows"]] == [
        "benzodiazepine",
        "central nervous system depressant",
        "medication",
    ]
    assert state["accepted_rows"][0]["level_grounding"]["status"] == "certifying"
    assert state["accepted_rows"][0]["level_grounding"]["source_family"] == "openfda-pharm-class"
    assert state["accepted_rows"][0]["level_count"] == 2.0
    assert max(row["level_count"] for row in state["accepted_rows"]) >= 100.0
    assert not any(row["reason"] in {"chain_below_floor", "too_few_levels"} for row in state["diagnostic_rows"])


def test_deterministic_lookup_reference_type_miss_goes_to_model_not_cache(monkeypatch, tmp_path: Path) -> None:
    # a runtime type WITH a reference source (drug/openFDA) that MISSES must not fall back to the
    # lattice_profiles.json cache (unreliable) -- it must return empty candidates so the graph routes
    # to the model.
    import cloak.lattice.producer.graph as graph_module

    monkeypatch.setattr(graph_module, "reference_candidates_for", lambda item: None)  # openFDA miss

    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created": "2026-07-07",
                "sources": {},
                "profiles": {"drug": {"aspirin": {"aliases": [], "levels": ["nsaid", "medication"], "source_ids": [], "count": 1}}},
            }
        )
    )
    state = {"current_item": {"surface": "aspirin", "runtime_type": "drug"}, "profiles_path": profiles}

    result = deterministic_lookup(state)

    assert result == {"current_candidates": []}


def test_deterministic_lookup_falls_back_to_profile_cache_when_no_reference_hit(monkeypatch, tmp_path: Path) -> None:
    import cloak.lattice.producer.graph as graph_module

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
    feedback = retry["current_item"]["rejection_feedback"]
    assert feedback[0]["reason"] == "missing_aliases"
    assert feedback[0]["level"] == "architecture and engineering occupation"
    assert "same-entity aliases" in feedback[0]["repair_hint"]  # actionable, not just the code
    assert retry["rejected_rows"] == []


def test_rejection_feedback_repair_hint_is_specific_per_reason(tmp_path: Path) -> None:
    state = make_initial_state(
        run_id="hint-smoke",
        run_dir=tmp_path / "run",
        profiles_path=tmp_path / "profiles.json",
        proposed_out=tmp_path / "proposed.json",
    )
    state["current_item"] = {"item_id": "drug:albuterol", "runtime_type": "drug", "surface": "albuterol"}
    state["accepted_rows"] = []
    state["rejected_rows"] = [
        {"item_id": "drug:albuterol", "level": "salbutamol inhaler", "reason": "self_leak"},
    ]
    state["diagnostic_rows"] = [
        {"item_id": "drug:albuterol", "level": "bronchodilator", "reason": "count_disagreement",
         "recorded_count": 25.0},
        {"item_id": "drug:albuterol", "level": "short-acting beta-2 agonist",
         "reason": "unreused_near_duplicate_label", "near_duplicates": ["beta-2 adrenergic agonist"]},
    ]

    retry = requeue_rejected_item(state)
    hints = {f["reason"]: f.get("repair_hint", "") for f in retry["current_item"]["rejection_feedback"]}
    assert "25.0" in hints["count_disagreement"]  # concrete recorded size, not boilerplate
    assert "beta-2 adrenergic agonist" in hints["unreused_near_duplicate_label"]  # the label to reuse
    assert "surface" in hints["self_leak"]
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


def test_retry_proposal_node_uses_state_model_for_escalation(monkeypatch, tmp_path: Path) -> None:
    seen = {}

    def fake_propose(item, **kwargs):
        seen["item"] = item
        seen.update(kwargs)
        return {"candidates": [{"level": "software privacy professional"}]}

    monkeypatch.setattr("cloak.lattice.producer.graph.propose_with_llama_swap", fake_propose)
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

    # the node now threads the configured model through (was hardcoded to Qwen); escalation falls
    # back to the same model when no distinct escalation_model is set.
    assert seen["model"] == "gemma 4 (E4B)"
    assert seen["escalation_model"] == "gemma 4 (E4B)"
    assert result["current_candidates"][0]["level"] == "software privacy professional"


def test_first_pass_proposal_node_uses_state_model(monkeypatch, tmp_path: Path) -> None:
    seen = {}

    def fake_propose(item, **kwargs):
        seen["item"] = item
        seen.update(kwargs)
        return {"candidates": [{"level": "software privacy professional"}]}

    monkeypatch.setattr("cloak.lattice.producer.graph.propose_with_llama_swap", fake_propose)
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

    assert seen["model"] == "gemma 4 (E4B)"


def test_proposal_node_defaults_to_qwen36_when_no_model_configured(monkeypatch, tmp_path: Path) -> None:
    seen = {}

    def fake_propose(item, **kwargs):
        seen.update(kwargs)
        return {"candidates": [{"level": "software privacy professional"}]}

    monkeypatch.setattr("cloak.lattice.producer.graph.propose_with_llama_swap", fake_propose)
    state = make_initial_state(
        run_id="default-model",
        run_dir=tmp_path / "run",
        profiles_path=tmp_path / "profiles.json",
        proposed_out=tmp_path / "proposed.json",
    )
    Path(state["run_dir"]).mkdir()
    (Path(state["run_dir"]) / "proposals.jsonl").touch()
    state["current_item"] = {"item_id": "profession:x", "runtime_type": "profession", "surface": "x"}

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


def test_propose_node_consumes_prefetched_payload_without_inline_call(monkeypatch, tmp_path: Path) -> None:
    from concurrent.futures import Future

    import cloak.lattice.producer.graph as graph_module
    from cloak.lattice.producer.graph import _PREFETCHER

    profiles = tmp_path / "profiles.json"
    _profiles(profiles)
    state = make_initial_state(
        run_id="prefetch-consume",
        run_dir=tmp_path / "run",
        profiles_path=profiles,
        proposed_out=tmp_path / "proposed.json",
    )
    state["current_item"] = {"item_id": "drug:aleve", "runtime_type": "drug", "surface": "aleve"}
    (tmp_path / "run").mkdir()

    payload = {"candidates": [{"level": "nsaid", "proposed_count": 900, "count_evidence": "e", "selector": "s", "rationale": "r"}]}
    future: Future = Future()
    future.set_result(payload)
    _PREFETCHER._futures["drug:aleve"] = future
    monkeypatch.setattr(
        graph_module,
        "propose_with_llama_swap",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("inline propose must not run when a prefetched payload exists")),
    )
    try:
        result = propose_with_llama_swap_node(state)
    finally:
        _PREFETCHER._futures.clear()
    assert [c["level"] for c in result["current_candidates"]] == ["nsaid"]
    # a failed prefetch future falls back to the inline call
    failed: Future = Future()
    failed.set_exception(RuntimeError("boom"))
    _PREFETCHER._futures["drug:aleve"] = failed
    monkeypatch.setattr(graph_module, "propose_with_llama_swap", lambda *a, **k: payload)
    try:
        result = propose_with_llama_swap_node(state)
    finally:
        _PREFETCHER._futures.clear()
    assert [c["level"] for c in result["current_candidates"]] == ["nsaid"]


def test_will_call_model_mirrors_routing(monkeypatch) -> None:
    import cloak.lattice.producer.graph as graph_module
    from cloak.lattice.producer.graph import _will_call_model

    monkeypatch.setattr(graph_module, "reference_candidates_for", lambda item: [])
    monkeypatch.setattr(graph_module, "has_reference_source", lambda rt: rt == "drug")
    assert _will_call_model({"item_id": "drug:x", "runtime_type": "drug"}) is True  # reference miss -> model
    assert _will_call_model({"item_id": "LOC:paris", "runtime_type": "LOC"}) is False  # no reference source
    assert _will_call_model({"item_id": "drug:x", "runtime_type": "drug", "eligible": False}) is False
    assert _will_call_model({"item_id": "drug:x", "runtime_type": "drug", "task_kind": "generated-universe"}) is False
    assert _will_call_model({"item_id": "LOC:paris", "runtime_type": "LOC", "force_model_proposal": True}) is True

    insufficient = [{"level": "retinoid", "member_set": frozenset({"a"}), "source_family": "openfda-pharm-class"}]
    monkeypatch.setattr(graph_module, "reference_candidates_for", lambda item: insufficient)
    assert _will_call_model({"item_id": "drug:accutane", "runtime_type": "drug"}) is True  # augment path

    sufficient = [
        {"level": "l1", "member_set": frozenset(f"m{i}" for i in range(150))},
        {"level": "l2", "member_set": frozenset(f"m{i}" for i in range(200))},
    ]
    monkeypatch.setattr(graph_module, "reference_candidates_for", lambda item: sufficient)
    assert _will_call_model({"item_id": "drug:x", "runtime_type": "drug"}) is False  # chain sufficient
