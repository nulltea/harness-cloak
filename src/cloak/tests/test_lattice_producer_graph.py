import json
from pathlib import Path

from cloak.lattice_producer.graph import build_graph, run_producer
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
