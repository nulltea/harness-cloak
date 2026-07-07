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
