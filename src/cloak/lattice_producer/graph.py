"""LangGraph orchestration for the offline lattice producer."""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from cloak.lattice import bucket_date, bucket_quantity, geonames_chain
from cloak.lattice_profiles import lookup_levels
from cloak.lattice_producer.counts import compile_level_counts
from cloak.lattice_producer.coverage import write_category_coverage
from cloak.lattice_producer.gates import gate_candidates
from cloak.lattice_producer.io import append_jsonl_unique, read_jsonl
from cloak.lattice_producer.merge import ensure_proposed_artifact, persist_proposed_artifact, validate_proposed_artifact
from cloak.lattice_producer.propose import ensure_local_base_url, extract_candidate_levels, propose_with_llama_swap
from cloak.lattice_producer.queue import build_or_load_queue
from cloak.lattice_producer.state import ProducerState, make_initial_state, thread_id_for_run


def _jsonl_path(state: ProducerState, name: str) -> Path:
    return Path(state["run_dir"]) / name


def _load_queue(state: ProducerState) -> list[dict[str, Any]]:
    return read_jsonl(state["queue_path"])


def _append_experiment_log(state: ProducerState, lines: list[str]) -> None:
    log = _jsonl_path(state, "EXPERIMENT_LOG.md")
    with log.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")


def initialize_run(state: ProducerState) -> ProducerState:
    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    brief = run_dir / "EXPERIMENT_BRIEF.md"
    if not brief.exists():
        brief.write_text(
            "\n".join(
                [
                    "# Lattice Producer Run",
                    "",
                    f"- run_id: {state['run_id']}",
                    f"- created: {date.today()}",
                    f"- profiles: {state['profiles_path']}",
                    f"- proposed_out: {state['proposed_out']}",
                    f"- model: {state.get('model', '')}",
                    f"- offline_only: {state.get('offline_only', False)}",
                    "",
                    "Goal: incrementally stage proposed lattice profile rows for review.",
                    "",
                ]
            )
        )
    log = run_dir / "EXPERIMENT_LOG.md"
    if not log.exists():
        log.write_text(
            "\n".join(
                [
                    "# Experiment Log",
                    "",
                    "Current hypothesis: initialize producer queue.",
                    "Latest result: pending.",
                    "Next planned step: build coverage.",
                    "",
                    "## Processed Entries",
                    "",
                ]
            )
        )
    for name in ("generated_universe.jsonl", "proposals.jsonl", "accepted.jsonl", "rejected.jsonl", "diagnostics.jsonl"):
        _jsonl_path(state, name).touch(exist_ok=True)
    if not state.get("offline_only"):
        ensure_local_base_url(state["base_url"])
    return state


def build_category_coverage_node(state: ProducerState) -> ProducerState:
    rows = write_category_coverage(state["profiles_path"], _jsonl_path(state, "coverage_gaps.json"), category=state.get("category"))
    return {"coverage_gaps": sum(1 for row in rows if row.get("generated_universe_required"))}


def build_or_load_queue_node(state: ProducerState) -> ProducerState:
    queue_path = Path(state["queue_path"])
    explicit = queue_path if queue_path.exists() else None
    items = build_or_load_queue(state["run_dir"], state["profiles_path"], explicit_queue=explicit, category=state.get("category"))
    if explicit is not None:
        Path(state["run_dir"]).mkdir(parents=True, exist_ok=True)
        Path(state["run_dir"], "queue.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in items))
    return {"queue_path": str(Path(state["run_dir"]) / "queue.jsonl")}


def select_next_item(state: ProducerState) -> ProducerState:
    queue = _load_queue(state)
    seen = {
        str(row.get("item_id"))
        for name in ("accepted.jsonl", "rejected.jsonl", "diagnostics.jsonl")
        for row in read_jsonl(_jsonl_path(state, name))
        if row.get("item_id")
    }
    idx = int(state.get("queue_index", 0))
    processed = int(state.get("processed", 0))
    max_items = state.get("max_items")
    while idx < len(queue):
        item = queue[idx]
        idx += 1
        if str(item.get("item_id")) in seen:
            continue
        if max_items is not None and processed >= int(max_items):
            return {"current_item": None, "queue_index": idx}
        return {"current_item": item, "queue_index": idx}
    return {"current_item": None, "queue_index": idx}


def route_selected(state: ProducerState) -> Literal["validate_proposed_artifact", "generate_universe_entries", "deterministic_lookup", "record_item_result"]:
    item = state.get("current_item")
    if not item:
        return "validate_proposed_artifact"
    if not item.get("eligible", True):
        return "record_item_result"
    if item.get("task_kind") == "generated-universe" or item.get("entry_origin") == "generated-universe":
        return "generate_universe_entries"
    return "deterministic_lookup"


def generate_universe_entries(state: ProducerState) -> ProducerState:
    item = dict(state["current_item"] or {})
    levels = [str(level).strip() for level in item.get("proposed_levels", []) if str(level).strip()]
    if not levels and not state.get("offline_only"):
        proposal = propose_with_llama_swap(
            item,
            profiles_path=state["profiles_path"],
            run_dir=state["run_dir"],
            prompt_version=state["prompt_version"],
            max_context_rows=state["max_context_rows"],
            base_url=state["base_url"],
            model=state["model"],
            escalation_model=state.get("escalation_model"),
        )
        append_jsonl_unique(_jsonl_path(state, "proposals.jsonl"), [{**proposal, "item_id": item.get("item_id")}])
        entries = proposal.get("entries") or []
        if entries:
            added_entries = []
            candidates = []
            for idx, entry in enumerate(entries[: int(state.get("max_generated_entries_per_category", 20))]):
                generated = {
                    "item_id": f"{item.get('item_id')}:generated:{idx}",
                    "runtime_type": item.get("runtime_type"),
                    "detector_label_family": item.get("detector_label_family"),
                    "canonical_value": entry.get("canonical_value"),
                    "aliases": entry.get("aliases", []),
                    "proposed_levels": entry.get("proposed_levels", entry.get("levels", [])),
                    "proposed_groundings": entry.get("proposed_groundings", {}),
                    "entry_origin": "generated-universe",
                    "generation_rationale": entry.get("generation_rationale", "model generated proposed universe entry"),
                }
                if generated["canonical_value"]:
                    added_entries.append(generated)
                    candidates.extend(
                        {"level": level, "entry_origin": "generated-universe"}
                        for level in generated["proposed_levels"]
                        if str(level).strip()
                    )
            append_jsonl_unique(_jsonl_path(state, "generated_universe.jsonl"), added_entries)
            return {
                "current_candidates": candidates,
                "generated_entries": int(state.get("generated_entries", 0)) + len(added_entries),
            }
        levels = [candidate["level"] for candidate in extract_candidate_levels(proposal)]
    if not levels:
        return {"current_candidates": [], "generated_entries": state.get("generated_entries", 0)}
    entry = {
        "item_id": item.get("item_id"),
        "runtime_type": item.get("runtime_type"),
        "detector_label_family": item.get("detector_label_family"),
        "canonical_value": item.get("canonical_value") or item.get("surface"),
        "aliases": item.get("aliases", []),
        "proposed_levels": levels,
        "proposed_groundings": item.get("proposed_groundings", {}),
        "entry_origin": "generated-universe",
        "generation_rationale": item.get("generation_rationale", "seeded from queue"),
    }
    append_jsonl_unique(_jsonl_path(state, "generated_universe.jsonl"), [entry])
    candidates = [{"level": level, "entry_origin": "generated-universe"} for level in levels]
    item["entry_origin"] = "generated-universe"
    item["canonical_value"] = entry["canonical_value"]
    return {
        "current_item": item,
        "current_candidates": candidates,
        "generated_entries": int(state.get("generated_entries", 0)) + 1,
    }


def deterministic_lookup(state: ProducerState) -> ProducerState:
    item = state["current_item"] or {}
    surface = str(item.get("surface") or item.get("canonical_value") or "")
    runtime_type = str(item.get("runtime_type") or "")
    levels = lookup_levels(surface, runtime_type, state["profiles_path"])
    if not levels and runtime_type == "LOC":
        levels = geonames_chain(surface) or []
    if not levels and runtime_type in {"DATETIME", "age"}:
        levels = bucket_date(surface) or []
    if not levels and runtime_type == "QUANTITY":
        levels = bucket_quantity(surface) or []
    return {"current_candidates": [{"level": level, "source_family": "deterministic"} for level in levels]}


def route_after_deterministic(state: ProducerState) -> Literal["compile_level_counts", "propose_with_llama_swap", "record_item_result"]:
    if state.get("current_candidates"):
        return "compile_level_counts"
    if state.get("offline_only"):
        return "record_item_result"
    return "propose_with_llama_swap"


def propose_with_llama_swap_node(state: ProducerState) -> ProducerState:
    item = state["current_item"] or {}
    proposal = propose_with_llama_swap(
        item,
        profiles_path=state["profiles_path"],
        run_dir=state["run_dir"],
        prompt_version=state["prompt_version"],
        max_context_rows=state["max_context_rows"],
        base_url=state["base_url"],
        model=state["model"],
        escalation_model=state.get("escalation_model"),
    )
    append_jsonl_unique(_jsonl_path(state, "proposals.jsonl"), [{**proposal, "item_id": item.get("item_id")}])
    return {"current_candidates": extract_candidate_levels(proposal)}


def route_after_proposal(state: ProducerState) -> Literal["compile_level_counts", "record_item_result"]:
    return "compile_level_counts" if state.get("current_candidates") else "record_item_result"


def compile_level_counts_node(state: ProducerState) -> ProducerState:
    return {
        "current_candidates": compile_level_counts(
            state["current_item"] or {},
            list(state.get("current_candidates", [])),
            generated_universe_path=_jsonl_path(state, "generated_universe.jsonl"),
        )
    }


def gate_candidates_node(state: ProducerState) -> ProducerState:
    result = gate_candidates(state["current_item"] or {}, list(state.get("current_candidates", [])))
    return {"accepted_rows": result.accepted, "rejected_rows": result.rejected, "diagnostic_rows": result.diagnostics}


def route_after_gate(state: ProducerState) -> Literal["persist_proposed_artifact", "record_item_result"]:
    return "persist_proposed_artifact" if state.get("accepted_rows") else "record_item_result"


def persist_proposed_artifact_node(state: ProducerState) -> ProducerState:
    persist_proposed_artifact(
        state["profiles_path"],
        state["proposed_out"],
        run_id=state["run_id"],
        item=state["current_item"] or {},
        accepted=list(state.get("accepted_rows", [])),
    )
    return {"proposed_persisted": int(state.get("proposed_persisted", 0)) + 1}


def record_item_result(state: ProducerState) -> ProducerState:
    item = state.get("current_item") or {}
    accepted = list(state.get("accepted_rows", []))
    rejected = list(state.get("rejected_rows", []))
    diagnostics = list(state.get("diagnostic_rows", []))
    if not item.get("eligible", True):
        rejected = [{**item, "reason": item.get("skip_reason", "ineligible"), "record_id": f"{item.get('item_id')}:rejected"}]
    elif not accepted and not rejected and not diagnostics:
        diagnostics = [{**item, "reason": "no_candidates"}]
    accepted = [
        {**row, "record_id": row.get("record_id") or f"{row.get('item_id')}:{row.get('level')}:accepted"}
        for row in accepted
    ]
    rejected = [
        {**row, "record_id": row.get("record_id") or f"{row.get('item_id')}:{row.get('reason', row.get('level'))}:rejected"}
        for row in rejected
    ]
    diagnostics = [
        {**row, "record_id": row.get("record_id") or f"{row.get('item_id')}:{row.get('reason', row.get('level'))}:diagnostic"}
        for row in diagnostics
    ]
    append_jsonl_unique(_jsonl_path(state, "accepted.jsonl"), accepted, key="record_id")
    append_jsonl_unique(_jsonl_path(state, "rejected.jsonl"), rejected, key="record_id")
    append_jsonl_unique(_jsonl_path(state, "diagnostics.jsonl"), diagnostics, key="record_id")
    _append_experiment_log(
        state,
        [
            f"- item_id: {item.get('item_id')}",
            f"  processed_at: {datetime.now(timezone.utc).isoformat()}",
            f"  runtime_type: {item.get('runtime_type')}",
            f"  accepted: {len(accepted)}",
            f"  rejected: {len(rejected)}",
            f"  diagnostics: {len(diagnostics)}",
            f"  next: {'validate proposal' if state.get('max_items') is not None and int(state.get('processed', 0)) + 1 >= int(state.get('max_items')) else 'select next item'}",
            "",
        ],
    )
    return {
        "processed": int(state.get("processed", 0)) + 1,
        "accepted": int(state.get("accepted", 0)) + len(accepted),
        "rejected": int(state.get("rejected", 0)) + len(rejected),
        "diagnostics": int(state.get("diagnostics", 0)) + len(diagnostics),
        "current_item": None,
        "current_candidates": [],
        "accepted_rows": [],
        "rejected_rows": [],
        "diagnostic_rows": [],
    }


def should_continue(state: ProducerState) -> Literal["select_next_item", "validate_proposed_artifact"]:
    max_items = state.get("max_items")
    if max_items is not None and int(state.get("processed", 0)) >= int(max_items):
        return "validate_proposed_artifact"
    return "select_next_item"


def validate_proposed_artifact_node(state: ProducerState) -> ProducerState:
    proposed = Path(state["proposed_out"])
    if not proposed.exists():
        ensure_proposed_artifact(state["profiles_path"], proposed, run_id=state["run_id"])
    errors = validate_proposed_artifact(proposed)
    if errors:
        return {"errors": [*state.get("errors", []), *[{"stage": "validate", "message": e} for e in errors]], "needs_review": False}
    return {"needs_review": True}


def route_after_validate(state: ProducerState) -> Literal["review_interrupt", "finalize_run"]:
    return "review_interrupt" if state.get("needs_review") else "finalize_run"


def review_interrupt(state: ProducerState) -> ProducerState:
    decision = state.get("review_decision")
    if decision is None:
        decision = interrupt(
            {
                "run_id": state["run_id"],
                "proposed_out": state["proposed_out"],
                "coverage": str(_jsonl_path(state, "coverage.json")),
                "accepted": state.get("accepted", 0),
                "rejected": state.get("rejected", 0),
                "diagnostics": state.get("diagnostics", 0),
            }
        )
    return {"review_decision": str(decision)}


def _write_review_report(state: ProducerState, status: str) -> None:
    run_dir = Path(state["run_dir"])
    proposals = read_jsonl(run_dir / "proposals.jsonl")
    accepted = read_jsonl(run_dir / "accepted.jsonl")
    diagnostics = read_jsonl(run_dir / "diagnostics.jsonl")
    rejected = read_jsonl(run_dir / "rejected.jsonl")
    lines = [
        "# Lattice Producer Review Report",
        "",
        f"- run_id: {state['run_id']}",
        f"- status: {status}",
        f"- proposed_out: {state['proposed_out']}",
        f"- accepted: {len(accepted)}",
        f"- rejected: {len(rejected)}",
        f"- diagnostics: {len(diagnostics)}",
        "",
        "## Accepted Levels",
        "",
    ]
    if accepted:
        for row in accepted:
            grounding = row.get("level_grounding") or {}
            aliases = ", ".join(str(alias) for alias in row.get("aliases", []) if str(alias).strip()) or "none"
            lines.extend(
                [
                    f"- item_id: {row.get('item_id')}",
                    f"  level: {row.get('level')}",
                    f"  aliases: {aliases}",
                    f"  count: {row.get('level_count')}",
                    f"  grounding_status: {grounding.get('status')}",
                    f"  source_family: {grounding.get('source_family') or row.get('source_family')}",
                    f"  selector: {grounding.get('selector') or row.get('selector')}",
                ]
            )
            if grounding.get("count_evidence") or row.get("count_evidence"):
                lines.append(f"  count_evidence: {grounding.get('count_evidence') or row.get('count_evidence')}")
            if row.get("rationale"):
                lines.append(f"  rationale: {row.get('rationale')}")
    else:
        lines.append("- none")
    lines.extend(["", "## Diagnostics", ""])
    if diagnostics:
        for row in diagnostics:
            lines.extend(
                [
                    f"- item_id: {row.get('item_id')}",
                    f"  reason: {row.get('reason')}",
                    f"  surface: {row.get('surface')}",
                    f"  level: {row.get('level')}",
                    f"  count: {row.get('level_count')}",
                    f"  grounding_status: {(row.get('level_grounding') or {}).get('status')}",
                ]
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Rejected", ""])
    if rejected:
        for row in rejected:
            lines.extend([f"- item_id: {row.get('item_id')}", f"  reason: {row.get('reason')}"])
    else:
        lines.append("- none")
    lines.extend(["", "## Raw Proposal Summary", ""])
    if proposals:
        for row in proposals:
            candidates = extract_candidate_levels(row)
            candidate_text = ", ".join(candidate["level"] for candidate in candidates) or "none"
            aliases = ", ".join(str(alias) for alias in row.get("aliases", []) if str(alias).strip()) or "none"
            lines.extend([f"- item_id: {row.get('item_id')}", f"  aliases: {aliases}", f"  candidate_levels: {candidate_text}"])
            for candidate in candidates:
                if candidate.get("proposed_count") or candidate.get("count_evidence") or candidate.get("rationale"):
                    lines.append(
                        "  - level: {level}; proposed_count: {count}; evidence: {evidence}; rationale: {rationale}".format(
                            level=candidate.get("level"),
                            count=candidate.get("proposed_count", candidate.get("level_count")),
                            evidence=candidate.get("count_evidence", ""),
                            rationale=candidate.get("rationale", ""),
                        )
                    )
    else:
        lines.append("- none")
    lines.append("")
    (run_dir / "REVIEW_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def finalize_run(state: ProducerState) -> ProducerState:
    status = "failed" if state.get("errors") else "complete"
    if state.get("review_decision") == "reject":
        status = "rejected"
    elif state.get("review_decision") == "approve" and state.get("allow_canonical_overwrite"):
        shutil.copyfile(state["proposed_out"], state["profiles_path"])
    coverage = {
        "run_id": state["run_id"],
        "accepted": state.get("accepted", 0),
        "rejected": state.get("rejected", 0),
        "diagnostics": state.get("diagnostics", 0),
        "status": status,
    }
    Path(state["run_dir"], "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    _write_review_report(state, status)
    _append_experiment_log(
        state,
        [
            "## Final Status",
            "",
            f"- status: {status}",
            f"- accepted: {state.get('accepted', 0)}",
            f"- rejected: {state.get('rejected', 0)}",
            f"- diagnostics: {state.get('diagnostics', 0)}",
            "- next: review proposed artifact",
            "",
        ],
    )
    return {"final_status": status}


def build_graph() -> StateGraph:
    graph = StateGraph(ProducerState)
    graph.add_node("initialize_run", initialize_run)
    graph.add_node("build_category_coverage", build_category_coverage_node)
    graph.add_node("build_or_load_queue", build_or_load_queue_node)
    graph.add_node("select_next_item", select_next_item)
    graph.add_node("generate_universe_entries", generate_universe_entries)
    graph.add_node("deterministic_lookup", deterministic_lookup)
    graph.add_node("propose_with_llama_swap", propose_with_llama_swap_node)
    graph.add_node("compile_level_counts", compile_level_counts_node)
    graph.add_node("gate_candidates", gate_candidates_node)
    graph.add_node("persist_proposed_artifact", persist_proposed_artifact_node)
    graph.add_node("record_item_result", record_item_result)
    graph.add_node("validate_proposed_artifact", validate_proposed_artifact_node)
    graph.add_node("review_interrupt", review_interrupt)
    graph.add_node("finalize_run", finalize_run)
    graph.add_edge(START, "initialize_run")
    graph.add_edge("initialize_run", "build_category_coverage")
    graph.add_edge("build_category_coverage", "build_or_load_queue")
    graph.add_edge("build_or_load_queue", "select_next_item")
    graph.add_conditional_edges("select_next_item", route_selected)
    graph.add_edge("generate_universe_entries", "compile_level_counts")
    graph.add_conditional_edges("deterministic_lookup", route_after_deterministic)
    graph.add_conditional_edges("propose_with_llama_swap", route_after_proposal)
    graph.add_edge("compile_level_counts", "gate_candidates")
    graph.add_conditional_edges("gate_candidates", route_after_gate)
    graph.add_edge("persist_proposed_artifact", "record_item_result")
    graph.add_conditional_edges("record_item_result", should_continue)
    graph.add_conditional_edges("validate_proposed_artifact", route_after_validate)
    graph.add_edge("review_interrupt", "finalize_run")
    graph.add_edge("finalize_run", END)
    return graph


def _checkpointer(run_dir: str | Path):
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(run_dir) / "checkpoints.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn)


def run_producer(
    *,
    run_dir: str | Path,
    profiles_path: str | Path,
    proposed_out: str | Path,
    queue_path: str | Path | None = None,
    run_id: str | None = None,
    base_url: str = "http://localhost:8060/v1",
    model: str = "",
    escalation_model: str | None = None,
    offline_only: bool = False,
    max_items: int | None = None,
    max_context_rows: int = 8,
    max_generated_entries_per_category: int = 20,
    review_decision: str | None = None,
    allow_canonical_overwrite: bool = False,
    category: str | None = None,
) -> ProducerState:
    run_id = run_id or Path(run_dir).name
    state = make_initial_state(
        run_id=run_id,
        run_dir=run_dir,
        profiles_path=profiles_path,
        proposed_out=proposed_out,
        queue_path=queue_path,
        base_url=base_url,
        model=model,
        escalation_model=escalation_model,
        offline_only=offline_only,
        max_items=max_items,
        max_context_rows=max_context_rows,
        max_generated_entries_per_category=max_generated_entries_per_category,
        review_decision=review_decision,
        allow_canonical_overwrite=allow_canonical_overwrite,
    )
    if category:
        state["category"] = category
    app = build_graph().compile(checkpointer=_checkpointer(run_dir))
    result = app.invoke(state, {"configurable": {"thread_id": thread_id_for_run(run_id)}})
    if isinstance(result, Command):
        raise RuntimeError("unexpected command result")
    return result
