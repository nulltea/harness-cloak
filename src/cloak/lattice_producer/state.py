"""Typed state for the LangGraph lattice producer."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, TypedDict


QWEN36_ESCALATION_MODEL = "Qwen3.6-35B-A3B"


class ProducerState(TypedDict, total=False):
    run_id: str
    run_dir: str
    profiles_path: str
    proposed_out: str
    queue_path: str
    current_item: dict[str, Any] | None
    current_candidates: list[dict[str, Any]]
    accepted_rows: list[dict[str, Any]]
    rejected_rows: list[dict[str, Any]]
    diagnostic_rows: list[dict[str, Any]]
    queue_index: int
    prompt_version: str
    model: str
    escalation_model: str | None
    base_url: str
    offline_only: bool
    max_items: int | None
    normalize_every: int
    queue_exhausted: bool
    max_context_rows: int
    max_generated_entries_per_category: int
    thinking_budget_tokens: int
    processed: int
    accepted: int
    rejected: int
    diagnostics: int
    proposed_persisted: int
    coverage_gaps: int
    generated_entries: int
    errors: list[dict[str, Any]]
    needs_review: bool
    review_decision: str | None
    allow_canonical_overwrite: bool
    category: str | None
    categories: list[str]
    final_status: str | None


def thread_id_for_run(run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:24]
    return f"lattice-producer:{digest}"


def make_initial_state(
    *,
    run_id: str,
    run_dir: str | Path,
    profiles_path: str | Path,
    proposed_out: str | Path,
    queue_path: str | Path | None = None,
    prompt_version: str = "lattice-producer-v1",
    model: str = QWEN36_ESCALATION_MODEL,
    escalation_model: str | None = None,
    base_url: str = "http://localhost:8060/v1",
    offline_only: bool = False,
    max_items: int | None = None,
    normalize_every: int = 50,
    max_context_rows: int = 8,
    max_generated_entries_per_category: int = 20,
    thinking_budget_tokens: int = -1,
    review_decision: str | None = None,
    allow_canonical_overwrite: bool = False,
    category: str | None = None,
    categories: list[str] | None = None,
) -> ProducerState:
    run_dir = Path(run_dir)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "profiles_path": str(profiles_path),
        "proposed_out": str(proposed_out),
        "queue_path": str(queue_path or run_dir / "queue.jsonl"),
        "current_item": None,
        "current_candidates": [],
        "accepted_rows": [],
        "rejected_rows": [],
        "diagnostic_rows": [],
        "queue_index": 0,
        "prompt_version": prompt_version,
        "model": QWEN36_ESCALATION_MODEL,
        "escalation_model": escalation_model or QWEN36_ESCALATION_MODEL,
        "base_url": base_url,
        "offline_only": offline_only,
        "max_items": max_items,
        "normalize_every": normalize_every,
        "queue_exhausted": False,
        "max_context_rows": max_context_rows,
        "max_generated_entries_per_category": max_generated_entries_per_category,
        "thinking_budget_tokens": thinking_budget_tokens,
        "processed": 0,
        "accepted": 0,
        "rejected": 0,
        "diagnostics": 0,
        "proposed_persisted": 0,
        "coverage_gaps": 0,
        "generated_entries": 0,
        "errors": [],
        "needs_review": False,
        "review_decision": review_decision,
        "allow_canonical_overwrite": allow_canonical_overwrite,
        "category": category,
        "categories": categories or [],
        "final_status": None,
    }
