#!/usr/bin/env python
"""Run the LangGraph generalization lattice producer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cloak.lattice_producer.graph import run_producer
from cloak.lattice_producer.coverage import normalize_category_filters

PROPOSED_ROOT = Path("data/lattice_profiles/proposed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--queue")
    parser.add_argument("--base-url", default="http://localhost:8060/v1")
    parser.add_argument("--model", default="")
    parser.add_argument("--escalation-model")
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--review-decision", choices=["approve", "reject", "approve-proposed-only"])
    parser.add_argument("--allow-canonical-overwrite", action="store_true")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-generated-entries-per-category", type=int, default=20)
    parser.add_argument("--max-context-rows", type=int, default=8)
    parser.add_argument("--thinking-budget-tokens", type=int, default=-1)
    parser.add_argument("--category", action="append", default=[])
    args = parser.parse_args(argv)
    args.category = normalize_category_filters(categories=args.category)
    return args


def main() -> int:
    args = parse_args()
    if args.workers != 1:
        raise SystemExit("--workers > 1 is intentionally unsupported until a saturation probe justifies it")
    out_path = Path(args.out)
    proposed_root = (Path.cwd() / PROPOSED_ROOT).resolve()
    resolved_out = (Path.cwd() / out_path).resolve() if not out_path.is_absolute() else out_path.resolve()
    if proposed_root not in [resolved_out, *resolved_out.parents]:
        raise SystemExit(f"--out must be under {PROPOSED_ROOT}/")
    result = run_producer(
        run_dir=Path(args.run_dir),
        profiles_path=Path(args.profiles),
        proposed_out=out_path,
        queue_path=Path(args.queue) if args.queue else None,
        base_url=args.base_url,
        model=args.model,
        escalation_model=args.escalation_model,
        offline_only=args.offline_only,
        max_items=args.max_items,
        max_context_rows=args.max_context_rows,
        max_generated_entries_per_category=args.max_generated_entries_per_category,
        thinking_budget_tokens=args.thinking_budget_tokens,
        review_decision=args.review_decision,
        allow_canonical_overwrite=args.allow_canonical_overwrite,
        categories=args.category,
    )
    print(
        "status={status} accepted={accepted} rejected={rejected} diagnostics={diagnostics} proposed={proposed}".format(
            status=result.get("final_status"),
            accepted=result.get("accepted", 0),
            rejected=result.get("rejected", 0),
            diagnostics=result.get("diagnostics", 0),
            proposed=out_path,
        )
    )
    return 1 if result.get("final_status") in {"failed", "rejected"} else 0


if __name__ == "__main__":
    sys.exit(main())
