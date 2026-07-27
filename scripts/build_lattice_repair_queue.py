#!/usr/bin/env python
"""Compile environment and QA audit evidence into a lattice-producer repair queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cloak.lattice.producer.repair_queue import build_repair_queue


def _load_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"audit must be a JSON object: {path}")
    nested = payload.get("qa_audit")
    if isinstance(nested, dict):
        return nested
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment-audit", action="append", required=True,
        help="environment-audit JSON sidecar; may be specified more than once",
    )
    parser.add_argument(
        "--qa-audit", action="append", default=[],
        help="QA audit JSON sidecar or utility artifact; may be specified more than once",
    )
    parser.add_argument("--profiles", default="data/lattice_profiles/lattice_profiles.json")
    parser.add_argument("--out", required=True, help="producer-ready JSONL queue")
    parser.add_argument(
        "--triage-out",
        help="non-runnable JSONL requiring explicit identity-evidence promotion",
    )
    parser.add_argument("--report", help="JSON report for manual and non-producer evidence")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    environment_audits = [_load_audit(Path(path)) for path in args.environment_audit]
    qa_audits = [_load_audit(Path(path)) for path in args.qa_audit]
    queue = build_repair_queue(
        environment_audits, qa_audits, profiles_path=args.profiles,
    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(
        json.dumps(item, sort_keys=True) + "\n"
        for item in queue["producer_items"]
    ))
    triage_output = Path(args.triage_out) if args.triage_out else output.with_name(
        output.name + ".triage.jsonl",
    )
    triage_output.write_text("".join(
        json.dumps(item, sort_keys=True) + "\n"
        for item in queue["triage_items"]
    ))
    report = Path(args.report) if args.report else output.with_name(
        output.name + ".report.json",
    )
    report.write_text(json.dumps({
        "producer_item_count": len(queue["producer_items"]),
        "triage_item_count": len(queue["triage_items"]),
        "profile_triage": queue["triage_items"],
        "manual_review": queue["manual_review"],
        "non_producer": queue["non_producer"],
    }, indent=1, sort_keys=True))
    print(
        f"wrote {output}: producer_items={len(queue['producer_items'])} "
        f"triage_items={len(queue['triage_items'])} "
        f"manual_review={len(queue['manual_review'])} "
        f"non_producer={len(queue['non_producer'])}; triage={triage_output}; report={report}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
