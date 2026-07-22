#!/usr/bin/env python3
"""Build a matched Ranker-v2 architecture-spike report from local measurements."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from cloak.train.ranker_architecture_diagnostics import (
    ArmMeasurement,
    MatchedArmContract,
    run_architecture_spike,
)


DEFAULT_OUTPUT = Path(
    "results/ranker_v2/architecture/spike/architecture-spike-report.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _read_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("architecture measurements must be a mapping")
    required = {
        "matched_contract", "measurements", "registered_thresholds",
        "aci_context", "non_aci_manifest_hash",
    }
    if set(payload) != required:
        raise ValueError("architecture measurement manifest fields are incomplete")
    return payload


def _write_report(path: Path, report: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    payload = _read_payload(args.measurements)
    contract = MatchedArmContract.from_mapping(payload["matched_contract"])
    rows = tuple(
        ArmMeasurement.from_mapping(row) for row in payload["measurements"]
    )
    by_identity = {(row.family, row.arm): row for row in rows}
    if len(by_identity) != len(rows):
        raise ValueError("architecture measurement manifest repeats an arm")

    def evaluator(family, arm, supplied_contract):
        if supplied_contract != contract:
            raise ValueError("architecture runner contract changed during evaluation")
        try:
            return by_identity[(family, arm)]
        except KeyError as exc:
            raise ValueError(f"architecture measurement is missing: {family}:{arm}") from exc

    report = run_architecture_spike(
        contract,
        evaluator,
        registered_thresholds=payload["registered_thresholds"],
        aci_context=payload["aci_context"],
        non_aci_manifest_hash=payload["non_aci_manifest_hash"],
    )
    _write_report(args.out, report)
    print(
        f"ARCHITECTURE_SPIKE {report['promotion_verdict']['verdict']} "
        f"out={args.out}",
        flush=True,
    )
    return report


if __name__ == "__main__":
    main()
