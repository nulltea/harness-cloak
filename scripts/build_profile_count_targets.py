"""Build strict or diagnostic own-profile count targets for Ranker-v2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cloak.ranker.count_reward import CountGateError
from cloak.ranker.profile_count import build_profile_count_targets


def _canonical_json(payload: dict) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ) + "\n"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gate-report", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--strict", action="store_true",
        help="fail closed unless every selectable decision has validated targets",
    )
    mode.add_argument(
        "--diagnostic", action="store_true",
        help="publish a diagnostics-only artifact with explicit eligibility flags",
    )
    args = parser.parse_args()

    environment = json.loads(Path(args.environment).read_text())
    try:
        artifact = build_profile_count_targets(environment, strict=args.strict)
    except CountGateError as error:
        _write_json(Path(args.gate_report), error.report)
        summary = error.report["summary"]
        print(
            "profile count gate FAIL: "
            f"eligible={summary['privacy_head_eligible_decisions']} "
            f"ineligible={summary['privacy_head_ineligible_decisions']} "
            f"gaps={summary['gap_count']}",
            file=sys.stderr,
        )
        for decision in error.report.get("decisions", []):
            if not decision["privacy_head_eligible"]:
                print(
                    f"GAP {decision['runtime_type']} {decision.get('profile_id')} "
                    f"{decision['decision_id']}: "
                    + ",".join(decision.get("gap_reasons", [])),
                    file=sys.stderr,
                )
        raise SystemExit(2) from None

    report = artifact["gate_report"]
    _write_json(Path(args.gate_report), report)
    _write_json(Path(args.out), artifact)
    summary = report["summary"]
    print(
        f"profile count artifact published: mode={artifact['gate_mode']} "
        f"strict_verdict={report['strict_verdict']} "
        f"eligible={summary['privacy_head_eligible_decisions']} "
        f"ineligible={summary['privacy_head_ineligible_decisions']} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
