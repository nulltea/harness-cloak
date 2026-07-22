"""Build the immutable Ranker-v2 count reward state and clause-level gate report."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from cloak.train.count_reward import CountGateError, _build_count_reward_state


COUNT_DEFECT_ISSUE = Path("docs/issues/2026-07-22-lattice-profile-count-defects.md")


def _canonical_json(payload: dict) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ) + "\n"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(payload))


def _append_gap_issue_summary(report: dict, issue_path: Path) -> None:
    gaps = report.get("gaps", [])
    if not gaps:
        return
    signature_payload = [
        (row["decision_id"], row["action_id"]) for row in gaps
    ]
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    marker = f"<!-- count-reward-provisional-gaps:{signature} -->"
    current = issue_path.read_text() if issue_path.exists() else "# Count data defects\n"
    if marker in current:
        return
    by_type = Counter(str(row["runtime_type"]) for row in gaps)
    lines = [
        "",
        "## Count-reward provisional gate inventory",
        "",
        marker,
        "",
        f"The provisional count gate confirmed **{len(gaps)}** unresolved level actions for",
        f"environment `{report.get('environment_hash')}`. These are existing profile-data",
        "defects, not runtime fallbacks: their entire decisions receive flat-count semantics",
        "until source evidence is repaired and confirmed.",
        "",
        "Gap actions by runtime type: "
        + ", ".join(f"`{key}` {by_type[key]}" for key in sorted(by_type))
        + ".",
        "",
        "Concrete gate inventory:",
        "",
    ]
    for row in gaps:
        lines.append(
            f"- `{row['runtime_type']}` / `{row.get('profile_id')}` / "
            f"`{row['decision_id']}` / `{row['action_id']}` / `{row.get('fill')}`: "
            + ", ".join(row.get("gap_reasons", []))
            + "."
        )
    issue_path.parent.mkdir(parents=True, exist_ok=True)
    issue_path.write_text(current.rstrip() + "\n" + "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gate-report", required=True)
    parser.add_argument(
        "--provisional", action="store_true",
        help="publish flat-count tags for coverage gaps; mappings/monotonicity remain hard",
    )
    args = parser.parse_args()
    environment = json.loads(Path(args.environment).read_text())
    try:
        state = _build_count_reward_state(environment, provisional=args.provisional)
    except CountGateError as error:
        _write_json(Path(args.gate_report), error.report)
        print(
            f"count gate FAIL: gaps={error.report['summary']['gap_count']} "
            f"missing_policy_mappings={len(error.report['missing_policy_mappings'])} "
            f"nonmonotone_profiles={len(error.report['nonmonotone_profiles'])}",
            file=sys.stderr,
        )
        for gap in error.report.get("gaps", []):
            print(
                f"GAP {gap['runtime_type']} {gap.get('profile_id')} "
                f"{gap['decision_id']} {gap['action_id']} {gap.get('fill')!r}: "
                + ",".join(gap.get("gap_reasons", [])),
                file=sys.stderr,
            )
        raise SystemExit(2) from None
    report = state["gate_report"]
    hash_payload = dict(state)
    encoded = json.dumps(
        hash_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    state["artifact_hash"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    _write_json(Path(args.gate_report), report)
    _write_json(Path(args.out), state)
    if args.provisional:
        _append_gap_issue_summary(report, COUNT_DEFECT_ISSUE)
    print(
        f"count gate {report['verdict']}: mode={report['mode']} "
        f"gaps={report['summary']['gap_count']} "
        f"tags={len(state['provisional_decision_tags'])} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
