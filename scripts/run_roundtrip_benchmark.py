#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from bench.metrics import score_traces
from bench.report import write_json_outputs, write_markdown_report
from bench.runner import StubRemote, run_suite, validate_model_args, write_dry_run, write_traces
from bench.schema import BenchmarkConfig


def main() -> int:
    args = _parse_args()
    config = BenchmarkConfig(
        suite=args.suite,
        limit=args.limit,
        seed=args.seed,
        detector_version=args.detector_version,
        substitutor_version=args.substitutor,
        privacy_setting=args.privacy_setting,
        remote_model=args.remote_model,
        extractor_version=args.extractor_version,
        attacker_version=args.attacker_version,
        output_dir=str(args.output_dir),
        detector_model=args.detector_model,
        detector_fine_dem=args.detector_fine_dem,
        extractor_model=args.extractor_model,
        attack_docp_model=args.attack_docp_model,
        attack_reconstruction_model=args.attack_reconstruction_model,
        attack_leak_model=args.attack_leak_model,
    )
    out = Path(args.output_dir)
    if args.dry_run:
        items = write_dry_run(config, out)
        print(f"dry-run: wrote {len(items)} items to {out}")
        return 0

    remote = StubRemote() if args.stub_remote else None
    try:
        validate_model_args(config, use_remote=remote is None)
        traces = run_suite(config, remote=remote)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    write_dry_run(config, out)
    write_traces(out, traces)
    scores = score_traces(traces, config)
    write_json_outputs(scores, out)
    write_markdown_report(scores, out)
    print(f"scored {len(traces)} traces; wrote {out / 'traces.jsonl'}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the roundtrip privacy/utility benchmark.")
    parser.add_argument("--suite", default="primary_utility")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--detector-version", default="current")
    parser.add_argument("--detector-model")
    parser.add_argument("--detector-fine-dem", action="store_true")
    parser.add_argument("--substitutor", default="current")
    parser.add_argument("--privacy-setting", default="tau=0.02")
    parser.add_argument("--remote-model")
    parser.add_argument("--extractor-version", default="current")
    parser.add_argument("--extractor-model")
    parser.add_argument("--attacker-version", default="offline-v1")
    parser.add_argument("--attack-docp-model")
    parser.add_argument("--attack-reconstruction-model")
    parser.add_argument("--attack-leak-model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stub-remote", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
