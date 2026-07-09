#!/usr/bin/env python3
"""Cache-hot microbenchmark for the frozen extractor."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from cloak import frozen_extractor as fx
from extractor_determinism_gate import (
    DEFAULT_FIXTURES,
    install_stub_prepass,
    load_fixtures,
    stub_models,
    write_fixtures,
)


DEFAULT_OUTPUT = Path("results/extractor_microbench.json")


def run_benchmark(args: argparse.Namespace) -> dict:
    if args.make_fixtures:
        write_fixtures(args.fixtures)
    fixtures = load_fixtures(args.fixtures)

    if args.stub:
        install_stub_prepass()
        models = stub_models()
    else:
        models = fx.load_models(device=args.device)

    for record in fixtures:
        fx.extract(record["doc_p"], record["R"], record["out_p"], models=models)

    cuda_peak = None
    torch = _try_torch()
    if torch is not None and _cuda_available(torch):
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    timings = []
    residue_counts = Counter()
    for record in fixtures:
        start = time.perf_counter()
        _, stats = fx.extract(record["doc_p"], record["R"], record["out_p"], models=models)
        timings.append(time.perf_counter() - start)
        residue_counts[str(len(stats.get("entries", [])))] += 1

    if torch is not None and _cuda_available(torch):
        try:
            cuda_peak = int(torch.cuda.max_memory_allocated())
        except Exception:
            cuda_peak = None

    return {
        "device": args.device,
        "extractor_version": fx.extractor_version(),
        "fixtures": str(args.fixtures),
        "n_docs": len(fixtures),
        "p50_wall_seconds_per_doc": _percentile(timings, 50),
        "p95_wall_seconds_per_doc": _percentile(timings, 95),
        "peak_cuda_memory_allocated_bytes": cuda_peak,
        "residue_count_distribution": dict(sorted(residue_counts.items())),
        "stub": bool(args.stub),
    }


def _try_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _cuda_available(torch) -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    idx = round((percentile / 100) * (len(sorted_values) - 1))
    return float(sorted_values[idx])


def write_results(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--make-fixtures", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stub", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = run_benchmark(args)
    write_results(args.output, results)
    print(json.dumps(results, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
