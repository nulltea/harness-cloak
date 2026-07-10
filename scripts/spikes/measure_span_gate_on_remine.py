"""One-off: measure the span gate on the cached large re-mine spans (no detector run).

Feeds results/mined_lattice_profile_spans_large.jsonl through the miner's gated
span-processing and reports per-type kept/dropped/retyped deltas vs a no-gate baseline
(gate artifacts temporarily pointed at empty paths). Term listings go to the report JSON.

  PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/measure_span_gate_on_remine.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from build_mined_lattice_profiles import DetectedSpan, build_rows_for_test

SPANS = Path("results/mined_lattice_profile_spans_large.jsonl")
OUT = Path("results/span_gate_mine_report.json")


def main() -> None:
    spans = [DetectedSpan(r["surface"], r["detector_label"], r["doc_id"], float(r["score"]))
             for r in map(json.loads, SPANS.read_text().splitlines()) if r]
    rows, stats = build_rows_for_test(spans)
    per_type_kept = {rt: len(entries) for rt, entries in rows.items()}
    report = {
        "spans_in": len(spans),
        "stats": {k: v for k, v in stats.items() if isinstance(v, (int, float, str))},
        "kept_per_type": per_type_kept,
        "kept_surfaces": {rt: sorted(entries) for rt, entries in rows.items()},
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    counts = Counter({k: v for k, v in stats.items() if isinstance(v, int)})
    print("spans_in:", len(spans))
    print("stats:", dict(counts))
    print("kept_per_type:", per_type_kept)
    print("report ->", OUT)


if __name__ == "__main__":
    main()
