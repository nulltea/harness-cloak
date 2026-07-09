#!/usr/bin/env python3
"""Chunk-bucketed run-quality re-measurement for the lattice producer.

Re-measures the "creativity collapse" signature documented in issue register #8
(docs/issues/2026-07-09-lattice-producer-generation-quality-issue-register.md) so a
post-overhaul run is directly comparable to the register's baseline tables.

Reads a completed *proposed* artifact plus the run's `accepted.jsonl` and prints:

1. Per 5 equal chronological chunks (accepted.jsonl file/append order = processing
   order of items): the % of items whose accepted levels are ALL generic sinks,
   the count of first-seen ("new") specific labels introduced in that chunk, and
   the top-3 most-frequent levels. Mirrors the register's per-chunk tables.
2. The chain-length histogram from the proposed artifact (how many entries have
   1, 2, 3... levels) — to confirm no length-1 entries survive the >=2-level floor
   from the overhaul.
3. The fraction of reused labels (labels appearing in >=2 entries) whose
   per-occurrence counts disagree by more than 4x — the count-agreement gate.

Generic-sink definition (operational, diagnostic-only — not production semantics):
a level string is a "generic sink" iff it is in `SINK_LEXICON` (the register's
observed catch-alls) OR it has <= `--sink-max-tokens` tokens (default 3) AND ends
with a broad head-word from `BROAD_HEADS` ("condition", "procedure", "service",
"activity", "substance", ...). This catches both the 2-token sinks ("medical
condition", "human activity") and the 3-token near-synonyms ("human medical
condition", "general medical condition") the register flagged. A "new specific
label" in a chunk = a level not seen in any earlier chunk AND not a generic sink.

Usage:
    python scripts/spikes/measure_lattice_run_quality.py <proposed.json> <accepted.jsonl>
    python scripts/spikes/measure_lattice_run_quality.py --self-check
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

# Register's observed catch-all labels, kept verbatim so anything matching them is
# always a sink regardless of the token heuristic.
SINK_LEXICON = {
    "medical condition",
    "human medical condition",
    "general medical condition",
    "medical procedure",
    "clinical service",
    "human activity",
    "chemical substance",
    "thing",
    "entity",
}

# Broad head-words: a short level ending in one of these is a generic sink.
BROAD_HEADS = {
    "condition", "disease", "disorder", "syndrome",
    "procedure", "service", "activity", "process", "intervention",
    "substance", "compound", "chemical", "agent", "material", "drug",
    "entity", "thing", "object", "concept", "phenomenon", "matter",
}


def is_generic_sink(level: str, max_tokens: int = 3) -> bool:
    lvl = level.strip().lower()
    if lvl in SINK_LEXICON:
        return True
    tokens = lvl.split()
    if 0 < len(tokens) <= max_tokens and tokens[-1] in BROAD_HEADS:
        return True
    return False


def load_items(accepted_path: Path) -> "OrderedDict[str, list[str]]":
    """Group accepted rows by item_id in first-appearance (=processing) order.

    Returns item_id -> list of level strings (chain for that item).
    """
    items: "OrderedDict[str, list[str]]" = OrderedDict()
    with accepted_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            item_id = str(row.get("item_id", ""))
            level = str(row.get("level", "")).strip()
            if not item_id or not level:
                continue
            items.setdefault(item_id, []).append(level)
    return items


def chunk_bounds(n: int, k: int = 5) -> "list[tuple[int, int]]":
    """Split n items into k as-even-as-possible contiguous chunks (remainder to
    the earliest chunks, matching the register's ~129/129/129/129/130 shape)."""
    base, rem = divmod(n, k)
    bounds, start = [], 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def measure_chunks(items: "OrderedDict[str, list[str]]", max_tokens: int, k: int = 5):
    item_list = list(items.items())  # [(item_id, [levels...])]
    n = len(item_list)
    seen_labels: set[str] = set()
    rows = []
    for ci, (lo, hi) in enumerate(chunk_bounds(n, k), start=1):
        chunk = item_list[lo:hi]
        if not chunk:
            rows.append((ci, 0, 0.0, 0, []))
            continue
        fully_generic = 0
        new_specific = 0
        level_freq: Counter[str] = Counter()
        for _item_id, levels in chunk:
            if levels and all(is_generic_sink(l, max_tokens) for l in levels):
                fully_generic += 1
            for l in levels:
                level_freq[l] += 1
        # New-specific-label pass: a label first appearing in this chunk that is
        # not a generic sink. Scan in item order, updating seen set as we go.
        for _item_id, levels in chunk:
            for l in levels:
                if l not in seen_labels:
                    if not is_generic_sink(l, max_tokens):
                        new_specific += 1
                    seen_labels.add(l)
        fg_pct = 100.0 * fully_generic / len(chunk)
        top3 = level_freq.most_common(3)
        rows.append((ci, len(chunk), fg_pct, new_specific, top3))
    return rows, n


def iter_entries(proposed: dict):
    """Yield (surface, entry_dict) over every profile entry in the artifact."""
    for _category, surfaces in (proposed.get("profiles") or {}).items():
        if isinstance(surfaces, dict):
            for surface, entry in surfaces.items():
                if isinstance(entry, dict):
                    yield surface, entry


def chain_length_histogram(proposed: dict) -> "Counter[int]":
    hist: Counter[int] = Counter()
    for _surface, entry in iter_entries(proposed):
        hist[len(entry.get("levels") or [])] += 1
    return hist


def count_disagreement(proposed: dict) -> "tuple[int, int, list]":
    """Fraction of reused labels (>=2 entries) whose per-occurrence counts span
    more than 4x. Returns (n_disagree, n_reused, worst_examples)."""
    label_counts: dict[str, list[float]] = {}
    for _surface, entry in iter_entries(proposed):
        for label, cnt in (entry.get("level_counts") or {}).items():
            try:
                c = float(cnt)
            except (TypeError, ValueError):
                continue
            if c > 0:
                label_counts.setdefault(str(label), []).append(c)
    reused = {lbl: cs for lbl, cs in label_counts.items() if len(cs) >= 2}
    disagree = []
    for lbl, cs in reused.items():
        lo, hi = min(cs), max(cs)
        ratio = hi / lo
        if ratio > 4.0:
            disagree.append((lbl, len(cs), lo, hi, ratio))
    disagree.sort(key=lambda r: r[4], reverse=True)
    return len(disagree), len(reused), disagree[:5]


def report(proposed_path: Path, accepted_path: Path, max_tokens: int) -> None:
    items = load_items(accepted_path)
    proposed = json.loads(proposed_path.read_text())

    print(f"proposed artifact : {proposed_path}")
    print(f"accepted.jsonl    : {accepted_path}")
    print(f"generic-sink rule : lexicon + <= {max_tokens} tokens ending in a broad head-word")
    print()

    # 1. Per-chunk temporal collapse table (register #8).
    rows, n = measure_chunks(items, max_tokens)
    print(f"=== 1. Per-chunk run quality ({n} items, 5 chronological chunks) ===")
    print(f"{'chunk':<6}{'items':>6}{'fully_generic':>15}{'new_specific':>14}   top-3 levels (freq)")
    for ci, size, fg_pct, new_spec, top3 in rows:
        top = ", ".join(f"{lbl} ({c})" for lbl, c in top3) or "-"
        print(f"{ci:<6}{size:>6}{fg_pct:>14.1f}%{new_spec:>14}   {top}")
    print()

    # 2. Chain-length histogram (>=2-level floor check).
    hist = chain_length_histogram(proposed)
    total_entries = sum(hist.values())
    print(f"=== 2. Chain-length histogram ({total_entries} entries) ===")
    for length in sorted(hist):
        print(f"  length {length}: {hist[length]}")
    len1 = hist.get(1, 0)
    len0 = hist.get(0, 0)
    flag = "OK (no length-1 entries)" if (len1 == 0 and len0 == 0) else \
        f"WARNING: {len1} length-1 and {len0} length-0 entries remain (>=2-level floor breached)"
    print(f"  -> {flag}")
    print()

    # 3. Count-disagreement rate (count-agreement gate).
    n_dis, n_reused, worst = count_disagreement(proposed)
    rate = (100.0 * n_dis / n_reused) if n_reused else 0.0
    print("=== 3. Count-disagreement among reused labels (>4x span) ===")
    print(f"  reused labels (>=2 entries): {n_reused}")
    print(f"  disagreeing (>4x span)     : {n_dis} ({rate:.1f}%)")
    if worst:
        print("  worst offenders:")
        for lbl, occ, lo, hi, ratio in worst:
            print(f"    {lbl!r}: {occ} occ, {lo:g} -> {hi:g} ({ratio:.0f}x)")


def demo() -> None:
    """Self-check: asserts chunking, new-label, and count-disagreement logic on
    known synthetic input. No framework — pure asserts."""
    import tempfile

    # Generic-sink classification.
    assert is_generic_sink("medical condition")
    assert is_generic_sink("human medical condition")      # 3-token near-synonym
    assert is_generic_sink("human activity")
    assert not is_generic_sink("alcoholic cardiomyopathy")  # 2 tokens, non-broad head
    assert not is_generic_sink("acute myocardial infarction")  # 3 tokens, non-broad head

    # Chunking: 12 items into 5 chunks -> 3,3,2,2,2 (remainder to earliest).
    assert chunk_bounds(12, 5) == [(0, 3), (3, 6), (6, 8), (8, 10), (10, 12)]
    assert chunk_bounds(5, 5) == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    # Build a synthetic accepted.jsonl: 10 items. Chunk1&2 (items 0-3) coin fresh
    # specific labels; from chunk3 on every item collapses to a generic sink only.
    accepted_rows = []
    specifics = ["asthma", "eczema", "gout", "lupus"]  # items 0-3, each also has a sink
    for i in range(10):
        item_id = f"health-condition:item{i}"
        if i < 4:
            # specific + generic parent -> NOT fully generic
            accepted_rows.append({"item_id": item_id, "level": specifics[i]})
            accepted_rows.append({"item_id": item_id, "level": "medical condition"})
        else:
            # collapsed: two generic sinks only -> fully generic
            accepted_rows.append({"item_id": item_id, "level": "human medical condition"})
            accepted_rows.append({"item_id": item_id, "level": "medical condition"})

    with tempfile.TemporaryDirectory() as td:
        acc = Path(td) / "accepted.jsonl"
        acc.write_text("\n".join(json.dumps(r) for r in accepted_rows) + "\n")
        items = load_items(acc)
        assert len(items) == 10
        assert items["health-condition:item0"] == ["asthma", "medical condition"]

        rows, n = measure_chunks(items, max_tokens=3)
        assert n == 10
        # chunk_bounds(10,5) = (0,2),(2,4),(4,6),(6,8),(8,10)
        # chunk1 (items 0,1): 0 fully-generic, 2 new specific (asthma,eczema)
        c1 = rows[0]
        assert c1[1] == 2 and c1[2] == 0.0 and c1[3] == 2, c1
        # chunk2 (items 2,3): 0 fully-generic, 2 new specific (gout,lupus)
        c2 = rows[1]
        assert c2[1] == 2 and c2[2] == 0.0 and c2[3] == 2, c2
        # chunk3 (items 4,5): both fully-generic, 0 new specific (only sinks)
        c3 = rows[2]
        assert c3[1] == 2 and c3[2] == 100.0 and c3[3] == 0, c3
        # top-3 in chunk3: "medical condition" and "human medical condition"
        top_labels = {lbl for lbl, _ in c3[4]}
        assert "medical condition" in top_labels, c3[4]

    # Count-disagreement + chain-length histogram on a synthetic artifact.
    proposed = {
        "profiles": {
            "health-condition": {
                "s1": {"levels": ["asthma", "medical condition"],
                       "level_counts": {"asthma": 1000.0, "medical condition": 4100.0}},
                "s2": {"levels": ["eczema", "medical condition"],
                       "level_counts": {"eczema": 500.0, "medical condition": 8500000000.0}},  # 2M x disagreement
                "s3": {"levels": ["gout", "chronic disease", "medical condition"],
                       "level_counts": {"gout": 200.0, "chronic disease": 3000.0,
                                        "medical condition": 4200.0}},  # within 4x of s1's 4100
            }
        }
    }
    hist = chain_length_histogram(proposed)
    assert hist == Counter({2: 2, 3: 1}), hist  # no length-1 entries
    n_dis, n_reused, worst = count_disagreement(proposed)
    # "medical condition" reused 3x: 4100 / 8.5e9 -> >4x disagreement.
    assert n_reused == 1, n_reused  # only "medical condition" appears in >=2 entries
    assert n_dis == 1, n_dis
    assert worst[0][0] == "medical condition"

    print("self-check OK: sink classification, chunking, new-label counting, "
          "chain-length histogram, and count-disagreement all pass.")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("proposed", nargs="?", help="proposed artifact JSON")
    ap.add_argument("accepted", nargs="?", help="run's accepted.jsonl")
    ap.add_argument("--sink-max-tokens", type=int, default=3,
                    help="max token count for the broad-head-word sink heuristic (default 3)")
    ap.add_argument("--self-check", action="store_true", help="run the built-in self-check and exit")
    args = ap.parse_args(argv)

    if args.self_check:
        demo()
        return 0
    if not args.proposed or not args.accepted:
        ap.error("both <proposed.json> and <accepted.jsonl> are required (or pass --self-check)")
    report(Path(args.proposed), Path(args.accepted), args.sink_max_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
