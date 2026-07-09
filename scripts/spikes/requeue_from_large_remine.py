"""Re-mine clinical samples with the fixed chunker + gliner-pii-large, then update the full-run queue.

Mining uses the exact miner path (scripts/build_mined_lattice_profiles.detect_clinical_spans) which
now inherits the fixed _chunks (word-boundary + overlap windows) and the encoder-window word cap —
the cap actually binds for large (max_len 768 vs 2048 base).

Queue update rules (provenance-safe; matching uses the miner's own ProfileIndex fuzzy matcher,
not exact keys, so singular/plural/punctuation variants of a surface count as "the same surface" —
otherwise 102 still-detected removals and 88 near-duplicate additions leak through):
- REMOVE a queue item only if the OLD mining run detected its surface (results/
  mined_lattice_profile_spans.jsonl, stock base model) AND the new run does NOT fuzzy-detect it —
  items sourced from fine/common profiles or producer proposals are never touched.
- ADD a surface the new run detects that the old run did not fuzzy-detect and the queue does not
  fuzzy-contain; skip generic surfaces (miner rule), surfaces covered by the common profile (miner
  rule), cross-type confirmed noise surfaces, and families outside the queue's three (drug /
  health-condition / medical-procedure). Aliases copied from a fine-profile match when one exists
  (mirrors the queue builder).

    PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/requeue_from_large_remine.py            # mine + dry-run
    PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/requeue_from_large_remine.py --apply    # reuse spans, write queue
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from build_mined_lattice_profiles import (
    LABEL_TO_RUNTIME_TYPE,
    ProfileIndex,
    _is_generic_surface,
    _norm,
    detect_clinical_spans,
)
from cloak.detect import is_noise_span

QUEUE = Path("data/lattice_runs/full-run/queue.jsonl")
OLD_SPANS = Path("results/mined_lattice_profile_spans.jsonl")
NEW_SPANS = Path("results/mined_lattice_profile_spans_large.jsonl")
MODEL = "knowledgator/gliner-pii-large-v1.0"
QUEUE_FAMILIES = {"drug", "health-condition", "medical-procedure"}
REPORT = Path("results/requeue_from_large_remine.json")


def span_set(path: Path) -> set[tuple[str, str]]:
    out = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        rt = LABEL_TO_RUNTIME_TYPE.get(_norm(row["detector_label"]))
        s = _norm(row["surface"])
        if rt and len(s) >= 2 and not _is_generic_surface(rt, s):
            out.add((rt, s))
    return out


def _index(keys) -> ProfileIndex:
    """ProfileIndex over (runtime_type, surface) pairs, for fuzzy `.find` membership."""
    art: dict = {"profiles": {}}
    for rt, s in keys:
        art["profiles"].setdefault(rt, {})[s] = {}
    return ProfileIndex(art)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not NEW_SPANS.exists():
        spans = detect_clinical_spans(model=MODEL)  # threshold 0.3, miner defaults
        with NEW_SPANS.open("w") as f:
            for sp in spans:
                f.write(json.dumps(sp.__dict__, sort_keys=True) + "\n")
        print(f"mined {len(spans)} spans -> {NEW_SPANS}")
    else:
        print(f"reusing {NEW_SPANS}")

    old, new = span_set(OLD_SPANS), span_set(NEW_SPANS)
    queue = [json.loads(l) for l in QUEUE.read_text().splitlines() if l.strip()]
    queue_keys = {(it["runtime_type"], _norm(it["surface"])) for it in queue}
    old_idx, new_idx, queue_idx = _index(old), _index(new), _index(queue_keys)

    # remove: was detected by the old run, still in the queue, and NOT fuzzy-detected now
    remove_keys = {(rt, s) for rt, s in queue_keys if (rt, s) in old and not new_idx.find(rt, s)}
    remove_idx = _index(remove_keys)
    kept = [it for it in queue if (it["runtime_type"], _norm(it["surface"])) not in remove_keys]

    common_index = ProfileIndex(json.loads(Path("data/lattice_profiles/comm_lattice_profiles.json").read_text()))
    fine_index = ProfileIndex(json.loads(Path("data/lattice_profiles/fine_lattice_profiles.json").read_text()))
    additions = []
    for rt, s in sorted(new):
        if rt not in QUEUE_FAMILIES:
            continue
        if old_idx.find(rt, s) or queue_idx.find(rt, s):   # already known / already queued (fuzzy)
            continue
        if remove_idx.find(rt, s):
            continue
        if common_index.find(rt, s):
            continue
        if is_noise_span(s, rt):
            continue
        fine = fine_index.find(rt, s)
        aliases = list(fine[1].get("aliases", [])) if fine else []
        additions.append({"aliases": aliases, "canonical_value": s, "detector_label_family": rt,
                          "item_id": f"{rt}:{s}", "runtime_type": rt, "surface": s,
                          "task_kind": "level-proposal"})

    by_type = lambda keys: {t: sum(1 for rt, _ in keys if rt == t) for t in sorted({rt for rt, _ in keys})}
    print(f"old mined set: {len(old)} | new mined set: {len(new)} | queue: {len(queue)}")
    print(f"removals (old-mined, no longer detected, in queue): {len(remove_keys)} {by_type(remove_keys)}")
    print(f"additions (newly detected, queue-eligible): {len(additions)} "
          f"{by_type({(a['runtime_type'], a['surface']) for a in additions})}")
    REPORT.write_text(json.dumps({"model": MODEL, "removals": sorted(remove_keys),
                                  "additions": [a["item_id"] for a in additions]}, indent=2) + "\n")
    print(f"detail -> {REPORT}")

    if not args.apply:
        print("(dry run -- rerun with --apply to write the queue)")
        return 0

    shutil.copy(QUEUE, QUEUE.with_name(QUEUE.name + ".prelarge-bak"))
    tmp = QUEUE.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(it, sort_keys=True) + "\n" for it in kept + additions))
    tmp.rename(QUEUE)  # atomic swap; the live producer re-reads on its next resume
    print(f"queue written: {len(queue)} -> {len(kept) + len(additions)} items "
          f"(-{len(remove_keys)} +{len(additions)}); backup {QUEUE.name}.prelarge-bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
