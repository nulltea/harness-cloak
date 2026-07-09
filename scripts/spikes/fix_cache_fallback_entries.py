#!/usr/bin/env python
"""Offline repair of cache-fallback lattice entries, reusing already-logged model proposals.

Context: before the reference-miss->model fix (commit 639c9b5), a reference-source miss
(openFDA/DOID/ICD) fell back to the lattice_profiles.json cache and produced cache-anchored
entries, even though the augment step already CALLED the model and logged a good chain in
proposals.jsonl. This rebuilds those entries from their logged model proposals through the exact
model-free path the fixed pipeline uses (extract -> compile -> gate -> persist -> coherence), so a
resumed run skips them. No new API calls.

Cache-anchored entry := an accepted entry with a `deterministic-aset` level and NO real reference
anchor (openfda-pharm-class / doid-is-a / icd10pcs-prefix). Usage:

    PYTHONPATH=src python scripts/spikes/fix_cache_fallback_entries.py [--apply]

Without --apply it prints a dry-run summary and changes nothing.
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from cloak.lattice_producer.counts import compile_level_counts
from cloak.lattice_producer.gates import gate_candidates
from cloak.lattice_producer.merge import persist_proposed_artifact
from cloak.lattice_producer.propose import extract_candidate_levels
from cloak.lattice_producer.coherence import normalize_coherence
from cloak.lattice_producer.io import atomic_write_json, read_jsonl

RUN_ID = "drug-health-procedure-nemotron-3-super"
RD = Path(f"data/lattice_runs/{RUN_ID}")
OUT = Path(f"data/lattice_profiles/proposed/{RUN_ID}.proposed.json")
PROFILES = "data/lattice_profiles/lattice_profiles.json"
REF = {"drug": "openfda-pharm-class", "health-condition": "doid-is-a", "medical-procedure": "icd10pcs-prefix"}


def _cache_anchored(accepted: list[dict]) -> set[str]:
    sf = defaultdict(set)
    for r in accepted:
        sf[(r.get("runtime_type"), r.get("item_id"))].add((r.get("level_grounding") or {}).get("source_family"))
    return {it for (rt, it), s in sf.items() if REF.get(rt) not in s and "deterministic-aset" in s}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = ap.parse_args()

    accepted = read_jsonl(RD / "accepted.jsonl")
    diagnostics = read_jsonl(RD / "diagnostics.jsonl")
    queue = {row["item_id"]: row for row in read_jsonl(RD / "queue.jsonl")}
    proposals = read_jsonl(RD / "proposals.jsonl")

    targets = _cache_anchored(accepted)
    latest_prop = {}
    for p in proposals:
        if p.get("item_id") in targets:
            latest_prop[p["item_id"]] = p  # last proposal wins

    print(f"cache-anchored entries to repair: {len(targets)}")
    print(f"of which have a logged model proposal: {sum(1 for it in targets if it in latest_prop)}")

    # Work against a copy of the real artifact with the 43 removed, so the gate sees the SAME
    # vocabulary a resumed run would (dry run uses a temp copy; apply mutates OUT in place). This
    # makes the preview faithful -- near-dup/no-domain checks depend on that vocabulary state.
    if args.apply:
        for f in ("accepted.jsonl", "diagnostics.jsonl"):
            shutil.copy(RD / f, RD / f"{f}.prefix-bak")
        shutil.copy(OUT, OUT.with_name(OUT.name + ".prefix-bak"))
        working = OUT
    else:
        working = OUT.with_name(OUT.name + ".dryrun-tmp")
        shutil.copy(OUT, working)
    art = json.loads(working.read_text())
    for it in targets:
        rt, surf = it.split(":", 1)
        art.get("profiles", {}).get(rt, {}).pop(surf.strip().lower(), None)
        art.get("profiles", {}).get(rt, {}).pop(surf, None)
    atomic_write_json(working, art)

    new_accepted_rows, new_diag_rows = [], []
    rebuilt_levels = {}
    diag_reasons = collections.Counter()
    accepted_count = diag_count = 0
    for it in sorted(targets):
        item = queue.get(it) or {"item_id": it, "runtime_type": it.split(":", 1)[0], "surface": it.split(":", 1)[1]}
        cands = extract_candidate_levels(latest_prop[it]) if it in latest_prop else []
        compiled = compile_level_counts(item, cands, generated_universe_path=RD / "generated_universe.jsonl")
        result = gate_candidates(item, compiled, proposed_out=str(working))
        if result.accepted:
            accepted_count += 1
            rebuilt_levels[it] = [r["level"] for r in result.accepted]
            rows = [
                {**r, "item_id": it, "runtime_type": item["runtime_type"],
                 "record_id": r.get("record_id") or f"{it}:{r.get('level')}:accepted"}
                for r in result.accepted
            ]
            new_accepted_rows.extend(rows)
            # persist to the working artifact (temp in dry run, OUT in apply) so later items see it
            persist_proposed_artifact(PROFILES, working, run_id=RUN_ID, item=item, accepted=result.accepted)
        else:
            diag_count += 1
            reason = (result.diagnostics[0].get("reason") if result.diagnostics else "no_candidates")
            diag_reasons[reason] += 1
            new_diag_rows.append({**item, "reason": reason, "record_id": f"{it}:{reason}:diagnostic"})

    print(f"\nrebuilt: {accepted_count} accepted, {diag_count} diverted to diagnostics")
    print("diagnostic reasons:", dict(diag_reasons))
    print("sample rebuilt chains:")
    for it, lv in list(rebuilt_levels.items())[:10]:
        print(f"  {it.split(':',1)[1]:38} -> {lv}")

    if not args.apply:
        working.unlink(missing_ok=True)
        print("\n(dry run -- rerun with --apply to write)")
        return 0

    # rewrite accepted/diagnostics: keep non-target rows, append rebuilt
    kept_acc = [r for r in accepted if r.get("item_id") not in targets]
    kept_diag = [r for r in diagnostics if r.get("item_id") not in targets]
    (RD / "accepted.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in kept_acc + new_accepted_rows))
    (RD / "diagnostics.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in kept_diag + new_diag_rows))

    # final coherence pass over the whole artifact
    art = json.loads(OUT.read_text())
    normalize_coherence(art)
    OUT.write_text(json.dumps(art, indent=2, sort_keys=True))
    print(f"\napplied. accepted.jsonl: {len(kept_acc)} kept + {len(new_accepted_rows)} rebuilt rows; "
          f"coherence re-run. backups: *.prefix-bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
