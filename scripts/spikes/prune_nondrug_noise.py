#!/usr/bin/env python
"""Prune non-drug noise surfaces from the drug inventory (queue + run output + proposed artifact).

Mined drug surfaces include detector noise that isn't a drug: devices (ace wrap), fragrances/
excipients, numeric junk, and lab/imaging/clinical abbreviations misfiled as drugs (ecg, mri, cbc,
a1c, ...). The model can only hallucinate lattices for these, so they're removed from the queue (so
a resumed run skips them) and from any already-processed run state. Reviewed decision: keep short
real-drug tokens (azo/mmr/pcp/cla/pop) and homeopathic botanicals.

    PYTHONPATH=src python scripts/spikes/prune_nondrug_noise.py [--apply]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

RUN_ID = "drug-health-procedure-nemotron-3-super"
RD = Path(f"data/lattice_runs/{RUN_ID}")
OUT = Path(f"data/lattice_profiles/proposed/{RUN_ID}.proposed.json")
QUEUE = Path("data/lattice_runs/full-run/queue.jsonl")

_DEVICE = re.compile(r"\b(wrap|wraps|bandage|gauze|tape|swab|kit|dressing|brace|splint|sponge|applicator|cloth|wipe|wipes)\b")
_LAB = re.compile(r"\b(a1c|panel|assay|titer|screen)\b")
_JUNK_NUM = re.compile(r"^[\d][\d .]*$")
_FRAGRANCE = re.compile(r"aldehyde$|cinnamaldehyde|fragrance|\blimonene\b|\blinalool\b")
# real drug/supplement/vaccine surfaces that would otherwise trip the heuristics -> never prune
_KEEP = re.compile(r"tocopherol|ascorbic|lipoic|cholecalciferol|niacin|riboflavin|thiamine|folic|"
                   r"pantothen|biotin|aminobutyric|retino|calciferol|menadione|pyridoxine|cobalamin")
_KEEP_TOKENS = {"azo", "mmr", "pcp", "cla", "pop"}  # reviewed: real drugs/vaccine, keep
# homeopathic botanicals: reviewed decision is KEEP (registered NDC products), so NOT pruned here.


def is_noise(surface: str) -> bool:
    s = surface.strip().lower()
    if _KEEP.search(s) or s in _KEEP_TOKENS:
        return False
    if _DEVICE.search(s) or _LAB.search(s) or _JUNK_NUM.match(s) or _FRAGRANCE.search(s):
        return True
    # short/abbreviation tokens misfiled as drugs (ecg, mri, cbc, bun, ...), excluding kept tokens
    return len(s.replace(" ", "")) <= 3 and not re.search(r"\d", s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    # compute the prune set over the FULL drug inventory in the queue (all 538), not just the
    # handful already processed into the run artifact.
    q = [json.loads(l) for l in QUEUE.read_text().splitlines() if l.strip()]
    drug_surfaces = sorted({it["surface"] for it in q if it.get("runtime_type") == "drug" and it.get("surface")})
    prune = sorted(s for s in drug_surfaces if is_noise(s))
    prune_ids = {f"drug:{s}" for s in prune}
    print(f"drug surfaces in queue: {len(drug_surfaces)} | pruning as non-drug noise: {len(prune)}")
    print(prune)

    # also count how many are already-processed (in accepted/diagnostics) vs queue-only
    acc = [json.loads(l) for l in (RD / "accepted.jsonl").read_text().splitlines() if l.strip()]
    in_accepted = {r["item_id"] for r in acc if r.get("item_id") in prune_ids}
    print(f"of those, already in accepted.jsonl: {len(in_accepted)} (will be removed from run output too)")

    if not args.apply:
        print("\n(dry run -- rerun with --apply to write)")
        return 0

    # backups
    for p in (QUEUE, RD / "accepted.jsonl", RD / "diagnostics.jsonl", OUT):
        shutil.copy(p, p.with_name(p.name + ".preprune-bak"))

    # 1. queue: drop pruned drug items
    q = [json.loads(l) for l in QUEUE.read_text().splitlines() if l.strip()]
    q_kept = [it for it in q if it.get("item_id") not in prune_ids]
    QUEUE.write_text("".join(json.dumps(it, sort_keys=True) + "\n" for it in q_kept))

    # 2. accepted / diagnostics: drop pruned rows
    for name in ("accepted.jsonl", "diagnostics.jsonl"):
        rows = [json.loads(l) for l in (RD / name).read_text().splitlines() if l.strip()]
        kept = [r for r in rows if r.get("item_id") not in prune_ids]
        (RD / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in kept))

    # 3. proposed artifact: drop pruned drug entries (only those already processed exist there)
    art = json.loads(OUT.read_text())
    for s in prune:
        art["profiles"].get("drug", {}).pop(s, None)
    OUT.write_text(json.dumps(art, indent=2, sort_keys=True))

    print(f"\napplied. queue: {len(q)} -> {len(q_kept)} items; removed {len(in_accepted)} accepted rows; "
          f"{len(prune)} drug entries dropped from artifact. backups: *.preprune-bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
