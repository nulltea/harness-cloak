"""Safe auto data-fix for lattice_profiles.json: scan an arms artifact for detected drug spans
that resolve to NO profile entry (the `missing_generalization` class) and, for the unambiguous
brand cases, add the surface as an alias to its existing generic entry via openFDA NDC.

Strict by design (never invents entries/levels, never resolves an ambiguous brand), so it is safe
to run before an arms rebuild. Two-pass workflow:  build_arms_artifact  ->  this script  ->
build_arms_artifact again (now the drug's generalization chain bakes via the new alias).

    python scripts/resolve_drug_aliases.py --arms /tmp/arms.json          # dry run (report only)
    python scripts/resolve_drug_aliases.py --arms /tmp/arms.json --apply  # persist fixes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cloak.lattice_profiles import (
    DEFAULT_NDC_PATH,
    DEFAULT_PROFILE_PATH,
    lookup_entry,
    resolve_drug_generic,
    resolve_missing_drug_aliases,
)


def _detected_drug_surfaces(arms: dict, corpus: str) -> set[str]:
    surfaces: set[str] = set()
    for doc in (arms.get(corpus) or {}).values():
        if not isinstance(doc, dict):
            continue
        walk = doc.get("tau_walk")
        rows = walk[1] if isinstance(walk, list) and len(walk) > 1 else []
        floor = doc.get("all_floor")
        rows = list(rows) + (floor[1] if isinstance(floor, list) and len(floor) > 1 else [])
        for row in rows:
            if isinstance(row, dict) and str(row.get("type")) == "drug" and row.get("surface"):
                surfaces.add(str(row["surface"]))
    return surfaces


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", required=True)
    parser.add_argument("--corpus", default="clinical")
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE_PATH))
    parser.add_argument("--ndc", default=str(DEFAULT_NDC_PATH))
    parser.add_argument("--apply", action="store_true", help="persist fixes to the profile")
    args = parser.parse_args(argv)

    arms = json.loads(Path(args.arms).read_text())
    surfaces = _detected_drug_surfaces(arms, args.corpus)
    missing = sorted(s for s in surfaces if lookup_entry(s, "drug", args.profile) is None)
    resolvable = {s: g for s in missing if (g := resolve_drug_generic(s, args.ndc))}
    unresolved = [s for s in missing if s not in resolvable]

    print(f"detected drug surfaces: {len(surfaces)}; missing generalization: {len(missing)}")
    for surface, generic in sorted(resolvable.items()):
        print(f"  RESOLVABLE  {surface!r} -> generic {generic!r}")
    for surface in unresolved:
        print(f"  UNRESOLVED  {surface!r} (absent generic or ambiguous brand -- review)")

    if args.apply:
        fixed = resolve_missing_drug_aliases(
            missing, profile_path=args.profile, ndc_path=args.ndc)
        print(f"\napplied {len(fixed)} alias fixes to {args.profile}:")
        for surface, canonical in sorted(fixed.items()):
            print(f"  + {surface!r} -> {canonical!r}")
        if len(fixed) < len(resolvable):
            print("  (resolvable-but-not-applied = generic exists in NDC but not as a profile "
                  "entry with levels; needs a new entry, out of strict safe-alias scope)")
    else:
        print("\ndry run; re-run with --apply to persist")


if __name__ == "__main__":
    main()
