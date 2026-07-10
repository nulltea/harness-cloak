"""One-off: merge the nemotron-3-super drug-health-procedure proposal into
data/lattice_profiles/lattice_profiles.json.

lattice_profiles.json is itself a build output of scripts/merge_lattice_profiles.py (from
comm_lattice_profiles.json + mined_lattice_profiles.json) -- this reuses the same
merge_profile_artifacts() logic (now level_counts/level_grounding aware) rather than hand-editing
the JSON, so the result stays structurally consistent with what that script produces.

It also reschematizes the pre-existing (non-drug/health/procedure) profiles to the proposal's
entry schema by stamping entry_origin="observed-surface" on every base row -- these rows are real
dataset surfaces (GeoNames/CLDR/ESCO/...), so "observed-surface" is factual. level_counts and
level_grounding are deliberately NOT synthesized for them: those are per-level anonymity-set sizes
that only a producer run grounds, and fabricating them from the single top-level `count` would
invent privacy numbers (empirical-honesty rule). merge_profile_artifacts() preserves untouched
profiles verbatim, so the stamped entry_origin flows through; the 3 incoming profiles are deep-copied
with their own entry_origin/level_counts/level_grounding intact.

The incoming runtime types (drug/health-condition/medical-procedure) REPLACE, not union with, any
same-typed profiles already in the base: a regenerated proposal supersedes the prior run's rows, so
we drop those types from the base before merging. This makes the merge idempotent whether the base
is a fresh 9-profile prep or already carries a previous drug/health/procedure merge.

Usage: python <this> [incoming.json]   (defaults to the live nemotron proposal path)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from merge_lattice_profiles import merge_profile_artifacts  # noqa: E402

BASE = Path("data/lattice_profiles/lattice_profiles.json")
INCOMING = Path("data/lattice_profiles/proposed/drug-health-procedure-nemotron-3-super.proposed.json")


def main() -> None:
    incoming_path = Path(sys.argv[1]) if len(sys.argv) > 1 else INCOMING
    base = json.loads(BASE.read_text())
    incoming = json.loads(incoming_path.read_text())

    dropped = {rt: len(base["profiles"].get(rt, {})) for rt in incoming.get("profiles", {}) if rt in base.get("profiles", {})}
    for rt in incoming.get("profiles", {}):
        base.get("profiles", {}).pop(rt, None)
    print(f"dropped superseded base profiles: {dropped}")

    stamped = 0
    for entries in base.get("profiles", {}).values():
        for row in entries.values():
            if "entry_origin" not in row:
                row["entry_origin"] = "observed-surface"
                stamped += 1
    print(f"stamped entry_origin on {stamped} existing rows")

    merged = merge_profile_artifacts(base, incoming)

    BASE.write_text(json.dumps(merged, indent=2, sort_keys=True))

    with_counts = {
        rt: sum(1 for row in entries.values() if "level_counts" in row)
        for rt, entries in merged["profiles"].items()
        if rt in ("drug", "health-condition")
    }
    print(f"wrote {BASE}")
    print(f"profiles: {{{', '.join(f'{k}: {len(v)}' for k, v in merged['profiles'].items())}}}")
    print(f"entries with level_counts: {with_counts}")


if __name__ == "__main__":
    main()
