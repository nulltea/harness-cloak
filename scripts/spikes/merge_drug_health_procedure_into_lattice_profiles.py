"""One-off: merge the coherence-cleaned + openFDA-EPC-patched drug-health-procedure proposal
into data/lattice_profiles/lattice_profiles.json.

lattice_profiles.json is itself a build output of scripts/merge_lattice_profiles.py (from
comm_lattice_profiles.json + mined_lattice_profiles.json) -- this reuses the same
merge_profile_artifacts() logic (now level_counts/level_grounding aware) rather than hand-editing
the JSON, so the result stays structurally consistent with what that script produces.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from merge_lattice_profiles import merge_profile_artifacts  # noqa: E402

BASE = Path("data/lattice_profiles/lattice_profiles.json")
INCOMING = Path("data/lattice_profiles/proposed/drug-health-procedure.proposed.cleaned.json")


def main() -> None:
    base = json.loads(BASE.read_text())
    incoming = json.loads(INCOMING.read_text())
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
