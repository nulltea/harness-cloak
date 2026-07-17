"""One-time migration: remap the anchor labels whose real-world magnitudes were corrected in
data/lattice_sources/reference/*_class_anchors.json (class-universe sizes -> runtime-type-
conditional anonymity-set sizes) into an existing lattice_profiles artifact.

Only these labels change; every other count is left untouched because the artifact's original
model-proposed counts were already overwritten by the old corpus-membership pass and cannot be
recovered here (that requires a producer re-run from the cached proposals). Verifies each
touched chain stays monotone.

Usage: PYTHONPATH=src python scripts/spikes/apply_anchor_count_fix.py data/lattice_profiles/lattice_profiles.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from cloak.lattice_producer.coherence import load_reference_anchors

# ONLY the labels whose anchor values were corrected (class-universe -> conditional magnitude).
# Other anchored labels are deliberately left alone: the artifact's stored counts for them can
# legitimately differ from the anchor table (per-run isotonic pooling), and remapping them here
# would reintroduce nonmonotone chains (e.g. meclizine's antihistamine rung).
CORRECTED_LABELS = {
    "drug": ("therapeutic agent", "pharmaceutical compound", "chemical substance"),
    "health-condition": ("clinical finding", "medical condition"),
}


def main(path: str) -> None:
    artifact_path = Path(path)
    artifact = json.loads(artifact_path.read_text())
    changed_rows = 0
    changed_labels: dict[str, int] = {}
    for runtime_type, labels in CORRECTED_LABELS.items():
        anchors = {label: load_reference_anchors(runtime_type)[label] for label in labels}
        entries = artifact.get("profiles", {}).get(runtime_type, {})
        for name, row in entries.items():
            levels = row.get("levels") or []
            counts = row.get("level_counts") or {}
            touched = False
            for level in levels:
                anchor = anchors.get(level)
                if anchor is None or counts.get(level) == float(anchor):
                    continue
                counts[level] = float(anchor)
                grounding = (row.get("level_grounding") or {}).get(level)
                if isinstance(grounding, dict):
                    grounding["count_basis"] = "real-world-reference-estimate"
                    grounding["count_evidence"] = (
                        f"'{level}' set from the corrected runtime-type-conditional magnitude in "
                        f"data/lattice_sources/reference/ (class-universe sizes were wrong here); "
                        f"still not certifying"
                    )
                changed_labels[level] = changed_labels.get(level, 0) + 1
                touched = True
            if touched:
                row["level_counts"] = counts
                if levels:
                    row["count"] = counts.get(levels[0], row.get("count"))
                changed_rows += 1
                values = [counts[level] for level in levels if counts.get(level) is not None]
                assert values == sorted(values), f"{runtime_type}/{name} nonmonotone after fix: {list(zip(levels, values))}"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    print(f"changed_rows={changed_rows}")
    for label, count in sorted(changed_labels.items()):
        print(f"  {label}: {count} rows")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/lattice_profiles/lattice_profiles.json")
